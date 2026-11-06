"""Reusable CUDA decode graphs over the existing LFM2.5 packed weights."""
from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from climbing_gym.inference.lfm25_cuda_kernels import conv_update, linear, prepare_qkv, rms_norm, swiglu
from climbing_gym.inference.lfm25_fp8 import rms_norm_fp8, swiglu_fp8
from climbing_gym.inference.lfm25_engine import LFM25GenerationOutput, LFM25HybridCache, _make_generator, apply_sequence_stops

if TYPE_CHECKING:
    from climbing_gym.inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig


@dataclass
class _DecodeGraph:
    cache: LFM25HybridCache
    input_tokens: torch.Tensor
    logits: torch.Tensor
    graph: torch.cuda.CUDAGraph
    active: torch.Tensor | None = None
    compact_graphs: dict = field(default_factory=dict)


class LFM25CUDADecoder:
    """Generation accelerator with bounded graph/cache storage.

    Each graph owns stable cache addresses. Prompts prefill those buffers
    directly when each request has one rollout. Subsequent tokens replay the
    same graph. Completed
    rows stay in the batch with masked outputs, avoiding new graphs each time
    a request stops. Packed weights stay shared with the reference engine, so
    its in-place policy synchronization also updates captured graphs.
    """

    supported_conv_kernel_sizes = (3,)

    def __init__(self, engine: LFM25Engine, *, max_graphs: int = 2, bf16_weight_packing: bool = True,
                 max_prefill_tokens: int = 4096, finish_check_interval: int = 32,
                 skip_finished_attention: bool = True, fp8_mlp: bool = False, fp8_kv: bool = False,
                 adaptive_batching: bool = False, fp8_lm_head: bool = False):
        if engine.device.type != "cuda" or engine.dtype not in (torch.float16, torch.bfloat16):
            raise ValueError("LFM2.5 CUDA graphs require CUDA and float16 or bfloat16 weights")
        if engine.config.conv_kernel_size not in self.supported_conv_kernel_sizes or engine.config.conv_bias:
            raise ValueError(f"Native CUDA graphs require bias-free convolutions of width {self.supported_conv_kernel_sizes}")
        if engine.config.head_dim not in (32, 64, 128, 256):
            raise ValueError("LFM2.5 CUDA graphs require head_dim in {32, 64, 128, 256}")
        if max_graphs <= 0:
            raise ValueError("max_graphs must be > 0")
        self.engine = engine
        self.max_graphs = max_graphs
        if max_prefill_tokens <= 0:
            raise ValueError("max_prefill_tokens must be positive")
        self.max_prefill_tokens = max_prefill_tokens
        if type(finish_check_interval) is not int or finish_check_interval <= 0:
            raise ValueError("finish_check_interval must be a positive integer")
        self.finish_check_interval = finish_check_interval
        self.skip_finished_attention = bool(skip_finished_attention)
        self.bf16_weight_packing = bf16_weight_packing and engine.dtype == torch.bfloat16
        self.packed_weights = {}
        self.fp8_mlp = bool(fp8_mlp)
        self.fp8_kv = bool(fp8_kv)
        self.adaptive_batching = bool(adaptive_batching)
        self.fp8_lm_head = bool(fp8_lm_head)
        self.fp8_weights = {}
        if self.fp8_mlp and torch.cuda.get_device_capability(engine.device) < (8, 9):
            raise ValueError("Native FP8 MLP requires NVIDIA compute capability 8.9 or later")
        if self.fp8_kv and torch.cuda.get_device_capability(engine.device) < (8, 9):
            raise ValueError("Native FP8 KV cache requires NVIDIA compute capability 8.9 or later")
        if self.fp8_lm_head and torch.cuda.get_device_capability(engine.device) < (8, 9):
            raise ValueError("Native FP8 vocabulary head requires NVIDIA compute capability 8.9 or later")
        self.graphs: OrderedDict[tuple[int, int, bool], _DecodeGraph] = OrderedDict()
        self._active_session = None
        # Keep one capture stream across graph eviction/clear. Cycling through
        # pooled streams while recapturing from successive rollout workers can
        # deadlock an allocation inside capture once the stream pool wraps.
        self._capture_stream = torch.cuda.Stream(device=engine.device)

    def clear(self) -> None:
        if self._active_session is not None:
            self._active_session.close()
        self.graphs.clear()
        self.packed_weights.clear()
        self.fp8_weights.clear()

    def refresh_weights(self) -> None:
        for packed in self.packed_weights.values():
            packed.refresh()
        for packed in self.fp8_weights.values():
            packed.refresh()

    def _prepare_fp8_weights(self):
        if not (self.fp8_mlp or self.fp8_lm_head) or self.fp8_weights:
            return
        from climbing_gym.inference.lfm25_fp8 import PackedFP8Weight

        matrices = [weight for layer in self.engine.weights.layers
                    for weight in (layer.w13_weight, layer.w2_weight)] if self.fp8_mlp else []
        if self.fp8_lm_head:
            matrices.append(self.engine.weights.lm_head_weight)
        self.fp8_weights = {id(weight): PackedFP8Weight.from_weight(weight) for weight in matrices}

    def _prepare_weights(self) -> None:
        if not self.bf16_weight_packing or self.packed_weights:
            return
        from climbing_gym.inference.lfm25_bf16 import PackedBF16Weight

        matrices = [] if self.fp8_lm_head else [self.engine.weights.lm_head_weight]
        # The MLP and vocabulary matrices dominate decode traffic. Packing
        # smaller operator matrices added residency without a measurable gain.
        if not self.fp8_mlp:
            for layer in self.engine.weights.layers:
                matrices.extend((layer.w13_weight, layer.w2_weight))
        self.packed_weights = {
            id(weight): PackedBF16Weight.from_weight(weight) for weight in matrices
            if weight.size(1) % 2 == 0 and weight.size(1) <= 16384
        }

    def _linear(self, x, weight):
        packed = self.packed_weights.get(id(weight)) if x.size(0) == 1 else None
        return packed.linear(x) if packed is not None else linear(x, weight)

    def _forward(self, cache: LFM25HybridCache, tokens: torch.Tensor,
                 active: torch.Tensor | None = None, *, slot_ids=None, positions=None) -> torch.Tensor:
        engine = self.engine
        cfg = engine.config
        linear = self._linear
        positions = cache.seq_lens if positions is None else positions
        hidden = F.embedding(tokens, engine.weights.embed_tokens)
        residual = None
        for layer in engine.weights.layers:
            operator_input, residual = rms_norm(hidden, layer.operator_norm_weight, cfg.norm_eps, residual)
            if layer.attention is not None:
                attention = layer.attention
                q, k, v = prepare_qkv(
                    linear(operator_input, attention.qkv_weight),
                    attention.q_norm_weight, attention.k_norm_weight,
                    engine._rope_cos, engine._rope_sin, positions,
                    cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.norm_eps,
                )
                if self.fp8_kv:
                    from climbing_gym.inference.lfm25_fp8_kv import decode
                    idx = attention.cache_index
                    attended = decode(q, k, v, cache.k_cache[idx], cache.v_cache[idx],
                                      cache.k_scales[idx], cache.v_scales[idx], positions,
                                      active=active, scale=engine._scaling, slot_ids=slot_ids)
                elif slot_ids is not None:
                    from climbing_gym.inference.lfm25_batching import append_bf16_kv
                    idx = attention.cache_index
                    append_bf16_kv(k, v, cache.k_cache[idx], cache.v_cache[idx], positions, slot_ids, active)
                    attended = flash_attn_with_kvcache(
                        q, cache.k_cache[idx], cache.v_cache[idx],
                        cache_seqlens=torch.where(active, positions + 1, 1),
                        cache_batch_idx=slot_ids.clamp_min(0), softmax_scale=engine._scaling, causal=True)
                else:
                    attended = flash_attn_with_kvcache(
                        q, cache.k_cache[attention.cache_index], cache.v_cache[attention.cache_index],
                        k=k, v=v, cache_seqlens=cache.seq_lens,
                        softmax_scale=engine._scaling, causal=True,
                    )
                operator_output = linear(attended.reshape(tokens.numel(), -1), attention.out_weight)
            else:
                conv = layer.conv
                assert conv is not None
                projected = linear(operator_input, conv.in_weight)
                convolved = conv_update(projected, conv.conv_weight, cache.conv_state[conv.cache_index], active, slot_ids)
                operator_output = linear(convolved, conv.out_weight)
            if self.fp8_mlp:
                ffn_input, scales, residual = rms_norm_fp8(operator_output, layer.ffn_norm_weight, cfg.norm_eps, residual)
                gate_up = self.fp8_weights[id(layer.w13_weight)].linear_quantized(ffn_input, scales, dtype=engine.dtype)
                activated, scales = swiglu_fp8(gate_up)
                hidden = self.fp8_weights[id(layer.w2_weight)].linear_quantized(activated, scales, dtype=engine.dtype)
            else:
                ffn_input, residual = rms_norm(operator_output, layer.ffn_norm_weight, cfg.norm_eps, residual)
                hidden = linear(swiglu(linear(ffn_input, layer.w13_weight)), layer.w2_weight)
        if self.fp8_lm_head:
            hidden, scales, _ = rms_norm_fp8(hidden, engine.weights.embedding_norm_weight, cfg.norm_eps, residual)
        else:
            hidden, _ = rms_norm(hidden, engine.weights.embedding_norm_weight, cfg.norm_eps, residual)
        if slot_ids is None:
            cache.seq_lens.add_(1 if active is None else active.to(torch.int32))
        else:
            from climbing_gym.inference.lfm25_batching import advance_positions
            advance_positions(cache.seq_lens, positions, slot_ids, active)
        if self.fp8_lm_head:
            return self.fp8_weights[id(engine.weights.lm_head_weight)].linear_quantized(hidden, scales, dtype=engine.dtype)
        return linear(hidden, engine.weights.lm_head_weight)

    def _select_graph(self, entry, live_slots):
        """Select a compute batch; all variants retain the same physical cache."""
        maximum = entry.input_tokens.numel()
        if not self.adaptive_batching or maximum <= 16:
            return entry.graph, maximum
        size = 16
        while size < len(live_slots):
            size *= 2
        if size >= maximum:
            return entry.graph, maximum
        if size not in entry.compact_graphs:
            from climbing_gym.inference.lfm25_batching import capture_compact
            entry.compact_graphs[size] = capture_compact(self, entry, size)
        compact = entry.compact_graphs[size]
        compact.set_slots(live_slots)
        return compact.graph, size

    @torch.inference_mode()
    def _get_graph(self, batch_size: int, cache_len: int, *, masked: bool = False) -> _DecodeGraph:
        # Large batches multiply unused cache positions across hundreds of
        # requests. Finer rounding avoids a large allocation jump when one
        # prompt crosses a 128-token boundary.
        alignment = 32 if batch_size >= 64 else 128
        capacity = min(self.engine.max_model_len, ((cache_len + alignment - 1) // alignment) * alignment)
        key = (batch_size, capacity, masked)
        if key in self.graphs:
            self.graphs.move_to_end(key)
            return self.graphs[key]
        # Evict before allocating the next cache, limiting peak residency too.
        if len(self.graphs) >= self.max_graphs:
            self.graphs.popitem(last=False)
        self._prepare_fp8_weights()
        if batch_size == 1:
            # Drop an evicted large cache before allocating encoded weights.
            self._prepare_weights()
        elif not any(key[0] == 1 for key in self.graphs):
            # No surviving graph uses these extra single-request copies.
            self.packed_weights.clear()
        cache = self.engine._empty_cache(max_slots=batch_size, max_seq_len=capacity, fp8_kv=self.fp8_kv)
        tokens = torch.zeros(batch_size, device=self.engine.device, dtype=torch.long)
        active = torch.ones(batch_size, device=self.engine.device, dtype=torch.bool) if masked else None
        # Preserve prefill logits for newly admitted rows while other rows
        # consume pending tokens. The addresses survive every phase change.
        saved_logits = torch.zeros((batch_size, self.engine.config.vocab_size),
                                   device=self.engine.device, dtype=self.engine.dtype) if masked else None

        def forward():
            logits = self._forward(cache, tokens) if active is None else self._forward(cache, tokens, active)
            if saved_logits is not None:
                saved_logits.copy_(torch.where(active[:, None], logits, saved_logits))
                return saved_logits
            return logits
        stream = self._capture_stream
        current = torch.cuda.current_stream(self.engine.device)
        stream.wait_stream(current)
        with torch.cuda.stream(stream):
            # Initialize cuBLAS / FlashAttention and compile Triton before capture.
            for _ in range(2):
                cache.seq_lens.zero_()
                forward()
            cache.seq_lens.zero_()
        current.wait_stream(stream)
        torch.cuda.synchronize(self.engine.device)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            logits = forward()
        current.wait_stream(stream)
        entry = _DecodeGraph(cache, tokens, logits, graph, active)
        self.graphs[key] = entry
        if batch_size >= 64:
            # Uncaptured warmup scratch is no longer needed. Releasing it here
            # avoids reserving two sets of workspaces at the VRAM limit.
            torch.cuda.empty_cache()
        return entry

    @torch.inference_mode()
    def generate(self, input_ids, cu_seqlens, max_seqlen: int, repeats: int, sampling: LFM25SamplingConfig):
        if self._active_session is not None:
            raise RuntimeError("Close the native generation session before calling generate")
        if repeats <= 0:
            raise ValueError("repeats must be > 0")
        engine = self.engine
        cache_len = int(max_seqlen) + sampling.max_new_tokens
        if cache_len > engine.max_model_len:
            raise ValueError(f"max_seq_len={cache_len} exceeds engine max_model_len={engine.max_model_len}")
        batch_size = (cu_seqlens.numel() - 1) * repeats
        entry = self._get_graph(batch_size, cache_len, masked=self.adaptive_batching) if sampling.max_new_tokens > 1 else None
        decode_graph = entry.graph if entry is not None else None
        # Write prompts directly into the graph's cache. This avoids keeping
        # two prompt caches and a gather temporary at the maximum batch size.
        prefix = engine.prefill(
            input_ids, cu_seqlens, max_seqlen, max_cache_len=max_seqlen,
            cache=entry.cache if entry is not None and repeats == 1 else None,
            max_prefill_tokens=self.max_prefill_tokens,
            logits_buffer=entry.logits if entry is not None and repeats == 1 else None,
        )
        if entry is not None and repeats != 1:
            source_slots = torch.arange(prefix.batch_size, device=engine.device).repeat_interleave(repeats)
            entry.cache.copy_prefix_from(prefix.cache, source_slots, max_seqlen)
        logits = prefix.next_token_logits if repeats == 1 else prefix.next_token_logits.repeat_interleave(repeats, dim=0)
        seen = engine._build_seen_token_mask(prefix, repeats, sampling)
        shape = (prefix.batch_size, repeats)
        del prefix  # Release prompt logits after the first sampled token.
        generator = _make_generator(engine.device, sampling.seed)
        tokens = torch.zeros((batch_size, sampling.max_new_tokens), device=engine.device, dtype=torch.long)
        logprobs = torch.zeros_like(tokens, dtype=torch.float32) if sampling.return_logprobs else None
        k = sampling.behavior_topk
        if k > engine.config.vocab_size:
            raise ValueError('Behavior head exceeds the vocabulary')
        head_ids = torch.empty((*tokens.shape, k), device=engine.device, dtype=torch.int32) if k else None
        head_logprobs = torch.empty_like(head_ids, dtype=torch.float32) if k else None
        lengths = torch.zeros(batch_size, device=engine.device, dtype=torch.int32)
        reasons = torch.zeros(batch_size, device=engine.device, dtype=torch.int8)
        alive = torch.ones(batch_size, device=engine.device, dtype=torch.bool)
        fixed_length = sampling.ignore_eos and not sampling.stop_token_ids and not sampling.stop_token_sequences and sampling.stop_table is None
        stop_patterns = [torch.tensor(s, device=engine.device) for s in sampling.stop_token_sequences]
        stop_states = torch.zeros(batch_size, device=engine.device, dtype=torch.long)
        fused_sampling = sampling.repetition_penalty == 1.0 and (
            sampling.temperature == 0.0 or (sampling.top_k == 0 and sampling.top_p == 1.0)
        )
        rows = torch.arange(batch_size, device=engine.device)
        for step in range(sampling.max_new_tokens):
            if fused_sampling:
                from climbing_gym.inference.lfm25_sampling import sample_logits

                sampled, raw_logprobs = sample_logits(
                    logits, temperature=sampling.temperature, seed=sampling.seed,
                    step=step, return_logprobs=sampling.return_logprobs,
                )
            else:
                sampled, raw_logprobs, _ = engine._sample(logits, sampling, generator=generator, seen_token_mask=seen)
            if k:
                from climbing_gym.inference.lfm25_sampling import behavior_topk
                ids, probs = behavior_topk(logits, sampled, raw_logprobs, k)
                head_ids[:, step] = ids
                head_logprobs[:, step] = probs
            tokens[:, step] = sampled if fixed_length else torch.where(alive, sampled, 0)
            if logprobs is not None:
                logprobs[:, step] = raw_logprobs if fixed_length else torch.where(alive, raw_logprobs, 0)
            if seen is not None:
                seen[rows, sampled] = True
            if not fixed_length:
                lengths.add_(alive.to(torch.int32))
                current_reasons = engine._finish_reasons(sampled, sampling)
                if sampling.stop_table is not None:
                    from climbing_gym.inference.text_stops import advance_text_stops
                    stop_states, current_reasons = advance_text_stops(sampling.stop_table, stop_states, sampled, current_reasons)
                if stop_patterns:
                    current_reasons = apply_sequence_stops(tokens[:, :step + 1], current_reasons, stop_patterns)
                reasons.copy_(torch.where(alive, current_reasons, reasons))
                alive.logical_and_(current_reasons == 0)
                # EOS/stop masking still happens on every token. Only polling
                # the host is amortized, avoiding a CUDA synchronization on
                # every decode step. Extra work after the last row stops is
                # masked and cannot extend any returned completion.
                check_interval = self.finish_check_interval if batch_size >= 16 else 1
                if (step + 1) % check_interval == 0 or step + 1 == sampling.max_new_tokens:
                    if self.adaptive_batching:
                        live_slots = [i for i, live in enumerate(alive.cpu().tolist()) if live]
                        if not live_slots:
                            break
                        if step + 1 < sampling.max_new_tokens:
                            decode_graph, _ = self._select_graph(entry, live_slots)
                    elif not bool(alive.any().item()):
                        break
            if step + 1 < sampling.max_new_tokens:
                assert entry is not None
                if not fixed_length and self.skip_finished_attention:
                    # Finished rows have masked outputs and never become live
                    # again. Stop reading their growing KV histories while
                    # preserving fixed graph shapes and every live row's state.
                    entry.cache.seq_lens.masked_fill_(~alive, 0)
                entry.input_tokens.copy_(sampled)
                if entry.active is not None:
                    entry.active.copy_(alive)
                decode_graph.replay()
                logits = entry.logits
        if fixed_length:
            lengths.fill_(sampling.max_new_tokens)
        reasons.masked_fill_(reasons == 0, 3)
        return LFM25GenerationOutput(
            token_ids=tokens.view(*shape, sampling.max_new_tokens), lengths=lengths.view(*shape),
            logprobs=None if logprobs is None else logprobs.view(*shape, sampling.max_new_tokens),
            finish_reasons=reasons.view(*shape),
            topk_token_ids=None if head_ids is None else head_ids.view(*shape, sampling.max_new_tokens, k),
            topk_logprobs=None if head_logprobs is None else head_logprobs.view(*shape, sampling.max_new_tokens, k),
        )
