from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
import torch.nn.functional as F
from safetensors.torch import load_file

from climbing_gym.models.lfm2 import LFM2Config

try:
    from flash_attn import flash_attn_func, flash_attn_with_kvcache
except Exception:
    flash_attn_func = None
    flash_attn_with_kvcache = None


@dataclass(frozen=True, slots=True)
class LFM25SamplingConfig:
    temperature: float = 1.0
    top_k: int = 0
    top_p: float = 1.0
    repetition_penalty: float = 1.0
    max_new_tokens: int = 256
    stop_token_ids: tuple[int, ...] = ()
    return_logprobs: bool = True
    seed: int = 0
    ignore_eos: bool = False
    stop_token_sequences: tuple[tuple[int, ...], ...] = ()
    stop_table: Any = None
    behavior_topk: int = 0

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < 0:
            raise ValueError("top_k must be 0 or positive")
        if not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        if self.repetition_penalty <= 0.0:
            raise ValueError("repetition_penalty must be > 0")
        if self.max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be > 0")
        if type(self.behavior_topk) is not int or not 0 <= self.behavior_topk <= 128:
            raise ValueError('behavior_topk must be an integer between 0 and 128')
        if self.behavior_topk and (not self.return_logprobs or self.temperature != 1.0
                                  or self.top_k or self.top_p != 1.0 or self.repetition_penalty != 1.0):
            raise ValueError('behavior_topk requires actual full-support temperature-1 logprobs')
        if any(not sequence for sequence in self.stop_token_sequences):
            raise ValueError("Stop token sequences must be nonempty")


@dataclass(frozen=True, slots=True)
class LFM25AttentionWeights:
    qkv_weight: torch.Tensor
    out_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    cache_index: int


@dataclass(frozen=True, slots=True)
class LFM25ConvWeights:
    in_weight: torch.Tensor
    conv_weight: torch.Tensor
    out_weight: torch.Tensor
    cache_index: int


@dataclass(frozen=True, slots=True)
class LFM25LayerWeights:
    operator_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    attention: LFM25AttentionWeights | None = None
    conv: LFM25ConvWeights | None = None

    @property
    def is_attention_layer(self) -> bool:
        return self.attention is not None


@dataclass(frozen=True, slots=True)
class LFM25PackedWeights:
    config: LFM2Config
    embed_tokens: torch.Tensor
    embedding_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    layers: tuple[LFM25LayerWeights, ...]
    device: torch.device
    dtype: torch.dtype

    @property
    def num_attention_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.attention is not None)

    @property
    def num_conv_layers(self) -> int:
        return sum(1 for layer in self.layers if layer.conv is not None)

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "LFM25PackedWeights":
        path = Path(model_dir)
        config = LFM2Config.from_pretrained(path)
        state = _load_state_dict(path)
        return cls.from_state_dict(config, state, device=device, dtype=dtype)

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> "LFM25PackedWeights":
        first_param = next(model.parameters())
        resolved_device = device or first_param.device
        resolved_dtype = dtype or first_param.dtype
        state = model.state_dict()
        config = _coerce_lfm2_config(getattr(model, "config", None))
        return cls.from_state_dict(config, state, device=resolved_device, dtype=resolved_dtype)

    @classmethod
    def from_state_dict(
        cls,
        config: LFM2Config,
        state_dict: Mapping[str, torch.Tensor],
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> "LFM25PackedWeights":
        config.validate()

        def tensor(key: str) -> torch.Tensor:
            value = _lookup_state_tensor(state_dict, key)
            return value.detach().to(device=device, dtype=dtype).contiguous()

        embed_tokens = tensor("model.embed_tokens.weight")
        lm_head_weight = (
            tensor("model.lm_head.weight")
            if _has_state_tensor(state_dict, "model.lm_head.weight")
            else embed_tokens
        )

        layers: list[LFM25LayerWeights] = []
        attn_cache_idx = 0
        conv_cache_idx = 0
        for layer_idx, layer_type in enumerate(config.layer_types):
            prefix = f"model.layers.{layer_idx}"
            w13_weight = torch.cat(
                [
                    tensor(f"{prefix}.feed_forward.w1.weight"),
                    tensor(f"{prefix}.feed_forward.w3.weight"),
                ],
                dim=0,
            ).contiguous()
            common = {
                "operator_norm_weight": tensor(f"{prefix}.operator_norm.weight"),
                "ffn_norm_weight": tensor(f"{prefix}.ffn_norm.weight"),
                "w13_weight": w13_weight,
                "w2_weight": tensor(f"{prefix}.feed_forward.w2.weight"),
            }
            if layer_type == "full_attention":
                qkv_weight = torch.cat(
                    [
                        tensor(f"{prefix}.self_attn.q_proj.weight"),
                        tensor(f"{prefix}.self_attn.k_proj.weight"),
                        tensor(f"{prefix}.self_attn.v_proj.weight"),
                    ],
                    dim=0,
                ).contiguous()
                layers.append(
                    LFM25LayerWeights(
                        **common,
                        attention=LFM25AttentionWeights(
                            qkv_weight=qkv_weight,
                            out_weight=tensor(f"{prefix}.self_attn.out_proj.weight"),
                            q_norm_weight=tensor(f"{prefix}.self_attn.q_layernorm.weight"),
                            k_norm_weight=tensor(f"{prefix}.self_attn.k_layernorm.weight"),
                            cache_index=attn_cache_idx,
                        ),
                    )
                )
                attn_cache_idx += 1
            elif layer_type == "conv":
                conv_weight = tensor(f"{prefix}.conv.conv.weight")
                if conv_weight.ndim != 3 or conv_weight.size(1) != 1:
                    raise ValueError(f"Expected depthwise conv weight [H,1,K], got {tuple(conv_weight.shape)}")
                layers.append(
                    LFM25LayerWeights(
                        **common,
                        conv=LFM25ConvWeights(
                            in_weight=tensor(f"{prefix}.conv.in_proj.weight"),
                            conv_weight=conv_weight[:, 0, :].contiguous(),
                            out_weight=tensor(f"{prefix}.conv.out_proj.weight"),
                            cache_index=conv_cache_idx,
                        ),
                    )
                )
                conv_cache_idx += 1
            else:
                raise ValueError(f"Unsupported LFM2 layer type: {layer_type!r}")

        return cls(
            config=config,
            embed_tokens=embed_tokens,
            embedding_norm_weight=tensor("model.embedding_norm.weight"),
            lm_head_weight=lm_head_weight,
            layers=tuple(layers),
            device=device,
            dtype=dtype,
        )

    @torch.inference_mode()
    def copy_from_model_(self, model: Any) -> None:
        """Copy a training model into the existing packed inference buffers.

        The packed tensors are deliberately updated in place.  Besides avoiding
        a second model-sized allocation, this is required for ``torch.compile``
        decode graphs, which retain references to the original packed tensors.
        All source shapes are checked before the first destination is mutated so
        an incompatible model cannot leave the engine partially synchronized.
        """

        source_config = _coerce_lfm2_config(getattr(model, "config", None))
        _validate_sync_config(self.config, source_config)

        state_dict = model.state_dict()
        if not isinstance(state_dict, Mapping):
            raise TypeError("model.state_dict() must return a mapping of tensor names to tensors")

        copies: list[tuple[str, torch.Tensor, torch.Tensor]] = []

        def add_copy(name: str, destination: torch.Tensor, source: torch.Tensor) -> None:
            if not torch.is_tensor(source):
                raise TypeError(f"LFM25 sync source {name!r} is not a tensor")
            if tuple(source.shape) != tuple(destination.shape):
                raise ValueError(
                    f"LFM25 sync shape mismatch for {name!r}: "
                    f"expected {tuple(destination.shape)}, got {tuple(source.shape)}"
                )
            copies.append((name, destination, source))

        add_copy(
            "model.embed_tokens.weight",
            self.embed_tokens,
            _lookup_state_tensor(state_dict, "model.embed_tokens.weight"),
        )
        add_copy(
            "model.embedding_norm.weight",
            self.embedding_norm_weight,
            _lookup_state_tensor(state_dict, "model.embedding_norm.weight"),
        )
        lm_head_source = (
            _lookup_state_tensor(state_dict, "model.lm_head.weight")
            if _has_state_tensor(state_dict, "model.lm_head.weight")
            else _lookup_state_tensor(state_dict, "model.embed_tokens.weight")
        )
        add_copy("model.lm_head.weight", self.lm_head_weight, lm_head_source)

        for layer_idx, (layer_type, destination_layer) in enumerate(
            zip(source_config.layer_types, self.layers, strict=True)
        ):
            prefix = f"model.layers.{layer_idx}"
            add_copy(
                f"{prefix}.operator_norm.weight",
                destination_layer.operator_norm_weight,
                _lookup_state_tensor(state_dict, f"{prefix}.operator_norm.weight"),
            )
            add_copy(
                f"{prefix}.ffn_norm.weight",
                destination_layer.ffn_norm_weight,
                _lookup_state_tensor(state_dict, f"{prefix}.ffn_norm.weight"),
            )
            mlp_rows = int(source_config.mlp_hidden_size)
            add_copy(
                f"{prefix}.feed_forward.w1.weight",
                destination_layer.w13_weight[:mlp_rows],
                _lookup_state_tensor(state_dict, f"{prefix}.feed_forward.w1.weight"),
            )
            add_copy(
                f"{prefix}.feed_forward.w3.weight",
                destination_layer.w13_weight[mlp_rows:],
                _lookup_state_tensor(state_dict, f"{prefix}.feed_forward.w3.weight"),
            )
            add_copy(
                f"{prefix}.feed_forward.w2.weight",
                destination_layer.w2_weight,
                _lookup_state_tensor(state_dict, f"{prefix}.feed_forward.w2.weight"),
            )

            if layer_type == "full_attention":
                attention = destination_layer.attention
                if attention is None or destination_layer.conv is not None:
                    raise ValueError(f"LFM25 packed layer {layer_idx} does not match full_attention config")
                q_rows = int(source_config.num_attention_heads * source_config.head_dim)
                kv_rows = int(source_config.num_key_value_heads * source_config.head_dim)
                add_copy(
                    f"{prefix}.self_attn.q_proj.weight",
                    attention.qkv_weight[:q_rows],
                    _lookup_state_tensor(state_dict, f"{prefix}.self_attn.q_proj.weight"),
                )
                add_copy(
                    f"{prefix}.self_attn.k_proj.weight",
                    attention.qkv_weight[q_rows : q_rows + kv_rows],
                    _lookup_state_tensor(state_dict, f"{prefix}.self_attn.k_proj.weight"),
                )
                add_copy(
                    f"{prefix}.self_attn.v_proj.weight",
                    attention.qkv_weight[q_rows + kv_rows :],
                    _lookup_state_tensor(state_dict, f"{prefix}.self_attn.v_proj.weight"),
                )
                add_copy(
                    f"{prefix}.self_attn.out_proj.weight",
                    attention.out_weight,
                    _lookup_state_tensor(state_dict, f"{prefix}.self_attn.out_proj.weight"),
                )
                add_copy(
                    f"{prefix}.self_attn.q_layernorm.weight",
                    attention.q_norm_weight,
                    _lookup_state_tensor(state_dict, f"{prefix}.self_attn.q_layernorm.weight"),
                )
                add_copy(
                    f"{prefix}.self_attn.k_layernorm.weight",
                    attention.k_norm_weight,
                    _lookup_state_tensor(state_dict, f"{prefix}.self_attn.k_layernorm.weight"),
                )
            elif layer_type == "conv":
                conv = destination_layer.conv
                if conv is None or destination_layer.attention is not None:
                    raise ValueError(f"LFM25 packed layer {layer_idx} does not match conv config")
                add_copy(
                    f"{prefix}.conv.in_proj.weight",
                    conv.in_weight,
                    _lookup_state_tensor(state_dict, f"{prefix}.conv.in_proj.weight"),
                )
                conv_source = _lookup_state_tensor(state_dict, f"{prefix}.conv.conv.weight")
                expected_conv_shape = (conv.conv_weight.size(0), 1, conv.conv_weight.size(1))
                if not torch.is_tensor(conv_source):
                    raise TypeError(f"LFM25 sync source {prefix + '.conv.conv.weight'!r} is not a tensor")
                if tuple(conv_source.shape) != expected_conv_shape:
                    raise ValueError(
                        f"LFM25 sync shape mismatch for {prefix + '.conv.conv.weight'!r}: "
                        f"expected {expected_conv_shape}, got {tuple(conv_source.shape)}"
                    )
                copies.append((f"{prefix}.conv.conv.weight", conv.conv_weight, conv_source[:, 0, :]))
                add_copy(
                    f"{prefix}.conv.out_proj.weight",
                    conv.out_weight,
                    _lookup_state_tensor(state_dict, f"{prefix}.conv.out_proj.weight"),
                )
            else:
                raise ValueError(f"Unsupported LFM2 layer type during synchronization: {layer_type!r}")

        for _, destination, source in copies:
            destination.copy_(source.detach(), non_blocking=False)


@dataclass(slots=True)
class LFM25HybridCache:
    k_cache: torch.Tensor
    v_cache: torch.Tensor
    conv_state: torch.Tensor
    seq_lens: torch.Tensor
    k_scales: torch.Tensor | None = None
    v_scales: torch.Tensor | None = None

    def rows(self, start: int, end: int) -> "LFM25HybridCache":
        return LFM25HybridCache(
            self.k_cache[:, start:end], self.v_cache[:, start:end],
            self.conv_state[:, start:end], self.seq_lens[start:end],
            None if self.k_scales is None else self.k_scales[:, start:end],
            None if self.v_scales is None else self.v_scales[:, start:end],
        )

    def store_prefill(self, layer: int, k: torch.Tensor, v: torch.Tensor, *, start: int = 0) -> None:
        if k.ndim == 3:
            k, v = k.unsqueeze(0), v.unsqueeze(0)
        end, length = start + k.size(0), k.size(1)
        kc, vc = self.k_cache[layer, start:end, :length], self.v_cache[layer, start:end, :length]
        if self.k_scales is None:
            kc.copy_(k)
            vc.copy_(v)
        else:
            from climbing_gym.inference.lfm25_fp8_kv import store_kv
            store_kv(k, v, kc, vc, self.k_scales[layer, start:end, :length],
                     self.v_scales[layer, start:end, :length])

    def copy_prefix_from(self, source: "LFM25HybridCache", slots: torch.Tensor, length: int) -> None:
        compatible = (self.k_cache.dtype == source.k_cache.dtype
                      and (self.k_scales is None) == (source.k_scales is None))
        for layer in range(self.k_cache.size(0)):
            # Float8 indexing support varies across PyTorch versions. Byte
            # copies also avoid dequantizing and requantizing an exact cache.
            def selected(value):
                part = value[layer, :, :length]
                return (part.view(torch.uint8).index_select(0, slots).view(part.dtype)
                        if part.element_size() == 1 else part.index_select(0, slots))
            k, v = selected(source.k_cache), selected(source.v_cache)
            if compatible:
                self.k_cache[layer, :, :length].copy_(k)
                self.v_cache[layer, :, :length].copy_(v)
                if source.k_scales is not None:
                    self.k_scales[layer, :, :length].copy_(selected(source.k_scales))
                    self.v_scales[layer, :, :length].copy_(selected(source.v_scales))
                continue
            if source.k_scales is not None:
                k = (k.float() * source.k_scales[layer, :, :length].index_select(0, slots)[..., None]).to(self.conv_state.dtype)
                v = (v.float() * source.v_scales[layer, :, :length].index_select(0, slots)[..., None]).to(self.conv_state.dtype)
            self.store_prefill(layer, k, v)
        self.conv_state.copy_(source.conv_state.index_select(1, slots))
        self.seq_lens.copy_(source.seq_lens.index_select(0, slots))

    @property
    def max_slots(self) -> int:
        return int(self.seq_lens.numel())

    @property
    def max_seq_len(self) -> int:
        return int(self.k_cache.size(2))


@dataclass(frozen=True, slots=True)
class LFM25PrefixBatch:
    cache: LFM25HybridCache
    next_token_logits: torch.Tensor
    prompt_lengths: torch.Tensor
    input_ids: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.prompt_lengths.numel())


@dataclass(slots=True)
class LFM25DecodeBatch:
    cache: LFM25HybridCache
    active_slot_ids: torch.Tensor
    prompt_indices: torch.Tensor
    repeat_indices: torch.Tensor
    alive: torch.Tensor

    @property
    def batch_size(self) -> int:
        return int(self.prompt_indices.numel())


@dataclass(frozen=True, slots=True)
class LFM25StepOutput:
    token_ids: torch.Tensor
    raw_logprobs: torch.Tensor | None
    sampling_logprobs: torch.Tensor | None
    finished: torch.Tensor
    finish_reasons: torch.Tensor
    logits: torch.Tensor


@dataclass(frozen=True, slots=True)
class LFM25GenerationOutput:
    token_ids: torch.Tensor
    lengths: torch.Tensor
    logprobs: torch.Tensor | None
    finish_reasons: torch.Tensor
    topk_token_ids: torch.Tensor | None = None
    topk_logprobs: torch.Tensor | None = None


class LFM25Engine:
    """LFM2.5-only in-process inference runtime with dense hybrid caches.

    This is the exact PyTorch reference path for the model-specific runtime: it
    uses packed QKV and MLP weights, consumes token IDs directly, keeps KV and
    short-convolution state in contiguous slot caches, and exposes explicit
    prefill, fork, and decode-step operations.
    """

    def __init__(
        self,
        weights: LFM25PackedWeights,
        *,
        max_model_len: int | None = None,
    ) -> None:
        self.weights = weights
        self.config = weights.config
        self.device = weights.device
        self.dtype = weights.dtype
        default_len = min(int(self.config.max_position_embeddings), 32768)
        self.max_model_len = int(max_model_len or default_len)
        if self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0")
        if self.max_model_len > int(self.config.max_position_embeddings):
            raise ValueError("max_model_len cannot exceed config.max_position_embeddings")
        self._num_query_heads = int(self.config.num_attention_heads)
        self._num_kv_heads = int(self.config.num_key_value_heads)
        self._head_dim = int(self.config.head_dim)
        self._kv_repeats = self._num_query_heads // self._num_kv_heads
        self._scaling = self._head_dim**-0.5
        self._max_vectorized_attention_cells = 131072
        self._flash_decode_min_cache_len = 1
        self._compiled_append_tokens = None
        self._cuda_decoder = None
        self.policy_version = 0
        self._build_rope_cache()

    @classmethod
    def from_pretrained(
        cls,
        model_dir: str | Path,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        max_model_len: int | None = None,
    ) -> "LFM25Engine":
        if cls is LFM25Engine:
            metadata = json.loads((Path(model_dir) / "config.json").read_text())
            if metadata.get("model_type") == "lfm3":
                from climbing_gym.inference.lfm3_engine import LFM3Engine

                return LFM3Engine.from_pretrained(model_dir, device=device, dtype=dtype, max_model_len=max_model_len)
        resolved_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        resolved_dtype = _resolve_dtype(dtype, device=resolved_device)
        weights = LFM25PackedWeights.from_pretrained(
            model_dir,
            device=resolved_device,
            dtype=resolved_dtype,
        )
        return cls(weights, max_model_len=max_model_len)

    @classmethod
    def from_model(
        cls,
        model: Any,
        *,
        device: str | torch.device | None = None,
        dtype: str | torch.dtype | None = None,
        max_model_len: int | None = None,
    ) -> "LFM25Engine":
        first_param = next(model.parameters())
        resolved_device = torch.device(device or first_param.device)
        resolved_dtype = _resolve_dtype(dtype or first_param.dtype, device=resolved_device)
        weights = LFM25PackedWeights.from_model(model, device=resolved_device, dtype=resolved_dtype)
        return cls(weights, max_model_len=max_model_len)

    @torch.inference_mode()
    def sync_from_model(
        self,
        model: Any,
        *,
        drain_timeout: int | float = 300,
    ) -> "LFM25Engine":
        """Synchronize policy weights without replacing packed buffers.

        A bare token engine has no asynchronous work to drain; ``drain_timeout``
        is accepted so it can be used directly by ``RolloutCoordinator``.  The
        async adapter supplies the actual execution/drain lock.
        """

        timeout = float(drain_timeout)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("drain_timeout must be finite and >= 0")
        if self._cuda_decoder is not None and self._cuda_decoder._active_session is not None:
            raise RuntimeError("Close the native generation session before synchronizing weights")
        self.weights.copy_from_model_(model)
        if self._cuda_decoder is not None:
            self._cuda_decoder.refresh_weights()
        self.policy_version += 1
        return self

    def release_inference_buffers(self) -> None:
        """Free completed-session caches and graphs, retaining the actor weights."""
        decoder = self._cuda_decoder
        if decoder is None:
            return
        if decoder._active_session is not None:
            raise RuntimeError("Close the native generation session before releasing inference buffers")
        if not (decoder.graphs or decoder.packed_weights or decoder.fp8_weights):
            return
        torch.cuda.synchronize(self.device)
        decoder.clear()
        torch.cuda.empty_cache()

    def generation_session(self, *, max_batch_size: int, max_new_tokens: int,
                           cache_len: int | None = None, behavior_topk: int = 0,
                           profile_timings: bool = False):
        """Reserve reusable native CUDA state for continuation and slot refill."""
        if self._cuda_decoder is None:
            raise RuntimeError("generation_session requires enable_cuda_graphs()")
        from climbing_gym.inference.lfm25_session import LFM25GenerationSession
        return LFM25GenerationSession(self._cuda_decoder, max_batch_size=max_batch_size,
                                      max_new_tokens=max_new_tokens, cache_len=cache_len,
                                      behavior_topk=behavior_topk, profile_timings=profile_timings)

    @torch.inference_mode()
    def prefill(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        *,
        max_cache_len: int | None = None,
        cache: LFM25HybridCache | None = None,
        max_prefill_tokens: int | None = None,
        logits_buffer: torch.Tensor | None = None,
    ) -> LFM25PrefixBatch:
        input_ids = input_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        cu_seqlens = cu_seqlens.to(device=self.device, dtype=torch.long).reshape(-1)
        if cu_seqlens.numel() < 2:
            raise ValueError("cu_seqlens must contain at least [0, total_tokens]")
        if int(cu_seqlens[0].item()) != 0:
            raise ValueError("cu_seqlens must start at 0")
        if int(cu_seqlens[-1].item()) != int(input_ids.numel()):
            raise ValueError("cu_seqlens[-1] must equal input_ids.numel()")
        prompt_lengths = cu_seqlens[1:] - cu_seqlens[:-1]
        if torch.any(prompt_lengths <= 0):
            raise ValueError("empty prompts are not supported")
        if int(prompt_lengths.max().item()) > int(max_seqlen):
            raise ValueError("max_seqlen is smaller than at least one prompt")

        num_prompts = int(prompt_lengths.numel())
        cache_len = int(max_cache_len or max_seqlen)
        if cache is None:
            cache = self._empty_cache(max_slots=num_prompts, max_seq_len=cache_len)
        elif cache.max_slots != num_prompts or cache.max_seq_len < cache_len:
            raise ValueError("destination cache does not fit the prompt batch")
        if max_prefill_tokens is not None and max_prefill_tokens <= 0:
            raise ValueError("max_prefill_tokens must be positive")
        logits_shape = (num_prompts, self.config.vocab_size)
        if logits_buffer is None:
            next_logits = torch.empty(logits_shape, device=self.device, dtype=self.dtype)
        else:
            if logits_buffer.shape != logits_shape or logits_buffer.dtype != self.dtype or logits_buffer.device != self.weights.embed_tokens.device:
                raise ValueError("logits buffer must match the prompt batch, vocabulary, device, and model dtype")
            next_logits = logits_buffer

        if max_prefill_tokens is not None and num_prompts > 1:
            lengths_cpu = prompt_lengths.tolist()
            start = 0
            while start < num_prompts:
                end, padded_length = start, 0
                while end < num_prompts:
                    candidate = max(padded_length, lengths_cpu[end])
                    if end > start and (end - start + 1) * candidate > max_prefill_tokens:
                        break
                    padded_length = candidate
                    end += 1
                chunk_cache = cache.rows(start, end)
                lengths = prompt_lengths[start:end]
                offsets = torch.arange(padded_length, device=self.device)[None, :]
                indices = cu_seqlens[start:end, None] + offsets
                padded = input_ids[indices.clamp_max(input_ids.numel() - 1)]
                padded = torch.where(offsets < lengths[:, None], padded, 0)
                logits = self._prefill_equal_length_prompts(chunk_cache, padded, prompt_lengths=lengths)
                next_logits[start:end].copy_(logits)
                start = end
        elif num_prompts > 1 and bool(torch.all(prompt_lengths == prompt_lengths[0]).item()):
            logits = self._prefill_equal_length_prompts(
                cache,
                input_ids.view(num_prompts, int(prompt_lengths[0].item())),
            )
            next_logits.copy_(logits)
        else:
            for prompt_idx in range(num_prompts):
                start = int(cu_seqlens[prompt_idx].item())
                end = int(cu_seqlens[prompt_idx + 1].item())
                logits = self._prefill_prompt(cache, prompt_idx, input_ids[start:end])
                next_logits[prompt_idx].copy_(logits[0])

        return LFM25PrefixBatch(
            cache=cache,
            next_token_logits=next_logits,
            prompt_lengths=prompt_lengths.to(dtype=torch.int32),
            input_ids=input_ids,
        )

    @torch.inference_mode()
    def fork(
        self,
        prefix: LFM25PrefixBatch,
        repeats: int,
        *,
        max_cache_len: int | None = None,
    ) -> LFM25DecodeBatch:
        if repeats <= 0:
            raise ValueError("repeats must be > 0")
        num_prompts = prefix.batch_size
        total_slots = num_prompts * int(repeats)
        cache_len = int(max_cache_len or prefix.cache.max_seq_len)
        if cache_len < prefix.cache.max_seq_len:
            raise ValueError("max_cache_len cannot be smaller than the prefix cache")
        cache = self._empty_cache(max_slots=total_slots, max_seq_len=cache_len)
        source_slots = torch.arange(num_prompts, device=self.device, dtype=torch.long).repeat_interleave(repeats)

        cache.copy_prefix_from(prefix.cache, source_slots, prefix.cache.max_seq_len)

        return LFM25DecodeBatch(
            cache=cache,
            active_slot_ids=torch.arange(total_slots, device=self.device, dtype=torch.long),
            prompt_indices=torch.arange(num_prompts, device=self.device, dtype=torch.long).repeat_interleave(repeats),
            repeat_indices=torch.arange(repeats, device=self.device, dtype=torch.long).repeat(num_prompts),
            alive=torch.ones(total_slots, device=self.device, dtype=torch.bool),
        )

    @torch.inference_mode()
    def decode_step(
        self,
        state: LFM25DecodeBatch,
        input_tokens: torch.Tensor,
        sampling: LFM25SamplingConfig | None = None,
    ) -> LFM25StepOutput:
        sampling = sampling or LFM25SamplingConfig()
        active_slot_ids = state.active_slot_ids.to(device=self.device, dtype=torch.long)
        input_tokens = input_tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        if input_tokens.numel() != active_slot_ids.numel():
            raise ValueError("input_tokens must have one token per active slot")
        logits = self._append_decode_tokens(state.cache, active_slot_ids, input_tokens)
        generator = _make_generator(self.device, sampling.seed)
        next_tokens, raw_logprobs, sampling_logprobs = self._sample(logits, sampling, generator=generator)
        finish_reasons = self._finish_reasons(next_tokens, sampling)
        finished = finish_reasons != 0
        return LFM25StepOutput(
            token_ids=next_tokens,
            raw_logprobs=raw_logprobs if sampling.return_logprobs else None,
            sampling_logprobs=sampling_logprobs if sampling.return_logprobs else None,
            finished=finished,
            finish_reasons=finish_reasons,
            logits=logits,
        )

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        repeats: int,
        sampling: LFM25SamplingConfig | None = None,
    ) -> LFM25GenerationOutput:
        sampling = sampling or LFM25SamplingConfig()
        if self._cuda_decoder is not None:
            return self._cuda_decoder.generate(input_ids, cu_seqlens, max_seqlen, repeats, sampling)
        if sampling.behavior_topk:
            raise ValueError('behavior_topk requires native CUDA graphs')
        max_cache_len = int(max_seqlen) + int(sampling.max_new_tokens)
        prefix = self.prefill(
            input_ids=input_ids,
            cu_seqlens=cu_seqlens,
            max_seqlen=max_seqlen,
            max_cache_len=max_cache_len,
        )
        state = self.fork(prefix, repeats=repeats, max_cache_len=max_cache_len)
        flat_batch = state.batch_size
        generator = _make_generator(self.device, sampling.seed)
        seen_token_mask = self._build_seen_token_mask(prefix, repeats, sampling)
        if sampling.ignore_eos and not sampling.stop_token_ids and not sampling.stop_token_sequences and sampling.stop_table is None:
            next_logits = prefix.next_token_logits.repeat_interleave(repeats, dim=0)
            return self._generate_fixed_length(
                prefix,
                state,
                next_logits,
                repeats,
                sampling,
                generator=generator,
                seen_token_mask=seen_token_mask,
            )

        token_ids = torch.zeros(
            (flat_batch, sampling.max_new_tokens),
            device=self.device,
            dtype=torch.long,
        )
        lengths = torch.zeros(flat_batch, device=self.device, dtype=torch.int32)
        finish_reasons = torch.zeros(flat_batch, device=self.device, dtype=torch.int8)
        logprobs = (
            torch.zeros((flat_batch, sampling.max_new_tokens), device=self.device, dtype=torch.float32)
            if sampling.return_logprobs
            else None
        )

        next_logits = prefix.next_token_logits.repeat_interleave(repeats, dim=0)
        next_tokens, raw_logprobs, _ = self._sample(
            next_logits,
            sampling,
            generator=generator,
            seen_token_mask=seen_token_mask,
        )
        alive = torch.ones(flat_batch, device=self.device, dtype=torch.bool)
        stop_patterns = [torch.tensor(s, device=self.device) for s in sampling.stop_token_sequences]
        stop_states = torch.zeros(flat_batch, device=self.device, dtype=torch.long)

        for step_idx in range(sampling.max_new_tokens):
            active_indices = alive.nonzero(as_tuple=False).squeeze(-1)
            if active_indices.numel() == 0:
                break

            current_tokens = next_tokens.index_select(0, active_indices)
            if seen_token_mask is not None:
                seen_token_mask[active_indices, current_tokens] = True
            token_ids[active_indices, step_idx] = current_tokens
            lengths[active_indices] = step_idx + 1
            if logprobs is not None and raw_logprobs is not None:
                logprobs[active_indices, step_idx] = raw_logprobs.index_select(0, active_indices)

            current_reasons = self._finish_reasons(current_tokens, sampling)
            if sampling.stop_table is not None:
                from climbing_gym.inference.text_stops import advance_text_stops
                next_states, current_reasons = advance_text_stops(sampling.stop_table,
                    stop_states[active_indices], current_tokens, current_reasons)
                stop_states[active_indices] = next_states
            if stop_patterns:
                current_reasons = apply_sequence_stops(
                    token_ids.index_select(0, active_indices)[:, :step_idx + 1], current_reasons, stop_patterns)
            just_finished = current_reasons != 0
            if torch.any(just_finished):
                finished_indices = active_indices.index_select(0, just_finished.nonzero(as_tuple=False).squeeze(-1))
                finish_reasons[finished_indices] = current_reasons[just_finished]
                alive[finished_indices] = False

            if step_idx == sampling.max_new_tokens - 1:
                break

            continue_indices = alive.nonzero(as_tuple=False).squeeze(-1)
            if continue_indices.numel() == 0:
                break
            continue_tokens = next_tokens.index_select(0, continue_indices)
            continue_slots = continue_indices.to(device=self.device, dtype=torch.long)
            logits = self._append_decode_tokens(state.cache, continue_slots, continue_tokens)
            current_seen_token_mask = None
            if seen_token_mask is not None:
                current_seen_token_mask = (
                    seen_token_mask
                    if continue_indices.numel() == flat_batch
                    else seen_token_mask.index_select(0, continue_indices)
                )
            sampled, sampled_raw_logprobs, _ = self._sample(
                logits,
                sampling,
                generator=generator,
                seen_token_mask=current_seen_token_mask,
            )
            next_tokens = next_tokens.clone()
            next_tokens[continue_indices] = sampled
            if raw_logprobs is not None and sampled_raw_logprobs is not None:
                raw_logprobs = raw_logprobs.clone()
                raw_logprobs[continue_indices] = sampled_raw_logprobs

        unfinished = finish_reasons == 0
        finish_reasons[unfinished] = 3
        shaped = (prefix.batch_size, repeats)
        return LFM25GenerationOutput(
            token_ids=token_ids.view(*shaped, sampling.max_new_tokens),
            lengths=lengths.view(*shaped),
            logprobs=None if logprobs is None else logprobs.view(*shaped, sampling.max_new_tokens),
            finish_reasons=finish_reasons.view(*shaped),
        )

    def _generate_fixed_length(
        self,
        prefix: LFM25PrefixBatch,
        state: LFM25DecodeBatch,
        next_logits: torch.Tensor,
        repeats: int,
        sampling: LFM25SamplingConfig,
        *,
        generator: torch.Generator,
        seen_token_mask: torch.Tensor | None,
    ) -> LFM25GenerationOutput:
        flat_batch = state.batch_size
        token_ids = torch.empty(
            (flat_batch, sampling.max_new_tokens),
            device=self.device,
            dtype=torch.long,
        )
        lengths = torch.full(
            (flat_batch,),
            int(sampling.max_new_tokens),
            device=self.device,
            dtype=torch.int32,
        )
        finish_reasons = torch.full((flat_batch,), 3, device=self.device, dtype=torch.int8)
        logprobs = (
            torch.empty((flat_batch, sampling.max_new_tokens), device=self.device, dtype=torch.float32)
            if sampling.return_logprobs
            else None
        )
        slot_ids = state.active_slot_ids
        next_tokens, raw_logprobs, _ = self._sample(
            next_logits,
            sampling,
            generator=generator,
            seen_token_mask=seen_token_mask,
        )

        for step_idx in range(sampling.max_new_tokens):
            token_ids[:, step_idx] = next_tokens
            if seen_token_mask is not None:
                seen_token_mask[slot_ids, next_tokens] = True
            if logprobs is not None and raw_logprobs is not None:
                logprobs[:, step_idx] = raw_logprobs
            if step_idx == sampling.max_new_tokens - 1:
                break
            logits = self._append_decode_tokens(
                state.cache,
                slot_ids,
                next_tokens,
                slots_are_prefix=True,
            )
            next_tokens, raw_logprobs, _ = self._sample(
                logits,
                sampling,
                generator=generator,
                seen_token_mask=seen_token_mask,
            )

        shaped = (prefix.batch_size, repeats)
        return LFM25GenerationOutput(
            token_ids=token_ids.view(*shaped, sampling.max_new_tokens),
            lengths=lengths.view(*shaped),
            logprobs=None if logprobs is None else logprobs.view(*shaped, sampling.max_new_tokens),
            finish_reasons=finish_reasons.view(*shaped),
        )

    def _build_seen_token_mask(
        self,
        prefix: LFM25PrefixBatch,
        repeats: int,
        sampling: LFM25SamplingConfig,
    ) -> torch.Tensor | None:
        if float(sampling.repetition_penalty) == 1.0:
            return None
        mask = torch.zeros(
            (prefix.batch_size * repeats, self.config.vocab_size),
            device=self.device,
            dtype=torch.bool,
        )
        offsets = torch.cat(
            (
                torch.zeros(1, device=self.device, dtype=torch.long),
                prefix.prompt_lengths.to(dtype=torch.long).cumsum(dim=0),
            )
        ).tolist()
        for prompt_index, (start, end) in enumerate(zip(offsets[:-1], offsets[1:], strict=True)):
            prompt_tokens = torch.unique(prefix.input_ids[int(start) : int(end)])
            repeat_rows = torch.arange(
                prompt_index * repeats,
                (prompt_index + 1) * repeats,
                device=self.device,
                dtype=torch.long,
            )
            mask[repeat_rows[:, None], prompt_tokens[None, :]] = True
        return mask

    def reset(self, state: LFM25DecodeBatch | LFM25PrefixBatch) -> None:
        state.cache.k_cache.zero_()
        state.cache.v_cache.zero_()
        state.cache.conv_state.zero_()
        state.cache.seq_lens.zero_()
        if isinstance(state, LFM25DecodeBatch):
            state.alive.fill_(True)
            state.active_slot_ids.copy_(torch.arange(state.batch_size, device=self.device, dtype=torch.long))

    def compile_decode(self, *, mode: str = "reduce-overhead", fullgraph: bool = False) -> None:
        self._compiled_append_tokens = torch.compile(
            self._append_tokens,
            mode=mode,
            fullgraph=fullgraph,
            dynamic=False,
        )

    def enable_cuda_graphs(self, *, max_graphs: int = 2, bf16_weight_packing: bool = True,
                           max_prefill_tokens: int = 4096, fp8_mlp: bool = False,
                           fp8_kv: bool = False, adaptive_batching: bool = False,
                           fp8_lm_head: bool = False) -> None:
        """Accelerate generation while retaining the prefill/fork/decode API.

        Captures are lazy and cached by batch size and context capacity. Weights
        are shared with this engine, including in-place training weight sync.
        Requires Triton and FlashAttention; CPU/reference paths are unchanged.
        """
        from climbing_gym.inference.lfm25_cuda import LFM25CUDADecoder

        if self._cuda_decoder is not None and self._cuda_decoder._active_session is not None:
            raise RuntimeError("Close the native generation session before reconfiguring CUDA graphs")
        self._cuda_decoder = LFM25CUDADecoder(self, max_graphs=max_graphs, bf16_weight_packing=bf16_weight_packing,
                                             max_prefill_tokens=max_prefill_tokens, fp8_mlp=fp8_mlp,
                                             fp8_kv=fp8_kv, adaptive_batching=adaptive_batching,
                                             fp8_lm_head=fp8_lm_head)

    def disable_cuda_graphs(self) -> None:
        if self._cuda_decoder is not None:
            self._cuda_decoder.clear()
        self._cuda_decoder = None

    def _build_rope_cache(self) -> None:
        inv_freq = 1.0 / (
            self.config.rope_theta
            ** (torch.arange(0, self._head_dim, 2, dtype=torch.float32, device=self.device) / self._head_dim)
        )
        positions = torch.arange(self.max_model_len, dtype=torch.float32, device=self.device)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self._rope_cos = freqs.cos()
        self._rope_sin = freqs.sin()

    def _prefill_prompt(
        self,
        cache: LFM25HybridCache,
        slot_idx: int,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        seqlen = int(input_ids.numel())
        if seqlen <= 0:
            raise ValueError("empty prompts are not supported")
        if seqlen > cache.max_seq_len:
            raise RuntimeError("cache capacity exceeded")
        if seqlen > self.max_model_len:
            raise RuntimeError("position exceeds engine max_model_len")

        positions = torch.arange(seqlen, device=self.device, dtype=torch.long)
        hidden = F.embedding(input_ids, self.weights.embed_tokens)
        for layer in self.weights.layers:
            residual = hidden
            operator_input = _rms_norm(hidden, layer.operator_norm_weight, self.config.norm_eps)
            if layer.attention is not None:
                operator_output = self._attention_prefill_layer(layer.attention, cache, slot_idx, positions, operator_input)
            elif layer.conv is not None:
                operator_output = self._conv_prefill_layer(layer.conv, cache, slot_idx, operator_input)
            else:
                raise RuntimeError("malformed LFM2 layer: neither attention nor conv")

            hidden = residual + operator_output
            ffn_input = _rms_norm(hidden, layer.ffn_norm_weight, self.config.norm_eps)
            w1, w3 = self._reference_mlp_projections(ffn_input, layer.w13_weight)
            hidden = hidden + F.linear(F.silu(w1) * w3, layer.w2_weight)

        cache.seq_lens[slot_idx] = seqlen
        hidden = _rms_norm(hidden[-1:], self.weights.embedding_norm_weight, self.config.norm_eps)
        return F.linear(hidden, self.weights.lm_head_weight)

    def _prefill_equal_length_prompts(
        self,
        cache: LFM25HybridCache,
        input_ids: torch.Tensor,
        *,
        prompt_lengths: torch.Tensor | None = None,
        return_hidden: bool = False,
    ) -> torch.Tensor:
        input_ids = input_ids.to(device=self.device, dtype=torch.long)
        batch_size = int(input_ids.size(0))
        seqlen = int(input_ids.size(1))
        if batch_size <= 0 or seqlen <= 0:
            raise ValueError("empty prompts are not supported")
        if batch_size > cache.max_slots:
            raise RuntimeError("cache slot capacity exceeded")
        if seqlen > cache.max_seq_len:
            raise RuntimeError("cache capacity exceeded")
        if seqlen > self.max_model_len:
            raise RuntimeError("position exceeds engine max_model_len")

        positions = torch.arange(seqlen, device=self.device, dtype=torch.long)
        hidden = F.embedding(input_ids, self.weights.embed_tokens)
        for layer in self.weights.layers:
            residual = hidden
            operator_input = _rms_norm(hidden, layer.operator_norm_weight, self.config.norm_eps)
            if layer.attention is not None:
                operator_output = self._attention_prefill_equal_length_layer(
                    layer.attention,
                    cache,
                    positions,
                    operator_input,
                )
            elif layer.conv is not None:
                operator_output = self._conv_prefill_equal_length_layer(
                    layer.conv,
                    cache,
                    operator_input,
                    prompt_lengths=prompt_lengths,
                )
            else:
                raise RuntimeError("malformed LFM2 layer: neither attention nor conv")

            hidden = residual + operator_output
            ffn_input = _rms_norm(hidden, layer.ffn_norm_weight, self.config.norm_eps)
            w1, w3 = self._reference_mlp_projections(ffn_input, layer.w13_weight)
            hidden = hidden + F.linear(F.silu(w1) * w3, layer.w2_weight)

        if return_hidden:
            return _rms_norm(hidden, self.weights.embedding_norm_weight, self.config.norm_eps)
        if prompt_lengths is None:
            cache.seq_lens[:batch_size] = seqlen
            last_hidden = hidden[:, -1]
        else:
            cache.seq_lens[:batch_size].copy_(prompt_lengths)
            last_hidden = hidden[torch.arange(batch_size, device=self.device), prompt_lengths - 1]
        hidden = _rms_norm(last_hidden, self.weights.embedding_norm_weight, self.config.norm_eps)
        return F.linear(hidden, self.weights.lm_head_weight)

    @torch.inference_mode()
    def score_completions(
        self,
        prompts: Sequence[Sequence[int]],
        completions: Sequence[Sequence[int]],
        *,
        max_prefill_tokens: int = 4096,
        head_chunk_size: int = 256,
    ) -> list[tuple[float, ...]]:
        """Raw, untempered completion log probabilities using native prefill.

        Includes every supplied completion token (including EOS), excludes the
        prompt, and does not consume sampling RNG or modify session caches.
        The chunked head avoids sequence-by-vocabulary logit allocations.
        """
        if len(prompts) != len(completions):
            raise ValueError('prompts and completions must have equal lengths')
        if max_prefill_tokens <= 0 or head_chunk_size <= 0:
            raise ValueError('scoring chunk sizes must be positive')
        sequences = [list(p) + list(c) for p, c in zip(prompts, completions)]
        if any(not p or not c for p, c in zip(prompts, completions)):
            raise ValueError('scoring requires nonempty prompts and completions')
        if any(len(s) > self.max_model_len for s in sequences):
            raise ValueError('scored sequence exceeds max_model_len')
        result: list[tuple[float, ...]] = [()] * len(sequences)
        order = sorted(range(len(sequences)), key=lambda i: len(sequences[i]))
        start = 0
        while start < len(order):
            end = start + 1
            while end < len(order) and (end + 1 - start) * len(sequences[order[end]]) <= max_prefill_tokens:
                end += 1
            indices = order[start:end]
            length = max(len(sequences[i]) for i in indices)
            ids = torch.zeros((len(indices), length), device=self.device, dtype=torch.long)
            for row, i in enumerate(indices):
                ids[row, :len(sequences[i])] = torch.tensor(sequences[i], device=self.device)
            lengths = torch.tensor([len(sequences[i]) for i in indices], device=self.device, dtype=torch.int32)
            cache = self._empty_cache(max_slots=len(indices), max_seq_len=length)
            hidden = self._prefill_equal_length_prompts(cache, ids, prompt_lengths=lengths, return_hidden=True)
            selected = torch.cat([hidden[row, len(prompts[i])-1:len(sequences[i])-1]
                                  for row, i in enumerate(indices)])
            targets = torch.tensor([t for i in indices for t in completions[i]], device=self.device, dtype=torch.long)
            logps = []
            for offset in range(0, len(targets), head_chunk_size):
                logits = F.linear(selected[offset:offset+head_chunk_size], self.weights.lm_head_weight).float()
                target = targets[offset:offset+head_chunk_size]
                logps.append(logits.gather(1, target[:, None]).squeeze(1) - torch.logsumexp(logits, dim=-1))
            flat = torch.cat(logps).cpu().tolist()
            cursor = 0
            for i in indices:
                size = len(completions[i])
                result[i] = tuple(flat[cursor:cursor+size])
                cursor += size
            del cache, hidden, selected, logits
            start = end
        return result

    def _empty_cache(self, *, max_slots: int, max_seq_len: int, fp8_kv: bool = False) -> LFM25HybridCache:
        if max_slots <= 0:
            raise ValueError("max_slots must be > 0")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be > 0")
        if max_seq_len > self.max_model_len:
            raise ValueError(f"max_seq_len={max_seq_len} exceeds engine max_model_len={self.max_model_len}")

        if fp8_kv:
            # Keep each head's compressed history contiguous. The logical
            # [layers, batch, tokens, heads, dim] view preserves prefill APIs.
            shape = (self.weights.num_attention_layers, max_slots, self._num_kv_heads, max_seq_len)
            return LFM25HybridCache(
                torch.zeros((*shape, self._head_dim), device=self.device, dtype=torch.float8_e4m3fn).permute(0, 1, 3, 2, 4),
                torch.zeros((*shape, self._head_dim), device=self.device, dtype=torch.float8_e4m3fn).permute(0, 1, 3, 2, 4),
                torch.zeros((self.weights.num_conv_layers, max_slots, self.config.hidden_size, self.config.conv_kernel_size),
                            device=self.device, dtype=self.dtype),
                torch.zeros(max_slots, device=self.device, dtype=torch.int32),
                torch.zeros(shape, device=self.device, dtype=torch.float32).permute(0, 1, 3, 2),
                torch.zeros(shape, device=self.device, dtype=torch.float32).permute(0, 1, 3, 2),
            )

        return LFM25HybridCache(
            k_cache=torch.zeros(
                (
                    self.weights.num_attention_layers,
                    max_slots,
                    max_seq_len,
                    self._num_kv_heads,
                    self._head_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            v_cache=torch.zeros(
                (
                    self.weights.num_attention_layers,
                    max_slots,
                    max_seq_len,
                    self._num_kv_heads,
                    self._head_dim,
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            conv_state=torch.zeros(
                (
                    self.weights.num_conv_layers,
                    max_slots,
                    self.config.hidden_size,
                    self.config.conv_kernel_size,
                ),
                device=self.device,
                dtype=self.dtype,
            ),
            seq_lens=torch.zeros(max_slots, device=self.device, dtype=torch.int32),
        )

    def _append_tokens(
        self,
        cache: LFM25HybridCache,
        slot_ids: torch.Tensor,
        input_tokens: torch.Tensor,
        *,
        compute_logits: bool = True,
        slots_are_prefix: bool = False,
    ) -> torch.Tensor | None:
        slot_ids = slot_ids.to(device=self.device, dtype=torch.long).reshape(-1)
        input_tokens = input_tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        if slot_ids.numel() != input_tokens.numel():
            raise ValueError("slot_ids and input_tokens must have matching lengths")
        if torch.any(slot_ids < 0) or torch.any(slot_ids >= cache.max_slots):
            raise ValueError("slot_ids contain an out-of-range cache slot")

        positions = cache.seq_lens.index_select(0, slot_ids).to(torch.long)
        if torch.any(positions >= cache.max_seq_len):
            raise RuntimeError("cache capacity exceeded")
        if torch.any(positions >= self.max_model_len):
            raise RuntimeError("position exceeds engine max_model_len")

        hidden = F.embedding(input_tokens, self.weights.embed_tokens)
        for layer in self.weights.layers:
            residual = hidden
            operator_input = _rms_norm(hidden, layer.operator_norm_weight, self.config.norm_eps)
            if layer.attention is not None:
                operator_output = self._attention_layer(
                    layer.attention,
                    cache,
                    slot_ids,
                    positions,
                    operator_input,
                    slots_are_prefix=slots_are_prefix,
                )
            elif layer.conv is not None:
                operator_output = self._conv_layer(
                    layer.conv,
                    cache,
                    slot_ids,
                    operator_input,
                    slots_are_prefix=slots_are_prefix,
                )
            else:
                raise RuntimeError("malformed LFM2 layer: neither attention nor conv")

            hidden = residual + operator_output
            ffn_input = _rms_norm(hidden, layer.ffn_norm_weight, self.config.norm_eps)
            w1, w3 = self._reference_mlp_projections(ffn_input, layer.w13_weight)
            hidden = hidden + F.linear(F.silu(w1) * w3, layer.w2_weight)

        cache.seq_lens[slot_ids] += 1
        if not compute_logits:
            return None
        hidden = _rms_norm(hidden, self.weights.embedding_norm_weight, self.config.norm_eps)
        return F.linear(hidden, self.weights.lm_head_weight)

    def _append_decode_tokens(
        self,
        cache: LFM25HybridCache,
        slot_ids: torch.Tensor,
        input_tokens: torch.Tensor,
        *,
        slots_are_prefix: bool = False,
    ) -> torch.Tensor:
        if self._compiled_append_tokens is not None:
            return self._compiled_append_tokens(
                cache,
                slot_ids,
                input_tokens,
                slots_are_prefix=slots_are_prefix,
            )
        logits = self._append_tokens(cache, slot_ids, input_tokens, slots_are_prefix=slots_are_prefix)
        if logits is None:
            raise RuntimeError("decode append must compute logits")
        return logits

    def _reference_mlp_projections(self, hidden, weight):
        if hidden.dtype == torch.bfloat16:
            w1, w3 = weight.chunk(2, dim=0)
            return F.linear(hidden, w1), F.linear(hidden, w3)
        return F.linear(hidden, weight).chunk(2, dim=-1)

    def _attention_layer(
        self,
        weights: LFM25AttentionWeights,
        cache: LFM25HybridCache,
        slot_ids: torch.Tensor,
        positions: torch.Tensor,
        hidden: torch.Tensor,
        *,
        slots_are_prefix: bool = False,
    ) -> torch.Tensor:
        q_end = self.config.hidden_size
        kv_width = self._num_kv_heads * self._head_dim
        if hidden.dtype == torch.bfloat16:
            qkv = torch.cat([F.linear(hidden, weight) for weight in weights.qkv_weight.split(
                (q_end, kv_width, kv_width), dim=0,
            )], dim=-1)
        else:
            qkv = F.linear(hidden, weights.qkv_weight)
        q = qkv[:, :q_end].view(-1, self._num_query_heads, self._head_dim)
        k = qkv[:, q_end : q_end + kv_width].view(-1, self._num_kv_heads, self._head_dim)
        v = qkv[:, q_end + kv_width :].view(-1, self._num_kv_heads, self._head_dim)
        q = _rms_norm(q, weights.q_norm_weight, self.config.norm_eps)
        k = _rms_norm(k, weights.k_norm_weight, self.config.norm_eps)
        q, k = self._apply_rope(q, k, positions)

        layer_k_cache = cache.k_cache[weights.cache_index]
        layer_v_cache = cache.v_cache[weights.cache_index]
        if (
            flash_attn_with_kvcache is not None
            and q.is_cuda
            and q.dtype in (torch.float16, torch.bfloat16)
            and int(layer_k_cache.size(1)) >= self._flash_decode_min_cache_len
        ):
            attended = flash_attn_with_kvcache(
                q.unsqueeze(1),
                layer_k_cache,
                layer_v_cache,
                k=k.unsqueeze(1),
                v=v.unsqueeze(1),
                cache_seqlens=positions.to(dtype=torch.int32),
                cache_batch_idx=slot_ids.to(dtype=torch.int32),
                softmax_scale=self._scaling,
                causal=True,
            ).squeeze(1)
            return F.linear(attended.reshape(q.size(0), -1), weights.out_weight)

        layer_k_cache[slot_ids, positions] = k
        layer_v_cache[slot_ids, positions] = v
        attended = self._attend_cached(
            q,
            layer_k_cache,
            layer_v_cache,
            slot_ids,
            positions,
            slots_are_prefix=slots_are_prefix,
        )
        return F.linear(attended, weights.out_weight)

    def _attention_prefill_layer(
        self,
        weights: LFM25AttentionWeights,
        cache: LFM25HybridCache,
        slot_idx: int,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        q_end = self.config.hidden_size
        kv_width = self._num_kv_heads * self._head_dim
        # Match the checkpoint implementation's projection shapes. cuBLAS can
        # choose a different BF16 reduction for a packed QKV matrix, which
        # perturbs cached keys throughout a long prompt.
        q = F.linear(hidden, weights.qkv_weight[:q_end]).view(-1, self._num_query_heads, self._head_dim)
        k = F.linear(hidden, weights.qkv_weight[q_end:q_end + kv_width]).view(-1, self._num_kv_heads, self._head_dim)
        v = F.linear(hidden, weights.qkv_weight[q_end + kv_width:]).view(-1, self._num_kv_heads, self._head_dim)
        q = _rms_norm(q, weights.q_norm_weight, self.config.norm_eps)
        k = _rms_norm(k, weights.k_norm_weight, self.config.norm_eps)
        q, k = self._apply_rope(q, k, positions)

        seqlen = int(hidden.size(0))
        cache.store_prefill(weights.cache_index, k, v, start=slot_idx)

        if flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            attended = flash_attn_func(q[None], k[None], v[None], causal=True, softmax_scale=self._scaling)
            return F.linear(attended.reshape(seqlen, -1), weights.out_weight)
        qh = q.permute(1, 0, 2).unsqueeze(0)
        kh = k.repeat_interleave(self._kv_repeats, dim=1).permute(1, 0, 2).unsqueeze(0)
        vh = v.repeat_interleave(self._kv_repeats, dim=1).permute(1, 0, 2).unsqueeze(0)
        attended = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        return F.linear(attended.squeeze(0).permute(1, 0, 2).reshape(seqlen, -1), weights.out_weight)

    def _attention_prefill_equal_length_layer(
        self,
        weights: LFM25AttentionWeights,
        cache: LFM25HybridCache,
        positions: torch.Tensor,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, seqlen = int(hidden.size(0)), int(hidden.size(1))
        q_end = self.config.hidden_size
        kv_width = self._num_kv_heads * self._head_dim
        q = F.linear(hidden, weights.qkv_weight[:q_end]).view(batch_size, seqlen, self._num_query_heads, self._head_dim)
        k = F.linear(hidden, weights.qkv_weight[q_end:q_end + kv_width]).view(batch_size, seqlen, self._num_kv_heads, self._head_dim)
        v = F.linear(hidden, weights.qkv_weight[q_end + kv_width:]).view(batch_size, seqlen, self._num_kv_heads, self._head_dim)
        q = _rms_norm(q, weights.q_norm_weight, self.config.norm_eps)
        k = _rms_norm(k, weights.k_norm_weight, self.config.norm_eps)
        q, k = self._apply_rope_prefill(q, k, positions)

        cache.store_prefill(weights.cache_index, k, v)

        if flash_attn_func is not None and q.is_cuda and q.dtype in (torch.float16, torch.bfloat16):
            attended = flash_attn_func(q, k, v, causal=True, softmax_scale=self._scaling)
            return F.linear(attended.reshape(batch_size, seqlen, -1), weights.out_weight)
        qh = q.transpose(1, 2)
        kh = k.repeat_interleave(self._kv_repeats, dim=2).transpose(1, 2)
        vh = v.repeat_interleave(self._kv_repeats, dim=2).transpose(1, 2)
        attended = F.scaled_dot_product_attention(qh, kh, vh, is_causal=True)
        return F.linear(attended.transpose(1, 2).contiguous().view(batch_size, seqlen, -1), weights.out_weight)

    def _conv_layer(
        self,
        weights: LFM25ConvWeights,
        cache: LFM25HybridCache,
        slot_ids: torch.Tensor,
        hidden: torch.Tensor,
        *,
        slots_are_prefix: bool = False,
    ) -> torch.Tensor:
        bcx = F.linear(hidden, weights.in_weight)
        b_gate, c_gate, x_proj = bcx.chunk(3, dim=-1)
        bx = b_gate * x_proj
        layer_state = cache.conv_state[weights.cache_index]
        if slots_are_prefix and self.config.conv_kernel_size == 3:
            state = layer_state[: bx.size(0)]
            state[:, :, 0].copy_(state[:, :, 1])
            state[:, :, 1].copy_(state[:, :, 2])
            state[:, :, 2].copy_(bx)
            conv_out = (state * weights.conv_weight.unsqueeze(0)).sum(dim=-1)
            return F.linear(c_gate * conv_out, weights.out_weight)
        state = layer_state.index_select(0, slot_ids)
        state = torch.roll(state, shifts=-1, dims=-1)
        state[:, :, -1] = bx
        layer_state[slot_ids] = state
        conv_out = (state * weights.conv_weight.unsqueeze(0)).sum(dim=-1)
        return F.linear(c_gate * conv_out, weights.out_weight)

    def _conv_prefill_layer(
        self,
        weights: LFM25ConvWeights,
        cache: LFM25HybridCache,
        slot_idx: int,
        hidden: torch.Tensor,
    ) -> torch.Tensor:
        seqlen = int(hidden.size(0))
        bcx = F.linear(hidden, weights.in_weight)
        b_gate, c_gate, x_proj = bcx.chunk(3, dim=-1)
        bx = b_gate * x_proj
        conv_in = bx.transpose(0, 1).unsqueeze(0)
        conv_weight = weights.conv_weight.unsqueeze(1)
        conv_out = F.conv1d(
            conv_in,
            conv_weight,
            padding=int(self.config.conv_kernel_size) - 1,
            groups=self.config.hidden_size,
        )[..., :seqlen]
        conv_out = conv_out.squeeze(0).transpose(0, 1).contiguous()

        state = cache.conv_state[weights.cache_index, slot_idx]
        state.zero_()
        keep = min(int(self.config.conv_kernel_size), seqlen)
        state[:, -keep:].copy_(bx[-keep:].transpose(0, 1))
        return F.linear(c_gate * conv_out, weights.out_weight)

    def _conv_prefill_equal_length_layer(
        self,
        weights: LFM25ConvWeights,
        cache: LFM25HybridCache,
        hidden: torch.Tensor,
        *,
        prompt_lengths: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch_size = int(hidden.size(0))
        seqlen = int(hidden.size(1))
        bcx = F.linear(hidden, weights.in_weight)
        b_gate, c_gate, x_proj = bcx.chunk(3, dim=-1)
        bx = b_gate * x_proj
        conv_in = bx.transpose(1, 2)
        conv_weight = weights.conv_weight.unsqueeze(1)
        conv_out = F.conv1d(
            conv_in,
            conv_weight,
            padding=int(self.config.conv_kernel_size) - 1,
            groups=self.config.hidden_size,
        )[..., :seqlen]
        conv_out = conv_out.transpose(1, 2).contiguous()

        state = cache.conv_state[weights.cache_index, :batch_size]
        state.zero_()
        if prompt_lengths is None:
            keep = min(int(self.config.conv_kernel_size), seqlen)
            state[:, :, -keep:].copy_(bx[:, -keep:].transpose(1, 2))
        else:
            indices = prompt_lengths[:, None] + torch.arange(-self.config.conv_kernel_size, 0, device=self.device)[None, :]
            history = bx.gather(1, indices.clamp_min(0)[:, :, None].expand(-1, -1, self.config.hidden_size))
            history = torch.where(indices[:, :, None] >= 0, history, 0)
            state.copy_(history.transpose(1, 2))
        return F.linear(c_gate * conv_out, weights.out_weight)

    def _apply_rope(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self._rope_cos.index_select(0, positions).unsqueeze(1)
        sin = self._rope_sin.index_select(0, positions).unsqueeze(1)
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)

    def _apply_rope_prefill(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self._rope_cos.index_select(0, positions).view(1, -1, 1, self._head_dim // 2)
        sin = self._rope_sin.index_select(0, positions).view(1, -1, 1, self._head_dim // 2)
        return _apply_rotary(q, cos, sin), _apply_rotary(k, cos, sin)

    def _attend_cached(
        self,
        q: torch.Tensor,
        k_cache: torch.Tensor,
        v_cache: torch.Tensor,
        slot_ids: torch.Tensor,
        positions: torch.Tensor,
        *,
        slots_are_prefix: bool = False,
    ) -> torch.Tensor:
        cache_seqlen = int(k_cache.size(1))
        max_seqlen = cache_seqlen
        if q.size(0) * max_seqlen > self._max_vectorized_attention_cells:
            max_seqlen = int(positions.max().item()) + 1
        if q.size(0) >= 8 and q.size(0) * max_seqlen <= self._max_vectorized_attention_cells:
            if slots_are_prefix:
                k_hist = k_cache[: q.size(0), :max_seqlen]
                v_hist = v_cache[: q.size(0), :max_seqlen]
            else:
                k_hist = k_cache.index_select(0, slot_ids)[:, :max_seqlen]
                v_hist = v_cache.index_select(0, slot_ids)[:, :max_seqlen]
            k_hist = k_hist.repeat_interleave(self._kv_repeats, dim=2).permute(0, 2, 1, 3)
            v_hist = v_hist.repeat_interleave(self._kv_repeats, dim=2).permute(0, 2, 1, 3)
            scores = torch.einsum("bhd,bhtd->bht", q, k_hist).float() * self._scaling
            valid = torch.arange(max_seqlen, device=q.device).view(1, 1, -1) <= positions.view(-1, 1, 1)
            scores = scores.masked_fill(~valid, torch.finfo(scores.dtype).min)
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            return torch.einsum("bht,bhtd->bhd", probs, v_hist).reshape(q.size(0), -1)

        outputs: list[torch.Tensor] = []
        for row_idx in range(q.size(0)):
            slot = int(slot_ids[row_idx].item())
            seqlen = int(positions[row_idx].item()) + 1
            k_hist = k_cache[slot, :seqlen].repeat_interleave(self._kv_repeats, dim=1).permute(1, 0, 2)
            v_hist = v_cache[slot, :seqlen].repeat_interleave(self._kv_repeats, dim=1).permute(1, 0, 2)
            scores = torch.einsum("hd,htd->ht", q[row_idx], k_hist) * self._scaling
            probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=q.dtype)
            out = torch.einsum("ht,htd->hd", probs, v_hist).reshape(-1)
            outputs.append(out)
        return torch.stack(outputs, dim=0)

    def _sample(
        self,
        logits: torch.Tensor,
        sampling: LFM25SamplingConfig,
        *,
        generator: torch.Generator,
        seen_token_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        if sampling.temperature == 0.0 and not sampling.return_logprobs and (
            seen_token_mask is None or sampling.repetition_penalty == 1.0
        ):
            return logits.argmax(dim=-1), None, None
        logits_fp32 = _apply_repetition_penalty(
            logits.float(),
            seen_token_mask=seen_token_mask,
            penalty=float(sampling.repetition_penalty),
        )
        if sampling.temperature == 0.0:
            token_ids = torch.argmax(logits_fp32, dim=-1)
            if not sampling.return_logprobs:
                return token_ids.to(torch.long), None, None
            selected_logits = logits_fp32.gather(1, token_ids[:, None]).squeeze(1)
            selected_raw = selected_logits - torch.logsumexp(logits_fp32, dim=-1)
            return token_ids.to(torch.long), selected_raw, selected_raw

        sample_logits = logits_fp32 / float(sampling.temperature)
        filtered_logits = _apply_top_k_top_p(sample_logits, top_k=sampling.top_k, top_p=sampling.top_p)
        probs = torch.softmax(filtered_logits, dim=-1)
        token_ids = torch.multinomial(probs, num_samples=1, generator=generator).squeeze(1)
        if not sampling.return_logprobs:
            return token_ids.to(torch.long), None, None
        selected_raw_logits = logits_fp32.gather(1, token_ids[:, None]).squeeze(1)
        selected_sampling_logits = filtered_logits.gather(1, token_ids[:, None]).squeeze(1)
        selected_raw = selected_raw_logits - torch.logsumexp(logits_fp32, dim=-1)
        selected_sampling = selected_sampling_logits - torch.logsumexp(filtered_logits, dim=-1)
        return token_ids.to(torch.long), selected_raw, selected_sampling

    def _finish_reasons(self, token_ids: torch.Tensor, sampling: LFM25SamplingConfig) -> torch.Tensor:
        reasons = torch.zeros_like(token_ids, dtype=torch.int8, device=self.device)
        if not sampling.ignore_eos:
            reasons[token_ids == int(self.config.eos_token_id)] = 1
        for stop_id in sampling.stop_token_ids:
            reasons[(reasons == 0) & (token_ids == int(stop_id))] = 2
        return reasons


def apply_sequence_stops(history, reasons, patterns):
    """Match generated-token suffixes, retaining stop tokens and their logprobs."""
    for pattern in patterns:
        if history.shape[1] >= pattern.numel():
            match = (history[:, -pattern.numel():] == pattern).all(dim=1)
            reasons = torch.where((reasons == 0) & match, 2, reasons)
    return reasons


def _load_state_dict(model_dir: Path) -> dict[str, torch.Tensor]:
    safetensors = sorted(model_dir.glob("*.safetensors"))
    if safetensors:
        return load_file(safetensors[0], device="cpu")
    pts = sorted(model_dir.glob("*.pt"))
    bins = sorted(model_dir.glob("*.bin"))
    if pts or bins:
        payload = torch.load((pts or bins)[0], map_location="cpu", weights_only=False)
        if isinstance(payload, dict) and isinstance(payload.get("state_dict"), dict):
            return payload["state_dict"]
        if isinstance(payload, dict) and isinstance(payload.get("model"), dict):
            return payload["model"]
        if isinstance(payload, dict):
            return payload
    raise FileNotFoundError(f"No safetensors, .pt, or .bin weights found in: {model_dir}")


_SYNC_CONFIG_FIELDS = (
    "vocab_size",
    "hidden_size",
    "mlp_hidden_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "max_position_embeddings",
    "rope_theta",
    "norm_eps",
    "conv_kernel_size",
    "conv_bias",
    "layer_types",
    "tie_embedding",
    "pad_token_id",
    "bos_token_id",
    "eos_token_id",
)


def _coerce_lfm2_config(config: Any) -> LFM2Config:
    """Normalize checkpoint and model configurations into the native schema."""

    if isinstance(config, LFM2Config):
        return config
    if config is None:
        raise TypeError("LFM25 weight synchronization requires a model with an LFM2 config")

    to_dict = getattr(config, "to_dict", None)
    if not callable(to_dict):
        raise TypeError(
            "LFM25 weight synchronization requires an LFM2 config exposing to_dict()"
        )
    payload = to_dict()
    if not isinstance(payload, Mapping):
        raise TypeError("LFM2 config.to_dict() must return a mapping")
    return LFM2Config.from_dict(dict(payload))


def _validate_sync_config(expected: LFM2Config, source: LFM2Config) -> None:
    source.validate()
    mismatches = [
        f"{field}: engine={getattr(expected, field)!r}, model={getattr(source, field)!r}"
        for field in _SYNC_CONFIG_FIELDS
        if getattr(expected, field) != getattr(source, field)
    ]
    if mismatches:
        raise ValueError(
            "Cannot synchronize an incompatible LFM2 config ("
            + "; ".join(mismatches)
            + ")"
        )


def _lookup_state_tensor(state_dict: Mapping[str, torch.Tensor], key: str) -> torch.Tensor:
    candidates = [key]
    if key.startswith("model."):
        candidates.append(key.removeprefix("model."))
    else:
        candidates.append(f"model.{key}")
    for candidate in candidates:
        value = state_dict.get(candidate)
        if value is not None:
            return value
    raise KeyError(f"Missing tensor {key!r} in LFM2 checkpoint")


def _has_state_tensor(state_dict: Mapping[str, torch.Tensor], key: str) -> bool:
    if key in state_dict:
        return True
    if key.startswith("model."):
        return key.removeprefix("model.") in state_dict
    return f"model.{key}" in state_dict


def _resolve_dtype(dtype: str | torch.dtype | None, *, device: torch.device) -> torch.dtype:
    if isinstance(dtype, torch.dtype):
        return dtype
    if dtype is None or str(dtype).lower() == "auto":
        return torch.bfloat16 if device.type == "cuda" else torch.float32
    normalized = str(dtype).strip().lower()
    mapping = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return mapping[normalized]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype: {dtype!r}") from exc


def _rms_norm(hidden: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    input_dtype = hidden.dtype
    normalized = hidden.float()
    variance = normalized.pow(2).mean(dim=-1, keepdim=True)
    normalized = normalized * torch.rsqrt(variance + eps)
    return (normalized.to(input_dtype) * weight).to(input_dtype)


def _apply_rotary(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    # Match Transformers' model-dtype coefficients and intermediate products.
    # Keeping these in FP32 changes BF16 attention appreciably at long contexts.
    x1, x2 = torch.chunk(x, 2, dim=-1)
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


def _apply_top_k_top_p(logits: torch.Tensor, *, top_k: int, top_p: float) -> torch.Tensor:
    filtered = logits
    if top_k > 0 and top_k < logits.size(-1):
        threshold = torch.topk(filtered, k=top_k, dim=-1).values[:, -1, None]
        filtered = filtered.masked_fill(filtered < threshold, -math.inf)

    if top_p < 1.0:
        sorted_logits, sorted_indices = torch.sort(filtered, descending=True, dim=-1)
        sorted_probs = torch.softmax(sorted_logits, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[:, 0] = False
        sorted_logits = sorted_logits.masked_fill(remove, -math.inf)
        restored = torch.full_like(filtered, -math.inf)
        filtered = restored.scatter(1, sorted_indices, sorted_logits)
    return filtered


def _apply_repetition_penalty(
    logits: torch.Tensor,
    *,
    seen_token_mask: torch.Tensor | None,
    penalty: float,
) -> torch.Tensor:
    if seen_token_mask is None or penalty == 1.0:
        return logits
    if penalty <= 0.0:
        raise ValueError("repetition penalty must be > 0")
    if seen_token_mask.shape != logits.shape:
        raise ValueError("seen_token_mask must have the same shape as logits")
    penalized = torch.where(logits < 0, logits * penalty, logits / penalty)
    return torch.where(seen_token_mask, penalized, logits)


def _make_generator(device: torch.device, seed: int) -> torch.Generator:
    generator = torch.Generator(device=device.type if device.type == "cuda" else "cpu")
    generator.manual_seed(int(seed))
    return generator


def flatten_token_batches(input_ids: Sequence[Sequence[int]], *, device: str | torch.device | None = None) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Convert a Python batch of token IDs into the packed prefill format."""

    if not input_ids:
        raise ValueError("input_ids must contain at least one prompt")
    resolved_device = torch.device(device or "cpu")
    lengths = [len(item) for item in input_ids]
    if any(length <= 0 for length in lengths):
        raise ValueError("empty prompts are not supported")
    flat = torch.tensor([token for item in input_ids for token in item], device=resolved_device, dtype=torch.long)
    cu = torch.zeros(len(lengths) + 1, device=resolved_device, dtype=torch.int32)
    cu[1:] = torch.tensor(lengths, device=resolved_device, dtype=torch.int32).cumsum(0)
    return flat, cu, max(lengths)


def describe_lfm25_checkpoint(model_dir: str | Path) -> dict[str, object]:
    config = LFM2Config.from_pretrained(model_dir)
    attention_layers = [idx for idx, kind in enumerate(config.layer_types) if kind == "full_attention"]
    conv_layers = [idx for idx, kind in enumerate(config.layer_types) if kind == "conv"]
    kv_bytes_per_token = len(attention_layers) * 2 * config.num_key_value_heads * config.head_dim * 2
    conv_state_bytes_per_sequence = len(conv_layers) * config.hidden_size * config.conv_kernel_size * 2
    return {
        "model_dir": str(Path(model_dir).expanduser().resolve()),
        "num_hidden_layers": config.num_hidden_layers,
        "attention_layers": attention_layers,
        "conv_layers": conv_layers,
        "hidden_size": config.hidden_size,
        "num_attention_heads": config.num_attention_heads,
        "num_key_value_heads": config.num_key_value_heads,
        "head_dim": config.head_dim,
        "mlp_hidden_size": config.mlp_hidden_size,
        "conv_kernel_size": config.conv_kernel_size,
        "vocab_size": config.vocab_size,
        "tie_embedding": config.tie_embedding,
        "max_position_embeddings": config.max_position_embeddings,
        "kv_bytes_per_token_per_sequence_bf16": kv_bytes_per_token,
        "conv_state_bytes_per_sequence_bf16": conv_state_bytes_per_sequence,
    }


def write_checkpoint_description(model_dir: str | Path, output_path: str | Path) -> Path:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(describe_lfm25_checkpoint(model_dir), indent=2) + "\n", encoding="utf-8")
    return path
