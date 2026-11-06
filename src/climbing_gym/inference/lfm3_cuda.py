"""LFM3 CUDA graphs over the native generation/sampling/session scheduler."""
from __future__ import annotations

import torch
import torch.nn.functional as F
from flash_attn import flash_attn_with_kvcache

from climbing_gym.inference.lfm25_cuda import LFM25CUDADecoder
from climbing_gym.inference.lfm25_cuda_kernels import rms_norm, swiglu
from climbing_gym.inference.lfm25_fp8 import PackedFP8Weight, swiglu_fp8
from climbing_gym.inference.lfm3_kernels import conv_update, prepare_qkv, head_gate, sandwich_norm


class LFM3CUDADecoder(LFM25CUDADecoder):
    supported_conv_kernel_sizes = (7,)

    def __init__(self, engine, *, fp8_operators=False, **kwargs):
        self.fp8_operators = fp8_operators
        super().__init__(engine, **kwargs)

    def _prepare_fp8_weights(self):
        super()._prepare_fp8_weights()
        if self.fp8_operators:
            for layer in self.engine.weights.layers:
                op = layer.attention or layer.conv
                matrices = (op.qkv_weight, op.out_weight) if layer.attention else (op.in_weight, op.out_weight)
                for weight in matrices:
                    if id(weight) not in self.fp8_weights:
                        self.fp8_weights[id(weight)] = PackedFP8Weight.from_weight(weight)

    def _project(self, x, weight):
        packed = self.fp8_weights.get(id(weight))
        return packed.linear(x) if packed is not None else self._linear(x, weight)

    def _forward(self, cache, tokens, active=None, *, slot_ids=None, positions=None):
        engine, cfg = self.engine, self.engine.config
        positions = cache.seq_lens if positions is None else positions
        residual = F.embedding(tokens, engine.weights.embed_tokens)
        op_input, _ = rms_norm(residual, engine.weights.layers[0].operator_norm_weight, cfg.norm_eps)
        for index, layer in enumerate(engine.weights.layers):
            if layer.attention is not None:
                a = layer.attention
                packed = self._project(op_input, a.qkv_weight)
                q, k, v = prepare_qkv(packed, a.q_norm_weight, a.k_norm_weight,
                                      *engine._rope_by_type[a.layer_type], positions,
                                      cfg.num_attention_heads, cfg.num_key_value_heads, cfg.head_dim, cfg.norm_eps)
                if self.fp8_kv:
                    from climbing_gym.inference.lfm25_fp8_kv import decode
                    i = a.cache_index
                    attended = decode(q, k, v, cache.k_cache[i], cache.v_cache[i], cache.k_scales[i], cache.v_scales[i],
                                      positions, active=active, scale=engine._scaling, slot_ids=slot_ids, window=a.window)
                elif slot_ids is not None:
                    from climbing_gym.inference.lfm25_batching import append_bf16_kv
                    i = a.cache_index
                    append_bf16_kv(k, v, cache.k_cache[i], cache.v_cache[i], positions, slot_ids, active)
                    attended = flash_attn_with_kvcache(
                        q, cache.k_cache[i], cache.v_cache[i], cache_seqlens=torch.where(active, positions+1, 1),
                        cache_batch_idx=slot_ids.clamp_min(0), softmax_scale=engine._scaling, causal=True,
                        window_size=(a.window-1, 0) if a.window else (-1, -1))
                else:
                    attended = flash_attn_with_kvcache(
                        q, cache.k_cache[a.cache_index], cache.v_cache[a.cache_index], k=k, v=v,
                        cache_seqlens=positions, cache_batch_idx=slot_ids,
                        softmax_scale=engine._scaling, causal=True,
                        window_size=(a.window-1, 0) if a.window else (-1, -1))
                out = self._project(head_gate(attended, packed, cfg.num_attention_heads, cfg.head_dim), a.out_weight)
            else:
                c = layer.conv
                bx = self._project(op_input, c.in_weight)
                out = self._project(conv_update(bx, c.conv_weight, cache.conv_state[c.cache_index], active, slot_ids), c.out_weight)
            ffn_input, residual = sandwich_norm(out, layer.post_operator_norm_weight, residual, layer.ffn_norm_weight, cfg.norm_eps)
            gate_up = self._project(ffn_input, layer.w13_weight)
            if self.fp8_mlp:
                activated, scales = swiglu_fp8(gate_up)
                out = self.fp8_weights[id(layer.w2_weight)].linear_quantized(activated, scales, dtype=engine.dtype)
            else:
                out = self._linear(swiglu(gate_up), layer.w2_weight)
            pre = (engine.weights.layers[index+1].operator_norm_weight if index+1 < len(engine.weights.layers)
                   else engine.weights.embedding_norm_weight)
            op_input, residual = sandwich_norm(out, layer.post_ffn_norm_weight, residual, pre, cfg.norm_eps)
        if slot_ids is None:
            cache.seq_lens.add_(1 if active is None else active.int())
        else:
            from climbing_gym.inference.lfm25_batching import advance_positions
            advance_positions(cache.seq_lens, positions, slot_ids, active)
        return self._project(op_input, engine.weights.lm_head_weight)
