"""Native LFM3 runtime using the existing hybrid scheduler and E4M3 kernels.

No Transformers model code is loaded or executed. Prefill uses BF16 GEMMs;
decode optionally uses dynamically scaled W8A8 projections and FP8 KV storage.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F

from climbing_gym.inference.lfm25_engine import (
    LFM25ConvWeights, LFM25Engine, _load_state_dict, _resolve_dtype, _rms_norm,
)


@dataclass(frozen=True)
class LFM3Config:
    vocab_size: int
    hidden_size: int
    intermediate_size: int
    num_hidden_layers: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    layer_types: tuple[str, ...]
    max_position_embeddings: int = 262144
    norm_eps: float = 1e-5
    conv_kernel_size: int = 7
    conv_bias: bool = False
    sliding_window: int = 256
    eos_token_id: int = 124900
    pad_token_id: int = 124893
    full_rope_theta: float = 1000000.0
    full_rotary_factor: float = 0.5
    sliding_rope_theta: float = 10000.0
    sliding_rotary_factor: float = 1.0

    @classmethod
    def from_pretrained(cls, path):
        data = json.loads((Path(path) / 'config.json').read_text())
        if data.get('model_type') != 'lfm3':
            raise ValueError('Expected an LFM3 checkpoint')
        if data.get('quantization_config'):
            raise ValueError('Load the BF16 LFM3 checkpoint; native FP8 packing is performed on device')
        if data.get('conv_bias', False) or not data.get('use_sandwich_norm', False):
            raise ValueError('Native LFM3 requires bias-free convolutions and sandwich normalization')
        rope = data['rope_parameters']
        if any(p.get('rope_type', 'default') != 'default' for p in rope.values()):
            raise ValueError('Only default LFM3 RoPE is supported')
        keys = ('vocab_size', 'hidden_size', 'intermediate_size', 'num_hidden_layers',
                'num_attention_heads', 'num_key_value_heads', 'head_dim', 'max_position_embeddings',
                'norm_eps', 'sliding_window', 'eos_token_id', 'pad_token_id')
        cfg = cls(**{k: data[k] for k in keys}, layer_types=tuple(data['layer_types']),
                  conv_kernel_size=data['conv_L_cache'],
                  full_rope_theta=rope['full_attention']['rope_theta'],
                  full_rotary_factor=rope['full_attention'].get('partial_rotary_factor', 1.),
                  sliding_rope_theta=rope['sliding_attention']['rope_theta'],
                  sliding_rotary_factor=rope['sliding_attention'].get('partial_rotary_factor', 1.))
        cfg.validate()
        return cfg

    def validate(self):
        if len(self.layer_types) != self.num_hidden_layers or set(self.layer_types) - {'conv', 'full_attention', 'sliding_attention'}:
            raise ValueError('Invalid LFM3 layer types')
        if self.num_attention_heads % self.num_key_value_heads or self.head_dim % 2:
            raise ValueError('Invalid attention head dimensions')
        if self.hidden_size != self.num_attention_heads * self.head_dim:
            raise ValueError('Native LFM3 requires hidden_size == num_attention_heads * head_dim')
        if self.conv_kernel_size != 7 or self.sliding_window <= 0:
            raise ValueError('Expected width-7 convolutions and a positive sliding window')
        if self.conv_bias or not 0 < self.full_rotary_factor <= 1 or not 0 < self.sliding_rotary_factor <= 1:
            raise ValueError('Expected bias-free convolutions and rotary factors in (0, 1]')


@dataclass(frozen=True)
class LFM3AttentionWeights:
    qkv_weight: torch.Tensor  # Q, K, V, then one output gate per query head
    out_weight: torch.Tensor
    q_norm_weight: torch.Tensor
    k_norm_weight: torch.Tensor
    cache_index: int
    layer_type: str
    window: int


@dataclass(frozen=True)
class LFM3LayerWeights:
    operator_norm_weight: torch.Tensor
    ffn_norm_weight: torch.Tensor
    post_operator_norm_weight: torch.Tensor
    post_ffn_norm_weight: torch.Tensor
    w13_weight: torch.Tensor
    w2_weight: torch.Tensor
    attention: LFM3AttentionWeights | None = None
    conv: LFM25ConvWeights | None = None


@dataclass(frozen=True)
class LFM3PackedWeights:
    config: LFM3Config
    embed_tokens: torch.Tensor
    embedding_norm_weight: torch.Tensor
    lm_head_weight: torch.Tensor
    layers: tuple[LFM3LayerWeights, ...]
    device: torch.device
    dtype: torch.dtype

    @property
    def num_attention_layers(self):
        return sum(layer.attention is not None for layer in self.layers)

    @property
    def num_conv_layers(self):
        return sum(layer.conv is not None for layer in self.layers)

    @classmethod
    def from_state_dict(cls, cfg, state, *, device, dtype):
        cfg.validate()
        def get(key):
            return state[key].detach().to(device=device, dtype=dtype).contiguous()
        layers = []
        ai = ci = 0
        for i, kind in enumerate(cfg.layer_types):
            p = f'model.layers.{i}'
            common = dict(operator_norm_weight=get(p+'.operator_norm.weight'), ffn_norm_weight=get(p+'.ffn_norm.weight'),
                          post_operator_norm_weight=get(p+'.post_operator_norm.weight'),
                          post_ffn_norm_weight=get(p+'.post_ffn_norm.weight'),
                          w13_weight=torch.cat([get(p+'.feed_forward.w1.weight'), get(p+'.feed_forward.w3.weight')]),
                          w2_weight=get(p+'.feed_forward.w2.weight'))
            if kind == 'conv':
                conv = LFM25ConvWeights(get(p+'.conv.in_proj.weight'), get(p+'.conv.conv.weight')[:, 0].contiguous(),
                                       get(p+'.conv.out_proj.weight'), ci)
                layers.append(LFM3LayerWeights(**common, conv=conv))
                ci += 1
            else:
                attn = LFM3AttentionWeights(
                    torch.cat([get(p+f'.self_attn.{name}_proj.weight') for name in ('q', 'k', 'v', 'gate')]),
                    get(p+'.self_attn.o_proj.weight'), get(p+'.self_attn.q_norm.weight'), get(p+'.self_attn.k_norm.weight'),
                    ai, kind, cfg.sliding_window if kind == 'sliding_attention' else 0)
                layers.append(LFM3LayerWeights(**common, attention=attn))
                ai += 1
        embedding = get('model.embed_tokens.weight')
        head = get('lm_head.weight') if 'lm_head.weight' in state else embedding
        return cls(cfg, embedding, get('model.embedding_norm.weight'), head, tuple(layers), torch.device(device), dtype)


class LFM3Engine(LFM25Engine):
    """LFM3 architecture with the native prefill, generation, and session API."""

    @classmethod
    def from_pretrained(cls, model_dir, *, device=None, dtype=None, max_model_len=None):
        device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        dtype = _resolve_dtype(dtype, device=device)
        cfg = LFM3Config.from_pretrained(model_dir)
        weights = LFM3PackedWeights.from_state_dict(cfg, _load_state_dict(Path(model_dir)), device=device, dtype=dtype)
        return cls(weights, max_model_len=max_model_len)

    @classmethod
    def from_model(cls, *args, **kwargs):
        raise NotImplementedError('Load LFM3 inference weights from a local checkpoint with from_pretrained')

    def _build_rope_cache(self):
        self._rope_by_type = {}
        for kind, theta, factor in (
            ('full_attention', self.config.full_rope_theta, self.config.full_rotary_factor),
            ('sliding_attention', self.config.sliding_rope_theta, self.config.sliding_rotary_factor)):
            # Keep the full-head frequency progression; zero frequencies in the tail.
            inv = theta ** (-torch.arange(0, self._head_dim, 2, device=self.device, dtype=torch.float32) / self._head_dim)
            inv[int(factor * self._head_dim) // 2:] = 0
            angles = torch.arange(self.max_model_len, device=self.device, dtype=torch.float32)[:, None] * inv
            self._rope_by_type[kind] = (angles.cos(), angles.sin())

    def enable_cuda_graphs(self, *, max_graphs=1, bf16_weight_packing=False, max_prefill_tokens=4096,
                           fp8_mlp=False, fp8_kv=False, adaptive_batching=False, fp8_lm_head=False,
                           fp8_operators=False):
        from climbing_gym.inference.lfm3_cuda import LFM3CUDADecoder
        if self._cuda_decoder is not None and self._cuda_decoder._active_session is not None:
            raise RuntimeError('Close the native generation session before reconfiguring CUDA graphs')
        if self._cuda_decoder is not None:
            self.disable_cuda_graphs()
        self._cuda_decoder = LFM3CUDADecoder(
            self, max_graphs=max_graphs, bf16_weight_packing=bf16_weight_packing,
            max_prefill_tokens=max_prefill_tokens, fp8_mlp=fp8_mlp, fp8_kv=fp8_kv,
            adaptive_batching=adaptive_batching, fp8_lm_head=fp8_lm_head, fp8_operators=fp8_operators)

    def _prefill_prompt(self, cache, slot_idx, input_ids):
        return self._prefill_equal_length_prompts(cache.rows(slot_idx, slot_idx+1), input_ids.reshape(1, -1))

    def _empty_cache(self, *, max_slots, max_seq_len, fp8_kv=None):
        if fp8_kv is None:
            fp8_kv = self._cuda_decoder is not None and self._cuda_decoder.fp8_kv
        return super()._empty_cache(max_slots=max_slots, max_seq_len=max_seq_len, fp8_kv=fp8_kv)

    def _prefill_equal_length_prompts(self, cache, input_ids, *, prompt_lengths=None, return_hidden=False):
        from climbing_gym.inference.lfm25_engine import flash_attn_func
        b, s = input_ids.shape
        if s > cache.max_seq_len or s > self.max_model_len:
            raise RuntimeError('cache capacity exceeded')
        positions = torch.arange(s, device=self.device)
        hidden = F.embedding(input_ids, self.weights.embed_tokens)
        cfg = self.config
        lengths = torch.full((b,), s, device=self.device, dtype=torch.long) if prompt_lengths is None else prompt_lengths.long()
        for layer in self.weights.layers:
            op = _rms_norm(hidden, layer.operator_norm_weight, cfg.norm_eps)
            if layer.attention is not None:
                a = layer.attention
                qkv = F.linear(op, a.qkv_weight)
                q, k, v, gate = qkv.split((cfg.hidden_size, self._num_kv_heads*self._head_dim,
                                          self._num_kv_heads*self._head_dim, self._num_query_heads), dim=-1)
                q = _rms_norm(q.reshape(b, s, self._num_query_heads, self._head_dim), a.q_norm_weight, cfg.norm_eps)
                k = _rms_norm(k.reshape(b, s, self._num_kv_heads, self._head_dim), a.k_norm_weight, cfg.norm_eps)
                v = v.reshape(b, s, self._num_kv_heads, self._head_dim).contiguous()
                cos, sin = (x[positions].to(self.dtype)[None, :, None, :].repeat(1, 1, 1, 2).float()
                            for x in self._rope_by_type[a.layer_type])
                def rotate(x):
                    left, right = x.float().chunk(2, dim=-1)
                    return (x.float()*cos + torch.cat((-right, left), dim=-1)*sin).to(self.dtype)
                q, k = rotate(q), rotate(k)
                cache.store_prefill(a.cache_index, k, v)
                window = (a.window - 1, 0) if a.window else (-1, -1)
                if hidden.is_cuda and flash_attn_func is not None:
                    attended = flash_attn_func(q, k, v, softmax_scale=self._scaling, causal=True, window_size=window)
                else:
                    causal = positions[:, None] >= positions[None, :]
                    if a.window:
                        causal &= positions[:, None] - positions[None, :] < a.window
                    attended = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                                              attn_mask=causal, enable_gqa=True).transpose(1, 2)
                out = F.linear((attended * gate.sigmoid()[..., None]).reshape(b, s, cfg.hidden_size), a.out_weight)
            else:
                c = layer.conv
                bp, cp, xp = F.linear(op, c.in_weight).chunk(3, dim=-1)
                bx = (bp * xp).transpose(1, 2).contiguous()
                convolved = F.conv1d(bx, c.conv_weight[:, None], padding=cfg.conv_kernel_size-1,
                                     groups=cfg.hidden_size)[..., :s]
                out = F.linear(cp * convolved.transpose(1, 2), c.out_weight)
                offsets = lengths[:, None] - cfg.conv_kernel_size + torch.arange(cfg.conv_kernel_size, device=self.device)
                state = bx.gather(2, offsets.clamp_min(0)[:, None, :].expand(-1, cfg.hidden_size, -1))
                cache.conv_state[c.cache_index].copy_(state * (offsets >= 0)[:, None, :])
            hidden = hidden + _rms_norm(out, layer.post_operator_norm_weight, cfg.norm_eps)
            ffn = _rms_norm(hidden, layer.ffn_norm_weight, cfg.norm_eps)
            gate, up = F.linear(ffn, layer.w13_weight).chunk(2, dim=-1)
            out = F.linear(F.silu(gate) * up, layer.w2_weight)
            hidden = hidden + _rms_norm(out, layer.post_ffn_norm_weight, cfg.norm_eps)
        if return_hidden:
            return _rms_norm(hidden, self.weights.embedding_norm_weight, cfg.norm_eps)
        cache.seq_lens.copy_(lengths)
        last = hidden[torch.arange(b, device=self.device), lengths-1]
        return F.linear(_rms_norm(last, self.weights.embedding_norm_weight, cfg.norm_eps), self.weights.lm_head_weight)

    def _append_tokens(self, cache, slot_ids, input_tokens, *, compute_logits=True, slots_are_prefix=False):
        from climbing_gym.inference.lfm3_cuda import LFM3CUDADecoder
        if self.device.type != 'cuda':
            raise ValueError('LFM3 cached decode requires CUDA')
        slot_ids = slot_ids.to(device=self.device, dtype=torch.int32).reshape(-1)
        input_tokens = input_tokens.to(device=self.device, dtype=torch.long).reshape(-1)
        if slot_ids.numel() != input_tokens.numel() or bool(((slot_ids < 0) | (slot_ids >= cache.max_slots)).any()):
            raise ValueError('Expected one input token per valid cache slot')
        positions = cache.seq_lens.index_select(0, slot_ids.long())
        if bool((positions >= cache.max_seq_len).any()):
            raise RuntimeError('cache capacity exceeded')
        decoder = self._cuda_decoder or LFM3CUDADecoder(self, bf16_weight_packing=False)
        decoder._prepare_fp8_weights()
        logits = decoder._forward(cache, input_tokens, torch.ones_like(positions, dtype=torch.bool),
                                  slot_ids=slot_ids.int(), positions=positions)
        return logits if compute_logits else None

    def sync_from_model(self, *args, **kwargs):
        raise NotImplementedError('LFM3 training weight synchronization is not implemented')
