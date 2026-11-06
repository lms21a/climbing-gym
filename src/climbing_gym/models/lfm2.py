from __future__ import annotations

import contextvars
import json
import logging
import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from safetensors.torch import load_file
from torch import nn

from climbing_gym.models.lfm2_compat import normalize_lfm2_config
from climbing_gym.models.config import BaseModelConfig
from climbing_gym.models.kernels.lfm2_fused import (
    LFM2_FUSED_KERNELS_AVAILABLE,
    fused_rms_norm,
    fused_rotary_emb,
    fused_silu_mul,
)
from climbing_gym.models.kernels.lfm2_shortconv import (
    TRITON_AVAILABLE,
    depthwise_causal_conv_k3,
    gated_depthwise_causal_conv_k3,
)

try:
    from flash_attn import flash_attn_varlen_func
except Exception as exc:
    flash_attn_varlen_func = None
    _flash_attn_import_error: Exception | None = exc
else:
    _flash_attn_import_error = None


logger = logging.getLogger(__name__)
_flash_attn_warning_emitted = False


def flash_attention_available() -> bool:
    return flash_attn_varlen_func is not None


def warn_if_flash_attention_unavailable(*, component: str = "LFM2") -> None:
    global _flash_attn_warning_emitted
    if flash_attention_available() or _flash_attn_warning_emitted:
        return
    detail = ""
    if _flash_attn_import_error is not None:
        detail = f" Import error: {type(_flash_attn_import_error).__name__}: {_flash_attn_import_error}"
    message = (
        f"FLASH ATTENTION IS NOT ACTIVE for {component} even though attn_backend='flash'. "
        "The model is falling back to the eager attention implementation, which can severely "
        f"degrade training throughput and memory behavior.{detail}"
    )
    logger.warning(message)
    warnings.warn(message, RuntimeWarning, stacklevel=2)
    _flash_attn_warning_emitted = True


@dataclass
class FlashAttnCtx:
    cu_seqlens: torch.Tensor | None = None
    max_seqlen: int | None = None


_flash_attn_ctx_var = contextvars.ContextVar("flash_attn_ctx", default=FlashAttnCtx())


def get_flash_attn_ctx() -> FlashAttnCtx:
    return _flash_attn_ctx_var.get()


def set_flash_attn_ctx(ctx: FlashAttnCtx) -> None:
    _flash_attn_ctx_var.set(ctx)


def build_flash_attn_ctx_from_positions(positions: torch.Tensor) -> FlashAttnCtx:
    if positions.size(0) != 1:
        raise ValueError("Packed flash-attention context expects batch_size == 1.")
    seqlen = positions.size(1)
    starts = (positions.reshape(-1) == 0).nonzero(as_tuple=False).squeeze(-1)
    ends = torch.cat(
        [
            starts[1:],
            torch.tensor([seqlen], device=positions.device, dtype=starts.dtype),
        ]
    )
    cu_seqlens = torch.cat(
        [
            torch.tensor([0], device=positions.device, dtype=torch.int32),
            ends.to(torch.int32),
        ]
    )
    return FlashAttnCtx(cu_seqlens=cu_seqlens, max_seqlen=int(seqlen))


def _set_flash_attn_ctx_from_positions(positions: torch.Tensor) -> None:
    if positions.size(0) == 1:
        set_flash_attn_ctx(build_flash_attn_ctx_from_positions(positions))
    else:
        set_flash_attn_ctx(FlashAttnCtx())


class FlashAttn(nn.Module):
    def __init__(self, n_qheads: int, n_kvheads: int, head_dim: int):
        super().__init__()
        self.enable_gqa = n_qheads != n_kvheads
        self.n_qheads = n_qheads
        self.n_kvheads = n_kvheads
        self.head_dim = head_dim

    @torch.compiler.disable
    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        bsz, seqlen = q.shape[:2]
        ctx = get_flash_attn_ctx()
        cu_seqlens = ctx.cu_seqlens
        max_seqlen = ctx.max_seqlen
        if cu_seqlens is None or max_seqlen is None:
            cu_seqlens = torch.arange(0, (bsz + 1) * seqlen, seqlen, device=q.device, dtype=torch.int32)
            max_seqlen = seqlen

        q = q.view(-1, self.n_qheads, self.head_dim)
        k = k.view(-1, self.n_kvheads, self.head_dim)
        v = v.view(-1, self.n_kvheads, self.head_dim)

        if flash_attn_varlen_func is not None:
            out = flash_attn_varlen_func(
                q=q,
                k=k,
                v=v,
                causal=True,
                cu_seqlens_q=cu_seqlens,
                cu_seqlens_k=cu_seqlens,
                max_seqlen_q=max_seqlen,
                max_seqlen_k=max_seqlen,
            )
            return out.view(bsz, seqlen, -1)

        qh = q.view(bsz, seqlen, self.n_qheads, self.head_dim).transpose(1, 2)
        kh = k.view(bsz, seqlen, self.n_kvheads, self.head_dim).transpose(1, 2)
        vh = v.view(bsz, seqlen, self.n_kvheads, self.head_dim).transpose(1, 2)
        if self.enable_gqa:
            reps = self.n_qheads // self.n_kvheads
            kh = kh[:, :, None, :, :].expand(bsz, self.n_kvheads, reps, seqlen, self.head_dim).reshape(
                bsz, self.n_qheads, seqlen, self.head_dim
            )
            vh = vh[:, :, None, :, :].expand(bsz, self.n_kvheads, reps, seqlen, self.head_dim).reshape(
                bsz, self.n_qheads, seqlen, self.head_dim
            )

        scores = torch.matmul(qh, kh.transpose(2, 3)) * (self.head_dim ** -0.5)
        if ctx.cu_seqlens is None:
            allowed = torch.tril(torch.ones(seqlen, seqlen, dtype=torch.bool, device=scores.device))
            allowed = allowed.unsqueeze(0).unsqueeze(0)
        else:
            if bsz != 1:
                raise ValueError("Packed flash-attention fallback expects batch_size == 1.")
            cu = cu_seqlens.to(device=scores.device, dtype=torch.long)
            idx = torch.arange(seqlen, device=scores.device, dtype=torch.long)
            seq_ids = torch.empty(seqlen, device=scores.device, dtype=torch.long)
            for i in range(cu.numel() - 1):
                seq_ids[cu[i] : cu[i + 1]] = i
            allowed = ((seq_ids[:, None] == seq_ids[None, :]) & (idx[:, None] >= idx[None, :])).unsqueeze(0).unsqueeze(0)

        scores = scores.masked_fill(~allowed, torch.finfo(scores.dtype).min)
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32).to(dtype=qh.dtype)
        return torch.matmul(probs, vh).transpose(1, 2).contiguous().view(bsz, seqlen, -1)


@dataclass
class LFM2Config(BaseModelConfig):
    vocab_size: int = 65536
    hidden_size: int = 2048
    mlp_hidden_size: int = 8192
    num_hidden_layers: int = 16
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    max_position_embeddings: int = 128000
    rope_theta: float = 1_000_000.0
    norm_eps: float = 1e-5
    conv_kernel_size: int = 3
    conv_bias: bool = False
    layer_types: tuple[str, ...] = (
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
        "full_attention",
        "conv",
    )
    tie_embedding: bool = True
    attn_backend: str = "flash"
    shortconv_backend: str = "torch"
    shortconv_block_t: int = 256
    shortconv_num_warps: int = 4
    fused_backbone_kernels: bool = False
    # Explicit MLP backward recomputes expanded activations, preserving weights.
    recompute_mlp: bool = False
    # Recompute decoder-layer activations during backward to bound accumulated
    # gradient + activation memory; no change to native inference or weights.
    checkpoint_layers: bool = False
    # Experimental: full-model BF16 per-parameter parity gate is not yet met.
    fused_cispo_head: bool = False
    pad_token_id: int = 0
    bos_token_id: int = 1
    eos_token_id: int = 7

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "LFM2Config":
        normalized = normalize_lfm2_config(payload)
        if normalized.get("mlp_hidden_size") is None:
            normalized["mlp_hidden_size"] = _resolve_lfm2_mlp_hidden_size(normalized)
        if normalized.get("conv_kernel_size") is None and normalized.get("conv_L_cache") is not None:
            normalized["conv_kernel_size"] = normalized["conv_L_cache"]
        if normalized.get("shortconv_block_t") is None and normalized.get("shortconv_triton_block_t") is not None:
            normalized["shortconv_block_t"] = normalized["shortconv_triton_block_t"]
        if normalized.get("shortconv_num_warps") is None and normalized.get("shortconv_triton_num_warps") is not None:
            normalized["shortconv_num_warps"] = normalized["shortconv_triton_num_warps"]
        return super().from_dict(normalized)

    def validate(self) -> None:
        if len(self.layer_types) != self.num_hidden_layers:
            raise ValueError("len(layer_types) must match num_hidden_layers")
        if self.hidden_size % self.num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads")
        if self.num_attention_heads % self.num_key_value_heads != 0:
            raise ValueError("num_attention_heads must be divisible by num_key_value_heads")
        if self.mlp_hidden_size <= 0:
            raise ValueError("mlp_hidden_size must be > 0")
        if self.conv_kernel_size <= 0:
            raise ValueError("conv_kernel_size must be > 0")
        if any(layer_type not in {"conv", "full_attention"} for layer_type in self.layer_types):
            raise ValueError("layer_types entries must be either 'conv' or 'full_attention'")
        if self.shortconv_backend not in {"torch", "triton_k3"}:
            raise ValueError("shortconv_backend must be one of {'torch', 'triton_k3'}")
        if self.shortconv_backend == "triton_k3":
            if self.conv_kernel_size != 3:
                raise ValueError("shortconv_backend='triton_k3' requires conv_kernel_size=3")
            if self.shortconv_block_t <= 0:
                raise ValueError("shortconv_block_t must be > 0")
            if self.shortconv_num_warps <= 0:
                raise ValueError("shortconv_num_warps must be > 0")

    @property
    def head_dim(self) -> int:
        return self.hidden_size // self.num_attention_heads


def _resolve_lfm2_mlp_hidden_size(payload: dict[str, Any]) -> int:
    if payload.get("intermediate_size") is None:
        return 8192

    intermediate_size = int(payload["intermediate_size"])
    if not bool(payload.get("block_auto_adjust_ff_dim", True)):
        return intermediate_size

    multiple = int(payload.get("block_multiple_of", 256))
    if multiple <= 0:
        raise ValueError("block_multiple_of must be > 0 when normalizing legacy LFM2 configs")

    adjusted_size = int(2 * intermediate_size / 3)
    adjusted_size = int(float(payload.get("block_ffn_dim_multiplier", 1.0)) * adjusted_size)
    return multiple * ((adjusted_size + multiple - 1) // multiple)


def apply_rotary_emb(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    x1, x2 = torch.chunk(x.float(), 2, dim=-1)
    y1 = x1 * cos - x2 * sin
    y2 = x2 * cos + x1 * sin
    return torch.cat((y1, y2), dim=-1).to(x.dtype)


class LFM2RotaryEmbedding(nn.Module):
    def __init__(
        self,
        head_dim: int,
        max_position_embeddings: int,
        base: float,
        *,
        use_fused_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.use_fused_kernel = bool(use_fused_kernel)
        inv_freq = 1.0 / (base ** (torch.arange(0, head_dim, 2, dtype=torch.float) / head_dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cache", freqs.cos(), persistent=False)
        self.register_buffer("sin_cache", freqs.sin(), persistent=False)

    def forward(self, positions: torch.Tensor, query: torch.Tensor, key: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        cos = self.cos_cache[positions]
        sin = self.sin_cache[positions]
        if (
            self.use_fused_kernel
            and LFM2_FUSED_KERNELS_AVAILABLE
            and query.is_cuda
            and key.is_cuda
        ):
            return fused_rotary_emb(query, key, cos, sin)
        cos = cos.unsqueeze(2)
        sin = sin.unsqueeze(2)
        return apply_rotary_emb(query, cos, sin), apply_rotary_emb(key, cos, sin)


@lru_cache(2)
def get_rope(
    head_dim: int,
    max_position: int,
    base: float,
    use_fused_kernel: bool = False,
) -> LFM2RotaryEmbedding:
    return LFM2RotaryEmbedding(
        head_dim,
        max_position,
        base,
        use_fused_kernel=use_fused_kernel,
    )


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    if n_rep == 1:
        return hidden_states
    bsz, n_kv_heads, seqlen, head_dim = hidden_states.shape
    hidden_states = hidden_states[:, :, None, :, :].expand(bsz, n_kv_heads, n_rep, seqlen, head_dim)
    return hidden_states.reshape(bsz, n_kv_heads * n_rep, seqlen, head_dim)


def build_packed_causal_mask(positions: torch.Tensor) -> torch.Tensor:
    seq_start = positions.eq(0).to(torch.long)
    segment_ids = torch.cumsum(seq_start, dim=1) - 1
    same_segment = segment_ids.unsqueeze(2) == segment_ids.unsqueeze(1)
    causal = positions.unsqueeze(2) >= positions.unsqueeze(1)
    return (same_segment & causal).unsqueeze(1)


class LFM2RMSNorm(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-6,
        *,
        use_fused_kernel: bool = False,
    ) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        self.use_fused_kernel = bool(use_fused_kernel)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        if (
            self.use_fused_kernel
            and LFM2_FUSED_KERNELS_AVAILABLE
            and hidden_states.is_cuda
        ):
            return fused_rms_norm(
                hidden_states,
                self.weight,
                eps=self.variance_epsilon,
            )
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


class LFM2MLP(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        self.use_fused_kernel = bool(config.fused_backbone_kernels)
        self.recompute = bool(config.recompute_mlp)
        self.w1 = nn.Linear(config.hidden_size, config.mlp_hidden_size, bias=False)
        self.w3 = nn.Linear(config.hidden_size, config.mlp_hidden_size, bias=False)
        self.w2 = nn.Linear(config.mlp_hidden_size, config.hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.recompute and torch.is_grad_enabled():
            from climbing_gym.models.kernels.mlp_recompute import recomputed_swiglu_mlp
            return recomputed_swiglu_mlp(x, self.w1.weight, self.w3.weight, self.w2.weight,
                                        fused=self.use_fused_kernel and LFM2_FUSED_KERNELS_AVAILABLE)
        gate = self.w1(x)
        up = self.w3(x)
        if (
            self.use_fused_kernel
            and LFM2_FUSED_KERNELS_AVAILABLE
            and gate.is_cuda
        ):
            hidden = fused_silu_mul(gate, up)
        else:
            hidden = F.silu(gate) * up
        return self.w2(hidden)


class LFM2Attention(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.num_attention_heads = config.num_attention_heads
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        self.scaling = self.head_dim ** -0.5
        self.q_proj = nn.Linear(config.hidden_size, self.num_attention_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=False)
        self.q_layernorm = LFM2RMSNorm(
            self.head_dim,
            eps=config.norm_eps,
            use_fused_kernel=config.fused_backbone_kernels,
        )
        self.k_layernorm = LFM2RMSNorm(
            self.head_dim,
            eps=config.norm_eps,
            use_fused_kernel=config.fused_backbone_kernels,
        )
        self.rotary_emb = get_rope(
            config.head_dim,
            config.max_position_embeddings,
            config.rope_theta,
            config.fused_backbone_kernels,
        )
        self.attn: FlashAttn | None = None
        if config.attn_backend == "flash":
            if flash_attention_available():
                self.attn = FlashAttn(self.num_attention_heads, self.num_key_value_heads, self.head_dim)
            else:
                warn_if_flash_attention_unavailable(component="LFM2")

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        bsz, seqlen, _ = hidden_states.shape
        q = self.q_proj(hidden_states).view(bsz, seqlen, self.num_attention_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(bsz, seqlen, self.num_key_value_heads, self.head_dim)
        q = self.q_layernorm(q)
        k = self.k_layernorm(k)
        q, k = self.rotary_emb(positions, q, k)
        if self.attn is not None:
            return self.out_proj(self.attn(q, k, v))
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        k = repeat_kv(k, self.num_key_value_groups)
        v = repeat_kv(v, self.num_key_value_groups)
        attn_weights = torch.matmul(q, k.transpose(2, 3)) * self.scaling
        attn_weights = attn_weights.masked_fill(~build_packed_causal_mask(positions), torch.finfo(attn_weights.dtype).min)
        attn_weights = torch.softmax(attn_weights, dim=-1, dtype=torch.float32).to(q.dtype)
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().reshape(bsz, seqlen, -1)
        return self.out_proj(attn_output)


def _replace_packed_shortconv_boundary_outputs(
    conv_out: torch.Tensor,
    *,
    bx: torch.Tensor,
    weight: torch.Tensor,
    bias: torch.Tensor | None,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Replace packed-boundary outputs with an isolation-safe convolution.

    Attention already treats every ``positions == 0`` token as a new packed
    segment. A plain Conv1d does not. Subtracting the cross-boundary terms from
    its output is not sufficient in low precision: those terms have already
    participated in the convolution accumulator, so BF16 rounding can leave a
    residual after cancellation. Instead, recompute only boundary outputs from
    lagged inputs that are masked *before* multiplication and accumulation.
    """

    if positions.ndim != 2 or tuple(positions.shape) != (
        int(bx.size(0)),
        int(bx.size(2)),
    ):
        raise ValueError(
            "short-convolution positions must have shape [batch, sequence_length]"
        )
    kernel_size = int(weight.size(-1))
    if kernel_size <= 1:
        return conv_out

    # Lag zero is valid at every token. Conv1d is cross-correlation, so the
    # current-token coefficient is the final kernel element.
    safe_out = bx * weight[:, 0, kernel_size - 1].view(1, -1, 1)
    absolute_positions = torch.arange(
        bx.size(-1),
        device=positions.device,
        dtype=positions.dtype,
    ).view(1, -1)
    for lag in range(1, kernel_size):
        shifted = F.pad(bx, (lag, 0))[..., : bx.size(-1)]
        coefficient = weight[:, 0, kernel_size - 1 - lag].view(1, -1, 1)
        valid_history = ((positions >= lag) & (absolute_positions >= lag)).unsqueeze(1)
        safe_shifted = torch.where(valid_history, shifted, torch.zeros_like(shifted))
        safe_out = safe_out + safe_shifted * coefficient
    if bias is not None:
        safe_out = safe_out + bias.view(1, -1, 1)

    boundary = (positions < kernel_size - 1).unsqueeze(1)
    return torch.where(boundary, safe_out, conv_out)


class LFM2ShortConv(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.layer_idx = layer_idx
        self.l_cache = config.conv_kernel_size
        self.bias = config.conv_bias
        self.shortconv_backend = config.shortconv_backend
        self.triton_block_t = config.shortconv_block_t
        self.triton_num_warps = config.shortconv_num_warps
        self.use_fused_gates = bool(config.fused_backbone_kernels)
        if self.shortconv_backend == "triton_k3" and not TRITON_AVAILABLE:
            raise RuntimeError("shortconv_backend='triton_k3' was requested but Triton is not available.")
        self.conv = nn.Conv1d(
            in_channels=config.hidden_size,
            out_channels=config.hidden_size,
            kernel_size=self.l_cache,
            groups=config.hidden_size,
            bias=self.bias,
            padding=self.l_cache - 1,
        )
        self.in_proj = nn.Linear(config.hidden_size, 3 * config.hidden_size, bias=self.bias)
        self.out_proj = nn.Linear(config.hidden_size, config.hidden_size, bias=self.bias)

    def forward(
        self,
        hidden_states: torch.Tensor,
        positions: torch.Tensor,
    ) -> torch.Tensor:
        seqlen = hidden_states.size(1)
        bcx_projected = self.in_proj(hidden_states)
        use_fused_gates = (
            self.use_fused_gates
            and self.shortconv_backend == "triton_k3"
            and hidden_states.is_cuda
            and bcx_projected.shape[0] == 1
            and self.l_cache == 3
            and not self.bias
        )
        if use_fused_gates:
            y = gated_depthwise_causal_conv_k3(
                bcx_projected,
                self.conv.weight,
                positions=positions,
                block_size=self.triton_block_t,
                num_warps=self.triton_num_warps,
            )
            return self.out_proj(y)

        bcx = bcx_projected.transpose(1, 2)
        b_gate, c_gate, x_proj = torch.chunk(bcx, 3, dim=1)
        bx = b_gate * x_proj
        use_triton = self.shortconv_backend == "triton_k3" and hidden_states.is_cuda and bx.shape[0] == 1 and self.l_cache == 3
        if use_triton:
            conv_out = depthwise_causal_conv_k3(
                bx,
                self.conv.weight,
                positions=positions,
                block_t=self.triton_block_t,
                num_warps=self.triton_num_warps,
            )
            if self.bias and self.conv.bias is not None:
                conv_out = conv_out + self.conv.bias.view(1, -1, 1)
            conv_out = conv_out[..., :seqlen]
        else:
            conv_out = self.conv(bx)[..., :seqlen]
        if not use_triton:
            conv_out = _replace_packed_shortconv_boundary_outputs(
                conv_out,
                bx=bx,
                weight=self.conv.weight,
                bias=self.conv.bias,
                positions=positions,
            )
        y = c_gate * conv_out
        y = y.transpose(1, 2).contiguous()
        return self.out_proj(y)


class LFM2DecoderLayer(nn.Module):
    def __init__(self, config: LFM2Config, layer_idx: int):
        super().__init__()
        self.is_attention_layer = config.layer_types[layer_idx] == "full_attention"
        if self.is_attention_layer:
            self.self_attn = LFM2Attention(config, layer_idx)
        else:
            self.conv = LFM2ShortConv(config, layer_idx)
        self.feed_forward = LFM2MLP(config)
        self.operator_norm = LFM2RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            use_fused_kernel=config.fused_backbone_kernels,
        )
        self.ffn_norm = LFM2RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            use_fused_kernel=config.fused_backbone_kernels,
        )

    def forward(self, hidden_states: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        operator_input = self.operator_norm(hidden_states)
        hidden_states = (
            self.self_attn(operator_input, positions)
            if self.is_attention_layer
            else self.conv(operator_input, positions)
        )
        hidden_states = hidden_states + residual
        hidden_states = hidden_states + self.feed_forward(self.ffn_norm(hidden_states))
        return hidden_states


def _checkpointed_layer(layer, hidden_states, positions, flash_ctx):
    # Autograd recomputation can run on a worker that does not inherit the
    # forward thread's ContextVar. Restore packed boundaries in that worker.
    if flash_ctx is not None:
        set_flash_attn_ctx(flash_ctx)
    return layer(hidden_states, positions)


class LFM2Model(nn.Module):
    def __init__(self, config: LFM2Config):
        super().__init__()
        config.validate()
        self.config = config
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList([LFM2DecoderLayer(config, idx) for idx in range(config.num_hidden_layers)])
        self.embedding_norm = LFM2RMSNorm(
            config.hidden_size,
            eps=config.norm_eps,
            use_fused_kernel=config.fused_backbone_kernels,
        )
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        if config.tie_embedding:
            self.lm_head.weight = self.embed_tokens.weight

    def get_hidden_states(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        if self.config.attn_backend == "flash":
            _set_flash_attn_ctx_from_positions(positions)
        hidden_states = self.embed_tokens(input_ids)
        for layer in self.layers:
            # Policy-gradient updates intentionally use eval() for behavior
            # parity. Autograd, not the dropout/training flag, determines
            # whether configured activation checkpointing is needed.
            if self.config.checkpoint_layers and torch.is_grad_enabled():
                from torch.utils.checkpoint import checkpoint
                flash_ctx = get_flash_attn_ctx() if self.config.attn_backend == 'flash' else None
                hidden_states = checkpoint(_checkpointed_layer, layer, hidden_states, positions, flash_ctx,
                                           use_reentrant=False, preserve_rng_state=False)
            else:
                hidden_states = layer(hidden_states, positions)
        return self.embedding_norm(hidden_states)

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        return self.lm_head(self.get_hidden_states(input_ids, positions))

    @property
    def ignore_index(self) -> int:
        return self.config.eos_token_id

    def from_pretrained(self, model_dir: str | Path) -> "LFM2Model":
        model_path = Path(model_dir)
        safes = sorted(model_path.glob("*.safetensors"))
        pts = sorted(model_path.glob("*.pt"))
        bins = sorted(model_path.glob("*.bin"))
        if safes:
            checkpoint = load_file(safes[0])
        elif pts:
            checkpoint = torch.load(pts[0], map_location="cpu")
        elif bins:
            checkpoint = torch.load(bins[0], map_location="cpu")
        else:
            raise FileNotFoundError(f"No .safetensors, .pt, or .bin files found in: {model_path}")
        if isinstance(checkpoint, dict) and "state_dict" in checkpoint and isinstance(checkpoint["state_dict"], dict):
            checkpoint = checkpoint["state_dict"]
        if isinstance(checkpoint, dict) and "model" in checkpoint and isinstance(checkpoint["model"], dict):
            checkpoint = checkpoint["model"]
        if not isinstance(checkpoint, dict):
            raise TypeError("Checkpoint did not contain a state dict")
        remapped = {}
        for key, value in checkpoint.items():
            remapped[key.replace("model.", "", 1) if key.startswith("model.") else key] = value
        missing, unexpected = self.load_state_dict(remapped, strict=False)
        missing = [key for key in missing if key != "lm_head.weight"]
        if missing:
            raise RuntimeError(f"Missing keys when loading LFM2 weights: {missing[:16]}")
        if unexpected:
            raise RuntimeError(f"Unexpected keys when loading LFM2 weights: {unexpected[:16]}")
        return self

    def save_pretrained(self, save_dir: str | Path) -> Path:
        path = Path(save_dir)
        path.mkdir(parents=True, exist_ok=True)
        config_path = path / "config.json"
        config_path.write_text(
            json.dumps(self.config.to_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        weight_path = path / "pytorch_model.bin"
        state_dict = {
            _export_lfm2_hf_key(key, tie_embedding=bool(self.config.tie_embedding)): value.detach().cpu()
            for key, value in self.state_dict().items()
            if _should_export_lfm2_key(key, tie_embedding=bool(self.config.tie_embedding))
        }
        torch.save(state_dict, weight_path)
        return weight_path

    def save_as_hf(self, save_dir: str | Path) -> Path:
        return self.save_pretrained(save_dir)


def _should_export_lfm2_key(key: str, *, tie_embedding: bool) -> bool:
    return not (tie_embedding and key == "lm_head.weight")


def _export_lfm2_hf_key(key: str, *, tie_embedding: bool) -> str:
    del tie_embedding
    if key == "lm_head.weight":
        return key
    return f"model.{key}"
