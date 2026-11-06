from __future__ import annotations

import torch

try:
    from liger_kernel.ops.rms_norm import LigerRMSNormFunction
    from liger_kernel.ops.rope import LigerRopeFunction
    from liger_kernel.ops.swiglu import LigerSiLUMulFunction

    LFM2_FUSED_KERNELS_AVAILABLE = True
except Exception:
    LigerRMSNormFunction = None
    LigerRopeFunction = None
    LigerSiLUMulFunction = None
    LFM2_FUSED_KERNELS_AVAILABLE = False


def fused_rms_norm(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    *,
    eps: float,
) -> torch.Tensor:
    """Apply LFM2's Llama-style RMSNorm with one fused CUDA kernel."""

    if not LFM2_FUSED_KERNELS_AVAILABLE or LigerRMSNormFunction is None:
        raise RuntimeError("Fused LFM2 RMSNorm requires liger-kernel.")
    return LigerRMSNormFunction.apply(
        hidden_states,
        weight,
        eps,
        0.0,
        "llama",
        True,
        None,
    )


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """Fuse SiLU and the SwiGLU elementwise product."""

    if not LFM2_FUSED_KERNELS_AVAILABLE or LigerSiLUMulFunction is None:
        raise RuntimeError("Fused LFM2 SwiGLU requires liger-kernel.")
    return LigerSiLUMulFunction.apply(gate, up)


def fused_rotary_emb(
    query: torch.Tensor,
    key: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fuse Q/K rotary embedding for LFM2's ``[B,T,H,D]`` layout."""

    if not LFM2_FUSED_KERNELS_AVAILABLE or LigerRopeFunction is None:
        raise RuntimeError("Fused LFM2 rotary embedding requires liger-kernel.")

    # Liger accepts logical [B,H,T,D] tensors and internally materializes the
    # contiguous [B,T,H,D] representation consumed by its Triton kernel.
    query_heads_first, key_heads_first = LigerRopeFunction.apply(
        query.transpose(1, 2),
        key.transpose(1, 2),
        cos,
        sin,
        None,
        1,
    )
    return (
        query_heads_first.transpose(1, 2),
        key_heads_first.transpose(1, 2),
    )


__all__ = [
    "LFM2_FUSED_KERNELS_AVAILABLE",
    "fused_rms_norm",
    "fused_rotary_emb",
    "fused_silu_mul",
]
