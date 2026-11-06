"""Small fused inference kernels for LFM2.5 (no training/autograd state)."""
from __future__ import annotations

import torch
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def _gemv_kernel(X, W, Y, K: tl.constexpr, N: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    col = tl.arange(0, BK)
    w = tl.load(W + row[:, None] * K + col[None, :], (row[:, None] < N) & (col[None, :] < K), other=0).to(tl.float32)
    x = tl.load(X + col, col < K, other=0).to(tl.float32)
    tl.store(Y + row, tl.sum(w * x[None, :], 1), row < N)


def linear(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Use a bandwidth-oriented GEMV for single requests, cuBLAS for batches."""
    if x.size(0) != 1 or weight.size(1) > 16384:
        return F.linear(x, weight)
    rows, width = weight.shape
    output = torch.empty((1, rows), device=x.device, dtype=x.dtype)
    block_rows = 4 if width <= 2048 and rows >= 16384 else 1
    _gemv_kernel[(triton.cdiv(rows, block_rows),)](
        x, weight, output, width, rows, block_rows, triton.next_power_of_2(width), num_warps=4,
    )
    return output


@triton.jit
def _rms_kernel(X, W, R, Y, SUM, H: tl.constexpr, EPS: tl.constexpr, ADD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    if ADD:
        r = tl.load(R + row * H + col, col < H, other=0).to(tl.float32)
        x = (x + r).to(X.dtype.element_ty).to(tl.float32)
        tl.store(SUM + row * H + col, x, col < H)
    var = tl.sum(x * x, 0) / H
    normalized = (x * tl.rsqrt(var + EPS)).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < H, other=0).to(tl.float32)
    tl.store(Y + row * H + col, normalized * w, col < H)


def rms_norm(x: torch.Tensor, weight: torch.Tensor, eps: float, residual: torch.Tensor | None = None):
    output = torch.empty_like(x)
    summed = torch.empty_like(x) if residual is not None else x
    _rms_kernel[(x.numel() // x.size(-1),)](
        x, weight, residual if residual is not None else x, output, summed,
        x.size(-1), eps, residual is not None, triton.next_power_of_2(x.size(-1)),
        enable_fp_fusion=False,
    )
    return output, summed


@triton.jit
def _swiglu_kernel(X, Y, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    gate = tl.load(X + row * 2 * H + col, col < H, other=0).to(tl.float32)
    up = tl.load(X + row * 2 * H + H + col, col < H, other=0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H + col, activated * up, col < H)


def swiglu(x: torch.Tensor) -> torch.Tensor:
    width = x.size(-1) // 2
    output = torch.empty((x.size(0), width), device=x.device, dtype=x.dtype)
    _swiglu_kernel[(x.size(0), triton.cdiv(width, 256))](x, output, width, 256)
    return output


@triton.jit
def _conv_kernel(BCX, W, STATE, Y, ACTIVE, H: tl.constexpr, BLOCK: tl.constexpr,
                 MASKED: tl.constexpr, SLOTS=None, INDEXED: tl.constexpr = False):
    row = tl.program_id(0)
    slot = tl.maximum(tl.load(SLOTS + row), 0) if INDEXED else row
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(BCX + row * 3 * H + col, col < H, other=0).to(tl.float32)
    c = tl.load(BCX + row * 3 * H + H + col, col < H, other=0).to(tl.float32)
    x = tl.load(BCX + row * 3 * H + 2 * H + col, col < H, other=0).to(tl.float32)
    bx = (b * x).to(BCX.dtype.element_ty).to(tl.float32)
    s0 = tl.load(STATE + (slot * H + col) * 3 + 1, col < H, other=0).to(tl.float32)
    s1 = tl.load(STATE + (slot * H + col) * 3 + 2, col < H, other=0).to(tl.float32)
    w0 = tl.load(W + col * 3, col < H, other=0).to(tl.float32)
    w1 = tl.load(W + col * 3 + 1, col < H, other=0).to(tl.float32)
    w2 = tl.load(W + col * 3 + 2, col < H, other=0).to(tl.float32)
    # Match the reference cached convolution's low-precision products and
    # FP32 reduction, including the cast before the output gate.
    p0 = (s0 * w0).to(BCX.dtype.element_ty).to(tl.float32)
    p1 = (s1 * w1).to(BCX.dtype.element_ty).to(tl.float32)
    p2 = (bx * w2).to(BCX.dtype.element_ty).to(tl.float32)
    conv = (p0 + p1 + p2).to(BCX.dtype.element_ty).to(tl.float32)
    active = tl.load(ACTIVE + row) if MASKED else True
    tl.store(STATE + (slot * H + col) * 3, s0, (col < H) & active)
    tl.store(STATE + (slot * H + col) * 3 + 1, s1, (col < H) & active)
    tl.store(STATE + (slot * H + col) * 3 + 2, bx, (col < H) & active)
    tl.store(Y + row * H + col, c * conv, col < H)


def conv_update(bcx: torch.Tensor, weight: torch.Tensor, state: torch.Tensor,
                active: torch.Tensor | None = None, slot_ids: torch.Tensor | None = None) -> torch.Tensor:
    width = weight.size(0)
    output = torch.empty((bcx.size(0), width), device=bcx.device, dtype=bcx.dtype)
    _conv_kernel[(bcx.size(0), triton.cdiv(width, 256))](
        bcx, weight, state, output, active, width, 256, active is not None,
        slot_ids, slot_ids is not None, enable_fp_fusion=False)
    return output


@triton.jit
def _qkv_kernel(X, QW, KW, COS, SIN, POS, Q, K, V,
                QH: tl.constexpr, KH: tl.constexpr, D: tl.constexpr, EPS: tl.constexpr):
    row = tl.program_id(0)
    head = tl.program_id(1)
    d = tl.arange(0, D)
    width: tl.constexpr = (QH + 2 * KH) * D
    x = tl.load(X + row * width + head * D + d).to(tl.float32)
    if head < QH:
        w = tl.load(QW + d).to(tl.float32)
    else:
        w = tl.load(KW + d).to(tl.float32)
    normalized = (x * tl.rsqrt(tl.sum(x * x, 0) / D + EPS)).to(X.dtype.element_ty).to(tl.float32)
    normalized = (normalized * w).to(X.dtype.element_ty).to(tl.float32)
    other = tl.gather(normalized, (d + D // 2) % D, axis=0)
    position = tl.load(POS + row)
    cos = tl.load(COS + position * (D // 2) + d % (D // 2)).to(X.dtype.element_ty).to(tl.float32)
    sin = tl.load(SIN + position * (D // 2) + d % (D // 2)).to(X.dtype.element_ty).to(tl.float32)
    left = (normalized * cos).to(X.dtype.element_ty).to(tl.float32)
    right = (other * sin).to(X.dtype.element_ty).to(tl.float32)
    rotated = left + tl.where(d < D // 2, -right, right)
    if head < QH:
        tl.store(Q + (row * QH + head) * D + d, rotated)
    else:
        kh = head - QH
        tl.store(K + (row * KH + kh) * D + d, rotated)
        v = tl.load(X + row * width + (QH + KH + kh) * D + d)
        tl.store(V + (row * KH + kh) * D + d, v)


def prepare_qkv(x, q_weight, k_weight, cos, sin, positions, q_heads: int, kv_heads: int, head_dim: int, eps: float):
    q = torch.empty((x.size(0), 1, q_heads, head_dim), device=x.device, dtype=x.dtype)
    k = torch.empty((x.size(0), 1, kv_heads, head_dim), device=x.device, dtype=x.dtype)
    v = torch.empty_like(k)
    _qkv_kernel[(x.size(0), q_heads + kv_heads)](
        x, q_weight, k_weight, cos, sin, positions, q, k, v,
        q_heads, kv_heads, head_dim, eps, num_warps=4, enable_fp_fusion=False,
    )
    return q, k, v
