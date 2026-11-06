"""LFM3 inference: width-7 gated convolution, FP32 RoPE, and head gates."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _conv(BCX, W, STATE, Y, ACTIVE, SLOTS, H: tl.constexpr, K: tl.constexpr,
          MASKED: tl.constexpr, INDEXED: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    slot = tl.maximum(tl.load(SLOTS + row), 0) if INDEXED else row
    d = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    b = tl.load(BCX + row * 3 * H + d, d < H, 0).to(tl.float32)
    c = tl.load(BCX + row * 3 * H + H + d, d < H, 0).to(tl.float32)
    x = tl.load(BCX + row * 3 * H + 2 * H + d, d < H, 0).to(tl.float32)
    bx = (b * x).to(BCX.dtype.element_ty).to(tl.float32)
    acc = tl.full((BLOCK,), 0, tl.float32)
    enabled = tl.load(ACTIVE + row) if MASKED else True
    for k in tl.static_range(K - 1):
        old = tl.load(STATE + (slot * H + d) * K + k + 1, d < H, 0)
        w = tl.load(W + d * K + k, d < H, 0).to(tl.float32)
        acc += old.to(tl.float32) * w
        tl.store(STATE + (slot * H + d) * K + k, old, (d < H) & enabled)
    w = tl.load(W + d * K + K - 1, d < H, 0).to(tl.float32)
    acc += bx * w
    tl.store(STATE + (slot * H + d) * K + K - 1, bx, (d < H) & enabled)
    value = acc.to(BCX.dtype.element_ty).to(tl.float32) * c
    tl.store(Y + row * H + d, value, d < H)


def conv_update(bcx, weight, state, active=None, slot_ids=None):
    h, k = weight.shape
    out = torch.empty((bcx.size(0), h), device=bcx.device, dtype=bcx.dtype)
    _conv[(bcx.size(0), triton.cdiv(h, 256))](
        bcx, weight, state, out, active, slot_ids, h, k,
        active is not None, slot_ids is not None, 256, enable_fp_fusion=False)
    return out


@triton.jit
def _qkv(X, QW, KW, COS, SIN, POS, Q, K, V, QH: tl.constexpr, KH: tl.constexpr,
         D: tl.constexpr, WIDTH: tl.constexpr, EPS: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    d = tl.arange(0, D)
    value = tl.load(X + row * WIDTH + head * D + d).to(tl.float32)
    w = tl.load(QW + d).to(tl.float32) if head < QH else tl.load(KW + d).to(tl.float32)
    norm = (value * tl.rsqrt(tl.sum(value * value, 0) / D + EPS)).to(X.dtype.element_ty).to(tl.float32)
    norm = (norm * w).to(X.dtype.element_ty).to(tl.float32)
    other = tl.gather(norm, (d + D // 2) % D, axis=0)
    pos = tl.load(POS + row)
    cos = tl.load(COS + pos * (D // 2) + d % (D // 2)).to(X.dtype.element_ty).to(tl.float32)
    sin = tl.load(SIN + pos * (D // 2) + d % (D // 2)).to(X.dtype.element_ty).to(tl.float32)
    # LFM3 does both products and the sum in FP32 (unlike LFM2).
    rotated = norm * cos + tl.where(d < D // 2, -other, other) * sin
    if head < QH:
        tl.store(Q + (row * QH + head) * D + d, rotated)
    else:
        kh = head - QH
        tl.store(K + (row * KH + kh) * D + d, rotated)
        tl.store(V + (row * KH + kh) * D + d,
                 tl.load(X + row * WIDTH + (QH + KH + kh) * D + d))


def prepare_qkv(x, qw, kw, cos, sin, positions, qh, kh, dim, eps):
    q = torch.empty((x.size(0), 1, qh, dim), device=x.device, dtype=x.dtype)
    k = torch.empty((x.size(0), 1, kh, dim), device=x.device, dtype=x.dtype)
    v = torch.empty_like(k)
    _qkv[(x.size(0), qh + kh)](x, qw, kw, cos, sin, positions, q, k, v,
                              qh, kh, dim, x.size(1), eps, num_warps=4, enable_fp_fusion=False)
    return q, k, v


@triton.jit
def _gate(X, PACKED, Y, H: tl.constexpr, D: tl.constexpr, WIDTH: tl.constexpr, OFFSET: tl.constexpr,
          BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    value = tl.load(X + row * H * D + col, col < H * D, 0).to(tl.float32)
    gate = tl.load(PACKED + row * WIDTH + OFFSET + col // D, col < H * D, 0).to(tl.float32)
    gate = (1.0 / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H * D + col, value * gate, col < H * D)


def head_gate(attended, packed, heads, dim):
    out = torch.empty((attended.size(0), heads * dim), device=attended.device, dtype=attended.dtype)
    _gate[(out.size(0), triton.cdiv(heads * dim, 256))](
        attended, packed, out, heads, dim, packed.size(1), packed.size(1) - heads, 256)
    return out


@triton.jit
def _sandwich(X, POST, RES, PRE, Y, SUM, H: tl.constexpr, EPS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    d = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + d, d < H, 0).to(tl.float32)
    post = tl.load(POST + d, d < H, 0).to(tl.float32)
    x = (x * tl.rsqrt(tl.sum(x * x, 0) / H + EPS)).to(X.dtype.element_ty).to(tl.float32)
    x = (x * post).to(X.dtype.element_ty).to(tl.float32)
    r = tl.load(RES + row * H + d, d < H, 0).to(tl.float32)
    x = (x + r).to(X.dtype.element_ty).to(tl.float32)
    tl.store(SUM + row * H + d, x, d < H)
    pre = tl.load(PRE + d, d < H, 0).to(tl.float32)
    norm = (x * tl.rsqrt(tl.sum(x * x, 0) / H + EPS)).to(X.dtype.element_ty).to(tl.float32)
    tl.store(Y + row * H + d, norm * pre, d < H)


def sandwich_norm(x, post_weight, residual, pre_weight, eps):
    out, summed = torch.empty_like(x), torch.empty_like(x)
    _sandwich[(x.size(0),)](x, post_weight, residual, pre_weight, out, summed,
                          x.size(1), eps, triton.next_power_of_2(x.size(1)), enable_fp_fusion=False)
    return out, summed
