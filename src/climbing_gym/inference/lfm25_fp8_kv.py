"""E4M3 KV storage and grouped-query decode for the native hybrid cache.

Each token and KV head has independent FP32 K/V scales. Queries, outputs,
convolution state and prefill attention retain the model's original precision.
Decode reads compressed history directly, with FP32 softmax/accumulation.
"""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _to_e4m3(value):
    # Triton 3.6 lowers the implicit store cast through truncated FP16 on
    # SM89. Convert directly to avoid that extra rounding near FP8 midpoints.
    return tl.inline_asm_elementwise(
        "{ .reg .b16 lo, hi; "
        "cvt.rn.satfinite.e4m3x2.f32 lo, $2, $1; "
        "cvt.rn.satfinite.e4m3x2.f32 hi, $4, $3; "
        "mov.b32 $0, {lo, hi}; }",
        constraints="=r,f,f,f,f", args=[value], dtype=tl.float8e4nv,
        is_pure=True, pack=4,
    )


@triton.jit
def _store_kv(K, V, KC, VC, KS, VS, POS, ACTIVE,
              S: tl.constexpr, H: tl.constexpr, D: tl.constexpr,
              K0: tl.constexpr, K1: tl.constexpr, K2: tl.constexpr,
              V0: tl.constexpr, V1: tl.constexpr, V2: tl.constexpr,
              C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
              APPEND: tl.constexpr, MASKED: tl.constexpr, BH: tl.constexpr, BD: tl.constexpr,
              SLOTS=None, INDEXED: tl.constexpr = False):
    row = tl.program_id(0)
    batch, token = row // S, row % S
    enabled = True
    if MASKED:
        enabled = tl.load(ACTIVE + batch)
    if enabled:
        slot = tl.load(SLOTS + batch) if INDEXED else batch
        position = tl.load(POS + batch) if APPEND else token
        h, d = tl.arange(0, BH), tl.arange(0, BD)
        mask = (h[:, None] < H) & (d[None, :] < D)
        k = tl.load(K + batch * K0 + token * K1 + h[:, None] * K2 + d[None, :], mask, 0).to(tl.float32)
        v = tl.load(V + batch * V0 + token * V1 + h[:, None] * V2 + d[None, :], mask, 0).to(tl.float32)
        sk = tl.maximum(tl.max(tl.abs(k), 1) / 448.0, 1.0e-12)
        sv = tl.maximum(tl.max(tl.abs(v), 1) / 448.0, 1.0e-12)
        offset = slot * C0 + position * C1 + h[:, None] * C2 + d[None, :]
        tl.store(KC + offset, _to_e4m3(k / sk[:, None]), mask)
        tl.store(VC + offset, _to_e4m3(v / sv[:, None]), mask)
        scale_offset = (slot * C0 + position * C1 + h * C2) // D
        tl.store(KS + scale_offset, sk, h < H)
        tl.store(VS + scale_offset, sv, h < H)


def store_kv(k, v, k_cache, v_cache, k_scales, v_scales, *, positions=None, active=None, slot_ids=None):
    """Write [batch, tokens, KV heads, dim] into contiguous cache row views."""
    batch, tokens, heads, dim = k.shape
    _store_kv[(batch * tokens,)](
        k, v, k_cache, v_cache, k_scales, v_scales,
        positions if positions is not None else k_scales,
        active if active is not None else k_scales,
        tokens, heads, dim, *k.stride()[:3], *v.stride()[:3],
        *k_cache.stride()[:3], positions is not None, active is not None,
        triton.next_power_of_2(heads), triton.next_power_of_2(dim),
        slot_ids, slot_ids is not None, num_warps=4,
    )


@triton.jit
def _decode(Q, K, V, KS, VS, POS, ACTIVE, OUT, PARTIAL, MAXIMA, TOTALS,
            H: tl.constexpr, G: tl.constexpr, D: tl.constexpr, CAP: tl.constexpr,
            C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
            SCALE: tl.constexpr, MASKED: tl.constexpr, SPLITS: tl.constexpr,
            M: tl.constexpr, N: tl.constexpr, BD: tl.constexpr,
            SLOTS=None, INDEXED: tl.constexpr = False, WINDOW: tl.constexpr = 0):
    batch, head, split = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    g, d, n = tl.arange(0, M), tl.arange(0, BD), tl.arange(0, N)
    enabled = True
    if MASKED:
        enabled = tl.load(ACTIVE + batch)
    maximum = tl.full((M,), -1.0e30, tl.float32)
    total = tl.zeros((M,), tl.float32)
    accum = tl.zeros((M, BD), tl.float32)
    if enabled:
        slot = tl.load(SLOTS + batch) if INDEXED else batch
        q = tl.load(Q + ((batch * H + head) * G + g[:, None]) * D + d[None, :],
                    (g[:, None] < G) & (d[None, :] < D), 0)
        length = tl.load(POS + batch) + 1
        span: tl.constexpr = tl.cdiv(WINDOW if WINDOW else CAP, SPLITS * N) * N
        first = tl.maximum(0, length - WINDOW) if WINDOW else 0
        for start in range(first + split * span, tl.minimum(first + (split + 1) * span, length), N):
            index = start + n
            offset = slot * C0 + index * C1 + head * C2
            k = tl.load(K + offset[None, :] + d[:, None],
                        (index[None, :] < length) & (d[:, None] < D), 0.0).to(q.dtype)
            sk = tl.load(KS + offset // D, index < length, 0)
            scores = tl.dot(q, k) * (SCALE * 1.4426950408889634) * sk[None, :]
            scores = tl.where(index[None, :] < length, scores, -1.0e30)
            updated_max = tl.maximum(maximum, tl.max(scores, 1))
            alpha = tl.exp2(maximum - updated_max)
            p = tl.exp2(scores - updated_max[:, None])
            total = total * alpha + tl.sum(p, 1)
            sv = tl.load(VS + offset // D, index < length, 0)
            v = tl.load(V + offset[:, None] + d[None, :],
                        (index[:, None] < length) & (d[None, :] < D), 0.0).to(q.dtype)
            accum = tl.dot((p * sv[None, :]).to(q.dtype), v, accum * alpha[:, None])
            maximum = updated_max
    if SPLITS == 1:
        offset = ((batch * H + head) * G + g[:, None]) * D + d[None, :]
        tl.store(OUT + offset, accum / tl.maximum(total[:, None], 1.0e-20),
                 (g[:, None] < G) & (d[None, :] < D))
    else:
        row = ((batch * H + head) * SPLITS + split) * G + g
        tl.store(PARTIAL + row[:, None] * D + d[None, :], accum,
                 (g[:, None] < G) & (d[None, :] < D))
        tl.store(MAXIMA + row, maximum, g < G)
        tl.store(TOTALS + row, total, g < G)


@triton.jit
def _combine(PARTIAL, MAXIMA, TOTALS, OUT, H: tl.constexpr, G: tl.constexpr,
              D: tl.constexpr, SPLITS: tl.constexpr, BS: tl.constexpr, BD: tl.constexpr):
    batch, head, group = tl.program_id(0), tl.program_id(1), tl.program_id(2)
    s, d = tl.arange(0, BS), tl.arange(0, BD)
    row = ((batch * H + head) * SPLITS + s) * G + group
    maximum = tl.load(MAXIMA + row, s < SPLITS, -1.0e30)
    scale = tl.exp2(maximum - tl.max(maximum, 0))
    total = tl.sum(scale * tl.load(TOTALS + row, s < SPLITS, 0), 0)
    partial = tl.load(PARTIAL + row[:, None] * D + d[None, :],
                      (s[:, None] < SPLITS) & (d[None, :] < D), 0)
    value = tl.sum(partial * scale[:, None], 0) / tl.maximum(total, 1.0e-20)
    tl.store(OUT + ((batch * H + head) * G + group) * D + d, value, d < D)


def attend(q, k_cache, v_cache, k_scales, v_scales, positions, *, active=None, scale=None, config=None, slot_ids=None, window=0):
    """Attend to compressed history, including the token at ``positions``."""
    batch, _, query_heads, dim = q.shape
    _, capacity, heads, _ = k_cache.shape
    groups = query_heads // heads
    # Large batches already expose enough parallelism; short batches split
    # histories to occupy the GPU without changing the reserved cache.
    splits = min(16, max(1, triton.cdiv(128, batch * heads)))
    if slot_ids is not None and batch <= 16 and capacity >= 1024:
        # A minimum-size compute bucket may contain only one live request.
        # Split its history so inactive padding cannot strand most SMs.
        splits = max(splits, 8)
    block, warps, splits = config or (256 if batch >= 16 and dim <= 64 else 64, 4, splits)
    out = torch.empty_like(q)
    partial = torch.empty((batch, heads, splits, groups, dim), device=q.device, dtype=torch.float32) if splits > 1 else out
    maxima = torch.empty((batch, heads, splits, groups), device=q.device, dtype=torch.float32) if splits > 1 else out
    totals = torch.empty_like(maxima) if splits > 1 else out
    _decode[(batch, heads, splits)](
        q, k_cache, v_cache, k_scales, v_scales, positions, active if active is not None else positions,
        out, partial, maxima, totals, heads, groups, dim, capacity, *k_cache.stride()[:3],
        scale if scale is not None else dim ** -0.5, active is not None, splits,
        max(16, triton.next_power_of_2(groups)), block, triton.next_power_of_2(dim),
        slot_ids, slot_ids is not None, window,
        num_warps=warps, num_stages=2,
    )
    if splits > 1:
        _combine[(batch, heads, groups)](
            partial, maxima, totals, out, heads, groups, dim, splits,
            triton.next_power_of_2(splits), triton.next_power_of_2(dim), num_warps=4)
    return out


def decode(q, k, v, k_cache, v_cache, k_scales, v_scales, positions, *, active=None, scale=None, slot_ids=None, window=0):
    store_kv(k, v, k_cache, v_cache, k_scales, v_scales, positions=positions, active=active, slot_ids=slot_ids)
    return attend(q, k_cache, v_cache, k_scales, v_scales, positions, active=active, scale=scale, slot_ids=slot_ids, window=window)
