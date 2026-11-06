"""Native E4M3 projection kernels for SM89+ with FP32 accumulation.

Weights use one scale per output channel, activations one scale per token.
Quantized buffers are refreshed in place after policy sync so captured CUDA
graphs continue to reference the current weights. BF16 originals serve prefill.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _quantize_rows(X, Q, S, K: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * K + col, col < K, other=0).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(x), 0) / 448.0, 1.0e-12)
    tl.store(Q + row * K + col, tl.minimum(tl.maximum(x / scale, -448.0), 448.0), col < K)
    tl.store(S + row, scale)


@triton.jit
def _rms_quantize(X, W, R, Q, S, SUM, H: tl.constexpr, EPS: tl.constexpr,
                  ADD: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    x = tl.load(X + row * H + col, col < H, other=0).to(tl.float32)
    if ADD:
        residual = tl.load(R + row * H + col, col < H, other=0).to(tl.float32)
        x = (x + residual).to(X.dtype.element_ty).to(tl.float32)
        tl.store(SUM + row * H + col, x, col < H)
    normalized = (x * tl.rsqrt(tl.sum(x * x, 0) / H + EPS)).to(X.dtype.element_ty).to(tl.float32)
    w = tl.load(W + col, col < H, other=0).to(tl.float32)
    value = (normalized * w).to(X.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0) / 448.0, 1.0e-12)
    tl.store(Q + row * H + col, tl.minimum(tl.maximum(value / scale, -448.0), 448.0), col < H)
    tl.store(S + row, scale)


@triton.jit
def _swiglu_quantize(X, Q, S, H: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    gate = tl.load(X + row * 2 * H + col, col < H, other=0).to(tl.float32)
    up = tl.load(X + row * 2 * H + H + col, col < H, other=0).to(tl.float32)
    activated = (gate / (1.0 + tl.exp(-gate))).to(X.dtype.element_ty).to(tl.float32)
    value = (activated * up).to(X.dtype.element_ty).to(tl.float32)
    scale = tl.maximum(tl.max(tl.abs(value), 0) / 448.0, 1.0e-12)
    tl.store(Q + row * H + col, tl.minimum(tl.maximum(value / scale, -448.0), 448.0), col < H)
    tl.store(S + row, scale)


def rms_norm_fp8(x, weight, eps, residual=None):
    quantized = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    summed = torch.empty_like(x) if residual is not None else x
    _rms_quantize[(x.size(0),)](
        x, weight, residual if residual is not None else x, quantized, scales, summed,
        x.size(1), eps, residual is not None, triton.next_power_of_2(x.size(1)), enable_fp_fusion=False)
    return quantized, scales, summed


def swiglu_fp8(x):
    width = x.size(1) // 2
    quantized = torch.empty((x.size(0), width), device=x.device, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    _swiglu_quantize[(x.size(0),)](
        x, quantized, scales, width, triton.next_power_of_2(width), num_warps=8, enable_fp_fusion=False)
    return quantized, scales


@triton.jit
def _matmul(A, W, AS, WS, Y, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr,
            BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr, SPLIT_K: tl.constexpr = 1):
    m = tl.program_id(0) * BM + tl.arange(0, BM)
    n = tl.program_id(1) * BN + tl.arange(0, BN)
    k = tl.arange(0, BK)
    accum = tl.zeros((BM, BN), tl.float32)
    iterations: tl.constexpr = tl.cdiv(K, BK * SPLIT_K)
    for start in range(iterations):
        kk = (tl.program_id(2) * iterations + start) * BK + k
        a = tl.load(A + m[:, None] * K + kk[None, :], (m[:, None] < M) & (kk[None, :] < K), other=0.0)
        w = tl.load(W + n[None, :] * K + kk[:, None], (n[None, :] < N) & (kk[:, None] < K), other=0.0)
        accum = tl.dot(a, w, accum)
    if SPLIT_K == 1:
        accum *= tl.load(AS + m, m < M, other=0)[:, None]
        accum *= tl.load(WS + n, n < N, other=0)[None, :]
    offset = tl.program_id(2) * M * N + m[:, None] * N + n[None, :]
    tl.store(Y + offset, accum, (m[:, None] < M) & (n[None, :] < N))


@triton.jit
def _reduce_split(PARTIAL, AS, WS, Y, M: tl.constexpr, N: tl.constexpr,
                   SPLIT_K: tl.constexpr, BLOCK: tl.constexpr):
    index = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    total = tl.full((BLOCK,), 0.0, tl.float32)
    for part in tl.static_range(SPLIT_K):
        total += tl.load(PARTIAL + part * M * N + index, index < M * N, other=0)
    total *= tl.load(AS + index // N, index < M * N, other=0)
    total *= tl.load(WS + index % N, index < M * N, other=0)
    tl.store(Y + index, total, index < M * N)


def quantize_rows(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if x.ndim != 2 or not x.is_cuda or not x.is_contiguous() or x.dtype not in (torch.bfloat16, torch.float16):
        raise ValueError("FP8 quantization requires a contiguous 2D CUDA BF16/FP16 tensor")
    if min(x.shape) <= 0 or x.size(1) > 32768:
        raise ValueError("FP8 row width must be in [1, 32768]")
    q = torch.empty_like(x, dtype=torch.float8_e4m3fn)
    scales = torch.empty(x.size(0), device=x.device, dtype=torch.float32)
    _quantize_rows[(x.size(0),)](x, q, scales, x.size(1), triton.next_power_of_2(x.size(1)))
    return q, scales


@dataclass
class PackedFP8Weight:
    original: torch.Tensor
    values: torch.Tensor
    scales: torch.Tensor

    @classmethod
    def from_weight(cls, weight: torch.Tensor):
        if torch.cuda.get_device_capability(weight.device) < (8, 9):
            raise ValueError("Native FP8 projections require NVIDIA compute capability 8.9 or later")
        values, scales = quantize_rows(weight)
        return cls(weight, values, scales)

    def refresh(self):
        _quantize_rows[(self.original.size(0),)](
            self.original, self.values, self.scales, self.original.size(1),
            triton.next_power_of_2(self.original.size(1)))

    def linear(self, x: torch.Tensor, *, config: tuple[int, ...] | None = None):
        values, scales = quantize_rows(x)
        return self.linear_quantized(values, scales, dtype=x.dtype, config=config)

    def linear_quantized(self, x, scales, *, dtype=torch.bfloat16, config=None):
        batch, width = x.shape
        rows = self.values.size(0)
        if width != self.values.size(1):
            raise ValueError("FP8 activation and weight widths differ")
        # Independently tuned on the 4090: gate/up has many output channels;
        # down needs narrower tiles to expose enough parallelism at batch 128.
        default = (64, 128, 128, 4, 3) if rows >= 8192 and batch >= 64 else (32, 64, 128, 4, 3)
        if rows >= 8192 and batch < 64:
            default = (16 if batch <= 16 else 32, 128, 128, 4, 3)
        if batch >= 64 and rows <= 4096 and width >= 4096:
            default = (32, 128, 128, 4, 3, 2)
        if batch <= 64 and rows <= 4096 and width >= 4096:
            # Narrow drain batches need more down-projection partitions to
            # occupy the GPU; measured separately from the batch-128 tiles.
            default = (32, 128, 64, 4, 3, 4)
        selected = config or default
        bm, bn, bk, warps, stages = selected[:5]
        split_k = selected[5] if len(selected) > 5 else 1
        output = torch.empty((batch, rows), device=x.device, dtype=dtype)
        partial = torch.empty((split_k, batch, rows), device=x.device, dtype=torch.float32) if split_k > 1 else output
        _matmul[(triton.cdiv(batch, bm), triton.cdiv(rows, bn), split_k)](
            x, self.values, scales, self.scales, partial, batch, rows, width,
            bm, bn, bk, split_k, num_warps=warps, num_stages=stages)
        if split_k > 1:
            _reduce_split[(triton.cdiv(batch * rows, 256),)](
                partial, scales, self.scales, output, batch, rows, split_k, 256)
        return output
