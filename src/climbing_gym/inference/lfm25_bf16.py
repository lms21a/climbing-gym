"""Lossless BF16 weight streams for memory-bound decode.

Sign and mantissa use eight bits; common exponents use a four-bit code.
Code zero reads the original BF16 value, so zeros, unusual exponents and all
other bit patterns remain exact. Original matrices stay available for GEMM
prefill and batched decode. This trades extra resident memory for less decode
bandwidth; it is not quantization.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _pack(W, M, E, SIZE: tl.constexpr, BASE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    w0 = tl.load(W + 2 * i, 2 * i < SIZE, other=0).to(tl.uint16, bitcast=True).to(tl.int32)
    w1 = tl.load(W + 2 * i + 1, 2 * i + 1 < SIZE, other=0).to(tl.uint16, bitcast=True).to(tl.int32)
    e0, e1 = (w0 >> 7) & 255, (w1 >> 7) & 255
    c0 = tl.where((e0 >= BASE) & (e0 < BASE + 15), e0 - BASE + 1, 0)
    c1 = tl.where((e1 >= BASE) & (e1 < BASE + 15), e1 - BASE + 1, 0)
    tl.store(M + 2 * i, (w0 & 127) | ((w0 >> 8) & 128), 2 * i < SIZE)
    tl.store(M + 2 * i + 1, (w1 & 127) | ((w1 >> 8) & 128), 2 * i + 1 < SIZE)
    tl.store(E + i, c0 | (c1 << 4), 2 * i < SIZE)


@triton.jit
def _load_weight(W, M, E, offset, valid, BASE: tl.constexpr):
    m = tl.load(M + offset, valid, other=0).to(tl.int32)
    code = tl.load(E + offset // 2, valid, other=0).to(tl.int32)
    exponent = (code >> ((offset & 1) * 4)) & 15
    bits = (m & 127) | ((m & 128) << 8) | ((exponent + BASE - 1) << 7)
    decoded = bits.to(tl.uint16).to(tl.bfloat16, bitcast=True)
    original = tl.load(W + offset, valid & (exponent == 0), other=0)
    return tl.where(exponent == 0, original, decoded)


@triton.jit
def _unpack(W, M, E, Y, SIZE: tl.constexpr, BASE: tl.constexpr, BLOCK: tl.constexpr):
    i = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(Y + i, _load_weight(W, M, E, i, i < SIZE, BASE), i < SIZE)


@triton.jit
def _gemv(X, W, M, E, Y, K: tl.constexpr, N: tl.constexpr,
          BN: tl.constexpr, BK: tl.constexpr, BASE: tl.constexpr):
    row = tl.program_id(0) * BN + tl.arange(0, BN)
    col = tl.arange(0, BK)
    offset = row[:, None] * K + col[None, :]
    valid = (row[:, None] < N) & (col[None, :] < K)
    w = _load_weight(W, M, E, offset, valid, BASE).to(tl.float32)
    x = tl.load(X + col, col < K, other=0).to(tl.float32)
    tl.store(Y + row, tl.sum(w * x[None, :], 1), row < N)


@dataclass
class PackedBF16Weight:
    original: torch.Tensor
    mantissa: torch.Tensor
    exponent: torch.Tensor
    base: int = 110

    @classmethod
    def from_weight(cls, weight: torch.Tensor) -> PackedBF16Weight:
        if weight.dtype != torch.bfloat16 or not weight.is_cuda or not weight.is_contiguous():
            raise ValueError("lossless BF16 packing requires contiguous CUDA BF16 weights")
        if weight.ndim != 2 or weight.size(1) % 2:
            raise ValueError("lossless BF16 packing requires a matrix with even input width")
        result = cls(weight, torch.empty_like(weight, dtype=torch.uint8),
                     torch.empty((weight.size(0), weight.size(1) // 2), device=weight.device, dtype=torch.uint8))
        result.refresh()
        return result

    def refresh(self) -> None:
        """Repack in place after policy synchronization; graph addresses stay valid."""
        _pack[(triton.cdiv(self.original.numel() // 2, 256),)](
            self.original, self.mantissa, self.exponent, self.original.numel(), self.base, 256,
        )

    def unpack(self) -> torch.Tensor:
        output = torch.empty_like(self.original)
        _unpack[(triton.cdiv(output.numel(), 256),)](
            self.original, self.mantissa, self.exponent, output, output.numel(), self.base, 256,
        )
        return output

    def linear(self, x: torch.Tensor) -> torch.Tensor:
        rows, width = self.original.shape
        if x.shape != (1, width) or x.dtype != torch.bfloat16 or not x.is_contiguous():
            raise ValueError("packed BF16 GEMV requires one contiguous BF16 input row")
        output = torch.empty((1, rows), device=x.device, dtype=x.dtype)
        block_rows = 2 if width <= 2048 and rows < 65536 else 1
        _gemv[(triton.cdiv(rows, block_rows),)](
            x, self.original, self.mantissa, self.exponent, output,
            width, rows, block_rows, triton.next_power_of_2(width), self.base,
            num_warps=8 if width > 4096 else 4,
        )
        return output
