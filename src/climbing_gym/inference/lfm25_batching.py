"""Compact decode graphs over a shared physical KV/convolution slot pool."""
from __future__ import annotations

from dataclasses import dataclass

import torch
import triton
import triton.language as tl


@triton.jit
def _gather_inputs(INPUT, LENGTHS, ACTIVE, SLOTS, TOKENS, POSITIONS, MASK,
                   B: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row, row < B, -1)
    valid = (row < B) & (slot >= 0)
    tokens = tl.load(INPUT + slot, valid, 0)
    positions = tl.load(LENGTHS + slot, valid, 0)
    active = tl.load(ACTIVE + slot, valid, False)
    tl.store(TOKENS + row, tokens, row < B)
    tl.store(POSITIONS + row, positions, row < B)
    tl.store(MASK + row, active, row < B)


@triton.jit
def _scatter_logits(LOGITS, SAVED, SLOTS, ACTIVE, V: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(ACTIVE + row):
        slot = tl.load(SLOTS + row)
        col = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        value = tl.load(LOGITS + row * V + col, col < V, 0)
        tl.store(SAVED + slot * V + col, value, col < V)


@triton.jit
def _advance_positions(LENGTHS, POSITIONS, SLOTS, ACTIVE, B: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.arange(0, BLOCK)
    slot = tl.load(SLOTS + row, row < B, -1)
    active = tl.load(ACTIVE + row, row < B, False)
    pos = tl.load(POSITIONS + row, row < B, 0)
    tl.store(LENGTHS + slot, pos + 1, (row < B) & (slot >= 0) & active)


def advance_positions(lengths, positions, slots, active):
    _advance_positions[(1,)](lengths, positions, slots, active, slots.numel(),
                              triton.next_power_of_2(slots.numel()))


@triton.jit
def _append_bf16_kv(K, V, KC, VC, POSITIONS, SLOTS, ACTIVE,
                    H: tl.constexpr, D: tl.constexpr,
                    C0: tl.constexpr, C1: tl.constexpr, C2: tl.constexpr,
                    BLOCK: tl.constexpr):
    row, head = tl.program_id(0), tl.program_id(1)
    if tl.load(ACTIVE + row):
        slot = tl.load(SLOTS + row)
        position = tl.load(POSITIONS + row)
        d = tl.arange(0, BLOCK)
        k = tl.load(K + (row * H + head) * D + d, d < D, 0)
        v = tl.load(V + (row * H + head) * D + d, d < D, 0)
        offset = slot * C0 + position * C1 + head * C2 + d
        tl.store(KC + offset, k, d < D)
        tl.store(VC + offset, v, d < D)


def append_bf16_kv(k, v, k_cache, v_cache, positions, slots, active):
    _append_bf16_kv[(k.size(0), k.size(2))](
        k, v, k_cache, v_cache, positions, slots, active, k.size(2), k.size(3),
        *k_cache.stride()[:3], triton.next_power_of_2(k.size(3)), num_warps=4)


@dataclass
class CompactDecodeGraph:
    graph: torch.cuda.CUDAGraph
    slots: torch.Tensor
    buffers: tuple[torch.Tensor, ...]
    mapped_slots: tuple[int, ...] = ()

    def set_slots(self, slots):
        mapping = tuple(slots)
        if mapping != self.mapped_slots:
            values = mapping + (-1,) * (self.slots.numel() - len(mapping))
            self.slots.copy_(torch.tensor(values, device=self.slots.device, dtype=torch.int32))
            self.mapped_slots = mapping


@torch.inference_mode()
def capture_compact(decoder, entry, batch_size):
    """Capture without mutating any resident request, including during warmup."""
    device = decoder.engine.device
    slots = torch.full((batch_size,), -1, device=device, dtype=torch.int32)
    tokens = torch.zeros(batch_size, device=device, dtype=torch.long)
    positions = torch.zeros(batch_size, device=device, dtype=torch.int32)
    active = torch.zeros(batch_size, device=device, dtype=torch.bool)

    def forward():
        _gather_inputs[(1,)](entry.input_tokens, entry.cache.seq_lens, entry.active, slots,
                            tokens, positions, active, batch_size, triton.next_power_of_2(batch_size))
        logits = decoder._forward(entry.cache, tokens, active, slot_ids=slots, positions=positions)
        _scatter_logits[(batch_size, triton.cdiv(logits.size(1), 1024))](
            logits, entry.logits, slots, active, logits.size(1), 1024)

    # All slots are invalid until capture completes: KV/conv/positions and
    # prefill logits remain untouched even when other requests are live.
    stream, current = decoder._capture_stream, torch.cuda.current_stream(device)
    stream.wait_stream(current)
    with torch.cuda.stream(stream):
        for _ in range(2):
            forward()
    current.wait_stream(stream)
    torch.cuda.synchronize(device)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        forward()
    current.wait_stream(stream)
    cache = entry.cache
    buffers = tuple(t for t in (tokens, positions, active, entry.input_tokens, entry.active,
                               entry.logits, cache.k_cache, cache.v_cache, cache.conv_state,
                               cache.seq_lens, cache.k_scales, cache.v_scales) if t is not None)
    return CompactDecodeGraph(graph, slots, buffers)
