"""Chunked output projection with fused CISPO softmax/loss/gradient kernels.

Only a bounded chunk of logits exists at once. Forward retains row normalizers
and detached CISPO weights. Backward recomputes projection chunks and fuses the
upstream loss multiplier into the logits gradient before BF16 rounding. Output
diagnostics are nondifferentiable. All vocabulary entries normalize the loss.

Experimental: local objective/gradient tests pass, but small softmax rounding
differences amplify through the full BF16 model beyond our per-parameter parity
gate. Keep disabled for training pending further numerical validation.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _partial_lse(logits, partials, V: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr):
        row, tile = tl.program_id(0), tl.program_id(1)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        z = tl.load(logits + row * V + cols, cols < V, other=-float("inf")).to(tl.float32)
        m = tl.max(z, 0)
        value = m + tl.log(tl.sum(tl.exp(z - m), 0))
        tl.store(partials + row * PARTS + tile, value)

    @triton.jit
    def _cispo_rows(logits, partials, labels, old, advantages, lse_out, coefficients, stats,
                    V: tl.constexpr, N: tl.constexpr, PARTS: tl.constexpr, BLOCK: tl.constexpr,
                    SEQUENCES: tl.constexpr, LOW: tl.constexpr, HIGH: tl.constexpr):
        row = tl.program_id(0)
        tiles = tl.arange(0, BLOCK)
        values = tl.load(partials + row * PARTS + tiles, tiles < PARTS, other=-float("inf"))
        maximum = tl.max(values, 0)
        lse = maximum + tl.log(tl.sum(tl.exp(values - maximum), 0))
        label = tl.load(labels + row)
        selected = tl.load(logits + row * V + label).to(tl.float32)
        logp = selected - lse
        behavior = tl.load(old + row).to(tl.float32)
        difference = logp - behavior
        ratio = tl.exp(tl.minimum(20.0, tl.maximum(-20.0, difference)))
        weight = tl.minimum(1.0 + HIGH, tl.maximum(LOW, ratio))
        weighted_advantage = weight * tl.load(advantages + row).to(tl.float32)
        coefficient = weighted_advantage / SEQUENCES
        tl.store(lse_out + row, lse)
        tl.store(coefficients + row, weighted_advantage)
        tl.store(stats + row, -coefficient * logp)
        tl.store(stats + N + row, (ratio != weight).to(tl.float32))
        tl.store(stats + 2 * N + row, ratio)
        tl.store(stats + 3 * N + row, ratio - 1.0 - difference)
        tl.store(stats + 4 * N + row, tl.abs(tl.exp(logp) - tl.exp(behavior)))

    @triton.jit
    def _logits_gradient(logits, labels, lse, coefficients, scale, V: tl.constexpr, BLOCK: tl.constexpr):
        row, tile = tl.program_id(0), tl.program_id(1)
        cols = tile * BLOCK + tl.arange(0, BLOCK)
        z = tl.load(logits + row * V + cols, cols < V, other=0.0).to(tl.float32)
        probability = tl.exp(z - tl.load(lse + row))
        coefficient = tl.load(coefficients + row) * tl.load(scale)
        gradient = coefficient * (probability - (cols == tl.load(labels + row)).to(tl.float32))
        # Cross-entropy operates in FP32; the gradient back into the BF16
        # projection is rounded to the projection dtype, as in the reference.
        tl.store(logits + row * V + cols, gradient, cols < V)


def _loss_statistics(logits, labels, old, advantages, sequences, low, high):
    n, vocab = logits.shape
    if logits.is_cuda:
        if triton is None:
            raise RuntimeError("CUDA fused CISPO requires Triton")
        block = min(4096, triton.next_power_of_2(vocab))
        parts = triton.cdiv(vocab, block)
        partials = torch.empty((n, parts), device=logits.device, dtype=torch.float32)
        lse = torch.empty(n, device=logits.device, dtype=torch.float32)
        coefficient = torch.empty_like(lse)
        stats = torch.empty((5, n), device=logits.device, dtype=torch.float32)
        _partial_lse[(n, parts)](logits, partials, vocab, parts, block)
        _cispo_rows[(n,)](logits, partials, labels, old, advantages, lse, coefficient, stats,
                          vocab, n, parts, triton.next_power_of_2(parts), sequences, low, high)
        return stats, lse, coefficient
    logps = logits.float().log_softmax(-1)
    selected = logps.gather(1, labels[:, None]).squeeze(1)
    difference = selected - old.float()
    ratio = difference.clamp(-20, 20).exp()
    weight = ratio.clamp(low, 1.0 + high)
    weighted_advantage = weight * advantages.float()
    stats = torch.stack((-weighted_advantage * selected / sequences, (ratio != weight).float(), ratio,
                         ratio - 1 - difference, (selected.exp() - old.float().exp()).abs()))
    return stats, logits.float().logsumexp(-1), weighted_advantage


class _FusedCISPOHead(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, hidden, weight, labels, old, advantages, sequences, clip_low, clip_high, chunk_size):
        lse = torch.empty(len(hidden), device=hidden.device, dtype=torch.float32)
        coefficients = torch.empty_like(lse)
        totals = torch.zeros(5, device=hidden.device, dtype=torch.float32)
        for start in range(0, len(hidden), chunk_size):
            end = min(start + chunk_size, len(hidden))
            x = hidden[start:end]
            logits = (x @ weight.T).contiguous()
            stats, row_lse, row_coefficient = _loss_statistics(
                logits, labels[start:end], old[start:end], advantages[start:end],
                sequences, max(0.0, 1.0 - clip_low), clip_high,
            )
            totals.add_(stats.sum(1))
            lse[start:end] = row_lse
            coefficients[start:end] = row_coefficient
        totals[1:] /= len(hidden)
        ctx.save_for_backward(hidden, weight, labels, lse, coefficients)
        ctx.sequences, ctx.chunk_size = sequences, chunk_size
        outputs = tuple(totals.unbind())
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_loss, *unused):
        hidden, weight, labels, lse, coefficients = ctx.saved_tensors
        need_hidden, need_weight = ctx.needs_input_grad[:2]
        dh = torch.empty_like(hidden) if need_hidden else None
        dw = None
        scale = grad_loss.float() / ctx.sequences
        # Reverse chunk order matches the reference checkpointed head's
        # backward schedule and its BF16 weight-gradient accumulation.
        for start in reversed(range(0, len(hidden), ctx.chunk_size)):
            end = min(start + ctx.chunk_size, len(hidden))
            x = hidden[start:end]
            logits = (x @ weight.T).contiguous()
            n, vocab = logits.shape
            if logits.is_cuda:
                block = min(4096, triton.next_power_of_2(vocab))
                _logits_gradient[(n, triton.cdiv(vocab, block))](
                    logits, labels[start:end], lse[start:end], coefficients[start:end], scale, vocab, block,
                )
            else:
                gradient = (logits.float() - lse[start:end, None]).exp()
                gradient.scatter_add_(1, labels[start:end, None], -torch.ones((n, 1), device=logits.device))
                logits.copy_(gradient * (coefficients[start:end] * scale)[:, None])
            if need_hidden:
                dh[start:end] = logits @ weight
            if need_weight:
                chunk_gradient = (logits.T @ x).to(weight.dtype)
                if dw is None:
                    dw = chunk_gradient
                else:
                    dw.add_(chunk_gradient)
        return dh, dw, None, None, None, None, None, None, None


def fused_cispo_head(hidden, weight, labels, behavior_log_probs, advantages, *,
                     sequence_count, clip_low, clip_high, chunk_size=256):
    """Return loss, clip fraction, mean ratio, approximate KL, probability shift."""
    if hidden.ndim != 2 or weight.ndim != 2 or hidden.shape[1] != weight.shape[1]:
        raise ValueError("Expected hidden [tokens, hidden] and weight [vocabulary, hidden]")
    if not len(hidden) or sequence_count <= 0 or chunk_size <= 0:
        raise ValueError("Token count, sequence count, and chunk size must be positive")
    if not all(math.isfinite(v) and v >= 0 for v in (clip_low, clip_high)):
        raise ValueError("Clip epsilons must be finite and nonnegative")
    if any(x.shape != (len(hidden),) for x in (labels, behavior_log_probs, advantages)):
        raise ValueError("Labels, behavior log probabilities and advantages must have one value per token")
    if labels.dtype != torch.long or bool(((labels < 0) | (labels >= len(weight))).any()):
        raise ValueError("Labels must be int64 vocabulary indices")
    if any(x.device != hidden.device for x in (weight, labels, behavior_log_probs, advantages)):
        raise ValueError("All fused-head tensors must share a device")
    return _FusedCISPOHead.apply(hidden.contiguous(), weight, labels.contiguous(),
                                behavior_log_probs.detach().contiguous(), advantages.detach().contiguous(),
                                int(sequence_count), float(clip_low), float(clip_high), int(chunk_size))
