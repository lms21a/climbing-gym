"""Sparse MIS/score-centering loss with a single dense logits gradient.

The vocabulary projection stays in ordinary PyTorch autograd, preserving its
BF16 GEMMs and gradient accumulation order. The sparse objective's backward
is fused. Weights and centering coefficients are detached;
diagnostics are not differentiable. This implements first-order gradients.
"""
from __future__ import annotations

import math

import torch
from torch.autograd.function import once_differentiable

try:
    import triton
    import triton.language as tl
    from triton.language.extra.cuda.libdevice import exp as cuda_exp
except ImportError:
    triton = None
    tl = None


if triton is not None:
    @triton.jit
    def _scaled_coefficients(COEF, IDS, Y, G, SCALED, TOTAL, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        upstream = tl.load(G + row)
        sampled = tl.load(COEF + row * (K + 1)) * upstream
        total = sampled
        if K:
            k = tl.arange(0, BLOCK)
            ids = tl.load(IDS + row * K + k, k < K, other=-1)
            selected = tl.load(Y + row)
            head = -tl.load(COEF + row * (K + 1) + 1 + k, k < K, other=0.) * upstream
            # Sum after applying the upstream gradient and combining a sampled
            # action that also appears in H, just as gather backward does.
            head = head + tl.where(ids == selected, sampled, 0.)
            total = tl.sum(head, 0) + tl.where(tl.sum((ids == selected).to(tl.int32), 0) == 0, sampled, 0.)
            tl.store(SCALED + row * (K + 1) + 1 + k, head, k < K)
        tl.store(SCALED + row * (K + 1), sampled)
        tl.store(TOTAL + row, total)

    @triton.jit
    def _dense_gradient(Z, NORM, TOTAL, DZ, V: tl.constexpr, BLOCK: tl.constexpr):
        row, part = tl.program_id(0), tl.program_id(1)
        col = part * BLOCK + tl.arange(0, BLOCK)
        z = tl.load(Z + row * V + col, col < V, other=0.).to(tl.float32)
        lp = (z - tl.load(NORM + row * 2)) - tl.load(NORM + row * 2 + 1)
        gradient = -cuda_exp(lp) * tl.load(TOTAL + row)
        tl.store(DZ + row * V + col, gradient, col < V)

    @triton.jit
    def _sparse_gradient(Z, Y, IDS, NORM, SCALED, TOTAL, DZ,
                         V: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        maximum = tl.load(NORM + row * 2)
        log_total = tl.load(NORM + row * 2 + 1)
        total = tl.load(TOTAL + row)
        selected = tl.load(Y + row)
        in_head = False
        if K:
            k = tl.arange(0, BLOCK)
            ids = tl.load(IDS + row * K + k, k < K, other=0)
            lp = (tl.load(Z + row * V + ids).to(tl.float32) - maximum) - log_total
            gradient = tl.load(SCALED + row * (K + 1) + 1 + k, k < K, other=0.) - cuda_exp(lp) * total
            tl.store(DZ + row * V + ids, gradient, k < K)
            in_head = tl.sum(((ids == selected) & (k < K)).to(tl.int32), 0) > 0
        if not in_head:
            selected_lp = (tl.load(Z + row * V + selected).to(tl.float32) - maximum) - log_total
            selected_gradient = tl.load(SCALED + row * (K + 1)) - cuda_exp(selected_lp) * total
            tl.store(DZ + row * V + selected, selected_gradient)


class _CorrectedProjection(torch.autograd.Function):
    @staticmethod
    def forward(ctx, logits, targets, behavior, ids, qlp, lower, upper):
        from climbing_gym.offpolicy import corrected_token_terms
        n, vocab = logits.shape
        k = ids.size(1)
        # The tail probability 1-sum(p_H) is sensitive to a last-bit change near
        # one, especially at a hard MIS bound. Preserve the reference forward
        # exactly; retain only BF16 logits and two normalizers for backward.
        lp = logits.float().log_softmax(-1)
        current = lp.gather(1, targets[:, None])[:, 0]
        terms = corrected_token_terms(current, behavior,
            train_head_logprobs=lp.gather(1, ids.long()) if k else None,
            behavior_head_logprobs=qlp if k else None, lower=lower, upper=upper)
        maximum, max_id = logits.max(-1)
        log_total = -lp.gather(1, max_id[:, None])[:, 0]
        norm = torch.stack((maximum.float(), log_total), dim=-1)
        coefficients = (torch.cat((terms.weights[:, None], terms.head_coefficients), dim=-1)
                        if k else terms.weights[:, None])
        ctx.save_for_backward(logits, targets, ids, norm, coefficients)
        outputs = (terms.objective, current, terms.weights,
                   terms.sampler_head_mass if k else torch.zeros_like(current))
        ctx.mark_non_differentiable(*outputs[1:])
        return outputs

    @staticmethod
    @once_differentiable
    def backward(ctx, grad_objective, *unused):
        logits, targets, ids, norm, coefficients = ctx.saved_tensors
        n, vocab = logits.shape
        k = ids.size(1)
        scaled = torch.empty_like(coefficients)
        total = torch.empty(n, device=logits.device, dtype=torch.float32)
        gradient = torch.empty_like(logits)
        block = triton.next_power_of_2(max(k, 1))
        _scaled_coefficients[(n,)](coefficients, ids, targets, grad_objective.contiguous(), scaled, total, k, block)
        _dense_gradient[(n, triton.cdiv(vocab, 4096))](logits, norm, total, gradient, vocab, 4096)
        _sparse_gradient[(n,)](logits, targets, ids, norm, scaled, total, gradient, vocab, k, block)
        return gradient, None, None, None, None, None, None


def corrected_projection(logits, targets, behavior, ids, qlp, lower=.5, upper=5.):
    """Return objective, sampled log p, MIS weights and sampler head mass.

    Internal projection primitive: caller validates distinct vocabulary IDs and
    finite, aligned sampler metadata once per pack, outside checkpoint replay.
    CPU uses the reference objective. CUDA retains no dense FP32 LPs or gather
    gradients, and reconstructs them from the exact reference normalizers.
    """
    if (not logits.is_cuda or triton is None or ids.size(1) > 128
            or logits.dtype not in (torch.float32, torch.bfloat16, torch.float16)):
        from climbing_gym.offpolicy import corrected_token_terms
        lp = logits.float().log_softmax(-1)
        current = lp.gather(1, targets[:, None])[:, 0]
        terms = corrected_token_terms(current, behavior,
            train_head_logprobs=lp.gather(1, ids.long()) if ids.size(1) else None,
            behavior_head_logprobs=qlp if ids.size(1) else None, lower=lower, upper=upper)
        mass = terms.sampler_head_mass if ids.size(1) else torch.zeros_like(current)
        return terms.objective, current.detach(), terms.weights, mass
    if logits.ndim != 2 or not logits.is_contiguous():
        raise ValueError('Expected contiguous [tokens, vocabulary] logits')
    n = logits.size(0)
    if targets.shape != (n,) or behavior.shape != (n,) or ids.ndim != 2 or ids.size(0) != n or qlp.shape != ids.shape:
        raise ValueError('Expected aligned sampled-token vectors and sparse heads')
    if targets.dtype != torch.int64 or ids.dtype not in (torch.int32, torch.int64):
        raise ValueError('Expected integer target and sparse-head indices')
    if any(x.device != logits.device for x in (targets, behavior, ids, qlp)):
        raise ValueError('Projection inputs must share a device')
    if not 0 < lower <= 1 <= upper or not math.isfinite(upper):
        raise ValueError('MIS requires 0 < lower <= 1 <= finite upper')
    return _CorrectedProjection.apply(logits, targets.contiguous(), behavior.detach().contiguous(),
                                      ids.contiguous(), qlp.detach().contiguous(), float(lower), float(upper))
