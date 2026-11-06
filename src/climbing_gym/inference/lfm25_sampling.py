"""Bounded-memory greedy and categorical sampling for BF16/FP16 rollouts."""
from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _partial_sample(L, SCORE, INDEX, MAXIMUM, TOTAL, SEED,
                    V: tl.constexpr, PARTS: tl.constexpr, TEMP: tl.constexpr,
                    LOGPROBS: tl.constexpr, BLOCK: tl.constexpr,
                    STEPS=None, TEMPERATURES=None, PER_ROW: tl.constexpr = False):
    row, part = tl.program_id(0), tl.program_id(1)
    col = part * BLOCK + tl.arange(0, BLOCK)
    logits = tl.load(L + row * V + col, col < V, other=-float("inf")).to(tl.float32)
    if PER_ROW:
        temperature = tl.load(TEMPERATURES + row)
        seed = (tl.load(SEED + row).to(tl.uint64)
                + tl.load(STEPS + row).to(tl.uint64) * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
        uniform = tl.maximum(tl.rand(seed, col), 1.0e-30)
        score = tl.where(temperature > 0,
                         logits / tl.maximum(temperature, 1.0e-30) - tl.log(-tl.log(uniform)), logits)
    elif TEMP > 0:
        uniform = tl.maximum(tl.rand(SEED, row * V + col), 1.0e-30)
        score = logits / TEMP - tl.log(-tl.log(uniform))
    else:
        score = logits
    maximum_score = tl.max(score, 0)
    index = tl.min(tl.where((score == maximum_score) & (col < V), col, 2147483647), 0)
    offset = row * PARTS + part
    tl.store(SCORE + offset, maximum_score)
    tl.store(INDEX + offset, index)
    if LOGPROBS:
        maximum = tl.max(logits, 0)
        total = tl.sum(tl.exp(logits - tl.where(maximum == -float("inf"), 0.0, maximum)), 0)
        tl.store(MAXIMUM + offset, maximum)
        tl.store(TOTAL + offset, total)


@triton.jit
def _finish_sample(L, SCORE, INDEX, MAXIMUM, TOTAL, TOKENS, LOGP,
                   V: tl.constexpr, PARTS: tl.constexpr, LOGPROBS: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    part = tl.arange(0, BLOCK)
    offset = row * PARTS + part
    scores = tl.load(SCORE + offset, part < PARTS, other=-float("inf"))
    indices = tl.load(INDEX + offset, part < PARTS, other=2147483647)
    best = tl.max(scores, 0)
    token = tl.min(tl.where(scores == best, indices, 2147483647), 0)
    tl.store(TOKENS + row, token)
    if LOGPROBS:
        maxima = tl.load(MAXIMUM + offset, part < PARTS, other=-float("inf"))
        sums = tl.load(TOTAL + offset, part < PARTS, other=0)
        maximum = tl.max(maxima, 0)
        total = tl.sum(sums * tl.exp(maxima - maximum), 0)
        selected = tl.load(L + row * V + token).to(tl.float32)
        tl.store(LOGP + row, selected - (maximum + tl.log(total)))


def sample_logits(logits: torch.Tensor, *, temperature: float, seed: int,
                  step: int, return_logprobs: bool) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Sample with Gumbel-max, preserving the full categorical distribution.

    Only tile reductions are materialized; no full FP32 logits, probabilities,
    or random-noise matrices are allocated. A distinct Philox key at each step
    prevents counter wrap from repeating noise on long, large-batch rollouts.
    Returned log probabilities are from the original, untempered logits.
    """
    if logits.ndim != 2 or not logits.is_cuda or logits.dtype not in (torch.bfloat16, torch.float16, torch.float32):
        raise ValueError("sampling requires a 2D CUDA BF16/FP16/FP32 logits tensor")
    logits = logits.contiguous()
    batch, vocab = logits.shape
    if batch == 0 or vocab == 0 or temperature < 0:
        raise ValueError("sampling requires nonempty logits and nonnegative temperature")
    parts = triton.cdiv(vocab, 4096)
    scores = torch.empty((batch, parts), device=logits.device, dtype=torch.float32)
    indices = torch.empty((batch, parts), device=logits.device, dtype=torch.int32)
    maxima = torch.empty_like(scores) if return_logprobs else scores
    totals = torch.empty_like(scores) if return_logprobs else scores
    tokens = torch.empty(batch, device=logits.device, dtype=torch.long)
    logprobs = torch.empty(batch, device=logits.device, dtype=torch.float32) if return_logprobs else None
    key = (int(seed) + int(step) * 0x9E3779B97F4A7C15) & 0x7FFFFFFFFFFFFFFF
    _partial_sample[(batch, parts)](
        logits, scores, indices, maxima, totals, key, vocab, parts, temperature, return_logprobs, 4096,
        num_warps=8,
    )
    _finish_sample[(batch,)](
        logits, scores, indices, maxima, totals, tokens, logprobs if logprobs is not None else scores,
        vocab, parts, return_logprobs, triton.next_power_of_2(parts),
    )
    return tokens, logprobs


class RowSampler:
    """Reusable full-support sampler with a separate RNG stream per request.

    Seeds and generated-token positions travel with a request, so replacing a
    neighboring slot or changing the polling interval cannot advance its RNG.
    """

    def __init__(self, logits):
        self.batch, self.vocab = logits.shape
        self.parts = triton.cdiv(self.vocab, 4096)
        self.scores = torch.empty((self.batch, self.parts), device=logits.device, dtype=torch.float32)
        self.indices = torch.empty_like(self.scores, dtype=torch.int32)
        self.maxima = torch.empty_like(self.scores)
        self.totals = torch.empty_like(self.scores)
        self.tokens = torch.empty(self.batch, device=logits.device, dtype=torch.long)
        self.logprobs = torch.empty(self.batch, device=logits.device, dtype=torch.float32)

    def sample(self, logits, seeds, steps, temperatures):
        _partial_sample[(self.batch, self.parts)](
            logits, self.scores, self.indices, self.maxima, self.totals,
            seeds, self.vocab, self.parts, 1., True, 4096,
            steps, temperatures, True, num_warps=8,
        )
        _finish_sample[(self.batch,)](
            logits, self.scores, self.indices, self.maxima, self.totals,
            self.tokens, self.logprobs, self.vocab, self.parts, True,
            triton.next_power_of_2(self.parts),
        )
        return self.tokens, self.logprobs


@triton.jit
def _head_logprobs(L, SELECTED, LOGP, VALUES, OUT, V: tl.constexpr, K: tl.constexpr,
                   BLOCK: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, BLOCK)
    selected = tl.load(SELECTED + row)
    log_z = tl.load(L + row * V + selected).to(tl.float32) - tl.load(LOGP + row)
    values = tl.load(VALUES + row * K + col, col < K, 0).to(tl.float32)
    tl.store(OUT + row * K + col, values - log_z, col < K)


def behavior_topk(logits, selected, logprobs, k):
    """Capture a sparse distribution without changing the categorical draw.

    Reuse the sampler's partition function through its selected-token logprob.
    The head is metadata; it never filters the distribution being sampled.
    """
    if not 0 < k <= min(logits.size(1), 128):
        raise ValueError('Behavior head exceeds the vocabulary or supported size')
    values, ids = torch.topk(logits, k, dim=-1, sorted=True)
    probs = torch.empty(values.shape, dtype=torch.float32, device=logits.device)
    _head_logprobs[(logits.size(0),)](logits, selected, logprobs, values, probs,
                                     logits.size(1), k, triton.next_power_of_2(k))
    return ids, probs
