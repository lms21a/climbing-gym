"""Resumable, fixed-slot native generation with continuous request admission.

A stopped request retains all tokens *except* its last sampled token in the
hybrid cache. Resuming consumes that pending token exactly once, then only the
new suffix. A fresh prompt replaces both attention and convolution history.
"""
from __future__ import annotations

from dataclasses import dataclass
from contextlib import contextmanager
from typing import Sequence

import torch
import triton
import triton.language as tl

from climbing_gym.inference.lfm25_engine import LFM25HybridCache, LFM25SamplingConfig, flatten_token_batches
from climbing_gym.inference.lfm25_sampling import RowSampler
from climbing_gym.inference.types import BehaviorTopK


@triton.jit
def _record_head(IDS, PROBS, OUT_IDS, OUT_PROBS, LENGTHS, ALIVE,
                 K: tl.constexpr, CAP: tl.constexpr, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(ALIVE + row):
        length = tl.load(LENGTHS + row)
        col = tl.arange(0, BLOCK)
        offset = (row * CAP + length) * K + col
        tl.store(OUT_IDS + offset, tl.load(IDS + row * K + col, col < K, 0), col < K)
        tl.store(OUT_PROBS + offset, tl.load(PROBS + row * K + col, col < K, 0), col < K)


@triton.jit
def _record_sample(SAMPLED, LOGP, TOKENS, LOGPROBS, LENGTHS, LIMITS, REASONS,
                   ALIVE, PENDING, INPUT, STOP_STATES, STOP_OFFSETS, STOP_TABLE,
                   IGNORE_EOS, RNG_POSITIONS, EOS: tl.constexpr, V: tl.constexpr, CAP: tl.constexpr):
    row = tl.program_id(0)
    if tl.load(ALIVE + row):
        token = tl.load(SAMPLED + row)
        length = tl.load(LENGTHS + row)
        tl.store(TOKENS + row * CAP + length, token)
        tl.store(LOGPROBS + row * CAP + length, tl.load(LOGP + row))
        state = tl.load(STOP_STATES + row)
        offset = tl.load(STOP_OFFSETS + row)
        value = tl.load(STOP_TABLE + offset + state * V + token)
        tl.store(STOP_STATES + row, value & 65535)
        reason = tl.where((token == EOS) & ~tl.load(IGNORE_EOS + row), 1,
                          tl.where(value >= 65536, 2,
                                   tl.where(length + 1 >= tl.load(LIMITS + row), 3, 0)))
        tl.store(LENGTHS + row, length + 1)
        tl.store(RNG_POSITIONS + row, tl.load(RNG_POSITIONS + row) + 1)
        tl.store(REASONS + row, reason)
        tl.store(ALIVE + row, reason == 0)
        tl.store(PENDING + row, reason == 0)
        tl.store(INPUT + row, token)


@dataclass(frozen=True, slots=True)
class LFM25ContinuationHandle:
    """A single-use lease on one session slot; invalid after release/close."""

    session: object
    slot: int
    version: int
    prefix_token_ids: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class LFM25SessionResult:
    handle: LFM25ContinuationHandle
    prompt_token_ids: tuple[int, ...]
    token_ids: tuple[int, ...]
    logprobs: tuple[float, ...] | None
    finish_reason: int
    behavior_topk: BehaviorTopK | None = None


@dataclass(frozen=True, slots=True)
class LFM25PrefixSnapshot:
    """Opaque immutable lease; tensor storage belongs to the creating session."""

    session: object
    identity: int
    policy_version: int
    prefix_token_ids: tuple[int, ...]


@dataclass
class _SnapshotState:
    lease: LFM25PrefixSnapshot
    cache: LFM25HybridCache
    logits: torch.Tensor
    pending_token: int | None
    seed: int
    rng_position: int
    stop_state: int


@dataclass
class _Request:
    prompt: tuple[int, ...]
    sampling: LFM25SamplingConfig
    version: int
    delivered: bool = False
    handle: LFM25ContinuationHandle | None = None


class LFM25GenerationSession:
    """Own one reusable CUDA cache and admit work into individually free rows.

    ``poll`` advances at most ``steps`` tokens and returns newly finished rows.
    Call ``resume`` with an exact extended prefix, or ``release`` and ``admit``
    for a fresh context. Other live rows retain their state during either action.
    Sampling supports greedy/full categorical draws and token or ASCII text
    stops. Filtered sampling continues to use the ordinary engine.generate API.
    """

    @torch.inference_mode()
    def __init__(self, decoder, *, max_batch_size: int, max_new_tokens: int,
                 cache_len: int | None = None, behavior_topk: int = 0,
                 profile_timings: bool = False):
        if decoder._active_session is not None:
            raise RuntimeError("A native generation session is already open")
        if type(max_batch_size) is not int or max_batch_size <= 0:
            raise ValueError("max_batch_size must be a positive integer")
        if type(max_new_tokens) is not int or max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be a positive integer")
        self.decoder, self.engine = decoder, decoder.engine
        self.batch_size, self.max_new_tokens = max_batch_size, max_new_tokens
        if type(behavior_topk) is not int or not 0 <= behavior_topk <= min(128, self.engine.config.vocab_size):
            raise ValueError('behavior_topk must be an integer between 0 and min(vocab, 128)')
        self.behavior_topk = behavior_topk
        self.entry = decoder._get_graph(max_batch_size, cache_len or self.engine.max_model_len, masked=True)
        self.entry.active.zero_()
        self.entry.cache.seq_lens.zero_()
        self.requests: list[_Request | None] = [None] * max_batch_size
        self.versions = [0] * max_batch_size
        self.closed = False
        self.policy_version = self.engine.policy_version
        self._snapshots = {}
        self._next_snapshot = 0
        self._timing_events = []
        self.profile_timings = profile_timings
        device = self.engine.device

        def rows(dtype, value=0):
            return torch.full((max_batch_size,), value, device=device, dtype=dtype)

        self.lengths = rows(torch.int32)
        self.rng_positions = rows(torch.int64)
        self.limits = rows(torch.int32)
        self.reasons = rows(torch.int8)
        self.alive = rows(torch.bool)
        self.pending = rows(torch.bool)
        self.seeds = rows(torch.int64)
        self.temperatures = rows(torch.float32)
        self.ignore_eos = rows(torch.bool)
        self.stop_states = rows(torch.int64)
        self.stop_offsets = rows(torch.int64)
        self.tokens = torch.empty((max_batch_size, max_new_tokens), device=device, dtype=torch.long)
        self.logprobs = torch.empty_like(self.tokens, dtype=torch.float32)
        self.head_ids = (torch.empty((max_batch_size, max_new_tokens, behavior_topk),
                                    dtype=torch.int32, device=device) if behavior_topk else None)
        self.head_logprobs = torch.empty_like(self.head_ids, dtype=torch.float32) if behavior_topk else None
        self.sampler = RowSampler(self.entry.logits)
        self.stop_table = torch.zeros(self.engine.config.vocab_size, device=device, dtype=torch.int32)
        self._stop_tables = {}
        self.stats = dict(prefilled_tokens=0, resumed_prefix_tokens=0, appended_tokens=0,
                          admitted_requests=0, resumed_requests=0, decode_steps=0,
                          slot_refills=0, peak_live_slots=0, decode_row_steps=0,
                          reserved_row_steps=0, adaptive_graph_captures=0, decode_batches={},
                          forked_requests=0, copied_prefix_tokens=0, snapshot_prefilled_tokens=0,
                          snapshot_count=0, peak_snapshots=0, peak_snapshot_bytes=0)
        decoder._active_session = self

    def _check_open(self):
        if (self.closed or self.decoder._active_session is not self
                or self.engine._cuda_decoder is not self.decoder):
            raise RuntimeError("Native generation session is closed")
        if self.engine.policy_version != self.policy_version:
            raise ValueError('Session cache belongs to another policy version')

    @contextmanager
    def _timed(self, phase):
        if not self.profile_timings:
            yield
            return
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        try:
            yield
        finally:
            end.record()
            self._timing_events.append((phase, start, end))

    def timing_stats(self):
        """Resolve GPU phase times once at collection end, outside the hot path."""
        if self._timing_events:
            self._timing_events[-1][2].synchronize()
        result = {}
        for phase, start, end in self._timing_events:
            result[phase + '_gpu_seconds'] = result.get(phase + '_gpu_seconds', 0.) + start.elapsed_time(end) / 1000
        return result

    def _validate(self, prompt, sampling):
        if sampling.behavior_topk > self.behavior_topk:
            raise ValueError('Request behavior head exceeds the session reservation')
        if not prompt:
            raise ValueError("Empty prompts are not supported")
        if any(type(t) is not int or not 0 <= t < self.engine.config.vocab_size for t in prompt):
            raise ValueError("Prompt token outside the model vocabulary")
        if len(prompt) + sampling.max_new_tokens > self.entry.cache.max_seq_len:
            raise ValueError("Request exceeds reserved session context capacity")
        if sampling.max_new_tokens > self.max_new_tokens:
            raise ValueError("Request exceeds the session generation buffer")
        if sampling.top_k or sampling.top_p != 1.0 or sampling.repetition_penalty != 1.0:
            raise ValueError("Sessions support greedy/full-support sampling without repetition penalties")
        if sampling.stop_token_sequences:
            raise ValueError("Sessions use stop_token_ids or stop_table; encode string stops with build_stop_table")
        if any(not 0 <= t < self.engine.config.vocab_size for t in sampling.stop_token_ids):
            raise ValueError("Stop token outside the model vocabulary")
        table = sampling.stop_table
        if table is not None and (table.ndim != 2 or table.size(1) != self.engine.config.vocab_size
                                  or table.dtype != torch.int32):
            raise ValueError("stop_table must be an int32 state-by-vocabulary table")

    def _stop_offset(self, sampling):
        key = (id(sampling.stop_table), sampling.stop_token_ids)
        if sampling.stop_table is None and not sampling.stop_token_ids:
            return 0
        if key not in self._stop_tables:
            table = (torch.zeros((1, self.engine.config.vocab_size), dtype=torch.int32, device=self.engine.device)
                     if sampling.stop_table is None else sampling.stop_table.to(self.engine.device).clone())
            if sampling.stop_token_ids:
                table[:, list(sampling.stop_token_ids)] = 65536
            offset = self.stop_table.numel()
            self.stop_table = torch.cat((self.stop_table, table.reshape(-1)))
            # Hold the source alive: a later table must not reuse its Python id.
            self._stop_tables[key] = (offset, sampling.stop_table)
        return self._stop_tables[key][0]

    def _configure(self, slot, prompt, sampling, *, rng_position=0, stop_state=0):
        self.versions[slot] += 1
        self.requests[slot] = _Request(tuple(prompt), sampling, self.versions[slot])
        self.lengths[slot] = 0
        self.reasons[slot] = 0
        self.stop_states[slot] = stop_state
        self.rng_positions[slot] = rng_position
        self.limits[slot] = sampling.max_new_tokens
        self.seeds[slot] = sampling.seed & 0x7FFFFFFFFFFFFFFF
        self.temperatures[slot] = sampling.temperature
        self.ignore_eos[slot] = sampling.ignore_eos
        self.stop_offsets[slot] = self._stop_offset(sampling)
        self.alive[slot] = True
        self.pending[slot] = False

    @torch.inference_mode()
    def admit(self, prompts: Sequence[Sequence[int]], samplings: Sequence[LFM25SamplingConfig],
              *, slots: Sequence[int] | None = None) -> list[int]:
        self._check_open()
        if len(prompts) != len(samplings):
            raise ValueError("One sampling configuration is required per prompt")
        free = [i for i, request in enumerate(self.requests) if request is None]
        slots = list(free[:len(prompts)] if slots is None else slots)
        if len(slots) != len(prompts) or len(set(slots)) != len(slots) or any(i not in free for i in slots):
            raise ValueError("Admission requires distinct free session slots")
        work = sorted(zip(slots, prompts, samplings))
        for _, prompt, sampling in work:
            self._validate(prompt, sampling)
        # Consecutive rows can prefill directly into cache views, with the same
        # bounded token budget as ordinary generation and no full-cache copies.
        start = 0
        while start < len(work):
            end = start + 1
            while end < len(work) and work[end][0] == work[end - 1][0] + 1:
                end += 1
            first, last = work[start][0], work[end - 1][0] + 1
            cache = self.entry.cache
            view = cache.rows(first, last)
            flat = flatten_token_batches([p for _, p, _ in work[start:end]], device=self.engine.device)
            with self._timed('prefill'):
                self.engine.prefill(*flat, cache=view, max_prefill_tokens=self.decoder.max_prefill_tokens,
                                    logits_buffer=self.entry.logits[first:last])
            start = end
        for slot, prompt, sampling in work:
            self.stats['slot_refills'] += int(self.versions[slot] > 0)
            self._configure(slot, prompt, sampling)
            self.stats['prefilled_tokens'] += len(prompt)
            self.stats['admitted_requests'] += 1
        self.stats['peak_live_slots'] = max(self.stats['peak_live_slots'], sum(r is not None for r in self.requests))
        return slots

    def _store_snapshot(self, prompt, cache, logits, pending_token=None, seed=0, rng_position=0, stop_state=0):
        self._next_snapshot += 1
        lease = LFM25PrefixSnapshot(self, self._next_snapshot, self.policy_version, tuple(prompt))
        self._snapshots[lease.identity] = _SnapshotState(lease, cache, logits, pending_token, seed, rng_position, stop_state)
        self.stats['snapshot_count'] += 1
        self.stats['peak_snapshots'] = max(self.stats['peak_snapshots'], len(self._snapshots))
        size = sum(t.numel() * t.element_size() for s in self._snapshots.values()
                   for t in (s.cache.k_cache, s.cache.v_cache, s.cache.conv_state, s.cache.seq_lens,
                             s.cache.k_scales, s.cache.v_scales, s.logits) if t is not None)
        self.stats['peak_snapshot_bytes'] = max(self.stats['peak_snapshot_bytes'], size)
        return lease

    @torch.inference_mode()
    def prefill_snapshot(self, prompt: Sequence[int]) -> LFM25PrefixSnapshot:
        """Prefill one exact text prefix under the current weights, once."""
        self._check_open()
        prompt = tuple(prompt)
        self._validate(prompt, LFM25SamplingConfig(max_new_tokens=1))
        cache = self.engine._empty_cache(max_slots=1, max_seq_len=len(prompt),
                                        fp8_kv=self.entry.cache.k_scales is not None)
        with self._timed('prefill'):
            prefix = self.engine.prefill(*flatten_token_batches([prompt], device=self.engine.device),
                cache=cache, max_prefill_tokens=self.decoder.max_prefill_tokens)
        self.stats['prefilled_tokens'] += len(prompt)
        self.stats['snapshot_prefilled_tokens'] += len(prompt)
        return self._store_snapshot(prompt, cache, prefix.next_token_logits)

    @torch.inference_mode()
    def snapshot(self, handle: LFM25ContinuationHandle) -> LFM25PrefixSnapshot:
        """Copy only valid KV bytes/scales, conv state, logits and pending token."""
        request = self._request_for(handle)
        length = len(handle.prefix_token_ids) - 1
        if int(self.entry.cache.seq_lens[handle.slot]) != length:
            raise ValueError('Cached position does not match the pending-token prefix')
        with self._timed('cache_copy'):
            source = self.entry.cache.rows(handle.slot, handle.slot + 1)
            cache = LFM25HybridCache(source.k_cache[:, :, :length].clone(), source.v_cache[:, :, :length].clone(),
                source.conv_state.clone(), source.seq_lens.clone(),
                None if source.k_scales is None else source.k_scales[:, :, :length].clone(),
                None if source.v_scales is None else source.v_scales[:, :, :length].clone())
            logits = self.entry.logits[handle.slot:handle.slot + 1].clone()
        return self._store_snapshot(handle.prefix_token_ids, cache, logits, handle.prefix_token_ids[-1],
            request.sampling.seed, int(self.rng_positions[handle.slot]), int(self.stop_states[handle.slot]))

    def _snapshot_for(self, snapshot):
        self._check_open()
        if snapshot.policy_version != self.engine.policy_version:
            raise ValueError('Snapshot belongs to another policy version')
        state = self._snapshots.get(snapshot.identity)
        if snapshot.session is not self or state is None or state.lease is not snapshot:
            raise ValueError('Snapshot is released or belongs to another session')
        return state

    @torch.inference_mode()
    def fork(self, snapshot: LFM25PrefixSnapshot, samplings: Sequence[LFM25SamplingConfig],
             *, slots: Sequence[int] | None = None) -> list[int]:
        """Admit independent mutable children into any free rows, without prefill."""
        state = self._snapshot_for(snapshot)
        free = [i for i, r in enumerate(self.requests) if r is None]
        slots = list(free[:len(samplings)] if slots is None else slots)
        if len(slots) != len(samplings) or len(set(slots)) != len(slots) or any(i not in free for i in slots):
            raise ValueError('Fork requires distinct free session slots')
        for sampling in samplings:
            self._validate(snapshot.prefix_token_ids, sampling)
        with self._timed('cache_copy'):
            length = state.cache.max_seq_len
            for slot in slots:
                dst = self.entry.cache.rows(slot, slot + 1)
                dst.k_cache[:, :, :length].copy_(state.cache.k_cache)
                dst.v_cache[:, :, :length].copy_(state.cache.v_cache)
                if dst.k_scales is not None:
                    dst.k_scales[:, :, :length].copy_(state.cache.k_scales)
                    dst.v_scales[:, :, :length].copy_(state.cache.v_scales)
                dst.conv_state.copy_(state.cache.conv_state)
                dst.seq_lens.copy_(state.cache.seq_lens)
                self.entry.logits[slot].copy_(state.logits[0])
        if slots and state.pending_token is not None:
            # The last sampled token is not in KV yet. Consume it exactly once
            # in each child before exposing the next-token distribution.
            with self._timed('append'):
                saved = self.entry.input_tokens.clone()
                self.entry.input_tokens[slots] = state.pending_token
                self.entry.active.zero_()
                self.entry.active[slots] = True
                graph, _ = self._select_graph(slots)
                graph.replay()
                self.entry.input_tokens.copy_(saved)
        for slot, sampling in zip(slots, samplings):
            self.stats['slot_refills'] += int(self.versions[slot] > 0)
            self._configure(slot, snapshot.prefix_token_ids, sampling)
        self.stats['forked_requests'] += len(slots)
        self.stats['copied_prefix_tokens'] += length * len(slots)
        self.stats['peak_live_slots'] = max(self.stats['peak_live_slots'], sum(r is not None for r in self.requests))
        return slots

    def release_snapshot(self, snapshot: LFM25PrefixSnapshot) -> None:
        self._snapshot_for(snapshot)
        del self._snapshots[snapshot.identity]

    def _request_for(self, handle):
        self._check_open()
        if handle.session is not self or not 0 <= handle.slot < self.batch_size:
            raise ValueError("Continuation handle belongs to another session")
        request = self.requests[handle.slot]
        if (request is None or request.version != handle.version
                or not request.delivered or request.handle is not handle):
            raise ValueError("Continuation handle is stale or already consumed")
        return request

    def _select_graph(self, slots):
        captures = len(self.entry.compact_graphs)
        graph, size = self.decoder._select_graph(self.entry, slots)
        self.stats['adaptive_graph_captures'] += len(self.entry.compact_graphs) - captures
        return graph, size

    @torch.inference_mode()
    def resume(self, handles: Sequence[LFM25ContinuationHandle], prompts: Sequence[Sequence[int]],
               samplings: Sequence[LFM25SamplingConfig], *, preserve_rng: bool = False) -> None:
        if not len(handles) == len(prompts) == len(samplings):
            raise ValueError("One prompt and sampling configuration is required per handle")
        if len({h.slot for h in handles}) != len(handles):
            raise ValueError("Cannot resume a handle twice")
        appends = []
        rng_states = []
        for handle, prompt, sampling in zip(handles, prompts, samplings):
            request = self._request_for(handle)
            self._validate(prompt, sampling)
            prefix = handle.prefix_token_ids
            if tuple(prompt[:len(prefix)]) != prefix:
                raise ValueError("Continuation requires the exact cached token prefix")
            if preserve_rng and (tuple(prompt) != prefix or sampling.seed != request.sampling.seed
                                 or sampling.stop_table is not request.sampling.stop_table
                                 or sampling.stop_token_ids != request.sampling.stop_token_ids):
                raise ValueError('Internal pauses must preserve the exact context, seed and stop rules')
            rng_states.append((int(self.rng_positions[handle.slot]), int(self.stop_states[handle.slot]))
                              if preserve_rng else (0, 0))
            # The final generated token has not yet entered KV/conv state.
            appends.append((handle.slot, (prefix[-1], *prompt[len(prefix):])))
        if not appends:
            return
        width = max(len(tokens) for _, tokens in appends)
        tokens_cpu = torch.zeros((width, self.batch_size), dtype=torch.long)
        mask_cpu = torch.zeros((width, self.batch_size), dtype=torch.bool)
        for slot, tokens in appends:
            tokens_cpu[:len(tokens), slot] = torch.tensor(tokens)
            mask_cpu[:len(tokens), slot] = True
        tokens_gpu, masks_gpu = tokens_cpu.to(self.engine.device), mask_cpu.to(self.engine.device)
        # Existing live rows also have pending tokens; restore their input
        # buffer after forced appends to the resumed rows.
        pending_tokens = self.entry.input_tokens.clone()
        graph, _ = self._select_graph([slot for slot, _ in appends])
        for i in range(width):
            self.entry.input_tokens.copy_(tokens_gpu[i])
            self.entry.active.copy_(masks_gpu[i])
            graph.replay()
        self.entry.input_tokens.copy_(pending_tokens)
        for handle, prompt, sampling, (position, stop_state) in zip(handles, prompts, samplings, rng_states):
            self._configure(handle.slot, prompt, sampling, rng_position=position, stop_state=stop_state)
            self.stats['resumed_prefix_tokens'] += len(handle.prefix_token_ids) - 1
            self.stats['appended_tokens'] += len(prompt) - len(handle.prefix_token_ids) + 1
            self.stats['resumed_requests'] += 1

    @torch.inference_mode()
    def poll(self, *, steps: int | None = None) -> list[LFM25SessionResult]:
        self._check_open()
        steps = self.decoder.finish_check_interval if steps is None else steps
        if type(steps) is not int or steps <= 0:
            raise ValueError("steps must be a positive integer")
        live_slots = [i for i, r in enumerate(self.requests) if r is not None and not r.delivered]
        if not live_slots:
            return []
        graph, batch_size = self._select_graph(live_slots)
        for _ in range(steps):
            self.entry.active.copy_(self.pending)
            graph.replay()
            sampled, logp = self.sampler.sample(self.entry.logits, self.seeds, self.rng_positions, self.temperatures)
            if self.behavior_topk:
                from climbing_gym.inference.lfm25_sampling import behavior_topk
                ids, probs = behavior_topk(self.entry.logits, sampled, logp, self.behavior_topk)
                _record_head[(self.batch_size,)](ids, probs, self.head_ids, self.head_logprobs,
                    self.lengths, self.alive, self.behavior_topk, self.max_new_tokens,
                    triton.next_power_of_2(self.behavior_topk))
            _record_sample[(self.batch_size,)](
                sampled, logp, self.tokens, self.logprobs, self.lengths, self.limits,
                self.reasons, self.alive, self.pending, self.entry.input_tokens,
                self.stop_states, self.stop_offsets, self.stop_table, self.ignore_eos,
                self.rng_positions,
                self.engine.config.eos_token_id, self.engine.config.vocab_size, self.max_new_tokens,
            )
        self.stats['decode_steps'] += steps
        self.stats['decode_row_steps'] += batch_size * steps
        self.stats['reserved_row_steps'] += self.batch_size * steps
        batches = self.stats['decode_batches']
        batches[str(batch_size)] = batches.get(str(batch_size), 0) + steps
        reasons = self.reasons.cpu().tolist()
        slots = [i for i, r in enumerate(self.requests) if r is not None and not r.delivered and reasons[i]]
        if not slots:
            return []
        indices = torch.tensor(slots, device=self.engine.device)
        lengths = self.lengths[indices].cpu().tolist()
        max_length = max(lengths)
        tokens = self.tokens[:, :max_length].index_select(0, indices).cpu().tolist()
        logprobs = self.logprobs[:, :max_length].index_select(0, indices).cpu().tolist()
        results = []
        for slot, length, ids, probs in zip(slots, lengths, tokens, logprobs):
            request = self.requests[slot]
            request.delivered = True
            ids = tuple(ids[:length])
            handle = LFM25ContinuationHandle(self, slot, request.version, request.prompt + ids)
            request.handle = handle
            k = request.sampling.behavior_topk
            head = (BehaviorTopK(self.head_ids[slot, :length, :k].cpu(),
                                self.head_logprobs[slot, :length, :k].cpu()) if k else None)
            results.append(LFM25SessionResult(handle, request.prompt, ids,
                           tuple(probs[:length]) if request.sampling.return_logprobs else None, reasons[slot], head))
        return results

    @torch.inference_mode()
    def release(self, handle: LFM25ContinuationHandle) -> None:
        self._request_for(handle)
        self.requests[handle.slot] = None
        self.versions[handle.slot] += 1
        self.alive[handle.slot] = False
        self.pending[handle.slot] = False
        self.entry.cache.seq_lens[handle.slot] = 0

    def close(self):
        if not self.closed:
            self.closed = True
            self.requests = [None] * self.batch_size
            self._snapshots.clear()
            if self.decoder._active_session is self:
                self.decoder._active_session = None

    def __enter__(self):
        self._check_open()
        return self

    def __exit__(self, *_):
        self.close()
