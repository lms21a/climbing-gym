from __future__ import annotations

import asyncio
import math
import threading
import uuid
from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import torch
from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer

from climbing_gym.inference.config import InferenceEngineConfig, SamplingConfig
from climbing_gym.inference.lfm25_engine import (
    LFM25Engine,
    LFM25GenerationOutput,
    LFM25SamplingConfig,
    flatten_token_batches,
)
from climbing_gym.inference.types import BehaviorTopK, GenerationOutput, GenerationResult, PromptInput
from climbing_gym.inference.text_stops import trim_after_stop


@dataclass(slots=True)
class _QueuedRequest:
    prompt: PromptInput
    prompt_text: str
    prompt_token_ids: tuple[int, ...]
    sampling: SamplingConfig
    request_id: str
    future: asyncio.Future[GenerationResult]


class LFM25AsyncInferenceEngine:
    """Async batching adapter around the in-process LFM2.5 token engine."""

    def __init__(
        self,
        config: InferenceEngineConfig,
        *,
        batch_wait_seconds: float = 0.001,
        max_batch_size: int | None = None,
        compile_decode: bool = True,
        cuda_graphs: bool = False,
        bf16_weight_packing: bool = True,
        max_cuda_graphs: int = 1,
        max_prefill_tokens: int = 4096,
        fp8_mlp: bool = False,
        fp8_kv: bool = False,
        adaptive_batching: bool = False,
        fp8_lm_head: bool = False,
        fp8_operators: bool = False,
    ) -> None:
        self.config = config
        self.batch_wait_seconds = max(0.0, float(batch_wait_seconds))
        self.max_batch_size = _resolve_max_batch_size(config, max_batch_size)
        if fp8_mlp and not cuda_graphs:
            raise ValueError("Native FP8 MLP requires cuda_graphs=True")
        if fp8_kv and not cuda_graphs:
            raise ValueError("Native FP8 KV cache requires cuda_graphs=True")
        if adaptive_batching and not cuda_graphs:
            raise ValueError("Native adaptive batching requires cuda_graphs=True")
        if fp8_lm_head and not cuda_graphs:
            raise ValueError("Native FP8 vocabulary head requires cuda_graphs=True")
        if fp8_operators and not cuda_graphs:
            raise ValueError("Native FP8 operators require cuda_graphs=True")
        self.tokenizer = load_lfm2_tokenizer(config.model_path)
        self._engine = LFM25Engine.from_pretrained(
            config.model_path,
            dtype=config.dtype,
            device="cuda" if torch.cuda.is_available() else "cpu",
            max_model_len=config.max_model_len,
        )
        if cuda_graphs:
            operator_options = {}
            if fp8_operators:
                from climbing_gym.inference.lfm3_engine import LFM3Engine

                if not isinstance(self._engine, LFM3Engine):
                    raise ValueError("FP8 operator projections currently require LFM3")
                operator_options["fp8_operators"] = True
            self._engine.enable_cuda_graphs(bf16_weight_packing=bf16_weight_packing,
                                           max_graphs=max_cuda_graphs, max_prefill_tokens=max_prefill_tokens,
                                           fp8_mlp=fp8_mlp, fp8_kv=fp8_kv,
                                           adaptive_batching=adaptive_batching, fp8_lm_head=fp8_lm_head,
                                           **operator_options)
        elif compile_decode and self._engine.device.type == "cuda":
            self._engine.compile_decode()
        self._pending: list[_QueuedRequest] = []
        self._pending_lock: asyncio.Lock | None = None
        self._flush_task: asyncio.Task[None] | None = None
        self._closed = False
        self._engine_lock = threading.Lock()
        self._seed_lock = threading.Lock()
        self._generation_index = 0

    @classmethod
    def from_components(
        cls,
        *,
        engine: LFM25Engine,
        tokenizer: Any,
        config: InferenceEngineConfig | None = None,
        batch_wait_seconds: float = 0.0,
        max_batch_size: int = 64,
    ) -> "LFM25AsyncInferenceEngine":
        self = cls.__new__(cls)
        self.config = config or InferenceEngineConfig(model_path="")
        self.batch_wait_seconds = max(0.0, float(batch_wait_seconds))
        self.max_batch_size = _resolve_max_batch_size(self.config, max_batch_size)
        self.tokenizer = tokenizer
        self._engine = engine
        self._pending = []
        self._pending_lock = None
        self._flush_task = None
        self._closed = False
        self._engine_lock = threading.Lock()
        self._seed_lock = threading.Lock()
        self._generation_index = 0
        return self

    @property
    def raw_engine(self) -> LFM25Engine:
        return self._engine

    def supports_token_prompt_bridge(self) -> bool:
        return True

    def tokenize_prompt(
        self,
        prompt: PromptInput,
        *,
        assistant_prefill: str | None = None,
        chat_template_enable_thinking: bool | None = None,
    ) -> tuple[int, ...]:
        """Render and tokenize a prompt without running native generation."""

        resolved_prefill = (
            str(self.config.assistant_prefill)
            if assistant_prefill is None
            else str(assistant_prefill)
        )
        _, _, prompt_token_ids = self._resolve_prompt(
            prompt,
            assistant_prefill=resolved_prefill,
            chat_template_enable_thinking=chat_template_enable_thinking,
        )
        return prompt_token_ids

    @classmethod
    def tokenize_for_config(
        cls, config: InferenceEngineConfig, tokenizer: Any, prompt: PromptInput,
        *, assistant_prefill: str | None = None,
        chat_template_enable_thinking: bool | None = None,
    ) -> tuple[int, ...]:
        """Render a prompt exactly as generation does, without loading a model.

        Used to prepare representative preflight workloads using only a local
        tokenizer. The temporary adapter owns no native engine or GPU state.
        """
        renderer = cls.__new__(cls)
        renderer.config = config
        renderer.tokenizer = tokenizer
        return renderer.tokenize_prompt(
            prompt, assistant_prefill=assistant_prefill,
            chat_template_enable_thinking=chat_template_enable_thinking,
        )

    def sync_from_model(
        self,
        model: Any,
        *,
        drain_timeout: int | float = 300,
    ) -> LFM25Engine:
        """Wait for native execution to drain, then update packed weights in place."""

        timeout = float(drain_timeout)
        if not math.isfinite(timeout) or timeout < 0.0:
            raise ValueError("drain_timeout must be finite and >= 0")
        if self._closed:
            raise RuntimeError("LFM25AsyncInferenceEngine is closed")
        if not self._engine_lock.acquire(timeout=timeout):
            raise TimeoutError(
                f"Timed out after {timeout:g}s waiting for native LFM2.5 generation to drain"
            )
        try:
            return self._engine.sync_from_model(model, drain_timeout=drain_timeout)
        finally:
            self._engine_lock.release()

    def close(self) -> None:
        self._closed = True
        pending = list(self._pending)
        self._pending.clear()
        for request in pending:
            if not request.future.done():
                request.future.cancel()

    async def generate(
        self,
        prompt: PromptInput,
        *,
        sampling: SamplingConfig | Any | None = None,
        request_id: str | None = None,
        chat_template_enable_thinking: bool | None = None,
        assistant_prefill: str = "",
    ) -> GenerationResult:
        if self._closed:
            raise RuntimeError("LFM25AsyncInferenceEngine is closed")

        resolved_sampling = self._resolve_sampling(sampling)
        resolved_prompt, prompt_text, prompt_token_ids = self._resolve_prompt(
            prompt,
            assistant_prefill=(
                str(assistant_prefill)
                if assistant_prefill
                else str(self.config.assistant_prefill)
            ),
            chat_template_enable_thinking=chat_template_enable_thinking,
        )
        loop = asyncio.get_running_loop()
        future: asyncio.Future[GenerationResult] = loop.create_future()
        queued = _QueuedRequest(
            prompt=resolved_prompt,
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            sampling=resolved_sampling,
            request_id=request_id or f"req-{uuid.uuid4().hex}",
            future=future,
        )
        await self._enqueue(queued)
        return await future

    async def generate_many(
        self,
        prompts: Sequence[PromptInput],
        *,
        sampling: SamplingConfig | Any | None = None,
    ) -> list[GenerationResult]:
        return await asyncio.gather(*(self.generate(prompt, sampling=sampling) for prompt in prompts))

    async def generate_chains(
        self,
        requests: Sequence[tuple[PromptInput, SamplingConfig]],
        *,
        on_result: Callable[[int, GenerationResult], tuple[PromptInput, SamplingConfig] | None],
        max_new_tokens: int,
        reuse_cache: bool = True,
        poll_interval: int | None = None,
    ) -> list[list[GenerationResult]]:
        """Continuously execute independent, possibly multi-phase requests.

        The synchronous callback receives the original request index and each
        completed phase, and returns the next prompt/sampling pair or None.
        Exact extensions consume only new tokens; other prompts start with a
        fresh KV/conv context. Completed chains immediately admit queued work.
        Callbacks execute on the native worker and must not call this engine.
        Per-chain/per-phase RNG streams are independent of slot assignment.
        """
        if self._closed:
            raise RuntimeError("LFM25AsyncInferenceEngine is closed")
        if not requests:
            return []
        cancelled = threading.Event()
        loop = asyncio.get_running_loop()
        worker = loop.run_in_executor(
            None, self._generate_chains_sync, tuple(requests), on_result,
            max_new_tokens, reuse_cache, poll_interval, cancelled,
        )
        try:
            return await asyncio.shield(worker)
        except asyncio.CancelledError:
            cancelled.set()
            # Drain before releasing the caller, including the cache lease.
            try:
                await asyncio.shield(worker)
            except Exception:
                pass
            raise

    def _native_sampling(self, sampling, seed):
        native = _to_native_sampling(sampling, seed=seed, tokenizer=self.tokenizer)
        if sampling.stop:
            from dataclasses import replace
            from climbing_gym.inference.text_stops import build_stop_table
            if not hasattr(self, '_stop_tables'):
                self._stop_tables = {}
            key = tuple(sampling.stop)
            if key not in self._stop_tables:
                self._stop_tables[key] = build_stop_table(
                    self.tokenizer, key, vocab_size=self._engine.config.vocab_size).to(self._engine.device)
            native = replace(native, stop_token_sequences=(), stop_table=self._stop_tables[key])
        return native

    def _generate_chains_sync(self, requests, on_result, max_new_tokens,
                              reuse_cache, poll_interval, cancelled):
        with self._engine_lock:
            if self._closed:
                raise RuntimeError("LFM25AsyncInferenceEngine is closed")
            if cancelled.is_set():
                return [[] for _ in requests]
            seed = self._next_generation_seed()
            results = [[] for _ in requests]
            phases = [0] * len(requests)
            current = {}

            def prepare(index, request):
                prompt, sampling = request
                sampling = self._resolve_sampling(sampling)
                resolved, _, ids = self._resolve_prompt(prompt, assistant_prefill=str(self.config.assistant_prefill))
                # Full 64-bit keys mix chain and phase without dependence on
                # which other requests happened to finish at the same time.
                key = (seed + index * 0x517CC1B727220A95 + phases[index] * 0x6C8E9CF570932BD5) & 0x7FFFFFFFFFFFFFFF
                native = self._native_sampling(sampling, key)
                return index, resolved, ids, sampling, native

            queue = iter(enumerate(requests))
            head_k = max(int(dict(self._resolve_sampling(s).extra_params).get('behavior_topk', 0))
                         for _, s in requests)
            with self._engine.generation_session(
                max_batch_size=min(self.max_batch_size, len(requests)),
                max_new_tokens=max_new_tokens,
                **({'behavior_topk': head_k} if head_k else {}),
            ) as session:
                initial = []
                for _ in range(session.batch_size):
                    item = next(queue, None)
                    if item is not None:
                        initial.append(prepare(*item))
                slots = session.admit([r[2] for r in initial], [r[4] for r in initial])
                current.update(zip(slots, initial))
                while current and not cancelled.is_set() and not self._closed:
                    finished = session.poll(steps=poll_interval)
                    resume, fresh = [], []
                    for output in finished:
                        slot = output.handle.slot
                        index, prompt, ids, sampling, _ = current.pop(slot)
                        finish, stop = _decode_finish_reason(output.finish_reason)
                        text = self.tokenizer.decode(output.token_ids, skip_special_tokens=False)
                        if stop == 'stop' and sampling.stop:
                            text = trim_after_stop(text, sampling.stop)
                        result = GenerationResult(
                            request_id=f'chain-{index}-phase-{phases[index]}',
                            prompt=prompt, prompt_token_ids=ids,
                            outputs=(GenerationOutput(0, text, output.token_ids, output.logprobs, finish, stop,
                                                       getattr(output, 'behavior_topk', None)),),
                            finished=True, raw_output=None,
                        )
                        results[index].append(result)
                        following = on_result(index, result)
                        if following is not None:
                            phases[index] += 1
                            prepared = prepare(index, following)
                            prefix = output.handle.prefix_token_ids
                            if reuse_cache and prepared[2][:len(prefix)] == prefix:
                                resume.append((output.handle, prepared))
                            else:
                                session.release(output.handle)
                                fresh.append((slot, prepared))
                        else:
                            session.release(output.handle)
                            item = next(queue, None)
                            if item is not None:
                                fresh.append((slot, prepare(*item)))
                    if resume:
                        session.resume([h for h, _ in resume], [r[2] for _, r in resume], [r[4] for _, r in resume])
                        current.update((h.slot, r) for h, r in resume)
                    if fresh:
                        session.admit([r[2] for _, r in fresh], [r[4] for _, r in fresh], slots=[s for s, _ in fresh])
                        current.update(fresh)
                self.last_session_stats = dict(session.stats)
            if self._closed and not cancelled.is_set():
                raise RuntimeError("LFM25AsyncInferenceEngine closed during generation")
            return results

    async def abort_async(self, request_id: str | Sequence[str]) -> None:
        ids = {str(request_id)} if isinstance(request_id, str) else {str(item) for item in request_id}
        lock = self._lock()
        async with lock:
            remaining: list[_QueuedRequest] = []
            for request in self._pending:
                if request.request_id in ids:
                    request.future.cancel()
                else:
                    remaining.append(request)
            self._pending = remaining

    def abort(self, request_id: str | Sequence[str]) -> None:
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            asyncio.run(self.abort_async(request_id))
        else:
            loop.create_task(self.abort_async(request_id))

    def release_inference_buffers(self) -> None:
        """Drain native execution before freeing caches used by its CUDA graphs."""
        with self._engine_lock:
            self._engine.release_inference_buffers()

    def sleep(self, *, level: int = 1, mode: str = "abort") -> None:
        del level, mode
        self.release_inference_buffers()

    def wake(self, *, tags: Sequence[str] | None = None) -> None:
        del tags

    def _lock(self) -> asyncio.Lock:
        if self._pending_lock is None:
            self._pending_lock = asyncio.Lock()
        return self._pending_lock

    async def _enqueue(self, request: _QueuedRequest) -> None:
        lock = self._lock()
        async with lock:
            self._pending.append(request)
            if self._flush_task is None or self._flush_task.done():
                self._flush_task = asyncio.create_task(self._flush_pending())

    async def _flush_pending(self) -> None:
        if self.batch_wait_seconds > 0:
            await asyncio.sleep(self.batch_wait_seconds)
        while True:
            async with self._lock():
                if not self._pending:
                    return
                first = self._pending[0]
                key = _sampling_key(first.sampling)
                batch: list[_QueuedRequest] = []
                remaining: list[_QueuedRequest] = []
                for request in self._pending:
                    request_key = _sampling_key(request.sampling)
                    if len(batch) < self.max_batch_size and request_key == key:
                        batch.append(request)
                    else:
                        remaining.append(request)
                self._pending = remaining
            if batch:
                await self._run_batch(batch)

    async def _run_batch(self, batch: Sequence[_QueuedRequest]) -> None:
        loop = asyncio.get_running_loop()
        try:
            results = await loop.run_in_executor(None, self._generate_batch_sync, tuple(batch))
        except Exception as exc:
            for request in batch:
                if not request.future.done():
                    request.future.set_exception(exc)
            return
        for request, result in zip(batch, results, strict=True):
            if not request.future.done():
                request.future.set_result(result)

    def _generate_batch_sync(self, batch: tuple[_QueuedRequest, ...]) -> list[GenerationResult]:
        encoded = [list(request.prompt_token_ids) for request in batch]
        sampling = batch[0].sampling
        with self._engine_lock:
            native_sampling = self._native_sampling(sampling, self._next_generation_seed())
            flat, cu, max_len = flatten_token_batches(encoded, device=self._engine.device)
            output = self._engine.generate(
                flat,
                cu,
                max_len,
                repeats=1,
                sampling=native_sampling,
            )
            if self._engine.device.type == "cuda":
                torch.cuda.synchronize()
            return self._build_results(batch, output)

    def _next_generation_seed(self) -> int:
        with self._seed_lock:
            generation_index = self._generation_index
            self._generation_index += 1
        return (int(self.config.seed) + generation_index) % (1 << 64)

    def _build_results(
        self,
        batch: Sequence[_QueuedRequest],
        output: LFM25GenerationOutput,
    ) -> list[GenerationResult]:
        # Transfer each batch tensor once, rather than synchronizing the device
        # for every row's length, tokens, logprobs and finish reason.
        lengths = output.lengths[:, 0].detach().cpu()
        max_length = int(lengths.max().item()) if lengths.numel() else 0
        token_ids = output.token_ids[:, 0, :max_length].detach().cpu()
        finish_reasons = output.finish_reasons[:, 0].detach().cpu()
        logprobs = (None if output.logprobs is None else
                    output.logprobs[:, 0, :max_length].detach().cpu())
        head_ids = (None if output.topk_token_ids is None else output.topk_token_ids[:, 0, :max_length].cpu())
        head_logprobs = (None if output.topk_logprobs is None else output.topk_logprobs[:, 0, :max_length].cpu())
        results: list[GenerationResult] = []
        for row, request in enumerate(batch):
            length = int(lengths[row].item())
            row_token_ids = tuple(int(token_id) for token_id in token_ids[row, :length].detach().cpu().tolist())
            row_logprobs = (
                None
                if logprobs is None
                else tuple(float(value) for value in logprobs[row, :length].detach().cpu().tolist())
            )
            finish_reason, stop_reason = _decode_finish_reason(int(finish_reasons[row].item()))
            text = self.tokenizer.decode(row_token_ids, skip_special_tokens=False)
            if stop_reason == "stop" and request.sampling.stop:
                # String stops may finish inside the final sampled token. Keep
                # its ID/logprob intact, but expose text only through the stop.
                text = trim_after_stop(text, request.sampling.stop)
            results.append(
                GenerationResult(
                    request_id=request.request_id,
                    prompt=request.prompt,
                    prompt_token_ids=request.prompt_token_ids,
                    outputs=(
                        GenerationOutput(
                            index=0,
                            text=text,
                            token_ids=row_token_ids,
                            completion_logprobs=row_logprobs,
                            finish_reason=finish_reason,
                            stop_reason=stop_reason,
                            # Own only this request's head: a CPU view would
                            # retain/serialize the complete batch storage.
                            behavior_topk=(None if head_ids is None else
                                           BehaviorTopK(head_ids[row, :length].clone(),
                                                        head_logprobs[row, :length].clone())),
                        ),
                    ),
                    finished=True,
                    # Sparse metadata is already owned on CPU. Holding the raw
                    # output here would keep its full-batch GPU head alive.
                    raw_output=output if head_ids is None else None,
                )
            )
        return results

    def _resolve_sampling(self, sampling: SamplingConfig | Any | None) -> SamplingConfig:
        if sampling is None:
            return self.config.default_sampling
        if isinstance(sampling, SamplingConfig):
            return sampling
        return SamplingConfig(
            temperature=float(getattr(sampling, "temperature", 0.0)),
            top_k=int(getattr(sampling, "top_k", -1)),
            top_p=float(getattr(sampling, "top_p", 1.0)),
            max_tokens=int(getattr(sampling, "max_tokens", getattr(sampling, "max_new_tokens", 256))),
            n=int(getattr(sampling, "n", 1)),
            logprobs=getattr(sampling, "logprobs", None),
        )

    def _resolve_prompt(
        self,
        prompt: PromptInput,
        *,
        assistant_prefill: str = "",
        chat_template_enable_thinking: bool | None = None,
    ) -> tuple[PromptInput, str, tuple[int, ...]]:
        prefill = str(assistant_prefill or "")
        if isinstance(prompt, Mapping):
            prompt_text = str(prompt.get("prompt", prompt.get("text", "")) or "")
            raw_token_ids = prompt.get("prompt_token_ids")
            prompt_token_ids = (
                self._encode_text(prompt_text)
                if raw_token_ids is None
                else tuple(int(token_id) for token_id in raw_token_ids or ())
            )
            if prefill:
                prompt_text = f"{prompt_text}{prefill}"
                prompt_token_ids = prompt_token_ids + self._encode_text(prefill)
                resolved = dict(prompt)
                if "prompt" in resolved:
                    resolved["prompt"] = prompt_text
                elif "text" in resolved:
                    resolved["text"] = prompt_text
                resolved["prompt_token_ids"] = list(prompt_token_ids)
                return resolved, prompt_text, prompt_token_ids
            return prompt, prompt_text, prompt_token_ids
        if isinstance(prompt, str):
            rendered_prompt = self._apply_chat_template(
                prompt,
                enable_thinking=chat_template_enable_thinking,
            )
            prompt_text = f"{rendered_prompt}{prefill}"
            return prompt_text, prompt_text, self._encode_text(prompt_text)
        prompt_token_ids = tuple(int(token_id) for token_id in prompt)
        prompt_text = self.tokenizer.decode(prompt_token_ids, skip_special_tokens=False)
        if prefill:
            prompt_text = f"{prompt_text}{prefill}"
            prompt_token_ids = prompt_token_ids + self._encode_text(prefill)
        return list(prompt_token_ids), prompt_text, prompt_token_ids

    def _encode_text(self, text: str) -> tuple[int, ...]:
        return tuple(int(token_id) for token_id in self.tokenizer.encode(str(text), add_special_tokens=False))

    def _apply_chat_template(
        self,
        prompt: str,
        *,
        enable_thinking: bool | None,
    ) -> str:
        if not bool(self.config.apply_chat_template):
            return str(prompt)
        apply_chat_template = getattr(self.tokenizer, "apply_chat_template", None)
        if not callable(apply_chat_template):
            raise TypeError(
                "apply_chat_template=true requires a tokenizer with apply_chat_template(...)"
            )
        resolved_enable_thinking = (
            bool(self.config.chat_template_enable_thinking)
            if enable_thinking is None
            else bool(enable_thinking)
        )
        kwargs: dict[str, Any] = {
            "tokenize": False,
            "add_generation_prompt": True,
            "enable_thinking": resolved_enable_thinking,
        }
        messages = [{"role": "user", "content": str(prompt)}]
        try:
            return str(apply_chat_template(messages, **kwargs))
        except TypeError as exc:
            message = str(exc)
            if "enable_thinking" not in message or (
                "unexpected keyword" not in message
                and "got an unexpected" not in message
            ):
                raise
            kwargs.pop("enable_thinking")
            return str(apply_chat_template(messages, **kwargs))


def _sampling_key(sampling: SamplingConfig) -> tuple[Any, ...]:
    return (
        float(sampling.temperature),
        int(sampling.top_k),
        float(sampling.top_p),
        int(sampling.max_tokens),
        int(sampling.n),
        tuple(sampling.stop),
        sampling.logprobs,
        tuple(sorted(dict(sampling.extra_params).items())),
    )


def _to_native_sampling(sampling: SamplingConfig, *, seed: int, tokenizer=None) -> LFM25SamplingConfig:
    if int(sampling.n) != 1:
        raise ValueError("LFM25AsyncInferenceEngine requires sampling.n == 1")
    if sampling.stop and tokenizer is None:
        raise ValueError("A tokenizer is required for string stop sequences")
    top_k = 0 if int(sampling.top_k) == -1 else int(sampling.top_k)
    return LFM25SamplingConfig(
        temperature=float(sampling.temperature),
        top_k=top_k,
        top_p=float(sampling.top_p),
        repetition_penalty=float(
            dict(sampling.extra_params).get("repetition_penalty", 1.0)
        ),
        max_new_tokens=int(sampling.max_tokens),
        return_logprobs=sampling.logprobs is not None,
        seed=seed,
        ignore_eos=bool(dict(sampling.extra_params).get("ignore_eos", False)),
        stop_token_sequences=tuple(dict.fromkeys(
            tuple(tokenizer.encode(prefix + s, add_special_tokens=False))
            for s in sampling.stop for prefix in ('', ' ', '\n', '>'))),
        behavior_topk=dict(sampling.extra_params).get('behavior_topk', 0),
    )


def _decode_finish_reason(reason: int) -> tuple[str | None, str | None]:
    if reason == 1:
        return "stop", "eos"
    if reason == 2:
        return "stop", "stop"
    if reason == 3:
        return "length", None
    return None, None


__all__ = ["LFM25AsyncInferenceEngine"]


def _resolve_max_batch_size(config: InferenceEngineConfig, requested: int | None) -> int:
    if requested is not None and (type(requested) is not int or requested <= 0):
        raise ValueError("max_batch_size must be a positive integer")
    maximum = requested if requested is not None else config.max_num_seqs or 64
    return min(maximum, config.max_num_seqs) if config.max_num_seqs is not None else maximum
