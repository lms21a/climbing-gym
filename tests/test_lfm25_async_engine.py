from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from climbing_gym.inference.config import InferenceEngineConfig, SamplingConfig
from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
from climbing_gym.inference.lfm25_engine import LFM25GenerationOutput


class _FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        return [ord(char) for char in str(text)]

    def decode(self, token_ids, skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        return "".join(chr(int(token_id)) for token_id in token_ids)

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize: bool,
        add_generation_prompt: bool,
        enable_thinking: bool,
    ) -> str:
        assert tokenize is False
        assert add_generation_prompt is True
        return f"<chat thinking={int(enable_thinking)}>{messages[0]['content']}</chat>"


class _FakeTokenEngine:
    device = torch.device("cpu")

    def __init__(self) -> None:
        self.calls = []
        self.sync_calls = []

    def generate(self, input_ids, cu_seqlens, max_seqlen, repeats, sampling):
        self.calls.append(
            {
                "input_ids": tuple(int(token_id) for token_id in input_ids.tolist()),
                "cu_seqlens": tuple(int(offset) for offset in cu_seqlens.tolist()),
                "max_seqlen": int(max_seqlen),
                "repeats": int(repeats),
                "sampling": sampling,
            }
        )
        batch = int(cu_seqlens.numel()) - 1
        token_ids = torch.zeros((batch, 1, 3), dtype=torch.long)
        lengths = torch.ones((batch, 1), dtype=torch.int32)
        finish_reasons = torch.full((batch, 1), 3, dtype=torch.int8)
        logprobs = torch.full((batch, 1, 3), -0.5, dtype=torch.float32) if sampling.return_logprobs else None
        for row in range(batch):
            token_ids[row, 0, 0] = ord("x") + row
        return LFM25GenerationOutput(
            token_ids=token_ids,
            lengths=lengths,
            logprobs=logprobs,
            finish_reasons=finish_reasons,
        )

    def sync_from_model(self, model, *, drain_timeout):
        self.sync_calls.append((model, drain_timeout))
        return self


def test_sparse_behavior_results_own_only_their_request_storage():
    from dataclasses import replace
    import weakref

    class HeadEngine(_FakeTokenEngine):
        def generate(self, *args, **kwargs):
            output = super().generate(*args, **kwargs)
            batch = output.token_ids.size(0)
            ids = torch.arange(8,dtype=torch.int32).expand(batch,1,3,8).clone()
            self.batch_head = weakref.ref(ids)
            return replace(output, topk_token_ids=ids,
                           topk_logprobs=torch.full((batch,1,3,8),-4.,dtype=torch.float32))

    raw = HeadEngine()
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw,tokenizer=_FakeTokenizer(),
                                                       max_batch_size=8,batch_wait_seconds=.01)
    sample = SamplingConfig(temperature=1.,max_tokens=3,logprobs=1,extra_params={'behavior_topk':8})

    async def run():
        return await asyncio.gather(engine.generate('a',sampling=sample),engine.generate('b',sampling=sample))

    results = asyncio.run(run())
    for result in results:
        head = result.outputs[0].behavior_topk
        assert head.token_ids.shape == head.logprobs.shape == (1,8)
        assert head.token_ids.untyped_storage().nbytes() == 8*4
        assert head.logprobs.untyped_storage().nbytes() == 8*4
        assert result.raw_output is None
    assert raw.batch_head() is None
    engine.close()


def test_async_engine_batches_concurrent_generate_calls() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        batch_wait_seconds=0.01,
        max_batch_size=8,
    )

    async def run():
        sampling = SamplingConfig(
            temperature=1.0,
            top_k=-1,
            top_p=1.0,
            max_tokens=3,
            logprobs=1,
            extra_params={"ignore_eos": True, "repetition_penalty": 1.05},
        )
        return await asyncio.gather(
            engine.generate("ab", sampling=sampling, request_id="r1"),
            engine.generate(
                {"prompt": "cd", "prompt_token_ids": [99, 100]},
                sampling=sampling,
                request_id="r2",
            ),
        )

    first, second = asyncio.run(run())

    assert len(fake_engine.calls) == 1
    call = fake_engine.calls[0]
    assert call["input_ids"] == (ord("a"), ord("b"), 99, 100)
    assert call["cu_seqlens"] == (0, 2, 4)
    assert call["max_seqlen"] == 2
    assert call["repeats"] == 1
    assert call["sampling"].return_logprobs is True
    assert call["sampling"].top_k == 0
    assert call["sampling"].ignore_eos is True
    assert call["sampling"].repetition_penalty == 1.05

    assert first.request_id == "r1"
    assert first.prompt_token_ids == (ord("a"), ord("b"))
    assert first.text == "x"
    assert first.token_ids == (ord("x"),)
    assert first.completion_logprobs == (-0.5,)
    assert first.outputs[0].finish_reason == "length"

    assert second.request_id == "r2"
    assert second.prompt_token_ids == (99, 100)
    assert second.text == "y"
    assert second.token_ids == (ord("y"),)


def test_async_engine_batches_variable_length_prompts() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        batch_wait_seconds=0.01,
        max_batch_size=8,
    )

    async def run():
        sampling = SamplingConfig(temperature=1.0, max_tokens=1)
        return await asyncio.gather(
            engine.generate("a", sampling=sampling),
            engine.generate("long", sampling=sampling),
        )

    asyncio.run(run())

    assert len(fake_engine.calls) == 1
    assert fake_engine.calls[0]["cu_seqlens"] == (0, 1, 5)
    assert fake_engine.calls[0]["max_seqlen"] == 4


def test_async_engine_appends_assistant_prefill_to_token_prompt() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        batch_wait_seconds=0.0,
    )

    async def run():
        return await engine.generate(
            {"prompt": "ab", "prompt_token_ids": [1, 2]},
            sampling=SamplingConfig(max_tokens=1),
            assistant_prefill="!",
        )

    result = asyncio.run(run())

    assert fake_engine.calls[0]["input_ids"] == (1, 2, ord("!"))
    assert result.prompt_token_ids == (1, 2, ord("!"))
    assert result.prompt == {"prompt": "ab!", "prompt_token_ids": [1, 2, ord("!")]}


def test_async_engine_applies_chat_template_then_configured_prefill() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        config=InferenceEngineConfig(
            model_path="",
            apply_chat_template=True,
            chat_template_enable_thinking=False,
            assistant_prefill="<thinking>go\n",
        ),
        batch_wait_seconds=0.0,
    )

    result = asyncio.run(
        engine.generate("ab", sampling=SamplingConfig(max_tokens=1))
    )

    expected = "<chat thinking=0>ab</chat><thinking>go\n"
    assert fake_engine.calls[0]["input_ids"] == tuple(ord(char) for char in expected)
    assert result.prompt == expected
    assert result.prompt_token_ids == tuple(ord(char) for char in expected)


def test_async_engine_tokenizes_privileged_prompt_without_generation() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        config=InferenceEngineConfig(
            model_path="",
            apply_chat_template=True,
            chat_template_enable_thinking=False,
            assistant_prefill="policy-prefill",
        ),
    )

    default_ids = engine.tokenize_prompt("question")
    critic_ids = engine.tokenize_prompt(
        "question",
        assistant_prefill="privileged-answer",
        chat_template_enable_thinking=True,
    )

    assert default_ids == tuple(ord(char) for char in "<chat thinking=0>question</chat>policy-prefill")
    assert critic_ids == tuple(ord(char) for char in "<chat thinking=1>question</chat>privileged-answer")
    assert fake_engine.calls == []


def test_async_engine_advances_generation_seed_once_per_native_batch() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        config=InferenceEngineConfig(model_path="", seed=41),
        batch_wait_seconds=0.01,
        max_batch_size=8,
    )

    async def run() -> None:
        sampling = SamplingConfig(temperature=1.0, max_tokens=1)
        await asyncio.gather(
            engine.generate("aa", sampling=sampling),
            engine.generate("bb", sampling=sampling),
        )
        await engine.generate("cc", sampling=sampling)

    asyncio.run(run())

    assert len(fake_engine.calls) == 2
    assert [call["sampling"].seed for call in fake_engine.calls] == [41, 42]

    second_fake_engine = _FakeTokenEngine()
    second_engine = LFM25AsyncInferenceEngine.from_components(
        engine=second_fake_engine,
        tokenizer=_FakeTokenizer(),
        config=InferenceEngineConfig(model_path="", seed=41),
        batch_wait_seconds=0.0,
    )
    asyncio.run(second_engine.generate("aa", sampling=SamplingConfig(temperature=1.0, max_tokens=1)))
    assert second_fake_engine.calls[0]["sampling"].seed == 41


def test_async_engine_seed_reservations_are_thread_safe() -> None:
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=_FakeTokenEngine(),
        tokenizer=_FakeTokenizer(),
        config=InferenceEngineConfig(model_path="", seed=1000),
    )

    with ThreadPoolExecutor(max_workers=8) as pool:
        seeds = list(pool.map(lambda _: engine._next_generation_seed(), range(100)))

    assert sorted(seeds) == list(range(1000, 1100))
    assert engine._next_generation_seed() == 1100


def test_async_engine_syncs_under_generation_lock_and_honors_drain_timeout() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
    )
    model = object()

    assert engine.sync_from_model(model, drain_timeout=3) is fake_engine
    assert fake_engine.sync_calls == [(model, 3)]

    engine._engine_lock.acquire()
    try:
        with pytest.raises(TimeoutError, match="waiting for native LFM2.5 generation"):
            engine.sync_from_model(model, drain_timeout=0.001)
    finally:
        engine._engine_lock.release()


def test_async_engine_rejects_multiple_outputs() -> None:
    fake_engine = _FakeTokenEngine()
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=fake_engine,
        tokenizer=_FakeTokenizer(),
        batch_wait_seconds=0.0,
    )

    async def run():
        return await engine.generate("ab", sampling=SamplingConfig(n=2))

    try:
        asyncio.run(run())
    except ValueError as exc:
        assert "sampling.n == 1" in str(exc)
    else:
        raise AssertionError("expected ValueError")


@pytest.mark.parametrize("requested", [0, -1, 1.5, True])
def test_native_adapter_rejects_invalid_batch_limits(requested):
    with pytest.raises(ValueError, match="positive integer"):
        LFM25AsyncInferenceEngine.from_components(
            engine=_FakeTokenEngine(), tokenizer=_FakeTokenizer(), max_batch_size=requested)


def test_native_adapter_respects_configured_request_limit():
    engine = LFM25AsyncInferenceEngine.from_components(
        engine=_FakeTokenEngine(), tokenizer=_FakeTokenizer(), max_batch_size=128,
        config=InferenceEngineConfig(model_path="", max_num_seqs=16))
    assert engine.max_batch_size == 16


def test_native_adapter_sleep_releases_under_generation_lock():
    class Engine(_FakeTokenEngine):
        def release_inference_buffers(self):
            assert adapter._engine_lock.locked()
            self.released = True

    raw = Engine()
    adapter = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer())
    adapter.sleep()
    assert raw.released
