from dataclasses import replace

import pytest
import torch

from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig, flatten_token_batches
from test_lfm25_cuda import _engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
KV_PRECISIONS = [False, pytest.param(True, marks=pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason='Native FP8 requires SM89+',
))]


def collect(session, count, steps=1):
    results = []
    while len(results) < count:
        results.extend(session.poll(steps=steps))
    return results


@pytest.mark.parametrize('fp8_kv', KV_PRECISIONS)
@torch.inference_mode()
def test_resume_preserves_pending_token_conv_state_and_exact_prefix(fp8_kv):
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, fp8_kv=fp8_kv)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=5, ignore_eos=True)
    with engine.generation_session(max_batch_size=2, max_new_tokens=20, cache_len=128) as session:
        session.admit([[1, 2, 3], [1, 8, 9, 10]], [sampling, replace(sampling, max_new_tokens=20)])
        first = collect(session, 1, steps=7)[0]
        assert first.handle.slot == 0 and first.finish_reason == 3 and len(first.token_ids) == 5
        assert session.entry.cache.seq_lens[0].item() == 7
        assert session.entry.cache.seq_lens[1].item() == 10
        saved_conv = session.entry.cache.conv_state[:, 0].clone()
        # Finished rows must survive several graph replays while a peer runs.
        session.poll(steps=3)
        torch.testing.assert_close(session.entry.cache.conv_state[:, 0], saved_conv, atol=0, rtol=0)
        peer_conv = session.entry.cache.conv_state[:, 1].clone()
        peer_len = session.entry.cache.seq_lens[1].clone()
        prompt = first.handle.prefix_token_ids + (40, 41, 42)
        graph = session.entry.graph
        session.resume([first.handle], [prompt], [sampling])
        assert session.entry.graph is graph
        assert session.entry.cache.seq_lens[0].item() == len(prompt)
        torch.testing.assert_close(session.entry.cache.conv_state[:, 1], peer_conv, atol=0, rtol=0)
        torch.testing.assert_close(session.entry.cache.seq_lens[1], peer_len, atol=0, rtol=0)
        # Independent full-prefix prefill validates KV, convolution, positions,
        # and the final pending token rather than reconstructing via resume.
        prefix = engine.prefill(*flatten_token_batches([prompt]), max_cache_len=128)
        torch.testing.assert_close(session.entry.logits[0], prefix.next_token_logits[0], atol=0.08, rtol=0.03)
        torch.testing.assert_close(session.entry.cache.conv_state[:, 0], prefix.cache.conv_state[:, 0], atol=0.004, rtol=0.04)
        with pytest.raises(ValueError, match="stale"):
            session.resume([first.handle], [prompt], [sampling])
        final = collect(session, 1)[0]
        assert final.prompt_token_ids == prompt
        assert final.token_ids[0] == prefix.next_token_logits[0].argmax().item()
        assert session.stats['prefilled_tokens'] == 7
        assert session.stats['appended_tokens'] == 4


@pytest.mark.parametrize('fp8_kv', KV_PRECISIONS)
@torch.inference_mode()
def test_refill_and_poll_interval_do_not_change_request_rng_or_stops(fp8_kv):
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, fp8_kv=fp8_kv)
    sampling = LFM25SamplingConfig(temperature=0.8, max_new_tokens=13, seed=97, ignore_eos=True)

    def run(interval):
        with engine.generation_session(max_batch_size=2, max_new_tokens=20, cache_len=128) as session:
            session.admit([[1, 4, 8], [1, 9]], [sampling, replace(sampling, max_new_tokens=2, seed=98)])
            results = {}
            refilled = False
            for _ in range(40):
                for result in session.poll(steps=interval):
                    key = result.prompt_token_ids
                    results[key] = result
                    session.release(result.handle)
                    if key == (1, 9):
                        assert (1, 4, 8) not in results  # refill before the long peer finishes
                        session.admit([[1, 13, 14, 15]], [replace(sampling, max_new_tokens=3, seed=99)], slots=[1])
                        refilled = True
                if len(results) == 3:
                    assert refilled and session.stats['slot_refills'] == 1
                    return results
            raise AssertionError("session failed to drain")

    first, second = run(1), run(4)
    for key in first:
        assert first[key].token_ids == second[key].token_ids
        assert first[key].logprobs == second[key].logprobs
    stop = first[(1, 4, 8)].token_ids[0]
    with engine.generation_session(max_batch_size=2, max_new_tokens=20, cache_len=128) as session:
        session.admit([[1, 4, 8]], [replace(sampling, stop_token_ids=(stop,))])
        result = collect(session, 1, steps=17)[0]
        assert result.token_ids == (stop,) and result.finish_reason == 2
        assert result.logprobs == first[(1, 4, 8)].logprobs[:1]
        # Empty-suffix continuation still must consume the pending token.
        session.resume([result.handle], [result.handle.prefix_token_ids], [replace(sampling, max_new_tokens=1)])
        assert session.entry.cache.seq_lens[0].item() == 4
        assert len(collect(session, 1, steps=5)[0].token_ids) == 1


@torch.inference_mode()
def test_session_rejects_invalid_reuse_and_sync_and_reuses_graph():
    model, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1)
    sampling = LFM25SamplingConfig(max_new_tokens=1, ignore_eos=True)
    with engine.generation_session(max_batch_size=2, max_new_tokens=2, cache_len=128) as session:
        graph = session.entry.graph
        with pytest.raises(RuntimeError, match="session"):
            engine.sync_from_model(model)
        with pytest.raises(RuntimeError, match="session"):
            engine.enable_cuda_graphs()
        with pytest.raises(RuntimeError, match="session"):
            engine.generate(*flatten_token_batches([[1]]), repeats=1, sampling=sampling)
        with pytest.raises(ValueError, match="capacity"):
            session.admit([[1] * 128], [sampling])
        assert all(r is None for r in session.requests)
        session.admit([[1, 2]], [sampling])
        result = collect(session, 1)[0]
        with pytest.raises(ValueError, match="stale"):
            session.release(replace(result.handle, prefix_token_ids=(1,)))
        with pytest.raises(ValueError, match="exact cached"):
            session.resume([result.handle], [[1, 3, 4]], [sampling])
        session.release(result.handle)
        with pytest.raises(ValueError, match="stale"):
            session.release(result.handle)
    with pytest.raises(RuntimeError, match="closed"):
        session.resume([result.handle], [result.handle.prefix_token_ids], [sampling])
    with engine.generation_session(max_batch_size=2, max_new_tokens=2, cache_len=128) as fresh:
        assert fresh.entry.graph is graph
        engine.disable_cuda_graphs()
        with pytest.raises(RuntimeError, match="closed"):
            fresh.poll()
    engine.sync_from_model(model)


@torch.inference_mode()
def test_row_sampler_is_independent_of_slot_and_returns_raw_logprobs():
    from climbing_gym.inference.lfm25_sampling import RowSampler
    torch.manual_seed(31)
    logits = torch.randn(3, 513, device="cuda", dtype=torch.bfloat16)
    seeds = torch.tensor([42, 999, 73], device="cuda")
    steps = torch.tensor([0, 8191, 4096], device="cuda", dtype=torch.int32)
    temperatures = torch.tensor([0., 0.7, 1.], device="cuda")
    sampler = RowSampler(logits)
    tokens, logprobs = sampler.sample(logits, seeds, steps, temperatures)
    tokens, logprobs = tokens.clone(), logprobs.clone()
    permutation = torch.tensor([2, 0, 1], device="cuda")
    actual, actual_lp = sampler.sample(logits[permutation], seeds[permutation], steps[permutation], temperatures[permutation])
    assert torch.equal(actual, tokens[permutation])
    assert torch.equal(actual_lp, logprobs[permutation])
    expected = logits.float().log_softmax(-1).gather(1, tokens[:, None])[:, 0]
    torch.testing.assert_close(logprobs, expected, atol=2e-6, rtol=1e-5)


@torch.inference_mode()
def test_row_sampler_preserves_the_full_categorical_distribution():
    from climbing_gym.inference.lfm25_sampling import RowSampler
    probabilities = torch.tensor([0.1, 0.2, 0.3, 0.4], device='cuda')
    logits = probabilities.log().expand(8192, -1).contiguous()
    sampler = RowSampler(logits)
    seeds = torch.arange(8192, device='cuda', dtype=torch.int64) + 20260912
    tokens, _ = sampler.sample(logits, seeds, torch.full_like(seeds, 8191),
                               torch.ones(8192, device='cuda'))
    observed = torch.bincount(tokens, minlength=4).float() / 8192
    torch.testing.assert_close(observed, probabilities, atol=0.025, rtol=0)


@pytest.mark.parametrize('fp8_kv', KV_PRECISIONS)
@torch.inference_mode()
def test_text_stop_states_reset_at_phase_boundary_and_preserve_stop_token(fp8_kv):
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, fp8_kv=fp8_kv)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=4, ignore_eos=True)
    with engine.generation_session(max_batch_size=1, max_new_tokens=4, cache_len=128) as session:
        session.admit([[1, 8, 9]], [sampling])
        reference = collect(session, 1)[0]
        session.release(reference.handle)
        table = torch.zeros((2, engine.config.vocab_size), dtype=torch.int32)
        table[0, reference.token_ids[0]] = 1
        table[1, reference.token_ids[1]] = 65536
        session.admit([[1, 8, 9]], [replace(sampling, stop_table=table)])
        actual = collect(session, 1, steps=7)[0]
        assert actual.token_ids == reference.token_ids[:2]
        assert actual.finish_reason == 2
        session.resume([actual.handle], [actual.handle.prefix_token_ids], [replace(sampling, stop_table=table)])
        assert session.stop_states[0].item() == 0


@pytest.mark.parametrize('fp8_kv', KV_PRECISIONS)
def test_async_chains_resume_refill_fresh_context_and_cleanup_on_callback_error(fp8_kv):
    import asyncio
    from climbing_gym.inference import LFM25AsyncInferenceEngine, SamplingConfig
    from test_lfm25_async_engine import _FakeTokenizer

    _, raw = _engine()
    raw.enable_cuda_graphs(max_graphs=1, fp8_kv=fp8_kv)
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer(), max_batch_size=2)
    sampling = SamplingConfig(temperature=0, max_tokens=3, logprobs=1, extra_params={'ignore_eos': True})
    requests = [([1, 2 + i], sampling) for i in range(5)]
    phases = [0] * 5

    def advance(index, result):
        assert result.finished and len(result.completion_logprobs) == 3
        phases[index] += 1
        if phases[index] == 1:
            return result.prompt_token_ids + result.token_ids + (40,), sampling
        if phases[index] == 2:
            return [1, 30 + index], sampling
        return None

    results = asyncio.run(engine.generate_chains(requests, on_result=advance, max_new_tokens=3, poll_interval=2))
    assert [len(chain) for chain in results] == [3] * 5
    assert engine.last_session_stats['resumed_requests'] == 5
    assert engine.last_session_stats['admitted_requests'] == 10
    assert engine.last_session_stats['prefilled_tokens'] == 20
    assert raw._cuda_decoder._active_session is None

    def fail(*_):
        raise ValueError('callback failed')

    with pytest.raises(ValueError, match='callback failed'):
        asyncio.run(engine.generate_chains(requests, on_result=fail, max_new_tokens=3))
    assert raw._cuda_decoder._active_session is None
    assert engine._engine_lock.acquire(blocking=False)
    engine._engine_lock.release()


def test_async_chain_cancellation_drains_the_worker_and_cache_lease():
    import asyncio
    from climbing_gym.inference import LFM25AsyncInferenceEngine, SamplingConfig
    from test_lfm25_async_engine import _FakeTokenizer

    _, raw = _engine()
    raw.enable_cuda_graphs(max_graphs=1)
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer(), max_batch_size=2)
    sampling = SamplingConfig(max_tokens=1, extra_params={'ignore_eos': True})

    async def run():
        loop = asyncio.get_running_loop()
        task = None
        cancellation_requested = False

        def advance(index, result):
            nonlocal cancellation_requested
            if not cancellation_requested:
                cancellation_requested = True
                loop.call_soon_threadsafe(task.cancel)
            return [1, 7], sampling

        task = asyncio.create_task(engine.generate_chains(
            [([1, 2], sampling)], on_result=advance, max_new_tokens=1, poll_interval=1))
        with pytest.raises(asyncio.CancelledError):
            await task
        assert raw._cuda_decoder._active_session is None
        assert engine._engine_lock.acquire(blocking=False)
        engine._engine_lock.release()

    asyncio.run(run())


@torch.inference_mode()
def test_capacity_limit_does_not_overrun_finished_rows_during_delayed_poll():
    _, raw = _engine()
    raw.enable_cuda_graphs(max_graphs=1)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=1, ignore_eos=True)
    with raw.generation_session(max_batch_size=2, max_new_tokens=2, cache_len=128) as session:
        session.admit([[1] * 127], [sampling])
        result = collect(session, 1, steps=35)[0]
        assert session.entry.cache.seq_lens[0].item() == 127
        assert len(result.token_ids) == 1
        with pytest.raises(ValueError, match='capacity'):
            session.resume([result.handle], [result.handle.prefix_token_ids], [sampling])
