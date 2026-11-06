import asyncio
from dataclasses import replace

import pytest
import torch

from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig, flatten_token_batches
from test_lfm25_cuda import _engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@torch.inference_mode()
def test_sampler_head_matches_independent_full_softmax_including_sample_outside_head():
    from climbing_gym.inference.lfm25_sampling import behavior_topk, sample_logits
    torch.manual_seed(73)
    logits = torch.randn(32, 65536, dtype=torch.bfloat16, device='cuda')
    sampled, lp = sample_logits(logits, temperature=1., seed=19, step=0, return_logprobs=True)
    ids, probs = behavior_topk(logits, sampled, lp, 32)
    expected = logits.float().log_softmax(-1).gather(1, ids)
    torch.testing.assert_close(probs, expected, atol=2e-6, rtol=1e-6)
    assert bool((~(ids == sampled[:, None]).any(-1)).any())


@torch.inference_mode()
def test_session_head_survives_stops_refills_and_continuation_without_changing_draws():
    _, engine = _engine()
    engine.enable_cuda_graphs(adaptive_batching=True, fp8_kv=True, fp8_mlp=True, fp8_lm_head=True)
    samples = [LFM25SamplingConfig(max_new_tokens=n, ignore_eos=True, seed=51+i) for i, n in enumerate((2, 5, 9))]
    prompts = [[1, 3], [1, 5], [1, 9]]

    def collect(k):
        result = {}
        with engine.generation_session(max_batch_size=32, max_new_tokens=10, behavior_topk=k) as s:
            s.admit(prompts, [replace(p, behavior_topk=k) for p in samples], slots=[0, 29, 31])
            while len(result) < 3:
                for row in s.poll(steps=2):
                    result[row.prompt_token_ids] = row
                    s.release(row.handle)
        return result

    before, after = collect(0), collect(32)
    for key, output in after.items():
        assert output.token_ids == before[key].token_ids
        assert output.logprobs == before[key].logprobs
        head = output.behavior_topk
        assert head.token_ids.shape == (len(output.token_ids), 32)
        assert bool(torch.isfinite(head.logprobs).all())
        for token, logp, ids, lp in zip(output.token_ids, output.logprobs, head.token_ids, head.logprobs):
            found = ids == token
            if bool(found.any()):
                assert lp[found].item() == pytest.approx(logp, abs=1e-6)
    sample = replace(samples[0], behavior_topk=32)
    with engine.generation_session(max_batch_size=2, max_new_tokens=10, behavior_topk=32) as s:
        s.admit([prompts[0]], [sample])
        first = s.poll(steps=3)[0]
        saved = first.behavior_topk.logprobs.clone()
        s.resume([first.handle], [first.handle.prefix_token_ids+(15,)], [sample])
        resumed = s.poll(steps=3)[0]
        s.release(resumed.handle)
        s.admit([[1, 20]], [sample])
        assert len(s.poll(steps=3)) == 1
        torch.testing.assert_close(first.behavior_topk.logprobs, saved, atol=0, rtol=0)


@torch.inference_mode()
def test_direct_generation_exposes_behavior_head():
    _, engine = _engine()
    engine.enable_cuda_graphs(fp8_mlp=True, fp8_kv=True, fp8_lm_head=True)
    args = flatten_token_batches([[1, 9], [1, 4, 7]], device='cuda')
    sample = LFM25SamplingConfig(max_new_tokens=7, seed=35, ignore_eos=True)
    before = engine.generate(*args, repeats=1, sampling=sample)
    after = engine.generate(*args, repeats=1, sampling=replace(sample, behavior_topk=32))
    torch.testing.assert_close(before.token_ids, after.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(before.logprobs, after.logprobs, atol=0, rtol=0)
    assert after.topk_token_ids.shape == (2, 1, 7, 32)


def test_async_chains_preserve_behavior_metadata_through_refill():
    from climbing_gym.inference import LFM25AsyncInferenceEngine, SamplingConfig
    from test_lfm25_async_engine import _FakeTokenizer
    _, raw = _engine()
    raw.enable_cuda_graphs(adaptive_batching=True)
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer(), max_batch_size=2)
    sample = SamplingConfig(temperature=1., max_tokens=3, logprobs=1,
                            extra_params={'behavior_topk': 32, 'ignore_eos': True})
    results = asyncio.run(engine.generate_chains([([1, i+2], sample) for i in range(5)],
        on_result=lambda i, r: None, max_new_tokens=3, poll_interval=2))
    assert len(results) == 5
    assert engine.last_session_stats['slot_refills'] == 3
    assert all(chain[0].outputs[0].behavior_topk.token_ids.shape == (3, 32) for chain in results)


def test_ordinary_async_outputs_release_gpu_heads_and_own_cpu_slices():
    from climbing_gym.inference import LFM25AsyncInferenceEngine, SamplingConfig
    from test_lfm25_async_engine import _FakeTokenizer
    _, raw = _engine()
    raw.enable_cuda_graphs(fp8_mlp=True,fp8_kv=True,fp8_lm_head=True,adaptive_batching=True)
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw,tokenizer=_FakeTokenizer(),max_batch_size=8)
    sample = SamplingConfig(temperature=1.,max_tokens=8,logprobs=1,
                           extra_params={'behavior_topk':32,'ignore_eos':True})
    results = asyncio.run(engine.generate_many([[1,3],[1,7,8],[1,15]],sampling=sample))
    for result in results:
        head = result.outputs[0].behavior_topk
        assert result.raw_output is None
        assert head.token_ids.device.type == head.logprobs.device.type == 'cpu'
        assert head.token_ids.shape == head.logprobs.shape == (8,32)
        assert head.token_ids.untyped_storage().nbytes() == 8*32*4
        assert head.logprobs.untyped_storage().nbytes() == 8*32*4
        assert bool(torch.isfinite(head.logprobs).all())
    engine.close()
