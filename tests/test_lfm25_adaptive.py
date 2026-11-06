from dataclasses import replace

import pytest
import torch

from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig, flatten_token_batches
from test_lfm25_cuda import _engine

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


@pytest.mark.parametrize('fp8_kv,fp8_mlp', [(False, False), (True, False), (True, True)])
@torch.inference_mode()
def test_compact_capture_and_replay_share_history_and_preserve_unselected_slots(fp8_kv, fp8_mlp):
    if (fp8_kv or fp8_mlp) and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, adaptive_batching=True, fp8_kv=fp8_kv, fp8_mlp=fp8_mlp)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=20, ignore_eos=True)
    with engine.generation_session(max_batch_size=32, max_new_tokens=20, cache_len=128) as session:
        session.admit([[1, 2 + i, 40 + i] for i in range(32)],
                      [replace(sampling, max_new_tokens=1) if i < 29 else sampling for i in range(32)])
        assert len(session.poll(steps=2)) == 29
        entry, cache = session.entry, session.entry.cache
        fields = [cache.k_cache, cache.v_cache, cache.conv_state, cache.seq_lens, entry.logits]
        if fp8_kv:
            fields += [cache.k_scales, cache.v_scales]
        saved = [t.clone() for t in fields]
        graph, batch = engine._cuda_decoder._select_graph(entry, [29, 30, 31])
        assert batch == 16 and len(engine._cuda_decoder.graphs) == 1
        for actual, before in zip(fields, saved):
            assert torch.equal(actual.contiguous().view(torch.uint8), before.contiguous().view(torch.uint8))
        compact = entry.compact_graphs[16]
        assert any(t is cache.k_cache for t in compact.buffers)
        assert any(t is cache.conv_state for t in compact.buffers)
        # A separate 16-row cache with the same histories is an independent
        # reference for slot indirection, positions, and convolution updates.
        ref = engine._empty_cache(max_slots=16, max_seq_len=128, fp8_kv=fp8_kv)
        for source, target in [(cache.k_cache, ref.k_cache), (cache.v_cache, ref.v_cache),
                               (cache.conv_state, ref.conv_state)]:
            target[:, :3].copy_(source[:, 29:32])
        ref.seq_lens[:3].copy_(cache.seq_lens[29:32])
        if fp8_kv:
            ref.k_scales[:, :3].copy_(cache.k_scales[:, 29:32])
            ref.v_scales[:, :3].copy_(cache.v_scales[:, 29:32])
        tokens = torch.zeros(16, device='cuda', dtype=torch.long)
        tokens[:3].copy_(entry.input_tokens[29:32])
        active = torch.arange(16, device='cuda') < 3
        expected = engine._cuda_decoder._forward(ref, tokens, active)
        entry.active.copy_(session.pending)
        graph.replay()
        torch.testing.assert_close(entry.logits[29:32], expected[:3], atol=.008, rtol=.008)
        torch.testing.assert_close(cache.conv_state[:, 29:32], ref.conv_state[:, :3], atol=0, rtol=0)
        torch.testing.assert_close(cache.seq_lens[29:32], ref.seq_lens[:3], atol=0, rtol=0)
        for actual, before in zip(fields, saved):
            # Row zero is also the harmless read target for padding rows.
            axis = 0 if actual is cache.seq_lens or actual is entry.logits else 1
            old = before[:29] if axis == 0 else before[:, :29]
            now = actual[:29] if axis == 0 else actual[:, :29]
            assert torch.equal(now.contiguous().view(torch.uint8), old.contiguous().view(torch.uint8))


@pytest.mark.parametrize('fp8_kv', [False, True])
@torch.inference_mode()
def test_sparse_sessions_resume_refill_and_keep_request_state_across_graph_changes(fp8_kv):
    if fp8_kv and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, adaptive_batching=True, fp8_kv=fp8_kv)
    sample = LFM25SamplingConfig(temperature=.8, max_new_tokens=20, ignore_eos=True, seed=91)
    with engine.generation_session(max_batch_size=32, max_new_tokens=20, cache_len=128) as session:
        session.admit([[1, 4], [1, 8], [1, 17], [1, 31]],
                      [replace(sample, max_new_tokens=2), replace(sample, max_new_tokens=7), sample, sample],
                      slots=[0, 8, 17, 31])
        first = session.poll(steps=2)[0]
        assert first.handle.slot == 0
        graph = session.entry.compact_graphs[16].graph
        peers = session.entry.cache.conv_state[:, [8, 17, 31]].clone()
        lengths = session.entry.cache.seq_lens[[8, 17, 31]].clone()
        session.resume([first.handle], [first.handle.prefix_token_ids + (40, 41)],
                       [replace(sample, max_new_tokens=3)])
        torch.testing.assert_close(session.entry.cache.conv_state[:, [8, 17, 31]], peers, atol=0, rtol=0)
        torch.testing.assert_close(session.entry.cache.seq_lens[[8, 17, 31]], lengths, atol=0, rtol=0)
        resumed = session.poll(steps=3)[0]
        assert resumed.handle.slot == 0 and len(resumed.token_ids) == 3
        session.release(resumed.handle)
        session.admit([[1, 22, 23]], [replace(sample, max_new_tokens=4)], slots=[0])
        results = []
        for _ in range(12):
            results.extend(session.poll(steps=3))
            if len(results) == 4:
                break
        assert len(results) == 4
        assert session.entry.compact_graphs[16].graph is graph
        assert session.stats['slot_refills'] == 1
        assert session.stats['decode_row_steps'] < session.stats['reserved_row_steps']
        assert session.stats['decode_batches'] == {'16': session.stats['decode_steps']}
        for result in results:
            session.release(result.handle)
        assert session.poll(steps=1) == []


@torch.inference_mode()
def test_fp8_head_refresh_and_clear_preserve_graph_addresses_and_behavior_logprobs():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    model, engine = _engine()
    engine.enable_cuda_graphs(fp8_lm_head=True, max_graphs=1)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=5, ignore_eos=True)
    flat = flatten_token_batches([[1, 8, 9], [1, 10]])
    before = engine.generate(*flat, repeats=1, sampling=sampling)
    entry = next(iter(engine._cuda_decoder.graphs.values()))
    head = engine._cuda_decoder.fp8_weights[id(engine.weights.lm_head_weight)]
    pointer = head.values.data_ptr()
    model.lm_head.weight.normal_(std=.04)
    engine.sync_from_model(model)
    after = engine.generate(*flat, repeats=1, sampling=sampling)
    assert head.values.data_ptr() == pointer
    assert not torch.equal(before.logprobs, after.logprobs)
    fresh = type(engine).from_model(model, device='cuda', dtype=torch.bfloat16, max_model_len=256)
    fresh.enable_cuda_graphs(fp8_lm_head=True, max_graphs=1)
    expected = fresh.generate(*flat, repeats=1, sampling=sampling)
    torch.testing.assert_close(after.token_ids, expected.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(after.logprobs, expected.logprobs, atol=0, rtol=0)
    engine._cuda_decoder.clear()
    rebuilt = engine.generate(*flat, repeats=1, sampling=sampling)
    torch.testing.assert_close(rebuilt.logprobs, after.logprobs, atol=0, rtol=0)


@pytest.mark.parametrize('fp8_kv', [False, True])
@torch.inference_mode()
def test_direct_generation_shrinks_after_stops_and_preserves_greedy_results(fp8_kv):
    if fp8_kv and torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    _, engine = _engine()
    flat = flatten_token_batches([[1, 2, 40 + i] for i in range(32)])
    engine.enable_cuda_graphs(max_graphs=1, fp8_kv=fp8_kv)
    first = engine.generate(*flat, repeats=1, sampling=LFM25SamplingConfig(temperature=0, max_new_tokens=1))
    stops = tuple(first.token_ids[:29, 0, 0].tolist())
    assert not set(first.token_ids[29:, 0, 0].tolist()).intersection(stops)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=65, ignore_eos=True, stop_token_ids=stops)
    reference = engine.generate(*flat, repeats=1, sampling=sampling)
    engine.enable_cuda_graphs(max_graphs=1, fp8_kv=fp8_kv, adaptive_batching=True)
    actual = engine.generate(*flat, repeats=1, sampling=sampling)
    torch.testing.assert_close(actual.lengths, reference.lengths, atol=0, rtol=0)
    torch.testing.assert_close(actual.token_ids, reference.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(actual.finish_reasons, reference.finish_reasons, atol=0, rtol=0)
    assert 16 in next(iter(engine._cuda_decoder.graphs.values())).compact_graphs


@torch.inference_mode()
def test_actual_behavior_logprobs_follow_batch_changes_with_fp8_head():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, adaptive_batching=True, fp8_mlp=True,
                              fp8_kv=True, fp8_lm_head=True)
    prompts = [[1, 2 + i, 40 + i] for i in range(32)]
    sample = LFM25SamplingConfig(temperature=.8, seed=91, max_new_tokens=65, ignore_eos=True)
    limits = [1] * 24 + [65] * 8
    samplings = [replace(sample, max_new_tokens=n, seed=91 + i) for i, n in enumerate(limits)]
    with engine.generation_session(max_batch_size=32, max_new_tokens=65, cache_len=128) as session:
        session.admit(prompts, samplings)
        outputs = []
        while len(outputs) < 32:
            outputs.extend(session.poll(steps=32))
        assert session.stats['decode_batches'] == {'32': 32, '16': 64}
    outputs.sort(key=lambda r: r.handle.slot)
    tokens = torch.zeros(32, 65, device='cuda', dtype=torch.long)
    for i, result in enumerate(outputs):
        tokens[i, :len(result.token_ids)] = torch.tensor(result.token_ids, device='cuda')
    # Independently teacher-force the saved trajectory, with the original
    # projection batch at each step; compute log-softmax outside the sampler.
    with engine.generation_session(max_batch_size=32, max_new_tokens=65, cache_len=128) as session:
        session.admit(prompts, samplings)
        replayed = [[] for _ in range(32)]
        graph = session.entry.graph
        for step in range(65):
            live = [i for i, n in enumerate(limits) if step < n]
            if step:
                if step == 32:
                    graph, _ = engine._cuda_decoder._select_graph(session.entry, live)
                session.entry.input_tokens.copy_(tokens[:, step - 1])
                session.entry.active.copy_(torch.tensor([step < n for n in limits], device='cuda'))
                graph.replay()
            values = session.entry.logits.float().log_softmax(-1).gather(1, tokens[:, step, None])[:, 0].tolist()
            for row in live:
                replayed[row].append(values[row])
        for row, result in enumerate(outputs):
            torch.testing.assert_close(torch.tensor(replayed[row]), torch.tensor(result.logprobs), atol=2e-6, rtol=1e-6)


def test_async_chains_cross_graph_sizes_with_continuation_and_refill():
    import asyncio
    from climbing_gym.inference import LFM25AsyncInferenceEngine, SamplingConfig
    from test_lfm25_async_engine import _FakeTokenizer

    _, raw = _engine()
    raw.enable_cuda_graphs(max_graphs=1, adaptive_batching=True)
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer(), max_batch_size=32)
    sample = SamplingConfig(temperature=0, max_tokens=8, logprobs=1, extra_params={'ignore_eos': True})
    requests = [([1, 20 + i], replace(sample, max_tokens=1 if i < 24 else 8)) for i in range(40)]
    phases = [0] * 40

    def advance(index, result):
        phases[index] += 1
        assert len(result.completion_logprobs) == len(result.token_ids)
        if index % 3 == 0 and phases[index] == 1:
            return result.prompt_token_ids + result.token_ids + (90,), replace(sample, max_tokens=3)
        return None

    results = asyncio.run(engine.generate_chains(requests, on_result=advance, max_new_tokens=8, poll_interval=2))
    assert [len(chain) for chain in results] == [2 if i % 3 == 0 else 1 for i in range(40)]
    assert engine.last_session_stats['resumed_requests'] == 14
    assert engine.last_session_stats['admitted_requests'] == 40
    assert {'32', '16'} <= engine.last_session_stats['decode_batches'].keys()
    assert raw._cuda_decoder._active_session is None


def test_compact_recapture_survives_worker_stream_reuse(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys

    script = tmp_path / 'recapture.py'
    script.write_text('''
from concurrent.futures import ThreadPoolExecutor
from test_lfm25_cuda import _engine
from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig
_, engine = _engine()
engine.enable_cuda_graphs(max_graphs=1, adaptive_batching=True)
sample = LFM25SamplingConfig(temperature=0, max_new_tokens=1, ignore_eos=True)
def collect():
    with engine.generation_session(max_batch_size=32, max_new_tokens=1, cache_len=128) as session:
        session.admit([[1, 4, 8]], [sample], slots=[31])
        assert len(session.poll(steps=2)) == 1
        assert 16 in session.entry.compact_graphs
    engine._cuda_decoder.clear()
for i in range(64):
    with ThreadPoolExecutor(max_workers=1) as worker:
        worker.submit(collect).result()
print('completed 64 recaptures')
''')
    repo = Path(__file__).resolve().parents[1]
    env = dict(os.environ, PYTHONPATH=os.pathsep.join((str(repo / 'src'), str(repo / 'tests'))))
    result = subprocess.run([sys.executable, str(script)], env=env, capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr
    assert 'completed 64 recaptures' in result.stdout


@torch.inference_mode()
def test_indexed_long_history_uses_split_attention_with_inactive_padding():
    if torch.cuda.get_device_capability() < (8, 9):
        pytest.skip('FP8 requires SM89+')
    from climbing_gym.inference.lfm25_fp8_kv import decode, store_kv
    from test_lfm25_fp8_kv import cache_tensors

    torch.manual_seed(31)
    kc, vc, ks, vs = cache_tensors(32, 1057, 2, 64)
    history_k = torch.randn(32, 1057, 2, 64, device='cuda', dtype=torch.bfloat16)
    history_v = torch.randn_like(history_k)
    store_kv(history_k, history_v, kc, vc, ks, vs)
    preserved = [t[[0, 4, 17]].clone() for t in (kc, vc, ks, vs)]
    q = torch.randn(16, 1, 8, 64, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(16, 1, 2, 64, device='cuda', dtype=q.dtype)
    v = torch.randn_like(k)
    slots = torch.tensor([31, 7, 23] + [-1] * 13, device='cuda', dtype=torch.int32)
    positions = torch.tensor([1056, 511, 13] + [0] * 13, device='cuda', dtype=torch.int32)
    active = torch.arange(16, device='cuda') < 3
    actual = decode(q, k, v, kc, vc, ks, vs, positions, active=active, slot_ids=slots)
    for row, (slot, length) in enumerate([(31, 1057), (7, 512), (23, 14)]):
        key = (kc[slot, :length].float() * ks[slot, :length, :, None]).repeat_interleave(4, 1).permute(1, 2, 0)
        value = (vc[slot, :length].float() * vs[slot, :length, :, None]).repeat_interleave(4, 1).permute(1, 0, 2)
        expected = ((q[row, 0].float()[:, None] @ key) / 8).softmax(-1) @ value
        torch.testing.assert_close(actual[row, 0].float(), expected[:, 0], atol=.003, rtol=.02)
    assert torch.count_nonzero(actual[3:]) == 0
    for values, before in zip((kc, vc, ks, vs), preserved):
        torch.testing.assert_close(values[[0, 4, 17]].float(), before.float(), atol=0, rtol=0)
