import pytest
import torch

from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig, flatten_token_batches
from climbing_gym.inference.lfm25_fp8_kv import attend, store_kv
from test_lfm25_cuda import _engine

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9),
    reason='Native FP8 requires SM89+',
)


def cache_tensors(batch, capacity, heads, dim, *, layout='HND'):
    if layout == 'HND':
        k = torch.zeros(batch, heads, capacity, dim, device='cuda', dtype=torch.float8_e4m3fn).transpose(1, 2)
        scales = torch.zeros(batch, heads, capacity, device='cuda', dtype=torch.float32).transpose(1, 2)
    else:
        k = torch.zeros(batch, capacity, heads, dim, device='cuda', dtype=torch.float8_e4m3fn)
        scales = torch.zeros(batch, capacity, heads, device='cuda', dtype=torch.float32)
    return k, torch.zeros_like(k), scales, torch.zeros_like(scales)


@pytest.mark.parametrize('layout', ['NHD', 'HND'])
@pytest.mark.parametrize('dtype', [torch.float16, torch.bfloat16])
@torch.inference_mode()
def test_scaled_storage_handles_strided_prefill_views_zeros_and_outliers(layout, dtype):
    torch.manual_seed(18)
    kc, vc, ks, vs = cache_tensors(5, 79, 3, 64, layout=layout)
    k = torch.randn(2, 37, 3, 64, device='cuda', dtype=dtype)
    v = torch.randn_like(k)
    k[0, 0].zero_()
    v[0, 0].zero_()
    k[1, :, 1] *= 100
    v[1, :, 2] *= 0.0001
    store_kv(k, v, kc[2:4, :37], vc[2:4, :37], ks[2:4, :37], vs[2:4, :37])
    for source, values, scales in ((k, kc, ks), (v, vc, vs)):
        expected_scale = (source.float().abs().amax(-1) / 448).clamp_min(1e-12)
        expected = (source.float() / expected_scale[..., None]).clamp(-448, 448).to(values.dtype)
        torch.testing.assert_close(scales[2:4, :37], expected_scale, atol=0, rtol=0)
        # Reciprocal-based division may pick the other equally near E4M3
        # neighbor at a rounding midpoint. Require an optimal rounding error.
        normalized = source.float() / expected_scale[..., None]
        actual_error = (values[2:4, :37].float() - normalized).abs()
        optimal_error = (expected.float() - normalized).abs()
        assert torch.all(actual_error <= optimal_error + normalized.abs() * 3e-7 + 1e-7)
        assert torch.count_nonzero(values[:2].float()) == 0
        assert torch.count_nonzero(values[2:4, 37:].float()) == 0
        assert torch.count_nonzero(values[4:].float()) == 0


@pytest.mark.parametrize('dim,groups,splits', [(32, 1, 1), (64, 4, 1), (64, 4, 4), (128, 2, 4), (256, 1, 2)])
@torch.inference_mode()
def test_decode_matches_independent_fp32_softmax_with_ragged_lengths_and_inactive_rows(dim, groups, splits):
    torch.manual_seed(92)
    batch, capacity, heads = 4, 277, 2
    kc, vc, ks, vs = cache_tensors(batch, capacity, heads, dim)
    k = torch.randn(batch, 251, heads, dim, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    store_kv(k, v, kc[:, :251], vc[:, :251], ks[:, :251], vs[:, :251])
    q = torch.randn(batch, 1, heads * groups, dim, device='cuda', dtype=k.dtype)
    positions = torch.tensor([0, 36, 250, 250], device='cuda', dtype=torch.int32)
    active = torch.tensor([True, True, True, False], device='cuda')
    actual = attend(q, kc, vc, ks, vs, positions, active=active, config=(64, 4, splits))
    for row, length in enumerate([1, 37, 251]):
        key = (kc[row, :length].float() * ks[row, :length, :, None]).repeat_interleave(groups, 1).permute(1, 2, 0)
        value = (vc[row, :length].float() * vs[row, :length, :, None]).repeat_interleave(groups, 1).permute(1, 0, 2)
        expected = ((q[row, 0].float()[:, None] @ key) * dim**-0.5).softmax(-1) @ value
        torch.testing.assert_close(actual[row, 0].float(), expected[:, 0], atol=0.012, rtol=0.025)
    assert torch.count_nonzero(actual[3]) == 0


@torch.inference_mode()
def test_append_preserves_inactive_histories_and_uses_independent_positions():
    kc, vc, ks, vs = cache_tensors(3, 19, 2, 64)
    k = torch.randn(3, 1, 2, 64, device='cuda', dtype=torch.bfloat16)
    v = torch.randn_like(k)
    positions = torch.tensor([3, 18, 7], device='cuda', dtype=torch.int32)
    active = torch.tensor([True, False, True], device='cuda')
    store_kv(k, v, kc, vc, ks, vs, positions=positions, active=active)
    assert torch.count_nonzero(kc[1].float()) == 0
    assert torch.count_nonzero(vc[1].float()) == 0
    assert torch.count_nonzero(ks[1]) == 0
    for row, pos in [(0, 3), (2, 7)]:
        assert ks[row, pos].min() > 0
        torch.testing.assert_close(kc[row, pos].float() * ks[row, pos, :, None], k[row, 0].float(), atol=.12, rtol=.05)
        assert torch.count_nonzero(ks[row, :pos]) == 0
        assert torch.count_nonzero(ks[row, pos + 1:]) == 0


@pytest.mark.parametrize('fp8_mlp', [False, True])
@torch.inference_mode()
def test_quantized_cache_prefill_repeats_policy_sync_and_clear(fp8_mlp):
    model, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, fp8_mlp=fp8_mlp, fp8_kv=True)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=9, ignore_eos=True, return_logprobs=True)
    flat = flatten_token_batches([[1, 4, 8], [1, 9, 12]])
    repeated = engine.generate(*flat, repeats=2, sampling=sampling)
    entry = next(iter(engine._cuda_decoder.graphs.values()))
    assert entry.cache.k_cache.dtype == torch.float8_e4m3fn
    assert entry.cache.k_cache.stride(2) == engine.config.head_dim
    torch.testing.assert_close(repeated.token_ids[:, 0], repeated.token_ids[:, 1], atol=0, rtol=0)
    ptr = entry.cache.k_cache.data_ptr()
    with torch.no_grad():
        model.lm_head.weight.normal_(std=.03)
    engine.sync_from_model(model)
    after = engine.generate(*flat, repeats=2, sampling=sampling)
    assert entry.cache.k_cache.data_ptr() == ptr
    assert not torch.equal(after.logprobs, repeated.logprobs)
    fresh = type(engine).from_model(model, device='cuda', dtype=torch.bfloat16, max_model_len=256)
    fresh.enable_cuda_graphs(max_graphs=1, fp8_mlp=fp8_mlp, fp8_kv=True)
    expected = fresh.generate(*flat, repeats=2, sampling=sampling)
    torch.testing.assert_close(after.token_ids, expected.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(after.logprobs, expected.logprobs, atol=0, rtol=0)
    engine._cuda_decoder.clear()
    rebuilt = engine.generate(*flat, repeats=2, sampling=sampling)
    torch.testing.assert_close(rebuilt.token_ids, after.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(rebuilt.logprobs, after.logprobs, atol=0, rtol=0)


@torch.inference_mode()
def test_session_logprobs_match_the_actual_compressed_cache_behavior_distribution():
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, fp8_mlp=True, fp8_kv=True)
    prompts = [[1, 4, 8], [1, 9, 12, 17]]
    sampling = LFM25SamplingConfig(temperature=.8, seed=99, max_new_tokens=17,
                                  ignore_eos=True, return_logprobs=True)
    with engine.generation_session(max_batch_size=2, max_new_tokens=17, cache_len=128) as session:
        session.admit(prompts, [sampling] * 2)
        results = sorted(session.poll(steps=19), key=lambda r: r.handle.slot)
    assert len(results) == 2
    tokens = torch.tensor([r.token_ids for r in results], device='cuda')
    recorded = torch.tensor([r.logprobs for r in results], device='cuda')
    with engine.generation_session(max_batch_size=2, max_new_tokens=17, cache_len=128) as session:
        session.admit(prompts, [sampling] * 2)
        replayed = []
        for step in range(17):
            probabilities = session.entry.logits.float().log_softmax(-1)
            replayed.append(probabilities.gather(1, tokens[:, step, None])[:, 0])
            if step < 16:
                session.entry.input_tokens.copy_(tokens[:, step])
                session.entry.active.fill_(True)
                session.entry.graph.replay()
        torch.testing.assert_close(torch.stack(replayed, 1), recorded, atol=2e-6, rtol=1e-6)


@pytest.mark.parametrize('prefill_tokens', [None, 4])
@torch.inference_mode()
def test_quantized_prefill_can_fork_to_the_reference_decode_cache(prefill_tokens):
    _, engine = _engine()
    cache = engine._empty_cache(max_slots=2, max_seq_len=128, fp8_kv=True)
    prefix = engine.prefill(*flatten_token_batches([[1, 4, 8], [1, 9]]), cache=cache,
                            max_prefill_tokens=prefill_tokens)
    state = engine.fork(prefix, repeats=2)
    slots = torch.tensor([0, 0, 1, 1], device='cuda')
    for values, scales, actual in [(cache.k_cache, cache.k_scales, state.cache.k_cache),
                                   (cache.v_cache, cache.v_scales, state.cache.v_cache)]:
        expected = (values.float() * scales[..., None]).to(engine.dtype).index_select(1, slots)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)
    assert state.cache.k_scales is None
    torch.testing.assert_close(state.cache.seq_lens, cache.seq_lens[slots], atol=0, rtol=0)
