from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import subprocess
import sys

import pytest
import torch
import torch.nn.functional as F

from climbing_gym.inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig, _rms_norm, flatten_token_batches
from climbing_gym.models.lfm2 import LFM2Config, LFM2Model

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("dtype", (torch.float32, torch.bfloat16, torch.float16))
@torch.inference_mode()
def test_fused_decode_kernels_match_reference(dtype):
    pytest.importorskip("triton")
    from climbing_gym.inference.lfm25_cuda_kernels import conv_update, linear, prepare_qkv, rms_norm, swiglu

    torch.manual_seed(17)
    batch, width = 3, 2048
    x = torch.randn(batch, width, device="cuda", dtype=dtype)
    residual = torch.randn_like(x)
    weight = torch.randn(width, device="cuda", dtype=dtype)
    output, summed = rms_norm(x, weight, 1e-5, residual)
    atol = 2e-5 if dtype == torch.float32 else 0.04 if dtype == torch.bfloat16 else 0.004
    rtol = 1e-5 if dtype == torch.float32 else 0.01
    torch.testing.assert_close(summed, x + residual, atol=0, rtol=0)
    torch.testing.assert_close(output, _rms_norm(x + residual, weight, 1e-5), atol=atol, rtol=rtol)
    projected = torch.randn(batch, 2 * 10752, device="cuda", dtype=dtype)
    gate, up = projected.chunk(2, dim=-1)
    torch.testing.assert_close(swiglu(projected), F.silu(gate) * up, atol=atol, rtol=rtol)
    bcx = torch.randn(batch, width * 3, device="cuda", dtype=dtype)
    state = torch.randn(batch, width, 3, device="cuda", dtype=dtype)
    conv_weight = torch.randn(width, 3, device="cuda", dtype=dtype)
    expected_state = state.roll(-1, -1)
    b, c, projection = bcx.chunk(3, dim=-1)
    expected_state[:, :, -1] = b * projection
    expected_output = c * (expected_state * conv_weight[None]).sum(-1)
    torch.testing.assert_close(conv_update(bcx, conv_weight, state), expected_output, atol=atol, rtol=rtol)
    torch.testing.assert_close(state, expected_state, atol=0, rtol=0)
    matrix = torch.randn(512, 10752, device="cuda", dtype=dtype)
    vector = torch.randn(1, 10752, device="cuda", dtype=dtype)
    torch.testing.assert_close(linear(vector, matrix), F.linear(vector, matrix), atol=atol * 16, rtol=rtol)

    qkv = torch.randn(batch, 3072, device="cuda", dtype=dtype)
    q_weight = torch.randn(64, device="cuda", dtype=dtype)
    k_weight = torch.randn(64, device="cuda", dtype=dtype)
    positions = torch.tensor([3, 27, 101], device="cuda", dtype=torch.int32)
    freq = torch.arange(128, device="cuda")[:, None] / (1e7 ** (torch.arange(0, 64, 2, device="cuda") / 64))
    cos, sin = freq.cos(), freq.sin()
    q, k, v = prepare_qkv(qkv, q_weight, k_weight, cos, sin, positions, 32, 8, 64, 1e-5)
    from climbing_gym.inference.lfm25_engine import _apply_rotary
    expected_q = _rms_norm(qkv[:, :2048].reshape(batch, 32, 64), q_weight, 1e-5)
    expected_k = _rms_norm(qkv[:, 2048:2560].reshape(batch, 8, 64), k_weight, 1e-5)
    torch.testing.assert_close(q[:, 0], _apply_rotary(expected_q, cos[positions.long(), None], sin[positions.long(), None]), atol=atol, rtol=rtol)
    torch.testing.assert_close(k[:, 0], _apply_rotary(expected_k, cos[positions.long(), None], sin[positions.long(), None]), atol=atol, rtol=rtol)
    torch.testing.assert_close(v[:, 0], qkv[:, 2560:].reshape(batch, 8, 64), atol=0, rtol=0)


@torch.inference_mode()
def test_bf16_packing_preserves_every_bit_pattern_and_refreshes():
    from climbing_gym.inference.lfm25_bf16 import PackedBF16Weight

    bits = torch.arange(65536, device="cuda", dtype=torch.int32).to(torch.int16).reshape(256, 256)
    weights = bits.view(torch.bfloat16)
    packed = PackedBF16Weight.from_weight(weights)
    assert torch.equal(packed.unpack().view(torch.int16), bits)
    # Includes rare exponents, signed zero, infinities and NaN payloads.
    weights.copy_(weights.flip(0))
    packed.refresh()
    assert torch.equal(packed.unpack().view(torch.int16), weights.view(torch.int16))


@pytest.mark.parametrize("width", (128, 2048, 10752))
@torch.inference_mode()
def test_bf16_packed_gemv_matches_unpacked(width):
    from climbing_gym.inference.lfm25_bf16 import PackedBF16Weight
    from climbing_gym.inference.lfm25_cuda_kernels import linear

    torch.manual_seed(319)
    weights = (torch.randn(256, width, device="cuda") * 0.025).bfloat16()
    weights[0, :4] = torch.tensor([0, 1e-20, 1000, -1e-20], device="cuda", dtype=torch.bfloat16)
    x = torch.randn(1, width, device="cuda", dtype=torch.bfloat16)
    packed = PackedBF16Weight.from_weight(weights)
    assert torch.equal(packed.unpack().view(torch.int16), weights.view(torch.int16))
    torch.testing.assert_close(packed.linear(x), linear(x, weights), rtol=0.008, atol=0.015625)


@torch.inference_mode()
def test_rotary_bf16_matches_transformers_intermediate_rounding():
    from transformers.models.lfm2.modeling_lfm2 import apply_rotary_pos_emb
    from climbing_gym.inference.lfm25_engine import _apply_rotary

    torch.manual_seed(18)
    q = torch.randn(3, 1, 32, 64, device="cuda", dtype=torch.bfloat16)
    k = torch.randn(3, 1, 8, 64, device="cuda", dtype=torch.bfloat16)
    frequencies = torch.tensor([27, 1331, 8000], device="cuda")[:, None, None] / (
        1e7 ** (torch.arange(0, 64, 2, device="cuda") / 64)
    )
    cos, sin = frequencies.cos().bfloat16(), frequencies.sin().bfloat16()
    expected_q, expected_k = apply_rotary_pos_emb(q, k, cos.repeat(1, 1, 2), sin.repeat(1, 1, 2), unsqueeze_dim=2)
    torch.testing.assert_close(_apply_rotary(q, cos[:, :, None], sin[:, :, None]), expected_q, atol=0, rtol=0)
    torch.testing.assert_close(_apply_rotary(k, cos[:, :, None], sin[:, :, None]), expected_k, atol=0, rtol=0)


def _engine():
    pytest.importorskip("flash_attn")
    torch.manual_seed(42)
    config = LFM2Config(
        vocab_size=257, hidden_size=128, mlp_hidden_size=192, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=256,
        rope_theta=1e7, layer_types=("conv", "full_attention", "conv", "full_attention"),
        attn_backend="eager",
    )
    model = LFM2Model(config).eval()
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.02)
    engine = LFM25Engine.from_model(model, device="cuda", dtype=torch.bfloat16, max_model_len=256)
    return model, engine


def test_cuda_graph_recapture_survives_rollout_worker_and_stream_pool_reuse(tmp_path):
    # Isolate a possible CUDA deadlock so a regression fails with a timeout
    # instead of wedging the pytest process. The unfixed decoder hangs around
    # capture 33 after cycling through the stream pool.
    repo = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    env['PYTHONPATH'] = str(repo / 'src')
    env['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
    result = subprocess.run(
        [sys.executable, str(repo / 'tests/helpers/native_graph_recapture.py'),
         '--iterations', '64', '--output', str(tmp_path / 'recaptures.jsonl')],
        env=env, capture_output=True, text=True, timeout=90,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert len((tmp_path / 'recaptures.jsonl').read_text().splitlines()) == 64


@torch.inference_mode()
def test_fp32_cached_decode_uses_supported_attention():
    model, _ = _engine()
    engine = LFM25Engine.from_model(model, device="cuda", dtype=torch.float32, max_model_len=256)
    prompt = [1, 2, 3]
    prefix = engine.prefill(*flatten_token_batches([prompt]), max_cache_len=128)
    state = engine.fork(prefix, repeats=1)
    result = engine.decode_step(state, torch.tensor([4], device="cuda"))
    expected = model(torch.tensor([prompt + [4]]), positions=torch.arange(4)[None])[0, -1]
    torch.testing.assert_close(result.logits[0].cpu(), expected, atol=1e-4, rtol=1e-4)


@torch.inference_mode()
def test_cuda_graph_teacher_forcing_matches_eager_with_unequal_lengths():
    _, engine = _engine()
    engine.enable_cuda_graphs()
    prompts = [[1, 2, 3], [1, 4, 5, 6, 7]]
    prefix = engine.prefill(*flatten_token_batches(prompts), max_cache_len=128)
    state = engine.fork(prefix, repeats=1)
    entry = engine._cuda_decoder._get_graph(2, 128)
    entry.cache.k_cache.copy_(prefix.cache.k_cache)
    entry.cache.v_cache.copy_(prefix.cache.v_cache)
    entry.cache.conv_state.copy_(prefix.cache.conv_state)
    entry.cache.seq_lens.copy_(prefix.cache.seq_lens)
    for step in range(4):
        tokens = torch.tensor([21 + step, 31 + step], device="cuda")
        reference = engine._append_decode_tokens(state.cache, state.active_slot_ids, tokens)
        entry.input_tokens.copy_(tokens)
        entry.graph.replay()
        torch.testing.assert_close(entry.logits, reference, atol=0.13, rtol=0.025)
        torch.testing.assert_close(entry.cache.seq_lens, state.cache.seq_lens)


@pytest.mark.parametrize('stop_all', [False, True])
@torch.inference_mode()
def test_amortized_finish_polling_preserves_tokens_logprobs_and_stops(stop_all):
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1)
    inputs = flatten_token_batches([[1, 3 + i, 7] for i in range(16)])
    sampling = LFM25SamplingConfig(temperature=1, seed=42, max_new_tokens=37,
                                  ignore_eos=True, return_logprobs=True)
    initial = engine.generate(*inputs, repeats=1, sampling=sampling)
    stops = tuple(set(initial.token_ids[:, 0, 0].tolist())) if stop_all else (int(initial.token_ids[0, 0, 0]),)
    sampling = replace(sampling, stop_token_ids=stops)
    engine._cuda_decoder.finish_check_interval = 1
    engine._cuda_decoder.skip_finished_attention = False
    reference = engine.generate(*inputs, repeats=1, sampling=sampling)
    engine._cuda_decoder.finish_check_interval = 32
    engine._cuda_decoder.skip_finished_attention = True
    actual = engine.generate(*inputs, repeats=1, sampling=sampling)
    for name in ('token_ids', 'lengths', 'finish_reasons', 'logprobs'):
        torch.testing.assert_close(getattr(actual, name), getattr(reference, name), rtol=0, atol=0)
    if stop_all:
        assert actual.lengths.eq(1).all()
        assert actual.token_ids[:, :, 1:].count_nonzero() == 0


@torch.inference_mode()
def test_cuda_graph_reuse_reset_stops_and_weight_sync():
    model, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=4, ignore_eos=True, return_logprobs=True)
    inputs = flatten_token_batches([[1, 11, 12], [1, 21]])
    first = engine.generate(*inputs, repeats=2, sampling=sampling)
    graph = next(iter(engine._cuda_decoder.graphs.values())).graph
    engine.generate(*flatten_token_batches([[1, 100], [1, 200, 201]]), repeats=2, sampling=sampling)
    repeated = engine.generate(*inputs, repeats=2, sampling=sampling)
    assert next(iter(engine._cuda_decoder.graphs.values())).graph is graph
    torch.testing.assert_close(first.token_ids, repeated.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(first.logprobs, repeated.logprobs, atol=0, rtol=0)
    stopped = engine.generate(*inputs, repeats=2, sampling=replace(sampling, stop_token_ids=(int(first.token_ids[0, 0, 0]),)))
    assert stopped.lengths[0, 0].item() == 1
    assert stopped.finish_reasons[0, 0].item() == 2
    assert stopped.token_ids[0, 0, 1:].count_nonzero().item() == 0
    torch.manual_seed(9)
    updated = LFM2Model(model.config).eval()
    for parameter in updated.parameters():
        if parameter.ndim > 1:
            parameter.normal_(std=0.02)
    engine.sync_from_model(updated)
    after_sync = engine.generate(*inputs, repeats=2, sampling=sampling)
    assert next(iter(engine._cuda_decoder.graphs.values())).graph is graph
    expected_engine = LFM25Engine.from_model(updated, device="cuda", dtype=torch.bfloat16, max_model_len=256)
    expected_engine.enable_cuda_graphs()
    expected = expected_engine.generate(*inputs, repeats=2, sampling=sampling)
    torch.testing.assert_close(after_sync.token_ids, expected.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(after_sync.logprobs, expected.logprobs, atol=0, rtol=0)
    assert not torch.equal(first.logprobs, after_sync.logprobs)


@torch.inference_mode()
def test_cuda_graph_stochastic_sampling_is_seeded_and_cache_is_bounded():
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1)
    inputs = flatten_token_batches([[1, 2, 3]])
    sampling = LFM25SamplingConfig(temperature=1, top_k=16, top_p=0.95, repetition_penalty=1.1,
                                 max_new_tokens=6, ignore_eos=True, seed=17)
    first = engine.generate(*inputs, repeats=2, sampling=sampling)
    second = engine.generate(*inputs, repeats=2, sampling=sampling)
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.logprobs, second.logprobs)
    engine.generate(*inputs, repeats=1, sampling=sampling)
    assert len(engine._cuda_decoder.graphs) == 1
    assert next(iter(engine._cuda_decoder.graphs))[0] == 1
    engine.disable_cuda_graphs()
    assert engine._cuda_decoder is None


@torch.inference_mode()
def test_packed_decode_graph_refreshes_after_policy_sync():
    model, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1)
    inputs = flatten_token_batches([[1, 2, 3]])
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=4, ignore_eos=True, return_logprobs=True)
    original = engine.generate(*inputs, repeats=1, sampling=sampling)
    graph = next(iter(engine._cuda_decoder.graphs.values())).graph
    addresses = [p.mantissa.data_ptr() for p in engine._cuda_decoder.packed_weights.values()]
    assert addresses
    with torch.no_grad():
        for parameter in model.parameters():
            if parameter.ndim > 1:
                parameter.normal_(std=0.03)
    engine.sync_from_model(model)
    updated = engine.generate(*inputs, repeats=1, sampling=sampling)
    assert next(iter(engine._cuda_decoder.graphs.values())).graph is graph
    assert addresses == [p.mantissa.data_ptr() for p in engine._cuda_decoder.packed_weights.values()]
    for packed in engine._cuda_decoder.packed_weights.values():
        assert torch.equal(packed.unpack().view(torch.int16), packed.original.view(torch.int16))
    assert not torch.equal(original.logprobs, updated.logprobs)
    fresh = LFM25Engine.from_model(model, device="cuda", dtype=torch.bfloat16, max_model_len=256)
    fresh.enable_cuda_graphs()
    expected = fresh.generate(*inputs, repeats=1, sampling=sampling)
    torch.testing.assert_close(updated.logprobs, expected.logprobs, atol=0, rtol=0)
    torch.testing.assert_close(updated.token_ids, expected.token_ids, atol=0, rtol=0)


@pytest.mark.parametrize("budget", (4, 24, 4096))
@torch.inference_mode()
def test_chunked_prefill_preserves_ragged_prompts_and_reuses_cache(budget):
    model, _ = _engine()
    # FP32 isolates padding/cache indexing from BF16 GEMM shape-dependent
    # rounding; checkpoint parity separately gates BF16 numerical agreement.
    engine = LFM25Engine.from_model(model, device="cuda", dtype=torch.float32, max_model_len=256)
    prompts = [[1], [1, 2], [1, 3, 4, 5, 6], [1, 7, 8], [1, 9, 10, 11]]
    inputs = flatten_token_batches(prompts)
    expected = engine.prefill(*inputs, max_cache_len=128)
    destination = engine._empty_cache(max_slots=len(prompts), max_seq_len=128)
    # A reused cache contains old data beyond the new prompt and old conv state.
    destination.k_cache.fill_(0.5)
    destination.v_cache.fill_(-0.5)
    destination.conv_state.fill_(1)
    logits_buffer = torch.empty_like(expected.next_token_logits)
    actual = engine.prefill(*inputs, cache=destination, max_prefill_tokens=budget, logits_buffer=logits_buffer)
    assert actual.cache is destination
    assert actual.next_token_logits is logits_buffer
    torch.testing.assert_close(actual.next_token_logits, expected.next_token_logits, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(actual.cache.seq_lens, expected.cache.seq_lens)
    torch.testing.assert_close(actual.cache.conv_state, expected.cache.conv_state, atol=1e-6, rtol=1e-4)
    for row, prompt in enumerate(prompts):
        torch.testing.assert_close(actual.cache.k_cache[:, row, :len(prompt)],
                                   expected.cache.k_cache[:, row, :len(prompt)], atol=1e-5, rtol=1e-4)
    slots = torch.arange(len(prompts), device="cuda")
    for step in range(3):
        tokens = slots + 40 + step
        logits = engine._append_decode_tokens(actual.cache, slots, tokens)
        reference = engine._append_decode_tokens(expected.cache, slots, tokens)
        torch.testing.assert_close(logits, reference, atol=1e-5, rtol=1e-4)


@pytest.mark.parametrize("dtype", (torch.bfloat16, torch.float16, torch.float32))
@torch.inference_mode()
def test_fused_sampling_greedy_and_raw_logprobs(dtype):
    from climbing_gym.inference.lfm25_sampling import sample_logits

    torch.manual_seed(45)
    logits = torch.randn(9, 128003, dtype=dtype, device="cuda")
    logits[0, :8192] = -torch.inf  # Fully masked tiles must not spoil the normalizer.
    logits[1, 12] = logits[1, 125000] = 20  # Argmax ties choose the first token.
    tokens, logprobs = sample_logits(logits, temperature=0, seed=0, step=0, return_logprobs=True)
    torch.testing.assert_close(tokens, logits.argmax(-1), atol=0, rtol=0)
    expected = logits.float().log_softmax(-1).gather(1, tokens[:, None]).squeeze(1)
    torch.testing.assert_close(logprobs, expected, atol=3e-6, rtol=2e-6)
    sampled, raw = sample_logits(logits, temperature=0.7, seed=31, step=4095, return_logprobs=True)
    expected = logits.float().log_softmax(-1).gather(1, sampled[:, None]).squeeze(1)
    torch.testing.assert_close(raw, expected, atol=3e-6, rtol=2e-6)


@torch.inference_mode()
def test_fused_categorical_sampling_distribution_and_seed():
    from climbing_gym.inference.lfm25_sampling import sample_logits

    probabilities = torch.tensor([0.05, 0.15, 0.3, 0.5], device="cuda")
    logits = (probabilities.log() * 0.7).expand(20000, -1).contiguous()
    kwargs = dict(temperature=0.7, seed=17, step=0, return_logprobs=False)
    tokens, logprobs = sample_logits(logits, **kwargs)
    assert logprobs is None
    torch.testing.assert_close(tokens.bincount(minlength=4) / tokens.numel(), probabilities, atol=0.012, rtol=0)
    assert torch.equal(tokens, sample_logits(logits, **kwargs)[0])
    for seed, step in ((18, 0), (17, 64), (17, 4095)):
        other = sample_logits(logits, **{**kwargs, "seed": seed, "step": step})[0]
        assert not torch.equal(tokens, other)


@torch.inference_mode()
def test_fused_sampling_generation_reuses_direct_prefill_cache():
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, max_prefill_tokens=8)
    inputs = flatten_token_batches([[1, 11, 12], [1], [1, 21]])
    sampling = LFM25SamplingConfig(temperature=1, max_new_tokens=6, ignore_eos=True, seed=17)
    first = engine.generate(*inputs, repeats=1, sampling=sampling)
    graph = next(iter(engine._cuda_decoder.graphs.values())).graph
    engine.generate(*flatten_token_batches([[1, 44], [1, 55, 66], [1, 7, 8, 9]]), repeats=1, sampling=sampling)
    second = engine.generate(*inputs, repeats=1, sampling=sampling)
    assert next(iter(engine._cuda_decoder.graphs.values())).graph is graph
    assert torch.equal(first.token_ids, second.token_ids)
    assert torch.equal(first.logprobs, second.logprobs)
    one = engine.generate(*inputs, repeats=2, sampling=replace(sampling, max_new_tokens=1))
    assert one.token_ids.shape == (3, 2, 1)


@torch.inference_mode()
def test_large_batch_graph_cache_rounding_and_eviction():
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, max_prefill_tokens=128)
    prompts = [[1] + [row + 2] * (row % 10) for row in range(64)]
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=20, ignore_eos=True)
    output = engine.generate(*flatten_token_batches(prompts), repeats=1, sampling=sampling)
    assert output.token_ids.shape == (64, 1, 20)
    assert next(iter(engine._cuda_decoder.graphs.values())).cache.max_seq_len == 32
    assert not engine._cuda_decoder.packed_weights
    prompts[0] += [70] * 12
    engine.generate(*flatten_token_batches(prompts), repeats=1, sampling=sampling)
    assert len(engine._cuda_decoder.graphs) == 1
    assert next(iter(engine._cuda_decoder.graphs.values())).cache.max_seq_len == 64


@torch.inference_mode()
def test_graph_eviction_releases_unused_single_request_weight_copies():
    _, engine = _engine()
    engine.enable_cuda_graphs(max_graphs=1, max_prefill_tokens=128)
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=4, ignore_eos=True)
    inputs = flatten_token_batches([[1, 2, 3]])
    original = engine.generate(*inputs, repeats=1, sampling=sampling)
    assert engine._cuda_decoder.packed_weights
    engine.generate(*flatten_token_batches([[1, 2, 3]] * 64), repeats=1, sampling=sampling)
    assert not engine._cuda_decoder.packed_weights
    repeated = engine.generate(*inputs, repeats=1, sampling=sampling)
    assert engine._cuda_decoder.packed_weights
    torch.testing.assert_close(original.token_ids, repeated.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(original.logprobs, repeated.logprobs, atol=0, rtol=0)
