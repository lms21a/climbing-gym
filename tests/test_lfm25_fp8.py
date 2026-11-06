from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from climbing_gym.inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig, flatten_token_batches
from climbing_gym.models.lfm2 import LFM2Config, LFM2Model

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or torch.cuda.get_device_capability() < (8, 9), reason="SM89+ required")


@torch.inference_mode()
def test_fused_activation_quantization_matches_separate_kernels():
    from climbing_gym.inference.lfm25_cuda_kernels import rms_norm, swiglu
    from climbing_gym.inference.lfm25_fp8 import quantize_rows, rms_norm_fp8, swiglu_fp8

    torch.manual_seed(10)
    x = torch.randn(17, 2048, device="cuda", dtype=torch.bfloat16)
    residual = torch.randn_like(x)
    weight = torch.randn(2048, device="cuda", dtype=torch.bfloat16)
    normalized, expected_sum = rms_norm(x, weight, 1e-5, residual)
    expected, scales = quantize_rows(normalized)
    actual, actual_scales, summed = rms_norm_fp8(x, weight, 1e-5, residual)
    torch.testing.assert_close(summed, expected_sum, atol=0, rtol=0)
    torch.testing.assert_close(actual_scales, scales, atol=0, rtol=0)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))
    projected = torch.randn(17, 16384, device="cuda", dtype=torch.bfloat16)
    expected, scales = quantize_rows(swiglu(projected))
    actual, actual_scales = swiglu_fp8(projected)
    torch.testing.assert_close(actual_scales, scales, atol=0, rtol=0)
    assert torch.equal(actual.view(torch.uint8), expected.view(torch.uint8))


@pytest.mark.parametrize("batch,width,outputs", [(3, 129, 193), (128, 2048, 512), (32, 8192, 128)])
@torch.inference_mode()
def test_fp8_projection_matches_independently_dequantized_operands(batch, width, outputs):
    from climbing_gym.inference.lfm25_fp8 import PackedFP8Weight, quantize_rows

    torch.manual_seed(42)
    x = torch.randn(batch, width, device="cuda", dtype=torch.bfloat16)
    weight = (torch.randn(outputs, width, device="cuda") * 0.025).bfloat16()
    x[0].zero_()
    weight[0].zero_()
    packed = PackedFP8Weight.from_weight(weight)
    q, scale = quantize_rows(x)
    expected = (q.float() * scale[:, None]) @ (packed.values.float() * packed.scales[:, None]).T
    actual = packed.linear(x)
    torch.testing.assert_close(actual.float(), expected, atol=0.02, rtol=0.015)
    assert torch.isfinite(actual).all()
    assert actual[0].count_nonzero() == actual[:, 0].count_nonzero() == 0
    reference = F.linear(x, weight).float()
    assert (actual.float() - reference).norm() / reference.norm() < 0.065
    split = packed.linear(x, config=(32, 64, 128, 4, 3, 2))
    torch.testing.assert_close(split.float(), expected, atol=0.02, rtol=0.015)


def model_and_engine():
    torch.manual_seed(19)
    cfg = LFM2Config(vocab_size=257, hidden_size=128, mlp_hidden_size=192, num_hidden_layers=2,
                    num_attention_heads=4, num_key_value_heads=2, max_position_embeddings=128,
                    layer_types=("conv", "full_attention"), attn_backend="eager")
    model = LFM2Model(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            if p.ndim > 1:
                p.normal_(std=0.025)
    engine = LFM25Engine.from_model(model, device="cuda", dtype=torch.bfloat16, max_model_len=128)
    engine.enable_cuda_graphs(fp8_mlp=True, max_graphs=1)
    return model, engine


@torch.inference_mode()
def test_fp8_graph_sync_updates_quantized_buffers_without_recapture():
    _, engine = model_and_engine()
    prompts = flatten_token_batches([[1, 2, 3], [1, 4]])
    sampling = LFM25SamplingConfig(temperature=0, max_new_tokens=5, ignore_eos=True)
    before = engine.generate(*prompts, repeats=1, sampling=sampling)
    decoder = engine._cuda_decoder
    graph = next(iter(decoder.graphs.values())).graph
    pointers = [(p.values.data_ptr(), p.scales.data_ptr()) for p in decoder.fp8_weights.values()]
    updated, _ = model_and_engine()
    for p in updated.parameters():
        if p.ndim > 1:
            p.normal_(std=0.05)
    engine.sync_from_model(updated)
    after = engine.generate(*prompts, repeats=1, sampling=sampling)
    assert next(iter(decoder.graphs.values())).graph is graph
    assert pointers == [(p.values.data_ptr(), p.scales.data_ptr()) for p in decoder.fp8_weights.values()]
    fresh = LFM25Engine.from_model(updated, device="cuda", dtype=torch.bfloat16, max_model_len=128)
    fresh.enable_cuda_graphs(fp8_mlp=True, max_graphs=1)
    expected = fresh.generate(*prompts, repeats=1, sampling=sampling)
    torch.testing.assert_close(after.token_ids, expected.token_ids, atol=0, rtol=0)
    torch.testing.assert_close(after.logprobs, expected.logprobs, atol=0, rtol=0)
    assert not torch.equal(before.logprobs, after.logprobs)
    decoder.clear()  # The trainer releases inference state before learning.
    assert not decoder.graphs and not decoder.fp8_weights
    restarted = engine.generate(*prompts, repeats=1, sampling=sampling)
    torch.testing.assert_close(restarted.logprobs, after.logprobs, atol=0, rtol=0)


@torch.inference_mode()
def test_fp8_returns_its_own_behavior_logprobs_and_sessions_refill():
    _, engine = model_and_engine()
    prompts = [[1, 2, 3], [1, 4]]
    sampling = LFM25SamplingConfig(temperature=1, seed=17, max_new_tokens=3, ignore_eos=True)
    result = engine.generate(*flatten_token_batches(prompts), repeats=1, sampling=sampling)
    entry = next(iter(engine._cuda_decoder.graphs.values()))
    engine.prefill(*flatten_token_batches(prompts), cache=entry.cache)
    entry.input_tokens.copy_(result.token_ids[:, 0, 0])
    entry.graph.replay()
    expected = entry.logits.float().log_softmax(-1).gather(1, result.token_ids[:, 0, 1:2]).squeeze(1)
    torch.testing.assert_close(result.logprobs[:, 0, 1], expected, atol=2e-6, rtol=0)
    with engine.generation_session(max_batch_size=2, max_new_tokens=3) as session:
        session.admit(prompts, [sampling, sampling])
        finished = session.poll(steps=4)
        assert len(finished) == 2
        for output in finished:
            assert len(output.token_ids) == len(output.logprobs) == 3
            session.release(output.handle)
        session.admit(prompts[::-1], [sampling, sampling])
        refilled = session.poll(steps=4)
        assert len(refilled) == 2
        assert session.stats["slot_refills"] == 2
        assert all(torch.isfinite(torch.tensor(r.logprobs)).all() for r in refilled)


@torch.inference_mode()
def test_fp8_does_not_also_losslessly_pack_unused_bf16_mlps():
    _, engine = model_and_engine()
    engine.generate(*flatten_token_batches([[1, 2, 3]]), repeats=1,
                    sampling=LFM25SamplingConfig(temperature=0, max_new_tokens=3))
    decoder = engine._cuda_decoder
    assert decoder.fp8_weights
    assert id(engine.weights.lm_head_weight) in decoder.packed_weights
    assert not set(decoder.fp8_weights).intersection(decoder.packed_weights)


def test_fp8_async_chains_refill_and_resume_with_stable_graph_weights():
    import asyncio
    from climbing_gym.inference import LFM25AsyncInferenceEngine, SamplingConfig
    from test_lfm25_async_engine import _FakeTokenizer

    _, raw = model_and_engine()
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer(), max_batch_size=2)
    sampling = SamplingConfig(temperature=1, max_tokens=3, logprobs=1, extra_params={"ignore_eos": True})
    requests = [([1, 2 + i], sampling) for i in range(7)]
    phases = [0] * len(requests)

    def advance(index, result):
        assert len(result.completion_logprobs) == 3
        phases[index] += 1
        if phases[index] == 1:
            return result.prompt_token_ids + result.token_ids + (40,), sampling
        return None

    results = asyncio.run(engine.generate_chains(requests, on_result=advance, max_new_tokens=3, poll_interval=2))
    assert all(len(chain) == 2 for chain in results)
    assert engine.last_session_stats["resumed_requests"] == 7
    assert engine.last_session_stats["slot_refills"] == 5
    assert len(raw._cuda_decoder.graphs) == 1
    assert raw._cuda_decoder._active_session is None
    engine.close()
