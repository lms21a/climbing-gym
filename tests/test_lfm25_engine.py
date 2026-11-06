from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from climbing_gym.inference.lfm25_engine import (
    LFM25Engine,
    LFM25SamplingConfig,
    _apply_repetition_penalty,
    describe_lfm25_checkpoint,
    flatten_token_batches,
)
from climbing_gym.models import LFM2Config, LFM2Model


def _tiny_cfg() -> LFM2Config:
    return LFM2Config(
        vocab_size=257,
        hidden_size=64,
        mlp_hidden_size=128,
        num_hidden_layers=4,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
        layer_types=("conv", "full_attention", "conv", "full_attention"),
        attn_backend="eager",
        shortconv_backend="torch",
    )


def _make_model_and_engine() -> tuple[LFM2Model, LFM25Engine]:
    torch.manual_seed(0)
    model = LFM2Model(_tiny_cfg()).eval()
    engine = LFM25Engine.from_model(model, device="cpu", dtype=torch.float32, max_model_len=128)
    return model, engine


def _packed_named_tensors(engine: LFM25Engine) -> dict[str, torch.Tensor]:
    tensors = {
        "embed_tokens": engine.weights.embed_tokens,
        "embedding_norm": engine.weights.embedding_norm_weight,
        "lm_head": engine.weights.lm_head_weight,
    }
    for layer_idx, layer in enumerate(engine.weights.layers):
        prefix = f"layers.{layer_idx}"
        tensors[f"{prefix}.operator_norm"] = layer.operator_norm_weight
        tensors[f"{prefix}.ffn_norm"] = layer.ffn_norm_weight
        tensors[f"{prefix}.w13"] = layer.w13_weight
        tensors[f"{prefix}.w2"] = layer.w2_weight
        if layer.attention is not None:
            tensors[f"{prefix}.qkv"] = layer.attention.qkv_weight
            tensors[f"{prefix}.attn_out"] = layer.attention.out_weight
            tensors[f"{prefix}.q_norm"] = layer.attention.q_norm_weight
            tensors[f"{prefix}.k_norm"] = layer.attention.k_norm_weight
        if layer.conv is not None:
            tensors[f"{prefix}.conv_in"] = layer.conv.in_weight
            tensors[f"{prefix}.conv_weight"] = layer.conv.conv_weight
            tensors[f"{prefix}.conv_out"] = layer.conv.out_weight
    return tensors


def test_packed_weight_shapes_match_lfm25_layout() -> None:
    _, engine = _make_model_and_engine()
    attn_layer = next(layer for layer in engine.weights.layers if layer.attention is not None)
    conv_layer = next(layer for layer in engine.weights.layers if layer.conv is not None)

    assert attn_layer.attention is not None
    assert conv_layer.conv is not None
    assert attn_layer.attention.qkv_weight.shape == (128, 64)
    assert conv_layer.conv.in_weight.shape == (192, 64)
    assert conv_layer.conv.conv_weight.shape == (64, 3)
    assert attn_layer.w13_weight.shape == (256, 64)
    assert engine.weights.num_attention_layers == 2
    assert engine.weights.num_conv_layers == 2


def test_prefill_logits_match_full_model_for_packed_prompts() -> None:
    model, engine = _make_model_and_engine()
    prompts = [[1, 11, 12, 13], [1, 21, 22]]
    flat, cu, max_len = flatten_token_batches(prompts)

    prefix = engine.prefill(flat, cu, max_len, max_cache_len=8)

    for idx, prompt in enumerate(prompts):
        ids = torch.tensor([prompt], dtype=torch.long)
        positions = torch.arange(len(prompt), dtype=torch.long).unsqueeze(0)
        expected = model(ids, positions=positions)[0, -1]
        assert torch.allclose(prefix.next_token_logits[idx], expected, atol=1e-4, rtol=1e-4)
    assert prefix.cache.seq_lens.tolist() == [4, 3]


def test_equal_length_prefill_logits_match_full_model() -> None:
    model, engine = _make_model_and_engine()
    prompts = [[1, 11, 12, 13], [1, 21, 22, 23], [1, 31, 32, 33]]
    flat, cu, max_len = flatten_token_batches(prompts)

    prefix = engine.prefill(flat, cu, max_len, max_cache_len=8)

    for idx, prompt in enumerate(prompts):
        ids = torch.tensor([prompt], dtype=torch.long)
        positions = torch.arange(len(prompt), dtype=torch.long).unsqueeze(0)
        expected = model(ids, positions=positions)[0, -1]
        assert torch.allclose(prefix.next_token_logits[idx], expected, atol=1e-4, rtol=1e-4)
    assert prefix.cache.seq_lens.tolist() == [4, 4, 4]


def test_cached_decode_matches_full_recompute() -> None:
    model, engine = _make_model_and_engine()
    prompt = [1, 5, 6, 7]
    flat, cu, max_len = flatten_token_batches([prompt])
    prefix = engine.prefill(flat, cu, max_len, max_cache_len=8)
    state = engine.fork(prefix, repeats=1, max_cache_len=8)

    step = engine.decode_step(
        state,
        torch.tensor([8], dtype=torch.long),
        LFM25SamplingConfig(temperature=0.0, max_new_tokens=1),
    )

    full = torch.tensor([prompt + [8]], dtype=torch.long)
    positions = torch.arange(full.size(1), dtype=torch.long).unsqueeze(0)
    expected_logits = model(full, positions=positions)[0, -1]
    assert torch.allclose(step.logits[0], expected_logits, atol=1e-4, rtol=1e-4)
    assert state.cache.seq_lens.tolist() == [5]


def test_forked_prefixes_decode_like_repeated_prefills() -> None:
    _, engine = _make_model_and_engine()
    prompt = [1, 2, 3]
    flat, cu, max_len = flatten_token_batches([prompt])
    prefix = engine.prefill(flat, cu, max_len, max_cache_len=8)
    state = engine.fork(prefix, repeats=2, max_cache_len=8)

    step = engine.decode_step(
        state,
        torch.tensor([4, 4], dtype=torch.long),
        LFM25SamplingConfig(temperature=0.0, max_new_tokens=1),
    )

    assert torch.allclose(step.logits[0], step.logits[1], atol=1e-6, rtol=1e-6)
    assert state.cache.seq_lens.tolist() == [4, 4]


def test_generate_returns_prompt_repeat_shaped_gpu_style_tensors() -> None:
    _, engine = _make_model_and_engine()
    flat, cu, max_len = flatten_token_batches([[1, 2, 3], [1, 4]])

    output = engine.generate(
        flat,
        cu,
        max_len,
        repeats=3,
        sampling=LFM25SamplingConfig(temperature=0.0, max_new_tokens=2, ignore_eos=True),
    )

    assert output.token_ids.shape == (2, 3, 2)
    assert output.lengths.shape == (2, 3)
    assert output.logprobs is not None
    assert output.logprobs.shape == (2, 3, 2)
    assert torch.all(output.lengths == 2)
    assert torch.all(output.finish_reasons == 3)


def test_sync_from_model_updates_packed_weights_in_place_and_keeps_compiled_decode() -> None:
    _, engine = _make_model_and_engine()
    torch.manual_seed(17)
    source_model = LFM2Model(_tiny_cfg()).eval()
    expected_engine = LFM25Engine.from_model(
        source_model,
        device="cpu",
        dtype=torch.float32,
        max_model_len=128,
    )
    identities = {
        name: (id(tensor), tensor.data_ptr())
        for name, tensor in _packed_named_tensors(engine).items()
    }
    compiled_decode_marker = object()
    engine._compiled_append_tokens = compiled_decode_marker

    result = engine.sync_from_model(source_model, drain_timeout=0)

    assert result is engine
    assert engine._compiled_append_tokens is compiled_decode_marker
    actual_tensors = _packed_named_tensors(engine)
    expected_tensors = _packed_named_tensors(expected_engine)
    assert actual_tensors.keys() == expected_tensors.keys()
    for name, actual in actual_tensors.items():
        assert (id(actual), actual.data_ptr()) == identities[name]
        assert torch.equal(actual, expected_tensors[name]), name


def test_sync_from_model_validates_config_and_all_shapes_before_copying() -> None:
    _, engine = _make_model_and_engine()
    original_embed = engine.weights.embed_tokens.clone()
    incompatible_model = LFM2Model(replace(_tiny_cfg(), eos_token_id=99)).eval()

    with pytest.raises(ValueError, match="incompatible LFM2 config.*eos_token_id"):
        engine.sync_from_model(incompatible_model)
    assert torch.equal(engine.weights.embed_tokens, original_embed)

    malformed_model = LFM2Model(_tiny_cfg()).eval()
    malformed_model.embed_tokens.weight = torch.nn.Parameter(torch.randn(256, 64))
    with pytest.raises(ValueError, match="shape mismatch.*embed_tokens"):
        engine.sync_from_model(malformed_model)
    assert torch.equal(engine.weights.embed_tokens, original_embed)


def test_sync_from_model_accepts_foreign_config_protocol() -> None:
    from types import SimpleNamespace

    model, _ = _make_model_and_engine()
    config = model.config.to_dict()
    config["intermediate_size"] = config.pop("mlp_hidden_size")
    config["block_auto_adjust_ff_dim"] = False
    config["conv_L_cache"] = config.pop("conv_kernel_size")
    model.config = SimpleNamespace(**config, to_dict=lambda: config)
    engine = LFM25Engine.from_model(model, device="cpu", dtype=torch.float32, max_model_len=128)
    before = engine.weights.embed_tokens.clone()
    with torch.no_grad():
        model.embed_tokens.weight.add_(1.0)
    engine.sync_from_model(model)
    assert engine.config.mlp_hidden_size == 128
    assert engine.config.conv_kernel_size == 3
    assert not torch.equal(engine.weights.embed_tokens, before)
    assert torch.equal(engine.weights.embed_tokens, model.embed_tokens.weight)


def test_repetition_penalty_adjusts_only_seen_token_logits() -> None:
    logits = torch.tensor([[3.0, -2.0, 2.0]])
    seen = torch.tensor([[True, True, False]])

    adjusted = _apply_repetition_penalty(logits, seen_token_mask=seen, penalty=2.0)

    assert torch.equal(adjusted, torch.tensor([[1.5, -4.0, 2.0]]))


def test_describe_lfm25_checkpoint_reports_cache_sizes(tmp_path) -> None:
    config = _tiny_cfg()
    (tmp_path / "config.json").write_text(
        """
        {
          "vocab_size": 257,
          "hidden_size": 64,
          "mlp_hidden_size": 128,
          "num_hidden_layers": 4,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "max_position_embeddings": 128,
          "layer_types": ["conv", "full_attention", "conv", "full_attention"]
        }
        """,
        encoding="utf-8",
    )

    summary = describe_lfm25_checkpoint(tmp_path)

    assert summary["attention_layers"] == [1, 3]
    assert summary["conv_layers"] == [0, 2]
    assert summary["kv_bytes_per_token_per_sequence_bf16"] == 2 * 2 * config.num_key_value_heads * config.head_dim * 2
    assert summary["conv_state_bytes_per_sequence_bf16"] == 2 * config.hidden_size * config.conv_kernel_size * 2
