from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import warnings

import pytest
import torch
import torch.nn.functional as F

import climbing_gym.models.lfm2 as lfm2_mod
from climbing_gym.models import LFM2Config, LFM2Model, build_config, build_model, list_model_names
from climbing_gym.models.benchmarks import lfm2 as lfm2_bench
from climbing_gym.models.kernels import (
    LFM2_FUSED_KERNELS_AVAILABLE,
    TRITON_AVAILABLE,
    depthwise_causal_conv_k3,
    depthwise_causal_conv_k3_reference,
    gated_depthwise_causal_conv_k3,
)


def _tiny_cfg() -> LFM2Config:
    return LFM2Config(
        vocab_size=257,
        hidden_size=64,
        mlp_hidden_size=128,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=4,
        max_position_embeddings=128,
        layer_types=("conv", "full_attention"),
        attn_backend="eager",
        shortconv_backend="torch",
    )


def _cuda_runtime_usable() -> bool:
    if not torch.cuda.is_available():
        return False
    try:
        torch.cuda.current_device()
    except Exception:
        return False
    return True


def test_lfm2_registry_builds_config_and_model(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "vocab_size": 257,
          "hidden_size": 64,
          "mlp_hidden_size": 128,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "max_position_embeddings": 128
        }
        """,
        encoding="utf-8",
    )

    assert list_model_names() == ["lfm2"]

    config = build_config("lfm2", model_dir=str(tmp_path), attn_backend="eager")
    assert isinstance(config, LFM2Config)
    assert config.hidden_size == 64
    assert config.attn_backend == "eager"

    model = build_model("lfm2", config=_tiny_cfg())
    assert isinstance(model, LFM2Model)


def test_lfm2_save_pretrained_roundtrip(tmp_path: Path) -> None:
    torch.manual_seed(0)
    model = LFM2Model(_tiny_cfg())

    export_dir = tmp_path / "export"
    saved_path = model.save_pretrained(export_dir)
    restored = LFM2Model(_tiny_cfg()).from_pretrained(export_dir)

    assert saved_path == export_dir / "pytorch_model.bin"
    assert (export_dir / "config.json").exists()
    assert (export_dir / "pytorch_model.bin").exists()

    original_state = model.state_dict()
    restored_state = restored.state_dict()
    assert original_state.keys() == restored_state.keys()
    for key in original_state:
        assert torch.allclose(original_state[key], restored_state[key])


def test_lfm2_save_pretrained_exports_hf_style_weight_keys(tmp_path: Path) -> None:
    model = LFM2Model(_tiny_cfg())
    export_dir = tmp_path / "export"

    model.save_pretrained(export_dir)
    payload = torch.load(export_dir / "pytorch_model.bin", map_location="cpu", weights_only=False)

    assert "model.embed_tokens.weight" in payload
    assert "model.layers.0.conv.conv.weight" in payload
    assert "model.layers.1.self_attn.q_proj.weight" in payload
    assert "embed_tokens.weight" not in payload


def test_lfm2_legacy_config_payload_is_normalized(tmp_path: Path) -> None:
    config_path = tmp_path / "config.json"
    config_path.write_text(
        """
        {
          "vocab_size": 257,
          "hidden_size": 64,
          "intermediate_size": 96,
          "block_auto_adjust_ff_dim": true,
          "block_multiple_of": 16,
          "conv_L_cache": 5,
          "num_hidden_layers": 2,
          "num_attention_heads": 8,
          "num_key_value_heads": 4,
          "max_position_embeddings": 128,
          "layer_types": ["conv", "full_attention"]
        }
        """,
        encoding="utf-8",
    )

    config = build_config("lfm2", model_dir=str(tmp_path), attn_backend="eager")

    assert isinstance(config, LFM2Config)
    assert config.mlp_hidden_size == 64
    assert config.conv_kernel_size == 5


def test_lfm2_config_drops_legacy_export_and_init_fields() -> None:
    config = LFM2Config()

    for field_name in (
        "architectures",
        "dtype",
        "transformers_version",
        "initializer_range",
        "num_heads",
        "intermediate_size",
        "block_auto_adjust_ff_dim",
        "block_ffn_dim_multiplier",
        "block_multiple_of",
        "block_mlp_init_scale",
        "block_out_init_scale",
        "conv_L_cache",
        "conv_use_xavier_init",
        "use_cache",
        "use_pos_enc",
    ):
        assert not hasattr(config, field_name)


def test_lfm2_forward_backward_tiny() -> None:
    torch.manual_seed(0)
    model = LFM2Model(_tiny_cfg())
    model.train()

    input_ids = torch.randint(0, model.config.vocab_size, (1, 12))
    positions = torch.tensor([list(range(6)) + list(range(6))], dtype=torch.long)
    targets = torch.randint(0, model.config.vocab_size, (1, 12))

    logits = model(input_ids, positions=positions)
    assert logits.shape == (1, 12, model.config.vocab_size)
    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
    loss.backward()
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_lfm2_fused_backbone_config_preserves_cpu_results_and_state_keys() -> None:
    torch.manual_seed(41)
    reference_config = _tiny_cfg()
    fused_config = _tiny_cfg()
    fused_config.fused_backbone_kernels = True
    reference = LFM2Model(reference_config).eval()
    fused = LFM2Model(fused_config).eval()
    fused.load_state_dict(reference.state_dict())

    assert fused.state_dict().keys() == reference.state_dict().keys()
    input_ids = torch.randint(0, reference.config.vocab_size, (1, 12))
    positions = torch.tensor([list(range(6)) + list(range(6))])
    with torch.no_grad():
        expected = reference(input_ids, positions)
        actual = fused(input_ids, positions)
    assert torch.equal(actual, expected)


def test_lfm2_fused_backbone_forward_backward_parity_cuda() -> None:
    if not (
        LFM2_FUSED_KERNELS_AVAILABLE
        and TRITON_AVAILABLE
        and _cuda_runtime_usable()
    ):
        pytest.skip("CUDA, Triton, and Liger kernels are required.")

    torch.manual_seed(43)
    lfm2_mod.get_rope.cache_clear()
    reference_config = _tiny_cfg()
    reference_config.shortconv_backend = "triton_k3"
    fused_config = _tiny_cfg()
    fused_config.shortconv_backend = "triton_k3"
    fused_config.fused_backbone_kernels = True
    try:
        reference = LFM2Model(reference_config).to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        fused = LFM2Model(fused_config).to(
            device="cuda",
            dtype=torch.bfloat16,
        )
        fused.load_state_dict(reference.state_dict())
        input_ids = torch.randint(
            0,
            reference.config.vocab_size,
            (1, 12),
            device="cuda",
        )
        positions = torch.tensor(
            [list(range(6)) + list(range(6))],
            device="cuda",
        )
        upstream = torch.randn(
            1,
            12,
            reference.config.hidden_size,
            device="cuda",
        )

        expected = reference.get_hidden_states(input_ids, positions)
        actual = fused.get_hidden_states(input_ids, positions)
        (expected.float() * upstream).sum().backward()
        (actual.float() * upstream).sum().backward()

        assert torch.allclose(actual, expected, atol=1.6e-2, rtol=1e-2)
        for (expected_name, expected_parameter), (actual_name, actual_parameter) in zip(
            reference.named_parameters(),
            fused.named_parameters(),
            strict=True,
        ):
            assert actual_name == expected_name
            assert actual_parameter.grad is not None
            assert expected_parameter.grad is not None
            assert torch.isfinite(actual_parameter.grad).all()
            assert torch.allclose(
                actual_parameter.grad,
                expected_parameter.grad,
                atol=6.3e-2,
                rtol=5e-2,
            ), actual_name
    finally:
        lfm2_mod.get_rope.cache_clear()


def test_lfm2_packed_forward_matches_independent_sequences() -> None:
    torch.manual_seed(0)
    model = LFM2Model(_tiny_cfg()).eval()
    first = torch.randint(0, model.config.vocab_size, (1, 7))
    second = torch.randint(0, model.config.vocab_size, (1, 5))
    packed = torch.cat((first, second), dim=1)
    packed_positions = torch.tensor(
        [list(range(first.size(1))) + list(range(second.size(1)))],
        dtype=torch.long,
    )

    with torch.no_grad():
        packed_logits = model(packed, positions=packed_positions)
        first_logits = model(
            first,
            positions=torch.arange(first.size(1)).unsqueeze(0),
        )
        second_logits = model(
            second,
            positions=torch.arange(second.size(1)).unsqueeze(0),
        )

    expected = torch.cat((first_logits, second_logits), dim=1)
    assert torch.allclose(packed_logits, expected, atol=1e-5, rtol=1e-5)


def _assert_shortconv_packed_isolation(
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    torch.manual_seed(7)
    config = _tiny_cfg()
    conv = lfm2_mod.LFM2ShortConv(config, layer_idx=0).to(
        device=device,
        dtype=dtype,
    )
    first_length = 7
    second_length = 5
    second = torch.randn(
        1,
        second_length,
        config.hidden_size,
        device=device,
        dtype=dtype,
    )
    packed_a = torch.cat(
        (
            torch.randn(
                1,
                first_length,
                config.hidden_size,
                device=device,
                dtype=dtype,
            ),
            second,
        ),
        dim=1,
    ).requires_grad_()
    packed_b = torch.cat(
        (
            torch.randn(
                1,
                first_length,
                config.hidden_size,
                device=device,
                dtype=dtype,
            ),
            second,
        ),
        dim=1,
    )
    positions = torch.tensor(
        [list(range(first_length)) + list(range(second_length))],
        device=device,
        dtype=torch.long,
    )

    output_a = conv(packed_a, positions)
    output_b = conv(packed_b, positions)

    assert torch.equal(output_a[:, first_length:], output_b[:, first_length:])
    output_a[:, first_length:].float().square().sum().backward()
    assert packed_a.grad is not None
    assert torch.count_nonzero(packed_a.grad[:, :first_length]).item() == 0


def test_lfm2_shortconv_packed_boundary_isolation_cpu() -> None:
    _assert_shortconv_packed_isolation(
        device=torch.device("cpu"),
        dtype=torch.float32,
    )


def test_lfm2_shortconv_packed_boundary_isolation_cuda_bfloat16() -> None:
    if not _cuda_runtime_usable():
        pytest.skip("CUDA is required for the BF16 packed-boundary regression.")
    _assert_shortconv_packed_isolation(
        device=torch.device("cuda"),
        dtype=torch.bfloat16,
    )


@pytest.mark.parametrize("attn_backend", ["eager", "flash"])
def test_lfm2_packed_bfloat16_cuda_second_sequence_is_invariant(
    attn_backend: str,
) -> None:
    if not _cuda_runtime_usable():
        pytest.skip("CUDA is required for the BF16 packed-boundary regression.")
    if attn_backend == "flash" and not lfm2_mod.flash_attention_available():
        pytest.skip("FlashAttention is not installed.")

    torch.manual_seed(8)
    # Rotary modules are globally cached by shape; keep this CUDA model from
    # leaving the shared test-process cache on the accelerator.
    lfm2_mod.get_rope.cache_clear()
    config = _tiny_cfg()
    config.attn_backend = attn_backend
    try:
        model = LFM2Model(config).to(device="cuda", dtype=torch.bfloat16).eval()
        first_length = 7
        second_length = 5
        second = torch.randint(
            0,
            model.config.vocab_size,
            (1, second_length),
            device="cuda",
        )
        packed_a = torch.cat(
            (
                torch.randint(
                    0,
                    model.config.vocab_size,
                    (1, first_length),
                    device="cuda",
                ),
                second,
            ),
            dim=1,
        )
        packed_b = torch.cat(
            (
                torch.randint(
                    0,
                    model.config.vocab_size,
                    (1, first_length),
                    device="cuda",
                ),
                second,
            ),
            dim=1,
        )
        positions = torch.tensor(
            [list(range(first_length)) + list(range(second_length))],
            device="cuda",
            dtype=torch.long,
        )

        with torch.no_grad():
            output_a = model(packed_a, positions=positions)
            output_b = model(packed_b, positions=positions)

        assert torch.equal(output_a[:, first_length:], output_b[:, first_length:])
    finally:
        lfm2_mod.get_rope.cache_clear()


def test_lfm2_compile_step_tiny() -> None:
    if torch.cuda.is_available() and not _cuda_runtime_usable():
        pytest.skip("torch.compile probes CUDA even for CPU graphs on this host.")

    torch.manual_seed(0)
    model = LFM2Model(_tiny_cfg())
    model.train()
    compiled_model = torch.compile(model, dynamic=False)

    @torch.compile(dynamic=False)
    def compiled_step(inp: torch.Tensor, positions: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        logits = compiled_model(inp, positions=positions)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), target.view(-1))

    input_ids = torch.randint(0, model.config.vocab_size, (1, 8))
    positions = torch.arange(8, dtype=torch.long).unsqueeze(0)
    targets = torch.randint(0, model.config.vocab_size, (1, 8))

    try:
        loss = compiled_step(input_ids, positions, targets)
        loss.backward()
    except Exception as exc:
        if "cudaGetDeviceCount" in str(exc) or "CUDA initialization" in str(exc):
            pytest.skip("torch.compile hit an unusable CUDA runtime on this host.")
        raise

    assert torch.isfinite(loss)
    assert model.embed_tokens.weight.grad is not None
    assert torch.isfinite(model.embed_tokens.weight.grad).all()


def test_lfm2_warns_when_flash_attention_is_requested_but_unavailable(monkeypatch) -> None:
    config = _tiny_cfg()
    config.layer_types = ("full_attention", "full_attention")
    config.num_hidden_layers = 2
    config.attn_backend = "flash"

    monkeypatch.setattr(lfm2_mod, "flash_attn_varlen_func", None)
    monkeypatch.setattr(lfm2_mod, "_flash_attn_import_error", ImportError("flash_attn missing"))
    monkeypatch.setattr(lfm2_mod, "_flash_attn_warning_emitted", False)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        model = LFM2Model(config)

    assert model.layers[0].self_attn.attn is None
    assert any("FLASH ATTENTION IS NOT ACTIVE" in str(item.message) for item in caught)


def test_lfm2_shortconv_reference_matches_conv1d() -> None:
    torch.manual_seed(0)
    bx = torch.randn(1, 4, 8)
    weight = torch.randn(4, 1, 3)

    expected = F.conv1d(bx, weight, bias=None, stride=1, padding=2, groups=4)[..., :8]
    actual = depthwise_causal_conv_k3_reference(bx, weight)
    assert torch.allclose(actual, expected)


def test_lfm2_triton_shortconv_parity_cuda() -> None:
    if not (TRITON_AVAILABLE and _cuda_runtime_usable()):
        return

    torch.manual_seed(0)
    bx = torch.randn(1, 4, 16, device="cuda", dtype=torch.float32, requires_grad=True)
    weight = torch.randn(4, 1, 3, device="cuda", dtype=torch.float32, requires_grad=True)

    ref = depthwise_causal_conv_k3_reference(bx, weight)
    out = depthwise_causal_conv_k3(bx, weight)
    assert torch.allclose(out, ref, atol=1e-5, rtol=1e-4)


def test_lfm2_triton_fused_gated_shortconv_parity_cuda() -> None:
    if not (TRITON_AVAILABLE and _cuda_runtime_usable()):
        pytest.skip("CUDA and Triton are required for fused shortconv parity.")

    torch.manual_seed(47)
    positions = torch.tensor(
        [list(range(17)) + list(range(13)) + list(range(19))],
        device="cuda",
    )
    channels = 65
    bcx = torch.randn(
        1,
        positions.shape[1],
        3 * channels,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    weight = torch.randn(
        channels,
        1,
        3,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    reference_bcx = bcx.detach().clone().requires_grad_()
    reference_weight = weight.detach().clone().requires_grad_()
    b_gate, c_gate, x_proj = reference_bcx.transpose(1, 2).chunk(3, dim=1)
    expected = (
        c_gate
        * depthwise_causal_conv_k3(
            b_gate * x_proj,
            reference_weight,
            positions=positions,
        )
    ).transpose(1, 2).contiguous()
    actual = gated_depthwise_causal_conv_k3(
        bcx,
        weight,
        positions=positions,
    )

    upstream = torch.randn_like(actual)
    (expected * upstream).sum().backward()
    (actual * upstream).sum().backward()

    assert torch.equal(actual, expected)
    assert torch.equal(bcx.grad, reference_bcx.grad)
    assert torch.equal(weight.grad, reference_weight.grad)


def test_lfm2_triton_shortconv_masks_packed_boundaries_in_kernel_cuda() -> None:
    if not (TRITON_AVAILABLE and _cuda_runtime_usable()):
        pytest.skip("CUDA and Triton are required for packed shortconv parity.")

    torch.manual_seed(31)
    positions = torch.tensor(
        [[0, 1, 2, 3, 0, 1, 2, 0, 1]],
        device="cuda",
        dtype=torch.long,
    )
    bx = torch.randn(
        1,
        5,
        positions.shape[1],
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    weight = torch.randn(
        5,
        1,
        3,
        device="cuda",
        dtype=torch.float32,
        requires_grad=True,
    )
    reference_bx = bx.detach().clone().requires_grad_(True)
    reference_weight = weight.detach().clone().requires_grad_(True)

    reference_tokens = []
    for token_index in range(positions.shape[1]):
        token = reference_bx[:, :, token_index] * reference_weight[:, 0, 2]
        if int(positions[0, token_index]) >= 1:
            token = token + reference_bx[:, :, token_index - 1] * reference_weight[:, 0, 1]
        if int(positions[0, token_index]) >= 2:
            token = token + reference_bx[:, :, token_index - 2] * reference_weight[:, 0, 0]
        reference_tokens.append(token)
    reference = torch.stack(reference_tokens, dim=-1)
    actual = depthwise_causal_conv_k3(bx, weight, positions=positions)
    upstream = torch.randn_like(actual)
    reference.backward(upstream)
    actual.backward(upstream)

    assert torch.allclose(actual, reference, atol=1e-5, rtol=1e-4)
    assert torch.allclose(bx.grad, reference_bx.grad, atol=1e-5, rtol=1e-4)
    assert torch.allclose(weight.grad, reference_weight.grad, atol=1e-5, rtol=1e-4)


def test_lfm2_benchmark_parser_defaults() -> None:
    args = lfm2_bench.build_arg_parser().parse_args([])
    assert args.model_path == "saved_models/LFM2.5-1.2B-Base"


def test_run_torch_compile_benchmark_smoke(monkeypatch) -> None:
    stats = lfm2_bench.TimingStats(mean_ms=1.0, p50_ms=1.0, p90_ms=1.0)

    class _FakeModel(torch.nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.config = SimpleNamespace(vocab_size=16)

        def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
            return torch.zeros(
                input_ids.size(0),
                input_ids.size(1),
                self.config.vocab_size,
                dtype=torch.float32,
                device=input_ids.device,
            )

    monkeypatch.setattr(lfm2_bench, "_load_lfm2_model", lambda *args, **kwargs: _FakeModel())
    monkeypatch.setattr(lfm2_bench, "_timed_forward", lambda *args, **kwargs: stats)
    monkeypatch.setattr(lfm2_bench, "_timed_fwdbwd", lambda *args, **kwargs: stats)
    monkeypatch.setattr(torch, "compile", lambda model, **kwargs: model)

    result = lfm2_bench.run_torch_compile_benchmark(
        "saved_models/LFM2.5-1.2B-Base",
        batch_size=1,
        seq_len=8,
        warmup=1,
        iters=1,
        device="cpu",
        dtype="float32",
    )

    assert result.compiled_forward == stats
    assert result.compiled_fwdbwd == stats
    assert result.compile_wall_s >= 0.0
