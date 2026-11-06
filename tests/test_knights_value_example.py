"""The production profile/bundled banks are verified without any model calls."""

from dataclasses import replace
import importlib
import importlib.util
import json
from pathlib import Path

import pytest

from climbing_gym.config import ConfigManager
from climbing_gym.train import resolve_config, validate_config

EXAMPLE = Path(__file__).parents[1] / "examples" / "knights_value_mis_sc"


@pytest.fixture
def recipe(monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    return importlib.import_module("knights_recipe")


@pytest.mark.parametrize(
    "profile,train,dev,test,batch,tokens",
    [
        ("production", 512, 96, 384, 256, 4096),
        ("smoke", 8, 6, 6, 2, 256),
    ],
)
def test_standalone_preparation_verifies_exact_data_and_production_contract(
    recipe, tmp_path, profile, train, dev, test, batch, tokens
):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    root = tmp_path / profile
    manifest = recipe.prepare(root, model, profile=profile)
    assert [manifest["data"][split]["rows"] for split in ("train", "dev", "test")] == [
        train,
        dev,
        test,
    ]
    cfg = resolve_config(ConfigManager.from_toml(root / "train.toml").config)
    validate_config(cfg)
    assert cfg.inference.native_max_batch_size == cfg.inference.max_num_seqs == batch
    assert cfg.rollout.max_inflight_requests == cfg.rollout.max_total_requests == batch
    assert (
        cfg.inference.sampling.max_tokens
        == cfg.trainer.flashreinforce_fixed_token_normalizer
        == tokens
    )
    assert cfg.trainer.value_epochs == 2 and cfg.trainer.value_max_grad_norm == 10
    assert cfg.trainer.policy_max_grad_norm == 1 and cfg.trainer.value_mis_alpha == 0.4
    assert cfg.trainer.gamma == cfg.trainer.gae_lambda == 1
    assert cfg.policy_optimizer.type == "cpu_master_adamw"
    assert cfg.value_optimizer.type == "mapped_master_adamw"
    assert not cfg.trainer.optimizer_pin_state
    assert not cfg.inference.native_bf16_weight_packing  # Original FP8-native setting.
    assert not cfg.inference.chat_template_enable_thinking
    assert cfg.inference.assistant_prefill == "<thinking>\n"
    assert manifest["historical_initialization_verified"] is False
    assert cfg.wandb.mode == "online" and cfg.wandb.project == "simple-pt"
    if profile == "production":
        assert cfg.trainer.num_steps == 300
        assert (
            cfg.trainer.value_warmup_min_batches,
            cfg.trainer.value_warmup_max_batches,
            cfg.trainer.value_warmup_consecutive_ready,
        ) == (10, 30, 3)
        assert set(manifest["data"]["dev"]["per_cell"].values()) == {16}
        assert set(manifest["data"]["test"]["per_cell"].values()) == {64}
        assert manifest["data"]["dev"]["samples_per_puzzle"] == 4
    with pytest.raises(FileExistsError):
        recipe.prepare(root, model, profile=profile)


def test_readiness_route_rejects_legacy_ppo_warmup_field(recipe, tmp_path):
    model = tmp_path / "model"
    model.mkdir()
    (model / "config.json").write_text("{}")
    root = tmp_path / "run"
    recipe.prepare(root, model)
    cfg = resolve_config(ConfigManager.from_toml(root / "train.toml").config)
    with pytest.raises(ValueError, match="readiness-gated"):
        validate_config(
            replace(cfg, trainer=replace(cfg.trainer, critic_warmup_steps=10))
        )


def test_actor_retention_publishes_latest_exact_state_before_pruning(recipe, tmp_path):
    def boundary(update):
        path = tmp_path / "checkpoints" / f"step_{update - 1}"
        (path / "policy").mkdir(parents=True)
        (path / "policy/weights.bin").write_bytes(bytes([update]))
        for name in ("value_model.pt", "training_state.pt", "dataset_state.json"):
            (path / name).write_text("{}")
        (path / "metadata.json").write_text(json.dumps({"step": update - 1}))
        (path / "value_warmup_state.json").write_text(
            json.dumps({"state": {"ready": True, "actor_updates": update}})
        )
        return {"step": update - 1, "checkpoint_path": str(path)}

    for update in (10, 20, 30, 40):
        recipe.retain_checkpoints(tmp_path, boundary(update), final_updates=40)
    assert sorted(p.name for p in (tmp_path / "checkpoints").iterdir()) == [
        "step_29",
        "step_39",
    ]
    assert (tmp_path / "policy_exports/update_020/weights.bin").read_bytes() == bytes(
        [20]
    )
    assert (tmp_path / "policy_exports/update_040/weights.bin").read_bytes() == bytes(
        [40]
    )
    pointer = json.loads((tmp_path / "latest_actor_checkpoint.json").read_text())
    assert pointer["actor_updates"] == 40 and pointer["exact_resume"]
    incomplete = boundary(50)
    (Path(incomplete["checkpoint_path"]) / "dataset_state.json").unlink()
    with pytest.raises(RuntimeError, match="incomplete"):
        recipe.retain_checkpoints(tmp_path, incomplete, final_updates=50)
    assert (tmp_path / "checkpoints/step_29").exists()


def test_prompt_tokenizer_parity_is_checked_before_any_benchmark(recipe, monkeypatch):
    monkeypatch.syspath_prepend(str(EXAMPLE))
    spec = importlib.util.spec_from_file_location(
        "knights_value_runner", EXAMPLE / "run.py"
    )
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)

    class Tokenizer:
        def apply_chat_template(self, *args, **kwargs):
            assert kwargs["enable_thinking"] is False
            return "public prompt"

        def encode(self, text, **kwargs):
            assert text.endswith("<thinking>\n")
            return [1, 2, 3]

    checksum = recipe.digest(json.dumps([1, 2, 3], separators=(",", ":")).encode())
    row = dict(id="p", prompt="public prompt", source_prompt_token_ids_sha256=checksum)
    assert runner.prompt_ids([row], Tokenizer()) == [[1, 2, 3]]
    with pytest.raises(ValueError, match="Tokenizer changed"):
        runner.prompt_ids(
            [dict(row, source_prompt_token_ids_sha256="wrong")], Tokenizer()
        )
