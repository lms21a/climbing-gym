"""Portable release contracts; no checkpoint weights, GPU work, or model evaluation."""
from pathlib import Path
import subprocess
import sys
import tomllib
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("command", ["--help", "--version"])
def test_public_cli_is_lazy(command):
    script = (
        "import sys; sys.path.insert(0, sys.argv.pop(1)); "
        "from climbing_gym.cli import main\n"
        "try: main(sys.argv[1:])\n"
        "except SystemExit as exc: assert exc.code == 0\n"
        "assert 'torch' not in sys.modules; assert 'wandb' not in sys.modules"
    )
    completed = subprocess.run(
        [sys.executable, "-I", "-S", "-c", script, str(ROOT / "src"), command],
        check=True, capture_output=True, text=True,
    )
    assert "climbing-gym" in completed.stdout


@pytest.mark.parametrize("command,module", [
    ("train", "train"), ("evaluate", "eval"), ("sft", "sft"),
    ("project", "sampling"), ("preflight", "preflight"),
    ("benchmark", "inference.benchmark"),
])
def test_cli_dispatches_to_real_workflow(command, module, monkeypatch):
    import climbing_gym.cli as cli
    calls = []
    def importer(name):
        assert name == "climbing_gym." + module
        return SimpleNamespace(main=lambda argv: calls.append(argv))
    monkeypatch.setattr(cli, "import_module", importer)
    assert cli.main([command, "--example-option", "value"]) == 0
    assert calls == [["--example-option", "value"]]


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("train_*.toml")), ids=lambda p: p.stem)
def test_all_training_examples_validate_without_weights(path, tmp_path, monkeypatch):
    from climbing_gym.config import ConfigManager
    from climbing_gym.preflight import resolve_limits
    from climbing_gym.train import resolve_config, validate_config
    monkeypatch.setenv("CLIMBING_GYM_MODEL", str(tmp_path / "checkpoint-not-loaded"))
    config = resolve_config(ConfigManager.from_toml(path).config)
    validate_config(config)
    assert config.model.path == str(tmp_path / "checkpoint-not-loaded")
    assert config.policy_optimizer.type == "cpu_master_adamw"
    assert config.trainer.optimizer_workspace_bytes == 48 * 1024**2
    assert config.wandb.enabled and config.wandb.mode == "online"
    assert config.inference.sampling.max_tokens == 4096
    assert config.inference.sampling.logprobs == 1
    assert resolve_limits(config.inference, config.rollout)["representative_batch_size"] >= 2
    env = config.env.build()
    rows = config.data.load_rows()
    assert len(rows) == 3
    for i, row in enumerate(rows):
        assert env.build_example(row, dataset_index=i).prompt


@pytest.mark.parametrize("path", sorted((ROOT / "configs").glob("eval_*.toml")), ids=lambda p: p.stem)
def test_all_evaluation_examples_parse_without_weights(path, tmp_path, monkeypatch):
    import climbing_gym.eval as evaluation
    monkeypatch.setenv("CLIMBING_GYM_MODEL", str(tmp_path / "checkpoint-not-loaded"))
    # Exercise the actual standalone loader and dataset/environment resolution,
    # replacing only model construction and the prerequisite measured report.
    monkeypatch.setattr(evaluation, "_build_inference_engine", lambda *args: SimpleNamespace(close=lambda: None))
    monkeypatch.setattr(evaluation, "validate_preflight", lambda config: {"test_only": True})
    run = evaluation.load_run_from_toml(path)
    assert run.total_examples == 3
    assert run.logging_config.trace_path == str(ROOT / "evaluations" / path.stem.removeprefix("eval_") / "traces.md")
    assert not run.rollout_coordinator.config.resolved_runner().use_dynamic_sampling


def test_sft_example_and_wandb_path_resolution(tmp_path, monkeypatch):
    from climbing_gym.sft import _make_sft_config
    monkeypatch.setenv("CLIMBING_GYM_MODEL", str(tmp_path / "model"))
    monkeypatch.setenv("PACKAGE_TEST_LOGS", "relative-logs")
    path = ROOT / "configs/sft.toml"
    payload = tomllib.loads(path.read_text())
    payload["wandb"]["dir"] = "${PACKAGE_TEST_LOGS}/sft"
    config = _make_sft_config(path, payload)
    assert config.wandb.dir == str(path.parent / "relative-logs/sft")
    assert config.sft.thinking_tag == "thinking"
    assert not config.sft.apply_chat_template
    assert config.data.dataset_path == str(ROOT / "runs/projection/projected_train.jsonl")


def test_environment_paths_are_relative_to_config_and_missing_vars_fail(tmp_path, monkeypatch):
    from climbing_gym.config import ConfigManager
    from climbing_gym.config.manager import _resolve_path
    monkeypatch.setenv("PACKAGE_TEST_PATH", "relative/model")
    assert _resolve_path("${PACKAGE_TEST_PATH}", base_dir=tmp_path) == str(tmp_path / "relative/model")
    monkeypatch.delenv("PACKAGE_TEST_MISSING", raising=False)
    with pytest.raises(ValueError, match="Unset environment variable.*PACKAGE_TEST_MISSING"):
        _resolve_path("$PACKAGE_TEST_MISSING/model", base_dir=tmp_path)
    monkeypatch.setenv("CLIMBING_GYM_MODEL", "../models/local")
    path = ROOT / "configs/train_mis.toml"
    assert ConfigManager.from_toml(path).config.model.path == str(ROOT / "models/local")
