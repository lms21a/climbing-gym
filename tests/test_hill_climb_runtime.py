"""CPU checks for historical stage identity and exact paired-cohort continuation."""

from copy import deepcopy
from dataclasses import replace
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.config import ConfigManager, Env
from climbing_gym.trainers.checkpoints import CheckpointManager, CheckpointManagerConfig
from rl_fixtures import TinyPolicy
from test_flashreinforce import config
from test_offpolicy import make_rows


EXAMPLE = Path(__file__).resolve().parents[1] / "examples/futoshiki_hill_climb"
sys.path.insert(0, str(EXAMPLE))
recipe = importlib.import_module("recipe")
prepare = importlib.import_module("prepare")
rl = importlib.import_module("rl")
tasks = importlib.import_module("tasks")


class Policy(TinyPolicy):
    def save_as_hf(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "weights.pt")


class Tokenizer:
    def save_pretrained(self, path):
        (Path(path) / "tokenizer.json").write_text("{}")


class Engine:
    def __init__(self):
        self.events = []

    def sync_from_model(self, *args, **kwargs):
        self.events.append("sync")

    def release_inference_buffers(self):
        self.events.append("release")


class Logger:
    def __init__(self):
        self.metrics = []

    def log_step(self, **payload):
        self.metrics.append(payload)


def make_run(folder, model):
    folder.mkdir(parents=True)
    (folder / "data").mkdir()
    (folder / "data/train.jsonl").write_text(
        "".join(json.dumps({"id": i}) + "\n" for i in range(4))
    )
    cfg = config(
        batch_size=2,
        flashreinforce_objective="mis_sc",
        flashreinforce_fused_head=False,
        policy_max_grad_norm=100.0,
    )
    spec = dict(
        recipe.stage(folder, "tier2_rl100"),
        steps=2,
        batch_size=2,
        cohort_size=4,
        checkpoint_interval=1,
    )
    return SimpleNamespace(
        folder=folder,
        spec=spec,
        config=cfg,
        policy_model=model,
        policy_optimizer=torch.optim.AdamW(model.parameters(), lr=0.01),
        engine=Engine(),
        tokenizer=Tokenizer(),
        wandb_logger=Logger(),
        checkpoint_manager=CheckpointManager(
            CheckpointManagerConfig(root_dir=str(folder / "checkpoints"))
        ),
        release_inference_buffers_before_update=True,
    )


def test_recipe_retains_real_ancestry_and_algorithm_semantics():
    stages = recipe.load_recipe()["stages"]
    assert (
        stages["gsm8k_16"]["steps"] == 16
        and stages["gsm8k_16"]["retained_updates"] == 10
    )
    assert stages["countdown3_13"]["input_policy"] == "gsm8k_16@10"
    assert stages["futoshiki4_rl750"]["max_tokens"] == 4096
    assert stages["futoshiki4_rl1000"]["max_tokens"] == 8192
    assert stages["futoshiki4_rl1000"]["train_slice"] == [96000, 128000]
    assert stages["projection_reset"]["input_policy"] == "futoshiki4_rl1000"
    assert stages["projection_reset"]["experts"] == {
        "5x5_g11_i5": "specialist_easy250",
        "5x5_g9_i5": "specialist_tier2@70",
    }
    for name in ["tier2_rl100", "tier3_rl100", "tier4_rl100"]:
        spec = stages[name]
        assert (
            spec["steps"] == 100
            and spec["cohort_size"] == 256
            and spec["batch_size"] == 128
        )
        assert (
            spec["objective"] == "mis_sc"
            and spec["normalization"] == "legacy_response_mean"
        )
    assert stages["tier2_rl100"]["optimizer_transfer"] is False
    assert stages["tier3_rl100"]["optimizer_transfer"] is True


def test_historical_data_prepare_uses_real_rows_and_native_portable_config(tmp_path):
    root = tmp_path / "run"
    prepare.prepare(
        root,
        tmp_path / "base",
        artifacts={"projection_reset": tmp_path / "student"},
        stages=["tier2_rl100"],
    )
    folder = root / "stages/tier2_rl100"
    rows = recipe.read_rows(folder / "data/train.jsonl")
    assert len(rows) == 12800 and {r["tier"] for r in rows} == {"5x5_g9_i5"}
    assert len({r["id"] for r in rows}) == 12800
    assert (
        recipe.digest(folder / "data/train.jsonl")
        == "a97d18e2651db447a315ce74b8cf491c853367e8df8213ba1c372082f5c85482"
    )
    cfg = ConfigManager.from_toml(folder / "train.toml").config
    assert (
        cfg.inference.native_max_batch_size == cfg.rollout.max_inflight_requests == 256
    )
    assert cfg.trainer.flashreinforce_length_normalization == "legacy_response_mean"
    assert cfg.policy_optimizer.type == "cpu_master_adamw"
    assert cfg.env.reward == "exact_match_strict"
    assert cfg.model.path == str(tmp_path / "student")
    assert (
        cfg.wandb.enabled
        and cfg.wandb.project == "climbing-gym"
        and cfg.wandb.entity is None
    )
    assert (
        Env(name="futoshiki", reward="exact_match_strict").build().reward_profile
        == "exact_match_strict"
    )
    with pytest.raises(ValueError, match="manifest differs"):
        prepare.prepare(root, tmp_path / "different", stages=["tier2_rl100"])


def test_fresh_generator_reserves_disjoint_heldout_targets_and_terminates():
    from climbing_gym.futoshiki import Puzzle, solve

    banks = prepare.generate_disjoint_banks(
        size=4,
        settings=[(6, 3)],
        counts=dict(train=8, monitor=2, test=2, calibration=2),
        seed=414,
        max_attempts_per_row=100,
    )
    owners = {}
    for split, rows in banks.items():
        assert len(rows) == (8 if split == "train" else 2)
        for row in rows:
            target = tuple(row["solution"])
            assert target not in owners or owners[target] == split
            owners[target] = split
            assert solve(Puzzle.from_row(row)) == [target]


def test_pending_second_half_survives_checkpoint_without_recollection(
    tmp_path, monkeypatch
):
    torch.manual_seed(112)
    initial = Policy()
    original = make_rows(initial)
    rows = [original[0], original[1], original[0], original[1]]
    traces = tuple(
        replace(t, group_id=i, example=replace(t.example, dataset_index=i))
        for i, t in enumerate(rows)
    )
    calls = []

    def collect(run, rows, *, start, behavior_version):
        calls.append((str(run.folder), start, behavior_version))
        assert start == 0 and behavior_version == 0
        return traces, dict(tokens_per_second=100.0, generated_tokens=16)

    monkeypatch.setattr(rl, "collect_native", collect)
    full = make_run(tmp_path / "full", deepcopy(initial))
    interrupted = make_run(tmp_path / "interrupted", deepcopy(initial))
    rl.train_loop(full, [{}] * 4, evaluations=False)
    save = rl.save_checkpoint

    def stop_after_first(run, **kwargs):
        path = save(run, **kwargs)
        if kwargs["completed"] == 1:
            raise RuntimeError("simulated process interruption")
        return path

    monkeypatch.setattr(rl, "save_checkpoint", stop_after_first)
    with pytest.raises(RuntimeError, match="simulated"):
        rl.train_loop(interrupted, [{}] * 4, evaluations=False)
    checkpoint = interrupted.folder / "checkpoints/step_0"
    assert (checkpoint / "policy/tokenizer.json").exists()
    metadata = recipe.read_json(checkpoint / "metadata.json")
    assert metadata["pending_cohort_responses"] == 2
    resumed = make_run(tmp_path / "resumed", Policy())
    resumed.policy_model.load_state_dict(
        torch.load(checkpoint / "policy/weights.pt", weights_only=True)
    )
    rl.restore_optimizer_checkpoint(resumed.policy_optimizer, checkpoint)
    completed, next_index, pending, behavior = rl.resume_state(resumed, checkpoint)
    assert completed == 1 and next_index == 4 and len(pending) == 2 and behavior == 0
    monkeypatch.setattr(rl, "save_checkpoint", save)
    rl.train_loop(
        resumed,
        [{}] * 4,
        completed=completed,
        next_index=next_index,
        pending=pending,
        behavior_version=behavior,
        evaluations=False,
    )
    assert len(calls) == 2  # One original collection per branch, none on resume.
    for a, b in zip(
        full.policy_model.parameters(), resumed.policy_model.parameters(), strict=True
    ):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
        assert resumed.policy_optimizer.state[b]["step"] == 2
    assert [
        m["metrics"]["hill_climb/policy_lag"] for m in full.wandb_logger.metrics
    ] == [0, 1]
    assert (
        resumed.wandb_logger.metrics[0]["metrics"]["hill_climb/reused_pending_cohort"]
        == 1
    )
    (resumed.folder / "data/train.jsonl").write_text("different bank\n")
    with pytest.raises(ValueError, match="dataset differs"):
        rl.resume_state(resumed, checkpoint)


def test_precursor_grade_preserves_terminal_numeric_and_truncation_contract():
    response = "Calculate 2+2=4.\n</thinking>\n<answer>4</answer>"
    assert tasks.grade_gsm8k(response, "4")["reward"] == 1
    assert tasks.grade_gsm8k(response, "4", truncated=True)["reward"] == 0
    assert tasks.grade_gsm8k(response + " and more", "4")["reward"] == 0
    assert tasks.grade_gsm8k(response + "<answer>4</answer>", "4")["reward"] == 0
    assert (
        tasks.grade_countdown(
            "Use 2+3*4.\n</thinking>\n<answer>2+3*4</answer>",
            {"numbers": [2, 3, 4], "operators": "+-*/", "target": 14},
        )["reward"]
        == 1
    )


def test_endpoint_workloads_include_8k_test_before_4k_training(tmp_path, monkeypatch):
    spec = recipe.stage(tmp_path, "futoshiki4_rl750")
    folder = tmp_path / "stage"
    (folder / "data").mkdir(parents=True)
    (folder / "data/test.jsonl").write_text("{}\n")
    workloads = rl.execution_workloads(spec, folder)
    assert workloads == {(4096, 8192): "train", (8192, 9216): "test"}
    checked = []
    monkeypatch.setattr(
        rl,
        "validate_preflight",
        lambda cfg: (
            checked.append(
                (cfg.inference.sampling.max_tokens, cfg.inference.max_model_len)
            )
            or {"accepted": True}
        ),
    )
    rl.validate_evaluation_reports(config(), spec, folder)
    assert checked == [(8192, 9216)]


def test_evaluation_uses_panel_budget_repeats_and_pass8(tmp_path, monkeypatch):
    run = make_run(tmp_path / "eval_run", Policy())
    run.spec = dict(
        run.spec,
        max_tokens=4096,
        evaluation={
            "test": dict(max_tokens=8192, context=9216, repeats=8, max_examples=2)
        },
    )
    (run.folder / "data/test.jsonl").write_text("{}\n{}\n")
    base = make_rows(run.policy_model)
    traces = tuple([base[0]] + [base[1]] * 15)
    contexts = []
    monkeypatch.setattr(
        rl, "ensure_context", lambda actual, context: contexts.append(context)
    )

    def collect(actual, rows, **kwargs):
        assert len(rows) == 16
        assert actual.spec["max_tokens"] == 8192
        return traces, {"tokens_per_second": 1.0}

    monkeypatch.setattr(rl, "collect_native", collect)
    summary = rl.evaluate_panel(run, "test", completed=750)
    assert contexts == [9216]
    assert summary["samples"] == 16 and summary["max_tokens"] == 8192
    assert summary["pass_at_1"] == 1 / 16
    assert summary["pass_at_4"] == 0.25
    assert summary["pass_at_8"] == 0.5
    trace = run.folder / "eval/update_750/test.traces.md"
    assert "Complete model response:" in trace.read_text()
    assert "Expected answer:" in trace.read_text()


def test_regenerated_4x4_phases_slice_one_shared_bank(tmp_path, monkeypatch):
    banks = prepare.generate_disjoint_banks(
        size=4,
        settings=[(6, 3)],
        counts=dict(train=8, monitor=2, test=2, calibration=2),
        seed=812,
    )
    recipe.write_json(tmp_path / ".generated/futoshiki4.json", banks)
    original = recipe.stage

    def small(root, name):
        spec = original(root, name)
        spec["train_slice"] = [0, 4] if name == "futoshiki4_rl750" else [4, 8]
        return spec

    monkeypatch.setattr(prepare, "stage", small)
    prepare.regenerate_futoshiki(tmp_path, "futoshiki4_rl750")
    prepare.regenerate_futoshiki(tmp_path, "futoshiki4_rl1000")
    first = recipe.read_rows(tmp_path / "stages/futoshiki4_rl750/data/train.jsonl")
    second = recipe.read_rows(tmp_path / "stages/futoshiki4_rl1000/data/train.jsonl")
    assert first + second == json.loads(json.dumps(banks["train"]))
    assert not {row["id"] for row in first} & {row["id"] for row in second}


def test_status_shows_pending_prerequisite_and_resume_availability(tmp_path, capsys):
    root = tmp_path / "run"
    prepare.prepare(root, tmp_path / "base", stages=["tier2_rl100"])
    assert rl.main(["status", "--root", str(root)]) == 0
    line = capsys.readouterr().out.strip()
    assert line.startswith("tier2_rl100 ")
    payload = json.loads(line.split(" ", 1)[1])
    assert payload["state"] == "pending" and payload["ready"] is False
    assert payload["exact_resume_checkpoints"] == []
    assert payload["prerequisites"]["projection_reset"]["available"] is False


def test_standalone_evaluation_validates_checkpoint_panel_before_loading_model(
    tmp_path, monkeypatch
):
    root = tmp_path / "run"
    prepare.prepare(root, tmp_path / "base", stages=["tier2_rl100"])
    checkpoint = tmp_path / "selected_checkpoint"
    (checkpoint / "policy").mkdir(parents=True)
    seen = []

    def preflight(cfg):
        seen.append(cfg)
        raise RuntimeError("stop before model loading")

    monkeypatch.setattr(rl, "validate_preflight", preflight)
    with pytest.raises(RuntimeError, match="before model loading"):
        rl.load_run(
            root, "tier2_rl100", checkpoint, evaluation=True, evaluation_panel="test"
        )
    assert seen[0].inference.model_path == str(checkpoint / "policy")
    assert seen[0].inference.sampling.max_tokens == 8192
    assert seen[0].inference.preflight_report_path.endswith(
        "evaluate_selected_checkpoint_test.json"
    )


def test_pruned_optimizer_is_never_a_fresh_continuation(tmp_path):
    checkpoint = tmp_path / "policy_only"
    (checkpoint / "policy").mkdir(parents=True)
    optimizer = torch.optim.AdamW(Policy().parameters(), lr=0.01)
    with pytest.raises(ValueError, match="policy-only or pruned"):
        rl.restore_optimizer_checkpoint(optimizer, checkpoint)
