"""CPU-only integration tests; model responses below are explicit fixtures."""
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from climbing_gym.config import Wandb
from climbing_gym.envs.contracts import EnvEvaluation, EnvExample, EnvFeedback, RewardResult
from climbing_gym.eval import EvalConfig, EvalLoggingConfig, ResolvedEvalRun, _trajectory_record, run_eval, summarize_trajectories
from climbing_gym.evaluation_traces import select_traces, write_trace_report
from climbing_gym.rollout.contracts import Trajectory, TrajectoryTurn
from climbing_gym.trainers.logger import WandbLogger


def trajectory(index=0, *, passed=True, difficulty="easy", response="<think>checked</think>\\boxed{2}"):
    example = EnvExample("What is 1+1?", "2", {"difficulty": difficulty}, index)
    evaluation = EnvEvaluation(example, response, "2", EnvFeedback(success=passed), RewardResult(float(passed), {"exact_match": float(passed)}))
    turn = TrajectoryTurn(1, example.prompt, response, (1, 2), (3, 4, 5), (-.1, -.2, -.3), "stop", "eos")
    return Trajectory(index, 0, example, (turn,), response, True, "single_turn", evaluation)


def test_disabled_logging_does_not_import_wandb_and_is_durable(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("disabled logger imported a package")
    monkeypatch.setattr("climbing_gym.trainers.logger.importlib.import_module", forbidden)
    cfg = Wandb(enabled=False, dir="logs")
    logger = WandbLogger(cfg, config_path=tmp_path / "run.toml", resolved_config={"model": {"path": Path("weights")}, "seed": 12})
    logger.log_step(step=3, metrics={"loss": 1.25, "eval/pass_at_1": .5})
    payload = json.loads(logger.metrics_path.read_text())
    assert payload == {"step": 3, "loss": 1.25, "eval/pass_at_1": .5}
    assert json.loads(logger.config_path.read_text())["config"]["seed"] == 12
    assert logger.log_dir == tmp_path / "logs"
    logger.close()
    logger.close()
    with pytest.raises(RuntimeError, match="closed"):
        logger.log_step(step=4, metrics={})


def test_reused_directory_preserves_every_runs_configuration(tmp_path):
    cfg = Wandb(enabled=False, dir=str(tmp_path))
    first = WandbLogger(cfg, config_path=tmp_path / "run.toml", resolved_config={"seed": 1})
    first.log_step(step=0, metrics={"loss": 3})
    original_config = first.config_path.read_bytes()
    original_metrics = first.metrics_path.read_bytes()
    first.close()
    second = WandbLogger(cfg, config_path=tmp_path / "run.toml", resolved_config={"seed": 2})
    second.close()
    assert second.config_path != first.config_path
    assert first.config_path.read_bytes() == original_config
    assert first.metrics_path.read_bytes() == original_metrics
    assert json.loads(second.config_path.read_text())["config"]["seed"] == 2


def test_online_uses_owned_run_full_configuration_and_finish(tmp_path, monkeypatch):
    calls = []
    run = SimpleNamespace(log=lambda payload, **kw: calls.append((payload, kw)), finish=lambda **kw: calls.append(("finish", kw)))
    def init(**kwargs):
        calls.append(kwargs)
        return run
    monkeypatch.setattr("climbing_gym.trainers.logger.importlib.import_module", lambda name: SimpleNamespace(init=init))
    with WandbLogger(Wandb(dir=str(tmp_path)), config_path=tmp_path / "job.toml", resolved_config={"model": {"path": "/model"}, "trainer": {"steps": 3}}) as logger:
        logger.log_step(step=1, metrics={"timing/full_step_seconds": 4.5})
    assert calls[0]["mode"] == "online"
    assert calls[0]["entity"] == "lukas928" and calls[0]["project"] == "simple-pt"
    assert calls[0]["config"]["trainer"] == {"steps": 3}
    assert calls[-1] == ("finish", {"exit_code": 0})


def test_mode_disabled_no_init_even_when_enabled(tmp_path, monkeypatch):
    monkeypatch.setattr("climbing_gym.trainers.logger.importlib.import_module", lambda name: pytest.fail("W&B import"))
    logger = WandbLogger(Wandb(enabled=True, mode="disabled", dir=str(tmp_path)), config_path=tmp_path / "x.toml")
    logger.close()


def test_initialization_error_preserves_configuration(tmp_path, monkeypatch):
    def missing(name):
        raise ModuleNotFoundError("no wandb")
    monkeypatch.setattr("climbing_gym.trainers.logger.importlib.import_module", missing)
    with pytest.raises(RuntimeError, match="authenticate.*offline"):
        WandbLogger(Wandb(dir=str(tmp_path)), config_path=tmp_path / "run.toml", resolved_config={"seed": 123})
    assert json.loads((tmp_path / "resolved_config.json").read_text())["config"]["seed"] == 123


def test_trace_stratification_lossless_fences_and_no_overwriting(tmp_path):
    response = "<think>\n```text\n" + "all output " * 2000 + "\n```\n</think>42"
    items = [trajectory(index, passed=passed, difficulty=tier, response=response)
             for index, (tier, passed) in enumerate((("easy", True), ("easy", False), ("hard", True), ("hard", False)))]
    selected = select_traces(reversed(items), max_traces=2)
    assert {(t.example.row["difficulty"], t.evaluation.feedback.success) for t in selected} == {("easy", True), ("easy", False), ("hard", True), ("hard", False)}
    assert select_traces(items) == select_traces(reversed(items))
    output = write_trace_report(tmp_path / "traces.md", items, max_traces=2)
    text = output.read_text()
    assert text.count(response) == 8  # complete per-turn and final response
    assert "````text" in text
    assert "Expected answer" in text and '"2"' in text
    second = write_trace_report(output, items)
    assert second != output and output.read_text() == text


def test_eval_mock_preserves_metrics_actual_fixture_traces_and_closes(tmp_path):
    items = [trajectory(), trajectory(1, passed=False, difficulty="hard")]
    states = []
    engine = SimpleNamespace(close=lambda: states.append("closed"))
    coordinator = SimpleNamespace(collect=lambda: SimpleNamespace(trajectories=items, stats={"output_tokens_per_second": 10}),
                                  env=SimpleNamespace(load_dataset_state_dict=lambda state: states.append(state)))
    run = ResolvedEvalRun(coordinator, engine, EvalConfig(), EvalLoggingConfig(summary_path=str(tmp_path / "summary.json"),
                          trajectory_path=str(tmp_path / "trajectories.jsonl")), 2, 1, initial_dataset_state={"cursor": 0})
    summary = run_eval(run)
    assert summary["pass_at_1"] == .5 and summary["reward_mean"] == .5
    assert summary["rollout/output_tokens_per_second"] == 10
    assert states == [{"cursor": 0}, "closed"]
    assert len(run.trace_paths) == 1 and run.trace_paths[0].exists()
    saved = json.loads((tmp_path / "summary.json").read_text())
    assert saved["trace_path"] == str(run.trace_paths[0])
    records = [json.loads(line) for line in (tmp_path / "trajectories.jsonl").read_text().splitlines()]
    assert records[0]["prompt"] == items[0].example.prompt
    assert records[0]["expected_answer"] == "2"
    assert records[0]["turns"][0]["response"] == items[0].turns[0].response


def test_eval_engine_closed_on_collection_failure(tmp_path):
    closed = []
    def fail():
        raise RuntimeError("inference failed")
    run = ResolvedEvalRun(SimpleNamespace(collect=fail), SimpleNamespace(close=lambda: closed.append(True)),
                          EvalConfig(), EvalLoggingConfig(trace_path=str(tmp_path / "x.md")), 1, 1)
    with pytest.raises(RuntimeError, match="inference failed"):
        run_eval(run)
    assert closed == [True]


def test_summary_ordered_pass_k_and_positive_dense_reward_failure():
    a = trajectory(passed=False)
    a = replace(a, evaluation=replace(a.evaluation, reward=RewardResult(.5)))
    b = replace(trajectory(passed=True), rollout_index=1)
    summary = summarize_trajectories([b, a], pass_k=2, num_rollouts=2)
    assert summary["pass_at_1"] == 0 and summary["pass_at_2"] == 1
    assert summary["avg_at_2"] == .75
    assert _trajectory_record(a)["passed"] is False
