"""CPU toy policies exercise the real warmup/critic/actor and checkpoint path."""

from copy import deepcopy
from dataclasses import replace
import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.dataset_checkpoint import restore_dataset_checkpoint
from climbing_gym.train import resolve_config, run_training, validate_config
from climbing_gym.trainers.checkpoints import (
    CheckpointManager,
    CheckpointManagerConfig,
    restore_training_state,
)
from climbing_gym.value_mis_sc import build_critic
from climbing_gym.value_warmup import WarmupState, load_state, save_state
from test_flashreinforce import config
from test_training_runtime import Coordinator, run_fixture
from test_value_mis_sc import TinyBackbone, Tokenizer


class Policy(TinyBackbone):
    def save_as_hf(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), Path(path) / "weights.pt")


class SavingTokenizer(Tokenizer):
    def save_pretrained(self, path):
        (Path(path) / "tokenizer_config.json").write_text("{}")


class Cursor:
    _dataset_cursor = True

    def __init__(self):
        self.position = 0

    def dataset_state_dict(self):
        return {"position": self.position}

    def load_dataset_state_dict(self, payload):
        self.position = payload["position"]


class FreshCoordinator(Coordinator):
    def __init__(self, engine, rows):
        super().__init__(engine, rows)
        self.env = Cursor()
        self.rollout_runner = SimpleNamespace(_unfinished_groups=[], _next_group_id=0)
        self.draws = []

    def collect(self):
        self.draws.append((self.env.position, float(torch.rand(()))))
        self.env.position += 1
        self.rollout_runner._next_group_id += len(self.rows)
        return super().collect()


def make_run(root, *, minimum=3, maximum=5, consecutive=3, num_steps=1):
    cfg = config(
        value_epochs=2,
        policy_max_grad_norm=1.0,
        value_max_grad_norm=10.0,
        flashreinforce_fused_head=False,
    )
    cfg = resolve_config(
        replace(
            cfg,
            trainer=replace(
                cfg.trainer,
                algorithm="value_mis_sc",
                advantage_estimator="length_adaptive_gae",
                num_steps=num_steps,
                value_warmup_min_batches=minimum,
                value_warmup_max_batches=maximum,
                value_warmup_consecutive_ready=consecutive,
            ),
            checkpoint=CheckpointManagerConfig(root_dir=str(root), save_final=True),
            inference=replace(
                cfg.inference,
                sampling=replace(
                    cfg.inference.sampling, extra_params={"behavior_topk": 16}
                ),
            ),
        )
    )
    validate_config(cfg)
    run = run_fixture(cfg, Policy())
    critic = build_critic(run.policy_model)
    return replace(
        run,
        policy_optimizer=torch.optim.AdamW(run.policy_model.parameters(), lr=1e-3),
        value_model=critic,
        value_optimizer=torch.optim.AdamW(
            (p for p in critic.parameters() if p.requires_grad), lr=1e-3
        ),
        tokenizer=SavingTokenizer(),
        checkpoint_manager=CheckpointManager(cfg.checkpoint),
        rollout_coordinator=FreshCoordinator(run.engine, run.rollout_coordinator.rows),
    )


def gate(monkeypatch, *, good=True):
    import climbing_gym.value_mis_sc as values

    original = values.value_diagnostics

    def diagnostics(data, predictions):
        return {
            **original(data, predictions),
            "mc_ev": 0.2 if good else -0.2,
            "mc_mse": 0.2 if good else 0.3,
            "constant_baseline_mse": 0.25,
            "calibration_bias": 0.01,
        }

    monkeypatch.setattr(values, "value_diagnostics", diagnostics)


def test_gate_requires_prefit_streak_and_minimum_and_resets_on_failure():
    trainer = make_run(
        Path("/unused"), minimum=4, maximum=6, consecutive=2
    ).config.trainer
    state = WarmupState()
    good = dict(
        mc_ev=0.1, mc_mse=0.2, constant_baseline_mse=0.25, calibration_bias=0.01
    )
    assert state.observe(good, trainer)
    assert state.observe(good, trainer)
    assert not state.ready  # Sufficient streak alone does not bypass the minimum.
    assert not state.observe(dict(good, calibration_bias=0.08), trainer)
    assert state.consecutive == 0  # Historical comparison is strictly less than .08.
    state.observe(good, trainer)
    assert not state.ready
    state.observe(good, trainer)
    assert state.ready and state.batches == 5


def test_real_critic_warmup_keeps_actor_and_optimizer_untouched_then_trains(
    monkeypatch, tmp_path
):
    gate(monkeypatch)
    torch.manual_seed(13)
    run = make_run(tmp_path)
    actor = deepcopy(run.policy_model.state_dict())
    critic = deepcopy(run.value_model.state_dict())
    observed = []

    def observe(payload):
        observed.append(payload)
        if payload.get("phase") == "critic_warmup":
            assert not run.policy_optimizer.state
            assert all(
                torch.equal(actor[k], v)
                for k, v in run.policy_model.state_dict().items()
            )
            assert payload["metrics"]["policy/optimizer_steps"] == 0

    run_training(run, observe_step=observe)
    assert [p["batch"] for p in observed if p.get("phase") == "critic_warmup"] == [
        1,
        2,
        3,
    ]
    assert any(
        not torch.equal(actor[k], v) for k, v in run.policy_model.state_dict().items()
    )
    assert any(
        not torch.equal(critic[k], v) for k, v in run.value_model.state_dict().items()
    )
    state = json.loads((tmp_path / "step_0/value_warmup_state.json").read_text())[
        "state"
    ]
    assert state == dict(
        batches=3, consecutive=3, ready=True, failed=False, actor_updates=1
    )
    assert [p["step"] for p in run.wandb_logger.metrics] == [0, 1, 2, 3]
    assert not (tmp_path / "critic_warmup/step_0").exists()
    assert (tmp_path / "critic_warmup/step_2/policy/tokenizer_config.json").is_file()
    assert (tmp_path / "step_0/policy/tokenizer_config.json").is_file()
    assert json.loads((tmp_path / "latest_value_warmup.json").read_text())["ready"]


def test_failed_gate_saves_failed_phase_and_never_starts_actor(monkeypatch, tmp_path):
    gate(monkeypatch, good=False)
    run = make_run(tmp_path, minimum=1, maximum=2, consecutive=1)
    actor = deepcopy(run.policy_model.state_dict())
    with pytest.raises(RuntimeError, match="actor updates have not started"):
        run_training(run)
    assert not run.policy_optimizer.state
    assert all(
        torch.equal(actor[k], v) for k, v in run.policy_model.state_dict().items()
    )
    checkpoint = tmp_path / "critic_warmup/step_1"
    cfg = replace(
        run.config, resume=replace(run.config.resume, checkpoint_dir=str(checkpoint))
    )
    state = load_state(cfg, actor_start_step=0)
    assert state.failed and not state.ready
    resumed = replace(run, config=cfg, value_warmup_state=state)
    before = len(run.rollout_coordinator.draws)
    with pytest.raises(RuntimeError, match="readiness failed"):
        run_training(resumed)
    assert len(run.rollout_coordinator.draws) == before


@pytest.mark.parametrize("completed_warmup", [2, 3])
def test_exact_midwarmup_resume_matches_uninterrupted_phases_and_parameters(
    monkeypatch, tmp_path, completed_warmup
):
    gate(monkeypatch)
    torch.manual_seed(18)
    reference = make_run(tmp_path / "reference")
    run_training(reference)
    checkpoint = tmp_path / f"reference/critic_warmup/step_{completed_warmup - 1}"
    resumed = make_run(tmp_path / "resumed")
    resumed.policy_model.load_state_dict(
        torch.load(checkpoint / "policy/weights.pt", weights_only=True)
    )
    resumed.value_model.load_state_dict(
        torch.load(checkpoint / "value_model.pt", weights_only=True)["state_dict"]
    )
    # This fake collector stores behavior probabilities rather than decoding.
    # Use the same frozen actor's rows after restoring its weights.
    resumed.rollout_coordinator.rows = deepcopy(reference.rollout_coordinator.rows)
    cfg = replace(
        resumed.config,
        resume=replace(resumed.config.resume, checkpoint_dir=str(checkpoint)),
    )
    validate_config(cfg)
    restore_training_state(
        checkpoint,
        policy_optimizer=resumed.policy_optimizer,
        value_optimizer=resumed.value_optimizer,
    )
    restore_dataset_checkpoint(resumed.rollout_coordinator, checkpoint)
    state = load_state(cfg, actor_start_step=0)
    assert state.batches == completed_warmup and state.consecutive == completed_warmup
    assert state.ready == (completed_warmup == 3)
    run_training(replace(resumed, config=cfg, value_warmup_state=state))
    assert (
        resumed.rollout_coordinator.draws
        == reference.rollout_coordinator.draws[completed_warmup:]
    )
    for expected, actual in zip(
        reference.policy_model.parameters(), resumed.policy_model.parameters()
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    for expected, actual in zip(
        reference.value_model.parameters(), resumed.value_model.parameters()
    ):
        torch.testing.assert_close(expected, actual, rtol=0, atol=0)
    assert [p["step"] for p in resumed.wandb_logger.metrics] == list(
        range(completed_warmup, 4)
    )


def test_phase_resume_rejects_missing_changed_or_skipped_state(tmp_path):
    run = make_run(tmp_path)
    directory = tmp_path / "legacy"
    directory.mkdir()
    cfg = replace(
        run.config, resume=replace(run.config.resume, checkpoint_dir=str(directory))
    )
    with pytest.raises(ValueError, match="lacks readiness"):
        load_state(cfg, actor_start_step=0)
    save_state(directory, WarmupState(batches=2, consecutive=2), cfg)
    (directory / "dataset_state.json").write_text("{}")
    with pytest.raises(ValueError, match="counters or actor phase"):
        load_state(cfg, actor_start_step=1)
    changed = replace(cfg, trainer=replace(cfg.trainer, value_warmup_min_batches=4))
    with pytest.raises(ValueError, match="configuration differs"):
        load_state(changed, actor_start_step=0)
    changed_method = replace(cfg, trainer=replace(cfg.trainer, value_epochs=3))
    with pytest.raises(ValueError, match="configuration differs"):
        load_state(changed_method, actor_start_step=0)
    disabled = replace(
        cfg,
        trainer=replace(
            cfg.trainer, value_warmup_min_batches=0, value_warmup_max_batches=0
        ),
    )
    with pytest.raises(ValueError, match="must retain"):
        load_state(disabled, actor_start_step=0)
