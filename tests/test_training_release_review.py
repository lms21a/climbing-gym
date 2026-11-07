"""Release regressions for numerical failure boundaries and configuration."""

from copy import deepcopy
from dataclasses import replace

import pytest
import torch

from climbing_gym.mis_ppo import update_policy
from climbing_gym.config import Optimizer
from climbing_gym.ppo import build_value_model_from_policy
from climbing_gym.train import validate_config
from test_flashreinforce import config
from test_training_runtime import run_fixture


def mis_config(**changes):
    cfg = config()
    return replace(
        cfg,
        trainer=replace(
            cfg.trainer,
            algorithm="mis_ppo",
            advantage_estimator="value",
            mis_token_ratio_min=0.1,
            mis_token_ratio_max=10.0,
            mis_trajectory_ratio_min=0.1,
            mis_trajectory_ratio_max=10.0,
            **changes,
        ),
    )


@pytest.mark.parametrize(
    "field",
    [
        "mis_token_ratio_min",
        "mis_token_ratio_max",
        "mis_trajectory_ratio_min",
        "mis_trajectory_ratio_max",
    ],
)
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_mis_bounds_fail_before_rollouts(field, value):
    cfg = mis_config()
    with pytest.raises(ValueError, match="bounds must be finite"):
        validate_config(replace(cfg, trainer=replace(cfg.trainer, **{field: value})))


def test_nonfinite_critic_gradient_does_not_apply_actor_or_optimizer_step():
    torch.manual_seed(49)
    run = run_fixture(mis_config())
    critic = build_value_model_from_policy(run.policy_model)
    actor_optimizer = torch.optim.AdamW(run.policy_model.parameters(), lr=0.01)
    critic_optimizer = torch.optim.AdamW(critic.parameters(), lr=0.01)
    run = replace(
        run,
        policy_optimizer=actor_optimizer,
        value_model=critic,
        value_optimizer=critic_optimizer,
    )
    actor_before = deepcopy(run.policy_model.state_dict())
    critic_before = deepcopy(critic.state_dict())
    hook = next(critic.value_head.parameters()).register_hook(
        lambda gradient: torch.full_like(gradient, float("nan"))
    )
    try:
        with pytest.raises(RuntimeError, match="non-finite"):
            update_policy(run, run.rollout_coordinator.rows)
    finally:
        hook.remove()
    assert not actor_optimizer.state and not critic_optimizer.state
    for name, tensor in run.policy_model.state_dict().items():
        torch.testing.assert_close(tensor, actor_before[name], rtol=0, atol=0)
    for name, tensor in critic.state_dict().items():
        torch.testing.assert_close(tensor, critic_before[name], rtol=0, atol=0)


@pytest.mark.parametrize("field", ["policy_max_grad_norm", "value_max_grad_norm"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0, 0.0])
def test_mis_invalid_clip_limit_fails_before_rollouts(field, value):
    cfg = mis_config()
    with pytest.raises(ValueError, match=field):
        validate_config(replace(cfg, trainer=replace(cfg.trainer, **{field: value})))


@pytest.mark.parametrize("algorithm", ["ppo", "grpo", "cispo"])
def test_legacy_trainers_keep_explicit_no_clipping_convention(algorithm):
    cfg = config()
    cfg = replace(
        cfg,
        trainer=replace(
            cfg.trainer,
            algorithm=algorithm,
            advantage_estimator="grpo",
            loss_aggregation="seq_mean_token_sum"
            if algorithm == "cispo"
            else "token_mean",
            policy_max_grad_norm=0.0,
            value_max_grad_norm=0.0,
        ),
    )
    validate_config(cfg)


@pytest.mark.parametrize("field", ["learning_rate", "weight_decay"])
@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1.0])
def test_optimizer_rejects_nonfinite_or_negative_settings(field, value):
    with pytest.raises(ValueError, match=field):
        Optimizer(type="master_adamw", **{field: value})


def test_optimizer_requires_actual_fused_boolean_and_keeps_zero_lr():
    with pytest.raises(ValueError, match="fused"):
        Optimizer(fused="false")
    assert (
        Optimizer(learning_rate=0.0, weight_decay=0.0, fused=False).learning_rate == 0.0
    )


@pytest.mark.parametrize(
    "changes,expected",
    [
        ({"privileged_critic": True}, "ordinary critic"),
        ({"loss_aggregation": "seq_mean_token_sum"}, "token_mean"),
        ({"loss_aggregation": "sample_mean"}, "token_mean"),
    ],
)
def test_ppo_rejects_ignored_critic_or_reduction_options(changes, expected):
    cfg = config()
    trainer = replace(
        cfg.trainer,
        algorithm="ppo",
        advantage_estimator="gae",
        loss_aggregation="token_mean",
    )
    with pytest.raises(ValueError, match=expected):
        validate_config(replace(cfg, trainer=replace(trainer, **changes)))


@pytest.mark.parametrize("high", [None, 0.2])
def test_ppo_keeps_symmetric_clip_aliases(high):
    cfg = config()
    cfg = replace(
        cfg,
        trainer=replace(
            cfg.trainer,
            algorithm="ppo",
            advantage_estimator="gae",
            loss_aggregation="token_mean",
            clip_epsilon=0.2,
            clip_epsilon_high=high,
        ),
    )
    validate_config(cfg)


def test_ppo_rejects_ignored_asymmetric_clipping():
    cfg = config()
    cfg = replace(
        cfg,
        trainer=replace(
            cfg.trainer,
            algorithm="ppo",
            advantage_estimator="gae",
            loss_aggregation="token_mean",
            clip_epsilon=0.2,
            clip_epsilon_high=0.4,
        ),
    )
    with pytest.raises(ValueError, match="symmetric clipping"):
        validate_config(cfg)
