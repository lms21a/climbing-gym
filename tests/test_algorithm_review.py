"""Regressions for integration boundaries found in the independent RL review."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.config import Optimizer
from climbing_gym.grpo import _build_policy_optimizer
from climbing_gym.mis_ppo import trajectory_is_truncated, update_policy
from climbing_gym.train import resolve_config, validate_config
from test_flashreinforce import config
from test_training_runtime import run_fixture
from test_value_mis_sc import TinyBackbone, Tokenizer


def value_config(**changes):
    cfg = config()
    return resolve_config(replace(cfg,
        trainer=replace(cfg.trainer, algorithm='value_mis_sc',
                        advantage_estimator='length_adaptive_gae', **changes),
        inference=replace(cfg.inference, sampling=replace(cfg.inference.sampling,
                           extra_params={'behavior_topk': 16}))))


@pytest.mark.parametrize('alpha', [0., -1., float('nan'), float('inf')])
def test_invalid_value_alpha_fails_before_collection(alpha):
    with pytest.raises(ValueError, match='value_mis_alpha'):
        validate_config(value_config(value_mis_alpha=alpha))


@pytest.mark.parametrize('reason,finish,expected', [
    ('single_turn', 'length', True), ('single_turn', 'stop', False),
    ('single_turn', 'eos', False), ('budget_exhausted', 'stop', True),
    ('max_turns_reached', 'stop', True), ('unknown', 'length', False),
])
def test_direct_length_cap_is_distinguished_from_eos_and_string_stop(reason, finish, expected):
    row = SimpleNamespace(terminal_reason=reason,
                          turns=(SimpleNamespace(finish_reason=finish),))
    assert trajectory_is_truncated(row) is expected


def test_native_direct_bootstrap_changes_only_capped_outcomes():
    class ConstantCritic(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.value = torch.nn.Parameter(torch.tensor(.5))

        def forward(self, inputs, positions=None):
            return self.value.expand_as(inputs)

    cfg = config(mis_bootstrap_truncated_trajectories=True)
    run = run_fixture(cfg)
    rows = []
    for index, row in enumerate(run.rollout_coordinator.rows):
        rows.append(replace(row,
            evaluation=replace(row.evaluation, reward=replace(row.evaluation.reward, reward=0.)),
            turns=(replace(row.turns[0], finish_reason='stop' if index == 0 else 'length'),)))
    critic = ConstantCritic()
    run = replace(run, value_model=critic, value_optimizer=torch.optim.SGD(critic.parameters(), lr=.1))
    _, audit = update_policy(run, rows)
    assert [record['reward'] for record in audit] == [0., .5, .5]


@pytest.mark.parametrize('kind', ['adamw', 'master_adamw', 'cpu_master_adamw'])
def test_shared_critic_optimizer_registers_only_trainable_parameters(kind):
    model = TinyBackbone()
    model.embed_tokens.weight.requires_grad_(False)
    optimizer = _build_policy_optimizer(model, {'type': kind, 'learning_rate': 1e-5})
    actual = {id(p) for group in optimizer.param_groups for p in group['params']}
    assert actual == {id(p) for p in model.parameters() if p.requires_grad}


def test_mapped_critic_loader_excludes_frozen_tensors(monkeypatch, tmp_path):
    import climbing_gym.grpo as grpo
    import climbing_gym.train as train
    import climbing_gym.models.lfm2_compat as compat

    cfg = replace(value_config(value_optimizer_backing_path=str(tmp_path/'critic.bin')),
                  value_optimizer=Optimizer(type='mapped_master_adamw'))
    base = run_fixture(cfg, TinyBackbone())
    monkeypatch.setattr(train.ConfigManager, 'from_toml', lambda _: SimpleNamespace(config=cfg))
    monkeypatch.setattr(grpo, 'load_run_from_toml', lambda *args, **kwargs: base)
    monkeypatch.setattr(compat, 'load_lfm2_tokenizer', lambda _: Tokenizer())
    run = train.load_run_from_toml('unused.toml')
    run.value_optimizer.initialize_state()
    expected = {id(p) for p in run.value_model.parameters() if p.requires_grad}
    actual = {id(p) for p in run.value_optimizer.state}
    assert actual == expected
    expected_bytes = 12 * sum(p.numel() for p in run.value_model.parameters() if p.requires_grad)
    assert run.value_optimizer.backing_path.stat().st_size == expected_bytes


def actor_only_config(**changes):
    cfg = config()
    return replace(cfg,
        trainer=replace(cfg.trainer, algorithm='mis_ppo', advantage_estimator='grpo',
                        mis_use_value=False, **changes),
        rollout=replace(cfg.rollout, num_rollouts=2, batch_size=1, effective_batch_size=1,
                        max_total_requests=2, max_inflight_requests=2))


@pytest.mark.parametrize('changes', [
    {'entropy_coef': .01}, {'mis_kl_coef': .01},
    {'mis_bootstrap_truncated_trajectories': True}, {'mis_normalize_advantages': False},
])
def test_actor_only_mis_rejects_ignored_objective_settings(changes):
    with pytest.raises(ValueError, match='Actor-only MIS-PPO'):
        validate_config(actor_only_config(**changes))


def test_actor_only_mis_accepts_source_objective_and_rejects_singleton_baseline():
    cfg = actor_only_config(entropy_coef=0., mis_kl_coef=0.)
    validate_config(cfg)
    with pytest.raises(ValueError, match='at least two rollouts'):
        validate_config(replace(cfg, rollout=replace(cfg.rollout, num_rollouts=1)))
