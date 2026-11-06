"""Exercise both native updater paths with real packed trajectory metadata."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.mis_ppo import update_policy
from climbing_gym.ppo import build_value_model_from_policy
from test_flashreinforce import config
from rl_fixtures import TinyPolicy, trajectory


class Policy(TinyPolicy):
    def forward(self, inputs, positions=None):
        return self.lm_head(self.get_hidden_states(inputs, positions))


@pytest.mark.parametrize('with_value', [False, True])
def test_native_mis_updater_changes_parameters_and_uses_global_trajectory_gate(with_value):
    torch.manual_seed(25)
    policy = Policy()
    rows = []
    for i, (reward, length) in enumerate(((1., 3), (0., 4))):
        row = trajectory(reward, length=length, group=0, rollout=i)
        turn = row.turns[0]
        ids = torch.tensor([turn.prompt_token_ids + turn.completion_token_ids])
        lp = policy(ids[:, :-1]).detach().log_softmax(-1)
        sampled = lp.gather(-1, ids[:, 1:, None]).squeeze(-1)[0, 1:]
        rows.append(replace(row, turns=(replace(turn, completion_logprobs=tuple(sampled.tolist())),)))
    original = [p.detach().clone() for p in policy.parameters()]
    cfg = config(batch_size=2, policy_max_packed_tokens=6, policy_max_grad_norm=100.)
    run = SimpleNamespace(config=cfg, policy_model=policy,
        policy_optimizer=torch.optim.SGD(policy.parameters(), lr=.1))
    if with_value:
        run.value_model = build_value_model_from_policy(policy)
        run.value_optimizer = torch.optim.SGD(run.value_model.parameters(), lr=.1)
    metrics, audit = update_policy(run, rows)
    assert metrics['policy/optimizer_steps'] == 1.
    assert metrics['mis_ppo/trajectory_keep_rate'] == 1.
    assert metrics['mis_ppo/valid_tokens'] == 7.
    assert all(record['trajectory_admitted'] for record in audit)
    assert any(not torch.equal(a, b) for a, b in zip(original, policy.parameters()))
    if with_value:
        assert metrics['mis_ppo/value_loss'] > 0
