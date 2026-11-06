from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace
import importlib
from pathlib import Path

import pytest
import torch

from climbing_gym.grouped_mis import classify_groups, fixed_length_loss, grouped_advantages
from climbing_gym.grouped_sampling import SurplusGroupQueue
from climbing_gym.inference.types import BehaviorTopK
from climbing_gym.offpolicy import update_grouped_corrected_policy
from test_flashreinforce import config
from rl_fixtures import TinyPolicy, trajectory


def grouped_rows(policy):
    result = []
    for i, (reward, length) in enumerate(((1., 3), (0., 5), (0., 1), (1., 4))):
        group = i//2
        row = trajectory(reward, length=length, group=group)
        row = replace(row, group_id=group, rollout_index=i%2,
                      example=replace(row.example, dataset_index=group))
        turn = row.turns[0]
        ids = torch.tensor(turn.prompt_token_ids+turn.completion_token_ids)
        lp = policy.lm_head(policy.embed_tokens(ids[:-1])).detach().log_softmax(-1)[1:]
        q = (lp+torch.linspace(-.8, .8, 16)).softmax(-1)
        sampled = torch.tensor(turn.completion_token_ids)
        head = BehaviorTopK(torch.arange(16, dtype=torch.int32).expand(length, -1).clone(), q.log())
        turn = replace(turn, completion_logprobs=tuple(q.gather(1, sampled[:, None]).log()[:, 0].tolist()),
                       behavior_topk=head)
        result.append(replace(row, turns=(turn,)))
    return result


def test_fixed_global_denominator_does_not_discount_tokens_in_long_failed_responses():
    values = torch.full((10,), -2., requires_grad=True)
    advantages = torch.tensor([-.5]*2+[-.5]*8)
    fixed_length_loss(values, advantages, total_sequences=2, max_tokens=16).backward()
    torch.testing.assert_close(values.grad, torch.full((10,), .5/32))
    assert values.grad[2:].sum() == pytest.approx(4*values.grad[:2].sum())


def test_dynamic_sampling_retains_entire_mixed_groups_only():
    result = classify_groups([0., 0., 1., 0., 1., 1.], 2)
    assert result == dict(mixed=[1], all_wrong=[0], all_correct=[2])
    for values in ([0., 1., 0.], [0., .5], []):
        with pytest.raises(ValueError):
            classify_groups(values, 2)


def test_group_baselines_remain_independent_and_have_no_std_scaling():
    model = TinyPolicy()
    data = grouped_rows(model)
    advantages, metrics = grouped_advantages(data, group_size=2, groups_per_update=2)
    assert advantages == [.5, -.5, -.5, .5]
    assert metrics['group/std_normalization'] == 0.
    # An incomplete or homogeneous group must not silently enter a learner update.
    with pytest.raises(ValueError, match='Incomplete retained'):
        grouped_advantages(data[:-1], group_size=2, groups_per_update=2)
    wrong = replace(data[0], evaluation=replace(data[0].evaluation,
        reward=replace(data[0].evaluation.reward, reward=0.)))
    with pytest.raises(ValueError, match='mixed'):
        grouped_advantages([wrong]+data[1:], group_size=2, groups_per_update=2)


def test_groups_with_different_reward_variances_keep_raw_centered_advantages():
    source = grouped_rows(TinyPolicy())
    data = []
    for group, rewards in enumerate(([1.]+[0.]*7, [1.]*4+[0.]*4)):
        for rollout, reward in enumerate(rewards):
            row = source[group*2]
            data.append(replace(row, rollout_index=rollout, evaluation=replace(row.evaluation,
                reward=replace(row.evaluation.reward, reward=reward))))
    advantages, _ = grouped_advantages(data, group_size=8, groups_per_update=2)
    assert advantages == [.875]+[-.125]*7+[.5]*4+[-.5]*4


@pytest.mark.parametrize('fused', [False, True])
@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required'))])
def test_grouped_mis_sc_update_is_invariant_to_memory_packing(fused, device):
    torch.manual_seed(819)
    original = TinyPolicy()
    data = grouped_rows(original)
    models, audits = [], []
    for pack, chunk in [(6, 2), (128, 512)]:
        model = deepcopy(original).to(device)
        cfg = config(flashreinforce_objective='mis_sc', flashreinforce_head_chunk_size=chunk,
            flashreinforce_fused_head=fused, policy_max_packed_tokens=pack, policy_max_grad_norm=100.,
            flashreinforce_fixed_token_normalizer=7)  # Ungrouped option must not affect Dr. GRPO's 16.
        cfg = replace(cfg, trainer=replace(cfg.trainer, train_device=device, algorithm='grpo', advantage_estimator='grpo',
            loss_aggregation='fixed_max_tokens'),
            rollout=replace(cfg.rollout, num_rollouts=2, effective_batch_size=2),
            inference=replace(cfg.inference, sampling=replace(cfg.inference.sampling, max_tokens=16)))
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        metrics, audit = update_grouped_corrected_policy(SimpleNamespace(config=cfg,
            policy_model=model, policy_optimizer=optimizer), data)
        assert metrics['group/fixed_token_normalizer'] == 16
        assert metrics['offpolicy/score_centering'] == 1.
        assert not any(k.startswith('flashreinforce/') for k in metrics)
        assert metrics['policy/optimizer_steps'] == 1.
        models.append(model)
        audits.append(audit)
    for a, b in zip(models[0].parameters(), models[1].parameters()):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    assert [a['group_id'] for a in audits[0]] == [0, 0, 1, 1]
    assert any(not torch.equal(a, b.cpu()) for a, b in zip(original.parameters(), models[0].parameters()))
