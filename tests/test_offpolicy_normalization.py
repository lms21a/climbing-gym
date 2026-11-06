"""Length-bias regressions through the production packed MIS/SC update."""
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.flashreinforce import update_policy, validate_config
from climbing_gym.inference.types import BehaviorTopK
from climbing_gym.offpolicy import (
    corrected_response_loss, corrected_token_terms, resolve_response_normalization,
)
from test_flashreinforce import config
from rl_fixtures import TinyPolicy, trajectory


class BranchPolicy(torch.nn.Module):
    """One learned branch followed by effectively deterministic continuation."""
    def __init__(self):
        super().__init__()
        self.lm_head = torch.nn.Linear(2, 3, bias=False)
        with torch.no_grad():
            self.lm_head.weight.copy_(torch.tensor([[0., -80.], [0., -80.], [-80., 0.]]))

    def get_hidden_states(self, inputs, positions):
        first_prediction = inputs == 9
        return torch.stack((first_prediction, ~first_prediction), dim=-1).float()


class RecordingSGD(torch.optim.SGD):
    def step(self, closure=None):
        self.gradients = [p.grad.detach().clone() for g in self.param_groups for p in g['params']]
        return super().step(closure)


def branch_update(*, reward, mode='fixed', fixed=None, method='mis_sc', pack=128, fused=True):
    policy = BranchPolicy()
    data = []
    # The first two paths have equal outcomes and different lengths. Opposite-
    # outcome filler prompts make the actual finite batch mean exactly 1/2.
    for i, (result, length, choice) in enumerate([
            (reward, 2, 0), (reward, 16, 1), (1-reward, 16, 0), (1-reward, 16, 1)]):
        row = trajectory(result, length=1, group=i)
        tokens = (choice,) + (2,)*(length-1)
        prompt = (20+i, 9)
        ids = torch.tensor(prompt+tokens)
        lp = policy.lm_head(policy.get_hidden_states(ids[:-1], None)).detach().log_softmax(-1)[1:]
        sampled = lp.gather(1, torch.tensor(tokens)[:, None])[:, 0]
        head = BehaviorTopK(torch.arange(3, dtype=torch.int32).expand(length, -1).clone(), lp)
        capped = (i == 1 and not result) or (i == 2 and not result)
        turn = replace(row.turns[0], prompt_token_ids=prompt, completion_token_ids=tokens,
            completion_logprobs=tuple(sampled.tolist()), behavior_topk=head,
            finish_reason='length' if capped else 'stop')
        data.append(replace(row, turns=(turn,)))
    cfg = config(batch_size=4, flashreinforce_objective=method,
        flashreinforce_length_normalization=mode, flashreinforce_fixed_token_normalizer=fixed,
        flashreinforce_fused_head=fused, flashreinforce_checkpoint_head=False,
        flashreinforce_head_chunk_size=3, policy_max_packed_tokens=pack, policy_max_grad_norm=100.)
    cfg = replace(cfg, inference=replace(cfg.inference, sampling=replace(cfg.inference.sampling,
        max_tokens=16, extra_params={'behavior_topk': 3})))
    validate_config(cfg)
    optimizer = RecordingSGD(policy.parameters(), lr=.1)
    metrics, audit = update_policy(SimpleNamespace(config=cfg, policy_model=policy, policy_optimizer=optimizer), data)
    return optimizer.gradients[0], metrics, audit


@pytest.mark.parametrize('reward', [0., 1.])
@pytest.mark.parametrize('mode', ['fixed', 'legacy_response_mean'])
@pytest.mark.parametrize('method', ['mis', 'mis_sc'])
@pytest.mark.parametrize('pack', [18, 128])
@pytest.mark.parametrize('fused', [False, True])
def test_equal_outcome_paths_have_no_length_preference_under_fixed_default(reward, mode, method, pack, fused):
    gradient, metrics, audit = branch_update(reward=reward, mode=mode, method=method, pack=pack, fused=fused)
    assert metrics['policy/reward_mean'] == .5
    assert metrics['trainer/truncated_fraction'] == .25
    assert len(audit) == 4
    assert sorted(a['advantage'] for a in audit) == [-.5, -.5, .5, .5]
    assert all(a['retained_token_fraction'] == 1. for a in audit)
    if mode == 'fixed':
        # No preference between branch 0 (short) and 1 (long), for successes OR
        # failures, even with an included length-capped failure and split packs.
        torch.testing.assert_close(gradient[:2, 0], torch.zeros(2), atol=1e-8, rtol=0)
        assert metrics['flashreinforce/fixed_token_normalizer'] == 16
    else:
        expected = -(reward-.5)/8 * (1/2-1/16)
        torch.testing.assert_close(gradient[:2, 0], torch.tensor([expected, -expected]), atol=1e-8, rtol=0)
        assert metrics['flashreinforce/fixed_token_normalizer'] == 0
    assert metrics['flashreinforce/legacy_response_mean'] == float(mode != 'fixed')


def test_explicit_fixed_divisor_controls_production_gradient_scale():
    from copy import deepcopy
    from test_offpolicy import make_rows
    torch.manual_seed(25)
    original = TinyPolicy()
    data = make_rows(original)
    gradients = []
    for fixed in (8, 16):
        policy = deepcopy(original)
        optimizer = RecordingSGD(policy.parameters(), lr=.1)
        cfg = config(flashreinforce_objective='mis_sc', flashreinforce_fixed_token_normalizer=fixed,
                     policy_max_grad_norm=100.)
        metrics, _ = update_policy(SimpleNamespace(config=cfg, policy_model=policy, policy_optimizer=optimizer), data)
        gradients.append(optimizer.gradients)
        assert metrics['flashreinforce/fixed_token_normalizer'] == fixed
    assert max(g.abs().max() for g in gradients[0]) > 1e-3
    for first, second in zip(*gradients):
        torch.testing.assert_close(first, 2*second)


def test_masking_and_partial_memory_packs_keep_original_batch_and_fixed_divisor():
    current = torch.full((8,), -2., requires_grad=True)
    behavior = current.detach().clone()
    behavior[0] = -8.  # Rejected MIS ratio; it must not renormalize other tokens.
    terms = corrected_token_terms(current, behavior)
    corrected_response_loss(terms.objective, torch.tensor([.5, -.5]), (2, 6),
        total_sequences=4, token_normalizer=16).backward()
    expected = torch.tensor([0., -.5]+[.5]*6)/64
    torch.testing.assert_close(current.grad, expected)


def test_fixed_default_uses_effective_sampling_budget_and_serializes_explicit_scale():
    cfg = config(flashreinforce_objective='mis')
    assert resolve_response_normalization(cfg) == ('fixed', cfg.inference.sampling.max_tokens)
    cfg = replace(cfg, rollout=replace(cfg.rollout, sampling=replace(cfg.inference.sampling, max_tokens=79)))
    assert resolve_response_normalization(cfg) == ('fixed', 79)
    cfg = replace(cfg, trainer=replace(cfg.trainer, flashreinforce_fixed_token_normalizer=1024))
    assert resolve_response_normalization(cfg) == ('fixed', 1024)
    assert cfg.to_dict()['trainer']['flashreinforce_fixed_token_normalizer'] == 1024


@pytest.mark.parametrize('options', [
    {'flashreinforce_length_normalization': 'response_mean'},
    {'flashreinforce_fixed_token_normalizer': 0},
    {'flashreinforce_fixed_token_normalizer': -1},
    {'flashreinforce_fixed_token_normalizer': True},
    {'flashreinforce_fixed_token_normalizer': 1.5},
    {'flashreinforce_fixed_token_normalizer': float('nan')},
    {'flashreinforce_length_normalization': 'legacy_response_mean', 'flashreinforce_fixed_token_normalizer': 16},
])
def test_invalid_normalization_fails_before_inference(options):
    with pytest.raises(ValueError):
        validate_config(config(flashreinforce_objective='mis', **options))


def test_legacy_reduction_exactly_matches_historical_formula():
    objective = torch.linspace(-2., -.1, 7, requires_grad=True)
    advantage = torch.tensor([.5, -.5])
    actual = corrected_response_loss(objective, advantage, (2, 5), total_sequences=3,
                                     normalization='legacy_response_mean')
    expected = -(objective * torch.tensor([.5/2/3]*2+[-.5/5/3]*5)).sum()
    torch.testing.assert_close(actual, expected, atol=0, rtol=0)
