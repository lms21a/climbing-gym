from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.config import Env, Harness, Inference, PPOConfig, PPOTrainer, Rollout
from climbing_gym.flashreinforce import (
    centered_advantages, flashreinforce_loss, sampled_action_kl, update_policy,
    validate_config, validate_trajectories,
)
from climbing_gym.inference.config import SamplingConfig
from rl_fixtures import TinyPolicy, trajectory


def config(batch_size=3, **options):
    return PPOConfig(
        env=Env(name='futoshiki', reward='exact_match'), harness=Harness(name='direct'),
        trainer=PPOTrainer(algorithm='flashreinforce', advantage_estimator='batch_centered',
                           loss_aggregation='sample_mean', compile_losses=False,
                           train_device='cpu', train_dtype=torch.float32, **options),
        rollout=Rollout(num_rollouts=1, batch_size=batch_size, effective_batch_size=batch_size,
                        max_total_requests=batch_size, use_dynamic_sampling=False),
        inference=Inference(backend='lfm25_native', sampling=SamplingConfig(temperature=1, logprobs=1)),
    )


def test_center_over_independent_prompts_without_std_or_leave_one_out():
    assert centered_advantages([0, 1, 0]) == pytest.approx([-1/3, 2/3, -1/3])
    assert centered_advantages([0, 0, 0]) == [0, 0, 0]
    assert centered_advantages([1, 1]) == [0, 0]
    for invalid in ([], [1], [0, float('nan')]):
        with pytest.raises(ValueError):
            centered_advantages(invalid)


def test_equation_11_detached_ratio_and_advantage_gradient():
    behavior = torch.tensor([.1, .3, .6, .2, .1]).log().requires_grad_()
    current = torch.tensor([.2, .2, .5, .4, .3]).log().requires_grad_()
    adv = torch.tensor([.5, -.5], requires_grad=True)
    result = flashreinforce_loss(current, behavior, adv, lengths=[2, 3], batch_size=2, delta=10)
    result.loss.backward()
    expected = -torch.tensor([2, 2/3, 5/6, 2, 3]) * torch.tensor([.5/4, .5/4, -.5/6, -.5/6, -.5/6])
    torch.testing.assert_close(current.grad, expected)
    assert behavior.grad is None and adv.grad is None


def test_proxy_direction_matches_bernoulli_distribution_kl():
    p = torch.tensor([.1, .2, .7, .95], dtype=torch.float64)
    q = torch.tensor([.4, .3, .8, .5], dtype=torch.float64)
    expected = torch.distributions.kl_divergence(torch.distributions.Bernoulli(p), torch.distributions.Bernoulli(q))
    torch.testing.assert_close(sampled_action_kl(p.log(), q.log()), expected)
    assert torch.isfinite(sampled_action_kl(torch.tensor([0., -1000.]), torch.tensor([-1e-9, -999.]))).all()


def test_reject_entire_trajectory_keep_original_denominator_and_baseline():
    behavior = torch.full((4,), .5).log()
    current = torch.tensor([.9, .5, .5, .5]).log().requires_grad_()
    result = flashreinforce_loss(current, behavior, torch.tensor([.5, -.5]),
                                 lengths=[2, 2], batch_size=2, delta=.1)
    result.loss.backward()
    assert result.admitted.tolist() == [False, True]
    torch.testing.assert_close(current.grad, torch.tensor([0., 0., .125, .125]))


def test_sample_mean_removes_length_multiplier_and_is_packing_invariant():
    old = torch.full((7,), -2.)
    full = old.clone().requires_grad_()
    result = flashreinforce_loss(full, old, torch.tensor([.5, -.5]), lengths=[2, 5], batch_size=2)
    result.loss.backward()
    split = old.clone().requires_grad_()
    for start, end, adv in ((0, 2, .5), (2, 7, -.5)):
        flashreinforce_loss(split[start:end], old[start:end], torch.tensor([adv]),
                            lengths=[end-start], batch_size=2).loss.backward()
    torch.testing.assert_close(split.grad, full.grad)
    assert full.grad[:2].sum().abs() == pytest.approx(full.grad[2:].sum().abs())


def test_large_admitted_ratio_is_not_clipped_at_twenty_or_ppo_bounds():
    current = torch.tensor([-30., -2.], requires_grad=True)
    old = torch.tensor([-60., -2.])
    result = flashreinforce_loss(current, old, torch.tensor([.5, -.5]), lengths=[1, 1], batch_size=2)
    result.loss.backward()
    assert result.admitted.tolist() == [True, True]
    assert current.grad[0].item() == pytest.approx(-.25 * torch.exp(torch.tensor(30.)).item())


def test_rejected_extreme_ratio_cannot_poison_accepted_gradient():
    current = torch.tensor([-.7, -2.], requires_grad=True)
    result = flashreinforce_loss(current, torch.tensor([-1000., -2.]), torch.tensor([.5, -.5]),
                                 lengths=[1, 1], batch_size=2)
    result.loss.backward()
    torch.testing.assert_close(current.grad, torch.tensor([0., .25]))
    assert torch.isfinite(result.loss)


@pytest.mark.parametrize('all_rewards', [[0, 0], [1, 1]])
def test_uniform_outcome_batches_have_zero_policy_gradient(all_rewards):
    current = torch.full((2,), -2., requires_grad=True)
    flashreinforce_loss(current, current.detach(), torch.tensor(centered_advantages(all_rewards)),
                        lengths=[1, 1], batch_size=2).loss.backward()
    torch.testing.assert_close(current.grad, torch.zeros_like(current))


def test_one_optimizer_step_matches_independent_reference_across_memory_packs():
    torch.manual_seed(39)
    policy = TinyPolicy()
    rows = [trajectory(r, length=n, group=i) for i, (r, n) in enumerate(((1, 3), (0, 5), (0, 1)))]
    for i, row in enumerate(rows):
        turn = row.turns[0]
        ids = torch.tensor([turn.prompt_token_ids + turn.completion_token_ids])
        lp = policy.lm_head(policy.embed_tokens(ids[:, :-1])).log_softmax(-1)
        selected = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0, 1:]
        rows[i] = replace(row, turns=(replace(turn, completion_logprobs=tuple(selected.detach().tolist())),))
    expected = deepcopy(policy)
    reference_optimizer = torch.optim.SGD(expected.parameters(), lr=.1)
    for row, adv in zip(rows, centered_advantages([1, 0, 0])):
        t = row.turns[0]
        ids = torch.tensor([t.prompt_token_ids + t.completion_token_ids])
        lp = expected.lm_head(expected.embed_tokens(ids[:, :-1])).log_softmax(-1)
        selected = lp.gather(-1, ids[:, 1:].unsqueeze(-1)).squeeze(-1)[0, 1:]
        (-adv * selected.mean() / 3).backward()
    reference_optimizer.step()

    class CountingSGD(torch.optim.SGD):
        count = 0
        def step(self, *args, **kwargs):
            self.count += 1
            return super().step(*args, **kwargs)

    for max_tokens in (6, 128):
        actual = deepcopy(policy)
        optimizer = CountingSGD(actual.parameters(), lr=.1)
        cfg = config(policy_max_packed_tokens=max_tokens, policy_max_grad_norm=100)
        metrics, audit = update_policy(SimpleNamespace(config=cfg, policy_model=actual,
                                                       policy_optimizer=optimizer), rows)
        assert optimizer.count == 1 and metrics['policy/optimizer_steps'] == 1
        assert [row['dataset_index'] for row in audit] == [0, 1, 2]
        for p, q in zip(actual.parameters(), expected.parameters()):
            torch.testing.assert_close(p, q)


@pytest.mark.parametrize('section,changes', [
    ('trainer', dict(policy_epochs=2)), ('trainer', dict(loss_aggregation='token_mean')),
    ('trainer', dict(advantage_estimator='grpo')), ('trainer', dict(entropy_coef=.01)),
    ('trainer', dict(flashreinforce_delta=float('nan'))), ('trainer', dict(critic_warmup_steps=20)),
    ('rollout', dict(num_rollouts=8)), ('rollout', dict(use_dynamic_sampling=True)),
    ('rollout', dict(max_total_requests=64)), ('harness', dict(name='infinity_think')),
])
def test_invalid_algorithm_combinations_fail_before_inference(section, changes):
    cfg = config()
    validate_config(cfg)
    with pytest.raises(ValueError):
        validate_config(replace(cfg, **{section: replace(getattr(cfg, section), **changes)}))


@pytest.mark.parametrize('changes', [dict(temperature=.7), dict(top_p=.9), dict(top_k=5),
                                    dict(logprobs=None), dict(n=2), dict(extra_params={'repetition_penalty':1.2})])
def test_sampling_logprobs_must_be_the_actual_full_support_behavior(changes):
    cfg = config()
    with pytest.raises(ValueError):
        validate_config(replace(cfg, inference=replace(cfg.inference, sampling=replace(cfg.inference.sampling, **changes))))


def test_missing_behavior_or_duplicate_prompts_are_not_silently_dropped():
    a, b = trajectory(group=0), trajectory(group=1)
    validate_trajectories([a, b], 2)
    with pytest.raises(RuntimeError, match='different prompts'):
        validate_trajectories([a, a], 2)
    with pytest.raises(RuntimeError, match='Missing actual behavior'):
        validate_trajectories([a, replace(b, turns=(replace(b.turns[0], completion_logprobs=None),))], 2)




@pytest.mark.skipif(not torch.cuda.is_available(), reason='Native sampler uses CUDA')
def test_native_stored_sampling_logprob_matches_actual_temperature_one_distribution():
    from climbing_gym.inference.lfm25_sampling import sample_logits
    logits = torch.randn((4, 4099), device='cuda', dtype=torch.bfloat16)
    tokens, logprobs = sample_logits(logits, temperature=1, seed=12, step=1, return_logprobs=True)
    expected = logits.float().log_softmax(-1).gather(-1, tokens[:, None]).squeeze(-1)
    torch.testing.assert_close(logprobs, expected, atol=2e-6, rtol=2e-6)
