"""PPO losses, temporal credit, and independent critic initialization."""
import pytest
import torch

from climbing_gym.ppo import (
    BoundedValueHead, build_value_model_from_policy, compute_gae,
    compute_bpco_trajectory_token_targets, packed_ppo_policy_loss,
    packed_value_loss, scale_ppo_loss_for_packed_batch,
)
from rl_fixtures import TinyPolicy, trajectory


def test_gae_terminal_outcome_and_length_adaptive_targets():
    advantages, returns = compute_gae([0., 0., 1.], [.2, .3, .4], gamma=1., gae_lambda=1.)
    assert advantages == pytest.approx([.8, .7, .6])
    assert returns == pytest.approx([1., 1., 1.])
    data = [trajectory(1., length=4), trajectory(0., length=2)]
    values = {(0, 0): (.5,)*4, (1, 0): (.5,)*2}
    adv, targets, stats = compute_bpco_trajectory_token_targets(data,
        sequence_values=values, gamma=1., length_adaptive_gae_alpha=.4)
    assert targets == {(0, 0): (1.,)*4, (1, 0): (0.,)*2}
    assert adv[(0, 0)][-1] == pytest.approx(.5)
    assert adv[(1, 0)][-1] == pytest.approx(-.5)
    assert abs(adv[(0, 0)][0]) < .5


def test_ppo_clip_blocks_only_the_wrong_side_of_the_surrogate():
    lp = torch.tensor([1.5, .5, .9]).log().requires_grad_()
    loss, clipped, ratio, kl = packed_ppo_policy_loss(target_log_probs=lp,
        old_infer_logprobs=torch.zeros(3), token_advantages=torch.tensor([1., -1., 1.]),
        valid_mask=torch.ones(3, dtype=torch.bool), clip_epsilon=.2)
    loss.backward()
    torch.testing.assert_close(lp.grad, torch.tensor([0., 0., -.3]))
    assert clipped == pytest.approx(2/3)


def test_value_loss_ignores_prompt_tokens_and_global_packed_scaling():
    values = torch.tensor([100., .2, .4], requires_grad=True)
    loss, _, _ = packed_value_loss(predicted_values=values,
        target_returns=torch.ones(3), valid_mask=torch.tensor([False, True, True]))
    assert loss.detach().item() == pytest.approx(.25)
    scaled = scale_ppo_loss_for_packed_batch(loss, valid_count=2, total_valid_tokens=4)
    scaled.backward()
    torch.testing.assert_close(values.grad, torch.tensor([0., -.2, -.15]))


def test_critic_is_independent_and_bounded_head_has_correct_range():
    actor = TinyPolicy()
    critic = build_value_model_from_policy(actor)
    for actor_param, critic_param in zip(actor.parameters(), critic.backbone.parameters()):
        assert actor_param.data_ptr() != critic_param.data_ptr()
        torch.testing.assert_close(actor_param, critic_param)
    head = BoundedValueHead(8, reward_min=-2., reward_max=3.)
    values = head(torch.randn(128, 8)*100.)
    assert (values > -2.).all() and (values < 3.).all()
    values.mean().backward()
    assert head.proj.weight.grad.abs().sum() > 0
