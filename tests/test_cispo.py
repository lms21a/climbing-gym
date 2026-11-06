import pytest
import torch
from climbing_gym.cispo import cispo_loss

def test_cispo_clipped_tokens_keep_signed_policy_gradient():
    old = torch.log(torch.tensor([0.1, 0.8, 0.5]))
    current = torch.log(torch.tensor([0.5, 0.1, 0.5])).requires_grad_()
    advantages = torch.tensor([1.0, -1.0, 0.0])
    loss, clip, ratio, kl, shift = cispo_loss(
        current, old, advantages, sequence_count=2, clip_low=1.0, clip_high=0.26,
    )
    loss.backward()
    # The positive token remains trainable beyond the upper clip boundary;
    # the negative token uses its actual low importance weight.
    torch.testing.assert_close(current.grad, torch.tensor([-1.26 / 2, 0.125 / 2, 0.0]))
    assert clip.item() == pytest.approx(1 / 3)
    assert all(torch.isfinite(x) for x in (loss, ratio, kl, shift))

def test_cispo_microbatch_accumulation_preserves_trajectory_sum_objective():
    old = torch.full((5,), -2.0)
    advantages = torch.tensor([0.5, 0.5, -0.5, -0.5, -0.5])
    full = old.clone().requires_grad_()
    loss = cispo_loss(full, old, advantages, sequence_count=2, clip_low=1, clip_high=.26)[0]
    loss.backward()
    split = old.clone().requires_grad_()
    for segment in (slice(0, 2), slice(2, 5)):
        chunk = cispo_loss(split[segment], old[segment], advantages[segment], sequence_count=1, clip_low=1, clip_high=.26)[0]
        (chunk / 2).backward()
    torch.testing.assert_close(split.grad, full.grad)
    # No division by each response's length.
    assert full.grad[0].abs() == full.grad[-1].abs()
