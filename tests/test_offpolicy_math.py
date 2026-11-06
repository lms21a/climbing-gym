import pytest
import torch
from climbing_gym.offpolicy import corrected_token_terms, masked_importance_weights

def test_mis_masks_extreme_ratios_before_exp_and_detaches_weights():
    ratios = torch.tensor([-1000., -.8, 0., 1., 1000.], requires_grad=True)
    weights = masked_importance_weights(ratios)
    torch.testing.assert_close(weights, torch.tensor([0., 0., 1., torch.exp(torch.tensor(1.)).item(), 0.]))
    assert not weights.requires_grad and bool(torch.isfinite(weights).all())
    for lo, hi in ((0., 5.), (2., 5.), (.5, float('inf')), (.5, .9)):
        with pytest.raises(ValueError):
            masked_importance_weights(ratios, lower=lo, upper=hi)


@pytest.mark.parametrize('sparse', [False, True])
def test_centered_weighted_score_has_zero_expected_gradient_for_constant_advantage(sparse):
    # Enumerate every possible sampled action instead of estimating with Monte Carlo.
    logits = torch.tensor([.8, -.3, -.6, .1], requires_grad=True)
    lp = logits.log_softmax(-1)
    p = lp.detach().exp()
    if sparse:
        q = torch.cat((torch.tensor([.01, .34]), .65 * p[2:]/p[2:].sum()))
        head = torch.tensor([0, 1])
    else:
        q = torch.tensor([.01, .3, .29, .4])
        head = torch.arange(4)
    terms = corrected_token_terms(lp, q.log(),
        train_head_logprobs=lp[head].expand(4, -1),
        behavior_head_logprobs=q[head].log().expand(4, -1))
    loss = -(q * terms.objective).sum()
    gradient = torch.autograd.grad(loss, logits, retain_graph=True)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=8e-8, rtol=0)
    uncentered = corrected_token_terms(lp, q.log())
    gradient_without_sc = torch.autograd.grad(-(q * uncentered.objective).sum(), logits)[0]
    assert gradient_without_sc.abs().max() > .01


def test_matching_policies_recover_policy_gradient_without_extra_ratio():
    logits = torch.tensor([.2, .5, -.1], requires_grad=True)
    lp = logits.log_softmax(-1)
    terms = corrected_token_terms(lp[:1], lp[:1].detach(),
        train_head_logprobs=lp[None], behavior_head_logprobs=lp.detach()[None])
    actual = torch.autograd.grad(terms.objective.sum(), logits, retain_graph=True)[0]
    expected = torch.autograd.grad(lp[0], logits)[0]
    torch.testing.assert_close(actual, expected)
    assert terms.weights.item() == 1.
