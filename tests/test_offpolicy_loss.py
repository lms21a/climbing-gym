"""First-order parity of the sparse fused loss, including masked samples."""
import pytest
import torch

from climbing_gym.models.kernels.offpolicy_loss import corrected_projection
from climbing_gym.offpolicy import corrected_token_terms


def reference(logits, y, old, ids, qlp):
    lp = logits.float().log_softmax(-1)
    current = lp.gather(1, y[:, None])[:, 0]
    terms = corrected_token_terms(current, old, train_head_logprobs=lp.gather(1, ids) if ids.size(1) else None,
                                  behavior_head_logprobs=qlp if ids.size(1) else None)
    return terms.objective, current.detach(), terms.weights, (terms.sampler_head_mass if ids.size(1)
                                                              else torch.zeros_like(current))


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('vocab,k', [(19, 0), (19, 1), (19, 19), (257, 32), (65536, 32), (65536, 128)])
def test_sparse_projection_matches_reference_values_and_gradients(device, dtype, vocab, k):
    torch.manual_seed(984)
    z = (torch.randn(7, vocab, device=device) * 3).to(dtype)
    y = torch.randint(vocab, (7,), device=device)
    q = (z.float()+torch.randn_like(z.float())*.8).log_softmax(-1)
    qlp, ids = q.topk(k, dim=-1)
    if k:
        # Cover both overlaps and actions outside the head in one batch.
        y[0] = ids[0, 0]
    old = q.gather(1, y[:, None])[:, 0]
    old[1:3] += torch.tensor([1000., -1000.], device=device)
    old.requires_grad_()
    qlp.requires_grad_()
    upstream = torch.tensor([1., -.1, -.5, 0., 1e-6, -3., .8], device=device)
    tensors, grads = [], []
    for func in (reference, corrected_projection):
        logits = z.detach().clone().requires_grad_()
        outputs = func(logits, y, old, ids, qlp)
        grads.append(torch.autograd.grad((outputs[0]*upstream).sum(), logits)[0])
        assert all(not t.requires_grad for t in outputs[1:])
        assert bool(torch.isfinite(outputs[0]).all())
        tensors.append(outputs)
    for expected, actual in zip(*tensors):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(grads[1], grads[0], rtol=.008 if dtype == torch.bfloat16 else 5e-4,
                                atol=2e-5 if dtype == torch.bfloat16 else 3e-6)
    assert not grads[1].requires_grad
    assert old.grad is None and qlp.grad is None
    if k:
        # Masking the sampled action must still leave the SC contribution.
        assert tensors[1][2][1:3].count_nonzero() == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('sparse', [False, True])
def test_fused_constant_advantage_has_zero_expected_gradient(sparse):
    logits = torch.tensor([.8, -.3, -.6, .1], device='cuda', requires_grad=True)
    p = logits.detach().softmax(-1)
    if sparse:
        q = torch.cat((torch.tensor([.01, .34], device='cuda'), .65*p[2:]/p[2:].sum()))
        head = torch.tensor([0, 1], device='cuda')
    else:
        q = torch.tensor([.01, .3, .29, .4], device='cuda')
        head = torch.arange(4, device='cuda')
    objective = corrected_projection(logits.expand(4, -1).contiguous(), torch.arange(4, device='cuda'),
                                      q.log(), head.expand(4, -1).contiguous(), q[head].log().expand(4, -1).contiguous())[0]
    assert torch.autograd.grad(objective[0], logits, retain_graph=True)[0].abs().max() > .01
    gradient = torch.autograd.grad(-(q*objective).sum(), logits)[0]
    torch.testing.assert_close(gradient, torch.zeros_like(gradient), atol=1e-7, rtol=0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_near_deterministic_forward_preserves_tail_masks_exactly():
    # At 1-sum(p_H) ~= 1e-6, a one-ulp normalization change can flip the
    # tail's hard MIS mask. These prefixes motivated preserving the forward.
    torch.manual_seed(701)
    logits = torch.full((8, 257), -18., device='cuda')
    logits[:, :3] = torch.randn(8, 3, device='cuda')
    logits = logits.to(torch.bfloat16).requires_grad_()
    qlp, ids = (logits.detach().float()+.01*torch.randn_like(logits.float())).log_softmax(-1).topk(32)
    targets = torch.tensor([0, 1, 256, 2, 0, 1, 3, 2],device='cuda')
    old = torch.full((8,), -1., device='cuda')
    expected = reference(logits, targets, old, ids, qlp)
    actual = corrected_projection(logits, targets, old, ids.int(), qlp)
    for a, b in zip(expected, actual):
        torch.testing.assert_close(a, b, atol=0, rtol=0)
