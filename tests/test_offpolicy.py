from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch

from climbing_gym.flashreinforce import update_policy, validate_config
from climbing_gym.inference.types import BehaviorTopK
from climbing_gym.offpolicy import corrected_token_terms, masked_importance_weights
from test_flashreinforce import config
from rl_fixtures import TinyPolicy, trajectory


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


def make_rows(policy):
    rows = []
    for i, (reward, length) in enumerate(((1., 3), (0., 5), (0., 1))):
        row = trajectory(reward, length=length, group=i)
        turn = row.turns[0]
        ids = torch.tensor(turn.prompt_token_ids+turn.completion_token_ids)
        lp = policy.lm_head(policy.embed_tokens(ids[:-1])).detach().log_softmax(-1)[1:]
        q = (lp + torch.linspace(-.8, .8, 16)).softmax(-1)
        sampled = torch.tensor(turn.completion_token_ids)
        head = BehaviorTopK(torch.arange(16, dtype=torch.int32).expand(length, -1).clone(), q.log())
        turn = replace(turn, completion_logprobs=tuple(q.gather(1, sampled[:, None]).log()[:, 0].tolist()),
                       behavior_topk=head)
        rows.append(replace(row, turns=(turn,)))
    return rows


@pytest.mark.parametrize('method', ['mis', 'mis_sc'])
def test_corrected_update_is_invariant_to_memory_packing_and_head_chunks(method):
    torch.manual_seed(819)
    original = TinyPolicy()
    rows = make_rows(original)
    results = []
    for pack, chunk in ((6, 2), (128, 512)):
        model = deepcopy(original)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        cfg = config(flashreinforce_objective=method, flashreinforce_head_chunk_size=chunk,
                     policy_max_packed_tokens=pack, policy_max_grad_norm=100.)
        metrics, audit = update_policy(SimpleNamespace(config=cfg, policy_model=model, policy_optimizer=optimizer), rows)
        assert metrics['policy/optimizer_steps'] == 1.
        assert metrics['offpolicy/weight_max'] <= 5.
        assert [a['dataset_index'] for a in audit] == [0, 1, 2]
        results.append(model)
    for first, second in zip(results[0].parameters(), results[1].parameters()):
        torch.testing.assert_close(first, second, atol=2e-7, rtol=2e-6)
    assert any(not torch.equal(a, b) for a, b in zip(original.parameters(), results[0].parameters()))


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('method', ['mis', 'mis_sc'])
@pytest.mark.parametrize('checkpoint_head', [True, False])
def test_cuda_fused_update_matches_reference_for_full_packed_update(method, checkpoint_head):
    torch.manual_seed(819)
    original = TinyPolicy()
    rows = make_rows(original)
    results, diagnostics = [], []
    for fused in (False, True):
        model = deepcopy(original).to('cuda')
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        cfg = config(flashreinforce_objective=method, flashreinforce_head_chunk_size=2,
            flashreinforce_fused_head=fused, flashreinforce_checkpoint_head=checkpoint_head,
            policy_max_packed_tokens=128, policy_max_grad_norm=100.)
        cfg = replace(cfg, trainer=replace(cfg.trainer, train_device='cuda'))
        metrics, audit = update_policy(SimpleNamespace(config=cfg, policy_model=model, policy_optimizer=optimizer), rows)
        results.append(model)
        diagnostics.append((metrics, audit))
    for first, second in zip(results[0].parameters(), results[1].parameters()):
        torch.testing.assert_close(first, second, atol=2e-7, rtol=2e-6)
    for key in ('policy/loss', 'offpolicy/masked_fraction', 'offpolicy/sampler_head_mass'):
        assert diagnostics[0][0][key] == diagnostics[1][0][key]
    assert diagnostics[0][1] == diagnostics[1][1]


@pytest.mark.parametrize('kind', ['duplicate', 'out_of_range', 'nonfinite', 'positive', 'mass', 'misaligned', 'empty'])
def test_corrected_update_rejects_invalid_sparse_metadata(kind):
    from climbing_gym.offpolicy import packed_corrected_terms
    from climbing_gym.trainers.packing import pack_trajectories
    model = TinyPolicy()
    rows = make_rows(model)
    head = rows[0].turns[0].behavior_topk
    ids, lp = head.token_ids.clone(), head.logprobs.clone()
    if kind == 'duplicate':
        ids[0, 1] = ids[0, 0]
    elif kind == 'out_of_range':
        ids[0, 0] = 16
    elif kind == 'nonfinite':
        lp[0, 0] = float('nan')
    elif kind == 'positive':
        lp[0, 0] = .01
    elif kind == 'mass':
        lp[0, :] = -.01
    elif kind == 'misaligned':
        ids, lp = ids[:-1], lp[:-1]
    else:
        ids, lp = ids[:, :0], lp[:, :0]
    rows[0] = replace(rows[0], turns=(replace(rows[0].turns[0], behavior_topk=BehaviorTopK(ids, lp)),))
    with pytest.raises(ValueError, match='sampler head'):
        packed_corrected_terms(model, pack_trajectories(rows, max_tokens=128)[0], rows,
                               score_centering=True, lower=.5, upper=5.)


def test_score_centering_requires_recorded_probabilities_and_valid_configuration():
    cfg = config(flashreinforce_objective='mis_sc')
    with pytest.raises(ValueError, match='behavior_topk'):
        validate_config(cfg)
    sampling = replace(cfg.inference.sampling, extra_params={'behavior_topk': 32})
    validate_config(replace(cfg, inference=replace(cfg.inference, sampling=sampling)))
    for mode in ('unknown', ''):
        with pytest.raises(ValueError, match='objective'):
            validate_config(replace(cfg, trainer=replace(cfg.trainer, flashreinforce_objective=mode)))








@pytest.mark.parametrize('field', ['flashreinforce_fused_head', 'flashreinforce_checkpoint_head'])
def test_head_backend_flags_require_actual_booleans(field):
    with pytest.raises(ValueError, match='booleans'):
        validate_config(config(**{field: 'false'}))
