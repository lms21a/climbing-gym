from copy import deepcopy
from dataclasses import replace
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from climbing_gym.offpolicy import corrected_token_terms
from climbing_gym.ppo import _collect_sequence_values
from climbing_gym.trainers.packing import pack_trajectories
from climbing_gym.value_mis_sc import (
    MappedCriticAdamW, build_critic, critic_prompt_overrides, fit_critic,
    freeze_targets, update_actor,
)
from test_flashreinforce import config
from test_offpolicy import make_rows
from rl_fixtures import TinyPolicy, trajectory


class TinyBackbone(TinyPolicy):
    def __init__(self):
        super().__init__()
        self.feed_forward = nn.Linear(8, 8)
        self.ffn_norm = nn.LayerNorm(8)
        self.attention = nn.Linear(8, 8)

    def get_hidden_states(self, input_ids, positions):
        h = self.embed_tokens(input_ids)
        return self.feed_forward(self.ffn_norm(self.attention(h)))


class Tokenizer:
    def __init__(self):
        self.prompts = []

    def apply_chat_template(self, messages, **kwargs):
        self.prompts.append(messages[0]['content'])
        return messages[0]['content']

    def encode(self, text, **kwargs):
        return [8, 9, 10]


def test_mapped_optimizer_cpu_state_round_trip_is_exact(tmp_path):
    parameter = nn.Parameter(torch.tensor([.25, -.5], dtype=torch.bfloat16))
    optimizer = MappedCriticAdamW([parameter], backing_path=tmp_path/'initial.bin', lr=5e-6, weight_decay=0.)
    optimizer.initialize_state()
    state = optimizer.state[parameter]
    state['step'] = 7
    state['master_param'].add_(1e-6)
    state['exp_avg'].fill_(.125)
    state['exp_avg_sq'].fill_(.25)
    # A durable checkpoint owns its bytes independently of the disposable mmap.
    snapshot = deepcopy(optimizer.state_dict())
    resumed = MappedCriticAdamW([parameter], backing_path=tmp_path/'resumed.bin', lr=1., weight_decay=0.)
    resumed.restore_exact(snapshot)
    restored = resumed.state[parameter]
    assert restored['step'] == 7
    for name in ('master_param', 'exp_avg', 'exp_avg_sq'):
        assert restored[name].dtype == torch.float32
        torch.testing.assert_close(restored[name], state[name], rtol=0, atol=0)
        assert restored[name].data_ptr() != state[name].data_ptr()
    assert resumed.param_groups[0]['lr'] == 5e-6


def test_critic_device_dtype_freezing_and_actor_isolation():
    actor = TinyBackbone().to(torch.bfloat16)
    critic = build_critic(actor)
    assert all(p.dtype == torch.bfloat16 for p in critic.backbone.parameters())
    assert all(p.dtype == torch.float32 for p in critic.value_head.parameters())
    prediction = critic(torch.tensor([[1, 2, 3]]), torch.arange(3))
    assert prediction.dtype == torch.float32
    assert ((prediction > 0) & (prediction < 1)).all()
    for name, p in critic.backbone.named_parameters():
        assert p.requires_grad == ('feed_forward.' in name or 'ffn_norm.' in name)
    assert all(p.requires_grad for p in actor.parameters())
    for a, b in zip(actor.parameters(), critic.backbone.parameters(), strict=True):
        assert a.data_ptr() != b.data_ptr()
    assert critic.value_head.proj.weight.requires_grad
    assert critic.value_head.proj.weight.shape == (1, 8)


def test_privileged_prompt_is_reference_only_and_pre_action_values_align():
    tokenizer = Tokenizer()
    data = [trajectory(1, length=4), trajectory(0, length=2)]
    data[0] = replace(data[0], final_response='UNSEEN_FUTURE',
                      example=replace(data[0].example, target='REFERENCE'))
    original = [t.turns[0].prompt_token_ids for t in data]
    overrides = critic_prompt_overrides(data, tokenizer)
    assert 'REFERENCE' in tokenizer.prompts[0]
    assert 'UNSEEN_FUTURE' not in tokenizer.prompts[0]
    assert [t.turns[0].prompt_token_ids for t in data] == original

    class IdentityValue(nn.Module):
        def forward(self, ids, positions):
            return ids.float() / 16

    cfg = config(value_max_packed_tokens=32)
    run = SimpleNamespace(config=cfg, tokenizer=tokenizer, value_model=IdentityValue())
    _, advantages, targets, values, _ = freeze_targets(run, data, policy_lambda=1.)
    assert values[(0, 0)] == (10/16, 3/16, 4/16, 5/16)
    assert values[(1, 0)] == (10/16, 3/16)
    for key, vs in values.items():
        reward = data[key[0]].evaluation.reward.reward
        assert targets[key] == (reward,) * len(vs)
        assert advantages[key] == pytest.approx([reward-v for v in vs])
    assert overrides == {(0, 0): (8, 9, 10), (1, 0): (8, 9, 10)}


@pytest.mark.parametrize('pack_cap,chunk', [(6, 2), (128, 512)])
def test_actor_matches_single_ratio_centered_gradient_with_raw_token_advantages(pack_cap, chunk):
    torch.manual_seed(17)
    initial = TinyPolicy()
    data = make_rows(initial)
    actual, expected = deepcopy(initial), deepcopy(initial)
    advantages = {(i, 0): tuple(.2 * (j-2+i) for j in range(len(t.turns[0].completion_token_ids)))
                  for i, t in enumerate(data)}
    cfg = config(flashreinforce_objective='mis_sc', policy_max_packed_tokens=pack_cap,
        flashreinforce_head_chunk_size=chunk, flashreinforce_fused_head=False,
        policy_max_grad_norm=100., flashreinforce_length_normalization='fixed',
        flashreinforce_fixed_token_normalizer=4096)
    optimizer = torch.optim.SGD(actual.parameters(), lr=.1)
    update_actor(SimpleNamespace(config=cfg, policy_model=actual, policy_optimizer=optimizer), data, advantages)
    loss = 0
    for i, t in enumerate(data):
        turn = t.turns[0]
        ids = torch.tensor(turn.prompt_token_ids+turn.completion_token_ids)
        lp = expected.lm_head(expected.embed_tokens(ids[:-1])).log_softmax(-1)[1:]
        sampled = lp.gather(1, torch.tensor(turn.completion_token_ids)[:, None])[:, 0]
        terms = corrected_token_terms(sampled, torch.tensor(turn.completion_logprobs),
            train_head_logprobs=lp, behavior_head_logprobs=turn.behavior_topk.logprobs)
        loss -= (terms.objective*torch.tensor(advantages[(i, 0)])).sum()/(len(data)*4096)
    loss.backward()
    with torch.no_grad():
        for p in expected.parameters():p.add_(p.grad, alpha=-.1)
    for a, b in zip(actual.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_critic_mc_gradient_uses_only_pre_action_completion_states_and_global_denominator(device):
    torch.manual_seed(123)
    data = [trajectory(1, length=4), trajectory(0, length=2)]
    critic = build_critic(TinyBackbone().to(device))
    original = deepcopy(critic)
    expected = deepcopy(critic)
    targets = {(0, 0): (1.,)*4, (1, 0): (0.,)*2}
    packs = pack_trajectories(data, max_tokens=6)
    cfg = config()
    cfg = replace(cfg, trainer=replace(cfg.trainer, train_device=device))
    run = SimpleNamespace(config=cfg, value_model=critic,
                          value_optimizer=torch.optim.SGD(critic.parameters(), lr=.1))
    fit_critic(run, data, packs, targets, epochs=1, max_grad_norm=100.)
    loss = 0.
    for t in data:
        turn = t.turns[0]
        ids = torch.tensor([turn.prompt_token_ids+turn.completion_token_ids[:-1]], device=device)
        values = expected(ids, torch.arange(ids.numel(), device=device))[0, len(turn.prompt_token_ids)-1:]
        loss += .5*(values-t.evaluation.reward.reward).square().sum()/6
    loss.backward()
    with torch.no_grad():
        for p in expected.parameters():
            if p.grad is not None:p.add_(p.grad, alpha=-.1)
    for a, b in zip(critic.parameters(), expected.parameters(), strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    for (name, p), old in zip(critic.backbone.named_parameters(), original.backbone.parameters(), strict=True):
        if not p.requires_grad:assert torch.equal(p, old), name


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
def test_mapped_optimizer_checkpoint_continuation_is_exact(tmp_path):
    from climbing_gym.cpu_master_adamw import CPUStateMasterAdamW
    torch.manual_seed(12)
    parameter = nn.Parameter(torch.randn(13, device='cuda', dtype=torch.bfloat16))
    expected = nn.Parameter(parameter.detach().clone())
    actual_opt = MappedCriticAdamW([parameter], backing_path=tmp_path/'live.bin', lr=5e-6,
                                 weight_decay=0., chunk_elements=7)
    expected_opt = CPUStateMasterAdamW([expected], lr=5e-6, weight_decay=0., pin_state=False, chunk_elements=7)
    for step in range(4):
        parameter.grad = torch.randn_like(parameter)
        expected.grad = parameter.grad.clone()
        actual_opt.step(); expected_opt.step(); actual_opt.evict()
        assert torch.equal(parameter, expected)
        for key in ('master_param','exp_avg','exp_avg_sq'):
            assert torch.equal(actual_opt.state[parameter][key],expected_opt.state[expected][key])
        if step == 1:
            torch.save(actual_opt.state_dict(), tmp_path/'checkpoint.pt')
            saved = torch.load(tmp_path/'checkpoint.pt', map_location='cpu', mmap=True, weights_only=True)
            actual_opt = MappedCriticAdamW([parameter], backing_path=tmp_path/'restored.bin', lr=1.,
                                         weight_decay=0., chunk_elements=7)
            actual_opt.restore_exact(saved)
            assert actual_opt.state[parameter]['step'] == 2
    assert actual_opt.state[parameter]['step'] == 4
