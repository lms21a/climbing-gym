"""Full-response actor--critic updates using the existing single MIS+SC score.

The critic observes a reference answer, never the current response's future or
outcome. Values are taken at pre-action positions. Targets are raw MC outcomes;
actor advantages are frozen before either model is fitted on the collection.
"""
from __future__ import annotations

import math
import mmap
from pathlib import Path
import time

import torch

from .cpu_master_adamw import CPUStateMasterAdamW
from .offpolicy import packed_corrected_terms
from .ppo import BoundedValueHead, ClonedBoundedValueModel, _collect_sequence_values, compute_bpco_trajectory_token_targets
from .trainers.packing import pack_trajectories


class MappedCriticAdamW(CPUStateMasterAdamW):
    """Exact FP32 Adam states backed by a disposable local mmap, not swap.

    Checkpoints remain ordinary immutable torch files. This backing file is a
    working cache only, recreated and restored from a checkpoint on restart.
    Bounded GPU staging and the optimizer arithmetic are unchanged.
    """

    def __init__(self, params, *, backing_path, **kwargs):
        super().__init__(params, pin_state=False, **kwargs)
        self.backing_path = Path(backing_path)
        self.mapping = None

    def initialize_state(self):
        if self.mapping is not None:
            return
        params = [p for g in self.param_groups for p in g['params']]
        self.backing_path.parent.mkdir(parents=True, exist_ok=True)
        size = sum(p.numel() * 12 for p in params)
        with self.backing_path.open('w+b') as f:
            f.truncate(size)
            self.mapping = mmap.mmap(f.fileno(), size, access=mmap.ACCESS_WRITE)
        offset = 0
        with torch.no_grad():
            for p in params:
                state = self.state[p]
                state['step'] = 0
                for key in ('master_param', 'exp_avg', 'exp_avg_sq'):
                    tensor = torch.frombuffer(self.mapping, dtype=torch.float32,
                                              count=p.numel(), offset=offset).view(p.shape)
                    tensor.copy_(p.detach()) if key == 'master_param' else tensor.zero_()
                    state[key] = tensor
                    offset += p.numel() * 4

    def restore_exact(self, saved):
        self.initialize_state()
        if len(saved['param_groups']) != len(self.param_groups):
            raise ValueError('Critic optimizer parameter groups changed')
        for live, old in zip(self.param_groups, saved['param_groups'], strict=True):
            if len(live['params']) != len(old['params']):
                raise ValueError('Critic optimizer parameter coverage changed')
            for p, index in zip(live['params'], old['params'], strict=True):
                src = saved['state'][index]
                self.state[p]['step'] = int(src['step'])
                for key in ('master_param', 'exp_avg', 'exp_avg_sq'):
                    if src[key].shape != p.shape or src[key].dtype != torch.float32:
                        raise ValueError('Invalid exact critic state')
                    self.state[p][key].copy_(src[key])
            for key, value in old.items():
                if key != 'params':
                    live[key] = value

    def evict(self):
        """Publish dirty cache pages and free them before generation/actor work."""
        if self.mapping is not None:
            self.mapping.flush()
            self.mapping.madvise(mmap.MADV_DONTNEED)


class Float32BoundedValueHead(BoundedValueHead):
    """Keep the small value projection and its input arithmetic in FP32."""

    def forward(self, hidden_states):
        return super().forward(hidden_states.float())


def build_critic(policy):
    critic = ClonedBoundedValueModel(policy, reward_min=0., reward_max=1.)
    reference = next(policy.parameters())
    critic.to(device=reference.device, dtype=reference.dtype)
    critic.value_head = Float32BoundedValueHead(critic.config.hidden_size, reward_min=0., reward_max=1.).to(reference.device)
    # SAO-style frozen token mixers, extended to the hybrid LFM convolution
    # mixers. Freeze embeddings too; all 16 feed-forward layers remain trained.
    for name, parameter in critic.backbone.named_parameters():
        parameter.requires_grad_('feed_forward.' in name or 'ffn_norm.' in name)
    torch.nn.init.normal_(critic.value_head.proj.weight, std=1. / (critic.config.hidden_size + 1))
    return critic.eval()


def critic_prompt_overrides(data, tokenizer):
    result = {}
    for i, trajectory in enumerate(data):
        if len(trajectory.turns) != 1:
            raise ValueError('This experiment requires single-turn full responses')
        content = (trajectory.example.prompt + '\n\nTraining-only value estimation context:\n'
                   'Verified reference answer: ' + str(trajectory.example.target) + '\n'
                   'Estimate whether the policy will eventually answer correctly from each prefix.')
        text = tokenizer.apply_chat_template([dict(role='user', content=content)], tokenize=False,
                    add_generation_prompt=True, enable_thinking=False) + '<thinking>\n'
        result[(i, trajectory.turns[0].turn_index)] = tuple(tokenizer.encode(text, add_special_tokens=False))
    return result


def value_diagnostics(data, values):
    predictions, outcomes, progress = [], [], []
    for i, trajectory in enumerate(data):
        vals = values[(i, trajectory.turns[0].turn_index)]
        predictions.extend(vals)
        outcomes.extend([float(trajectory.evaluation.reward.reward)] * len(vals))
        progress.extend(min(3, 4 * j // len(vals)) for j in range(len(vals)))
    v, y = torch.tensor(predictions), torch.tensor(outcomes)
    if not len(v) or not torch.isfinite(v).all() or (v < 0).any() or (v > 1).any():
        raise ValueError('Invalid bounded value predictions')
    variance = float(y.var(unbiased=False))
    error = y - v
    result = dict(mc_mse=float(error.square().mean()), mc_ev=1.-float(error.var(unbiased=False))/variance if variance else 0.,
        outcome_variance=variance, prediction_mean=float(v.mean()), target_mean=float(y.mean()),
        calibration_bias=float((v-y).mean()), prediction_min=float(v.min()), prediction_max=float(v.max()),
        constant_baseline_mse=variance, ev_defined=float(variance > 0))
    positions = torch.tensor(progress)
    for k in range(4):
        selected = positions == k
        if selected.any():
            result[f'quarter_{k}_mse'] = float(error[selected].square().mean())
            result[f'quarter_{k}_bias'] = float((v-y)[selected].mean())
    return result


def freeze_targets(run, data, *, alpha=.4, policy_lambda=None):
    overrides = critic_prompt_overrides(data, run.tokenizer)
    packs = pack_trajectories(data, max_tokens=run.config.trainer.value_max_packed_tokens,
                              prompt_token_overrides=overrides)
    values = _collect_sequence_values(run.value_model, batches=packs, device=run.config.trainer.train_device)
    advantages, targets, stats = compute_bpco_trajectory_token_targets(data, sequence_values=values,
        gamma=1., length_adaptive_gae_alpha=alpha, policy_gae_lambda=policy_lambda)
    diagnostics = value_diagnostics(data, values)
    # A lambda=1 control diagnostic is free: no extra inference or optimizer step.
    mc = {key: tuple(target-v for target, v in zip(targets[key], vals, strict=True))
          for key, vals in values.items()}
    pairs = [(a,b) for key in advantages for a,b in zip(advantages[key],mc[key],strict=True)]
    diagnostics['gae_mc_sign_disagreement'] = sum(a*b < 0 for a,b in pairs)/len(pairs)
    diagnostics['gae_mc_mean_abs_difference'] = math.fsum(abs(a-b) for a,b in pairs)/len(pairs)
    return packs, advantages, targets, values, dict(diagnostics, **stats)


def fit_critic(run, data, packs, targets, *, epochs=2, max_grad_norm=10.):
    model, optimizer = run.value_model, run.value_optimizer
    model.eval()
    total = sum(len(t.turns[0].completion_token_ids) for t in data)
    stats = {}
    started = time.perf_counter()
    for epoch in range(epochs):
        optimizer.zero_grad(set_to_none=True)
        loss_sum = 0.
        for number, pack in enumerate(packs, 1):
            if getattr(run, 'training_progress', None):
                run.training_progress('critic_backward', epoch=epoch+1, pack=number, packs=len(packs))
            batch = pack.to(run.config.trainer.train_device)
            predictions = model(batch.inputs, positions=batch.position_ids).reshape(-1).float()
            mask = batch.valid_completion_mask().reshape(-1)
            target = batch.completion_value_tensor([targets[(i,t)] for i,t in
                zip(pack.trajectory_indices, pack.turn_indices, strict=True)]).to(predictions.device).reshape(-1)
            loss = .5 * ((predictions[mask]-target[mask]).square().sum()) / total
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite critic loss')
            loss.backward()
            loss_sum += float(loss.detach())
        head = [p for p in model.value_head.parameters() if p.grad is not None]
        backbone = [p for p in model.backbone.parameters() if p.grad is not None]
        norm = lambda ps: float(torch.linalg.vector_norm(torch.stack([p.grad.float().norm() for p in ps])))
        stats.update(head_grad_norm=norm(head), backbone_grad_norm=norm(backbone))
        grad = float(torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm, error_if_nonfinite=True))
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        stats.update(loss=loss_sum, grad_norm=grad,
                     changed_parameters=float(getattr(optimizer, 'last_quantized_change_count', -1)))
    stats.update(seconds=time.perf_counter()-started, optimizer_steps=epochs)
    return stats


def update_actor(run, data, advantages):
    """One ratio, the existing SC term, fixed denominator, all response tokens."""
    from .training_memory import prepare_policy_update

    trainer = run.config.trainer
    policy, optimizer = run.policy_model, run.policy_optimizer
    if trainer.flashreinforce_objective != 'mis_sc':
        raise ValueError('Value learner requires the established MIS+SC objective')
    prepare_policy_update(run)
    policy.eval()
    optimizer.zero_grad(set_to_none=True)
    packs = pack_trajectories(data, max_tokens=trainer.policy_max_packed_tokens)
    loss_sum, diagnostics = 0., []
    for number, pack in enumerate(packs, 1):
        if getattr(run, 'training_progress', None):
            run.training_progress('actor_backward', pack=number, packs=len(packs))
        objective, current, behavior, weights, masses = packed_corrected_terms(policy,
            pack.to(trainer.train_device), data, score_centering=True,
            lower=trainer.offpolicy_ratio_min, upper=trainer.offpolicy_ratio_max,
            chunk_size=trainer.flashreinforce_head_chunk_size, fused_head=trainer.flashreinforce_fused_head,
            checkpoint_head=trainer.flashreinforce_checkpoint_head, metadata_batch=pack)
        adv = torch.tensor([a for i,t in zip(pack.trajectory_indices,pack.turn_indices,strict=True)
                            for a in advantages[(i,t)]], device=objective.device).detach()
        if adv.numel() != objective.numel():
            raise ValueError('Token advantages are misaligned')
        loss = -(objective * adv).sum() / (len(data)*trainer.flashreinforce_fixed_token_normalizer)
        if not torch.isfinite(loss):
            raise FloatingPointError('Nonfinite value-MIS+SC actor loss')
        loss.backward()
        loss_sum += float(loss.detach())
        diagnostics.append(torch.stack((current,behavior,weights,masses)).detach().cpu())
    grad = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer.policy_max_grad_norm, error_if_nonfinite=True))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    current, behavior, weights, masses = torch.cat(diagnostics,dim=1).unbind()
    return dict(loss=loss_sum,grad_norm=grad,optimizer_steps=1.,tokens=current.numel(),
        ratio_mean=float((current-behavior).exp().mean()),masked_fraction=float((weights==0).float().mean()),
        sampler_head_mass=float(masses.mean()),
        changed_parameters=float(getattr(optimizer,'last_quantized_change_count',-1)))
