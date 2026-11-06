"""Token MIS and sparse score centering for full-support native rollouts.

Score centering follows arXiv:2609.20807, Appendix A, Equations 10--12.
The sampler head is exact; its omitted tail is modeled proportional to the
learner's tail. Importance weights and centering coefficients are detached.
This is a per-token correction, not an exact trajectory-density correction.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint



def masked_importance_weights(log_ratio, *, lower=.5, upper=5.):
    if not 0 < lower <= 1 <= upper or not math.isfinite(upper):
        raise ValueError('MIS requires 0 < lower <= 1 <= finite upper')
    values = log_ratio.detach().float()
    mask = (values >= math.log(lower)) & (values <= math.log(upper))
    # Mask before exponentiating: even rejected extreme ratios stay finite.
    return torch.where(mask, torch.where(mask, values, 0.).exp(), 0.)


@dataclass
class CorrectedTerms:
    objective: torch.Tensor
    weights: torch.Tensor
    centering: torch.Tensor
    sampler_head_mass: torch.Tensor | None
    head_coefficients: torch.Tensor | None = None


def corrected_token_terms(train_logprobs, behavior_logprobs, *, train_head_logprobs=None,
                          behavior_head_logprobs=None, lower=.5, upper=5., eps=1e-6):
    """Return w(a) log p(a) minus the expected weighted score surrogate."""
    if train_logprobs.shape != behavior_logprobs.shape or train_logprobs.ndim != 1:
        raise ValueError('Expected aligned flat sampled-token logprobs')
    current = train_logprobs.float()
    behavior = behavior_logprobs.detach().float()
    weight = masked_importance_weights(current - behavior, lower=lower, upper=upper)
    centering = torch.zeros_like(current)
    head_mass = None
    coefficients = None
    if (train_head_logprobs is None) != (behavior_head_logprobs is None):
        raise ValueError('Score centering requires both learner and sampler head logprobs')
    if train_head_logprobs is not None:
        if (train_head_logprobs.ndim != 2 or train_head_logprobs.size(0) != current.numel()
                or train_head_logprobs.shape != behavior_head_logprobs.shape
                or train_head_logprobs.size(1) == 0):
            raise ValueError('Expected aligned [tokens, k] learner/sampler heads')
        head = train_head_logprobs.float()
        with torch.no_grad():
            p = head.detach().exp()
            qlp = behavior_head_logprobs.detach().float()
            q = qlp.exp()
            head_mass = q.sum(-1)
            p_tail = (1 - p.sum(-1)).clamp_min(eps)
            q_tail = (1 - head_mass).clamp_min(eps)
            log_rho = q_tail.log() - p_tail.log()
            alpha = log_rho.exp() * masked_importance_weights(-log_rho, lower=lower, upper=upper)
            head_weights = masked_importance_weights(head.detach() - qlp, lower=lower, upper=upper)
            coefficients = q * head_weights - alpha[:, None] * p
        centering = (coefficients * head).sum(-1)
    return CorrectedTerms(weight * current - centering, weight, centering, head_mass, coefficients)


@dataclass(frozen=True)
class CorrectedTokenSelection:
    """Explicit generated-token spans in flattened, shifted packed inputs.

    Each span uses next-token prediction positions (completion starts at p-1).
    Feedback and chat delimiters outside these spans have no direct loss.
    """
    spans: tuple[slice, ...]
    logprobs: tuple[tuple[float, ...], ...]
    heads: tuple[Any, ...]

    def validate(self, token_count):
        if not self.spans or len(self.spans) != len(self.logprobs) or len(self.spans) != len(self.heads):
            raise ValueError('Misaligned explicit completion selection')
        previous = 0
        for span, values in zip(self.spans, self.logprobs):
            if (span.step not in (None, 1) or not 0 <= previous <= span.start < span.stop <= token_count
                    or len(values) != span.stop-span.start):
                raise ValueError('Overlapping, unordered or invalid completion span')
            previous = span.stop


def packed_corrected_terms(model, batch, trajectories, *, score_centering, lower, upper, chunk_size=512,
                           fused_head=True, checkpoint_head=True, metadata_batch=None, token_selection=None):
    """Share the vocabulary projection for samples and sparse heads.

    The fused first-order loss retains no dense FP32 logprobs or gather gradients.
    Disabling head checkpointing saves one projection, retaining BF16 logits
    for all completion positions in this memory pack instead.
    """
    source = batch if metadata_batch is None else metadata_batch
    if token_selection is None:
        prompt_lens = source.prompt_lens.tolist()
        completion_lens = tuple(len(seq)-p for seq, p in zip(source.sequences, prompt_lens))
        spans, offset = [], 0
        for seq, p in zip(source.sequences, prompt_lens):
            spans.append(slice(offset+p-1, offset+len(seq)-1))
            offset += len(seq)-1
        behavior_rows = source.resolved_completion_logprobs()
        heads = [trajectories[i].turns[0].behavior_topk for i in batch.trajectory_indices] if score_centering else []
    else:
        token_selection.validate(source.token_count)
        spans = token_selection.spans
        completion_lens = tuple(s.stop-s.start for s in spans)
        behavior_rows = token_selection.logprobs
        heads = token_selection.heads
    hidden = model.get_hidden_states(batch.inputs, batch.position_ids)
    flat = hidden.reshape(-1, hidden.size(-1))
    target_flat = batch.targets.reshape(-1)
    states = flat[spans[0]] if len(spans) == 1 else torch.cat([flat[s] for s in spans])
    targets = target_flat[spans[0]] if len(spans) == 1 else torch.cat([target_flat[s] for s in spans])
    behavior = torch.tensor([v for row in behavior_rows for v in row],
                            device=states.device, dtype=torch.float32)
    if score_centering:
        if any(h is None for h in heads):
            raise ValueError('Score centering requires recorded sampler head probabilities')
        if any(h.token_ids.size(0) != n or h.token_ids.size(1) == 0 for h, n in zip(heads, completion_lens)):
            raise ValueError('Invalid or misaligned sampler head')
        ids = torch.cat([h.token_ids for h in heads])
        qlp = torch.cat([h.logprobs for h in heads])
        if (not bool(torch.isfinite(qlp).all()) or bool((qlp > 0).any()) or bool((ids < 0).any())
                or bool((ids >= model.lm_head.weight.size(0)).any())
                or bool((qlp.exp().sum(-1) > 1.00001).any())
                or bool((ids.sort(-1).values.diff(dim=-1) == 0).any())):
            raise ValueError('Invalid or misaligned sampler head')
        # Native metadata is int32. Only the reference gather needs int64.
        ids = ids.to(states.device, dtype=ids.dtype if fused_head and states.is_cuda else torch.long)
        qlp = qlp.to(states.device)
    else:
        ids = torch.empty((len(states), 0), device=states.device, dtype=torch.long)
        qlp = torch.empty_like(ids, dtype=torch.float32)

    def project(h, y, old, selected_ids, selected_lp):
        if fused_head:
            from climbing_gym.models.kernels.offpolicy_loss import corrected_projection
            return corrected_projection(model.lm_head(h), y, old, selected_ids, selected_lp,
                                        lower=lower, upper=upper)
        lp = F.log_softmax(model.lm_head(h).float(), dim=-1)
        current = lp.gather(1, y[:, None])[:, 0]
        terms = corrected_token_terms(current, old,
            train_head_logprobs=lp.gather(1, selected_ids) if score_centering else None,
            behavior_head_logprobs=selected_lp if score_centering else None,
            lower=lower, upper=upper)
        head_mass = terms.sampler_head_mass if score_centering else torch.zeros_like(current)
        return terms.objective, current.detach(), terms.weights, head_mass

    chunks = [(checkpoint(project, *values, use_reentrant=False, preserve_rng_state=False)
               if checkpoint_head else project(*values))
              for values in zip(*(x.split(chunk_size) for x in (states, targets, behavior, ids, qlp)))]
    objective, current, weights, masses = (torch.cat(values) for values in zip(*chunks))
    return objective, current, behavior, weights, masses


def update_grouped_corrected_policy(run: Any, trajectories: Sequence[Any]):
    """Dr. GRPO group baseline with the same token MIS + SC correction."""
    return update_corrected_policy(run, trajectories, grouped=True)


def resolve_response_normalization(config):
    """Resolve the ungrouped MIS divisor without inspecting emitted responses."""
    trainer = config.trainer
    mode = trainer.flashreinforce_length_normalization
    if mode not in {'fixed', 'legacy_response_mean'}:
        raise ValueError('flashreinforce_length_normalization must be fixed or legacy_response_mean')
    fixed = trainer.flashreinforce_fixed_token_normalizer
    if fixed is not None and (type(fixed) is not int or fixed <= 0):
        raise ValueError('flashreinforce_fixed_token_normalizer must be a positive integer or None')
    if mode == 'legacy_response_mean':
        if fixed is not None:
            raise ValueError('legacy_response_mean cannot use a fixed token normalizer')
        return mode, None
    sampling = config.rollout.sampling or config.inference.sampling
    fixed = fixed if fixed is not None else getattr(sampling, 'max_tokens', None)
    if type(fixed) is not int or fixed <= 0:
        raise ValueError('Fixed MIS normalization requires a positive sampling max_tokens or explicit normalizer')
    return mode, fixed


def corrected_response_loss(objective, advantages, lengths, *, total_sequences,
                            normalization='fixed', token_normalizer=None):
    """Accumulate one memory pack using the original global batch denominator.

    A fixed divisor controls global gradient scale and does not constrain response
    length. Historical per-response averaging is available only by explicit name.
    """
    if (objective.ndim != 1 or advantages.shape != (len(lengths),)
            or not lengths or any(type(n) is not int or n <= 0 for n in lengths)
            or sum(lengths) != objective.numel() or total_sequences < len(lengths)):
        raise ValueError('Invalid corrected response loss layout')
    sizes = torch.tensor(lengths, device=objective.device)
    ids = torch.repeat_interleave(torch.arange(len(lengths), device=objective.device), sizes,
                                  output_size=objective.numel())
    token_advantages = advantages.detach()[ids]
    if normalization == 'fixed':
        if type(token_normalizer) is not int or token_normalizer <= 0:
            raise ValueError('Fixed MIS normalization requires a positive integer token normalizer')
        from climbing_gym.grouped_mis import fixed_length_loss
        return fixed_length_loss(objective, token_advantages,
                                 total_sequences=total_sequences, max_tokens=token_normalizer)
    if normalization != 'legacy_response_mean' or token_normalizer is not None:
        raise ValueError('Use fixed or legacy_response_mean normalization')
    return -(objective * (token_advantages / sizes[ids] / total_sequences)).sum()


def update_corrected_policy(run: Any, trajectories: Sequence[Any], *, grouped=False):
    """One batch-centered REINFORCE update with MIS, optionally SC.

    This replaces FlashREINFORCE's whole-trajectory gate; it does not multiply
    an existing PPO/IS ratio a second time. The original full-batch denominator
    remains unchanged when tokens are masked.
    """
    from .trainers.packing import pack_trajectories
    from .training_memory import prepare_policy_update
    from climbing_gym.flashreinforce import centered_advantages, sampled_action_kl, validate_trajectories
    trainer = run.config.trainer
    group_metrics = {}
    if grouped:
        from climbing_gym.grouped_mis import grouped_advantages
        advantages, group_metrics = grouped_advantages(trajectories,
            group_size=run.config.rollout.num_rollouts,
            groups_per_update=run.config.rollout.effective_batch_size)
    else:
        validate_trajectories(trajectories, run.config.rollout.batch_size)
        normalization, token_normalizer = resolve_response_normalization(run.config)
    score_centering = trainer.flashreinforce_objective == 'mis_sc'
    rewards = [float(t.evaluation.reward.reward) for t in trajectories]
    if not grouped:
        advantages = centered_advantages(rewards)
    policy, optimizer = run.policy_model, run.policy_optimizer
    prepare_policy_update(run)
    policy.eval()
    optimizer.zero_grad(set_to_none=True)
    batches = pack_trajectories(trajectories, max_tokens=trainer.policy_max_packed_tokens)
    diagnostic_rows, audit_layout = [], []
    audits, losses = [], []
    progress = getattr(run, 'training_progress', None)
    for pack_index, batch in enumerate(batches):
        if progress is not None:
            progress('pack_forward_backward', pack=pack_index+1, packs=len(batches), tokens=batch.token_count)
        device_batch = batch.to(trainer.train_device)
        objective, current, behavior, weights, mass = packed_corrected_terms(
            policy, device_batch, trajectories, score_centering=score_centering,
            lower=trainer.offpolicy_ratio_min, upper=trainer.offpolicy_ratio_max,
            chunk_size=trainer.flashreinforce_head_chunk_size,
            fused_head=trainer.flashreinforce_fused_head,
            checkpoint_head=trainer.flashreinforce_checkpoint_head,
            metadata_batch=batch)
        adv = torch.tensor([advantages[i] for i in batch.trajectory_indices], device=current.device)
        if grouped:
            sizes = torch.tensor(batch.completion_lens, device=current.device)
            ids = torch.repeat_interleave(torch.arange(len(sizes), device=current.device), sizes,
                                          output_size=current.numel())
            from climbing_gym.grouped_mis import fixed_length_loss
            budget = run.config.inference.sampling.max_tokens
            if any(n > budget for n in batch.completion_lens):
                raise ValueError('Completion exceeds the fixed Dr. GRPO token budget')
            loss = fixed_length_loss(objective, adv[ids], total_sequences=len(trajectories), max_tokens=budget)
        else:
            loss = corrected_response_loss(objective, adv, batch.completion_lens,
                total_sequences=len(trajectories), normalization=normalization,
                token_normalizer=token_normalizer)
        if not bool(torch.isfinite(loss) & torch.isfinite(current).all()):
            raise FloatingPointError('Non-finite corrected policy objective')
        loss.backward()
        losses.append(float(loss.detach()))
        diagnostic_rows.append(torch.stack((current, behavior, weights, mass)).cpu())
        audit_layout.extend(zip(batch.trajectory_indices, batch.completion_lens))
    grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer.policy_max_grad_norm,
                                                    error_if_nonfinite=True))
    if progress is not None:
        progress('optimizer', pack=len(batches), packs=len(batches))
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    current, behavior, weights, masses = torch.cat(diagnostic_rows, dim=1).unbind()
    log_ratio = current - behavior
    offset = 0
    for i, n in audit_layout:
        c, b, w = current[offset:offset+n], behavior[offset:offset+n], weights[offset:offset+n]
        audits.append(dict(trajectory_index=i, dataset_index=trajectories[i].example.dataset_index,
            reward=rewards[i], advantage=advantages[i], generated_tokens=n,
            sequence_kl=float(sampled_action_kl(b, c).mean()),
            retained_token_fraction=float((w > 0).float().mean())))
        offset += n
    ratio = log_ratio.exp()
    metrics = dict(
        **{'policy/loss': math.fsum(losses), 'policy/grad_norm': grad_norm,
           'policy/optimizer_steps': 1., 'policy/reward_mean': math.fsum(rewards)/len(rewards),
           'policy/pre_update_mean_abs_probability_shift': float((current.exp()-behavior.exp()).abs().mean()),
           'policy/pre_update_mean_abs_logprob_shift': float(log_ratio.abs().mean()),
           'policy/sample_nll': float(-current.mean()),
           'offpolicy/ratio_mean': float(ratio.mean()), 'offpolicy/ratio_p01': float(ratio.quantile(.01)),
           'offpolicy/ratio_p99': float(ratio.quantile(.99)), 'offpolicy/ratio_max': float(ratio.max()),
           'offpolicy/masked_fraction': float((weights == 0).float().mean()),
           'offpolicy/weight_max': float(weights.max()),
           'offpolicy/weight_ess_fraction': float(weights.sum().square()/(weights.square().sum().clamp_min(1e-30)*weights.numel())),
           'offpolicy/sampler_head_mass': float(masses.mean()) if score_centering else 0.,
           'offpolicy/sampler_head_mass_min': float(masses.min()) if score_centering else 0.,
           'offpolicy/score_centering': float(score_centering),
           'flashreinforce/sequence_kl_mean': math.fsum(a['sequence_kl'] for a in audits)/len(audits),
           'flashreinforce/rejected_fraction': 0., 'flashreinforce/behavior_policy_lag': 0.,
           'trainer/generated_tokens': float(current.numel()),
           'trainer/avg_completion_length': current.numel()/len(trajectories),
           'trainer/truncated_fraction': sum(t.turns[0].finish_reason == 'length' for t in trajectories)/len(trajectories),
           'trainer/memory_packs': float(len(batches))})
    if not grouped:
        metrics['flashreinforce/fixed_token_normalizer'] = token_normalizer or 0
        metrics['flashreinforce/legacy_response_mean'] = float(normalization == 'legacy_response_mean')
    if grouped:
        metrics.update(group_metrics)
        metrics['group/fixed_token_normalizer'] = run.config.inference.sampling.max_tokens
        for key in list(metrics):
            if key.startswith('flashreinforce/'):
                metrics['group/'+key.split('/', 1)[1]] = metrics.pop(key)
        for audit in audits:
            trajectory = trajectories[audit['trajectory_index']]
            audit.update(group_id=trajectory.group_id, rollout_index=trajectory.rollout_index,
                behavior_version=trajectory.metadata.get('behavior_policy_version'),
                learner_version=trajectory.metadata.get('learner_policy_version'),
                source_wave_file=trajectory.metadata.get('source_wave_file'),
                source_group=trajectory.metadata.get('source_group'))
    return metrics, sorted(audits, key=lambda a: a['trajectory_index'])
