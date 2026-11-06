"""FlashREINFORCE, equations (5)--(11), using stored native sampling logprobs.

The loss accepts complete variable-length trajectories, including stale behavior
probabilities. This single-GPU integration uses the direct harness: one complete
trajectory per prompt and a fresh collection before each accumulated update.
Native inference and learning alternate, so there is no intentional policy lag.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

if TYPE_CHECKING:
    from climbing_gym.trainers.packing import Batch


def centered_advantages(rewards: Sequence[float]) -> list[float]:
    """One baseline over the entire fresh batch, before any trust rejection."""
    if len(rewards) < 2 or not all(math.isfinite(float(r)) for r in rewards):
        raise ValueError("FlashREINFORCE needs at least two finite batch rewards")
    mean = math.fsum(float(r) for r in rewards) / len(rewards)
    return [float(r) - mean for r in rewards]


@torch.no_grad()
def sampled_action_kl(behavior_log_probs: torch.Tensor, log_probs: torch.Tensor) -> torch.Tensor:
    """Bernoulli KL(behavior || learner), not the reverse KL or a ratio proxy.

    Evaluate the small proxy in FP64. Smooth only probabilities rounded to one
    at FP32 precision; the importance ratio itself is never clipped or smoothed.
    expm1 keeps the complementary probability accurate near one.
    """
    lp = behavior_log_probs.detach().double().clamp(max=-torch.finfo(torch.float32).eps)
    lq = log_probs.detach().double().clamp(max=-torch.finfo(torch.float32).eps)
    p = lp.exp()
    return (p * (lp - lq) + (-lp.expm1()) *
            ((-lp.expm1()).log() - (-lq.expm1()).log())).clamp_min(0)


@dataclass(frozen=True)
class FlashLoss:
    loss: torch.Tensor
    sequence_kl: torch.Tensor
    admitted: torch.Tensor
    admitted_token_ratios: torch.Tensor
    probability_shift: torch.Tensor


def flashreinforce_loss(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    lengths: Sequence[int],
    batch_size: int,
    delta: float = 0.003,
) -> FlashLoss:
    """Eq. (11) for one memory pack containing COMPLETE trajectories.

    Advantages have already been centered over the full rollout batch. Divide
    by that original B even when some trajectories fail the gate or reside in
    other memory packs. Memory packs accumulate gradients into ONE update.
    """
    if not lengths or any(n <= 0 for n in lengths):
        raise ValueError("Every trajectory must have at least one generated token")
    if batch_size < len(lengths) or batch_size < 2 or delta < 0 or not math.isfinite(delta):
        raise ValueError("Invalid full batch size or sequence trust threshold")
    if log_probs.ndim != 1 or behavior_log_probs.shape != log_probs.shape:
        raise ValueError("Expected aligned, flat learner and behavior logprobs")
    if sum(lengths) != log_probs.numel() or advantages.shape != (len(lengths),):
        raise ValueError("Trajectory lengths/advantages do not align with token logprobs")
    for values in (log_probs, behavior_log_probs):
        if not torch.isfinite(values).all() or (values > 0).any():
            raise ValueError("Sampling and learner logprobs must be finite and nonpositive")
    if not torch.isfinite(advantages).all():
        raise ValueError("Advantages must be finite")
    current = log_probs.float()
    behavior = behavior_log_probs.detach().float()
    sizes = torch.tensor(lengths, device=current.device, dtype=torch.long)
    ids = torch.repeat_interleave(torch.arange(len(lengths), device=current.device), sizes)
    with torch.no_grad():
        token_kl = sampled_action_kl(behavior, current)
        sequence_kl = torch.zeros(len(lengths), device=current.device, dtype=torch.float64)
        sequence_kl.scatter_add_(0, ids, token_kl)
        sequence_kl /= sizes
        admitted = sequence_kl <= delta
        token_admitted = admitted[ids]
        # Gate first so extreme rejected ratios cannot create 0 * inf NaNs.
        admitted_log_ratio = torch.where(token_admitted, current.detach() - behavior, 0.0)
        ratios = admitted_log_ratio.exp()
        if not torch.isfinite(ratios).all():
            raise FloatingPointError("Non-finite admitted importance ratio; update aborted (no clipping)")
        weights = (ratios * token_admitted * advantages.detach().float()[ids]
                   / sizes[ids] / batch_size)
    return FlashLoss(
        loss=-(weights * current).sum(),
        sequence_kl=sequence_kl,
        admitted=admitted,
        admitted_token_ratios=ratios[token_admitted],
        probability_shift=(current.detach().exp() - behavior.exp()).abs(),
    )


def packed_log_probs(model: Any, batch: Batch, *, chunk_size: int = 256) -> torch.Tensor:
    """FP32 selected-token log probabilities; checkpoint vocabulary-head chunks."""
    mask = batch.valid_completion_mask().reshape(-1)
    hidden = model.get_hidden_states(batch.inputs, batch.position_ids)
    states = hidden.reshape(-1, hidden.shape[-1])[mask]
    labels = batch.targets.reshape(-1)[mask]

    def selected(h: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        return -F.cross_entropy(model.lm_head(h).float(), y, reduction="none")

    return torch.cat([checkpoint(selected, h, y, use_reentrant=False)
                      for h, y in zip(states.split(chunk_size), labels.split(chunk_size))])


def validate_config(config: Any) -> None:
    trainer, rollout = config.trainer, config.rollout
    if trainer.algorithm != "flashreinforce" or trainer.advantage_estimator != "batch_centered":
        raise ValueError("Use algorithm='flashreinforce' and advantage_estimator='batch_centered'")
    if trainer.loss_aggregation != "sample_mean" or trainer.policy_epochs != 1:
        raise ValueError("FlashREINFORCE requires sample_mean and exactly one full-batch update")
    if trainer.critic_warmup_steps or trainer.entropy_coef or trainer.format_reward_enabled:
        raise ValueError("The base method has no critic, entropy objective, or auxiliary format reward")
    if trainer.justrl2_length_penalty or trainer.outcome_reward_weight != 1:
        raise ValueError("FlashREINFORCE consumes raw terminal rewards without reward reweighting")
    if trainer.compile_losses:
        raise ValueError("This runtime uses eager loss validation; set compile_losses=false")
    if trainer.flashreinforce_delta < 0 or not math.isfinite(trainer.flashreinforce_delta):
        raise ValueError("flashreinforce_delta must be finite and nonnegative")
    if trainer.flashreinforce_objective not in {'sequence_kl', 'mis', 'mis_sc'}:
        raise ValueError('flashreinforce_objective must be sequence_kl, mis or mis_sc')
    if trainer.flashreinforce_objective in {'mis', 'mis_sc'}:
        from climbing_gym.offpolicy import resolve_response_normalization
        resolve_response_normalization(config)
    if not 0 < trainer.offpolicy_ratio_min <= 1 <= trainer.offpolicy_ratio_max < math.inf:
        raise ValueError('Require 0 < offpolicy_ratio_min <= 1 <= finite offpolicy_ratio_max')
    if type(trainer.flashreinforce_head_chunk_size) is not int or trainer.flashreinforce_head_chunk_size <= 0:
        raise ValueError('flashreinforce_head_chunk_size must be a positive integer')
    if type(trainer.flashreinforce_fused_head) is not bool or type(trainer.flashreinforce_checkpoint_head) is not bool:
        raise ValueError('flashreinforce_fused_head and flashreinforce_checkpoint_head must be booleans')
    if trainer.policy_max_grad_norm <= 0 or not math.isfinite(trainer.policy_max_grad_norm):
        raise ValueError("Use a positive finite policy_max_grad_norm")
    if config.harness.name != "direct":
        raise ValueError("This runtime requires complete direct trajectories (no per-turn gates)")
    if rollout.num_rollouts != 1 or rollout.use_dynamic_sampling:
        raise ValueError("Use one unfiltered rollout per prompt, with dynamic sampling disabled")
    if rollout.batch_size < 2 or rollout.effective_batch_size != rollout.batch_size:
        raise ValueError("Use at least two distinct prompts and a matching effective batch size")
    if rollout.max_total_requests != rollout.batch_size:
        raise ValueError("Set max_total_requests=batch_size to prevent surplus rollouts")
    if config.inference.backend not in {"lfm25_native", "lfm25", "native_lfm25"}:
        raise ValueError("FlashREINFORCE must use the native inference engine")
    sampling = rollout.sampling or config.inference.sampling
    if (sampling is None or sampling.logprobs is None or sampling.temperature != 1.0
            or sampling.top_k != -1 or sampling.top_p != 1.0 or sampling.n != 1
            or set(sampling.extra_params) - {'behavior_topk'}):
        raise ValueError("Native stored logprobs are raw: require temperature=1, full support, n=1, no processors")
    head_k = dict(sampling.extra_params).get('behavior_topk', 0)
    if type(head_k) is not int or not 0 <= head_k <= 128:
        raise ValueError('behavior_topk must be an integer between 0 and 128')
    if trainer.flashreinforce_objective == 'mis_sc' and head_k == 0:
        raise ValueError('MIS + score centering requires sampling.extra_params.behavior_topk')


def validate_trajectories(trajectories: Sequence[Any], expected_batch_size: int) -> None:
    if len(trajectories) != expected_batch_size:
        raise RuntimeError("Incomplete fresh rollout batch")
    if len({t.example.dataset_index for t in trajectories}) != expected_batch_size:
        raise RuntimeError("FlashREINFORCE requires different prompts within each batch")
    for trajectory in trajectories:
        if trajectory.rollout_index != 0 or len(trajectory.turns) != 1:
            raise RuntimeError("Expected exactly one direct rollout per prompt")
        turn = trajectory.turns[0]
        if not turn.prompt_token_ids or not turn.completion_token_ids:
            raise RuntimeError("Empty prompt or trajectory; refusing to silently drop training data")
        if turn.completion_logprobs is None or len(turn.completion_logprobs) != len(turn.completion_token_ids):
            raise RuntimeError("Missing actual behavior sampling probabilities")
        if not all(math.isfinite(p) and p <= 0 for p in turn.completion_logprobs):
            raise RuntimeError("Invalid behavior sampling log probabilities")
        if float(trajectory.metadata.get("policy_objective_weight", 1)) != 1:
            raise RuntimeError("Unexpected per-trajectory objective reweighting")


def update_policy(run: Any, trajectories: Sequence[Any]) -> tuple[dict[str, float], list[dict[str, Any]]]:
    """Accumulate all complete trajectories, then take exactly one optimizer step."""
    from .trainers.packing import pack_trajectories
    from .training_memory import prepare_policy_update
    config = run.config
    trainer = config.trainer
    if trainer.flashreinforce_objective != 'sequence_kl':
        from climbing_gym.offpolicy import update_corrected_policy
        return update_corrected_policy(run, trajectories)
    batch_size = len(trajectories)
    validate_trajectories(trajectories, config.rollout.batch_size)
    rewards = [float(t.evaluation.reward.reward) for t in trajectories]
    if config.env.name == "futoshiki" and config.env.reward == "exact_match":
        if any(r not in (0.0, 1.0) for r in rewards):
            raise RuntimeError("Binary Futoshiki reward must be exactly zero or one")
    advantages = centered_advantages(rewards)
    batches = pack_trajectories(trajectories, max_tokens=trainer.policy_max_packed_tokens)
    packed_ids = [idx for batch in batches for idx in batch.trajectory_indices]
    if sorted(packed_ids) != list(range(batch_size)):
        raise RuntimeError("Every trajectory must occur in exactly one complete memory pack")
    policy, optimizer = run.policy_model, run.policy_optimizer
    prepare_policy_update(run)
    # No dropout: these logprobs must describe the learner's actual policy.
    policy.eval()
    optimizer.zero_grad(set_to_none=True)
    loss_sum, shift_sum, token_count = 0.0, 0.0, 0
    kl_rows, admitted_rows, ratios = [], [], []
    learner_rows, behavior_rows = [], []
    audit = []
    for batch in batches:
        device_batch = batch.to(trainer.train_device)
        current = packed_log_probs(policy, device_batch, chunk_size=trainer.flashreinforce_head_chunk_size)
        behavior = torch.tensor([v for row in batch.resolved_completion_logprobs() for v in row],
                                device=current.device, dtype=torch.float32)
        local_advantages = torch.tensor([advantages[i] for i in batch.trajectory_indices], device=current.device)
        result = flashreinforce_loss(current, behavior, local_advantages,
                                     lengths=batch.completion_lens, batch_size=batch_size,
                                     delta=trainer.flashreinforce_delta)
        if not torch.isfinite(result.loss):
            raise FloatingPointError("Non-finite FlashREINFORCE loss; update aborted")
        result.loss.backward()
        loss_sum += result.loss.detach().item()
        shift_sum += result.probability_shift.sum().item()
        token_count += current.numel()
        kl_rows.extend(result.sequence_kl.cpu().tolist())
        admitted_rows.extend(result.admitted.cpu().tolist())
        ratios.append(result.admitted_token_ratios.cpu())
        learner_rows.append(current.detach().cpu())
        behavior_rows.append(behavior.cpu())
        for idx, drift, admitted in zip(batch.trajectory_indices, kl_rows[-batch.sequence_count:],
                                        admitted_rows[-batch.sequence_count:]):
            audit.append({"trajectory_index": idx, "dataset_index": trajectories[idx].example.dataset_index,
                          "reward": rewards[idx], "advantage": advantages[idx],
                          "generated_tokens": len(trajectories[idx].turns[0].completion_token_ids),
                          "sequence_kl": drift, "admitted": admitted})
    grad_norm = float(torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer.policy_max_grad_norm,
                                                    error_if_nonfinite=True))
    probability_shift = shift_sum / token_count
    if probability_shift > trainer.behavior_parity_max_mean_abs_probability_shift:
        raise RuntimeError(f"Behavior/training probability mismatch: {probability_shift:.6f}")
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    ratio = torch.cat(ratios)
    all_current, all_behavior = torch.cat(learner_rows), torch.cat(behavior_rows)
    log_ratio = all_current.double() - all_behavior.double()
    all_ratio = log_ratio.exp()
    lengths = [len(t.turns[0].completion_token_ids) for t in trajectories]
    metrics = {
        "policy/loss": loss_sum, "policy/grad_norm": grad_norm,
        "policy/optimizer_steps": 1.0, "policy/reward_mean": math.fsum(rewards) / batch_size,
        "policy/master_update_norm": float(getattr(optimizer, "last_master_update_norm", 0)),
        "policy/pre_update_mean_abs_probability_shift": probability_shift,
        "policy/pre_update_mean_abs_logprob_shift": float(log_ratio.abs().mean()),
        "policy/sample_nll": float(-all_current.mean()),
        "offpolicy/ratio_mean": float(all_ratio.mean()),
        "offpolicy/ratio_p01": float(all_ratio.quantile(.01)),
        "offpolicy/ratio_p99": float(all_ratio.quantile(.99)),
        "offpolicy/ratio_max": float(all_ratio.max()),
        "offpolicy/weight_max": float(ratio.max()) if ratio.numel() else 0.,
        "flashreinforce/advantage_mean": math.fsum(advantages) / batch_size,
        "flashreinforce/zero_advantage_fraction": sum(a == 0 for a in advantages) / batch_size,
        "flashreinforce/sequence_kl_mean": math.fsum(kl_rows) / batch_size,
        "flashreinforce/sequence_kl_max": max(kl_rows),
        "flashreinforce/rejected_fraction": 1 - sum(admitted_rows) / batch_size,
        "flashreinforce/admitted_count": float(sum(admitted_rows)),
        "flashreinforce/admitted_ratio_mean": float(ratio.mean()) if ratio.numel() else 0.0,
        "flashreinforce/admitted_ratio_max": float(ratio.max()) if ratio.numel() else 0.0,
        "flashreinforce/behavior_policy_lag": 0.0,
        "trainer/generated_tokens": float(sum(lengths)),
        "trainer/avg_completion_length": sum(lengths) / batch_size,
        "trainer/truncated_fraction": sum(t.turns[0].finish_reason == "length" for t in trajectories) / batch_size,
        "trainer/memory_packs": float(len(batches)),
    }
    return metrics, sorted(audit, key=lambda row: row["trajectory_index"])
