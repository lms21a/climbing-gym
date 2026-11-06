"""MIS-PPO dual token/trajectory filtering with native packed rollout updates.

Numerical objectives ported from multiturn_mis_ppo: retained samples are
on-policy actor terms. This differs from token MIS importance weighting.
"""
from __future__ import annotations
from typing import Dict, List, Sequence, Tuple
import math
import torch

TRUNCATED_TERMINATED_REASONS = frozenset({"max_turns_reached", "budget_exhausted"})

def build_valid_completion_mask(*, targets, position_ids, seq_indices, prompt_lens, pad_id):
    """Legacy MIS-PPO's pad-aware next-token mask, kept for numerical parity."""
    return (position_ids.squeeze(0) >= prompt_lens[seq_indices] - 1) & (targets.squeeze(0) != pad_id)

def unwrap_training_model(model):
    return getattr(model, "_orig_mod", model)

def resolve_actor_critic_model(model):
    if hasattr(model, "forward_with_value"):
        return model
    return unwrap_training_model(model)

def resolve_policy_export_model(model):
    base = unwrap_training_model(model)
    return getattr(base, "base_model", base)

def trajectory_is_truncated(trajectory) -> bool:
    reason = str(getattr(trajectory, "terminal_reason", getattr(trajectory, "terminated_reason", ""))).strip().lower()
    if reason in TRUNCATED_TERMINATED_REASONS:
        return True
    # DirectHarness finishes its single request even when native decoding hits
    # the token cap. Preserve legacy multi-turn termination semantics, and use
    # the actual sampler finish reason for this otherwise-complete direct case.
    turns = getattr(trajectory, "turns", ())
    return (reason == "single_turn" and len(turns) == 1
            and str(turns[0].finish_reason).strip().lower() == "length")

def _resolve_ratio_bounds(
    ratio_min: float,
    ratio_max: float,
    *,
    label: str,
) -> tuple[float, float]:
    lower = float(ratio_min)
    upper = float(ratio_max)
    if lower <= 0.0 or upper <= 0.0:
        raise ValueError(f"{label} bounds must be > 0, got lower={lower} upper={upper}")
    if lower > upper:
        raise ValueError(f"{label} lower bound must be <= upper bound, got lower={lower} upper={upper}")
    return lower, upper

def scale_mis_ppo_loss_for_packed_batch(
    loss: torch.Tensor,
    *,
    valid_count: float | torch.Tensor,
    total_valid_tokens: float,
) -> torch.Tensor:
    total = float(total_valid_tokens)
    if total <= 0.0:
        return loss * 0.0

    batch_valid = (
        float(valid_count.detach().item())
        if isinstance(valid_count, torch.Tensor)
        else float(valid_count)
    )
    if batch_valid <= 0.0:
        return loss * 0.0
    return loss * (batch_valid / total)

def count_packed_valid_completion_tokens(
    *,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    seq_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
) -> float:
    valid_mask = build_valid_completion_mask(
        targets=targets,
        position_ids=position_ids,
        seq_indices=seq_indices,
        prompt_lens=prompt_lens,
        pad_id=pad_id,
    )
    return float(valid_mask.sum().item())

def summarize_packed_mis_trajectory_ratios(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    seq_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    sequence_trajectory_ids: torch.Tensor | None,
    pad_id: int,
    num_trajectories: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    logits = model(inputs, positions=position_ids)
    lse = torch.logsumexp(logits, dim=-1)
    tgt_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (tgt_logits - lse).squeeze(0).to(dtype=torch.float32)

    valid_mask = build_valid_completion_mask(
        targets=targets,
        position_ids=position_ids,
        seq_indices=seq_indices,
        prompt_lens=prompt_lens,
        pad_id=pad_id,
    )
    valid_float = valid_mask.to(dtype=torch.float32)
    token_log_ratio = target_log_probs - old_infer_logprobs.to(dtype=torch.float32)

    if sequence_trajectory_ids is None:
        trajectory_ids_per_sequence = torch.arange(
            int(prompt_lens.shape[0]),
            device=seq_indices.device,
            dtype=torch.long,
        )
    else:
        trajectory_ids_per_sequence = sequence_trajectory_ids.to(
            device=seq_indices.device,
            dtype=torch.long,
        )
    trajectory_token_ids = trajectory_ids_per_sequence[seq_indices]

    traj_log_ratio_sum = torch.zeros(
        num_trajectories,
        dtype=torch.float32,
        device=target_log_probs.device,
    )
    traj_valid_counts = torch.zeros(
        num_trajectories,
        dtype=torch.float32,
        device=target_log_probs.device,
    )
    traj_log_ratio_sum.scatter_add_(0, trajectory_token_ids, token_log_ratio * valid_float)
    traj_valid_counts.scatter_add_(0, trajectory_token_ids, valid_float)
    return traj_log_ratio_sum, traj_valid_counts

def summarize_packed_mis_trajectory_ratios_from_logits(
    *,
    logits: torch.Tensor,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    seq_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    sequence_trajectory_ids: torch.Tensor | None,
    pad_id: int,
    num_trajectories: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    lse = torch.logsumexp(logits, dim=-1)
    tgt_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (tgt_logits - lse).squeeze(0).to(dtype=torch.float32)

    valid_mask = build_valid_completion_mask(
        targets=targets,
        position_ids=position_ids,
        seq_indices=seq_indices,
        prompt_lens=prompt_lens,
        pad_id=pad_id,
    )
    valid_float = valid_mask.to(dtype=torch.float32)
    token_log_ratio = target_log_probs - old_infer_logprobs.to(dtype=torch.float32)

    if sequence_trajectory_ids is None:
        trajectory_ids_per_sequence = torch.arange(
            int(prompt_lens.shape[0]),
            device=seq_indices.device,
            dtype=torch.long,
        )
    else:
        trajectory_ids_per_sequence = sequence_trajectory_ids.to(
            device=seq_indices.device,
            dtype=torch.long,
        )
    trajectory_token_ids = trajectory_ids_per_sequence[seq_indices]

    traj_log_ratio_sum = torch.zeros(
        num_trajectories,
        dtype=torch.float32,
        device=target_log_probs.device,
    )
    traj_valid_counts = torch.zeros(
        num_trajectories,
        dtype=torch.float32,
        device=target_log_probs.device,
    )
    traj_log_ratio_sum.scatter_add_(0, trajectory_token_ids, token_log_ratio * valid_float)
    traj_valid_counts.scatter_add_(0, trajectory_token_ids, valid_float)
    return traj_log_ratio_sum, traj_valid_counts

def extract_packed_completion_edge_values(
    *,
    values: torch.Tensor,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    seq_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    pad_id: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    valid_mask = build_valid_completion_mask(
        targets=targets,
        position_ids=position_ids,
        seq_indices=seq_indices,
        prompt_lens=prompt_lens,
        pad_id=pad_id,
    )
    num_sequences = int(prompt_lens.shape[0])
    first_values = torch.full(
        (num_sequences,),
        float("nan"),
        dtype=torch.float32,
        device=values.device,
    )
    last_values = torch.full(
        (num_sequences,),
        float("nan"),
        dtype=torch.float32,
        device=values.device,
    )
    values_fp32 = values.to(dtype=torch.float32)

    for local_seq_idx in range(num_sequences):
        local_mask = (seq_indices == local_seq_idx) & valid_mask
        local_token_idx = torch.nonzero(local_mask, as_tuple=False).squeeze(-1)
        if local_token_idx.numel() <= 0:
            continue
        first_values[local_seq_idx] = values_fp32[local_token_idx[0]]
        last_values[local_seq_idx] = values_fp32[local_token_idx[-1]]

    return first_values, last_values

def normalize_sequence_advantages(
    sequence_advantages: List[float],
    *,
    eps: float,
) -> Tuple[List[float], float, float]:
    if not sequence_advantages:
        return [], 0.0, 0.0

    tensor = torch.tensor(sequence_advantages, dtype=torch.float32)
    mean = float(tensor.mean().item())
    std = float(tensor.std(unbiased=False).item())
    denom = max(std, float(eps))
    normalized = ((tensor - mean) / denom).tolist()
    return [float(value) for value in normalized], mean, std

def should_sync_rollout_policy(
    *,
    step: int,
    last_synced_step: int | None,
    update_interval: int,
    force_sync: bool,
) -> bool:
    interval = max(1, int(update_interval))
    if step <= 0 or last_synced_step is None:
        return True
    if force_sync:
        return True
    return (int(step) - int(last_synced_step)) >= interval

def compose_truncation_bootstrap_return(
    *,
    trajectory_reward: float,
    trajectory_raw_reward: float,
    bootstrap_value: float,
    outcome_reward_weight: float,
) -> float:
    shaping_reward = float(trajectory_reward) - (float(outcome_reward_weight) * float(trajectory_raw_reward))
    return shaping_reward + (float(outcome_reward_weight) * float(bootstrap_value))

def packed_mis_ppo_loss(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    seq_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    token_advantages: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    pad_id: int,
    token_ratio_min: float,
    token_ratio_max: float,
    trajectory_ratio_min: float,
    trajectory_ratio_max: float,
    fallback_to_valid: bool,
    *,
    sequence_trajectory_ids: torch.Tensor | None = None,
    precomputed_trajectory_keep_mask: torch.Tensor | None = None,
    precomputed_trajectory_keep_rate: float | torch.Tensor | None = None,
    precomputed_mean_trajectory_ratio: float | torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    logits = model(inputs, positions=position_ids)

    lse = torch.logsumexp(logits, dim=-1)
    tgt_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (tgt_logits - lse).squeeze(0)
    target_log_probs_fp32 = target_log_probs.to(dtype=torch.float32)

    valid_mask = build_valid_completion_mask(
        targets=targets,
        position_ids=position_ids,
        seq_indices=seq_indices,
        prompt_lens=prompt_lens,
        pad_id=pad_id,
    )
    valid_float = valid_mask.to(dtype=torch.float32)
    valid_count = valid_float.sum().clamp(min=1.0)

    token_log_ratio = target_log_probs_fp32 - old_infer_logprobs.to(dtype=torch.float32)
    token_log_ratio_min = math.log(token_ratio_min)
    token_log_ratio_max = math.log(token_ratio_max)
    token_keep_mask = valid_mask & (token_log_ratio >= token_log_ratio_min) & (token_log_ratio <= token_log_ratio_max)
    token_keep_float = token_keep_mask.to(dtype=torch.float32)

    num_sequences = int(prompt_lens.shape[0])
    if sequence_trajectory_ids is None:
        trajectory_ids_per_sequence = torch.arange(
            num_sequences,
            device=seq_indices.device,
            dtype=torch.long,
        )
    else:
        trajectory_ids_per_sequence = sequence_trajectory_ids.to(
            device=seq_indices.device,
            dtype=torch.long,
        )
    trajectory_token_ids = trajectory_ids_per_sequence[seq_indices]

    if precomputed_trajectory_keep_mask is None:
        num_trajectories = int(trajectory_ids_per_sequence.max().item()) + 1 if num_sequences > 0 else 0
        traj_log_ratio_sum = torch.zeros(
            num_trajectories,
            dtype=torch.float32,
            device=target_log_probs.device,
        )
        traj_valid_counts = torch.zeros(
            num_trajectories,
            dtype=torch.float32,
            device=target_log_probs.device,
        )
        traj_log_ratio_sum.scatter_add_(0, trajectory_token_ids, token_log_ratio * valid_float)
        traj_valid_counts.scatter_add_(0, trajectory_token_ids, valid_float)

        safe_traj_counts = traj_valid_counts.clamp(min=1.0)
        traj_mean_log_ratio = traj_log_ratio_sum / safe_traj_counts
        traj_geom_ratio = torch.exp(traj_mean_log_ratio)
        traj_has_valid = traj_valid_counts > 0.0
        trajectory_log_ratio_min = math.log(trajectory_ratio_min)
        trajectory_log_ratio_max = math.log(trajectory_ratio_max)
        trajectory_keep_mask = (
            (traj_mean_log_ratio >= trajectory_log_ratio_min)
            & (traj_mean_log_ratio <= trajectory_log_ratio_max)
            & traj_has_valid
        )
        if torch.any(traj_has_valid):
            trajectory_keep_rate = trajectory_keep_mask[traj_has_valid].to(dtype=torch.float32).mean()
            mean_trajectory_ratio = traj_geom_ratio[traj_has_valid].mean()
        else:
            trajectory_keep_rate = torch.tensor(0.0, dtype=torch.float32, device=target_log_probs.device)
            mean_trajectory_ratio = torch.tensor(0.0, dtype=torch.float32, device=target_log_probs.device)
    else:
        trajectory_keep_mask = precomputed_trajectory_keep_mask.to(
            device=target_log_probs.device,
            dtype=torch.bool,
        )
        trajectory_keep_rate = torch.tensor(
            float(
                precomputed_trajectory_keep_rate
                if precomputed_trajectory_keep_rate is not None
                else 0.0
            ),
            dtype=torch.float32,
            device=target_log_probs.device,
        )
        mean_trajectory_ratio = torch.tensor(
            float(
                precomputed_mean_trajectory_ratio
                if precomputed_mean_trajectory_ratio is not None
                else 0.0
            ),
            dtype=torch.float32,
            device=target_log_probs.device,
        )

    trajectory_keep_token_float = trajectory_keep_mask[trajectory_token_ids].to(dtype=torch.float32)

    train_mask = valid_float * token_keep_float * trajectory_keep_token_float
    fallback_applied = torch.tensor(0.0, dtype=torch.float32, device=target_log_probs.device)
    if fallback_to_valid and torch.all(train_mask == 0):
        train_mask = valid_float
        fallback_applied = torch.tensor(1.0, dtype=torch.float32, device=target_log_probs.device)

    active_tokens = train_mask.sum()
    active_count = active_tokens.clamp(min=1.0)
    loss = -(token_advantages.to(dtype=torch.float32) * target_log_probs_fp32 * train_mask).sum() / active_count

    log_probs = logits.to(dtype=torch.float32) - lse.to(dtype=torch.float32).unsqueeze(-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
    entropy = (token_entropy * valid_float).sum() / valid_count

    token_retention_rate = (token_keep_float * valid_float).sum() / valid_count
    active_token_rate = active_tokens / valid_count
    mean_token_ratio = (torch.exp(token_log_ratio) * valid_float).sum() / valid_count

    seq_active_counts = torch.zeros(num_sequences, dtype=torch.float32, device=target_log_probs.device)
    seq_active_counts.scatter_add_(0, seq_indices, train_mask)
    active_sequences = (seq_active_counts > 0.0).to(dtype=torch.float32).sum()

    return (
        loss,
        entropy,
        token_retention_rate,
        trajectory_keep_rate,
        active_token_rate,
        mean_token_ratio,
        mean_trajectory_ratio,
        active_tokens,
        active_sequences,
        valid_count,
        fallback_applied,
    )

def packed_mis_ppo_actor_critic_loss(
    model,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    position_ids: torch.Tensor,
    seq_indices: torch.Tensor,
    prompt_lens: torch.Tensor,
    sequence_returns: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    pad_id: int,
    token_ratio_min: float,
    token_ratio_max: float,
    trajectory_ratio_min: float,
    trajectory_ratio_max: float,
    fallback_to_valid: bool,
    *,
    value_loss_coef: float,
    kl_coef: float,
    entropy_coef: float,
    normalize_advantages: bool,
    adv_norm_eps: float,
    sequence_trajectory_ids: torch.Tensor | None = None,
    precomputed_trajectory_keep_mask: torch.Tensor | None = None,
    precomputed_trajectory_keep_rate: float | torch.Tensor | None = None,
    precomputed_mean_trajectory_ratio: float | torch.Tensor | None = None,
) -> Tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    actor_critic_model = resolve_actor_critic_model(model)
    logits, values = actor_critic_model.forward_with_value(inputs, position_ids)

    lse = torch.logsumexp(logits, dim=-1)
    tgt_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (tgt_logits - lse).squeeze(0)
    target_log_probs_fp32 = target_log_probs.to(dtype=torch.float32)
    values_fp32 = values.squeeze(0).to(dtype=torch.float32)

    valid_mask = build_valid_completion_mask(
        targets=targets,
        position_ids=position_ids,
        seq_indices=seq_indices,
        prompt_lens=prompt_lens,
        pad_id=pad_id,
    )
    valid_float = valid_mask.to(dtype=torch.float32)
    valid_count = valid_float.sum().clamp(min=1.0)
    sequence_returns_fp32 = sequence_returns.to(dtype=torch.float32)
    token_return_targets = sequence_returns_fp32[seq_indices]

    token_log_ratio = target_log_probs_fp32 - old_infer_logprobs.to(dtype=torch.float32)
    token_log_ratio_min = math.log(token_ratio_min)
    token_log_ratio_max = math.log(token_ratio_max)
    token_keep_mask = valid_mask & (token_log_ratio >= token_log_ratio_min) & (token_log_ratio <= token_log_ratio_max)
    token_keep_float = token_keep_mask.to(dtype=torch.float32)

    num_sequences = int(prompt_lens.shape[0])
    if sequence_trajectory_ids is None:
        trajectory_ids_per_sequence = torch.arange(
            num_sequences,
            device=seq_indices.device,
            dtype=torch.long,
        )
    else:
        trajectory_ids_per_sequence = sequence_trajectory_ids.to(
            device=seq_indices.device,
            dtype=torch.long,
        )
    trajectory_token_ids = trajectory_ids_per_sequence[seq_indices]

    if precomputed_trajectory_keep_mask is None:
        num_trajectories = int(trajectory_ids_per_sequence.max().item()) + 1 if num_sequences > 0 else 0
        traj_log_ratio_sum = torch.zeros(
            num_trajectories,
            dtype=torch.float32,
            device=target_log_probs.device,
        )
        traj_valid_counts = torch.zeros(
            num_trajectories,
            dtype=torch.float32,
            device=target_log_probs.device,
        )
        traj_log_ratio_sum.scatter_add_(0, trajectory_token_ids, token_log_ratio * valid_float)
        traj_valid_counts.scatter_add_(0, trajectory_token_ids, valid_float)

        safe_traj_counts = traj_valid_counts.clamp(min=1.0)
        traj_mean_log_ratio = traj_log_ratio_sum / safe_traj_counts
        traj_geom_ratio = torch.exp(traj_mean_log_ratio)
        traj_has_valid = traj_valid_counts > 0.0
        trajectory_log_ratio_min = math.log(trajectory_ratio_min)
        trajectory_log_ratio_max = math.log(trajectory_ratio_max)
        trajectory_keep_mask = (
            (traj_mean_log_ratio >= trajectory_log_ratio_min)
            & (traj_mean_log_ratio <= trajectory_log_ratio_max)
            & traj_has_valid
        )
        if torch.any(traj_has_valid):
            trajectory_keep_rate = trajectory_keep_mask[traj_has_valid].to(dtype=torch.float32).mean()
            mean_trajectory_ratio = traj_geom_ratio[traj_has_valid].mean()
        else:
            trajectory_keep_rate = torch.tensor(0.0, dtype=torch.float32, device=target_log_probs.device)
            mean_trajectory_ratio = torch.tensor(0.0, dtype=torch.float32, device=target_log_probs.device)
    else:
        trajectory_keep_mask = precomputed_trajectory_keep_mask.to(
            device=target_log_probs.device,
            dtype=torch.bool,
        )
        trajectory_keep_rate = torch.tensor(
            float(
                precomputed_trajectory_keep_rate
                if precomputed_trajectory_keep_rate is not None
                else 0.0
            ),
            dtype=torch.float32,
            device=target_log_probs.device,
        )
        mean_trajectory_ratio = torch.tensor(
            float(
                precomputed_mean_trajectory_ratio
                if precomputed_mean_trajectory_ratio is not None
                else 0.0
            ),
            dtype=torch.float32,
            device=target_log_probs.device,
        )

    trajectory_keep_token_float = trajectory_keep_mask[trajectory_token_ids].to(dtype=torch.float32)
    train_mask = valid_float * token_keep_float * trajectory_keep_token_float
    fallback_applied = torch.tensor(0.0, dtype=torch.float32, device=target_log_probs.device)
    if fallback_to_valid and torch.all(train_mask == 0):
        train_mask = valid_float
        fallback_applied = torch.tensor(1.0, dtype=torch.float32, device=target_log_probs.device)

    raw_token_advantages = token_return_targets - values_fp32.detach()
    if bool(normalize_advantages):
        advantage_mean = (raw_token_advantages * valid_float).sum() / valid_count
        centered_advantages = raw_token_advantages - advantage_mean
        advantage_var = ((centered_advantages * centered_advantages) * valid_float).sum() / valid_count
        advantage_scale = torch.sqrt(advantage_var.clamp(min=0.0) + float(adv_norm_eps))
        token_advantages = centered_advantages / advantage_scale
    else:
        token_advantages = raw_token_advantages

    active_tokens = train_mask.sum()
    active_count = active_tokens.clamp(min=1.0)
    actor_loss = -(token_advantages.to(dtype=torch.float32) * target_log_probs_fp32 * train_mask).sum() / active_count

    value_error = values_fp32 - token_return_targets
    value_loss = 0.5 * ((value_error * value_error) * valid_float).sum() / valid_count

    log_probs = logits.to(dtype=torch.float32) - lse.to(dtype=torch.float32).unsqueeze(-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
    entropy = (token_entropy * valid_float).sum() / valid_count
    approx_kl = ((((torch.exp(token_log_ratio) - 1.0) - token_log_ratio) * valid_float).sum() / valid_count)
    total_loss = (
        actor_loss
        + float(value_loss_coef) * value_loss
        + float(kl_coef) * approx_kl
        - float(entropy_coef) * entropy
    )

    token_retention_rate = (token_keep_float * valid_float).sum() / valid_count
    active_token_rate = active_tokens / valid_count
    mean_token_ratio = (torch.exp(token_log_ratio) * valid_float).sum() / valid_count
    mean_value = (values_fp32 * valid_float).sum() / valid_count
    mean_return = (token_return_targets * valid_float).sum() / valid_count

    seq_active_counts = torch.zeros(num_sequences, dtype=torch.float32, device=target_log_probs.device)
    seq_active_counts.scatter_add_(0, seq_indices, train_mask)
    active_sequences = (seq_active_counts > 0.0).to(dtype=torch.float32).sum()

    return (
        total_loss,
        actor_loss,
        value_loss,
        entropy,
        token_retention_rate,
        trajectory_keep_rate,
        active_token_rate,
        mean_token_ratio,
        mean_trajectory_ratio,
        active_tokens,
        active_sequences,
        valid_count,
        fallback_applied,
        approx_kl,
        mean_value,
        mean_return,
    )

def compute_grpo_advantages(
    trajectory_rewards: Sequence[float],
    *,
    num_rollouts: int | None = None,
    group_ids: Sequence[int] | None = None,
    eps: float,
) -> Tuple[List[float], Dict[str, float]]:
    if not trajectory_rewards:
        return [], {
            "grpo_group_reward_mean": 0.0,
            "grpo_group_reward_std_mean": 0.0,
            "grpo_zero_std_prompt_rate": 0.0,
        }

    if group_ids is not None:
        if len(group_ids) != len(trajectory_rewards):
            raise ValueError(
                "group_ids length must match trajectory_rewards length. "
                f"Got len(group_ids)={len(group_ids)} and len(trajectory_rewards)={len(trajectory_rewards)}."
            )
        group_to_indices: Dict[int, List[int]] = {}
        for idx, group_id in enumerate(group_ids):
            group_to_indices.setdefault(int(group_id), []).append(idx)

        advantages = [0.0] * len(trajectory_rewards)
        group_means: List[float] = []
        group_stds: List[float] = []
        zero_std_count = 0

        for indices in group_to_indices.values():
            grouped = torch.tensor(
                [trajectory_rewards[idx] for idx in indices],
                dtype=torch.float32,
            )
            group_mean = grouped.mean()
            group_std = grouped.std(unbiased=False)
            group_means.append(float(group_mean.item()))
            group_stds.append(float(group_std.item()))
            if float(group_std.item()) <= eps:
                zero_std_count += 1
                continue

            normalized = (grouped - group_mean) / group_std
            for local_idx, traj_idx in enumerate(indices):
                advantages[traj_idx] = float(normalized[local_idx].item())

        metrics = {
            "grpo_group_reward_mean": float(sum(group_means) / max(1, len(group_means))),
            "grpo_group_reward_std_mean": float(sum(group_stds) / max(1, len(group_stds))),
            "grpo_zero_std_prompt_rate": float(zero_std_count / max(1, len(group_stds))),
        }
        return advantages, metrics

    if num_rollouts is None or num_rollouts <= 0:
        raise ValueError(f"num_rollouts must be > 0, got {num_rollouts}")
    if len(trajectory_rewards) % num_rollouts != 0:
        raise ValueError(
            "trajectory_rewards length must be divisible by num_rollouts. "
            f"Got {len(trajectory_rewards)} and num_rollouts={num_rollouts}."
        )

    grouped = torch.tensor(trajectory_rewards, dtype=torch.float32).view(-1, num_rollouts)
    group_mean = grouped.mean(dim=1, keepdim=True)
    group_std = grouped.std(dim=1, keepdim=True, unbiased=False)

    valid = group_std > eps
    safe_std = torch.where(valid, group_std, torch.ones_like(group_std))
    advantages = (grouped - group_mean) / safe_std
    advantages = torch.where(valid.expand_as(advantages), advantages, torch.zeros_like(advantages))

    metrics = {
        "grpo_group_reward_mean": float(group_mean.mean().item()),
        "grpo_group_reward_std_mean": float(group_std.mean().item()),
        "grpo_zero_std_prompt_rate": float((~valid.squeeze(-1)).float().mean().item()),
    }
    return advantages.reshape(-1).tolist(), metrics


def update_policy(run, trajectories):
    """Fit a native collection using the released dual-filter packed update.

    The trajectory gate is computed across *all* turns/packs before backward.
    Per-pack active-token means are weighted by that pack's share of valid
    tokens, as in the original trainer (including its explicit fallback).
    When a value model is supplied, it receives outcome returns on every valid
    completion position; actor advantages use detached pre-action values.
    """
    from .trainers.packing import pack_trajectories
    from .training_memory import prepare_policy_update

    if not trajectories:
        raise ValueError("MIS-PPO requires a nonempty native rollout collection")
    trainer = run.config.trainer
    token_min, token_max = _resolve_ratio_bounds(
        getattr(trainer, "mis_token_ratio_min", .95),
        getattr(trainer, "mis_token_ratio_max", 1.05), label="token")
    trajectory_min, trajectory_max = _resolve_ratio_bounds(
        getattr(trainer, "mis_trajectory_ratio_min", .95),
        getattr(trainer, "mis_trajectory_ratio_max", 1.05), label="trajectory")
    fallback = getattr(trainer, "mis_fallback_to_valid", False)
    if type(fallback) is not bool:
        raise ValueError("mis_fallback_to_valid must be a boolean")
    policy = run.policy_model
    critic = getattr(run, "value_model", None)
    if critic is not None and getattr(run, "value_optimizer", None) is None:
        raise ValueError("A MIS-PPO value model requires its value_optimizer")
    prepare_policy_update(run)
    policy.eval()
    if critic is not None:
        critic.eval()

    class ActorCritic:
        def forward_with_value(self, inputs, positions):
            return policy(inputs, positions=positions), critic(inputs, positions=positions)

    actor_critic = ActorCritic()
    device = trainer.train_device
    # Keep CPU metadata packs so the prepass never retains device activations.
    packs = pack_trajectories(trajectories, max_tokens=trainer.policy_max_packed_tokens)
    if not packs:
        raise ValueError("MIS-PPO requires generated completion tokens")
    configured_pad_id = getattr(getattr(policy, "config", None), "pad_token_id", None)
    pad_id = -1 if configured_pad_id is None else int(configured_pad_id)
    returns = [float(t.evaluation.reward.reward) for t in trajectories]
    if not all(math.isfinite(r) for r in returns):
        raise ValueError("MIS-PPO requires finite outcomes")
    count = len(trajectories)
    sums = torch.zeros(count, device=device)
    counts = torch.zeros(count, device=device)
    last_values = {}
    prepared = []
    with torch.no_grad():
        for pack in packs:
            old = pack.completion_value_tensor(pack.resolved_completion_logprobs()).reshape(-1)
            mask = build_valid_completion_mask(targets=pack.targets, position_ids=pack.position_ids,
                seq_indices=pack.seq_indices, prompt_lens=pack.prompt_lens, pad_id=pad_id)
            if not torch.isfinite(old[mask]).all() or (old[mask] > 0).any():
                raise ValueError("MIS-PPO requires actual finite native behavior logprobs")
            valid_count = float(mask.sum())
            prepared.append((pack, old, valid_count))
            batch = pack.to(device)
            logits = policy(batch.inputs, positions=batch.position_ids)
            ids = torch.tensor(pack.trajectory_indices, device=device)
            part_sum, part_count = summarize_packed_mis_trajectory_ratios_from_logits(
                logits=logits, targets=batch.targets, position_ids=batch.position_ids,
                seq_indices=batch.seq_indices, prompt_lens=batch.prompt_lens,
                old_infer_logprobs=old.to(device), sequence_trajectory_ids=ids,
                pad_id=pad_id, num_trajectories=count)
            sums += part_sum
            counts += part_count
            del logits  # Do not retain the final dense vocabulary projection through backward.
            if critic is not None and getattr(trainer, "mis_bootstrap_truncated_trajectories", False):
                values = critic(batch.inputs, positions=batch.position_ids).reshape(-1)
                _, edges = extract_packed_completion_edge_values(values=values,
                    targets=batch.targets, position_ids=batch.position_ids,
                    seq_indices=batch.seq_indices, prompt_lens=batch.prompt_lens, pad_id=pad_id)
                for index, turn, edge in zip(pack.trajectory_indices, pack.turn_indices, edges.tolist()):
                    if math.isfinite(edge) and (index not in last_values or turn > last_values[index][0]):
                        last_values[index] = (turn, edge)
    total_valid = sum(n for _, _, n in prepared)
    if total_valid <= 0:
        raise ValueError("MIS-PPO collection contains no valid completion tokens")
    has_tokens = counts > 0
    mean_log_ratio = sums / counts.clamp_min(1)
    keep = has_tokens & (mean_log_ratio >= math.log(trajectory_min)) & (mean_log_ratio <= math.log(trajectory_max))
    keep_rate = float(keep[has_tokens].float().mean())
    mean_ratio = float(mean_log_ratio[has_tokens].exp().mean())
    for index, (_, edge) in last_values.items():
        if trajectory_is_truncated(trajectories[index]):
            raw = float(getattr(trajectories[index].evaluation.reward, "raw_reward", returns[index]))
            returns[index] = compose_truncation_bootstrap_return(trajectory_reward=returns[index],
                trajectory_raw_reward=raw, bootstrap_value=edge,
                outcome_reward_weight=getattr(trainer, "mis_outcome_reward_weight", 1.))
    advantages, group_metrics = compute_grpo_advantages(returns,
        group_ids=[t.group_id for t in trajectories], eps=getattr(trainer, "mis_adv_norm_eps", 1e-6))
    run.policy_optimizer.zero_grad(set_to_none=True)
    if critic is not None:
        run.value_optimizer.zero_grad(set_to_none=True)
    metrics = dict(group_metrics)
    totals = {}
    for pack, old, valid_count in prepared:
        batch = pack.to(device)
        ids = torch.tensor(pack.trajectory_indices, device=device)
        common = dict(inputs=batch.inputs, targets=batch.targets, position_ids=batch.position_ids,
            seq_indices=batch.seq_indices, prompt_lens=batch.prompt_lens, old_infer_logprobs=old.to(device),
            pad_id=pad_id, token_ratio_min=token_min, token_ratio_max=token_max,
            trajectory_ratio_min=trajectory_min, trajectory_ratio_max=trajectory_max,
            fallback_to_valid=fallback, sequence_trajectory_ids=ids,
            precomputed_trajectory_keep_mask=keep, precomputed_trajectory_keep_rate=keep_rate,
            precomputed_mean_trajectory_ratio=mean_ratio)
        if critic is None:
            adv = pack.completion_value_tensor([advantages[i] for i in pack.trajectory_indices]).reshape(-1).to(device)
            result = packed_mis_ppo_loss(policy, token_advantages=adv, **common)
            names = ("loss", "entropy", "token_retention_rate", "trajectory_keep_rate", "active_token_rate",
                "mean_token_ratio", "mean_trajectory_ratio", "active_tokens", "active_sequences", "valid_tokens", "fallback_rate")
        else:
            result = packed_mis_ppo_actor_critic_loss(actor_critic,
                sequence_returns=torch.tensor([returns[i] for i in pack.trajectory_indices], device=device),
                value_loss_coef=getattr(trainer, "mis_value_loss_coef", .5),
                kl_coef=getattr(trainer, "mis_kl_coef", 0.), entropy_coef=trainer.entropy_coef,
                normalize_advantages=getattr(trainer, "mis_normalize_advantages", True),
                adv_norm_eps=getattr(trainer, "mis_adv_norm_eps", 1e-6), **common)
            names = ("loss", "actor_loss", "value_loss", "entropy", "token_retention_rate", "trajectory_keep_rate",
                "active_token_rate", "mean_token_ratio", "mean_trajectory_ratio", "active_tokens", "active_sequences",
                "valid_tokens", "fallback_rate", "approx_kl", "mean_value", "mean_return")
        if not torch.isfinite(result[0]):
            raise FloatingPointError("Nonfinite MIS-PPO loss")
        weighted = scale_mis_ppo_loss_for_packed_batch(result[0], valid_count=valid_count, total_valid_tokens=total_valid)
        weighted.backward()
        for name, value in zip(names, result):
            weight = 1. if name in {"active_tokens", "active_sequences", "valid_tokens"} else valid_count / total_valid
            totals[name] = totals.get(name, 0.) + float(value.detach()) * weight
    norm = torch.nn.utils.clip_grad_norm_(policy.parameters(), trainer.policy_max_grad_norm, error_if_nonfinite=True)
    run.policy_optimizer.step()
    run.policy_optimizer.zero_grad(set_to_none=True)
    if critic is not None:
        value_norm = torch.nn.utils.clip_grad_norm_(critic.parameters(), trainer.value_max_grad_norm, error_if_nonfinite=True)
        run.value_optimizer.step()
        run.value_optimizer.zero_grad(set_to_none=True)
        metrics["value/grad_norm"] = float(value_norm)
    metrics.update({f"mis_ppo/{key}": value for key, value in totals.items()})
    metrics.update({"policy/optimizer_steps": 1., "policy/grad_norm": float(norm),
                    "policy/reward_mean": math.fsum(returns)/count})
    audit = [dict(dataset_index=t.example.dataset_index, group_id=t.group_id,
        rollout_index=t.rollout_index, reward=returns[i], trajectory_admitted=bool(keep[i]))
        for i, t in enumerate(trajectories)]
    return metrics, audit
