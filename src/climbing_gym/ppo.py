from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import dataclass, replace
import json
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from climbing_gym.config import (
    ConfigManager,
    PPOConfig,
    PPOTrainer as PPOTrainerConfig,
    Wandb as WandbConfig,
)
from climbing_gym.eval import (
    EvalConfig as PeriodicEvalConfig,
    EvalLoggingConfig,
    ResolvedEvalRun,
    _trajectory_record,
    build_eval_run,
    run_eval,
)
from climbing_gym.inference import LFM25AsyncInferenceEngine
from climbing_gym.models import build_model
from climbing_gym.models.kernels.dppo_loss import (
    TRITON_DPPO_AVAILABLE,
    fused_linear_dppo_loss,
)
from climbing_gym.models.kernels.master_weight_adamw import (
    TRITON_MASTER_ADAMW_AVAILABLE,
    compress_fp32_master_to_cpu,
    fused_master_weight_adamw_step_,
    master_weight_adamw_block_count,
    restore_fp32_master,
)
from climbing_gym.models.kernels.lfm2_shortconv import (
    TRITON_AVAILABLE as TRITON_SHORTCONV_AVAILABLE,
)
from climbing_gym.models.kernels.lfm2_fused import (
    LFM2_FUSED_KERNELS_AVAILABLE,
)
from climbing_gym.rollout import Trajectory, TrajectoryBatch
from climbing_gym.rollout import RolloutCoordinator
from climbing_gym.trainers.common import (
    collect_rollout_batch_with_trainables,
    compute_grad_norm,
    offload_to_cpu,
    reload_to_device,
)
from climbing_gym.trainers.checkpoints import CheckpointManager
from climbing_gym.trainers.logger import WandbLogger
from climbing_gym.trainers.packing import Batch, extract_completion_values, pack_trajectories


def scale_ppo_loss_for_packed_batch(
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


def scale_bpco_loss_for_microbatch(
    loss: torch.Tensor,
    *,
    loss_aggregation: str,
    microbatch_valid_tokens: float | torch.Tensor,
    microbatch_sequence_count: int,
    minibatch_valid_tokens: float,
    minibatch_sequence_count: int,
) -> torch.Tensor:
    """Scale one packed bin into its logical trajectory-minibatch objective.

    Packing is only a memory optimization for BPCO. A token-mean loss is
    weighted by its valid-token count, while the released BPCO
    sequence-mean/token-sum reduction is weighted by its sequence count.
    Summing these scaled microbatch losses therefore reproduces the objective
    that would be computed if the whole logical minibatch fit in one pack.
    """

    mode = str(loss_aggregation).strip().lower()
    if mode == "token_mean":
        numerator = (
            float(microbatch_valid_tokens.detach().item())
            if isinstance(microbatch_valid_tokens, torch.Tensor)
            else float(microbatch_valid_tokens)
        )
        denominator = float(minibatch_valid_tokens)
    elif mode == "seq_mean_token_sum":
        numerator = float(microbatch_sequence_count)
        denominator = float(minibatch_sequence_count)
    else:
        raise ValueError(
            "loss_aggregation must be 'token_mean' or 'seq_mean_token_sum'"
        )
    if numerator <= 0.0 or denominator <= 0.0:
        return loss * 0.0
    return loss * (numerator / denominator)


def resolve_hidden_size(config: Any) -> int:
    for key in ("hidden_size", "d_model", "n_embd"):
        value = getattr(config, key, None)
        if value is not None:
            return int(value)
    raise AttributeError(f"Could not resolve hidden size from config type {type(config).__name__}")


class ValueHead(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.fc1 = nn.Linear(hidden_size, hidden_size, bias=False)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.fc1.weight, mean=0.0, std=0.02)
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(hidden_states))).squeeze(-1)


class BoundedValueHead(nn.Module):
    """Linear scalar head mapped differentiably into a known reward range."""

    def __init__(self, hidden_size: int, *, reward_min: float, reward_max: float) -> None:
        super().__init__()
        if not float(reward_min) < float(reward_max):
            raise ValueError("reward_min must be smaller than reward_max")
        self.proj = nn.Linear(hidden_size, 1, bias=False)
        nn.init.normal_(self.proj.weight, mean=0.0, std=0.02)
        self.reward_min = float(reward_min)
        self.reward_max = float(reward_max)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        raw_values = self.proj(hidden_states).squeeze(-1).to(dtype=torch.float32)
        unit_values = 0.5 + torch.atan(raw_values) / torch.pi
        return self.reward_min + (self.reward_max - self.reward_min) * unit_values


class ClonedValueModel(nn.Module):
    """Standalone critic whose backbone is initialized from the policy model."""

    def __init__(self, policy_model: nn.Module) -> None:
        super().__init__()
        if not hasattr(policy_model, "get_hidden_states"):
            raise TypeError("Policy model must expose get_hidden_states(...) to build a PPO value model.")
        self.backbone = deepcopy(policy_model)
        self.config = getattr(self.backbone, "config", None)
        self.value_head = ValueHead(resolve_hidden_size(self.config))

    def forward(self, input_ids: torch.Tensor, positions: torch.Tensor) -> torch.Tensor:
        hidden_states = self.backbone.get_hidden_states(input_ids, positions)
        return self.value_head(hidden_states)


def build_value_model_from_policy(policy_model: nn.Module) -> ClonedValueModel:
    return ClonedValueModel(policy_model)


class ClonedBoundedValueModel(ClonedValueModel):
    """Training-only BPCO critic initialized from the policy backbone."""

    def __init__(
        self,
        policy_model: nn.Module,
        *,
        reward_min: float,
        reward_max: float,
    ) -> None:
        nn.Module.__init__(self)
        if not hasattr(policy_model, "get_hidden_states"):
            raise TypeError(
                "Policy model must expose get_hidden_states(...) to build a BPCO critic."
            )
        self.backbone = deepcopy(policy_model)
        self.config = getattr(self.backbone, "config", None)
        self.value_head = BoundedValueHead(
            resolve_hidden_size(self.config),
            reward_min=reward_min,
            reward_max=reward_max,
        )


def build_bounded_value_model_from_policy(
    policy_model: nn.Module,
    *,
    reward_min: float,
    reward_max: float,
) -> ClonedBoundedValueModel:
    return ClonedBoundedValueModel(
        policy_model,
        reward_min=reward_min,
        reward_max=reward_max,
    )


def maybe_compile(
    target: Any,
    *,
    dynamic: bool,
    fullgraph: bool,
    backend: str,
    mode: str,
) -> Any:
    return torch.compile(
        target,
        dynamic=dynamic,
        fullgraph=fullgraph,
        backend=backend,
        mode=mode,
    )


def packed_ppo_policy_loss(
    *,
    target_log_probs: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    token_advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = valid_mask.to(dtype=target_log_probs.dtype)
    log_ratio = target_log_probs - old_infer_logprobs
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, min=1.0 - clip_epsilon, max=1.0 + clip_epsilon)
    unclipped = ratio * token_advantages
    clipped = clipped_ratio * token_advantages
    token_objective = torch.minimum(unclipped, clipped) * valid

    valid_count = valid.sum().clamp_min(1.0)
    loss = -token_objective.sum() / valid_count
    clip_fraction = (((ratio - 1.0).abs() > clip_epsilon).to(valid.dtype) * valid).sum() / valid_count
    mean_ratio = (ratio * valid).sum() / valid_count
    approx_kl = ((((ratio - 1.0) - log_ratio) * valid).sum()) / valid_count
    return loss, clip_fraction, mean_ratio, approx_kl


def packed_ppo_policy_loss_from_model(
    policy_model: Any,
    *,
    batch: Batch,
    token_advantages: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    clip_epsilon: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = policy_model(batch.inputs, positions=batch.position_ids)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=batch.targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (target_logits - lse).squeeze(0).to(dtype=torch.float32)

    valid_mask = batch.valid_completion_mask()
    valid_float = valid_mask.to(dtype=torch.float32)
    valid_count = valid_float.sum()
    safe_valid_count = valid_count.clamp_min(1.0)

    policy_loss, clip_fraction, mean_ratio, approx_kl = packed_ppo_policy_loss(
        target_log_probs=target_log_probs,
        old_infer_logprobs=old_infer_logprobs.to(dtype=torch.float32),
        token_advantages=token_advantages.to(dtype=torch.float32),
        valid_mask=valid_mask,
        clip_epsilon=clip_epsilon,
    )

    log_probs = logits.to(dtype=torch.float32) - lse.to(dtype=torch.float32).unsqueeze(-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
    entropy = (token_entropy * valid_float).sum() / safe_valid_count
    total_loss = policy_loss - float(entropy_coef) * entropy
    return total_loss, policy_loss, entropy, clip_fraction, mean_ratio, approx_kl, valid_count


def _aggregate_masked_token_objective(
    token_values: torch.Tensor,
    *,
    valid: torch.Tensor,
    aggregation: str,
    seq_indices: torch.Tensor | None,
    sequence_count: int | None,
) -> torch.Tensor:
    mode = str(aggregation).strip().lower()
    masked = token_values * valid
    if mode == "token_mean":
        return masked.sum() / valid.sum().clamp_min(1.0)
    if mode != "seq_mean_token_sum":
        raise ValueError(
            "loss_aggregation must be 'token_mean' or 'seq_mean_token_sum'"
        )
    if seq_indices is None or sequence_count is None:
        raise ValueError(
            "seq_mean_token_sum requires seq_indices and sequence_count"
        )
    resolved_sequence_count = int(sequence_count)
    if resolved_sequence_count <= 0:
        return masked.sum() * 0.0
    indices = seq_indices.to(device=masked.device, dtype=torch.long)
    if indices.numel() != masked.numel():
        raise ValueError("seq_indices must align with the packed token objective")
    sequence_sums = torch.zeros(
        resolved_sequence_count,
        device=masked.device,
        dtype=masked.dtype,
    )
    sequence_sums.scatter_add_(0, indices, masked)
    sequence_valid_counts = torch.zeros_like(sequence_sums)
    sequence_valid_counts.scatter_add_(0, indices, valid)
    valid_sequences = sequence_valid_counts > 0
    valid_sequence_float = valid_sequences.to(dtype=masked.dtype)
    return (
        (sequence_sums * valid_sequence_float).sum()
        / valid_sequence_float.sum().clamp_min(1.0)
    )


def packed_dppo_policy_loss(
    *,
    target_log_probs: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    token_advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    clip_epsilon: float,
    log_ratio_clip: float = 20.0,
    loss_aggregation: str = "token_mean",
    seq_indices: torch.Tensor | None = None,
    sequence_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Binary-TV DPPO objective from BPCO Equation 2.

    ``behavior_logprobs`` must be the sampled-token log probabilities recorded
    by the rollout engine. DPPO clips absolute sampled-token probability shifts,
    rather than applying a fixed bound to every probability ratio.
    """

    epsilon = float(clip_epsilon)
    if epsilon <= 0.0:
        raise ValueError("DPPO clip_epsilon must be > 0")
    ratio_cap = float(log_ratio_clip)
    if ratio_cap <= 0.0:
        raise ValueError("dppo_log_ratio_clip must be > 0")

    valid = valid_mask.to(dtype=torch.float32)
    target_lp = target_log_probs.to(dtype=torch.float32)
    behavior_lp = behavior_logprobs.to(dtype=torch.float32).detach()
    advantages = token_advantages.to(dtype=torch.float32).detach()
    log_ratio = torch.clamp(target_lp - behavior_lp, min=-ratio_cap, max=ratio_cap)
    ratio = torch.exp(log_ratio)

    behavior_prob = torch.exp(behavior_lp).clamp_min(torch.finfo(torch.float32).tiny)
    current_prob = torch.exp(target_lp)
    dynamic_radius = epsilon / behavior_prob
    lower_ratio = 1.0 - dynamic_radius
    upper_ratio = 1.0 + dynamic_radius
    clipped_ratio = torch.minimum(torch.maximum(ratio, lower_ratio), upper_ratio)

    unclipped_objective = ratio * advantages
    clipped_objective = clipped_ratio * advantages
    token_objective = torch.minimum(unclipped_objective, clipped_objective) * valid

    valid_count = valid.sum().clamp_min(1.0)
    loss = -_aggregate_masked_token_objective(
        token_objective,
        valid=valid,
        aggregation=loss_aggregation,
        seq_indices=seq_indices,
        sequence_count=sequence_count,
    )
    clipped_tokens = (clipped_objective < unclipped_objective).to(dtype=torch.float32)
    clip_fraction = (clipped_tokens * valid).sum() / valid_count
    mean_ratio = (ratio * valid).sum() / valid_count
    approx_kl = ((((ratio - 1.0) - log_ratio) * valid).sum()) / valid_count
    mean_abs_probability_shift = (
        ((current_prob - behavior_prob).abs() * valid).sum() / valid_count
    )
    return loss, clip_fraction, mean_ratio, approx_kl, mean_abs_probability_shift


def packed_dppo_policy_loss_from_model(
    policy_model: Any,
    *,
    batch: Batch,
    token_advantages: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    clip_epsilon: float,
    entropy_coef: float,
    log_ratio_clip: float = 20.0,
    loss_aggregation: str = "token_mean",
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    valid_mask = batch.valid_completion_mask()
    valid_float = valid_mask.to(dtype=torch.float32)
    valid_count = valid_float.sum()
    safe_valid_count = valid_count.clamp_min(1.0)

    # BPCO never needs prompt-token logits. Projecting the full packed sequence
    # through a 65K-token LM head is especially wasteful for short completions,
    # and retaining that full logits tensor is the actor path's memory peak.
    # Keep torch.compile on its established full-logits path; eager CUDA uses a
    # fused, chunked Triton head below.
    split_head_available = (
        callable(getattr(policy_model, "get_hidden_states", None))
        and hasattr(policy_model, "lm_head")
        and not torch.compiler.is_compiling()
    )
    if split_head_available:
        hidden_states = policy_model.get_hidden_states(
            batch.inputs,
            batch.position_ids,
        )
        hidden_states_2d = hidden_states.reshape(-1, hidden_states.shape[-1])
        completion_hidden_states = hidden_states_2d[valid_mask]
        completion_targets = batch.targets.reshape(-1)[valid_mask]
        completion_advantages = token_advantages.reshape(-1)[valid_mask]
        completion_behavior_logprobs = behavior_logprobs.reshape(-1)[valid_mask]
        completion_seq_indices = batch.seq_indices.reshape(-1)[valid_mask]
        completion_token_count = int(completion_hidden_states.shape[0])

        if completion_token_count == 0:
            connected_zero = (
                hidden_states.sum() * 0.0
                + policy_model.lm_head.weight.sum() * 0.0
            ).to(dtype=torch.float32)
            metric_zero = connected_zero.detach()
            return (
                connected_zero,
                connected_zero,
                metric_zero,
                metric_zero,
                metric_zero,
                metric_zero,
                metric_zero,
                metric_zero,
            )

        aggregation = str(loss_aggregation).strip().lower()
        if aggregation not in {"token_mean", "seq_mean_token_sum"}:
            raise ValueError(
                "loss_aggregation must be 'token_mean' or 'seq_mean_token_sum'"
            )
        loss_denominator = (
            float(batch.sequence_count)
            if aggregation == "seq_mean_token_sum"
            else float(completion_token_count)
        )
        use_fused_head = (
            TRITON_DPPO_AVAILABLE
            and completion_hidden_states.is_cuda
            and float(entropy_coef) == 0.0
            # Below these sizes, one compact cuBLAS projection is faster than
            # the memory-bounded custom autograd path. Large packs cross over
            # once optimizer state leaves little allocator headroom.
            and int(batch.inputs.numel()) >= 3072
            and completion_token_count >= 2048
        )
        if use_fused_head:
            (
                policy_loss,
                clip_fraction,
                mean_ratio,
                approx_kl,
                probability_shift,
            ) = fused_linear_dppo_loss(
                completion_hidden_states,
                policy_model.lm_head.weight,
                completion_targets,
                completion_behavior_logprobs,
                completion_advantages,
                clip_epsilon=clip_epsilon,
                log_ratio_clip=log_ratio_clip,
                loss_denominator=loss_denominator,
            )
            entropy = policy_loss.detach().new_zeros(())
            return (
                policy_loss,
                policy_loss,
                entropy,
                clip_fraction,
                mean_ratio,
                approx_kl,
                probability_shift,
                valid_count,
            )

        logits = policy_model.lm_head(completion_hidden_states)
        lse = torch.logsumexp(logits, dim=-1)
        target_logits = logits.gather(
            dim=-1,
            index=completion_targets.unsqueeze(-1),
        ).squeeze(-1)
        target_log_probs = (target_logits - lse).to(dtype=torch.float32)
        compact_valid_mask = torch.ones_like(target_log_probs, dtype=torch.bool)
        policy_loss, clip_fraction, mean_ratio, approx_kl, probability_shift = (
            packed_dppo_policy_loss(
                target_log_probs=target_log_probs,
                behavior_logprobs=completion_behavior_logprobs,
                token_advantages=completion_advantages,
                valid_mask=compact_valid_mask,
                clip_epsilon=clip_epsilon,
                log_ratio_clip=log_ratio_clip,
                loss_aggregation=loss_aggregation,
                seq_indices=completion_seq_indices,
                sequence_count=batch.sequence_count,
            )
        )
        if float(entropy_coef) == 0.0:
            entropy = logits.new_zeros((), dtype=torch.float32)
        else:
            log_probs = logits.to(dtype=torch.float32) - lse.to(
                dtype=torch.float32
            ).unsqueeze(-1)
            probs = torch.exp(log_probs)
            entropy = -(probs * log_probs).sum(dim=-1).mean()
        total_loss = policy_loss - float(entropy_coef) * entropy
        return (
            total_loss,
            policy_loss,
            entropy,
            clip_fraction,
            mean_ratio,
            approx_kl,
            probability_shift,
            valid_count,
        )

    logits = policy_model(batch.inputs, positions=batch.position_ids)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=batch.targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (target_logits - lse).squeeze(0).to(dtype=torch.float32)
    policy_loss, clip_fraction, mean_ratio, approx_kl, probability_shift = (
        packed_dppo_policy_loss(
            target_log_probs=target_log_probs,
            behavior_logprobs=behavior_logprobs,
            token_advantages=token_advantages,
            valid_mask=valid_mask,
            clip_epsilon=clip_epsilon,
            log_ratio_clip=log_ratio_clip,
            loss_aggregation=loss_aggregation,
            seq_indices=batch.seq_indices,
            sequence_count=batch.sequence_count,
        )
    )

    if float(entropy_coef) == 0.0:
        entropy = logits.new_zeros((), dtype=torch.float32)
    else:
        log_probs = logits.to(dtype=torch.float32) - lse.to(dtype=torch.float32).unsqueeze(-1)
        probs = torch.exp(log_probs)
        token_entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
        entropy = (token_entropy * valid_float).sum() / safe_valid_count
    total_loss = policy_loss - float(entropy_coef) * entropy
    return (
        total_loss,
        policy_loss,
        entropy,
        clip_fraction,
        mean_ratio,
        approx_kl,
        probability_shift,
        valid_count,
    )


def packed_value_loss(
    *,
    predicted_values: torch.Tensor,
    target_returns: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = valid_mask.to(dtype=torch.float32)
    valid_count = valid.sum().clamp_min(1.0)
    predicted_values_fp32 = predicted_values.to(dtype=torch.float32)
    target_returns_fp32 = target_returns.to(dtype=torch.float32)
    error = predicted_values_fp32 - target_returns_fp32
    loss = 0.5 * ((error * error) * valid).sum() / valid_count
    mean_value = (predicted_values_fp32 * valid).sum() / valid_count
    mean_return = (target_returns_fp32 * valid).sum() / valid_count
    return loss, mean_value, mean_return


def packed_value_loss_from_model(
    value_model: Any,
    *,
    batch: Batch,
    target_returns: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_values = value_model(batch.inputs, positions=batch.position_ids).squeeze(0)
    valid_mask = batch.valid_completion_mask()
    loss, mean_value, mean_return = packed_value_loss(
        predicted_values=predicted_values,
        target_returns=target_returns,
        valid_mask=valid_mask,
    )
    valid_count = valid_mask.to(dtype=torch.float32).sum()
    return loss, mean_value, mean_return, valid_count


def packed_bpco_value_loss(
    *,
    predicted_values: torch.Tensor,
    target_returns: torch.Tensor,
    valid_mask: torch.Tensor,
    loss_aggregation: str = "token_mean",
    seq_indices: torch.Tensor | None = None,
    sequence_count: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Paper Equation 6: un-clipped outcome Monte Carlo mean-squared error."""

    valid = valid_mask.to(dtype=torch.float32)
    valid_count = valid.sum().clamp_min(1.0)
    predicted_values_fp32 = predicted_values.to(dtype=torch.float32)
    target_returns_fp32 = target_returns.to(dtype=torch.float32)
    error = predicted_values_fp32 - target_returns_fp32
    loss = _aggregate_masked_token_objective(
        error * error,
        valid=valid,
        aggregation=loss_aggregation,
        seq_indices=seq_indices,
        sequence_count=sequence_count,
    )
    mean_value = (predicted_values_fp32 * valid).sum() / valid_count
    mean_return = (target_returns_fp32 * valid).sum() / valid_count
    return loss, mean_value, mean_return


def packed_bpco_value_loss_from_model(
    value_model: Any,
    *,
    batch: Batch,
    target_returns: torch.Tensor,
    loss_aggregation: str = "token_mean",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    predicted_values = value_model(batch.inputs, positions=batch.position_ids).squeeze(0)
    valid_mask = batch.valid_completion_mask()
    loss, mean_value, mean_return = packed_bpco_value_loss(
        predicted_values=predicted_values,
        target_returns=target_returns,
        valid_mask=valid_mask,
        loss_aggregation=loss_aggregation,
        seq_indices=batch.seq_indices,
        sequence_count=batch.sequence_count,
    )
    valid_count = valid_mask.to(dtype=torch.float32).sum()
    return loss, mean_value, mean_return, valid_count


def compute_gae(
    rewards: Sequence[float],
    values: Sequence[float],
    *,
    gamma: float,
    gae_lambda: float,
) -> tuple[list[float], list[float]]:
    if len(rewards) != len(values):
        raise ValueError("rewards and values must have the same length")
    advantages = [0.0] * len(rewards)
    returns = [0.0] * len(rewards)
    gae = 0.0
    next_value = 0.0
    for idx in range(len(rewards) - 1, -1, -1):
        is_terminal = idx == len(rewards) - 1
        next_non_terminal = 0.0 if is_terminal else 1.0
        delta = float(rewards[idx]) + (float(gamma) * next_value * next_non_terminal) - float(values[idx])
        gae = delta + (float(gamma) * float(gae_lambda) * next_non_terminal * gae)
        advantages[idx] = float(gae)
        returns[idx] = float(gae + float(values[idx]))
        next_value = float(values[idx])
    return advantages, returns


def compute_trajectory_token_gae(
    trajectories: Sequence[Trajectory],
    *,
    sequence_values: dict[tuple[int, int], tuple[float, ...]],
    gamma: float,
    gae_lambda: float,
) -> tuple[
    dict[tuple[int, int], tuple[float, ...]],
    dict[tuple[int, int], tuple[float, ...]],
    dict[str, float],
]:
    sequence_advantages: dict[tuple[int, int], tuple[float, ...]] = {}
    sequence_returns: dict[tuple[int, int], tuple[float, ...]] = {}
    all_advantages: list[float] = []
    all_returns: list[float] = []

    for trajectory_index, trajectory in enumerate(trajectories):
        flattened_values: list[float] = []
        ranges: list[tuple[tuple[int, int], int, int]] = []
        cursor = 0
        for turn in trajectory.turns:
            key = (int(trajectory_index), int(turn.turn_index))
            token_values = tuple(float(value) for value in sequence_values.get(key, ()))
            if not token_values:
                continue
            if len(token_values) != len(turn.completion_token_ids):
                raise ValueError(
                    "Value predictions must match the completion token count for each trajectory turn. "
                    f"trajectory_index={trajectory_index} turn_index={turn.turn_index} "
                    f"expected={len(turn.completion_token_ids)} got={len(token_values)}"
                )
            start = cursor
            cursor += len(token_values)
            flattened_values.extend(token_values)
            ranges.append((key, start, cursor))

        if not flattened_values:
            continue

        rewards = [0.0] * len(flattened_values)
        rewards[-1] = float(trajectory.evaluation.reward.reward)
        flat_advantages, flat_returns = compute_gae(
            rewards,
            flattened_values,
            gamma=gamma,
            gae_lambda=gae_lambda,
        )
        all_advantages.extend(flat_advantages)
        all_returns.extend(flat_returns)
        for key, start, end in ranges:
            sequence_advantages[key] = tuple(float(value) for value in flat_advantages[start:end])
            sequence_returns[key] = tuple(float(value) for value in flat_returns[start:end])

    advantage_mean = float(sum(all_advantages) / len(all_advantages)) if all_advantages else 0.0
    return_mean = float(sum(all_returns) / len(all_returns)) if all_returns else 0.0
    advantage_variance = (
        float(sum((value - advantage_mean) ** 2 for value in all_advantages) / len(all_advantages))
        if all_advantages
        else 0.0
    )
    return sequence_advantages, sequence_returns, {
        "advantage_mean": advantage_mean,
        "advantage_std": float(advantage_variance**0.5),
        "return_mean": return_mean,
        "gae_token_count": float(len(all_advantages)),
    }


def compute_bpco_trajectory_token_targets(
    trajectories: Sequence[Trajectory],
    *,
    sequence_values: Mapping[tuple[int, int], tuple[float, ...]],
    gamma: float,
    length_adaptive_gae_alpha: float,
    policy_gae_lambda: float | None = None,
) -> tuple[
    dict[tuple[int, int], tuple[float, ...]],
    dict[tuple[int, int], tuple[float, ...]],
    dict[str, float],
]:
    """Build raw GAE policy advantages and Monte Carlo critic targets.

    BPCO decouples the estimators: the policy uses response-length-adaptive
    GAE by default; policy_gae_lambda overrides it for controlled comparisons.
    Every critic state regresses directly to the observed terminal outcome.
    No centering or variance normalization is applied.
    """

    alpha = float(length_adaptive_gae_alpha)
    if alpha <= 0.0:
        raise ValueError("length_adaptive_gae_alpha must be > 0")
    if policy_gae_lambda is not None and not 0.0 <= float(policy_gae_lambda) <= 1.0:
        raise ValueError("policy_gae_lambda must be in [0, 1]")

    sequence_advantages: dict[tuple[int, int], tuple[float, ...]] = {}
    sequence_mc_returns: dict[tuple[int, int], tuple[float, ...]] = {}
    all_advantages: list[float] = []
    all_returns: list[float] = []
    lambdas: list[float] = []
    clamped_short_responses = 0

    for trajectory_index, trajectory in enumerate(trajectories):
        flattened_values: list[float] = []
        ranges: list[tuple[tuple[int, int], int, int]] = []
        cursor = 0
        for turn in trajectory.turns:
            key = (int(trajectory_index), int(turn.turn_index))
            token_values = tuple(float(value) for value in sequence_values.get(key, ()))
            if not token_values:
                continue
            if len(token_values) != len(turn.completion_token_ids):
                raise ValueError(
                    "BPCO critic values must match each completion token count. "
                    f"trajectory_index={trajectory_index} turn_index={turn.turn_index} "
                    f"expected={len(turn.completion_token_ids)} got={len(token_values)}"
                )
            start = cursor
            cursor += len(token_values)
            flattened_values.extend(token_values)
            ranges.append((key, start, cursor))

        response_length = len(flattened_values)
        if response_length <= 0:
            continue

        raw_lambda = (
            1.0 - 1.0 / (alpha * float(response_length))
            if policy_gae_lambda is None else float(policy_gae_lambda)
        )
        gae_lambda = min(1.0, max(0.0, raw_lambda))
        if gae_lambda != raw_lambda:
            clamped_short_responses += 1
        lambdas.append(gae_lambda)

        outcome = float(trajectory.evaluation.reward.reward)
        token_rewards = [0.0] * response_length
        token_rewards[-1] = outcome
        flat_advantages, _bootstrapped_returns = compute_gae(
            token_rewards,
            flattened_values,
            gamma=float(gamma),
            gae_lambda=gae_lambda,
        )
        flat_mc_returns = [outcome] * response_length
        all_advantages.extend(flat_advantages)
        all_returns.extend(flat_mc_returns)

        for key, start, end in ranges:
            sequence_advantages[key] = tuple(flat_advantages[start:end])
            sequence_mc_returns[key] = tuple(flat_mc_returns[start:end])

    advantage_mean = float(sum(all_advantages) / len(all_advantages)) if all_advantages else 0.0
    advantage_variance = (
        float(sum((value - advantage_mean) ** 2 for value in all_advantages) / len(all_advantages))
        if all_advantages
        else 0.0
    )
    return sequence_advantages, sequence_mc_returns, {
        "advantage_mean": advantage_mean,
        "advantage_std": float(advantage_variance**0.5),
        "return_mean": float(sum(all_returns) / len(all_returns)) if all_returns else 0.0,
        "gae_token_count": float(len(all_advantages)),
        "la_gae_lambda_mean": float(sum(lambdas) / len(lambdas)) if lambdas else 0.0,
        "la_gae_lambda_min": float(min(lambdas)) if lambdas else 0.0,
        "la_gae_lambda_max": float(max(lambdas)) if lambdas else 0.0,
        "la_gae_short_response_clamp_count": float(clamped_short_responses),
    }


def format_privileged_answer(target: Any) -> str:
    if isinstance(target, Sequence) and not isinstance(target, (str, bytes)):
        return " ".join(str(item) for item in target)
    return str(target)


def build_privileged_critic_prompt(
    prompt: str,
    target: Any,
    *,
    template: str,
) -> str:
    answer = format_privileged_answer(target)
    try:
        context = str(template).format(answer=answer)
    except (IndexError, KeyError) as exc:
        raise ValueError(
            "privileged_answer_template may only reference the {answer} field"
        ) from exc
    if not context.strip():
        raise ValueError("privileged_answer_template must render non-empty context")
    return f"{str(prompt).rstrip()}\n\n{context.strip()}\n"


def build_privileged_prompt_token_overrides(
    trajectories: Sequence[Trajectory],
    *,
    tokenize_prompt: Callable[[str], Sequence[int]],
    template: str,
) -> dict[tuple[int, int], tuple[int, ...]]:
    """Tokenize answer-privileged critic prefixes without mutating rollouts."""

    overrides: dict[tuple[int, int], tuple[int, ...]] = {}
    for trajectory_index, trajectory in enumerate(trajectories):
        critic_prompt = build_privileged_critic_prompt(
            trajectory.example.prompt,
            trajectory.example.target,
            template=template,
        )
        critic_token_ids = tuple(int(token_id) for token_id in tokenize_prompt(critic_prompt))
        if not critic_token_ids:
            raise ValueError("privileged critic prompt tokenization produced no tokens")
        for turn in trajectory.turns:
            key = (int(trajectory_index), int(turn.turn_index))
            overrides[key] = critic_token_ids
    return overrides


def _append_training_trajectories(
    path: str,
    *,
    step: int,
    trajectories: Sequence[Trajectory],
) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for trajectory in trajectories:
            record = _trajectory_record(trajectory)
            record["step"] = int(step)
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _append_training_metrics(path: str, payload: Mapping[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(dict(payload), sort_keys=True) + "\n")


def normalize_sequence_payloads(
    sequence_payloads: Mapping[tuple[int, int], tuple[float, ...]],
    *,
    epsilon: float = 1e-8,
) -> dict[tuple[int, int], tuple[float, ...]]:
    flattened: list[float] = [
        float(value)
        for values in sequence_payloads.values()
        for value in values
    ]
    if not flattened:
        return {
            key: tuple(float(value) for value in values)
            for key, values in sequence_payloads.items()
        }
    mean = float(sum(flattened) / len(flattened))
    variance = float(sum((value - mean) ** 2 for value in flattened) / len(flattened))
    denom = max(float(variance**0.5), float(epsilon))
    return {
        key: tuple(float((float(value) - mean) / denom) for value in values)
        for key, values in sequence_payloads.items()
    }


def compute_value_explained_variance(
    *,
    sequence_values: Mapping[tuple[int, int], tuple[float, ...]],
    sequence_returns: Mapping[tuple[int, int], tuple[float, ...]],
) -> float:
    predicted: list[float] = []
    targets: list[float] = []
    for key, target_values in sequence_returns.items():
        predicted_values = sequence_values.get(key, ())
        if len(predicted_values) != len(target_values):
            continue
        predicted.extend(float(value) for value in predicted_values)
        targets.extend(float(value) for value in target_values)

    if len(targets) < 2:
        return 0.0

    target_mean = float(sum(targets) / len(targets))
    target_variance = float(sum((value - target_mean) ** 2 for value in targets) / len(targets))
    if target_variance <= 0.0:
        return 0.0

    residuals = [
        target - pred
        for pred, target in zip(predicted, targets, strict=False)
    ]
    residual_mean = float(sum(residuals) / len(residuals))
    residual_variance = float(
        sum((residual - residual_mean) ** 2 for residual in residuals)
        / len(residuals)
    )
    return float(1.0 - (residual_variance / target_variance))


@dataclass(frozen=True, slots=True)
class ResolvedPPORun:
    policy_model: nn.Module
    policy_optimizer: torch.optim.Optimizer
    value_model: nn.Module
    value_optimizer: torch.optim.Optimizer
    rollout_coordinator: RolloutCoordinator
    config: PPOConfig
    engine: Any
    checkpoint_manager: CheckpointManager
    wandb_logger: WandbLogger
    eval_run: ResolvedEvalRun | None = None
    start_step: int = 0


_NATIVE_LFM25_BACKENDS = {"lfm25", "lfm25_native", "native_lfm25"}


def _trainer_algorithm(config: PPOConfig) -> str:
    return str(config.trainer.algorithm or "ppo").strip().lower()


def _validate_bpco_config(config: PPOConfig) -> None:
    if _trainer_algorithm(config) != "bpco":
        return
    backend = str(config.inference.backend or "").strip().lower()
    if backend not in _NATIVE_LFM25_BACKENDS:
        raise ValueError("BPCO training requires inference.backend='lfm25_native'.")
    if str(config.harness.name or "").strip().lower() != "direct":
        raise ValueError(
            "BPCO privileged-critic isolation currently requires harness.name='direct'."
        )
    if int(config.rollout.num_rollouts) != 1:
        raise ValueError("BPCO is a single-rollout method; rollout.num_rollouts must be 1.")
    if bool(config.rollout.use_dynamic_sampling):
        raise ValueError("BPCO requires rollout.use_dynamic_sampling=false.")
    if (
        str(config.resume.checkpoint_dir).strip()
        and not bool(config.resume.allow_weights_only)
    ):
        raise ValueError(
            "BPCO resume is disabled because the current checkpoint format does not "
            "save FP32 master weights, optimizer moments, or data/RNG state. Set "
            "resume.allow_weights_only=true to explicitly accept a weights-only "
            "continuation."
        )
    if float(config.trainer.gamma) != 1.0:
        raise ValueError("BPCO outcome Monte Carlo targets currently require trainer.gamma=1.0.")
    if float(config.trainer.entropy_coef) != 0.0:
        raise ValueError("The paper's BPCO recipe requires trainer.entropy_coef=0.")
    if not float(config.trainer.reward_min) < float(config.trainer.reward_max):
        raise ValueError("trainer.reward_min must be smaller than trainer.reward_max.")
    if float(config.trainer.length_adaptive_gae_alpha) <= 0.0:
        raise ValueError("trainer.length_adaptive_gae_alpha must be > 0.")
    if int(config.trainer.trajectory_minibatch_size) <= 0:
        raise ValueError("trainer.trajectory_minibatch_size must be > 0.")
    if str(config.trainer.loss_aggregation).strip().lower() not in {
        "token_mean",
        "seq_mean_token_sum",
    }:
        raise ValueError(
            "trainer.loss_aggregation must be 'token_mean' or "
            "'seq_mean_token_sum'."
        )
    if float(
        config.trainer.behavior_parity_max_mean_abs_probability_shift
    ) <= 0.0:
        raise ValueError(
            "trainer.behavior_parity_max_mean_abs_probability_shift must be > 0."
        )
    if config.trainer.value_minibatch_prompt_groups is not None:
        raise ValueError(
            "BPCO privileged critic packing currently requires "
            "trainer.value_minibatch_prompt_groups to be omitted."
        )
    sampling = config.rollout.sampling or config.inference.sampling
    if float(sampling.temperature) != 1.0:
        raise ValueError(
            "BPCO requires sampling.temperature=1.0 so stored rollout logprobs "
            "are the actual behavior-policy probabilities."
        )
    if int(sampling.top_k) != -1 or float(sampling.top_p) != 1.0:
        raise ValueError(
            "BPCO requires unfiltered sampling (top_k=-1 and top_p=1.0) so "
            "stored rollout logprobs are the actual behavior-policy probabilities."
        )
    if int(sampling.n) != 1:
        raise ValueError("BPCO requires sampling.n=1.")
    repetition_penalty = float(
        dict(sampling.extra_params).get("repetition_penalty", 1.0)
    )
    if repetition_penalty != 1.0:
        raise ValueError(
            "BPCO requires repetition_penalty=1.0 so rollout and actor "
            "probabilities use the same distribution."
        )
    if config.trainer.train_dtype == torch.float16:
        raise ValueError(
            "BPCO master-weight training currently supports BF16 or FP32, not FP16."
        )
    if config.trainer.train_dtype == torch.bfloat16:
        for label, optimizer in (
            ("policy_optimizer", config.policy_optimizer),
            ("value_optimizer", config.value_optimizer),
        ):
            if str(optimizer.type).strip().lower() != "master_adamw":
                raise ValueError(
                    f"BPCO with {config.trainer.train_dtype} requires "
                    f"{label}.type='master_adamw' to preserve paper-scale updates."
                )


def _build_rollout_engine(config: PPOConfig) -> Any:
    backend = str(config.inference.backend or "lfm25_native").strip().lower()
    engine_config = config.inference.build_engine_config()
    if backend in _NATIVE_LFM25_BACKENDS:
        return LFM25AsyncInferenceEngine(
            engine_config,
            batch_wait_seconds=float(config.inference.native_batch_wait_seconds),
            max_batch_size=config.inference.native_max_batch_size,
            compile_decode=bool(config.inference.compile_decode),
            cuda_graphs=bool(config.inference.native_cuda_graphs),
            max_cuda_graphs=int(config.inference.native_cuda_max_graphs),
            max_prefill_tokens=int(config.inference.native_prefill_tokens),
            bf16_weight_packing=bool(config.inference.native_bf16_weight_packing),
            fp8_mlp=bool(config.inference.native_fp8_mlp),
            fp8_kv=bool(config.inference.native_fp8_kv),
            adaptive_batching=bool(config.inference.native_adaptive_batching),
            fp8_lm_head=bool(config.inference.native_fp8_lm_head),
        )
    raise ValueError(
        f"Unsupported inference.backend={backend!r}; expected 'lfm25_native'."
    )


class MasterWeightAdamW(torch.optim.Optimizer):
    """AdamW for low-precision parameters with persistent FP32 master weights.

    The first and second moments intentionally retain the parameter dtype to
    fit a full actor or critic beside the native engine on one 24 GB GPU. The
    FP32 master copy preserves paper-scale updates that would otherwise round
    to zero when applied directly to BF16 parameters.
    """

    def __init__(
        self,
        params: Any,
        *,
        lr: float,
        weight_decay: float,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        if lr < 0.0:
            raise ValueError("learning rate must be >= 0")
        if weight_decay < 0.0:
            raise ValueError("weight_decay must be >= 0")
        if not 0.0 <= betas[0] < 1.0 or not 0.0 <= betas[1] < 1.0:
            raise ValueError("AdamW betas must be in [0, 1)")
        if eps < 0.0:
            raise ValueError("AdamW eps must be >= 0")
        super().__init__(
            params,
            defaults={
                "lr": float(lr),
                "weight_decay": float(weight_decay),
                "betas": tuple(float(value) for value in betas),
                "eps": float(eps),
            },
        )
        self.last_master_update_norm = 0.0
        self.last_quantized_change_count = 0

    _LOW_BITS_KEY = "master_param_low_bits"
    _TIE_INDICES_KEY = "master_param_tie_round_down_indices"

    @torch.no_grad()
    def offload_state_to_cpu(self) -> None:
        """Offload state, encoding BF16 masters losslessly in half the space.

        A BF16 parameter already contains the rounded high 16 bits of its FP32
        master. Persisting only the master's low word (plus a tiny sparse table
        for exact halfway ties) avoids keeping the redundant high word in host
        RAM. This is what keeps a full actor and critic out of swap on 24 GiB
        workstations.
        """

        for parameter, state in self.state.items():
            master_parameter = state.get("master_param")
            if (
                isinstance(master_parameter, torch.Tensor)
                and master_parameter.is_cuda
                and parameter.dtype == torch.bfloat16
                and TRITON_MASTER_ADAMW_AVAILABLE
            ):
                low_bits, tie_indices = compress_fp32_master_to_cpu(
                    master_parameter
                )
                state[self._LOW_BITS_KEY] = low_bits
                state[self._TIE_INDICES_KEY] = tie_indices
                del state["master_param"]
            elif isinstance(master_parameter, torch.Tensor):
                state["master_param"] = master_parameter.detach().cpu()

            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    state[key] = value.detach().cpu()

    @torch.no_grad()
    def reload_state_to_device(self, device: str | torch.device) -> None:
        """Restore an offloaded optimizer state on the active device."""

        resolved_device = torch.device(device)
        for parameter, state in self.state.items():
            low_bits = state.get(self._LOW_BITS_KEY)
            tie_indices = state.get(self._TIE_INDICES_KEY)
            if isinstance(low_bits, torch.Tensor):
                if not isinstance(tie_indices, torch.Tensor):
                    raise RuntimeError("Compressed master state is missing tie indices.")
                state["master_param"] = restore_fp32_master(
                    parameter,
                    low_bits,
                    tie_indices,
                    device=resolved_device,
                )
                del state[self._LOW_BITS_KEY]
                del state[self._TIE_INDICES_KEY]
            else:
                master_parameter = state.get("master_param")
                if isinstance(master_parameter, torch.Tensor):
                    state["master_param"] = master_parameter.to(resolved_device)

            for key in ("exp_avg", "exp_avg_sq"):
                value = state.get(key)
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(resolved_device)

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        if closure is None:
            loss = None
        else:
            with torch.enable_grad():
                loss = closure()
        active: list[
            tuple[
                torch.Tensor,
                torch.Tensor,
                dict[str, Any],
                Mapping[str, Any],
            ]
        ] = []
        for group in self.param_groups:
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("MasterWeightAdamW does not support sparse gradients")
                if parameter.dtype not in {torch.bfloat16, torch.float32}:
                    raise TypeError(
                        "MasterWeightAdamW requires BF16 or FP32 model parameters; "
                        f"got {parameter.dtype}."
                    )
                state = self.state[parameter]
                if self._LOW_BITS_KEY in state:
                    raise RuntimeError(
                        "MasterWeightAdamW state is compressed; reload it before step()."
                    )
                if not state:
                    state["step"] = 0
                    state["master_param"] = parameter.detach().to(dtype=torch.float32).clone()
                    state["exp_avg"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                    state["exp_avg_sq"] = torch.zeros_like(
                        parameter,
                        memory_format=torch.preserve_format,
                    )
                state["step"] = int(state["step"]) + 1
                active.append((parameter, grad, state, group))

        use_fused_cuda = bool(active) and TRITON_MASTER_ADAMW_AVAILABLE and all(
            parameter.is_cuda
            and grad.is_cuda
            and parameter.is_contiguous()
            and grad.is_contiguous()
            and state["master_param"].is_contiguous()
            and state["exp_avg"].is_contiguous()
            and state["exp_avg_sq"].is_contiguous()
            for parameter, grad, state, _group in active
        )
        if use_fused_cuda:
            max_blocks = max(
                master_weight_adamw_block_count(parameter.numel())
                for parameter, _grad, _state, _group in active
            )
            device = active[0][0].device
            update_sq_blocks = torch.empty(
                max_blocks,
                device=device,
                dtype=torch.float32,
            )
            changed_blocks = torch.empty(
                max_blocks,
                device=device,
                dtype=torch.int32,
            )
            update_sq_sum = torch.zeros((), device=device, dtype=torch.float32)
            changed_count = torch.zeros((), device=device, dtype=torch.int64)
            for parameter, grad, state, group in active:
                beta1, beta2 = (float(value) for value in group["betas"])
                parameter_update_sq, parameter_changed = (
                    fused_master_weight_adamw_step_(
                        parameter,
                        grad,
                        state["master_param"],
                        state["exp_avg"],
                        state["exp_avg_sq"],
                        step=int(state["step"]),
                        learning_rate=float(group["lr"]),
                        weight_decay=float(group["weight_decay"]),
                        beta1=beta1,
                        beta2=beta2,
                        epsilon=float(group["eps"]),
                        update_sq_blocks=update_sq_blocks,
                        changed_blocks=changed_blocks,
                    )
                )
                update_sq_sum.add_(parameter_update_sq)
                changed_count.add_(parameter_changed)
            self.last_master_update_norm = float(update_sq_sum.sqrt().item())
            self.last_quantized_change_count = int(changed_count.item())
            return loss

        update_sq_sum_float = 0.0
        changed_count_int = 0
        for parameter, grad, state, group in active:
            lr = float(group["lr"])
            weight_decay = float(group["weight_decay"])
            beta1, beta2 = (float(value) for value in group["betas"])
            eps = float(group["eps"])
            step = int(state["step"])
            master_param = state["master_param"]
            exp_avg = state["exp_avg"]
            exp_avg_sq = state["exp_avg_sq"]
            exp_avg.mul_(beta1).add_(grad, alpha=1.0 - beta1)
            exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1.0 - beta2)

            bias_correction1 = 1.0 - beta1**step
            bias_correction2 = 1.0 - beta2**step
            # FP32 states already have the target dtype: to() can alias them.
            # Update scratch must not overwrite the persistent Adam moments.
            delta = exp_avg.to(dtype=torch.float32, copy=True)
            denom = exp_avg_sq.to(dtype=torch.float32, copy=True).sqrt_()
            denom.div_(bias_correction2**0.5).add_(eps)
            delta.div_(denom).mul_(-lr / bias_correction1)
            if weight_decay:
                delta.add_(master_param, alpha=-lr * weight_decay)

            update_sq_sum_float += float(torch.sum(delta * delta).item())
            master_param.add_(delta)
            quantized = master_param.to(dtype=parameter.dtype)
            changed_count_int += int(
                torch.count_nonzero(quantized != parameter).item()
            )
            parameter.copy_(quantized)

        self.last_master_update_norm = float(update_sq_sum_float**0.5)
        self.last_quantized_change_count = int(changed_count_int)
        return loss


def _build_policy_optimizer(policy_model, optimizer_config, *, trainer=None):
    from climbing_gym.grpo import _build_policy_optimizer as build
    return build(policy_model, optimizer_config, trainer=trainer)


def _build_value_optimizer_from_config(value_model, optimizer_config, *, trainer=None):
    from climbing_gym.grpo import _build_policy_optimizer as build
    return build(value_model, optimizer_config, trainer=trainer)


def _load_resume_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Resume checkpoint metadata must parse to a mapping.")
    return dict(payload)


def _load_value_model_checkpoint(value_model: nn.Module, checkpoint_path: Path) -> None:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if isinstance(payload, Mapping) and isinstance(payload.get("state_dict"), Mapping):
        state_dict = dict(payload["state_dict"])
    elif isinstance(payload, Mapping):
        state_dict = dict(payload)
    else:
        raise TypeError("Value checkpoint must contain a state dict mapping.")
    missing, unexpected = value_model.load_state_dict(state_dict, strict=False)
    if missing:
        raise ValueError(f"Missing value checkpoint keys: {sorted(missing)!r}")
    if unexpected:
        raise ValueError(f"Unexpected value checkpoint keys: {sorted(unexpected)!r}")


def _apply_bpco_lfm2_kernel_defaults(
    model_config: dict[str, Any],
    *,
    algorithm: str,
    model_name: str,
    train_device: str | torch.device,
) -> None:
    """Select the measured CUDA fast paths unless the job overrides them."""

    if (
        algorithm != "bpco"
        or model_name != "lfm2"
        or torch.device(train_device).type != "cuda"
    ):
        return
    if TRITON_SHORTCONV_AVAILABLE and "shortconv_backend" not in model_config:
        model_config["shortconv_backend"] = "triton_k3"
    if (
        LFM2_FUSED_KERNELS_AVAILABLE
        and "fused_backbone_kernels" not in model_config
    ):
        model_config["fused_backbone_kernels"] = True


def load_run_from_toml(config_path: str | Path) -> ResolvedPPORun:
    manager = ConfigManager.from_toml(config_path)
    path = manager.config_path
    config = manager.config
    trainer = config.trainer
    wandb_config = config.wandb
    algorithm = _trainer_algorithm(config)
    if algorithm != "ppo":
        raise ValueError(
            f"Unsupported trainer.algorithm={algorithm!r}; expected 'ppo'."
        )
    _validate_bpco_config(config)
    from climbing_gym.train import validate_config
    from climbing_gym.preflight import validate_preflight
    validate_config(config)
    preflight_report = validate_preflight(config)
    model_name = str(config.model.name or "").strip().lower()
    model_path = str(config.model.path)
    if not model_name:
        raise ValueError("`model.name` is required.")
    if not model_path:
        raise ValueError("`model.path` is required.")

    if config.job.print_config:
        print(json.dumps(manager.to_dict(), indent=2, sort_keys=True))

    resume_checkpoint_dir = str(config.resume.checkpoint_dir)
    resume_policy_path = ""
    resume_value_path = ""
    start_step = 0
    if resume_checkpoint_dir:
        checkpoint_dir = Path(resume_checkpoint_dir)
        resume_policy_path = str((checkpoint_dir / "policy").resolve())
        resume_value_path = str((checkpoint_dir / "value_model.pt").resolve())
        if not Path(resume_policy_path).is_dir():
            raise FileNotFoundError(f"Resume policy checkpoint directory not found: {resume_policy_path}")
        if not Path(resume_value_path).is_file():
            raise FileNotFoundError(f"Resume value checkpoint not found: {resume_value_path}")
        if bool(config.resume.continue_step_numbers):
            metadata = _load_resume_metadata(checkpoint_dir)
            start_step = int(metadata.get("step", -1)) + 1

    engine: Any | None = None
    wandb_logger: WandbLogger | None = None
    try:
        model_config = dict(config.model.config)
        _apply_bpco_lfm2_kernel_defaults(
            model_config,
            algorithm=algorithm,
            model_name=model_name,
            train_device=trainer.train_device,
        )
        policy_model = build_model(
            model_name,
            model_dir=(resume_policy_path or model_path),
            **model_config,
        )
        policy_optimizer = _build_policy_optimizer(policy_model, config.policy_optimizer.to_dict(), trainer=trainer)
        value_model = build_value_model_from_policy(policy_model)
        if resume_value_path:
            _load_value_model_checkpoint(value_model, Path(resume_value_path))
        value_optimizer = _build_value_optimizer_from_config(value_model, config.value_optimizer.to_dict(), trainer=trainer)

        if resume_checkpoint_dir:
            from climbing_gym.trainers.checkpoints import restore_training_state
            restore_training_state(resume_checkpoint_dir, policy_optimizer=policy_optimizer,
                                   value_optimizer=value_optimizer, allow_weights_only=config.resume.allow_weights_only)

        default_sampling = config.inference.sampling
        rollout_sampling = config.rollout.sampling
        effective_sampling = rollout_sampling or default_sampling
        if effective_sampling is None or effective_sampling.logprobs is None:
            raise ValueError("PPO requires rollout sampling with `logprobs` enabled in `[rollout.sampling]` or `[inference.sampling]`.")

        engine = _build_rollout_engine(config)

        env = config.env.build()
        dataset_rows = config.data.load_rows()
        env.configure_dataset(
            dataset_rows,
            seed=int(config.data.seed),
            shuffle=bool(config.data.shuffle),
            epoch=int(config.data.epoch),
        )

        harness = config.harness.build()
        rollout = config.rollout.build_rollout_config()
        rollout_coordinator = RolloutCoordinator(
            engine=engine,
            env=env,
            harness=harness,
            config=config.rollout.build_coordinator_config(),
        )
        if config.rollout.use_dynamic_sampling and config.harness.name == "direct" and config.rollout.num_rollouts >= 2:
            from climbing_gym.native_dynamic import NativeDynamicRunner
            audit_parent = (Path(config.train_logging.metrics_path).parent if config.train_logging.metrics_path
                            else Path(config.checkpoint.root_dir or Path.cwd() / "runs"))
            rollout_coordinator.rollout_runner = NativeDynamicRunner(
                config.rollout.build_runner_config(), audit_dir=audit_parent / "dynamic_sampling")
        if resume_checkpoint_dir and config.resume.continue_step_numbers:
            from climbing_gym.dataset_checkpoint import restore_dataset_checkpoint
            restore_dataset_checkpoint(rollout_coordinator, Path(resume_checkpoint_dir))
        eval_run: ResolvedEvalRun | None = None
        if config.eval.enabled:
            eval_run = build_eval_run(
                engine=engine,
                harness=harness,
                data_config=config.eval.data,
                env_config=config.eval.env,
                rollout_config=config.eval.rollout,
                eval_config=PeriodicEvalConfig(
                    max_examples=int(config.eval.max_examples),
                    pass_k=int(config.eval.pass_k),
                    seed=int(config.eval.seed),
                    shuffle=bool(config.eval.shuffle),
                ),
                logging_config=EvalLoggingConfig(
                    summary_path=str(config.eval.logging.summary_path),
                    trajectory_path=str(config.eval.logging.trajectory_path),
                    trace_path=str(config.eval.logging.trace_path),
                    max_traces=int(config.eval.logging.max_traces),
                ),
                close_engine_on_finish=False,
            )
            from climbing_gym.preflight import validate_eval_preflight
            eval_run = replace(eval_run, preflight=validate_eval_preflight(config, eval_run))
        checkpoint_manager = CheckpointManager(config.checkpoint)
        wandb_logger = WandbLogger(wandb_config, config_path=path, resolved_config={**config.to_dict(), "preflight": preflight_report,
                                                "eval_preflight": eval_run.preflight if eval_run is not None else None})

        return ResolvedPPORun(
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            value_model=value_model,
            value_optimizer=value_optimizer,
            rollout_coordinator=rollout_coordinator,
            config=config,
            engine=engine,
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
            eval_run=eval_run,
            start_step=int(start_step),
        )
    except Exception:
        if wandb_logger is not None:
            wandb_logger.close()
        if engine is not None:
            engine.close()
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run climbing_gym PPO training from a single TOML config.")
    parser.add_argument("config", type=str, help="Path to the PPO TOML config.")
    return parser


def _collect_sequence_values(
    value_model: nn.Module,
    *,
    batches: Sequence[Batch],
    device: torch.device,
) -> dict[tuple[int, int], tuple[float, ...]]:
    sequence_values: dict[tuple[int, int], tuple[float, ...]] = {}
    value_model.eval()
    pending: list[tuple[Batch, torch.Tensor]] = []
    with torch.inference_mode():
        for batch in batches:
            device_batch = batch.to(device)
            packed_values = value_model(device_batch.inputs, positions=device_batch.position_ids).squeeze(0)
            pending.append((batch, packed_values.detach().reshape(-1)))

    # A per-pack .cpu() forces a device synchronization after every critic
    # forward. Queue all no-grad forwards first and transfer the tiny scalar
    # value stream once so the GPU can execute the prepass continuously.
    if not pending:
        return sequence_values
    packed_cpu_values = torch.cat([values for _batch, values in pending]).cpu()
    offset = 0
    for batch, packed_values in pending:
        token_count = int(packed_values.numel())
        batch_values = packed_cpu_values[offset : offset + token_count]
        offset += token_count
        per_sequence = extract_completion_values(batch, batch_values)
        for local_index, values in enumerate(per_sequence):
            key = (
                int(batch.trajectory_indices[local_index]),
                int(batch.turn_indices[local_index]),
            )
            sequence_values[key] = tuple(float(value) for value in values)
    return sequence_values


def _build_completion_targets(
    batch: Batch,
    *,
    sequence_payloads: dict[tuple[int, int], tuple[float, ...]],
) -> torch.Tensor:
    return batch.completion_value_tensor(
        [
            sequence_payloads[(int(batch.trajectory_indices[idx]), int(batch.turn_indices[idx]))]
            for idx in range(batch.sequence_count)
        ]
    )


def _remap_batch_trajectory_indices(
    batch: Batch,
    *,
    local_to_global: Sequence[int],
) -> Batch:
    return Batch(
        inputs=batch.inputs,
        targets=batch.targets,
        position_ids=batch.position_ids,
        seq_indices=batch.seq_indices,
        prompt_lens=batch.prompt_lens,
        sequences=batch.sequences,
        trajectory_indices=tuple(int(local_to_global[int(idx)]) for idx in batch.trajectory_indices),
        group_ids=batch.group_ids,
        rollout_indices=batch.rollout_indices,
        turn_indices=batch.turn_indices,
        completion_logprobs=batch.completion_logprobs,
        packing_efficiency=batch.packing_efficiency,
    )


def _group_trajectories_by_prompt_group(
    trajectories: Sequence[Trajectory],
) -> list[list[tuple[int, Trajectory]]]:
    groups_by_id: dict[int, list[tuple[int, Trajectory]]] = {}
    ordered_group_ids: list[int] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        group_id = int(trajectory.group_id)
        if group_id not in groups_by_id:
            groups_by_id[group_id] = []
            ordered_group_ids.append(group_id)
        groups_by_id[group_id].append((int(trajectory_index), trajectory))
    return [groups_by_id[group_id] for group_id in ordered_group_ids]


def _pack_value_minibatch_groups(
    grouped_trajectories: Sequence[Sequence[tuple[int, Trajectory]]],
    *,
    prompt_groups_per_minibatch: int,
    max_tokens: int,
) -> list[list[Batch]]:
    if prompt_groups_per_minibatch <= 0:
        raise ValueError("value_minibatch_prompt_groups must be > 0 when provided.")

    minibatches: list[list[Batch]] = []
    for start_idx in range(0, len(grouped_trajectories), int(prompt_groups_per_minibatch)):
        group_chunk = grouped_trajectories[start_idx : start_idx + int(prompt_groups_per_minibatch)]
        chunk_pairs = [pair for group in group_chunk for pair in group]
        if not chunk_pairs:
            continue
        local_to_global = [trajectory_index for trajectory_index, _ in chunk_pairs]
        local_trajectories = [trajectory for _, trajectory in chunk_pairs]
        packed_batches = [
            _remap_batch_trajectory_indices(batch, local_to_global=local_to_global)
            for batch in pack_trajectories(local_trajectories, max_tokens=max_tokens)
        ]
        if packed_batches:
            minibatches.append(packed_batches)
    return minibatches


def _pack_bpco_trajectory_minibatches(
    trajectories: Sequence[Trajectory],
    *,
    trajectory_indices: Sequence[int],
    trajectory_minibatch_size: int,
    max_tokens: int,
    prompt_token_overrides: Mapping[tuple[int, int], Sequence[int]] | None = None,
) -> list[list[Batch]]:
    """Pack ordered logical trajectory minibatches into memory microbatches."""

    minibatch_size = int(trajectory_minibatch_size)
    if minibatch_size <= 0:
        raise ValueError("trajectory_minibatch_size must be > 0.")

    ordered_indices = [int(index) for index in trajectory_indices]
    if any(index < 0 or index >= len(trajectories) for index in ordered_indices):
        raise IndexError("trajectory_indices contains an out-of-range trajectory index.")

    logical_minibatches: list[list[Batch]] = []
    for start_idx in range(0, len(ordered_indices), minibatch_size):
        global_indices = ordered_indices[start_idx : start_idx + minibatch_size]
        local_trajectories = [trajectories[index] for index in global_indices]
        local_prompt_overrides: dict[tuple[int, int], Sequence[int]] | None = None
        if prompt_token_overrides is not None:
            local_prompt_overrides = {}
            for local_index, global_index in enumerate(global_indices):
                for turn in trajectories[global_index].turns:
                    global_key = (int(global_index), int(turn.turn_index))
                    if global_key in prompt_token_overrides:
                        local_prompt_overrides[(local_index, int(turn.turn_index))] = (
                            prompt_token_overrides[global_key]
                        )

        packed_batches = [
            _remap_batch_trajectory_indices(batch, local_to_global=global_indices)
            for batch in pack_trajectories(
                local_trajectories,
                max_tokens=max_tokens,
                prompt_token_overrides=local_prompt_overrides,
            )
        ]
        if packed_batches:
            logical_minibatches.append(packed_batches)
    return logical_minibatches


def main(
    argv: Sequence[str] | None = None,
    *,
    required_algorithm: str | None = None,
    setup_run: Callable[[ResolvedPPORun], ResolvedPPORun] | None = None,
    observe_step: Callable[[Mapping[str, Any]], None] | None = None,
) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    run = load_run_from_toml(args.config)
    if setup_run is not None:
        run = setup_run(run)
    policy_model = run.policy_model
    policy_optimizer = run.policy_optimizer
    value_model = run.value_model
    value_optimizer = run.value_optimizer
    rollout_coordinator = run.rollout_coordinator
    checkpoint_manager = run.checkpoint_manager
    wandb_logger = run.wandb_logger
    config = run.config
    trainer_config = config.trainer
    algorithm = _trainer_algorithm(config)
    if required_algorithm is not None and algorithm != str(required_algorithm).strip().lower():
        raise ValueError(
            f"This entrypoint requires trainer.algorithm={required_algorithm!r}; got {algorithm!r}."
        )
    use_bpco = algorithm == "bpco"
    loss_aggregation = str(trainer_config.loss_aggregation).strip().lower()
    eval_run = run.eval_run
    start_step = int(run.start_step)

    rollout_coordinator.rollout_config = config.rollout.build_rollout_config()

    num_steps = int(trainer_config.num_steps)
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    critic_warmup_steps = int(getattr(trainer_config, "critic_warmup_steps", 0))
    if critic_warmup_steps < 0:
        raise ValueError("critic_warmup_steps must be >= 0")
    remaining_warmup_steps = max(0, critic_warmup_steps - start_step)
    total_training_steps = remaining_warmup_steps + num_steps
    final_step = start_step + total_training_steps - 1
    train_device = torch.device(trainer_config.train_device)
    if use_bpco:
        if trainer_config.train_dtype is None:
            policy_model.to("cpu")
            value_model.to("cpu")
        else:
            policy_model.to(device="cpu", dtype=trainer_config.train_dtype)
            value_model.to(device="cpu", dtype=trainer_config.train_dtype)
    elif trainer_config.train_dtype is None:
        policy_model.to(train_device)
        value_model.to(train_device)
    else:
        policy_model.to(device=train_device, dtype=trainer_config.train_dtype)
        value_model.to(device=train_device, dtype=trainer_config.train_dtype)

    policy_loss_target = (
        packed_dppo_policy_loss_from_model
        if use_bpco
        else packed_ppo_policy_loss_from_model
    )
    if bool(trainer_config.compile_losses):
        compiled_policy_loss_fn = maybe_compile(
            policy_loss_target,
            dynamic=bool(trainer_config.compile_dynamic),
            fullgraph=bool(trainer_config.compile_fullgraph),
            backend=str(trainer_config.compile_backend),
            mode=str(trainer_config.compile_mode),
        )
        compiled_value_loss_fn = maybe_compile(
            packed_bpco_value_loss_from_model
            if use_bpco
            else packed_value_loss_from_model,
            dynamic=bool(trainer_config.compile_dynamic),
            fullgraph=bool(trainer_config.compile_fullgraph),
            backend=str(trainer_config.compile_backend),
            mode=str(trainer_config.compile_mode),
        )
    else:
        compiled_policy_loss_fn = policy_loss_target
        compiled_value_loss_fn = (
            packed_bpco_value_loss_from_model
            if use_bpco
            else packed_value_loss_from_model
        )

    try:
        for step in range(start_step, start_step + total_training_steps):
            critic_warmup_active = step < critic_warmup_steps
            step_started_at = time.perf_counter()

            rollout_started_at = time.perf_counter()
            if use_bpco:
                # A full actor, full critic, their optimizer states, and the
                # native engine do not all fit concurrently on one 24 GB GPU.
                # Keep trainables offloaded after rollout and load only the
                # critic or actor for its own phase below.
                offload_to_cpu(policy_model, policy_optimizer)
                offload_to_cpu(value_model, value_optimizer)
                rollout_coordinator.sync_weights(
                    policy_model,
                    drain_timeout=trainer_config.drain_timeout,
                )
                trajectory_batch = rollout_coordinator.collect()
            else:
                trajectory_batch = collect_rollout_batch_with_trainables(
                    rollout_coordinator=rollout_coordinator,
                    trainables=((policy_model, policy_optimizer), (value_model, value_optimizer)),
                    sync_model=policy_model,
                    device=train_device,
                    dtype=trainer_config.train_dtype,
                    drain_timeout=trainer_config.drain_timeout,
                    release_inference_buffers_before_update=trainer_config.release_inference_buffers_before_update,
                )
            rollout_time_seconds = time.perf_counter() - rollout_started_at
            _append_training_trajectories(
                str(config.train_logging.trajectory_path),
                step=step,
                trajectories=trajectory_batch.trajectories,
            )

            value_started_at = time.perf_counter()
            privileged_prompt_tokens: dict[tuple[int, int], tuple[int, ...]] | None = None
            if use_bpco:
                for trajectory in trajectory_batch.trajectories:
                    outcome = float(trajectory.evaluation.reward.reward)
                    if not (
                        float(trainer_config.reward_min)
                        <= outcome
                        <= float(trainer_config.reward_max)
                    ):
                        raise ValueError(
                            "Observed reward falls outside the configured critic range: "
                            f"reward={outcome} range=[{trainer_config.reward_min}, "
                            f"{trainer_config.reward_max}]"
                        )
                if trainer_config.privileged_critic:
                    tokenize_prompt = getattr(run.engine, "tokenize_prompt", None)
                    if not callable(tokenize_prompt):
                        raise TypeError(
                            "BPCO's privileged critic requires the rollout engine to provide "
                            "tokenize_prompt(prompt)."
                        )
                    privileged_prompt_tokens = build_privileged_prompt_token_overrides(
                        trajectory_batch.trajectories,
                        tokenize_prompt=tokenize_prompt,
                        template=str(trainer_config.privileged_answer_template),
                    )
            value_batches = list(
                pack_trajectories(
                    trajectory_batch.trajectories,
                    max_tokens=trainer_config.value_max_packed_tokens,
                    prompt_token_overrides=privileged_prompt_tokens,
                )
            )
            policy_batches = list(
                pack_trajectories(
                    trajectory_batch.trajectories,
                    max_tokens=trainer_config.policy_max_packed_tokens,
                )
            )

            offload_to_cpu(policy_model, policy_optimizer)
            reload_to_device(
                value_model,
                value_optimizer,
                device=train_device,
                dtype=trainer_config.train_dtype,
            )
            sequence_values = _collect_sequence_values(
                value_model,
                batches=value_batches,
                device=train_device,
            )
            if use_bpco:
                sequence_advantages, sequence_returns, gae_metrics = (
                    compute_bpco_trajectory_token_targets(
                        trajectory_batch.trajectories,
                        sequence_values=sequence_values,
                        gamma=trainer_config.gamma,
                        length_adaptive_gae_alpha=(
                            trainer_config.length_adaptive_gae_alpha
                        ),
                        policy_gae_lambda=trainer_config.bpco_policy_gae_lambda,
                    )
                )
                policy_sequence_advantages = sequence_advantages
            else:
                sequence_advantages, sequence_returns, gae_metrics = (
                    compute_trajectory_token_gae(
                        trajectory_batch.trajectories,
                        sequence_values=sequence_values,
                        gamma=trainer_config.gamma,
                        gae_lambda=trainer_config.gae_lambda,
                    )
                )
                policy_sequence_advantages = normalize_sequence_payloads(
                    sequence_advantages
                )

            total_value_valid_tokens = float(sum(batch.valid_token_count for batch in value_batches))
            value_loss_mean = 0.0
            mean_value = 0.0
            mean_return = 0.0
            value_grad_norm = 0.0
            value_update_norm = 0.0
            value_changed_parameter_count = 0.0
            value_optimizer_steps = 0
            value_explained_variance = compute_value_explained_variance(
                sequence_values=sequence_values,
                sequence_returns=sequence_returns,
            )

            value_model.train()
            if total_value_valid_tokens > 0.0:
                total_value_weight = total_value_valid_tokens * float(trainer_config.value_epochs)
                total_value_loss_weight = (
                    float(sum(batch.sequence_count for batch in value_batches))
                    * float(trainer_config.value_epochs)
                    if use_bpco and loss_aggregation == "seq_mean_token_sum"
                    else total_value_weight
                )
                if use_bpco:
                    value_grad_norms: list[float] = []
                    value_metric_terms: list[torch.Tensor] = []
                    for epoch_idx in range(trainer_config.value_epochs):
                        epoch_trajectory_indices = list(
                            range(len(trajectory_batch.trajectories))
                        )
                        if (
                            trainer_config.shuffle_value_batches
                            and len(epoch_trajectory_indices) > 1
                        ):
                            random.Random(
                                trainer_config.seed + step + epoch_idx
                            ).shuffle(epoch_trajectory_indices)
                        epoch_minibatches = _pack_bpco_trajectory_minibatches(
                            trajectory_batch.trajectories,
                            trajectory_indices=epoch_trajectory_indices,
                            trajectory_minibatch_size=(
                                trainer_config.trajectory_minibatch_size
                            ),
                            max_tokens=trainer_config.value_max_packed_tokens,
                            prompt_token_overrides=privileged_prompt_tokens,
                        )
                        for minibatch_batches in epoch_minibatches:
                            minibatch_valid_tokens = float(
                                sum(
                                    batch.valid_token_count
                                    for batch in minibatch_batches
                                )
                            )
                            minibatch_sequence_count = sum(
                                batch.sequence_count for batch in minibatch_batches
                            )
                            if (
                                minibatch_valid_tokens <= 0.0
                                or minibatch_sequence_count <= 0
                            ):
                                continue

                            value_optimizer.zero_grad(set_to_none=True)
                            for batch in minibatch_batches:
                                target_returns = _build_completion_targets(
                                    batch,
                                    sequence_payloads=sequence_returns,
                                ).to(train_device)
                                device_batch = batch.to(train_device)
                                (
                                    loss,
                                    batch_mean_value,
                                    batch_mean_return,
                                    valid_count,
                                ) = compiled_value_loss_fn(
                                    value_model,
                                    batch=device_batch,
                                    target_returns=target_returns,
                                    loss_aggregation=loss_aggregation,
                                )
                                scale_bpco_loss_for_microbatch(
                                    loss,
                                    loss_aggregation=loss_aggregation,
                                    microbatch_valid_tokens=float(
                                        batch.valid_token_count
                                    ),
                                    microbatch_sequence_count=batch.sequence_count,
                                    minibatch_valid_tokens=minibatch_valid_tokens,
                                    minibatch_sequence_count=(
                                        minibatch_sequence_count
                                    ),
                                ).backward()

                                weight = float(batch.valid_token_count)
                                loss_weight = (
                                    float(batch.sequence_count)
                                    if loss_aggregation == "seq_mean_token_sum"
                                    else weight
                                )
                                value_metric_terms.append(
                                    torch.stack(
                                        (
                                            loss.detach() * loss_weight,
                                            batch_mean_value.detach() * weight,
                                            batch_mean_return.detach() * weight,
                                        )
                                    )
                                )

                            value_grad_norms.append(
                                float(
                                    torch.nn.utils.clip_grad_norm_(
                                        value_model.parameters(),
                                        trainer_config.value_max_grad_norm,
                                    ).item()
                                )
                                if trainer_config.value_max_grad_norm > 0.0
                                else compute_grad_norm(value_model)
                            )
                            value_optimizer.step()
                            value_optimizer_steps += 1
                            value_update_norm += float(
                                getattr(
                                    value_optimizer,
                                    "last_master_update_norm",
                                    0.0,
                                )
                            )
                            value_changed_parameter_count += float(
                                getattr(
                                    value_optimizer,
                                    "last_quantized_change_count",
                                    0,
                                )
                            )
                            value_optimizer.zero_grad(set_to_none=True)

                    if value_metric_terms:
                        (
                            value_loss_mean,
                            mean_value,
                            mean_return,
                        ) = map(
                            float,
                            torch.stack(value_metric_terms)
                            .sum(dim=0)
                            .cpu()
                            .tolist(),
                        )
                    value_loss_mean /= total_value_loss_weight
                    mean_value /= total_value_weight
                    mean_return /= total_value_weight
                    value_grad_norm = (
                        float(sum(value_grad_norms) / len(value_grad_norms))
                        if value_grad_norms
                        else 0.0
                    )
                elif trainer_config.value_minibatch_prompt_groups is None:
                    value_optimizer.zero_grad(set_to_none=True)
                    for epoch_idx in range(trainer_config.value_epochs):
                        epoch_batches = list(value_batches)
                        if trainer_config.shuffle_value_batches and len(epoch_batches) > 1:
                            random.Random(trainer_config.seed + step + epoch_idx).shuffle(epoch_batches)
                        for batch in epoch_batches:
                            target_returns = _build_completion_targets(
                                batch,
                                sequence_payloads=sequence_returns,
                            ).to(train_device)
                            device_batch = batch.to(train_device)
                            (
                                loss,
                                batch_mean_value,
                                batch_mean_return,
                                valid_count,
                            ) = compiled_value_loss_fn(
                                value_model,
                                batch=device_batch,
                                target_returns=target_returns,
                            )
                            scale_ppo_loss_for_packed_batch(
                                loss,
                                valid_count=valid_count,
                                total_valid_tokens=total_value_weight,
                            ).backward()

                            weight = float(batch.valid_token_count)
                            value_loss_mean += (
                                float(loss.detach().item()) * weight
                            )
                            mean_value += float(batch_mean_value.detach().item()) * weight
                            mean_return += float(batch_mean_return.detach().item()) * weight

                    value_loss_mean /= total_value_loss_weight
                    mean_value /= total_value_weight
                    mean_return /= total_value_weight
                    value_grad_norm = (
                        float(
                            torch.nn.utils.clip_grad_norm_(
                                value_model.parameters(),
                                trainer_config.value_max_grad_norm,
                            ).item()
                        )
                        if trainer_config.value_max_grad_norm > 0.0
                        else compute_grad_norm(value_model)
                    )
                    value_optimizer.step()
                    value_optimizer.zero_grad(set_to_none=True)
                else:
                    grouped_trajectories = _group_trajectories_by_prompt_group(trajectory_batch.trajectories)
                    grad_norms: list[float] = []
                    for epoch_idx in range(trainer_config.value_epochs):
                        epoch_groups = list(grouped_trajectories)
                        if trainer_config.shuffle_value_batches and len(epoch_groups) > 1:
                            random.Random(trainer_config.seed + step + epoch_idx).shuffle(epoch_groups)
                        epoch_minibatches = _pack_value_minibatch_groups(
                            epoch_groups,
                            prompt_groups_per_minibatch=int(trainer_config.value_minibatch_prompt_groups),
                            max_tokens=trainer_config.value_max_packed_tokens,
                        )
                        for minibatch_batches in epoch_minibatches:
                            minibatch_total_valid_tokens = float(
                                sum(batch.valid_token_count for batch in minibatch_batches)
                            )
                            if minibatch_total_valid_tokens <= 0.0:
                                continue
                            value_optimizer.zero_grad(set_to_none=True)
                            for batch in minibatch_batches:
                                target_returns = _build_completion_targets(
                                    batch,
                                    sequence_payloads=sequence_returns,
                                ).to(train_device)
                                device_batch = batch.to(train_device)
                                loss, batch_mean_value, batch_mean_return, valid_count = compiled_value_loss_fn(
                                    value_model,
                                    batch=device_batch,
                                    target_returns=target_returns,
                                )
                                scale_ppo_loss_for_packed_batch(
                                    loss,
                                    valid_count=valid_count,
                                    total_valid_tokens=minibatch_total_valid_tokens,
                                ).backward()

                                weight = float(batch.valid_token_count)
                                value_loss_mean += float(loss.detach().item()) * weight
                                mean_value += float(batch_mean_value.detach().item()) * weight
                                mean_return += float(batch_mean_return.detach().item()) * weight

                            grad_norm = (
                                float(
                                    torch.nn.utils.clip_grad_norm_(
                                        value_model.parameters(),
                                        trainer_config.value_max_grad_norm,
                                    ).item()
                                )
                                if trainer_config.value_max_grad_norm > 0.0
                                else compute_grad_norm(value_model)
                            )
                            grad_norms.append(grad_norm)
                            value_optimizer.step()
                            value_optimizer_steps += 1
                            value_update_norm = float(
                                getattr(
                                    value_optimizer,
                                    "last_master_update_norm",
                                    0.0,
                                )
                            )
                            value_changed_parameter_count += float(
                                getattr(
                                    value_optimizer,
                                    "last_quantized_change_count",
                                    0,
                                )
                            )
                            value_optimizer.zero_grad(set_to_none=True)

                    value_loss_mean /= total_value_weight
                    mean_value /= total_value_weight
                    mean_return /= total_value_weight
                    value_grad_norm = (
                        float(sum(grad_norms) / len(grad_norms))
                        if grad_norms
                        else 0.0
                    )
            value_time_seconds = time.perf_counter() - value_started_at

            policy_started_at = time.perf_counter()
            total_policy_valid_tokens = float(sum(batch.valid_token_count for batch in policy_batches))
            policy_loss_mean = 0.0
            entropy_mean = 0.0
            clip_fraction_mean = 0.0
            ratio_mean = 0.0
            approx_kl_mean = 0.0
            probability_shift_mean = 0.0
            policy_grad_norm = 0.0
            policy_update_norm = 0.0
            policy_changed_parameter_count = 0.0
            policy_optimizer_steps = 0
            policy_grad_norms: list[float] = []
            pre_update_probability_shift = 0.0
            pre_update_mean_ratio = 0.0
            pre_update_approx_kl = 0.0

            if not critic_warmup_active:
                offload_to_cpu(value_model, value_optimizer)
                reload_to_device(
                    policy_model,
                    policy_optimizer,
                    device=train_device,
                    dtype=trainer_config.train_dtype,
                )

                policy_model.train()
                policy_optimizer.zero_grad(set_to_none=True)
                if total_policy_valid_tokens > 0.0:
                    total_policy_weight = total_policy_valid_tokens * float(trainer_config.policy_epochs)
                    total_policy_loss_weight = (
                        float(sum(batch.sequence_count for batch in policy_batches))
                        * float(trainer_config.policy_epochs)
                        if use_bpco and loss_aggregation == "seq_mean_token_sum"
                        else total_policy_weight
                    )
                    if use_bpco:
                        policy_metric_terms: list[torch.Tensor] = []
                        for epoch_idx in range(trainer_config.policy_epochs):
                            epoch_trajectory_indices = list(
                                range(len(trajectory_batch.trajectories))
                            )
                            if (
                                trainer_config.shuffle_policy_batches
                                and len(epoch_trajectory_indices) > 1
                            ):
                                random.Random(
                                    trainer_config.seed
                                    + 10_000
                                    + step
                                    + epoch_idx
                                ).shuffle(epoch_trajectory_indices)
                            epoch_minibatches = _pack_bpco_trajectory_minibatches(
                                trajectory_batch.trajectories,
                                trajectory_indices=epoch_trajectory_indices,
                                trajectory_minibatch_size=(
                                    trainer_config.trajectory_minibatch_size
                                ),
                                max_tokens=trainer_config.policy_max_packed_tokens,
                            )
                            for minibatch_batches in epoch_minibatches:
                                minibatch_valid_tokens = float(
                                    sum(
                                        batch.valid_token_count
                                        for batch in minibatch_batches
                                    )
                                )
                                minibatch_sequence_count = sum(
                                    batch.sequence_count
                                    for batch in minibatch_batches
                                )
                                if (
                                    minibatch_valid_tokens <= 0.0
                                    or minibatch_sequence_count <= 0
                                ):
                                    continue

                                is_pre_update_minibatch = policy_optimizer_steps == 0
                                pre_update_metric_weight = 0.0
                                pre_update_probability_shift_sum = 0.0
                                pre_update_mean_ratio_sum = 0.0
                                pre_update_approx_kl_sum = 0.0
                                minibatch_metric_terms: list[torch.Tensor] = []
                                policy_optimizer.zero_grad(set_to_none=True)
                                for batch in minibatch_batches:
                                    token_advantages = _build_completion_targets(
                                        batch,
                                        sequence_payloads=policy_sequence_advantages,
                                    ).to(train_device)
                                    behavior_logprobs = batch.completion_value_tensor(
                                        batch.resolved_completion_logprobs()
                                    ).to(train_device)
                                    device_batch = batch.to(train_device)
                                    (
                                        total_loss,
                                        policy_loss,
                                        entropy,
                                        clip_fraction,
                                        mean_ratio,
                                        approx_kl,
                                        probability_shift,
                                        valid_count,
                                    ) = compiled_policy_loss_fn(
                                        policy_model,
                                        batch=device_batch,
                                        token_advantages=token_advantages,
                                        behavior_logprobs=behavior_logprobs,
                                        clip_epsilon=trainer_config.clip_epsilon,
                                        entropy_coef=trainer_config.entropy_coef,
                                        log_ratio_clip=(
                                            trainer_config.dppo_log_ratio_clip
                                        ),
                                        loss_aggregation=loss_aggregation,
                                    )
                                    scale_bpco_loss_for_microbatch(
                                        total_loss,
                                        loss_aggregation=loss_aggregation,
                                        microbatch_valid_tokens=float(
                                            batch.valid_token_count
                                        ),
                                        microbatch_sequence_count=(
                                            batch.sequence_count
                                        ),
                                        minibatch_valid_tokens=(
                                            minibatch_valid_tokens
                                        ),
                                        minibatch_sequence_count=(
                                            minibatch_sequence_count
                                        ),
                                    ).backward()

                                    weight = float(batch.valid_token_count)
                                    loss_weight = (
                                        float(batch.sequence_count)
                                        if loss_aggregation
                                        == "seq_mean_token_sum"
                                        else weight
                                    )
                                    metric_term = torch.stack(
                                        (
                                            policy_loss.detach() * loss_weight,
                                            entropy.detach() * weight,
                                            clip_fraction.detach() * weight,
                                            mean_ratio.detach() * weight,
                                            approx_kl.detach() * weight,
                                            probability_shift.detach() * weight,
                                        )
                                    )
                                    policy_metric_terms.append(metric_term)
                                    minibatch_metric_terms.append(metric_term)
                                    if is_pre_update_minibatch:
                                        pre_update_metric_weight += weight

                                if (
                                    is_pre_update_minibatch
                                    and pre_update_metric_weight > 0.0
                                ):
                                    pre_update_metric_values = (
                                        torch.stack(minibatch_metric_terms)
                                        .sum(dim=0)
                                        .cpu()
                                        .tolist()
                                    )
                                    pre_update_mean_ratio_sum = float(
                                        pre_update_metric_values[3]
                                    )
                                    pre_update_approx_kl_sum = float(
                                        pre_update_metric_values[4]
                                    )
                                    pre_update_probability_shift_sum = float(
                                        pre_update_metric_values[5]
                                    )
                                    pre_update_probability_shift = (
                                        pre_update_probability_shift_sum
                                        / pre_update_metric_weight
                                    )
                                    pre_update_mean_ratio = (
                                        pre_update_mean_ratio_sum
                                        / pre_update_metric_weight
                                    )
                                    pre_update_approx_kl = (
                                        pre_update_approx_kl_sum
                                        / pre_update_metric_weight
                                    )
                                    parity_limit = float(
                                        trainer_config.behavior_parity_max_mean_abs_probability_shift
                                    )
                                    if (
                                        pre_update_probability_shift
                                        > parity_limit
                                    ):
                                        raise RuntimeError(
                                            "Native/training behavior-policy parity check failed: "
                                            f"mean_abs_probability_shift={pre_update_probability_shift:.6g} "
                                            f"> configured limit={parity_limit:.6g}."
                                        )

                                policy_grad_norms.append(
                                    float(
                                        torch.nn.utils.clip_grad_norm_(
                                            policy_model.parameters(),
                                            trainer_config.policy_max_grad_norm,
                                        ).item()
                                    )
                                    if trainer_config.policy_max_grad_norm > 0.0
                                    else compute_grad_norm(policy_model)
                                )
                                policy_optimizer.step()
                                policy_optimizer_steps += 1
                                policy_update_norm += float(
                                    getattr(
                                        policy_optimizer,
                                        "last_master_update_norm",
                                        0.0,
                                    )
                                )
                                policy_changed_parameter_count += float(
                                    getattr(
                                        policy_optimizer,
                                        "last_quantized_change_count",
                                        0,
                                    )
                                )
                                policy_optimizer.zero_grad(set_to_none=True)
                    else:
                        for epoch_idx in range(trainer_config.policy_epochs):
                            epoch_batches = list(policy_batches)
                            if (
                                trainer_config.shuffle_policy_batches
                                and len(epoch_batches) > 1
                            ):
                                random.Random(
                                    trainer_config.seed
                                    + 10_000
                                    + step
                                    + epoch_idx
                                ).shuffle(epoch_batches)
                            for batch in epoch_batches:
                                token_advantages = _build_completion_targets(
                                    batch,
                                    sequence_payloads=policy_sequence_advantages,
                                ).to(train_device)
                                behavior_logprobs = batch.completion_value_tensor(
                                    batch.resolved_completion_logprobs()
                                ).to(train_device)
                                device_batch = batch.to(train_device)
                                (
                                    total_loss,
                                    policy_loss,
                                    entropy,
                                    clip_fraction,
                                    mean_ratio,
                                    approx_kl,
                                    valid_count,
                                ) = compiled_policy_loss_fn(
                                    policy_model,
                                    batch=device_batch,
                                    token_advantages=token_advantages,
                                    old_infer_logprobs=behavior_logprobs,
                                    clip_epsilon=trainer_config.clip_epsilon,
                                    entropy_coef=trainer_config.entropy_coef,
                                )
                                probability_shift = policy_loss.new_zeros(())
                                scale_ppo_loss_for_packed_batch(
                                    total_loss,
                                    valid_count=valid_count,
                                    total_valid_tokens=total_policy_weight,
                                ).backward()

                                weight = float(batch.valid_token_count)
                                policy_loss_mean += (
                                    float(policy_loss.detach().item()) * weight
                                )
                                entropy_mean += (
                                    float(entropy.detach().item()) * weight
                                )
                                clip_fraction_mean += (
                                    float(clip_fraction.detach().item()) * weight
                                )
                                ratio_mean += (
                                    float(mean_ratio.detach().item()) * weight
                                )
                                approx_kl_mean += (
                                    float(approx_kl.detach().item()) * weight
                                )
                                probability_shift_mean += (
                                    float(probability_shift.detach().item())
                                    * weight
                                )

                    if use_bpco and policy_metric_terms:
                        (
                            policy_loss_mean,
                            entropy_mean,
                            clip_fraction_mean,
                            ratio_mean,
                            approx_kl_mean,
                            probability_shift_mean,
                        ) = map(
                            float,
                            torch.stack(policy_metric_terms)
                            .sum(dim=0)
                            .cpu()
                            .tolist(),
                        )
                    policy_loss_mean /= total_policy_loss_weight
                    entropy_mean /= total_policy_weight
                    clip_fraction_mean /= total_policy_weight
                    ratio_mean /= total_policy_weight
                    approx_kl_mean /= total_policy_weight
                    probability_shift_mean /= total_policy_weight
                    if use_bpco:
                        policy_grad_norm = (
                            float(sum(policy_grad_norms) / len(policy_grad_norms))
                            if policy_grad_norms
                            else 0.0
                        )
                    else:
                        policy_grad_norm = (
                            float(
                                torch.nn.utils.clip_grad_norm_(
                                    policy_model.parameters(),
                                    trainer_config.policy_max_grad_norm,
                                ).item()
                            )
                            if trainer_config.policy_max_grad_norm > 0.0
                            else compute_grad_norm(policy_model)
                        )
                        policy_optimizer.step()
                        policy_optimizer.zero_grad(set_to_none=True)
            policy_time_seconds = time.perf_counter() - policy_started_at
            step_time_seconds = time.perf_counter() - step_started_at
            reward_mean = (
                float(
                    sum(
                        float(trajectory.evaluation.reward.reward)
                        for trajectory in trajectory_batch.trajectories
                    )
                    / len(trajectory_batch.trajectories)
                )
                if trajectory_batch.trajectories
                else 0.0
            )
            avg_completion_length = (
                float(
                    sum(
                        len(turn.completion_token_ids)
                        for trajectory in trajectory_batch.trajectories
                        for turn in trajectory.turns
                    )
                    / sum(len(trajectory.turns) for trajectory in trajectory_batch.trajectories)
                )
                if trajectory_batch.trajectories
                and any(trajectory.turns for trajectory in trajectory_batch.trajectories)
                else 0.0
            )
            avg_turn_count = (
                float(
                    sum(len(trajectory.turns) for trajectory in trajectory_batch.trajectories)
                    / len(trajectory_batch.trajectories)
                )
                if trajectory_batch.trajectories
                else 0.0
            )
            effective_group_target = float(trajectory_batch.stats.get("effective_group_target", 0.0))
            effective_group_count = float(trajectory_batch.stats.get("effective_group_count", 0.0))
            effective_group_fraction = (
                float(effective_group_count / effective_group_target)
                if effective_group_target > 0.0
                else 0.0
            )
            tokens_per_second = (
                float(total_policy_valid_tokens / step_time_seconds)
                if step_time_seconds > 0.0
                else 0.0
            )

            metrics = {
                "policy/reward_mean": float(reward_mean),
                "policy/loss": float(policy_loss_mean),
                "policy/clip_fraction": float(clip_fraction_mean),
                "policy/grad_norm": float(policy_grad_norm),
                "policy/entropy": float(entropy_mean),
                "policy/approx_kl": float(approx_kl_mean),
                "value/loss": float(value_loss_mean),
                "value/grad_norm": float(value_grad_norm),
                "value/explained_variance": float(value_explained_variance),
                "trainer/avg_completion_length": float(avg_completion_length),
                "trainer/avg_turn_count": float(avg_turn_count),
                "trainer/effective_group_fraction": float(effective_group_fraction),
                "trainer/tokens_per_second": float(tokens_per_second),
                "trainer/rollout_time_seconds": float(rollout_time_seconds),
                "trainer/value_time_seconds": float(value_time_seconds),
                "trainer/policy_time_seconds": float(policy_time_seconds),
                "trainer/step_time_seconds": float(step_time_seconds),
            }
            if use_bpco:
                metrics.update(
                    {
                        "policy/mean_abs_probability_shift": float(
                            probability_shift_mean
                        ),
                        "policy/mean_ratio": float(ratio_mean),
                        "policy/pre_update_approx_kl": float(
                            pre_update_approx_kl
                        ),
                        "policy/pre_update_mean_abs_probability_shift": float(
                            pre_update_probability_shift
                        ),
                        "policy/pre_update_mean_ratio": float(
                            pre_update_mean_ratio
                        ),
                        "policy/master_update_norm_sum": float(policy_update_norm),
                        "policy/optimizer_steps": float(policy_optimizer_steps),
                        "policy/quantized_parameter_change_events": float(
                            policy_changed_parameter_count
                        ),
                        "policy/advantage_mean": float(
                            gae_metrics["advantage_mean"]
                        ),
                        "policy/advantage_std": float(
                            gae_metrics["advantage_std"]
                        ),
                        "value/mean_prediction": float(mean_value),
                        "value/mean_mc_target": float(mean_return),
                        "value/master_update_norm_sum": float(value_update_norm),
                        "value/optimizer_steps": float(value_optimizer_steps),
                        "value/quantized_parameter_change_events": float(
                            value_changed_parameter_count
                        ),
                        "bpco/length_adaptive_lambda_mean": float(
                            gae_metrics["la_gae_lambda_mean"]
                        ),
                        "bpco/length_adaptive_lambda_min": float(
                            gae_metrics["la_gae_lambda_min"]
                        ),
                        "bpco/length_adaptive_lambda_max": float(
                            gae_metrics["la_gae_lambda_max"]
                        ),
                        "bpco/short_response_lambda_clamps": float(
                            gae_metrics["la_gae_short_response_clamp_count"]
                        ),
                        "bpco/privileged_critic": float(trainer_config.privileged_critic),
                        "bpco/trajectory_minibatch_size": float(
                            trainer_config.trajectory_minibatch_size
                        ),
                        "bpco/seq_mean_token_sum": float(
                            loss_aggregation == "seq_mean_token_sum"
                        ),
                    }
                )
                for info_key in (
                    "exact_match",
                    "valid_code",
                    "position_accuracy",
                    "symbol_overlap",
                    "clue_consistency",
                    "copied_clue_guess",
                    "reward_format",
                ):
                    values = [
                        float(trajectory.evaluation.reward.info.get(info_key, 0.0))
                        for trajectory in trajectory_batch.trajectories
                    ]
                    metrics[f"reward/{info_key}_mean"] = (
                        float(sum(values) / len(values)) if values else 0.0
                    )
            if eval_run is not None and int(config.eval.interval) > 0 and (int(step) + 1) % int(config.eval.interval) == 0:
                offload_to_cpu(policy_model, policy_optimizer)
                offload_to_cpu(value_model, value_optimizer)
                eval_run.rollout_coordinator.sync_weights(
                    policy_model,
                    drain_timeout=trainer_config.drain_timeout,
                )
                eval_summary = run_eval(eval_run)
                if trainer_config.release_inference_buffers_before_update:
                    release = getattr(run.engine, "release_inference_buffers", None)
                    if callable(release):
                        release()
                if not use_bpco:
                    reload_to_device(
                        policy_model,
                        policy_optimizer,
                        device=train_device,
                        dtype=trainer_config.train_dtype,
                    )
                    reload_to_device(
                        value_model,
                        value_optimizer,
                        device=train_device,
                        dtype=trainer_config.train_dtype,
                    )
                metrics.update(
                    {
                        f"eval/{key}": float(value)
                        for key, value in eval_summary.items()
                    }
                )
            checkpoint_path = checkpoint_manager.maybe_save(
                step=step,
                is_final=step == final_step,
                policy_model=policy_model,
                value_model=value_model,
                policy_optimizer=policy_optimizer,
                value_optimizer=value_optimizer,
            )
            if checkpoint_path is not None:
                from climbing_gym.dataset_checkpoint import save_dataset_checkpoint
                save_dataset_checkpoint(rollout_coordinator, Path(checkpoint_path))
            generated_tokens = sum(len(turn.completion_token_ids) for row in trajectory_batch.trajectories for turn in row.turns)
            metrics["trainer/rollout_tokens_per_second"] = generated_tokens / max(rollout_time_seconds, 1e-9)
            metrics["trainer/full_step_time_seconds"] = time.perf_counter() - step_started_at
            wandb_logger.log_step(step=step, metrics=metrics)
            payload = {
                "step": int(step),
                "metrics": metrics,
            }
            if checkpoint_path is not None:
                payload["checkpoint_path"] = str(checkpoint_path)
            _append_training_metrics(
                str(config.train_logging.metrics_path),
                payload,
            )
            if observe_step is not None:
                observe_step(payload)
            print(
                json.dumps(
                    payload,
                    sort_keys=True,
                )
            )
    finally:
        run.wandb_logger.close()
        run.engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
