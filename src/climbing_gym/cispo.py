"""Critic-free CISPO with centered outcome advantages and native rollouts.

Uses the GRPO data/checkpoint runtime, but clips detached importance weights
instead of the PPO surrogate. The loss averages trajectory token sums: memory
packing and response length do not introduce per-response normalization.
"""
from __future__ import annotations

from typing import Any, Sequence, TYPE_CHECKING

import torch
import torch.nn.functional as F

if TYPE_CHECKING:
    from climbing_gym.trainers.packing import Batch


def cispo_loss(
    log_probs: torch.Tensor,
    behavior_log_probs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    sequence_count: int,
    clip_low: float,
    clip_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if sequence_count <= 0 or clip_low < 0 or clip_high < 0:
        raise ValueError("CISPO requires positive sequence_count and nonnegative clip epsilons")
    log_ratio = log_probs.float() - behavior_log_probs.float()
    ratio = log_ratio.detach().clamp(-20, 20).exp()
    weight = ratio.clamp(min=max(0.0, 1.0 - clip_low), max=1.0 + clip_high)
    loss = -(weight * advantages.float() * log_probs.float()).sum() / sequence_count
    clip_fraction = (ratio != weight).float().mean()
    approx_kl = (ratio - 1.0 - log_ratio.detach()).mean()
    probability_shift = (log_probs.detach().exp() - behavior_log_probs.exp()).abs().mean()
    return loss, clip_fraction, ratio.mean(), approx_kl, probability_shift


def packed_cispo_policy_loss_from_model(
    policy_model: Any,
    *,
    batch: Batch,
    token_advantages: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, ...]:
    if entropy_coef != 0:
        raise ValueError("This CISPO experiment uses no entropy objective")
    mask = batch.valid_completion_mask().reshape(-1)
    hidden = policy_model.get_hidden_states(batch.inputs, batch.position_ids)
    completion_hidden = hidden.reshape(-1, hidden.shape[-1])[mask]
    targets = batch.targets.reshape(-1)[mask]
    if getattr(getattr(policy_model, "config", None), "fused_cispo_head", False):
        from climbing_gym.models.kernels.fused_cispo_head import fused_cispo_head
        loss, clipped, ratio, kl, shift = fused_cispo_head(
            completion_hidden, policy_model.lm_head.weight, targets,
            old_infer_logprobs.reshape(-1)[mask], token_advantages.reshape(-1)[mask],
            sequence_count=batch.sequence_count, clip_low=clip_epsilon_low, clip_high=clip_epsilon_high,
        )
        zero = loss.detach() * 0.0
        return loss, loss, zero, clipped, ratio, kl, mask.sum(), shift
    # Float32 cross entropy avoids BF16 log-sum-exp rounding in IS ratios.
    # Checkpoint head chunks to avoid retaining full vocabulary activations.
    from torch.utils.checkpoint import checkpoint

    def selected_log_probs(states: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return -F.cross_entropy(policy_model.lm_head(states).float(), labels, reduction="none")

    log_probs = torch.cat([
        checkpoint(selected_log_probs, states, labels, use_reentrant=False)
        for states, labels in zip(completion_hidden.split(256), targets.split(256))
    ])
    loss, clipped, ratio, kl, shift = cispo_loss(
        log_probs,
        old_infer_logprobs.reshape(-1)[mask],
        token_advantages.reshape(-1)[mask],
        sequence_count=batch.sequence_count,
        clip_low=clip_epsilon_low,
        clip_high=clip_epsilon_high,
    )
    zero = loss.detach() * 0.0
    return loss, loss, zero, clipped, ratio, kl, mask.sum(), shift


def main(argv: Sequence[str] | None = None) -> int:
    from climbing_gym.config import ConfigManager
    from climbing_gym.grpo import build_arg_parser, main as train

    args = build_arg_parser().parse_args(argv)
    config = ConfigManager.from_toml(args.config).config
    if config.trainer.algorithm != "cispo":
        raise ValueError("CISPO entrypoint requires trainer.algorithm='cispo'")
    if config.trainer.advantage_estimator != "grpo" or config.trainer.format_reward_enabled:
        raise ValueError("Use centered scalar outcome advantages without auxiliary format rewards")
    return train([args.config])


if __name__ == "__main__":
    raise SystemExit(main())
