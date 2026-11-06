"""Tests for MIS-PPO style dual-ratio filtering."""

from __future__ import annotations

import math

import pytest
import torch

from climbing_gym.mis_ppo import (
    compose_truncation_bootstrap_return,
    compute_grpo_advantages,
    packed_mis_ppo_actor_critic_loss,
    packed_mis_ppo_loss,
    should_sync_rollout_policy,
)


class _FakeTokenLogprobModel:
    def __init__(self, target_log_probs: list[float], *, dtype: torch.dtype = torch.float32) -> None:
        logits = []
        for log_prob in target_log_probs:
            prob = math.exp(log_prob)
            logits.append([log_prob, math.log1p(-prob)])
        self._logits = torch.tensor([logits], dtype=dtype)

    def __call__(self, inputs, positions=None):
        del inputs, positions
        return self._logits


class _FakeActorCriticTokenModel(_FakeTokenLogprobModel):
    def __init__(
        self,
        target_log_probs: list[float],
        values: list[float],
        *,
        dtype: torch.dtype = torch.float32,
    ) -> None:
        super().__init__(target_log_probs, dtype=dtype)
        self._values = torch.tensor([values], dtype=dtype)

    def forward_with_value(self, inputs, positions=None):
        del inputs, positions
        return self._logits, self._values


def test_mis_ppo_compute_advantages_group_normalization():
    advantages, metrics = compute_grpo_advantages(
        [1.0, 3.0, 2.0, 2.0],
        num_rollouts=2,
        eps=1e-6,
    )

    assert len(advantages) == 4
    assert advantages[0] == pytest.approx(-1.0)
    assert advantages[1] == pytest.approx(1.0)
    assert advantages[2] == pytest.approx(0.0)
    assert advantages[3] == pytest.approx(0.0)
    assert metrics["grpo_zero_std_prompt_rate"] == pytest.approx(0.5)


def test_packed_mis_ppo_loss_filters_out_of_range_tokens_but_keeps_valid_trajectory():
    target_log_probs = torch.tensor([-0.2, -0.3, -0.4], dtype=torch.float32)
    token_ratios = torch.tensor([0.97, 1.0, 1.08], dtype=torch.float32)
    old_infer_logprobs = target_log_probs - torch.log(token_ratios)

    (
        loss,
        _,
        token_retention_rate,
        trajectory_keep_rate,
        active_token_rate,
        mean_token_ratio,
        mean_trajectory_ratio,
        active_tokens,
        active_sequences,
        valid_count,
        fallback_applied,
    ) = packed_mis_ppo_loss(
        _FakeTokenLogprobModel(target_log_probs.tolist()),
        inputs=torch.tensor([[5, 6, 7]], dtype=torch.long),
        targets=torch.tensor([[0, 0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 2]], dtype=torch.long),
        seq_indices=torch.tensor([0, 0, 0], dtype=torch.long),
        prompt_lens=torch.tensor([1], dtype=torch.long),
        token_advantages=torch.ones(3, dtype=torch.float32),
        old_infer_logprobs=old_infer_logprobs,
        pad_id=99,
        token_ratio_min=0.95,
        token_ratio_max=1.05,
        trajectory_ratio_min=0.95,
        trajectory_ratio_max=1.05,
        fallback_to_valid=False,
    )

    expected_geom_ratio = math.exp(sum(math.log(float(value)) for value in token_ratios.tolist()) / 3.0)
    assert float(valid_count) == pytest.approx(3.0)
    assert float(token_retention_rate) == pytest.approx(2.0 / 3.0)
    assert float(trajectory_keep_rate) == pytest.approx(1.0)
    assert float(active_token_rate) == pytest.approx(2.0 / 3.0)
    assert float(mean_token_ratio) == pytest.approx(float(token_ratios.mean().item()))
    assert float(mean_trajectory_ratio) == pytest.approx(expected_geom_ratio)
    assert float(active_tokens) == pytest.approx(2.0)
    assert float(active_sequences) == pytest.approx(1.0)
    assert float(fallback_applied) == pytest.approx(0.0)
    assert float(loss) == pytest.approx(0.25)


def test_packed_mis_ppo_loss_trajectory_filter_can_drop_all_tokens():
    target_log_probs = torch.tensor([-0.2, -0.3], dtype=torch.float32)
    token_ratios = torch.tensor([0.8, 0.8], dtype=torch.float32)
    old_infer_logprobs = target_log_probs - torch.log(token_ratios)

    (
        loss,
        _,
        token_retention_rate,
        trajectory_keep_rate,
        active_token_rate,
        _,
        mean_trajectory_ratio,
        active_tokens,
        active_sequences,
        valid_count,
        fallback_applied,
    ) = packed_mis_ppo_loss(
        _FakeTokenLogprobModel(target_log_probs.tolist()),
        inputs=torch.tensor([[5, 6]], dtype=torch.long),
        targets=torch.tensor([[0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        seq_indices=torch.tensor([0, 0], dtype=torch.long),
        prompt_lens=torch.tensor([1], dtype=torch.long),
        token_advantages=torch.ones(2, dtype=torch.float32),
        old_infer_logprobs=old_infer_logprobs,
        pad_id=99,
        token_ratio_min=0.5,
        token_ratio_max=1.5,
        trajectory_ratio_min=0.95,
        trajectory_ratio_max=1.05,
        fallback_to_valid=False,
    )

    assert float(valid_count) == pytest.approx(2.0)
    assert float(token_retention_rate) == pytest.approx(1.0)
    assert float(trajectory_keep_rate) == pytest.approx(0.0)
    assert float(active_token_rate) == pytest.approx(0.0)
    assert float(mean_trajectory_ratio) == pytest.approx(0.8)
    assert float(active_tokens) == pytest.approx(0.0)
    assert float(active_sequences) == pytest.approx(0.0)
    assert float(fallback_applied) == pytest.approx(0.0)
    assert float(loss) == pytest.approx(0.0)


def test_packed_mis_ppo_loss_trajectory_filter_spans_multiple_turns_from_same_rollout():
    target_log_probs = torch.tensor([-0.2, -0.3, -0.4, -0.5], dtype=torch.float32)
    token_ratios = torch.tensor([1.0, 1.0, 0.8, 0.8], dtype=torch.float32)
    old_infer_logprobs = target_log_probs - torch.log(token_ratios)

    (
        loss,
        _,
        token_retention_rate,
        trajectory_keep_rate,
        active_token_rate,
        _,
        mean_trajectory_ratio,
        active_tokens,
        active_sequences,
        valid_count,
        fallback_applied,
    ) = packed_mis_ppo_loss(
        _FakeTokenLogprobModel(target_log_probs.tolist()),
        inputs=torch.tensor([[5, 6, 7, 8]], dtype=torch.long),
        targets=torch.tensor([[0, 0, 0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1, 0, 1]], dtype=torch.long),
        seq_indices=torch.tensor([0, 0, 1, 1], dtype=torch.long),
        prompt_lens=torch.tensor([1, 1], dtype=torch.long),
        token_advantages=torch.ones(4, dtype=torch.float32),
        old_infer_logprobs=old_infer_logprobs,
        pad_id=99,
        token_ratio_min=0.5,
        token_ratio_max=1.5,
        trajectory_ratio_min=0.95,
        trajectory_ratio_max=1.05,
        fallback_to_valid=False,
        sequence_trajectory_ids=torch.tensor([0, 0], dtype=torch.long),
    )

    expected_geom_ratio = math.exp(sum(math.log(float(value)) for value in token_ratios.tolist()) / 4.0)
    assert float(valid_count) == pytest.approx(4.0)
    assert float(token_retention_rate) == pytest.approx(1.0)
    assert float(trajectory_keep_rate) == pytest.approx(0.0)
    assert float(active_token_rate) == pytest.approx(0.0)
    assert float(mean_trajectory_ratio) == pytest.approx(expected_geom_ratio)
    assert float(active_tokens) == pytest.approx(0.0)
    assert float(active_sequences) == pytest.approx(0.0)
    assert float(fallback_applied) == pytest.approx(0.0)
    assert float(loss) == pytest.approx(0.0)




def test_should_sync_rollout_policy_respects_interval_and_forced_sync():
    assert should_sync_rollout_policy(
        step=0,
        last_synced_step=None,
        update_interval=4,
        force_sync=False,
    )
    assert not should_sync_rollout_policy(
        step=2,
        last_synced_step=0,
        update_interval=4,
        force_sync=False,
    )
    assert should_sync_rollout_policy(
        step=4,
        last_synced_step=0,
        update_interval=4,
        force_sync=False,
    )
    assert should_sync_rollout_policy(
        step=3,
        last_synced_step=0,
        update_interval=4,
        force_sync=True,
    )


def test_compose_truncation_bootstrap_return_preserves_shaping_reward():
    target = compose_truncation_bootstrap_return(
        trajectory_reward=0.55,
        trajectory_raw_reward=0.0,
        bootstrap_value=0.35,
        outcome_reward_weight=1.0,
    )
    assert target == pytest.approx(0.90)

    weighted_target = compose_truncation_bootstrap_return(
        trajectory_reward=0.65,
        trajectory_raw_reward=1.0,
        bootstrap_value=0.40,
        outcome_reward_weight=0.5,
    )
    assert weighted_target == pytest.approx(0.35)


def test_packed_mis_ppo_loss_accepts_float32_behavior_logprobs_with_bfloat_model_outputs():
    target_log_probs = torch.tensor([-0.2, -0.3], dtype=torch.float32)
    token_ratios = torch.tensor([1.0, 1.0], dtype=torch.float32)
    old_infer_logprobs = target_log_probs - torch.log(token_ratios)

    (
        loss,
        _,
        _,
        _,
        active_token_rate,
        _,
        _,
        active_tokens,
        active_sequences,
        valid_count,
        _,
    ) = packed_mis_ppo_loss(
        _FakeTokenLogprobModel(target_log_probs.tolist(), dtype=torch.bfloat16),
        inputs=torch.tensor([[5, 6]], dtype=torch.long),
        targets=torch.tensor([[0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        seq_indices=torch.tensor([0, 0], dtype=torch.long),
        prompt_lens=torch.tensor([1], dtype=torch.long),
        token_advantages=torch.ones(2, dtype=torch.float32),
        old_infer_logprobs=old_infer_logprobs.to(dtype=torch.float32),
        pad_id=99,
        token_ratio_min=0.95,
        token_ratio_max=1.05,
        trajectory_ratio_min=0.95,
        trajectory_ratio_max=1.05,
        fallback_to_valid=False,
    )

    assert float(valid_count) == pytest.approx(2.0)
    assert float(active_token_rate) == pytest.approx(1.0)
    assert float(active_tokens) == pytest.approx(2.0)
    assert float(active_sequences) == pytest.approx(1.0)
    assert float(loss) > 0.0


def test_packed_mis_ppo_actor_critic_loss_adds_value_loss_and_value_metrics():
    target_log_probs = torch.tensor([-0.2, -0.3], dtype=torch.float32)
    old_infer_logprobs = target_log_probs.clone()

    (
        total_loss,
        actor_loss,
        value_loss,
        _entropy,
        _token_retention_rate,
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
    ) = packed_mis_ppo_actor_critic_loss(
        _FakeActorCriticTokenModel(target_log_probs.tolist(), [0.2, 0.4]),
        inputs=torch.tensor([[5, 6]], dtype=torch.long),
        targets=torch.tensor([[0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        seq_indices=torch.tensor([0, 0], dtype=torch.long),
        prompt_lens=torch.tensor([1], dtype=torch.long),
        sequence_returns=torch.tensor([1.0], dtype=torch.float32),
        old_infer_logprobs=old_infer_logprobs,
        pad_id=99,
        token_ratio_min=0.95,
        token_ratio_max=1.05,
        trajectory_ratio_min=0.95,
        trajectory_ratio_max=1.05,
        fallback_to_valid=False,
        value_loss_coef=1.0,
        kl_coef=0.0,
        entropy_coef=0.0,
        normalize_advantages=False,
        adv_norm_eps=1e-6,
    )

    assert float(valid_count) == pytest.approx(2.0)
    assert float(trajectory_keep_rate) == pytest.approx(1.0)
    assert float(active_token_rate) == pytest.approx(1.0)
    assert float(mean_token_ratio) == pytest.approx(1.0)
    assert float(mean_trajectory_ratio) == pytest.approx(1.0)
    assert float(active_tokens) == pytest.approx(2.0)
    assert float(active_sequences) == pytest.approx(1.0)
    assert float(fallback_applied) == pytest.approx(0.0)
    assert float(approx_kl) == pytest.approx(0.0)
    assert float(actor_loss) == pytest.approx(0.17)
    assert float(value_loss) == pytest.approx(0.25)
    assert float(total_loss) == pytest.approx(0.42)
    assert float(mean_value) == pytest.approx(0.3)
    assert float(mean_return) == pytest.approx(1.0)


def test_packed_mis_ppo_actor_critic_loss_supports_kl_and_entropy_regularization():
    target_log_probs = torch.tensor([-0.2, -0.3], dtype=torch.float32)
    token_ratios = torch.tensor([1.10, 0.90], dtype=torch.float32)
    old_infer_logprobs = target_log_probs - torch.log(token_ratios)

    (
        total_loss,
        actor_loss,
        value_loss,
        entropy,
        _token_retention_rate,
        _trajectory_keep_rate,
        _active_token_rate,
        _mean_token_ratio,
        _mean_trajectory_ratio,
        _active_tokens,
        _active_sequences,
        _valid_count,
        _fallback_applied,
        approx_kl,
        _mean_value,
        _mean_return,
    ) = packed_mis_ppo_actor_critic_loss(
        _FakeActorCriticTokenModel(target_log_probs.tolist(), [0.2, 0.4]),
        inputs=torch.tensor([[5, 6]], dtype=torch.long),
        targets=torch.tensor([[0, 0]], dtype=torch.long),
        position_ids=torch.tensor([[0, 1]], dtype=torch.long),
        seq_indices=torch.tensor([0, 0], dtype=torch.long),
        prompt_lens=torch.tensor([1], dtype=torch.long),
        sequence_returns=torch.tensor([1.0], dtype=torch.float32),
        old_infer_logprobs=old_infer_logprobs,
        pad_id=99,
        token_ratio_min=0.5,
        token_ratio_max=1.5,
        trajectory_ratio_min=0.5,
        trajectory_ratio_max=1.5,
        fallback_to_valid=False,
        value_loss_coef=1.0,
        kl_coef=0.2,
        entropy_coef=0.1,
        normalize_advantages=False,
        adv_norm_eps=1e-6,
    )

    expected_base_loss = float(actor_loss) + float(value_loss)
    expected_total_loss = expected_base_loss + (0.2 * float(approx_kl)) - (0.1 * float(entropy))

    assert float(approx_kl) > 0.0
    assert float(entropy) > 0.0
    assert float(total_loss) == pytest.approx(expected_total_loss)
