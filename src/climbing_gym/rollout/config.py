from __future__ import annotations

from dataclasses import dataclass

from climbing_gym.inference.config import SamplingConfig


@dataclass(frozen=True, slots=True)
class RolloutConfig:
    """Logical rollout request settings."""
    num_rollouts: int
    sampling: SamplingConfig | object | None = None
    max_total_requests: int | None = None

    def __post_init__(self) -> None:
        if self.num_rollouts <= 0:
            raise ValueError("num_rollouts must be > 0")
        if self.max_total_requests is not None and self.max_total_requests <= 0:
            raise ValueError("max_total_requests must be > 0 when provided")


@dataclass(frozen=True, slots=True)
class RolloutRunnerConfig:
    """Execution settings for the async rollout runner."""
    effective_batch_size: int = 1
    max_inflight_requests: int | None = None
    reward_std_epsilon: float = 1e-8
    use_dynamic_sampling: bool = True
    stall_timeout_seconds: float = 120.0
    progress_log_interval_seconds: float = 15.0

    def __post_init__(self) -> None:
        if self.effective_batch_size <= 0:
            raise ValueError("effective_batch_size must be > 0")
        if self.max_inflight_requests is not None and self.max_inflight_requests <= 0:
            raise ValueError("max_inflight_requests must be > 0 when provided")
        if self.reward_std_epsilon < 0:
            raise ValueError("reward_std_epsilon must be >= 0")
        if self.stall_timeout_seconds < 0:
            raise ValueError("stall_timeout_seconds must be >= 0")
        if self.progress_log_interval_seconds < 0:
            raise ValueError("progress_log_interval_seconds must be >= 0")
