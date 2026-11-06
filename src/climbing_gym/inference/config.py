from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

DEFAULT_STAGED_CHECKPOINT_ROOT = Path.home() / ".cache" / "climbing-gym" / "checkpoints"


@dataclass(frozen=True, slots=True)
class SamplingConfig:
    """Sampling controls for native rollouts."""

    temperature: float = 0.0
    top_k: int = -1
    top_p: float = 1.0
    max_tokens: int = 256
    n: int = 1
    stop: tuple[str, ...] = ()
    logprobs: int | None = None
    extra_params: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.temperature < 0:
            raise ValueError("temperature must be >= 0")
        if self.top_k < -1 or self.top_k == 0:
            raise ValueError("top_k must be -1 or >= 1")
        if not 0.0 <= self.top_p <= 1.0:
            raise ValueError("top_p must be between 0 and 1")
        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0")
        if self.n <= 0:
            raise ValueError("n must be > 0")
        if self.logprobs is not None and self.logprobs <= 0:
            raise ValueError("logprobs must be > 0 when provided")



@dataclass(frozen=True, slots=True)
class InferenceEngineConfig:
    """Configuration for the local, single-GPU async inference engine."""

    model_path: str
    dtype: str = "auto"
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool = False
    enable_sleep_mode: bool = True
    apply_chat_template: bool = False
    chat_template_enable_thinking: bool = True
    assistant_prefill: str = ""
    default_sampling: SamplingConfig = field(default_factory=SamplingConfig)
    staged_checkpoint_root: Path = field(default_factory=lambda: DEFAULT_STAGED_CHECKPOINT_ROOT)

    def __post_init__(self) -> None:
        if not 0.0 < self.gpu_memory_utilization <= 1.0:
            raise ValueError("gpu_memory_utilization must be between 0 and 1")
        if self.max_model_len is not None and self.max_model_len <= 0:
            raise ValueError("max_model_len must be > 0 when provided")
        if self.max_num_seqs is not None and self.max_num_seqs <= 0:
            raise ValueError("max_num_seqs must be > 0 when provided")
