from __future__ import annotations

from dataclasses import dataclass, field
from collections.abc import Mapping as MappingABC, Sequence
import math
from numbers import Real
from pathlib import Path
from typing import Any, Mapping

DEFAULT_STAGED_CHECKPOINT_ROOT = Path.home() / ".cache" / "climbing-gym" / "checkpoints"


def _finite_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool) and math.isfinite(value)


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
        if not _finite_real(self.temperature) or self.temperature < 0:
            raise ValueError("temperature must be finite and >= 0")
        if type(self.top_k) is not int or self.top_k < -1 or self.top_k == 0:
            raise ValueError("top_k must be -1 or >= 1")
        if not _finite_real(self.top_p) or not 0.0 < self.top_p <= 1.0:
            raise ValueError("top_p must be finite and in (0, 1]")
        if type(self.max_tokens) is not int or self.max_tokens <= 0:
            raise ValueError("max_tokens must be a positive integer")
        if type(self.n) is not int or self.n <= 0:
            raise ValueError("n must be a positive integer")
        if self.logprobs is not None and (type(self.logprobs) is not int or self.logprobs <= 0):
            raise ValueError("logprobs must be a positive integer when provided")
        if (not isinstance(self.stop, Sequence) or isinstance(self.stop, (str, bytes))
                or any(not isinstance(item, str) or not item for item in self.stop)):
            raise ValueError("stop must be a sequence of nonempty strings")
        object.__setattr__(self, "stop", tuple(self.stop))
        if not isinstance(self.extra_params, MappingABC):
            raise ValueError("extra_params must be a mapping")
        unknown = set(self.extra_params) - {"repetition_penalty", "ignore_eos", "behavior_topk"}
        if unknown:
            raise ValueError(f"Unsupported native sampling extra_params: {sorted(map(str, unknown))}")
        penalty = self.extra_params.get("repetition_penalty", 1.0)
        if not _finite_real(penalty) or penalty <= 0:
            raise ValueError("repetition_penalty must be finite and > 0")
        if type(self.extra_params.get("ignore_eos", False)) is not bool:
            raise ValueError("ignore_eos must be a boolean")
        head = self.extra_params.get("behavior_topk", 0)
        if type(head) is not int or not 0 <= head <= 128:
            raise ValueError("behavior_topk must be an integer between 0 and 128")



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
