"""Inference request/response contracts for the native engines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch


PromptInput = str | Sequence[int] | Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class BehaviorTopK:
    """Unfiltered sampler head, aligned with generated tokens; normally CPU tensors."""

    token_ids: torch.Tensor
    logprobs: torch.Tensor

    def __post_init__(self):
        if self.token_ids.ndim != 2 or self.token_ids.shape != self.logprobs.shape:
            raise ValueError('Behavior head IDs/logprobs must be aligned [tokens, k] tensors')
        if self.token_ids.dtype not in (torch.int32, torch.int64) or self.logprobs.dtype != torch.float32:
            raise ValueError('Behavior head requires integer IDs and FP32 logprobs')


@dataclass(frozen=True, slots=True)
class GenerationOutput:
    index: int
    text: str
    token_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...] | None
    finish_reason: str | None
    stop_reason: str | None
    behavior_topk: BehaviorTopK | None = None


@dataclass(frozen=True, slots=True)
class GenerationResult:
    request_id: str
    prompt: PromptInput
    prompt_token_ids: tuple[int, ...]
    outputs: tuple[GenerationOutput, ...]
    finished: bool
    raw_output: Any

    @property
    def text(self) -> str:
        if not self.outputs:
            return ""
        return self.outputs[0].text

    @property
    def token_ids(self) -> tuple[int, ...]:
        if not self.outputs:
            return ()
        return self.outputs[0].token_ids

    @property
    def completion_logprobs(self) -> tuple[float, ...] | None:
        if not self.outputs:
            return None
        return self.outputs[0].completion_logprobs


__all__ = ["BehaviorTopK", "GenerationOutput", "GenerationResult", "PromptInput"]
