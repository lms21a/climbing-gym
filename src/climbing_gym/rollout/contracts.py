from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, runtime_checkable

from climbing_gym.envs.contracts import EnvEvaluation, EnvExample
from climbing_gym.inference.types import BehaviorTopK


@dataclass(frozen=True, slots=True)
class HarnessRequest:
    prompt: Any
    sampling: Any | None = None


@dataclass(frozen=True, slots=True)
class TrajectoryTurn:
    turn_index: int
    prompt: str
    response: str
    prompt_token_ids: tuple[int, ...]
    completion_token_ids: tuple[int, ...]
    completion_logprobs: tuple[float, ...] | None
    finish_reason: str | None
    stop_reason: str | None
    metadata: Mapping[str, Any] = field(default_factory=dict)
    behavior_topk: BehaviorTopK | None = None


@dataclass(frozen=True, slots=True)
class Trajectory:
    group_id: int
    rollout_index: int
    example: EnvExample
    turns: tuple[TrajectoryTurn, ...]
    final_response: str
    finished: bool
    terminal_reason: str
    evaluation: EnvEvaluation
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(slots=True)
class HarnessState:
    rollout_index: int
    turns: list[TrajectoryTurn] = field(default_factory=list)
    finished: bool = False
    final_response: str = ""
    terminal_reason: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


def build_trajectory_turn(
    *,
    turn_index: int,
    prompt: str,
    response: str,
    result: Any,
    metadata: Mapping[str, Any],
) -> TrajectoryTurn:
    output = result.outputs[0] if getattr(result, "outputs", ()) else None
    return TrajectoryTurn(
        turn_index=turn_index,
        prompt=str(prompt),
        response=str(response),
        prompt_token_ids=tuple(getattr(result, "prompt_token_ids", ()) or ()),
        completion_token_ids=tuple(getattr(output, "token_ids", ()) or ()),
        completion_logprobs=(
            None
            if getattr(output, "completion_logprobs", None) is None
            else tuple(getattr(output, "completion_logprobs", ()) or ())
        ),
        finish_reason=(None if output is None else output.finish_reason),
        stop_reason=(None if output is None else output.stop_reason),
        metadata=dict(metadata),
        behavior_topk=getattr(output, 'behavior_topk', None),
    )


@runtime_checkable
class Harness(Protocol):
    def create_state(self, example: EnvExample, rollout_index: int) -> HarnessState:
        ...

    def next_request(self, example: EnvExample, state: HarnessState) -> HarnessRequest | None:
        ...

    def apply_result(self, example: EnvExample, state: HarnessState, result: Any) -> None:
        ...

    def force_finish(self, state: HarnessState, reason: str) -> None:
        ...
