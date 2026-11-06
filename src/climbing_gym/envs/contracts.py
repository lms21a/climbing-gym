from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import Any, Callable, Mapping, Sequence

from climbing_gym.data import DatasetCursor


RewardFn = Callable[["RewardRequest"], "RewardResult | float | int"]


@dataclass(frozen=True, slots=True)
class EnvExample:
    prompt: str
    target: Any
    row: Mapping[str, Any] = field(default_factory=dict)
    dataset_index: int | None = None


@dataclass(frozen=True, slots=True)
class EnvBatch:
    indices: tuple[int, ...]
    rows: tuple[Mapping[str, Any], ...]
    examples: tuple[EnvExample, ...]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


@dataclass(frozen=True, slots=True)
class EnvFeedback:
    done: bool = True
    success: bool | None = None
    message: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardResult:
    reward: float
    info: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class RewardRequest:
    env_name: str
    example: EnvExample
    response: str
    parsed_response: Any
    feedback: EnvFeedback | None
    turn_count: int = 1
    turn_metadata: tuple[Mapping[str, Any], ...] = field(default_factory=tuple)


@dataclass(frozen=True, slots=True)
class EnvEvaluation:
    example: EnvExample
    response: str
    parsed_response: Any
    feedback: EnvFeedback | None
    reward: RewardResult


def _coerce_reward_result(result: RewardResult | float | int) -> RewardResult:
    if isinstance(result, RewardResult):
        return result
    return RewardResult(reward=float(result))


class BaseEnv:
    task_name: str = "task"
    prompt_key: str = "prompt"
    target_key: str = "target"
    reward_fn: RewardFn
    _example_queue: deque[EnvExample] | None = None
    _dataset_cursor: DatasetCursor | None = None

    def build_prompt(self, row: Mapping[str, Any]) -> str:
        return str(row[self.prompt_key])

    def build_target(self, row: Mapping[str, Any]) -> Any:
        return row[self.target_key]

    def build_example(
        self,
        row: Mapping[str, Any],
        *,
        dataset_index: int | None = None,
    ) -> EnvExample:
        return EnvExample(
            prompt=self.build_prompt(row),
            target=self.build_target(row),
            row=dict(row),
            dataset_index=(None if dataset_index is None else int(dataset_index)),
        )

    def build_examples(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        dataset_indices: Sequence[int] | None = None,
    ) -> list[EnvExample]:
        if dataset_indices is not None and len(dataset_indices) != len(rows):
            raise ValueError("dataset_indices length must match rows length")
        return [
            self.build_example(
                row,
                dataset_index=(
                    None
                    if dataset_indices is None
                    else int(dataset_indices[row_idx])
                ),
            )
            for row_idx, row in enumerate(rows)
        ]

    def load_rows(self, rows: Sequence[Mapping[str, Any]]) -> None:
        resolved_rows = tuple(dict(row) for row in rows)
        self.load_examples(self.build_examples(resolved_rows))

    def load_examples(self, examples: Sequence[EnvExample]) -> None:
        self._example_queue = deque(examples)

    def enqueue_examples(self, examples: Sequence[EnvExample]) -> None:
        if self._example_queue is None:
            self._example_queue = deque()
        self._example_queue.extend(examples)

    @property
    def pending_example_count(self) -> int:
        if self._example_queue is None:
            return 0
        return len(self._example_queue)

    def configure_dataset(
        self,
        rows: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        shuffle: bool = True,
        epoch: int = 0,
    ) -> None:
        self._dataset_cursor = DatasetCursor(
            rows,
            seed=seed,
            shuffle=shuffle,
            epoch=epoch,
        )

    def dataset_state_dict(self) -> dict[str, Any]:
        if self._dataset_cursor is None:
            raise RuntimeError("Dataset is not configured")
        pending = [example.dataset_index for example in (self._example_queue or ())]
        if any(index is None for index in pending):
            raise ValueError("Cannot checkpoint queued examples without dataset indices")
        return {"cursor": self._dataset_cursor.state_dict(), "pending_indices": pending}

    def load_dataset_state_dict(self, payload: Mapping[str, Any]) -> None:
        if self._dataset_cursor is None:
            raise RuntimeError("Dataset is not configured")
        pending = [int(index) for index in payload["pending_indices"]]
        if any(index < 0 or index >= self._dataset_cursor.size for index in pending):
            raise ValueError("Invalid pending dataset index")
        self._dataset_cursor.load_state_dict(payload["cursor"])
        self.load_examples(self.build_examples(
            [self._dataset_cursor.dataset[index] for index in pending],
            dataset_indices=pending,
        ))

    def next_batch(self, batch_size: int) -> EnvBatch:
        if self._dataset_cursor is None:
            raise RuntimeError("BaseEnv.next_batch() called before dataset rows were configured.")
        data_batch = self._dataset_cursor.next_batch(batch_size)
        rows = tuple(data_batch.rows)
        examples = tuple(
            self.build_examples(
                rows,
                dataset_indices=data_batch.indices,
            )
        )
        return EnvBatch(
            indices=tuple(int(index) for index in data_batch.indices),
            rows=rows,
            examples=examples,
        )

    def next_example(self) -> EnvExample | None:
        if self._example_queue is None:
            raise RuntimeError("BaseEnv.next_example() called before rows were loaded.")
        if not self._example_queue:
            return None
        return self._example_queue.popleft()

    def parse_response(self, response: str) -> Any:
        return str(response)

    def build_feedback(
        self,
        example: EnvExample,
        response: str,
        *,
        parsed_response: Any | None = None,
    ) -> EnvFeedback | None:
        del example, response, parsed_response
        return None

    def evaluate_response(
        self,
        example: EnvExample,
        response: str,
        *,
        turn_count: int = 1,
        turn_metadata: Sequence[Mapping[str, Any]] | None = None,
    ) -> EnvEvaluation:
        parsed_response = self.parse_response(response)
        feedback = self.build_feedback(example, response, parsed_response=parsed_response)
        reward = _coerce_reward_result(
            self.reward_fn(
                RewardRequest(
                    env_name=self.task_name,
                    example=example,
                    response=str(response),
                    parsed_response=parsed_response,
                    feedback=feedback,
                    turn_count=max(1, int(turn_count)),
                    turn_metadata=tuple(dict(metadata) for metadata in (turn_metadata or ())),
                )
            )
        )
        return EnvEvaluation(
            example=example,
            response=str(response),
            parsed_response=parsed_response,
            feedback=feedback,
            reward=reward,
        )
