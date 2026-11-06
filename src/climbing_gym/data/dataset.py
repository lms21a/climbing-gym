from __future__ import annotations

from dataclasses import dataclass
import random
import hashlib
import json
from typing import Any, Mapping, Sequence


def _build_order(num_rows: int, *, seed: int, shuffle: bool, epoch: int) -> tuple[int, ...]:
    if num_rows < 0:
        raise ValueError("num_rows must be >= 0")
    order = list(range(num_rows))
    if shuffle and num_rows > 1:
        random.Random(seed + epoch).shuffle(order)
    return tuple(order)


@dataclass(frozen=True, slots=True)
class DataState:
    order: tuple[int, ...]
    cursor: int = 0
    epoch: int = 0
    seed: int = 0
    shuffle: bool = True

    def __post_init__(self) -> None:
        if self.cursor < 0 or self.cursor > len(self.order):
            raise ValueError("cursor must be between 0 and len(order)")
        if self.epoch < 0:
            raise ValueError("epoch must be >= 0")

    @property
    def size(self) -> int:
        return len(self.order)

    @property
    def remaining(self) -> int:
        return max(0, len(self.order) - self.cursor)

    @property
    def exhausted(self) -> bool:
        return self.cursor >= len(self.order)


@dataclass(frozen=True, slots=True)
class DataBatch:
    indices: tuple[int, ...]
    rows: tuple[Mapping[str, Any], ...]

    def __len__(self) -> int:
        return len(self.indices)

    @property
    def is_empty(self) -> bool:
        return not self.indices


@dataclass(slots=True)
class DatasetCursor:
    dataset: Sequence[Mapping[str, Any]]
    state: DataState

    def __init__(
        self,
        dataset: Sequence[Mapping[str, Any]],
        *,
        seed: int = 0,
        shuffle: bool = True,
        epoch: int = 0,
    ) -> None:
        self.dataset = dataset
        self.state = create_data_state(
            len(dataset),
            seed=seed,
            shuffle=shuffle,
            epoch=epoch,
        )

    @property
    def size(self) -> int:
        return int(self.state.size)

    @property
    def exhausted(self) -> bool:
        return bool(self.state.exhausted)

    def state_dict(self) -> dict[str, Any]:
        # Include row content and order: a cursor from a different bank is unsafe.
        fingerprint = hashlib.sha256(json.dumps(
            list(self.dataset), sort_keys=True, separators=(",", ":"),
        ).encode()).hexdigest()
        return {"dataset_sha256": fingerprint, "size": self.size,
                "cursor": self.state.cursor, "epoch": self.state.epoch,
                "seed": self.state.seed, "shuffle": self.state.shuffle}

    def load_state_dict(self, payload: Mapping[str, Any]) -> None:
        current = self.state_dict()
        if any(payload[key] != current[key] for key in ("dataset_sha256", "size", "seed", "shuffle")):
            raise ValueError("Dataset cursor checkpoint does not match dataset, seed, or shuffle")
        epoch = int(payload["epoch"])
        self.state = DataState(
            order=_build_order(self.size, seed=self.state.seed,
                               shuffle=self.state.shuffle, epoch=epoch),
            cursor=int(payload["cursor"]), epoch=epoch,
            seed=self.state.seed, shuffle=self.state.shuffle,
        )

    def advance_epoch(self) -> None:
        self.state = advance_data_epoch(self.state)

    def next_batch(self, batch_size: int) -> DataBatch:
        if self.size <= 0:
            return DataBatch(indices=(), rows=())

        batch, next_state = next_data_batch(self.dataset, self.state, batch_size)
        if batch.is_empty:
            self.state = advance_data_epoch(self.state)
            batch, next_state = next_data_batch(self.dataset, self.state, batch_size)
        self.state = next_state
        return batch


def create_data_state(
    num_rows: int,
    *,
    seed: int = 0,
    shuffle: bool = True,
    epoch: int = 0,
) -> DataState:
    return DataState(
        order=_build_order(num_rows, seed=seed, shuffle=shuffle, epoch=epoch),
        cursor=0,
        epoch=epoch,
        seed=seed,
        shuffle=shuffle,
    )


def advance_data_epoch(state: DataState) -> DataState:
    next_epoch = state.epoch + 1
    return DataState(
        order=_build_order(
            state.size,
            seed=state.seed,
            shuffle=state.shuffle,
            epoch=next_epoch,
        ),
        cursor=0,
        epoch=next_epoch,
        seed=state.seed,
        shuffle=state.shuffle,
    )


def select_rows(rows: Sequence[Mapping[str, Any]], indices: Sequence[int]) -> DataBatch:
    resolved_rows = tuple(dict(rows[int(index)]) for index in indices)
    return DataBatch(
        indices=tuple(int(index) for index in indices),
        rows=resolved_rows,
    )


def next_data_batch(
    rows: Sequence[Mapping[str, Any]],
    state: DataState,
    batch_size: int,
) -> tuple[DataBatch, DataState]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if len(rows) != state.size:
        raise ValueError("rows length must match state size")
    if state.exhausted:
        return DataBatch(indices=(), rows=()), state

    next_cursor = min(state.cursor + batch_size, state.size)
    indices = state.order[state.cursor:next_cursor]
    batch = select_rows(rows, indices)
    new_state = DataState(
        order=state.order,
        cursor=next_cursor,
        epoch=state.epoch,
        seed=state.seed,
        shuffle=state.shuffle,
    )
    return batch, new_state
