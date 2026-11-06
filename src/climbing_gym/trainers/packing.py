from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import torch

from climbing_gym.rollout.contracts import Trajectory


def _completion_token_count(sequence: Sequence[int], prompt_len: int) -> int:
    return max(0, len(sequence) - int(prompt_len))


@dataclass(frozen=True, slots=True)
class Batch:
    """Packed training batch built from rollout trajectories."""

    inputs: torch.Tensor
    targets: torch.Tensor
    position_ids: torch.Tensor
    seq_indices: torch.Tensor
    prompt_lens: torch.Tensor
    sequences: tuple[tuple[int, ...], ...]
    trajectory_indices: tuple[int, ...]
    group_ids: tuple[int, ...]
    rollout_indices: tuple[int, ...]
    turn_indices: tuple[int, ...]
    completion_logprobs: tuple[tuple[float, ...] | None, ...]
    packing_efficiency: float

    def __len__(self) -> int:
        return len(self.sequences)

    @property
    def sequence_count(self) -> int:
        return len(self.sequences)

    @property
    def token_count(self) -> int:
        return int(self.inputs.numel())

    @property
    def completion_lens(self) -> tuple[int, ...]:
        prompt_lens = self.prompt_lens.tolist()
        return tuple(
            _completion_token_count(sequence, int(prompt_len))
            for sequence, prompt_len in zip(self.sequences, prompt_lens)
        )

    @property
    def valid_token_count(self) -> float:
        return float(self.valid_completion_mask().sum().item())

    def to(self, device: torch.device | str) -> "Batch":
        resolved_device = torch.device(device)
        return Batch(
            inputs=self.inputs.to(resolved_device),
            targets=self.targets.to(resolved_device),
            position_ids=self.position_ids.to(resolved_device),
            seq_indices=self.seq_indices.to(resolved_device),
            prompt_lens=self.prompt_lens.to(resolved_device),
            sequences=self.sequences,
            trajectory_indices=self.trajectory_indices,
            group_ids=self.group_ids,
            rollout_indices=self.rollout_indices,
            turn_indices=self.turn_indices,
            completion_logprobs=self.completion_logprobs,
            packing_efficiency=self.packing_efficiency,
        )

    def valid_completion_mask(self) -> torch.Tensor:
        return build_valid_completion_mask(self)

    def completion_value_tensor(
        self,
        values: Sequence[Any],
        *,
        fill_value: float = 0.0,
    ) -> torch.Tensor:
        return build_completion_value_tensor(self, values, fill_value=fill_value)

    def resolved_completion_logprobs(self) -> tuple[tuple[float, ...], ...]:
        resolved: list[tuple[float, ...]] = []
        prompt_lens = self.prompt_lens.tolist()

        for idx, (sequence, prompt_len, row) in enumerate(
            zip(self.sequences, prompt_lens, self.completion_logprobs)
        ):
            completion_len = _completion_token_count(sequence, int(prompt_len))
            if completion_len <= 0:
                resolved.append(())
                continue
            if row is None:
                raise ValueError(
                    "Missing completion_logprobs for packed sequence. "
                    f"sequence_index={idx} expected_completion_tokens={completion_len}. "
                    "Configure rollout sampling to request vLLM logprobs."
                )
            values = tuple(float(value) for value in row)
            if len(values) != completion_len:
                raise ValueError(
                    "completion_logprobs length must match completion token count. "
                    f"sequence_index={idx} expected={completion_len} got={len(values)}"
                )
            resolved.append(values)

        return tuple(resolved)


@dataclass(frozen=True, slots=True)
class _PackedSequence:
    sequence: tuple[int, ...]
    prompt_len: int
    trajectory_index: int
    group_id: int
    rollout_index: int
    turn_index: int
    completion_logprobs: tuple[float, ...] | None


def _pack_lengths_bfd(lengths: Sequence[int], max_tokens: int) -> tuple[list[list[int]], float]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be > 0")
    if not lengths:
        return [], 0.0
    if any(length <= 0 for length in lengths):
        raise ValueError("All packed sequence lengths must be > 0")

    indexed_lengths = sorted(enumerate(lengths), key=lambda item: -item[1])
    bins: list[list[int]] = []
    remaining_capacity: list[int] = []

    for seq_idx, seq_len in indexed_lengths:
        if seq_len > max_tokens:
            raise ValueError(
                "Encountered a sequence longer than max_tokens: "
                f"length={seq_len} max_tokens={max_tokens}"
            )

        best_bin_idx = -1
        best_remaining = max_tokens + 1
        for bin_idx, capacity in enumerate(remaining_capacity):
            remaining = capacity - seq_len
            if 0 <= remaining < best_remaining:
                best_bin_idx = bin_idx
                best_remaining = remaining

        if best_bin_idx >= 0:
            bins[best_bin_idx].append(seq_idx)
            remaining_capacity[best_bin_idx] -= seq_len
            continue

        bins.append([seq_idx])
        remaining_capacity.append(max_tokens - seq_len)

    total_tokens = sum(lengths)
    total_capacity = len(bins) * max_tokens
    efficiency = total_tokens / total_capacity if total_capacity > 0 else 0.0
    return bins, efficiency


def _flatten_trajectory_turns(
    trajectories: Sequence[Trajectory],
    *,
    prompt_token_overrides: Mapping[tuple[int, int], Sequence[int]] | None = None,
) -> list[_PackedSequence]:
    sequences: list[_PackedSequence] = []
    for trajectory_index, trajectory in enumerate(trajectories):
        for turn in trajectory.turns:
            key = (int(trajectory_index), int(turn.turn_index))
            prompt_token_ids = (
                turn.prompt_token_ids
                if prompt_token_overrides is None or key not in prompt_token_overrides
                else tuple(int(token_id) for token_id in prompt_token_overrides[key])
            )
            if not prompt_token_ids or not turn.completion_token_ids:
                continue
            sequence = tuple(prompt_token_ids) + tuple(turn.completion_token_ids)
            prompt_len = len(prompt_token_ids)
            if prompt_len <= 0 or len(sequence) <= 1:
                continue
            if _completion_token_count(sequence, prompt_len) <= 0:
                continue
            sequences.append(
                _PackedSequence(
                    sequence=sequence,
                    prompt_len=prompt_len,
                    trajectory_index=int(trajectory_index),
                    group_id=int(trajectory.group_id),
                    rollout_index=int(trajectory.rollout_index),
                    turn_index=int(turn.turn_index),
                    completion_logprobs=(
                        None if turn.completion_logprobs is None else tuple(turn.completion_logprobs)
                    ),
                )
            )
    return sequences


def _build_batch(
    sequences: Sequence[_PackedSequence],
    *,
    packing_efficiency: float,
) -> Batch:
    all_inputs: list[int] = []
    all_targets: list[int] = []
    all_positions: list[int] = []
    all_seq_indices: list[int] = []
    prompt_lens: list[int] = []

    for local_seq_idx, sample in enumerate(sequences):
        shifted_len = len(sample.sequence) - 1
        if shifted_len <= 0:
            continue
        all_inputs.extend(sample.sequence[:-1])
        all_targets.extend(sample.sequence[1:])
        all_positions.extend(range(shifted_len))
        all_seq_indices.extend([local_seq_idx] * shifted_len)
        prompt_lens.append(int(sample.prompt_len))

    return Batch(
        inputs=torch.tensor(all_inputs, dtype=torch.long).unsqueeze(0),
        targets=torch.tensor(all_targets, dtype=torch.long).unsqueeze(0),
        position_ids=torch.tensor(all_positions, dtype=torch.long).unsqueeze(0),
        seq_indices=torch.tensor(all_seq_indices, dtype=torch.long),
        prompt_lens=torch.tensor(prompt_lens, dtype=torch.long),
        sequences=tuple(sample.sequence for sample in sequences),
        trajectory_indices=tuple(sample.trajectory_index for sample in sequences),
        group_ids=tuple(sample.group_id for sample in sequences),
        rollout_indices=tuple(sample.rollout_index for sample in sequences),
        turn_indices=tuple(sample.turn_index for sample in sequences),
        completion_logprobs=tuple(sample.completion_logprobs for sample in sequences),
        packing_efficiency=float(packing_efficiency),
    )


def pack_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    max_tokens: int,
    prompt_token_overrides: Mapping[tuple[int, int], Sequence[int]] | None = None,
) -> list[Batch]:
    """Pack trajectory turns, optionally with critic-only prompt tokens."""
    flattened = _flatten_trajectory_turns(
        trajectories,
        prompt_token_overrides=prompt_token_overrides,
    )
    if not flattened:
        return []

    lengths = [len(sample.sequence) - 1 for sample in flattened]
    bins, efficiency = _pack_lengths_bfd(lengths, max_tokens)
    return [
        _build_batch([flattened[idx] for idx in bin_indices], packing_efficiency=efficiency)
        for bin_indices in bins
    ]


def pack_sequences(
    sequences: Sequence[Sequence[int]],
    *,
    prompt_lens: Sequence[int],
    max_tokens: int,
) -> list[Batch]:
    """Pack prompt/completion token sequences into training batches."""
    if len(sequences) != len(prompt_lens):
        raise ValueError(
            "prompt_lens length must match sequences length. "
            f"Got len(sequences)={len(sequences)} and len(prompt_lens)={len(prompt_lens)}."
        )

    flattened: list[_PackedSequence] = []
    for sequence_index, (sequence, prompt_len) in enumerate(zip(sequences, prompt_lens)):
        token_ids = tuple(int(token_id) for token_id in sequence)
        resolved_prompt_len = int(prompt_len)
        if resolved_prompt_len <= 0 or len(token_ids) <= 1:
            continue
        if _completion_token_count(token_ids, resolved_prompt_len) <= 0:
            continue
        flattened.append(
            _PackedSequence(
                sequence=token_ids,
                prompt_len=resolved_prompt_len,
                trajectory_index=sequence_index,
                group_id=0,
                rollout_index=0,
                turn_index=0,
                completion_logprobs=None,
            )
        )

    if not flattened:
        return []

    lengths = [len(sample.sequence) - 1 for sample in flattened]
    bins, efficiency = _pack_lengths_bfd(lengths, max_tokens)
    return [
        _build_batch([flattened[idx] for idx in bin_indices], packing_efficiency=efficiency)
        for bin_indices in bins
    ]


def build_valid_completion_mask(batch: Batch) -> torch.Tensor:
    token_prompt_lens = batch.prompt_lens[batch.seq_indices]
    token_positions = batch.position_ids.squeeze(0)
    return token_positions >= (token_prompt_lens - 1)


def _align_completion_values(source: Any, completion_len: int, *, fill_value: float) -> list[float]:
    if completion_len <= 0:
        return []
    if isinstance(source, torch.Tensor):
        values = [float(value) for value in source.detach().cpu().reshape(-1).tolist()]
    elif isinstance(source, Sequence) and not isinstance(source, (str, bytes)):
        values = [float(value) for value in source]
    else:
        return [float(source)] * completion_len
    if len(values) < completion_len:
        values.extend(float(fill_value) for _ in range(completion_len - len(values)))
    return values[:completion_len]


def build_completion_value_tensor(
    batch: Batch,
    values: Sequence[Any],
    *,
    fill_value: float = 0.0,
) -> torch.Tensor:
    """Project per-sequence completion values into the packed shifted-token layout."""
    if len(values) != len(batch):
        raise ValueError("values length must match batch.sequence_count")

    packed_values: list[float] = []
    prompt_lens = batch.prompt_lens.tolist()
    for sequence, prompt_len, source in zip(batch.sequences, prompt_lens, values):
        shifted_len = len(sequence) - 1
        completion_len = _completion_token_count(sequence, int(prompt_len))
        aligned = _align_completion_values(source, completion_len, fill_value=fill_value)
        token_values = [float(fill_value)] * shifted_len
        completion_start = max(0, int(prompt_len) - 1)
        for offset, value in enumerate(aligned):
            token_values[completion_start + offset] = float(value)
        packed_values.extend(token_values)

    return torch.tensor(packed_values, dtype=torch.float32)


def extract_completion_values(
    batch: Batch,
    values: Sequence[Any] | torch.Tensor,
) -> tuple[tuple[float, ...], ...]:
    """Extract per-sequence completion-region values from a packed token tensor."""
    if isinstance(values, torch.Tensor):
        flat_values = values.detach().cpu().reshape(-1)
    else:
        flat_values = torch.tensor([float(value) for value in values], dtype=torch.float32)

    expected = int(batch.seq_indices.numel())
    if int(flat_values.numel()) != expected:
        raise ValueError(
            "Packed values length must match the shifted packed token count. "
            f"expected={expected} got={int(flat_values.numel())}"
        )

    valid_mask = build_valid_completion_mask(batch).detach().cpu()
    seq_indices = batch.seq_indices.detach().cpu()
    extracted: list[tuple[float, ...]] = []
    for sequence_idx in range(batch.sequence_count):
        mask = (seq_indices == sequence_idx) & valid_mask
        extracted.append(tuple(float(value) for value in flat_values[mask].tolist()))
    return tuple(extracted)


__all__ = [
    "Batch",
    "build_completion_value_tensor",
    "build_valid_completion_mask",
    "extract_completion_values",
    "pack_trajectories",
]
