from __future__ import annotations

import pytest

from climbing_gym.data import (
    advance_data_epoch,
    create_data_state,
    DatasetCursor,
    next_data_batch,
    select_rows,
)


def test_create_data_state_is_deterministic_for_seed() -> None:
    state_a = create_data_state(6, seed=7, shuffle=True)
    state_b = create_data_state(6, seed=7, shuffle=True)
    state_c = create_data_state(6, seed=8, shuffle=True)

    assert state_a.order == state_b.order
    assert state_c.order == create_data_state(6, seed=8, shuffle=True).order


def test_next_data_batch_walks_rows_without_repeats_until_exhausted() -> None:
    rows = [{"value": value} for value in range(5)]
    state = create_data_state(len(rows), shuffle=False)

    batch_1, state = next_data_batch(rows, state, 2)
    batch_2, state = next_data_batch(rows, state, 2)
    batch_3, state = next_data_batch(rows, state, 2)
    batch_4, state = next_data_batch(rows, state, 2)

    assert batch_1.indices == (0, 1)
    assert batch_2.indices == (2, 3)
    assert batch_3.indices == (4,)
    assert batch_4.is_empty is True
    assert state.exhausted is True


def test_advance_data_epoch_resets_cursor_and_rebuilds_order() -> None:
    rows = [{"value": value} for value in range(6)]
    state = create_data_state(len(rows), seed=3, shuffle=True)
    _, state = next_data_batch(rows, state, 4)

    next_state = advance_data_epoch(state)
    expected_state = create_data_state(len(rows), seed=3, shuffle=True, epoch=1)

    assert next_state.epoch == 1
    assert next_state.cursor == 0
    assert next_state.order == expected_state.order


def test_select_rows_returns_copied_rows_in_requested_order() -> None:
    rows = [{"value": "a"}, {"value": "b"}, {"value": "c"}]

    batch = select_rows(rows, [2, 0])

    assert batch.indices == (2, 0)
    assert [row["value"] for row in batch.rows] == ["c", "a"]
    assert batch.rows[0] is not rows[2]


def test_next_data_batch_rejects_row_count_mismatch() -> None:
    state = create_data_state(3, shuffle=False)

    with pytest.raises(ValueError, match="rows length must match state size"):
        next_data_batch([{"value": 1}], state, 1)


def test_dataset_cursor_cycles_across_epochs() -> None:
    cursor = DatasetCursor(
        [{"value": 0}, {"value": 1}, {"value": 2}],
        shuffle=False,
    )

    batch_1 = cursor.next_batch(2)
    batch_2 = cursor.next_batch(2)
    batch_3 = cursor.next_batch(2)

    assert batch_1.indices == (0, 1)
    assert batch_2.indices == (2,)
    assert batch_3.indices == (0, 1)
