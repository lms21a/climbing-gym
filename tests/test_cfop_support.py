from __future__ import annotations

from dataclasses import FrozenInstanceError

import pytest

pytest.importorskip("cfop_solver")

from climbing_gym.envs.cfop_support import (
    CFOPParseError,
    FACE_ORDER,
    FaceletStateError,
    format_facelet_state,
    normalize_facelet_state,
    parse_cfop_response,
    parse_cfop_response_strict,
)


SOLVED_STATE = "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB"


def test_facelet_state_normalization_and_prompt_formatting() -> None:
    assert normalize_facelet_state(f"\n{SOLVED_STATE}\n") == SOLVED_STATE
    assert format_facelet_state(SOLVED_STATE) == "\n".join(
        (
            "U: UUU/UUU/UUU",
            "R: RRR/RRR/RRR",
            "F: FFF/FFF/FFF",
            "D: DDD/DDD/DDD",
            "L: LLL/LLL/LLL",
            "B: BBB/BBB/BBB",
        )
    )
    assert FACE_ORDER == ("U", "R", "F", "D", "L", "B")


@pytest.mark.parametrize(
    ("state", "message"),
    [
        ("U" * 53, "exactly 54"),
        ("u" + SOLVED_STATE[1:], "uppercase URFDLB"),
        ("U" * 54, "exactly nine"),
        (SOLVED_STATE[:27] + " " + SOLVED_STATE[28:], "uppercase URFDLB"),
    ],
)
def test_facelet_state_validation_rejects_noncanonical_values(
    state: str,
    message: str,
) -> None:
    with pytest.raises(FaceletStateError, match=message):
        normalize_facelet_state(state)


def test_facelet_state_validation_rejects_non_strings() -> None:
    with pytest.raises(FaceletStateError, match="must be a string"):
        normalize_facelet_state(123)  # type: ignore[arg-type]


def test_parse_full_cfop_response_derives_moves_and_verifier_boundaries() -> None:
    response = "\n".join(
        (
            "SETUP: x",
            "CROSS: R U R'",
            "F2L FR: U R U' R'",
            "F2L FL,BL: U",
            "OLL: F R U R' U' F'",
            "PLL: M2 U M2 U2 M2 U M2",
            "END",
        )
    )

    parsed = parse_cfop_response(response)

    assert parsed.valid
    assert parsed.error is None
    assert parsed.error_code is None
    assert parsed.error_line is None
    assert parsed.moves == (
        "x R U R' U R U' R' U F R U R' U' F' M2 U M2 U2 M2 U M2"
    )
    assert parsed.move_count == 22
    assert parsed.move_tokens[:4] == ("x", "R", "U", "R'")
    assert parsed.phase_boundaries == (
        {"stage": "setup", "after_move": 1},
        {"stage": "cross", "after_move": 4},
        {"stage": "f2l_pair", "after_move": 8, "f2l_pairs": ["fr"]},
        {
            "stage": "f2l_pair",
            "after_move": 9,
            "f2l_pairs": ["fl", "bl"],
        },
        {"stage": "oll", "after_move": 15},
        {"stage": "pll", "after_move": 22},
    )
    assert parsed.canonical_text == response
    assert [phase.stage for phase in parsed.phases] == [
        "setup",
        "cross",
        "f2l_pair",
        "f2l_pair",
        "oll",
        "pll",
    ]


def test_parse_allows_no_setup_no_f2l_and_empty_move_lists() -> None:
    response = "CROSS:\nOLL:\nPLL:\nEND\n"

    parsed = parse_cfop_response(response)

    assert parsed.valid
    assert parsed.moves == ""
    assert parsed.move_tokens == ()
    assert parsed.move_count == 0
    assert parsed.phase_boundaries == (
        {"stage": "cross", "after_move": 0},
        {"stage": "oll", "after_move": 0},
        {"stage": "pll", "after_move": 0},
    )
    assert parsed.canonical_text == response.rstrip()


def test_parse_result_is_frozen_and_boundary_mappings_are_fresh() -> None:
    parsed = parse_cfop_response("CROSS:\nOLL:\nPLL:\nEND")
    with pytest.raises(FrozenInstanceError):
        parsed.valid = False  # type: ignore[misc]

    first = parsed.phase_boundaries
    first[0]["stage"] = "changed"
    assert parsed.phase_boundaries[0]["stage"] == "cross"


@pytest.mark.parametrize(
    ("response", "code", "line"),
    [
        ("", "empty_response", None),
        ("cross:\nOLL:\nPLL:\nEND", "expected_cross", 1),
        ("CROSS:\n\nOLL:\nPLL:\nEND", "blank_line", 2),
        ("CROSS:  R\nOLL:\nPLL:\nEND", "invalid_whitespace", 1),
        ("CROSS: R3\nOLL:\nPLL:\nEND", "invalid_move", 1),
        ("CROSS: R’\nOLL:\nPLL:\nEND", "non_ascii", 1),
        ("CROSS:\nSETUP: x\nOLL:\nPLL:\nEND", "expected_oll", 2),
        ("CROSS:\nF2L RF: R\nOLL:\nPLL:\nEND", "invalid_f2l_pair", 2),
        ("CROSS:\nF2L FR, FR: R\nOLL:\nPLL:\nEND", "invalid_f2l_pair", 2),
        (
            "CROSS:\nF2L FR: R\nF2L FR: U\nOLL:\nPLL:\nEND",
            "duplicate_f2l_pair",
            3,
        ),
        ("CROSS:\nPLL:\nOLL:\nEND", "expected_oll", 2),
        ("CROSS:\nOLL:\nPLL:", "expected_end", 4),
        ("CROSS:\nOLL:\nPLL:\nEND\nprose", "trailing_content", 5),
        ("```text\nCROSS:\nOLL:\nPLL:\nEND\n```", "expected_cross", 1),
    ],
)
def test_parse_rejects_noncanonical_responses_without_raising(
    response: str,
    code: str,
    line: int | None,
) -> None:
    parsed = parse_cfop_response(response)

    assert not parsed.valid
    assert parsed.error
    assert parsed.error_code == code
    assert parsed.error_line == line
    assert parsed.moves == ""
    assert parsed.phase_boundaries == ()
    assert parsed.canonical_text == ""


def test_strict_parser_raises_classified_error() -> None:
    with pytest.raises(CFOPParseError) as error:
        parse_cfop_response_strict("CROSS:\nOLL: nope\nPLL:\nEND")

    assert error.value.code == "invalid_move"
    assert error.value.line == 2
    assert str(error.value) == "line 2: invalid move token 'nope'"
