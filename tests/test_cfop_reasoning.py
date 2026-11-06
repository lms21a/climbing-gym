from __future__ import annotations

import pytest


cfop_solver = pytest.importorskip("cfop_solver")

from climbing_gym.envs import (  # noqa: E402
    CFOPCubeEnv,
    CFOPPhaseEnv,
    build_cfop_phase_reasoning_prompt,
    build_cfop_reasoning_prompt,
    cfop_dense_v2_reward,
    cfop_phase_v1_reward,
)
from climbing_gym.envs.cfop_reasoning import (  # noqa: E402
    CFOP_REASONING_HEADER,
    parse_cfop_phase_reasoning_response,
    parse_cfop_reasoning_response,
)


HARD_STATE = "UUFUUFRRFLLLRRDRRDBBUDFFLFFBDDBDDBLLRRDLLLUUUDFFBBUBBR"
HARD_MOVES = (
    "L B' R' U R L' U2 L U' B U B' U2 B U' B' "
    "U2 Rw' U' R U' R' U2 Rw U' R2 U R U R' U' R' U' R' U R' U"
)
HARD_ANSWER = "\n".join(
    (
        "CROSS[FR,BL]: L B'",
        "F2L FL: R' U R L' U2 L",
        "F2L BR: U' B U B' U2 B U' B'",
        "OLL: U2 Rw' U' R U' R' U2 Rw",
        "PLL: U' R2 U R U R' U' R' U' R' U R' U",
        "END",
    )
)
HARD_REASONING = "\n".join(
    (
        "<think>",
        CFOP_REASONING_HEADER,
        (
            "STEP 1 | stage=CROSS[FR,BL] | from=none | to=xxcross "
            "| moves=2 | gain=BL | preserve=FR | break=-"
        ),
        (
            "STEP 2 | stage=F2L FL | from=xxcross | to=xxxcross "
            "| moves=6 | gain=FL | preserve=FR,BL | break=-"
        ),
        (
            "STEP 3 | stage=F2L BR | from=xxxcross | to=f2l "
            "| moves=8 | gain=BR | preserve=FR,FL,BL | break=-"
        ),
        (
            "STEP 4 | stage=OLL | from=f2l | to=oll "
            "| moves=8 | gain=- | preserve=FR,FL,BL,BR | break=-"
        ),
        (
            "STEP 5 | stage=PLL | from=oll | to=pll "
            "| moves=13 | gain=- | preserve=FR,FL,BL,BR | break=-"
        ),
        "</think>",
        HARD_ANSWER,
    )
)
HARD_CROSS_REASONING = "\n".join(
    (
        "<think>",
        CFOP_REASONING_HEADER,
        (
            "STEP 1 | stage=CROSS[FR,BL] | from=none | to=xxcross "
            "| moves=2 | gain=BL | preserve=FR | break=-"
        ),
        "</think>",
        "CROSS[FR,BL]: L B'",
        "END",
    )
)


@pytest.fixture(scope="module")
def reasoning_env() -> CFOPCubeEnv:
    return CFOPCubeEnv(
        reward_fn=cfop_dense_v2_reward,
        verifier_threads=1,
        response_format="reasoning_v1",
    )


@pytest.fixture(scope="module")
def cross_reasoning_env() -> CFOPPhaseEnv:
    return CFOPPhaseEnv(
        reward_fn=cfop_phase_v1_reward,
        verifier_threads=1,
        response_format="reasoning_v1",
        requested_stage="cross",
    )


def test_reasoning_prompt_is_state_only_and_declares_closed_envelope() -> None:
    prompt = build_cfop_reasoning_prompt(HARD_STATE)

    assert HARD_STATE not in prompt
    assert "STATE (facelets-urfdlb-v1):" in prompt
    assert "CFOP_PLAN_V1" in prompt
    assert "<think>\n" in prompt
    assert "\n</think>\n" in prompt
    assert "one STEP per answer phase" in prompt
    assert prompt.endswith("OUTPUT:\n")


def test_canonical_grounded_reasoning_solves_with_full_reward(
    reasoning_env: CFOPCubeEnv,
) -> None:
    parsed = parse_cfop_reasoning_response(HARD_REASONING)
    assert parsed.valid
    assert parsed.answer is not None
    assert parsed.answer.canonical_text == HARD_ANSWER
    assert parsed.canonical_text == HARD_REASONING
    assert len(parsed.steps) == 5

    example = reasoning_env.build_example(
        {
            "state": HARD_STATE,
            "completion": HARD_REASONING,
            "known_solution": HARD_MOVES,
        }
    )
    evaluation = reasoning_env.evaluate_response(example, HARD_REASONING)

    assert evaluation.feedback is not None
    assert evaluation.feedback.success is True
    assert evaluation.parsed_response.moves == HARD_MOVES
    assert evaluation.reward.reward == pytest.approx(1.8)
    assert evaluation.reward.info["reasoning_required"] == pytest.approx(1.0)
    assert evaluation.reward.info["reasoning_envelope_valid"] == pytest.approx(
        1.0
    )
    assert evaluation.reward.info["reasoning_grounded_valid"] == pytest.approx(
        1.0
    )
    assert evaluation.reward.info["reasoning_grounding_issues"] == ()


@pytest.mark.parametrize(
    ("response", "error_code"),
    (
        (HARD_ANSWER, "missing_think_open"),
        (
            HARD_REASONING.replace("</think>", ""),
            "missing_or_duplicate_think_close",
        ),
        (
            HARD_REASONING.replace("<think>", "<think>\n<think>", 1),
            "duplicate_think_open",
        ),
        (
            HARD_REASONING.replace("| moves=2 |", "| moves=3 |", 1),
            "reasoning_move_count_mismatch",
        ),
    ),
)
def test_malformed_or_inconsistent_reasoning_is_not_executed(
    reasoning_env: CFOPCubeEnv,
    response: str,
    error_code: str,
) -> None:
    parsed = parse_cfop_reasoning_response(response)
    assert parsed.valid is False
    assert parsed.error_code == error_code

    example = reasoning_env.build_example(
        {
            "state": HARD_STATE,
            "known_solution": HARD_MOVES,
        }
    )
    evaluation = reasoning_env.evaluate_response(example, response)

    assert evaluation.feedback is not None
    assert evaluation.feedback.success is False
    assert evaluation.parsed_response.moves == ""
    assert evaluation.reward.info["format_valid"] == pytest.approx(0.0)
    assert evaluation.reward.info["strict_success"] == pytest.approx(0.0)
    assert evaluation.reward.info["reasoning_envelope_valid"] == pytest.approx(
        0.0
    )
    assert evaluation.reward.info["reasoning_grounded_valid"] == pytest.approx(
        0.0
    )


def test_valid_envelope_with_false_milestone_claim_is_rejected(
    reasoning_env: CFOPCubeEnv,
) -> None:
    ungrounded = HARD_REASONING.replace(
        "from=none | to=xxcross",
        "from=cross | to=xxcross",
        1,
    )
    parsed = parse_cfop_reasoning_response(ungrounded)
    assert parsed.valid

    example = reasoning_env.build_example(
        {
            "state": HARD_STATE,
            "known_solution": HARD_MOVES,
        }
    )
    evaluation = reasoning_env.evaluate_response(example, ungrounded)

    assert evaluation.feedback is not None
    assert evaluation.feedback.success is False
    assert evaluation.reward.info["format_valid"] == pytest.approx(1.0)
    assert evaluation.reward.info["solved"] == pytest.approx(1.0)
    assert evaluation.reward.info["reasoning_envelope_valid"] == pytest.approx(
        1.0
    )
    assert evaluation.reward.info["reasoning_grounded_valid"] == pytest.approx(
        0.0
    )
    assert evaluation.reward.info["strict_cfop_trace"] == pytest.approx(0.0)
    assert evaluation.reward.info["strict_success"] == pytest.approx(0.0)
    assert evaluation.reward.info["reasoning_grounding_issues"][0]["step"] == 1
    assert evaluation.reward.info["solved_reward"] == pytest.approx(0.0)


def test_grounded_cross_phase_reasoning_receives_strict_phase_success(
    cross_reasoning_env: CFOPPhaseEnv,
) -> None:
    parsed = parse_cfop_phase_reasoning_response(HARD_CROSS_REASONING)
    assert parsed.valid
    assert parsed.answer is not None
    assert parsed.answer.moves == "L B'"

    example = cross_reasoning_env.build_example(
        {
            "task_type": "phase",
            "stage": "cross",
            "state": HARD_STATE,
            "known_solution": "L B'",
            "stage_end_state": cfop_solver.apply_moves(HARD_STATE, "L B'"),
            "completion": HARD_CROSS_REASONING,
        }
    )
    evaluation = cross_reasoning_env.evaluate_response(
        example,
        HARD_CROSS_REASONING,
    )

    assert evaluation.feedback is not None
    assert evaluation.feedback.success is True
    assert evaluation.reward.reward == pytest.approx(1.0)
    assert evaluation.reward.info["phase_goal_success"] is True
    assert evaluation.reward.info["cross_solved"] is True
    assert evaluation.reward.info["cross_edges_solved"] == 4
    assert evaluation.reward.info["reasoning_envelope_valid"] is True
    assert evaluation.reward.info["reasoning_grounded_valid"] is True
    assert evaluation.reward.info["strict_success"] is True


@pytest.mark.parametrize(
    "response",
    (
        HARD_CROSS_REASONING.replace(
            "from=none | to=xxcross",
            "from=cross | to=xxcross",
        ),
        HARD_CROSS_REASONING.replace("CROSS[FR,BL]", "CROSS[FR]"),
        HARD_CROSS_REASONING.replace("</think>\n", ""),
    ),
)
def test_cross_phase_rejects_ungrounded_labels_or_unclosed_reasoning(
    cross_reasoning_env: CFOPPhaseEnv,
    response: str,
) -> None:
    example = cross_reasoning_env.build_example(
        {
            "task_type": "phase",
            "stage": "cross",
            "state": HARD_STATE,
            "known_solution": "L B'",
        }
    )
    evaluation = cross_reasoning_env.evaluate_response(example, response)

    assert evaluation.feedback is not None
    assert evaluation.feedback.success is False
    assert evaluation.reward.info["strict_success"] is False


def test_cross_reasoning_prompt_does_not_reveal_teacher_pair_set() -> None:
    prompt = build_cfop_phase_reasoning_prompt(HARD_STATE, "cross")

    assert "CROSS[FR,BL]" not in prompt
    assert "CFOP_PLAN_V1" in prompt
    assert prompt.endswith("OUTPUT:\n")
