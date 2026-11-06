from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from typing import Any

from climbing_gym.envs.cfop_reasoning import (
    CFOP_PHASE_REASONING_PROTOCOL,
    CFOP_REASONING_PROTOCOL,
    ParsedCFOPReasoning,
    parse_cfop_phase_reasoning_response,
    parse_cfop_reasoning_response,
    reasoning_grounding_issues,
)
from climbing_gym.envs.cfop_support import (
    F2L_PAIR_ORDER,
    ParsedCFOPResponse,
    format_facelet_state,
    parse_cfop_phase_response,
    parse_cfop_response,
)
from climbing_gym.envs.contracts import (
    BaseEnv,
    EnvExample,
    EnvFeedback,
    RewardFn,
    RewardRequest,
    RewardResult,
)


CFOP_PROTOCOL_RULE_V1 = (
    "After the first physically solved atomic prefix, every later atomic prefix "
    "must remain physically solved. Whole-cube rotations may then canonicalize "
    "the solved frame, but the first canonically solved prefix must be the final "
    "atomic move. Solve-then-disturb traces and moves after canonical solve are "
    "invalid. "
    "Passing v1 establishes ordered milestone and phase-invariant validity; it "
    "does not prove that moves originated from a particular human algorithm. "
    "Boundary milestone jumps are diagnostic and are not failures in v1."
)
STRICT_CFOP_TRACE_RULE_V1 = CFOP_PROTOCOL_RULE_V1
CFOP_PROTOCOL_RULE_V2 = (
    f"{CFOP_PROTOCOL_RULE_V1} "
    "V2 additionally requires phase fidelity: CROSS and F2L phases may not "
    "newly advance the cube into OLL or PLL milestones, except for a single "
    "move that immediately solves the cube with all later phases empty. OLL "
    "may also be skipped when the final F2L phase first completes F2L on its "
    "final atomic move, that same endpoint is already OLL-solved, and the OLL "
    "move list is empty. "
    "OLL may advance directly to PLL when one uninterrupted last-layer sequence "
    "also solves permutation, but it may not reach an OLL-solved state at a "
    "proper internal prefix and then continue with hidden PLL work. PLL must "
    "finish the solve. This rejects packing downstream work into an earlier "
    "phase."
)
STRICT_CFOP_TRACE_RULE_V2 = CFOP_PROTOCOL_RULE_V2
CFOP_PHASE_PROTOCOL_RULE_V1 = (
    "A stage-local Cross response must begin in the canonical frame, contain "
    "exactly one CROSS phase, solve all four Cross edges at its endpoint, and "
    "declare exactly every F2L pair solved there. It may advance through F2L "
    "but may not hide OLL, PLL, or a full solve. A required reasoning envelope "
    "must be self-consistent and match verifier-derived before/after progress."
)

_FORMAT_REWARD = 0.05
_NOTATION_REWARD = 0.05
_PHASE_COMPONENT_REWARD = 0.10
_INVARIANT_REWARD = 0.10
_SOLVED_REWARD = 1.00
_MAX_EFFICIENCY_REWARD = 0.20
_EXPECTED_CFOP_SOLVER_VERSION = "0.4.0"
_EXPECTED_VERIFICATION_SCHEMA = "cfop-verification/v1"
_EXPECTED_STATE_FORMAT = "facelets-urfdlb-v1"
_MILESTONE_RANK = {
    "none": 0,
    "cross": 1,
    "xcross": 2,
    "xxcross": 3,
    "xxxcross": 4,
    "f2l": 5,
    "oll": 6,
    "pll": 7,
}


class CFOPCompatibilityError(RuntimeError):
    """Raised when the installed CFOPSolver contract is incompatible."""


def _validate_cfop_solver_module(module: Any) -> None:
    required_attributes = (
        "Cube",
        "SOLVED_STATE",
        "STATE_FORMAT",
        "TrainingVerifier",
        "VERIFICATION_SCHEMA",
        "__version__",
    )
    missing = [
        attribute
        for attribute in required_attributes
        if not hasattr(module, attribute)
    ]
    if missing:
        raise CFOPCompatibilityError(
            f"cfop_solver is missing required attributes: {missing!r}"
        )

    schema = module.VERIFICATION_SCHEMA
    if schema != _EXPECTED_VERIFICATION_SCHEMA:
        raise CFOPCompatibilityError(
            "cfop_solver.VERIFICATION_SCHEMA must be "
            f"{_EXPECTED_VERIFICATION_SCHEMA!r}, got {schema!r}"
        )
    state_format = module.STATE_FORMAT
    if state_format != _EXPECTED_STATE_FORMAT:
        raise CFOPCompatibilityError(
            "cfop_solver.STATE_FORMAT must be "
            f"{_EXPECTED_STATE_FORMAT!r}, got {state_format!r}"
        )
    version = str(module.__version__)
    if version != _EXPECTED_CFOP_SOLVER_VERSION:
        raise CFOPCompatibilityError(
            f"cfop_solver must be pinned to {_EXPECTED_CFOP_SOLVER_VERSION}, "
            f"got {version!r}"
        )


def _as_mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _as_move_string(value: Any) -> str:
    if isinstance(value, str):
        return " ".join(value.split())
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return " ".join(str(token).strip() for token in value if str(token).strip())
    return str(value or "").strip()


def _solution_text(value: Any) -> str:
    if isinstance(value, str):
        return value.strip()
    if isinstance(value, Mapping):
        for key in ("moves", "solution", "candidate"):
            text = _as_move_string(value.get(key))
            if text:
                return text
    return ""


def _append_unique(values: list[str], candidate: Any) -> None:
    text = _solution_text(candidate)
    if text and text not in values:
        values.append(text)


def _known_solutions_from_row(row: Mapping[str, Any]) -> tuple[str, ...]:
    known: list[str] = []

    known_solutions = row.get("known_solutions")
    if isinstance(known_solutions, Sequence) and not isinstance(
        known_solutions,
        (str, bytes, bytearray),
    ):
        for solution in known_solutions:
            _append_unique(known, solution)
    else:
        _append_unique(known, known_solutions)
    _append_unique(known, row.get("known_solution"))
    _append_unique(known, row.get("solution"))

    solutions = row.get("solutions")
    if isinstance(solutions, Sequence) and not isinstance(
        solutions,
        (str, bytes, bytearray),
    ):
        for solution in solutions:
            if isinstance(solution, Mapping) and solution.get("verified") is False:
                continue
            _append_unique(known, solution)

    completion = row.get("completion")
    if completion is not None:
        parsed_completion = parse_cfop_response(str(completion))
        if parsed_completion.valid:
            _append_unique(known, parsed_completion.moves)

    return tuple(known)


def _progress_component_values(progress: Mapping[str, Any]) -> tuple[float, float, float, float]:
    if not progress:
        return (0.0, 0.0, 0.0, 0.0)

    canonical_orientation = bool(progress.get("canonical_orientation", False))
    cross_edges = max(0, min(4, int(progress.get("cross_edges_solved", 0) or 0)))
    cross_component = (float(cross_edges) / 4.0) if canonical_orientation else 0.0

    solved_pairs = progress.get("solved_pairs")
    pair_count = (
        len(solved_pairs)
        if isinstance(solved_pairs, Sequence)
        and not isinstance(solved_pairs, (str, bytes, bytearray))
        else 0
    )
    f2l_component = (
        float(max(0, min(4, pair_count))) / 4.0
        if bool(progress.get("cross_solved", False))
        else 0.0
    )

    oriented = max(
        0,
        min(8, int(progress.get("last_layer_oriented_cubies", 0) or 0)),
    )
    oll_component = (
        float(oriented) / 8.0
        if bool(progress.get("f2l_solved", False))
        else 0.0
    )

    permuted = max(
        0,
        min(8, int(progress.get("last_layer_permuted_cubies", 0) or 0)),
    )
    pll_component = (
        float(permuted) / 8.0
        if bool(progress.get("oll_solved", False))
        else 0.0
    )
    return (
        cross_component,
        f2l_component,
        oll_component,
        pll_component,
    )


def _solved_efficiency_reward(verifier: Mapping[str, Any], *, eligible: bool) -> float:
    if not eligible:
        return 0.0
    known = _as_mapping(verifier.get("known_solutions"))
    best_move_count = known.get("best_move_count")
    move_count = verifier.get("move_count")
    if not isinstance(best_move_count, int) or not isinstance(move_count, int):
        return 0.0
    if move_count <= best_move_count:
        return _MAX_EFFICIENCY_REWARD
    if best_move_count <= 0 or move_count <= 0:
        return 0.0
    return _MAX_EFFICIENCY_REWARD * min(
        1.0,
        float(best_move_count) / float(move_count),
    )


def cfop_dense_v2_reward(request: RewardRequest) -> RewardResult:
    """Dense v2 reward with a maximum of 1.8 and no teacher-string matching."""

    feedback_metadata = (
        request.feedback.metadata
        if request.feedback is not None and isinstance(request.feedback.metadata, Mapping)
        else {}
    )
    verifier = _as_mapping(feedback_metadata.get("verifier"))
    verifier_parse = _as_mapping(verifier.get("parse"))
    progress = _as_mapping(verifier.get("progress"))

    format_valid = bool(feedback_metadata.get("format_valid", False))
    candidate_notation_nonempty = bool(str(request.response).strip())
    notation_valid = bool(
        verifier_parse.get("moves_valid", False)
        and (format_valid or candidate_notation_nonempty)
    )
    verifier_valid = bool(verifier_parse.get("valid", False))
    strict_cfop_trace = bool(feedback_metadata.get("strict_cfop_trace", False))
    phase_fidelity_valid = bool(
        feedback_metadata.get("phase_fidelity_valid", False)
    )
    premature_full_solve = bool(
        feedback_metadata.get("premature_full_solve", False)
    )
    solved = bool(verifier.get("solved", False))
    strict_success = bool(
        format_valid and verifier_valid and strict_cfop_trace and solved
    )

    format_reward = _FORMAT_REWARD if format_valid else 0.0
    notation_reward = _NOTATION_REWARD if notation_valid else 0.0

    (
        cross_progress,
        f2l_progress,
        oll_progress,
        pll_progress,
    ) = _progress_component_values(progress)
    progress_eligible = bool(
        format_valid
        and verifier_valid
        and phase_fidelity_valid
        and not premature_full_solve
        and (not solved or strict_cfop_trace)
    )
    if not progress_eligible:
        cross_progress = 0.0
        f2l_progress = 0.0
        oll_progress = 0.0
        pll_progress = 0.0
    progress_reward = _PHASE_COMPONENT_REWARD * (
        cross_progress + f2l_progress + oll_progress + pll_progress
    )

    invariant_reward = _INVARIANT_REWARD if strict_cfop_trace else 0.0
    solved_reward = _SOLVED_REWARD if strict_success else 0.0
    efficiency_reward = _solved_efficiency_reward(
        verifier,
        eligible=strict_success,
    )
    reward = (
        format_reward
        + notation_reward
        + progress_reward
        + invariant_reward
        + solved_reward
        + efficiency_reward
    )

    known_solutions = _as_mapping(verifier.get("known_solutions"))
    phase_invariants = _as_mapping(verifier.get("phase_invariants"))
    info: dict[str, Any] = {
        "format_valid": float(format_valid),
        "notation_valid": float(notation_valid),
        "candidate_notation_nonempty": float(candidate_notation_nonempty),
        "state_valid": float(bool(verifier_parse.get("state_valid", False))),
        "verifier_valid": float(verifier_valid),
        "solved": float(solved),
        "strict_success": float(strict_success),
        "strict_cfop_trace": float(strict_cfop_trace),
        "phase_fidelity_valid": float(phase_fidelity_valid),
        "phase_fidelity_issues": feedback_metadata.get(
            "phase_fidelity_issues",
            (),
        ),
        "phase_fidelity_exceptions": feedback_metadata.get(
            "phase_fidelity_exceptions",
            (),
        ),
        "protocol_valid": float(strict_cfop_trace),
        "premature_full_solve": float(premature_full_solve),
        "all_moves_in_opening": float(
            bool(feedback_metadata.get("all_moves_in_opening", False))
        ),
        "early_terminal_solve": float(
            bool(feedback_metadata.get("early_terminal_solve", False))
        ),
        "early_terminal_stage": feedback_metadata.get("early_terminal_stage"),
        "early_terminal_solve_offset": feedback_metadata.get(
            "early_terminal_solve_offset"
        ),
        "terminal_solve_at_boundary": float(
            bool(feedback_metadata.get("terminal_solve_at_boundary", False))
        ),
        "terminal_suffix_valid": float(
            bool(feedback_metadata.get("terminal_suffix_valid", False))
        ),
        "first_solved_offset": feedback_metadata.get("first_solved_offset"),
        "solve_then_disturb": float(
            bool(feedback_metadata.get("solve_then_disturb", False))
        ),
        "pre_cross_canonical": float(
            bool(feedback_metadata.get("pre_cross_canonical", False))
        ),
        "pre_oll_f2l_ready": float(
            bool(feedback_metadata.get("pre_oll_f2l_ready", False))
        ),
        "pair_declarations_valid": float(
            bool(feedback_metadata.get("pair_declarations_valid", False))
        ),
        "pair_declaration_issues": feedback_metadata.get(
            "pair_declaration_issues",
            (),
        ),
        "reasoning_required": float(
            bool(feedback_metadata.get("reasoning_required", False))
        ),
        "reasoning_envelope_valid": float(
            bool(feedback_metadata.get("reasoning_envelope_valid", False))
        ),
        "reasoning_grounded_valid": float(
            bool(feedback_metadata.get("reasoning_grounded_valid", False))
        ),
        "reasoning_grounding_issues": feedback_metadata.get(
            "reasoning_grounding_issues",
            (),
        ),
        "cross_progress": float(cross_progress),
        "f2l_progress": float(f2l_progress),
        "oll_progress": float(oll_progress),
        "pll_progress": float(pll_progress),
        "progress_eligible": float(progress_eligible),
        "invalid_solved_trace": float(solved and not strict_cfop_trace),
        "progress_reward": float(progress_reward),
        "format_reward": float(format_reward),
        "notation_reward": float(notation_reward),
        "invariant_reward": float(invariant_reward),
        "solved_reward": float(solved_reward),
        "efficiency_reward": float(efficiency_reward),
        "max_reward": 1.8,
        "move_count": verifier.get("move_count"),
        "best_known_move_count": known_solutions.get("best_move_count"),
        "excess_moves": known_solutions.get("excess_moves"),
        "phase_invariant_status": phase_invariants.get("status", "unavailable"),
        "parser_error": feedback_metadata.get("parser_error"),
        "strict_cfop_trace_rule": STRICT_CFOP_TRACE_RULE_V2,
        "protocol_rule": CFOP_PROTOCOL_RULE_V2,
        "phase_prefix_progress": feedback_metadata.get(
            "phase_prefix_progress",
            (),
        ),
        "oll_internal_prefix_progress": feedback_metadata.get(
            "oll_internal_prefix_progress",
            (),
        ),
        "final_f2l_internal_prefix_progress": feedback_metadata.get(
            "final_f2l_internal_prefix_progress",
            (),
        ),
        "phase_move_counts": feedback_metadata.get(
            "phase_move_counts",
            (),
        ),
        "boundary_milestone_jumps": feedback_metadata.get(
            "boundary_milestone_jumps",
            (),
        ),
        "pre_oll_solved_offsets": feedback_metadata.get(
            "pre_oll_solved_offsets",
            (),
        ),
        "solved_move_offsets": feedback_metadata.get(
            "solved_move_offsets",
            (),
        ),
        "physical_solved_offsets": feedback_metadata.get(
            "physical_solved_offsets",
            (),
        ),
        "canonical_solved_offsets": feedback_metadata.get(
            "canonical_solved_offsets",
            (),
        ),
        "first_physical_solved_offset": feedback_metadata.get(
            "first_physical_solved_offset"
        ),
        "first_canonical_solved_offset": feedback_metadata.get(
            "first_canonical_solved_offset"
        ),
        "solved_then_unsolved_offsets": feedback_metadata.get(
            "solved_then_unsolved_offsets",
            (),
        ),
        "post_canonical_move_offsets": feedback_metadata.get(
            "post_canonical_move_offsets",
            (),
        ),
        "verifier": dict(verifier),
    }
    return RewardResult(reward=float(reward), info=info)

def _build_training_verifier(threads: int) -> Any:
    try:
        import cfop_solver
    except ImportError as error:
        raise RuntimeError(
            "CFOPCubeEnv requires cfop-solver==0.4.0. Install the optional Rust bindings "
            "with uv as described in docs/cfop.md."
        ) from error
    _validate_cfop_solver_module(cfop_solver)
    return cfop_solver.TrainingVerifier(
        threads=max(0, int(threads)),
        features="none",
    )


def _sequential_solve_diagnostics(
    state: str,
    parsed: ParsedCFOPResponse,
) -> Mapping[str, Any]:

    try:
        import cfop_solver
    except ImportError as error:
        raise RuntimeError(
            "CFOPCubeEnv requires cfop-solver==0.4.0. Install the optional Rust bindings "
            "with uv as described in docs/cfop.md."
        ) from error
    _validate_cfop_solver_module(cfop_solver)
    cube = cfop_solver.Cube(state)
    physical_solved_offsets: list[int] = []
    canonical_solved_offsets: list[int] = []
    solved_then_unsolved_offsets: list[int] = []
    first_physical_solved_offset: int | None = None
    if cube.is_solved:
        physical_solved_offsets.append(0)
        first_physical_solved_offset = 0
    if cube.state == cfop_solver.SOLVED_STATE:
        canonical_solved_offsets.append(0)

    for offset, move in enumerate(
        parsed.move_tokens,
        start=1,
    ):
        cube.apply(move)
        if cube.is_solved:
            physical_solved_offsets.append(offset)
            if first_physical_solved_offset is None:
                first_physical_solved_offset = offset
        elif first_physical_solved_offset is not None:
            solved_then_unsolved_offsets.append(offset)
        if cube.state == cfop_solver.SOLVED_STATE:
            canonical_solved_offsets.append(offset)

    first_canonical_solved_offset = (
        canonical_solved_offsets[0]
        if canonical_solved_offsets
        else None
    )
    post_canonical_move_offsets = (
        tuple(range(first_canonical_solved_offset + 1, parsed.move_count + 1))
        if first_canonical_solved_offset is not None
        else ()
    )
    terminal_suffix_valid = bool(
        not solved_then_unsolved_offsets
        and first_canonical_solved_offset is not None
        and first_canonical_solved_offset == parsed.move_count
    )
    return {
        "physical_solved_offsets": tuple(physical_solved_offsets),
        "canonical_solved_offsets": tuple(canonical_solved_offsets),
        "first_physical_solved_offset": first_physical_solved_offset,
        "first_canonical_solved_offset": first_canonical_solved_offset,
        "solved_then_unsolved_offsets": tuple(solved_then_unsolved_offsets),
        "post_canonical_move_offsets": post_canonical_move_offsets,
        "terminal_suffix_valid": terminal_suffix_valid,
    }


def build_cfop_prompt(state: str) -> str:
    """Build the canonical v2 state-only CFOP prompt."""

    formatted_state = format_facelet_state(state)
    return (
        "Solve this Rubik's Cube completely using a valid CFOP phase trace.\n"
        "Think through the solve internally, phase by phase, and verify each "
        "CFOP boundary before responding.\n"
        "The state is the only cube input. It uses 54 facelets in URFDLB order "
        "(nine facelets per face).\n\n"
        "STATE (facelets-urfdlb-v1):\n"
        f"{formatted_state}\n\n"
        "Return only ASCII text in this exact grammar:\n"
        "SETUP: <x/y/z whole-cube rotations only>  (optional line)\n"
        "CROSS[FR,BL]: <space-separated moves>  (brackets list every F2L "
        "pair solved at the CROSS endpoint; omit brackets when none are solved)\n"
        "F2L FR: <non-empty space-separated moves>  (zero or more F2L lines; "
        "pair labels are FR, FL, BL, BR and may be comma-separated)\n"
        "OLL: <space-separated moves>\n"
        "PLL: <space-separated moves>\n"
        "END\n\n"
        "CROSS, OLL, and PLL move lists may be empty when that phase goal is "
        "already satisfied; every emitted F2L line must have at least one move. "
        "Do not include prose, Markdown fences, or blank lines. After the cube is "
        "physically solved, only whole-cube rotations that keep it solved may "
        "canonicalize its frame. Stop immediately once it is canonically solved.\n"
        "OUTPUT:\n"
    )


def build_cfop_reasoning_prompt(state: str) -> str:
    """Build the canonical grounded-reasoning full-solve prompt."""

    formatted_state = format_facelet_state(state)
    return (
        "Solve this Rubik's Cube completely using a valid CFOP phase trace.\n"
        "Reason through the selected phase transitions in a compact, grounded "
        "CFOP_PLAN_V1 block, then return the strict answer.\n"
        "The state is the only cube input. It uses 54 facelets in URFDLB order "
        "(nine facelets per face).\n\n"
        "STATE (facelets-urfdlb-v1):\n"
        f"{formatted_state}\n\n"
        "Return only ASCII text in this exact envelope:\n"
        "<think>\n"
        "CFOP_PLAN_V1\n"
        "STEP 1 | stage=CROSS | from=none | to=cross | moves=<count> "
        "| gain=- | preserve=- | break=-\n"
        "STEP 2 | stage=F2L FR | from=cross | to=xcross | moves=<count> "
        "| gain=FR | preserve=- | break=-\n"
        "...one STEP per answer phase...\n"
        "</think>\n"
        "CROSS[FR,BL]: <space-separated moves>\n"
        "F2L FR: <non-empty space-separated moves>\n"
        "OLL: <space-separated moves>\n"
        "PLL: <space-separated moves>\n"
        "END\n\n"
        "Every STEP must match the corresponding answer phase exactly: stage "
        "label, move count, verifier-derived milestone before and after, newly "
        "gained, preserved, and broken F2L pairs. Use '-' for an empty pair set "
        "and canonical FR,FL,BL,BR order otherwise. CROSS brackets list "
        "every pair already solved at its endpoint. F2L labels list exactly "
        "the newly gained pair or pairs. CROSS, OLL, and PLL move lists may be "
        "empty only when their goal is already satisfied. Do not include "
        "additional prose, Markdown fences, or blank lines. Stop immediately "
        "after the canonically solved answer's END line.\n"
        "OUTPUT:\n"
    )


def build_cfop_phase_reasoning_prompt(state: str, stage: str) -> str:
    """Build the grounded stage-local prompt used by phase curricula."""

    normalized_stage = str(stage).strip().lower()
    if normalized_stage != "cross":
        raise ValueError(
            "grounded phase reasoning currently supports only the Cross gate"
        )
    formatted_state = format_facelet_state(state)
    return (
        "Produce one valid CFOP Cross transition for this Rubik's Cube state.\n"
        "Reason through the transition in one compact, grounded CFOP_PLAN_V1 "
        "STEP, then return only the stage-local answer.\n"
        "The state is the only cube input. It uses 54 facelets in URFDLB order "
        "(nine facelets per face).\n\n"
        "STATE (facelets-urfdlb-v1):\n"
        f"{formatted_state}\n\n"
        "Return only ASCII text in this exact envelope:\n"
        "<think>\n"
        "CFOP_PLAN_V1\n"
        "STEP 1 | stage=CROSS | from=none | to=cross | moves=<count> "
        "| gain=- | preserve=- | break=-\n"
        "</think>\n"
        "CROSS: <space-separated moves>\n"
        "END\n\n"
        "If the Cross endpoint also has solved F2L pairs, declare every one in "
        "both stage labels as CROSS[FR,FL,...], using canonical FR,FL,BL,BR "
        "order. The STEP stage and move count must exactly match the answer. "
        "Its from/to milestones and gained, preserved, and broken pair sets "
        "must match cube replay; use '-' for an empty set. The answer must "
        "solve Cross in the canonical frame without hiding later-phase work. "
        "Do not include prose, Markdown fences, blank lines, or any phase after "
        "Cross. Stop immediately after END.\n"
        "OUTPUT:\n"
    )


def build_cfop_phase_prompt(
    state: str,
    stage: str,
    f2l_pairs: Sequence[str] = (),
) -> str:
    """Build a v2 state-only SFT prompt for one requested CFOP phase."""

    normalized_stage = str(stage).strip().lower()
    if normalized_stage == "f2l":
        normalized_stage = "f2l_pair"
    if normalized_stage not in {"cross", "f2l_pair", "oll", "pll"}:
        raise ValueError(
            "stage must be one of: cross, f2l_pair (or f2l), oll, pll"
        )

    raw_pairs: Sequence[str]
    if isinstance(f2l_pairs, str):
        raw_pairs = (f2l_pairs,)
    else:
        raw_pairs = f2l_pairs
    pairs = tuple(str(pair).strip().upper() for pair in raw_pairs)
    if normalized_stage in {"cross", "f2l_pair"}:
        if normalized_stage == "f2l_pair" and not pairs:
            raise ValueError("f2l_pair prompts require at least one F2L pair")
        invalid_pairs = [pair for pair in pairs if pair not in F2L_PAIR_ORDER]
        if invalid_pairs:
            raise ValueError(f"invalid F2L pair labels: {invalid_pairs!r}")
        if len(set(pairs)) != len(pairs):
            raise ValueError("F2L pair labels must be unique")
    if normalized_stage == "f2l_pair":
        header = f"F2L {','.join(pairs)}"
    elif normalized_stage == "cross":
        header = f"CROSS[{','.join(pairs)}]" if pairs else "CROSS"
    else:
        if pairs:
            raise ValueError(
                "f2l_pairs may only be provided for cross or f2l_pair prompts"
            )
        header = normalized_stage.upper()

    formatted_state = format_facelet_state(state)
    move_placeholder = (
        "<non-empty space-separated moves>"
        if normalized_stage == "f2l_pair"
        else "<space-separated moves>"
    )
    return (
        f"Produce only the {header} move sequence for this Rubik's Cube state.\n"
        "Think through the requested transition internally and verify its "
        "endpoint before responding.\n"
        "The state uses 54 facelets in URFDLB order (nine facelets per face).\n\n"
        "STATE (facelets-urfdlb-v1):\n"
        f"{formatted_state}\n\n"
        "Return exactly two ASCII lines, with no prose, Markdown fences, or "
        "blank lines:\n"
        f"{header}: {move_placeholder}\n"
        "END\n"
        + (
            "The F2L move list must contain at least one move.\n"
            if normalized_stage == "f2l_pair"
            else (
                "The move list may be empty when the requested phase goal is "
                "already satisfied.\n"
            )
        )
        + "OUTPUT:\n"
    )


def _phase_prefix_summary(
    *,
    phase_index: int,
    stage: str,
    after_move: int,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    parse = _as_mapping(result.get("parse"))
    return {
        "phase_index": int(phase_index),
        "stage": str(stage),
        "after_move": int(after_move),
        "valid": bool(parse.get("valid", False)),
        "solved": bool(result.get("solved", False)),
        "move_count": result.get("move_count"),
        "progress": dict(_as_mapping(result.get("progress"))),
    }


def _phase_boundary_diagnostics(
    parsed: ParsedCFOPResponse,
    prefix_progress: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    if len(prefix_progress) != len(parsed.phases) + 1:
        return (), (), ()

    phase_move_counts: list[Mapping[str, Any]] = []
    milestone_jumps: list[Mapping[str, Any]] = []
    pair_declaration_issues: list[Mapping[str, Any]] = []
    for phase_index, phase in enumerate(parsed.phases):
        before = prefix_progress[phase_index]
        after = prefix_progress[phase_index + 1]
        before_progress = _as_mapping(before.get("progress"))
        after_progress = _as_mapping(after.get("progress"))
        before_milestone = str(before_progress.get("milestone", "none"))
        after_milestone = str(after_progress.get("milestone", "none"))
        phase_record: dict[str, Any] = {
            "phase_index": phase_index,
            "stage": phase.stage,
            "move_count": phase.move_count,
            "after_move": after.get("after_move"),
        }
        if phase.f2l_pairs:
            phase_record["f2l_pairs"] = list(phase.f2l_pairs)
        phase_move_counts.append(phase_record)

        jump_size = (
            _MILESTONE_RANK.get(after_milestone, 0)
            - _MILESTONE_RANK.get(before_milestone, 0)
        )
        if jump_size > 1:
            milestone_jumps.append(
                {
                    "phase_index": phase_index,
                    "stage": phase.stage,
                    "before_milestone": before_milestone,
                    "after_milestone": after_milestone,
                    "move_count": phase.move_count,
                    "jump_size": jump_size,
                }
            )

        before_pairs = set(before_progress.get("solved_pairs", ()))
        after_pairs = set(after_progress.get("solved_pairs", ()))
        declared_pairs = set(phase.f2l_pairs)
        issue: dict[str, Any] | None = None
        if phase.stage == "cross" and declared_pairs != after_pairs:
            issue = {
                "kind": "cross_pair_commitment_mismatch",
                "expected_pairs": sorted(after_pairs),
            }
        elif phase.stage == "f2l_pair":
            gained_pairs = after_pairs - before_pairs
            broken_pairs = before_pairs - after_pairs
            if declared_pairs != gained_pairs or broken_pairs:
                issue = {
                    "kind": "f2l_pair_identity_mismatch",
                    "expected_pairs": sorted(gained_pairs),
                    "gained_pairs": sorted(gained_pairs),
                    "broken_pairs": sorted(broken_pairs),
                }
        if issue is not None:
            issue.update(
                {
                    "phase_index": phase_index,
                    "stage": phase.stage,
                    "declared_pairs": sorted(declared_pairs),
                    "before_pairs": sorted(before_pairs),
                    "after_pairs": sorted(after_pairs),
                }
            )
            pair_declaration_issues.append(issue)
    return (
        tuple(phase_move_counts),
        tuple(milestone_jumps),
        tuple(pair_declaration_issues),
    )


def _phase_fidelity_diagnostics(
    parsed: ParsedCFOPResponse,
    prefix_progress: Sequence[Mapping[str, Any]],
    final_f2l_internal_prefix_progress: Sequence[Mapping[str, Any]],
    oll_internal_prefix_progress: Sequence[Mapping[str, Any]],
) -> tuple[
    tuple[Mapping[str, Any], ...],
    tuple[Mapping[str, Any], ...],
]:
    if len(prefix_progress) != len(parsed.phases) + 1:
        return (
            (
                {
                    "kind": "missing_phase_prefix_diagnostics",
                    "expected_prefixes": len(parsed.phases) + 1,
                    "actual_prefixes": len(prefix_progress),
                },
            ),
            (),
        )

    issues: list[Mapping[str, Any]] = []
    exceptions: list[Mapping[str, Any]] = []
    f2l_rank = _MILESTONE_RANK["f2l"]
    last_f2l_phase_index = next(
        (
            phase_index
            for phase_index in range(len(parsed.phases) - 1, -1, -1)
            if parsed.phases[phase_index].stage == "f2l_pair"
        ),
        None,
    )
    oll_phase_index = next(
        (
            phase_index
            for phase_index, phase in enumerate(parsed.phases)
            if phase.stage == "oll"
        ),
        None,
    )
    for phase_index, phase in enumerate(parsed.phases):
        if phase.stage not in {"cross", "f2l_pair"}:
            continue
        before_progress = _as_mapping(
            prefix_progress[phase_index].get("progress")
        )
        after_progress = _as_mapping(
            prefix_progress[phase_index + 1].get("progress")
        )
        before_milestone = str(before_progress.get("milestone", "none"))
        after_milestone = str(after_progress.get("milestone", "none"))
        before_rank = _MILESTONE_RANK.get(before_milestone, 0)
        after_rank = _MILESTONE_RANK.get(after_milestone, 0)
        if after_rank <= f2l_rank or after_rank <= before_rank:
            continue
        record = {
            "phase_index": phase_index,
            "stage": phase.stage,
            "before_milestone": before_milestone,
            "after_milestone": after_milestone,
            "move_count": phase.move_count,
        }
        single_move_terminal_skip = bool(
            phase.move_count == 1
            and bool(prefix_progress[phase_index + 1].get("solved", False))
            and all(
                later_phase.move_count == 0
                for later_phase in parsed.phases[phase_index + 1 :]
            )
        )
        if single_move_terminal_skip:
            exceptions.append(
                {
                    "kind": "single_move_terminal_skip",
                    **record,
                }
            )
            continue
        final_f2l_oll_skip = bool(
            phase.stage == "f2l_pair"
            and phase_index == last_f2l_phase_index
            and after_milestone == "oll"
            and oll_phase_index is not None
            and oll_phase_index > phase_index
            and parsed.phases[oll_phase_index].move_count == 0
            and not any(
                bool(
                    _as_mapping(prefix.get("progress")).get(
                        "f2l_solved",
                        False,
                    )
                )
                for prefix in final_f2l_internal_prefix_progress
                if prefix.get("phase_index") == phase_index
            )
        )
        if final_f2l_oll_skip:
            exceptions.append(
                {
                    "kind": "final_f2l_oll_skip",
                    "first_f2l_solved_at_phase_endpoint": True,
                    "oll_move_count": 0,
                    **record,
                }
            )
            continue
        issues.append(
            {
                "kind": "downstream_milestone_in_early_phase",
                **record,
            }
        )
    for prefix in oll_internal_prefix_progress:
        progress = _as_mapping(prefix.get("progress"))
        if not bool(progress.get("oll_solved", False)):
            continue
        issues.append(
            {
                "kind": "pll_work_packed_into_oll",
                "stage": "oll",
                "after_move": prefix.get("after_move"),
                "oll_moves_remaining": prefix.get("oll_moves_remaining"),
                "milestone": progress.get("milestone"),
                "solved": bool(prefix.get("solved", False)),
            }
        )
        break
    return tuple(issues), tuple(exceptions)


@dataclass(slots=True)
class CFOPCubeEnv(BaseEnv):
    reward_fn: RewardFn
    verifier: Any | None = field(default=None, repr=False)
    verifier_threads: int = 0
    use_compatible_row_prompt: bool = False
    response_format: str = "direct"
    task_name: str = "cfop_cube"
    prompt_key: str = "prompt"
    target_key: str = "completion"

    def __post_init__(self) -> None:
        if int(self.verifier_threads) < 0:
            raise ValueError("verifier_threads must be non-negative")
        normalized_response_format = str(self.response_format).strip().lower()
        if normalized_response_format not in {"direct", "reasoning_v1"}:
            raise ValueError(
                "response_format must be 'direct' or 'reasoning_v1'"
            )
        self.response_format = normalized_response_format
        if self.verifier is None:
            self.verifier = _build_training_verifier(self.verifier_threads)

    @staticmethod
    def _resolve_state(row: Mapping[str, Any]) -> str:
        state = str(row.get("state", "") or "").strip()
        if not state:
            raise KeyError("CFOPCubeEnv requires a compact `state` facelet string")
        return state

    @staticmethod
    def _validate_environment_row(row: Mapping[str, Any]) -> None:
        if row.get("environment_eligible") is False:
            raise ValueError(
                "CFOPCubeEnv cannot consume a row with environment_eligible=false"
            )
        if row.get("sft_only") is True:
            raise ValueError("CFOPCubeEnv cannot consume an sft_only row")

    def build_prompt(self, row: Mapping[str, Any]) -> str:
        self._validate_environment_row(row)
        state = self._resolve_state(row)
        canonical_prompt = (
            build_cfop_reasoning_prompt(state)
            if self.response_format == "reasoning_v1"
            else build_cfop_prompt(state)
        )
        supplied_prompt = str(row.get(self.prompt_key, "") or "")
        if (
            self.use_compatible_row_prompt
            and supplied_prompt
            and supplied_prompt == canonical_prompt
        ):
            return supplied_prompt
        return canonical_prompt

    def build_target(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        self._validate_environment_row(row)
        state = self._resolve_state(row)
        completion = row.get(self.target_key)
        return {
            "state": state,
            "completion": "" if completion is None else str(completion),
            "known_solutions": _known_solutions_from_row(row),
        }

    def parse_response(self, response: str) -> ParsedCFOPResponse:
        if self.response_format == "reasoning_v1":
            reasoning = parse_cfop_reasoning_response(response)
            if not reasoning.valid or reasoning.answer is None:
                return ParsedCFOPResponse(
                    raw_response=str(response),
                    valid=False,
                    error=reasoning.error,
                    error_code=reasoning.error_code,
                    error_line=reasoning.error_line,
                )
            return replace(
                reasoning.answer,
                raw_response=str(response),
                canonical_text=reasoning.canonical_text,
            )
        return parse_cfop_response(response)

    def _verify_items(
        self,
        items: Sequence[Mapping[str, Any]],
    ) -> list[Mapping[str, Any]]:
        if self.verifier is None:
            raise RuntimeError("CFOPCubeEnv verifier was not initialized")
        verify_batch = getattr(self.verifier, "verify_batch", None)
        if not callable(verify_batch):
            raise CFOPCompatibilityError(
                "CFOP verifier must expose TrainingVerifier.verify_batch()"
            )
        batch = verify_batch([dict(item) for item in items])
        if not isinstance(batch, Mapping):
            raise CFOPCompatibilityError(
                "TrainingVerifier.verify_batch() must return a mapping"
            )
        if batch.get("schema") != _EXPECTED_VERIFICATION_SCHEMA:
            raise CFOPCompatibilityError(
                "TrainingVerifier batch schema must be "
                f"{_EXPECTED_VERIFICATION_SCHEMA!r}, got {batch.get('schema')!r}"
            )
        if batch.get("state_format") != _EXPECTED_STATE_FORMAT:
            raise CFOPCompatibilityError(
                "TrainingVerifier batch state_format must be "
                f"{_EXPECTED_STATE_FORMAT!r}, got {batch.get('state_format')!r}"
            )
        results = batch.get("results")
        if not isinstance(results, Sequence):
            raise CFOPCompatibilityError(
                "TrainingVerifier.verify_batch() did not return a results sequence"
            )
        resolved = [
            _as_mapping(result)
            for result in results
        ]
        if len(resolved) != len(items):
            raise CFOPCompatibilityError(
                "TrainingVerifier returned a different number of results than inputs"
            )
        return resolved

    def _verification_diagnostics(
        self,
        example: EnvExample,
        parsed: ParsedCFOPResponse,
        response: str,
    ) -> tuple[
        Mapping[str, Any],
        tuple[Mapping[str, Any], ...],
        Mapping[str, Any],
    ]:
        state = self._resolve_state(example.row)
        known_solutions = _known_solutions_from_row(example.row)
        candidate_moves = parsed.moves if parsed.valid else str(response).strip()
        boundaries = list(parsed.phase_boundaries) if parsed.valid else []

        item_id = (
            str(example.row.get("id"))
            if example.row.get("id") is not None
            else (
                str(example.dataset_index)
                if example.dataset_index is not None
                else None
            )
        )
        main_item: dict[str, Any] = {
            "state": state,
            "moves": candidate_moves,
            "known_solutions": list(known_solutions),
            "phase_boundaries": boundaries,
        }
        if item_id is not None:
            main_item["id"] = item_id

        prefix_specs: list[tuple[int, str, int]] = [(-1, "initial", 0)]
        if parsed.valid:
            prefix_specs.extend(
                (
                    phase_index,
                    str(boundary.get("stage", "")),
                    int(boundary.get("after_move", 0)),
                )
                for phase_index, boundary in enumerate(boundaries)
            )
        oll_internal_specs: list[tuple[int, int]] = []
        final_f2l_internal_specs: list[tuple[int, int, int]] = []
        move_offset = 0
        if parsed.valid:
            last_f2l_phase_index = next(
                (
                    phase_index
                    for phase_index in range(
                        len(parsed.phases) - 1,
                        -1,
                        -1,
                    )
                    if parsed.phases[phase_index].stage == "f2l_pair"
                ),
                None,
            )
            for phase_index, phase in enumerate(parsed.phases):
                phase_start = move_offset
                move_offset += phase.move_count
                if phase_index == last_f2l_phase_index:
                    final_f2l_internal_specs.extend(
                        (
                            phase_index,
                            offset,
                            move_offset - offset,
                        )
                        for offset in range(phase_start + 1, move_offset)
                    )
                if phase.stage == "oll":
                    oll_internal_specs.extend(
                        (offset, move_offset - offset)
                        for offset in range(phase_start + 1, move_offset)
                    )

        diagnostic_offsets = sorted(
            {
                after_move for _, _, after_move in prefix_specs
            }
            .union(offset for _, offset, _ in final_f2l_internal_specs)
            .union(offset for offset, _ in oll_internal_specs)
        )
        prefix_items = [
            {
                "state": state,
                "moves": " ".join(parsed.move_tokens[:offset]),
            }
            for offset in diagnostic_offsets
        ]

        results = self._verify_items([main_item, *prefix_items])
        verifier = results[0]
        result_by_offset = dict(
            zip(
                diagnostic_offsets,
                results[1:],
                strict=True,
            )
        )
        prefix_summaries = tuple(
            _phase_prefix_summary(
                phase_index=phase_index,
                stage=stage,
                after_move=after_move,
                result=result_by_offset[after_move],
            )
            for phase_index, stage, after_move in prefix_specs
        )
        verifier_parse = _as_mapping(verifier.get("parse"))
        solve_diagnostics = (
            dict(_sequential_solve_diagnostics(state, parsed))
            if parsed.valid and bool(verifier_parse.get("state_valid", False))
            else {}
        )
        solve_diagnostics["oll_internal_prefix_progress"] = tuple(
            {
                "after_move": offset,
                "oll_moves_remaining": moves_remaining,
                "solved": bool(result_by_offset[offset].get("solved", False)),
                "progress": dict(
                    _as_mapping(result_by_offset[offset].get("progress"))
                ),
            }
            for offset, moves_remaining in oll_internal_specs
        )
        solve_diagnostics["final_f2l_internal_prefix_progress"] = tuple(
            {
                "phase_index": phase_index,
                "after_move": offset,
                "f2l_moves_remaining": moves_remaining,
                "solved": bool(result_by_offset[offset].get("solved", False)),
                "progress": dict(
                    _as_mapping(result_by_offset[offset].get("progress"))
                ),
            }
            for phase_index, offset, moves_remaining
            in final_f2l_internal_specs
        )
        return verifier, prefix_summaries, solve_diagnostics

    @staticmethod
    def _trace_flags(
        parsed: ParsedCFOPResponse,
        verifier: Mapping[str, Any],
        prefix_progress: Sequence[Mapping[str, Any]],
        solve_diagnostics: Mapping[str, Any],
        *,
        pair_declarations_valid: bool,
        phase_fidelity_valid: bool,
    ) -> Mapping[str, Any]:
        early_terminal_solve = False
        early_terminal_stage: str | None = None
        early_terminal_solve_offset: int | None = None
        terminal_solve_at_boundary = False
        first_solved_offset_value = solve_diagnostics.get(
            "first_physical_solved_offset"
        )
        first_solved_offset = (
            int(first_solved_offset_value)
            if isinstance(first_solved_offset_value, int)
            else None
        )
        if first_solved_offset is not None:
            cumulative_moves = 0
            for phase_index, phase in enumerate(parsed.phases):
                cumulative_moves += int(phase.move_count)
                if cumulative_moves != first_solved_offset:
                    continue
                terminal_solve_at_boundary = all(
                    later_phase.move_count == 0
                    for later_phase in parsed.phases[phase_index + 1 :]
                )
                if phase.stage != "pll":
                    early_terminal_solve = True
                    early_terminal_stage = phase.stage
                    early_terminal_solve_offset = first_solved_offset
                break
        solved_then_unsolved_offsets = tuple(
            solve_diagnostics.get("solved_then_unsolved_offsets", ())
        )
        post_canonical_move_offsets = tuple(
            solve_diagnostics.get("post_canonical_move_offsets", ())
        )
        solve_then_disturb = bool(solved_then_unsolved_offsets)
        terminal_suffix_valid = bool(
            solve_diagnostics.get("terminal_suffix_valid", False)
        )
        requires_terminal_solve = bool(verifier.get("solved", False))
        premature_full_solve = bool(
            solve_then_disturb
            or post_canonical_move_offsets
            or (requires_terminal_solve and not terminal_suffix_valid)
        )
        saw_oll = any(
            str(prefix.get("stage", "")) == "oll"
            for prefix in prefix_progress
        )
        saw_pll = any(str(prefix.get("stage", "")) == "pll" for prefix in prefix_progress)

        cross_boundaries = [
            boundary
            for boundary in parsed.phase_boundaries
            if str(boundary.get("stage", "")) == "cross"
        ]
        all_moves_in_opening = bool(
            parsed.move_count > 0
            and cross_boundaries
            and int(cross_boundaries[-1].get("after_move", 0)) == parsed.move_count
        )

        verifier_parse = _as_mapping(verifier.get("parse"))
        phase_invariants = _as_mapping(verifier.get("phase_invariants"))
        prefixes_valid = bool(prefix_progress) and all(
            bool(prefix.get("valid", False)) for prefix in prefix_progress
        )
        cross_phase_index = next(
            (
                phase_index
                for phase_index, phase in enumerate(parsed.phases)
                if phase.stage == "cross"
            ),
            None,
        )
        oll_phase_index = next(
            (
                phase_index
                for phase_index, phase in enumerate(parsed.phases)
                if phase.stage == "oll"
            ),
            None,
        )
        pre_cross_progress = (
            _as_mapping(prefix_progress[cross_phase_index].get("progress"))
            if cross_phase_index is not None
            and cross_phase_index < len(prefix_progress)
            else {}
        )
        pre_oll_progress = (
            _as_mapping(prefix_progress[oll_phase_index].get("progress"))
            if oll_phase_index is not None
            and oll_phase_index < len(prefix_progress)
            else {}
        )
        pre_cross_canonical = bool(
            pre_cross_progress.get("canonical_orientation", False)
        )
        pre_oll_f2l_ready = bool(pre_oll_progress.get("f2l_solved", False))
        protocol_valid = bool(
            parsed.valid
            and verifier_parse.get("valid", False)
            and phase_invariants.get("status") == "passed"
            and prefixes_valid
            and saw_oll
            and saw_pll
            and pair_declarations_valid
            and phase_fidelity_valid
            and pre_cross_canonical
            and pre_oll_f2l_ready
            and terminal_suffix_valid
            and not premature_full_solve
        )
        return {
            "premature_full_solve": premature_full_solve,
            "all_moves_in_opening": all_moves_in_opening,
            "protocol_valid": protocol_valid,
            "strict_cfop_trace": protocol_valid,
            "early_terminal_solve": early_terminal_solve,
            "early_terminal_stage": early_terminal_stage,
            "early_terminal_solve_offset": early_terminal_solve_offset,
            "terminal_solve_at_boundary": terminal_solve_at_boundary,
            "terminal_suffix_valid": terminal_suffix_valid,
            "first_solved_offset": first_solved_offset,
            "solve_then_disturb": solve_then_disturb,
            "pre_cross_canonical": pre_cross_canonical,
            "pre_oll_f2l_ready": pre_oll_f2l_ready,
            "phase_fidelity_valid": phase_fidelity_valid,
        }

    def build_feedback(
        self,
        example: EnvExample,
        response: str,
        *,
        parsed_response: ParsedCFOPResponse | None = None,
    ) -> EnvFeedback:
        parsed = self.parse_response(response) if parsed_response is None else parsed_response
        reasoning_required = self.response_format == "reasoning_v1"
        reasoning_parse: ParsedCFOPReasoning | None = (
            parse_cfop_reasoning_response(response)
            if reasoning_required
            else None
        )
        verifier_error: str | None = None
        try:
            (
                verifier,
                prefix_progress,
                solve_diagnostics,
            ) = self._verification_diagnostics(
                example,
                parsed,
                response,
            )
        except CFOPCompatibilityError:
            raise
        except Exception as error:
            verifier_error = f"{type(error).__name__}: {error}"
            verifier = {}
            prefix_progress = ()
            solve_diagnostics = {}
        solved_move_offsets = tuple(
            solve_diagnostics.get("physical_solved_offsets", ())
        )
        oll_phase_index = next(
            (
                phase_index
                for phase_index, phase in enumerate(parsed.phases)
                if phase.stage == "oll"
            ),
            None,
        )
        oll_start_offset = (
            sum(
                phase.move_count
                for phase in parsed.phases[:oll_phase_index]
            )
            if oll_phase_index is not None
            else 0
        )
        pre_oll_solved_offsets = tuple(
            offset
            for offset in solved_move_offsets
            if offset <= oll_start_offset
        )
        (
            phase_move_counts,
            boundary_milestone_jumps,
            pair_declaration_issues,
        ) = _phase_boundary_diagnostics(
            parsed,
            prefix_progress,
        )
        pair_declarations_valid = bool(
            parsed.valid and not pair_declaration_issues
        )
        (
            phase_fidelity_issues,
            phase_fidelity_exceptions,
        ) = _phase_fidelity_diagnostics(
            parsed,
            prefix_progress,
            tuple(
                solve_diagnostics.get(
                    "final_f2l_internal_prefix_progress",
                    (),
                )
            ),
            tuple(
                solve_diagnostics.get(
                    "oll_internal_prefix_progress",
                    (),
                )
            ),
        )
        phase_fidelity_valid = bool(
            parsed.valid and not phase_fidelity_issues
        )
        reasoning_issues = (
            reasoning_grounding_issues(reasoning_parse, prefix_progress)
            if reasoning_parse is not None
            else ()
        )
        reasoning_envelope_valid = bool(
            reasoning_parse is not None and reasoning_parse.valid
        )
        reasoning_grounded_for_protocol = bool(
            (not reasoning_required)
            or (reasoning_envelope_valid and not reasoning_issues)
        )
        trace_flags = self._trace_flags(
            parsed,
            verifier,
            prefix_progress,
            solve_diagnostics,
            pair_declarations_valid=pair_declarations_valid,
            phase_fidelity_valid=phase_fidelity_valid,
        )
        premature_full_solve = bool(trace_flags["premature_full_solve"])
        strict_cfop_trace = bool(
            trace_flags["strict_cfop_trace"]
            and reasoning_grounded_for_protocol
        )
        early_terminal_solve = bool(trace_flags["early_terminal_solve"])
        early_terminal_stage = trace_flags["early_terminal_stage"]
        early_terminal_solve_offset = trace_flags["early_terminal_solve_offset"]
        verifier_parse = _as_mapping(verifier.get("parse"))
        phase_invariants = _as_mapping(verifier.get("phase_invariants"))
        solved = bool(verifier.get("solved", False))
        success = bool(
            parsed.valid
            and verifier_parse.get("valid", False)
            and solved
            and strict_cfop_trace
        )

        if success:
            if early_terminal_solve:
                message = (
                    "Cube solved with a valid terminal CFOP skip at the "
                    f"{early_terminal_stage} boundary in "
                    f"{verifier.get('move_count', parsed.move_count)} moves."
                )
            else:
                message = (
                    "Cube solved with a valid CFOP protocol trace in "
                    f"{verifier.get('move_count', parsed.move_count)} moves."
                )
        elif not parsed.valid:
            message = f"Invalid CFOP response format: {parsed.error or 'unknown parse error'}"
        elif verifier_error is not None:
            message = f"CFOP verification failed: {verifier_error}"
        elif reasoning_required and reasoning_issues:
            message = (
                "Rejected grounded CFOP reasoning: "
                f"{reasoning_issues[0]!r}"
            )
        elif premature_full_solve:
            message = (
                "Rejected CFOP protocol trace: the cube was solved mid-phase or "
                "was disturbed after an early solve."
            )
        elif not pair_declarations_valid:
            message = (
                "Rejected CFOP pair declarations: "
                f"{pair_declaration_issues[0]!r}"
            )
        elif not phase_fidelity_valid:
            message = (
                "Rejected CFOP phase fidelity: "
                f"{phase_fidelity_issues[0]!r}"
            )
        elif not bool(trace_flags["pre_cross_canonical"]):
            message = (
                "Rejected CFOP protocol trace: the cube frame was not canonical "
                "before CROSS began."
            )
        elif not bool(trace_flags["pre_oll_f2l_ready"]):
            message = (
                "Rejected CFOP protocol trace: F2L was not complete before OLL began."
            )
        elif not bool(verifier_parse.get("valid", False)):
            errors = verifier_parse.get("errors")
            message = f"Invalid cube state or move notation: {errors!r}"
        elif solved:
            violation = _as_mapping(phase_invariants.get("first_violation"))
            detail = (
                violation.get("message")
                or phase_invariants.get("error")
                or phase_invariants.get("status")
            )
            message = f"Cube solved, but the CFOP phase trace is invalid: {detail}"
        else:
            progress = _as_mapping(verifier.get("progress"))
            message = (
                "Cube is not solved. Highest verified CFOP milestone: "
                f"{progress.get('milestone', 'none')}."
            )

        return EnvFeedback(
            done=True,
            success=success,
            message=message,
            metadata={
                "format_valid": bool(parsed.valid),
                "parser_error": parsed.error,
                "parser_error_code": parsed.error_code,
                "parser_error_line": parsed.error_line,
                "canonical_response": parsed.canonical_text,
                "strict_cfop_trace": strict_cfop_trace,
                "protocol_valid": strict_cfop_trace,
                "strict_cfop_trace_rule": STRICT_CFOP_TRACE_RULE_V2,
                "protocol_rule": CFOP_PROTOCOL_RULE_V2,
                "premature_full_solve": premature_full_solve,
                "all_moves_in_opening": bool(trace_flags["all_moves_in_opening"]),
                "early_terminal_solve": early_terminal_solve,
                "early_terminal_stage": early_terminal_stage,
                "early_terminal_solve_offset": early_terminal_solve_offset,
                "terminal_solve_at_boundary": bool(
                    trace_flags["terminal_solve_at_boundary"]
                ),
                "terminal_suffix_valid": bool(
                    trace_flags["terminal_suffix_valid"]
                ),
                "first_solved_offset": trace_flags["first_solved_offset"],
                "solve_then_disturb": bool(trace_flags["solve_then_disturb"]),
                "physical_solved_offsets": solved_move_offsets,
                "canonical_solved_offsets": tuple(
                    solve_diagnostics.get("canonical_solved_offsets", ())
                ),
                "first_physical_solved_offset": solve_diagnostics.get(
                    "first_physical_solved_offset"
                ),
                "first_canonical_solved_offset": solve_diagnostics.get(
                    "first_canonical_solved_offset"
                ),
                "solved_then_unsolved_offsets": tuple(
                    solve_diagnostics.get("solved_then_unsolved_offsets", ())
                ),
                "post_canonical_move_offsets": tuple(
                    solve_diagnostics.get("post_canonical_move_offsets", ())
                ),
                "pre_cross_canonical": bool(trace_flags["pre_cross_canonical"]),
                "pre_oll_f2l_ready": bool(trace_flags["pre_oll_f2l_ready"]),
                "pair_declarations_valid": pair_declarations_valid,
                "pair_declaration_issues": pair_declaration_issues,
                "reasoning_protocol": (
                    CFOP_REASONING_PROTOCOL if reasoning_required else None
                ),
                "reasoning_required": reasoning_required,
                "reasoning_envelope_valid": reasoning_envelope_valid,
                "reasoning_grounded_valid": bool(
                    reasoning_required
                    and reasoning_grounded_for_protocol
                ),
                "reasoning_grounding_issues": reasoning_issues,
                "reasoning_plan_steps": (
                    tuple(
                        {
                            "index": step.index,
                            "stage": step.stage_label,
                            "from_milestone": step.from_milestone,
                            "to_milestone": step.to_milestone,
                            "move_count": step.move_count,
                            "gained_pairs": step.gained_pairs,
                            "preserved_pairs": step.preserved_pairs,
                            "broken_pairs": step.broken_pairs,
                        }
                        for step in reasoning_parse.steps
                    )
                    if reasoning_parse is not None
                    else ()
                ),
                "phase_fidelity_valid": phase_fidelity_valid,
                "phase_fidelity_issues": phase_fidelity_issues,
                "phase_fidelity_exceptions": phase_fidelity_exceptions,
                "oll_internal_prefix_progress": tuple(
                    solve_diagnostics.get(
                        "oll_internal_prefix_progress",
                        (),
                    )
                ),
                "final_f2l_internal_prefix_progress": tuple(
                    solve_diagnostics.get(
                        "final_f2l_internal_prefix_progress",
                        (),
                    )
                ),
                "phase_prefix_progress": prefix_progress,
                "phase_move_counts": phase_move_counts,
                "boundary_milestone_jumps": boundary_milestone_jumps,
                "solved_move_offsets": solved_move_offsets,
                "pre_oll_solved_offsets": pre_oll_solved_offsets,
                "verifier_error": verifier_error,
                "verifier": dict(verifier),
            },
        )


def cfop_phase_v1_reward(request: RewardRequest) -> RewardResult:
    """Dense stage-local reward with strict phase success capped at 1.0."""

    metadata = (
        request.feedback.metadata
        if request.feedback is not None
        and isinstance(request.feedback.metadata, Mapping)
        else {}
    )
    reasoning_required = bool(metadata.get("reasoning_required", False))
    format_valid = bool(metadata.get("format_valid", False))
    notation_valid = bool(metadata.get("notation_valid", False))
    envelope_valid = bool(metadata.get("reasoning_envelope_valid", False))
    grounded_valid = bool(metadata.get("reasoning_grounded_valid", False))
    invariant_valid = bool(metadata.get("phase_invariant_valid", False))
    pair_valid = bool(metadata.get("pair_declarations_valid", False))
    fidelity_valid = bool(metadata.get("phase_fidelity_valid", False))
    phase_goal_success = bool(metadata.get("phase_goal_success", False))
    strict_success = bool(metadata.get("strict_success", False))
    cross_progress = float(metadata.get("cross_progress", 0.0) or 0.0)
    cross_progress = max(0.0, min(1.0, cross_progress))

    reward_components = {
        "format_reward": 0.05 if format_valid else 0.0,
        "notation_reward": 0.05 if notation_valid else 0.0,
        "reasoning_envelope_reward": (
            0.05 if (not reasoning_required or envelope_valid) else 0.0
        ),
        "reasoning_grounding_reward": (
            0.10 if (not reasoning_required or grounded_valid) else 0.0
        ),
        "phase_invariant_reward": 0.10 if invariant_valid else 0.0,
        "pair_declaration_reward": 0.10 if pair_valid else 0.0,
        "phase_fidelity_reward": 0.05 if fidelity_valid else 0.0,
        "cross_progress_reward": 0.10 * cross_progress,
        "phase_goal_reward": 0.15 if phase_goal_success else 0.0,
        "strict_success_reward": 0.25 if strict_success else 0.0,
    }
    reward = float(sum(reward_components.values()))
    return RewardResult(
        reward=reward,
        info={
            **dict(metadata),
            **reward_components,
            "reward": reward,
        },
    )


@dataclass(slots=True)
class CFOPPhaseEnv(CFOPCubeEnv):
    """Strict verifier-backed environment for one local CFOP transition."""

    requested_stage: str = "cross"
    task_name: str = "cfop_phase"

    def __post_init__(self) -> None:
        CFOPCubeEnv.__post_init__(self)
        self.requested_stage = str(self.requested_stage).strip().lower()
        if self.requested_stage != "cross":
            raise ValueError("CFOPPhaseEnv currently supports requested_stage='cross'")

    def _validate_phase_row(self, row: Mapping[str, Any]) -> None:
        if str(row.get("task_type", "")).strip().lower() != "phase":
            raise ValueError("CFOPPhaseEnv requires task_type='phase'")
        row_stage = str(row.get("stage", "")).strip().lower()
        if row_stage != self.requested_stage:
            raise ValueError(
                "CFOPPhaseEnv row stage does not match requested_stage: "
                f"{row_stage!r} != {self.requested_stage!r}"
            )

    def build_prompt(self, row: Mapping[str, Any]) -> str:
        self._validate_phase_row(row)
        state = self._resolve_state(row)
        canonical_prompt = (
            build_cfop_phase_reasoning_prompt(state, self.requested_stage)
            if self.response_format == "reasoning_v1"
            else build_cfop_phase_prompt(state, self.requested_stage)
        )
        supplied_prompt = str(row.get(self.prompt_key, "") or "")
        if (
            self.use_compatible_row_prompt
            and supplied_prompt
            and supplied_prompt == canonical_prompt
        ):
            return supplied_prompt
        return canonical_prompt

    def build_target(self, row: Mapping[str, Any]) -> Mapping[str, Any]:
        self._validate_phase_row(row)
        completion = row.get(self.target_key)
        return {
            "state": self._resolve_state(row),
            "stage": self.requested_stage,
            "completion": "" if completion is None else str(completion),
            "known_solutions": _known_solutions_from_row(row),
            "teacher_end_state": str(row.get("stage_end_state", "")),
        }

    def parse_response(self, response: str) -> ParsedCFOPResponse:
        if self.response_format == "reasoning_v1":
            reasoning = parse_cfop_phase_reasoning_response(response)
            if not reasoning.valid or reasoning.answer is None:
                return ParsedCFOPResponse(
                    raw_response=str(response),
                    valid=False,
                    error=reasoning.error,
                    error_code=reasoning.error_code,
                    error_line=reasoning.error_line,
                )
            return replace(
                reasoning.answer,
                raw_response=str(response),
                canonical_text=reasoning.canonical_text,
            )
        return parse_cfop_phase_response(response)

    def build_feedback(
        self,
        example: EnvExample,
        response: str,
        *,
        parsed_response: ParsedCFOPResponse | None = None,
    ) -> EnvFeedback:
        parsed = (
            self.parse_response(response)
            if parsed_response is None
            else parsed_response
        )
        reasoning_required = self.response_format == "reasoning_v1"
        reasoning_parse = (
            parse_cfop_phase_reasoning_response(response)
            if reasoning_required
            else None
        )
        state = self._resolve_state(example.row)
        known_solutions = _known_solutions_from_row(example.row)
        phase = parsed.phases[0] if parsed.valid and len(parsed.phases) == 1 else None
        boundaries = list(parsed.phase_boundaries) if phase is not None else []
        item_id = (
            str(example.row.get("row_id"))
            if example.row.get("row_id") is not None
            else (
                str(example.dataset_index)
                if example.dataset_index is not None
                else None
            )
        )
        main_item: dict[str, Any] = {
            "state": state,
            "moves": parsed.moves if parsed.valid else "",
            "known_solutions": list(known_solutions),
            "phase_boundaries": boundaries,
        }
        if item_id is not None:
            main_item["id"] = item_id

        verifier_error: str | None = None
        try:
            verifier, initial = self._verify_items(
                [
                    main_item,
                    {
                        "state": state,
                        "moves": "",
                    },
                ]
            )
        except CFOPCompatibilityError:
            raise
        except Exception as error:
            verifier_error = f"{type(error).__name__}: {error}"
            verifier = {}
            initial = {}

        verifier_parse = _as_mapping(verifier.get("parse"))
        initial_parse = _as_mapping(initial.get("parse"))
        initial_progress = _as_mapping(initial.get("progress"))
        final_progress = _as_mapping(verifier.get("progress"))
        prefix_progress = (
            {
                "phase_index": -1,
                "stage": "initial",
                "after_move": 0,
                "valid": bool(initial_parse.get("valid", False)),
                "solved": bool(initial.get("solved", False)),
                "move_count": initial.get("move_count"),
                "progress": dict(initial_progress),
            },
            {
                "phase_index": 0,
                "stage": self.requested_stage,
                "after_move": parsed.move_count if parsed.valid else 0,
                "valid": bool(verifier_parse.get("valid", False)),
                "solved": bool(verifier.get("solved", False)),
                "move_count": verifier.get("move_count"),
                "progress": dict(final_progress),
            },
        )
        reasoning_issues = (
            reasoning_grounding_issues(reasoning_parse, prefix_progress)
            if reasoning_parse is not None
            else ()
        )
        reasoning_envelope_valid = bool(
            reasoning_parse is not None and reasoning_parse.valid
        )
        reasoning_grounded_valid = bool(
            reasoning_envelope_valid and not reasoning_issues
        )

        stage_matches = bool(
            phase is not None and phase.stage == self.requested_stage
        )
        phase_invariants = _as_mapping(verifier.get("phase_invariants"))
        phase_invariant_valid = bool(
            parsed.valid
            and phase_invariants.get("status") == "passed"
        )
        pair_delta = _as_mapping(verifier.get("pair_delta"))
        final_pairs = tuple(
            pair.lower()
            for pair in F2L_PAIR_ORDER
            if pair.lower()
            in {
                str(value).strip().lower()
                for value in tuple(pair_delta.get("final_solved", ()) or ())
            }
        )
        declared_pairs = (
            tuple(str(pair).lower() for pair in phase.f2l_pairs)
            if phase is not None
            else ()
        )
        pair_declarations_valid = bool(
            stage_matches and declared_pairs == final_pairs
        )
        pre_cross_canonical = bool(
            initial_progress.get("canonical_orientation", False)
        )
        final_milestone = str(final_progress.get("milestone", "none"))
        phase_fidelity_valid = bool(
            stage_matches
            and final_milestone in _MILESTONE_RANK
            and _MILESTONE_RANK[final_milestone] <= _MILESTONE_RANK["f2l"]
            and not bool(verifier.get("solved", False))
        )
        cross_edges_solved = max(
            0,
            min(4, int(final_progress.get("cross_edges_solved", 0) or 0)),
        )
        cross_progress = (
            float(cross_edges_solved) / 4.0
            if (
                parsed.valid
                and bool(verifier_parse.get("valid", False))
                and pre_cross_canonical
            )
            else 0.0
        )
        phase_goal_success = bool(
            parsed.valid and final_progress.get("cross_solved", False)
        )
        notation_valid = bool(
            parsed.valid and verifier_parse.get("moves_valid", False)
        )
        protocol_valid = bool(
            parsed.valid
            and verifier_parse.get("valid", False)
            and stage_matches
            and phase_invariant_valid
            and pair_declarations_valid
            and pre_cross_canonical
            and phase_fidelity_valid
            and (
                not reasoning_required
                or reasoning_grounded_valid
            )
        )
        success = bool(protocol_valid and phase_goal_success)

        if success:
            message = (
                "Cross solved with a valid grounded stage-local trace in "
                f"{parsed.move_count} moves."
            )
        elif not parsed.valid:
            message = (
                "Invalid stage-local CFOP response: "
                f"{parsed.error or 'unknown parse error'}"
            )
        elif verifier_error is not None:
            message = f"CFOP phase verification failed: {verifier_error}"
        elif reasoning_required and reasoning_issues:
            message = (
                "Rejected grounded stage-local reasoning: "
                f"{reasoning_issues[0]!r}"
            )
        elif not stage_matches:
            message = "Response does not contain exactly the requested Cross phase."
        elif not pair_declarations_valid:
            message = (
                "CROSS labels do not match the endpoint solved-pair set: "
                f"declared={declared_pairs!r}, actual={final_pairs!r}."
            )
        elif not phase_fidelity_valid:
            message = (
                "Cross response hides downstream OLL, PLL, or full-solve work."
            )
        elif not pre_cross_canonical:
            message = "Cross must begin in the canonical cube frame."
        elif not phase_goal_success:
            message = (
                "Cross is incomplete. Verified Cross edges: "
                f"{cross_edges_solved}/4."
            )
        else:
            message = "Cross response failed the strict phase protocol."

        return EnvFeedback(
            done=True,
            success=success,
            message=message,
            metadata={
                "format_valid": bool(parsed.valid),
                "notation_valid": notation_valid,
                "verifier_valid": bool(
                    parsed.valid and verifier_parse.get("valid", False)
                ),
                "solved": bool(verifier.get("solved", False)),
                "strict_success": success,
                "strict_cfop_trace": protocol_valid,
                "protocol_valid": protocol_valid,
                "protocol_rule": CFOP_PHASE_PROTOCOL_RULE_V1,
                "requested_stage": self.requested_stage,
                "phase_goal_success": phase_goal_success,
                "cross_solved": phase_goal_success,
                "cross_edges_solved": cross_edges_solved,
                "cross_progress": cross_progress,
                "phase_invariant_valid": phase_invariant_valid,
                "phase_invariant_status": phase_invariants.get("status"),
                "phase_fidelity_valid": phase_fidelity_valid,
                "pre_cross_canonical": pre_cross_canonical,
                "pair_declarations_valid": pair_declarations_valid,
                "declared_pairs": declared_pairs,
                "final_pairs": final_pairs,
                "reasoning_protocol": (
                    CFOP_PHASE_REASONING_PROTOCOL
                    if reasoning_required
                    else None
                ),
                "reasoning_required": reasoning_required,
                "reasoning_envelope_valid": reasoning_envelope_valid,
                "reasoning_grounded_valid": reasoning_grounded_valid,
                "reasoning_grounding_issues": reasoning_issues,
                "reasoning_plan_steps": (
                    tuple(
                        {
                            "index": step.index,
                            "stage": step.stage_label,
                            "from_milestone": step.from_milestone,
                            "to_milestone": step.to_milestone,
                            "move_count": step.move_count,
                            "gained_pairs": step.gained_pairs,
                            "preserved_pairs": step.preserved_pairs,
                            "broken_pairs": step.broken_pairs,
                        }
                        for step in reasoning_parse.steps
                    )
                    if reasoning_parse is not None
                    else ()
                ),
                "parser_error": parsed.error,
                "parser_error_code": parsed.error_code,
                "parser_error_line": parsed.error_line,
                "canonical_response": parsed.canonical_text,
                "phase_prefix_progress": prefix_progress,
                "move_count": (
                    verifier.get("move_count")
                    if isinstance(verifier.get("move_count"), int)
                    else parsed.move_count
                ),
                "verifier_error": verifier_error,
                "verifier": dict(verifier),
            },
        )


__all__ = [
    "CFOPCompatibilityError",
    "CFOP_PHASE_PROTOCOL_RULE_V1",
    "CFOP_PROTOCOL_RULE_V1",
    "CFOP_PROTOCOL_RULE_V2",
    "CFOPCubeEnv",
    "CFOPPhaseEnv",
    "STRICT_CFOP_TRACE_RULE_V1",
    "STRICT_CFOP_TRACE_RULE_V2",
    "build_cfop_phase_reasoning_prompt",
    "build_cfop_phase_prompt",
    "build_cfop_prompt",
    "build_cfop_reasoning_prompt",
    "cfop_dense_v2_reward",
    "cfop_phase_v1_reward",
]
