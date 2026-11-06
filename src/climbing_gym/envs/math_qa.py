from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Any, Mapping

from climbing_gym.envs.contracts import (
    BaseEnv,
    EnvExample,
    EnvFeedback,
    RewardFn,
    RewardRequest,
    RewardResult,
)


_ANSWER_TAG_RE = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.DOTALL | re.IGNORECASE)
_ANSWER_OPEN_RE = re.compile(r"<\s*answer\b", re.IGNORECASE)
_ANSWER_CLOSE_RE = re.compile(r"<\s*/\s*answer\b", re.IGNORECASE)
_ANSWER_END_RE = re.compile(r"\s*(?:<\|im_end\|>)?\s*\Z")
_PREFILLED_THINKING_ANSWER_RE = re.compile(
    r"\s*(?P<thinking>.+?)\s*</thinking>\s*"
    r"<answer>\s*(?P<answer>.*?)\s*</answer>\s*(?:<\|im_end\|>)?\s*",
    re.DOTALL | re.IGNORECASE,
)


def normalize_math_answer(value: Any) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    # Canonical GSM8K rows keep the worked solution and final answer in the
    # same `answer` field, separated by this marker.  Normalize only the gold
    # answer so callers can train directly from the upstream dataset instead
    # of requiring a separately materialized `parsed_answer` column.
    if "####" in text:
        text = text.rsplit("####", 1)[-1].strip()
    text = re.sub(r"^\$+|\$+$", "", text).strip()
    text = text.replace(",", "")
    text = re.sub(r"\s+", "", text)
    if text.endswith("."):
        text = text[:-1]
    return text


def _extract_last_answer_tag(text: str) -> str:
    matches = list(_ANSWER_TAG_RE.finditer(str(text)))
    return matches[-1].group(1).strip() if matches else ""


def _extract_last_boxed_answer(text: str) -> str:
    raw = str(text)
    marker = r"\boxed"
    start = raw.rfind(marker)
    while start >= 0:
        idx = start + len(marker)
        while idx < len(raw) and raw[idx].isspace():
            idx += 1
        if idx < len(raw) and raw[idx] == "{":
            depth = 0
            for end in range(idx, len(raw)):
                char = raw[end]
                if char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        return raw[idx + 1 : end].strip()
        start = raw.rfind(marker, 0, start)
    return ""


def _extract_last_non_empty_line(text: str) -> str:
    for line in reversed(str(text).splitlines()):
        stripped = line.strip()
        if stripped:
            return stripped
    return str(text).strip()


@dataclass(frozen=True, slots=True)
class ParsedMathQAResponse:
    final_answer: str
    raw_response: str
    boxed_answer: str
    answer_tag_answer: str
    has_boxed_answer: bool
    has_answer_tag: bool
    answer_tag_count: int
    strict_thinking_answer_format: bool


def _math_reward_components(request: RewardRequest) -> dict[str, float]:
    parsed = request.parsed_response
    if not isinstance(parsed, ParsedMathQAResponse):
        raise TypeError("math reward requires ParsedMathQAResponse")
    final_answer = str(parsed.final_answer)
    target = str(request.example.target)
    return {
        "exact_match": float(final_answer == target),
        "reward_format": float(parsed.strict_thinking_answer_format),
        "has_boxed_answer": float(parsed.has_boxed_answer),
        "has_answer_tag": float(parsed.has_answer_tag),
        "answer_tag_count": float(parsed.answer_tag_count),
    }


def exact_match_math_reward(request: RewardRequest) -> RewardResult:
    info = _math_reward_components(request)
    return RewardResult(
        reward=float(info["exact_match"]),
        info=info,
    )


def _single_final_answer_components(
    parsed: ParsedMathQAResponse, target: Any,
) -> dict[str, float]:
    raw = parsed.raw_response
    matches = list(_ANSWER_TAG_RE.finditer(raw))
    open_count = len(_ANSWER_OPEN_RE.findall(raw))
    close_count = len(_ANSWER_CLOSE_RE.findall(raw))
    valid = bool(
        len(matches) == open_count == close_count == 1
        and matches[0].group(1).strip()
        and "<" not in matches[0].group(1)
        and _ANSWER_END_RE.fullmatch(raw[matches[0].end():])
    )
    answer_correct = bool(
        matches and normalize_math_answer(matches[-1].group(1)) == str(target)
    )
    return {
        "answer_correct": float(answer_correct),
        "single_final_answer_valid": float(valid),
        "answer_tag_open_count": float(open_count),
        "answer_tag_close_count": float(close_count),
        "exact_match": float(valid and answer_correct),
    }


def exact_match_single_answer_math_reward(request: RewardRequest) -> RewardResult:
    """Reward a correct, single terminal answer block; never reward answer loops.

    Reasoning length and style are unrestricted. There is no format bonus:
    malformed, repeated, missing, nonterminal, or incorrect answers earn zero.
    """
    info = _math_reward_components(request)
    info.update(_single_final_answer_components(request.parsed_response, request.example.target))
    return RewardResult(reward=info["exact_match"], info=info)


def exact_match_with_format_math_reward(
    request: RewardRequest,
    *,
    format_weight: float = 0.1,
) -> RewardResult:
    """Exact-match outcome with a strict thinking/answer format bonus."""

    if float(format_weight) < 0.0:
        raise ValueError("format_weight must be >= 0")
    info = _math_reward_components(request)
    return RewardResult(
        reward=float(info["exact_match"] + float(format_weight) * info["reward_format"]),
        info=info,
    )


def _soft_turn_target_alignment(turn_count: int, *, target_turn_count: int = 2) -> float:
    resolved_turn_count = max(1, int(turn_count))
    target = max(1, int(target_turn_count))
    return max(0.0, 1.0 - (abs(float(resolved_turn_count - target)) / float(target)))


def exact_match_math_soft_target_2_turns_reward(request: RewardRequest) -> RewardResult:
    final_answer = str(getattr(request.parsed_response, "final_answer", ""))
    target = str(request.example.target)
    exact_match = float(final_answer == target)
    turn_count = max(1, int(request.turn_count))
    turn_alignment = _soft_turn_target_alignment(turn_count, target_turn_count=2)
    turn_bonus = (-0.1 + (0.2 * turn_alignment)) if exact_match > 0.0 else 0.0
    return RewardResult(
        reward=float(exact_match + turn_bonus),
        info={
            "exact_match": exact_match,
            "turn_count": float(turn_count),
            "turn_target": 2.0,
            "turn_target_alignment": float(turn_alignment),
            "turn_bonus": float(turn_bonus),
            "has_boxed_answer": float(getattr(request.parsed_response, "has_boxed_answer", False)),
            "has_answer_tag": float(getattr(request.parsed_response, "has_answer_tag", False)),
        },
    )


def _turn_summary_payloads(request: RewardRequest) -> tuple[str, ...]:
    payloads: list[str] = []
    for metadata in request.turn_metadata:
        payloads.append(str(metadata.get("summary", "")).strip())
    return tuple(payloads)


def _turn_format_flags(request: RewardRequest) -> tuple[tuple[bool, ...], bool, bool]:
    turn_metadata = tuple(request.turn_metadata)
    turn_count = max(1, int(request.turn_count))
    expected_intermediate_turns = max(0, turn_count - 1)

    intermediate_flags: list[bool] = []
    for turn_index in range(expected_intermediate_turns):
        metadata = turn_metadata[turn_index] if turn_index < len(turn_metadata) else {}
        intermediate_flags.append(bool(metadata.get("strict_intermediate_format", False)))

    final_metadata = turn_metadata[-1] if turn_metadata else {}
    strict_final_turn = bool(final_metadata.get("strict_final_format", False))
    strict_format_ok = (
        len(turn_metadata) == turn_count
        and len(intermediate_flags) == expected_intermediate_turns
        and all(intermediate_flags)
        and strict_final_turn
    )
    return tuple(intermediate_flags), strict_final_turn, strict_format_ok


def exact_match_math_summary_bootstrap_2_turns_reward(request: RewardRequest) -> RewardResult:
    final_answer = str(getattr(request.parsed_response, "final_answer", ""))
    target = str(request.example.target)
    exact_match = float(final_answer == target)
    turn_count = max(1, int(request.turn_count))

    summary_payloads = _turn_summary_payloads(request)
    expected_summary_turns = max(0, turn_count - 1)
    intermediate_summary_count = sum(bool(payload) for payload in summary_payloads[:expected_summary_turns])
    intermediate_summary_coverage = (
        float(intermediate_summary_count / expected_summary_turns)
        if expected_summary_turns > 0
        else 0.0
    )
    final_turn_has_summary = bool(summary_payloads[-1]) if summary_payloads else False
    perfect_summary_chain = (
        expected_summary_turns > 0
        and intermediate_summary_count == expected_summary_turns
        and not final_turn_has_summary
    )

    if turn_count == 1:
        structure_reward = -0.2
    else:
        structure_reward = 0.1 * intermediate_summary_coverage
        structure_reward += 0.08 if not final_turn_has_summary else -0.08
        if turn_count == 2:
            structure_reward += 0.12
        elif turn_count == 3:
            structure_reward += 0.04
        else:
            structure_reward -= 0.05 * float(min(turn_count - 3, 2))
        if perfect_summary_chain:
            structure_reward += 0.1

    shaped_structure_reward = structure_reward if exact_match > 0.0 else (0.25 * structure_reward)
    reward = float(exact_match + shaped_structure_reward)
    return RewardResult(
        reward=reward,
        info={
            "exact_match": exact_match,
            "turn_count": float(turn_count),
            "summary_turn_count": float(sum(bool(payload) for payload in summary_payloads)),
            "expected_summary_turns": float(expected_summary_turns),
            "intermediate_summary_count": float(intermediate_summary_count),
            "intermediate_summary_coverage": float(intermediate_summary_coverage),
            "final_turn_has_summary": float(final_turn_has_summary),
            "perfect_summary_chain": float(perfect_summary_chain),
            "structure_reward": float(structure_reward),
            "shaped_structure_reward": float(shaped_structure_reward),
            "has_boxed_answer": float(getattr(request.parsed_response, "has_boxed_answer", False)),
            "has_answer_tag": float(getattr(request.parsed_response, "has_answer_tag", False)),
        },
    )


def exact_match_math_min_2_turns_reward(request: RewardRequest) -> RewardResult:
    final_answer = str(getattr(request.parsed_response, "final_answer", ""))
    target = str(request.example.target)
    exact_match = float(final_answer == target)
    turn_count = max(1, int(request.turn_count))
    eligible_turn_count = float(turn_count >= 2)
    reward = float(exact_match * eligible_turn_count)
    return RewardResult(
        reward=reward,
        info={
            "exact_match": exact_match,
            "turn_count": float(turn_count),
            "eligible_turn_count": eligible_turn_count,
            "has_boxed_answer": float(getattr(request.parsed_response, "has_boxed_answer", False)),
            "has_answer_tag": float(getattr(request.parsed_response, "has_answer_tag", False)),
        },
    )


def exact_match_math_underturn_penalty_2_turns_reward(request: RewardRequest) -> RewardResult:
    final_answer = str(getattr(request.parsed_response, "final_answer", ""))
    target = str(request.example.target)
    exact_match = float(final_answer == target)
    turn_count = max(1, int(request.turn_count))
    target_turn_count = 2
    under_turn_gap = max(0, target_turn_count - turn_count)
    eligible_turn_count = float(under_turn_gap <= 0)
    under_turn_penalty = float(under_turn_gap)
    reward = float(exact_match if eligible_turn_count > 0.0 else -under_turn_penalty)
    return RewardResult(
        reward=reward,
        info={
            "exact_match": exact_match,
            "turn_count": float(turn_count),
            "target_turn_count": float(target_turn_count),
            "eligible_turn_count": eligible_turn_count,
            "under_turn_gap": float(under_turn_gap),
            "under_turn_penalty": float(under_turn_penalty),
            "has_boxed_answer": float(getattr(request.parsed_response, "has_boxed_answer", False)),
            "has_answer_tag": float(getattr(request.parsed_response, "has_answer_tag", False)),
        },
    )


def exact_match_math_hybrid_2_turns_reward(request: RewardRequest) -> RewardResult:
    final_answer = str(getattr(request.parsed_response, "final_answer", ""))
    target = str(request.example.target)
    exact_match = float(final_answer == target)
    turn_count = max(1, int(request.turn_count))
    summary_payloads = _turn_summary_payloads(request)
    expected_summary_turns = max(0, turn_count - 1)
    intermediate_summary_count = sum(bool(payload) for payload in summary_payloads[:expected_summary_turns])
    final_turn_has_summary = bool(summary_payloads[-1]) if summary_payloads else False
    strict_intermediate_turns, strict_final_turn, strict_format_ok = _turn_format_flags(request)
    reward = float(exact_match * float(strict_format_ok))
    return RewardResult(
        reward=reward,
        info={
            "exact_match": exact_match,
            "turn_count": float(turn_count),
            "summary_turn_count": float(sum(bool(payload) for payload in summary_payloads)),
            "expected_summary_turns": float(expected_summary_turns),
            "intermediate_summary_count": float(intermediate_summary_count),
            "final_turn_has_summary": float(final_turn_has_summary),
            "strict_intermediate_turn_count": float(sum(strict_intermediate_turns)),
            "expected_intermediate_turn_count": float(expected_summary_turns),
            "strict_final_turn": float(strict_final_turn),
            "strict_format_ok": float(strict_format_ok),
            "format_reward": float(strict_format_ok),
            "has_boxed_answer": float(getattr(request.parsed_response, "has_boxed_answer", False)),
            "has_answer_tag": float(getattr(request.parsed_response, "has_answer_tag", False)),
        },
    )


def exact_match_math_bounded_summary_2_turns_reward(request: RewardRequest) -> RewardResult:
    final_answer = str(getattr(request.parsed_response, "final_answer", ""))
    target = str(request.example.target)
    exact_match = float(final_answer == target)
    turn_count = max(1, int(request.turn_count))

    summary_payloads = _turn_summary_payloads(request)
    expected_summary_turns = max(0, turn_count - 1)
    intermediate_summary_count = sum(bool(payload) for payload in summary_payloads[:expected_summary_turns])
    intermediate_summary_coverage = (
        float(intermediate_summary_count / expected_summary_turns)
        if expected_summary_turns > 0
        else 0.0
    )
    final_turn_has_summary = bool(summary_payloads[-1]) if summary_payloads else False
    perfect_summary_chain = (
        expected_summary_turns > 0
        and intermediate_summary_count == expected_summary_turns
        and not final_turn_has_summary
    )

    if turn_count == 1:
        structure_reward = -0.05
    else:
        structure_reward = 0.08 * intermediate_summary_coverage
        structure_reward += 0.05 if not final_turn_has_summary else -0.05
        if turn_count == 2:
            structure_reward += 0.08
        elif turn_count == 3:
            structure_reward += 0.03
        else:
            structure_reward -= 0.03 * float(min(turn_count - 3, 2))
        if perfect_summary_chain:
            structure_reward += 0.04
    structure_reward = float(max(-0.05, min(0.2, structure_reward)))

    if exact_match > 0.0:
        shaped_structure_reward = structure_reward
        reward = float(exact_match + shaped_structure_reward)
    else:
        positive_structure = max(0.0, structure_reward)
        negative_structure = min(0.0, structure_reward)
        shaped_structure_reward = float((0.1 * positive_structure) + (0.25 * negative_structure))
        reward = float(shaped_structure_reward)

    return RewardResult(
        reward=reward,
        info={
            "exact_match": exact_match,
            "turn_count": float(turn_count),
            "summary_turn_count": float(sum(bool(payload) for payload in summary_payloads)),
            "expected_summary_turns": float(expected_summary_turns),
            "intermediate_summary_count": float(intermediate_summary_count),
            "intermediate_summary_coverage": float(intermediate_summary_coverage),
            "final_turn_has_summary": float(final_turn_has_summary),
            "perfect_summary_chain": float(perfect_summary_chain),
            "structure_reward": float(structure_reward),
            "shaped_structure_reward": float(shaped_structure_reward),
            "has_boxed_answer": float(getattr(request.parsed_response, "has_boxed_answer", False)),
            "has_answer_tag": float(getattr(request.parsed_response, "has_answer_tag", False)),
        },
    )


@dataclass(slots=True)
class MathQAEnv(BaseEnv):
    reward_fn: RewardFn
    task_name: str = "math_qa"
    prompt_key: str = "question"
    target_key: str = "parsed_answer"
    require_thinking_answer_format: bool = False
    require_single_final_answer: bool = False

    def build_prompt(self, row: Mapping[str, Any]) -> str:
        question = str(row.get(self.prompt_key, row.get("problem", row.get("prompt", ""))))
        if not question:
            raise KeyError("MathQAEnv requires one of: question, problem, or prompt")
        if self.require_thinking_answer_format:
            instruction = (
                "Solve the following problem. The assistant response is already "
                "prefilled with an opening <thinking> tag. Continue the reasoning "
                "inside that block, close it with </thinking>, then emit exactly "
                "<answer>final answer</answer>. Put only the final answer inside "
                "<answer>, and do not add text outside these blocks."
            )
        else:
            instruction = (
                "Solve the following problem. You may show reasoning, "
                "but the final answer must appear in \\boxed{...}."
            )
        return f"{instruction}\n\nProblem:\n{question}"

    def build_target(self, row: Mapping[str, Any]) -> str:
        if self.target_key in row and row.get(self.target_key) is not None:
            return normalize_math_answer(row.get(self.target_key))
        return normalize_math_answer(row.get("answer", ""))

    def parse_response(self, response: str) -> ParsedMathQAResponse:
        raw_response = str(response).strip()
        answer_matches = list(_ANSWER_TAG_RE.finditer(raw_response))
        answer_tag_answer = answer_matches[-1].group(1).strip() if answer_matches else ""
        boxed_answer = _extract_last_boxed_answer(raw_response)
        fallback_answer = _extract_last_non_empty_line(raw_response)
        final_answer = normalize_math_answer(
            answer_tag_answer if self.require_single_final_answer
            else (boxed_answer or answer_tag_answer or fallback_answer)
        )
        format_match = _PREFILLED_THINKING_ANSWER_RE.fullmatch(raw_response)
        strict_thinking_answer_format = bool(
            format_match
            and len(answer_matches) == 1
            and format_match.group("thinking").strip()
            and format_match.group("answer").strip()
            and "<thinking" not in raw_response.lower()
            and raw_response.lower().count("</thinking>") == 1
        )
        return ParsedMathQAResponse(
            final_answer=final_answer,
            raw_response=raw_response,
            boxed_answer=boxed_answer,
            answer_tag_answer=answer_tag_answer,
            has_boxed_answer=bool(boxed_answer),
            has_answer_tag=bool(answer_tag_answer),
            answer_tag_count=len(answer_matches),
            strict_thinking_answer_format=strict_thinking_answer_format,
        )

    def build_feedback(
        self,
        example: EnvExample,
        response: str,
        *,
        parsed_response: ParsedMathQAResponse | None = None,
    ) -> EnvFeedback:
        parsed = self.parse_response(response) if parsed_response is None else parsed_response
        success = str(parsed.final_answer) == str(example.target)
        valid_final_answer = True
        if self.require_single_final_answer:
            components = _single_final_answer_components(parsed, example.target)
            success = bool(components["exact_match"])
            valid_final_answer = bool(components["single_final_answer_valid"])
        if success:
            message = "Correct final answer."
        elif not valid_final_answer:
            message = "Emit exactly one complete <answer>...</answer> block at the end."
        elif not parsed.final_answer:
            message = (
                "No final answer found. Put the final answer in \\boxed{...} "
                "or <answer>...</answer>."
            )
        else:
            message = (
                "Incorrect final answer. "
                f"Expected `{example.target}` but received `{parsed.final_answer}`."
            )
        return EnvFeedback(
            done=True,
            success=success,
            message=message,
            metadata={
                "has_boxed_answer": int(parsed.has_boxed_answer),
                "has_answer_tag": int(parsed.has_answer_tag),
            },
        )
