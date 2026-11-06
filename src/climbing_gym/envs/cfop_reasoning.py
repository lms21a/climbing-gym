from __future__ import annotations

from dataclasses import dataclass
import json
import re
from typing import Any, Mapping, Sequence

from climbing_gym.envs.cfop_support import (
    F2L_PAIR_ORDER,
    CFOPPhase,
    ParsedCFOPResponse,
    parse_cfop_phase_response,
    parse_cfop_phase_response_strict,
    parse_cfop_response,
    parse_cfop_response_strict,
)


CFOP_REASONING_PROTOCOL = "cfop-reasoning/v1"
CFOP_PHASE_REASONING_PROTOCOL = "cfop-phase-reasoning/v1"
CFOP_REASONING_HEADER = "CFOP_PLAN_V1"

_STEP_RE = re.compile(
    r"^STEP (?P<index>[1-9][0-9]*)"
    r" \| stage=(?P<stage>[^|]+)"
    r" \| from=(?P<from>[a-z0-9]+)"
    r" \| to=(?P<to>[a-z0-9]+)"
    r" \| moves=(?P<moves>[0-9]+)"
    r" \| gain=(?P<gain>[-A-Z,]+)"
    r" \| preserve=(?P<preserve>[-A-Z,]+)"
    r" \| break=(?P<break>[-A-Z,]+)$"
)


@dataclass(frozen=True, slots=True)
class CFOPReasoningStep:
    index: int
    stage_label: str
    from_milestone: str
    to_milestone: str
    move_count: int
    gained_pairs: tuple[str, ...] = ()
    preserved_pairs: tuple[str, ...] = ()
    broken_pairs: tuple[str, ...] = ()

    @property
    def canonical_line(self) -> str:
        return (
            f"STEP {self.index} | stage={self.stage_label} "
            f"| from={self.from_milestone} | to={self.to_milestone} "
            f"| moves={self.move_count} "
            f"| gain={_render_pairs(self.gained_pairs)} "
            f"| preserve={_render_pairs(self.preserved_pairs)} "
            f"| break={_render_pairs(self.broken_pairs)}"
        )


@dataclass(frozen=True, slots=True)
class ParsedCFOPReasoning:
    raw_response: str
    valid: bool
    reasoning_text: str = ""
    answer_text: str = ""
    steps: tuple[CFOPReasoningStep, ...] = ()
    answer: ParsedCFOPResponse | None = None
    canonical_text: str = ""
    error: str | None = None
    error_code: str | None = None
    error_line: int | None = None


def _ordered_pairs(values: Sequence[str]) -> tuple[str, ...]:
    normalized = {str(value).strip().upper() for value in values}
    invalid = normalized.difference(F2L_PAIR_ORDER)
    if invalid:
        raise ValueError(f"invalid F2L pair labels: {sorted(invalid)!r}")
    return tuple(pair for pair in F2L_PAIR_ORDER if pair in normalized)


def _render_pairs(values: Sequence[str]) -> str:
    pairs = _ordered_pairs(values)
    return ",".join(pairs) if pairs else "-"


def _parse_pairs(value: str) -> tuple[str, ...]:
    text = str(value).strip()
    if text == "-":
        return ()
    raw = text.split(",")
    if not raw or any(not part for part in raw):
        raise ValueError("pair sets must be '-' or comma-separated labels")
    pairs = _ordered_pairs(raw)
    if tuple(raw) != pairs:
        raise ValueError(
            "pair labels must be unique and use canonical FR,FL,BL,BR order"
        )
    return pairs


def _phase_stage_label(phase: CFOPPhase) -> str:
    if phase.stage == "cross":
        pairs = _render_pairs(phase.f2l_pairs)
        return "CROSS" if pairs == "-" else f"CROSS[{pairs}]"
    if phase.stage == "f2l_pair":
        return f"F2L {_render_pairs(phase.f2l_pairs)}"
    return phase.stage.upper()


def _reasoning_error(
    raw_response: str,
    message: str,
    *,
    code: str,
    line: int | None = None,
    reasoning_text: str = "",
    answer_text: str = "",
    steps: Sequence[CFOPReasoningStep] = (),
    answer: ParsedCFOPResponse | None = None,
) -> ParsedCFOPReasoning:
    prefix = f"line {line}: " if line is not None else ""
    return ParsedCFOPReasoning(
        raw_response=str(raw_response),
        valid=False,
        reasoning_text=str(reasoning_text),
        answer_text=str(answer_text),
        steps=tuple(steps),
        answer=answer,
        error=f"{prefix}{message}",
        error_code=code,
        error_line=line,
    )


def _parse_cfop_reasoning_response(
    response: str,
    *,
    phase_local: bool,
) -> ParsedCFOPReasoning:
    """Strictly parse a grounded reasoning envelope and its CFOP answer."""

    raw = str(response)
    if not raw.startswith("<think>\n"):
        return _reasoning_error(
            raw,
            "response must begin with '<think>'",
            code="missing_think_open",
            line=1,
        )
    if raw.count("<think>") != 1:
        return _reasoning_error(
            raw,
            "response must contain exactly one '<think>' tag",
            code="duplicate_think_open",
        )
    if raw.count("</think>") != 1:
        return _reasoning_error(
            raw,
            "response must contain exactly one '</think>' tag",
            code="missing_or_duplicate_think_close",
        )

    separator = "\n</think>\n"
    if separator not in raw:
        return _reasoning_error(
            raw,
            "'</think>' must occupy its own line and be followed by the answer",
            code="invalid_think_close",
        )
    reasoning_text, answer_text = raw[len("<think>\n") :].split(
        separator,
        1,
    )
    if not reasoning_text:
        return _reasoning_error(
            raw,
            "reasoning block must not be empty",
            code="empty_reasoning",
        )
    if not answer_text:
        return _reasoning_error(
            raw,
            "strict CFOP answer is missing after '</think>'",
            code="missing_answer",
            reasoning_text=reasoning_text,
        )
    if "<think>" in reasoning_text or "</think>" in reasoning_text:
        return _reasoning_error(
            raw,
            "nested think tags are not allowed",
            code="nested_think",
            reasoning_text=reasoning_text,
            answer_text=answer_text,
        )

    lines = reasoning_text.splitlines()
    if not lines or lines[0] != CFOP_REASONING_HEADER:
        return _reasoning_error(
            raw,
            f"reasoning must begin with {CFOP_REASONING_HEADER}",
            code="missing_reasoning_header",
            line=2,
            reasoning_text=reasoning_text,
            answer_text=answer_text,
        )
    if len(lines) == 1:
        return _reasoning_error(
            raw,
            "reasoning plan must contain at least one STEP",
            code="missing_reasoning_steps",
            reasoning_text=reasoning_text,
            answer_text=answer_text,
        )

    steps: list[CFOPReasoningStep] = []
    for offset, line in enumerate(lines[1:], start=3):
        if not line:
            return _reasoning_error(
                raw,
                "blank lines are not allowed in the reasoning plan",
                code="reasoning_blank_line",
                line=offset,
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                steps=steps,
            )
        match = _STEP_RE.fullmatch(line)
        if match is None:
            return _reasoning_error(
                raw,
                "invalid CFOP_PLAN_V1 STEP grammar",
                code="invalid_reasoning_step",
                line=offset,
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                steps=steps,
            )
        try:
            gained_pairs = _parse_pairs(match.group("gain"))
            preserved_pairs = _parse_pairs(match.group("preserve"))
            broken_pairs = _parse_pairs(match.group("break"))
        except ValueError as exc:
            return _reasoning_error(
                raw,
                str(exc),
                code="invalid_reasoning_pairs",
                line=offset,
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                steps=steps,
            )
        step = CFOPReasoningStep(
            index=int(match.group("index")),
            stage_label=match.group("stage").strip(),
            from_milestone=match.group("from"),
            to_milestone=match.group("to"),
            move_count=int(match.group("moves")),
            gained_pairs=gained_pairs,
            preserved_pairs=preserved_pairs,
            broken_pairs=broken_pairs,
        )
        if step.index != len(steps) + 1:
            return _reasoning_error(
                raw,
                "STEP indices must be contiguous and start at 1",
                code="invalid_reasoning_step_index",
                line=offset,
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                steps=steps,
            )
        steps.append(step)

    answer = (
        parse_cfop_phase_response(answer_text)
        if phase_local
        else parse_cfop_response(answer_text)
    )
    if not answer.valid:
        return _reasoning_error(
            raw,
            (
                "invalid strict stage-local CFOP answer: "
                if phase_local
                else "invalid strict CFOP answer: "
            )
            + str(answer.error),
            code="invalid_reasoning_answer",
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            steps=steps,
            answer=answer,
        )
    if len(steps) != len(answer.phases):
        return _reasoning_error(
            raw,
            "reasoning STEP count must equal the number of answer phases",
            code="reasoning_phase_count_mismatch",
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            steps=steps,
            answer=answer,
        )
    for step, phase in zip(steps, answer.phases, strict=True):
        expected_label = _phase_stage_label(phase)
        if step.stage_label != expected_label:
            return _reasoning_error(
                raw,
                f"STEP {step.index} stage must be {expected_label!r}",
                code="reasoning_stage_mismatch",
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                steps=steps,
                answer=answer,
            )
        if step.move_count != phase.move_count:
            return _reasoning_error(
                raw,
                f"STEP {step.index} move count does not match its answer phase",
                code="reasoning_move_count_mismatch",
                reasoning_text=reasoning_text,
                answer_text=answer_text,
                steps=steps,
                answer=answer,
            )

    canonical_reasoning = "\n".join(
        (CFOP_REASONING_HEADER, *(step.canonical_line for step in steps))
    )
    canonical_text = (
        f"<think>\n{canonical_reasoning}\n</think>\n"
        f"{answer.canonical_text}"
    )
    if raw != canonical_text:
        return _reasoning_error(
            raw,
            "reasoning response is not in canonical form",
            code="noncanonical_reasoning",
            reasoning_text=reasoning_text,
            answer_text=answer_text,
            steps=steps,
            answer=answer,
        )
    return ParsedCFOPReasoning(
        raw_response=raw,
        valid=True,
        reasoning_text=reasoning_text,
        answer_text=answer_text,
        steps=tuple(steps),
        answer=answer,
        canonical_text=canonical_text,
    )


def parse_cfop_reasoning_response(response: str) -> ParsedCFOPReasoning:
    """Parse a grounded full-program reasoning response."""

    return _parse_cfop_reasoning_response(response, phase_local=False)


def parse_cfop_phase_reasoning_response(
    response: str,
) -> ParsedCFOPReasoning:
    """Parse a grounded one-phase reasoning response."""

    return _parse_cfop_reasoning_response(response, phase_local=True)


def reasoning_grounding_issues(
    parsed: ParsedCFOPReasoning,
    prefix_progress: Sequence[Mapping[str, Any]],
) -> tuple[Mapping[str, Any], ...]:
    """Compare every plan claim to verifier-derived phase endpoints."""

    if not parsed.valid or parsed.answer is None:
        return (
            {
                "kind": "invalid_reasoning_envelope",
                "error": parsed.error,
            },
        )
    if len(prefix_progress) != len(parsed.steps) + 1:
        return (
            {
                "kind": "missing_reasoning_prefix_diagnostics",
                "expected": len(parsed.steps) + 1,
                "actual": len(prefix_progress),
            },
        )

    issues: list[Mapping[str, Any]] = []
    for step_index, step in enumerate(parsed.steps):
        before = prefix_progress[step_index].get("progress", {})
        after = prefix_progress[step_index + 1].get("progress", {})
        before_map = before if isinstance(before, Mapping) else {}
        after_map = after if isinstance(after, Mapping) else {}
        before_pairs = _ordered_pairs(
            tuple(before_map.get("solved_pairs", ()) or ())
        )
        after_pairs = _ordered_pairs(
            tuple(after_map.get("solved_pairs", ()) or ())
        )
        actual_gain = tuple(
            pair for pair in F2L_PAIR_ORDER
            if pair in set(after_pairs).difference(before_pairs)
        )
        actual_preserve = tuple(
            pair for pair in F2L_PAIR_ORDER
            if pair in set(after_pairs).intersection(before_pairs)
        )
        actual_break = tuple(
            pair for pair in F2L_PAIR_ORDER
            if pair in set(before_pairs).difference(after_pairs)
        )
        expected = {
            "from_milestone": str(before_map.get("milestone", "none")),
            "to_milestone": str(after_map.get("milestone", "none")),
            "gained_pairs": actual_gain,
            "preserved_pairs": actual_preserve,
            "broken_pairs": actual_break,
        }
        actual = {
            "from_milestone": step.from_milestone,
            "to_milestone": step.to_milestone,
            "gained_pairs": step.gained_pairs,
            "preserved_pairs": step.preserved_pairs,
            "broken_pairs": step.broken_pairs,
        }
        if actual != expected:
            issues.append(
                {
                    "kind": "reasoning_grounding_mismatch",
                    "step": step.index,
                    "stage": step.stage_label,
                    "expected": expected,
                    "actual": actual,
                }
            )
    return tuple(issues)


def render_cfop_reasoning_completion(row: Mapping[str, Any]) -> str:
    """Render a canonical plan from normalized, verifier-audited row data."""

    answer_text = str(row.get("completion", ""))
    answer = parse_cfop_response_strict(answer_text)
    audits_value = row.get("source_stage_boundary_audits_json", "[]")
    audits = (
        json.loads(audits_value)
        if isinstance(audits_value, str)
        else audits_value
    )
    if not isinstance(audits, Sequence) or isinstance(
        audits,
        (str, bytes, bytearray),
    ):
        raise TypeError("source_stage_boundary_audits_json must be a sequence")
    if len(audits) != len(answer.phases):
        raise ValueError(
            "stage-boundary audit count must match answer phase count"
        )

    steps: list[CFOPReasoningStep] = []
    for index, (phase, audit_value) in enumerate(
        zip(answer.phases, audits, strict=True),
        start=1,
    ):
        if not isinstance(audit_value, Mapping):
            raise TypeError("each stage-boundary audit must be a mapping")
        audit = dict(audit_value)
        if str(audit.get("declared_stage", "")) != phase.stage:
            raise ValueError("audit stage does not match answer phase")
        if int(audit.get("move_count", -1)) != phase.move_count:
            raise ValueError("audit move count does not match answer phase")
        if _ordered_pairs(audit.get("f2l_pairs", ())) != _ordered_pairs(
            phase.f2l_pairs
        ):
            raise ValueError("audit F2L pair labels do not match answer phase")
        steps.append(
            CFOPReasoningStep(
                index=index,
                stage_label=_phase_stage_label(phase),
                from_milestone=str(audit.get("start_milestone", "none")),
                to_milestone=str(audit.get("end_milestone", "none")),
                move_count=phase.move_count,
                gained_pairs=_ordered_pairs(audit.get("gained_pairs", ())),
                preserved_pairs=_ordered_pairs(
                    audit.get("preserved_pairs", ())
                ),
                broken_pairs=_ordered_pairs(audit.get("broken_pairs", ())),
            )
        )

    reasoning = "\n".join(
        (CFOP_REASONING_HEADER, *(step.canonical_line for step in steps))
    )
    rendered = f"<think>\n{reasoning}\n</think>\n{answer.canonical_text}"
    parsed = parse_cfop_reasoning_response(rendered)
    if not parsed.valid or parsed.canonical_text != rendered:
        raise ValueError(
            f"rendered reasoning completion failed validation: {parsed.error}"
        )
    return rendered


def render_cfop_phase_reasoning_completion(
    row: Mapping[str, Any],
) -> str:
    """Render one verifier-grounded stage-local reasoning target."""

    answer_text = str(row.get("completion", ""))
    answer = parse_cfop_phase_response_strict(answer_text)
    if len(answer.phases) != 1:
        raise ValueError("stage-local reasoning requires exactly one phase")
    phase = answer.phases[0]

    audits_value = row.get("source_stage_boundary_audits_json", "[]")
    audits = (
        json.loads(audits_value)
        if isinstance(audits_value, str)
        else audits_value
    )
    if not isinstance(audits, Sequence) or isinstance(
        audits,
        (str, bytes, bytearray),
    ):
        raise TypeError("source_stage_boundary_audits_json must be a sequence")
    stage_index = int(row.get("stage_index", -1))
    if stage_index < 0 or stage_index >= len(audits):
        raise ValueError("stage-local row has no matching boundary audit")
    audit_value = audits[stage_index]
    if not isinstance(audit_value, Mapping):
        raise TypeError("stage boundary audit must be a mapping")
    audit = dict(audit_value)
    if str(audit.get("declared_stage", "")) != phase.stage:
        raise ValueError("audit stage does not match stage-local answer")
    if int(audit.get("move_count", -1)) != phase.move_count:
        raise ValueError("audit move count does not match stage-local answer")
    if _ordered_pairs(audit.get("f2l_pairs", ())) != _ordered_pairs(
        phase.f2l_pairs
    ):
        raise ValueError("audit F2L pair labels do not match stage-local answer")

    step = CFOPReasoningStep(
        index=1,
        stage_label=_phase_stage_label(phase),
        from_milestone=str(audit.get("start_milestone", "none")),
        to_milestone=str(audit.get("end_milestone", "none")),
        move_count=phase.move_count,
        gained_pairs=_ordered_pairs(audit.get("gained_pairs", ())),
        preserved_pairs=_ordered_pairs(audit.get("preserved_pairs", ())),
        broken_pairs=_ordered_pairs(audit.get("broken_pairs", ())),
    )
    reasoning = f"{CFOP_REASONING_HEADER}\n{step.canonical_line}"
    rendered = f"<think>\n{reasoning}\n</think>\n{answer.canonical_text}"
    parsed = parse_cfop_phase_reasoning_response(rendered)
    if not parsed.valid or parsed.canonical_text != rendered:
        raise ValueError(
            "rendered phase reasoning completion failed validation: "
            f"{parsed.error}"
        )
    return rendered


__all__ = [
    "CFOP_PHASE_REASONING_PROTOCOL",
    "CFOP_REASONING_HEADER",
    "CFOP_REASONING_PROTOCOL",
    "CFOPReasoningStep",
    "ParsedCFOPReasoning",
    "parse_cfop_phase_reasoning_response",
    "parse_cfop_reasoning_response",
    "reasoning_grounding_issues",
    "render_cfop_phase_reasoning_completion",
    "render_cfop_reasoning_completion",
]
