"""Deterministic selection and lossless Markdown reports for real trajectories."""
from __future__ import annotations

from collections import defaultdict
import json
from pathlib import Path
import re
from typing import Any, Iterable, Mapping, Sequence


def _fenced(text: Any) -> str:
    value = str(text)
    longest = max((len(match.group()) for match in re.finditer(r"`+", value)), default=0)
    fence = "`" * max(3, longest + 1)
    return f"{fence}text\n{value}\n{fence}"


def difficulty_tier(trajectory: Any) -> str:
    row = trajectory.example.row
    for key in ("difficulty", "difficulty_tier", "tier", "level", "num_people", "n", "size"):
        if key in row:
            return f"{key}={row[key]}"
    metadata = row.get("metadata", {})
    if isinstance(metadata, Mapping):
        for key in ("difficulty", "difficulty_tier", "tier", "level"):
            if key in metadata:
                return f"{key}={metadata[key]}"
    return "unspecified"


def trajectory_passed(trajectory: Any) -> bool:
    feedback = trajectory.evaluation.feedback
    if feedback is not None and feedback.success is not None:
        return bool(feedback.success)
    return float(trajectory.evaluation.reward.reward) > 0.0


def select_traces(trajectories: Iterable[Any], *, max_traces: int = 8) -> list[Any]:
    """Round-robin tier/outcome strata, deterministic regardless of finish order.

    The cap is a preference: at least one available trace in every tier/outcome
    stratum is included, so rare failures and difficulty tiers are never omitted.
    ``max_traces=0`` selects every trajectory.
    """
    if max_traces < 0:
        raise ValueError("max_traces must be >= 0")
    ordered = sorted(trajectories, key=lambda item: (
        -1 if item.example.dataset_index is None else int(item.example.dataset_index),
        int(item.group_id), int(item.rollout_index)))
    if max_traces == 0:
        return ordered
    strata: dict[tuple[str, bool], list[Any]] = defaultdict(list)
    for item in ordered:
        strata[(difficulty_tier(item), trajectory_passed(item))].append(item)
    selected: list[Any] = []
    keys = sorted(strata)
    limit = max(max_traces, len(keys))
    offset = 0
    while len(selected) < min(limit, len(ordered)):
        for key in keys:
            if offset < len(strata[key]):
                selected.append(strata[key][offset])
                if len(selected) >= min(limit, len(ordered)):
                    break
        offset += 1
    return selected


def write_trace_report(path: str | Path, trajectories: Sequence[Any], *,
                       summary: Mapping[str, Any] | None = None,
                       max_traces: int = 8, tokenizer: Any | None = None) -> Path:
    """Save complete per-turn prompts/responses, target, outcome and final response.

    Files use exclusive creation, preserving every periodic evaluation report.
    A colliding requested name gains a numeric suffix. Responses are never cut.
    """
    requested = Path(path).expanduser().resolve()
    requested.parent.mkdir(parents=True, exist_ok=True)
    traces = select_traces(trajectories, max_traces=max_traces)
    lines = ["# Evaluation traces", "", f"Inspected {len(traces)} of {len(trajectories)} actual trajectories.", ""]
    if summary is not None:
        lines += ["## Metrics", "", _fenced(json.dumps(dict(summary), indent=2, sort_keys=True, default=str)), ""]
    if not traces:
        lines += ["No trajectories were returned by the model.", ""]
    for index, trajectory in enumerate(traces, 1):
        passed = trajectory_passed(trajectory)
        lines += [f"## Trace {index}", "",
                  f"Dataset index: {trajectory.example.dataset_index}; rollout: {trajectory.rollout_index}.", "",
                  f"Difficulty: {difficulty_tier(trajectory)}.", "",
                  f"Outcome: {'PASS' if passed else 'FAIL'}; reward: {float(trajectory.evaluation.reward.reward):g}; "
                  f"terminal reason: {trajectory.terminal_reason}.", "",
                  "### Expected answer", "", _fenced(json.dumps(trajectory.example.target, ensure_ascii=False, indent=2, default=str)), "",
                  "### Original task prompt", "", _fenced(trajectory.example.prompt), ""]
        for turn in trajectory.turns:
            actual_prompt = (tokenizer.decode(list(turn.prompt_token_ids), skip_special_tokens=False)
                             if tokenizer is not None and turn.prompt_token_ids else turn.prompt)
            lines += [f"### Turn {turn.turn_index}: actual prompt", "", _fenced(actual_prompt), "",
                      f"### Turn {turn.turn_index}: complete model response", "", _fenced(turn.response), ""]
        lines += ["### Complete final model response", "", _fenced(trajectory.final_response), "",
                  "### Verifier details", "", _fenced(json.dumps(dict(trajectory.evaluation.reward.info),
                                                                       ensure_ascii=False, indent=2, default=str)), ""]
    suffix = 0
    while True:
        output = requested if suffix == 0 else requested.with_name(f"{requested.stem}-{suffix:04d}{requested.suffix}")
        try:
            handle = output.open("x", encoding="utf-8")
            break
        except FileExistsError:
            suffix += 1
    with handle:
        handle.write("\n".join(lines) + "\n")
    return output


__all__ = ["difficulty_tier", "select_traces", "trajectory_passed", "write_trace_report"]
