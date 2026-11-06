from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any

from climbing_gym.envs.contracts import EnvExample
from climbing_gym.rollout.contracts import HarnessRequest, HarnessState, build_trajectory_turn


_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL | re.IGNORECASE)
_SUMMARY_RE = re.compile(r"<summary>\s*(.*?)\s*</summary>", re.DOTALL | re.IGNORECASE)

_INFINITY_THINK_INSTRUCTION = (
    "You are solving a problem over multiple turns.\n"
    "Rules:\n"
    "1) Every turn must begin with exactly one <think>...</think> block.\n"
    "2) If you are still working, immediately follow </think> with exactly one <summary>...</summary> block.\n"
    "3) If you are finished, do not emit a <summary> block.\n"
    "4) On the final turn, immediately follow </think> with the answer/explanation in plain text.\n"
    "5) Continue from the latest summary state when history is provided.\n"
    "6) Each summary should be concise and carry forward only the key state needed next turn.\n"
)


def clean_summary_payload(text: str) -> str:
    payload = str(text or "").strip()
    if not payload:
        return ""
    payload = payload.replace("<summary>", "").replace("</summary>", "")
    lines = [line.strip() for line in payload.splitlines() if line.strip()]
    return "\n".join(lines).strip()


def extract_summary_payload(text: str) -> str:
    matches = list(_SUMMARY_RE.finditer(str(text)))
    if not matches:
        return ""
    return clean_summary_payload(matches[-1].group(1))


def _strip_tag_payload(text: str) -> str:
    return str(text or "").strip()


def _analyze_turn_format(text: str) -> dict[str, Any]:
    raw_text = str(text or "")
    think_matches = list(_THINK_RE.finditer(raw_text))
    summary_matches = list(_SUMMARY_RE.finditer(raw_text))

    first_think = think_matches[0] if think_matches else None
    first_summary = summary_matches[0] if summary_matches else None
    think_payload = _strip_tag_payload(first_think.group(1)) if first_think else ""
    summary_payload = clean_summary_payload(first_summary.group(1)) if first_summary else ""

    text_before_think = raw_text[: first_think.start()] if first_think else raw_text
    text_after_think = raw_text[first_think.end() :] if first_think else ""
    text_between_think_and_summary = (
        raw_text[first_think.end() : first_summary.start()]
        if first_think is not None and first_summary is not None
        else ""
    )
    text_after_summary = raw_text[first_summary.end() :] if first_summary is not None else ""

    has_text_before_think = bool(text_before_think.strip())
    has_text_between_think_and_summary = bool(text_between_think_and_summary.strip())
    has_text_after_summary = bool(text_after_summary.strip())
    has_text_after_think = bool(text_after_think.strip())
    has_answer_after_think = bool(text_after_think.strip()) and first_summary is None
    starts_with_think = bool(first_think is not None and not has_text_before_think)

    strict_intermediate_format = (
        len(think_matches) == 1
        and len(summary_matches) == 1
        and starts_with_think
        and bool(think_payload)
        and bool(summary_payload)
        and first_summary is not None
        and first_think is not None
        and first_summary.start() >= first_think.end()
        and not has_text_between_think_and_summary
        and not has_text_after_summary
    )
    strict_final_format = (
        len(think_matches) == 1
        and len(summary_matches) == 0
        and starts_with_think
        and bool(think_payload)
        and has_answer_after_think
    )

    return {
        "summary_payload": summary_payload,
        "has_think": bool(think_matches),
        "has_summary": bool(summary_matches),
        "think_count": int(len(think_matches)),
        "summary_count": int(len(summary_matches)),
        "starts_with_think": starts_with_think,
        "has_text_before_think": has_text_before_think,
        "has_text_between_think_and_summary": has_text_between_think_and_summary,
        "has_text_after_summary": has_text_after_summary,
        "has_text_after_think": has_text_after_think,
        "has_answer_after_think": has_answer_after_think,
        "strict_intermediate_format": strict_intermediate_format,
        "strict_final_format": strict_final_format,
    }


def _build_history_block(history_summaries: list[str]) -> str:
    if not history_summaries:
        return ""
    lines = ["<history>"]
    for index, summary in enumerate(history_summaries, start=1):
        lines.append(f"[{index}] {summary}")
    lines.append("</history>")
    return "\n".join(lines)


@dataclass(slots=True)
class _InfinityThinkState(HarnessState):
    next_turn_index: int = 1
    history_summaries: list[str] = field(default_factory=list)


class InfinityThinkHarness:
    def __init__(self, *, max_turns: int) -> None:
        if max_turns <= 0:
            raise ValueError("max_turns must be > 0")
        self.max_turns = int(max_turns)

    def create_state(self, example: EnvExample, rollout_index: int) -> _InfinityThinkState:
        del example
        state = _InfinityThinkState(
            rollout_index=int(rollout_index),
            metadata={"harness": "infinity_think"},
        )
        self._refresh_metadata(state)
        return state

    def _refresh_metadata(self, state: _InfinityThinkState) -> None:
        state.metadata["harness"] = "infinity_think"
        state.metadata["history_summaries"] = tuple(state.history_summaries)
        state.metadata["next_turn_index"] = int(state.next_turn_index)

    def _build_prompt(self, example: EnvExample, state: _InfinityThinkState) -> str:
        history_block = _build_history_block(state.history_summaries)
        parts = [
            _INFINITY_THINK_INSTRUCTION.strip(),
            "",
            "Problem:",
            str(example.prompt).strip(),
        ]
        if history_block:
            parts.extend(["", history_block])
        parts.extend(
            [
                "",
                f"Turn {state.next_turn_index}/{self.max_turns}.",
                "Required format if still working: <think>...</think><summary>...</summary>",
                "Required format if finished: <think>...</think> followed by the answer/explanation in plain text.",
                "Do not place any text before <think>.",
                "Do not include a <summary> block on the final turn.",
                "Summaries should keep only the partial results and open questions needed for the next turn.",
            ]
        )
        return "\n".join(parts).strip()

    def next_request(self, example: EnvExample, state: _InfinityThinkState) -> HarnessRequest | None:
        if state.finished:
            return None
        return HarnessRequest(prompt=self._build_prompt(example, state))

    def apply_result(self, example: EnvExample, state: _InfinityThinkState, result: Any) -> None:
        del example
        format_info = _analyze_turn_format(getattr(result, "text", ""))
        summary_payload = str(format_info["summary_payload"])
        turn = build_trajectory_turn(
            turn_index=int(state.next_turn_index),
            prompt=str(getattr(result, "prompt", "")),
            response=str(getattr(result, "text", "")),
            result=result,
            metadata={
                "harness": "infinity_think",
                "summary": summary_payload,
                "history_length": int(len(state.history_summaries)),
                "has_think": bool(format_info["has_think"]),
                "has_summary": bool(format_info["has_summary"]),
                "think_count": int(format_info["think_count"]),
                "summary_count": int(format_info["summary_count"]),
                "starts_with_think": bool(format_info["starts_with_think"]),
                "has_text_before_think": bool(format_info["has_text_before_think"]),
                "has_text_between_think_and_summary": bool(format_info["has_text_between_think_and_summary"]),
                "has_text_after_summary": bool(format_info["has_text_after_summary"]),
                "has_text_after_think": bool(format_info["has_text_after_think"]),
                "has_answer_after_think": bool(format_info["has_answer_after_think"]),
                "strict_intermediate_format": bool(format_info["strict_intermediate_format"]),
                "strict_final_format": bool(format_info["strict_final_format"]),
            },
        )
        state.turns.append(turn)
        state.final_response = turn.response

        if summary_payload:
            if state.next_turn_index >= self.max_turns:
                state.finished = True
                state.terminal_reason = "max_turns_reached"
                self._refresh_metadata(state)
                return
            state.history_summaries.append(summary_payload)
            state.next_turn_index += 1
            self._refresh_metadata(state)
            return

        state.finished = True
        state.terminal_reason = "response_complete"
        self._refresh_metadata(state)

    def force_finish(self, state: _InfinityThinkState, reason: str) -> None:
        if state.finished:
            return
        state.finished = True
        if not state.final_response and state.turns:
            state.final_response = state.turns[-1].response
        state.terminal_reason = str(reason)
        self._refresh_metadata(state)
