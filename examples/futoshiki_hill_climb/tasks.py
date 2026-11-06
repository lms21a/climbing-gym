"""Exact precursor prompt/reward contracts, ported from the frozen Oct 2 runs.

Only the supplied policy's tokenizer is used. No original workspace or models
are imported. Futoshiki uses the package's strict solver-backed environment.
"""

from decimal import Decimal, InvalidOperation
import re
from climbing_gym.countdown import build_prompt, score_response
from climbing_gym.envs.contracts import BaseEnv, EnvFeedback, RewardResult
from climbing_gym.projection_sampling import FORMAT_INSTRUCTION as PLAN_INSTRUCTION

PREFILL = "<thinking>\n"
INSTRUCTION = """Solve the math question below.
Write your reasoning inside <thinking>...</thinking>.
After closing the thinking block, put your final numeric answer inside <answer>...</answer>.
End your response after </answer>.
"""
EOS_TEXT = ("<|im_end|>", "<|endoftext|>")
NUMBER = re.compile(r"[+-]?(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?\Z")
ANSWER = re.compile(r"<answer>(.*?)</answer>", re.S)
TAGS = re.compile(r"</?(?:thinking|answer)>")
FORMAT = re.compile(r"<thinking>(.*?)</thinking>\s*<answer>(.*?)</answer>\s*\Z", re.S)


def numeric(text):
    text = str(text).strip()
    if not NUMBER.fullmatch(text):
        return None
    try:
        return Decimal(text.replace(",", ""))
    except InvalidOperation:
        return None


def grade_gsm8k(response, target, *, format_weight=0.0, truncated=False):
    """Inspect the entire generated suffix: NEVER split or trim at </answer>."""
    full = PREFILL + str(response)
    clean = full.rstrip()
    # Only an actual terminal marker is ignorable. Interior EOS + text fails.
    for marker in EOS_TEXT:
        if clean.endswith(marker):
            clean = clean[: -len(marker)].rstrip()
            break
    matches = list(ANSWER.finditer(clean))
    opens, closes = clean.count("<answer>"), clean.count("</answer>")
    single = len(matches) == opens == closes == 1
    # Commit to the FIRST answer attempt. Later guesses cannot rescue a wrong
    # or malformed first attempt, and continuation cannot erase a correct one.
    first_open = clean.find("<answer>")
    first_close = clean.find("</answer>")
    first = (
        matches[0]
        if matches and matches[0].start() == first_open and first_open < first_close
        else None
    )
    predicted = numeric(first[1]) if first else None
    gold = numeric(str(target).rsplit("####", 1)[-1])
    if gold is None:
        raise ValueError("GSM8K target is not numeric")
    correct = bool(predicted is not None and predicted == gold)
    ordered = TAGS.findall(clean) == [
        "<thinking>",
        "</thinking>",
        "<answer>",
        "</answer>",
    ]
    match = FORMAT.fullmatch(clean)
    generated_reasoning = bool(
        match
        and match[1].strip() != PREFILL[len("<thinking>") :].strip()
        and match[1].strip()
    )
    valid = bool(
        ordered
        and match
        and generated_reasoning
        and predicted is not None
        and "<|" not in clean
        and not truncated
    )
    # Inspect after the FIRST closing tag, including malformed/duplicate answers.
    tail = clean.split("</answer>", 1)[1] if "</answer>" in clean else ""
    continued = bool(tail.strip())
    if continued:
        valid = False
    return dict(
        reward=float(correct and valid),
        exact_match=float(correct),
        reward_format=float(valid),
        correctness_reward=float(correct and valid),
        format_reward=0.0,
        format_valid=float(valid),
        continued_after_answer=float(continued),
        answer_tag_count=float(len(matches)),
        multiple_answers=float(opens > 1 or closes > 1),
        missing_answer_open=float(closes > 0 and opens == 0),
        single_answer_exact_match=float(single and correct),
        numeric_first_answer=float(predicted is not None),
        truncated=float(truncated),
        success=float(correct and valid),
    )


def grade_countdown(response, target, *, truncated=False):
    row = target
    finish_reason = "length" if truncated else "stop"
    result = score_response(response, row)
    text = response.rstrip()
    for marker in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(marker):
            text = text[: -len(marker)].rstrip()
            break
    match = re.fullmatch(r"(.*?)</thinking>\s*<answer>(.*?)</answer>\s*", text, re.S)
    ordered = re.findall(r"</?(?:thinking|answer)>", text) == [
        "</thinking>",
        "<answer>",
        "</answer>",
    ]
    valid = bool(
        match
        and match[1].strip()
        and match[2].strip()
        and ordered
        and "<|" not in text
        and finish_reason != "length"
    )
    return dict(
        result,
        exact_match=float(result["exact"]),
        format_valid=valid,
        joint_success=bool(valid and result["exact"]),
        success=float(valid and result["exact"]),
        reward=float(valid and result["exact"]),
        truncated=float(truncated),
    )


class HistoricalJointEnv(BaseEnv):
    def __init__(self, kind, tokenizer):
        if kind not in {"gsm8k", "countdown"}:
            raise ValueError("Expected a GSM8K or Countdown precursor")
        self.kind, self.tokenizer = kind, tokenizer
        self.task_name = kind
        self.reward_fn = self.reward

    def build_prompt(self, row):
        if self.kind == "gsm8k":
            content = (
                str(row["question"])
                + "\n\n"
                + PLAN_INSTRUCTION
                + " Provide only the final numeric answer inside the answer tags."
            )
        else:
            content = build_prompt(row) + "\n\n" + PLAN_INSTRUCTION
        return (
            self.tokenizer.apply_chat_template(
                [dict(role="user", content=content)],
                tokenize=False,
                add_generation_prompt=True,
            )
            + PREFILL
        )

    def build_target(self, row):
        if self.kind == "gsm8k":
            return str(row["answer"]).rsplit("####", 1)[-1].strip()
        return {key: row[key] for key in ["numbers", "operators", "target"]}

    def reward(self, request):
        truncated = any(
            m.get("finish_reason") == "length" for m in request.turn_metadata
        )
        grade = grade_gsm8k if self.kind == "gsm8k" else grade_countdown
        info = grade(request.response, request.example.target, truncated=truncated)
        return RewardResult(reward=float(info["reward"]), info=info)

    def build_feedback(self, example, response, *, parsed_response=None):
        grade = grade_gsm8k if self.kind == "gsm8k" else grade_countdown
        return EnvFeedback(
            done=True, success=bool(grade(response, example.target)["success"])
        )
