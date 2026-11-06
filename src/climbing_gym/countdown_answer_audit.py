"""Format-tolerant final-expression diagnostics, separate from the frozen tag score."""
from __future__ import annotations

import re
from typing import Any, Mapping

from climbing_gym.countdown import score_expression, score_response


def score_countdown_answer(response: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Accept a terminal answer tag, terminal boxed math, or a bare final expression.

    Only the last explicitly submitted answer is considered. In particular, a correct
    expression earlier in the scratchpad cannot rescue an incorrect final answer.
    Box contents are simple arithmetic, not arbitrary LaTeX or executable code.
    """
    text = re.sub(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$", "", response.strip()).strip()
    tags = list(re.finditer(r"<answer>\s*(.*?)\s*</answer>", text, re.S | re.I))
    boxed = re.search(r"\\boxed\{([^{}]*)\}\s*\$?\s*[.;]?\s*$", text)
    if tags and not text[tags[-1].end():].strip():
        expression = tags[-1].group(1).strip()
        kind = "answer_tag"
        wrapped = re.fullmatch(r"\$?\s*\\boxed\{([^{}]*)\}\s*\$?", expression)
        if wrapped:
            expression = wrapped.group(1)
    elif boxed:
        expression, kind = boxed.group(1), "boxed"
    elif tags:
        # A completed answer followed by prose is not a terminal expression.
        expression, kind = "", "nonterminal_answer"
    else:
        expression = text.splitlines()[-1] if text else ""
        expression = re.sub(r"^(?:final answer|answer)\s*:\s*", "", expression.strip(), flags=re.I)
        expression, kind = expression.strip("`$"), "bare_final_line"
    for latex, ascii_text in ((r"\times", "*"), (r"\cdot", "*"), (r"\div", "/"),
                              (r"\left", ""), (r"\right", "")):
        expression = expression.replace(latex, ascii_text)
    equation_valid = True
    if "=" in expression:
        parts = expression.split("=")
        equation_valid = len(parts) == 2 and parts[1].strip() == str(row["target"])
        expression = parts[0].strip()
    score = score_expression(expression, row)
    return {**score, "exact": bool(score["exact"] and equation_valid),
            "answer_kind": kind, "expression": expression, "equation_valid": equation_valid,
            "strict_format_exact": score_response(response, row)["exact"]}
