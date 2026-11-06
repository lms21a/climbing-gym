"""Deterministic, outcome-independent detection of sustained generation loops.

Three signals: stagnant revisits to arithmetic candidates, near-verbatim
third-and-later repeated 12-token spans, and short exact tandem repetitions.
Detection scans the entire response: a later answer cannot erase an earlier loop.
This is a repetition heuristic, not a semantic judge of reasoning progress.
"""
from __future__ import annotations

import ast
from bisect import bisect_left, bisect_right
from collections import Counter
import math
import re
from typing import Any, Sequence


DETECTOR_VERSION = "doomloop-v1.1"
_TRANSLATE = str.maketrans({"×": "*", "÷": "/", "−": "-", "–": "-"})
_TOKEN = re.compile(r"[\w]+|[+*/()=-]", re.UNICODE)
_CUE = r"\b(?:let['’]s try|let me try|how about|what if|what about|alternatively|maybe|consider|try)\b"
_BOUNDARY = re.compile(r"\n|(?=" + _CUE + r")", re.I)
_EXPLICIT = re.compile(_CUE, re.I)
_EXPRESSION = re.compile(r"[0-9()+*/\- \t]+")


def _canonical(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and type(node.value) is int:
        return str(node.value)
    if not isinstance(node, ast.BinOp) or type(node.op) not in (ast.Add, ast.Sub, ast.Mult, ast.Div):
        raise ValueError("Not integer arithmetic")
    symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}[type(node.op)]
    if symbol in "+*":
        # Normalize both operand order and associative regrouping, but never
        # merge different candidate expressions merely because values agree.
        def parts(child):
            if isinstance(child, ast.BinOp) and type(child.op) is type(node.op):
                return parts(child.left) + parts(child.right)
            return [_canonical(child)]
        return "(" + symbol.join(sorted(parts(node))) + ")"
    return f"({_canonical(node.left)}{symbol}{_canonical(node.right)})"


def arithmetic_candidates(text: str, numbers: Sequence[int] = ()) -> list[dict[str, Any]]:
    """Extract first substantial arithmetic expression per attempt/line.

    Consecutive explanatory copies collapse, while an explicit new attempt is
    retained. Expressions need not be correct or use the right multiplicities.
    Offsets refer to the original response (normalization is one-for-one).
    """
    normalized = text.translate(_TRANSLATE)
    boundaries = sorted({0, len(text), *(m.start() for m in _BOUNDARY.finditer(normalized))})
    output = []
    inputs = set(numbers)
    for start, end in zip(boundaries, boundaries[1:]):
        segment = normalized[start:end]
        explicit = bool(_EXPLICIT.match(segment.lstrip()))
        for match in _EXPRESSION.finditer(segment):
            expression = match.group().strip()
            if not 5 <= len(expression) <= 512 or not re.search(r"[+*/-]", expression):
                continue
            try:
                tree = ast.parse(expression, mode="eval")
                nodes = list(ast.walk(tree))
                literals = [node.value for node in nodes if isinstance(node, ast.Constant)]
                if len(nodes) > 128 or len(literals) < 3 or any(type(n) is not int for n in literals):
                    continue
                if inputs and sum(n in inputs for n in literals) < 2:
                    continue
                key = _canonical(tree.body)
            except (SyntaxError, ValueError, RecursionError):
                continue
            if explicit or not output or output[-1]["canonical"] != key:
                output.append({"expression": expression, "canonical": key,
                               "start": start + match.start(), "end": start + match.end()})
            break
    return output


def _candidate_signal(candidates):
    counts = Counter()
    repeats = []
    revisits = []
    for item in candidates:
        counts[item["canonical"]] += 1
        repeats.append(int(counts[item["canonical"]] >= 3))
        revisits.append(int(counts[item["canonical"]] >= 2))
    best, best_end, running, revisited = 0, 0, 0, 0
    for i, value in enumerate(repeats):
        running += value
        revisited += revisits[i]
        if i >= 16:
            running -= repeats[i - 16]
            revisited -= revisits[i - 16]
        # Exploration gets no penalty: all 16 candidates must be revisits.
        if i >= 15 and revisited == 16 and running > best:
            best, best_end = running, i + 1
    score = max(0.0, (best - 11) / 5)
    evidence = None
    if score:
        window = candidates[best_end - 16:best_end]
        evidence = {"start": window[0]["start"], "end": window[-1]["end"],
                    "third_or_later_visits": best, "window_candidates": 16,
                    "examples": [item['expression'] for item in window[-4:]]}
    return score, evidence


def _text_signal(tokens, spans, novel_candidates):
    if len(tokens) < 128:
        return 0.0, None
    seen = Counter()
    difference = [0] * (len(tokens) + 1)
    for i in range(len(tokens) - 11):
        gram = tuple(tokens[i:i + 12])
        seen[gram] += 1
        if seen[gram] >= 3:
            difference[i] += 1
            difference[i + 12] -= 1
    coverage = []
    active = 0
    for value in difference[:-1]:
        active += value
        coverage.append(int(active > 0))
    # Repeated boilerplate around a genuinely new arithmetic candidate must
    # not turn continued exploration into a text-loop penalty.
    starts = [s[0] for s in spans]
    novelty_difference = [0] * (len(tokens) + 1)
    for candidate in novel_candidates:
        left = bisect_left(starts, candidate['start'])
        right = bisect_right(starts, candidate['end'])
        novelty_difference[left] += 1
        novelty_difference[right] -= 1
    novelty = []
    active = 0
    for value in novelty_difference[:-1]:
        active += value
        novelty.append(int(active > 0))
    best, best_end, running, novel = 0, 0, 0, 0
    for i, value in enumerate(coverage):
        running += value
        novel += novelty[i]
        if i >= 128:
            running -= coverage[i - 128]
            novel -= novelty[i - 128]
        if i >= 127 and novel == 0 and running > best:
            best, best_end = running, i + 1
    score = min(1.0, max(0.0, (best / 128 - .95) / .05)) if best >= 122 else 0.0
    evidence = None if not score else {
        "start": spans[best_end - 128][0], "end": spans[best_end - 1][1],
        "repeated_tokens": best, "window_tokens": 128, "ngram_tokens": 12,
    }
    return score, evidence


def _tandem_signal(tokens, spans):
    best, evidence = 0.0, None
    # Short cycles are often too short to fill the text detector's 128-token
    # window. Require >= 48 tokens AND at least 8 complete copies of a unit.
    for period in range(1, 7):
        length = 0
        for i in range(period, len(tokens)):
            length = length + 1 if tokens[i] == tokens[i - period] else 0
            total = length + period
            if total >= 48 and total // period >= 8:
                # Formatting separators are not repeated reasoning.
                if not any(any(char.isalnum() for char in token) for token in tokens[i - period + 1:i + 1]):
                    continue
                score = min(1.0, (total - 32) / 64)
                if score > best:
                    best = score
                    evidence = {"start": spans[i - total + 1][0], "end": spans[i][1],
                                "period_tokens": period, "repeated_tokens": total}
    return best, evidence


def detect_doom_loop(response: str, numbers: Sequence[int] = ()) -> dict[str, Any]:
    """Return severity in [0, 1] and compact, inspectable evidence.

    No dependence on correctness, target, finish reason, or overall response
    length normalization. Arithmetic detection requires 16 consecutive old
    proposals, at least 12 visited for the third time or more. Text detection
    requires >=122/128 repeated lexical tokens with no new arithmetic proposal.
    Exact short cycles require >=48 lexical tokens. Not a paraphrase detector.
    """
    matches = list(_TOKEN.finditer(response.translate(_TRANSLATE)))
    tokens = [m.group().lower() for m in matches]
    spans = [(m.start(), m.end()) for m in matches]
    candidates = arithmetic_candidates(response, numbers)
    seen = set()
    novel_candidates = []
    for candidate in candidates:
        if candidate['canonical'] not in seen:
            novel_candidates.append(candidate)
            seen.add(candidate['canonical'])
    signals = {"arithmetic_cycle": _candidate_signal(candidates),
               "repeated_text": _text_signal(tokens, spans, novel_candidates),
               "short_cycle": _tandem_signal(tokens, spans)}
    scores = {name: score for name, (score, _) in signals.items()}
    score = max(scores.values())
    return {"version": DETECTOR_VERSION, "detected": score > 0, "score": score,
            "lexical_tokens": len(tokens), "candidate_count": len(candidates),
            "unique_candidates": len({c['canonical'] for c in candidates}),
            "signals": scores, "evidence": {name: evidence for name, (_, evidence) in signals.items() if evidence}}


def validate_penalty_weight(weight: float) -> float:
    weight = float(weight)
    if not math.isfinite(weight) or not 0 <= weight < 1:
        raise ValueError("doomloop_penalty_weight must be finite and in [0, 1)")
    return weight
