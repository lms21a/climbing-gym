"""Procedural integer Countdown with an exact solver and a non-executing verifier."""
from __future__ import annotations

import ast
from collections import Counter
from dataclasses import asdict, dataclass
from functools import lru_cache
import random
import re
from typing import Any, Mapping, Sequence


@dataclass(frozen=True)
class CountdownTier:
    name: str
    count: int
    operators: str
    min_number: int = 1
    max_number: int = 12
    min_target: int = 1
    max_target: int = 200
    require_non_additive: bool = False


TIERS = (
    CountdownTier("2_addsub", 2, "+-", max_target=24),
    CountdownTier("3_addsub", 3, "+-", max_target=36),
    CountdownTier("3_multiply", 3, "+-*", require_non_additive=True),
    CountdownTier("4_multiply", 4, "+-*", require_non_additive=True),
    CountdownTier("4_all", 4, "+-*/", require_non_additive=True),
    CountdownTier("5_all", 5, "+-*/", require_non_additive=True),
    CountdownTier("6_all", 6, "+-*/", require_non_additive=True),
)


def apply_operation(a: int, b: int, op: str) -> int | None:
    if op == "+":
        return a + b
    if op == "*":
        return a * b
    if op == "-":
        return a - b if a > b else None
    if op == "/":
        return a // b if b > 0 and a % b == 0 else None
    raise ValueError(f"Unsupported operation: {op}")


def solve_countdown(numbers: Sequence[int], operators: str = "+-*/") -> dict[int, str]:
    """All reachable targets, with one witness each; every intermediate is a positive integer.

    Subset DP considers each indexed input exactly once, including duplicate values.
    No intermediate-value cutoff or search heuristic is used. Limited to six inputs.
    """
    numbers = tuple(numbers)
    if not 2 <= len(numbers) <= 6 or any(type(n) is not int or not 1 <= n <= 100 for n in numbers):
        raise ValueError("Use 2–6 integer inputs in [1, 100]")
    if not operators or len(set(operators)) != len(operators) or set(operators) - set("+-*/"):
        raise ValueError("operators must be a nonempty subset of +-*/ without duplicates")

    @lru_cache(None)
    def visit(mask: int) -> dict[int, str]:
        if mask & (mask - 1) == 0:
            value = numbers[mask.bit_length() - 1]
            return {value: str(value)}
        out: dict[int, str] = {}
        left = (mask - 1) & mask
        while left:
            right = mask ^ left
            if left < right:
                for a, ae in visit(left).items():
                    for b, be in visit(right).items():
                        for op in operators:
                            value = apply_operation(a, b, op)
                            if value is not None and value not in out:
                                out[value] = f"({ae} {op} {be})"
                            if op in "-/":
                                value = apply_operation(b, a, op)
                                if value is not None and value not in out:
                                    out[value] = f"({be} {op} {ae})"
            left = (left - 1) & mask
        return out

    return visit((1 << len(numbers)) - 1)


def random_expression(numbers: Sequence[int], operators: str, rng: random.Random) -> tuple[int, str] | None:
    """Documented guessing baseline: uniform ordered pair and operator at each merge.

    Invalid moves end the attempt; no retry, lookahead, or target information.
    This is not uniform sampling over legal expressions.
    """
    items = [(n, str(n)) for n in numbers]
    while len(items) > 1:
        i, j = rng.sample(range(len(items)), 2)
        a, ae = items[i]
        b, be = items[j]
        op = rng.choice(operators)
        value = apply_operation(a, b, op)
        if value is None:
            return None
        items = [item for k, item in enumerate(items) if k not in (i, j)]
        items.append((value, f"({ae} {op} {be})"))
    return items[0]


def build_prompt(row: Mapping[str, Any]) -> str:
    return (
        f"Use the numbers {', '.join(map(str, row['numbers']))} to make {row['target']}.\n"
        "Use every given number exactly once, including repeated numbers. "
        f"Allowed operations: {' '.join(row['operators'])}. Parentheses are allowed. "
        "Every intermediate result must be a positive integer; division must be exact. "
        "Do not join digits or introduce extra numbers.\n"
        "Work through the problem, then put one final arithmetic expression inside "
        "<answer>...</answer>."
    )


def generate_bank(tier: CountdownTier, count: int, seed: int, baseline_attempts: int = 256) -> list[dict[str, Any]]:
    if count <= 0 or baseline_attempts <= 0:
        raise ValueError("count and baseline_attempts must be positive")
    rng = random.Random(seed)
    baseline_rng = random.Random(seed + 1_000_003)
    rows: list[dict[str, Any]] = []
    seen: set[tuple[tuple[int, ...], int]] = set()
    for _ in range(count * 100):
        numbers = [rng.randint(tier.min_number, tier.max_number) for _ in range(tier.count)]
        solutions = solve_countdown(numbers, tier.operators)
        excluded = solve_countdown(numbers, "+-") if tier.require_non_additive else {}
        targets = sorted(t for t in solutions if tier.min_target <= t <= tier.max_target and t not in excluded)
        rng.shuffle(targets)
        target = next((t for t in targets if (tuple(sorted(numbers)), t) not in seen), None)
        if target is None:
            continue
        seen.add((tuple(sorted(numbers)), target))
        baseline_correct = 0
        baseline_valid = 0
        for _ in range(baseline_attempts):
            attempt = random_expression(numbers, tier.operators, baseline_rng)
            baseline_valid += attempt is not None
            baseline_correct += attempt is not None and attempt[0] == target
        row = {
            "id": f"{tier.name}:{len(rows):04d}", "tier": tier.name,
            "numbers": numbers, "target": target, "operators": tier.operators,
            "solution": solutions[target], "tier_config": asdict(tier),
            "reachable_target_count": len(solutions), "eligible_target_count": len(targets),
            "random_baseline_attempts": baseline_attempts,
            "random_baseline_correct": baseline_correct, "random_baseline_valid": baseline_valid,
        }
        row["prompt"] = build_prompt(row)
        if not score_expression(row["solution"], row)["exact"]:
            raise AssertionError("Solver witness failed independent AST verification")
        rows.append(row)
        if len(rows) == count:
            return rows
    raise RuntimeError(f"Could not generate {count} unique puzzles for {tier.name}")


def score_expression(expression: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Parse a small arithmetic AST; never eval/exec model-generated text."""
    result = {"syntax_valid": False, "numbers_correct": False, "operations_allowed": False,
              "intermediates_valid": False, "legal": False, "exact": False,
              "value": None, "error": "invalid_syntax"}
    expression = expression.strip().replace("×", "*").replace("÷", "/").replace("−", "-")
    if not expression or len(expression) > 512 or not re.fullmatch(r"[0-9+*/()\s-]+", expression):
        return result
    try:
        tree = ast.parse(expression, mode="eval")
    except (SyntaxError, ValueError, RecursionError):
        return result
    nodes = list(ast.walk(tree))
    allowed = (ast.Expression, ast.BinOp, ast.Constant, ast.Add, ast.Sub, ast.Mult, ast.Div)
    if len(nodes) > 128 or any(not isinstance(node, allowed) for node in nodes):
        return result
    constants = [node.value for node in nodes if isinstance(node, ast.Constant)]
    if any(type(n) is not int or n <= 0 or n > 100 for n in constants):
        return result
    result["syntax_valid"] = True
    result["numbers_correct"] = Counter(constants) == Counter(row["numbers"])
    symbol = {ast.Add: "+", ast.Sub: "-", ast.Mult: "*", ast.Div: "/"}
    result["operations_allowed"] = all(symbol[type(node.op)] in row["operators"] for node in nodes if isinstance(node, ast.BinOp))
    if not result["numbers_correct"]:
        result["error"] = "wrong_numbers"
        return result
    if not result["operations_allowed"]:
        result["error"] = "forbidden_operation"
        return result

    def calculate(node: ast.AST) -> int:
        if isinstance(node, ast.Constant):
            return node.value
        value = apply_operation(calculate(node.left), calculate(node.right), symbol[type(node.op)])
        if value is None:
            raise ValueError("nonpositive or fractional intermediate")
        return value

    try:
        result["value"] = calculate(tree.body)
    except (ValueError, RecursionError):
        result["error"] = "invalid_intermediate"
        return result
    result.update(intermediates_valid=True, legal=True, exact=result["value"] == row["target"])
    result["error"] = "" if result["exact"] else "wrong_target"
    return result


_ANSWER = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.S | re.I)
_EOS = re.compile(r"(?:<\|im_end\|>|<\|endoftext\|>)\s*$")


def score_response(response: str, row: Mapping[str, Any]) -> dict[str, Any]:
    """Strict terminal-answer score plus separately labeled format-tolerant diagnostics.

    Never search the scratchpad for a correct candidate. The diagnostic fallback uses
    only the last nonempty line; multiple answer blocks use the last block diagnostically.
    """
    text = _EOS.sub("", response.strip()).strip()
    matches = list(_ANSWER.finditer(text))
    opens = len(re.findall(r"<\s*answer\b", text, re.I))
    closes = len(re.findall(r"<\s*/\s*answer\b", text, re.I))
    terminal = bool(len(matches) == opens == closes == 1 and not text[matches[0].end():].strip())
    expression = matches[-1].group(1).strip() if matches else (text.splitlines()[-1].strip() if text else "")
    if not matches:
        expression = re.sub(r"^(?:final answer|answer)\s*:\s*", "", expression, flags=re.I)
    expression = expression.strip("`$").strip()
    equation_valid = True
    if "=" in expression:
        parts = expression.split("=")
        equation_valid = len(parts) == 2 and parts[1].strip() == str(row["target"])
        expression = parts[0].strip()
    result = score_expression(expression, row)
    semantic_exact = bool(result["exact"] and equation_valid)
    return {**result, "exact": bool(semantic_exact and terminal), "semantic_exact": semantic_exact,
            "single_terminal_answer": terminal, "answer_tag_count": len(matches),
            "equation_valid": equation_valid, "expression": expression}
