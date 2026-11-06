from itertools import permutations, product
import importlib.util
import json
from pathlib import Path

import pytest

from climbing_gym.countdown import CountdownTier, TIERS, generate_bank, score_expression, score_response, solve_countdown


def row(numbers=(2, 3, 4), target=14, operators="+-*/"):
    return {"numbers": list(numbers), "target": target, "operators": operators}


@pytest.mark.parametrize("expression", ["(3*4)+2", "2+4*3", "(2 + (3 × 4))"])
def test_accepts_alternative_solutions(expression):
    assert score_expression(expression, row())["exact"]


@pytest.mark.parametrize("expression", ["14", "2+3+4+5", "3*4+2+2", "23-4", "__import__('os').system('true')", "2**3+4", "2//3+4", "-2+3*4", "2.0+3*4", "True+3*4", "(3*4)+2 # hi"])
def test_rejects_illegal_or_executable_expressions(expression):
    assert not score_expression(expression, row())["legal"]


def test_duplicate_numbers_and_intermediate_rules():
    assert score_expression("(2+2)*3", row((2, 2, 3), 12))["exact"]
    assert not score_expression("2*3", row((2, 2, 3), 6))["legal"]
    assert not score_expression("(3-4)+2", row((3, 4, 2), 1))["legal"]
    assert not score_expression("(3/2)*4", row((3, 2, 4), 6))["legal"]
    assert score_expression("(4/2)*3", row((4, 2, 3), 6))["exact"]
    assert not score_expression("(3*4)+2", row(operators="+-"))["legal"]


def test_answer_contract_and_diagnostics_do_not_mine_scratchpad():
    assert score_response("Thinking.</thinking><answer>2+3*4</answer><|im_end|>", row())["exact"]
    assert score_response("<answer>2+3*4 = 14</answer>", row())["exact"]
    assert not score_response("<answer>2+3*4 = 15</answer>", row())["semantic_exact"]
    for text in ["2+3*4", "<answer>2+3*4</answer> more text", "<answer>2+3*4</answer><answer>2+3*4</answer>"]:
        result = score_response(text, row())
        assert result["semantic_exact"] and not result["exact"]
    assert not score_response("2+3*4\nI cannot solve it.", row())["semantic_exact"]
    assert not score_response("<answer>2+3*4</answer><answer>2+3+4</answer>", row())["semantic_exact"]
    assert not score_response("<answer><answer>2+3*4</answer>", row())["exact"]


def test_solver_agrees_with_independent_three_number_bruteforce():
    from fractions import Fraction
    ops = {"+": lambda a, b: a+b, "-": lambda a, b: a-b, "*": lambda a, b: a*b, "/": lambda a, b: a/b}
    for numbers in [(2, 3, 4), (1, 2, 2), (3, 6, 9)]:
        expected = set()
        for a, b, c in permutations(map(Fraction, numbers)):
            for op1, op2 in product(ops.values(), repeat=2):
                for nested_left in (True, False):
                    try:
                        intermediate = op1(a, b) if nested_left else op2(b, c)
                        if intermediate <= 0 or intermediate.denominator != 1:
                            continue
                        value = op2(intermediate, c) if nested_left else op1(a, intermediate)
                        if value > 0 and value.denominator == 1:
                            expected.add(int(value))
                    except ZeroDivisionError:
                        pass
        actual = solve_countdown(numbers)
        assert set(actual) == expected
        for target, expression in actual.items():
            assert score_expression(expression, row(numbers, target))["exact"]


def test_generator_is_deterministic_unique_and_obeys_tiers():
    for tier in TIERS:
        bank = generate_bank(tier, 3, 42, baseline_attempts=8)
        assert bank == generate_bank(tier, 3, 42, baseline_attempts=8)
        assert len({(tuple(sorted(r["numbers"])), r["target"]) for r in bank}) == 3
        for puzzle in bank:
            assert score_expression(puzzle["solution"], puzzle)["exact"]
            if tier.require_non_additive:
                assert puzzle["target"] not in solve_countdown(puzzle["numbers"], "+-")


def test_environment_registry_and_feedback_match_exact_reward():
    from climbing_gym.config import Env
    env = Env(name="countdown").build()
    example = env.build_example(row())
    assert "2, 3, 4" in example.prompt
    for response, correct in [("<answer>2+3*4</answer>", True), ("2+3*4", False), ("<answer>2+3+4</answer>", False)]:
        evaluated = env.evaluate_response(example, response)
        assert evaluated.feedback.success == correct
        assert evaluated.reward.reward == float(correct)
    with pytest.raises(ValueError):
        Env(name="countdown", reward="bogus").build()
