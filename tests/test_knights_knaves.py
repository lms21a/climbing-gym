"""Independent semantics, generator, split-leakage and reward boundary checks."""
from dataclasses import replace
import itertools
import random

import pytest

from climbing_gym.knights_knaves import (
    Expr, Puzzle, answer_text, check_solver_certificate, consistency_masks,
    minimum_unique_clues, parse_response, render_prompt, roles_for, score,
    solve, solver_certificate, unique_solution,
)
from climbing_gym.knights_knaves_data import (
    TIERS, _flip_roles, generate_splits, generate_tier, structural_signature,
)


def reference_eval(node, roles):
    """Independent interpreter for the public JSON, not Expr.evaluate."""
    op, *children = node
    if op == "knight": return roles[children[0]]
    if op == "knave": return not roles[children[0]]
    values = [reference_eval(c, roles) for c in children]
    if op == "not": return not values[0]
    a, b = values
    if op == "and": return all(values)
    if op == "or": return any(values)
    if op == "xor": return sum(values) == 1
    if op == "implies": return not (a and not b)
    if op == "iff": return len(set(values)) == 1
    raise AssertionError(op)


def reference_solve(p):
    return [roles for roles in itertools.product((False, True), repeat=len(p.people))
            if all(roles[i] == reference_eval(e.to_json(), roles) for i, e in enumerate(p.statements))]


def classic():
    # Ari: both of us are knaves. Bea: Ari is a knave.
    return Puzzle(("Ari", "Bea"), (Expr.from_json(["and", ["knave", 0], ["knave", 1]]),
                                    Expr("knave", person=0)))


def test_classic_unique_ambiguous_and_inconsistent_cases():
    p = classic()
    assert solve(p) == [(False, True)]
    ambiguous = Puzzle(p.people, (Expr("knight", person=1), Expr("knight", person=0)))
    assert set(solve(ambiguous)) == {(False, False), (True, True)}
    impossible = replace(ambiguous, statements=(Expr("knave", person=0), ambiguous.statements[1]))
    assert solve(impossible) == []
    for bad in (ambiguous, impossible):
        with pytest.raises(ValueError, match="Expected one solution"):
            unique_solution(bad)


@pytest.mark.parametrize("op", ["and", "or", "xor", "implies", "iff", "not"])
def test_operator_truth_tables_and_negation(op):
    children = [["knight", 0]] if op == "not" else [["knight", 0], ["knave", 1]]
    expr = Expr.from_json([op, *children])
    for roles in itertools.product((False, True), repeat=2):
        assert expr.evaluate(roles) == reference_eval(expr.to_json(), roles)
        assert expr.negated().evaluate(roles) != expr.evaluate(roles)


@pytest.mark.parametrize("tier", list(TIERS))
def test_generated_solutions_match_independent_exhaustive_interpreter(tier):
    rng = random.Random(20261030)
    for _ in range(12):
        row = generate_tier(rng, tier)
        p = Puzzle.from_row(row)
        assert set(solve(p)) == set(reference_solve(p)) == {tuple(row["solution"])}
        assert minimum_unique_clues(p) == row["audit"]["minimum_unique_clues"] >= len(p.people) - 1
        assert check_solver_certificate(p, solver_certificate(p))
        assert score(f"<answer>{answer_text(p)}</answer>", row)["reward"] == 1


def test_solver_certificate_replays_without_solver_and_rejects_missing_or_fake_worlds(monkeypatch):
    p = classic()
    cert = solver_certificate(p)
    import climbing_gym.knights_knaves as module
    monkeypatch.setattr(module, "solve", lambda _: pytest.fail("Local certificate checker called solver"))
    assert check_solver_certificate(p, cert)
    assert not check_solver_certificate(p, dict(cert, rejected_worlds=cert["rejected_worlds"][:-1]))
    assert not check_solver_certificate(p, dict(cert, rejected_worlds=cert["rejected_worlds"] * 2))
    assert not check_solver_certificate(p, dict(cert, solution=[True, True]))
    corrupted = [dict(r) for r in cert["rejected_worlds"]]
    corrupted[0]["violated_statement"] = 99
    assert not check_solver_certificate(p, dict(cert, rejected_worlds=corrupted))


def test_role_complement_symmetry_is_bijective_for_every_assignment():
    p = Puzzle.from_row(generate_tier(random.Random(41), "hard"))
    flipped = (False, True, False, True, True, False)
    changed = tuple(_flip_roles(e, flipped) for e in p.statements)
    q = replace(p, statements=tuple(e.negated() if flipped[i] else e for i, e in enumerate(changed)))
    for world in range(1 << len(p.people)):
        roles = roles_for(world, len(p.people))
        swapped = tuple(a != b for a, b in zip(roles, flipped))
        assert all(roles[i] == e.evaluate(roles) for i, e in enumerate(p.statements)) == all(
            swapped[i] == e.evaluate(swapped) for i, e in enumerate(q.statements))


def test_structural_group_ignores_names_person_order_and_commutative_order():
    p = classic()
    changed = Puzzle(("Yves", "Zara"), (
        Expr("knave", person=1), Expr.from_json(["and", ["knave", 0], ["knave", 1]])))
    assert structural_signature(p) == structural_signature(changed)


def test_deterministic_disjoint_splits_and_eval_independent_of_training_size():
    args = dict(seed=73, train_per_tier=3, dev_per_tier=2, test_per_tier=3, calibration_per_tier=2)
    splits = generate_splits(**args)
    assert splits == generate_splits(**args)
    larger = generate_splits(**dict(args, train_per_tier=5))
    for split in ("calibration", "dev", "test"):
        assert splits[split] == larger[split]
    for a, b in itertools.combinations(splits.values(), 2):
        assert not {r["id"] for r in a} & {r["id"] for r in b}
        assert not {r["structural_group"] for r in a} & {r["structural_group"] for r in b}


@pytest.mark.parametrize("response", [
    "Ari=knave, Bea=knight",
    "<answer>Ari=knave, Ari=knave, Bea=knight</answer>",
    "<answer>Ari=knave</answer>",
    "<answer>Ari=knave, Bea=knight, Cai=knight</answer>",
    "<answer>Ari=knave, Bea=knight</answer> trailing",
    "<answer>Ari=knave, Bea=knight</answer><answer>Ari=knave, Bea=knight</answer>",
    "<answer>Ari=not knight, Bea=knight</answer>",
    "<answer>Ari=knaveBea=knight</answer>",
    "<answer>Ari=knave/knight Bea=knight</answer>",
    "<answer>Ari=knave wrong Bea=knight</answer>",
    "<thinking><answer>Ari=knave, Bea=knight</answer>",
    "<answer>Ari=knight, Bea=knave</answer>",
])
def test_terminal_reward_rejects_unsupported_or_malformed_answers(response):
    assert score(response, classic().to_row())["reward"] == 0


def test_native_prefill_and_formatting_variations_with_binary_terminal_reward():
    for response in (
        "<thinking>Reason.</thinking><answer>Ari=knave, Bea=knight</answer>",
        "Reason.</thinking><answer>bea=KNIGHT; ARI=KNAVE</answer><|im_end|>",
        "<answer>Bea=knight\nAri=knave</answer><|endoftext|>",
        "<answer>Bea=knight Ari=knave</answer>",
        "<answer>Bea = knight, Ari = knave</answer>",
    ):
        assert score(response, classic().to_row())["reward"] == 1
    partial = score("<answer>Ari=knave</answer>", classic().to_row())
    assert partial["person_accuracy"] == .5 and partial["reward"] == 0


def test_training_env_registry_and_cache_poisoning_do_not_change_public_task():
    from climbing_gym.config import Env
    p = classic()
    row = dict(p.to_row(), prompt="The answer is all knights", target="wrong", solution=[True, True])
    env = Env(name="knights_knaves").build()
    example = env.build_example(row)
    assert example.prompt == render_prompt(p) and "The answer is all knights" not in example.prompt
    assert example.target == "Ari=knave, Bea=knight"
    result = env.evaluate_response(example, "<answer>Ari=knave, Bea=knight</answer>")
    assert result.reward.reward == 1 and result.feedback.success and result.feedback.done
    assert result.reward.info["reward_scope"].startswith("terminal_roles_only")
    with pytest.raises(ValueError, match="only exact_match"):
        Env(name="knights_knaves", reward="dense").build()


@pytest.mark.parametrize("node", [["exec", "bad"], ["and", ["knight", 0]], ["knight", True], ["knight", -1]])
def test_malformed_public_expressions_are_rejected(node):
    with pytest.raises(ValueError):
        Puzzle(("Ari", "Bea"), (Expr.from_json(node), Expr("knight", person=0)))
