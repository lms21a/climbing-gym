"""Seeded puzzle generation and name-invariant held-out grouping."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import random

from climbing_gym.knights_knaves import (
    COMMUTATIVE, Expr, Puzzle, answer_text, check_solver_certificate,
    consistency_masks, minimum_unique_clues, roles_for, solver_certificate, unique_solution,
)


TIERS = {
    "easy": dict(people=3, max_depth=1, operators=["and", "or"]),
    "medium": dict(people=4, max_depth=2, operators=["and", "or", "implies"]),
    "hard": dict(people=6, max_depth=2, operators=["and", "or", "implies", "xor", "iff"]),
    "expert": dict(people=8, max_depth=3, operators=["and", "or", "implies", "xor", "iff"]),
}
NAMES = tuple("Ari Bea Cai Dara Eli Faye Gus Hana Ivo Juno Kai Lea Milo Nora Oren Pia Quin Rhea Sol Tia Uma Vera Wes Xena Yves Zara".split())


def _hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def structural_signature(puzzle):
    """Conservative graph-color fingerprint, invariant to names and person order.

    Commutative child order is ignored. Color-refinement collisions are grouped
    together, so they can reduce sample diversity but cannot leak isomorphic
    renamed instances between splits. This is not Boolean-equivalence grouping.
    """
    def pattern(expr, colors):
        if not expr.args:
            return (expr.op, colors[expr.person])
        children = [pattern(e, colors) for e in expr.args]
        if expr.op in COMMUTATIVE:
            children.sort(key=repr)
        return (expr.op, *children)
    colors = ["person"] * len(puzzle.people)
    for _ in range(len(colors) + 1):
        colors = [_hash((colors[i], pattern(e, colors))) for i, e in enumerate(puzzle.statements)]
    return _hash(sorted((colors[i], pattern(e, colors)) for i, e in enumerate(puzzle.statements)))


def _expression(rng, count, depth, operators):
    if depth == 0 or rng.random() < .20:
        return Expr(rng.choice(("knight", "knave")), person=rng.randrange(count))
    return Expr(rng.choice(operators), tuple(_expression(rng, count, depth - 1, operators) for _ in range(2)))


def _flip_roles(expr, flipped):
    if not expr.args:
        return expr.negated() if flipped[expr.person] else expr
    return Expr(expr.op, tuple(_flip_roles(e, flipped) for e in expr.args))


def _connected(puzzle):
    reached = {0}
    edges = [{i} | e.references() for i, e in enumerate(puzzle.statements)]
    while True:
        more = set().union(*(edge for edge in edges if edge & reached))
        if more <= reached:
            return len(reached) == len(puzzle.people)
        reached |= more


def generate_tier(rng: random.Random, tier: str, *, max_attempts=20000):
    if tier not in TIERS:
        raise ValueError(f"Unknown Knights and Knaves tier: {tier!r}")
    settings = TIERS[tier]
    count = settings["people"]
    all_worlds = (1 << (1 << count)) - 1
    for attempt in range(1, max_attempts + 1):
        statements = tuple(_expression(rng, count, settings["max_depth"], settings["operators"]) for _ in range(count))
        puzzle = Puzzle(tuple(NAMES[:count]), statements)
        if not _connected(puzzle):
            continue
        # Exclude direct truth/lie self-claims and constant utterances.
        if any(not e.args and e.person == i for i, e in enumerate(statements)):
            continue
        truth = [sum(e.evaluate(roles_for(w, count)) << w for w in range(1 << count)) for e in statements]
        if any(t in (0, all_worlds) for t in truth):
            continue
        masks = consistency_masks(puzzle)
        candidates = all_worlds
        for mask in masks:
            candidates &= mask
        if candidates.bit_count() != 1:
            continue
        minimum = minimum_unique_clues(puzzle, masks)
        if minimum < count - 1:
            continue
        # Independent per-person role flips preserve the solution set bijectively.
        # They make each final answer bit fair, removing an all-knight/knave prior.
        flipped = tuple(bool(rng.getrandbits(1)) for _ in range(count))
        changed = tuple(_flip_roles(e, flipped) for e in statements)
        changed = tuple(e.negated() if flipped[i] else e for i, e in enumerate(changed))
        # Randomize people, identifiers and statement order together.
        order = rng.sample(range(count), count)
        inverse = {old: new for new, old in enumerate(order)}
        def relabel(e):
            return Expr(e.op, tuple(relabel(a) for a in e.args), inverse[e.person] if not e.args else -1)
        puzzle = Puzzle(tuple(rng.sample(NAMES, count)), tuple(relabel(changed[i]) for i in order))
        certificate = solver_certificate(puzzle)
        if not check_solver_certificate(puzzle, certificate):
            raise RuntimeError("Local replay rejected an exhaustive solver certificate")
        solution = unique_solution(puzzle)
        row = puzzle.to_row()
        row.update(id=puzzle.identity, task="knights_knaves", tier=tier,
                   structural_group=structural_signature(puzzle), solution=list(solution),
                   target=answer_text(puzzle, solution),
                   audit=dict(unique_solution=True, solver_certificate_verified=True,
                              worlds=1 << count, minimum_unique_clues=minimum,
                              nominal_max_expression_depth=settings["max_depth"],
                              connected=True, role_balance="independent random role-complement symmetry",
                              generator_attempts=attempt))
        return row
    raise RuntimeError(f"Could not generate tier {tier!r} after {max_attempts} attempts")


def generate_splits(*, seed=20261030, train_per_tier=256, dev_per_tier=16,
                    test_per_tier=32, calibration_per_tier=32):
    counts = dict(calibration=calibration_per_tier, dev=dev_per_tier,
                  test=test_per_tier, train=train_per_tier)
    if any(type(n) is not int or n < 0 for n in counts.values()):
        raise ValueError("Split sizes must be nonnegative integers")
    data, reserved, ids = {}, set(), set()
    for split_index, (split, per_tier) in enumerate(counts.items()):
        panel = []
        for tier_index, tier in enumerate(TIERS):
            # Separate RNG per tier/split: changing training size cannot change evals.
            rng = random.Random(seed + split_index * 1000003 + tier_index * 10007)
            groups = set()
            attempts = 0
            while sum(r["tier"] == tier for r in panel) < per_tier:
                attempts += 1
                if attempts > max(10000, per_tier * 100):
                    raise RuntimeError(f"Insufficient distinct structures for {split}/{tier}")
                row = generate_tier(rng, tier)
                if row["structural_group"] in reserved or row["structural_group"] in groups or row["id"] in ids:
                    continue
                ids.add(row["id"])
                groups.add(row["structural_group"])
                row["split"] = split
                panel.append(row)
            reserved.update(groups)
        data[split] = panel
    return data


def summarize(rows):
    return dict(problems=len(rows), tiers=dict(Counter(r["tier"] for r in rows)),
                unique_ids=len({r["id"] for r in rows}),
                unique_structural_groups=len({r["structural_group"] for r in rows}),
                knight_fraction=sum(sum(r["solution"]) for r in rows) / max(1, sum(len(r["people"]) for r in rows)))
