"""Exact Knights and Knaves semantics, solver, prompts, and terminal rewards.

One Boolean variable per person: True means knight. A public statement is
consistent precisely when its truth value equals its speaker's role. No Python
eval, cached answer, model judge, or assumption about the number of knights is
used. The small supported worlds can be enumerated exhaustively.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import itertools
import json
import re
from typing import Mapping


OPERATORS = {"and", "or", "xor", "implies", "iff"}
COMMUTATIVE = {"and", "or", "xor", "iff"}
MAX_PEOPLE = 10


@dataclass(frozen=True, slots=True)
class Expr:
    op: str
    args: tuple["Expr", ...] = ()
    person: int = -1

    @classmethod
    def from_json(cls, value, *, depth=0):
        if depth > 8 or not isinstance(value, (list, tuple)) or not value:
            raise ValueError("Invalid or excessively nested statement")
        op = value[0]
        if op in ("knight", "knave") and len(value) == 2 and type(value[1]) is int:
            return cls(op, person=value[1])
        arity = 1 if op == "not" else 2 if op in OPERATORS else -1
        if len(value) != arity + 1:
            raise ValueError(f"Invalid statement operator or arity: {op!r}")
        return cls(op, tuple(cls.from_json(v, depth=depth + 1) for v in value[1:]))

    def to_json(self):
        return [self.op, self.person] if not self.args else [self.op, *(a.to_json() for a in self.args)]

    def evaluate(self, roles):
        if self.op in ("knight", "knave"):
            return bool(roles[self.person]) == (self.op == "knight")
        a = self.args[0].evaluate(roles)
        if self.op == "not":
            return not a
        b = self.args[1].evaluate(roles)
        return {"and": a and b, "or": a or b, "xor": a != b,
                "implies": (not a) or b, "iff": a == b}[self.op]

    def negated(self):
        if self.op in ("knight", "knave"):
            return Expr("knave" if self.op == "knight" else "knight", person=self.person)
        if self.op == "not":
            return self.args[0]
        a, b = self.args
        if self.op in ("and", "or"):
            return Expr("or" if self.op == "and" else "and", (a.negated(), b.negated()))
        if self.op == "implies":
            return Expr("and", (a, b.negated()))
        return Expr("xor" if self.op == "iff" else "iff", self.args)

    def references(self):
        return {self.person} if not self.args else set().union(*(a.references() for a in self.args))

    def render(self, people):
        if self.op in ("knight", "knave"):
            return f"{people[self.person]} is a {self.op}"
        a = self.args[0].render(people)
        if self.op == "not":
            return f"it is not the case that ({a})"
        b = self.args[1].render(people)
        return {"and": f"({a}) and ({b})", "or": f"({a}) or ({b})",
                "xor": f"exactly one of ({a}) and ({b}) is true",
                "implies": f"if ({a}), then ({b})",
                "iff": f"({a}) if and only if ({b})"}[self.op]


@dataclass(frozen=True, slots=True)
class Puzzle:
    people: tuple[str, ...]
    statements: tuple[Expr, ...]  # statement i is spoken by person i

    def __post_init__(self):
        if not 2 <= len(self.people) <= MAX_PEOPLE or len(self.statements) != len(self.people):
            raise ValueError("Require 2..10 people and exactly one statement per person")
        if len({n.casefold() for n in self.people}) != len(self.people):
            raise ValueError("Names must be distinct, ignoring case")
        if any(not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", n) for n in self.people):
            raise ValueError("Names must be single identifiers")
        for expr in self.statements:
            # Roundtrip validates manually constructed expressions as well.
            if Expr.from_json(expr.to_json()) != expr:
                raise ValueError("Invalid expression")
            if any(not 0 <= i < len(self.people) for i in expr.references()):
                raise ValueError("Statement references an unknown person")

    @classmethod
    def from_row(cls, row):
        # Intentionally ignore prompt, target, solution and audit fields.
        return cls(tuple(row["people"]), tuple(Expr.from_json(s) for s in row["statements"]))

    def to_row(self):
        return dict(people=list(self.people), statements=[s.to_json() for s in self.statements])

    @property
    def identity(self):
        return hashlib.sha256(json.dumps(self.to_row(), sort_keys=True).encode()).hexdigest()


def roles_for(mask, count):
    return tuple(bool(mask & (1 << i)) for i in range(count))


def consistency_masks(puzzle):
    """Bit j is set for statement i iff world j satisfies that statement."""
    masks = [0] * len(puzzle.people)
    for world in range(1 << len(puzzle.people)):
        roles = roles_for(world, len(puzzle.people))
        for i, expr in enumerate(puzzle.statements):
            if roles[i] == expr.evaluate(roles):
                masks[i] |= 1 << world
    return tuple(masks)


def solve(puzzle):
    candidates = (1 << (1 << len(puzzle.people))) - 1
    for mask in consistency_masks(puzzle):
        candidates &= mask
    return [roles_for(i, len(puzzle.people)) for i in range(1 << len(puzzle.people)) if candidates & (1 << i)]


def unique_solution(puzzle):
    solutions = solve(puzzle)
    if len(solutions) != 1:
        raise ValueError(f"Expected one solution, found {len(solutions)}")
    return solutions[0]


def answer_text(puzzle, roles=None):
    roles = unique_solution(puzzle) if roles is None else roles
    return ", ".join(f"{name}={'knight' if role else 'knave'}" for name, role in zip(puzzle.people, roles))


def solver_certificate(puzzle):
    """Every rejected world has a concrete statement it contradicts."""
    roles = unique_solution(puzzle)
    rejected = []
    for world in range(1 << len(puzzle.people)):
        assignment = roles_for(world, len(puzzle.people))
        if assignment == roles:
            continue
        i = next(i for i, expr in enumerate(puzzle.statements) if assignment[i] != expr.evaluate(assignment))
        rejected.append(dict(world=world, violated_statement=i))
    return dict(solution=list(roles), rejected_worlds=rejected)


def check_solver_certificate(puzzle, certificate):
    """Replay the exhaustive certificate locally, without calling solve()."""
    try:
        solution = tuple(certificate["solution"])
        if len(solution) != len(puzzle.people) or any(type(r) is not bool for r in solution):
            return False
        if any(solution[i] != e.evaluate(solution) for i, e in enumerate(puzzle.statements)):
            return False
        selected = sum(int(v) << i for i, v in enumerate(solution))
        seen = {selected}
        for rejected in certificate["rejected_worlds"]:
            world, i = rejected["world"], rejected["violated_statement"]
            if type(world) is not int or type(i) is not int or world in seen:
                return False
            if not 0 <= world < (1 << len(puzzle.people)) or not 0 <= i < len(puzzle.people):
                return False
            roles = roles_for(world, len(puzzle.people))
            if roles[i] == puzzle.statements[i].evaluate(roles):
                return False
            seen.add(world)
        return len(seen) == 1 << len(puzzle.people)
    except (KeyError, TypeError, ValueError, IndexError):
        return False


def minimum_unique_clues(puzzle, masks=None):
    """Exact minimum count of public statements needed to identify the world."""
    masks = consistency_masks(puzzle) if masks is None else masks
    for size in range(1, len(masks) + 1):
        for subset in itertools.combinations(masks, size):
            candidates = (1 << (1 << len(puzzle.people))) - 1
            for mask in subset:
                candidates &= mask
            if candidates and candidates.bit_count() == 1:
                return size
    return None


def render_prompt(puzzle):
    statements = "\n".join(f'{name} says: "{expr.render(puzzle.people)}."'
                           for name, expr in zip(puzzle.people, puzzle.statements))
    format_example = ", ".join(f"{n}=knight/knave" for n in puzzle.people)
    return (
        "Solve this Knights and Knaves puzzle. Every person is either a knight or a knave. "
        "Knights always make true statements; knaves always make false statements. "
        "Each person makes exactly one complete statement. For a knave, the whole statement is false; "
        "its individual parts need not all be false. There is no fixed number of knights or knaves.\n"
        "Logical 'or' includes the case where both parts are true. 'Exactly one' excludes that case. "
        "'If P then Q' is false only when P is true and Q is false. "
        "'P if and only if Q' means P and Q have the same truth value.\n\n"
        f"People: {', '.join(puzzle.people)}.\n{statements}\n\n"
        "There is exactly one solution. Determine every person's role using only these rules and statements. "
        "Explain your reasoning inside <thinking>...</thinking>, then give exactly one final "
        f"<answer>{format_example}</answer> block, replacing each knight/knave with one role. "
        "Include every person exactly once and write nothing after </answer>."
    )


@dataclass(frozen=True, slots=True)
class ParsedAnswer:
    assignments: tuple[tuple[str, bool], ...] = ()
    error: str | None = None


def parse_response(response):
    text = str(response).strip()
    for terminal in ("<|im_end|>", "<|endoftext|>"):
        if text.endswith(terminal):
            text = text[:-len(terminal)].rstrip()
    if text.count("<answer>") != 1 or text.count("</answer>") != 1:
        return ParsedAnswer(error="require_one_answer_block")
    match = re.search(r"<answer>(.*?)</answer>\s*$", text, re.DOTALL)
    if not match:
        return ParsedAnswer(error="nonterminal_or_unclosed_answer")
    # An answer hidden inside an unclosed thinking block is not terminal.
    prefix = text[:match.start()]
    if prefix.count("<thinking>") > 1 or prefix.count("</thinking>") > 1:
        return ParsedAnswer(error="multiple_thinking_blocks")
    if "<thinking>" in prefix and "</thinking>" not in prefix:
        return ParsedAnswer(error="unclosed_thinking")
    body = match.group(1).strip()
    entries = list(re.finditer(r"([A-Za-z][A-Za-z0-9_]*)\s*=\s*(knight|knave)\b", body, re.IGNORECASE))
    if not entries or body[:entries[0].start()].strip() or body[entries[-1].end():].strip():
        return ParsedAnswer(error="invalid_role_assignment")
    assignments = []
    previous_end = None
    for entry in entries:
        if previous_end is not None and not re.fullmatch(r"[\s,;]+", body[previous_end:entry.start()]):
            return ParsedAnswer(error="invalid_role_assignment")
        name, role = entry.groups()
        assignments.append((name.casefold(), role.lower() == "knight"))
        previous_end = entry.end()
    if len({n for n, _ in assignments}) != len(assignments):
        return ParsedAnswer(error="duplicate_person")
    return ParsedAnswer(tuple(assignments))


def score(response, row: Mapping):
    puzzle = Puzzle.from_row(row)
    solution = unique_solution(puzzle)
    expected = {n.casefold(): r for n, r in zip(puzzle.people, solution)}
    parsed = parse_response(response)
    submitted = dict(parsed.assignments)
    complete = parsed.error is None and submitted.keys() == expected.keys()
    exact = complete and submitted == expected
    consistency = None
    if complete:
        roles = tuple(submitted[n.casefold()] for n in puzzle.people)
        consistency = sum(roles[i] == e.evaluate(roles) for i, e in enumerate(puzzle.statements)) / len(roles)
    return dict(reward=float(exact), exact_match=exact, format_valid=complete,
                person_accuracy=sum(submitted.get(n) == r for n, r in expected.items()) / len(expected),
                statement_consistency=consistency,
                error=parsed.error or (None if complete else "missing_or_unknown_people"),
                reward_scope="terminal_roles_only; natural_language_reasoning_is_not_verified")
