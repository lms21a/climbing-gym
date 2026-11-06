"""Algorithmic guides for model-native Knights/Knaves prose projection."""
import re
from climbing_gym.knights_knaves import Puzzle, solver_certificate, check_solver_certificate, roles_for, answer_text, score


def evaluation_steps(expr, roles, names):
    if not expr.args:
        return f'{names[expr.person]} is {expr.op}={expr.evaluate(roles)}'
    children=[evaluation_steps(a,roles,names) for a in expr.args]
    return '('+f' {expr.op} '.join(children)+f')={expr.evaluate(roles)}' if expr.op!='not' else f'not({children[0]})={expr.evaluate(roles)}'


def solver_trace(row):
    puzzle=Puzzle.from_row(row);cert=solver_certificate(puzzle)
    if not check_solver_certificate(puzzle,cert):raise ValueError('Invalid independent certificate')
    lines=['Verified exhaustive consistency check (True=knight, False=knave):']
    for rejection in cert['rejected_worlds']:
        roles=roles_for(rejection['world'],len(puzzle.people));i=rejection['violated_statement']
        lines.append(f"Reject [{answer_text(puzzle,roles)}]: {puzzle.people[i]}'s statement: "
                     +evaluation_steps(puzzle.statements[i],roles,puzzle.people)
                     +f'; assigned role={roles[i]}, so statement and role disagree.')
    roles=tuple(cert['solution'])
    lines.append('Unique surviving assignment: '+answer_text(puzzle,roles))
    for name,expr,role in zip(puzzle.people,puzzle.statements,roles):
        lines.append(f'{name}: '+evaluation_steps(expr,roles,puzzle.people)+f'; assigned role={role}, agrees.')
    return '\n'.join(lines)


def answer_suffix(row):
    return '\n</thinking>\n<answer>'+answer_text(Puzzle.from_row(row))+'</answer>'


def conclusion_matches(prose, row):
    """Check the last explicit role commitments, allowing earlier exploration.

    This checks the conclusion, not every intermediate deduction. Missing or
    unparseable conclusions are rejected rather than silently repaired.
    """
    puzzle = Puzzle.from_row(row)
    expected = solver_certificate(puzzle)['solution']
    plain = prose.replace('**', '').replace('`', '')
    for name, role in zip(puzzle.people, expected, strict=True):
        matches = list(re.finditer(r'\b' + re.escape(name)
            + r'\s*(?:=|is|:)\s*(?:a\s+)?(knight|knave)\b', plain, re.IGNORECASE))
        if not matches or (matches[-1].group(1).lower() == 'knight') != role:
            return False
    return True
