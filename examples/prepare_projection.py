"""Create a verified algorithmic Knights/Knaves guide without a teacher model.

Run: python examples/prepare_projection.py > experts.jsonl
Pass that file to `python -m climbing_gym.sampling --rows experts.jsonl ...`.
"""
import json
from climbing_gym.knights_knaves import Puzzle, answer_text, render_prompt
from climbing_gym.knights_solver_projection import solver_trace

# Alice says Bob is a knave. Bob says Alice is a knave or Bob is a knave.
# Exhaustive checking leaves Alice=knave, Bob=knight.
task = dict(people=['Alice', 'Bob'], statements=[['knave', 1], ['or', ['knave', 0], ['knave', 1]]])
puzzle = Puzzle.from_row(task)
answer = answer_text(puzzle)
print(json.dumps(dict(id='algorithmic-knights-example', tier='two_people', task=task,
    messages=[dict(role='user', content=render_prompt(puzzle))], reference_source='algorithmic',
    reference_answer=answer, expert='<thinking>\n'+solver_trace(task)+'\n</thinking>\n<answer>'+answer+'</answer>')))
