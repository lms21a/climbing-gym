"""Preserve native sampler data needed for importance weighting and auditing."""
from types import SimpleNamespace

import torch

from climbing_gym.inference.types import BehaviorTopK, GenerationOutput, GenerationResult
from climbing_gym.rollout.contracts import build_trajectory_turn


def test_native_logprobs_and_behavior_head_survive_turn_conversion():
    head = BehaviorTopK(torch.tensor([[7, 8], [9, 10]]), torch.tensor([[-.1, -.2], [-.3, -.4]]))
    output = GenerationOutput(0, 'answer', (7, 9), (-.11, -.31), 'stop', None, head)
    result = GenerationResult('r', 'prompt', (1, 2), (output,), True, None)
    turn = build_trajectory_turn(turn_index=1, prompt='prompt', response='answer', result=result,
                                 metadata={'policy_objective_weight': .5})
    assert turn.prompt_token_ids == (1, 2)
    assert turn.completion_token_ids == (7, 9)
    assert turn.completion_logprobs == (-.11, -.31)
    assert turn.behavior_topk is head
    assert turn.metadata['policy_objective_weight'] == .5


def test_absent_logprobs_are_distinct_from_empty_completion_logprobs():
    for logprobs in (None, ()):
        result = SimpleNamespace(prompt_token_ids=(1,), outputs=(SimpleNamespace(
            token_ids=(), completion_logprobs=logprobs, finish_reason='length', stop_reason=None),))
        turn = build_trajectory_turn(turn_index=1, prompt='p', response='', result=result, metadata={})
        assert turn.completion_logprobs == logprobs
