from types import SimpleNamespace
import torch
from torch import nn
from climbing_gym.envs.contracts import EnvEvaluation, EnvExample, RewardResult
from climbing_gym.rollout import Trajectory, TrajectoryTurn

class TinyPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        self.config = SimpleNamespace(hidden_size=8, pad_token_id=0)
        self.embed_tokens = nn.Embedding(16, 8)
        self.lm_head = nn.Linear(8, 16, bias=False)

    def get_hidden_states(self, input_ids, positions):
        return self.embed_tokens(input_ids)

def trajectory(reward=1.0, *, length=4, group=0, rollout=0):
    example = EnvExample(prompt="puzzle", target=12, row={}, dataset_index=group)
    evaluation = EnvEvaluation(example=example, response="answer", parsed_response=None,
                               feedback=None, reward=RewardResult(reward=reward))
    turn = TrajectoryTurn(turn_index=0, prompt="puzzle", response="answer", prompt_token_ids=(1, 2),
                          completion_token_ids=tuple(range(3, 3 + length)),
                          completion_logprobs=(-3.0,) * length, finish_reason="stop", stop_reason="eos")
    return Trajectory(group_id=group, rollout_index=rollout, example=example, turns=(turn,),
                      final_response="answer", finished=True, terminal_reason="single_turn", evaluation=evaluation)
