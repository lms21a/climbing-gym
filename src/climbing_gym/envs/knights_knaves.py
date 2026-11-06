"""Single-turn adapter usable by the shared PPO/GRPO/FlashREINFORCE trainers."""
from dataclasses import dataclass

from climbing_gym.envs.contracts import BaseEnv, EnvFeedback, RewardRequest, RewardResult
from climbing_gym.knights_knaves import Puzzle, answer_text, parse_response, render_prompt, score


def exact_knights_knaves_reward(request: RewardRequest) -> RewardResult:
    info = score(request.response, request.example.row)
    return RewardResult(float(info["reward"]), info)


@dataclass(slots=True)
class KnightsKnavesEnv(BaseEnv):
    task_name: str = "knights_knaves"
    reward_fn: object = exact_knights_knaves_reward

    def build_prompt(self, row):
        return render_prompt(Puzzle.from_row(row))

    def build_target(self, row):
        return answer_text(Puzzle.from_row(row))

    def parse_response(self, response):
        return parse_response(response)

    def build_feedback(self, example, response, *, parsed_response=None):
        info = score(response, example.row)
        return EnvFeedback(done=True, success=bool(info["exact_match"]), metadata=info)
