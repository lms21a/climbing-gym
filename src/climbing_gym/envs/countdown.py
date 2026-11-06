"""Countdown adapter for the shared evaluation and RL environment contracts."""
from dataclasses import dataclass
from typing import Any, Mapping

from climbing_gym.countdown import build_prompt, score_response
from climbing_gym.countdown_answer_audit import score_countdown_answer
from climbing_gym.doomloop import detect_doom_loop, validate_penalty_weight
from climbing_gym.envs.contracts import BaseEnv, EnvExample, EnvFeedback, RewardFn, RewardRequest, RewardResult


def exact_countdown_reward(request: RewardRequest) -> RewardResult:
    info = score_response(request.response, request.example.row)
    return RewardResult(reward=float(info["exact"]), info=info)


def terminal_expression_countdown_reward(request: RewardRequest) -> RewardResult:
    info = score_countdown_answer(request.response, request.example.row)
    return RewardResult(reward=float(info["exact"]), info=info)


def terminal_expression_doomloop_countdown_reward(
    request: RewardRequest, *, penalty_weight: float = 0.25, truncation_penalty_weight: float = 0.0,
) -> RewardResult:
    """Exact outcome minus repetition and optional reasoning-cutoff penalties.

    Correctness/pass@k remains in `exact` and feedback.success. Shaped reward is
    deliberately separate. With default weights, success remains positive even
    after both penalties; penalties also apply to failed attempts.
    """
    weight = validate_penalty_weight(penalty_weight)
    cap_weight = validate_penalty_weight(truncation_penalty_weight)
    info = score_countdown_answer(request.response, request.example.row)
    loop = detect_doom_loop(request.response, request.example.row["numbers"])
    # Every generated phase receives the final outcome; earlier loops cannot be
    # hidden by writing a clean summary or final answer in a later window.
    earlier = [m.get('doomloop') for m in request.turn_metadata if m.get('harness') == 'countdown_summary' and m.get('doomloop')]
    if earlier:
        loop = max([loop, *earlier], key=lambda item: item['score'])
    base = float(info["exact"])
    penalty = weight * loop["score"]
    cap_count = sum(bool(m.get('reasoning_cap_hit')) for m in request.turn_metadata)
    cap_penalty = min(0.25, cap_weight * cap_count)
    info.update(base_reward=base, doomloop_score=loop["score"], doomloop_penalty=penalty,
                doomloop=loop, reasoning_cap_count=cap_count, truncation_penalty=cap_penalty,
                summary_count=sum(m.get('phase') == 'summary' or bool(m.get('summary')) for m in request.turn_metadata),
                summary_failure_count=sum(bool(m.get('summary_failed')) for m in request.turn_metadata),
                forced_summary_count=sum(m.get('phase') == 'summary' and bool(m.get('forced_summary')) for m in request.turn_metadata))
    return RewardResult(reward=base - penalty - cap_penalty, info=info)


@dataclass(slots=True)
class CountdownEnv(BaseEnv):
    reward_fn: RewardFn = exact_countdown_reward
    task_name: str = "countdown"
    target_key: str = "target"
    accept_boxed_answers: bool = False

    def build_prompt(self, row: Mapping[str, Any]) -> str:
        return str(row.get("prompt") or build_prompt(row))

    def build_feedback(self, example: EnvExample, response: str, *, parsed_response=None) -> EnvFeedback:
        scorer = score_countdown_answer if self.accept_boxed_answers else score_response
        info = scorer(response, example.row)
        return EnvFeedback(done=True, success=info["exact"],
                           message="Correct Countdown expression." if info["exact"] else "Incorrect Countdown expression.",
                           metadata=info)
