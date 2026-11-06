from __future__ import annotations

from typing import Any

from climbing_gym.envs.contracts import EnvExample
from climbing_gym.rollout.contracts import HarnessRequest, HarnessState, build_trajectory_turn


class DirectHarness:
    def create_state(self, example: EnvExample, rollout_index: int) -> HarnessState:
        del example
        return HarnessState(
            rollout_index=int(rollout_index),
            metadata={"harness": "direct"},
        )

    def next_request(self, example: EnvExample, state: HarnessState) -> HarnessRequest | None:
        if state.finished or state.turns:
            return None
        return HarnessRequest(prompt=str(example.prompt))

    def apply_result(self, example: EnvExample, state: HarnessState, result: Any) -> None:
        del example
        turn = build_trajectory_turn(
            turn_index=1,
            prompt=str(getattr(result, "prompt", "")),
            response=str(getattr(result, "text", "")),
            result=result,
            metadata={"harness": "direct", "completion_token_count": len(result.outputs[0].token_ids)},
        )
        state.turns.append(turn)
        state.final_response = turn.response
        state.finished = True
        state.terminal_reason = "single_turn"

    def force_finish(self, state: HarnessState, reason: str) -> None:
        if state.finished:
            return
        state.finished = True
        if not state.final_response and state.turns:
            state.final_response = state.turns[-1].response
        state.terminal_reason = str(reason)
