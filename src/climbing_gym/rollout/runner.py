from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
import inspect
import logging
from statistics import pstdev
from typing import Any
import uuid

from climbing_gym.envs.contracts import BaseEnv, EnvExample
from climbing_gym.rollout.config import RolloutConfig, RolloutRunnerConfig
from climbing_gym.rollout.contracts import Harness, HarnessState, Trajectory


logger = logging.getLogger(__name__)


@dataclass(slots=True)
class _TrajectorySlot:
    rollout_index: int
    state: HarnessState
    pending_request_id: str | None = None
    discarded: bool = False


@dataclass(slots=True)
class _PromptGroupState:
    group_id: int
    example: EnvExample
    slots: list[_TrajectorySlot] = field(default_factory=list)


@dataclass(frozen=True, slots=True)
class _PendingRequest:
    request_id: str
    group: _PromptGroupState
    slot: _TrajectorySlot
    task: asyncio.Task[Any]


@dataclass(slots=True)
class _CollectState:
    env: BaseEnv
    engine: Any
    harness: Harness
    rollout_config: RolloutConfig
    loop: asyncio.AbstractEventLoop
    active_groups: list[_PromptGroupState]
    carried_in_unfinished_groups: int
    target_effective_groups: int
    max_active_groups: int
    max_inflight_requests: int
    max_inflight_rollouts_per_group: int
    reward_std_epsilon: float
    use_dynamic_sampling: bool
    stall_timeout_seconds: float
    progress_log_interval_seconds: float
    selected_groups: list[list[Trajectory]] = field(default_factory=list)
    effective_group_count: int = 0
    sampled_groups: int = 0
    dropped_zero_std_groups: int = 0
    requests_issued: int = 0
    requests_completed: int = 0
    aborted_request_count: int = 0
    pending_by_request_id: dict[str, _PendingRequest] = field(default_factory=dict)
    task_to_request_id: dict[asyncio.Task[Any], str] = field(default_factory=dict)
    last_progress_time: float = 0.0
    last_progress_log_time: float = 0.0


def _sampling_n(sampling: object | None) -> int | None:
    if sampling is None:
        return None
    value = getattr(sampling, "n", None)
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _validate_single_output_sampling(sampling: object | None) -> None:
    n = _sampling_n(sampling)
    if n is not None and n != 1:
        raise ValueError("RolloutRunner requires sampling.n == 1 for each engine request.")


class RolloutRunner:
    def __init__(self, runner_config: RolloutRunnerConfig = RolloutRunnerConfig()) -> None:
        self.runner_config = runner_config
        self.last_stats: dict[str, float] = {}
        self._unfinished_groups: list[_PromptGroupState] = []
        self._next_group_id = 0

    def _create_collect_state(
        self,
        *,
        env: BaseEnv,
        engine: Any,
        harness: Harness,
        rollout_config: RolloutConfig,
    ) -> _CollectState:
        active_groups = list(self._unfinished_groups)
        self._unfinished_groups = []
        loop = asyncio.get_running_loop()
        now = loop.time()
        target_effective_groups = int(self.runner_config.effective_batch_size)
        max_inflight_rollouts_per_group = int(rollout_config.num_rollouts)
        max_inflight_requests = (
            max(1, target_effective_groups * max_inflight_rollouts_per_group)
            if self.runner_config.max_inflight_requests is None
            else int(self.runner_config.max_inflight_requests)
        )
        max_active_groups = max(1, max_inflight_requests // max_inflight_rollouts_per_group)
        return _CollectState(
            env=env,
            engine=engine,
            harness=harness,
            rollout_config=rollout_config,
            loop=loop,
            active_groups=active_groups,
            carried_in_unfinished_groups=len(active_groups),
            target_effective_groups=target_effective_groups,
            max_active_groups=max_active_groups,
            max_inflight_requests=max_inflight_requests,
            max_inflight_rollouts_per_group=max_inflight_rollouts_per_group,
            reward_std_epsilon=float(self.runner_config.reward_std_epsilon),
            use_dynamic_sampling=bool(self.runner_config.use_dynamic_sampling),
            stall_timeout_seconds=float(self.runner_config.stall_timeout_seconds),
            progress_log_interval_seconds=float(self.runner_config.progress_log_interval_seconds),
            last_progress_time=now,
            last_progress_log_time=now,
        )

    def _within_request_budget(self, state: _CollectState) -> bool:
        if state.rollout_config.max_total_requests is None:
            return True
        return state.requests_issued < int(state.rollout_config.max_total_requests)

    def _group_pending_count(self, group: _PromptGroupState) -> int:
        return sum(1 for slot in group.slots if slot.pending_request_id is not None)

    def _group_is_finished(self, group: _PromptGroupState) -> bool:
        return all(slot.discarded or slot.state.finished for slot in group.slots)

    def _build_group(self, state: _CollectState, example: EnvExample) -> _PromptGroupState:
        group = _PromptGroupState(
            group_id=int(self._next_group_id),
            example=example,
            slots=[],
        )
        self._next_group_id += 1
        for rollout_index in range(int(state.rollout_config.num_rollouts)):
            group.slots.append(
                _TrajectorySlot(
                    rollout_index=int(rollout_index),
                    state=state.harness.create_state(example, rollout_index),
                )
            )
        return group

    def _refill_example_queue_from_dataset(self, state: _CollectState) -> int:
        pending_count = int(state.env.pending_example_count)
        refill_target = max(1, int(state.max_active_groups))
        refill_count = max(1, refill_target - pending_count)
        try:
            prompt_batch = state.env.next_batch(refill_count)
        except RuntimeError:
            return 0
        if prompt_batch.is_empty:
            return 0
        state.env.enqueue_examples(prompt_batch.examples)
        return len(prompt_batch)

    def _sample_group(self, state: _CollectState) -> _PromptGroupState | None:
        example = state.env.next_example()
        if example is None and self._refill_example_queue_from_dataset(state) > 0:
            example = state.env.next_example()
        if example is None:
            return None
        state.sampled_groups += 1
        return self._build_group(state, example)

    def _materialize_group(self, state: _CollectState, group: _PromptGroupState) -> list[Trajectory]:
        trajectories: list[Trajectory] = []
        for slot in sorted(group.slots, key=lambda item: item.rollout_index):
            if slot.discarded:
                continue
            turns = tuple(slot.state.turns)
            if not turns:
                continue
            final_response = str(slot.state.final_response or turns[-1].response)
            evaluation = state.env.evaluate_response(
                group.example,
                final_response,
                turn_count=len(turns),
                turn_metadata=tuple(dict(turn.metadata) for turn in turns),
            )
            trajectories.append(
                Trajectory(
                    group_id=int(group.group_id),
                    rollout_index=int(slot.rollout_index),
                    example=group.example,
                    turns=turns,
                    final_response=str(final_response),
                    finished=bool(slot.state.finished),
                    terminal_reason=str(slot.state.terminal_reason),
                    evaluation=evaluation,
                    metadata=dict(slot.state.metadata),
                )
            )
        return trajectories

    def _group_is_effective(
        self,
        state: _CollectState,
        trajectories: list[Trajectory],
    ) -> bool:
        if not trajectories:
            return False
        if int(state.rollout_config.num_rollouts) < 2:
            return True
        rewards = [float(traj.evaluation.reward.reward) for traj in trajectories]
        if len(rewards) < 2:
            return False
        return float(pstdev(rewards)) > state.reward_std_epsilon

    def _finalize_group(self, state: _CollectState, group: _PromptGroupState) -> None:
        if group in state.active_groups:
            state.active_groups.remove(group)
        trajectories = self._materialize_group(state, group)
        if not trajectories:
            return
        is_effective = self._group_is_effective(state, trajectories)
        if is_effective:
            state.effective_group_count += 1
            if len(state.selected_groups) < state.target_effective_groups:
                state.selected_groups.append(trajectories)
            return
        if not state.use_dynamic_sampling:
            if len(state.selected_groups) < state.target_effective_groups:
                state.selected_groups.append(trajectories)
            return
        state.dropped_zero_std_groups += 1

    def _force_finish_budget_exhausted_groups(self, state: _CollectState) -> None:
        for group in list(state.active_groups):
            for slot in group.slots:
                if (
                    slot.discarded
                    or slot.pending_request_id is not None
                    or slot.state.finished
                ):
                    continue
                if slot.state.turns:
                    state.harness.force_finish(slot.state, "budget_exhausted")
                else:
                    slot.discarded = True
            if self._group_is_finished(group):
                self._finalize_group(state, group)

    def _status_summary(self, state: _CollectState) -> str:
        return (
            f"selected={len(state.selected_groups)}/{state.target_effective_groups} "
            f"pending={len(state.pending_by_request_id)} active_groups={len(state.active_groups)} "
            f"sampled_groups={state.sampled_groups} carried_in={state.carried_in_unfinished_groups} "
            f"requests_issued={state.requests_issued} requests_completed={state.requests_completed}"
        )

    def _launch_request(
        self,
        state: _CollectState,
        *,
        group: _PromptGroupState,
        slot: _TrajectorySlot,
    ) -> bool:
        request = state.harness.next_request(group.example, slot.state)
        if request is None:
            if slot.state.finished:
                return False
            raise RuntimeError("Harness returned no request for an unfinished state.")

        sampling = request.sampling if request.sampling is not None else state.rollout_config.sampling
        _validate_single_output_sampling(sampling)

        request_id = f"rollout-{group.group_id}-{slot.rollout_index}-{uuid.uuid4().hex}"
        slot.pending_request_id = request_id
        task = asyncio.create_task(
            state.engine.generate(
                request.prompt,
                sampling=sampling,
                request_id=request_id,
            )
        )
        state.pending_by_request_id[request_id] = _PendingRequest(
            request_id=request_id,
            group=group,
            slot=slot,
            task=task,
        )
        state.task_to_request_id[task] = request_id
        state.requests_issued += 1
        return True

    def _issue_from_active_groups(self, state: _CollectState) -> bool:
        for group in list(state.active_groups):
            if self._group_pending_count(group) >= state.max_inflight_rollouts_per_group:
                continue
            for slot in group.slots:
                if (
                    slot.discarded
                    or slot.pending_request_id is not None
                    or slot.state.finished
                ):
                    continue
                if self._launch_request(state, group=group, slot=slot):
                    return True
        return False

    def _issue_ready_requests(self, state: _CollectState) -> None:
        while (
            len(state.pending_by_request_id) < state.max_inflight_requests
            and self._within_request_budget(state)
        ):
            if self._issue_from_active_groups(state):
                continue
            if len(state.active_groups) >= state.max_active_groups:
                break
            if (
                not state.use_dynamic_sampling
                and len(state.selected_groups) + len(state.active_groups)
                >= state.target_effective_groups
            ):
                break
            new_group = self._sample_group(state)
            if new_group is None:
                break
            state.active_groups.append(new_group)

    async def _abort_pending_requests(self, state: _CollectState) -> None:
        abort_fn = getattr(state.engine, "abort_async", None) or getattr(state.engine, "abort", None)
        tasks_to_cancel: list[asyncio.Task[Any]] = []
        for request_id, pending in list(state.pending_by_request_id.items()):
            if abort_fn is not None:
                result = abort_fn(request_id)
                if inspect.isawaitable(result):
                    await result
            state.aborted_request_count += 1
            pending.slot.pending_request_id = None
            state.pending_by_request_id.pop(request_id, None)
            state.task_to_request_id.pop(pending.task, None)
            if not pending.task.done():
                pending.task.cancel()
            tasks_to_cancel.append(pending.task)
        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)

    def _compute_wait_timeout(self, state: _CollectState) -> float | None:
        wait_timeout: float | None = None
        if state.progress_log_interval_seconds > 0.0:
            wait_timeout = state.progress_log_interval_seconds
        if state.stall_timeout_seconds > 0.0:
            remaining_stall_budget = (
                state.stall_timeout_seconds
                - (state.loop.time() - state.last_progress_time)
            )
            if remaining_stall_budget <= 0.0:
                raise RuntimeError(
                    f"Rollout runner stalled before wait: {self._status_summary(state)}"
                )
            wait_timeout = (
                remaining_stall_budget
                if wait_timeout is None
                else min(wait_timeout, remaining_stall_budget)
            )
        return wait_timeout

    def _handle_wait_timeout(self, state: _CollectState) -> None:
        now = state.loop.time()
        if (
            state.progress_log_interval_seconds > 0.0
            and now - state.last_progress_log_time >= state.progress_log_interval_seconds
        ):
            logger.info("Rollout runner heartbeat: %s", self._status_summary(state))
            state.last_progress_log_time = now
        if (
            state.stall_timeout_seconds > 0.0
            and now - state.last_progress_time >= state.stall_timeout_seconds
        ):
            raise RuntimeError(
                "Rollout runner stalled waiting for completions: "
                f"{self._status_summary(state)}"
            )

    def _process_completed_tasks(
        self,
        state: _CollectState,
        tasks: set[asyncio.Task[Any]],
    ) -> None:
        for task in tasks:
            request_id = state.task_to_request_id.pop(task)
            pending = state.pending_by_request_id.pop(request_id)
            pending.slot.pending_request_id = None
            result = task.result()
            state.requests_completed += 1
            state.last_progress_time = state.loop.time()
            state.harness.apply_result(pending.group.example, pending.slot.state, result)
            if self._group_is_finished(pending.group):
                self._finalize_group(state, pending.group)

    def _carry_out_groups(self, state: _CollectState) -> list[_PromptGroupState]:
        return [
            group
            for group in state.active_groups
            if not self._group_is_finished(group)
        ]

    def _update_last_stats(self, state: _CollectState, carried_out_groups: list[_PromptGroupState]) -> None:
        self.last_stats = {
            "effective_group_target": float(state.target_effective_groups),
            "selected_group_count": float(len(state.selected_groups)),
            "effective_group_count": float(state.effective_group_count),
            "sampled_group_count": float(state.sampled_groups),
            "dropped_zero_std_group_count": float(state.dropped_zero_std_groups),
            "carried_in_unfinished_group_count": float(state.carried_in_unfinished_groups),
            "carried_out_unfinished_group_count": float(len(carried_out_groups)),
            "requests_issued": float(state.requests_issued),
            "requests_completed": float(state.requests_completed),
            "aborted_request_count": float(state.aborted_request_count),
        }

    def _flatten_selected_groups(self, state: _CollectState) -> list[Trajectory]:
        return [trajectory for group in state.selected_groups for trajectory in group]

    async def collect(
        self,
        *,
        env: BaseEnv,
        engine: Any,
        harness: Harness,
        rollout_config: RolloutConfig,
    ) -> list[Trajectory]:
        _validate_single_output_sampling(rollout_config.sampling)
        state = self._create_collect_state(
            env=env,
            engine=engine,
            harness=harness,
            rollout_config=rollout_config,
        )

        logger.info("Rollout runner start: %s", self._status_summary(state))

        while len(state.selected_groups) < state.target_effective_groups:
            self._issue_ready_requests(state)

            if len(state.selected_groups) >= state.target_effective_groups:
                break

            if not state.pending_by_request_id:
                if not self._within_request_budget(state):
                    self._force_finish_budget_exhausted_groups(state)
                break

            wait_timeout = self._compute_wait_timeout(state)

            done, _ = await asyncio.wait(
                set(state.task_to_request_id),
                return_when=asyncio.FIRST_COMPLETED,
                timeout=wait_timeout,
            )
            if not done:
                self._handle_wait_timeout(state)
                continue

            self._process_completed_tasks(state, done)

        if state.pending_by_request_id and len(state.selected_groups) >= state.target_effective_groups:
            await self._abort_pending_requests(state)

        if not self._within_request_budget(state) and not state.pending_by_request_id:
            self._force_finish_budget_exhausted_groups(state)

        carried_out_groups = self._carry_out_groups(state)
        self._unfinished_groups = carried_out_groups
        self._update_last_stats(state, carried_out_groups)
        logger.info("Rollout runner complete: %s", self._status_summary(state))
        return self._flatten_selected_groups(state)
