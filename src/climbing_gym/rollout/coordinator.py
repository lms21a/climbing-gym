from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any

from climbing_gym.envs.contracts import BaseEnv
from climbing_gym.rollout.config import RolloutConfig, RolloutRunnerConfig
from climbing_gym.rollout.contracts import Harness, Trajectory
from climbing_gym.rollout.runner import RolloutRunner


def _sync_engine_weights(
    engine: Any,
    model: Any,
    *,
    drain_timeout: int = 300,
) -> Any:
    sync_from_model = getattr(engine, "sync_from_model", None)
    if callable(sync_from_model):
        return sync_from_model(model, drain_timeout=drain_timeout)

    sync = getattr(engine, "sync", None)
    if callable(sync):
        return sync(drain_timeout=drain_timeout)

    raise TypeError("Rollout engine must provide sync_from_model(model, ...) or sync(...).")


def _park_engine(engine: Any, *, level: int, mode: str) -> None:
    sleep = getattr(engine, "sleep", None)
    if not callable(sleep):
        return
    sleep(level=level, mode=mode)


@dataclass(frozen=True, slots=True)
class TrajectoryBatch:
    indices: tuple[int, ...]
    trajectories: tuple[Trajectory, ...]
    stats: dict[str, float]


def _resolve_trajectory_indices(trajectories: tuple[Trajectory, ...]) -> tuple[int, ...]:
    indices: list[int] = []
    seen: set[int] = set()
    for trajectory in trajectories:
        dataset_index = getattr(trajectory.example, "dataset_index", None)
        if dataset_index is None:
            continue
        resolved_index = int(dataset_index)
        if resolved_index in seen:
            continue
        seen.add(resolved_index)
        indices.append(resolved_index)
    return tuple(indices)


@dataclass(frozen=True, slots=True)
class RolloutCoordinatorConfig:
    num_rollouts: int
    sampling: object | None = None
    max_total_requests: int | None = None
    batch_size: int = 1
    engine_sleep_level: int = 1
    engine_sleep_mode: str = "abort"
    runner_config: RolloutRunnerConfig | None = None

    def resolved_runner(self) -> RolloutRunnerConfig:
        if self.runner_config is not None:
            return self.runner_config
        return RolloutRunnerConfig(effective_batch_size=int(self.batch_size))

    def build_rollout_config(self) -> RolloutConfig:
        return RolloutConfig(
            num_rollouts=int(self.num_rollouts),
            sampling=self.sampling,
            max_total_requests=self.max_total_requests,
        )


class RolloutCoordinator:
    def __init__(
        self,
        *,
        engine: Any,
        env: BaseEnv,
        harness: Harness,
        config: RolloutCoordinatorConfig,
        rollout_runner: RolloutRunner | None = None,
    ) -> None:
        self.engine = engine
        self.env = env
        self.harness = harness
        self.config = config
        self.rollout_config = config.build_rollout_config()
        runner_config = config.resolved_runner()
        self.rollout_runner = rollout_runner or RolloutRunner(runner_config=runner_config)

    @property
    def last_stats(self) -> dict[str, float]:
        return dict(getattr(self.rollout_runner, "last_stats", {}))

    def sync_weights(
        self,
        model: Any,
        *,
        drain_timeout: int = 300,
    ) -> Any:
        return _sync_engine_weights(
            self.engine,
            model,
            drain_timeout=drain_timeout,
        )

    def park(self) -> None:
        _park_engine(
            engine=self.engine,
            level=self.config.engine_sleep_level,
            mode=self.config.engine_sleep_mode,
        )

    def _top_up_example_queue(self) -> int:
        target_count = max(0, int(self.config.batch_size))
        deficit = max(0, target_count - int(self.env.pending_example_count))
        if deficit <= 0:
            return 0
        prompt_batch = self.env.next_batch(deficit)
        if prompt_batch.is_empty:
            return 0
        self.env.enqueue_examples(prompt_batch.examples)
        return len(prompt_batch)

    async def collect_async(self) -> TrajectoryBatch:
        pending_before = int(self.env.pending_example_count)
        loaded_count = self._top_up_example_queue()

        try:
            trajectories = await self.rollout_runner.collect(
                env=self.env,
                engine=self.engine,
                harness=self.harness,
                rollout_config=self.rollout_config,
            )
        finally:
            self.park()

        trajectory_tuple = tuple(trajectories)
        stats = self.last_stats
        cursor = getattr(self.env, "_dataset_cursor", None)
        if cursor is not None:
            stats.update({"data/cursor": float(cursor.state.cursor),
                          "data/epoch": float(cursor.state.epoch),
                          "data/size": float(cursor.size),
                          "data/rows_loaded": float(cursor.state.epoch * cursor.size + cursor.state.cursor),
                          "data/pending_examples": float(self.env.pending_example_count)})
        if (
            not trajectory_tuple
            and pending_before <= 0
            and loaded_count <= 0
            and float(stats.get("sampled_group_count", 0.0)) <= 0.0
            and float(stats.get("carried_in_unfinished_group_count", 0.0)) <= 0.0
        ):
            raise RuntimeError("RolloutCoordinator could not build a prompt batch from the provided rows.")

        return TrajectoryBatch(
            indices=_resolve_trajectory_indices(trajectory_tuple),
            trajectories=trajectory_tuple,
            stats=stats,
        )

    def collect(self) -> TrajectoryBatch:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.collect_async())
        raise RuntimeError(
            "RolloutCoordinator.collect() cannot be called from an active event loop; "
            "use `await rollout_coordinator.collect_async()` instead."
        )
