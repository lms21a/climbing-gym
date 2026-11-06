from __future__ import annotations

import asyncio
from collections import deque
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

from climbing_gym.envs import MathQAEnv, exact_match_math_reward
from climbing_gym.inference import GenerationOutput, GenerationResult, SamplingConfig
from climbing_gym.rollout import (
    DirectHarness,
    InfinityThinkHarness,
    RolloutCoordinator,
    RolloutCoordinatorConfig,
    RolloutConfig,
    RolloutRunner,
    RolloutRunnerConfig,
)


@dataclass
class _Plan:
    text: str
    prompt_token_ids: tuple[int, ...] = (11, 12)
    completion_token_ids: tuple[int, ...] = (21, 22)
    completion_logprobs: tuple[float, ...] | None = (-0.1, -0.2)
    finish_reason: str | None = "stop"
    stop_reason: str | None = None
    delay_seconds: float = 0.0


class _FakeRolloutEngine:
    def __init__(self, plans: list[_Plan]) -> None:
        self._plans = deque(plans)
        self._reserved: dict[str, _Plan] = {}
        self._lock = asyncio.Lock()
        self.generate_calls: list[tuple[str, object | None, str]] = []
        self.aborted: list[str] = []

    async def generate(self, prompt, *, sampling=None, request_id=None):
        resolved_request_id = str(request_id or "req-missing")
        async with self._lock:
            plan = self._plans.popleft()
            self._reserved[resolved_request_id] = plan
        self.generate_calls.append((str(prompt), sampling, resolved_request_id))
        try:
            if plan.delay_seconds > 0.0:
                await asyncio.sleep(plan.delay_seconds)
        except asyncio.CancelledError:
            raise
        async with self._lock:
            self._reserved.pop(resolved_request_id, None)
        return GenerationResult(
            request_id=resolved_request_id,
            prompt=str(prompt),
            prompt_token_ids=tuple(plan.prompt_token_ids),
            outputs=(
                GenerationOutput(
                    index=0,
                    text=str(plan.text),
                    token_ids=tuple(plan.completion_token_ids),
                    completion_logprobs=plan.completion_logprobs,
                    finish_reason=plan.finish_reason,
                    stop_reason=plan.stop_reason,
                ),
            ),
            finished=True,
            raw_output=plan,
        )

    async def abort_async(self, request_id: str) -> None:
        self.aborted.append(str(request_id))
        async with self._lock:
            plan = self._reserved.pop(str(request_id), None)
            if plan is not None:
                self._plans.appendleft(plan)


class _FakeCoordinatorEngine(_FakeRolloutEngine):
    def __init__(self, plans: list[_Plan]) -> None:
        super().__init__(plans)
        self.tokenizer = SimpleNamespace(pad_token_id=0)
        self.sync_from_model_calls: list[tuple[object, int]] = []
        self.sleep_calls: list[tuple[int, str]] = []

    def sync_from_model(self, model, *, drain_timeout=300):
        self.sync_from_model_calls.append((model, drain_timeout))
        return "checkpoint"

    def sleep(self, *, level=1, mode="abort"):
        self.sleep_calls.append((level, mode))


def _build_env() -> MathQAEnv:
    return MathQAEnv(reward_fn=exact_match_math_reward)


def test_direct_harness_returns_single_turn_trajectory() -> None:
    env = _build_env()
    env.load_rows([{"question": "What is 2 + 2?", "parsed_answer": "4"}])
    engine = _FakeRolloutEngine([_Plan(text="Work\n\\boxed{4}")])
    runner = RolloutRunner(RolloutRunnerConfig(effective_batch_size=1))

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=DirectHarness(),
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert trajectory.terminal_reason == "single_turn"
    assert trajectory.evaluation.reward.reward == 1.0
    assert len(trajectory.turns) == 1
    assert trajectory.turns[0].response.endswith("\\boxed{4}")


def test_infinity_think_harness_continues_and_injects_clean_history() -> None:
    env = _build_env()
    env.load_rows([{"question": "What is 2 + 2?", "parsed_answer": "4"}])
    engine = _FakeRolloutEngine(
        [
            _Plan(text="<think>step</think><summary>\n  first step  \n</summary>"),
            _Plan(text="<think>done</think>\\boxed{4}"),
        ]
    )
    runner = RolloutRunner(RolloutRunnerConfig(effective_batch_size=1))

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=InfinityThinkHarness(max_turns=2),
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(trajectories) == 1
    trajectory = trajectories[0]
    assert len(trajectory.turns) == 2
    assert trajectory.terminal_reason == "response_complete"
    assert trajectory.evaluation.reward.reward == 1.0
    assert "<history>\n[1] first step\n</history>" in engine.generate_calls[1][0]
    assert "Turn 2/2." in engine.generate_calls[1][0]


def test_infinity_think_reward_receives_turn_count() -> None:
    captured_turn_counts: list[int] = []

    def reward_fn(request) -> float:
        captured_turn_counts.append(int(request.turn_count))
        return 1.0

    env = MathQAEnv(reward_fn=reward_fn)
    env.load_rows([{"question": "What is 2 + 2?", "parsed_answer": "4"}])
    engine = _FakeRolloutEngine(
        [
            _Plan(text="<think>step</think><summary>first step</summary>"),
            _Plan(text="<think>done</think>\\boxed{4}"),
        ]
    )
    runner = RolloutRunner(RolloutRunnerConfig(effective_batch_size=1))

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=InfinityThinkHarness(max_turns=2),
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(trajectories) == 1
    assert len(trajectories[0].turns) == 2
    assert captured_turn_counts == [2]


def test_infinity_think_reward_receives_summary_metadata() -> None:
    captured_turn_metadata: list[tuple[dict[str, object], ...]] = []

    def reward_fn(request) -> float:
        captured_turn_metadata.append(tuple(dict(metadata) for metadata in request.turn_metadata))
        return 1.0

    env = MathQAEnv(reward_fn=reward_fn)
    env.load_rows([{"question": "What is 2 + 2?", "parsed_answer": "4"}])
    engine = _FakeRolloutEngine(
        [
            _Plan(text="<think>step</think><summary>carry the partial arithmetic</summary>"),
            _Plan(text="<think>done</think>\\boxed{4}"),
        ]
    )
    runner = RolloutRunner(RolloutRunnerConfig(effective_batch_size=1))

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=InfinityThinkHarness(max_turns=2),
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(trajectories) == 1
    assert len(captured_turn_metadata) == 1
    intermediate, final = captured_turn_metadata[0]
    assert intermediate["summary"] == "carry the partial arithmetic"
    assert intermediate["history_length"] == 0
    assert intermediate["strict_intermediate_format"] is True
    assert intermediate["strict_final_format"] is False
    assert final["summary"] == ""
    assert final["history_length"] == 1
    assert final["strict_intermediate_format"] is False
    assert final["strict_final_format"] is True


def test_infinity_think_marks_last_summary_turn_as_max_turns_reached() -> None:
    env = _build_env()
    env.load_rows([{"question": "What is 2 + 2?", "parsed_answer": "4"}])
    engine = _FakeRolloutEngine([_Plan(text="<think>step</think><summary>keep going</summary>")])
    runner = RolloutRunner(RolloutRunnerConfig(effective_batch_size=1))

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=InfinityThinkHarness(max_turns=1),
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(trajectories) == 1
    assert trajectories[0].terminal_reason == "max_turns_reached"


def test_dynamic_sampler_replaces_zero_std_groups_until_effective_target_is_met() -> None:
    env = _build_env()
    env.load_rows(
        [
            {"question": "q0", "parsed_answer": "1"},
            {"question": "q1", "parsed_answer": "1"},
        ]
    )
    engine = _FakeRolloutEngine(
        [
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{2}"),
        ]
    )
    runner = RolloutRunner(
        RolloutRunnerConfig(
            effective_batch_size=1,
            max_inflight_requests=2,
        )
    )

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=DirectHarness(),
            rollout_config=RolloutConfig(num_rollouts=2),
        )
    )

    assert len(trajectories) == 2
    assert [trajectory.evaluation.reward.reward for trajectory in trajectories] == [1.0, 0.0]
    assert runner.last_stats["sampled_group_count"] == 2.0
    assert runner.last_stats["dropped_zero_std_group_count"] == 1.0
    assert runner.last_stats["effective_group_count"] == 1.0


def test_finished_ineffective_groups_are_not_replayed_across_calls() -> None:
    env = _build_env()
    env.load_rows([{"question": "q0", "parsed_answer": "1"}])
    engine = _FakeRolloutEngine([_Plan(text="\\boxed{1}"), _Plan(text="\\boxed{1}")])
    runner = RolloutRunner(
        RolloutRunnerConfig(
            effective_batch_size=1,
            max_inflight_requests=2,
        )
    )

    first = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=DirectHarness(),
            rollout_config=RolloutConfig(num_rollouts=2),
        )
    )
    second = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=DirectHarness(),
            rollout_config=RolloutConfig(num_rollouts=2),
        )
    )

    assert first == []
    assert second == []
    assert runner.last_stats["sampled_group_count"] == 0.0
    assert runner.last_stats["carried_in_unfinished_group_count"] == 0.0


def test_runner_keeps_ineffective_groups_when_dynamic_sampling_is_disabled() -> None:
    env = _build_env()
    env.load_rows([{"question": "q0", "parsed_answer": "1"}])
    engine = _FakeRolloutEngine([_Plan(text="\\boxed{1}"), _Plan(text="\\boxed{1}")])
    runner = RolloutRunner(
        RolloutRunnerConfig(
            effective_batch_size=1,
            max_inflight_requests=2,
            use_dynamic_sampling=False,
        )
    )

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=DirectHarness(),
            rollout_config=RolloutConfig(num_rollouts=2),
        )
    )

    assert len(trajectories) == 2
    assert [trajectory.evaluation.reward.reward for trajectory in trajectories] == [1.0, 1.0]
    assert runner.last_stats["selected_group_count"] == 1.0
    assert runner.last_stats["effective_group_count"] == 0.0
    assert runner.last_stats["dropped_zero_std_group_count"] == 0.0


def test_unfinished_groups_carry_across_collect_calls() -> None:
    env = _build_env()
    env.load_rows(
        [
            {"question": "q0", "parsed_answer": "1"},
            {"question": "q1", "parsed_answer": "1"},
        ]
    )
    engine = _FakeRolloutEngine(
        [
            _Plan(text="\\boxed{1}"),
            _Plan(text="<think>step</think><summary>keep going</summary>"),
            _Plan(text="\\boxed{1}"),
        ]
    )
    runner = RolloutRunner(
        RolloutRunnerConfig(
            effective_batch_size=1,
            max_inflight_requests=2,
        )
    )
    harness = InfinityThinkHarness(max_turns=2)

    first = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=harness,
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )
    first_stats = dict(runner.last_stats)

    second = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=harness,
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(first) == 1
    assert first_stats["carried_out_unfinished_group_count"] == 1.0
    assert len(second) == 1
    assert len(second[0].turns) == 2
    assert runner.last_stats["sampled_group_count"] == 0.0


def test_runner_aborts_pending_requests_once_effective_target_is_reached() -> None:
    env = _build_env()
    env.load_rows(
        [
            {"question": "q0", "parsed_answer": "1"},
            {"question": "q1", "parsed_answer": "1"},
        ]
    )
    engine = _FakeRolloutEngine(
        [
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}", delay_seconds=0.1),
        ]
    )
    runner = RolloutRunner(
        RolloutRunnerConfig(
            effective_batch_size=1,
            max_inflight_requests=2,
        )
    )

    trajectories = asyncio.run(
        runner.collect(
            env=env,
            engine=engine,
            harness=DirectHarness(),
            rollout_config=RolloutConfig(num_rollouts=1),
        )
    )

    assert len(trajectories) == 1
    assert len(engine.aborted) == 1
    assert runner.last_stats["aborted_request_count"] == 1.0


def test_non_dynamic_runner_does_not_prefetch_past_bounded_target() -> None:
    engine = _FakeCoordinatorEngine(
        [
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}", delay_seconds=0.05),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
        ]
    )
    env = _build_env()
    env.configure_dataset(
        [
            {"question": "q0", "parsed_answer": "1"},
            {"question": "q1", "parsed_answer": "1"},
            {"question": "q2", "parsed_answer": "1"},
            {"question": "q3", "parsed_answer": "1"},
        ],
        shuffle=False,
    )
    coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=DirectHarness(),
        config=RolloutCoordinatorConfig(
            num_rollouts=1,
            batch_size=4,
            runner_config=RolloutRunnerConfig(
                effective_batch_size=4,
                max_inflight_requests=2,
                use_dynamic_sampling=False,
            ),
        ),
    )

    batch = coordinator.collect()

    assert sorted(batch.indices) == [0, 1, 2, 3]
    assert len(batch.trajectories) == 4
    assert len(engine.generate_calls) == 4
    assert batch.stats["sampled_group_count"] == pytest.approx(4.0)
    assert batch.stats["aborted_request_count"] == pytest.approx(0.0)


def test_runner_raises_when_requests_stall() -> None:
    env = _build_env()
    env.load_rows([{"question": "q0", "parsed_answer": "1"}])
    engine = _FakeRolloutEngine([_Plan(text="\\boxed{1}", delay_seconds=0.1)])
    runner = RolloutRunner(
        RolloutRunnerConfig(
            effective_batch_size=1,
            stall_timeout_seconds=0.01,
            progress_log_interval_seconds=0.0,
        )
    )

    with pytest.raises(RuntimeError, match="stalled waiting for completions"):
        asyncio.run(
            runner.collect(
                env=env,
                engine=engine,
                harness=DirectHarness(),
                rollout_config=RolloutConfig(num_rollouts=1),
            )
        )


def test_runner_rejects_multi_sample_generation_requests() -> None:
    env = _build_env()
    env.load_rows([{"question": "q0", "parsed_answer": "1"}])
    engine = _FakeRolloutEngine([])
    runner = RolloutRunner(RolloutRunnerConfig(effective_batch_size=1))

    with pytest.raises(ValueError, match="sampling.n == 1"):
        asyncio.run(
            runner.collect(
                env=env,
                engine=engine,
                harness=DirectHarness(),
                rollout_config=RolloutConfig(num_rollouts=1, sampling=SamplingConfig(n=2)),
            )
        )


def test_rollout_coordinator_collects_trajectories_and_hides_runner_details() -> None:
    engine = _FakeCoordinatorEngine([_Plan(text="Work\n\\boxed{4}")])
    env = _build_env()
    env.configure_dataset(
        [{"question": "What is 2 + 2?", "parsed_answer": "4"}],
        shuffle=False,
    )
    coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=DirectHarness(),
        config=RolloutCoordinatorConfig(
            num_rollouts=1,
            batch_size=1,
        ),
    )

    batch = coordinator.collect()

    assert batch.indices == (0,)
    assert len(batch.trajectories) == 1
    assert batch.trajectories[0].evaluation.reward.reward == 1.0
    assert engine.sync_from_model_calls == []
    assert engine.sleep_calls == [(1, "abort")]
    assert batch.stats["effective_group_count"] == pytest.approx(1.0)


def test_rollout_coordinator_syncs_weights_via_engine() -> None:
    engine = _FakeCoordinatorEngine([])
    env = _build_env()
    env.configure_dataset(
        [{"question": "What is 2 + 2?", "parsed_answer": "4"}],
        shuffle=False,
    )
    coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=DirectHarness(),
        config=RolloutCoordinatorConfig(
            num_rollouts=1,
            batch_size=1,
        ),
    )
    model = object()

    checkpoint = coordinator.sync_weights(model, drain_timeout=17)

    assert checkpoint == "checkpoint"
    assert engine.sync_from_model_calls == [(model, 17)]
    assert engine.sleep_calls == []


def test_rollout_coordinator_reports_emitted_indices_and_preserves_prefetched_examples() -> None:
    engine = _FakeCoordinatorEngine(
        [
            _Plan(text="<think>step</think><summary>keep going</summary>"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="<think>step</think><summary>keep going</summary>"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
        ]
    )
    env = _build_env()
    env.configure_dataset(
        [
            {"question": "q0", "parsed_answer": "1"},
            {"question": "q1", "parsed_answer": "1"},
            {"question": "q2", "parsed_answer": "1"},
            {"question": "q3", "parsed_answer": "1"},
        ],
        shuffle=False,
    )
    coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=InfinityThinkHarness(max_turns=2),
        config=RolloutCoordinatorConfig(
            num_rollouts=1,
            batch_size=2,
            runner_config=RolloutRunnerConfig(
                effective_batch_size=1,
                max_inflight_requests=2,
            ),
        ),
    )

    first = coordinator.collect()
    second = coordinator.collect()
    third = coordinator.collect()

    assert first.indices == (1,)
    assert [trajectory.example.row["question"] for trajectory in first.trajectories] == ["q1"]
    assert first.stats["carried_out_unfinished_group_count"] == pytest.approx(1.0)

    assert second.indices == (0,)
    assert [trajectory.example.row["question"] for trajectory in second.trajectories] == ["q0"]
    assert second.stats["carried_in_unfinished_group_count"] == pytest.approx(1.0)
    assert coordinator.env.pending_example_count == 1

    assert third.indices in ((2,), (3,))
    assert [trajectory.example.row["question"] for trajectory in third.trajectories] in (["q2"], ["q3"])
    assert third.stats["carried_in_unfinished_group_count"] == pytest.approx(1.0)
    assert third.stats["sampled_group_count"] == pytest.approx(1.0)


def test_rollout_coordinator_dynamic_sampling_refills_beyond_initial_batch_size() -> None:
    engine = _FakeCoordinatorEngine(
        [
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{1}"),
            _Plan(text="\\boxed{2}"),
        ]
    )
    env = _build_env()
    env.configure_dataset(
        [
            {"question": "q0", "parsed_answer": "1"},
            {"question": "q1", "parsed_answer": "1"},
        ],
        shuffle=False,
    )
    coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=DirectHarness(),
        config=RolloutCoordinatorConfig(
            num_rollouts=2,
            batch_size=1,
            runner_config=RolloutRunnerConfig(
                effective_batch_size=1,
                max_inflight_requests=2,
                use_dynamic_sampling=True,
            ),
        ),
    )

    batch = coordinator.collect()

    assert len(batch.trajectories) == 2
    assert [trajectory.evaluation.reward.reward for trajectory in batch.trajectories] == [1.0, 0.0]
    assert batch.indices == (1,)
    assert batch.stats["sampled_group_count"] == pytest.approx(2.0)
    assert batch.stats["dropped_zero_std_group_count"] == pytest.approx(1.0)
    assert batch.stats["effective_group_count"] == pytest.approx(1.0)


def test_rollout_coordinator_collect_async_supports_active_event_loops() -> None:
    engine = _FakeCoordinatorEngine([_Plan(text="Work\n\\boxed{4}")])
    env = _build_env()
    env.configure_dataset(
        [{"question": "What is 2 + 2?", "parsed_answer": "4"}],
        shuffle=False,
    )
    coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=DirectHarness(),
        config=RolloutCoordinatorConfig(
            num_rollouts=1,
            batch_size=1,
        ),
    )

    async def _run() -> None:
        with pytest.raises(RuntimeError, match="collect_async"):
            coordinator.collect()
        batch = await coordinator.collect_async()
        assert batch.indices == (0,)
        assert len(batch.trajectories) == 1

    asyncio.run(_run())


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"effective_batch_size": 0}, "effective_batch_size must be > 0"),
        ({"max_inflight_requests": 0}, "max_inflight_requests must be > 0 when provided"),
    ],
)
def test_rollout_runner_config_rejects_non_positive_values(kwargs, message) -> None:
    with pytest.raises(ValueError, match=message):
        RolloutRunnerConfig(**kwargs)
