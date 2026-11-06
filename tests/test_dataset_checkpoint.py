import json

import pytest

from climbing_gym.config import Env
from climbing_gym.dataset_checkpoint import restore_dataset_checkpoint, save_dataset_checkpoint
from climbing_gym.rollout import DirectHarness, RolloutCoordinator, RolloutCoordinatorConfig, RolloutRunnerConfig
from climbing_gym.native_dynamic import NativeDynamicRunner as CountdownDynamicRunner
from test_rollout import _FakeCoordinatorEngine, _Plan


GOOD = '<answer>2+3*4</answer>'
BAD = '<answer>2*3+4</answer>'


def coordinator(tmp_path, *, shuffle=False, rows=12):
    env = Env(name='countdown', reward='terminal_expression').build()
    env.configure_dataset([{'numbers': [2, 3, 4], 'target': 14, 'operators': '+-*/',
                            'tier': '3_multiply', 'id': str(i)} for i in range(rows)],
                          seed=17, shuffle=shuffle)
    engine = _FakeCoordinatorEngine([_Plan(text=t) for t in
        [BAD, BAD, GOOD, BAD, GOOD, BAD, GOOD, BAD] * 4])
    runner_config = RolloutRunnerConfig(effective_batch_size=1, max_inflight_requests=4,
                                        use_dynamic_sampling=True)
    return RolloutCoordinator(engine=engine, env=env, harness=DirectHarness(),
        config=RolloutCoordinatorConfig(num_rollouts=2, batch_size=3, max_total_requests=16,
                                        runner_config=runner_config),
        rollout_runner=CountdownDynamicRunner(runner_config, audit_dir=tmp_path))


def test_dynamic_cursor_retains_pending_and_advances_past_rejected_groups_on_resume(tmp_path):
    original = coordinator(tmp_path / 'original')
    first = original.collect()
    assert first.indices == (1,)  # Puzzle 0 was rejected, puzzle 2 is queued.
    assert first.stats['data/cursor'] == 3
    assert first.stats['data/pending_examples'] == 1
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    save_dataset_checkpoint(original, checkpoint)
    saved = json.loads((checkpoint / 'dataset_state.json').read_text())
    assert saved['env']['pending_indices'] == [2]
    second = original.collect()
    resumed = coordinator(tmp_path / 'resumed')
    # Start at the same engine responses as the original's next wave.
    resumed.engine._plans.popleft()
    resumed.engine._plans.popleft()
    resumed.engine._plans.popleft()
    resumed.engine._plans.popleft()
    assert restore_dataset_checkpoint(resumed, checkpoint)
    actual = resumed.collect()
    assert actual.indices == second.indices == (2,)
    assert actual.stats['data/cursor'] == second.stats['data/cursor']
    assert actual.trajectories[0].group_id == second.trajectories[0].group_id
    audits = [json.loads(line) for line in (tmp_path / 'original/groups.jsonl').read_text().splitlines()]
    assert [row['dataset_index'] for row in audits] == [0, 1, 2, 3]


def test_shuffled_resume_matches_uninterrupted_across_epoch_boundary(tmp_path):
    original = coordinator(tmp_path / 'original', shuffle=True, rows=5)
    original.env.enqueue_examples(original.env.next_batch(3).examples)
    original.env.next_example()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    save_dataset_checkpoint(original, checkpoint)
    resumed = coordinator(tmp_path / 'resumed', shuffle=True, rows=5)
    restore_dataset_checkpoint(resumed, checkpoint)
    assert [original.env.next_example().dataset_index for _ in range(2)] == [
        resumed.env.next_example().dataset_index for _ in range(2)]
    for _ in range(8):
        assert original.env.next_batch(2).indices == resumed.env.next_batch(2).indices
    assert original.env._dataset_cursor.state.epoch > 0


def test_resume_rejects_changed_bank_even_with_same_row_count(tmp_path):
    original = coordinator(tmp_path / 'original')
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    save_dataset_checkpoint(original, checkpoint)
    resumed = coordinator(tmp_path / 'resumed')
    resumed.env._dataset_cursor.dataset[0]['id'] = 'different puzzle'
    with pytest.raises(ValueError, match='does not match'):
        restore_dataset_checkpoint(resumed, checkpoint)


def test_dynamic_futoshiki_binary_group_rewards(tmp_path):
    from test_futoshiki import example, answer
    env = Env(name='futoshiki', reward='exact_match', args={'target_key': 'solution'}).build()
    row = example()
    good = answer({i: value for i, value in enumerate(row['solution']) if i not in row['givens']})
    env.configure_dataset([row] * 5, shuffle=False)
    engine = _FakeCoordinatorEngine([_Plan(text=text) for text in ['', '', good, '']])
    runner_config = RolloutRunnerConfig(effective_batch_size=1, max_inflight_requests=4,
                                        use_dynamic_sampling=True)
    run = RolloutCoordinator(engine=engine, env=env, harness=DirectHarness(),
        config=RolloutCoordinatorConfig(num_rollouts=2, batch_size=2, max_total_requests=4,
                                        runner_config=runner_config),
        rollout_runner=CountdownDynamicRunner(runner_config, audit_dir=tmp_path))
    batch = run.collect()
    assert batch.indices == (1,)
    assert [t.evaluation.reward.reward for t in batch.trajectories] == [1.0, 0.0]
    assert batch.stats['sampling/attempt_accuracy'] == .25
    assert batch.stats['data/cursor'] == 2


def test_partial_groups_never_save_a_cursor_that_loses_their_state(tmp_path):
    original = coordinator(tmp_path / 'original')
    original.rollout_runner._unfinished_groups = [object()]
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    with pytest.warns(UserWarning, match='unfinished rollout groups'):
        save_dataset_checkpoint(original, checkpoint)
    assert not (checkpoint / 'dataset_state.json').exists()


def test_native_generation_seed_counter_continues_on_resume(tmp_path):
    import threading
    original = coordinator(tmp_path / 'original')
    original.engine._generation_index = 137
    original.engine._seed_lock = threading.Lock()
    checkpoint = tmp_path / 'checkpoint'
    checkpoint.mkdir()
    save_dataset_checkpoint(original, checkpoint)
    resumed = coordinator(tmp_path / 'resumed')
    resumed.engine._generation_index = 0
    resumed.engine._seed_lock = threading.Lock()
    restore_dataset_checkpoint(resumed, checkpoint)
    assert resumed.engine._generation_index == 137


def test_dynamic_sampler_hard_budget_finishes_complete_waves_before_rejection(tmp_path):
    run = coordinator(tmp_path / 'original')
    from dataclasses import replace
    run.rollout_config = replace(run.rollout_config, max_total_requests=2)
    run.engine._plans.clear()
    run.engine._plans.extend([_Plan(text=BAD), _Plan(text=BAD)])
    with pytest.raises(RuntimeError, match='Request cap reached'):
        run.collect()
    assert run.rollout_runner.last_stats['requests_issued'] == 2
    assert run.rollout_runner.last_stats['aborted_request_count'] == 0
    assert run.rollout_runner._unfinished_groups == []


@pytest.mark.parametrize('concurrency,cap', [(3, 4), (4, 3), (4, 0)])
def test_dynamic_sampler_rejects_partial_group_limits_before_inference(tmp_path, concurrency, cap):
    from dataclasses import replace
    run = coordinator(tmp_path / 'original')
    run.rollout_runner.runner_config = replace(run.rollout_runner.runner_config, max_inflight_requests=concurrency)
    with pytest.raises(ValueError, match='whole groups|must be > 0'):
        run.rollout_config = replace(run.rollout_config, max_total_requests=cap)
        run.collect()
    assert len(run.engine._plans) == 32
