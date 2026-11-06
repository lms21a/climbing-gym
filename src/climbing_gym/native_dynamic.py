"""Fresh, full-wave dynamic sampling for the native dense-batch engine."""
from collections import defaultdict
from dataclasses import replace
import json
import logging
import math
from pathlib import Path
from statistics import pstdev
import time

from climbing_gym.rollout import DirectHarness, RolloutRunner


class NativeDynamicRunner(RolloutRunner):
    """Finish each inference wave, select variable-reward groups, then refill.

    Waves keep native decode batches full. Extra effective groups in the last
    wave are discarded, never reused after a policy update. Selection follows
    issue order, not response latency or reward magnitude.
    """

    def __init__(self, runner_config, *, audit_dir, group_reward_transform=None):
        super().__init__(runner_config)
        self.audit_dir = Path(audit_dir)
        self.audit_dir.mkdir(parents=True, exist_ok=True)
        self.update_index = 0
        self.group_reward_transform = group_reward_transform

    async def collect(self, *, env, engine, harness, rollout_config):
        config = self.runner_config
        n = rollout_config.num_rollouts
        cap = rollout_config.max_total_requests
        concurrency = config.max_inflight_requests
        if not config.use_dynamic_sampling or not isinstance(harness, DirectHarness) or n < 2:
            raise ValueError("Expected dynamic sampling with direct groups")
        max_calls = getattr(harness, 'max_generation_requests', 1)
        if not concurrency or concurrency % n or cap is None or cap % n or cap < config.effective_batch_size * n:
            raise ValueError("Concurrency and request cap must contain whole groups")
        target = config.effective_batch_size
        selected = []
        issued = tokens = effective = correct = penalized = waves = truncated = responses = 0
        zero_std = all_wrong = 0
        require_correct = bool(getattr(env.reward_fn, 'require_correct_in_group', False)
                               or getattr(self.group_reward_transform, 'require_correct_in_group', False))
        first_group_id = self._next_group_id
        started = time.monotonic()
        while len(selected) < target and issued < cap:
            wave_groups = min(concurrency // n, (cap - issued) // (n * max_calls))
            if wave_groups == 0:
                break
            wave = RolloutRunner(replace(config, effective_batch_size=wave_groups, use_dynamic_sampling=False))
            wave._next_group_id = self._next_group_id
            trajectories = await wave.collect(env=env, engine=engine, harness=harness,
                rollout_config=replace(rollout_config, max_total_requests=wave_groups * n * max_calls))
            self._next_group_id = wave._next_group_id
            if len(trajectories) != wave_groups * n or wave._unfinished_groups or wave.last_stats['aborted_request_count']:
                raise RuntimeError("Inference wave did not finish all fresh groups")
            groups = defaultdict(list)
            for trajectory in trajectories:
                groups[trajectory.group_id].append(trajectory)
            with (self.audit_dir / 'groups.jsonl').open('a') as handle:
                for group_id, group in sorted(groups.items()):
                    if self.group_reward_transform is not None:
                        group = self.group_reward_transform(group)
                    rewards = [float(t.evaluation.reward.reward) for t in group]
                    if len(group) != n or not all(math.isfinite(r) for r in rewards):
                        raise ValueError("Incomplete group or non-finite reward")
                    variable = pstdev(rewards) > config.reward_std_epsilon
                    has_correct = any(t.evaluation.reward.info.get('exact', t.evaluation.reward.info.get('exact_match', False)) for t in group)
                    excluded_all_wrong = require_correct and not has_correct
                    zero_std += int(not variable)
                    all_wrong += int(excluded_all_wrong)
                    useful = variable and not excluded_all_wrong
                    retained = useful and len(selected) < target
                    effective += int(useful)
                    if retained:
                        selected.append(group)
                    group_tokens = sum(len(turn.completion_token_ids) for t in group for turn in t.turns)
                    tokens += group_tokens
                    correct += sum(bool(t.evaluation.reward.info.get('exact', t.evaluation.reward.info.get('exact_match', False))) for t in group)
                    penalized += sum(t.evaluation.reward.info.get('doomloop_penalty', 0) > 0 for t in group)
                    truncated += sum(any(turn.finish_reason == 'length' for turn in t.turns) for t in group)
                    handle.write(json.dumps({'update': self.update_index, 'wave': waves, 'group_id': group_id,
                        'question_id': group[0].example.row.get('id'),
                        'dataset_index': group[0].example.dataset_index, 'rewards': rewards,
                        'effective': useful, 'selected': retained, 'completion_tokens': group_tokens,
                        'completion_lengths': [sum(len(turn.completion_token_ids) for turn in t.turns) for t in group],
                        'exact': [bool(t.evaluation.reward.info.get('exact', t.evaluation.reward.info.get('exact_match', False))) for t in group],
                        'excluded_all_wrong': excluded_all_wrong, 'zero_std': not variable}) + '\n')
            issued += int(wave.last_stats['requests_issued'])
            responses += len(trajectories)
            waves += 1
            logging.getLogger(__name__).info(
                'Dynamic sampling update %d: wave=%d selected=%d/%d requests=%d/%d correct=%d penalized=%d',
                self.update_index, waves, len(selected), target, issued, cap, correct, penalized)
        elapsed = time.monotonic() - started
        self.last_stats = {
            'effective_group_target': target,
            'selected_group_count': len(selected), 'effective_group_count': effective,
            'sampled_group_count': self._next_group_id - first_group_id, 'dropped_zero_std_group_count': zero_std,
            'sampling/excluded_all_wrong_groups': all_wrong,
            'requests_issued': issued, 'requests_completed': issued, 'aborted_request_count': 0,
            'carried_in_unfinished_group_count': 0, 'carried_out_unfinished_group_count': 0,
            'sampling/waves': waves, 'sampling/extra_effective_groups': effective - len(selected),
            'sampling/trajectories': responses,
            'sampling/attempt_accuracy': correct / max(1, responses),
            'sampling/doomloop_fraction': penalized / max(1, responses),
            'sampling/acceptance_fraction': effective / max(1, responses // n),
            'sampling/completion_tokens': tokens, 'sampling/tokens_per_second': tokens / max(elapsed, 1e-9),
            'sampling/seconds': elapsed, 'sampling/truncated_fraction': truncated / max(1, responses),
        }
        with (self.audit_dir / 'sampling.jsonl').open('a') as handle:
            handle.write(json.dumps({'update': self.update_index, 'stats': self.last_stats}) + '\n')
        if len(selected) != target:
            raise RuntimeError(f"Request cap reached with {len(selected)}/{target} variable-reward groups; no optimizer update")
        self.update_index += 1
        return [trajectory for group in selected for trajectory in group]
