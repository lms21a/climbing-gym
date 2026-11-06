"""Group-relative rewards with Dr. GRPO's two normalization corrections.

The token MIS / sparse score-centering surrogate is supplied by offpolicy.py.
Neither realized response lengths nor group reward standard deviations divide
the gradient. Dynamic sampling intentionally conditions on mixed-outcome groups.
"""
from __future__ import annotations

from collections import defaultdict
import math

import torch


def classify_groups(rewards, group_size):
    if group_size < 2 or not rewards or len(rewards) % group_size:
        raise ValueError('Expected complete groups with at least two rollouts')
    if any(r not in (0., 1.) for r in rewards):
        raise ValueError('Dynamic Futoshiki sampling requires binary exact-match rewards')
    result = dict(mixed=[], all_wrong=[], all_correct=[])
    for group in range(len(rewards)//group_size):
        successes = sum(rewards[group*group_size:(group+1)*group_size])
        key = 'all_wrong' if successes == 0 else 'all_correct' if successes == group_size else 'mixed'
        result[key].append(group)
    return result


def grouped_advantages(trajectories, *, group_size, groups_per_update):
    if len(trajectories) != group_size*groups_per_update or group_size < 2:
        raise ValueError('Incomplete retained group batch')
    groups = defaultdict(list)
    for index, trajectory in enumerate(trajectories):
        if len(trajectory.turns) != 1 or not trajectory.finished:
            raise ValueError('Expected complete single-turn rollouts')
        turn = trajectory.turns[0]
        if (not turn.prompt_token_ids or not turn.completion_token_ids
                or turn.completion_logprobs is None
                or len(turn.completion_logprobs) != len(turn.completion_token_ids)
                or not all(math.isfinite(p) and p <= 0 for p in turn.completion_logprobs)):
            raise ValueError('Missing native behavior token probabilities')
        if float(trajectory.metadata.get('policy_objective_weight', 1.)) != 1.:
            raise ValueError('Unexpected trajectory reweighting')
        groups[trajectory.group_id].append(index)
    if len(groups) != groups_per_update:
        raise ValueError('Incorrect number of retained prompts')
    advantages = [0.]*len(trajectories)
    seen = set()
    means = []
    for indices in groups.values():
        selected = [trajectories[i] for i in indices]
        if len(indices) != group_size or sorted(t.rollout_index for t in selected) != list(range(group_size)):
            raise ValueError('Incomplete or duplicated rollout indices within a group')
        dataset_indices = {t.example.dataset_index for t in selected}
        if (len(dataset_indices) != 1 or seen & dataset_indices
                or len({tuple(t.turns[0].prompt_token_ids) for t in selected}) != 1
                or len({t.example.prompt for t in selected}) != 1):
            raise ValueError('Group prompt identity mismatch or duplicate retained prompt')
        seen.update(dataset_indices)
        rewards = [float(t.evaluation.reward.reward) for t in selected]
        if any(r not in (0., 1.) for r in rewards) or not 0 < sum(rewards) < group_size:
            raise ValueError('Retain only mixed exact-match groups')
        mean = math.fsum(rewards)/group_size
        means.append(mean)
        for i, reward in zip(indices, rewards):
            advantages[i] = reward-mean
    return advantages, dict(
        **{'group/count': len(groups), 'group/size': group_size,
           'group/mean_reward': math.fsum(means)/len(means),
           'group/advantage_mean': math.fsum(advantages)/len(advantages),
           'group/std_normalization': 0.})


def fixed_length_loss(objective, token_advantages, *, total_sequences, max_tokens):
    if (objective.shape != token_advantages.shape or objective.ndim != 1
            or total_sequences <= 0 or max_tokens <= 0):
        raise ValueError('Invalid global Dr. GRPO loss normalization')
    # One global B * MAX_TOKENS denominator across every memory pack. No
    # renormalization after MIS masks, and no division by emitted token counts.
    return -(objective*token_advantages.detach()).sum()/(total_sequences*max_tokens)
