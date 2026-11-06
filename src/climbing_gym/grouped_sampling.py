"""Disk-backed, single-use surplus groups with an inclusive policy-lag bound."""
from __future__ import annotations

from pathlib import PurePosixPath


class SurplusGroupQueue:
    """Keep FIFO references to immutable native rollout archives, not tensors.

    Selecting for a future update must not expire groups against that future
    version: they can still be valid for the learner's current version. Only
    ``expire`` discards groups, and callers supply the actual learner version.
    """

    def __init__(self, max_policy_lag=4, entries=()):
        if type(max_policy_lag) is not int or max_policy_lag < 0:
            raise ValueError('Expected a nonnegative integer policy-lag limit')
        self.max_policy_lag = max_policy_lag
        self.entries = []
        self.extend(entries)

    def __len__(self):
        return len(self.entries)

    def extend(self, entries):
        new = [dict(entry) for entry in entries]
        seen = {entry['dataset_index'] for entry in self.entries}
        for entry in new:
            path = PurePosixPath(entry['wave_file'])
            if path.is_absolute() or '..' in path.parts:
                raise ValueError('Expected a relative immutable native-wave path')
            if any(type(entry.get(key)) is not int or entry[key] < 0
                   for key in ('group', 'dataset_index', 'behavior_version')):
                raise ValueError('Invalid complete-group archive reference')
            if entry['dataset_index'] in seen:
                raise ValueError('Duplicate surplus group')
            seen.add(entry['dataset_index'])
        self.entries.extend(new)

    def expire(self, learner_version):
        self._check_version(learner_version)
        if any(entry['behavior_version'] > learner_version for entry in self.entries):
            raise ValueError('Behavior policy is newer than the actual learner')
        expired = [entry for entry in self.entries
                   if learner_version-entry['behavior_version'] > self.max_policy_lag]
        self.entries = [entry for entry in self.entries
                        if learner_version-entry['behavior_version'] <= self.max_policy_lag]
        return expired

    def take(self, count, learner_version):
        self._check_version(learner_version)
        if type(count) is not int or count < 0:
            raise ValueError('Expected a nonnegative number of complete groups')
        selected, remaining = [], []
        for entry in self.entries:
            lag = learner_version-entry['behavior_version']
            if len(selected) < count and 0 <= lag <= self.max_policy_lag:
                selected.append(entry)
            else:
                remaining.append(entry)
        self.entries = remaining
        return selected

    def state_dict(self):
        return dict(max_policy_lag=self.max_policy_lag, entries=[dict(e) for e in self.entries])

    @staticmethod
    def _check_version(version):
        if type(version) is not int or version < 0:
            raise ValueError('Expected a nonnegative integer learner version')
