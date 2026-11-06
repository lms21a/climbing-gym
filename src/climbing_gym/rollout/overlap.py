"""Bounded cohort scheduling with a separately owned inference snapshot."""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import time


def one_batch_ahead(count, *, collect, snapshot, learn):
    """Overlap the next complete cohort with this cohort's learner update.

    ``snapshot(version)`` runs on the calling thread only after the previous
    collection has drained. It must copy learner weights into independent
    inference storage and establish any device-stream dependencies.
    ``collect(index, version)`` owns the producer thread. ``learn`` receives
    ``(payload, index, cycle_start)`` on the calling thread.

    Exactly one cohort is outstanding, preserving submission order without
    completion-time filtering. With one optimizer update per cohort, behavior
    versions are 0, 0, 1, 2, ...; maximum intentional lag is one update. The
    executor drains even if either callback fails, before caller teardown.
    """
    if type(count) is not int or count < 1:
        raise ValueError('count must be a positive integer')
    start=time.perf_counter()
    snapshot(0)
    with ThreadPoolExecutor(max_workers=1,thread_name_prefix='native-rollout') as pool:
        future=pool.submit(collect,0,0)
        for index in range(count):
            cycle=start if index==0 else time.perf_counter()
            payload=future.result()
            if index+1<count:
                snapshot(index)
                future=pool.submit(collect,index+1,index)
            learn(payload,index,cycle)
