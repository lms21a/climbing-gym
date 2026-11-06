from threading import Event, Lock

import pytest

from climbing_gym.rollout.overlap import one_batch_ahead


def test_overlap_keeps_cohorts_ordered_and_refreshes_only_drained_snapshots():
    lock=Lock()
    active=False
    policy_version=0
    engine_version=None
    started=[Event() for _ in range(4)]
    consumed=[]
    snapshots=[]

    def snapshot(version):
        nonlocal engine_version
        with lock:
            assert not active
            assert version==policy_version
            engine_version=version
            snapshots.append(version)

    def collect(index,version):
        nonlocal active
        with lock:
            assert not active
            assert engine_version==version
            active=True
        started[index].set()
        with lock:
            assert engine_version==version
            active=False
        return index,version

    def learn(payload,index,cycle):
        nonlocal policy_version
        if index+1<4:
            # The next producer actually runs before this update completes.
            assert started[index+1].wait(timeout=5)
        assert payload[0]==index
        assert index-payload[1] in (0,1)
        assert cycle>0
        consumed.append(payload)
        policy_version+=1

    one_batch_ahead(4,collect=collect,snapshot=snapshot,learn=learn)
    assert consumed==[(0,0),(1,0),(2,1),(3,2)]
    assert snapshots==[0,0,1,2]
    assert policy_version==4


def test_collection_failure_does_not_update_on_missing_or_partial_cohorts():
    learned=[]
    def collect(index,version):
        if index==1:
            raise RuntimeError('collector failed')
        return index
    with pytest.raises(RuntimeError,match='collector failed'):
        one_batch_ahead(3,collect=collect,snapshot=lambda version:None,
                        learn=lambda payload,index,cycle:learned.append(payload))
    assert learned==[0]


def test_learner_failure_drains_the_outstanding_producer_before_teardown():
    next_started=Event()
    next_finished=Event()
    def collect(index,version):
        if index==1:
            next_started.set()
            next_finished.set()
        return index
    def learn(payload,index,cycle):
        assert next_started.wait(timeout=5)
        raise RuntimeError('learner failed')
    with pytest.raises(RuntimeError,match='learner failed'):
        one_batch_ahead(3,collect=collect,snapshot=lambda version:None,learn=learn)
    assert next_finished.is_set()


@pytest.mark.parametrize('count',[0,-1,True,1.5])
def test_invalid_cohort_counts(count):
    with pytest.raises(ValueError):
        one_batch_ahead(count,collect=None,snapshot=None,learn=None)
