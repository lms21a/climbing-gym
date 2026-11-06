from copy import deepcopy
from types import SimpleNamespace
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch

from climbing_gym.cpu_master_adamw import CPUStateMasterAdamW
from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
from climbing_gym.inference.lfm25_engine import LFM25Engine
from climbing_gym.training_memory import (
    DEFAULT_OPTIMIZER_CHUNK_ELEMENTS, DEFAULT_OPTIMIZER_WORKSPACE_BYTES,
    optimizer_chunk_elements, training_memory_defaults,
)


def test_legacy_whole_parameter_setting_cannot_bypass_the_workspace_default():
    protocol = dict(optimizer_chunk_elements=2**31 - 1, optimizer_workspace_bytes=48 * 2**20,
                    optimizer_transfer_mode="whole_parameter_reusable_workspace",
                    async_collection=True, lookahead_collections=1, cache_resident=True,
                    native_batch_size=256, max_num_seqs=256, max_tokens=4096,
                    logprobs=1, behavior_topk=32)
    original = deepcopy(protocol)
    resolved = training_memory_defaults(protocol)
    assert protocol == original  # Frozen launch artifacts retain their original settings.
    assert resolved["optimizer_chunk_elements"] == DEFAULT_OPTIMIZER_CHUNK_ELEMENTS
    assert resolved["optimizer_workspace_bytes"] == DEFAULT_OPTIMIZER_WORKSPACE_BYTES
    assert resolved["release_inference_buffers_before_update"]
    assert not resolved["async_collection"] and not resolved["cache_resident"]
    assert not resolved["actor_learner_overlap"] and resolved["lookahead_collections"] == 0
    assert resolved["maximum_policy_lag"] == 1
    for key in ("native_batch_size", "max_num_seqs", "max_tokens", "logprobs", "behavior_topk"):
        assert resolved[key] == original[key]
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = CPUStateMasterAdamW([parameter], lr=.01, weight_decay=0., chunk_elements=2**31 - 1)
    assert optimizer.chunk_elements == resolved["optimizer_chunk_elements"]
    assert optimizer.workspace_bytes == resolved["optimizer_workspace_bytes"]


def test_explicit_overlap_opt_out_keeps_its_schedule_and_still_bounds_optimizer():
    resolved = training_memory_defaults(dict(release_inference_buffers_before_update=False,
                                            async_collection=True, cache_resident=True,
                                            lookahead_collections=1, optimizer_chunk_elements=2**31 - 1))
    assert resolved["async_collection"] and resolved["cache_resident"]
    assert resolved["lookahead_collections"] == 1
    assert resolved["optimizer_workspace_bytes"] == DEFAULT_OPTIMIZER_WORKSPACE_BYTES


def test_release_flag_requires_a_boolean():
    with pytest.raises(ValueError, match="must be a boolean"):
        training_memory_defaults(dict(release_inference_buffers_before_update="false"))


@pytest.mark.parametrize("chunk,budget", [(0, 48), (-1, 48), (1.5, 48), (True, 48),
                                         (1, 0), (1, 11), (1, 12.5)])
def test_invalid_workspace_geometry_fails_before_device_allocations(chunk, budget):
    with pytest.raises(ValueError):
        optimizer_chunk_elements(chunk, budget)


def test_smaller_explicit_budget_bounds_all_three_state_buffers():
    parameter = torch.nn.Parameter(torch.ones(1))
    optimizer = CPUStateMasterAdamW([parameter], lr=.01, weight_decay=0.,
                                   chunk_elements=2**31 - 1, workspace_bytes=1025)
    assert optimizer.chunk_elements == 85
    assert optimizer.workspace_bytes == 1020


def token_engine(decoder):
    engine = LFM25Engine.__new__(LFM25Engine)
    engine.device = torch.device("cuda")
    engine._cuda_decoder = decoder
    engine.weights = object()
    return engine


def test_release_keeps_weights_and_never_closes_an_active_session(monkeypatch):
    calls = []
    session = object()
    decoder = SimpleNamespace(_active_session=session, graphs={1: object()}, packed_weights={},
                              fp8_weights={}, clear=lambda: calls.append("clear"))
    engine = token_engine(decoder)
    weights = engine.weights
    monkeypatch.setattr(torch.cuda, "synchronize", lambda device: calls.append("sync"))
    monkeypatch.setattr(torch.cuda, "empty_cache", lambda: calls.append("empty"))
    with pytest.raises(RuntimeError, match="Close the native generation session"):
        engine.release_inference_buffers()
    assert calls == [] and decoder._active_session is session
    decoder._active_session = None
    engine.release_inference_buffers()
    assert calls == ["sync", "clear", "empty"]
    assert engine.weights is weights


def test_sleep_waits_for_native_execution_before_releasing_buffers():
    released = threading.Event()
    started = threading.Event()
    raw = SimpleNamespace(release_inference_buffers=released.set)
    engine = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=object())

    def park():
        started.set()
        engine.sleep()

    with ThreadPoolExecutor(max_workers=1) as pool:
        with engine._engine_lock:
            future = pool.submit(park)
            assert started.wait(timeout=2)
            assert not released.is_set()
        future.result(timeout=2)
    assert released.is_set()


@pytest.mark.parametrize("method", ["sequence_kl", "mis", "mis_sc"])
def test_shared_update_releases_before_the_first_learner_forward(method):
    from climbing_gym.flashreinforce import update_policy
    from test_flashreinforce import config
    from test_offpolicy import make_rows
    from rl_fixtures import TinyPolicy

    policy = TinyPolicy()
    traces = make_rows(policy)
    released = False

    def release():
        nonlocal released
        released = True

    forward = policy.get_hidden_states

    def checked_forward(*args, **kwargs):
        assert released, "Learner allocated activations while inference buffers were resident"
        return forward(*args, **kwargs)

    policy.get_hidden_states = checked_forward
    run = SimpleNamespace(config=config(flashreinforce_objective=method), policy_model=policy,
                          policy_optimizer=torch.optim.SGD(policy.parameters(), lr=.01),
                          engine=SimpleNamespace(release_inference_buffers=release))
    metrics, _ = update_policy(run, traces)
    assert released and metrics["policy/optimizer_steps"] == 1
