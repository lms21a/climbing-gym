"""Native adapter ownership and public sampling-boundary regressions (CPU)."""
import asyncio
from concurrent.futures import ThreadPoolExecutor
import threading
from types import SimpleNamespace
import weakref

import pytest

from climbing_gym.inference import InferenceEngineConfig, SamplingConfig
from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
from test_lfm25_async_engine import _FakeTokenEngine, _FakeTokenizer


def owned_adapter(monkeypatch, *, fail_setup=False, fail_release=False):
    events, references = [], []

    class Raw(_FakeTokenEngine):
        def __init__(self):
            super().__init__()
            self.decoder = SimpleNamespace(engine=self)  # Real decoder/engine cycle.
            self.fail_release = fail_release

        def release_inference_buffers(self):
            events.append('release')
            if self.fail_release:
                raise RuntimeError('active native session')

        def enable_cuda_graphs(self, **kwargs):
            if fail_setup:
                raise ValueError('graph setup failed')

    def load(*args, **kwargs):
        raw = Raw()
        references.append(weakref.ref(raw))
        return raw

    monkeypatch.setattr('climbing_gym.inference.lfm25_async_engine.load_lfm2_tokenizer', lambda *a: _FakeTokenizer())
    monkeypatch.setattr('climbing_gym.inference.lfm25_async_engine.LFM25Engine.from_pretrained', load)
    adapter = LFM25AsyncInferenceEngine(InferenceEngineConfig(model_path='fixture'),
                                       compile_decode=False, cuda_graphs=fail_setup)
    return adapter, events, references


def test_owned_close_retires_native_cycle_and_stop_tables(monkeypatch):
    adapter, events, references = owned_adapter(monkeypatch)
    class Buffer:
        pass
    adapter._stop_tables = {'fixture': Buffer()}
    buffer = weakref.ref(adapter._stop_tables['fixture'])
    assert references[0]() is not None
    adapter.close()
    assert references[0]() is None
    assert buffer() is None and adapter._stop_tables == {}
    assert events == ['release']
    adapter.close()  # Idempotent; later cleanup of already-retired buffers is safe.
    adapter.release_inference_buffers()
    with pytest.raises(RuntimeError, match='closed'):
        _ = adapter.raw_engine


def test_close_preserves_externally_supplied_engine_and_graphs():
    raw = _FakeTokenEngine()
    raw.graph = object()
    raw.release_inference_buffers = lambda: pytest.fail('adapter does not own shared engine')
    graph = raw.graph
    adapter = LFM25AsyncInferenceEngine.from_components(engine=raw, tokenizer=_FakeTokenizer())
    adapter.close()
    assert raw.graph is graph
    assert adapter._engine is None


def test_owned_close_waits_for_worker_before_releasing_buffers(monkeypatch):
    adapter, events, _ = owned_adapter(monkeypatch)
    entered, finish = threading.Event(), threading.Event()

    def worker():
        with adapter._engine_lock:
            events.append('worker entered')
            entered.set()
            assert finish.wait(timeout=3)
            events.append('worker drained')

    with ThreadPoolExecutor(max_workers=2) as pool:
        working = pool.submit(worker)
        assert entered.wait(timeout=3)
        closing = pool.submit(adapter.close)
        assert not closing.done()
        assert events == ['worker entered']
        finish.set()
        working.result(timeout=3)
        closing.result(timeout=3)
    assert events == ['worker entered', 'worker drained', 'release']


def test_owned_close_keeps_reference_if_active_session_prevents_cleanup(monkeypatch):
    adapter, _, references = owned_adapter(monkeypatch, fail_release=True)
    with pytest.raises(RuntimeError, match='active native session'):
        adapter.close()
    assert references[0]() is adapter._engine
    adapter._engine.fail_release = False
    adapter.close()
    assert references[0]() is None


def test_failed_native_setup_releases_owned_actor(monkeypatch):
    events = []
    class Raw(_FakeTokenEngine):
        def enable_cuda_graphs(self, **kwargs):
            raise ValueError('graph setup failed')
        def release_inference_buffers(self):
            events.append('released')
    monkeypatch.setattr('climbing_gym.inference.lfm25_async_engine.load_lfm2_tokenizer', lambda *a: _FakeTokenizer())
    monkeypatch.setattr('climbing_gym.inference.lfm25_async_engine.LFM25Engine.from_pretrained', lambda *a, **kw: Raw())
    with pytest.raises(ValueError, match='graph setup failed'):
        LFM25AsyncInferenceEngine(InferenceEngineConfig(model_path='fixture'), cuda_graphs=True)
    assert events == ['released']


def test_request_waiting_for_enqueue_cannot_revive_closed_adapter():
    adapter = LFM25AsyncInferenceEngine.from_components(engine=_FakeTokenEngine(), tokenizer=_FakeTokenizer())
    async def check():
        async with adapter._lock():
            task = asyncio.create_task(adapter.generate('fixture'))
            await asyncio.sleep(0)
            adapter.close()
        with pytest.raises(RuntimeError, match='closed'):
            await task
        assert not adapter._pending
    asyncio.run(check())


def test_retired_worker_fails_closed_before_dereferencing_native_engine():
    adapter = LFM25AsyncInferenceEngine.from_components(engine=_FakeTokenEngine(), tokenizer=_FakeTokenizer())
    adapter.close()
    request = SimpleNamespace(prompt_token_ids=(1, 2), sampling=SamplingConfig())
    with pytest.raises(RuntimeError, match='closed'):
        adapter._generate_batch_sync((request,))


@pytest.mark.parametrize('options', [
    {'temperature': float('nan')}, {'temperature': float('inf')},
    {'top_p': 0.}, {'top_p': float('nan')}, {'top_k': 1.5},
    {'max_tokens': 1.5}, {'max_tokens': True}, {'n': 1.5}, {'logprobs': True},
    {'stop': 'END'}, {'stop': ('',)}, {'extra_params': {'presence_penalty': .5}},
    {'extra_params': {'repetition_penalty': float('nan')}},
    {'extra_params': {'repetition_penalty': float('inf')}},
    {'extra_params': {'ignore_eos': 'false'}},
    {'extra_params': {'behavior_topk': 2.5}},
])
def test_public_native_sampling_rejects_invalid_or_ignored_options(options):
    with pytest.raises(ValueError):
        SamplingConfig(**options)


def test_foreign_sampling_preserves_stops_and_supported_extras():
    adapter = LFM25AsyncInferenceEngine.from_components(engine=object(), tokenizer=object())
    foreign = SimpleNamespace(temperature='1', top_k=-1, max_new_tokens=7,
        stop=['END'], extra_params={'ignore_eos': True, 'repetition_penalty': 1.1})
    converted = adapter._resolve_sampling(foreign)
    assert converted.max_tokens == 7 and converted.temperature == 1.
    assert converted.stop == ('END',)
    assert converted.extra_params == foreign.extra_params
    assert adapter._resolve_sampling(SimpleNamespace(stop=None, extra_params=None)).stop == ()
    with pytest.raises(TypeError, match='mapping'):
        adapter._resolve_sampling({'max_tokens': 7})


@pytest.mark.parametrize('maximum', [0., 1.])
@pytest.mark.parametrize('bad_gradient', [float('nan'), float('inf')])
def test_sft_failed_backward_preserves_weights_and_optimizer(tmp_path, monkeypatch, maximum, bad_gradient):
    import copy
    import torch
    import climbing_gym.sft as sft
    from test_sft import _TinyTrainModel, _FakeTokenizer as SFTTokenizer, _write_sft_toml_config

    checkpoint = tmp_path / 'checkpoints'
    config = _write_sft_toml_config(tmp_path, checkpoint_root=checkpoint)
    config.write_text(config.read_text().replace('max_grad_norm = 1.0', f'max_grad_norm = {maximum}'))
    model = _TinyTrainModel()
    before = copy.deepcopy(model.state_dict())
    next(model.parameters()).register_hook(lambda gradient: torch.full_like(gradient, bad_gradient))
    monkeypatch.setattr(sft, 'build_model', lambda *a, **kw: model)
    monkeypatch.setattr(sft, 'AutoTokenizer', SimpleNamespace(from_pretrained=lambda *a, **kw: SFTTokenizer()))
    monkeypatch.setattr(sft, 'maybe_compile', lambda function, **kwargs: function)
    run = sft.load_run_from_toml(config)
    monkeypatch.setattr(sft, 'load_run_from_toml', lambda _: run)
    with pytest.raises(FloatingPointError, match='gradient norm'):
        sft.main([str(config)])
    assert not run.policy_optimizer.state
    assert all(torch.equal(value, before[key]) for key, value in model.state_dict().items())
    assert not list(checkpoint.glob('step_*'))


@pytest.mark.parametrize('maximum', [float('nan'), float('inf'), -float('inf')])
def test_sft_rejects_nonfinite_clip_before_forward(tmp_path, monkeypatch, maximum):
    from dataclasses import replace
    import climbing_gym.sft as sft
    from test_sft import _TinyTrainModel, _FakeTokenizer as SFTTokenizer, _write_sft_toml_config

    config = _write_sft_toml_config(tmp_path, checkpoint_root=tmp_path / 'checkpoints')
    model = _TinyTrainModel()
    monkeypatch.setattr(sft, 'build_model', lambda *a, **kw: model)
    monkeypatch.setattr(sft, 'AutoTokenizer', SimpleNamespace(from_pretrained=lambda *a, **kw: SFTTokenizer()))
    run = sft.load_run_from_toml(config)
    run = replace(run, config=replace(run.config, sft=replace(run.config.sft, max_grad_norm=maximum)))
    monkeypatch.setattr(sft, 'load_run_from_toml', lambda _: run)
    monkeypatch.setattr(model, 'forward', lambda *a, **kw: pytest.fail('forward ran before rejecting invalid clip'))
    with pytest.raises(ValueError, match='max_grad_norm must be finite'):
        sft.main([str(config)])
