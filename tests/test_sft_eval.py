from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import climbing_gym.sft_eval as module
from climbing_gym.eval import EvalLoggingConfig
from climbing_gym.sft_eval import PeriodicSFTEvaluator, SFTEvaluationConfig


@dataclass
class FakeRun:
    engine: object
    logging_config: object
    close_engine_on_finish: bool = True


def test_periodic_eval_reuses_validated_native_run_syncs_current_model_and_closes(tmp_path, monkeypatch):
    calls = []
    class Engine:
        def release_inference_buffers(self): calls.append('release')
        def sync_from_model(self, model): calls.append(('sync', model))
        def close(self): calls.append('close')
    engine = Engine()
    def load(path):
        calls.append(('load_and_validate_preflight', path))
        return FakeRun(engine, EvalLoggingConfig())
    def evaluate(run):
        assert not run.close_engine_on_finish
        assert run.engine is engine
        path = Path(run.logging_config.trace_path)
        path.parent.mkdir(parents=True)
        path.write_text('Complete actual native traces are handled by standalone run_eval')
        return {'pass@1': .5}
    monkeypatch.setattr(module, 'load_run_from_toml', load)
    monkeypatch.setattr(module, 'run_eval', evaluate)
    model = torch.nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    hook = PeriodicSFTEvaluator(SFTEvaluationConfig('eval.toml', 2), model=model, optimizer=optimizer,
                               train_device=torch.device('cpu'), output_dir=tmp_path)
    assert hook.evaluate(0) == {}
    assert hook.evaluate(1)['eval/pass@1'] == .5
    assert hook.evaluate(3)['eval/pass@1'] == .5
    assert [c for c in calls if isinstance(c, tuple) and c[0]=='load_and_validate_preflight'] == [('load_and_validate_preflight', 'eval.toml')]
    assert (tmp_path/'step_1/traces.md').is_file()
    assert (tmp_path/'step_3/traces.md').is_file()
    assert model.training
    hook.close()
    assert calls[-1] == 'close'


def test_evaluation_failure_restores_training_device_and_mode(tmp_path, monkeypatch):
    calls = []
    engine = SimpleNamespace(release_inference_buffers=lambda: calls.append('release'),
                             sync_from_model=lambda model: None, close=lambda: None)
    monkeypatch.setattr(module, 'load_run_from_toml', lambda path: FakeRun(engine, EvalLoggingConfig()))
    def fail(run): raise RuntimeError('native evaluation failed')
    monkeypatch.setattr(module, 'run_eval', fail)
    model = torch.nn.Linear(2, 2); model.eval()
    optimizer = torch.optim.AdamW(model.parameters())
    hook = PeriodicSFTEvaluator(SFTEvaluationConfig('eval.toml', 1), model=model, optimizer=optimizer,
                               train_device=torch.device('cpu'), output_dir=tmp_path)
    with pytest.raises(RuntimeError, match='evaluation failed'):
        hook.evaluate(0)
    assert not model.training
    assert calls == ['release', 'release']


def test_sft_evaluation_config_requires_both_file_and_interval():
    with pytest.raises(ValueError): SFTEvaluationConfig('eval.toml', 0)
    with pytest.raises(ValueError): SFTEvaluationConfig('', 2)


def test_constructor_closes_owned_engine_after_initial_buffer_release_failure(tmp_path, monkeypatch):
    calls = []
    failure = RuntimeError('initial buffer release failed')
    def release():
        calls.append('release')
        raise failure
    def close():
        calls.append('close')
        raise OSError('engine cleanup failed')
    engine = SimpleNamespace(release_inference_buffers=release, close=close)
    monkeypatch.setattr(module, 'load_run_from_toml', lambda path: FakeRun(engine, EvalLoggingConfig()))
    model = torch.nn.Linear(2, 2)
    with pytest.raises(RuntimeError) as caught:
        PeriodicSFTEvaluator(SFTEvaluationConfig('eval.toml', 1), model=model,
            optimizer=torch.optim.AdamW(model.parameters()), train_device='cpu', output_dir=tmp_path)
    assert caught.value is failure
    assert calls == ['release', 'close']
    assert any('engine cleanup failed' in note for note in caught.value.__notes__)


def test_eval_release_failure_cannot_prevent_restoration_or_mask_eval_failure(tmp_path, monkeypatch):
    calls = []
    failure = RuntimeError('evaluation failed first')
    def release():
        calls.append('release')
        if calls.count('release') == 2:
            raise OSError('buffer cleanup failed')
    engine = SimpleNamespace(release_inference_buffers=release, sync_from_model=lambda model: None)
    monkeypatch.setattr(module, 'load_run_from_toml', lambda path: FakeRun(engine, EvalLoggingConfig()))
    class Model(torch.nn.Linear):
        def to(self, device):
            calls.append(('model.to', str(device)))
            return super().to(device)
        def train(self, mode=True):
            calls.append(('model.train', mode))
            return super().train(mode)
    model = Model(2, 2); model.eval()
    optimizer = torch.optim.AdamW(model.parameters())
    original_move = module._move_optimizer
    def move_optimizer(optimizer, device):
        calls.append(('optimizer.to', str(device)))
        original_move(optimizer, device)
    monkeypatch.setattr(module, '_move_optimizer', move_optimizer)
    def fail(run): raise failure
    monkeypatch.setattr(module, 'run_eval', fail)
    evaluator = PeriodicSFTEvaluator(SFTEvaluationConfig('eval.toml', 1), model=model,
        optimizer=optimizer, train_device='cpu', output_dir=tmp_path)
    with pytest.raises(RuntimeError) as caught:
        evaluator.evaluate(0)
    assert caught.value is failure
    assert not model.training
    assert calls[-3:] == [('model.to', 'cpu'), ('optimizer.to', 'cpu'), ('model.train', False)]
    assert any('buffer cleanup failed' in note for note in caught.value.__notes__)
