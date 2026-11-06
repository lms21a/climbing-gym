"""Independent CPU regressions for example continuation and native identities.

Tiny models and explicit doubles below never perform model evaluation or CUDA
work. They check ordered lifecycle/state transitions of the runnable examples.
"""
from copy import deepcopy
from dataclasses import asdict, replace
import asyncio
import importlib
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest
import torch

EXAMPLE = Path(__file__).resolve().parents[1] / 'examples/futoshiki_hill_climb'
sys.path.insert(0, str(EXAMPLE))
workflow = importlib.import_module('projection')
native = importlib.import_module('native_projection')
history = importlib.import_module('historical')
rl = importlib.import_module('rl')
from test_hill_climb_runtime import Policy, make_run


class TinySFT(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = torch.nn.Embedding(16, 4)
        self.lm_head = torch.nn.Linear(4, 16)

    def get_hidden_states(self, inputs, positions):
        return self.embed(inputs)

    def save_pretrained(self, path):
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        torch.save(self.state_dict(), path / 'weights.pt')
        (path / 'config.json').write_text('{"fixture":true}')


class TinyTokenizer:
    def save_pretrained(self, path):
        (Path(path) / 'tokenizer.json').write_text('{"fixture":"tokenizer"}')


@pytest.mark.parametrize('changed', ['config.json', 'weights.pt', 'tokenizer.json'])
def test_sft_resume_binds_saved_policy_and_tokenizer_bytes(tmp_path, changed):
    model = TinySFT()
    optimizer = history.FP32MasterAdamW(model.parameters())
    workflow.save_training(model, TinyTokenizer(), optimizer, tmp_path, {})
    resume = tmp_path / 'resume'
    state = workflow.read_json(resume / 'state.json')
    workflow.validate_saved_policy(resume, state)
    path = resume / 'policy' / changed
    path.write_bytes(path.read_bytes() + b'changed bytes')
    with pytest.raises(ValueError, match='policy or tokenizer'):
        workflow.validate_saved_policy(resume, state)


def test_sft_pending_epoch_offloads_before_evaluator_and_restores_before_update(tmp_path, monkeypatch):
    model = TinySFT()
    optimizer = history.FP32MasterAdamW(model.parameters())
    rows = [dict(id=str(i), row=dict(tier=history.TIERS[i % 2]),
                 prompt_token_ids=[1, 2], completion_token_ids=[3 + i, 7])
            for i in range(4)]
    protocol = replace(history.PROTOCOLS['projection_reset'], batch_size=2,
                       retained_epoch=2, head_chunk=2, packed_tokens=32)
    state = dict(next_epoch=1, next_batch=0, optimizer_updates=2,
                 pending_epoch=1, protocol=asdict(protocol),
                 data_sha256=workflow.sha(json.dumps(rows, sort_keys=True).encode()))
    workflow.save_training(model, TinyTokenizer(), optimizer, tmp_path, state)
    events = []
    move = workflow.move_training
    backward = workflow.backward_completion_loss

    def traced_move(*args):
        events.append(('move', args[2]))
        move(*args)

    def evaluate(epoch, checkpoint):
        events.append(('evaluate', epoch))
        if epoch == 1:
            assert events == [('move', 'cpu'), ('evaluate', 1)]
        return {}

    def traced_backward(*args, **kwargs):
        if not any(event[0] == 'backward' for event in events):
            assert events[:3] == [('move', 'cpu'), ('evaluate', 1), ('move', 'cpu')]
        events.append(('backward', None))
        return backward(*args, **kwargs)

    monkeypatch.setattr(workflow, 'move_training', traced_move)
    monkeypatch.setattr(workflow, 'backward_completion_loss', traced_backward)
    logger = SimpleNamespace(log_step=lambda **kwargs: None)
    assert workflow.train_loop(model, TinyTokenizer(), optimizer, rows, protocol,
                               tmp_path, logger, device='cpu', evaluator=evaluate,
                               resume_state=state) == 4
    assert workflow.read_json(tmp_path / 'resume/state.json')['pending_epoch'] is None


def test_failed_pending_epoch_keeps_pending_marker_and_does_not_reload(tmp_path, monkeypatch):
    protocol = history.PROTOCOLS['solver_projection_256']
    rows = []
    state = dict(next_epoch=1, next_batch=0, optimizer_updates=0, pending_epoch=1,
                 protocol=asdict(protocol),
                 data_sha256=workflow.sha(json.dumps(rows, sort_keys=True).encode()))
    model = TinySFT()
    optimizer = history.FP32MasterAdamW(model.parameters())
    workflow.save_training(model, TinyTokenizer(), optimizer, tmp_path, state)
    events = []
    monkeypatch.setattr(workflow, 'move_training', lambda *args: events.append(args[2]))

    def fail(*args):
        assert events == ['cpu']
        raise RuntimeError('pending evaluation failed')

    with pytest.raises(RuntimeError, match='pending evaluation failed'):
        workflow.train_loop(model, object(), optimizer,
                            rows, protocol, tmp_path,
                            SimpleNamespace(log_step=lambda **kw: None), device='cpu',
                            evaluator=fail, resume_state=state)
    assert events == ['cpu']
    assert workflow.read_json(tmp_path / 'resume/state.json')['pending_epoch'] == 1


def fixture_model(tmp_path):
    model = tmp_path / 'model'
    model.mkdir()
    (model / 'model.safetensors').write_bytes(b'fixture weights, never loaded')
    (model / 'tokenizer.json').write_text('{"fixture":"tokenizer"}')
    (model / 'config.json').write_text('{"rope_theta":10000}')
    return model


def test_preflight_identity_detects_config_change_without_relabeling_weight_hash(tmp_path):
    model = fixture_model(tmp_path)
    folder = workflow.stage_dir(tmp_path, 'projection_reset')
    (folder / 'data').mkdir(parents=True)
    (folder / 'data/calibration.jsonl').write_text('{}\n')
    before = workflow.identity(tmp_path, 'projection_reset', model, 'training', 128)
    (model / 'config.json').write_text('{"rope_theta":10000000}')
    after = workflow.identity(tmp_path, 'projection_reset', model, 'training', 128)
    assert before['model_sha256'] == after['model_sha256']
    assert before['model_config_sha256'] != after['model_config_sha256']
    assert before != after


def test_collection_detects_config_change_before_creating_native_engine(tmp_path, monkeypatch):
    model = fixture_model(tmp_path)
    folder = workflow.stage_dir(tmp_path, 'projection_reset')
    (folder / 'data').mkdir(parents=True)
    tier = '5x5_g11_i5'
    pool = folder / f'data/pool_{tier}.jsonl'
    pool.write_text('{}\n')
    identity = dict(model_sha256=workflow.model_digest(model),
                    model_config_sha256=workflow.model_config_digest(model),
                    tokenizer_sha256=workflow.tokenizer_digest(model),
                    pool_sha256=workflow.digest(pool),
                    protocol=asdict(history.PROTOCOLS['projection_reset']))
    workflow.write_json(folder / f'collection/{tier}/identity.json', identity)
    (model / 'config.json').write_text('{"rope_theta":10000000}')
    monkeypatch.setattr(workflow, 'validate_preflight', lambda *a: None)
    monkeypatch.setattr(workflow, 'identity', lambda *a: {})
    monkeypatch.setattr(workflow, 'NativeRuntime', lambda *a, **kw: pytest.fail('engine loaded before identity rejection'))
    with pytest.raises(ValueError, match='changed during collection resume'):
        asyncio.run(workflow.collect(tmp_path, model, tier=tier, report='unused.json'))


@pytest.mark.parametrize('changed', ['weights.pt', 'tokenizer.json', 'numerical_config'])
def test_rl_exact_resume_rejects_changed_assets_and_numerical_recipe(tmp_path, changed):
    run = make_run(tmp_path / 'run', Policy())
    checkpoint = rl.save_checkpoint(run, completed=1, next_index=4,
                                    pending=(), behavior_version=0)
    assert rl.resume_state(run, checkpoint) == (1, 4, (), 0)
    if changed == 'numerical_config':
        run.config = replace(run.config, trainer=replace(run.config.trainer,
                            policy_max_grad_norm=run.config.trainer.policy_max_grad_norm / 2))
    else:
        path = checkpoint / 'policy' / changed
        path.write_bytes(path.read_bytes() + b'changed bytes')
    with pytest.raises(ValueError, match='configuration differs|Policy or tokenizer bytes differ'):
        rl.resume_state(run, checkpoint)


@pytest.mark.parametrize('bad', [float('nan'), float('inf'), float('-inf'), 0.1])
def test_native_proposal_rejects_nonfinite_or_positive_behavior_logprob(monkeypatch, bad):
    import threading
    from climbing_gym.inference import SamplingConfig
    runtime = native.NativeRuntime.__new__(native.NativeRuntime)
    runtime.output = None
    runtime.tokenizer = SimpleNamespace(decode=lambda *a, **kw: 'fixture')

    async def generate(requests, *, on_result, **kwargs):
        result = SimpleNamespace(prompt_token_ids=(1, 2), outputs=(
            SimpleNamespace(token_ids=(3,), completion_logprobs=(bad,), finish_reason='stop'),))
        on_result(0, result)

    runtime.engine = SimpleNamespace(_seed_lock=threading.Lock(), _generation_index=0,
                                    generate_chains=generate)
    monkeypatch.setattr(torch.cuda, 'synchronize', lambda: None)
    monkeypatch.setattr(torch.cuda, 'reset_peak_memory_stats', lambda: None)
    request = ({'prompt_token_ids': (1, 2)}, SamplingConfig(max_tokens=3, logprobs=1))
    with pytest.raises(RuntimeError, match='sampled-token/logprob'):
        asyncio.run(runtime.generate([request], seed_offset=0, label='unit fixture'))


def test_projection_preflight_grades_prefix_and_suffix_but_preserves_actual_suffix(tmp_path, monkeypatch):
    from climbing_gym.futoshiki import variable_name
    model = fixture_model(tmp_path)
    task = json.loads((EXAMPLE.parent / 'data/futoshiki_smoke.jsonl').read_text().splitlines()[0])
    answer = ' '.join(f'{variable_name(i)}={v}' for i, v in enumerate(task['solution'])
                      if i not in {int(k) for k in task['givens']})
    complete = 'Checked.\n</thinking>\n<answer>' + answer + '</answer>'
    ids = [ord(c) for c in complete]
    cut = len(ids) // 2
    assert '<answer>' not in complete[cut:]  # Scoring just this suffix would fail.
    guide = dict(id=task['id'], row=task, expert_completion_token_ids=ids)
    folder = workflow.stage_dir(tmp_path, 'projection_reset')
    workflow.write_json(folder / 'data/unused.json', {})
    (folder / 'data/guides_recorded.jsonl').write_text(json.dumps(guide) + '\n' + json.dumps(dict(guide, id='second-fixture')) + '\n')

    class Runtime:
        tokenizer = SimpleNamespace(decode=lambda values, **kw: ''.join(map(chr, values)))

        def __init__(self, *args, **kwargs):
            pass

        def request(self, row, prefix):
            return ({'prompt_token_ids': [ord('Q')] + prefix}, None)

        async def generate(self, requests, **kwargs):
            return [dict(response=complete[len(request[0]['prompt_token_ids']) - 1:],
                         completion_token_ids=ids[len(request[0]['prompt_token_ids']) - 1:])
                    for request in requests], dict(
                generated_tokens=2 * len(ids) - cut, tokens_per_second=1., peak_allocated_bytes=1)

        def score(self, selected, candidates, **kwargs):
            assert candidates == [ids, ids]
            return [[-1.] * len(ids), [-1.] * len(ids)]

        def close(self):
            pass

    monkeypatch.setattr(workflow, 'NativeRuntime', Runtime)
    monkeypatch.setattr(torch.cuda, 'max_memory_allocated', lambda: 1)
    asyncio.run(workflow.preflight(tmp_path, 'projection_reset', model,
                                  purpose='projection', batch=64))
    trace = (folder / 'preflight/projection_traces.md').read_text()
    assert trace.count('Outcome: True') == 2
    assert complete[cut:] in trace
    assert 'Q' + complete[:cut] in trace


def test_cfop_documented_minimal_program_runs_against_optional_real_rust_binding():
    import re
    pytest.importorskip('cfop_solver')
    text = (EXAMPLE.parents[1] / 'docs/cfop.md').read_text()
    block = re.search(r'```python\n(.*?)\n```', text, re.S)
    namespace = {}
    exec(compile(block[1], 'docs/cfop.md example', 'exec'), namespace)
    assert namespace['result'].feedback.success


def test_interrupted_epoch_export_resumes_from_complete_rolling_snapshot(tmp_path, monkeypatch):
    model = TinySFT()
    optimizer = history.FP32MasterAdamW(model.parameters())
    state = dict(pending_epoch=1)
    workflow.save_training(model, TinyTokenizer(), optimizer, tmp_path, state)
    copy = workflow.shutil.copytree

    def partial_copy(source, target, **kwargs):
        Path(target).mkdir(parents=True)
        (Path(target) / 'weights.pt').write_bytes(b'incomplete output')
        raise RuntimeError('process failed during epoch export')

    monkeypatch.setattr(workflow.shutil, 'copytree', partial_copy)
    with pytest.raises(RuntimeError, match='epoch export'):
        workflow.publish_epoch_checkpoint(tmp_path, 1)
    assert not (tmp_path / 'checkpoints/epoch_1/policy').exists()
    assert workflow.read_json(tmp_path / 'resume/state.json')['pending_epoch'] == 1
    monkeypatch.setattr(workflow.shutil, 'copytree', copy)
    checkpoint = workflow.publish_epoch_checkpoint(tmp_path, 1, pending=True)
    assert workflow.saved_policy_manifest(checkpoint) == state['policy_sha256']
    assert not checkpoint.with_name('.policy.partial').exists()
    with pytest.raises(FileExistsError, match='immutable epoch'):
        workflow.publish_epoch_checkpoint(tmp_path, 1)
