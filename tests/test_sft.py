from __future__ import annotations

import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

import climbing_gym.sft as sft_mod
from climbing_gym.sft import (
    _prepare_sft_sequences,
    _resolve_dataset_path,
    SFTTrainerConfig,
    main,
    packed_sft_loss,
)
from climbing_gym.trainers.packing import pack_sequences


class _TinyTrainModel(nn.Module):
    def __init__(self, vocab_size: int = 256, hidden_size: int = 16) -> None:
        super().__init__()
        self.config = SimpleNamespace(pad_token_id=0, hidden_size=hidden_size)
        self.embed = nn.Embedding(vocab_size, hidden_size, padding_idx=0)
        self.proj = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, positions=None):
        del positions
        return self.proj(self.embed(input_ids))

    def save_pretrained(self, save_dir):
        save_path = Path(save_dir)
        save_path.mkdir(parents=True, exist_ok=True)
        (save_path / "config.json").write_text("{}", encoding="utf-8")
        torch.save(self.state_dict(), save_path / "pytorch_model.bin")
        return save_path / "pytorch_model.bin"


class _FakeTokenizer:
    eos_token_id = 255

    def encode(self, text, add_special_tokens=False):
        del add_special_tokens
        return [((ord(char) % 127) + 1) for char in str(text)]

    def apply_chat_template(
        self,
        messages,
        *,
        tokenize,
        add_generation_prompt,
        enable_thinking=True,
    ):
        del tokenize
        content = messages[0]["content"]
        suffix = "<think>\n" if add_generation_prompt and enable_thinking else ""
        return f"USER:{content}\nASSISTANT:{suffix}"


def test_raw_text_documents_preserve_text_and_supervise_all_targets():
    tokenizer = _FakeTokenizer()
    tokenizer.bos_token_id = 254
    text = "DOCUMENT\nPUZZLE\nSTRATEGY\nMOVE 1\nSOLVED"
    sequences, prompts = _prepare_sft_sequences(tokenizer, [{"text": text}],
        config=SFTTrainerConfig(text_key="text", loss_on_full_sequence=True))
    assert sequences == [[254] + tokenizer.encode(text) + [255]]
    batch = pack_sequences(sequences, prompt_lens=prompts, max_tokens=4096)[0]
    assert batch.targets.tolist()[0] == tokenizer.encode(text) + [255]
    assert batch.valid_token_count == len(sequences[0]) - 1


@pytest.mark.parametrize("options,match", [
    ({"loss_on_full_sequence": False}, "loss_on_full_sequence"),
    ({"apply_chat_template": True}, "transformations"),
    ({"max_seq_len": 2}, "refusing truncation"),
])
def test_raw_text_rejects_masking_wrappers_and_truncation(options, match):
    kwargs = {"text_key": "text", "loss_on_full_sequence": True, **options}
    with pytest.raises(ValueError, match=match):
        _prepare_sft_sequences(_FakeTokenizer(), [{"text": "DOCUMENT"}],
                               config=SFTTrainerConfig(**kwargs))


def _write_sft_toml_config(tmp_path: Path, *, checkpoint_root: Path) -> Path:
    dataset_path = tmp_path / "sft.jsonl"
    dataset_path.write_text(
        "".join(
            json.dumps(row) + "\n"
            for row in (
                {"question": "What is 1+1?", "solution": "<think>\n2\n</think>\n\\boxed{2}"},
                {"question": "What is 2+2?", "solution": "<think>\n4\n</think>\n\\boxed{4}"},
            )
        ),
        encoding="utf-8",
    )
    model_path = tmp_path / "stub-model"
    model_path.mkdir()
    config_path = tmp_path / "sft.toml"
    config_path.write_text(
        f"""
[model]
name = "lfm2"
path = "{model_path}"

[data]
dataset_path = "{dataset_path}"
split = "train"
shuffle = false

[sft]
prompt_key = "question"
completion_key = "solution"
batch_size = 2
num_epochs = 1
seed = 7
log_interval_steps = 1
max_seq_len = 256
max_packed_tokens = 256
shuffle = false
shuffle_packed_batches = false
train_device = "cpu"
max_grad_norm = 1.0
apply_chat_template = false
append_eos_token = true
compile_dynamic = false
compile_fullgraph = false
compile_backend = "eager"
compile_mode = "default"

[policy_optimizer]
learning_rate = 0.01

[checkpoint]
root_dir = "{checkpoint_root}"
save_interval = 0
save_final = true

[wandb]
enabled = false
""".strip()
        + "\n",
        encoding="utf-8",
    )
    return config_path


def test_packed_sft_loss_masks_prompt_tokens() -> None:
    batch = pack_sequences(
        [(11, 12, 13, 14)],
        prompt_lens=[2],
        max_tokens=16,
    )[0]
    valid_mask = batch.valid_completion_mask()
    assert valid_mask.tolist() == [False, True, True]

    loss, valid_count = packed_sft_loss(
        target_log_probs=torch.tensor(
            [math.log(0.9), math.log(0.25), math.log(0.5)],
            dtype=torch.float32,
        ),
        valid_mask=valid_mask,
    )
    expected = -((math.log(0.25) + math.log(0.5)) / 2.0)
    assert float(loss) == pytest.approx(expected)
    assert float(valid_count) == pytest.approx(2.0)


def test_prepare_sft_sequences_uses_optional_chat_template() -> None:
    tokenizer = _FakeTokenizer()
    config = SFTTrainerConfig(
        prompt_key="question",
        completion_key="solution",
        max_seq_len=256,
        apply_chat_template=True,
        chat_template_enable_thinking=True,
        append_eos_token=True,
    )
    sequences, prompt_lens = _prepare_sft_sequences(
        tokenizer,
        [{"question": "Solve x", "solution": "<think>\nwork\n</think>\n\\boxed{1}"}],
        config=config,
    )
    assert len(sequences) == 1
    assert prompt_lens == [len(tokenizer.encode("USER:Solve x\nASSISTANT:<think>\n"))]
    assert sequences[0][-1] == tokenizer.eos_token_id


def test_resolve_dataset_path_prefers_hf_dataset_child(tmp_path: Path) -> None:
    wrapper = tmp_path / "openmath_wrapper"
    child = wrapper / "hf_dataset_train_test"
    child.mkdir(parents=True)
    assert _resolve_dataset_path(str(wrapper)) == str(child.resolve())


def test_sft_main_runs_end_to_end_with_stub_model_and_tokenizer(
    tmp_path: Path,
    monkeypatch,
    capsys,
) -> None:
    checkpoint_root = tmp_path / "checkpoints"
    config_path = _write_sft_toml_config(tmp_path, checkpoint_root=checkpoint_root)

    monkeypatch.setattr(sft_mod, "build_model", lambda *args, **kwargs: _TinyTrainModel())
    monkeypatch.setattr(
        sft_mod,
        "AutoTokenizer",
        type("FakeAutoTokenizer", (), {"from_pretrained": staticmethod(lambda *args, **kwargs: _FakeTokenizer())}),
    )
    monkeypatch.setattr(sft_mod, "maybe_compile", lambda target, **kwargs: target)

    assert main([str(config_path)]) == 0
    captured = capsys.readouterr()
    assert "[sft] starting" in captured.out
    assert "[sft] step=0" in captured.out
    assert "[sft] completed last_step=0" in captured.out
    assert (checkpoint_root / "step_0" / "policy" / "config.json").is_file()


def test_exact_ids_preserve_prefix_mask_and_only_one_eos():
    tokenizer = _FakeTokenizer()
    seq, prompts = _prepare_sft_sequences(tokenizer, [dict(prompt_token_ids=[13, 14], completion_token_ids=[15, 255])],
        config=SFTTrainerConfig())
    assert seq == [[13, 14, 15, 255]]
    batch = pack_sequences(seq, prompt_lens=prompts, max_tokens=32)[0]
    assert batch.valid_completion_mask().tolist() == [False, True, True]
    with pytest.raises(ValueError, match='refusing truncation'):
        _prepare_sft_sequences(tokenizer, [dict(prompt_token_ids=[13, 14], completion_token_ids=[15, 255])],
            config=SFTTrainerConfig(max_seq_len=3))
    with pytest.raises(ValueError, match='transformations'):
        _prepare_sft_sequences(tokenizer, [dict(prompt_token_ids=[13], completion_token_ids=[15])],
            config=SFTTrainerConfig(apply_chat_template=True))


def test_lfm_thinking_tags_are_preserved_without_duplicate_open():
    tokenizer = _FakeTokenizer()
    prompt = 'USER: Solve\nASSISTANT:<thinking>\n'
    completion = '<thinking>\nwork\n</thinking>\n<answer>2</answer>'
    seq, prompts = _prepare_sft_sequences(tokenizer, [dict(prompt=prompt, completion=completion)],
        config=SFTTrainerConfig(thinking_tag='thinking', require_think_close=True))
    assert prompts == [len(tokenizer.encode(prompt))]
    assert seq == [tokenizer.encode(prompt)+tokenizer.encode('work\n</thinking>\n<answer>2</answer>')+[255]]


def test_checkpoint_resume_restores_optimizer_and_data_cursor(tmp_path, monkeypatch):
    def model_factory(*args, **kwargs):
        torch.manual_seed(123)
        model = _TinyTrainModel()
        checkpoint = Path(kwargs.get('model_dir', ''))/'pytorch_model.bin'
        if checkpoint.is_file():
            model.load_state_dict(torch.load(checkpoint, weights_only=True))
        return model
    monkeypatch.setattr(sft_mod, 'build_model', model_factory)
    monkeypatch.setattr(sft_mod, 'AutoTokenizer', type('FakeAutoTokenizer', (), {
        'from_pretrained': staticmethod(lambda *a, **k: _FakeTokenizer())}))
    monkeypatch.setattr(sft_mod, 'maybe_compile', lambda target, **kwargs: target)
    full_dir, resumed_dir = tmp_path/'full', tmp_path/'resumed'
    config = _write_sft_toml_config(tmp_path, checkpoint_root=full_dir)
    text = config.read_text().replace('batch_size = 2', 'batch_size = 1')
    config.write_text(text.replace('num_epochs = 1', 'num_epochs = 2'))
    assert main([str(config)]) == 0
    config.write_text(text.replace(str(full_dir), str(resumed_dir)))
    assert main([str(config)]) == 0
    first_state = torch.load(resumed_dir/'step_1/sft_state.pt', weights_only=False)
    assert (first_state['next_epoch'], first_state['next_batch_offset']) == (1, 0)
    config.write_text(text.replace(str(full_dir), str(resumed_dir)).replace('num_epochs = 1', 'num_epochs = 2')+
        f'\n[resume]\ncheckpoint_dir = "{resumed_dir / "step_1"}"\n')
    assert main([str(config)]) == 0
    expected = torch.load(full_dir/'step_3/policy/pytorch_model.bin', weights_only=True)
    actual = torch.load(resumed_dir/'step_3/policy/pytorch_model.bin', weights_only=True)
    for name in expected:
        torch.testing.assert_close(actual[name], expected[name], atol=0, rtol=0)
    assert torch.load(resumed_dir/'step_3/sft_state.pt', weights_only=False)['optimizer']['state']


def test_head_chunk_checkpointed_loss_matches_response_only_gradients():
    from climbing_gym.projection_sampling import backward_completion_loss
    class HeadModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.embed = nn.Embedding(32, 5)
            self.lm_head = nn.Linear(5, 32)
        def get_hidden_states(self, inputs, positions):
            return self.embed(inputs)
    torch.manual_seed(22)
    chunked = HeadModel()
    reference = HeadModel(); reference.load_state_dict(chunked.state_dict())
    batch = pack_sequences([[1, 2, 3, 4, 5]], prompt_lens=[2], max_tokens=16)[0]
    actual_loss = backward_completion_loss(chunked, batch, denominator=3, head_chunk=2)
    mask = batch.valid_completion_mask().flatten()
    hidden = reference.get_hidden_states(batch.inputs, batch.position_ids).reshape(-1, 5)[mask]
    expected_loss = torch.nn.functional.cross_entropy(reference.lm_head(hidden), batch.targets.flatten()[mask])
    expected_loss.backward()
    assert actual_loss == pytest.approx(float(expected_loss.detach()))
    for actual, expected in zip(chunked.parameters(), reference.parameters()):
        torch.testing.assert_close(actual.grad, expected.grad)


@pytest.mark.parametrize('stage', ['model_move', 'compile', 'dataset', 'resume', 'training', 'evaluator_setup'])
def test_main_cleans_resources_for_setup_and_training_failures(tmp_path, monkeypatch, stage):
    from dataclasses import replace
    from climbing_gym.sft_eval import SFTEvaluationConfig
    failure = RuntimeError('injected '+stage+' failure')
    calls = []
    config = _write_sft_toml_config(tmp_path, checkpoint_root=tmp_path/'checkpoints')
    config.write_text(config.read_text()+'\n[eval]\nconfig_file = "eval.toml"\ninterval_steps = 1\n')
    class Model(_TinyTrainModel):
        moves = 0
        def to(self, *args, **kwargs):
            self.moves += 1
            if stage == 'model_move' and self.moves == 2:
                raise failure
            return super().to(*args, **kwargs)
    class Logger:
        log_dir = tmp_path
        def __init__(self, *args, **kwargs): pass
        def log_step(self, **kwargs): pass
        def close(self):
            calls.append('logger.close')
            raise OSError('injected logger cleanup failure')
    class Evaluator:
        def __init__(self, *args, **kwargs):
            if stage == 'evaluator_setup':
                raise failure
        def evaluate(self, step): return {}
        def close(self):
            calls.append('evaluator.close')
            raise OSError('injected evaluator cleanup failure')
    monkeypatch.setattr(sft_mod, 'build_model', lambda *a, **k: Model())
    monkeypatch.setattr(sft_mod, 'AutoTokenizer', type('FakeAutoTokenizer', (), {
        'from_pretrained': staticmethod(lambda *a, **k: _FakeTokenizer())}))
    monkeypatch.setattr(sft_mod, 'WandbLogger', Logger)
    monkeypatch.setattr(sft_mod, 'PeriodicSFTEvaluator', Evaluator)
    def fail(*args, **kwargs): raise failure
    monkeypatch.setattr(sft_mod, 'maybe_compile', fail if stage == 'compile' else lambda target, **kwargs: target)
    if stage == 'dataset':
        monkeypatch.setattr(sft_mod, '_load_rows', fail)
    if stage == 'training':
        monkeypatch.setattr(sft_mod, 'packed_sft_loss_from_model', fail)
    if stage == 'resume':
        original_load = sft_mod.load_run_from_toml
        def load(*args):
            run = original_load(*args)
            return replace(run, resume_state=dict(dataset_sha256='changed'))
        monkeypatch.setattr(sft_mod, 'load_run_from_toml', load)
    with pytest.raises((RuntimeError, ValueError)) as caught:
        main([str(config)])
    if stage == 'resume':
        assert 'resume dataset' in str(caught.value)
    else:
        assert caught.value is failure
    expected = ['logger.close'] if stage == 'evaluator_setup' else ['evaluator.close', 'logger.close']
    assert calls == expected
    assert any('logger cleanup failure' in note for note in caught.value.__notes__)
    if stage != 'evaluator_setup':
        assert any('evaluator cleanup failure' in note for note in caught.value.__notes__)


def test_main_attempts_logger_close_when_evaluator_close_fails_after_success(tmp_path, monkeypatch):
    calls = []
    failure = OSError('evaluator close failed')
    run = SimpleNamespace(wandb_logger=SimpleNamespace(close=lambda: calls.append('logger.close')))
    monkeypatch.setattr(sft_mod, 'load_run_from_toml', lambda config: run)
    def train(run, cleanups):
        def close():
            calls.append('evaluator.close')
            raise failure
        cleanups.append(('native evaluator', close))
        return 0
    monkeypatch.setattr(sft_mod, '_train_sft', train)
    with pytest.raises(OSError) as caught:
        main(['unused.toml'])
    assert caught.value is failure
    assert calls == ['evaluator.close', 'logger.close']
