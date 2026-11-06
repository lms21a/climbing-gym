import pytest
import torch

from climbing_gym.inference.text_stops import build_stop_table, advance_text_stops, trim_after_stop


class PaddedTokenizer:
    pieces = ['<', '/', 'summary', '>', 'x']

    def __len__(self):
        return len(self.pieces)

    def decode(self, ids, **kwargs):
        # Padded model IDs have no tokenizer text.
        return ''.join(self.pieces[i] if i < len(self) else '' for i in ids)

    def encode(self, text, **kwargs):
        return [0, 1, 2, 3] if text.endswith('</summary>') else [4]


@pytest.mark.parametrize(('text', 'stops', 'expected'), [
    ('<answer>A1+B1</answer></', ('</answer>',), '<answer>A1+B1</answer>'),
    ('<summary>3+4=7</summary>"', ('</summary>',), '<summary>3+4=7</summary>'),
    ('first STOP later END', ('END', 'STOP'), 'first STOP'),
    ('abc trailing', ('abc', 'b'), 'ab'),
    ('unfinished </answer', ('</answer>',), 'unfinished </answer'),
    ('no requested stop', (), 'no requested stop'),
])
def test_response_text_keeps_the_first_completed_stop(text, stops, expected):
    assert trim_after_stop(text, stops) == expected


@pytest.mark.parametrize('tag', ['answer', 'summary'])
@pytest.mark.parametrize('finish_code', [1, 2, 3])
def test_native_adapter_trims_inside_final_token_only_for_string_stop(tag, finish_code):
    import asyncio
    from types import SimpleNamespace
    from climbing_gym.inference.config import SamplingConfig
    from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
    from climbing_gym.inference.lfm25_engine import LFM25GenerationOutput
    from test_lfm25_async_engine import _FakeTokenEngine

    class Tokenizer(PaddedTokenizer):
        pieces = [f'<{tag}>3+4=7', '</', tag, '></', 'prompt']

    class Engine(_FakeTokenEngine):
        config = SimpleNamespace(vocab_size=5)

        def generate(self, *args, **kwargs):
            return LFM25GenerationOutput(token_ids=torch.tensor([[[0, 1, 2, 3]]]),
                lengths=torch.tensor([[4]]), finish_reasons=torch.tensor([[finish_code]]),
                logprobs=torch.tensor([[[-.5, -.5, -.5, -.5]]]))

    tokenizer = Tokenizer()
    adapter = LFM25AsyncInferenceEngine.from_components(engine=Engine(), tokenizer=tokenizer)
    try:
        result = asyncio.run(adapter.generate({'prompt': 'prompt', 'prompt_token_ids': [4]},
            sampling=SamplingConfig(max_tokens=4, logprobs=1, stop=(f'</{tag}>',))))
        raw = tokenizer.decode([0, 1, 2, 3])
        assert result.text == (raw[:-2] if finish_code == 2 else raw)
        assert result.token_ids == (0, 1, 2, 3)
        assert result.completion_logprobs == (-.5,) * 4
        assert result.raw_output.lengths.item() == 4
    finally:
        adapter.close()


@pytest.mark.parametrize('device', ['cpu', pytest.param('cuda', marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason='CUDA required'))])
def test_padded_model_ids_preserve_partial_stop_without_out_of_bounds(device):
    tokenizer = PaddedTokenizer()
    table = build_stop_table(tokenizer, ('</summary>',), vocab_size=65536).to(device)
    assert table.shape[1] == 65536
    state = torch.zeros(1, dtype=torch.long, device=device)
    reasons = torch.zeros(1, dtype=torch.int8, device=device)
    for index, token in enumerate([65535, 0, 1, 64400, 2, 65535, 3]):
        state, reason = advance_text_stops(table, state, torch.tensor([token], device=device), reasons)
        assert reason.item() == (2 if index == 6 else 0)


def test_old_tokenizer_sized_table_reproduces_failure_on_padded_id():
    table = build_stop_table(PaddedTokenizer(), ('</summary>',))
    with pytest.raises(IndexError):
        advance_text_stops(table, torch.tensor([0]), torch.tensor([65535]), torch.tensor([0]))


def test_model_vocabulary_cannot_be_smaller_than_tokenizer():
    with pytest.raises(ValueError, match='Model vocabulary'):
        build_stop_table(PaddedTokenizer(), ('</summary>',), vocab_size=4)


def test_async_adapter_uses_model_vocab_for_stop_table():
    import asyncio
    from types import SimpleNamespace
    from climbing_gym.inference.config import SamplingConfig
    from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
    from test_lfm25_async_engine import _FakeTokenEngine

    engine = _FakeTokenEngine()
    engine.config = SimpleNamespace(vocab_size=65536)
    adapter = LFM25AsyncInferenceEngine.from_components(engine=engine, tokenizer=PaddedTokenizer())
    asyncio.run(adapter.generate({'prompt': 'x', 'prompt_token_ids': [4]},
        sampling=SamplingConfig(max_tokens=3, stop=('</summary>',))))
    assert engine.calls[0]['sampling'].stop_table.shape[1] == 65536


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@torch.inference_mode()
def test_native_cuda_decode_accepts_padded_id_inside_a_stop_sequence(monkeypatch):
    from climbing_gym.inference import lfm25_sampling
    from climbing_gym.inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig, flatten_token_batches
    from climbing_gym.models.lfm2 import LFM2Config, LFM2Model

    config = LFM2Config(vocab_size=65536, hidden_size=128, mlp_hidden_size=192,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=64, layer_types=('conv', 'full_attention'), attn_backend='eager')
    model = LFM2Model(config).eval()
    engine = LFM25Engine.from_model(model, device='cuda', dtype=torch.bfloat16, max_model_len=64)
    engine.enable_cuda_graphs()
    sequence = [0, 1, 65535, 2, 3]

    def forced_sample(logits, *, step, **kwargs):
        return (torch.full((logits.shape[0],), sequence[step], device=logits.device, dtype=torch.long),
                torch.zeros(logits.shape[0], device=logits.device))

    monkeypatch.setattr(lfm25_sampling, 'sample_logits', forced_sample)
    table = build_stop_table(PaddedTokenizer(), ('</summary>',), vocab_size=config.vocab_size).cuda()
    flat, cu, max_len = flatten_token_batches([[4, 4], [4, 4]], device='cuda')
    result = engine.generate(flat, cu, max_len, repeats=1, sampling=LFM25SamplingConfig(
        max_new_tokens=8, temperature=1, ignore_eos=True, return_logprobs=True, stop_table=table))
    assert result.lengths.tolist() == [[5], [5]]
    assert result.finish_reasons.tolist() == [[2], [2]]
    assert result.token_ids[0, 0, :5].tolist() == sequence
    assert engine._cuda_decoder.graphs
    engine._cuda_decoder.clear()
