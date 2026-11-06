import json
import sys

from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast

from climbing_gym.inference.benchmark import main
from climbing_gym.models import LFM2Config, LFM2Model


def test_native_speed_check_writes_timings_and_complete_traces(tmp_path, monkeypatch):
    config = LFM2Config(vocab_size=257, hidden_size=64, mlp_hidden_size=128,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, layer_types=('conv', 'full_attention'),
        attn_backend='eager', shortconv_backend='torch')
    LFM2Model(config).save_pretrained(tmp_path)
    backend = Tokenizer(WordLevel({'[UNK]': 0, 'Check': 1, 'each': 2}, unk_token='[UNK]'))
    backend.pre_tokenizer = Whitespace()
    PreTrainedTokenizerFast(tokenizer_object=backend, unk_token='[UNK]').save_pretrained(tmp_path)
    output = tmp_path / 'speed.json'
    monkeypatch.setattr(sys, 'argv', ['benchmark', '--model', str(tmp_path), '--device', 'cpu',
        '--batches', '1', '2', '--prompt-tokens', '4', '--output-tokens', '3',
        '--warmups', '1', '--repeats', '1', '--output', str(output)])
    main()
    report = json.loads(output.read_text())
    assert [b['batch_size'] for b in report['batches']] == [1, 2]
    assert [b['runs'][0]['output_tokens'] for b in report['batches']] == [3, 6]
    assert all(b['median_tokens_per_second'] > 0 for b in report['batches'])
    traces = output.with_suffix('.md').read_text()
    assert traces.count('Complete model response:') == 3
    assert 'sampled log probabilities finite' in traces
