from __future__ import annotations

from copy import deepcopy
import json

import pytest
import torch

from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer, normalize_lfm2_config
from climbing_gym.inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig, flatten_token_batches
from climbing_gym.models.lfm2 import LFM2Config


def _modern_config():
    return {
        "model_type": "lfm2", "architectures": ["Lfm2ForCausalLM"],
        "vocab_size": 128000, "hidden_size": 2048, "intermediate_size": 10752,
        "num_hidden_layers": 30, "num_attention_heads": 32, "num_key_value_heads": 8,
        "layer_types": ["full_attention" if i in (2, 5, 9, 13, 17, 21, 24, 27) else "conv" for i in range(30)],
        "conv_L_cache": 3, "conv_bias": False, "block_auto_adjust_ff_dim": False,
        "tie_word_embeddings": True, "max_position_embeddings": 128000,
        "rope_parameters": {"rope_theta": 10000000.0, "rope_type": "default"},
        "bos_token_id": 124894, "eos_token_id": 124900, "pad_token_id": 124893,
    }


def test_26b_config_preserves_architecture():
    payload = _modern_config()
    original = deepcopy(payload)
    native = LFM2Config.from_dict(payload)
    assert payload == original
    for config in (native,):
        assert config.rope_theta == 10000000.0
        assert config.hidden_size == 2048
        assert config.vocab_size == 128000
        assert config.num_hidden_layers == 30
        assert config.layer_types.count("full_attention") == 8
        assert config.layer_types.count("conv") == 22
        assert config.bos_token_id == 124894
        assert config.eos_token_id == 124900
        assert config.pad_token_id == 124893
        assert config.tie_embedding
    assert native.mlp_hidden_size == 10752


@pytest.mark.parametrize("config_cls", (LFM2Config,))
def test_modern_rope_takes_priority_over_transformers4_default(config_cls):
    payload = _modern_config()
    payload.update(rope_theta=1000000.0, tie_word_embeddings=False)
    config = config_cls.from_dict(payload)
    assert config.rope_theta == 10000000.0
    assert not config.tie_embedding


@pytest.mark.parametrize("rope", ({"rope_type": "linear", "factor": 2}, {"rope_type": "yarn"}, {"full_attention": {}}))
def test_scaled_or_layer_specific_rope_is_not_silently_ignored(rope):
    with pytest.raises(ValueError, match="Unsupported LFM2"):
        normalize_lfm2_config({"rope_parameters": rope})


def test_tokenizers_backend_checkpoint_loads_special_tokens_and_template(tmp_path):
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace

    backend = Tokenizer(WordLevel({"[UNK]": 0, "[BOS]": 1, "[EOS]": 2, "[PAD]": 3, "hello": 4}, unk_token="[UNK]"))
    backend.pre_tokenizer = Whitespace()
    backend.save(str(tmp_path / "tokenizer.json"))
    (tmp_path / "tokenizer_config.json").write_text(json.dumps({
        "tokenizer_class": "TokenizersBackend", "bos_token": "[BOS]", "eos_token": "[EOS]", "pad_token": "[PAD]",
    }))
    (tmp_path / "chat_template.jinja").write_text("{{ messages[0]['content'] }}")
    tokenizer = load_lfm2_tokenizer(str(tmp_path))
    assert tokenizer.encode("hello", add_special_tokens=False) == [4]
    assert (tokenizer.bos_token_id, tokenizer.eos_token_id, tokenizer.pad_token_id) == (1, 2, 3)
    assert tokenizer.apply_chat_template([{"role": "user", "content": "hello"}], tokenize=False) == "hello"


@torch.inference_mode()
def test_modern_30_layer_checkpoint_matches_transformers_prefill_and_decode():
    from transformers import Lfm2Config, Lfm2ForCausalLM

    payload = _modern_config()
    payload.update(vocab_size=257, hidden_size=64, intermediate_size=96, num_attention_heads=4,
                   num_key_value_heads=2, max_position_embeddings=128, pad_token_id=0, bos_token_id=1, eos_token_id=2)
    reference_config = Lfm2Config(**payload)
    reference_config.rope_theta = payload["rope_parameters"]["rope_theta"]
    reference_config._attn_implementation = "eager"
    torch.manual_seed(17)
    reference = Lfm2ForCausalLM(reference_config).eval()
    native = LFM25Engine.from_model(reference, device="cpu", dtype=torch.float32, max_model_len=128)
    prompts = [[1, 11, 12, 13, 14], [1, 20, 21]]
    prefix = native.prefill(*flatten_token_batches(prompts), max_cache_len=16)
    state = native.fork(prefix, repeats=1)
    for row, prompt in enumerate(prompts):
        expected = reference(torch.tensor([prompt]), use_cache=False).logits[0, -1]
        torch.testing.assert_close(prefix.next_token_logits[row], expected, atol=1e-5, rtol=1e-5)
    for step in range(3):
        next_tokens = torch.tensor([31 + step, 41 + step])
        output = native.decode_step(state, next_tokens, LFM25SamplingConfig(temperature=0))
        for row, token in enumerate(next_tokens.tolist()):
            prompts[row].append(token)
            expected = reference(torch.tensor([prompts[row]]), use_cache=False).logits[0, -1]
            torch.testing.assert_close(output.logits[row], expected, atol=1e-5, rtol=1e-5)


def test_greedy_generation_applies_repetition_penalty():
    from climbing_gym.models.lfm2 import LFM2Model

    config = LFM2Config(vocab_size=3, hidden_size=64, mlp_hidden_size=64, num_hidden_layers=1,
                       num_attention_heads=4, num_key_value_heads=2, layer_types=("conv",), attn_backend="eager")
    engine = LFM25Engine.from_model(LFM2Model(config))
    tokens, _, _ = engine._sample(
        torch.tensor([[3.0, 2.0, 1.0]]), LFM25SamplingConfig(temperature=0, repetition_penalty=2),
        generator=torch.Generator(), seen_token_mask=torch.tensor([[True, False, False]]),
    )
    assert tokens.tolist() == [1]
