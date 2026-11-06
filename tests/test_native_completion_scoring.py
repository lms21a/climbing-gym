import pytest
import torch

from climbing_gym.inference.lfm25_engine import LFM25Engine
from climbing_gym.models import LFM2Config, LFM2Model


def model_and_engine():
    torch.manual_seed(42)
    cfg=LFM2Config(vocab_size=257,hidden_size=64,mlp_hidden_size=128,num_hidden_layers=4,
        num_attention_heads=8,num_key_value_heads=4,max_position_embeddings=128,
        layer_types=('conv','full_attention','conv','full_attention'),attn_backend='eager',shortconv_backend='torch')
    model=LFM2Model(cfg).eval()
    return model,LFM25Engine.from_model(model,device='cpu',dtype=torch.float32,max_model_len=128)


def test_native_teacher_forcing_matches_full_model_with_padding_chunks_and_eos():
    model,engine=model_and_engine()
    prompts=[[1,11,12],[1,21]]
    completions=[[13,14,7],[22,23,24,25,7]]
    rng=torch.get_rng_state().clone()
    scores=engine.score_completions(prompts,completions,max_prefill_tokens=64,head_chunk_size=2)
    assert torch.equal(rng,torch.get_rng_state())
    for p,c,actual in zip(prompts,completions,scores,strict=True):
        with torch.no_grad():
            ids=torch.tensor([p+c])
            logits=model(ids,positions=torch.arange(ids.shape[1])[None])[0].float()
            expected=logits[len(p)-1:-1].log_softmax(-1).gather(1,torch.tensor(c)[:,None]).squeeze(1)
        assert len(actual)==len(c)
        assert torch.allclose(torch.tensor(actual),expected,atol=2e-5,rtol=2e-5)
    separate=[engine.score_completions([p],[c],max_prefill_tokens=1,head_chunk_size=1)[0]
              for p,c in zip(prompts,completions)]
    for a,b in zip(scores,separate):
        assert torch.allclose(torch.tensor(a),torch.tensor(b),atol=2e-5,rtol=2e-5)


def test_native_scoring_rejects_invalid_lengths_and_context():
    _,engine=model_and_engine()
    for p,c in (([[1]],[]),([[]],[[2]]),([[1]],[[]]),([[1]*128],[[2]])):
        with pytest.raises(ValueError):
            engine.score_completions(p,c)
    assert engine.score_completions([],[])==[]
