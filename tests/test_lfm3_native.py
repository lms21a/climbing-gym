from __future__ import annotations

import pytest
import torch
import torch.nn.functional as F

from climbing_gym.inference.lfm3_engine import LFM3Config, LFM3Engine, LFM3PackedWeights
from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig, flatten_token_batches, _rms_norm

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')


def tiny_engine():
    torch.manual_seed(173)
    cfg = LFM3Config(vocab_size=257, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
                     num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                     layer_types=('conv', 'full_attention', 'sliding_attention', 'conv'),
                     max_position_embeddings=512, eos_token_id=256, pad_token_id=0)
    state = {}
    def matrix(key, shape):
        state[key] = torch.randn(shape, device='cuda', dtype=torch.bfloat16) * 0.02
    def norm(key, width):
        state[key] = (1 + torch.randn(width, device='cuda') * 0.1).bfloat16()
    matrix('model.embed_tokens.weight', (257, 128))
    norm('model.embedding_norm.weight', 128)
    for i, kind in enumerate(cfg.layer_types):
        p = f'model.layers.{i}'
        for name in ('operator_norm', 'ffn_norm', 'post_operator_norm', 'post_ffn_norm'):
            norm(p+f'.{name}.weight', 128)
        for name, shape in [('w1', (192,128)), ('w3', (192,128)), ('w2', (128,192))]:
            matrix(p+f'.feed_forward.{name}.weight', shape)
        if kind == 'conv':
            for name, shape in [('in_proj', (384,128)), ('out_proj', (128,128)), ('conv', (128,1,7))]:
                matrix(p+f'.conv.{name}.weight', shape)
        else:
            for name, size in [('q',128), ('k',64), ('v',64), ('o',128), ('gate',4)]:
                matrix(p+f'.self_attn.{name}_proj.weight', (size,128))
            for name in ('q_norm', 'k_norm'):
                norm(p+f'.self_attn.{name}.weight', 32)
    return LFM3Engine(LFM3PackedWeights.from_state_dict(cfg, state, device='cuda', dtype=torch.bfloat16), max_model_len=512)


def test_native_factory_selects_lfm3_without_loading_transformers_model(tmp_path, monkeypatch):
    import json
    from climbing_gym.inference.lfm25_engine import LFM25Engine
    (tmp_path/'config.json').write_text(json.dumps({'model_type':'lfm3'}))
    sentinel = object()
    calls = []
    def load(path, **kwargs):
        calls.append((path,kwargs))
        return sentinel
    monkeypatch.setattr(LFM3Engine, 'from_pretrained', load)
    assert LFM25Engine.from_pretrained(tmp_path, device='cuda', dtype='bfloat16', max_model_len=4096) is sentinel
    assert calls == [(tmp_path,dict(device='cuda',dtype='bfloat16',max_model_len=4096))]


@torch.inference_mode()
def test_seven_tap_convolution_masks_and_indexed_slots():
    from climbing_gym.inference.lfm3_kernels import conv_update
    torch.manual_seed(7)
    b, h = 3, 137
    bcx = torch.randn(b, 3*h, device='cuda', dtype=torch.bfloat16)
    w = torch.randn(h, 7, device='cuda', dtype=torch.bfloat16)
    state = torch.randn(5, h, 7, device='cuda', dtype=torch.bfloat16)
    before = state.clone()
    slots = torch.tensor([3, 0, 4], device='cuda', dtype=torch.int32)
    active = torch.tensor([True, False, True], device='cuda')
    bp, cp, xp = bcx.chunk(3, dim=-1)
    window = torch.cat((before[slots.long(), :, 1:], (bp*xp)[..., None]), dim=-1)
    expected = (window.float()*w.float()).sum(-1).bfloat16() * cp
    out = conv_update(bcx, w, state, active, slots)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)
    torch.testing.assert_close(state[3], window[0], rtol=0, atol=0)
    torch.testing.assert_close(state[4], window[2], rtol=0, atol=0)
    torch.testing.assert_close(state[[0,1,2]], before[[0,1,2]], rtol=0, atol=0)


@torch.inference_mode()
def test_sandwich_norm_preserves_bf16_rounding():
    from climbing_gym.inference.lfm3_kernels import sandwich_norm
    torch.manual_seed(2)
    x, r = (torch.randn(3, 128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    post, pre = (torch.randn(128, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    expected_sum = _rms_norm(x, post, 1e-5) + r
    expected = _rms_norm(expected_sum, pre, 1e-5)
    out, summed = sandwich_norm(x, post, r, pre, 1e-5)
    torch.testing.assert_close(summed, expected_sum, rtol=0, atol=0)
    torch.testing.assert_close(out, expected, rtol=0, atol=0)


@pytest.mark.parametrize('kind', ['full_attention', 'sliding_attention'])
@torch.inference_mode()
def test_qkv_partial_rope_and_head_gate(kind):
    from climbing_gym.inference.lfm3_kernels import prepare_qkv, head_gate
    engine = tiny_engine()
    cfg = engine.config
    x = torch.randn(3, 260, device='cuda', dtype=torch.bfloat16)
    qw, kw = (torch.randn(32, device='cuda', dtype=torch.bfloat16) for _ in range(2))
    pos = torch.tensor([0, 17, 271], device='cuda', dtype=torch.int32)
    cos, sin = engine._rope_by_type[kind]
    q, k, v = prepare_qkv(x, qw, kw, cos, sin, pos, 4, 2, 32, cfg.norm_eps)
    co, si = (t[pos.long()].bfloat16().float().repeat(1, 2)[:,None] for t in (cos,sin))
    for actual, raw, weight in [(q[:,0], x[:,:128].reshape(3,4,32), qw),
                                (k[:,0], x[:,128:192].reshape(3,2,32), kw)]:
        norm = _rms_norm(raw, weight, cfg.norm_eps).float()
        expected = (norm*co + torch.cat((-norm[...,16:], norm[...,:16]),dim=-1)*si).bfloat16()
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    torch.testing.assert_close(v[:,0], x[:,192:256].reshape(3,2,32), rtol=0, atol=0)
    expected = (q[:,0] * x[:,-4:].sigmoid()[...,None]).reshape(3,128)
    torch.testing.assert_close(head_gate(q, x, 4, 32), expected, rtol=0, atol=0)
    if kind == 'full_attention':
        assert torch.equal(cos[:,8:], torch.ones_like(cos[:,8:]))
        assert torch.count_nonzero(sin[:,8:]) == 0


@pytest.mark.parametrize('window', [1, 31, 256])
@torch.inference_mode()
def test_fp8_sliding_attention_matches_dequantized_reference(window):
    from climbing_gym.inference.lfm25_fp8_kv import store_kv, attend
    torch.manual_seed(9)
    b,s,h,d,g=3,320,2,32,2
    k,v = (torch.randn(b,s,h,d,device='cuda',dtype=torch.bfloat16) for _ in range(2))
    kc,vc = (torch.empty(b,h,s,d,device='cuda',dtype=torch.float8_e4m3fn).transpose(1,2) for _ in range(2))
    ks,vs = (torch.empty(b,h,s,device='cuda',dtype=torch.float32).transpose(1,2) for _ in range(2))
    store_kv(k,v,kc,vc,ks,vs)
    q = torch.randn(b,1,h*g,d,device='cuda',dtype=torch.bfloat16)
    pos = torch.tensor([4,270,319],device='cuda',dtype=torch.int32)
    for splits in (1,4):
        out = attend(q,kc,vc,ks,vs,pos,window=window,config=(64,4,splits))
        kd,vd = kc.float()*ks[...,None], vc.float()*vs[...,None]
        allowed = ((torch.arange(s,device='cuda')[None,:] <= pos[:,None]) &
                   (torch.arange(s,device='cuda')[None,:] > pos[:,None]-window))
        expected = F.scaled_dot_product_attention(q.transpose(1,2).float(),kd.transpose(1,2),vd.transpose(1,2),
                                                  attn_mask=allowed[:,None,None,:],enable_gqa=True).transpose(1,2)
        torch.testing.assert_close(out.float(),expected,atol=.012,rtol=.02)


@pytest.mark.parametrize('lengths', [(3,5), (257,271)])
@torch.inference_mode()
def test_cached_decode_matches_full_prefix(lengths):
    engine = tiny_engine()
    prompts = [list(range(1, n+1)) for n in lengths]
    prompts = [[v % 250 for v in row] for row in prompts]
    prefix = engine.prefill(*flatten_token_batches(prompts,device='cuda'),max_cache_len=512,max_prefill_tokens=1024)
    state = engine.fork(prefix,1,max_cache_len=512)
    for step in range(3):
        tokens = torch.tensor([17+step,41+step],device='cuda')
        out = engine.decode_step(state,tokens,LFM25SamplingConfig(temperature=0))
        prompts = [p+[int(t)] for p,t in zip(prompts,tokens)]
        fresh = engine.prefill(*flatten_token_batches(prompts,device='cuda'),max_prefill_tokens=1024)
        torch.testing.assert_close(out.logits.float(), fresh.next_token_logits.float(), atol=.07, rtol=.04)
        assert torch.equal(out.logits.argmax(-1),fresh.next_token_logits.argmax(-1))


@pytest.mark.parametrize('fp8', [False, True])
@torch.inference_mode()
def test_graph_logprobs_and_session_refill(fp8):
    engine = tiny_engine()
    engine.enable_cuda_graphs(fp8_mlp=fp8,fp8_kv=fp8,fp8_lm_head=fp8,fp8_operators=fp8)
    prompts = [[1,2,3],[4,5]]
    sampling = LFM25SamplingConfig(temperature=1, max_new_tokens=4, ignore_eos=True, seed=17)
    result = engine.generate(*flatten_token_batches(prompts,device='cuda'),repeats=1,sampling=sampling)
    entry = next(iter(engine._cuda_decoder.graphs.values()))
    torch.testing.assert_close(entry.cache.seq_lens, torch.tensor([6,5],device='cuda',dtype=torch.int32))
    engine.prefill(*flatten_token_batches(prompts,device='cuda'),cache=entry.cache)
    entry.input_tokens.copy_(result.token_ids[:,0,0])
    entry.graph.replay()
    expected = entry.logits.float().log_softmax(-1).gather(1,result.token_ids[:,0,1:2]).squeeze(1)
    torch.testing.assert_close(result.logprobs[:,0,1],expected,rtol=0,atol=2e-6)
    with engine.generation_session(max_batch_size=2,max_new_tokens=4,cache_len=128) as session:
        for rows in (prompts,prompts[::-1]):
            session.admit(rows,[sampling]*2)
            completed = session.poll(steps=5)
            assert len(completed)==2
            for output in completed:
                assert len(output.token_ids)==len(output.logprobs)==4
                assert torch.isfinite(torch.tensor(output.logprobs)).all()
                session.release(output.handle)
        assert session.stats['slot_refills']==2


@pytest.mark.parametrize('fp8', [False, True])
@torch.inference_mode()
def test_compact_graph_preserves_unmapped_slots(fp8):
    from climbing_gym.inference.lfm25_batching import capture_compact
    engine = tiny_engine()
    engine.enable_cuda_graphs(fp8_mlp=fp8,fp8_kv=fp8,fp8_lm_head=fp8,fp8_operators=fp8)
    decoder = engine._cuda_decoder
    entry = decoder._get_graph(4,128,masked=True)
    engine.prefill(*flatten_token_batches([[1,2],[3,4],[5,6],[7,8]],device='cuda'),cache=entry.cache)
    entry.input_tokens.copy_(torch.tensor([10,11,12,13],device='cuda'))
    entry.active.fill_(True)
    snapshot = [t.clone() for t in (entry.cache.k_cache,entry.cache.v_cache,entry.cache.conv_state,entry.cache.seq_lens)]
    compact = capture_compact(decoder,entry,2)
    for actual, expected in zip((entry.cache.k_cache,entry.cache.v_cache,entry.cache.conv_state,entry.cache.seq_lens),snapshot):
        assert torch.equal(actual.view(torch.uint8),expected.view(torch.uint8))
    compact.set_slots([3])
    compact.graph.replay()
    torch.testing.assert_close(entry.cache.seq_lens,torch.tensor([2,2,2,3],device='cuda',dtype=torch.int32))
    for actual, expected in zip((entry.cache.k_cache,entry.cache.v_cache,entry.cache.conv_state),snapshot):
        assert torch.equal(actual[:,:3].contiguous().view(torch.uint8),expected[:,:3].contiguous().view(torch.uint8))
