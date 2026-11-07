"""Actual CUDA state transfer/resume parity for both native Adam modes."""
from copy import deepcopy
import pytest
import torch
from climbing_gym.cpu_master_adamw import CPUStateMasterAdamW
from climbing_gym.optimizer_resume import restore_master_optimizer
from climbing_gym.ppo import MasterWeightAdamW
from climbing_gym.trainers.common import offload_to_cpu, reload_to_device

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')

@pytest.mark.parametrize('host_state', [False, True])
def test_cuda_offload_resume_keeps_next_update_and_fp32_state_exact(host_state):
    torch.manual_seed(231)
    model = torch.nn.Linear(129, 17, bias=False, device='cuda', dtype=torch.bfloat16)
    cls = CPUStateMasterAdamW if host_state else MasterWeightAdamW
    options = dict(lr=1e-4, weight_decay=.01)
    if host_state:
        options.update(chunk_elements=257, workspace_bytes=257 * 12, pin_state=True)
    optimizer = cls(model.parameters(), **options)
    for parameter in model.parameters():
        parameter.grad = torch.full_like(parameter, .37)
    optimizer.step()
    resumed = deepcopy(model)
    offload_to_cpu(model, optimizer)
    saved = deepcopy(optimizer.state_dict())
    other = cls(resumed.parameters(), **options)
    restore_master_optimizer(other, saved)
    reload_to_device(model, optimizer, device='cuda')
    reload_to_device(resumed, other, device='cuda')
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        a.grad = torch.linspace(-.4, .5, a.numel(), device='cuda').reshape_as(a).to(a.dtype)
        b.grad = a.grad.clone()
    optimizer.step()
    other.step()
    for a, b in zip(model.parameters(), resumed.parameters(), strict=True):
        assert torch.equal(a, b)
        for key in ('master_param', 'exp_avg', 'exp_avg_sq'):
            x, y = optimizer.state[a][key], other.state[b][key]
            assert x.dtype == y.dtype
            assert x.dtype == (torch.float32 if host_state or key == 'master_param' else torch.bfloat16)
            assert torch.equal(x, y)
            assert x.device.type == ('cpu' if host_state else 'cuda')
    if host_state:
        assert optimizer.workspace_bytes_last_step <= 257 * 12
        assert other.workspace_bytes_last_step <= 257 * 12


@pytest.mark.parametrize('fp8', [False, True])
def test_owned_native_actor_close_reclaims_cuda_state_before_reload(monkeypatch, fp8):
    import asyncio
    import math
    import weakref

    from climbing_gym.inference import InferenceEngineConfig, SamplingConfig
    from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
    from test_lfm25_async_engine import _FakeTokenizer
    from test_lfm25_cuda import _engine

    references = []

    def load_native(*args, **kwargs):
        _, native = _engine()
        references.append(weakref.ref(native))
        return native

    monkeypatch.setattr('climbing_gym.inference.lfm25_async_engine.load_lfm2_tokenizer',
                        lambda *args: _FakeTokenizer())
    monkeypatch.setattr('climbing_gym.inference.lfm25_async_engine.LFM25Engine.from_pretrained',
                        load_native)
    for _ in range(2):
        adapter = LFM25AsyncInferenceEngine(
            InferenceEngineConfig(model_path='tiny-cuda-fixture', max_num_seqs=2),
            max_batch_size=2, compile_decode=False, cuda_graphs=True,
            fp8_mlp=fp8, fp8_kv=fp8, fp8_lm_head=fp8, adaptive_batching=True,
        )
        try:
            outputs = asyncio.run(adapter.generate_many(
                [{'prompt_token_ids': [1, 9]}, {'prompt_token_ids': [1, 4, 7]}],
                sampling=SamplingConfig(max_tokens=8, logprobs=1,
                                        extra_params={'ignore_eos': True}),
            ))
            assert all(len(result.outputs[0].token_ids) == 8 for result in outputs)
            assert all(math.isfinite(value) for result in outputs
                       for value in result.outputs[0].completion_logprobs)
            assert adapter.raw_engine._cuda_decoder.graphs
            decoder = weakref.ref(adapter.raw_engine._cuda_decoder)
            weight = weakref.ref(adapter.raw_engine.weights.lm_head_weight)
            torch.cuda.synchronize()
            allocated = torch.cuda.memory_allocated()
        finally:
            adapter.close()
        assert references[-1]() is None
        assert decoder() is None and weight() is None
        assert torch.cuda.memory_allocated() < allocated
