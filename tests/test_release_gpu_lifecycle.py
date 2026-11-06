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
