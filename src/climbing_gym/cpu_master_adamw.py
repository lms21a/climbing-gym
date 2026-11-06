"""Experimental host-resident FP32 Adam states for a resident GPU learner.

Reuse the native Triton optimizer update, moving bounded chunks of parameter states
onto the GPU at a time. No optimizer quantization or CPU arithmetic is used.
"""
from __future__ import annotations

import time

import torch

from climbing_gym.ppo import MasterWeightAdamW
from climbing_gym.training_memory import (
    DEFAULT_OPTIMIZER_CHUNK_ELEMENTS, DEFAULT_OPTIMIZER_WORKSPACE_BYTES,
    OPTIMIZER_STATE_BYTES_PER_ELEMENT, optimizer_chunk_elements,
)
from climbing_gym.models.kernels.master_weight_adamw import (
    fused_master_weight_adamw_step_, master_weight_adamw_block_count,
)


class CPUStateMasterAdamW(MasterWeightAdamW):
    def __init__(self, params, *, pin_state=True,
                 chunk_elements=DEFAULT_OPTIMIZER_CHUNK_ELEMENTS,
                 workspace_bytes=DEFAULT_OPTIMIZER_WORKSPACE_BYTES, **kwargs):
        super().__init__(params, **kwargs)
        self.chunk_elements = optimizer_chunk_elements(chunk_elements, workspace_bytes)
        self.workspace_bytes = self.chunk_elements * OPTIMIZER_STATE_BYTES_PER_ELEMENT
        self.pin_state = pin_state
        self._host_staging = None

    def host_state_copy(self, value):
        result = torch.empty(value.shape, dtype=torch.float32, device='cpu', pin_memory=self.pin_state)
        result.copy_(value.detach())
        return result

    def initialize_state(self):
        for group in self.param_groups:
            for parameter in group['params']:
                state=self.state[parameter]
                if not state:
                    state.update(step=0,
                        master_param=self.host_state_copy(parameter),
                        exp_avg=torch.zeros(parameter.shape,dtype=torch.float32,pin_memory=self.pin_state),
                        exp_avg_sq=torch.zeros(parameter.shape,dtype=torch.float32,pin_memory=self.pin_state))

    def offload_state_to_cpu(self):
        # The exact masters and moments already live in host memory.
        pass

    def reload_state_to_device(self,device):
        # Keep states on the host even when the policy is moved to CUDA.
        pass

    @torch.no_grad()
    def step(self,closure=None):
        if closure is None:
            loss=None
        else:
            with torch.enable_grad(): loss=closure()
        self.initialize_state()
        active=[(p,self.state[p],g) for g in self.param_groups for p in g['params'] if p.grad is not None]
        if not active: return loss
        if any(not p.is_cuda or p.grad.is_sparse for p,_,_ in active):
            raise ValueError('CPUStateMasterAdamW requires dense CUDA gradients')
        device=active[0][0].device
        size=min(self.chunk_elements, max(p.numel() for g in self.param_groups for p in g['params']))
        self.workspace_bytes_last_step = size * OPTIMIZER_STATE_BYTES_PER_ELEMENT
        max_blocks=master_weight_adamw_block_count(size)
        sq_blocks=torch.empty(max_blocks,device=device,dtype=torch.float32)
        changed_blocks=torch.empty(max_blocks,device=device,dtype=torch.int32)
        sq_sum=torch.zeros((),device=device,dtype=torch.float32)
        changed_sum=torch.zeros((),device=device,dtype=torch.int64)
        self.transfer_bytes_last_step=0
        started=time.perf_counter()
        if not self.pin_state and self._host_staging is None:
            self._host_staging=[torch.empty(size,dtype=torch.float32,pin_memory=True) for _ in range(3)]
        # Reuse a bounded workspace, including across parameters. Full embedding
        # states previously needed three 512 MiB allocations at optimizer time.
        device_buffers=[torch.empty(size,device=device,dtype=torch.float32) for _ in range(3)]
        keys=('master_param','exp_avg','exp_avg_sq')
        for parameter,state,group in active:
            if not parameter.is_contiguous() or not parameter.grad.is_contiguous():
                raise ValueError('CPUStateMasterAdamW requires contiguous parameters and gradients')
            state['step']=int(state['step'])+1
            beta1,beta2=group['betas']
            for start in range(0,parameter.numel(),size):
                end=min(start+size,parameter.numel())
                count=end-start
                host=[state[key].view(-1)[start:end] for key in keys]
                staging=([buffer[:count] for buffer in self._host_staging]
                         if not self.pin_state else host)
                if not self.pin_state:
                    for target,value in zip(staging,host):
                        target.copy_(value)
                buffers=[buffer[:count] for buffer in device_buffers]
                for target,value in zip(buffers,staging):
                    target.copy_(value,non_blocking=True)
                sq,changed=fused_master_weight_adamw_step_(
                    parameter.view(-1)[start:end],parameter.grad.view(-1)[start:end],*buffers,
                    step=state['step'],learning_rate=group['lr'],weight_decay=group['weight_decay'],
                    beta1=beta1,beta2=beta2,epsilon=group['eps'],
                    update_sq_blocks=sq_blocks,changed_blocks=changed_blocks)
                sq_sum.add_(sq);changed_sum.add_(changed)
                for target,buffer in zip(staging,buffers):
                    target.copy_(buffer,non_blocking=True)
                    self.transfer_bytes_last_step+=2*buffer.numel()*buffer.element_size()
                torch.cuda.current_stream(device).synchronize()
                if not self.pin_state:
                    for target,value in zip(host,staging):
                        target.copy_(value)
        self.last_master_update_norm=float(sq_sum.sqrt().item())
        self.last_quantized_change_count=int(changed_sum.item())
        self.transfer_seconds_last_step=time.perf_counter()-started
        return loss
