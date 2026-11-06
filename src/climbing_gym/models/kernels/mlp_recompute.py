"""SwiGLU MLP with an explicit backward that saves only its narrow input.

GEMMs use PyTorch's CUDA BLAS implementation. CUDA elementwise forward/backward
reuse the same Liger Triton kernels as the reference model, including its BF16
rounding. Expanded gate/up/product tensors are recomputed one layer at a time.
Only first-order differentiation is supported.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F
from torch.autograd.function import once_differentiable


def _product(gate, up, fused):
    if fused:
        from liger_kernel.ops.swiglu import swiglu_forward
        return swiglu_forward(gate, up)[2]
    return F.silu(gate) * up


class _RecomputedMLP(torch.autograd.Function):
    @staticmethod
    @torch.amp.custom_fwd(device_type="cuda")
    def forward(ctx, x, w1, w3, w2, fused):
        flat = x.reshape(-1, x.shape[-1])
        gate = F.linear(flat, w1)
        up = F.linear(flat, w3)
        output = F.linear(_product(gate, up, fused), w2)
        ctx.save_for_backward(x, w1, w3, w2)
        ctx.fused = fused
        return output.reshape(*x.shape[:-1], w2.shape[0])

    @staticmethod
    @once_differentiable
    @torch.amp.custom_bwd(device_type="cuda")
    def backward(ctx, grad_output):
        x, w1, w3, w2 = ctx.saved_tensors
        flat = x.reshape(-1, x.shape[-1])
        grad = grad_output.reshape(-1, grad_output.shape[-1]).contiguous()
        gate = F.linear(flat, w1)
        up = F.linear(flat, w3)
        product = _product(gate, up, ctx.fused)
        dw2 = grad.T @ product if ctx.needs_input_grad[3] else None
        del product
        dproduct = grad @ w2
        if ctx.fused:
            from liger_kernel.ops.swiglu import swiglu_backward
            # These are private recomputed tensors: the kernel can overwrite
            # gate/up with their derivatives without touching saved activations.
            dgate, dup = swiglu_backward(gate, up, dproduct)
        else:
            # Match the unfused SiLU -> multiply chain, including the multiply's
            # cast to the original dtype before SiLU backward.
            dup = dproduct * F.silu(gate)
            dgate = torch.ops.aten.silu_backward(dproduct * up, gate)
        # Autocast may use BF16 projections with FP32 activations. Cast the
        # branch gradients back before accumulating in the activation dtype.
        dx = ((dgate @ w1).to(x.dtype) + (dup @ w3).to(x.dtype)).reshape_as(x) if ctx.needs_input_grad[0] else None
        dw1 = dgate.T @ flat if ctx.needs_input_grad[1] else None
        dw3 = dup.T @ flat if ctx.needs_input_grad[2] else None
        return dx, dw1, dw3, dw2, None


def recomputed_swiglu_mlp(x, w1, w3, w2, *, fused=True):
    if x.ndim < 2 or any(w.ndim != 2 for w in (w1, w3, w2)):
        raise ValueError("Expected input [..., hidden] and three matrix weights")
    if w1.shape != w3.shape or w1.shape[1] != x.shape[-1] or w2.shape[1] != w1.shape[0]:
        raise ValueError("Incompatible SwiGLU projection shapes")
    return _RecomputedMLP.apply(x, w1, w3, w2, bool(fused and x.is_cuda))
