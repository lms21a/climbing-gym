from __future__ import annotations

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl

    TRITON_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_AVAILABLE = False


def depthwise_causal_conv_k3_reference(
    bx: torch.Tensor,
    conv_weight: torch.Tensor,
) -> torch.Tensor:
    if bx.ndim != 3 or conv_weight.ndim != 3:
        raise ValueError("Expected bx [B,H,T] and conv_weight [H,1,3].")
    if conv_weight.shape[2] != 3:
        raise ValueError(f"Expected kernel_size=3, got {conv_weight.shape[2]}.")
    return F.conv1d(
        bx,
        conv_weight,
        bias=None,
        stride=1,
        padding=2,
        groups=bx.shape[1],
    )[..., : bx.shape[-1]]


if TRITON_AVAILABLE:

    @triton.jit
    def _gated_dwconv_k3_fwd_kernel(
        bcx_ptr,
        w_ptr,
        positions_ptr,
        out_ptr,
        stride_bt,
        stride_bh,
        stride_wh,
        stride_wk,
        stride_pt,
        stride_ot,
        stride_oh,
        tlen,
        channels,
        BLOCK_SIZE: tl.constexpr,
    ):
        flat_offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        total = tlen * channels
        mask = flat_offsets < total
        token_offsets = flat_offsets // channels
        channel_offsets = flat_offsets - token_offsets * channels
        position = tl.load(
            positions_ptr + token_offsets * stride_pt,
            mask=mask,
            other=0,
        )

        current_base = token_offsets * stride_bt + channel_offsets * stride_bh
        b2 = tl.load(bcx_ptr + current_base, mask=mask, other=0.0)
        input_dtype = b2.dtype
        x2 = tl.load(
            bcx_ptr + current_base + 2 * channels * stride_bh,
            mask=mask,
            other=0.0,
        )
        c = tl.load(
            bcx_ptr + current_base + channels * stride_bh,
            mask=mask,
            other=0.0,
        )
        bx2 = (b2 * x2).to(input_dtype).to(tl.float32)

        token1 = token_offsets - 1
        mask1 = mask & (token1 >= 0) & (position >= 1)
        base1 = token1 * stride_bt + channel_offsets * stride_bh
        b1 = tl.load(bcx_ptr + base1, mask=mask1, other=0.0)
        x1 = tl.load(
            bcx_ptr + base1 + 2 * channels * stride_bh,
            mask=mask1,
            other=0.0,
        )
        bx1 = (b1 * x1).to(input_dtype).to(tl.float32)

        token0 = token_offsets - 2
        mask0 = mask & (token0 >= 0) & (position >= 2)
        base0 = token0 * stride_bt + channel_offsets * stride_bh
        b0 = tl.load(bcx_ptr + base0, mask=mask0, other=0.0)
        x0 = tl.load(
            bcx_ptr + base0 + 2 * channels * stride_bh,
            mask=mask0,
            other=0.0,
        )
        bx0 = (b0 * x0).to(input_dtype).to(tl.float32)

        weight_base = channel_offsets * stride_wh
        w0 = tl.load(w_ptr + weight_base, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(
            w_ptr + weight_base + stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            w_ptr + weight_base + 2 * stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        conv = (bx0 * w0 + bx1 * w1 + bx2 * w2).to(input_dtype)
        out = (c * conv).to(input_dtype)
        tl.store(
            out_ptr + token_offsets * stride_ot + channel_offsets * stride_oh,
            out,
            mask=mask,
        )


    @triton.jit
    def _gated_dwconv_k3_bwd_kernel(
        bcx_ptr,
        w_ptr,
        positions_ptr,
        grad_out_ptr,
        grad_bcx_ptr,
        grad_w_ptr,
        stride_bt,
        stride_bh,
        stride_wh,
        stride_wk,
        stride_pt,
        stride_gt,
        stride_gh,
        stride_gbt,
        stride_gbh,
        stride_gwh,
        stride_gwk,
        tlen,
        channels,
        BLOCK_T: tl.constexpr,
        BLOCK_H: tl.constexpr,
    ):
        token_offsets = tl.program_id(0) * BLOCK_T + tl.arange(0, BLOCK_T)
        channel_offsets = tl.program_id(1) * BLOCK_H + tl.arange(0, BLOCK_H)
        tokens = token_offsets[:, None]
        channel = channel_offsets[None, :]
        mask_t = tokens < tlen
        mask_h = channel < channels
        mask = mask_t & mask_h

        position = tl.load(
            positions_ptr + tokens * stride_pt,
            mask=mask_t,
            other=0,
        )
        base = tokens * stride_bt + channel * stride_bh
        b2 = tl.load(bcx_ptr + base, mask=mask, other=0.0)
        input_dtype = b2.dtype
        c2 = tl.load(
            bcx_ptr + base + channels * stride_bh,
            mask=mask,
            other=0.0,
        )
        x2 = tl.load(
            bcx_ptr + base + 2 * channels * stride_bh,
            mask=mask,
            other=0.0,
        )
        grad_y2 = tl.load(
            grad_out_ptr + tokens * stride_gt + channel * stride_gh,
            mask=mask,
            other=0.0,
        )
        bx2 = (b2 * x2).to(input_dtype)

        token1 = tokens - 1
        mask1 = mask & (token1 >= 0) & (position >= 1)
        base1 = token1 * stride_bt + channel * stride_bh
        b1 = tl.load(bcx_ptr + base1, mask=mask1, other=0.0)
        x1 = tl.load(
            bcx_ptr + base1 + 2 * channels * stride_bh,
            mask=mask1,
            other=0.0,
        )
        bx1 = (b1 * x1).to(input_dtype)

        token0 = tokens - 2
        mask0 = mask & (token0 >= 0) & (position >= 2)
        base0 = token0 * stride_bt + channel * stride_bh
        b0 = tl.load(bcx_ptr + base0, mask=mask0, other=0.0)
        x0 = tl.load(
            bcx_ptr + base0 + 2 * channels * stride_bh,
            mask=mask0,
            other=0.0,
        )
        bx0 = (b0 * x0).to(input_dtype)

        weight_base = channel * stride_wh
        w0 = tl.load(w_ptr + weight_base, mask=mask, other=0.0).to(tl.float32)
        w1 = tl.load(
            w_ptr + weight_base + stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)
        w2 = tl.load(
            w_ptr + weight_base + 2 * stride_wk,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        conv2 = (
            bx0.to(tl.float32) * w0
            + bx1.to(tl.float32) * w1
            + bx2.to(tl.float32) * w2
        ).to(input_dtype)
        grad_conv2 = (grad_y2 * c2).to(input_dtype)
        grad_c = (grad_y2 * conv2).to(input_dtype)

        future1 = tokens + 1
        mask_future1_t = future1 < tlen
        future1_position = tl.load(
            positions_ptr + future1 * stride_pt,
            mask=mask_future1_t,
            other=0,
        )
        mask_future1 = mask_h & mask_future1_t & (future1_position >= 1)
        future1_base = future1 * stride_bt + channel * stride_bh
        future1_c = tl.load(
            bcx_ptr + future1_base + channels * stride_bh,
            mask=mask_future1,
            other=0.0,
        )
        future1_grad_y = tl.load(
            grad_out_ptr + future1 * stride_gt + channel * stride_gh,
            mask=mask_future1,
            other=0.0,
        )
        grad_conv1 = (future1_grad_y * future1_c).to(input_dtype)

        future0 = tokens + 2
        mask_future0_t = future0 < tlen
        future0_position = tl.load(
            positions_ptr + future0 * stride_pt,
            mask=mask_future0_t,
            other=0,
        )
        mask_future0 = mask_h & mask_future0_t & (future0_position >= 2)
        future0_base = future0 * stride_bt + channel * stride_bh
        future0_c = tl.load(
            bcx_ptr + future0_base + channels * stride_bh,
            mask=mask_future0,
            other=0.0,
        )
        future0_grad_y = tl.load(
            grad_out_ptr + future0 * stride_gt + channel * stride_gh,
            mask=mask_future0,
            other=0.0,
        )
        grad_conv0 = (future0_grad_y * future0_c).to(input_dtype)

        grad_bx = (
            grad_conv0.to(tl.float32) * w0
            + grad_conv1.to(tl.float32) * w1
            + grad_conv2.to(tl.float32) * w2
        ).to(input_dtype)
        grad_b = (grad_bx * x2).to(input_dtype)
        grad_x = (grad_bx * b2).to(input_dtype)
        grad_base = tokens * stride_gbt + channel * stride_gbh
        tl.store(grad_bcx_ptr + grad_base, grad_b, mask=mask)
        tl.store(
            grad_bcx_ptr + grad_base + channels * stride_gbh,
            grad_c,
            mask=mask,
        )
        tl.store(
            grad_bcx_ptr + grad_base + 2 * channels * stride_gbh,
            grad_x,
            mask=mask,
        )

        # Every program owns one time tile. Reduce that tile before atomically
        # accumulating its three per-channel weight-gradient contributions.
        grad_conv_fp32 = grad_conv2.to(tl.float32)
        partial_w0 = tl.sum(grad_conv_fp32 * bx0.to(tl.float32), axis=0)
        partial_w1 = tl.sum(grad_conv_fp32 * bx1.to(tl.float32), axis=0)
        partial_w2 = tl.sum(grad_conv_fp32 * bx2.to(tl.float32), axis=0)
        grad_weight_base = channel_offsets * stride_gwh
        tl.atomic_add(
            grad_w_ptr + grad_weight_base,
            partial_w0,
            mask=channel_offsets < channels,
        )
        tl.atomic_add(
            grad_w_ptr + grad_weight_base + stride_gwk,
            partial_w1,
            mask=channel_offsets < channels,
        )
        tl.atomic_add(
            grad_w_ptr + grad_weight_base + 2 * stride_gwk,
            partial_w2,
            mask=channel_offsets < channels,
        )


    @triton.jit
    def _dwconv_k3_fwd_kernel(
        x_ptr,
        w_ptr,
        positions_ptr,
        out_ptr,
        stride_xh,
        stride_xt,
        stride_wh,
        stride_wk,
        stride_pt,
        stride_oh,
        stride_ot,
        tlen,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < tlen

        x_base = x_ptr + pid_h * stride_xh
        w_base = w_ptr + pid_h * stride_wh
        o_base = out_ptr + pid_h * stride_oh
        position = tl.load(positions_ptr + offs_t * stride_pt, mask=mask_t, other=0)

        w0 = tl.load(w_base + 0 * stride_wk).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_wk).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_wk).to(tl.float32)

        x0_idx = offs_t - 2
        x1_idx = offs_t - 1
        x2_idx = offs_t

        x0 = tl.load(
            x_base + x0_idx * stride_xt,
            mask=mask_t & (x0_idx >= 0) & (position >= 2),
            other=0.0,
        ).to(tl.float32)
        x1 = tl.load(
            x_base + x1_idx * stride_xt,
            mask=mask_t & (x1_idx >= 0) & (position >= 1),
            other=0.0,
        ).to(tl.float32)
        x2 = tl.load(x_base + x2_idx * stride_xt, mask=mask_t, other=0.0).to(tl.float32)

        out = x0 * w0 + x1 * w1 + x2 * w2
        tl.store(o_base + offs_t * stride_ot, out, mask=mask_t)


    @triton.jit
    def _dwconv_k3_bwd_input_kernel(
        grad_out_ptr,
        w_ptr,
        positions_ptr,
        grad_x_ptr,
        stride_gh,
        stride_gt,
        stride_wh,
        stride_wk,
        stride_pt,
        stride_xh,
        stride_xt,
        tlen,
        BLOCK_T: tl.constexpr,
    ):
        pid_t = tl.program_id(0)
        pid_h = tl.program_id(1)

        offs_t = pid_t * BLOCK_T + tl.arange(0, BLOCK_T)
        mask_t = offs_t < tlen

        go_base = grad_out_ptr + pid_h * stride_gh
        w_base = w_ptr + pid_h * stride_wh
        gx_base = grad_x_ptr + pid_h * stride_xh

        w0 = tl.load(w_base + 0 * stride_wk).to(tl.float32)
        w1 = tl.load(w_base + 1 * stride_wk).to(tl.float32)
        w2 = tl.load(w_base + 2 * stride_wk).to(tl.float32)

        g0_idx = offs_t + 2
        g1_idx = offs_t + 1
        g2_idx = offs_t

        p0 = tl.load(
            positions_ptr + g0_idx * stride_pt,
            mask=mask_t & (g0_idx < tlen),
            other=0,
        )
        p1 = tl.load(
            positions_ptr + g1_idx * stride_pt,
            mask=mask_t & (g1_idx < tlen),
            other=0,
        )

        g0 = tl.load(
            go_base + g0_idx * stride_gt,
            mask=mask_t & (g0_idx < tlen) & (p0 >= 2),
            other=0.0,
        ).to(tl.float32)
        g1 = tl.load(
            go_base + g1_idx * stride_gt,
            mask=mask_t & (g1_idx < tlen) & (p1 >= 1),
            other=0.0,
        ).to(tl.float32)
        g2 = tl.load(go_base + g2_idx * stride_gt, mask=mask_t, other=0.0).to(tl.float32)

        gx = g0 * w0 + g1 * w1 + g2 * w2
        tl.store(gx_base + offs_t * stride_xt, gx, mask=mask_t)


    @triton.jit
    def _dwconv_k3_bwd_weight_kernel(
        x_ptr,
        grad_out_ptr,
        positions_ptr,
        grad_w_ptr,
        stride_xh,
        stride_xt,
        stride_gh,
        stride_gt,
        stride_pt,
        stride_wh,
        stride_wk,
        tlen,
        BLOCK_T: tl.constexpr,
    ):
        pid_h = tl.program_id(0)

        x_base = x_ptr + pid_h * stride_xh
        go_base = grad_out_ptr + pid_h * stride_gh
        gw_base = grad_w_ptr + pid_h * stride_wh

        acc0 = tl.zeros((), dtype=tl.float32)
        acc1 = tl.zeros((), dtype=tl.float32)
        acc2 = tl.zeros((), dtype=tl.float32)

        start = 0
        while start < tlen:
            offs = start + tl.arange(0, BLOCK_T)
            mask = offs < tlen
            grad = tl.load(go_base + offs * stride_gt, mask=mask, other=0.0).to(tl.float32)
            position = tl.load(positions_ptr + offs * stride_pt, mask=mask, other=0)

            x0_idx = offs - 2
            x1_idx = offs - 1
            x2_idx = offs

            x0 = tl.load(
                x_base + x0_idx * stride_xt,
                mask=mask & (x0_idx >= 0) & (position >= 2),
                other=0.0,
            ).to(tl.float32)
            x1 = tl.load(
                x_base + x1_idx * stride_xt,
                mask=mask & (x1_idx >= 0) & (position >= 1),
                other=0.0,
            ).to(tl.float32)
            x2 = tl.load(x_base + x2_idx * stride_xt, mask=mask, other=0.0).to(tl.float32)

            acc0 += tl.sum(grad * x0, axis=0)
            acc1 += tl.sum(grad * x1, axis=0)
            acc2 += tl.sum(grad * x2, axis=0)
            start += BLOCK_T

        tl.store(gw_base + 0 * stride_wk, acc0)
        tl.store(gw_base + 1 * stride_wk, acc1)
        tl.store(gw_base + 2 * stride_wk, acc2)


    class _DepthwiseCausalConvK3Fn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            bx: torch.Tensor,
            conv_weight: torch.Tensor,
            positions: torch.Tensor,
            block_t: int,
            num_warps: int,
        ) -> torch.Tensor:
            if bx.ndim != 3 or conv_weight.ndim != 3:
                raise ValueError("Expected bx [B,H,T] and conv_weight [H,1,3].")
            if not bx.is_cuda or not conv_weight.is_cuda:
                raise ValueError("Triton shortconv requires CUDA tensors.")
            if conv_weight.device != bx.device:
                raise ValueError(
                    "Triton shortconv input and weights must share a CUDA device."
                )
            if not positions.is_cuda or positions.device != bx.device:
                raise ValueError(
                    "Triton shortconv positions must be on the input CUDA device."
                )
            bsz, channels, tlen = bx.shape
            if bsz != 1:
                raise ValueError(f"Triton shortconv currently requires bsz=1, got {bsz}.")
            if int(conv_weight.shape[0]) != int(channels):
                raise ValueError(
                    "Triton shortconv input and weight channel counts must match."
                )
            if conv_weight.shape[2] != 3:
                raise ValueError(f"Triton shortconv requires kernel_size=3, got {conv_weight.shape[2]}.")
            if positions.ndim != 2 or tuple(positions.shape) != (bsz, tlen):
                raise ValueError(
                    "Triton shortconv positions must have shape [batch, sequence_length]."
                )

            x2 = bx[0].contiguous()
            w2 = conv_weight[:, 0, :].contiguous()
            positions1d = positions[0].contiguous()
            out2 = torch.empty_like(x2)

            grid = (triton.cdiv(tlen, block_t), channels)
            _dwconv_k3_fwd_kernel[grid](
                x2,
                w2,
                positions1d,
                out2,
                x2.stride(0),
                x2.stride(1),
                w2.stride(0),
                w2.stride(1),
                positions1d.stride(0),
                out2.stride(0),
                out2.stride(1),
                tlen,
                BLOCK_T=block_t,
                num_warps=num_warps,
            )

            ctx.save_for_backward(x2, w2, positions1d)
            ctx.tlen = tlen
            ctx.block_t = block_t
            ctx.num_warps = num_warps
            return out2.unsqueeze(0)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            x2, w2, positions1d = ctx.saved_tensors
            tlen = int(ctx.tlen)
            block_t = int(ctx.block_t)
            num_warps = int(ctx.num_warps)
            channels = x2.shape[0]

            go2 = grad_out[0].contiguous()
            gx2 = torch.empty_like(x2)
            gw_fp32 = torch.empty((channels, 3), device=x2.device, dtype=torch.float32)

            grid_in = (triton.cdiv(tlen, block_t), channels)
            _dwconv_k3_bwd_input_kernel[grid_in](
                go2,
                w2,
                positions1d,
                gx2,
                go2.stride(0),
                go2.stride(1),
                w2.stride(0),
                w2.stride(1),
                positions1d.stride(0),
                gx2.stride(0),
                gx2.stride(1),
                tlen,
                BLOCK_T=block_t,
                num_warps=num_warps,
            )

            _dwconv_k3_bwd_weight_kernel[(channels,)](
                x2,
                go2,
                positions1d,
                gw_fp32,
                x2.stride(0),
                x2.stride(1),
                go2.stride(0),
                go2.stride(1),
                positions1d.stride(0),
                gw_fp32.stride(0),
                gw_fp32.stride(1),
                tlen,
                BLOCK_T=block_t,
                num_warps=num_warps,
            )

            grad_bx = gx2.unsqueeze(0)
            grad_w = gw_fp32.to(dtype=w2.dtype).unsqueeze(1)
            return grad_bx, grad_w, None, None, None


    class _GatedDepthwiseCausalConvK3Fn(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            bcx: torch.Tensor,
            conv_weight: torch.Tensor,
            positions: torch.Tensor,
            block_size: int,
            backward_block_t: int,
            backward_block_h: int,
            num_warps: int,
        ) -> torch.Tensor:
            if bcx.ndim != 3 or conv_weight.ndim != 3:
                raise ValueError("Expected bcx [B,T,3H] and conv_weight [H,1,3].")
            if not bcx.is_cuda or not conv_weight.is_cuda:
                raise ValueError("Triton gated shortconv requires CUDA tensors.")
            if conv_weight.device != bcx.device:
                raise ValueError(
                    "Triton gated shortconv input and weights must share a CUDA device."
                )
            if not positions.is_cuda or positions.device != bcx.device:
                raise ValueError(
                    "Triton gated shortconv positions must be on the input CUDA device."
                )
            bsz, tlen, projected_width = bcx.shape
            channels = int(conv_weight.shape[0])
            if bsz != 1:
                raise ValueError(
                    f"Triton gated shortconv currently requires bsz=1, got {bsz}."
                )
            if int(projected_width) != 3 * channels:
                raise ValueError(
                    "Triton gated shortconv requires bcx.shape[-1] == 3 * channels."
                )
            if conv_weight.shape[1:] != (1, 3):
                raise ValueError("Triton gated shortconv requires weight shape [H,1,3].")
            if positions.ndim != 2 or tuple(positions.shape) != (bsz, tlen):
                raise ValueError(
                    "Triton gated shortconv positions must have shape [batch, sequence_length]."
                )

            bcx2 = bcx[0].contiguous()
            weight2 = conv_weight[:, 0, :].contiguous()
            positions1d = positions[0].contiguous()
            out2 = torch.empty(
                (tlen, channels),
                device=bcx.device,
                dtype=bcx.dtype,
            )
            _gated_dwconv_k3_fwd_kernel[
                (triton.cdiv(tlen * channels, block_size),)
            ](
                bcx2,
                weight2,
                positions1d,
                out2,
                bcx2.stride(0),
                bcx2.stride(1),
                weight2.stride(0),
                weight2.stride(1),
                positions1d.stride(0),
                out2.stride(0),
                out2.stride(1),
                tlen,
                channels,
                BLOCK_SIZE=block_size,
                num_warps=num_warps,
            )

            ctx.save_for_backward(bcx2, weight2, positions1d)
            ctx.backward_block_t = int(backward_block_t)
            ctx.backward_block_h = int(backward_block_h)
            ctx.num_warps = int(num_warps)
            return out2.unsqueeze(0)

        @staticmethod
        def backward(ctx, grad_out: torch.Tensor):
            bcx2, weight2, positions1d = ctx.saved_tensors
            tlen = int(bcx2.shape[0])
            channels = int(weight2.shape[0])
            block_t = int(ctx.backward_block_t)
            block_h = int(ctx.backward_block_h)
            num_warps = int(ctx.num_warps)
            grad_out2 = grad_out[0].contiguous()
            grad_bcx2 = torch.empty_like(bcx2)
            grad_weight_fp32 = torch.zeros(
                (channels, 3),
                device=bcx2.device,
                dtype=torch.float32,
            )
            grid = (
                triton.cdiv(tlen, block_t),
                triton.cdiv(channels, block_h),
            )
            _gated_dwconv_k3_bwd_kernel[grid](
                bcx2,
                weight2,
                positions1d,
                grad_out2,
                grad_bcx2,
                grad_weight_fp32,
                bcx2.stride(0),
                bcx2.stride(1),
                weight2.stride(0),
                weight2.stride(1),
                positions1d.stride(0),
                grad_out2.stride(0),
                grad_out2.stride(1),
                grad_bcx2.stride(0),
                grad_bcx2.stride(1),
                grad_weight_fp32.stride(0),
                grad_weight_fp32.stride(1),
                tlen,
                channels,
                BLOCK_T=block_t,
                BLOCK_H=block_h,
                num_warps=num_warps,
            )
            grad_weight = grad_weight_fp32.to(weight2.dtype).unsqueeze(1)
            return (
                grad_bcx2.unsqueeze(0),
                grad_weight,
                None,
                None,
                None,
                None,
                None,
            )


def depthwise_causal_conv_k3(
    bx: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    positions: torch.Tensor | None = None,
    block_t: int = 256,
    num_warps: int = 4,
) -> torch.Tensor:
    if not TRITON_AVAILABLE:
        raise RuntimeError("Triton is not available; cannot run the Triton shortconv kernel.")
    if positions is None:
        positions = torch.arange(
            bx.shape[-1],
            device=bx.device,
            dtype=torch.long,
        ).unsqueeze(0).expand(bx.shape[0], -1)
    return _DepthwiseCausalConvK3Fn.apply(
        bx,
        conv_weight,
        positions,
        block_t,
        num_warps,
    )


def gated_depthwise_causal_conv_k3(
    bcx: torch.Tensor,
    conv_weight: torch.Tensor,
    *,
    positions: torch.Tensor,
    block_size: int = 256,
    backward_block_t: int = 8,
    backward_block_h: int = 64,
    num_warps: int = 4,
) -> torch.Tensor:
    """Fuse both LFM2 gates with packed boundary-aware depthwise Conv1d."""

    if not TRITON_AVAILABLE:
        raise RuntimeError(
            "Triton is not available; cannot run the fused gated shortconv kernel."
        )
    return _GatedDepthwiseCausalConvK3Fn.apply(
        bcx,
        conv_weight,
        positions,
        block_size,
        backward_block_t,
        backward_block_h,
        num_warps,
    )
