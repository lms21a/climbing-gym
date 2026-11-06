from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_MASTER_ADAMW_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_MASTER_ADAMW_AVAILABLE = False


_UPDATE_BLOCK_SIZE = 1024
_BITS_BLOCK_SIZE = 1024


if TRITON_MASTER_ADAMW_AVAILABLE:

    @triton.jit
    def _round_fp32_to_bf16_float(value):
        """Round like PyTorch BF16 stores, including ties-to-even.

        Triton's direct FP32-to-BF16 cast rounds exact halfway values upward
        on CUDA, while PyTorch uses round-to-nearest-even. Returning an FP32
        value with its low word cleared also prevents a later BF16 store from
        changing the result.
        """

        bits = tl.cast(value, tl.int32, bitcast=True)
        rounding_bias = 0x7FFF + ((bits >> 16) & 1)
        rounded_bits = (bits + rounding_bias) & -0x10000
        return tl.cast(rounded_bits, tl.float32, bitcast=True)

    @triton.jit
    def _master_adamw_kernel(
        parameter_ptr,
        grad_ptr,
        master_ptr,
        exp_avg_ptr,
        exp_avg_sq_ptr,
        update_sq_blocks_ptr,
        changed_blocks_ptr,
        n_elements,
        beta1,
        beta2,
        one_minus_beta1,
        one_minus_beta2,
        learning_rate,
        weight_decay,
        bias_correction1,
        sqrt_bias_correction2,
        epsilon,
        PARAMETER_IS_BF16: tl.constexpr,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements

        parameter = tl.load(parameter_ptr + offsets, mask=mask, other=0.0)
        grad = tl.load(grad_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        master = tl.load(master_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg = tl.load(exp_avg_ptr + offsets, mask=mask, other=0.0).to(tl.float32)
        exp_avg_sq = tl.load(exp_avg_sq_ptr + offsets, mask=mask, other=0.0).to(tl.float32)

        if PARAMETER_IS_BF16:
            # Preserve the established low-memory optimizer semantics: each
            # in-place BF16 moment operation rounds before the next operation,
            # and the update consumes those stored (quantized) moments.
            scaled_exp_avg = _round_fp32_to_bf16_float(beta1 * exp_avg)
            next_exp_avg = _round_fp32_to_bf16_float(
                scaled_exp_avg + one_minus_beta1 * grad
            )
            scaled_exp_avg_sq = _round_fp32_to_bf16_float(beta2 * exp_avg_sq)
            grad_sq = grad * grad
            next_exp_avg_sq = _round_fp32_to_bf16_float(
                scaled_exp_avg_sq + one_minus_beta2 * grad_sq
            )
        else:
            next_exp_avg = beta1 * exp_avg + one_minus_beta1 * grad
            grad_sq = grad * grad
            next_exp_avg_sq = beta2 * exp_avg_sq + one_minus_beta2 * grad_sq
        update_exp_avg = next_exp_avg.to(tl.float32)
        update_exp_avg_sq = next_exp_avg_sq.to(tl.float32)
        denominator = tl.sqrt(update_exp_avg_sq) / sqrt_bias_correction2 + epsilon
        delta = update_exp_avg * (-learning_rate / bias_correction1) / denominator
        delta += master * (-learning_rate * weight_decay)
        next_master = master + delta
        if PARAMETER_IS_BF16:
            next_parameter = _round_fp32_to_bf16_float(next_master)
        else:
            next_parameter = next_master

        tl.store(exp_avg_ptr + offsets, next_exp_avg, mask=mask)
        tl.store(exp_avg_sq_ptr + offsets, next_exp_avg_sq, mask=mask)
        tl.store(master_ptr + offsets, next_master, mask=mask)
        tl.store(parameter_ptr + offsets, next_parameter, mask=mask)

        update_sq = tl.where(mask, delta * delta, 0.0)
        changed = tl.where(mask & (next_parameter != parameter), 1, 0)
        tl.store(update_sq_blocks_ptr + block_id, tl.sum(update_sq, axis=0))
        tl.store(changed_blocks_ptr + block_id, tl.sum(changed, axis=0))


    @triton.jit
    def _split_master_bits_kernel(
        master_ptr,
        low_bits_ptr,
        tie_counts_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        master = tl.load(master_ptr + offsets, mask=mask, other=0.0)
        bits = tl.cast(master, tl.int32, bitcast=True)
        low_bits = bits & 0xFFFF
        high_bits = (bits >> 16) & 0xFFFF
        # At an exact halfway value, round-to-nearest-even maps both adjacent
        # FP32 high words to the same BF16 value. Record the round-down case.
        tie_round_down = mask & (low_bits == 0x8000) & ((high_bits & 1) == 0)
        tl.store(low_bits_ptr + offsets, low_bits.to(tl.int16), mask=mask)
        tl.store(
            tie_counts_ptr + block_id,
            tl.sum(tie_round_down.to(tl.int32), axis=0),
        )


    @triton.jit
    def _write_tie_indices_kernel(
        master_ptr,
        block_offsets_ptr,
        indices_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        block_id = tl.program_id(0)
        offsets = block_id * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        master = tl.load(master_ptr + offsets, mask=mask, other=0.0)
        bits = tl.cast(master, tl.int32, bitcast=True)
        low_bits = bits & 0xFFFF
        high_bits = (bits >> 16) & 0xFFFF
        tie_round_down = mask & (low_bits == 0x8000) & ((high_bits & 1) == 0)
        local_offsets = tl.cumsum(tie_round_down.to(tl.int32), axis=0) - 1
        block_offset = tl.load(block_offsets_ptr + block_id)
        tl.store(
            indices_ptr + block_offset + local_offsets,
            offsets.to(tl.int32),
            mask=tie_round_down,
        )


    @triton.jit
    def _join_master_bits_kernel(
        parameter_bits_ptr,
        low_bits_ptr,
        master_ptr,
        n_elements,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        quantized_high = (
            tl.load(parameter_bits_ptr + offsets, mask=mask, other=0).to(tl.int32)
            & 0xFFFF
        )
        low_bits = (
            tl.load(low_bits_ptr + offsets, mask=mask, other=0).to(tl.int32)
            & 0xFFFF
        )
        # Values above halfway rounded up. For halfway values, default to the
        # round-up source; the sparse patch kernel restores round-down sources.
        source_high = quantized_high - (low_bits >= 0x8000).to(tl.int32)
        bits = (source_high << 16) | low_bits
        master = tl.cast(bits, tl.float32, bitcast=True)
        tl.store(master_ptr + offsets, master, mask=mask)


    @triton.jit
    def _patch_round_down_ties_kernel(
        master_ptr,
        tie_indices_ptr,
        tie_count,
        BLOCK_SIZE: tl.constexpr,
    ):
        offsets = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = offsets < tie_count
        indices = tl.load(tie_indices_ptr + offsets, mask=mask, other=0).to(tl.int64)
        master = tl.load(master_ptr + indices, mask=mask, other=0.0)
        bits = tl.cast(master, tl.int32, bitcast=True) + 0x10000
        tl.store(
            master_ptr + indices,
            tl.cast(bits, tl.float32, bitcast=True),
            mask=mask,
        )


def master_weight_adamw_block_count(numel: int) -> int:
    return max(1, (int(numel) + _UPDATE_BLOCK_SIZE - 1) // _UPDATE_BLOCK_SIZE)


def fused_master_weight_adamw_step_(
    parameter: torch.Tensor,
    grad: torch.Tensor,
    master_parameter: torch.Tensor,
    exp_avg: torch.Tensor,
    exp_avg_sq: torch.Tensor,
    *,
    step: int,
    learning_rate: float,
    weight_decay: float,
    beta1: float,
    beta2: float,
    epsilon: float,
    update_sq_blocks: torch.Tensor,
    changed_blocks: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run one allocation-free FP32-master AdamW update on CUDA."""

    if not TRITON_MASTER_ADAMW_AVAILABLE:
        raise RuntimeError("Triton is unavailable; cannot use fused master AdamW.")
    tensors = (parameter, grad, master_parameter, exp_avg, exp_avg_sq)
    if any(not tensor.is_cuda for tensor in tensors):
        raise ValueError("Fused master AdamW requires CUDA tensors.")
    if any(not tensor.is_contiguous() for tensor in tensors):
        raise ValueError("Fused master AdamW requires contiguous tensors.")
    if any(int(tensor.numel()) != int(parameter.numel()) for tensor in tensors):
        raise ValueError("All fused master AdamW tensors must have the same size.")
    if parameter.dtype not in {torch.bfloat16, torch.float32}:
        raise TypeError("Fused master AdamW supports BF16 and FP32 parameters.")
    if master_parameter.dtype != torch.float32:
        raise TypeError("The master parameter must be FP32.")

    blocks = master_weight_adamw_block_count(parameter.numel())
    if update_sq_blocks.numel() < blocks or changed_blocks.numel() < blocks:
        raise ValueError("Optimizer reduction scratch buffers are too small.")
    beta1_correction = 1.0 - float(beta1) ** int(step)
    beta2_correction = 1.0 - float(beta2) ** int(step)
    _master_adamw_kernel[(blocks,)](
        parameter,
        grad,
        master_parameter,
        exp_avg,
        exp_avg_sq,
        update_sq_blocks,
        changed_blocks,
        parameter.numel(),
        float(beta1),
        float(beta2),
        1.0 - float(beta1),
        1.0 - float(beta2),
        float(learning_rate),
        float(weight_decay),
        beta1_correction,
        math.sqrt(beta2_correction),
        float(epsilon),
        PARAMETER_IS_BF16=parameter.dtype == torch.bfloat16,
        BLOCK_SIZE=_UPDATE_BLOCK_SIZE,
        num_warps=8,
    )
    return (
        update_sq_blocks[:blocks].sum(),
        changed_blocks[:blocks].sum(),
    )


def compress_fp32_master_to_cpu(
    master_parameter: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Losslessly encode an FP32 master using its BF16 parameter's high word.

    The returned low words use two bytes per parameter. Only ambiguous
    round-to-nearest-even halfway values need a sparse index side table.
    """

    if master_parameter.dtype != torch.float32:
        raise TypeError("Expected an FP32 master parameter.")
    if not master_parameter.is_cuda or not TRITON_MASTER_ADAMW_AVAILABLE:
        bits = master_parameter.detach().contiguous().view(torch.int32)
        low = (bits & 0xFFFF).to(torch.int16)
        high = (bits >> 16) & 0xFFFF
        ties = torch.nonzero(
            ((bits & 0xFFFF) == 0x8000) & ((high & 1) == 0),
            as_tuple=False,
        ).reshape(-1).to(torch.int32)
        return low.cpu(), ties.cpu()

    flat_master = master_parameter.detach().contiguous().view(-1)
    element_count = int(flat_master.numel())
    blocks = max(1, (element_count + _BITS_BLOCK_SIZE - 1) // _BITS_BLOCK_SIZE)
    low_bits = torch.empty(
        element_count,
        device=flat_master.device,
        dtype=torch.int16,
    )
    tie_counts = torch.empty(blocks, device=flat_master.device, dtype=torch.int32)
    _split_master_bits_kernel[(blocks,)](
        flat_master,
        low_bits,
        tie_counts,
        element_count,
        BLOCK_SIZE=_BITS_BLOCK_SIZE,
        num_warps=4,
    )
    block_offsets = torch.cumsum(tie_counts, dim=0) - tie_counts
    tie_count = int((block_offsets[-1] + tie_counts[-1]).item())
    tie_indices = torch.empty(
        tie_count,
        device=flat_master.device,
        dtype=torch.int32,
    )
    if tie_count:
        _write_tie_indices_kernel[(blocks,)](
            flat_master,
            block_offsets,
            tie_indices,
            element_count,
            BLOCK_SIZE=_BITS_BLOCK_SIZE,
            num_warps=4,
        )
    return low_bits.cpu(), tie_indices.cpu()


def restore_fp32_master(
    quantized_parameter: torch.Tensor,
    low_bits: torch.Tensor,
    tie_round_down_indices: torch.Tensor,
    *,
    device: torch.device | str,
) -> torch.Tensor:
    """Reconstruct the exact FP32 master encoded by ``compress_*``."""

    if quantized_parameter.dtype != torch.bfloat16:
        raise TypeError("Lossless split-master restoration requires a BF16 parameter.")
    resolved_device = torch.device(device)
    parameter = quantized_parameter.detach().to(resolved_device).contiguous().view(-1)
    low = low_bits.to(resolved_device).contiguous().view(-1)
    ties = tie_round_down_indices.to(resolved_device).contiguous().view(-1)
    if int(low.numel()) != int(parameter.numel()):
        raise ValueError("Compressed master low bits do not match the parameter size.")

    if resolved_device.type == "cuda" and TRITON_MASTER_ADAMW_AVAILABLE:
        master = torch.empty_like(parameter, dtype=torch.float32)
        blocks = max(
            1,
            (int(parameter.numel()) + _BITS_BLOCK_SIZE - 1) // _BITS_BLOCK_SIZE,
        )
        _join_master_bits_kernel[(blocks,)](
            parameter.view(torch.int16),
            low,
            master,
            parameter.numel(),
            BLOCK_SIZE=_BITS_BLOCK_SIZE,
            num_warps=4,
        )
        if ties.numel():
            tie_blocks = max(
                1,
                (int(ties.numel()) + _BITS_BLOCK_SIZE - 1) // _BITS_BLOCK_SIZE,
            )
            _patch_round_down_ties_kernel[(tie_blocks,)](
                master,
                ties,
                ties.numel(),
                BLOCK_SIZE=_BITS_BLOCK_SIZE,
                num_warps=4,
            )
        return master.view_as(quantized_parameter)

    quantized_high = parameter.view(torch.int16).to(torch.int32) & 0xFFFF
    low_int32 = low.to(torch.int32) & 0xFFFF
    source_high = quantized_high - (low_int32 >= 0x8000).to(torch.int32)
    bits = (source_high << 16) | low_int32
    if ties.numel():
        bits[ties.to(torch.long)] += 0x10000
    return bits.contiguous().view(torch.float32).view_as(quantized_parameter)


__all__ = [
    "TRITON_MASTER_ADAMW_AVAILABLE",
    "compress_fp32_master_to_cpu",
    "fused_master_weight_adamw_step_",
    "master_weight_adamw_block_count",
    "restore_fp32_master",
]
