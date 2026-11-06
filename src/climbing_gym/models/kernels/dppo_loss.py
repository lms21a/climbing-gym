from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl

    TRITON_DPPO_AVAILABLE = True
except Exception:
    triton = None
    tl = None
    TRITON_DPPO_AVAILABLE = False


# A full LFM2.5 vocabulary fits in two passes at this size. Keeping each pass
# below 64K avoids the severe register spilling seen with a single 65,536-wide
# Triton block.
_MAX_SOFTMAX_BLOCK = 32_768


if TRITON_DPPO_AVAILABLE:

    @triton.jit
    def _dppo_loss_and_grad_kernel(
        logits_ptr,
        targets_ptr,
        behavior_logprobs_ptr,
        advantages_ptr,
        metrics_ptr,
        logits_row_stride,
        metrics_row_stride,
        n_cols,
        loss_denominator,
        metric_denominator,
        epsilon,
        log_ratio_cap,
        BLOCK_SIZE: tl.constexpr,
    ):
        """Turn one logits row into its DPPO gradient and scalar metrics."""

        row = tl.program_id(0).to(tl.int64)
        logits_row = logits_ptr + row * logits_row_stride
        target = tl.load(targets_ptr + row)
        behavior_logprob = tl.load(behavior_logprobs_ptr + row).to(tl.float32)
        advantage = tl.load(advantages_ptr + row).to(tl.float32)
        target_logit = tl.load(logits_row + target).to(tl.float32)

        row_max = float("-inf")
        row_exp_sum = 0.0
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            block = tl.load(
                logits_row + offsets,
                mask=offsets < n_cols,
                other=float("-inf"),
            ).to(tl.float32)
            block_max = tl.max(block, axis=0)
            next_max = tl.maximum(row_max, block_max)
            row_exp_sum = (
                row_exp_sum * tl.exp(row_max - next_max)
                + tl.sum(tl.exp(block - next_max), axis=0)
            )
            row_max = next_max

        logsumexp = row_max + tl.log(row_exp_sum)
        target_logprob = target_logit - logsumexp
        raw_log_ratio = target_logprob - behavior_logprob
        log_ratio = tl.maximum(
            -log_ratio_cap,
            tl.minimum(raw_log_ratio, log_ratio_cap),
        )
        ratio = tl.exp(log_ratio)

        behavior_probability = tl.maximum(
            tl.exp(behavior_logprob),
            1.1754943508222875e-38,
        )
        current_probability = tl.exp(target_logprob)
        dynamic_radius = epsilon / behavior_probability
        lower_ratio = 1.0 - dynamic_radius
        upper_ratio = 1.0 + dynamic_radius
        clipped_ratio = tl.minimum(
            tl.maximum(ratio, lower_ratio),
            upper_ratio,
        )

        unclipped_objective = ratio * advantage
        clipped_objective = clipped_ratio * advantage
        use_clipped = clipped_objective < unclipped_objective
        token_objective = tl.minimum(unclipped_objective, clipped_objective)

        # Match autograd through clamp/minimum. Away from the measure-zero clamp
        # boundaries, a selected clipped branch has zero derivative.
        ratio_has_gradient = (raw_log_ratio >= -log_ratio_cap) & (
            raw_log_ratio <= log_ratio_cap
        )
        unclipped_derivative = tl.where(
            ratio_has_gradient,
            ratio * advantage,
            0.0,
        )
        clipped_has_gradient = (
            ratio_has_gradient
            & (ratio >= lower_ratio)
            & (ratio <= upper_ratio)
        )
        clipped_derivative = tl.where(
            clipped_has_gradient,
            ratio * advantage,
            0.0,
        )
        loss_logprob_derivative = -tl.where(
            use_clipped,
            clipped_derivative,
            unclipped_derivative,
        ) / loss_denominator

        # d log p(target) / d logits = one_hot(target) - softmax(logits).
        for start in range(0, n_cols, BLOCK_SIZE):
            offsets = start + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_cols
            block = tl.load(
                logits_row + offsets,
                mask=mask,
                other=float("-inf"),
            ).to(tl.float32)
            probability = tl.exp(block - row_max) / row_exp_sum
            one_hot = offsets == target
            grad = loss_logprob_derivative * (
                one_hot.to(tl.float32) - probability
            )
            tl.store(logits_row + offsets, grad, mask=mask)

        inverse_metric_denominator = 1.0 / metric_denominator
        metrics_row = metrics_ptr + row * metrics_row_stride
        tl.store(metrics_row + 0, -token_objective / loss_denominator)
        tl.store(
            metrics_row + 1,
            use_clipped.to(tl.float32) * inverse_metric_denominator,
        )
        tl.store(metrics_row + 2, ratio * inverse_metric_denominator)
        tl.store(
            metrics_row + 3,
            ((ratio - 1.0) - log_ratio) * inverse_metric_denominator,
        )
        tl.store(
            metrics_row + 4,
            tl.abs(current_probability - behavior_probability)
            * inverse_metric_denominator,
        )


    class _FusedLinearDPPOFunction(torch.autograd.Function):
        @staticmethod
        def forward(
            ctx,
            hidden_states: torch.Tensor,
            weight: torch.Tensor,
            targets: torch.Tensor,
            behavior_logprobs: torch.Tensor,
            advantages: torch.Tensor,
            loss_denominator: float,
            clip_epsilon: float,
            log_ratio_clip: float,
            chunk_tokens: int,
        ) -> tuple[
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
            torch.Tensor,
        ]:
            if hidden_states.ndim != 2 or weight.ndim != 2:
                raise ValueError("Expected hidden_states [T,H] and weight [V,H].")
            token_count, hidden_size = hidden_states.shape
            if int(weight.shape[1]) != int(hidden_size):
                raise ValueError("The linear head and hidden-state sizes must match.")
            if token_count <= 0:
                raise ValueError("The fused DPPO kernel requires at least one token.")
            if any(
                int(tensor.numel()) != int(token_count)
                for tensor in (targets, behavior_logprobs, advantages)
            ):
                raise ValueError("Targets, behavior logprobs, and advantages must align.")
            if not hidden_states.is_cuda or not weight.is_cuda:
                raise ValueError("The fused DPPO kernel requires CUDA tensors.")
            if hidden_states.dtype not in {
                torch.bfloat16,
                torch.float16,
                torch.float32,
            }:
                raise TypeError(f"Unsupported hidden-state dtype: {hidden_states.dtype}.")
            if weight.dtype != hidden_states.dtype:
                raise TypeError("The linear head and hidden states must use the same dtype.")
            if not math.isfinite(float(loss_denominator)) or float(loss_denominator) <= 0.0:
                raise ValueError("loss_denominator must be finite and positive.")

            hidden_states = hidden_states.contiguous()
            weight = weight.contiguous()
            targets = targets.reshape(-1).contiguous()
            behavior_logprobs = behavior_logprobs.reshape(-1).to(torch.float32).contiguous()
            advantages = advantages.reshape(-1).to(torch.float32).contiguous()

            vocab_size = int(weight.shape[0])
            block_size = min(_MAX_SOFTMAX_BLOCK, triton.next_power_of_2(vocab_size))
            resolved_chunk_tokens = max(1, min(int(chunk_tokens), int(token_count)))

            grad_hidden = torch.empty_like(hidden_states)
            # Model gradients are BF16/FP16 as well. Accumulating at most a few
            # large token chunks in that native dtype avoids a persistent
            # 512 MiB FP32 vocabulary-head buffer on 65K-vocabulary models.
            grad_weight = torch.zeros_like(weight)
            metric_sums = torch.zeros(5, device=hidden_states.device, dtype=torch.float32)

            for start in range(0, int(token_count), resolved_chunk_tokens):
                end = min(start + resolved_chunk_tokens, int(token_count))
                input_chunk = hidden_states[start:end]
                logits_chunk = torch.mm(input_chunk, weight.t())
                row_count = int(end - start)
                row_metrics = torch.empty(
                    (row_count, 5),
                    device=hidden_states.device,
                    dtype=torch.float32,
                )
                _dppo_loss_and_grad_kernel[(row_count,)](
                    logits_chunk,
                    targets[start:end],
                    behavior_logprobs[start:end],
                    advantages[start:end],
                    row_metrics,
                    logits_chunk.stride(0),
                    row_metrics.stride(0),
                    vocab_size,
                    float(loss_denominator),
                    float(token_count),
                    float(clip_epsilon),
                    float(log_ratio_clip),
                    BLOCK_SIZE=block_size,
                    num_warps=32,
                )
                grad_hidden[start:end] = torch.mm(logits_chunk, weight)
                chunk_grad_weight = torch.mm(logits_chunk.t(), input_chunk)
                grad_weight.add_(chunk_grad_weight)
                metric_sums.add_(row_metrics.sum(dim=0))

            ctx.save_for_backward(
                grad_hidden,
                grad_weight,
            )
            return tuple(metric_sums.unbind())

        @staticmethod
        def backward(ctx, grad_loss: torch.Tensor, *grad_metrics: torch.Tensor):
            del grad_metrics
            grad_hidden, grad_weight = ctx.saved_tensors
            # The surrounding microbatch scaler supplies a scalar multiplier.
            # Applying it here avoids retaining the full logits graph.
            grad_hidden.mul_(grad_loss)
            grad_weight.mul_(grad_loss)
            return (
                grad_hidden,
                grad_weight,
                None,
                None,
                None,
                None,
                None,
                None,
                None,
            )


def fused_linear_dppo_loss(
    hidden_states: torch.Tensor,
    weight: torch.Tensor,
    targets: torch.Tensor,
    behavior_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    *,
    clip_epsilon: float,
    log_ratio_clip: float,
    loss_denominator: float,
    chunk_tokens: int = 2048,
) -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Memory-bounded fused linear-head and Binary-TV DPPO objective.

    Only completion-token hidden states should be passed. The implementation
    chunks the vocabulary projection, overwrites each temporary logits tile
    with its gradient in a Triton row kernel, and stores only the final hidden
    and head gradients for autograd. It therefore avoids a persistent
    ``[packed_tokens, vocab]`` activation.
    """

    if not TRITON_DPPO_AVAILABLE:
        raise RuntimeError("Triton is unavailable; cannot use fused DPPO loss.")
    if float(clip_epsilon) <= 0.0:
        raise ValueError("clip_epsilon must be > 0.")
    if float(log_ratio_clip) <= 0.0:
        raise ValueError("log_ratio_clip must be > 0.")

    return _FusedLinearDPPOFunction.apply(
        hidden_states,
        weight,
        targets,
        behavior_logprobs,
        advantages,
        float(loss_denominator),
        float(clip_epsilon),
        float(log_ratio_clip),
        int(chunk_tokens),
    )


__all__ = ["TRITON_DPPO_AVAILABLE", "fused_linear_dppo_loss"]
