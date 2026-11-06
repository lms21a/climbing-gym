from __future__ import annotations

import gc
from typing import Any

import torch


def offload_to_cpu(model: Any, optimizer: Any | None) -> None:
    """Move a training model and optimizer state off the accelerator."""
    model.to("cpu")
    if optimizer is not None:
        optimizer_offload = getattr(optimizer, "offload_state_to_cpu", None)
        if callable(optimizer_offload):
            optimizer_offload()
        else:
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.detach().cpu()
        evict = getattr(optimizer, "evict", None)
        if callable(evict):
            evict()
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
    gc.collect()


def reload_to_device(
    model: Any,
    optimizer: Any | None,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
) -> None:
    """Bring a training model and optimizer state back onto the target device."""
    resolved_device = torch.device(device)
    if dtype is None:
        model.to(resolved_device)
    else:
        model.to(device=resolved_device, dtype=dtype)
    if optimizer is not None:
        optimizer_reload = getattr(optimizer, "reload_state_to_device", None)
        if callable(optimizer_reload):
            optimizer_reload(resolved_device)
        else:
            for state in optimizer.state.values():
                for key, value in list(state.items()):
                    if torch.is_tensor(value):
                        state[key] = value.to(resolved_device)


def offload_trainables_to_cpu(trainables: Any) -> None:
    """Move multiple ``(model, optimizer)`` pairs off the accelerator."""
    for model, optimizer in trainables:
        offload_to_cpu(model, optimizer)


def reload_trainables_to_device(
    trainables: Any,
    *,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
) -> None:
    """Bring multiple ``(model, optimizer)`` pairs back onto the target device."""
    for model, optimizer in trainables:
        reload_to_device(
            model,
            optimizer,
            device=device,
            dtype=dtype,
        )


def collect_rollout_batch_with_trainables(
    *,
    rollout_coordinator: Any,
    trainables: Any,
    sync_model: Any,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    drain_timeout: int = 300,
    release_inference_buffers_before_update: bool = True,
) -> Any:
    """Offload multiple train states, collect a rollout batch, then reload them."""
    offload_trainables_to_cpu(trainables)
    try:
        rollout_coordinator.sync_weights(
            sync_model,
            drain_timeout=drain_timeout,
        )
        return rollout_coordinator.collect()
    finally:
        if release_inference_buffers_before_update:
            release = getattr(getattr(rollout_coordinator, "engine", None), "release_inference_buffers", None)
            if callable(release):
                release()
        reload_trainables_to_device(
            trainables,
            device=device,
            dtype=dtype,
        )


def collect_rollout_batch(
    *,
    rollout_coordinator: Any,
    model: Any,
    optimizer: Any | None,
    device: str | torch.device = "cuda",
    dtype: torch.dtype | None = None,
    drain_timeout: int = 300,
    release_inference_buffers_before_update: bool = True,
) -> Any:
    """Offload train state, collect a rollout batch, then reload train state."""
    return collect_rollout_batch_with_trainables(
        rollout_coordinator=rollout_coordinator,
        trainables=((model, optimizer),),
        sync_model=model,
        device=device,
        dtype=dtype,
        drain_timeout=drain_timeout,
        release_inference_buffers_before_update=release_inference_buffers_before_update,
    )


def compute_grad_norm(model: Any) -> float:
    """Return the L2 norm across all gradients currently present on the model."""
    gradients = [
        parameter.grad.detach()
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    if not gradients:
        return 0.0
    # The previous loop called .item() once per parameter, serializing hundreds
    # of otherwise independent CUDA reductions. PyTorch's foreach path launches
    # them as a group and synchronizes only for the final scalar.
    return float(
        torch.nn.utils.get_total_norm(
            gradients,
            norm_type=2.0,
            foreach=None,
        ).item()
    )


__all__ = [
    "compute_grad_norm",
    "collect_rollout_batch_with_trainables",
    "offload_to_cpu",
    "offload_trainables_to_cpu",
    "reload_to_device",
    "reload_trainables_to_device",
    "collect_rollout_batch",
]
