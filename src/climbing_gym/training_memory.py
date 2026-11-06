"""Shared memory defaults for alternating native rollouts and learning."""
from __future__ import annotations

from typing import Any


DEFAULT_OPTIMIZER_WORKSPACE_BYTES = 48 * 1024 * 1024
OPTIMIZER_STATE_BYTES_PER_ELEMENT = 3 * 4  # FP32 master and two Adam moments.
DEFAULT_OPTIMIZER_CHUNK_ELEMENTS = (
    DEFAULT_OPTIMIZER_WORKSPACE_BYTES // OPTIMIZER_STATE_BYTES_PER_ELEMENT
)


def optimizer_chunk_elements(chunk_elements: int, workspace_bytes: int) -> int:
    """Bound actual allocations even when an older config requests whole tensors."""
    if type(chunk_elements) is not int or chunk_elements <= 0:
        raise ValueError("chunk_elements must be a positive integer")
    if type(workspace_bytes) is not int or workspace_bytes < OPTIMIZER_STATE_BYTES_PER_ELEMENT:
        raise ValueError("workspace_bytes must hold at least three FP32 elements")
    return min(chunk_elements, workspace_bytes // OPTIMIZER_STATE_BYTES_PER_ELEMENT)


def training_memory_defaults(protocol: dict[str, Any]) -> dict[str, Any]:
    """Resolve effective settings without rewriting a frozen run's protocol."""
    p = dict(protocol)
    chunk = optimizer_chunk_elements(
        p.get("optimizer_chunk_elements", DEFAULT_OPTIMIZER_CHUNK_ELEMENTS),
        p.get("optimizer_workspace_bytes", DEFAULT_OPTIMIZER_WORKSPACE_BYTES),
    )
    p.update(optimizer_chunk_elements=chunk,
             optimizer_workspace_bytes=chunk * OPTIMIZER_STATE_BYTES_PER_ELEMENT,
             optimizer_transfer_mode="bounded_chunk_reusable_workspace")
    p.setdefault("release_inference_buffers_before_update", True)
    if type(p["release_inference_buffers_before_update"]) is not bool:
        raise ValueError("release_inference_buffers_before_update must be a boolean")
    if p["release_inference_buffers_before_update"]:
        # Buffers can only be released once the entire collection has drained.
        p.update(cache_resident=False, actor_learner_overlap=False, async_collection=False,
                 lookahead_collections=0, maximum_policy_lag=1,
                 async_schedule="Sequential collection followed by two ordered learner updates",
                 inference_buffers="Release completed native KV/graph buffers before learning; actor weights stay resident")
    return p


def prepare_policy_update(run: Any) -> None:
    """Release a drained native actor's buffers before allocating learner state."""
    if not getattr(run, "release_inference_buffers_before_update", True):
        return
    release = getattr(getattr(run, "engine", None), "release_inference_buffers", None)
    if callable(release):
        release()
