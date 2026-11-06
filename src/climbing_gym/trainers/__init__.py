from climbing_gym.trainers.checkpoints import CheckpointManager, CheckpointManagerConfig
from climbing_gym.trainers.common import (
    collect_rollout_batch,
    collect_rollout_batch_with_trainables,
    compute_grad_norm,
    offload_to_cpu,
    offload_trainables_to_cpu,
    reload_to_device,
    reload_trainables_to_device,
)
from climbing_gym.trainers.packing import (
    Batch,
    build_completion_value_tensor,
    build_valid_completion_mask,
    extract_completion_values,
    pack_trajectories,
)


__all__ = [
    "Batch",
    "CheckpointManager",
    "CheckpointManagerConfig",
    "WandbLogger",
    "build_completion_value_tensor",
    "build_valid_completion_mask",
    "collect_rollout_batch",
    "collect_rollout_batch_with_trainables",
    "compute_grad_norm",
    "extract_completion_values",
    "offload_to_cpu",
    "offload_trainables_to_cpu",
    "pack_trajectories",
    "reload_to_device",
    "reload_trainables_to_device",
]


def __getattr__(name):
    if name == "WandbLogger":
        from climbing_gym.trainers.logger import WandbLogger
        return WandbLogger
    raise AttributeError(name)
