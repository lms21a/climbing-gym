from __future__ import annotations

from dataclasses import dataclass
import json
import random
from pathlib import Path
from typing import Any

import torch


@dataclass(frozen=True, slots=True)
class CheckpointManagerConfig:
    root_dir: str = ""
    save_interval: int = 0
    save_final: bool = True


class CheckpointManager:
    """Minimal checkpoint writer for policy checkpoints with optional value state."""

    def __init__(self, config: CheckpointManagerConfig) -> None:
        self.config = config

    @property
    def enabled(self) -> bool:
        return bool(str(self.config.root_dir).strip())

    def should_save(self, *, step: int, is_final: bool) -> bool:
        if not self.enabled:
            return False
        if is_final and self.config.save_final:
            return True
        interval = int(self.config.save_interval)
        return interval > 0 and (int(step) + 1) % interval == 0

    def save(
        self,
        *,
        step: int,
        policy_model: Any,
        value_model: Any | None = None,
        policy_optimizer: Any | None = None,
        value_optimizer: Any | None = None,
    ) -> Path | None:
        if not self.enabled:
            return None

        root = Path(self.config.root_dir)
        step_dir = root / f"step_{int(step)}"
        step_dir.mkdir(parents=True, exist_ok=True)

        policy_dir = step_dir / "policy"
        self._save_policy(policy_model, policy_dir)
        if value_model is not None:
            self._save_value(value_model, step_dir / "value_model.pt")
        if policy_optimizer is not None:
            payload = {
                "version": 1,
                "policy_optimizer": policy_optimizer.state_dict(),
                "value_optimizer": value_optimizer.state_dict() if value_optimizer is not None else None,
                "torch_rng_state": torch.get_rng_state(),
                "python_rng_state": random.getstate(),
                "cuda_rng_states": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
            }
            temporary = step_dir / "training_state.tmp"
            torch.save(payload, temporary)
            temporary.replace(step_dir / "training_state.pt")
        (step_dir / "metadata.json").write_text(
            json.dumps({"step": int(step)}, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return step_dir

    def maybe_save(
        self,
        *,
        step: int,
        is_final: bool,
        policy_model: Any,
        value_model: Any | None = None,
        policy_optimizer: Any | None = None,
        value_optimizer: Any | None = None,
    ) -> Path | None:
        if not self.should_save(step=step, is_final=is_final):
            return None
        return self.save(
            step=step,
            policy_model=policy_model,
            value_model=value_model,
            policy_optimizer=policy_optimizer,
            value_optimizer=value_optimizer,
        )

    def _save_policy(self, model: Any, save_dir: Path) -> None:
        save_as_hf = getattr(model, "save_as_hf", None)
        if callable(save_as_hf):
            save_as_hf(save_dir)
            return
        save_pretrained = getattr(model, "save_pretrained", None)
        if callable(save_pretrained):
            save_pretrained(save_dir)
            return
        raise TypeError("Policy checkpoint export requires save_as_hf(save_dir) or save_pretrained(save_dir).")

    def _save_value(self, model: Any, save_path: Path) -> None:
        state_dict = {
            key: value.detach().cpu()
            for key, value in model.state_dict().items()
        }
        torch.save({"state_dict": state_dict}, save_path)


__all__ = [
    "CheckpointManager",
    "CheckpointManagerConfig",
]


def restore_training_state(checkpoint_dir, *, policy_optimizer, value_optimizer=None, allow_weights_only=False):
    """Restore exact master/moment dtypes, optimizer settings, and RNG state.

    Legacy checkpoints without optimizer state require an explicit fresh-optimizer
    opt-in. Mapped critic state is copied into its newly created local backing.
    """
    path = Path(checkpoint_dir) / "training_state.pt"
    if not path.is_file():
        if allow_weights_only:
            return False
        raise ValueError("Checkpoint has no optimizer state; set resume.allow_weights_only=true for a fresh optimizer")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload.get("version") != 1:
        raise ValueError("Unsupported training-state checkpoint version")
    for optimizer, state in ((policy_optimizer, payload["policy_optimizer"]),
                              (value_optimizer, payload["value_optimizer"])):
        if optimizer is None:
            if state is not None:
                raise ValueError("Checkpoint has a critic optimizer but the run does not")
            continue
        if state is None:
            raise ValueError("Checkpoint is missing the critic optimizer")
        restore_exact = getattr(optimizer, "restore_exact", None)
        if callable(restore_exact):
            restore_exact(state)
        else:
            from climbing_gym.optimizer_resume import restore_master_optimizer
            # Dtype-preserving restoration also handles ordinary AdamW states.
            restore_master_optimizer(optimizer, state, require_all=False)
    torch.set_rng_state(payload["torch_rng_state"])
    random.setstate(payload["python_rng_state"])
    if payload["cuda_rng_states"] and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(payload["cuda_rng_states"])
    return True
