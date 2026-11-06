"""Run-scoped W&B logging with durable, unconditional local records."""
from __future__ import annotations

from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import importlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping
import uuid


def _json_value(value: Any) -> Any:
    if is_dataclass(value) and not isinstance(value, type):
        return _json_value(asdict(value))
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if hasattr(value, "to_dict"):
        return _json_value(value.to_dict())
    return str(value)


def _resolve_wandb_dir(config: Any, *, config_path: Path) -> Path:
    configured = str(getattr(config, "dir", "") or "").strip()
    override = os.environ.get("CLIMBING_GYM_WANDB_DIR", "").strip()
    selected = configured or override
    if selected:
        path = Path(selected).expanduser()
        return (path if path.is_absolute() else config_path.parent / path).resolve()
    root = Path(os.environ.get("CLIMBING_GYM_LOG_DIR", str(config_path.parent / "runs"))).expanduser()
    if not root.is_absolute():
        root = config_path.parent / root
    return (root / (datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ") + "-" + uuid.uuid4().hex[:8])).resolve()


class WandbLogger:
    """Always write local config/metrics; optionally mirror to an owned W&B run.

    ``enabled=False`` or ``mode='disabled'`` never imports or initializes W&B.
    Failures initializing/logging W&B are surfaced rather than silently changing
    an online run into an offline one. Local metrics are written first.
    """

    def __init__(self, config: Any, *, config_path: Path,
                 resolved_config: Any | None = None) -> None:
        self._config = config
        self._run: Any | None = None
        self._closed = False
        path = Path(config_path).expanduser().resolve()
        mode = str(getattr(config, "mode", None) or "online").lower().strip()
        if mode not in {"online", "offline", "disabled"}:
            raise ValueError("wandb.mode must be 'online', 'offline', or 'disabled'")
        self.log_dir = _resolve_wandb_dir(config, config_path=path)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.metrics_path = self.log_dir / "metrics.jsonl"
        configuration = _json_value(resolved_config if resolved_config is not None else {"wandb": config})
        self.config_path = self.log_dir / "resolved_config.json"
        # Pick both paths before writing anything, including after a failed
        # initialization that left a config but no metrics file.
        while self.metrics_path.exists() or self.config_path.exists():
            self.metrics_path = self.log_dir / f"metrics-{uuid.uuid4().hex[:12]}.jsonl"
            self.config_path = self.metrics_path.with_suffix(".config.json")
        self.config_path.write_text(json.dumps({"config_path": str(path), "config": configuration},
                                              indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
        self._metrics = self.metrics_path.open("a", encoding="utf-8")
        if not bool(getattr(config, "enabled", True)) or mode == "disabled":
            return
        try:
            wandb = importlib.import_module("wandb")
            init_kwargs = {"project": str(config.project), "entity": getattr(config, "entity", None),
                           "config": {"config_path": str(path), **configuration},
                           "dir": str(self.log_dir), "mode": mode}
            if getattr(config, "run_name", None):
                init_kwargs["name"] = str(config.run_name)
            self._run = wandb.init(**init_kwargs)
            if self._run is None:
                raise RuntimeError("wandb.init returned no run")
        except Exception as exc:
            self._metrics.close()
            self._closed = True
            raise RuntimeError(
                f"Could not start W&B {mode} run for {getattr(config, 'entity', None)}/{config.project}: {exc}. "
                "Install the wandb extra and authenticate for online logging, or explicitly set "
                "wandb.mode='offline' / wandb.enabled=false. "
                f"Resolved configuration is saved at {self.config_path}."
            ) from exc

    def log_step(self, *, step: int, metrics: Mapping[str, float]) -> None:
        if self._closed:
            raise RuntimeError("Cannot log to a closed WandbLogger")
        payload = _json_value({**{str(key): float(value) for key, value in metrics.items()}, "step": int(step)})
        self._metrics.write(json.dumps(payload, sort_keys=True, allow_nan=False) + "\n")
        self._metrics.flush()
        os.fsync(self._metrics.fileno())
        if self._run is not None:
            try:
                self._run.log(payload, step=int(step))
            except Exception as exc:
                raise RuntimeError(f"W&B metric logging failed; local metrics retained at {self.metrics_path}") from exc

    def close(self, *, exit_code: int = 0) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            if self._run is not None:
                self._run.finish(exit_code=int(exit_code))
        finally:
            self._metrics.close()

    def __enter__(self) -> "WandbLogger":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close(exit_code=1 if exc_type is not None else 0)


__all__ = ["WandbLogger"]
