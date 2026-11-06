"""Native model post-training and verified reasoning environments.

Top-level imports are lazy; importing Climbing Gym does not initialize Torch,
CUDA, model weights, or logging services. Import specialized modules for their
full algorithm and model APIs.
"""
from __future__ import annotations

from importlib import import_module
from typing import Any

__version__ = "0.1.0"

_EXPORTS = {
    "ConfigManager": ("config", "ConfigManager"),
    "LFM2Config": ("models", "LFM2Config"),
    "LFM2Model": ("models", "LFM2Model"),
    "LFM25Engine": ("inference", "LFM25Engine"),
    "LFM25AsyncInferenceEngine": ("inference", "LFM25AsyncInferenceEngine"),
    "SamplingConfig": ("inference", "SamplingConfig"),
    "RolloutCoordinator": ("rollout", "RolloutCoordinator"),
}
__all__ = ["__version__", *_EXPORTS]


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module, attribute = _EXPORTS[name]
    value = getattr(import_module(f"{__name__}.{module}"), attribute)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_EXPORTS))
