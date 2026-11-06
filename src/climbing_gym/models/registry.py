from __future__ import annotations

from importlib import import_module
from typing import Any

from climbing_gym.models.config import BaseModelConfig


_REGISTRY: dict[str, tuple[str, str, str]] = {
    "lfm2": ("climbing_gym.models.lfm2", "LFM2Config", "LFM2Model"),
}


def list_model_names() -> list[str]:
    return sorted(_REGISTRY)


def _load_registry_entry(name: str) -> tuple[type[BaseModelConfig], type[Any]]:
    try:
        module_path, config_name, model_name = _REGISTRY[str(name).strip().lower()]
    except KeyError as exc:
        raise KeyError(f"Unknown model name: {name!r}") from exc

    module = import_module(module_path)
    config_cls = getattr(module, config_name)
    model_cls = getattr(module, model_name)
    return config_cls, model_cls


def build_config(
    name: str,
    *,
    model_dir: str | None = None,
    **config_overrides: Any,
) -> BaseModelConfig:
    config_cls, _ = _load_registry_entry(name)
    config = config_cls() if model_dir is None else config_cls.from_pretrained(model_dir)
    for key, value in config_overrides.items():
        if not hasattr(config, key):
            raise AttributeError(f"Unknown config field for {name!r}: {key!r}")
        setattr(config, key, value)
    validate = getattr(config, "validate", None)
    if callable(validate):
        validate()
    return config


def build_model(
    name: str,
    *,
    model_dir: str | None = None,
    config: BaseModelConfig | None = None,
    **config_overrides: Any,
) -> Any:
    _, model_cls = _load_registry_entry(name)
    if config is not None and config_overrides:
        raise ValueError("Pass either `config` or config overrides, not both.")

    resolved_config = config if config is not None else build_config(name, model_dir=model_dir, **config_overrides)
    model = model_cls(resolved_config)
    if model_dir is not None:
        return model.from_pretrained(model_dir)
    return model
