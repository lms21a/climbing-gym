from __future__ import annotations

import json
from dataclasses import MISSING, fields
from pathlib import Path
from typing import Any


class BaseModelConfig:
    """Tiny shared config mixin for climbing_gym model files."""

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "BaseModelConfig":
        kwargs: dict[str, Any] = {}
        for field in fields(cls):
            if field.name not in payload:
                continue
            value = payload[field.name]
            if value is None:
                continue
            default = None if field.default is MISSING else field.default
            if isinstance(value, list) and isinstance(default, tuple):
                kwargs[field.name] = tuple(value)
            else:
                kwargs[field.name] = value
        config = cls(**kwargs)
        validate = getattr(config, "validate", None)
        if callable(validate):
            validate()
        return config

    @classmethod
    def from_json_file(cls, config_path: str | Path) -> "BaseModelConfig":
        path = Path(config_path)
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if not isinstance(payload, dict):
            raise TypeError(f"Expected config payload to be an object, got {type(payload)!r}.")
        return cls.from_dict(payload)

    @classmethod
    def from_pretrained(cls, model_dir: str | Path) -> "BaseModelConfig":
        config_path = Path(model_dir) / "config.json"
        if not config_path.exists():
            return cls()
        return cls.from_json_file(config_path)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {}
        for field in fields(self):
            value = getattr(self, field.name)
            payload[field.name] = list(value) if isinstance(value, tuple) else value
        return payload
