"""Shared artifact contract for the hill-climb example (no source-checkout imports)."""

from __future__ import annotations
import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load_recipe():
    data = read_json(HERE / "recipe.json")
    if data["schema_version"] != 1:
        raise ValueError("Unsupported hill-climb recipe version")
    return data


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_rows(path):
    with Path(path).open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def stage(root, stage_id):
    del root
    try:
        return dict(load_recipe()["stages"][stage_id], id=stage_id)
    except KeyError as exc:
        raise ValueError(f"Unknown hill-climb stage {stage_id!r}") from exc


def stage_dir(root, stage_id):
    stage(root, stage_id)
    return Path(root).expanduser().resolve() / "stages" / stage_id


def checkpoint_path(root, stage_id, updates=None):
    spec = stage(root, stage_id)
    if spec["kind"] == "sft":
        label = f"epoch_{updates or spec['retained_epoch']}"
    else:
        label = f"step_{(updates or spec.get('retained_updates', spec['steps'])) - 1}"
    return stage_dir(root, stage_id) / "checkpoints" / label


def model_path(root, reference):
    # @N chooses an explicitly retained intermediate policy, never a score-driven
    # automatic best checkpoint. Overrides allow starting at a known ancestor.
    reference, _, selected = reference.partition("@")
    manifest = read_json(Path(root) / "manifest.json")
    if reference in manifest.get("artifacts", {}):
        return Path(manifest["artifacts"][reference]).expanduser().resolve()
    if reference == "base":
        return Path(manifest["initial_model"]).expanduser().resolve()
    return (
        checkpoint_path(root, reference, int(selected) if selected else None) / "policy"
    )
