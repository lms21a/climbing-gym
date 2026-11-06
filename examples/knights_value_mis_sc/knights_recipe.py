"""Portable profile/data preparation; no SimplePT paths or model calls."""

from __future__ import annotations

from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import shutil
from pathlib import Path
import tomllib

from climbing_gym.knights_knaves import Puzzle, render_prompt, unique_solution

HERE = Path(__file__).resolve().parent


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write_toml(path, payload):
    """Write the simple nested TOML tables used by these two profiles."""
    lines = []

    def table(values, prefix=()):
        if prefix:
            lines.extend(["", "[" + ".".join(prefix) + "]"])
        for key, value in values.items():
            if not isinstance(value, dict) and value is not None:
                lines.append(f"{key} = {json.dumps(value, ensure_ascii=False)}")
        for key, value in values.items():
            if isinstance(value, dict):
                table(value, (*prefix, key))

    table(payload)
    Path(path).write_text("\n".join(lines).lstrip() + "\n")


def read_bank(split, provenance):
    compressed = (HERE / "data" / f"{split}.jsonl.gz").read_bytes()
    expected = provenance["data"][split]
    raw = gzip.decompress(compressed)
    if (
        digest(compressed) != expected["gzip_sha256"]
        or digest(raw) != expected["portable_sha256"]
    ):
        raise ValueError(f"Bundled {split} archive checksum differs")
    rows = [json.loads(line) for line in raw.splitlines()]
    if (
        len(rows) != expected["rows"]
        or dict(Counter(row["cell"] for row in rows)) != expected["per_cell"]
    ):
        raise ValueError(f"Bundled {split} panel counts differ")
    for row in rows:
        puzzle = Puzzle.from_row(row)
        if tuple(row["solution"]) != unique_solution(puzzle) or row[
            "prompt"
        ] != render_prompt(puzzle):
            raise ValueError(f"Solver or public prompt differs for {row['id']}")
    return rows


def select_panel(rows, cells, per_cell):
    return [
        row
        for cell in cells
        for row in [row for row in rows if row["cell"] == cell][:per_cell]
    ]


def evaluation_config(config, *, split, rows, repeats):
    return {
        "model": deepcopy(config["model"]),
        "data": {"dataset_path": f"data/{split}.jsonl", "shuffle": False},
        "env": deepcopy(config["env"]),
        "harness": deepcopy(config["harness"]),
        "inference": {
            **deepcopy(config["inference"]),
            "preflight_report_path": f"preflight/{split}.json",
        },
        "rollout": {
            "num_rollouts": repeats,
            "batch_size": rows,
            "effective_batch_size": rows,
            "max_inflight_requests": config["inference"]["native_max_batch_size"],
            "max_total_requests": rows * repeats,
            "use_dynamic_sampling": False,
        },
        "eval": {
            "max_examples": rows,
            "pass_k": 1,
            "shuffle": False,
            "seed": 2026110411,
        },
        "logging": {
            "summary_path": f"eval/{split}/summary.json",
            "trajectory_path": f"eval/{split}/trajectories.jsonl",
            "trace_path": f"eval/{split}/traces.md",
            "max_traces": 12,
        },
    }


def prepare(root, model, *, profile="production"):
    root, model = Path(root).resolve(), Path(model).resolve()
    if root.exists():
        raise FileExistsError(f"Refusing to overwrite an existing run: {root}")
    if not model.is_dir() or not (model / "config.json").is_file():
        raise ValueError(
            "Provide a local HF-format LFM2.5 checkpoint directory with config.json and tokenizer"
        )
    if profile not in {"production", "smoke"}:
        raise ValueError("Choose the production or smoke profile")
    provenance = json.loads((HERE / "provenance.json").read_text())
    banks = {split: read_bank(split, provenance) for split in ("train", "dev", "test")}
    ids = [set(row["id"] for row in banks[split]) for split in ("train", "dev", "test")]
    groups = [
        set(row["structural_group"] for row in banks[split])
        for split in ("train", "dev", "test")
    ]
    if any(
        ids[i] & ids[j] or groups[i] & groups[j]
        for i in range(3)
        for j in range(i + 1, 3)
    ):
        raise ValueError("Bundled training/held-out banks overlap")
    cells = provenance["evaluation_cells"]
    panels = {
        "train": banks["train"],
        "dev": select_panel(banks["dev"], cells, 16),
        "test": select_panel(banks["test"], cells, 64),
    }
    cfg = tomllib.loads((HERE / f"{profile}.toml").read_text())
    cfg["model"]["path"] = str(model)
    cfg["wandb"]["run_name"] = root.name
    if profile == "smoke":
        panels = {
            "train": panels["train"][:8],
            "dev": select_panel(banks["dev"], cells, 1),
            "test": select_panel(banks["test"], cells, 1),
        }
    repeats = 4 if profile == "production" else 1
    root.mkdir(parents=True)
    for folder in ("data", "preflight", "logs", "cache", "checkpoints", "eval"):
        (root / folder).mkdir()
    manifest = {
        "profile": profile,
        "model": str(model),
        "source": provenance,
        "historical_initialization_verified": False,
        "data": {},
    }
    weights = model / "pytorch_model.bin"
    if weights.is_file():
        with weights.open("rb") as stream:
            weight_sha = hashlib.file_digest(stream, "sha256").hexdigest()
        manifest["model_sha256"] = weight_sha
        manifest["historical_initialization_verified"] = (
            weight_sha == provenance["initial_checkpoint_sha256"]
        )
    for split, rows in panels.items():
        body = "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows)
        path = root / "data" / f"{split}.jsonl"
        path.write_text(body)
        manifest["data"][split] = {
            "rows": len(rows),
            "per_cell": dict(Counter(r["cell"] for r in rows)),
            "sha256": digest(body.encode()),
            "samples_per_puzzle": 1 if split == "train" else repeats,
        }
    cfg["eval"]["max_examples"] = len(panels["dev"])
    cfg["eval"]["rollout"].update(
        batch_size=len(panels["dev"]),
        effective_batch_size=len(panels["dev"]),
        num_rollouts=repeats,
        max_total_requests=len(panels["dev"]) * repeats,
    )
    write_toml(root / "train.toml", cfg)
    for split in ("dev", "test"):
        write_toml(
            root / f"eval_{split}.toml",
            evaluation_config(
                cfg, split=split, rows=len(panels[split]), repeats=repeats
            ),
        )
    (root / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n"
    )
    return manifest


def retain_checkpoints(root, payload, *, final_updates):
    """Bound this example's large actor/critic state; keep selected HF policies.

    Exact resumption uses the latest two full checkpoint directories. Policy
    exports contain no optimizer/readiness/dataset state and are eval artifacts.
    """
    if payload.get("phase") == "critic_warmup" or not payload.get("checkpoint_path"):
        return
    root = Path(root)
    current = Path(payload["checkpoint_path"])
    required = (
        "policy",
        "value_model.pt",
        "training_state.pt",
        "dataset_state.json",
        "value_warmup_state.json",
    )
    if not all((current / item).exists() for item in required):
        raise RuntimeError(
            "New exact actor checkpoint is incomplete; earlier states retained"
        )
    complete = []
    for directory in (root / "checkpoints").glob("step_*"):
        if not all((directory / item).exists() for item in required):
            continue
        metadata = json.loads((directory / "metadata.json").read_text())
        phase = json.loads((directory / "value_warmup_state.json").read_text())["state"]
        update = int(metadata["step"]) + 1
        if not phase["ready"] or phase["actor_updates"] != update:
            raise ValueError(
                "Actor checkpoint phase does not match its completed update"
            )
        complete.append((update, directory))
    complete.sort()
    if not complete or complete[-1][1].resolve() != current.resolve():
        raise ValueError(
            "New checkpoint is not the latest actor boundary; earlier states retained"
        )
    updates = complete[-1][0]
    if updates % 20 == 0 or updates == final_updates:
        archive = root / "policy_exports" / f"update_{updates:03}"
        if not archive.exists():
            temporary = archive.with_name(archive.name + ".incomplete")
            if temporary.exists():
                raise FileExistsError(
                    f"Inspect incomplete selected policy before retrying: {temporary}"
                )
            temporary.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(current / "policy", temporary)
            (temporary / "export.json").write_text(
                json.dumps(
                    {
                        "actor_updates": updates,
                        "source_checkpoint": str(current.resolve()),
                        "exact_resume": False,
                    }
                )
                + "\n"
            )
            temporary.replace(archive)
    pointer = root / "latest_actor_checkpoint.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(
            {
                "actor_updates": updates,
                "checkpoint_path": str(current.resolve()),
                "exact_resume": True,
            },
            indent=2,
        )
        + "\n"
    )
    temporary.replace(pointer)
    if Path(json.loads(pointer.read_text())["checkpoint_path"]) != current.resolve():
        raise RuntimeError(
            "Actor checkpoint pointer was not published; earlier states retained"
        )
    for _, directory in complete[:-2]:
        shutil.rmtree(directory)
