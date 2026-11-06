"""Materialize the complete historical stage graph, data, and portable TOML files."""

from __future__ import annotations
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil

from recipe import (
    HERE,
    load_recipe,
    stage,
    stage_dir,
    model_path,
    read_json,
    write_json,
    digest,
)


def emit_toml(value, prefix=""):
    lines = []
    leaves = {
        k: v for k, v in value.items() if not isinstance(v, dict) and v is not None
    }
    if prefix:
        lines += ["[" + prefix + "]"]
    for key, item in leaves.items():
        lines.append(key + " = " + json.dumps(item))
    for key, item in value.items():
        if isinstance(item, dict):
            lines += ["", emit_toml(item, prefix + "." + key if prefix else key)]
    return "\n".join(lines) + "\n"


def config_for(root, stage_id):
    spec = stage(root, stage_id)
    folder = stage_dir(root, stage_id)
    source = model_path(root, spec["input_policy"])
    fp8 = spec["fp8"]
    cohort = spec["cohort_size"]
    precursor = spec["reward"] == "historical_joint"
    # All file paths are relative to this generated train.toml, including models
    # from the same run. User-supplied base/registered models remain explicit.
    import os

    def relative(path):
        return os.path.relpath(Path(path), folder)

    cfg = dict(
        job=dict(description="Futoshiki hill climb: " + stage_id),
        model=dict(
            name="lfm2",
            path=relative(source),
            config=dict(
                shortconv_backend="triton_k3",
                fused_backbone_kernels=True,
                **spec.get(
                    "model_config", {"recompute_mlp": True, "checkpoint_layers": True}
                ),
            ),
        ),
        data=dict(dataset_path="data/train.jsonl", seed=spec["seed"], shuffle=False),
        env=dict(
            name=spec["environment"],
            reward="exact_match" if precursor else spec["reward"],
        ),
        harness=dict(name="direct"),
        inference=dict(
            backend="lfm25_native",
            model_path=relative(source),
            dtype="bfloat16",
            seed=spec["seed"],
            preflight_report_path="preflight/train.json",
            max_model_len=spec["context"],
            max_num_seqs=cohort,
            native_max_batch_size=cohort,
            native_cuda_graphs=True,
            native_cuda_max_graphs=1,
            native_prefill_tokens=4096,
            native_bf16_weight_packing=not fp8,
            native_fp8_mlp=fp8,
            native_fp8_kv=fp8,
            native_fp8_lm_head=fp8,
            native_adaptive_batching=fp8,
            compile_decode=False,
            apply_chat_template=not precursor,
            chat_template_enable_thinking=False,
            assistant_prefill="" if precursor else "<thinking>\n",
            sampling=dict(
                temperature=1.0,
                top_p=1.0,
                top_k=-1,
                max_tokens=spec["max_tokens"],
                logprobs=1,
                extra_params={"behavior_topk": 32} if fp8 else {},
            ),
        ),
        rollout=dict(
            num_rollouts=1,
            batch_size=cohort,
            effective_batch_size=cohort,
            max_total_requests=cohort,
            max_inflight_requests=cohort,
            use_dynamic_sampling=False,
            stall_timeout_seconds=1200.0,
        ),
        trainer=dict(
            algorithm="flashreinforce",
            num_steps=spec["steps"],
            seed=spec["seed"],
            train_device="cuda",
            train_dtype="bfloat16",
            policy_epochs=1,
            advantage_estimator="batch_centered",
            loss_aggregation="sample_mean",
            policy_max_packed_tokens=spec["pack_tokens"],
            policy_max_grad_norm=1.0,
            compile_losses=False,
            flashreinforce_objective=spec["objective"],
            flashreinforce_length_normalization=spec.get(
                "normalization", "legacy_response_mean"
            ),
            flashreinforce_head_chunk_size=spec["head_chunk"],
            flashreinforce_fused_head=fp8,
            flashreinforce_checkpoint_head=not fp8,
            offpolicy_ratio_min=0.5,
            offpolicy_ratio_max=5.0,
            optimizer_workspace_bytes=48 * 1024 * 1024,
            release_inference_buffers_before_update=True,
        ),
        policy_optimizer=dict(
            type="cpu_master_adamw",
            learning_rate=spec["learning_rate"],
            weight_decay=0.0,
        ),
        checkpoint=dict(
            root_dir="checkpoints",
            save_interval=spec["checkpoint_interval"],
            save_final=True,
        ),
        wandb=dict(
            enabled=True,
            entity="lukas928",
            project="simple-pt",
            mode="online",
            dir="logs",
            run_name="futoshiki-hill-" + stage_id,
        ),
        train_logging=dict(
            metrics_path="metrics.jsonl", trajectory_path="trajectories.jsonl"
        ),
    )
    return cfg


def materialize_asset(root, stage_id, asset_manifest):
    spec = stage(root, stage_id)
    folder = stage_dir(root, stage_id)
    selected = {}
    for split, entry in asset_manifest["assets"][spec["data_asset"]].items():
        source = HERE / "data" / entry["file"]
        if digest(source) != entry["gzip_sha256"]:
            raise ValueError("Bundled puzzle bank checksum differs: " + str(source))
        destination = folder / "data" / f"{split}.jsonl"
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_suffix(".tmp")
        sha = hashlib.sha256()
        source_sha = hashlib.sha256()
        count = 0
        bounds = spec.get("train_slice") if split == "train" else None
        with gzip.open(source, "rb") as stream, temporary.open("wb") as output:
            for index, line in enumerate(stream):
                source_sha.update(line)
                if bounds is None or bounds[0] <= index < bounds[1]:
                    output.write(line)
                    sha.update(line)
                    count += 1
        if source_sha.hexdigest() != entry["sha256"]:
            raise ValueError("Decompressed puzzle bank checksum differs")
        if bounds and count != bounds[1] - bounds[0]:
            raise ValueError("Historical training slice is incomplete")
        temporary.replace(destination)
        selected[split] = dict(
            rows=count,
            sha256=sha.hexdigest(),
            asset_sha256=entry["sha256"],
            source_rows=bounds or "all",
        )
    # Stable alias used by generic monitor configs; exact named panels remain.
    if "monitor" not in selected and "validation" in selected:
        shutil.copyfile(folder / "data/validation.jsonl", folder / "data/monitor.jsonl")
        selected["monitor"] = selected["validation"]
    write_json(
        folder / "data/manifest.json",
        dict(mode="historical_exact_rows", splits=selected),
    )
    return selected


def generate_disjoint_banks(*, size, settings, counts, seed, max_attempts_per_row=500):
    """Reserve held-out targets first, with finite rejection budgets."""
    from climbing_gym.futoshiki import generate, Puzzle, solve

    rng = random.Random(seed)
    seen = set()
    owners = {}
    splits = {}
    for split in [
        s for s in ("monitor", "test", "calibration", "train") if s in counts
    ]:
        rows = []
        count = counts[split]
        attempts = 0
        while len(rows) < count:
            attempts += 1
            if attempts > max(100, count * max_attempts_per_row):
                raise RuntimeError(
                    f"Could not fill {split}: {len(rows)}/{count}; reduce counts or relax constraints"
                )
            g, i = settings[len(rows) % len(settings)]
            row = generate(rng, size=size, givens_count=g, inequality_count=i)
            target = tuple(row["solution"])
            if row["id"] in seen or target in owners and owners[target] != split:
                continue
            if solve(Puzzle.from_row(row)) != [target]:
                raise ValueError("Generator returned an invalid target")
            seen.add(row["id"])
            owners[target] = split
            rows.append(row)
        splits[split] = rows
    return splits


def regenerate_futoshiki(root, stage_id):
    """Fresh method replication; the two 4x4 phases slice one shared bank."""
    spec = stage(root, stage_id)
    if spec["environment"] != "futoshiki":
        raise ValueError("Fresh regeneration applies only to Futoshiki")
    folder = stage_dir(root, stage_id)
    size = 4 if stage_id.startswith("futoshiki4") else 5
    tier = (
        [(11, 5)]
        if "easy" in stage_id
        else [(9, 5)]
        if "tier2" in stage_id
        else [(8, 7)]
        if "tier3" in stage_id
        else [(6, 8)]
        if "tier4" in stage_id
        else [(6, 8), (8, 7), (9, 5), (11, 5)]
    )
    if size == 4:
        tier = [(4, 5), (5, 4), (6, 3), (7, 3)]
        cache = Path(root) / ".generated/futoshiki4.json"
        if not cache.exists():
            banks = generate_disjoint_banks(
                size=4,
                settings=tier,
                counts=dict(train=128000, monitor=32, test=128, calibration=32),
                seed=20261010,
            )
            write_json(cache, banks)
        banks = read_json(cache)
        left, right = spec["train_slice"]
        banks["train"] = banks["train"][left:right]
    else:
        banks = generate_disjoint_banks(
            size=5,
            settings=tier,
            counts=dict(
                train=spec["steps"] * spec["batch_size"],
                monitor=32,
                test=128,
                calibration=32,
            ),
            seed=spec.get("generation_seed", spec["seed"]),
        )
    splits = {}
    for split, rows in banks.items():
        path = folder / "data" / f"{split}.jsonl"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
        splits[split] = dict(rows=len(rows), sha256=digest(path))
    write_json(
        folder / "data/manifest.json",
        dict(
            mode="regenerated_method_replication",
            seed=spec["seed"],
            shared_4x4_bank=size == 4,
            splits=splits,
        ),
    )
    return splits


def prepare(
    root, initial_model, *, artifacts=None, data_mode="historical", stages=None
):
    root = Path(root).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest = root / "manifest.json"
    identity = dict(
        schema_version=1,
        initial_model=str(Path(initial_model).expanduser().resolve()),
        artifacts={
            k: str(Path(v).expanduser().resolve()) for k, v in (artifacts or {}).items()
        },
        recipe_sha256=digest(HERE / "recipe.json"),
        data_mode=data_mode,
    )
    if manifest.exists() and read_json(manifest) != identity:
        raise ValueError("Existing run manifest differs; use a separate run directory")
    write_json(manifest, identity)
    data = read_json(HERE / "data/manifest.json")
    for stage_id in stages or load_recipe()["stages"]:
        spec = stage(root, stage_id)
        folder = stage_dir(root, stage_id)
        if spec["kind"] == "sft":
            import projection

            projection.prepare(root, stage_id)
            continue
        if (folder / "metrics.jsonl").exists():
            raise ValueError(
                "Refusing to rewrite a stage that already has training metrics"
            )
        selected = (
            regenerate_futoshiki(root, stage_id)
            if data_mode == "regenerate" and spec["environment"] == "futoshiki"
            else materialize_asset(root, stage_id, data)
        )
        write_json(
            folder / "stage.json",
            dict(
                specification=spec,
                data=selected,
                recipe_sha256=identity["recipe_sha256"],
            ),
        )
        (folder / "train.toml").write_text(emit_toml(config_for(root, stage_id)))
    return root


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--initial-model", required=True)
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        help="stage_id=existing local policy directory; no weights copied",
    )
    parser.add_argument(
        "--data-mode", choices=["historical", "regenerate"], default="historical"
    )
    parser.add_argument("--stage", action="append", dest="stages")
    args = parser.parse_args(argv)
    artifacts = {}
    for item in args.artifact:
        key, sep, value = item.partition("=")
        if not sep or key not in load_recipe()["stages"]:
            parser.error("--artifact must name a recipe stage and a local policy")
        artifacts[key] = value
    prepare(
        args.root,
        args.initial_model,
        artifacts=artifacts,
        data_mode=args.data_mode,
        stages=args.stages,
    )
    print(str(args.root.resolve()))


if __name__ == "__main__":
    main()
