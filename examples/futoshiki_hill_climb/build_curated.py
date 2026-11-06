"""Maintainer-only exporter of recorded, verified data; never reads model weights.

Accept explicit original run directories and the original tokenizer directory.
Portable consumers use projection.py prepare; they do not need those runs.
"""

import argparse
from collections import Counter
import gzip
import hashlib
import json
from pathlib import Path


def digest(data):
    return hashlib.sha256(data).hexdigest()


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def build(solver_run, reset_run, tokenizer_dir, output):
    output.mkdir(parents=True, exist_ok=True)
    metadata = dict(
        schema_version=1,
        generation_recipe_version="futoshiki-hill-climb-v1",
        tokenizer=dict(
            name="LiquidAI/LFM2.5-1.2B-Instruct",
            eos_token="<|im_end|>",
            eos_token_id=7,
            files={
                p.name: digest(p.read_bytes())
                for p in tokenizer_dir.iterdir()
                if p.name
                in {
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                }
            },
        ),
        corpora={},
    )
    for stage, root, projected, splits in [
        (
            "solver_projection_256",
            solver_run,
            "projected_256.jsonl",
            ("test", "calibration"),
        ),
        (
            "projection_reset",
            reset_run,
            "projected_1024.jsonl",
            ("dev", "final_test", "calibration"),
        ),
    ]:
        guide_path = (
            root
            / "data"
            / (
                "solver_experts.jsonl"
                if stage == "solver_projection_256"
                else "experts_1024.jsonl"
            )
        )
        guides = read(guide_path)
        by_id = {r["id"]: r for r in guides}
        train_path = root / "data" / projected
        training = read(train_path)
        keep = (
            "id",
            "row",
            "prompt_token_ids",
            "completion_token_ids",
            "completion",
            "teacher_sha256",
            "projector_sha256",
            "teacher_seeded",
            "teacher_origin_token_count",
            "teacher_seed_refinements",
        )
        rows = []
        for source in training:
            record = {k: source[k] for k in keep if k in source}
            guide = by_id[source["id"]]
            record.update(
                prompt=guide["prompt"],
                expected_answer=guide["reference_answer"],
                prompt_length=len(source["prompt_token_ids"]),
                completion_length=len(source["completion_token_ids"]),
            )
            rows.append(record)

        def save(name, values, source_path):
            data = "".join(
                json.dumps(
                    r, ensure_ascii=False, separators=(",", ":"), allow_nan=False
                )
                + "\n"
                for r in values
            ).encode()
            encoded = gzip.compress(data, mtime=0)
            filename = f"{stage}.{name}.jsonl.gz"
            (output / filename).write_bytes(encoded)
            metadata["corpora"][filename] = dict(
                stage=stage,
                kind=name,
                rows=len(values),
                source_run=root.name,
                source_file=source_path.name,
                source_sha256=digest(source_path.read_bytes()),
                curated_sha256=digest(data),
                gzip_sha256=digest(encoded),
                bytes=len(data),
                gzip_bytes=len(encoded),
                ordered_ids_sha256=digest("\n".join(r["id"] for r in values).encode()),
                tiers=dict(Counter(r.get("row", r).get("tier") for r in values)),
            )

        save("train_recorded", rows, train_path)
        keep_guides = (
            "id",
            "row",
            "expert",
            "expert_completion_token_ids",
            "reference_answer",
            "prompt",
            "prompt_token_ids",
            "teacher_sha256",
        )
        save(
            "guides_recorded",
            [{k: r[k] for k in keep_guides if k in r} for r in guides],
            guide_path,
        )
        for name in splits:
            source_path = root / f"data/{name}.jsonl"
            save(name, read(source_path), source_path)
        if stage == "projection_reset":
            for tier in ("5x5_g11_i5", "5x5_g9_i5"):
                source_path = root / f"data/pool_{tier}.jsonl"
                save("pool_" + tier, read(source_path), source_path)
    (output / "projection_manifest.json").write_text(
        json.dumps(metadata, indent=2, sort_keys=True) + "\n"
    )
    return metadata


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--solver-run", type=Path, required=True)
    parser.add_argument("--reset-run", type=Path, required=True)
    parser.add_argument("--tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "data")
    args = parser.parse_args()
    build(args.solver_run, args.reset_run, args.tokenizer_dir, args.output)
