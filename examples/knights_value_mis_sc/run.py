"""Explicit native preflight, training/resume, and held-out evaluation commands."""

from __future__ import annotations

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tomllib
import random

import torch

from climbing_gym.config import ConfigManager, Rollout
from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer
from climbing_gym.preflight import validate_preflight
from knights_recipe import digest, write_toml, retain_checkpoints


def policy_path(checkpoint):
    path = Path(checkpoint).resolve()
    return path / "policy" if (path / "policy").is_dir() else path


def model_key(model):
    return hashlib.sha256(str(Path(model).resolve()).encode()).hexdigest()[:12]


def report_path(root, model, panel):
    return Path(root) / "preflight" / f"{panel}.{model_key(model)}.json"


def prompt_ids(rows, tokenizer):
    prompts = []
    for row in rows:
        text = (
            tokenizer.apply_chat_template(
                [dict(role="user", content=row["prompt"])],
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            + "<thinking>\n"
        )
        tokens = tokenizer.encode(text, add_special_tokens=False)
        expected = row.get("source_prompt_token_ids_sha256")
        if (
            expected
            and digest(json.dumps(tokens, separators=(",", ":")).encode()) != expected
        ):
            raise ValueError(
                f"Tokenizer changed the historical public prompt: {row['id']}"
            )
        prompts.append(tokens)
    return prompts


def preflight(root, *, checkpoint=None, panel="all", prior=None):
    root = Path(root).resolve()
    config = ConfigManager.from_toml(root / "train.toml").config
    model = policy_path(checkpoint or config.model.path)
    tokenizer = load_lfm2_tokenizer(model)
    reports = {}
    for split in ("train", "dev", "test") if panel == "all" else (panel,):
        rows = [
            json.loads(line)
            for line in (root / "data" / f"{split}.jsonl").read_text().splitlines()
        ]
        prompts = prompt_ids(rows, tokenizer)
        prompts_file = root / "preflight" / f"prompts_{split}.json"
        prompts_file.write_text(json.dumps({"prompts": prompts}) + "\n")
        output = report_path(root, model, split)
        args = [
            sys.executable,
            "-m",
            "climbing_gym.inference.benchmark",
            "--model",
            str(model),
            "--prompts-json",
            str(prompts_file),
            "--batches",
            str(config.inference.native_max_batch_size),
            "--output-tokens",
            str(config.inference.sampling.max_tokens),
            "--max-model-len",
            str(config.inference.max_model_len),
            "--dtype",
            config.inference.dtype,
            "--device",
            "cuda",
            "--temperature",
            "1",
            "--top-k",
            "-1",
            "--top-p",
            "1",
            "--behavior-topk",
            "32",
            "--seed",
            str(config.inference.seed),
            "--max-prefill-tokens",
            str(config.inference.native_prefill_tokens),
            "--output",
            str(output),
        ]
        for key, flag in (
            ("native_cuda_graphs", "--cuda-graphs"),
            ("native_bf16_weight_packing", "--bf16-weight-packing"),
            ("native_fp8_mlp", "--fp8-mlp"),
            ("native_fp8_kv", "--fp8-kv"),
            ("native_fp8_lm_head", "--fp8-lm-head"),
            ("native_adaptive_batching", "--adaptive-batching"),
        ):
            if getattr(config.inference, key):
                args.append(flag)
        subprocess.run(args, check=True)
        rollout = (
            config.rollout
            if split == "train"
            else Rollout(
                **tomllib.loads((root / f"eval_{split}.toml").read_text())["rollout"]
            )
        )
        view = replace(
            config,
            inference=replace(
                config.inference,
                model_path=str(model),
                preflight_report_path=str(output),
                preflight_prior_report_path=str(Path(prior).resolve()) if prior else "",
            ),
            rollout=rollout,
        )
        reports[split] = validate_preflight(view)
    selection_path = root / "preflight" / f"selection.{model_key(model)}.json"
    selection = (
        json.loads(selection_path.read_text()) if selection_path.exists() else {}
    )
    selection.update(
        {
            split: {
                "report": str(report_path(root, model, split)),
                "prior": str(Path(prior).resolve()) if prior else "",
            }
            for split in reports
        }
    )
    selection_path.write_text(json.dumps(selection, indent=2) + "\n")
    print(json.dumps({"model": str(model), "reports": reports}, indent=2))


def selected_report(root, model, panel):
    path = root / "preflight" / f"selection.{model_key(model)}.json"
    if not path.is_file() or panel not in json.loads(path.read_text()):
        raise ValueError(
            f"Run preflight for this checkpoint and {panel} panel before launching"
        )
    return json.loads(path.read_text())[panel]


def training(root, *, resume=None):
    from climbing_gym.train import load_run_from_toml, run_training

    root = Path(root).resolve()
    cfg = tomllib.loads((root / "train.toml").read_text())
    model = policy_path(resume or cfg["model"]["path"])
    train_report, dev_report = (
        selected_report(root, model, split) for split in ("train", "dev")
    )
    cfg["inference"].update(
        model_path=str(model),
        preflight_report_path=train_report["report"],
        preflight_prior_report_path=train_report["prior"],
    )
    cfg["eval"].update(
        preflight_report_path=dev_report["report"],
        preflight_prior_report_path=dev_report["prior"],
    )
    if resume:
        checkpoint = Path(resume).resolve()
        metadata = json.loads((checkpoint / "metadata.json").read_text())
        completed = int(metadata["step"]) + 1
        cfg["trainer"]["num_steps"] -= completed
        if cfg["trainer"]["num_steps"] < 0:
            raise ValueError(
                "Checkpoint is beyond this profile's configured actor update target"
            )
        cfg["resume"] = dict(
            checkpoint_dir=str(checkpoint),
            continue_step_numbers=True,
            allow_weights_only=False,
        )
    path = root / "resolved_train.toml"
    write_toml(path, cfg)
    random.seed(cfg["trainer"]["seed"])
    torch.manual_seed(cfg["trainer"]["seed"])
    final_updates = tomllib.loads((root / "train.toml").read_text())["trainer"][
        "num_steps"
    ]
    return run_training(
        load_run_from_toml(path),
        observe_step=lambda payload: retain_checkpoints(
            root, payload, final_updates=final_updates
        ),
    )


def evaluate(root, *, checkpoint, panel):
    from climbing_gym.eval import main

    root = Path(root).resolve()
    model = policy_path(checkpoint)
    selected = selected_report(root, model, panel)
    cfg = tomllib.loads((root / f"eval_{panel}.toml").read_text())
    cfg = deepcopy(cfg)
    cfg["model"]["path"] = str(model)
    cfg["inference"].update(
        model_path=str(model),
        preflight_report_path=selected["report"],
        preflight_prior_report_path=selected["prior"],
    )
    folder = root / "eval" / model_key(model) / panel
    cfg["logging"].update(
        summary_path=str(folder / "summary.json"),
        trajectory_path=str(folder / "trajectories.jsonl"),
        trace_path=str(folder / "traces.md"),
    )
    path = root / f"resolved_eval_{panel}.toml"
    write_toml(path, cfg)
    return main([str(path)])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("preflight", "train", "evaluate"))
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument(
        "--resume", type=Path, help="Exact warmup or actor checkpoint directory"
    )
    parser.add_argument(
        "--panel", choices=("all", "train", "dev", "test"), default="all"
    )
    parser.add_argument(
        "--prior",
        type=Path,
        help="Relevant previous native benchmark report for comparison",
    )
    args = parser.parse_args()
    if args.command == "preflight":
        if args.resume:
            parser.error("Use --checkpoint for preflight of a resumed policy")
        return preflight(
            args.root, checkpoint=args.checkpoint, panel=args.panel, prior=args.prior
        )
    if args.command == "train":
        if args.checkpoint or args.panel != "all" or args.prior:
            parser.error(
                "Train accepts --resume; benchmark checkpoints/panels with preflight first"
            )
        return training(args.root, resume=args.resume)
    if not args.checkpoint or args.panel not in {"dev", "test"}:
        parser.error("Evaluate requires --checkpoint and --panel dev|test")
    return evaluate(args.root, checkpoint=args.checkpoint, panel=args.panel)


if __name__ == "__main__":
    main()
