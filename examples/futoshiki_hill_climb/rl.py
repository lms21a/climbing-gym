"""Native historical hill-climb collection, paired learner updates, and resume.

Unlike the generic one-update driver, a 256-response cohort here supplies two
ordered 128-response updates at policy lags zero and one. Every checkpoint owns
its pending CPU cohort, so restart never recollects or drops its second half.
"""

from __future__ import annotations
import argparse
from dataclasses import replace
import json
import hashlib
import math
from pathlib import Path
import random
from types import SimpleNamespace
import time

import torch
from climbing_gym.config import ConfigManager
from climbing_gym.grpo import _build_rollout_engine
from climbing_gym.models import build_model
from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer
from climbing_gym.cpu_master_adamw import CPUStateMasterAdamW
from climbing_gym.optimizer_resume import restore_master_optimizer
from climbing_gym.preflight import validate_preflight
from climbing_gym.rollout import Trajectory, TrajectoryTurn
from climbing_gym.inference.lfm25_engine import LFM25SamplingConfig
from climbing_gym.trainers.checkpoints import CheckpointManager, restore_training_state
from climbing_gym.trainers.common import offload_to_cpu, reload_to_device
from climbing_gym.trainers.logger import WandbLogger
from climbing_gym.train import update_policy, validate_config
from climbing_gym.sampling import write_traces
from recipe import (
    HERE,
    stage,
    stage_dir,
    checkpoint_path,
    model_path,
    read_json,
    write_json,
    read_rows,
    digest,
)
from tasks import HistoricalJointEnv

SEED_STRIDE = 0x517CC1B727220A95


def append(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as stream:
        stream.write(json.dumps(value, allow_nan=False) + "\n")


def task_env(spec, tokenizer):
    if spec["reward"] == "historical_joint":
        return HistoricalJointEnv(spec["environment"], tokenizer)
    from climbing_gym.config import Env

    return Env(name="futoshiki", reward="exact_match_strict").build()


def prompts_for(env, rows, tokenizer, spec):
    texts = []
    tokens = []
    for row in rows:
        text = env.build_prompt(row)
        if spec["reward"] != "historical_joint":
            text = (
                tokenizer.apply_chat_template(
                    [dict(role="user", content=text)],
                    tokenize=False,
                    add_generation_prompt=True,
                    enable_thinking=False,
                )
                + "<thinking>\n"
            )
        ids = tuple(tokenizer.encode(text, add_special_tokens=False))
        if len(ids) + spec["max_tokens"] > spec["context"]:
            raise ValueError("Prompt plus complete response exceeds context")
        texts.append(text)
        tokens.append(ids)
    return texts, tokens


@torch.inference_mode()
def collect_native(run, rows, *, start, behavior_version):
    """Port of the frozen native session sampler; every live slot is explicit."""
    spec = run.spec
    _, prompts = prompts_for(run.env, rows, run.tokenizer, spec)
    head = 32 if spec["objective"] == "mis_sc" else 0
    capacity = spec["cohort_size"]
    offset = spec.get("sampling_offset", 0) + start
    seed = (spec["seed"] + offset * SEED_STRIDE) & 0x7FFFFFFFFFFFFFFF
    eos = run.tokenizer.eos_token_id
    sampling = [
        LFM25SamplingConfig(
            temperature=1.0,
            max_new_tokens=spec["max_tokens"],
            stop_token_ids=(eos,),
            seed=(seed + i * SEED_STRIDE) & 0x7FFFFFFFFFFFFFFF,
            return_logprobs=True,
            behavior_topk=head,
        )
        for i in range(len(prompts))
    ]
    output = [None] * len(rows)
    queue = iter(range(len(rows)))
    raw = run.engine.raw_engine
    device = raw.device

    def sync():
        if device.type == "cuda":
            torch.cuda.synchronize(device)

    sync()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    began = time.perf_counter()
    with raw.generation_session(
        max_batch_size=capacity, max_new_tokens=spec["max_tokens"], behavior_topk=head
    ) as session:
        initial = [next(queue) for _ in range(min(capacity, len(rows)))]
        slots = session.admit(
            [prompts[i] for i in initial], [sampling[i] for i in initial]
        )
        live = dict(zip(slots, initial))
        while live:
            fresh = []
            for result in session.poll(steps=32):
                slot = result.handle.slot
                index = live.pop(slot)
                output[index] = result
                session.release(result.handle)
                following = next(queue, None)
                if following is not None:
                    fresh.append((slot, following))
            if fresh:
                session.admit(
                    [prompts[i] for _, i in fresh],
                    [sampling[i] for _, i in fresh],
                    slots=[s for s, _ in fresh],
                )
                live.update(fresh)
        stats = dict(session.stats)
    sync()
    seconds = time.perf_counter() - began
    traces = []
    for i, (row, prompt, result) in enumerate(zip(rows, prompts, output, strict=True)):
        ids = tuple(result.token_ids)
        lp = tuple(result.logprobs)
        if (
            not ids
            or len(ids) != len(lp)
            or not all(math.isfinite(p) and p <= 0 for p in lp)
        ):
            raise ValueError(
                "Native collection must retain every actual sampled-token probability"
            )
        response = run.tokenizer.decode(ids, skip_special_tokens=False)
        reason = "length" if result.finish_reason == 3 else "stop"
        example = run.env.build_example(row, dataset_index=start + i)
        evaluation = run.env.evaluate_response(
            example, response, turn_metadata=[{"finish_reason": reason}]
        )
        turn = TrajectoryTurn(
            turn_index=1,
            prompt=run.tokenizer.decode(prompt, skip_special_tokens=False),
            response=response,
            prompt_token_ids=prompt,
            completion_token_ids=ids,
            completion_logprobs=lp,
            finish_reason=reason,
            stop_reason=reason,
            behavior_topk=result.behavior_topk,
        )
        traces.append(
            Trajectory(
                group_id=start + i,
                rollout_index=0,
                example=example,
                turns=(turn,),
                final_response=response,
                finished=True,
                terminal_reason="single_turn",
                evaluation=evaluation,
                metadata={
                    "behavior_policy_version": behavior_version,
                    "learner_policy_version": behavior_version,
                },
            )
        )
    tokens = sum(len(t.turns[0].completion_token_ids) for t in traces)
    stats.update(
        collection_seconds=seconds,
        tokens_per_second=tokens / max(seconds, 1e-9),
        generated_tokens=tokens,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device)
        if device.type == "cuda"
        else 0,
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device)
        if device.type == "cuda"
        else 0,
    )
    return tuple(traces), stats


def restore_optimizer_checkpoint(optimizer, checkpoint):
    """Read current and frozen historical optimizer layouts without BF16 coercion."""
    checkpoint = Path(checkpoint)
    if not (checkpoint / "training_state.pt").exists():
        old = checkpoint / "trainer_state/optimizer.pt"
        if not old.exists():
            raise ValueError(
                "Exact optimizer state is unavailable (policy-only or pruned checkpoint); use a retained state checkpoint"
            )
        saved = torch.load(old, map_location="cpu", weights_only=False)
        restore_master_optimizer(optimizer, saved, require_all=False)
        rng_path = checkpoint / "trainer_state/rng.pt"
        if rng_path.exists():
            rng = torch.load(rng_path, map_location="cpu", weights_only=False)
            if "torch" in rng:
                torch.set_rng_state(rng["torch"])
            if "python" in rng:
                random.setstate(rng["python"])
            if torch.cuda.is_available() and rng.get("cuda"):
                torch.cuda.set_rng_state_all(rng["cuda"])
        return
    # Our checkpoint layout includes the version key; original expert scripts
    # use optimizer/torch_rng/python_rng and no version. Input is trusted local.
    saved = torch.load(
        checkpoint / "training_state.pt", map_location="cpu", weights_only=False
    )
    if saved.get("version") == 1:
        restore_training_state(checkpoint, policy_optimizer=optimizer)
    elif "optimizer" in saved:
        restore_master_optimizer(optimizer, saved["optimizer"], require_all=False)
        torch.set_rng_state(saved["torch_rng"])
        random.setstate(saved["python_rng"])
        if torch.cuda.is_available() and saved.get("cuda_rng"):
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    else:
        raise ValueError("Unknown exact optimizer checkpoint layout")


def configuration_sha256(config):
    encoded = json.dumps(
        config.to_dict(), sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    return hashlib.sha256(encoded.encode()).hexdigest()


def policy_manifest(checkpoint):
    policy = Path(checkpoint) / "policy"
    return {
        str(path.relative_to(policy)): digest(path)
        for path in sorted(policy.rglob("*"))
        if path.is_file()
    }


def verify_saved_identity(config, checkpoint):
    state = read_json(Path(checkpoint) / "hill_state.json")
    if state.get("configuration_sha256") != configuration_sha256(config):
        raise ValueError(
            "Resolved training configuration differs from the resume checkpoint"
        )
    if state.get("policy_sha256") != policy_manifest(checkpoint):
        raise ValueError("Policy or tokenizer bytes differ from the resume checkpoint")


def load_run(
    root, stage_id, resume=None, *, evaluation=False, evaluation_panel="monitor"
):
    spec = stage(root, stage_id)
    folder = stage_dir(root, stage_id)
    if read_json(Path(root) / "manifest.json")["recipe_sha256"] != digest(
        HERE / "recipe.json"
    ):
        raise ValueError(
            "Prepared recipe differs from the current recipe; use a separate prepared run"
        )
    if spec["kind"] != "rl":
        raise ValueError("Use projection.py for SFT stages")
    cfg = ConfigManager.from_toml(folder / "train.toml").config
    training_sha256 = digest(folder / "data/train.jsonl")
    recorded = read_json(folder / "data/manifest.json")["splits"]["train"]["sha256"]
    if training_sha256 != recorded:
        raise ValueError("Prepared ordered training bank checksum differs")
    validate_config(cfg)
    source = (
        Path(resume) / "policy" if resume else model_path(root, spec["input_policy"])
    )
    if not source.is_dir():
        raise FileNotFoundError(
            f"Run the prerequisite or register its artifact: {source}"
        )
    if evaluation:
        if resume is None:
            raise ValueError("Standalone evaluation requires an actual checkpoint")
        profile = spec.get("evaluation", {}).get(evaluation_panel)
        if profile is None:
            raise ValueError("No historical profile for this evaluation panel")
        inference = replace(
            cfg.inference,
            model_path=str(source),
            max_model_len=profile["context"],
            sampling=replace(cfg.inference.sampling, max_tokens=profile["max_tokens"]),
            preflight_report_path=str(
                checkpoint_report(folder, resume, evaluation_panel)
            ),
            preflight_prior_report_path="",
        )
        cfg = replace(cfg, inference=inference)
        preflight = validate_preflight(cfg)
        evaluation_preflight = {evaluation_panel: preflight}
    else:
        preflight = validate_preflight(cfg)
        evaluation_preflight = validate_evaluation_reports(cfg, spec, folder)
    if resume and not evaluation:
        verify_saved_identity(cfg, resume)
    tokenizer = load_lfm2_tokenizer(source)
    policy = build_model("lfm2", model_dir=str(source), **dict(cfg.model.config)).to(
        "cpu", torch.bfloat16
    )
    optimizer = CPUStateMasterAdamW(
        policy.parameters(),
        lr=spec["learning_rate"],
        weight_decay=0.0,
        pin_state=False,
        chunk_elements=cfg.trainer.optimizer_chunk_elements,
        workspace_bytes=cfg.trainer.optimizer_workspace_bytes,
    )
    if resume and not evaluation:
        restore_optimizer_checkpoint(optimizer, resume)
    elif not evaluation and spec["optimizer_transfer"]:
        restore_optimizer_checkpoint(
            optimizer, model_path(root, spec["input_policy"]).parent
        )
    inference = replace(cfg.inference, model_path=str(source))
    # One frozen native actor and the BF16 learner; only KV/graphs are released
    # during learner phases. CPU moments use a bounded reusable staging budget.
    engine = _build_rollout_engine(replace(cfg, inference=inference))
    try:
        logger = WandbLogger(
            cfg.wandb,
            config_path=folder / "train.toml",
            resolved_config={
                **cfg.to_dict(),
                "hill_climb": spec,
                "preflight": preflight,
                "evaluation_preflight": evaluation_preflight,
                "transition": {
                    "optimizer_transfer": spec["optimizer_transfer"],
                    "dataset_reset": not bool(resume),
                    "input_policy": str(source),
                },
            },
        )
        # Updaters see each 128-row learner batch, never the cohort's 256-row
        # collection geometry. Centering and response normalization stay local.
        learner_cfg = replace(
            cfg,
            rollout=replace(
                cfg.rollout,
                batch_size=spec["batch_size"],
                effective_batch_size=spec["batch_size"],
                max_total_requests=spec["batch_size"],
            ),
        )
        return SimpleNamespace(
            config=learner_cfg,
            collection_config=cfg,
            training_sha256=training_sha256,
            engine_context=inference.max_model_len,
            spec=spec,
            policy_model=policy,
            policy_optimizer=optimizer,
            value_model=None,
            engine=engine,
            tokenizer=tokenizer,
            env=task_env(spec, tokenizer),
            wandb_logger=logger,
            folder=folder,
            checkpoint_manager=CheckpointManager(cfg.checkpoint),
            release_inference_buffers_before_update=True,
        )
    except BaseException:
        engine.close()
        raise


def prune_optimizer_states(run, *, reserve_new=False):
    keep = run.spec.get("keep_optimizer_checkpoints", 2) - int(reserve_new)
    keep = max(1, keep)
    candidates = sorted(
        (run.folder / "checkpoints").glob("step_*/training_state.pt"),
        key=lambda path: int(path.parent.name.split("_")[-1]),
    )
    pinned = run.spec.get("retained_updates")
    for path in candidates[:-keep]:
        if pinned and path.parent.name == f"step_{pinned - 1}":
            continue
        path.unlink()
        state = path.parent / "cohort_state.pt"
        if state.exists():
            state.unlink()


def prune_cohorts(run):
    keep = run.spec.get("keep_raw_cohorts", 12)
    archives = sorted((run.folder / "cohorts").glob("cohort_*.pt"))
    for path in archives[:-keep] if keep else archives:
        path.unlink()


def save_checkpoint(run, *, completed, next_index, pending, behavior_version):
    final = run.folder / "checkpoints" / f"step_{completed - 1}"
    if final.exists():
        raise FileExistsError(f"Checkpoint already exists: {final}")
    prune_optimizer_states(run, reserve_new=True)
    manager = CheckpointManager(
        replace(
            run.checkpoint_manager.config, root_dir=str(run.folder / ".checkpoint_work")
        )
    )
    destination = manager.save(
        step=completed - 1,
        policy_model=run.policy_model,
        policy_optimizer=run.policy_optimizer,
    )
    if destination is None:
        return None
    run.tokenizer.save_pretrained(destination / "policy")
    # Raw trajectory metadata is deliberately CPU-owned, including top-32
    # distributions. This is a resumable state artifact, not model executable code.
    torch.save(
        {"pending": tuple(pending), "behavior_version": behavior_version},
        destination / "cohort_state.pt",
    )
    write_json(
        destination / "hill_state.json",
        dict(
            version=1,
            completed_updates=completed,
            next_dataset_index=next_index,
            training_sha256=run.training_sha256
            if hasattr(run, "training_sha256")
            else digest(run.folder / "data/train.jsonl"),
            stage_id=run.spec["id"],
            recipe_sha256=digest(HERE / "recipe.json"),
            cohort_sha256=digest(destination / "cohort_state.pt"),
            configuration_sha256=configuration_sha256(
                getattr(run, "collection_config", run.config)
            ),
            policy_sha256=policy_manifest(destination),
        ),
    )
    metadata = read_json(destination / "metadata.json")
    steps = sorted({int(s.get("step", 0)) for s in run.policy_optimizer.state.values()})
    metadata.update(
        completed_stage_updates=completed,
        cumulative_optimizer_updates=steps,
        pending_cohort_responses=len(pending),
        next_dataset_index=next_index,
    )
    write_json(destination / "metadata.json", metadata)
    final.parent.mkdir(parents=True, exist_ok=True)
    destination.replace(final)
    return final


def resume_state(run, checkpoint):
    verify_saved_identity(getattr(run, "collection_config", run.config), checkpoint)
    state = read_json(Path(checkpoint) / "hill_state.json")
    if (
        state["version"] != 1
        or state["stage_id"] != run.spec["id"]
        or state["training_sha256"] != digest(run.folder / "data/train.jsonl")
        or state["recipe_sha256"] != digest(HERE / "recipe.json")
    ):
        raise ValueError("Resume stage, recipe or exact ordered dataset differs")
    path = Path(checkpoint) / "cohort_state.pt"
    if digest(path) != state["cohort_sha256"]:
        raise ValueError("Pending cohort checksum differs")
    pending = torch.load(path, map_location="cpu", weights_only=False)
    return (
        state["completed_updates"],
        state["next_dataset_index"],
        tuple(pending["pending"]),
        pending["behavior_version"],
    )


def checkpoint_report(folder, checkpoint, panel):
    return Path(folder) / "preflight" / f"evaluate_{Path(checkpoint).name}_{panel}.json"


def execution_workloads(spec, folder):
    """Distinct startup benchmark budgets, including larger endpoint tests."""
    workloads = {(spec["max_tokens"], spec["context"]): "train"}
    for panel, profile in spec.get("evaluation", {}).items():
        if (folder / "data" / f"{panel}.jsonl").exists():
            workloads.setdefault((profile["max_tokens"], profile["context"]), panel)
    return workloads


def validate_evaluation_reports(config, spec, folder):
    reports = {}
    for (budget, context), panel in execution_workloads(spec, folder).items():
        if panel == "train":
            continue
        inference = replace(
            config.inference,
            max_model_len=context,
            sampling=replace(config.inference.sampling, max_tokens=budget),
            preflight_report_path=str(
                folder / "preflight" / f"eval_{budget}_{context}.json"
            ),
            preflight_prior_report_path="",
        )
        reports[panel] = validate_preflight(replace(config, inference=inference))
    return reports


def ensure_context(run, context):
    if getattr(run, "engine_context", run.spec["context"]) == context:
        return
    # Endpoint tests can use a larger context than training. Retire the old
    # actor before allocating the next; the learner is already on the CPU.
    run.engine.close()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    inference = replace(run.collection_config.inference, max_model_len=context)
    run.engine = _build_rollout_engine(
        replace(run.collection_config, inference=inference)
    )
    run.engine.sync_from_model(run.policy_model, drain_timeout=1200)
    run.engine_context = context


def evaluate_panel(run, panel, *, completed, output=None):
    path = run.folder / "data" / f"{panel}.jsonl"
    profile = run.spec.get("evaluation", {}).get(panel)
    if profile is None:
        raise ValueError(f"No historical evaluation profile for panel {panel!r}")
    rows = read_rows(path)[: profile["max_examples"]]
    repeats = profile["repeats"]
    ensure_context(run, profile["context"])
    expanded = [r for r in rows for _ in range(repeats)]
    seed = 20261001 + run.spec.get("evaluation_seed_offsets", {}).get(panel, 0)
    evaluation_run = SimpleNamespace(**vars(run))
    evaluation_run.spec = dict(
        run.spec,
        seed=seed,
        sampling_offset=0,
        max_tokens=profile["max_tokens"],
        context=profile["context"],
    )
    traces, stats = collect_native(
        evaluation_run, expanded, start=0, behavior_version=completed
    )
    good = [bool(t.evaluation.reward.reward) for t in traces]
    p4 = []
    p8 = []
    for index in range(0, len(good), repeats):
        successes = sum(good[index : index + repeats])
        k = min(4, repeats)
        p4.append(
            1 - math.comb(repeats - successes, k) / math.comb(repeats, k)
            if repeats - successes >= k
            else 1.0
        )
        if repeats >= 8:
            p8.append(float(successes > 0))
    summary = dict(
        stats,
        problems=len(rows),
        samples=len(traces),
        pass_at_1=sum(good) / len(good),
        pass_at_4=sum(p4) / len(p4),
        max_tokens=profile["max_tokens"],
        checkpoint_stage_update=completed,
        mean_tokens=sum(len(t.turns[0].completion_token_ids) for t in traces)
        / len(traces),
        cap_hit_rate=sum(t.turns[0].finish_reason == "length" for t in traces)
        / len(traces),
        panel=panel,
    )
    if repeats >= 8:
        summary["pass_at_8"] = sum(p8) / len(p8)
    destination = (
        Path(output) if output else run.folder / "eval" / f"update_{completed}"
    )
    write_json(destination / f"{panel}.json", summary)
    records = []
    for index, t in enumerate(traces):
        record = dict(
            id=str(t.example.row.get("id", index)),
            tier=t.example.row.get("tier", "unspecified"),
            prompt=t.turns[0].prompt,
            response=t.final_response,
            expected_answer=t.example.target,
            outcome="success" if good[index] else "failure",
        )
        append(destination / f"{panel}.trajectories.jsonl", record)
        records.append(record)
    # Balanced trace inspection over available tiers/outcomes, complete responses.
    chosen = []
    seen = set()
    for record in records:
        key = (record["tier"], record["outcome"])
        if key not in seen:
            chosen.append(record)
            seen.add(key)
    write_traces(destination / f"{panel}.traces.md", chosen[:8])
    return summary


def evaluate_panels(run, completed, *, final=False):
    summaries = {}
    for panel, profile in run.spec.get("evaluation", {}).items():
        if profile.get("final_only") and not final:
            continue
        if (run.folder / "data" / f"{panel}.jsonl").exists():
            summaries[panel] = evaluate_panel(run, panel, completed=completed)
    panels = [
        v for k, v in summaries.items() if k in {"expert", "next", "harder", "hardest"}
    ]
    if panels:
        samples = sum(v["samples"] for v in panels)
        pooled = dict(
            samples=samples,
            problems=sum(v["problems"] for v in panels),
            pass_at_1=sum(v["samples"] * v["pass_at_1"] for v in panels) / samples,
        )
        write_json(run.folder / "eval" / f"update_{completed}" / "pooled.json", pooled)
        summaries["pooled"] = pooled
    return {
        f"eval/{panel}/{key}": float(value)
        for panel, values in summaries.items()
        for key, value in values.items()
        if isinstance(value, (int, float))
    }


def train_loop(
    run,
    rows,
    *,
    completed=0,
    next_index=0,
    pending=(),
    behavior_version=0,
    evaluations=True,
):
    spec = run.spec
    batch = spec["batch_size"]
    pending = tuple(pending)
    if spec["cohort_size"] % batch:
        raise ValueError("Cohort must contain complete fixed learner batches")
    if evaluations and completed == 0:
        run.wandb_logger.log_step(step=0, metrics=evaluate_panels(run, 0))
    while completed < spec["steps"]:
        began = time.perf_counter()
        rollout_stats = {}
        if not pending:
            width = min(spec["cohort_size"], (spec["steps"] - completed) * batch)
            selected = [rows[(next_index + i) % len(rows)] for i in range(width)]
            offload_to_cpu(run.policy_model, run.policy_optimizer)
            run.engine.sync_from_model(run.policy_model, drain_timeout=1200)
            behavior_version = completed
            pending, rollout_stats = collect_native(
                run, selected, start=next_index, behavior_version=behavior_version
            )
            next_index += width
            archive = run.folder / "cohorts" / f"cohort_{completed:06d}.pt"
            archive.parent.mkdir(parents=True, exist_ok=True)
            torch.save(
                {
                    "trajectories": pending,
                    "stats": rollout_stats,
                    "behavior_version": behavior_version,
                },
                archive,
            )
            prune_cohorts(run)
            run.engine.release_inference_buffers()
        selected, pending = pending[:batch], pending[batch:]
        if len(selected) != batch:
            raise ValueError("Incomplete pending learner batch; no optimizer update")
        lag = completed - behavior_version
        if lag not in range(spec["cohort_size"] // batch):
            raise ValueError("Invalid cohort policy lag")
        selected = tuple(
            replace(t, metadata={**t.metadata, "learner_policy_version": completed})
            for t in selected
        )
        reload_to_device(
            run.policy_model,
            run.policy_optimizer,
            device=run.config.trainer.train_device,
            dtype=run.config.trainer.train_dtype,
        )
        learned = time.perf_counter()
        metrics, audit = update_policy(run, selected)
        learn_seconds = time.perf_counter() - learned
        completed += 1
        metrics.update(
            {
                "hill_climb/policy_lag": lag,
                "flashreinforce/behavior_policy_lag": lag,
                "hill_climb/completed_stage_updates": completed,
                "hill_climb/pending_responses": len(pending),
                "trainer/learner_seconds": learn_seconds,
                "trainer/rollout_tokens_per_second": rollout_stats.get(
                    "tokens_per_second", 0.0
                ),
                "trainer/rollout_generated_tokens": rollout_stats.get(
                    "generated_tokens", 0.0
                ),
                "hill_climb/reused_pending_cohort": float(not rollout_stats),
            }
        )
        # Per-response credit audits expose original behavior, unmodified rewards,
        # centered advantages, actual lengths, and exactly one use of each row.
        append(
            run.folder / "audit.jsonl",
            dict(
                update=completed,
                behavior_version=behavior_version,
                policy_lag=lag,
                rows=audit,
            ),
        )
        if evaluations and (
            completed % spec["eval_interval"] == 0 or completed == spec["steps"]
        ):
            offload_to_cpu(run.policy_model, run.policy_optimizer)
            run.engine.sync_from_model(run.policy_model, drain_timeout=1200)
            metrics.update(
                evaluate_panels(run, completed, final=completed == spec["steps"])
            )
            run.engine.release_inference_buffers()
        if (
            completed % spec["checkpoint_interval"] == 0
            or completed == spec["steps"]
            or completed == spec.get("retained_updates")
        ):
            save_checkpoint(
                run,
                completed=completed,
                next_index=next_index,
                pending=pending,
                behavior_version=behavior_version,
            )
        metrics["trainer/full_step_time_seconds"] = time.perf_counter() - began
        append(run.folder / "metrics.jsonl", dict(update=completed, metrics=metrics))
        run.wandb_logger.log_step(step=completed, metrics=metrics)
        write_json(
            run.folder / "status.json",
            dict(
                state="complete" if completed == spec["steps"] else "training",
                completed_updates=completed,
                target_updates=spec["steps"],
                next_dataset_index=next_index,
                pending_responses=len(pending),
            ),
        )
        print(
            json.dumps({"stage": spec["id"], "update": completed, "metrics": metrics}),
            flush=True,
        )
    return completed


def preflight(root, stage_id, prior=None, *, checkpoint=None, panel=None):
    from climbing_gym.inference.benchmark import main as benchmark

    spec = stage(root, stage_id)
    folder = stage_dir(root, stage_id)
    if read_json(Path(root) / "manifest.json")["recipe_sha256"] != digest(
        HERE / "recipe.json"
    ):
        raise ValueError(
            "Prepared recipe differs from the current recipe; use a separate prepared run"
        )
    source = (
        Path(checkpoint) / "policy"
        if checkpoint
        else model_path(root, spec["input_policy"])
    )
    tokenizer = load_lfm2_tokenizer(source)
    env = task_env(spec, tokenizer)
    workloads = execution_workloads(spec, folder)
    if checkpoint:
        panel = panel or "monitor"
        profile = spec.get("evaluation", {}).get(panel)
        if profile is None:
            raise ValueError("No historical profile for this evaluation panel")
        workloads = {(profile["max_tokens"], profile["context"]): panel}
    for (budget, context), panel in workloads.items():
        profile = spec.get("evaluation", {}).get(panel, {})
        rows = read_rows(folder / "data" / f"{panel}.jsonl")
        repeats = profile.get("repeats", 1)
        rows = [row for row in rows for _ in range(repeats)][: spec["cohort_size"]]
        texts, tokens = prompts_for(
            env, rows, tokenizer, dict(spec, max_tokens=budget, context=context)
        )
        prompts = folder / "preflight" / f"prompts_{budget}_{context}.json"
        write_json(prompts, {"prompts": tokens, "texts": texts})
        output = (
            folder
            / "preflight"
            / ("train.json" if panel == "train" else f"eval_{budget}_{context}.json")
        )
        if checkpoint:
            output = checkpoint_report(folder, checkpoint, panel)
        args = [
            "--model",
            str(source),
            "--batches",
            str(spec["cohort_size"]),
            "--output-tokens",
            str(budget),
            "--max-model-len",
            str(context),
            "--prompts-json",
            str(prompts),
            "--dtype",
            "bfloat16",
            "--temperature",
            "1",
            "--seed",
            str(spec["seed"]),
            "--cuda-graphs",
            "--park-gib",
            "2.5",
            "--output",
            str(output),
        ]
        args += (
            [
                "--fp8-mlp",
                "--fp8-kv",
                "--fp8-lm-head",
                "--adaptive-batching",
                "--behavior-topk",
                "32",
            ]
            if spec["fp8"]
            else ["--bf16-weight-packing"]
        )
        benchmark(args)
    # Store the prior path alongside the measured report. Training validates
    # matched native settings and logs the comparison once, before model load.
    if prior and not checkpoint:
        from prepare import emit_toml

        cfg = (
            read_json(folder / "generated_config.json")
            if (folder / "generated_config.json").exists()
            else None
        )
        import tomllib

        cfg = tomllib.loads((folder / "train.toml").read_text())
        cfg["inference"]["preflight_prior_report_path"] = str(Path(prior).resolve())
        (folder / "train.toml").write_text(emit_toml(cfg))
    return output if checkpoint else folder / "preflight/train.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=["preflight", "train", "evaluate", "status"])
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage")
    parser.add_argument("--prior", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--panel", default="monitor")
    args = parser.parse_args(argv)
    if args.command == "status":
        for stage_id, specification in read_json(HERE / "recipe.json")[
            "stages"
        ].items():
            folder = stage_dir(args.root, stage_id)
            if not folder.exists():
                continue
            status = (
                read_json(folder / "status.json")
                if (folder / "status.json").exists()
                else {"state": "pending"}
            )
            prerequisites = [
                specification["input_policy"],
                *specification.get("experts", {}).values(),
            ]
            status["prerequisites"] = {
                reference: {
                    "policy": str(model_path(args.root, reference)),
                    "available": model_path(args.root, reference).is_dir(),
                }
                for reference in prerequisites
            }
            selected = checkpoint_path(args.root, stage_id)
            status["retained_checkpoint"] = str(selected)
            status["retained_optimizer_available"] = (
                selected / "training_state.pt"
            ).exists()
            status["exact_resume_checkpoints"] = [
                str(path.parent)
                for path in sorted(
                    (folder / "checkpoints").glob("step_*/training_state.pt")
                )
                if (path.parent / "hill_state.json").exists()
                and (path.parent / "cohort_state.pt").exists()
            ]
            status["ready"] = all(
                item["available"] for item in status["prerequisites"].values()
            )
            if specification.get("optimizer_transfer"):
                parent = model_path(args.root, specification["input_policy"]).parent
                state_available = (parent / "training_state.pt").exists() or (
                    parent / "trainer_state/optimizer.pt"
                ).exists()
                status["input_optimizer_available"] = state_available
                status["ready"] = status["ready"] and state_available
            print(stage_id, json.dumps(status))
        return 0
    if not args.stage:
        parser.error("--stage is required")
    if args.command == "preflight":
        preflight(
            args.root,
            args.stage,
            args.prior,
            checkpoint=args.checkpoint,
            panel=args.panel,
        )
        return 0
    if args.command == "evaluate":
        if not args.checkpoint:
            parser.error("--checkpoint is required for evaluate")
        run = load_run(
            args.root,
            args.stage,
            args.checkpoint,
            evaluation=True,
            evaluation_panel=args.panel,
        )
        try:
            summary = evaluate_panel(
                run,
                args.panel,
                completed=read_json(args.checkpoint / "metadata.json").get(
                    "completed_stage_updates", 0
                ),
            )
            print(json.dumps(summary))
            return 0
        finally:
            run.wandb_logger.close()
            run.engine.close()
    run = load_run(args.root, args.stage, args.resume)
    try:
        if not args.resume and (run.folder / "metrics.jsonl").exists():
            raise ValueError("Stage already launched; resume an explicit checkpoint")
        state = resume_state(run, args.resume) if args.resume else (0, 0, (), 0)
        train_loop(
            run,
            read_rows(run.folder / "data/train.jsonl"),
            completed=state[0],
            next_index=state[1],
            pending=state[2],
            behavior_version=state[3],
        )
        return 0
    finally:
        run.wandb_logger.close()
        run.engine.close()


if __name__ == "__main__":
    raise SystemExit(main())
