"""Recorded-data replay and native historical projection/SFT for the hill climb.

prepare/audit/export run entirely on CPU. preflight/collect/project/train run
real models and must be launched separately after reviewing their setup.
"""

from __future__ import annotations
import argparse
from collections import Counter
from dataclasses import asdict
import gzip
import hashlib
import json
from pathlib import Path
import random
import shutil
import time

import torch

from climbing_gym.envs.futoshiki import score
from climbing_gym.futoshiki import (
    Puzzle,
    render_prompt,
    render_solve_trace,
    solve,
    variable_name,
)
from climbing_gym.projection_sampling import backward_completion_loss, format_audit
from climbing_gym.sampling import write_traces

try:
    from .historical import (
        PROTOCOLS,
        FP32MasterAdamW,
        ordinary_prompt,
        training_batches,
        learning_rate,
        candidate_audit,
    )
    from .native_projection import (
        NativeRuntime,
        evaluate,
        jsonl,
        model_digest,
        model_config_digest,
        tokenizer_digest,
        project_guides,
    )
    from .recipe import (
        digest,
        model_path,
        read_json,
        read_rows,
        stage,
        stage_dir,
        write_json,
    )
except ImportError:
    from historical import (
        PROTOCOLS,
        FP32MasterAdamW,
        ordinary_prompt,
        training_batches,
        learning_rate,
        candidate_audit,
    )
    from native_projection import (
        NativeRuntime,
        evaluate,
        jsonl,
        model_digest,
        model_config_digest,
        tokenizer_digest,
        project_guides,
    )
    from recipe import (
        digest,
        model_path,
        read_json,
        read_rows,
        stage,
        stage_dir,
        write_json,
    )

HERE = Path(__file__).resolve().parent


def sha(data):
    return hashlib.sha256(data).hexdigest()


def validate_records(rows, protocol, *, tokenizer=None):
    if len({r["id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate original puzzle IDs")
    for record in rows:
        row = record["row"]
        puzzle = Puzzle.from_row(row)
        if (
            record["id"] != puzzle.identity
            or row["id"] != record["id"]
            or row["prompt"] != render_prompt(puzzle)
        ):
            raise ValueError(
                "Puzzle ID/prompt provenance differs from structured constraints"
            )
        if not puzzle.accepts(tuple(row["solution"])):
            raise ValueError("Stored target violates puzzle constraints")
        ids = record["completion_token_ids"]
        prompt = record["prompt_token_ids"]
        if (
            not ids
            or not prompt
            or ids[-1] != 7
            or any(type(t) is not int or t < 0 for t in ids + prompt)
        ):
            raise ValueError("Expected nonempty exact IDs with LFM EOS=7")
        if len(ids) + len(prompt) > protocol.packed_tokens:
            raise ValueError("Original trajectory exceeds historical packing budget")
        if not score(record["completion"], row, reward_profile="exact_match_strict")[
            "exact_match"
        ]:
            raise ValueError("Recorded final answer failed the independent verifier")
        if not format_audit(record["completion"])["valid"]:
            raise ValueError("Recorded thinking/answer tags invalid")
        if tokenizer:
            if (
                list(tokenizer.encode(record["prompt"], add_special_tokens=False))
                != prompt
            ):
                raise ValueError(
                    "Tokenizer/chat template differs from exact historical prompt IDs"
                )
            text = tokenizer.decode(
                ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
            )
            if text != record["completion"]:
                raise ValueError(
                    "Completion IDs do not decode to recorded full response"
                )
    return dict(
        rows=len(rows),
        tiers=dict(Counter(r["row"]["tier"] for r in rows)),
        unique_solution_grids=len({tuple(r["row"]["solution"]) for r in rows}),
        teacher_seeded=sum(bool(r.get("teacher_seeded")) for r in rows),
        completion_tokens=sum(len(r["completion_token_ids"]) for r in rows),
        max_sequence_tokens=max(
            len(r["prompt_token_ids"]) + len(r["completion_token_ids"]) for r in rows
        ),
        ordered_ids_sha256=sha("\n".join(r["id"] for r in rows).encode()),
    )


def prepare(root, stage_id):
    """Expand byte-verified portable corpora, preserving original row order."""
    protocol = PROTOCOLS[stage_id]
    directory = stage_dir(root, stage_id)
    manifest = read_json(HERE / "data/projection_manifest.json")
    copied = {}
    for name, meta in manifest["corpora"].items():
        if meta["stage"] != stage_id:
            continue
        data = (HERE / "data" / name).read_bytes()
        if sha(data) != meta["gzip_sha256"]:
            raise ValueError(f"Curated gzip checksum failed: {name}")
        raw = gzip.decompress(data)
        if sha(raw) != meta["curated_sha256"]:
            raise ValueError(f"Curated row checksum failed: {name}")
        path = directory / "data" / (meta["kind"] + ".jsonl")
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists() and path.read_bytes() != raw:
            raise ValueError(f"Refusing to replace changed stage data: {path}")
        path.write_bytes(raw)
        copied[meta["kind"]] = meta
    rows = read_rows(directory / "data/train_recorded.jsonl")
    audit = validate_records(rows, protocol)
    training_ids = {r["id"] for r in rows}
    training_grids = {tuple(r["row"]["solution"]) for r in rows}
    for split in ("test", "dev", "final_test", "calibration"):
        path = directory / f"data/{split}.jsonl"
        if path.exists():
            held = read_rows(path)
            if training_ids & {r["id"] for r in held} or training_grids & {
                tuple(r["solution"]) for r in held
            }:
                raise ValueError(
                    "Recorded training/heldout ID or solution-grid overlap"
                )
    write_json(
        directory / "data_audit.json",
        dict(
            **audit,
            mode="recorded_data_replay",
            recipe_version=manifest["generation_recipe_version"],
            tokenizer=manifest["tokenizer"],
            assets=copied,
        ),
    )
    return audit


def export_samples(root, stage_id, output):
    rows = read_rows(stage_dir(root, stage_id) / "data/train_recorded.jsonl")
    selected = []
    for tier in sorted({r["row"]["tier"] for r in rows}):
        selected.append(next(r for r in rows if r["row"]["tier"] == tier))
    selected += [r for r in rows if r.get("teacher_seeded")][:2]
    write_traces(
        Path(output),
        [
            dict(
                id=r["id"],
                tier=r["row"]["tier"],
                prompt=r["prompt"],
                response=r["completion"],
                expected=r["expected_answer"],
                outcome="Archived projection; independent final-answer verifier passed; intermediate prose not certified",
            )
            for r in selected
        ],
    )


def regenerate_solver_guides(root, tokenizer):
    """Replay the algorithmic traces on the original selected puzzle order."""
    directory = stage_dir(root, "solver_projection_256")
    original = read_rows(directory / "data/guides_recorded.jsonl")
    guides = []
    for source in original:
        puzzle = Puzzle.from_row(source["row"])
        events = []
        solutions = solve(puzzle, trace=events)
        if solutions != [tuple(source["row"]["solution"])]:
            raise ValueError("Original solver guide puzzle is not uniquely solvable")
        response = (
            render_solve_trace(puzzle, events)
            + "\n</thinking>\n<answer>"
            + source["reference_answer"]
            + "</answer><|im_end|>"
        )
        if response != source["expert"]:
            raise ValueError(
                "Solver trace implementation differs from the original guide"
            )
        guide = dict(
            source,
            expert=response,
            expert_completion_token_ids=list(
                tokenizer.encode(response, add_special_tokens=False)
            ),
        )
        if (
            guide["expert_completion_token_ids"]
            != source["expert_completion_token_ids"]
        ):
            raise ValueError(
                "Solver guide tokenizer differs from the original exact IDs"
            )
        guides.append(guide)
    jsonl(directory / "data/guides_recollected.jsonl", guides)
    return len(guides)


def check_tokenizer(model):
    from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer

    meta = read_json(HERE / "data/projection_manifest.json")["tokenizer"]
    expected = meta["files"]["tokenizer.json"]
    if digest(Path(model) / "tokenizer.json") != expected:
        raise ValueError(
            "Historical exact-ID data requires the original LFM2.5-1.2B tokenizer vocabulary"
        )
    return load_lfm2_tokenizer(str(model))


def identity(root, stage_id, model, purpose, batch, guide_mode="recorded"):
    p = PROTOCOLS[stage_id]
    directory = stage_dir(root, stage_id)
    data = directory / (
        f"data/guides_{guide_mode}.jsonl"
        if purpose == "projection"
        else "data/calibration.jsonl"
    )
    return dict(
        stage=stage_id,
        model_sha256=model_digest(model),
        model_config_sha256=model_config_digest(model),
        tokenizer_sha256=tokenizer_digest(model),
        data_sha256=digest(data),
        purpose=purpose,
        native_batch_size=batch,
        max_num_seqs=batch,
        external_request_concurrency_limit=None,
        max_tokens=p.projection_budget
        if purpose == "projection"
        else (8192 if p.teacher_fallback else 4096),
        protocol=asdict(p),
        logprobs=1,
    )


def validate_preflight(path, expected):
    value = read_json(path)
    if value["identity"] != expected:
        raise ValueError("Preflight model, data, token budget or concurrency differs")
    report = value["timing"]
    if (
        report["generated_tokens"] <= 0
        or report["tokens_per_second"] <= 0
        or report["peak_allocated_bytes"] <= 0
    ):
        raise ValueError(
            "Preflight must contain actual native GPU timings and peak memory"
        )
    return value


async def preflight(
    root, stage_id, model, *, purpose, batch, prior=None, guide_mode="recorded"
):
    p = PROTOCOLS[stage_id]
    directory = stage_dir(root, stage_id)
    runtime = NativeRuntime(
        model,
        p,
        batch=batch,
        purpose=purpose,
        output=directory / "preflight",
        setup=True,
    )
    try:
        started = time.perf_counter()
        guides = read_rows(directory / f"data/guides_{guide_mode}.jsonl")
        if purpose == "projection":
            selected = (
                sorted(
                    guides,
                    key=lambda r: len(r["expert_completion_token_ids"]),
                    reverse=True,
                )[:128]
                if p.teacher_fallback
                else guides[:128]
            )
            prefixes = [
                r["expert_completion_token_ids"][
                    : len(r["expert_completion_token_ids"]) // 2
                ]
                if p.teacher_fallback and i % 2
                else []
                for i, r in enumerate(selected)
            ]
            requests = [
                runtime.request(row, prefix)
                for row, prefix in zip(selected, prefixes, strict=True)
            ]
            results, timing = await runtime.generate(
                requests,
                seed_offset=500000 if p.teacher_fallback else 600000,
                label="initial_projection_preflight",
            )
            candidates = [
                prefix + result["completion_token_ids"]
                for prefix, result in zip(prefixes, results, strict=True)
            ]
            scores = runtime.score(
                selected, candidates, label="initial_projection_preflight"
            )
            records = [
                dict(id=r["id"], proposal=out, base_logprobs=lp)
                for r, out, lp in zip(selected, results, scores, strict=True)
            ]
            jsonl(directory / "preflight/projection_proposals.jsonl", records)
        else:
            selected = read_rows(directory / "data/calibration.jsonl")[:128]
            budget = 8192 if p.teacher_fallback else 4096
            requests = []
            for row in selected:
                prompt = ordinary_prompt(runtime.tokenizer, row)
                requests.append(
                    (
                        dict(
                            prompt=prompt,
                            prompt_token_ids=tuple(
                                runtime.tokenizer.encode(
                                    prompt, add_special_tokens=False
                                )
                            ),
                        ),
                        runtime.sampling(budget, projection=False),
                    )
                )
            results, timing = await runtime.generate(
                requests, seed_offset=0, label="initial_training_preflight"
            )
        reference = (
            prior
            or read_json(HERE / "data/projection_benchmarks.json")[stage_id][purpose]
        )
        timing["full_step_seconds"] = time.perf_counter() - started
        timing["full_step_peak_allocated_bytes"] = torch.cuda.max_memory_allocated()
        comparison = dict(
            status="compared",
            tokens_per_second_ratio=timing["tokens_per_second"]
            / reference["tokens_per_second"],
            peak_memory_ratio=timing["peak_allocated_bytes"]
            / reference["peak_allocated_bytes"],
            prior=reference,
            note="Compare workload, output lengths, precision and GPU; review unexplained differences before launch.",
        )
        report = dict(
            identity=identity(root, stage_id, model, purpose, batch, guide_mode),
            timing=timing,
            comparison=comparison,
        )
        write_json(directory / f"preflight/{purpose}.json", report)
        traces = []
        for index, (row, request, out) in enumerate(
            zip(selected, requests, results, strict=True)
        ):
            task = row.get("row", row)
            # A projection request can continue a prefix already in its prompt.
            # Grade the whole candidate, while retaining the actual generated
            # suffix verbatim in the trace's response field.
            verified_response = (
                runtime.tokenizer.decode(
                    candidates[index], skip_special_tokens=False,
                    clean_up_tokenization_spaces=False,
                )
                if purpose == "projection" else out["response"]
            )
            traces.append(
                dict(
                    id=task["id"],
                    tier=task["tier"],
                    prompt=runtime.tokenizer.decode(
                        request[0]["prompt_token_ids"], skip_special_tokens=False
                    ),
                    response=out["response"],
                    expected=" ".join(
                        f"{variable_name(i)}={v}"
                        for i, v in enumerate(task["solution"])
                        if i not in dict((int(k), v) for k, v in task["givens"].items())
                    ),
                    outcome=bool(
                        score(
                            verified_response, task, reward_profile="exact_match_strict"
                        )["exact_match"]
                    ),
                )
            )
        write_traces(directory / f"preflight/{purpose}_traces.md", traces)
        return report
    finally:
        runtime.close()


async def collect(root, model, *, tier, report):
    """Repeat the original single-sample,512-puzzle specialist selection loop."""
    p = PROTOCOLS["projection_reset"]
    directory = stage_dir(root, p.stage)
    # Preflight is specific to this exact specialist, using ordinary prompts.
    validate_preflight(report, identity(root, p.stage, model, "training", 128))
    collection_identity = dict(
        model_sha256=model_digest(model),
        model_config_sha256=model_config_digest(model),
        tokenizer_sha256=tokenizer_digest(model),
        pool_sha256=digest(directory / f"data/pool_{tier}.jsonl"),
        protocol=asdict(p),
    )
    identity_path = directory / f"collection/{tier}/identity.json"
    if identity_path.exists() and read_json(identity_path) != collection_identity:
        raise ValueError(
            "Specialist weights, pool or protocol changed during collection resume"
        )
    write_json(identity_path, collection_identity)
    runtime = NativeRuntime(
        model, p, purpose="training", output=directory / "collection"
    )
    selected = []
    pool = read_rows(directory / f"data/pool_{tier}.jsonl")
    teacher_hash = model_digest(model)
    try:
        for offset in range(0, len(pool), 512):
            if len(selected) >= 1024:
                break
            rows = pool[offset : offset + 512]
            requests = []
            for row in rows:
                text = ordinary_prompt(runtime.tokenizer, row)
                requests.append(
                    (
                        dict(
                            prompt=text,
                            prompt_token_ids=tuple(
                                runtime.tokenizer.encode(text, add_special_tokens=False)
                            ),
                        ),
                        runtime.sampling(8192, projection=False),
                    )
                )
            archive = directory / f"collection/{tier}/batch_{offset // 512:04d}.jsonl"
            if archive.exists():
                records = read_rows(archive)
            else:
                records, _ = await runtime.generate(
                    requests, seed_offset=100000 + offset, label="specialist_" + tier
                )
                jsonl(archive, records)
            for row, request, result in zip(rows, requests, records, strict=True):
                audit = candidate_audit(
                    runtime.tokenizer,
                    result["completion_token_ids"],
                    row,
                    result["finish_reason"],
                )
                if audit["accepted"]:
                    selected.append(
                        dict(
                            id=row["id"],
                            row=row,
                            expert=result["response"],
                            expert_completion_token_ids=result["completion_token_ids"],
                            expert_completion_logprobs=result["completion_logprobs"],
                            reference_answer=audit["format"]["answer"],
                            prompt=request[0]["prompt"],
                            prompt_token_ids=list(request[0]["prompt_token_ids"]),
                            teacher_sha256=teacher_hash,
                        )
                    )
            jsonl(directory / f"collection/{tier}/selected.jsonl", selected)
        if len(selected) < 1024:
            raise RuntimeError(
                "Original specialist puzzle pool exhausted; fresh weights did not collect1024 verified guides. No duplicates or easier-puzzle substitution."
            )
        paths = [
            directory / f"collection/{t}/selected.jsonl"
            for t in ("5x5_g11_i5", "5x5_g9_i5")
        ]
        if all(path.exists() and len(read_rows(path)) >= 1024 for path in paths):
            buckets = [read_rows(path)[:1024] for path in paths]
            jsonl(
                directory / "data/guides_recollected.jsonl",
                [bucket[i] for i in range(1024) for bucket in buckets],
            )
    finally:
        runtime.close()


def saved_policy_manifest(policy):
    policy = Path(policy)
    return {
        str(path.relative_to(policy)): digest(path)
        for path in sorted(policy.rglob("*"))
        if path.is_file()
    }


def validate_saved_policy(directory, state):
    if "policy_sha256" not in state:
        raise ValueError("SFT resume lacks the saved model/tokenizer identity")
    if state.get("policy_sha256") != saved_policy_manifest(Path(directory) / "policy"):
        raise ValueError("SFT resume policy or tokenizer bytes differ")


def save_training(model, tokenizer, optimizer, directory, state):
    directory = Path(directory)
    partial = directory / "resume.partial"
    resume = directory / "resume"
    previous = directory / "resume.previous"
    if partial.exists():
        shutil.rmtree(partial)
    partial.mkdir(parents=True)
    model.to("cpu")
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.cpu()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model.save_pretrained(partial / "policy")
    if callable(getattr(tokenizer, "save_pretrained", None)):
        tokenizer.save_pretrained(partial / "policy")
    torch.save(optimizer.state_dict(), partial / "optimizer.pt")
    torch.save(
        dict(
            torch=torch.get_rng_state(),
            python=random.getstate(),
            cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        ),
        partial / "rng.pt",
    )
    state["policy_sha256"] = saved_policy_manifest(partial / "policy")
    write_json(partial / "state.json", state)
    if previous.exists():
        shutil.rmtree(previous)
    if resume.exists():
        resume.rename(previous)
    partial.rename(resume)
    if previous.exists():
        shutil.rmtree(previous)


def move_training(model, optimizer, device):
    model.to(device)
    for values in optimizer.state.values():
        for key, value in values.items():
            if isinstance(value, torch.Tensor):
                values[key] = value.to(device)


def publish_epoch_checkpoint(directory, epoch, *, pending=False):
    """Publish one immutable snapshot; interrupted copies remain unadvertised."""
    directory = Path(directory)
    checkpoint = directory / f"checkpoints/epoch_{epoch}/policy"
    if checkpoint.exists():
        if not pending:
            raise FileExistsError(f"Refusing to overwrite immutable epoch: {checkpoint}")
        state = read_json(directory / "resume/state.json")
        if saved_policy_manifest(checkpoint) != state["policy_sha256"]:
            raise ValueError("Immutable epoch differs from the pending saved policy")
        return checkpoint
    temporary = checkpoint.with_name(".policy.partial")
    if temporary.exists():
        shutil.rmtree(temporary)
    temporary.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(
        directory / "resume/policy", temporary,
        copy_function=__import__("os").link,
    )
    temporary.replace(checkpoint)
    return checkpoint


def train_loop(
    model,
    tokenizer,
    optimizer,
    rows,
    protocol,
    directory,
    logger,
    *,
    device,
    evaluator=None,
    resume_state=None,
):
    """Source token-normalized/chunked loss and schedules, with exact resume."""
    from climbing_gym.trainers.packing import pack_sequences

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    start_epoch = (resume_state or {}).get("next_epoch", 0)
    start_batch = (resume_state or {}).get("next_batch", 0)
    step = (resume_state or {}).get("optimizer_updates", 0)
    data_hash = sha(json.dumps(rows, sort_keys=True).encode())
    if resume_state and (
        resume_state["data_sha256"] != data_hash
        or resume_state["protocol"] != asdict(protocol)
    ):
        raise ValueError("SFT resume data/protocol differs")

    def save(next_epoch, next_batch, pending_epoch=None):
        state = dict(
            next_epoch=next_epoch,
            next_batch=next_batch,
            optimizer_updates=step,
            data_sha256=data_hash,
            protocol=asdict(protocol),
            pending_epoch=pending_epoch,
        )
        save_training(model, tokenizer, optimizer, directory, state)
        return state

    if resume_state and resume_state.get("pending_epoch") and evaluator:
        # train() restores both learner and Adam tensors to CUDA. A pending
        # evaluation bypasses save_training(), which normally offloads them.
        move_training(model, optimizer, "cpu")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        checkpoint = publish_epoch_checkpoint(
            directory, resume_state["pending_epoch"], pending=True
        )
        evaluator(
            resume_state["pending_epoch"],
            checkpoint,
        )
        write_json(
            directory / "resume/state.json", dict(resume_state, pending_epoch=None)
        )
        if start_epoch < protocol.retained_epoch:
            move_training(model, optimizer, device)
    for epoch in range(start_epoch, protocol.retained_epoch):
        batches = training_batches(rows, protocol, epoch)
        for batch_index, selected in enumerate(batches):
            if epoch == start_epoch and batch_index < start_batch:
                continue
            started = time.perf_counter()
            model.train()
            optimizer.zero_grad(set_to_none=True)
            if str(device).startswith("cuda"):
                torch.cuda.reset_peak_memory_stats()
            lr = learning_rate(protocol, step, len(rows))
            for group in optimizer.param_groups:
                group["lr"] = lr
            packed = pack_sequences(
                [r["prompt_token_ids"] + r["completion_token_ids"] for r in selected],
                prompt_lens=[len(r["prompt_token_ids"]) for r in selected],
                max_tokens=protocol.packed_tokens,
            )
            denominator = sum(b.valid_token_count for b in packed)
            if denominator <= 0:
                raise ValueError("No response/EOS training targets")
            loss = 0.0
            for batch in packed:
                loss += backward_completion_loss(
                    model,
                    batch.to(device),
                    denominator=denominator,
                    head_chunk=protocol.head_chunk,
                    autocast_dtype=torch.bfloat16
                    if str(device).startswith("cuda")
                    else None,
                )
            norm = torch.nn.utils.clip_grad_norm_(
                model.parameters(), protocol.grad_clip, error_if_nonfinite=True
            )
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if str(device).startswith("cuda"):
                torch.cuda.synchronize()
            metrics = dict(
                sft_loss=loss,
                learning_rate=lr,
                grad_norm=float(norm),
                supervised_tokens=denominator,
                training_seconds=time.perf_counter() - started,
                epoch=epoch + 1,
                optimizer_updates=step,
            )
            metrics["supervised_tokens_per_second"] = denominator / max(
                metrics["training_seconds"], 1e-9
            )
            if str(device).startswith("cuda"):
                metrics.update(
                    peak_allocated_bytes=torch.cuda.max_memory_allocated(),
                    peak_reserved_bytes=torch.cuda.max_memory_reserved(),
                )
            last = batch_index + 1 == len(batches)
            if last or (protocol.teacher_fallback and step % 32 == 0):
                state = save(
                    epoch + 1 if last else epoch,
                    0 if last else batch_index + 1,
                    epoch + 1 if last and evaluator else None,
                )
                if last:
                    checkpoint = publish_epoch_checkpoint(directory, epoch + 1)
                    if evaluator:
                        metrics.update(evaluator(epoch + 1, checkpoint))
                        write_json(
                            directory / "resume/state.json",
                            dict(state, pending_epoch=None),
                        )
                if epoch + 1 < protocol.retained_epoch or not last:
                    move_training(model, optimizer, device)
            metrics["full_step_seconds"] = time.perf_counter() - started
            logger.log_step(step=step - 1, metrics=metrics)
    return step


def train(
    root, stage_id, model, *, preflight_report, data_mode="recorded", eval_config=None
):
    from climbing_gym.models import build_model
    from climbing_gym.trainers.logger import WandbLogger
    from climbing_gym.config import Wandb
    import asyncio

    p = PROTOCOLS[stage_id]
    directory = stage_dir(root, stage_id)
    validate_preflight(
        preflight_report, identity(root, stage_id, model, "training", 128)
    )
    tokenizer = check_tokenizer(model)
    rows = read_rows(directory / f"data/train_{data_mode}.jsonl")
    validate_records(rows, p, tokenizer=tokenizer)
    resume = directory / "resume"
    if not resume.exists() and (directory / "resume.previous").exists():
        (directory / "resume.previous").rename(resume)
    resume_state = read_json(resume / "state.json") if resume.exists() else None
    if resume_state is not None:
        validate_saved_policy(resume, resume_state)
    torch.manual_seed(p.seed)
    random.seed(p.seed)
    policy = build_model(
        "lfm2",
        model_dir=str(resume / "policy" if resume.exists() else model),
        attn_backend="flash",
        shortconv_backend="triton_k3",
        fused_backbone_kernels=True,
        recompute_mlp=True,
        checkpoint_layers=p.teacher_fallback,
    )
    policy.to(
        device="cuda", dtype=torch.bfloat16 if p.teacher_fallback else torch.float32
    )
    optimizer = (
        FP32MasterAdamW(
            policy.parameters(), lr=p.learning_rate, weight_decay=p.weight_decay
        )
        if p.teacher_fallback
        else torch.optim.AdamW(
            policy.parameters(),
            lr=p.learning_rate,
            weight_decay=p.weight_decay,
            fused=True,
        )
    )
    if p.teacher_fallback:
        for parameter in policy.parameters():
            optimizer.state[parameter] = dict(
                step=0,
                master=parameter.detach().float().clone(),
                exp_avg=torch.zeros_like(parameter, dtype=torch.float32),
                exp_avg_sq=torch.zeros_like(parameter, dtype=torch.float32),
            )
    if resume.exists():
        optimizer.state.clear()
        optimizer.load_state_dict(
            torch.load(resume / "optimizer.pt", map_location="cpu", weights_only=True)
        )
        move_training(policy, optimizer, "cuda")
        rng = torch.load(resume / "rng.pt", map_location="cpu", weights_only=False)
        torch.set_rng_state(rng["torch"])
        random.setstate(rng["python"])
        if rng["cuda"] is not None:
            torch.cuda.set_rng_state_all(rng["cuda"])

    def evaluator(epoch, checkpoint):
        if eval_config:
            from climbing_gym.eval import load_run_from_toml, run_eval
            from dataclasses import replace

            value = load_run_from_toml(eval_config)
            try:
                value.engine.sync_from_model(policy)
                log = replace(
                    value.logging_config,
                    summary_path=str(directory / f"eval/epoch_{epoch}/summary.json"),
                    trajectory_path=str(
                        directory / f"eval/epoch_{epoch}/trajectories.jsonl"
                    ),
                    trace_path=str(directory / f"eval/epoch_{epoch}/traces.md"),
                )
                return {
                    "eval/" + k: v
                    for k, v in run_eval(replace(value, logging_config=log)).items()
                }
            finally:
                value.engine.close()
        runtime = NativeRuntime(
            checkpoint, p, purpose="training", output=directory / f"eval/epoch_{epoch}"
        )
        try:
            held = read_rows(
                directory
                / ("data/dev.jsonl" if p.teacher_fallback else "data/test.jsonl")
            )
            return asyncio.run(
                evaluate(
                    runtime,
                    held,
                    samples=4 if p.teacher_fallback else 8,
                    destination=directory / f"eval/epoch_{epoch}",
                )
            )
        finally:
            runtime.close()

    with WandbLogger(
        Wandb(dir=str(directory / "logs"), run_name=stage_id),
        config_path=HERE / "recipe.json",
        resolved_config=dict(
            protocol=asdict(p),
            data_mode=data_mode,
            data_sha256=digest(directory / f"data/train_{data_mode}.jsonl"),
            initial_model_sha256=model_digest(model),
            preflight=read_json(preflight_report),
        ),
    ) as logger:
        steps = train_loop(
            policy,
            tokenizer,
            optimizer,
            rows,
            p,
            directory,
            logger,
            device="cuda",
            evaluator=evaluator,
            resume_state=resume_state,
        )
    write_json(
        directory / "result.json",
        dict(
            retained_epoch=p.retained_epoch,
            optimizer_updates=steps,
            policy=str(directory / f"checkpoints/epoch_{p.retained_epoch}/policy"),
            mode=data_mode,
            claim="Same historical data/method; no guarantee of identical checkpoint hashes or accuracy.",
        ),
    )


def main(argv=None):
    import asyncio

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=[
            "prepare",
            "audit",
            "export",
            "solver-guides",
            "preflight",
            "collect",
            "project",
            "train",
        ],
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--stage", choices=sorted(PROTOCOLS), required=True)
    parser.add_argument("--student", type=Path)
    parser.add_argument("--preflight", type=Path)
    parser.add_argument(
        "--purpose", choices=["training", "projection"], default="projection"
    )
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--prior-benchmark", type=Path)
    parser.add_argument(
        "--guides", choices=["recorded", "recollected"], default="recorded"
    )
    parser.add_argument(
        "--data", choices=["recorded", "recollected"], default="recorded"
    )
    parser.add_argument("--tier", choices=["5x5_g11_i5", "5x5_g9_i5"])
    parser.add_argument("--eval-config", type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    if args.batch_size is not None and args.batch_size <= 0:
        parser.error("--batch-size must be positive")
    p = PROTOCOLS[args.stage]
    directory = stage_dir(args.root, args.stage)
    if args.command == "prepare":
        print(json.dumps(prepare(args.root, args.stage)))
        return 0
    if args.command == "audit":
        print(
            json.dumps(
                validate_records(read_rows(directory / "data/train_recorded.jsonl"), p)
            )
        )
        return 0
    if args.command == "export":
        export_samples(
            args.root, args.stage, args.output or directory / "recorded_samples.md"
        )
        return 0
    model = args.student or model_path(
        args.root, stage(args.root, args.stage)["input_policy"]
    )
    tokenizer = check_tokenizer(model)
    if args.command == "solver-guides":
        if args.stage != "solver_projection_256":
            parser.error(
                "solver-guides is only for the algorithmic 4x4 projection stage"
            )
        print(json.dumps(dict(guides=regenerate_solver_guides(args.root, tokenizer))))
        return 0
    if args.command == "preflight":
        batch = args.batch_size or (
            p.native_batch if args.purpose == "projection" else 128
        )
        asyncio.run(
            preflight(
                args.root,
                args.stage,
                model,
                purpose=args.purpose,
                batch=batch,
                prior=read_json(args.prior_benchmark) if args.prior_benchmark else None,
                guide_mode=args.guides,
            )
        )
        return 0
    if args.command == "collect":
        if args.stage != "projection_reset" or not args.tier or not args.preflight:
            parser.error(
                "collect requires projection_reset, --tier, --student specialist and its --preflight training report"
            )
        asyncio.run(collect(args.root, model, tier=args.tier, report=args.preflight))
        return 0
    if args.command == "train":
        train(
            args.root,
            args.stage,
            model,
            preflight_report=args.preflight or directory / "preflight/training.json",
            data_mode=args.data,
            eval_config=args.eval_config,
        )
        return 0
    if args.command == "project":
        batch = args.batch_size or p.native_batch
        validate_preflight(
            args.preflight or directory / "preflight/projection.json",
            identity(args.root, args.stage, model, "projection", batch, args.guides),
        )
        guides = read_rows(directory / f"data/guides_{args.guides}.jsonl")
        runtime = NativeRuntime(model, p, batch=batch, output=directory / "projection")
        try:
            initial = (
                read_rows(directory / "preflight/projection_proposals.jsonl")
                if not p.teacher_fallback
                else None
            )
            records = asyncio.run(
                project_guides(
                    runtime,
                    guides,
                    directory / "projection" / args.guides,
                    initial_preflight=initial,
                )
            )
            records = records if p.teacher_fallback else records[:256]
            jsonl(directory / "data/train_recollected.jsonl", records)
            write_json(
                directory / "projection/summary.json",
                validate_records(records, p, tokenizer=runtime.tokenizer),
            )
            write_traces(
                directory / "projection/traces.md",
                [
                    dict(
                        id=r["id"],
                        tier=r["row"]["tier"],
                        prompt=r["prompt"],
                        response=r["completion"],
                        expected=r["expected_answer"],
                        outcome="Verified final answer; intermediate prose unverified",
                    )
                    for r in records
                ],
            )
        finally:
            runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
