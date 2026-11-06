"""Native execution of the two historical, complete-trajectory projectors."""

from dataclasses import asdict
import hashlib
import json
import math
from pathlib import Path
import time

import torch

from climbing_gym.inference import (
    InferenceEngineConfig,
    LFM25AsyncInferenceEngine,
    SamplingConfig,
)
from climbing_gym.projection_sampling import ProjectionState
from climbing_gym.sampling import write_traces

try:
    from .historical import (
        candidate_audit,
        ordinary_prompt,
        proposal_prompt,
        suffix_budget,
    )
    from .recipe import digest, read_json, read_rows, write_json
except ImportError:
    from historical import (
        candidate_audit,
        ordinary_prompt,
        proposal_prompt,
        suffix_budget,
    )
    from recipe import digest, read_json, read_rows, write_json


def jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def model_digest(model):
    model = Path(model)
    files = sorted(p for p in model.iterdir() if p.suffix in {".bin", ".safetensors"})
    if not files:
        raise ValueError(f"No model weight files in {model}")
    if len(files) == 1:
        return digest(files[0])
    return hashlib.sha256(
        "".join(p.name + ":" + digest(p) + "\n" for p in files).encode()
    ).hexdigest()


def tokenizer_digest(model):
    files = {}
    for name in (
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "chat_template.jinja",
    ):
        path = Path(model) / name
        if path.exists():
            files[name] = digest(path)
    if "tokenizer.json" not in files:
        raise ValueError("Tokenizer identity requires tokenizer.json")
    return hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()


def model_config_digest(model):
    """Bind inference semantics without changing historical weight hashes."""
    path = Path(model) / "config.json"
    return digest(path) if path.exists() else None


class NativeRuntime:
    def __init__(
        self,
        model,
        protocol,
        *,
        batch=None,
        purpose="projection",
        output=None,
        setup=False,
    ):
        self.protocol = protocol
        self.purpose = purpose
        self.output = Path(output) if output else None
        self.model_identity = dict(
            weights_sha256=model_digest(model),
            config_sha256=model_config_digest(model),
            tokenizer_sha256=tokenizer_digest(model),
        )
        self.batch = batch or (
            protocol.native_batch if purpose == "projection" else 128
        )
        self.context = (
            protocol.context
            if purpose == "projection"
            else (9216 if protocol.teacher_fallback else 8192)
        )
        self.inference_seed = (
            protocol.inference_seed
            if protocol.teacher_fallback or (purpose == "projection" and not setup)
            else 20261001
        )
        self.engine = LFM25AsyncInferenceEngine(
            InferenceEngineConfig(
                model_path=str(model),
                dtype="bfloat16",
                seed=self.inference_seed,
                max_model_len=self.context,
                max_num_seqs=self.batch,
                apply_chat_template=False,
                assistant_prefill="",
            ),
            max_batch_size=self.batch,
            compile_decode=False,
            cuda_graphs=True,
            max_cuda_graphs=1,
            bf16_weight_packing=True,
            max_prefill_tokens=4096,
        )
        self.tokenizer = self.engine.tokenizer
        if (
            self.engine.max_batch_size != self.batch
            or self.engine.config.max_num_seqs != self.batch
        ):
            self.close()
            raise RuntimeError("Native batch/concurrency configuration differs")

    def emit(self, event, **values):
        record = dict(event=event, **values)
        print(json.dumps(record), flush=True)
        if self.output:
            self.output.mkdir(parents=True, exist_ok=True)
            with (self.output / "timings.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")

    def sampling(self, budget, *, projection=True):
        return SamplingConfig(
            temperature=0.6 if projection else 1.0,
            top_p=1.0,
            top_k=-1,
            max_tokens=budget,
            logprobs=1,
            stop=("<|endoftext|>",),
            extra_params=dict(repetition_penalty=1.1 if projection else 1.0),
        )

    def request(self, guide, prefix, *, limit=None):
        text, ids = proposal_prompt(self.tokenizer, guide, prefix)
        budget = (limit or self.protocol.projection_budget) - len(prefix)
        if budget <= 0 or len(ids) + budget > self.context:
            raise ValueError(
                "Historical proposal exceeds context or suffix budget; refusing truncation"
            )
        return dict(prompt=text, prompt_token_ids=ids), self.sampling(budget)

    async def generate(self, requests, *, seed_offset, label, max_tokens=None):
        if not requests:
            return [], dict(seconds=0.0, generated_tokens=0, tokens_per_second=0.0)
        max_tokens = max_tokens or max(s.max_tokens for _, s in requests)
        with self.engine._seed_lock:
            self.engine._generation_index = seed_offset
        outputs = {}
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()

        def complete(index, result):
            output = result.outputs[0]
            if tuple(result.prompt_token_ids) != tuple(
                requests[index][0]["prompt_token_ids"]
            ):
                raise RuntimeError("Native engine changed exact prompt IDs")
            ids = list(output.token_ids)
            lp = list(output.completion_logprobs or ())
            if (
                not ids
                or len(ids) != len(lp)
                or len(ids) > requests[index][1].max_tokens
                or not all(math.isfinite(value) and value <= 1e-6 for value in lp)
            ):
                raise RuntimeError("Invalid native sampled-token/logprob output")
            outputs[index] = dict(
                completion_token_ids=ids,
                completion_logprobs=lp,
                finish_reason=output.finish_reason,
                response=self.tokenizer.decode(
                    ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
                ),
            )

        await self.engine.generate_chains(
            requests,
            on_result=complete,
            max_new_tokens=max_tokens,
            reuse_cache=False,
            poll_interval=32,
        )
        torch.cuda.synchronize()
        seconds = time.perf_counter() - started
        stats = dict(self.engine.last_session_stats)
        if len(outputs) != len(requests) or stats["admitted_requests"] != len(requests):
            raise RuntimeError("Native request cap dropped part of the stage queue")
        if (
            stats["peak_live_slots"] != min(self.batch, len(requests))
            or stats["resumed_requests"]
        ):
            raise RuntimeError("Unexpected native session concurrency or continuation")
        records = [outputs[i] for i in range(len(requests))]
        tokens = sum(len(r["completion_token_ids"]) for r in records)
        timing = dict(
            seconds=seconds,
            generated_tokens=tokens,
            tokens_per_second=tokens / seconds,
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(),
            native_batch_size=self.batch,
            max_num_seqs=self.batch,
            external_request_concurrency_limit=None,
            max_total_requests=len(requests),
            max_completion_tokens=max_tokens,
            logprobs=1,
            inference_seed=self.inference_seed,
            session_stats=stats,
        )
        self.emit("native_generation", label=label, **timing)
        return records, timing

    def score(self, guides, candidates, *, label):
        self.engine.release_inference_buffers()
        started = time.perf_counter()
        scores = self.engine.raw_engine.score_completions(
            [g["prompt_token_ids"] for g in guides],
            candidates,
            max_prefill_tokens=4096,
            head_chunk_size=128 if self.protocol.teacher_fallback else 256,
        )
        torch.cuda.synchronize()
        for ids, lp in zip(candidates, scores, strict=True):
            if len(ids) != len(lp) or not all(
                math.isfinite(x) and x <= 1e-6 for x in lp
            ):
                raise ValueError(
                    "Raw complete-trajectory base scores are not finite and aligned"
                )
        self.emit(
            "native_scoring",
            label=label,
            seconds=time.perf_counter() - started,
            tokens=sum(map(len, candidates)),
            peak_allocated_bytes=torch.cuda.max_memory_allocated(),
        )
        return [list(lp) for lp in scores]

    def close(self):
        self.engine.close()


async def project_guides(runtime, guides, destination, *, initial_preflight=None):
    """Replay archives on resume; otherwise perform the original suffix updates.

    Seed source logprobs, when absent from the curated guide, remain explicitly
    unavailable (None). They are provenance only; acceptance always uses raw
    frozen-student ordinary-prompt scores, never these proposal logprobs.
    """
    p = runtime.protocol
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    identity = hashlib.sha256(
        json.dumps(
            dict(protocol=asdict(p), guides=guides, model=runtime.model_identity),
            sort_keys=True,
        ).encode()
    ).hexdigest()
    identity_path = destination / "identity.json"
    if identity_path.exists() and read_json(identity_path)["sha256"] != identity:
        raise ValueError("Projection guide/order/protocol changed during resume")
    write_json(identity_path, dict(sha256=identity))
    final = []
    for offset in range(0, len(guides), p.block_size):
        block_index = offset // p.block_size
        block = destination / f"block_{block_index:04d}"
        block.mkdir(exist_ok=True)
        rows = guides[offset : offset + p.block_size]
        if (block / "final.jsonl").exists():
            records = read_rows(block / "final.jsonl")
            if [r["id"] for r in records] != [r["id"] for r in rows]:
                raise ValueError("Archived block order changed")
            final += records
            continue
        states = [ProjectionState(row, p.seed) for row in rows]
        seed_origins = {}
        base_scores = {}
        initial_scores = {}

        # Each operation's result is saved before applying it, so restart
        # replays the same transitions without resampling or duplicating IDs.
        async def operation(indices, cuts, *, name, seed, initialize=False):
            started = time.perf_counter()
            archive_path = block / (name + ".jsonl")
            selected = [states[i] for i in indices]
            if archive_path.exists():
                archive = read_rows(archive_path)
            else:
                requests = []
                for state, cut in zip(selected, cuts, strict=True):
                    budget = suffix_budget(p, state.row, state.tokens, cut)
                    requests.append(
                        runtime.request(
                            state.row, state.tokens[:cut], limit=cut + budget
                        )
                    )
                if initialize and initial_preflight is not None and offset == 0:
                    prior = initial_preflight
                    if [r["id"] for r in prior] != [
                        s.row["id"] for s in selected[: len(prior)]
                    ]:
                        raise ValueError(
                            "Solver preflight proposals do not match guide order"
                        )
                    later, _ = await runtime.generate(
                        requests[len(prior) :], seed_offset=600128, label=name
                    )
                    generated = [r["proposal"] for r in prior] + later
                    candidates = [
                        s.tokens[:cut] + r["completion_token_ids"]
                        for s, cut, r in zip(selected, cuts, generated, strict=True)
                    ]
                    later_scores = (
                        runtime.score(
                            [s.row for s in selected[len(prior) :]],
                            candidates[len(prior) :],
                            label=name,
                        )
                        if later
                        else []
                    )
                    scores = [r["base_logprobs"] for r in prior] + later_scores
                else:
                    generated, _ = await runtime.generate(
                        requests, seed_offset=seed, label=name
                    )
                    candidates = [
                        s.tokens[:cut] + r["completion_token_ids"]
                        for s, cut, r in zip(selected, cuts, generated, strict=True)
                    ]
                    scores = runtime.score(
                        [s.row for s in selected], candidates, label=name
                    )
                archive = [
                    dict(
                        id=s.row["id"],
                        cut=cut,
                        proposal=r,
                        candidate=c,
                        base_logprobs=lp,
                        audit=candidate_audit(
                            runtime.tokenizer,
                            c,
                            s.row["row"],
                            r["finish_reason"],
                            solver=not p.teacher_fallback,
                        ),
                    )
                    for s, cut, r, c, lp in zip(
                        selected, cuts, generated, candidates, scores, strict=True
                    )
                ]
                jsonl(archive_path, archive)
            if [r["id"] for r in archive] != [s.row["id"] for s in selected]:
                raise ValueError("Archived request order differs")
            for state, record, cut in zip(selected, archive, cuts, strict=True):
                if (
                    record["cut"] != cut
                    or record["candidate"][:cut] != state.tokens[:cut]
                ):
                    raise ValueError("Archived exact prefix/cut differs")
                was_empty = not state.tokens
                if record["audit"]["accepted"]:
                    changed = state.revise(
                        record["candidate"],
                        record["proposal"]["completion_logprobs"],
                        record["base_logprobs"],
                        cut=cut,
                        initialize=was_empty,
                    )
                    if was_empty and not initialize and not p.teacher_fallback:
                        state.attempts += 1
                    if changed:
                        base_scores[state.row["id"]] = record["base_logprobs"]
                        if was_empty:
                            initial_scores[state.row["id"]] = state.score
                        if state.row["id"] in seed_origins:
                            seed_origins[state.row["id"]] = seed_origins[
                                state.row["id"]
                            ][:cut] + [False] * len(
                                record["proposal"]["completion_token_ids"]
                            )
                elif not initialize:
                    state.attempts += 1
            runtime.emit(
                "projection_transition",
                block=block_index,
                phase=name,
                full_step_seconds=time.perf_counter() - started,
                valid_chains=sum(bool(s.tokens) for s in states),
            )

        for iteration in range(-1, p.refinements):
            cuts = [
                0 if iteration < 0 or not s.tokens else s.cut(block_index, iteration)
                for s in states
            ]
            seed = (
                (600000 + offset * 100 + (iteration + 1) * 1000000)
                if p.teacher_fallback
                else (600000 if iteration < 0 else 610000 + 100000 * iteration)
            )
            await operation(
                list(range(len(states))),
                cuts,
                name=f"iteration_{iteration + 1:02d}",
                seed=seed,
                initialize=iteration < 0,
            )
        for retry in range(p.retries):
            missing = [i for i, s in enumerate(states) if not s.tokens]
            if not missing:
                break
            seed = (
                (40000000 + offset * 100 + retry * 1000000)
                if p.teacher_fallback
                else 3000000 + 100000 * retry
            )
            await operation(
                missing,
                [0] * len(missing),
                name=f"validity_retry_{retry:02d}",
                seed=seed,
                initialize=True,
            )
        missing = [i for i, s in enumerate(states) if not s.tokens]
        if missing and p.teacher_fallback:
            selected = [states[i] for i in missing]
            scores = runtime.score(
                [s.row for s in selected],
                [s.row["expert_completion_token_ids"] for s in selected],
                label="teacher_seed",
            )
            for state, lp in zip(selected, scores, strict=True):
                tokens = state.row["expert_completion_token_ids"]
                if not candidate_audit(
                    runtime.tokenizer, tokens, state.row["row"], "stop"
                )["accepted"]:
                    raise ValueError("Fallback reference is not independently verified")
                provenance = state.row.get(
                    "expert_completion_logprobs", [None] * len(tokens)
                )
                state.revise(tokens, provenance, lp, cut=0, initialize=True)
                seed_origins[state.row["id"]] = [True] * len(tokens)
                base_scores[state.row["id"]] = lp
                initial_scores[state.row["id"]] = state.score
        if any(not s.tokens for s in states):
            raise RuntimeError(
                "No verified projection for some original puzzles; no solver-stage copy fallback"
            )
        seeded = [i for i, s in enumerate(states) if s.row["id"] in seed_origins]
        for iteration in range(p.seed_refinements if seeded else 0):
            cuts = [
                states[i].cut(block_index, p.refinements + iteration) for i in seeded
            ]
            await operation(
                seeded,
                cuts,
                name=f"teacher_seed_refinement_{iteration:02d}",
                seed=80000000 + offset * 100 + iteration * 1000000,
            )
        records = []
        for state in states:
            audit = candidate_audit(
                runtime.tokenizer,
                state.tokens,
                state.row["row"],
                "stop",
                solver=not p.teacher_fallback,
            )
            if (
                not audit["accepted"]
                or len(state.row["prompt_token_ids"]) + len(state.tokens)
                > p.packed_tokens
            ):
                raise ValueError(
                    "Final projection violates verified complete SFT trajectory contract"
                )
            records.append(
                dict(
                    id=state.row["id"],
                    row=state.row["row"],
                    prompt=state.row["prompt"],
                    prompt_token_ids=state.row["prompt_token_ids"],
                    completion_token_ids=state.tokens,
                    completion=audit["response"],
                    expected_answer=state.row["reference_answer"],
                    base_logprobs=base_scores[state.row["id"]],
                    base_mean_logprob=state.score,
                    initial_projection_mean_logprob=initial_scores[state.row["id"]],
                    projection_attempts=state.attempts,
                    projection_acceptances=state.acceptances,
                    teacher_seeded=state.row["id"] in seed_origins,
                    teacher_origin_token_count=sum(
                        seed_origins.get(state.row["id"], [])
                    ),
                    teacher_sha256=state.row.get("teacher_sha256"),
                    generation_recipe_version="futoshiki-hill-climb-v1",
                )
            )
        jsonl(block / "final.jsonl", records)
        final += records
    return final


async def evaluate(runtime, rows, *, samples, destination):
    from climbing_gym.envs.futoshiki import score

    requests = []
    tasks = []
    budget = 8192 if runtime.protocol.teacher_fallback else 4096
    for row in rows:
        text = ordinary_prompt(runtime.tokenizer, row)
        ids = tuple(runtime.tokenizer.encode(text, add_special_tokens=False))
        for sample in range(samples):
            requests.append(
                (
                    dict(prompt=text, prompt_token_ids=ids),
                    runtime.sampling(budget, projection=False),
                )
            )
            tasks.append((row, text, sample))
    records, timing = await runtime.generate(
        requests, seed_offset=0, label="epoch_evaluation", max_tokens=budget
    )
    traces = []
    graded = []
    for record, (row, prompt, sample) in zip(records, tasks, strict=True):
        grade = score(record["response"], row, reward_profile="exact_match_strict")
        graded.append(
            dict(
                id=row["id"],
                tier=row["tier"],
                sample=sample,
                prompt=prompt,
                **record,
                grade=grade,
            )
        )
        traces.append(
            dict(
                id=f"{row['id']}/{sample}",
                tier=row["tier"],
                prompt=prompt,
                response=record["response"],
                expected=" ".join(
                    f"{chr(65 + i)}={v}"
                    for i, v in enumerate(row["solution"])
                    if i not in dict((int(k), v) for k, v in row["givens"].items())
                ),
                outcome=bool(grade["exact_match"]),
            )
        )
    summary = {
        tier: sum(r["grade"]["exact_match"] for r in graded if r["tier"] == tier)
        / sum(r["tier"] == tier for r in graded)
        for tier in sorted({r["tier"] for r in graded})
    }
    destination = Path(destination)
    jsonl(destination / "trajectories.jsonl", graded)
    write_json(destination / "summary.json", dict(pass_at_1=summary, timing=timing))
    # Actual full responses, with both successes/failures and every available tier.
    write_traces(destination / "traces.md", traces)
    return {f"eval/{tier}/pass_at_1": value for tier, value in summary.items()}
