"""Validate a saved native speed/memory benchmark once before a run starts.

This module never launches a speed probe and never acts as a training-time speed
watchdog. Callers keep the returned metadata beside resolved run configuration.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any, Mapping


def _read_report(path: str | Path) -> dict[str, Any]:
    report_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        raise ValueError(f"Cannot read native preflight report {report_path}: {exc}") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("config"), dict):
        raise ValueError(f"Invalid native benchmark report: {report_path}")
    return payload


def _positive(value: Any, label: str) -> float:
    number = float(value)
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"Preflight {label} must be finite and positive")
    return number


def resolve_limits(inference: Any, rollout: Any) -> dict[str, int | None]:
    """Mirror native adapter and rollout runner concurrency resolution."""
    groups = int(rollout.effective_batch_size)
    repeats = int(rollout.num_rollouts)
    batch = int(rollout.batch_size)
    for value, name in ((groups, "effective_batch_size"), (repeats, "num_rollouts"), (batch, "batch_size")):
        if value <= 0:
            raise ValueError(f"{name} must be positive")
    native = getattr(inference, "native_max_batch_size", None)
    seqs = getattr(inference, "max_num_seqs", None)
    inflight = getattr(rollout, "max_inflight_requests", None)
    for value, name in ((native, "native_max_batch_size"), (seqs, "max_num_seqs"), (inflight, "max_inflight_requests")):
        if value is not None and int(value) <= 0:
            raise ValueError(f"{name} must be positive when configured")
    engine_batch = int(native or seqs or 64)
    if seqs is not None:
        engine_batch = min(engine_batch, int(seqs))
    request_limit = int(inflight) if inflight is not None else groups * repeats
    # Explicit inflight can launch more active groups than the target effective
    # count. Include that cap, not only the number of ultimately kept groups.
    return {"batch_size": batch, "effective_batch_size": groups, "num_rollouts": repeats,
            "native_max_batch_size": None if native is None else int(native),
            "max_num_seqs": None if seqs is None else int(seqs),
            "max_inflight_requests": request_limit, "native_batch_size": engine_batch,
            "representative_batch_size": min(engine_batch, request_limit),
            "max_total_requests": getattr(rollout, "max_total_requests", None)}


def _matching_batch(report: Mapping[str, Any], batch_size: int) -> Mapping[str, Any]:
    for batch in report.get("batches", []):
        if int(batch.get("batch_size", 0)) == batch_size:
            _positive(batch.get("median_tokens_per_second", 0), "tokens_per_second")
            device = str(report["config"].get("device", "cuda"))
            if device.startswith("cuda"):
                _positive(batch.get("peak_allocated_bytes", 0), "peak_allocated_bytes")
                _positive(batch.get("peak_reserved_bytes", 0), "peak_reserved_bytes")
            runs = batch.get("runs", [])
            if not runs:
                raise ValueError("Native preflight report contains no measured rollouts")
            for run in runs:
                _positive(run.get("seconds", 0), "rollout seconds")
                _positive(run.get("tokens_per_second", 0), "rollout tokens_per_second")
                if int(run.get("output_tokens", 0)) != batch_size * int(report["config"]["output_tokens"]):
                    raise ValueError("Native preflight report did not complete the intended token budget")
            return batch
    raise ValueError(f"Native preflight report must benchmark resolved batch size {batch_size}")


def validate_preflight(config: Any, *, rollout_config: Any | None = None) -> dict[str, Any]:
    """Check representative benchmark compatibility and compare a prior baseline.

    Accepts a resolved training config, or an object with ``inference`` and
    ``rollout`` attributes. No GPU/network work occurs. A first benchmark records
    ``comparison.status='initial_baseline'``; supply a prior report thereafter.
    Throughput/memory ratios are recorded for human inspection, never used to
    stop an already running training job.
    """
    inference = config.inference
    rollout = rollout_config or config.rollout
    if str(getattr(inference, "backend", "lfm25_native")) not in {"lfm25", "native_lfm25", "lfm25_native"}:
        raise ValueError("Preflight requires the native inference backend")
    path = str(getattr(inference, "preflight_report_path", "") or os.environ.get("CLIMBING_GYM_PREFLIGHT_REPORT", "")).strip()
    if not path:
        raise ValueError("A native preflight benchmark is required before training/evaluation. Set "
                         "inference.preflight_report_path (or CLIMBING_GYM_PREFLIGHT_REPORT) to the JSON "
                         "from python -m climbing_gym.inference.benchmark using representative --prompts-json, "
                         "the intended --output-tokens, and the resolved native batch size.")
    report = _read_report(path)
    options = report["config"]
    if report.get("engine", "native_lfm25") != "native_lfm25":
        raise ValueError("Preflight artifact must come from the native inference benchmark")
    if report.get("workload", {}).get("kind", "representative") != "representative":
        raise ValueError("Preflight artifact must use a representative workload")
    sampling = rollout.sampling or inference.sampling
    if int(options.get("output_tokens", 0)) != int(sampling.max_tokens):
        raise ValueError("Native preflight output token budget differs from intended rollout max_tokens")
    if report.get("sampling", {}).get("return_logprobs") is not True:
        raise ValueError("Native preflight must measure generation with log probabilities")
    measured_sampling = report.get("sampling", {})
    for name, intended in (("temperature", float(sampling.temperature)),
                           ("top_k", 0 if sampling.top_k == -1 else int(sampling.top_k)),
                           ("top_p", float(sampling.top_p)),
                           ("repetition_penalty", float(sampling.extra_params.get("repetition_penalty", 1.0))),
                           ("behavior_topk", int(sampling.extra_params.get("behavior_topk", 0)))):
        if name in measured_sampling and measured_sampling[name] != intended:
            raise ValueError(f"Native preflight sampling {name} differs from intended rollout sampling")
    if not options.get("prompts_json"):
        raise ValueError("Native preflight requires representative --prompts-json; synthetic prompts are insufficient")
    if Path(str(options.get("model", ""))).expanduser().resolve() != Path(inference.model_path).expanduser().resolve():
        raise ValueError("Native preflight checkpoint differs from inference.model_path")
    for option, attribute in (("cuda_graphs", "native_cuda_graphs"), ("compile_decode", "compile_decode"),
                              ("fp8_mlp", "native_fp8_mlp"), ("fp8_kv", "native_fp8_kv"),
                              ("fp8_lm_head", "native_fp8_lm_head"), ("adaptive_batching", "native_adaptive_batching")):
        # compile_decode is ignored when CUDA graphs are active.
        if option == "compile_decode" and getattr(inference, "native_cuda_graphs", False):
            continue
        if bool(options.get(option, False)) != bool(getattr(inference, attribute, False)):
            raise ValueError(f"Native preflight {option} differs from inference.{attribute}")
    if getattr(inference, "native_cuda_graphs", False):
        if bool(options.get("bf16_weight_packing", False)) != bool(getattr(inference, "native_bf16_weight_packing", True)):
            raise ValueError("Native preflight bf16_weight_packing differs from inference.native_bf16_weight_packing; "
                             "benchmark requires --bf16-weight-packing for the default runtime setting")
        if int(options.get("max_prefill_tokens", 4096)) != int(getattr(inference, "native_prefill_tokens", 4096)):
            raise ValueError("Native preflight max_prefill_tokens differs from inference.native_prefill_tokens")
        if int(getattr(inference, "native_cuda_max_graphs", 1)) != 1:
            raise ValueError("Native benchmark currently validates native_cuda_max_graphs=1 only")
    if str(options.get("dtype", "auto")) != str(inference.dtype):
        raise ValueError("Native preflight dtype differs from inference.dtype")
    if getattr(inference, "max_model_len", None) is not None and int(report.get("max_model_len", 0)) != int(inference.max_model_len):
        raise ValueError("Native preflight context length differs from inference.max_model_len")
    limits = resolve_limits(inference, rollout)
    measured = _matching_batch(report, int(limits["representative_batch_size"]))
    prior_path = str(getattr(inference, "preflight_prior_report_path", "") or
                     os.environ.get("CLIMBING_GYM_PREFLIGHT_PRIOR_REPORT", "")).strip()
    comparison: dict[str, Any] = {"status": "initial_baseline", "prior_report_path": None}
    if prior_path:
        prior = _read_report(prior_path)
        if Path(str(prior["config"].get("model", ""))).expanduser().resolve() != Path(inference.model_path).expanduser().resolve():
            raise ValueError("Prior native benchmark checkpoint is not comparable to this preflight")
        current_hardware = report.get("hardware", {}).get("name")
        prior_hardware = prior.get("hardware", {}).get("name")
        if current_hardware and prior_hardware and current_hardware != prior_hardware:
            raise ValueError("Prior native benchmark used different hardware")
        if int(prior["config"].get("output_tokens", 0)) != int(sampling.max_tokens):
            raise ValueError("Prior benchmark token budget is not comparable to this preflight")
        previous = _matching_batch(prior, int(limits["representative_batch_size"]))
        comparison = {"status": "compared", "prior_report_path": str(Path(prior_path).expanduser().resolve()),
                      "tokens_per_second_ratio": float(measured["median_tokens_per_second"]) / _positive(previous["median_tokens_per_second"], "prior throughput")}
        for key in ("peak_allocated_bytes", "peak_reserved_bytes"):
            if measured.get(key) is not None and previous.get(key) is not None:
                comparison[f"{key}_ratio"] = float(measured[key]) / _positive(previous[key], f"prior {key}")
    return {"report_path": str(Path(path).expanduser().resolve()),
            "report_sha256": hashlib.sha256(Path(path).expanduser().read_bytes()).hexdigest(), "limits": limits,
            "max_tokens": int(sampling.max_tokens), "logprobs": True,
            "tokens_per_second": float(measured["median_tokens_per_second"]),
            "peak_allocated_bytes": measured.get("peak_allocated_bytes"),
            "peak_reserved_bytes": measured.get("peak_reserved_bytes"), "comparison": comparison}


def validate_eval_preflight(config: Any, eval_run: Any) -> dict[str, Any]:
    """Validate periodic-eval setup once, with optional separate saved report."""
    from dataclasses import replace
    from types import SimpleNamespace
    inference = config.inference
    eval_config = config.eval
    report = str(getattr(eval_config, "preflight_report_path", "") or "")
    prior = str(getattr(eval_config, "preflight_prior_report_path", "") or "")
    if report:
        inference = replace(inference, preflight_report_path=report,
                            preflight_prior_report_path=prior)
    elif prior:
        inference = replace(inference, preflight_prior_report_path=prior)
    coordinator = eval_run.rollout_coordinator.config
    runner = coordinator.resolved_runner()
    rollout = SimpleNamespace(
        sampling=coordinator.sampling, num_rollouts=coordinator.num_rollouts,
        batch_size=coordinator.batch_size, effective_batch_size=runner.effective_batch_size,
        max_inflight_requests=runner.max_inflight_requests,
        max_total_requests=coordinator.max_total_requests,
    )
    return validate_preflight(SimpleNamespace(inference=inference, rollout=rollout))


__all__ = ["prepare_prompts", "resolve_limits", "validate_eval_preflight", "validate_preflight"]


def prepare_prompts(config_path: str | Path, output_path: str | Path, *,
                    max_examples: int = 16, periodic_eval: bool = False,
                    evaluation: bool = False) -> dict[str, Any]:
    """Tokenize representative real first-turn requests without loading weights.

    Uses the configured dataset order/seed, environment, harness and native chat
    renderer. Multi-turn users should add representative continuation lengths to
    the workload as well, since the first request cannot predict their lengths.
    """
    from dataclasses import replace
    import random
    from climbing_gym.config.manager import ConfigManager
    from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
    from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer

    if periodic_eval and evaluation:
        raise ValueError("Choose either evaluation or periodic_eval")
    if max_examples <= 0:
        raise ValueError("max_examples must be positive")
    manager = ConfigManager.from_toml(config_path)
    config = manager.config
    section = config.eval if periodic_eval else config
    rows = list(section.data.load_rows())
    eval_settings = config.eval
    eval_profile = periodic_eval or evaluation
    seed = eval_settings.seed if eval_profile else section.data.seed
    if (eval_settings.shuffle if eval_profile else section.data.shuffle):
        random.Random(seed).shuffle(rows)
    if eval_profile and eval_settings.max_examples > 0:
        rows = rows[:eval_settings.max_examples]
    total_examples = len(rows)
    rows = rows[:max_examples]
    if not rows:
        raise ValueError("Preflight dataset resolved to zero representative rows")
    env = section.env.build()
    harness = config.harness.build()
    inference = config.inference
    tokenizer = load_lfm2_tokenizer(inference.model_path)
    prompts: list[list[int]] = []
    examples: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        example = env.build_example(row, dataset_index=index)
        request = harness.next_request(example, harness.create_state(example, 0))
        if request is None:
            raise ValueError("Harness returned no first request for preflight")
        tokens = LFM25AsyncInferenceEngine.tokenize_for_config(
            inference.build_engine_config(), tokenizer, request.prompt)
        prompts.append(list(tokens))
        examples.append({"prompt": tokenizer.decode(list(tokens), skip_special_tokens=False),
                         "expected_answer": example.target, "source_row": index})
    rollout = section.rollout
    # Evaluation collects all selected rows in a single coordinator batch.
    if eval_profile:
        rollout = replace(rollout, batch_size=total_examples, effective_batch_size=total_examples)
    sampling = rollout.sampling or inference.sampling
    result = {"prompts": prompts, "examples": examples,
              "config_path": str(manager.config_path), "periodic_eval": periodic_eval, "evaluation": evaluation,
              "limits": resolve_limits(inference, rollout),
              "output_tokens": sampling.max_tokens,
              "note": "Actual first-turn env/harness prompts, with native chat template and thinking prefill."}
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, ensure_ascii=False, indent=2, default=str) + "\n", encoding="utf-8")
    return result


def main(argv: list[str] | None = None) -> int:
    import argparse
    parser = argparse.ArgumentParser(description="Prepare real native benchmark prompts without loading model weights")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--max-examples", type=int, default=16)
    parser.add_argument("--evaluation", action="store_true", help="Use standalone evaluation row count/order/settings")
    parser.add_argument("--periodic-eval", action="store_true", help="Use the training config's nested eval dataset/settings")
    args = parser.parse_args(argv)
    result = prepare_prompts(args.config, args.output, max_examples=args.max_examples, periodic_eval=args.periodic_eval, evaluation=args.evaluation)
    print(json.dumps({"prompts_path": str(args.output.resolve()), "prompt_count": len(result["prompts"]),
                      "prompt_lengths": [len(row) for row in result["prompts"]],
                      "output_tokens": result["output_tokens"], "limits": result["limits"]}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
