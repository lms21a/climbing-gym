from __future__ import annotations

import argparse
from dataclasses import dataclass, field, replace
import json
from pathlib import Path
import random
import time
import tomllib
from typing import Any, Mapping, Sequence

from climbing_gym.config import Data as DataConfig
from climbing_gym.config import Env as EnvConfig
from climbing_gym.config import Harness as HarnessConfig
from climbing_gym.config import Inference as InferenceConfig
from climbing_gym.config import Rollout as RolloutConfigSection
from climbing_gym.config.manager import (
    _build_sampling_config,
    _coerce_table,
    _resolve_path,
)
from climbing_gym.inference import (
    LFM25AsyncInferenceEngine,
    SamplingConfig,
)
from climbing_gym.rollout import RolloutCoordinator, Trajectory
from climbing_gym.evaluation_traces import write_trace_report, difficulty_tier
from climbing_gym.preflight import validate_preflight


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _trajectory_passed(trajectory: Trajectory) -> bool:
    feedback = trajectory.evaluation.feedback
    if feedback is not None and feedback.success is not None:
        return bool(feedback.success)
    return float(trajectory.evaluation.reward.reward) > 0.0


def _trajectory_response_length(trajectory: Trajectory) -> int:
    return sum(len(turn.completion_token_ids) for turn in trajectory.turns)


def _trajectory_turn_count(trajectory: Trajectory) -> int:
    return len(trajectory.turns)


def _trajectory_record(trajectory: Trajectory) -> dict[str, Any]:
    final_turn = trajectory.turns[-1] if trajectory.turns else None
    record = {
        "dataset_index": (
            None
            if trajectory.example.dataset_index is None
            else int(trajectory.example.dataset_index)
        ),
        "group_id": int(trajectory.group_id),
        "rollout_index": int(trajectory.rollout_index),
        "reward": float(trajectory.evaluation.reward.reward),
        "passed": bool(_trajectory_passed(trajectory)),
        "response_length": int(_trajectory_response_length(trajectory)),
        "prompt_token_count": sum(len(turn.prompt_token_ids) for turn in trajectory.turns),
        "turn_count": int(_trajectory_turn_count(trajectory)),
        "terminal_reason": str(trajectory.terminal_reason),
        "finish_reason": None if final_turn is None else final_turn.finish_reason,
        "stop_reason": None if final_turn is None else final_turn.stop_reason,
        "finished": bool(trajectory.finished),
        "reward_info": dict(trajectory.evaluation.reward.info),
        "final_response": str(trajectory.final_response),
    }
    record["prompt"] = str(trajectory.example.prompt)
    record["expected_answer"] = trajectory.example.target
    record["difficulty"] = difficulty_tier(trajectory)
    record["turns"] = [
        {"turn_index": int(turn.turn_index), "prompt": str(turn.prompt),
         "response": str(turn.response), "prompt_token_ids": list(turn.prompt_token_ids),
         "completion_token_ids": list(turn.completion_token_ids), "finish_reason": turn.finish_reason,
         "stop_reason": turn.stop_reason}
        for turn in trajectory.turns
    ]
    return record


def summarize_trajectories(
    trajectories: Sequence[Trajectory],
    *,
    pass_k: int,
    num_rollouts: int,
) -> dict[str, float]:
    if pass_k <= 0:
        raise ValueError("pass_k must be > 0")
    if num_rollouts <= 0:
        raise ValueError("num_rollouts must be > 0")
    if pass_k > num_rollouts:
        raise ValueError(
            f"eval.pass_k ({pass_k}) must be <= rollout.num_rollouts ({num_rollouts})."
        )

    grouped: dict[int, list[Trajectory]] = {}
    for trajectory in trajectories:
        dataset_index = trajectory.example.dataset_index
        if dataset_index is None:
            raise ValueError("All eval trajectories must carry example.dataset_index.")
        grouped.setdefault(int(dataset_index), []).append(trajectory)

    if not grouped:
        return {
            "example_count": 0.0,
            "trajectory_count": 0.0,
            "reward_mean": 0.0,
            "avg_response_length": 0.0,
            "avg_turn_count": 0.0,
            "pass_at_1": 0.0,
            f"pass_at_{pass_k}": 0.0,
            f"avg_at_{pass_k}": 0.0,
        }

    all_rewards = [float(trajectory.evaluation.reward.reward) for trajectory in trajectories]
    all_lengths = [float(_trajectory_response_length(trajectory)) for trajectory in trajectories]
    all_turn_counts = [float(_trajectory_turn_count(trajectory)) for trajectory in trajectories]

    pass_at_1_values: list[float] = []
    pass_at_k_values: list[float] = []
    avg_at_k_values: list[float] = []

    for dataset_index in sorted(grouped):
        group = sorted(grouped[dataset_index], key=lambda item: int(item.rollout_index))
        top_1 = group[:1]
        top_k = group[:pass_k]
        pass_at_1_values.append(1.0 if any(_trajectory_passed(item) for item in top_1) else 0.0)
        pass_at_k_values.append(1.0 if any(_trajectory_passed(item) for item in top_k) else 0.0)
        avg_at_k_values.append(
            float(sum(float(item.evaluation.reward.reward) for item in top_k) / len(top_k))
            if top_k
            else 0.0
        )

    summary = {
        "example_count": float(len(grouped)),
        "trajectory_count": float(len(trajectories)),
        "reward_mean": float(sum(all_rewards) / len(all_rewards)),
        "avg_response_length": float(sum(all_lengths) / len(all_lengths)),
        "avg_turn_count": float(sum(all_turn_counts) / len(all_turn_counts)),
        "pass_at_1": float(sum(pass_at_1_values) / len(pass_at_1_values)),
        f"pass_at_{pass_k}": float(sum(pass_at_k_values) / len(pass_at_k_values)),
        f"avg_at_{pass_k}": float(sum(avg_at_k_values) / len(avg_at_k_values)),
    }
    for info_key in (
        "exact_match",
        "strict_success",
        "strict_cfop_trace",
        "phase_fidelity_valid",
        "format_valid",
        "notation_valid",
        "valid_code",
        "position_accuracy",
        "symbol_overlap",
        "clue_consistency",
        "copied_clue_guess",
        "reward_format",
        "base_reward",
        "doomloop_score",
        "doomloop_penalty",
    ):
        values = [
            float(trajectory.evaluation.reward.info[info_key])
            for trajectory in trajectories
            if info_key in trajectory.evaluation.reward.info
        ]
        if values:
            summary[f"reward/{info_key}_mean"] = float(sum(values) / len(values))
    return summary


@dataclass(frozen=True, slots=True)
class EvalLoggingConfig:
    summary_path: str = ""
    trajectory_path: str = ""
    trace_path: str = ""
    max_traces: int = 8

    def __post_init__(self) -> None:
        if self.max_traces < 0:
            raise ValueError("logging.max_traces must be >= 0")


@dataclass(frozen=True, slots=True)
class EvalConfig:
    max_examples: int = 0
    pass_k: int = 1
    seed: int = 0
    shuffle: bool = False

    def __post_init__(self) -> None:
        if self.max_examples < 0:
            raise ValueError("eval.max_examples must be >= 0")
        if self.pass_k <= 0:
            raise ValueError("eval.pass_k must be > 0")


@dataclass(frozen=True, slots=True)
class ResolvedEvalRun:
    rollout_coordinator: RolloutCoordinator
    engine: Any
    eval_config: EvalConfig
    logging_config: EvalLoggingConfig
    total_examples: int
    num_rollouts: int
    close_engine_on_finish: bool = True
    initial_dataset_state: Mapping[str, Any] | None = None
    trace_paths: list[Path] = field(default_factory=list, compare=False)
    preflight: Mapping[str, Any] = field(default_factory=dict, compare=False)


def _resolved_eval_rollout_config(
    rollout: RolloutConfigSection,
    *,
    batch_size: int,
) -> RolloutConfigSection:
    # Evaluation must retain successes and failures alike and visit each row
    # once. Dynamic training sampling discards zero-variance groups.
    return replace(rollout, batch_size=int(batch_size),
                   effective_batch_size=int(batch_size), use_dynamic_sampling=False)



def build_eval_run(
    *,
    engine: Any,
    harness: Any,
    data_config: DataConfig,
    env_config: EnvConfig,
    rollout_config: RolloutConfigSection,
    eval_config: EvalConfig,
    logging_config: EvalLoggingConfig,
    close_engine_on_finish: bool,
) -> ResolvedEvalRun:
    env = env_config.build()
    rows = list(data_config.load_rows())
    if eval_config.shuffle and len(rows) > 1:
        random.Random(eval_config.seed).shuffle(rows)
    if eval_config.max_examples > 0:
        rows = rows[: eval_config.max_examples]
    if not rows:
        raise ValueError("Eval dataset resolved to zero rows.")

    resolved_batch_size = len(rows)
    env.configure_dataset(rows, seed=eval_config.seed, shuffle=False, epoch=0)

    if eval_config.pass_k > int(rollout_config.num_rollouts):
        raise ValueError(
            f"eval.pass_k ({eval_config.pass_k}) must be <= rollout.num_rollouts ({rollout_config.num_rollouts})."
        )

    resolved_rollout = _resolved_eval_rollout_config(
        rollout_config,
        batch_size=resolved_batch_size,
    )
    rollout_coordinator = RolloutCoordinator(
        engine=engine,
        env=env,
        harness=harness,
        config=resolved_rollout.build_coordinator_config(),
    )
    return ResolvedEvalRun(
        rollout_coordinator=rollout_coordinator,
        engine=engine,
        eval_config=EvalConfig(
            max_examples=eval_config.max_examples,
            pass_k=eval_config.pass_k,
            seed=eval_config.seed,
            shuffle=eval_config.shuffle,
        ),
        logging_config=logging_config,
        total_examples=len(rows),
        num_rollouts=int(resolved_rollout.num_rollouts),
        close_engine_on_finish=bool(close_engine_on_finish),
        initial_dataset_state=env.dataset_state_dict(),
    )


def _build_inference_engine(
    config: Any,
    inference_options: Mapping[str, Any],
) -> Any:
    backend = str(inference_options.get("backend", "lfm25_native") or "lfm25_native").strip().lower()
    if backend in {"lfm25", "lfm25_native", "native_lfm25"}:
        return LFM25AsyncInferenceEngine(
            config,
            batch_wait_seconds=float(
                inference_options.get("native_batch_wait_seconds", 0.001)
            ),
            max_batch_size=(
                None
                if inference_options.get("native_max_batch_size") is None
                else int(inference_options["native_max_batch_size"])
            ),
            compile_decode=bool(inference_options.get("compile_decode", True)),
            cuda_graphs=bool(inference_options.get("native_cuda_graphs", False)),
            max_cuda_graphs=int(inference_options.get("native_cuda_max_graphs", 1)),
            max_prefill_tokens=int(inference_options.get("native_prefill_tokens", 4096)),
            bf16_weight_packing=bool(inference_options.get("native_bf16_weight_packing", True)),
            fp8_mlp=bool(inference_options.get("native_fp8_mlp", False)),
            fp8_kv=bool(inference_options.get("native_fp8_kv", False)),
            adaptive_batching=bool(inference_options.get("native_adaptive_batching", False)),
            fp8_lm_head=bool(inference_options.get("native_fp8_lm_head", False)),
        )
    raise ValueError(
        f"Unsupported inference.backend={backend!r}; expected 'lfm25_native'."
    )


def load_run_from_toml(config_path: str | Path) -> ResolvedEvalRun:
    path = Path(_resolve_path(config_path))
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("Eval TOML config must parse to a mapping.")

    base_dir = path.parent
    resolve_path = lambda value: _resolve_path(value, base_dir=base_dir)
    model_config = _coerce_table(payload, "model", default={})
    data_config = _coerce_table(payload, "data")
    env_config = _coerce_table(payload, "env")
    harness_config = _coerce_table(payload, "harness", default={"name": "direct"})
    inference_config = _coerce_table(payload, "inference", default={})
    rollout_config_raw = _coerce_table(payload, "rollout", default={})
    eval_config_raw = _coerce_table(payload, "eval", default={})
    logging_config_raw = _coerce_table(payload, "logging", default={})

    eval_config = EvalConfig(
        max_examples=int(eval_config_raw.get("max_examples", 0)),
        pass_k=int(eval_config_raw.get("pass_k", 1)),
        seed=int(eval_config_raw.get("seed", 0)),
        shuffle=bool(eval_config_raw.get("shuffle", False)),
    )
    if eval_config.max_examples < 0:
        raise ValueError("eval.max_examples must be >= 0")

    logging_config = EvalLoggingConfig(
        summary_path=resolve_path(logging_config_raw.get("summary_path")),
        trajectory_path=resolve_path(logging_config_raw.get("trajectory_path")),
        trace_path=resolve_path(logging_config_raw.get("trace_path")) or str(base_dir / "evaluations" / "traces.md"),
        max_traces=int(logging_config_raw.get("max_traces", 8)),
    )

    model_path = resolve_path(model_config.get("path"))
    engine_model_path = (
        resolve_path(inference_config.get("model_path"))
        or model_path
    )
    if not engine_model_path:
        raise ValueError("Eval requires `model.path` or `inference.model_path`.")

    default_sampling = _build_sampling_config(
        _coerce_table(inference_config, "sampling", default={})
    )
    rollout_sampling = _build_sampling_config(
        _coerce_table(rollout_config_raw, "sampling", default={})
    )

    resolved_inference_config = InferenceConfig(
        backend=str(inference_config.get("backend", "lfm25_native")),
        preflight_report_path=resolve_path(inference_config.get("preflight_report_path")),
        preflight_prior_report_path=resolve_path(inference_config.get("preflight_prior_report_path")),
        model_path=engine_model_path,
        dtype=str(inference_config.get("dtype", "auto") or "auto"),
        seed=int(inference_config.get("seed", eval_config.seed)),
        gpu_memory_utilization=float(inference_config.get("gpu_memory_utilization", 0.9)),
        max_model_len=(
            None
            if inference_config.get("max_model_len") is None
            else int(inference_config.get("max_model_len"))
        ),
        max_num_seqs=(
            None
            if inference_config.get("max_num_seqs") is None
            else int(inference_config.get("max_num_seqs"))
        ),
        enforce_eager=bool(inference_config.get("enforce_eager", False)),
        enable_sleep_mode=bool(inference_config.get("enable_sleep_mode", True)),
        apply_chat_template=bool(inference_config.get("apply_chat_template", False)),
        chat_template_enable_thinking=bool(inference_config.get("chat_template_enable_thinking", True)),
        assistant_prefill=str(inference_config.get("assistant_prefill", "") or ""),
        sampling=default_sampling or SamplingConfig(),
        staged_checkpoint_root=Path(resolve_path(inference_config.get("staged_checkpoint_root"))
                                    or Path.home() / ".cache" / "climbing-gym" / "checkpoints"),
        **{key: inference_config[key] for key in ("native_batch_wait_seconds", "native_max_batch_size", "native_cuda_graphs", "native_cuda_max_graphs",
           "native_prefill_tokens", "native_bf16_weight_packing", "compile_decode",
           "native_fp8_mlp", "native_fp8_kv", "native_fp8_lm_head", "native_adaptive_batching") if key in inference_config},
    )
    engine = _build_inference_engine(
        resolved_inference_config.build_engine_config(),
        inference_config,
    )

    try:
        run = build_eval_run(
            engine=engine,
            harness=HarnessConfig(
                name=str(harness_config.get("name", "direct") or "direct"),
                args={key: value for key, value in harness_config.items() if key != "name"},
            ).build(),
            data_config=DataConfig(
                dataset_name=str(data_config.get("dataset_name", "") or ""),
                dataset_config=str(data_config.get("dataset_config", "") or ""),
                dataset_path=resolve_path(data_config.get("dataset_path")),
                split=str(data_config.get("split", "train") or "train"),
                max_rows=int(data_config.get("max_rows", 0) or 0),
                seed=int(data_config.get("seed", eval_config.seed)),
                shuffle=bool(data_config.get("shuffle", True)),
                epoch=int(data_config.get("epoch", 0)),
            ),
            env_config=EnvConfig(
                name=str(env_config.get("name", "") or ""),
                reward=str(env_config.get("reward", "exact_match") or "exact_match"),
                args={
                    key: value
                    for key, value in env_config.items()
                    if key not in {"name", "reward"}
                },
            ),
            rollout_config=RolloutConfigSection(
                num_rollouts=int(rollout_config_raw.get("num_rollouts", 1)),
                sampling=rollout_sampling or default_sampling,
                max_total_requests=(
                    None
                    if rollout_config_raw.get("max_total_requests") is None
                    else int(rollout_config_raw.get("max_total_requests"))
                ),
                batch_size=1,
                effective_batch_size=1,
                max_inflight_requests=(
                    None
                    if rollout_config_raw.get("max_inflight_requests") is None
                    else int(rollout_config_raw.get("max_inflight_requests"))
                ),
                reward_std_epsilon=float(rollout_config_raw.get("reward_std_epsilon", 1e-8)),
                use_dynamic_sampling=bool(rollout_config_raw.get("use_dynamic_sampling", False)),
                stall_timeout_seconds=float(rollout_config_raw.get("stall_timeout_seconds", 120.0)),
                progress_log_interval_seconds=float(
                    rollout_config_raw.get("progress_log_interval_seconds", 15.0)
                ),
                engine_sleep_level=int(rollout_config_raw.get("engine_sleep_level", 1)),
                engine_sleep_mode=str(rollout_config_raw.get("engine_sleep_mode", "abort")),
            ),
            eval_config=eval_config,
            logging_config=logging_config,
            close_engine_on_finish=True,
        )
    except Exception:
        engine.close()
        raise

    # The engine has only been constructed; no model inference has run yet.
    # Resolve actual evaluation row count before validating concurrency.
    from types import SimpleNamespace
    coordinator = run.rollout_coordinator.config
    resolved_rollout = SimpleNamespace(
        sampling=coordinator.sampling, num_rollouts=coordinator.num_rollouts,
        batch_size=coordinator.batch_size,
        effective_batch_size=coordinator.resolved_runner().effective_batch_size,
        max_inflight_requests=coordinator.resolved_runner().max_inflight_requests,
        max_total_requests=coordinator.max_total_requests,
    )
    try:
        report = validate_preflight(SimpleNamespace(inference=resolved_inference_config, rollout=resolved_rollout))
        return replace(run, preflight=report)
    except Exception:
        engine.close()
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run climbing_gym evaluation from a single TOML config.")
    parser.add_argument("config", nargs="?", type=str, help="Path to the eval TOML config.")
    parser.add_argument("--config", dest="config_option", help="Path to the eval TOML config.")
    return parser


def _write_summary(path: str, summary: Mapping[str, Any]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )


def _write_trajectories(path: str, trajectories: Sequence[Trajectory]) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for trajectory in trajectories:
            handle.write(json.dumps(_trajectory_record(trajectory), sort_keys=True, default=_json_default) + "\n")


def run_eval(run: ResolvedEvalRun) -> dict[str, float]:
    start_time = time.perf_counter()
    all_trajectories: list[Trajectory] = []
    last_rollout_stats: dict[str, float] = {}

    try:
        if run.initial_dataset_state is not None:
            run.rollout_coordinator.env.load_dataset_state_dict(dict(run.initial_dataset_state))
        batch = run.rollout_coordinator.collect()
        all_trajectories.extend(batch.trajectories)
        last_rollout_stats = dict(batch.stats)
    finally:
        if run.close_engine_on_finish:
            run.engine.close()

    summary = summarize_trajectories(
        all_trajectories,
        pass_k=run.eval_config.pass_k,
        num_rollouts=run.num_rollouts,
    )
    summary["runtime_seconds"] = float(time.perf_counter() - start_time)
    summary["requested_examples"] = float(run.total_examples)
    summary["num_rollouts"] = float(run.num_rollouts)
    summary["rollout_batch_size"] = float(run.total_examples)
    if last_rollout_stats:
        for key, value in last_rollout_stats.items():
            summary[f"rollout/{key}"] = float(value)

    # Always preserve a complete actual trace report, including for periodic
    # evaluations with no explicitly configured JSON output paths.
    configured = run.logging_config.trace_path
    if not configured:
        anchor = run.logging_config.trajectory_path or run.logging_config.summary_path
        configured = str(Path(anchor).with_suffix(".traces.md")) if anchor else "evaluations/traces.md"
    trace_path = write_trace_report(configured, all_trajectories, summary=summary,
                                   max_traces=run.logging_config.max_traces,
                                   tokenizer=getattr(run.engine, "tokenizer", None))
    run.trace_paths.append(trace_path)
    _write_summary(run.logging_config.summary_path, {**summary, "trace_path": str(trace_path),
                                                    "preflight": dict(run.preflight)})
    _write_trajectories(run.logging_config.trajectory_path, all_trajectories)
    return summary


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    config_path = args.config_option or args.config
    if not config_path:
        build_arg_parser().error("a config path or --config is required")
    run = load_run_from_toml(config_path)
    summary = run_eval(run)
    print(json.dumps({**summary, "trace_path": str(run.trace_paths[-1])}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
