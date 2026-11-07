from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import os
import re
import tomllib
from typing import Any, Mapping

import torch

from climbing_gym.config.job import (
    Data,
    Env,
    Eval,
    EvalLogging,
    Harness,
    Inference,
    Job,
    Model,
    Optimizer,
    PPOConfig,
    PPOTrainer,
    Resume,
    Rollout,
    TrainLogging,
    Wandb,
)
from climbing_gym.inference.config import DEFAULT_STAGED_CHECKPOINT_ROOT, SamplingConfig
from climbing_gym.trainers.checkpoints import CheckpointManagerConfig


@dataclass(frozen=True, slots=True)
class ConfigManager:
    config: PPOConfig
    config_path: Path
    raw_config: dict[str, Any]

    @classmethod
    def from_toml(cls, config_path: str | Path) -> "ConfigManager":
        path = Path(_resolve_path(config_path))
        with path.open("rb") as handle:
            payload = tomllib.load(handle)
        if not isinstance(payload, Mapping):
            raise TypeError("PPO TOML config must parse to a mapping.")
        raw_config = dict(payload)
        resolve_path = lambda value: _resolve_path(value, base_dir=path.parent)

        trainer_table = dict(_coerce_table(raw_config, "trainer", default={}))
        if trainer_table.get("value_optimizer_backing_path"):
            trainer_table["value_optimizer_backing_path"] = _resolve_path(trainer_table["value_optimizer_backing_path"], base_dir=path.parent)
        trainer_table["train_dtype"] = _resolve_torch_dtype(trainer_table.get("train_dtype"))
        trainer = PPOTrainer(**trainer_table)

        model_table = _coerce_table(raw_config, "model")
        inference_table = _coerce_table(raw_config, "inference")
        rollout_table = _coerce_table(raw_config, "rollout")
        env_table = _coerce_table(raw_config, "env")
        harness_table = _coerce_table(raw_config, "harness", default={"name": "direct"})
        data_table = _coerce_table(raw_config, "data")
        checkpoint_table = _coerce_table(raw_config, "checkpoint", default={})
        resume_table = _coerce_table(raw_config, "resume", default={})
        wandb_table = _coerce_table(raw_config, "wandb", default={})
        train_logging_table = _coerce_table(raw_config, "train_logging", default={})
        eval_table = _coerce_table(raw_config, "eval", default={})
        job_table = _coerce_table(raw_config, "job", default={})

        model_path = resolve_path(model_table.get("path"))
        resume_checkpoint_dir = resolve_path(resume_table.get("checkpoint_dir"))
        inferred_resume_policy_path = (
            str((Path(resume_checkpoint_dir) / "policy").resolve())
            if resume_checkpoint_dir
            else ""
        )
        inference_model_path = (
            resolve_path(inference_table.get("model_path"))
            or inferred_resume_policy_path
            or model_path
        )
        default_sampling = _build_sampling_config(_coerce_table(inference_table, "sampling", default={})) or SamplingConfig()
        rollout_sampling = _build_sampling_config(_coerce_table(rollout_table, "sampling", default={}))
        eval_data_table = _coerce_table(eval_table, "data", default=data_table)
        eval_env_table = _coerce_table(eval_table, "env", default=env_table)
        eval_rollout_table = _coerce_table(eval_table, "rollout", default=rollout_table)
        eval_logging_table = _coerce_table(eval_table, "logging", default={})
        eval_rollout_sampling = _build_sampling_config(_coerce_table(eval_rollout_table, "sampling", default={}))

        config = PPOConfig(
            job=Job(
                config_file=str(path),
                description=str(job_table.get("description", "climbing_gym PPO run") or "climbing_gym PPO run"),
                print_config=bool(job_table.get("print_config", False)),
            ),
            model=Model(
                name=str(model_table.get("name", "") or ""),
                path=model_path,
                config=_coerce_table(model_table, "config", default={}),
            ),
            data=Data(
                dataset_name=str(data_table.get("dataset_name", "") or ""),
                dataset_config=str(data_table.get("dataset_config", "") or ""),
                dataset_path=resolve_path(data_table.get("dataset_path")),
                split=str(data_table.get("split", "train") or "train"),
                max_rows=int(data_table.get("max_rows", 0) or 0),
                seed=int(data_table.get("seed", trainer.seed)),
                shuffle=bool(data_table.get("shuffle", True)),
                epoch=int(data_table.get("epoch", 0)),
            ),
            env=Env(
                name=str(env_table.get("name", "") or ""),
                reward=str(env_table.get("reward", "exact_match") or "exact_match"),
                args={
                    key: value
                    for key, value in env_table.items()
                    if key not in {"name", "reward"}
                },
            ),
            harness=Harness(
                name=str(harness_table.get("name", "direct") or "direct"),
                args={
                    key: value
                    for key, value in harness_table.items()
                    if key != "name"
                },
            ),
            inference=Inference(
                backend=str(inference_table.get("backend", "lfm25_native") or "lfm25_native"),
                model_path=inference_model_path,
                preflight_report_path=resolve_path(inference_table.get("preflight_report_path")),
                preflight_prior_report_path=resolve_path(inference_table.get("preflight_prior_report_path")),
                dtype=str(inference_table.get("dtype", "auto") or "auto"),
                seed=int(inference_table.get("seed", trainer.seed)),
                gpu_memory_utilization=float(inference_table.get("gpu_memory_utilization", 0.9)),
                max_model_len=(
                    None
                    if inference_table.get("max_model_len") is None
                    else int(inference_table.get("max_model_len"))
                ),
                max_num_seqs=(
                    None
                    if inference_table.get("max_num_seqs") is None
                    else int(inference_table.get("max_num_seqs"))
                ),
                enforce_eager=bool(inference_table.get("enforce_eager", False)),
                enable_sleep_mode=bool(inference_table.get("enable_sleep_mode", True)),
                apply_chat_template=bool(inference_table.get("apply_chat_template", False)),
                chat_template_enable_thinking=bool(inference_table.get("chat_template_enable_thinking", True)),
                assistant_prefill=str(inference_table.get("assistant_prefill", "") or ""),
                native_batch_wait_seconds=float(
                    inference_table.get("native_batch_wait_seconds", 0.001)
                ),
                native_max_batch_size=(
                    None
                    if inference_table.get("native_max_batch_size") is None
                    else int(inference_table.get("native_max_batch_size"))
                ),
                compile_decode=bool(inference_table.get("compile_decode", True)),
                native_cuda_graphs=bool(inference_table.get("native_cuda_graphs", False)),
                native_cuda_max_graphs=int(inference_table.get("native_cuda_max_graphs", 1)),
                native_prefill_tokens=int(inference_table.get("native_prefill_tokens", 4096)),
                native_bf16_weight_packing=bool(inference_table.get("native_bf16_weight_packing", True)),
                native_fp8_mlp=bool(inference_table.get("native_fp8_mlp", False)),
                native_fp8_kv=bool(inference_table.get("native_fp8_kv", False)),
                native_adaptive_batching=bool(inference_table.get("native_adaptive_batching", False)),
                native_fp8_lm_head=bool(inference_table.get("native_fp8_lm_head", False)),
                sampling=default_sampling,
                staged_checkpoint_root=Path(
                    resolve_path(inference_table.get("staged_checkpoint_root"))
                    or DEFAULT_STAGED_CHECKPOINT_ROOT
                ),
            ),
            rollout=Rollout(
                num_rollouts=int(rollout_table.get("num_rollouts", 2)),
                sampling=rollout_sampling,
                max_total_requests=(
                    None
                    if rollout_table.get("max_total_requests") is None
                    else int(rollout_table.get("max_total_requests"))
                ),
                batch_size=int(rollout_table.get("batch_size", 1)),
                effective_batch_size=int(rollout_table.get("effective_batch_size", rollout_table.get("batch_size", 1))),
                max_inflight_requests=(
                    None
                    if rollout_table.get("max_inflight_requests") is None
                    else int(rollout_table.get("max_inflight_requests"))
                ),
                reward_std_epsilon=float(rollout_table.get("reward_std_epsilon", 1e-8)),
                use_dynamic_sampling=bool(rollout_table.get("use_dynamic_sampling", True)),
                stall_timeout_seconds=float(rollout_table.get("stall_timeout_seconds", 120.0)),
                progress_log_interval_seconds=float(rollout_table.get("progress_log_interval_seconds", 15.0)),
                engine_sleep_level=int(rollout_table.get("engine_sleep_level", 1)),
                engine_sleep_mode=str(rollout_table.get("engine_sleep_mode", "abort") or "abort"),
            ),
            trainer=trainer,
            policy_optimizer=Optimizer(
                type=str(_coerce_table(raw_config, "policy_optimizer", default={}).get("type", "adamw") or "adamw"),
                learning_rate=float(_coerce_table(raw_config, "policy_optimizer", default={}).get("learning_rate", 1e-6)),
                weight_decay=float(_coerce_table(raw_config, "policy_optimizer", default={}).get("weight_decay", 0.0)),
                fused=_coerce_optional_bool(_coerce_table(raw_config, "policy_optimizer", default={}).get("fused")),
            ),
            value_optimizer=Optimizer(
                type=str(_coerce_table(raw_config, "value_optimizer", default={}).get("type", "adamw") or "adamw"),
                learning_rate=float(
                    _coerce_table(raw_config, "value_optimizer", default={}).get("learning_rate", trainer.value_learning_rate)
                ),
                weight_decay=float(
                    _coerce_table(raw_config, "value_optimizer", default={}).get("weight_decay", trainer.value_weight_decay)
                ),
                fused=_coerce_optional_bool(_coerce_table(raw_config, "value_optimizer", default={}).get("fused")),
            ),
            checkpoint=CheckpointManagerConfig(
                root_dir=resolve_path(checkpoint_table.get("root_dir")),
                save_interval=int(checkpoint_table.get("save_interval", 0)),
                save_final=bool(checkpoint_table.get("save_final", True)),
            ),
            resume=Resume(
                checkpoint_dir=resume_checkpoint_dir,
                continue_step_numbers=bool(resume_table.get("continue_step_numbers", True)),
                allow_weights_only=bool(resume_table.get("allow_weights_only", False)),
            ),
            wandb=Wandb(
                enabled=bool(wandb_table.get("enabled", True)),
                project=str(wandb_table.get("project", "climbing-gym") or "climbing-gym"),
                entity=str(wandb_table.get("entity", "") or "").strip() or None,
                dir=resolve_path(wandb_table.get("dir")),
                run_name=str(wandb_table.get("run_name", "") or "").strip() or None,
                mode=str(wandb_table.get("mode", "online") or "online").strip() or None,
            ),
            train_logging=TrainLogging(
                trajectory_path=resolve_path(train_logging_table.get("trajectory_path")),
                metrics_path=resolve_path(train_logging_table.get("metrics_path")),
            ),
            eval=Eval(
                preflight_report_path=resolve_path(eval_table.get("preflight_report_path")),
                preflight_prior_report_path=resolve_path(eval_table.get("preflight_prior_report_path")),
                interval=int(eval_table.get("interval", 0) or 0),
                max_examples=int(eval_table.get("max_examples", 0) or 0),
                pass_k=int(eval_table.get("pass_k", 1) or 1),
                seed=int(eval_table.get("seed", trainer.seed)),
                shuffle=bool(eval_table.get("shuffle", False)),
                data=Data(
                    dataset_name=str(eval_data_table.get("dataset_name", "") or ""),
                    dataset_config=str(eval_data_table.get("dataset_config", "") or ""),
                    dataset_path=resolve_path(eval_data_table.get("dataset_path")),
                    split=str(eval_data_table.get("split", "train") or "train"),
                    max_rows=int(eval_data_table.get("max_rows", 0) or 0),
                    seed=int(eval_data_table.get("seed", eval_table.get("seed", trainer.seed))),
                    shuffle=bool(eval_data_table.get("shuffle", eval_table.get("shuffle", False))),
                    epoch=int(eval_data_table.get("epoch", 0)),
                ),
                env=Env(
                    name=str(eval_env_table.get("name", "") or ""),
                    reward=str(eval_env_table.get("reward", "exact_match") or "exact_match"),
                    args={
                        key: value
                        for key, value in eval_env_table.items()
                        if key not in {"name", "reward"}
                    },
                ),
                rollout=Rollout(
                    num_rollouts=int(eval_rollout_table.get("num_rollouts", 1)),
                    sampling=eval_rollout_sampling,
                    max_total_requests=(
                        None
                        if eval_rollout_table.get("max_total_requests") is None
                        else int(eval_rollout_table.get("max_total_requests"))
                    ),
                    batch_size=1,
                    effective_batch_size=int(
                        eval_rollout_table.get(
                            "effective_batch_size",
                            1,
                        )
                    ),
                    max_inflight_requests=(
                        None
                        if eval_rollout_table.get("max_inflight_requests") is None
                        else int(eval_rollout_table.get("max_inflight_requests"))
                    ),
                    reward_std_epsilon=float(eval_rollout_table.get("reward_std_epsilon", 1e-8)),
                    use_dynamic_sampling=bool(eval_rollout_table.get("use_dynamic_sampling", False)),
                    stall_timeout_seconds=float(eval_rollout_table.get("stall_timeout_seconds", 120.0)),
                    progress_log_interval_seconds=float(
                        eval_rollout_table.get("progress_log_interval_seconds", 15.0)
                    ),
                    engine_sleep_level=int(eval_rollout_table.get("engine_sleep_level", 1)),
                    engine_sleep_mode=str(eval_rollout_table.get("engine_sleep_mode", "abort") or "abort"),
                ),
                logging=EvalLogging(
                    summary_path=resolve_path(eval_logging_table.get("summary_path")),
                    trajectory_path=resolve_path(eval_logging_table.get("trajectory_path")),
                    trace_path=resolve_path(eval_logging_table.get("trace_path")),
                    max_traces=int(eval_logging_table.get("max_traces", 8)),
                ),
            ),
        )
        return cls(config=config, config_path=path, raw_config=raw_config)

    def to_dict(self) -> dict[str, Any]:
        return self.config.to_dict()


def _coerce_table(
    payload: Mapping[str, Any],
    key: str,
    *,
    default: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    value = payload.get(key, default if default is not None else {})
    if not isinstance(value, Mapping):
        raise TypeError(f"`{key}` must be a TOML table.")
    return dict(value)


def _resolve_path(raw: str | Path | None, *, base_dir: Path | None = None) -> str:
    if raw is None or not str(raw).strip():
        return ""
    text = str(raw).strip()
    variables = re.findall(r"\$(?:\{([A-Za-z_][A-Za-z0-9_]*)\}|([A-Za-z_][A-Za-z0-9_]*))", text)
    missing = sorted({a or b for a, b in variables if (a or b) not in os.environ})
    if missing:
        raise ValueError(f"Unset environment variable(s) in path: {', '.join(missing)}")
    path = Path(os.path.expandvars(text)).expanduser()
    if not path.is_absolute():
        path = (base_dir or Path.cwd()) / path
    return str(path.resolve())


def _resolve_torch_dtype(raw: Any) -> torch.dtype | None:
    if raw is None:
        return None
    value = str(raw).strip().lower()
    if not value or value == "auto":
        return None
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "float": torch.float32,
        "float16": torch.float16,
        "fp16": torch.float16,
        "half": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
    }
    try:
        return mapping[value]
    except KeyError as exc:
        raise ValueError(f"Unsupported torch dtype: {raw!r}") from exc


def _build_sampling_config(payload: Mapping[str, Any] | None) -> SamplingConfig | None:
    if payload is None:
        return None
    data = dict(payload)
    if not data:
        return None
    extra_params = data.pop("extra_params", {})
    if extra_params is None:
        extra_params = {}
    if not isinstance(extra_params, Mapping):
        raise TypeError("`extra_params` must be a TOML table when provided.")
    logprobs = data.pop("logprobs", None)
    return SamplingConfig(
        temperature=float(data.pop("temperature", 0.0)),
        top_k=int(data.pop("top_k", -1)),
        top_p=float(data.pop("top_p", 1.0)),
        max_tokens=int(data.pop("max_tokens", 256)),
        n=int(data.pop("n", 1)),
        stop=tuple(str(item) for item in data.pop("stop", ()) or ()),
        logprobs=None if logprobs is None else int(logprobs),
        extra_params=dict(extra_params),
    )


def _coerce_optional_bool(value: Any) -> bool | None:
    if value is None:
        return None
    return bool(value)


__all__ = ["ConfigManager"]
