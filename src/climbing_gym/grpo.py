from __future__ import annotations

import argparse
from dataclasses import dataclass, replace
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Callable, Mapping, Sequence

import torch
from torch import nn

from climbing_gym.config import (
    ConfigManager,
    PPOConfig,
)
from climbing_gym.eval import (
    EvalConfig as PeriodicEvalConfig,
    EvalLoggingConfig,
    ResolvedEvalRun,
    _trajectory_record,
    build_eval_run,
    run_eval,
)
from climbing_gym.inference import LFM25AsyncInferenceEngine
from climbing_gym.models import build_model
from climbing_gym.rollout import RolloutCoordinator, Trajectory, TrajectoryBatch
from climbing_gym.trainers.checkpoints import CheckpointManager
from climbing_gym.trainers.common import collect_rollout_batch, compute_grad_norm, offload_to_cpu, reload_to_device
from climbing_gym.trainers.logger import WandbLogger
from climbing_gym.trainers.packing import Batch, pack_trajectories


def scale_grpo_loss_for_packed_batch(
    loss: torch.Tensor,
    *,
    sequence_count: float | int | torch.Tensor,
    total_sequence_count: float | int,
) -> torch.Tensor:
    total = float(total_sequence_count)
    if total <= 0.0:
        return loss * 0.0

    batch_sequences = (
        float(sequence_count.detach().item())
        if isinstance(sequence_count, torch.Tensor)
        else float(sequence_count)
    )
    if batch_sequences <= 0.0:
        return loss * 0.0
    return loss * (batch_sequences / total)


def maybe_compile(
    target: Any,
    *,
    dynamic: bool,
    fullgraph: bool,
    backend: str,
    mode: str,
) -> Any:
    return torch.compile(
        target,
        dynamic=dynamic,
        fullgraph=fullgraph,
        backend=backend,
        mode=mode,
    )


def compute_dr_grpo_advantages(
    trajectory_rewards: Sequence[float],
    *,
    group_ids: Sequence[int],
    eps: float,
) -> tuple[list[float], dict[str, float]]:
    if not trajectory_rewards:
        return [], {
            "grpo_group_reward_mean": 0.0,
            "grpo_group_reward_std_mean": 0.0,
            "grpo_zero_std_prompt_rate": 0.0,
        }
    if len(group_ids) != len(trajectory_rewards):
        raise ValueError(
            "group_ids length must match trajectory_rewards length. "
            f"Got len(group_ids)={len(group_ids)} and len(trajectory_rewards)={len(trajectory_rewards)}."
        )

    group_to_indices: dict[int, list[int]] = {}
    for idx, group_id in enumerate(group_ids):
        group_to_indices.setdefault(int(group_id), []).append(idx)

    advantages = [0.0] * len(trajectory_rewards)
    group_means: list[float] = []
    group_stds: list[float] = []
    zero_std_count = 0

    for indices in group_to_indices.values():
        grouped = torch.tensor([trajectory_rewards[idx] for idx in indices], dtype=torch.float32)
        group_mean = grouped.mean()
        group_std = grouped.std(unbiased=False)
        centered = grouped - group_mean

        group_means.append(float(group_mean.item()))
        group_stds.append(float(group_std.item()))
        if float(group_std.item()) <= float(eps):
            zero_std_count += 1

        for local_idx, trajectory_idx in enumerate(indices):
            advantages[trajectory_idx] = float(centered[local_idx].item())

    return advantages, {
        "grpo_group_reward_mean": float(sum(group_means) / max(1, len(group_means))),
        "grpo_group_reward_std_mean": float(sum(group_stds) / max(1, len(group_stds))),
        "grpo_zero_std_prompt_rate": float(zero_std_count / max(1, len(group_stds))),
    }


def compute_sa_mrpo_advantages(
    objective_rewards: Mapping[str, Sequence[float]],
    *,
    objective_weights: Mapping[str, float],
    gamma: float,
    group_ids: Sequence[int],
    eps: float,
) -> tuple[list[float], dict[str, float]]:
    """Independently normalize objectives by prompt, then combine and normalize.

    Binary objective saturation is its active-batch mean. Saturation-aware weights
    keep an already-learned formatting objective from overwhelming sparse accuracy.
    """

    objective_names = list(objective_rewards)
    if not objective_names:
        raise ValueError("SA-MRPO requires at least one objective")
    if set(objective_names) != set(objective_weights):
        raise ValueError("SA-MRPO objective rewards and weights must have identical keys")
    lengths = {name: len(objective_rewards[name]) for name in objective_names}
    trajectory_count = lengths[objective_names[0]]
    if any(length != trajectory_count for length in lengths.values()):
        raise ValueError(f"SA-MRPO objective lengths differ: {lengths}")
    if len(group_ids) != trajectory_count:
        raise ValueError("SA-MRPO group_ids length must match objective rewards")
    if not math.isfinite(float(gamma)) or float(gamma) < 0.0:
        raise ValueError("SA-MRPO gamma must be finite and non-negative")

    group_to_indices: dict[int, list[int]] = {}
    for index, group_id in enumerate(group_ids):
        group_to_indices.setdefault(int(group_id), []).append(index)

    combined = torch.zeros(trajectory_count, dtype=torch.float64)
    metrics: dict[str, float] = {"sa_mrpo_gamma": float(gamma)}
    for objective_name in objective_names:
        weight = float(objective_weights[objective_name])
        if not math.isfinite(weight) or weight < 0.0:
            raise ValueError(f"Invalid SA-MRPO weight for {objective_name!r}: {weight}")
        rewards = torch.tensor(objective_rewards[objective_name], dtype=torch.float64)
        normalized = torch.zeros_like(rewards)
        group_stds: list[float] = []
        zero_std_count = 0
        for indices in group_to_indices.values():
            index_tensor = torch.tensor(indices, dtype=torch.long)
            grouped = rewards[index_tensor]
            group_std = grouped.std(unbiased=False)
            group_stds.append(float(group_std.item()))
            if float(group_std.item()) <= float(eps):
                zero_std_count += 1
                continue
            normalized[index_tensor] = (grouped - grouped.mean()) / group_std

        saturation = float(rewards.mean().item()) if trajectory_count else 0.0
        saturation = min(1.0, max(0.0, saturation))
        effective_weight = weight * ((1.0 - saturation) ** float(gamma))
        combined += effective_weight * normalized
        metric_name = "".join(
            character if character.isalnum() else "_"
            for character in objective_name.strip().lower()
        ).strip("_")
        metrics.update(
            {
                f"sa_mrpo_{metric_name}_reward_mean": saturation,
                f"sa_mrpo_{metric_name}_group_reward_std_mean": (
                    float(sum(group_stds) / len(group_stds)) if group_stds else 0.0
                ),
                f"sa_mrpo_{metric_name}_zero_std_prompt_rate": (
                    float(zero_std_count / len(group_stds)) if group_stds else 0.0
                ),
                f"sa_mrpo_{metric_name}_saturation": saturation,
                f"sa_mrpo_{metric_name}_effective_weight": effective_weight,
            }
        )

    combined_std = combined.std(unbiased=False) if trajectory_count else torch.tensor(0.0)
    if trajectory_count and float(combined_std.item()) > float(eps):
        combined = (combined - combined.mean()) / combined_std
    else:
        combined.zero_()
    metrics["sa_mrpo_advantage_std"] = (
        float(combined.std(unbiased=False).item()) if trajectory_count else 0.0
    )
    return combined.tolist(), metrics


def weight_trajectory_advantages(trajectories, advantages):
    """Apply explicit objective weights after centering, without changing rewards."""
    if len(trajectories) != len(advantages):
        raise ValueError("Trajectory/advantage lengths differ")
    result = []
    for trajectory, advantage in zip(trajectories, advantages):
        weight = float(trajectory.metadata.get("policy_objective_weight", 1.0))
        if not math.isfinite(weight) or weight < 0:
            raise ValueError("Policy objective weights must be finite and nonnegative")
        result.append(float(advantage) * weight)
    return result


def _build_sequence_scalar_payloads(
    trajectories: Sequence[Trajectory],
    *,
    trajectory_values: Sequence[float],
) -> dict[tuple[int, int], tuple[float, ...]]:
    sequence_payloads: dict[tuple[int, int], tuple[float, ...]] = {}
    for trajectory_index, trajectory in enumerate(trajectories):
        scalar_value = float(trajectory_values[trajectory_index])
        for turn in trajectory.turns:
            completion_len = len(turn.completion_token_ids)
            if completion_len <= 0:
                continue
            sequence_payloads[(int(trajectory_index), int(turn.turn_index))] = tuple(
                scalar_value for _ in range(completion_len)
            )
    return sequence_payloads


def _append_training_trajectories(
    path: str,
    *,
    step: int,
    trajectories: Sequence[Trajectory],
) -> None:
    if not path:
        return
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("a", encoding="utf-8") as handle:
        for trajectory in trajectories:
            record = _trajectory_record(trajectory)
            record["step"] = int(step)
            handle.write(json.dumps(record, sort_keys=True) + "\n")


def _build_policy_optimizer(
    policy_model: nn.Module,
    optimizer_config: Mapping[str, Any],
    *, trainer: Any | None = None,
) -> torch.optim.Optimizer:
    optimizer_name = str(optimizer_config.get("type", "adamw") or "adamw").strip().lower()
    # Actor parameters are normally all trainable. A frozen-token-mixer critic
    # must exclude its frozen tensors before host/mmap optimizers eagerly
    # initialize masters and moments for every registered parameter.
    parameters = [parameter for parameter in policy_model.parameters() if parameter.requires_grad]
    if optimizer_name in {"master_adamw", "cpu_master_adamw"}:
        if optimizer_name == "cpu_master_adamw":
            from climbing_gym.cpu_master_adamw import CPUStateMasterAdamW
            factory = CPUStateMasterAdamW
            extra = dict(chunk_elements=getattr(trainer, "optimizer_chunk_elements", 4194304),
                         workspace_bytes=getattr(trainer, "optimizer_workspace_bytes", 50331648))
        else:
            from climbing_gym.ppo import MasterWeightAdamW
            factory, extra = MasterWeightAdamW, {}
        return factory(parameters, lr=float(optimizer_config.get("learning_rate", 1e-6)),
                       weight_decay=float(optimizer_config.get("weight_decay", 0.0)), **extra)
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported policy optimizer type: {optimizer_name!r}")
    return torch.optim.AdamW(
        parameters,
        lr=float(optimizer_config.get("learning_rate", 1e-6)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        fused=bool(torch.cuda.is_available() if optimizer_config.get("fused") is None else optimizer_config["fused"]),
    )


def _load_resume_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Resume checkpoint metadata must parse to a mapping.")
    return dict(payload)


def _build_completion_targets(
    batch: Batch,
    *,
    sequence_payloads: Mapping[tuple[int, int], tuple[float, ...]],
) -> torch.Tensor:
    return batch.completion_value_tensor(
        [
            sequence_payloads[(int(batch.trajectory_indices[idx]), int(batch.turn_indices[idx]))]
            for idx in range(batch.sequence_count)
        ]
    )


def packed_grpo_policy_loss(
    *,
    target_log_probs: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    token_advantages: torch.Tensor,
    valid_mask: torch.Tensor,
    sequence_ids: torch.Tensor,
    sequence_count: int,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    valid = valid_mask.to(dtype=torch.float32)
    valid_count = valid.sum().clamp_min(1.0)

    log_ratio = target_log_probs.to(dtype=torch.float32) - old_infer_logprobs.to(dtype=torch.float32)
    ratio = torch.exp(log_ratio)
    clipped_ratio = torch.clamp(ratio, min=1.0 - float(clip_epsilon_low), max=1.0 + float(clip_epsilon_high))
    surrogate = torch.minimum(ratio * token_advantages.to(dtype=torch.float32), clipped_ratio * token_advantages.to(dtype=torch.float32))
    token_objective = surrogate * valid

    if int(sequence_count) <= 0:
        loss = token_objective.sum() * 0.0
    else:
        sequence_objective = torch.zeros(int(sequence_count), device=token_objective.device, dtype=token_objective.dtype)
        sequence_objective.scatter_add_(0, sequence_ids.to(device=token_objective.device, dtype=torch.long), token_objective)
        sequence_valid_counts = torch.zeros(int(sequence_count), device=valid.device, dtype=valid.dtype)
        sequence_valid_counts.scatter_add_(0, sequence_ids.to(device=valid.device, dtype=torch.long), valid)
        loss = -(sequence_objective / sequence_valid_counts.clamp_min(1.0)).mean()

    clip_fraction = (((ratio - clipped_ratio).abs() > 1e-6).to(dtype=torch.float32) * valid).sum() / valid_count
    mean_ratio = (ratio * valid).sum() / valid_count
    approx_kl = ((((ratio - 1.0) - log_ratio) * valid).sum()) / valid_count
    return loss, clip_fraction, mean_ratio, approx_kl, valid_count


def packed_grpo_policy_loss_from_model(
    policy_model: Any,
    *,
    batch: Batch,
    token_advantages: torch.Tensor,
    old_infer_logprobs: torch.Tensor,
    clip_epsilon_low: float,
    clip_epsilon_high: float,
    entropy_coef: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = policy_model(batch.inputs, positions=batch.position_ids)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=batch.targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (target_logits - lse).squeeze(0).to(dtype=torch.float32)

    valid_mask = batch.valid_completion_mask()
    policy_loss, clip_fraction, mean_ratio, approx_kl, valid_count = packed_grpo_policy_loss(
        target_log_probs=target_log_probs,
        old_infer_logprobs=old_infer_logprobs,
        token_advantages=token_advantages,
        valid_mask=valid_mask,
        sequence_ids=batch.seq_indices.to(device=target_log_probs.device, dtype=torch.long),
        sequence_count=batch.sequence_count,
        clip_epsilon_low=clip_epsilon_low,
        clip_epsilon_high=clip_epsilon_high,
    )

    valid_float = valid_mask.to(dtype=torch.float32)
    safe_valid_count = valid_float.sum().clamp_min(1.0)
    log_probs = logits.to(dtype=torch.float32) - lse.to(dtype=torch.float32).unsqueeze(-1)
    probs = torch.exp(log_probs)
    token_entropy = -(probs * log_probs).sum(dim=-1).squeeze(0)
    entropy = (token_entropy * valid_float).sum() / safe_valid_count
    total_loss = policy_loss - (float(entropy_coef) * entropy)
    return total_loss, policy_loss, entropy, clip_fraction, mean_ratio, approx_kl, valid_count


@dataclass(frozen=True, slots=True)
class ResolvedGRPORun:
    policy_model: nn.Module
    policy_optimizer: torch.optim.Optimizer
    rollout_coordinator: RolloutCoordinator
    config: PPOConfig
    engine: Any
    checkpoint_manager: CheckpointManager
    wandb_logger: WandbLogger
    eval_run: ResolvedEvalRun | None = None
    start_step: int = 0
    value_model: Any | None = None
    value_optimizer: Any | None = None
    tokenizer: Any | None = None
    training_progress: Any | None = None

    @property
    def release_inference_buffers_before_update(self):
        return self.config.trainer.release_inference_buffers_before_update


def _build_rollout_engine(config: PPOConfig) -> Any:
    backend = str(config.inference.backend or "lfm25_native").strip().lower()
    engine_config = config.inference.build_engine_config()
    if backend in {"lfm25", "lfm25_native", "native_lfm25"}:
        return LFM25AsyncInferenceEngine(
            engine_config,
            batch_wait_seconds=float(config.inference.native_batch_wait_seconds),
            max_batch_size=config.inference.native_max_batch_size,
            compile_decode=bool(config.inference.compile_decode),
            cuda_graphs=bool(config.inference.native_cuda_graphs),
            max_cuda_graphs=int(config.inference.native_cuda_max_graphs),
            max_prefill_tokens=int(config.inference.native_prefill_tokens),
            bf16_weight_packing=bool(config.inference.native_bf16_weight_packing),
            fp8_mlp=bool(config.inference.native_fp8_mlp),
            fp8_kv=bool(config.inference.native_fp8_kv),
            adaptive_batching=bool(config.inference.native_adaptive_batching),
            fp8_lm_head=bool(config.inference.native_fp8_lm_head),
        )
    raise ValueError(f"Unsupported inference.backend={backend!r}; only native LFM inference is supported.")


def load_run_from_toml(config_path: str | Path, *, config_override: PPOConfig | None = None,
                       restore_optimizer: bool = True) -> ResolvedGRPORun:
    manager = ConfigManager.from_toml(config_path)
    path = manager.config_path
    config = config_override or manager.config
    wandb_config = config.wandb
    from climbing_gym.train import validate_config
    validate_config(config)
    from climbing_gym.preflight import validate_preflight
    preflight_report = validate_preflight(config)

    model_name = str(config.model.name or "").strip().lower()
    model_path = str(config.model.path)
    if not model_name:
        raise ValueError("`model.name` is required.")
    if not model_path:
        raise ValueError("`model.path` is required.")

    if config.job.print_config:
        print(json.dumps(manager.to_dict(), indent=2, sort_keys=True))

    resume_checkpoint_dir = str(config.resume.checkpoint_dir)
    resume_policy_path = ""
    start_step = 0
    if resume_checkpoint_dir:
        checkpoint_dir = Path(resume_checkpoint_dir)
        resume_policy_path = str((checkpoint_dir / "policy").resolve())
        if not Path(resume_policy_path).is_dir():
            raise FileNotFoundError(f"Resume policy checkpoint directory not found: {resume_policy_path}")
        if bool(config.resume.continue_step_numbers):
            metadata = _load_resume_metadata(checkpoint_dir)
            start_step = int(metadata.get("step", -1)) + 1

    engine: Any | None = None
    wandb_logger: WandbLogger | None = None
    try:
        policy_model = build_model(
            model_name,
            model_dir=(resume_policy_path or model_path),
            **dict(config.model.config),
        )
        policy_optimizer = _build_policy_optimizer(policy_model, config.policy_optimizer.to_dict(), trainer=config.trainer)
        if resume_checkpoint_dir and restore_optimizer:
            from climbing_gym.trainers.checkpoints import restore_training_state
            restore_training_state(resume_checkpoint_dir, policy_optimizer=policy_optimizer,
                                   allow_weights_only=config.resume.allow_weights_only)

        default_sampling = config.inference.sampling
        rollout_sampling = config.rollout.sampling
        effective_sampling = rollout_sampling or default_sampling
        if effective_sampling is None or effective_sampling.logprobs is None:
            raise ValueError(
                "GRPO requires rollout sampling with `logprobs` enabled in `[rollout.sampling]` or `[inference.sampling]`."
            )

        engine = _build_rollout_engine(config)

        env = config.env.build()
        dataset_rows = config.data.load_rows()
        env.configure_dataset(
            dataset_rows,
            seed=int(config.data.seed),
            shuffle=bool(config.data.shuffle),
            epoch=int(config.data.epoch),
        )

        harness = config.harness.build()
        rollout_coordinator = RolloutCoordinator(
            engine=engine,
            env=env,
            harness=harness,
            config=config.rollout.build_coordinator_config(),
        )

        if config.rollout.use_dynamic_sampling and config.harness.name == "direct" and config.rollout.num_rollouts >= 2:
            from climbing_gym.native_dynamic import NativeDynamicRunner
            audit_parent = (Path(config.train_logging.metrics_path).parent if config.train_logging.metrics_path
                            else Path(config.checkpoint.root_dir or Path.cwd() / "runs"))
            rollout_coordinator.rollout_runner = NativeDynamicRunner(
                config.rollout.build_runner_config(), audit_dir=audit_parent / "dynamic_sampling")
        if resume_checkpoint_dir and config.resume.continue_step_numbers:
            from climbing_gym.dataset_checkpoint import restore_dataset_checkpoint
            restore_dataset_checkpoint(rollout_coordinator, Path(resume_checkpoint_dir))

        eval_run: ResolvedEvalRun | None = None
        if config.eval.enabled:
            eval_run = build_eval_run(
                engine=engine,
                harness=harness,
                data_config=config.eval.data,
                env_config=config.eval.env,
                rollout_config=config.eval.rollout,
                eval_config=PeriodicEvalConfig(
                    max_examples=int(config.eval.max_examples),
                    pass_k=int(config.eval.pass_k),
                    seed=int(config.eval.seed),
                    shuffle=bool(config.eval.shuffle),
                ),
                logging_config=EvalLoggingConfig(
                    summary_path=str(config.eval.logging.summary_path),
                    trajectory_path=str(config.eval.logging.trajectory_path),
                    trace_path=str(config.eval.logging.trace_path),
                    max_traces=int(config.eval.logging.max_traces),
                ),
                close_engine_on_finish=False,
            )
            from climbing_gym.preflight import validate_eval_preflight
            eval_run = replace(eval_run, preflight=validate_eval_preflight(config, eval_run))
        checkpoint_manager = CheckpointManager(config.checkpoint)
        wandb_logger = WandbLogger(wandb_config, config_path=path, resolved_config={**config.to_dict(), "preflight": preflight_report,
                                                "eval_preflight": eval_run.preflight if eval_run is not None else None})

        return ResolvedGRPORun(
            policy_model=policy_model,
            policy_optimizer=policy_optimizer,
            rollout_coordinator=rollout_coordinator,
            config=config,
            engine=engine,
            checkpoint_manager=checkpoint_manager,
            wandb_logger=wandb_logger,
            eval_run=eval_run,
            start_step=int(start_step),
        )
    except Exception:
        if wandb_logger is not None:
            wandb_logger.close()
        if engine is not None:
            engine.close()
        raise


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run climbing_gym GRPO training from a single TOML config.")
    parser.add_argument("config", type=str, help="Path to the GRPO TOML config.")
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    setup_run: Callable[[ResolvedGRPORun], ResolvedGRPORun] | None = None,
    observe_step: Callable[[dict[str, Any]], None] | None = None,
) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    seed = ConfigManager.from_toml(args.config).config.trainer.seed
    random.seed(seed)
    torch.manual_seed(seed)
    run = load_run_from_toml(args.config)
    if setup_run is not None:
        try:
            run = setup_run(run)
        except BaseException:
            run.wandb_logger.close()
            run.engine.close()
            raise
    policy_model = run.policy_model
    policy_optimizer = run.policy_optimizer
    rollout_coordinator = run.rollout_coordinator
    checkpoint_manager = run.checkpoint_manager
    wandb_logger = run.wandb_logger
    config = run.config
    trainer_config = config.trainer
    eval_run = run.eval_run
    start_step = int(run.start_step)

    rollout_coordinator.rollout_config = config.rollout.build_rollout_config()

    num_steps = int(trainer_config.num_steps)
    if num_steps < 0:
        raise ValueError("num_steps must be >= 0")
    final_step = start_step + num_steps - 1

    train_device = torch.device(trainer_config.train_device)
    if trainer_config.train_dtype is None:
        policy_model.to(train_device)
    else:
        policy_model.to(device=train_device, dtype=trainer_config.train_dtype)

    use_cispo = str(trainer_config.algorithm).lower() == "cispo"
    loss_fn = packed_grpo_policy_loss_from_model
    if use_cispo:
        from climbing_gym.cispo import packed_cispo_policy_loss_from_model

        loss_fn = packed_cispo_policy_loss_from_model
        if trainer_config.loss_aggregation != "seq_mean_token_sum":
            raise ValueError("CISPO requires explicit seq_mean_token_sum loss aggregation")
        if trainer_config.policy_epochs != 1:
            raise ValueError("This CISPO runtime supports one accumulated update per rollout batch")
    compiled_policy_loss_fn = maybe_compile(
        loss_fn,
        dynamic=bool(trainer_config.compile_dynamic),
        fullgraph=bool(trainer_config.compile_fullgraph),
        backend=str(trainer_config.compile_backend),
        mode=str(trainer_config.compile_mode),
    ) if trainer_config.compile_losses else loss_fn

    try:
        for step in range(start_step, start_step + num_steps):
            step_started_at = time.perf_counter()

            rollout_started_at = time.perf_counter()
            trajectory_batch: TrajectoryBatch = collect_rollout_batch(
                rollout_coordinator=rollout_coordinator,
                model=policy_model,
                optimizer=policy_optimizer,
                device=train_device,
                dtype=trainer_config.train_dtype,
                drain_timeout=trainer_config.drain_timeout,
                release_inference_buffers_before_update=trainer_config.release_inference_buffers_before_update,
            )
            rollout_time_seconds = time.perf_counter() - rollout_started_at
            _append_training_trajectories(
                str(config.train_logging.trajectory_path),
                step=step,
                trajectories=trajectory_batch.trajectories,
            )

            policy_started_at = time.perf_counter()
            policy_batches = list(
                pack_trajectories(
                    trajectory_batch.trajectories,
                    max_tokens=trainer_config.policy_max_packed_tokens,
                )
            )
            trajectory_rewards = tuple(float(trajectory.evaluation.reward.reward) for trajectory in trajectory_batch.trajectories)
            group_ids = [int(trajectory.group_id) for trajectory in trajectory_batch.trajectories]
            advantage_estimator = str(trainer_config.advantage_estimator).strip().lower()
            if advantage_estimator == "sa_mrpo":
                accuracy_rewards = [
                    float(trajectory.evaluation.reward.info.get("exact_match", 0.0))
                    for trajectory in trajectory_batch.trajectories
                ]
                objective_rewards: dict[str, Sequence[float]] = {"accuracy": accuracy_rewards}
                objective_weights = {"accuracy": float(trainer_config.outcome_reward_weight)}
                if bool(trainer_config.format_reward_enabled):
                    objective_rewards["format"] = [
                        float(trajectory.evaluation.reward.info.get("reward_format", 0.0))
                        for trajectory in trajectory_batch.trajectories
                    ]
                    objective_weights["format"] = float(trainer_config.format_reward_weight)
                group_advantages, reward_metrics = compute_sa_mrpo_advantages(
                    objective_rewards,
                    objective_weights=objective_weights,
                    gamma=float(trainer_config.sa_mrpo_gamma),
                    group_ids=group_ids,
                    eps=float(config.rollout.reward_std_epsilon),
                )
            elif advantage_estimator == "grpo":
                group_advantages, reward_metrics = compute_dr_grpo_advantages(
                    trajectory_rewards,
                    group_ids=group_ids,
                    eps=float(config.rollout.reward_std_epsilon),
                )
            else:
                raise ValueError(
                    f"Unsupported trainer.advantage_estimator={advantage_estimator!r}; "
                    "expected 'grpo' or 'sa_mrpo'."
                )
            sequence_advantages = _build_sequence_scalar_payloads(
                trajectory_batch.trajectories,
                trajectory_values=weight_trajectory_advantages(trajectory_batch.trajectories, group_advantages),
            )

            total_policy_sequences = float(sum(batch.sequence_count for batch in policy_batches))
            total_policy_valid_tokens = float(sum(batch.valid_token_count for batch in policy_batches))
            policy_loss_mean = 0.0
            entropy_mean = 0.0
            clip_fraction_mean = 0.0
            ratio_mean = 0.0
            approx_kl_mean = 0.0
            probability_shift_mean = 0.0
            policy_grad_norm = 0.0

            policy_model.train()
            policy_optimizer.zero_grad(set_to_none=True)
            if total_policy_sequences > 0.0:
                total_policy_sequence_weight = total_policy_sequences * float(trainer_config.policy_epochs)
                total_policy_token_weight = total_policy_valid_tokens * float(trainer_config.policy_epochs)
                for epoch_idx in range(trainer_config.policy_epochs):
                    epoch_batches = list(policy_batches)
                    if trainer_config.shuffle_policy_batches and len(epoch_batches) > 1:
                        random.Random(trainer_config.seed + step + epoch_idx).shuffle(epoch_batches)
                    for batch in epoch_batches:
                        token_advantages = _build_completion_targets(
                            batch,
                            sequence_payloads=sequence_advantages,
                        ).to(train_device)
                        old_infer_logprobs = batch.completion_value_tensor(
                            batch.resolved_completion_logprobs()
                        ).to(train_device)
                        device_batch = batch.to(train_device)
                        loss_result = compiled_policy_loss_fn(
                            policy_model,
                            batch=device_batch,
                            token_advantages=token_advantages,
                            old_infer_logprobs=old_infer_logprobs,
                            clip_epsilon_low=float(trainer_config.clip_epsilon),
                            clip_epsilon_high=float(
                                trainer_config.clip_epsilon
                                if trainer_config.clip_epsilon_high is None
                                else trainer_config.clip_epsilon_high
                            ),
                            entropy_coef=float(trainer_config.entropy_coef),
                        )
                        total_loss, policy_loss, entropy, clip_fraction, mean_ratio, approx_kl, _ = loss_result[:7]
                        scale_grpo_loss_for_packed_batch(
                            total_loss,
                            sequence_count=batch.sequence_count,
                            total_sequence_count=total_policy_sequence_weight,
                        ).backward()

                        if not torch.isfinite(total_loss.detach()):
                            raise FloatingPointError("Non-finite policy loss; optimizer update aborted")

                        sequence_weight = float(batch.sequence_count)
                        token_weight = float(batch.valid_token_count)
                        policy_loss_mean += float(policy_loss.detach().item()) * sequence_weight
                        entropy_mean += float(entropy.detach().item()) * token_weight
                        clip_fraction_mean += float(clip_fraction.detach().item()) * token_weight
                        ratio_mean += float(mean_ratio.detach().item()) * token_weight
                        approx_kl_mean += float(approx_kl.detach().item()) * token_weight
                        if use_cispo:
                            probability_shift_mean += float(loss_result[7].detach().item()) * token_weight

                policy_loss_mean /= total_policy_sequence_weight
                if total_policy_token_weight > 0.0:
                    entropy_mean /= total_policy_token_weight
                    clip_fraction_mean /= total_policy_token_weight
                    ratio_mean /= total_policy_token_weight
                    approx_kl_mean /= total_policy_token_weight
                policy_grad_norm = (
                    float(
                        torch.nn.utils.clip_grad_norm_(
                            policy_model.parameters(),
                            trainer_config.policy_max_grad_norm,
                        ).item()
                    )
                    if trainer_config.policy_max_grad_norm > 0.0
                    else compute_grad_norm(policy_model)
                )
                if not math.isfinite(policy_grad_norm):
                    raise FloatingPointError("Non-finite policy gradient; optimizer update aborted")
                if use_cispo and total_policy_token_weight > 0:
                    probability_shift_mean /= total_policy_token_weight
                    if probability_shift_mean > trainer_config.behavior_parity_max_mean_abs_probability_shift:
                        raise RuntimeError(f"Behavior/training probability mismatch: {probability_shift_mean:.6f}")
                policy_optimizer.step()
                policy_optimizer.zero_grad(set_to_none=True)
            policy_time_seconds = time.perf_counter() - policy_started_at
            step_time_seconds = time.perf_counter() - step_started_at

            reward_mean = (
                float(sum(trajectory_rewards) / len(trajectory_rewards))
                if trajectory_rewards
                else 0.0
            )
            avg_completion_length = (
                float(
                    sum(
                        len(turn.completion_token_ids)
                        for trajectory in trajectory_batch.trajectories
                        for turn in trajectory.turns
                    )
                    / sum(len(trajectory.turns) for trajectory in trajectory_batch.trajectories)
                )
                if trajectory_batch.trajectories and any(trajectory.turns for trajectory in trajectory_batch.trajectories)
                else 0.0
            )
            avg_turn_count = (
                float(sum(len(trajectory.turns) for trajectory in trajectory_batch.trajectories) / len(trajectory_batch.trajectories))
                if trajectory_batch.trajectories
                else 0.0
            )
            effective_group_target = float(trajectory_batch.stats.get("effective_group_target", 0.0))
            effective_group_count = float(trajectory_batch.stats.get("effective_group_count", 0.0))
            effective_group_fraction = (
                float(effective_group_count / effective_group_target)
                if effective_group_target > 0.0
                else 0.0
            )
            tokens_per_second = (
                float(total_policy_valid_tokens / step_time_seconds)
                if step_time_seconds > 0.0
                else 0.0
            )

            metrics = {
                "policy/reward_mean": float(reward_mean),
                "policy/loss": float(policy_loss_mean),
                "policy/clip_fraction": float(clip_fraction_mean),
                "policy/grad_norm": float(policy_grad_norm),
                "policy/entropy": float(entropy_mean),
                "policy/approx_kl": float(approx_kl_mean),
                "policy/mean_ratio": float(ratio_mean),
                "trainer/avg_completion_length": float(avg_completion_length),
                "trainer/avg_turn_count": float(avg_turn_count),
                "trainer/effective_group_fraction": float(effective_group_fraction),
                "trainer/tokens_per_second": float(tokens_per_second),
                "trainer/rollout_time_seconds": float(rollout_time_seconds),
                "trainer/policy_time_seconds": float(policy_time_seconds),
                "trainer/step_time_seconds": float(step_time_seconds),
                "grpo/group_reward_mean": float(reward_metrics.get("grpo_group_reward_mean", 0.0)),
                "grpo/group_reward_std_mean": float(reward_metrics.get("grpo_group_reward_std_mean", 0.0)),
                "grpo/zero_std_prompt_rate": float(reward_metrics.get("grpo_zero_std_prompt_rate", 0.0)),
            }
            metrics.update({key.replace("_", "/", 1): float(value) for key, value in reward_metrics.items() if key.startswith("sa_mrpo_")})
            metrics.update({key: float(value) for key, value in trajectory_batch.stats.items() if key.startswith(("data/", "sampling/", "two_pass/", "reconcile/", "selector/", "single/", "verify/", "repair/"))})
            if use_cispo:
                # No entropy objective or entropy diagnostic is evaluated.
                metrics.pop("policy/entropy")
                metrics["policy/pre_update_mean_abs_probability_shift"] = probability_shift_mean
                metrics["policy/master_update_norm"] = float(getattr(policy_optimizer, "last_master_update_norm", 0.0))
                metrics["policy/quantized_parameter_changes"] = float(getattr(policy_optimizer, "last_quantized_change_count", 0))
                metrics["trainer/mixed_group_fraction"] = 1.0 - metrics["grpo/zero_std_prompt_rate"]
                metrics["trainer/truncated_fraction"] = sum(
                    any(turn.finish_reason == "length" for turn in trajectory.turns)
                    for trajectory in trajectory_batch.trajectories
                ) / max(1, len(trajectory_batch.trajectories))
                reward_info = [trajectory.evaluation.reward.info for trajectory in trajectory_batch.trajectories]
                if any("answer_tag_count" in info for info in reward_info):
                    metrics["reward/repeated_answer_fraction"] = sum(
                        info.get("answer_tag_open_count", info.get("answer_tag_count", 0)) > 1
                        or info.get("answer_tag_close_count", info.get("answer_tag_count", 0)) > 1
                        for info in reward_info
                    ) / max(1, len(reward_info))
                for key in ("single_final_answer_valid", "answer_correct", "base_reward", "doomloop_score", "doomloop_penalty", "truncation_penalty", "reasoning_cap_count", "summary_count", "forced_summary_count", "summary_failure_count", "length_penalty", "length_penalty_weight", "completion_token_count"):
                    if any(key in info for info in reward_info):
                        metrics[f"reward/{key}_mean"] = sum(
                            float(info.get(key, 0)) for info in reward_info
                        ) / max(1, len(reward_info))
                if any("doomloop_score" in info for info in reward_info):
                    metrics["reward/doomloop_fraction"] = sum(
                        info.get("doomloop_score", 0) > 0 for info in reward_info
                    ) / max(1, len(reward_info))

            if eval_run is not None and int(config.eval.interval) > 0 and (int(step) + 1) % int(config.eval.interval) == 0:
                offload_to_cpu(policy_model, policy_optimizer)
                eval_run.rollout_coordinator.sync_weights(
                    policy_model,
                    drain_timeout=trainer_config.drain_timeout,
                )
                eval_summary = run_eval(eval_run)
                if trainer_config.release_inference_buffers_before_update:
                    release = getattr(run.engine, "release_inference_buffers", None)
                    if callable(release):
                        release()
                reload_to_device(
                    policy_model,
                    policy_optimizer,
                    device=train_device,
                    dtype=trainer_config.train_dtype,
                )
                metrics.update({f"eval/{key}": float(value) for key, value in eval_summary.items()})

            checkpoint_path = checkpoint_manager.maybe_save(
                step=step,
                is_final=step == final_step,
                policy_model=policy_model,
                value_model=None,
                policy_optimizer=policy_optimizer,
            )
            if checkpoint_path is not None:
                from climbing_gym.dataset_checkpoint import save_dataset_checkpoint
                save_dataset_checkpoint(rollout_coordinator, Path(checkpoint_path))
            generated_tokens = sum(len(turn.completion_token_ids) for row in trajectory_batch.trajectories for turn in row.turns)
            metrics["trainer/rollout_tokens_per_second"] = generated_tokens / max(rollout_time_seconds, 1e-9)
            metrics["trainer/full_step_time_seconds"] = time.perf_counter() - step_started_at
            wandb_logger.log_step(step=step, metrics=metrics)
            payload = {
                "step": int(step),
                "metrics": metrics,
            }
            if checkpoint_path is not None:
                payload["checkpoint_path"] = str(checkpoint_path)
            metrics_path = str(config.train_logging.metrics_path)
            if metrics_path:
                path = Path(metrics_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(json.dumps(payload, sort_keys=True) + "\n")
            print(json.dumps(payload, sort_keys=True))
            if observe_step is not None:
                observe_step(payload)
    finally:
        run.wandb_logger.close()
        run.engine.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
