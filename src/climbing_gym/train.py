"""Native training entry point for the supported policy and actor--critic methods.

The source PPO and CISPO loops retain their numerical reductions. The remaining
methods share collection, exact checkpoint continuation, logging, and alternating
actor/learner memory ownership here; their objectives live in algorithm modules.
"""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import random
import time
from typing import Any, Sequence

import torch

from .config import ConfigManager, PPOConfig
from .dataset_checkpoint import save_dataset_checkpoint
from .trainers.common import offload_to_cpu, reload_to_device
from .trainers.checkpoints import restore_training_state

ALGORITHMS = frozenset({"ppo", "grpo", "cispo", "flashreinforce", "mis", "mis_sc",
                        "grouped_mis", "grouped_mis_sc", "mis_ppo", "value_mis_sc"})


def resolve_config(config: PPOConfig) -> PPOConfig:
    """Make named MIS variants explicit in the persisted resolved configuration."""
    trainer = config.trainer
    algorithm = trainer.algorithm.lower()
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported trainer.algorithm={algorithm!r}; choose {sorted(ALGORITHMS)}")
    if algorithm in {"mis", "mis_sc", "grouped_mis", "grouped_mis_sc", "value_mis_sc"}:
        objective = "mis_sc" if algorithm.endswith("mis_sc") else "mis"
        trainer = replace(trainer, flashreinforce_objective=objective)
    if algorithm == "value_mis_sc" and trainer.flashreinforce_fixed_token_normalizer is None:
        sampling = config.rollout.sampling or config.inference.sampling
        trainer = replace(trainer, flashreinforce_fixed_token_normalizer=sampling.max_tokens)
    return replace(config, trainer=trainer)


def validate_config(config: PPOConfig) -> None:
    """Fail before allocating models for unsupported method combinations."""
    trainer, rollout = config.trainer, config.rollout
    algorithm = trainer.algorithm
    if algorithm not in ALGORITHMS:
        raise ValueError(f"Unsupported trainer.algorithm={algorithm!r}")
    if trainer.num_steps < 0 or trainer.policy_max_packed_tokens <= 0:
        raise ValueError("num_steps must be nonnegative and packing budget positive")
    if trainer.policy_epochs <= 0 or trainer.value_epochs <= 0:
        raise ValueError("Policy and critic epochs must be positive")
    limits = ["policy_max_grad_norm"]
    if algorithm in {"ppo", "value_mis_sc"} or (algorithm == "mis_ppo" and trainer.mis_use_value):
        limits.append("value_max_grad_norm")
    for name in limits:
        value = getattr(trainer, name)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be finite")
        if algorithm in {"mis_ppo", "value_mis_sc"} and value <= 0:
            raise ValueError(f"{name} must be positive for {algorithm}")
    from .value_warmup import validate_config as validate_warmup
    validate_warmup(config)
    from .training_memory import optimizer_chunk_elements
    optimizer_chunk_elements(trainer.optimizer_chunk_elements, trainer.optimizer_workspace_bytes)
    if type(trainer.optimizer_pin_state) is not bool:
        raise ValueError("optimizer_pin_state must be a boolean")
    if type(trainer.release_inference_buffers_before_update) is not bool:
        raise ValueError("release_inference_buffers_before_update must be a boolean")
    sampling = rollout.sampling or config.inference.sampling
    if sampling.logprobs is None:
        raise ValueError("Native training requires stored sampled-token log probabilities")
    if sampling.temperature != 1 or sampling.top_p != 1 or sampling.top_k != -1 or sampling.n != 1:
        raise ValueError("Native policy updates require temperature=1, full support, and n=1 for raw behavior probabilities")
    if config.env.name not in {"gsm8k", "math_qa", "countdown", "knights_knaves", "futoshiki", "cfop_cube", "cfop_phase"}:
        raise ValueError("Training supports GSM8K, Countdown, Knights and Knaves, Futoshiki, and CFOP")
    if algorithm == "value_mis_sc" and config.env.name in {"cfop_cube", "cfop_phase"}:
        raise ValueError("Value MIS+SC requires binary outcomes; CFOP dense rewards are incompatible")
    if rollout.use_dynamic_sampling and config.harness.name == "direct" and rollout.num_rollouts >= 2:
        n, concurrency, cap = rollout.num_rollouts, rollout.max_inflight_requests, rollout.max_total_requests
        if not concurrency or concurrency % n or cap is None or cap % n or cap < rollout.effective_batch_size * n:
            raise ValueError("Dynamic sampling requires concurrency and request cap to contain whole prompt groups")
    if algorithm == "ppo":
        if trainer.advantage_estimator not in {"gae", "grpo"}:
            raise ValueError("PPO uses GAE; choose advantage_estimator='gae' ('grpo' remains a historical alias)")
        if trainer.privileged_critic:
            raise ValueError("PPO uses an ordinary critic; privileged_critic is supported by value_mis_sc instead")
        if trainer.loss_aggregation != "token_mean":
            raise ValueError("PPO supports loss_aggregation='token_mean' only")
        if trainer.clip_epsilon_high is not None and trainer.clip_epsilon_high != trainer.clip_epsilon:
            raise ValueError("PPO uses symmetric clipping; omit clip_epsilon_high or set it equal to clip_epsilon")
        return
    if algorithm == "grpo":
        if trainer.advantage_estimator not in {"grpo", "sa_mrpo"}:
            raise ValueError("GRPO requires grpo or sa_mrpo advantages")
        return
    if algorithm == "cispo":
        if trainer.advantage_estimator != "grpo" or trainer.format_reward_enabled or trainer.entropy_coef:
            raise ValueError("CISPO requires centered scalar outcome advantages and no auxiliary objectives")
        if trainer.loss_aggregation != "seq_mean_token_sum" or trainer.policy_epochs != 1:
            raise ValueError("CISPO requires seq_mean_token_sum and one accumulated update")
        return
    if algorithm == "mis_ppo":
        expected_estimator = "value" if trainer.mis_use_value else "grpo"
        if trainer.advantage_estimator != expected_estimator:
            raise ValueError(f"MIS-PPO with mis_use_value={trainer.mis_use_value} requires advantage_estimator={expected_estimator!r}")
        if not trainer.mis_use_value:
            if rollout.num_rollouts < 2:
                raise ValueError("Actor-only MIS-PPO requires at least two rollouts per prompt for group advantages")
            if trainer.entropy_coef != 0 or trainer.mis_kl_coef != 0:
                raise ValueError("Actor-only MIS-PPO has no entropy or KL objective; set both coefficients to zero")
            if trainer.mis_bootstrap_truncated_trajectories:
                raise ValueError("Actor-only MIS-PPO cannot bootstrap without a value model")
            if not trainer.mis_normalize_advantages:
                raise ValueError("Actor-only MIS-PPO uses normalized group advantages; mis_normalize_advantages must be true")
        if trainer.policy_epochs != 1 or trainer.value_epochs != 1 or trainer.critic_warmup_steps:
            raise ValueError("MIS-PPO supports one accumulated actor/critic update per collection")
        if trainer.gamma != 1 or trainer.gae_lambda != 1 or trainer.format_reward_enabled:
            raise ValueError("MIS-PPO uses terminal outcome returns without GAE or auxiliary rewards")
        if trainer.privileged_critic or trainer.outcome_reward_weight != 1:
            raise ValueError("MIS-PPO uses an ordinary critic and raw environment outcomes")
        if trainer.compile_losses:
            raise ValueError("MIS-PPO validates gates eagerly; set compile_losses=false")
        if config.harness.name != "direct":
            raise ValueError("MIS-PPO currently collects complete direct responses")
        if sampling.temperature != 1 or sampling.top_p != 1 or sampling.top_k != -1 or sampling.n != 1:
            raise ValueError("MIS-PPO requires raw native probabilities (temperature=1 and full support)")
        from .mis_ppo import _resolve_ratio_bounds
        _resolve_ratio_bounds(trainer.mis_token_ratio_min, trainer.mis_token_ratio_max, label="token")
        _resolve_ratio_bounds(trainer.mis_trajectory_ratio_min, trainer.mis_trajectory_ratio_max, label="trajectory")
        return
    from .flashreinforce import validate_config as validate_flash
    # Validate the shared sampling/objective contract using its real validator.
    view = replace(config, trainer=replace(trainer, algorithm="flashreinforce"))
    if algorithm == "value_mis_sc":
        if trainer.critic_warmup_steps:
            raise ValueError("Value MIS+SC uses readiness-gated warmup; set critic_warmup_steps=0 and configure value_warmup_min_batches/max_batches")
        if trainer.gamma != 1 or trainer.gae_lambda != 1:
            raise ValueError("Value MIS+SC uses gamma=1 and Monte Carlo critic targets (gae_lambda=1)")
        if trainer.advantage_estimator != "length_adaptive_gae":
            raise ValueError("Value MIS+SC requires advantage_estimator='length_adaptive_gae'")
        view = replace(view, trainer=replace(view.trainer, advantage_estimator="batch_centered"))
    grouped = algorithm in {"grouped_mis", "grouped_mis_sc"}
    if grouped:
        if trainer.advantage_estimator != "grpo" or not rollout.use_dynamic_sampling or rollout.num_rollouts < 2:
            raise ValueError("Grouped MIS requires group-relative advantages, dynamic mixed-group sampling, and >=2 rollouts")
        if config.env.reward not in {"exact_match", "exact_match_strict", "exact_match_single_answer", "terminal_expression"}:
            raise ValueError("Grouped MIS requires binary exact-match rewards")
        if config.rollout.sampling and config.rollout.sampling.max_tokens != config.inference.sampling.max_tokens:
            raise ValueError("Grouped MIS uses inference.sampling.max_tokens as its fixed denominator; budgets must match")
        size = max(2, rollout.batch_size)
        view = replace(view, trainer=replace(view.trainer, advantage_estimator="batch_centered"),
                       rollout=replace(rollout, num_rollouts=1, use_dynamic_sampling=False,
                                       batch_size=size, effective_batch_size=size, max_total_requests=size))
    validate_flash(view)
    if algorithm == "value_mis_sc":
        if not math.isfinite(trainer.value_mis_alpha) or trainer.value_mis_alpha <= 0:
            raise ValueError("value_mis_alpha must be finite and positive")
        if trainer.value_mis_policy_lambda is not None and not 0 <= trainer.value_mis_policy_lambda <= 1:
            raise ValueError("value_mis_policy_lambda must lie in [0, 1]")
        if trainer.flashreinforce_length_normalization != "fixed":
            raise ValueError("Value MIS+SC uses a fixed configured token denominator")


def load_run_from_toml(config_path: str | Path):
    from .grpo import load_run_from_toml as load_base, _build_policy_optimizer
    from .ppo import build_value_model_from_policy, _load_value_model_checkpoint
    config = resolve_config(ConfigManager.from_toml(config_path).config)
    validate_config(config)
    run = load_base(config_path, config_override=config, restore_optimizer=False)
    try:
        critic = optimizer = tokenizer = None
        if config.trainer.algorithm == "value_mis_sc":
            from .value_mis_sc import build_critic
            from .models.lfm2_compat import load_lfm2_tokenizer
            critic = build_critic(run.policy_model)
            tokenizer = load_lfm2_tokenizer(config.model.path)
        elif config.trainer.algorithm == "mis_ppo" and config.trainer.mis_use_value:
            critic = build_value_model_from_policy(run.policy_model)
        if critic is not None:
            if config.value_optimizer.type == "mapped_master_adamw":
                from .value_mis_sc import MappedCriticAdamW
                backing = config.trainer.value_optimizer_backing_path
                if not backing:
                    root = Path(config.checkpoint.root_dir or Path.cwd() / "runs")
                    backing = root / "critic-optimizer-cache.bin"
                optimizer = MappedCriticAdamW((p for p in critic.parameters() if p.requires_grad), backing_path=backing,
                    lr=config.value_optimizer.learning_rate, weight_decay=config.value_optimizer.weight_decay,
                    chunk_elements=config.trainer.optimizer_chunk_elements,
                    workspace_bytes=config.trainer.optimizer_workspace_bytes)
            else:
                optimizer = _build_policy_optimizer(critic, config.value_optimizer.to_dict(), trainer=config.trainer)
            if config.resume.checkpoint_dir:
                _load_value_model_checkpoint(critic, Path(config.resume.checkpoint_dir) / "value_model.pt")
        if config.resume.checkpoint_dir:
            restore_training_state(config.resume.checkpoint_dir, policy_optimizer=run.policy_optimizer,
                                   value_optimizer=optimizer, allow_weights_only=config.resume.allow_weights_only)
        from .value_warmup import load_state
        state = load_state(config, actor_start_step=run.start_step)
        return replace(run, value_model=critic, value_optimizer=optimizer, tokenizer=tokenizer,
                       value_warmup_state=state)
    except BaseException:
        run.wandb_logger.close()
        run.engine.close()
        raise


def _reload_critic(run):
    trainer = run.config.trainer
    if trainer.algorithm == "value_mis_sc":
        # The bounded critic's small projection intentionally stays FP32.
        run.value_model.backbone.to(device=trainer.train_device, dtype=trainer.train_dtype)
        run.value_model.value_head.to(device=trainer.train_device, dtype=torch.float32)
        reload_to_device(run.value_model, run.value_optimizer, device=trainer.train_device)
    else:
        reload_to_device(run.value_model, run.value_optimizer, device=trainer.train_device, dtype=trainer.train_dtype)


def update_policy(run, trajectories, *, actor_update=True):
    """Execute one complete learner update; public for CPU integration tests."""
    algorithm = run.config.trainer.algorithm
    if not actor_update and algorithm != "value_mis_sc":
        raise ValueError("Critic-only readiness updates require value_mis_sc")
    if algorithm == "value_mis_sc" and actor_update:
        from .value_warmup import enabled as warmup_enabled
        state = getattr(run, "value_warmup_state", None)
        if warmup_enabled(run.config.trainer) and (state is None or not state.ready or state.failed):
            raise RuntimeError("Complete readiness-gated critic warmup before any value-MIS actor update")
    if algorithm == "mis_ppo":
        from .mis_ppo import update_policy as update
        return update(run, trajectories)
    if algorithm in {"grouped_mis", "grouped_mis_sc"}:
        from .offpolicy import update_grouped_corrected_policy
        return update_grouped_corrected_policy(run, trajectories)
    if algorithm == "value_mis_sc":
        from .training_memory import prepare_policy_update
        from .value_mis_sc import freeze_targets, fit_critic, update_actor
        prepare_policy_update(run)
        offload_to_cpu(run.policy_model, run.policy_optimizer)
        _reload_critic(run)
        packs, advantages, targets, _, diagnostics = freeze_targets(run, trajectories,
            alpha=run.config.trainer.value_mis_alpha, policy_lambda=run.config.trainer.value_mis_policy_lambda)
        critic_metrics = fit_critic(run, trajectories, packs, targets,
            epochs=run.config.trainer.value_epochs, max_grad_norm=run.config.trainer.value_max_grad_norm)
        offload_to_cpu(run.value_model, run.value_optimizer)
        metrics = {}
        if actor_update:
            reload_to_device(run.policy_model, run.policy_optimizer, device=run.config.trainer.train_device,
                             dtype=run.config.trainer.train_dtype)
            metrics = {f"policy/{k}": v for k, v in update_actor(run, trajectories, advantages).items()}
        metrics.update({f"value/{k}": v for k, v in critic_metrics.items()})
        metrics.update({f"value/frozen_{k}": v for k, v in diagnostics.items()})
        return metrics, []
    if algorithm in {"flashreinforce", "mis", "mis_sc"}:
        from .flashreinforce import update_policy as update
        return update(run, trajectories)
    raise ValueError(f"Algorithm {algorithm!r} uses its dedicated PPO/GRPO runtime")


def _append_json(path, payload):
    if path:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        with destination.open("a") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _save_warmup_checkpoint(run, state):
    """Publish a complete critic collection boundary with actor step still -1."""
    from .trainers.checkpoints import CheckpointManager
    from .value_warmup import save_state

    root = Path(run.config.checkpoint.root_dir)
    destination = root / "critic_warmup" / f"step_{state.batches - 1}"
    if (destination / "value_warmup_state.json").exists():
        raise FileExistsError(f"Warmup boundary already exists: {destination}; resume the latest checkpoint or choose a new root")
    manager = CheckpointManager(replace(run.config.checkpoint, root_dir=str(root / "critic_warmup")))
    # Numbered warmup directories never collide with actor step checkpoints.
    path = manager.save(step=state.batches - 1, policy_model=run.policy_model,
        value_model=run.value_model, policy_optimizer=run.policy_optimizer,
        value_optimizer=run.value_optimizer)
    save_dataset_checkpoint(run.rollout_coordinator, path)
    if not (path / "dataset_state.json").is_file():
        raise RuntimeError("Cannot publish warmup checkpoint without exact dataset continuation state")
    metadata = path / "metadata.json"
    temporary = metadata.with_suffix(".tmp")
    temporary.write_text(json.dumps({"step": -1, "value_warmup_batch": state.batches}) + "\n")
    temporary.replace(metadata)
    # The phase sidecar is written last; missing/incomplete checkpoints fail closed.
    _save_tokenizer(run, path)
    save_state(path, state, run.config)
    pointer = root / "latest_value_warmup.json"
    temporary = pointer.with_suffix(".tmp")
    temporary.write_text(json.dumps({"checkpoint_path": str(path.resolve()),
                                    "batches": state.batches, "ready": state.ready,
                                    "failed": state.failed}, indent=2) + "\n")
    temporary.replace(pointer)
    published = json.loads(pointer.read_text())
    required = ("policy", "value_model.pt", "training_state.pt", "dataset_state.json", "value_warmup_state.json")
    if Path(published["checkpoint_path"]) != path.resolve() or not all((path / name).exists() for name in required):
        raise RuntimeError("Published warmup checkpoint is incomplete; previous checkpoints retained")
    # Full critic optimizer states are large. Retain two complete recovery
    # boundaries, pruning only after the new pointer and every state file exist.
    import shutil
    complete = []
    for candidate in (root / "critic_warmup").glob("step_*"):
        marker = candidate / "value_warmup_state.json"
        if candidate.is_dir() and marker.is_file():
            batch = json.loads(marker.read_text())["state"]["batches"]
            complete.append((batch, candidate))
    for _, old in sorted(complete)[:-2]:
        shutil.rmtree(old)
    return path


def _run_value_warmup(run, state, *, observe_step=None):
    from .grpo import _append_training_trajectories

    trainer = run.config.trainer
    if state.failed:
        raise RuntimeError("Critic readiness failed at the maximum warmup batches; actor training has not started")
    while not state.ready:
        started = time.perf_counter()
        for model, optimizer in ((run.policy_model, run.policy_optimizer), (run.value_model, run.value_optimizer)):
            offload_to_cpu(model, optimizer)
        rollout_started = time.perf_counter()
        run.rollout_coordinator.sync_weights(run.policy_model, drain_timeout=trainer.drain_timeout)
        batch = run.rollout_coordinator.collect()
        rollout_seconds = time.perf_counter() - rollout_started
        _append_training_trajectories(run.config.train_logging.trajectory_path,
            step=-(state.batches + 1), trajectories=batch.trajectories)
        metrics, _ = update_policy(run, batch.trajectories, actor_update=False)
        diagnostics = {key.removeprefix("value/frozen_"): value for key, value in metrics.items()
                       if key.startswith("value/frozen_")}
        good = state.observe(diagnostics, trainer)
        generated = sum(len(turn.completion_token_ids) for row in batch.trajectories for turn in row.turns)
        metrics.update({"warmup/batch": state.batches, "warmup/consecutive_ready": state.consecutive,
            "warmup/gate_passed": float(good), "warmup/ready": float(state.ready),
            "policy/optimizer_steps": 0., "trainer/actor_updates": 0.,
            "trainer/rollout_time_seconds": rollout_seconds,
            "trainer/rollout_tokens_per_second": generated / max(rollout_seconds, 1e-9),
            "trainer/generated_tokens": generated})
        metrics.update({key: float(value) for key, value in batch.stats.items()})
        path = _save_warmup_checkpoint(run, state)
        metrics["trainer/full_step_time_seconds"] = time.perf_counter() - started
        payload = {"phase": "critic_warmup", "batch": state.batches,
                   "checkpoint_path": str(path), "metrics": metrics}
        _append_json(run.config.train_logging.metrics_path, payload)
        run.wandb_logger.log_step(step=state.batches - 1, metrics=metrics)
        print(json.dumps(payload, sort_keys=True), flush=True)
        if observe_step is not None:
            observe_step(payload)
        if state.failed:
            raise RuntimeError("Critic has not passed fresh-rollout calibration; actor updates have not started")


def _save_tokenizer(run, checkpoint_dir):
    save = getattr(getattr(run, "tokenizer", None), "save_pretrained", None)
    if callable(save):
        save(Path(checkpoint_dir) / "policy")


def run_training(run, *, observe_step=None):
    """Run native collection and learning, persisting complete continuation state."""
    from .grpo import _append_training_trajectories
    from .eval import run_eval
    config, trainer = run.config, run.config.trainer
    from .value_warmup import enabled as warmup_enabled, load_state, save_state
    warmup = None
    if warmup_enabled(trainer):
        warmup = run.value_warmup_state or load_state(config, actor_start_step=run.start_step)
        run = replace(run, value_warmup_state=warmup)
    reload_to_device(run.policy_model, run.policy_optimizer, device=trainer.train_device, dtype=trainer.train_dtype)
    if run.value_model is not None and trainer.algorithm != "value_mis_sc":
        _reload_critic(run)
    final_step = run.start_step + trainer.num_steps - 1
    try:
        if warmup is not None:
            _run_value_warmup(run, warmup, observe_step=observe_step)
        for step in range(run.start_step, final_step + 1):
            started = time.perf_counter()
            trainables = [(run.policy_model, run.policy_optimizer)]
            if run.value_model is not None:
                trainables.append((run.value_model, run.value_optimizer))
            for model, optimizer in trainables:
                offload_to_cpu(model, optimizer)
            rollout_started = time.perf_counter()
            run.rollout_coordinator.sync_weights(run.policy_model, drain_timeout=trainer.drain_timeout)
            batch = run.rollout_coordinator.collect()
            rollout_seconds = time.perf_counter() - rollout_started
            if trainer.release_inference_buffers_before_update:
                release = getattr(run.engine, "release_inference_buffers", None)
                if callable(release):
                    release()
            if trainer.algorithm != "value_mis_sc":
                for model, optimizer in trainables:
                    reload_to_device(model, optimizer, device=trainer.train_device, dtype=trainer.train_dtype)
            _append_training_trajectories(config.train_logging.trajectory_path, step=step, trajectories=batch.trajectories)
            update_started = time.perf_counter()
            metrics, audit = update_policy(run, batch.trajectories)
            if warmup is not None:
                warmup.actor_updates = step + 1
            update_seconds = time.perf_counter() - update_started
            generated = sum(len(turn.completion_token_ids) for row in batch.trajectories for turn in row.turns)
            rewards = [row.evaluation.reward.reward for row in batch.trajectories]
            metrics.setdefault("policy/reward_mean", math.fsum(rewards) / max(1, len(rewards)))
            metrics.update({"trainer/rollout_time_seconds": rollout_seconds,
                "trainer/rollout_tokens_per_second": generated / max(rollout_seconds, 1e-9),
                "trainer/policy_time_seconds": update_seconds, "trainer/generated_tokens": generated,
                "trainer/step_time_seconds": time.perf_counter() - started})
            metrics.update({key: float(value) for key, value in batch.stats.items()})
            if run.eval_run is not None and (step + 1) % config.eval.interval == 0:
                for model, optimizer in trainables:
                    offload_to_cpu(model, optimizer)
                run.eval_run.rollout_coordinator.sync_weights(run.policy_model, drain_timeout=trainer.drain_timeout)
                metrics.update({f"eval/{k}": v for k, v in run_eval(run.eval_run).items()})
                release = getattr(run.engine, "release_inference_buffers", None)
                if trainer.release_inference_buffers_before_update and callable(release):
                    release()
                if trainer.algorithm != "value_mis_sc":
                    for model, optimizer in trainables:
                        reload_to_device(model, optimizer, device=trainer.train_device, dtype=trainer.train_dtype)
            path = run.checkpoint_manager.maybe_save(step=step, is_final=step == final_step,
                policy_model=run.policy_model, value_model=run.value_model,
                policy_optimizer=run.policy_optimizer, value_optimizer=run.value_optimizer)
            if path is not None:
                _save_tokenizer(run, path)
                save_dataset_checkpoint(run.rollout_coordinator, Path(path))
                if warmup is not None:
                    save_state(path, warmup, config)
            metrics["trainer/full_step_time_seconds"] = time.perf_counter() - started
            metrics["trainer/actor_updates"] = step + 1
            run.wandb_logger.log_step(step=step + (warmup.batches if warmup is not None else 0), metrics=metrics)
            payload = {"step": step, "metrics": metrics}
            if path is not None:
                payload["checkpoint_path"] = str(path)
            _append_json(config.train_logging.metrics_path, payload)
            if audit and config.train_logging.metrics_path:
                _append_json(str(Path(config.train_logging.metrics_path).with_suffix(".audit.jsonl")),
                             {"step": step, "trajectories": audit})
            print(json.dumps(payload, sort_keys=True), flush=True)
            if observe_step is not None:
                observe_step(payload)
    finally:
        run.wandb_logger.close()
        run.engine.close()
    return 0


def main(argv: Sequence[str] | None = None):
    parser = argparse.ArgumentParser(description="Train a policy with the native inference engine")
    parser.add_argument("config", help="TOML configuration path")
    args = parser.parse_args(argv)
    config = resolve_config(ConfigManager.from_toml(args.config).config)
    validate_config(config)
    random.seed(config.trainer.seed)
    torch.manual_seed(config.trainer.seed)
    if config.trainer.algorithm == "ppo":
        from .ppo import main as train
        return train([args.config])
    if config.trainer.algorithm in {"grpo", "cispo"}:
        from .grpo import main as train
        return train([args.config])
    return run_training(load_run_from_toml(args.config))


if __name__ == "__main__":
    raise SystemExit(main())
