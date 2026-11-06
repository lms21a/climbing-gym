from __future__ import annotations

import csv
from dataclasses import asdict, dataclass, field, is_dataclass
import json
from pathlib import Path
from typing import Any

import torch

from climbing_gym.inference.config import (
    DEFAULT_STAGED_CHECKPOINT_ROOT,
    InferenceEngineConfig,
    SamplingConfig,
)
from climbing_gym.rollout.config import RolloutConfig, RolloutRunnerConfig
from climbing_gym.rollout.coordinator import RolloutCoordinatorConfig
from climbing_gym.trainers.checkpoints import CheckpointManagerConfig


@dataclass(frozen=True, slots=True)
class Job:
    config_file: str | None = None
    description: str = "climbing_gym PPO run"
    print_config: bool = False


@dataclass(frozen=True, slots=True)
class Model:
    name: str = ""
    path: str = ""
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class Data:
    dataset_name: str = ""
    dataset_config: str = ""
    dataset_path: str = ""
    split: str = "train"
    max_rows: int = 0
    seed: int = 0
    shuffle: bool = True
    epoch: int = 0

    def load_rows(self) -> Any:
        from climbing_gym.data.loading import load_dataset_source, resolve_split

        dataset_path = str(self.dataset_path).strip()
        if dataset_path:
            path = Path(dataset_path)
            if path.is_file():
                suffix = path.suffix.lower()
                if suffix == ".jsonl":
                    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
                    return rows[:int(self.max_rows)] if int(self.max_rows) > 0 else rows
                if suffix == ".json":
                    payload = json.loads(path.read_text(encoding="utf-8"))
                    if isinstance(payload, list):
                        return payload[: int(self.max_rows)] if int(self.max_rows) > 0 else payload
                    raise TypeError("JSON dataset files must contain a list of row objects.")
                if suffix == ".csv":
                    with path.open("r", encoding="utf-8", newline="") as handle:
                        rows = list(csv.DictReader(handle))
                    return rows[: int(self.max_rows)] if int(self.max_rows) > 0 else rows

        dataset_obj = load_dataset_source(
            str(self.dataset_name),
            str(self.dataset_config),
            dataset_path,
        )
        dataset_split = resolve_split(dataset_obj, str(self.split))
        max_rows = int(self.max_rows)
        if max_rows > 0:
            if hasattr(dataset_split, "select"):
                dataset_split = dataset_split.select(range(min(max_rows, len(dataset_split))))
            else:
                dataset_split = list(dataset_split)[:max_rows]
        return dataset_split


@dataclass(frozen=True, slots=True)
class Env:
    name: str = ""
    reward: str = "exact_match"
    args: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        env_name = str(self.name).strip().lower()
        if not env_name:
            raise ValueError("`env.name` is required.")

        if env_name in {"math_qa", "gsm8k"}:
            from climbing_gym.envs.math_qa import (
                MathQAEnv,
                exact_match_math_bounded_summary_2_turns_reward,
                exact_match_math_hybrid_2_turns_reward,
                exact_match_math_min_2_turns_reward,
                exact_match_math_reward,
                exact_match_single_answer_math_reward,
                exact_match_math_summary_bootstrap_2_turns_reward,
                exact_match_math_soft_target_2_turns_reward,
                exact_match_math_underturn_penalty_2_turns_reward,
                exact_match_with_format_math_reward,
            )

            reward_name = str(self.reward or "exact_match").strip().lower()
            if reward_name == "exact_match":
                reward_fn = exact_match_math_reward
            elif reward_name == "exact_match_single_answer":
                reward_fn = exact_match_single_answer_math_reward
            elif reward_name == "exact_match_with_format":
                from functools import partial

                reward_fn = partial(
                    exact_match_with_format_math_reward,
                    format_weight=float(self.args.get("format_reward_weight", 0.1)),
                )
            elif reward_name == "exact_match_bounded_summary_2_turns":
                reward_fn = exact_match_math_bounded_summary_2_turns_reward
            elif reward_name == "exact_match_hybrid_2_turns":
                reward_fn = exact_match_math_hybrid_2_turns_reward
            elif reward_name == "exact_match_min_2_turns":
                reward_fn = exact_match_math_min_2_turns_reward
            elif reward_name == "exact_match_underturn_penalty_2_turns":
                reward_fn = exact_match_math_underturn_penalty_2_turns_reward
            elif reward_name == "exact_match_summary_bootstrap_2_turns":
                reward_fn = exact_match_math_summary_bootstrap_2_turns_reward
            elif reward_name == "exact_match_soft_target_2_turns":
                reward_fn = exact_match_math_soft_target_2_turns_reward
            else:
                raise ValueError(f"Unsupported math_qa reward: {reward_name!r}")
            env_args = dict(self.args)
            env_args.pop("format_reward_weight", None)
            if reward_name == "exact_match_single_answer":
                # Keep feedback/pass@1 aligned with the reward's answer contract.
                env_args["require_single_final_answer"] = True
            return MathQAEnv(
                reward_fn=reward_fn,
                **env_args,
            )

        if env_name == "countdown":
            from functools import partial
            from climbing_gym.doomloop import validate_penalty_weight
            from climbing_gym.envs.countdown import (
                CountdownEnv, terminal_expression_countdown_reward,
                terminal_expression_doomloop_countdown_reward,
            )

            reward_name = str(self.reward or "exact_match").strip().lower()
            env_args = dict(self.args)
            if reward_name == "exact_match":
                env_args["accept_boxed_answers"] = False
                return CountdownEnv(**env_args)
            if reward_name == "terminal_expression":
                env_args["accept_boxed_answers"] = True
                return CountdownEnv(reward_fn=terminal_expression_countdown_reward, **env_args)
            if reward_name == "terminal_expression_doomloop":
                weight = validate_penalty_weight(env_args.pop("doomloop_penalty_weight", 0.25))
                cap_weight = validate_penalty_weight(env_args.pop('truncation_penalty_weight', 0.0))
                env_args["accept_boxed_answers"] = True
                return CountdownEnv(reward_fn=partial(terminal_expression_doomloop_countdown_reward,
                                                      penalty_weight=weight, truncation_penalty_weight=cap_weight), **env_args)
            raise ValueError(f"Unsupported countdown reward: {self.reward!r}")

        if env_name == "knights_knaves":
            from climbing_gym.envs.knights_knaves import KnightsKnavesEnv
            if str(self.reward or "exact_match").strip().lower() != "exact_match":
                raise ValueError("Knights and Knaves supports only exact_match reward")
            return KnightsKnavesEnv(**dict(self.args))

        if env_name == "futoshiki":
            from climbing_gym.envs.futoshiki import FutoshikiEnv, dense_futoshiki_reward, exact_futoshiki_reward, accuracy_futoshiki_reward, strict_exact_futoshiki_reward
            rewards = {"dense": dense_futoshiki_reward, "dense_accuracy": accuracy_futoshiki_reward, "exact_match": exact_futoshiki_reward, "exact_match_strict": strict_exact_futoshiki_reward}
            reward_name = str(self.reward or "dense").strip().lower()
            if reward_name not in rewards:
                raise ValueError(f"Unsupported Futoshiki reward: {reward_name!r}")
            env_args=dict(self.args)
            env_args['reward_profile']=reward_name
            return FutoshikiEnv(reward_fn=rewards[reward_name], **env_args)

        if env_name in {"cfop_cube", "cfop_phase"}:
            from climbing_gym.envs.cfop_cube import (
                CFOPCubeEnv, CFOPPhaseEnv, cfop_dense_v2_reward, cfop_phase_v1_reward,
            )
            expected = "cfop_phase_v1" if env_name == "cfop_phase" else "cfop_dense_v2"
            if str(self.reward).strip().lower() != expected:
                raise ValueError(f"{env_name} requires reward={expected!r}.")
            cls, reward = ((CFOPPhaseEnv, cfop_phase_v1_reward) if env_name == "cfop_phase"
                           else (CFOPCubeEnv, cfop_dense_v2_reward))
            return cls(reward_fn=reward, **dict(self.args))

        raise ValueError(f"Unsupported env.name={env_name!r}.")


@dataclass(frozen=True, slots=True)
class Harness:
    name: str = "direct"
    args: dict[str, Any] = field(default_factory=dict)

    def build(self) -> Any:
        from climbing_gym.rollout.harnesses import DirectHarness, InfinityThinkHarness

        harness_name = str(self.name or "direct").strip().lower()
        if harness_name == "direct":
            return DirectHarness()
        if harness_name == 'numbered_notebook':
            from climbing_gym.rollout.harnesses.numbered_notebook import NumberedNotebookHarness
            return NumberedNotebookHarness(**self.args)
        if harness_name == "infinity_think":
            max_turns = int(self.args.get("max_turns", 1))
            return InfinityThinkHarness(max_turns=max_turns)
        raise ValueError(f"Unsupported harness.name={harness_name!r}.")


@dataclass(frozen=True, slots=True)
class Optimizer:
    type: str = "adamw"
    learning_rate: float = 1e-6
    weight_decay: float = 0.0
    fused: bool | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "type": str(self.type),
            "learning_rate": float(self.learning_rate),
            "weight_decay": float(self.weight_decay),
            "fused": self.fused,
        }


@dataclass(frozen=True, slots=True)
class Resume:
    checkpoint_dir: str = ""
    continue_step_numbers: bool = True
    allow_weights_only: bool = False


@dataclass(frozen=True, slots=True)
class PPOTrainer:
    algorithm: str = "ppo"
    flashreinforce_delta: float = 0.003
    flashreinforce_objective: str = 'sequence_kl'
    # MIS/MIS_SC only: use one response-independent divisor for every token.
    # None resolves to the configured rollout sampling max_tokens, not batch lengths.
    flashreinforce_length_normalization: str = 'fixed'
    flashreinforce_fixed_token_normalizer: int | None = None
    flashreinforce_head_chunk_size: int = 512
    flashreinforce_fused_head: bool = True
    flashreinforce_checkpoint_head: bool = True
    value_optimizer_backing_path: str = ""
    value_mis_alpha: float = 0.4
    value_mis_policy_lambda: float | None = None
    # Zero/zero keeps generic value-MIS updates ungated. Historical Knights
    # uses 10/30 batches and three consecutive fresh, pre-fit successes.
    value_warmup_min_batches: int = 0
    value_warmup_max_batches: int = 0
    value_warmup_consecutive_ready: int = 3
    value_warmup_max_abs_bias: float = 0.08
    release_inference_buffers_before_update: bool = True
    optimizer_chunk_elements: int = 4194304
    optimizer_workspace_bytes: int = 50331648
    optimizer_pin_state: bool = True
    mis_use_value: bool = True
    mis_token_ratio_min: float = 0.95
    mis_token_ratio_max: float = 1.05
    mis_trajectory_ratio_min: float = 0.95
    mis_trajectory_ratio_max: float = 1.05
    mis_fallback_to_valid: bool = False
    mis_normalize_advantages: bool = True
    mis_adv_norm_eps: float = 1e-6
    mis_value_loss_coef: float = 0.5
    mis_kl_coef: float = 0.0
    mis_bootstrap_truncated_trajectories: bool = False
    mis_outcome_reward_weight: float = 1.0
    offpolicy_ratio_min: float = .5
    offpolicy_ratio_max: float = 5.
    num_steps: int = 1
    critic_warmup_steps: int = 0
    seed: int = 0
    train_device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dtype: torch.dtype | None = None
    drain_timeout: int = 300
    policy_max_packed_tokens: int = 4096
    value_max_packed_tokens: int = 4096
    trajectory_minibatch_size: int = 256
    value_minibatch_prompt_groups: int | None = None
    policy_epochs: int = 1
    value_epochs: int = 1
    shuffle_policy_batches: bool = True
    shuffle_value_batches: bool = True
    clip_epsilon: float = 0.2
    clip_epsilon_high: float | None = None
    advantage_estimator: str = "grpo"
    sa_mrpo_gamma: float = 0.25
    outcome_reward_weight: float = 1.0
    format_reward_enabled: bool = False
    format_reward_weight: float = 1.0
    dual_clip_ratio: float = 10.0
    entropy_coef: float = 0.0
    gamma: float = 1.0
    gae_lambda: float = 1.0
    length_adaptive_gae_alpha: float = 0.4
    justrl2_lambda_k: float = 0.5
    justrl2_value_bias: float = 0.5
    justrl2_value_clip: float = 0.2
    justrl2_critic_lr_warmup_steps: int = 10
    justrl2_require_warmup_ready: bool = True
    justrl2_length_penalty: float = 0.0
    justrl2_normalize_advantages: bool = True
    justrl2_target_method: str = 'justrl2'
    justrl2_policy_objective: str = 'ppo'
    compactionrl_gae_alpha: float = 1.5
    # None keeps BPCO's length-adaptive estimator; 1.0 uses exact MC advantages.
    bpco_policy_gae_lambda: float | None = None
    reward_min: float = 0.0
    reward_max: float = 1.0
    privileged_critic: bool = False
    privileged_answer_template: str = (
        "Training-only critic context:\nReference answer: {answer}"
    )
    dppo_log_ratio_clip: float = 20.0
    loss_aggregation: str = "token_mean"
    behavior_parity_max_mean_abs_probability_shift: float = 0.05
    future_kl_tau: float = 32.0
    future_kl_clip_ratio: float = 0.2
    future_kl_clip_high_only: bool = True
    future_kl_start: str = "include_current"
    future_kl_window: int = -1
    future_kl_average: bool = False
    future_kl_safety_threshold: float = 10.0
    future_kl_log_ratio_cap: float = 20.0
    dual_clip_sequence_threshold: int = 1
    policy_max_grad_norm: float = 1.0
    value_max_grad_norm: float = 1.0
    value_learning_rate: float = 5e-6
    value_weight_decay: float = 0.0
    vine_num_mc_rollouts: int = 0
    compile_losses: bool = True
    compile_dynamic: bool = True
    compile_fullgraph: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"


@dataclass(frozen=True, slots=True)
class Wandb:
    enabled: bool = True
    project: str = "simple-pt"
    entity: str | None = "lukas928"
    dir: str = ""
    run_name: str | None = None
    mode: str | None = "online"


@dataclass(frozen=True, slots=True)
class EvalLogging:
    summary_path: str = ""
    trajectory_path: str = ""
    trace_path: str = ""
    max_traces: int = 8


@dataclass(frozen=True, slots=True)
class TrainLogging:
    trajectory_path: str = ""
    metrics_path: str = ""


@dataclass(frozen=True, slots=True)
class Inference:
    backend: str = "lfm25_native"
    model_path: str = ""
    preflight_report_path: str = ""
    preflight_prior_report_path: str = ""
    dtype: str = "auto"
    seed: int = 0
    gpu_memory_utilization: float = 0.9
    max_model_len: int | None = None
    max_num_seqs: int | None = None
    enforce_eager: bool = False
    enable_sleep_mode: bool = True
    apply_chat_template: bool = False
    chat_template_enable_thinking: bool = True
    assistant_prefill: str = ""
    native_batch_wait_seconds: float = 0.001
    native_max_batch_size: int | None = None
    compile_decode: bool = True
    native_cuda_graphs: bool = False
    native_cuda_max_graphs: int = 1
    native_prefill_tokens: int = 4096
    native_bf16_weight_packing: bool = True
    native_fp8_mlp: bool = False
    native_fp8_kv: bool = False
    native_adaptive_batching: bool = False
    native_fp8_lm_head: bool = False
    sampling: SamplingConfig = field(default_factory=SamplingConfig)
    staged_checkpoint_root: Path = field(default_factory=lambda: DEFAULT_STAGED_CHECKPOINT_ROOT)

    def __post_init__(self) -> None:
        if self.backend not in {"lfm25_native", "lfm25", "native_lfm25"}:
            raise ValueError("Only the native LFM inference backend is supported")
        for name in ("native_max_batch_size", "max_num_seqs"):
            value = getattr(self, name)
            if value is not None and value <= 0:
                raise ValueError(f"{name} must be positive when configured")

    def build_engine_config(self) -> InferenceEngineConfig:
        if not str(self.model_path).strip():
            raise ValueError("`inference.model_path` is required.")
        return InferenceEngineConfig(
            model_path=str(self.model_path),
            dtype=str(self.dtype),
            seed=int(self.seed),
            gpu_memory_utilization=float(self.gpu_memory_utilization),
            max_model_len=self.max_model_len,
            max_num_seqs=self.max_num_seqs,
            enforce_eager=bool(self.enforce_eager),
            enable_sleep_mode=bool(self.enable_sleep_mode),
            apply_chat_template=bool(self.apply_chat_template),
            chat_template_enable_thinking=bool(self.chat_template_enable_thinking),
            assistant_prefill=str(self.assistant_prefill),
            default_sampling=self.sampling,
            staged_checkpoint_root=Path(self.staged_checkpoint_root),
        )


@dataclass(frozen=True, slots=True)
class Rollout:
    num_rollouts: int = 2
    sampling: SamplingConfig | None = None
    max_total_requests: int | None = None
    batch_size: int = 1
    effective_batch_size: int = 1
    max_inflight_requests: int | None = None
    reward_std_epsilon: float = 1e-8
    use_dynamic_sampling: bool = True
    stall_timeout_seconds: float = 120.0
    progress_log_interval_seconds: float = 15.0
    engine_sleep_level: int = 1
    engine_sleep_mode: str = "abort"

    def build_rollout_config(self) -> RolloutConfig:
        return RolloutConfig(
            num_rollouts=int(self.num_rollouts),
            sampling=self.sampling,
            max_total_requests=self.max_total_requests,
        )

    def build_runner_config(self) -> RolloutRunnerConfig:
        return RolloutRunnerConfig(
            effective_batch_size=int(self.effective_batch_size),
            max_inflight_requests=self.max_inflight_requests,
            reward_std_epsilon=float(self.reward_std_epsilon),
            use_dynamic_sampling=bool(self.use_dynamic_sampling),
            stall_timeout_seconds=float(self.stall_timeout_seconds),
            progress_log_interval_seconds=float(self.progress_log_interval_seconds),
        )

    def build_coordinator_config(self) -> RolloutCoordinatorConfig:
        rollout_config = self.build_rollout_config()
        return RolloutCoordinatorConfig(
            num_rollouts=int(rollout_config.num_rollouts),
            sampling=rollout_config.sampling,
            max_total_requests=rollout_config.max_total_requests,
            batch_size=int(self.batch_size),
            engine_sleep_level=int(self.engine_sleep_level),
            engine_sleep_mode=str(self.engine_sleep_mode),
            runner_config=self.build_runner_config(),
        )


@dataclass(frozen=True, slots=True)
class Eval:
    preflight_report_path: str = ""
    preflight_prior_report_path: str = ""
    interval: int = 0
    max_examples: int = 0
    pass_k: int = 1
    seed: int = 0
    shuffle: bool = False
    data: Data = field(default_factory=Data)
    env: Env = field(default_factory=Env)
    rollout: Rollout = field(default_factory=Rollout)
    logging: EvalLogging = field(default_factory=EvalLogging)

    @property
    def enabled(self) -> bool:
        return int(self.interval) > 0


@dataclass(frozen=True, slots=True)
class PPOConfig:
    job: Job = field(default_factory=Job)
    model: Model = field(default_factory=Model)
    data: Data = field(default_factory=Data)
    env: Env = field(default_factory=Env)
    harness: Harness = field(default_factory=Harness)
    inference: Inference = field(default_factory=Inference)
    rollout: Rollout = field(default_factory=Rollout)
    trainer: PPOTrainer = field(default_factory=PPOTrainer)
    policy_optimizer: Optimizer = field(default_factory=Optimizer)
    value_optimizer: Optimizer = field(default_factory=lambda: Optimizer(learning_rate=5e-6))
    checkpoint: CheckpointManagerConfig = field(default_factory=CheckpointManagerConfig)
    resume: Resume = field(default_factory=Resume)
    wandb: Wandb = field(default_factory=Wandb)
    train_logging: TrainLogging = field(default_factory=TrainLogging)
    eval: Eval = field(default_factory=Eval)

    def to_dict(self) -> dict[str, Any]:
        return _serialize(asdict(self))


JobConfig = PPOConfig


def _serialize(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _serialize(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_serialize(item) for item in value]
    if isinstance(value, tuple):
        return [_serialize(item) for item in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).removeprefix("torch.")
    if is_dataclass(value):
        return _serialize(asdict(value))
    return value


__all__ = [
    "Data",
    "Env",
    "Eval",
    "EvalLogging",
    "Harness",
    "Inference",
    "Job",
    "JobConfig",
    "Model",
    "Optimizer",
    "PPOConfig",
    "PPOTrainer",
    "Resume",
    "Rollout",
    "TrainLogging",
    "Wandb",
]
