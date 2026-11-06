"""Readiness state for the historical full-response Value MIS+SC critic.

The gate observes fresh-rollout diagnostics measured before fitting the critic.
It never evaluates a held-out panel or lets post-fit predictions admit the actor.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path


STATE_FILE = "value_warmup_state.json"


def enabled(trainer):
    return (
        trainer.value_warmup_min_batches != 0 or trainer.value_warmup_max_batches != 0
    )


def gate_config(trainer):
    return {
        "min_batches": trainer.value_warmup_min_batches,
        "max_batches": trainer.value_warmup_max_batches,
        "consecutive_ready": trainer.value_warmup_consecutive_ready,
        "max_abs_bias": trainer.value_warmup_max_abs_bias,
    }


def method_config(config):
    """Bind continuation to the objective, precision, and collection contract.

    Output paths, total requested updates, W&B metadata, and checkpoint paths
    can change on resume; the method and sampler must retain their meaning.
    Dataset identity/order are additionally checked by dataset_state.json.
    """
    trainer = config.trainer
    fields = (
        "value_mis_alpha",
        "value_mis_policy_lambda",
        "value_epochs",
        "policy_max_grad_norm",
        "value_max_grad_norm",
        "policy_max_packed_tokens",
        "value_max_packed_tokens",
        "gamma",
        "gae_lambda",
        "train_dtype",
        "flashreinforce_fixed_token_normalizer",
        "flashreinforce_head_chunk_size",
        "flashreinforce_fused_head",
        "flashreinforce_checkpoint_head",
        "offpolicy_ratio_min",
        "offpolicy_ratio_max",
    )
    sampling = config.rollout.sampling or config.inference.sampling
    payload = {
        "trainer": {key: getattr(trainer, key) for key in fields},
        "policy_optimizer": config.policy_optimizer.to_dict(),
        "value_optimizer": config.value_optimizer.to_dict(),
        "model_config": config.model.config,
        "sampling": asdict(sampling),
        "env": asdict(config.env),
        "harness": asdict(config.harness),
        "rollout": {
            key: getattr(config.rollout, key)
            for key in (
                "num_rollouts",
                "batch_size",
                "effective_batch_size",
                "max_inflight_requests",
                "max_total_requests",
                "use_dynamic_sampling",
            )
        },
        "inference": {
            key: getattr(config.inference, key)
            for key in (
                "dtype",
                "seed",
                "max_model_len",
                "native_max_batch_size",
                "max_num_seqs",
                "apply_chat_template",
                "chat_template_enable_thinking",
                "assistant_prefill",
                "native_cuda_graphs",
                "native_bf16_weight_packing",
                "compile_decode",
                "native_prefill_tokens",
                "native_fp8_mlp",
                "native_fp8_kv",
                "native_fp8_lm_head",
                "native_adaptive_batching",
            )
        },
    }
    return json.loads(json.dumps(payload, default=str))


def validate_config(config):
    trainer = config.trainer
    if not enabled(trainer):
        return
    if trainer.algorithm != "value_mis_sc":
        raise ValueError(
            "Readiness-gated value warmup requires algorithm='value_mis_sc'"
        )
    gate = gate_config(trainer)
    if any(
        type(gate[k]) is not int or gate[k] <= 0
        for k in ("min_batches", "max_batches", "consecutive_ready")
    ):
        raise ValueError(
            "Value warmup batch limits and consecutive_ready must be positive integers"
        )
    if (
        gate["min_batches"] > gate["max_batches"]
        or gate["consecutive_ready"] > gate["max_batches"]
    ):
        raise ValueError(
            "Value warmup minimum and consecutive count must fit the maximum batches"
        )
    if not math.isfinite(gate["max_abs_bias"]) or gate["max_abs_bias"] <= 0:
        raise ValueError("value_warmup_max_abs_bias must be finite and positive")
    if not config.checkpoint.root_dir:
        raise ValueError(
            "Readiness-gated warmup requires checkpoint.root_dir to preserve every boundary"
        )
    if config.resume.checkpoint_dir and (
        config.resume.allow_weights_only or not config.resume.continue_step_numbers
    ):
        raise ValueError(
            "Value warmup resume requires exact optimizer/RNG/dataset state and continue_step_numbers=true"
        )


@dataclass
class WarmupState:
    batches: int = 0
    consecutive: int = 0
    ready: bool = False
    failed: bool = False
    actor_updates: int = 0

    def observe(self, diagnostics, trainer):
        if self.ready or self.failed:
            raise RuntimeError("Cannot add a collection to completed value warmup")
        keys = ("mc_ev", "mc_mse", "constant_baseline_mse", "calibration_bias")
        if any(not math.isfinite(float(diagnostics[key])) for key in keys):
            raise FloatingPointError(
                "Value warmup received nonfinite pre-fit diagnostics"
            )
        good = (
            diagnostics["mc_ev"] > 0
            and diagnostics["mc_mse"] < diagnostics["constant_baseline_mse"]
            and abs(diagnostics["calibration_bias"]) < trainer.value_warmup_max_abs_bias
        )
        self.batches += 1
        self.consecutive = self.consecutive + 1 if good else 0
        self.ready = (
            self.batches >= trainer.value_warmup_min_batches
            and self.consecutive >= trainer.value_warmup_consecutive_ready
        )
        self.failed = (
            not self.ready and self.batches >= trainer.value_warmup_max_batches
        )
        return bool(good)


def save_state(checkpoint_dir, state, config):
    destination = Path(checkpoint_dir) / STATE_FILE
    payload = {
        "version": 1,
        "gate": gate_config(config.trainer),
        "method": method_config(config),
        "state": asdict(state),
    }
    temporary = destination.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(destination)


def load_state(config, *, actor_start_step):
    if not enabled(config.trainer):
        if (
            config.resume.checkpoint_dir
            and (Path(config.resume.checkpoint_dir) / STATE_FILE).exists()
        ):
            raise ValueError(
                "A readiness-gated value checkpoint must retain its warmup configuration on resume"
            )
        return None
    directory = config.resume.checkpoint_dir
    if not directory:
        if actor_start_step:
            raise ValueError("A fresh value warmup cannot begin after actor training")
        return WarmupState()
    path = Path(directory) / STATE_FILE
    if not path.is_file():
        raise ValueError(
            f"Value checkpoint lacks readiness phase state: {path}; use a complete value-warmup checkpoint"
        )
    payload = json.loads(path.read_text())
    if (
        payload.get("version") != 1
        or payload.get("gate") != gate_config(config.trainer)
        or payload.get("method") != method_config(config)
    ):
        raise ValueError(
            "Value warmup checkpoint version or readiness configuration differs"
        )
    if not (Path(directory) / "dataset_state.json").is_file():
        raise ValueError(
            "Value warmup checkpoint is missing the exact dataset continuation state"
        )
    state = WarmupState(**payload["state"])
    if (
        type(state.batches) is not int
        or type(state.consecutive) is not int
        or type(state.actor_updates) is not int
        or type(state.ready) is not bool
        or type(state.failed) is not bool
        or state.batches < 0
        or not 0 <= state.consecutive <= state.batches
        or state.batches > config.trainer.value_warmup_max_batches
        or state.actor_updates != actor_start_step
        or state.actor_updates < 0
        or (state.actor_updates and not state.ready)
        or (state.ready and state.failed)
    ):
        raise ValueError("Inconsistent value warmup checkpoint counters or actor phase")
    expected_ready = (
        state.batches >= config.trainer.value_warmup_min_batches
        and state.consecutive >= config.trainer.value_warmup_consecutive_ready
    )
    expected_failed = (
        not expected_ready and state.batches >= config.trainer.value_warmup_max_batches
    )
    if state.ready != expected_ready or state.failed != expected_failed:
        raise ValueError(
            "Value warmup checkpoint gate does not match its completed collections"
        )
    return state
