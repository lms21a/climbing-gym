from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import hashlib
from pathlib import Path
import random
import time
import tomllib
from typing import Any, Mapping, Sequence

from datasets import Dataset, DatasetDict, load_from_disk
import torch
from torch import nn
from transformers import AutoTokenizer

from climbing_gym.sft_data import prompt_prefills_open_think
from climbing_gym.sft_data import (
    adapt_completion_to_thinking_template,
    strip_response_xml_envelope,
)
from climbing_gym.config import Data as DataConfig
from climbing_gym.config import Job as JobConfig
from climbing_gym.config import Model as ModelConfig
from climbing_gym.config import Optimizer as OptimizerConfig
from climbing_gym.config import Resume as ResumeConfig
from climbing_gym.config import Wandb as WandbConfig
from climbing_gym.config.manager import (
    _coerce_optional_bool,
    _coerce_table,
    _resolve_path,
    _resolve_torch_dtype,
)
from climbing_gym.models import build_model
from climbing_gym.sft_eval import SFTEvaluationConfig, PeriodicSFTEvaluator, _cleanup_resources
from climbing_gym.trainers.checkpoints import CheckpointManager, CheckpointManagerConfig
from climbing_gym.trainers.common import compute_grad_norm
from climbing_gym.trainers.logger import WandbLogger
from climbing_gym.trainers.packing import Batch, pack_sequences


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.dtype):
        return str(value).replace("torch.", "")
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _load_resume_metadata(checkpoint_dir: Path) -> dict[str, Any]:
    metadata_path = checkpoint_dir / "metadata.json"
    if not metadata_path.is_file():
        return {}
    payload = json.loads(metadata_path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("Resume checkpoint metadata must parse to a mapping.")
    return dict(payload)


def _build_policy_optimizer(
    policy_model: nn.Module,
    optimizer_config: Mapping[str, Any],
) -> torch.optim.Optimizer:
    optimizer_name = str(optimizer_config.get("type", "adamw") or "adamw").strip().lower()
    if optimizer_name != "adamw":
        raise ValueError(f"Unsupported policy optimizer type: {optimizer_name!r}")
    return torch.optim.AdamW(
        policy_model.parameters(),
        lr=float(optimizer_config.get("learning_rate", 1e-6)),
        weight_decay=float(optimizer_config.get("weight_decay", 0.0)),
        fused=bool(optimizer_config.get("fused", torch.cuda.is_available())),
    )


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


def _resolve_dataset_path(dataset_path: str) -> str:
    resolved = str(dataset_path or "").strip()
    if not resolved:
        return resolved
    path = Path(resolved)
    if not path.is_dir():
        return resolved
    child = path / "hf_dataset_train_test"
    if child.is_dir():
        return str(child.resolve())
    return resolved


def _load_rows(data_config: DataConfig) -> Sequence[Mapping[str, Any]]:
    dataset_path = _resolve_dataset_path(str(data_config.dataset_path))
    if dataset_path:
        path = Path(dataset_path)
        if path.is_dir():
            dataset = load_from_disk(str(path))
            if isinstance(dataset, DatasetDict):
                split = str(data_config.split or "train")
                if split not in dataset:
                    raise KeyError(
                        f"Dataset split {split!r} not found. Available splits: {sorted(dataset.keys())}"
                    )
                rows: Dataset | Sequence[Mapping[str, Any]] = dataset[split]
            else:
                rows = dataset
            max_rows = int(data_config.max_rows)
            if max_rows > 0 and hasattr(rows, "select"):
                rows = rows.select(range(min(max_rows, len(rows))))
            elif max_rows > 0:
                rows = list(rows)[:max_rows]
            return rows

    if dataset_path != str(data_config.dataset_path):
        data_config = DataConfig(
            dataset_name=str(data_config.dataset_name),
            dataset_config=str(data_config.dataset_config),
            dataset_path=dataset_path,
            split=str(data_config.split),
            max_rows=int(data_config.max_rows),
            seed=int(data_config.seed),
            shuffle=bool(data_config.shuffle),
            epoch=int(data_config.epoch),
        )
    return data_config.load_rows()


def _apply_chat_template(
    tokenizer: Any,
    prompt: str,
    *,
    enable_thinking: bool,
) -> str:
    apply_chat_template = getattr(tokenizer, "apply_chat_template", None)
    if not callable(apply_chat_template):
        raise TypeError("sft.apply_chat_template=true requires a tokenizer with apply_chat_template(...).")

    template_kwargs: dict[str, Any] = {
        "tokenize": False,
        "add_generation_prompt": True,
        "enable_thinking": bool(enable_thinking),
    }
    messages = [{"role": "user", "content": prompt}]
    try:
        return str(apply_chat_template(messages, **template_kwargs))
    except TypeError as exc:
        message = str(exc)
        if "enable_thinking" in message and ("unexpected keyword" in message or "got an unexpected" in message):
            template_kwargs.pop("enable_thinking", None)
            return str(apply_chat_template(messages, **template_kwargs))
        raise
    except ValueError as exc:
        raise ValueError(
            "sft.apply_chat_template=true failed because the tokenizer does not expose a usable chat template."
        ) from exc


def _contains_subsequence(tokens: Sequence[int], needle: Sequence[int]) -> bool:
    if not needle:
        return True
    if len(tokens) < len(needle):
        return False
    for offset in range(len(tokens) - len(needle) + 1):
        if tuple(tokens[offset : offset + len(needle)]) == tuple(needle):
            return True
    return False


@dataclass(frozen=True, slots=True)
class SFTTrainerConfig:
    text_key: str = ""
    thinking_tag: str = "think"
    prompt_token_ids_key: str = "prompt_token_ids"
    completion_token_ids_key: str = "completion_token_ids"
    prompt_key: str = "prompt"
    completion_key: str = "completion"
    batch_size: int = 8
    num_epochs: int = 1
    seed: int = 0
    log_interval_steps: int = 1
    max_seq_len: int = 4096
    max_packed_tokens: int = 4096
    shuffle: bool = True
    shuffle_packed_batches: bool = True
    train_device: str | torch.device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dtype: torch.dtype | None = None
    max_grad_norm: float = 1.0
    loss_on_full_sequence: bool = False
    apply_chat_template: bool = False
    chat_template_enable_thinking: bool = True
    force_think_prefix: bool = False
    drop_response_xml: bool = False
    require_think_close: bool = False
    append_eos_token: bool = True
    compile_dynamic: bool = True
    compile_fullgraph: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"


@dataclass(frozen=True, slots=True)
class SFTConfig:
    job: JobConfig
    model: ModelConfig
    data: DataConfig
    sft: SFTTrainerConfig
    policy_optimizer: OptimizerConfig
    checkpoint: CheckpointManagerConfig
    resume: ResumeConfig
    wandb: WandbConfig
    eval: SFTEvaluationConfig = field(default_factory=SFTEvaluationConfig)


@dataclass(frozen=True, slots=True)
class ResolvedSFTRun:
    policy_model: nn.Module
    tokenizer: Any
    policy_optimizer: torch.optim.Optimizer
    config: SFTConfig
    checkpoint_manager: CheckpointManager
    wandb_logger: WandbLogger
    start_step: int = 0
    resume_state: Mapping[str, Any] | None = None


def packed_sft_loss(
    *,
    target_log_probs: torch.Tensor,
    valid_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    valid = valid_mask.to(dtype=torch.float32)
    valid_count = valid.sum()
    safe_valid_count = valid_count.clamp_min(1.0)
    token_nll = -target_log_probs.to(dtype=torch.float32)
    loss = (token_nll * valid).sum() / safe_valid_count
    return loss, valid_count


def packed_sft_loss_from_model(
    policy_model: Any,
    *,
    batch: Batch,
    loss_on_full_sequence: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    logits = policy_model(batch.inputs, positions=batch.position_ids)
    lse = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(dim=-1, index=batch.targets.unsqueeze(-1)).squeeze(-1)
    target_log_probs = (target_logits - lse).squeeze(0).to(dtype=torch.float32)

    if loss_on_full_sequence:
        valid_mask = torch.ones_like(target_log_probs, dtype=torch.float32)
    else:
        valid_mask = batch.valid_completion_mask().to(device=target_log_probs.device, dtype=torch.float32)

    loss, valid_count = packed_sft_loss(
        target_log_probs=target_log_probs,
        valid_mask=valid_mask,
    )
    predictions = logits.argmax(dim=-1).squeeze(0)
    safe_valid_count = valid_count.clamp_min(1.0)
    accuracy = (
        ((predictions == batch.targets.squeeze(0)).to(dtype=torch.float32) * valid_mask).sum()
        / safe_valid_count
    )
    return loss, accuracy, valid_count


def scale_sft_loss_for_packed_batch(
    loss: torch.Tensor,
    *,
    token_count: float | int | torch.Tensor,
    total_token_count: float | int,
) -> torch.Tensor:
    total = float(total_token_count)
    if total <= 0.0:
        return loss * 0.0
    batch_tokens = (
        float(token_count.detach().item())
        if isinstance(token_count, torch.Tensor)
        else float(token_count)
    )
    if batch_tokens <= 0.0:
        return loss * 0.0
    return loss * (batch_tokens / total)


def _prepare_sft_sequences(
    tokenizer: Any,
    rows: Sequence[Mapping[str, Any]],
    *,
    config: SFTTrainerConfig,
) -> tuple[list[list[int]], list[int]]:
    sequences: list[list[int]] = []
    prompt_lens: list[int] = []
    if config.text_key:
        if not config.loss_on_full_sequence:
            raise ValueError("text_key requires loss_on_full_sequence=true")
        if any((config.apply_chat_template, config.force_think_prefix,
                config.drop_response_xml, config.require_think_close)):
            raise ValueError("text_key cannot be combined with chat/reasoning transformations")
        for row in rows:
            document = row[config.text_key]
            if not isinstance(document, str) or not document.strip():
                raise ValueError("Raw training documents must be nonempty strings")
            tokens = list(tokenizer.encode(document, add_special_tokens=False))
            # BOS supplies context for predicting the first document token.
            bos = getattr(tokenizer, "bos_token_id", None)
            if bos is not None:
                tokens.insert(0, int(bos))
            if config.append_eos_token and tokenizer.eos_token_id is not None:
                tokens.append(int(tokenizer.eos_token_id))
            if len(tokens) > config.max_seq_len:
                raise ValueError(f"Raw document has {len(tokens)} tokens, exceeding max_seq_len={config.max_seq_len}; refusing truncation")
            if len(tokens) < 2:
                raise ValueError("Raw document must supply at least two sequence tokens")
            sequences.append(tokens)
            # Packing requires positive prompt_lens; full-sequence loss ignores it.
            prompt_lens.append(1)
        return sequences, prompt_lens
    if config.thinking_tag not in {"think", "thinking"}:
        raise ValueError("thinking_tag must be think or thinking")
    think_close_tokens = tuple(
        int(token)
        for token in tokenizer.encode(f"</{config.thinking_tag}>", add_special_tokens=False)
    ) if bool(config.require_think_close) else ()

    for row in rows:
        if config.prompt_token_ids_key in row or config.completion_token_ids_key in row:
            if not all(k in row for k in (config.prompt_token_ids_key, config.completion_token_ids_key)):
                raise ValueError("Exact-ID SFT requires both prompt and completion token IDs")
            if any((config.apply_chat_template, config.force_think_prefix, config.drop_response_xml)):
                raise ValueError("Exact-ID SFT cannot apply text transformations")
            prompt = list(row[config.prompt_token_ids_key])
            completion = list(row[config.completion_token_ids_key])
            if not prompt or not completion or any(type(t) is not int or t < 0 for t in prompt + completion):
                raise ValueError("Exact-ID SFT requires nonempty nonnegative integer token IDs")
            eos = getattr(tokenizer, "eos_token_id", None)
            if config.append_eos_token and eos is not None and completion[-1] != eos:
                completion.append(int(eos))
            if len(prompt) + len(completion) > config.max_seq_len:
                raise ValueError("Exact-ID SFT trajectory exceeds max_seq_len; refusing truncation")
            if think_close_tokens and not _contains_subsequence(completion, think_close_tokens):
                continue
            sequences.append(prompt + completion)
            prompt_lens.append(len(prompt))
            continue
        if config.prompt_key not in row:
            raise KeyError(f"Prompt key {config.prompt_key!r} missing from row.")
        if config.completion_key not in row:
            raise KeyError(f"Completion key {config.completion_key!r} missing from row.")

        prompt_text = str(row[config.prompt_key])
        if config.apply_chat_template:
            prompt_text = _apply_chat_template(
                tokenizer,
                prompt_text,
                enable_thinking=bool(config.chat_template_enable_thinking),
            )
        if config.force_think_prefix and f"<{config.thinking_tag}>" not in prompt_text.lower():
            prompt_text = prompt_text + f"<{config.thinking_tag}>\n"

        continue_from_open_think = prompt_prefills_open_think(prompt_text)
        prompt_tokens = list(tokenizer.encode(prompt_text, add_special_tokens=False))
        prompt_len = len(prompt_tokens)
        if prompt_len <= 0:
            continue

        completion_text = adapt_completion_to_thinking_template(
            str(row[config.completion_key]),
            continue_from_open_think=continue_from_open_think,
        )
        if config.drop_response_xml:
            completion_text = strip_response_xml_envelope(completion_text)
        completion_tokens = list(tokenizer.encode(completion_text, add_special_tokens=False))
        eos_token_id = getattr(tokenizer, "eos_token_id", None)
        if config.append_eos_token and eos_token_id is not None:
            if not completion_tokens or completion_tokens[-1] != eos_token_id:
                completion_tokens.append(int(eos_token_id))

        full_sequence = prompt_tokens + completion_tokens
        if len(full_sequence) > int(config.max_seq_len):
            full_sequence = full_sequence[: int(config.max_seq_len)]
        if len(full_sequence) < 2:
            continue

        effective_prompt_len = min(prompt_len, max(1, len(full_sequence) - 1))
        if think_close_tokens:
            completion_start = effective_prompt_len
            completion_region = full_sequence[completion_start:]
            if not _contains_subsequence(completion_region, think_close_tokens):
                continue

        sequences.append(full_sequence)
        prompt_lens.append(effective_prompt_len)

    return sequences, prompt_lens


def _make_sft_config(config_path: Path, payload: Mapping[str, Any]) -> SFTConfig:
    job_table = _coerce_table(payload, "job", default={})
    model_table = _coerce_table(payload, "model")
    data_table = _coerce_table(payload, "data")
    sft_table = dict(_coerce_table(payload, "sft", default={}))
    checkpoint_table = _coerce_table(payload, "checkpoint", default={})
    resume_table = _coerce_table(payload, "resume", default={})
    wandb_table = _coerce_table(payload, "wandb", default={})
    optimizer_table = _coerce_table(payload, "policy_optimizer", default={})
    eval_table = _coerce_table(payload, "eval", default={})

    sft_table["train_dtype"] = _resolve_torch_dtype(sft_table.get("train_dtype"))

    return SFTConfig(
        eval=SFTEvaluationConfig(config_file=_resolve_path(eval_table.get("config_file"), base_dir=config_path.parent),
                                 interval_steps=int(eval_table.get("interval_steps", 0))),
        job=JobConfig(
            config_file=str(config_path),
            description=str(job_table.get("description", "climbing_gym SFT run") or "climbing_gym SFT run"),
            print_config=bool(job_table.get("print_config", False)),
        ),
        model=ModelConfig(
            name=str(model_table.get("name", "") or ""),
            path=_resolve_path(model_table.get("path"), base_dir=config_path.parent),
            config=_coerce_table(model_table, "config", default={}),
        ),
        data=DataConfig(
            dataset_name=str(data_table.get("dataset_name", "") or ""),
            dataset_config=str(data_table.get("dataset_config", "") or ""),
            dataset_path=_resolve_path(data_table.get("dataset_path"), base_dir=config_path.parent),
            split=str(data_table.get("split", "train") or "train"),
            max_rows=int(data_table.get("max_rows", 0) or 0),
            seed=int(data_table.get("seed", sft_table.get("seed", 0)) or 0),
            shuffle=bool(data_table.get("shuffle", sft_table.get("shuffle", True))),
            epoch=int(data_table.get("epoch", 0) or 0),
        ),
        sft=SFTTrainerConfig(**sft_table),
        policy_optimizer=OptimizerConfig(
            type=str(optimizer_table.get("type", "adamw") or "adamw"),
            learning_rate=float(optimizer_table.get("learning_rate", 1e-6)),
            weight_decay=float(optimizer_table.get("weight_decay", 0.0)),
            fused=_coerce_optional_bool(optimizer_table.get("fused")),
        ),
        checkpoint=CheckpointManagerConfig(
            root_dir=_resolve_path(checkpoint_table.get("root_dir"), base_dir=config_path.parent),
            save_interval=int(checkpoint_table.get("save_interval", 0)),
            save_final=bool(checkpoint_table.get("save_final", True)),
        ),
        resume=ResumeConfig(
            checkpoint_dir=_resolve_path(resume_table.get("checkpoint_dir"), base_dir=config_path.parent),
            continue_step_numbers=bool(resume_table.get("continue_step_numbers", True)),
        ),
        wandb=WandbConfig(
            dir=_resolve_path(wandb_table.get("dir"), base_dir=config_path.parent),
            enabled=bool(wandb_table.get("enabled", True)),
            project=str(wandb_table.get("project", "climbing-gym") or "climbing-gym"),
            entity=str(wandb_table.get("entity", "") or "").strip() or None,
            run_name=str(wandb_table.get("run_name", "") or "").strip() or None,
            mode=str(wandb_table.get("mode", "online") or "online").strip(),
        ),
    )


def load_run_from_toml(config_path: str | Path) -> ResolvedSFTRun:
    path = Path(_resolve_path(config_path))
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    if not isinstance(payload, Mapping):
        raise TypeError("SFT TOML config must parse to a mapping.")
    config = _make_sft_config(path, payload)

    model_name = str(config.model.name or "").strip().lower()
    model_path = str(config.model.path)
    if not model_name:
        raise ValueError("`model.name` is required.")
    if not model_path:
        raise ValueError("`model.path` is required.")

    if config.job.print_config:
        print(json.dumps(asdict(config), indent=2, sort_keys=True, default=_json_default))

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

    policy_model = build_model(
        model_name,
        model_dir=(resume_policy_path or model_path),
        **dict(config.model.config),
    )
    tokenizer = AutoTokenizer.from_pretrained(
        resume_policy_path or model_path,
        trust_remote_code=True,
    )
    policy_optimizer = _build_policy_optimizer(policy_model, config.policy_optimizer.to_dict())
    resume_state = None
    if resume_checkpoint_dir:
        state_file = Path(resume_checkpoint_dir) / "sft_state.pt"
        if state_file.exists():
            resume_state = torch.load(state_file, map_location="cpu", weights_only=False)
            policy_optimizer.load_state_dict(resume_state["optimizer"])
            random.setstate(resume_state["python_rng"])
            torch.set_rng_state(resume_state["torch_rng"])
            if torch.cuda.is_available() and resume_state.get("cuda_rng") is not None:
                torch.cuda.set_rng_state_all(resume_state["cuda_rng"])
    checkpoint_manager = CheckpointManager(config.checkpoint)
    wandb_logger = WandbLogger(config.wandb, config_path=path, resolved_config=config)

    return ResolvedSFTRun(
        policy_model=policy_model,
        tokenizer=tokenizer,
        policy_optimizer=policy_optimizer,
        config=config,
        checkpoint_manager=checkpoint_manager,
        wandb_logger=wandb_logger,
        start_step=int(start_step),
        resume_state=resume_state,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run climbing_gym SFT training from a single TOML config.")
    parser.add_argument("config", type=str, help="Path to the SFT TOML config.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_arg_parser().parse_args(list(argv) if argv is not None else None)
    run = load_run_from_toml(args.config)
    cleanups = [('W&B/local logger', run.wandb_logger.close)]
    failure = None
    try:
        return _train_sft(run, cleanups)
    except BaseException as error:
        failure = error
        raise
    finally:
        _cleanup_resources(reversed(cleanups), failure)


def _train_sft(run: ResolvedSFTRun, cleanups) -> int:
    policy_model = run.policy_model
    tokenizer = run.tokenizer
    policy_optimizer = run.policy_optimizer
    config = run.config
    sft_config = config.sft
    checkpoint_manager = run.checkpoint_manager
    wandb_logger = run.wandb_logger

    train_device = torch.device(sft_config.train_device)
    evaluator = None
    if config.eval.config_file:
        policy_model.to('cpu')
        evaluator = PeriodicSFTEvaluator(config.eval, model=policy_model, optimizer=policy_optimizer,
            train_device=train_device, output_dir=wandb_logger.log_dir/'evaluations')
        cleanups.append(('native periodic evaluator', evaluator.close))
    if sft_config.train_dtype is None:
        policy_model.to(train_device)
    else:
        policy_model.to(device=train_device, dtype=sft_config.train_dtype)

    for state in policy_optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(train_device)

    compiled_loss_fn = maybe_compile(
        packed_sft_loss_from_model,
        dynamic=bool(sft_config.compile_dynamic),
        fullgraph=bool(sft_config.compile_fullgraph),
        backend=str(sft_config.compile_backend),
        mode=str(sft_config.compile_mode),
    )

    dataset_rows = _load_rows(config.data)
    total_rows = len(dataset_rows)
    if total_rows <= 0:
        raise ValueError("SFT dataset resolved to zero rows.")

    batch_size = int(sft_config.batch_size)
    if batch_size <= 0:
        raise ValueError("sft.batch_size must be > 0")
    num_epochs = int(sft_config.num_epochs)
    if num_epochs <= 0:
        raise ValueError("sft.num_epochs must be > 0")
    log_interval_steps = int(sft_config.log_interval_steps)
    if log_interval_steps <= 0:
        raise ValueError("sft.log_interval_steps must be > 0")

    total_optimizer_steps = (total_rows + batch_size - 1) // batch_size
    print(
        "[sft] "
        f"starting rows={total_rows} batch_size={batch_size} num_epochs={num_epochs} "
        f"steps_per_epoch={total_optimizer_steps} device={train_device} "
        f"dtype={sft_config.train_dtype or 'auto'}",
        flush=True,
    )

    fingerprint = hashlib.sha256(json.dumps(list(dataset_rows), sort_keys=True, default=_json_default).encode()).hexdigest()
    recipe = asdict(sft_config)
    recipe.pop('num_epochs')
    recipe_sha256 = hashlib.sha256(json.dumps(recipe, sort_keys=True, default=_json_default).encode()).hexdigest()
    resume_state = run.resume_state
    if resume_state and (resume_state["dataset_sha256"] != fingerprint
                         or resume_state["batch_size"] != batch_size
                         or resume_state["seed"] != sft_config.seed
                         or resume_state.get("recipe_sha256", recipe_sha256) != recipe_sha256):
        raise ValueError("SFT resume dataset, batch size or shuffle seed changed")
    start_epoch = int(resume_state["next_epoch"]) if resume_state else 0
    start_offset = int(resume_state["next_batch_offset"]) if resume_state else 0
    next_epoch, next_offset = start_epoch, start_offset

    def save_checkpoint(step, *, is_final):
        checkpoint_dir = checkpoint_manager.maybe_save(step=step, is_final=is_final,
                                                       policy_model=policy_model)
        if checkpoint_dir is not None:
            state = dict(optimizer=policy_optimizer.state_dict(), next_epoch=next_epoch,
                         next_batch_offset=next_offset, dataset_sha256=fingerprint,
                         batch_size=batch_size, seed=sft_config.seed, recipe_sha256=recipe_sha256,
                         python_rng=random.getstate(),
                         torch_rng=torch.get_rng_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)
            temp = checkpoint_dir / "sft_state.pt.tmp"
            torch.save(state, temp)
            temp.replace(checkpoint_dir / "sft_state.pt")
            if callable(getattr(tokenizer, "save_pretrained", None)):
                tokenizer.save_pretrained(checkpoint_dir / "policy")

    global_step = int(run.start_step)
    last_completed_step = global_step - 1

    for epoch in range(start_epoch, num_epochs):
        indices = list(range(total_rows))
        if sft_config.shuffle and len(indices) > 1:
            random.Random(int(sft_config.seed) + epoch).shuffle(indices)

        for batch_offset in range(start_offset if epoch == start_epoch else 0, total_rows, batch_size):
            next_offset = batch_offset + batch_size
            next_epoch = epoch
            if next_offset >= total_rows:
                next_epoch, next_offset = epoch + 1, 0
            step_started_at = time.perf_counter()
            batch_indices = indices[batch_offset : batch_offset + batch_size]
            rows = [dataset_rows[idx] for idx in batch_indices]
            sequences, prompt_lens = _prepare_sft_sequences(
                tokenizer,
                rows,
                config=sft_config,
            )
            packed_batches = pack_sequences(
                sequences,
                prompt_lens=prompt_lens,
                max_tokens=int(sft_config.max_packed_tokens),
            )
            if not packed_batches:
                continue

            total_supervised_tokens = float(
                sum(
                    (batch.targets.numel() if sft_config.loss_on_full_sequence else batch.valid_token_count)
                    for batch in packed_batches
                )
            )
            if total_supervised_tokens <= 0.0:
                continue

            policy_model.train()
            policy_optimizer.zero_grad(set_to_none=True)

            loss_sum = 0.0
            accuracy_sum = 0.0
            packing_efficiency_sum = 0.0
            packed_token_count = 0.0
            supervised_token_count = 0.0

            epoch_batches = list(packed_batches)
            if sft_config.shuffle_packed_batches and len(epoch_batches) > 1:
                random.Random(int(sft_config.seed) + global_step).shuffle(epoch_batches)

            for packed_batch in epoch_batches:
                device_batch = packed_batch.to(train_device)
                loss, accuracy, valid_count = compiled_loss_fn(
                    policy_model,
                    batch=device_batch,
                    loss_on_full_sequence=bool(sft_config.loss_on_full_sequence),
                )
                scale_sft_loss_for_packed_batch(
                    loss,
                    token_count=valid_count,
                    total_token_count=total_supervised_tokens,
                ).backward()

                weight = float(valid_count.detach().item())
                loss_sum += float(loss.detach().item()) * weight
                accuracy_sum += float(accuracy.detach().item()) * weight
                supervised_token_count += weight
                packed_token_count += float(packed_batch.token_count)
                packing_efficiency_sum += float(packed_batch.packing_efficiency) * float(packed_batch.sequence_count)

            grad_norm = compute_grad_norm(policy_model)
            max_grad_norm = float(sft_config.max_grad_norm)
            if max_grad_norm > 0.0:
                torch.nn.utils.clip_grad_norm_(policy_model.parameters(), max_grad_norm)
            policy_optimizer.step()
            policy_optimizer.zero_grad(set_to_none=True)

            mean_loss = loss_sum / max(1.0, supervised_token_count)
            mean_accuracy = accuracy_sum / max(1.0, supervised_token_count)
            mean_packing_efficiency = packing_efficiency_sum / max(
                1.0,
                float(sum(batch.sequence_count for batch in epoch_batches)),
            )
            step_time_seconds = time.perf_counter() - step_started_at

            metrics = {
                "sft_loss": mean_loss,
                "sft_token_accuracy": mean_accuracy,
                "sft_grad_norm": float(grad_norm),
                "sft_supervised_tokens": supervised_token_count,
                "sft_packed_tokens": packed_token_count,
                "sft_packing_efficiency": mean_packing_efficiency,
                "sft_examples": float(len(rows)),
                "sft_sequences": float(len(sequences)),
                "sft_packed_batches": float(len(epoch_batches)),
                "epoch": float(epoch),
                "step_time_seconds": step_time_seconds,
                "tokens_per_second": supervised_token_count / max(step_time_seconds, 1e-6),
            }
            if evaluator is not None:
                metrics.update(evaluator.evaluate(global_step))
            metrics['full_step_seconds'] = time.perf_counter()-step_started_at
            wandb_logger.log_step(step=global_step, metrics=metrics)
            if (global_step % log_interval_steps) == 0:
                print(
                    "[sft] "
                    f"step={global_step} epoch={epoch} batch_start={batch_offset} "
                    f"loss={mean_loss:.6f} acc={mean_accuracy:.4f} "
                    f"supervised_tokens={supervised_token_count:.0f} "
                    f"packed_tokens={packed_token_count:.0f} "
                    f"packing_eff={mean_packing_efficiency:.4f} "
                    f"grad_norm={grad_norm:.4f} "
                    f"step_time_s={step_time_seconds:.2f} "
                    f"tok_per_s={metrics['tokens_per_second']:.1f}",
                    flush=True,
                )

            save_checkpoint(global_step, is_final=False)
            last_completed_step = global_step
            global_step += 1

    if last_completed_step >= 0:
        save_checkpoint(last_completed_step, is_final=True)
        print(f"[sft] completed last_step={last_completed_step}", flush=True)


    return 0


if __name__ == "__main__":
    raise SystemExit(main())
