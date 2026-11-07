# Running Climbing Gym

## Configuration and examples

Every TOML path is relative to its configuration file, with `~`, `$VAR`, and
`${VAR}` expansion. An unset variable fails early with its name. The examples
require `CLIMBING_GYM_MODEL` to identify an existing, immutable local checkpoint
with weights, config, and tokenizer. They neither download weights nor assume a
SimplePT installation. CLI file arguments are relative to the shell's current
directory.

`configs/train_*.toml` provides standalone one-step examples for the ten training
methods and optional CFOP adapter; `configs/eval_*.toml` covers every environment.
`configs/sft.toml` consumes verified projection output. The
[complete Futoshiki recipe](../examples/futoshiki_hill_climb/README.md) adds staged
historical data replay and regeneration. The
[Knights Value MIS+SC example](../examples/knights_value_mis_sc/README.md) adds the
recent privileged-critic route with readiness warmup and exact continuation.
Models can be Base, Instruct, or Thinking;
choose chat templating and the prefill to match that checkpoint's tokenizer.

Strict arithmetic and puzzle examples append `<thinking>\n`, require a natural
`</thinking>` close, and finish with `<answer>...</answer>`. CFOP instead uses its
[verified phase-program protocol](cfop.md), with no incompatible thinking prefill;
its solver bindings are optional. Ordinary chat and raw prompt modes remain
available. Check saved actual prompts before long runs, especially when changing
a checkpoint's chat template.

The small batches make configurations easy to inspect. They are not throughput
recommendations. Resize them after measuring your GPU; do not carry a small
request cap into a large run without validation. The generated example puzzles
are for schema and verifier checks. Use separate training/evaluation datasets
and seeds for research claims.

## Native preflight

Before any training or evaluation:

1. Prepare actual environment/harness prompt IDs using `climbing-gym preflight`.
   Use `--evaluation` with standalone evaluation TOML or `--periodic-eval` with
   the nested evaluation section of a training TOML. No model weights load in
   this step; the local tokenizer is required. The resulting JSON reports the
   token budget, prompt lengths, and `limits.representative_batch_size`.
2. Inspect every concurrency limit: `native_max_batch_size`, `max_num_seqs`,
   rollout `max_inflight_requests`, logical/effective batch sizes, rollout count,
   and request budget. The representative native batch is the smaller of the
   effective engine batch and the active-request cap. Dynamic sampling must
   permit whole groups in both concurrency and request budget.
3. Benchmark those exact prompts with that batch, the intended token budget,
   and sampled-token log probabilities. The native benchmark always records
   sampled log probabilities and forces complete output budgets for timing.
   Include any `behavior_topk` sparse probability transfer in the measurement.
4. Inspect tokens/second and peak allocated/reserved GPU memory. Configure
   `inference.preflight_report_path` to that JSON. Use
   `inference.preflight_prior_report_path` for a comparable previous report.
   Without one, the run explicitly records `initial_baseline`. Priors must match
   model, workload, budget, native options and hardware; historical measurements
   are context, not automatically equivalent comparisons.
5. Launch, then inspect actual rollout throughput and full-step timings. Logs
   continue throughout training. Speed guards belong only to initial setup;
   there are no automatic training stops or speed probes for fluctuations.

A complete example for `configs/train_mis_sc.toml`:

```bash
uv run wandb login
uv run climbing-gym preflight --config configs/train_mis_sc.toml \
  --output artifacts/mis_sc/prompts.json
uv run climbing-gym benchmark --model "$CLIMBING_GYM_MODEL" \
  --prompts-json artifacts/mis_sc/prompts.json --batches 2 \
  --output-tokens 4096 --max-model-len 8192 --device cuda --dtype bfloat16 \
  --temperature 1 --top-k -1 --top-p 1 --behavior-topk 32 \
  --cuda-graphs --bf16-weight-packing --max-prefill-tokens 4096 \
  --output artifacts/mis_sc/speed.json
uv run climbing-gym train configs/train_mis_sc.toml
```

These flags match this example, including BF16 packing and sparse score-centering
metadata. Runtime defaults differ from benchmark defaults: runtime
`compile_decode=true` needs `--compile-decode` unless disabled in the config;
runtime `native_bf16_weight_packing=true` needs `--bf16-weight-packing`.
Match CUDA graphs, FP8 MLP/KV/head, adaptive batching, context size, prefill chunk,
precision, temperature, top-k/top-p, and repetition penalty explicitly.
The examples disable decode compilation and enable CUDA graphs. FP8 is optional,
requires compatible hardware (SM89+), and can change memory and numerical behavior.

One benchmark can include multiple batch sizes when train/evaluation share the
same token budget, probability metadata, model, and options. Different periodic
evaluation budgets require their own report via `[eval] preflight_report_path`.
Multi-turn training also requires representative continuation lengths beyond the
initial first-turn workload. Keep source benchmark/hardware details when comparing
runs; see [native provenance](porting/native_engine.md) for historical references.

## Memory and checkpoints

The RL examples use `cpu_master_adamw` with a 48 MiB bounded GPU workspace for
FP32 master weights/moments. Value MIS+SC uses mapped critic state with a disposable
backing file under `runs/`. Native buffers are released before learner updates.
These choices bound optimizer workspace; activations, logits, inference cache,
model copies, and actor/critic residency still consume GPU memory. Full-model PPO
holds both actor and critic during learning. Choose packing/context budgets for
your device. A one-step configuration is not proof a 1.2B run fits every GPU.

Checkpoints contain HF-style `policy/`, tokenizer artifacts, `metadata.json`,
optional `value_model.pt`, and exact optimizer/RNG state in `training_state.pt`.
Dataset continuation preserves cursor/order, queued groups, and native seed state
when available. Keep native batching unchanged for identical stochastic replay.
Set `[resume] checkpoint_dir="../runs/METHOD/checkpoints/step_..."` to the actual
saved directory. `allow_weights_only=true` explicitly permits older checkpoints
without exact optimizer state. Inspect metadata and logs to confirm continuation.
Mapped optimizer working files are disposable; durable checkpoints retain exact
FP32 state.

## Logging and evaluation

Training defaults to online W&B project `climbing-gym` under the current W&B
account, including configuration, step metrics, and periodic evaluations. `wandb`
is a base dependency, so the default install supports this behavior. Authenticate
before launch. Leave `[wandb] entity` unset to use your configured/default W&B
account, or set it explicitly to target a team or another entity. `[wandb] project`
can also be overridden. For a deliberately local run, set
`enabled=false`; `mode="offline"` explicitly selects SDK offline logging. Online
initialization failures are actionable errors, not silent local fallback.

Resolved configuration, flushed local metrics, trajectories, and checkpoints
remain local. Paths can be selected in `[train_logging]`, `[checkpoint]`, and
`[wandb]`. `CLIMBING_GYM_LOG_DIR` and `CLIMBING_GYM_WANDB_DIR` configure defaults.
Never store credentials in committed TOML.

Standalone evaluation uses `[logging]`; periodic RL evaluation uses
`[eval.logging]`. Each invocation saves complete actual model prompts, every
response, the expected answer and verifier outcome in Markdown. Trace selection
covers available difficulty and success/failure strata. `max_traces=0` saves all
trajectories; the default saves a representative subset without truncating any
response. Periodic evaluations preserve separate trace files. Evaluation disables
dynamic group rejection so uniform-success/failure groups remain in accuracy.

For periodic RL evaluation set `[eval] interval`, dataset/env/rollout overrides,
and an appropriate preflight report. For SFT set `[eval] config_file` and
`interval_steps` to a standalone evaluation TOML; its preflight is validated once
before training. Evaluations reuse the native engine and sync current weights.

## Projection to supervised training

Prepare a reference without a teacher model, run its independent preflight,
then generate verified training trajectories:

```bash
mkdir -p runs/projection
uv run python examples/prepare_projection.py > runs/projection/experts.jsonl
uv run climbing-gym project --model "$CLIMBING_GYM_MODEL" \
  --rows runs/projection/experts.jsonl --environment knights_knaves \
  --output runs/projection --batch-size 1 --max-tokens 4096 \
  --max-model-len 12288 --fixed-answer --preflight
# Inspect runs/projection/preflight.json and its complete trace report first.
uv run climbing-gym project --model "$CLIMBING_GYM_MODEL" \
  --rows runs/projection/experts.jsonl --environment knights_knaves \
  --output runs/projection --batch-size 1 --max-tokens 4096 \
  --max-model-len 12288 --fixed-answer
uv run climbing-gym sft configs/sft.toml
```

Projection accepts `reference_source="algorithmic"` or `"same_model_verified"`
only, and independently verifies references and final answers. Its proposal
conditioning may include the verified guide; target likelihood is scored under
the ordinary frozen base policy over the complete trajectory. Fixed-answer mode
preserves the reference answer verbatim. It is greedy mean raw-log-probability
ascent, without MH acceptance/proposal correction or detailed balance.
Intermediate prose deductions are not generally independently verified.

The driver saves complete `sampling_traces.md`, state, timing, all candidates,
and verified `projected_train.jsonl`. It resumes only with matching data/config
identity and a matching preflight. Supply `--prior-benchmark` when a comparable
prior is available; otherwise it records an initial baseline. Its synchronous
batch has no hidden request semaphore, but suffix budgets and completed chains
can reduce utilization.

SFT accepts text rows or exact `prompt_token_ids`/`completion_token_ids`. Exact-ID
rows preserve prompt masking and one EOS; incompatible transforms or truncation
fail. Projection prompts are already rendered: leave `apply_chat_template=false`.
The standard SFT loop uses ordinary AdamW and a full vocabulary head, so its memory
profile differs from bounded RL optimizers. Reduce packing and sequence lengths
for your GPU; do not discard the end of verified exact-ID trajectories. Pure SFT
has no rollout queue, but project policy still requires a representative native
speed/memory check before training; projection's matching-model preflight provides
that evidence. SFT periodic evaluation needs its own evaluation preflight.

## Development and local artifacts

```bash
uv run pytest -q
uv build
```

Run these checks locally. GPU tests skip when CUDA is unavailable. Artifacts,
weights, logs, credentials and caches are ignored. Keep a copy of actual evaluation
traces with each experiment. Source provenance and validation limitations live in
`docs/porting/`; this code has no runtime import from the original SimplePT tree.
