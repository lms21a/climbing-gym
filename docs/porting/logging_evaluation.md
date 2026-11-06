# W&B, local logging, native evaluation, and preflight

Source: `/home/lukas928/SimplePT/simple-pt-memory-defaults`, commit `6edae8f`,
`src/simple_pt_2/trainers/logger.py` and `src/simple_pt_2/eval.py`.
The source checkout remains unchanged. `climbing_gym.evaluation_traces` and
`climbing_gym.preflight` are new standalone helpers. Native benchmark metadata,
sampling options, and tokenizer-only native prompt rendering were extended with
parent approval after the inference owner finished. No external publication,
actual training, GPU inference, or real W&B run was performed by this agent.

## Logging contract

`WandbLogger(config, *, config_path: Path, resolved_config=None)` preserves the
trainer interface `log_step(step=..., metrics=...)` / `close()`. Runtime passes
its fully resolved configuration, including preflight measurements. W&B defaults
are online `lukas928/simple-pt`, configured by the shared `Wandb` dataclass.
An owned W&B Run receives initialization/configuration, training metrics and
periodic `eval/*` metrics. Run methods rather than process-global W&B methods
prevent one logger from finishing another run. `close(exit_code=...)` is
idempotent; context manager exit marks failed runs with exit code 1.

Every run writes `resolved_config.json` and flushed/fsynced `metrics.jsonl`
before remote metric logging. Non-finite metric values are JSON `null`.
Reusing an explicit directory selects unique metrics/configuration filenames and
preserves previous run files. The checkpoint manager remains runtime-owned.
W&B errors are actionable and never silently downgrade an online run.

`wandb.enabled=false` or `wandb.mode="disabled"` skips the W&B import entirely;
local files remain enabled. `mode="offline"` explicitly selects the SDK's
offline mode. Relative configured directories are resolved against the config
file. Directory priority is `wandb.dir`, `CLIMBING_GYM_WANDB_DIR`, then a unique
run directory under `CLIMBING_GYM_LOG_DIR` (default: config-parent `runs/`).

## Evaluation and actual traces

The existing public symbols `EvalConfig`, `EvalLoggingConfig`,
`ResolvedEvalRun`, `_trajectory_record`, `build_eval_run`, `run_eval`, and
`summarize_trajectories` remain. The CLI accepts a positional TOML config or
`--config CONFIG`. Relative dataset/model/log paths are config-relative.
Only the native LFM adapter is constructed. Shared environment configuration
supports GSM8K, Countdown, Knights and Knaves, and Futoshiki.

Evaluation fixes the selected dataset and restores its initial cursor for each
periodic invocation. It disables dynamic group rejection, which would otherwise
remove uniform successes/failures and bias accuracy. Summary pass@k retains the
source's first-k completion interpretation, ordered by rollout index. Explicit
feedback success takes precedence over positive dense rewards.

Each completed evaluation always creates a Markdown trace file, even with no
logging paths configured. `logging.trace_path` selects it; otherwise a JSON
logging path anchors its `.traces.md` sibling, or the default is
`evaluations/traces.md`. Standalone CLI defaults are config-relative. Colliding
names receive numeric suffixes so periodic traces survive. The summary JSON
contains `trace_path`; programmatic callers inspect `run.trace_paths`.
Training's nested configuration uses `[eval.logging]`.

`max_traces=8` is a preference: all available difficulty/outcome strata receive
at least one trace, even if their count exceeds eight. Set zero to retain every
trajectory. Selection is deterministic across completion orders. Each trace
includes original task prompt, each actual model prompt (decoded from actual
prompt token IDs if a tokenizer is available), complete response for every turn,
complete final response, expected answer, difficulty, pass/fail, reward and
verifier details. Markdown fences adapt to arbitrary embedded backticks. No
model response is truncated. Optional JSONL records likewise retain task prompt,
target and complete per-turn responses/token IDs.

## Preflight workflow and exact defaults

Preflight validates a saved benchmark once during startup; it never launches
live speed probes and never stops training on throughput fluctuations. Runtime
retains rollout throughput and full-step timing throughout training.

1. Prepare actual first-turn workload prompts without loading model weights:
   `python -m climbing_gym.preflight --config run.toml --output prompts.json`.
   Use `--evaluation` for standalone evaluation settings or `--periodic-eval`
   for the nested periodic evaluation configuration. Output reports the intended
   output budget and `limits.representative_batch_size`; first-turn variable
   prompt lengths and expected answers are saved alongside exact token IDs.
   Multi-turn workloads also need representative continuation lengths: a first
   request cannot predict them.
2. Run `python -m climbing_gym.inference.benchmark --model CHECKPOINT
   --prompts-json prompts.json --batches RESOLVED_BATCH --output-tokens BUDGET
   --output speed.json`, supplying the actual context, dtype and inference flags.
   Sampling log probabilities are always included. Several batch sizes may be
   benchmarked in one report for train/eval configurations sharing a token budget.
3. Set `[inference] preflight_report_path="speed.json"`. Set optional
   `preflight_prior_report_path` to a comparable earlier report. Environment
   equivalents are `CLIMBING_GYM_PREFLIGHT_REPORT` and
   `CLIMBING_GYM_PREFLIGHT_PRIOR_REPORT`.
4. Different periodic-eval budgets can use `[eval] preflight_report_path` and
   `preflight_prior_report_path` overrides. Runtime validates these once at
   startup through `validate_eval_preflight(config, eval_run)`.

**Match defaults explicitly:** benchmark `--compile-decode` defaults off, whereas
runtime `inference.compile_decode` defaults true. Add `--compile-decode` unless
runtime disables it or CUDA graphs supersede compilation. Benchmark
`--bf16-weight-packing` defaults off, whereas runtime
`inference.native_bf16_weight_packing` defaults true: add
`--bf16-weight-packing` when using graphs with default runtime settings. Match
`--cuda-graphs`, `--max-prefill-tokens`, `--fp8-mlp`, `--fp8-kv`,
`--fp8-lm-head`, `--adaptive-batching`, dtype/context and sampling settings.
Benchmark accepts `--temperature`, `--top-k` (`-1` disables it), `--top-p`,
`--repetition-penalty`, and `--behavior-topk`; sparse behavior heads include their
CPU transfer in timing. Runtime currently supports this benchmark's single
CUDA-graph setting. Benchmark temperature defaults 1, shared sampling defaults
0: pass the configured value explicitly. For fair comparison to the original
`scripts/benchmark_lfm25_inference.py`, set `OMP_NUM_THREADS=8`, because that
script sets PyTorch's CPU thread count to eight.

Validation checks positive batch/request caps, exact model/output budget,
representative prompts, sampled log probabilities, native acceleration options,
context/dtype, measured full output budgets, finite positive throughput and CUDA
peak allocated/reserved memory. It records report SHA256 and resolved native
batch, sequence/request caps. Native adapter defaults to batch 64; absent a
runner request cap, the cap is effective groups times rollout count. No
unrelated restrictive cap is imported. Current/prior ratios cover throughput
and both memory measures. First-use without a prior is explicitly
`comparison.status="initial_baseline"`; thereafter supply a relevant prior.
Prior reports must have the same native JSON schema, checkpoint, token budget,
batch, and (when supplied) hardware name. Historical source script output needs
explicit normalization for comparison; there is no ambiguous heuristic parser.

## Verification

CPU fixture tests use `/home/lukas928/SimplePT/.venv/bin/python`, with no real
W&B initialization or GPU/model evaluation:

`PYTHONPATH=src /home/lukas928/SimplePT/.venv/bin/python -m pytest
 tests/test_logging_evaluation.py tests/test_preflight.py -q`

The tests cover durable local records, complete configuration, owned W&B
lifecycle, online/disabled behavior, initialization errors, collision-safe logs,
lossless embedded Markdown/long responses, difficulty and success/failure
coverage, evaluation summaries/cursor restore/engine cleanup, all preflight
mismatch classes, first-baseline/prior ratios, independent periodic-eval budget,
real-request tokenizer-only preparation, and exact renderer parity for plain,
ordinary chat, thinking prefills, mappings and exact token-ID requests. Mocked
responses are test fixtures, not claims of model quality. GPU evaluation and
actual model trace reports belong to independent release verification.
