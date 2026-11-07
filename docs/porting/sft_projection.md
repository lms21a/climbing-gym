# SFT and native projection sampling

The port uses existing SimplePT implementations, with native execution extracted into a reusable sampler. Source worktrees were read only. No GPU training or model evaluation was performed during this port.

## Provenance

| Destination | Source and revision |
| --- | --- |
| `src/climbing_gym/sft.py` | `simple-pt-memory-defaults/src/simple_pt_2/sft.py`, worktree HEAD `6edae8f`; last path commit `b8ff387` |
| `src/climbing_gym/sft_data.py` | Focused formatting functions from `simple_pt/envs/sft_data.py` and `simple_pt/core/parsing_math.py` in that worktree |
| `src/climbing_gym/projection_sampling.py` | `simple-pt-futoshiki-expert-projection/src/simple_pt_2/projection_sampling.py`, path origin `720d7d4`; SHA256 `f6daab948fff27d3fa48985dfd25ae51e858a0a6ee6453744e3535e17ceb17ea` |
| `src/climbing_gym/constrained_projection.py` | `simple-pt-nemotron-e2s/src/simple_pt_2/constrained_projection.py`, origin `6f2a22d`; SHA256 `0bf011271e4b3cef776f0f52588a9ad79481cd02d9fc6130c6d07ca1476c9b7c` |
| `src/climbing_gym/sampling.py` | Reusable native driver extracted from the block/iteration flow in `simple-pt-nemotron-e2s/scripts/run_nemotron_projection.py`, using the above original primitives and immutable-answer adapter |
| `src/climbing_gym/knights_solver_projection.py` | Focused solver-guide/conclusion functions from memory-defaults, last path commit `6c3b969`; historical experiment-specific exporter omitted |

## Exact sampling semantics

The historical experiment calls its revision count `mcmc_steps`, but its acceptance rule is **greedy mean raw base-policy log-probability ascent**. It is not exact Metropolis–Hastings and has no acceptance ratio, proposal correction or detailed-balance guarantee. The driver calls this `revision_steps` to avoid suggesting otherwise.

`ProjectionState.revise` preserves strict `candidate_mean > current_mean` acceptance, unconditional block extension, sampled suffix log probabilities, and scoring over the entire current trajectory. Random cuts include zero and use the original deterministic `seed:row_id:block:iteration` seed. `NativeProjectionSampler` implements actual native batched proposals and native `score_completions`, not an abstract chain-state API. Generation and scoring use the same frozen engine. Expert text appears only in proposal conditioning; target scoring uses the ordinary task chat prompt plus `<thinking>\n` and exact completion IDs.

Block mode grows by `max_tokens / blocks` and applies greedy revisions within each block, retaining source EOS handling. This is the source's block-growing behavior: a nonterminal intermediate candidate is scored over its current complete prefix, not an imagined future answer. The fixed-answer mode instead projects a complete reasoning trajectory with an immutable reference answer and EOS. It starts from the verified reference, proposes reasoning continuations, requires a natural `</thinking>` close, appends the original answer verbatim, and scores **all** reasoning, answer, formatting and EOS tokens. Its deterministic cut function and boundary safety margin are the original constrained implementation.

Both proposal builders append the exact existing prefix IDs to encoded chat-prompt IDs. They never decode and re-encode continuation prefixes. The constrained candidate canonicalizer deliberately re-tokenizes the final composed envelope, as in the source. Full base likelihood is recomputed afterward.

The sampler requires an independent verifier and a declared `reference_source` of `algorithmic` or `same_model_verified`. It verifies each input reference, rejects stronger-teacher source declarations, and checks each final trajectory before exporting it for SFT. Environment adapters cover GSM8K, Countdown, Knights and Knaves, and Futoshiki. GSM8K checks the dataset's expected numeric answer; it does not claim to verify every prose deduction. Puzzle solvers independently verify final answers. Free-prose intermediate reasoning is not generally independently verified.

## Entry points and schema

```bash
python examples/prepare_projection.py > experts.jsonl
python -m climbing_gym.sampling --model /local/LFM2.5-1.2B-Instruct \
  --rows experts.jsonl --environment knights_knaves --output runs/projection \
  --batch-size 128 --max-tokens 4096 --fixed-answer --preflight \
  --prior-benchmark /local/relevant-prior-summary.json
# Inspect preflight.json and preflight_traces.md, then repeat without --preflight.
```

Each JSONL row supplies `id`, optional `tier`, `messages` ending in a user request, `expert` with a complete thinking/answer envelope, `reference_answer`, `reference_source`, and `task` with the environment's original structured puzzle or dataset answer. `examples/prepare_projection.py` creates an exhaustive-solver-verified Knights and Knaves reference without model inference or a teacher.

The CLI saves `sampling_state.json`, `timings.jsonl`, `projected_all.json`, verified `projected_train.jsonl`, and `sampling_traces.md`. Resume checks an identity hash of all prepared rows and the complete sampler configuration. Checkpoints occur between complete blocks or fixed-answer revisions; an interrupted block is recomputed with deterministic seeds. Synchronous native batches have one explicit `batch_size` limit and no hidden request semaphore. Same-budget requests are batched together; actual utilization can fall when chains terminate or suffix budgets differ.

Preflight benchmarks representative native proposals with the actual configured reasoning budget and sampled-token log probabilities, followed by raw base scoring; it reports generation speed, scoring time, full-step time and peak GPU allocation. It warms up first, records actual request count and batch capacity, and writes full actual prompt/response/expected/outcome traces. A prior summary contains `tokens_per_second` and `peak_allocated_bytes`; the result uses `comparison.status=compared` and explicit ratios. If no prior exists, omission records `initial_baseline` instead. Relevant source study priors are referenced by `reference_preflight` in the historical projection manifests; compare their workload and device before extracting a prior summary. This port did not manufacture benchmark numbers.

The CLI checks that a saved preflight's model path, row hash, native execution settings, context budget and sampler configuration match before sampling. Model paths should identify immutable local checkpoints; replacing checkpoint files in place requires rerunning preflight. During sampling it retains throughput and full-step timing without speed probes or automatic speed stops. The owner must inspect comparisons and investigate unexplained slowdowns. Shared logging infrastructure's general benchmark validator targets async rollout/evaluation configuration, so this synchronous projection driver retains its richer proposal-workload identity and aligns the comparison convention.

## SFT behavior and evaluation

`python -m climbing_gym.sft CONFIG.toml` retains source chat templating, response-only token masks, optional full-document loss, packed token-count-weighted accumulation, optimizer step logging, native-model checkpoint export and TOML loading. File paths resolve relative to the config file. New runs default to online W&B project `climbing-gym` under the current W&B account (with an optional entity override) and always retain local resolved configuration/metrics via the shared logger.

The original ordinary `<think>` format remains the default. Set `sft.thinking_tag = "thinking"` for the LFM `<thinking>...</thinking><answer>...</answer>` projection contract. The focused formatting helpers preserve both tag families. Ordinary tokenizer chat templates remain available via `sft.apply_chat_template`; projection records already contain fully rendered prompts and must not be templated again.

Rows with both `prompt_token_ids` and `completion_token_ids` take an exact-ID path. Prompt tokens are masked, EOS is supervised exactly once, transforms are rejected, and overlong exact-ID trajectories fail rather than truncate. Text-row truncation remains source-compatible. `backward_completion_loss` retains the source's checkpointed backbone/chunked-head response-only gradient implementation; it is separately available for memory-sensitive update drivers, while the standard SFT loop preserves the original full-head loss implementation.

SFT checkpoints retain `policy/`, tokenizer artifacts, `metadata.json`, and an atomically written `sft_state.pt` containing optimizer state, data hash, recipe hash, next epoch/batch cursor and Python/Torch/CUDA RNG. Resume allows increasing epoch count and verifies unchanged data/batching/tokenization/numerical recipe. Legacy policy-only checkpoints still load with their original step-number continuation; they do not acquire an exact data cursor retroactively.

Optional periodic evaluation reuses the standalone native evaluation configuration:

```toml
[eval]
config_file = "eval_knights.toml"
interval_steps = 100
```

`PeriodicSFTEvaluator` loads that TOML and validates its saved native preflight once before training starts. It reuses the native engine, offloads policy/AdamW tensors around each evaluation, syncs the current policy, restores training device/mode, releases inference buffers, merges `eval/*` metrics into the same W&B/local step, and saves separate complete trace/trajectory/summary files for every evaluation. It never launches speed probes during training. Native actor weights remain resident between evaluations, consistent with the existing training runtime; this is not a fully offloaded actor implementation.

Pure offline SFT has no inference request queue or rollout throughput to invent. Before agent-run training, perform the project-required representative native speed/memory check externally. Configure the evaluation TOML and its preflight when periodic evaluation is needed.

## Validation and remaining scope

CPU verification command:

```bash
PYTHONPATH=src /home/lukas928/SimplePT/.venv/bin/python -m pytest \
  tests/test_sft.py tests/test_projection_sampling.py tests/test_sft_eval.py -q
```

The scoped suite covers source response-only loss, EOS masks, raw documents/chat templates, exact-ID preservation, both thinking tags, strict greedy acceptance, complete-trajectory scoring, fixed-answer composition, deterministic zero-inclusive cuts, native tensor-interface end-to-end drivers, scoring prompts without expert leakage, resume identity checks, bitwise SFT model parity between continuous and resumed training, checkpointed-head gradient parity, preflight comparison labeling, opaque Markdown trace rendering, and periodic evaluation engine reuse/failure restoration.

Native GPU performance, LFM checkpoint integration and model-quality evaluation remain for the independent QA stage. Unit-test stubs are not model evaluations and their strings are not presented as model traces. Historical experiment orchestration, dataset-specific schedules, stronger-teacher studies and unsupported environments were intentionally excluded.
