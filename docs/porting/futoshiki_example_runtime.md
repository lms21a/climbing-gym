# Futoshiki hill-climb runtime example

The example is `examples/futoshiki_hill_climb/`. `recipe.json` declares the actual
stage graph rather than relabeling an unrelated Knights configuration. The
unpublished blog's primary curriculum is projection reset → nine givens/five
inequalities → eight/seven → six/eight. Earlier arithmetic, 4x4 RL, warmup,
regression, and same-lineage specialist branches remain executable prerequisites.

## Sources and numerical behavior

Read-only source evidence was taken from the frozen run protocols/snapshots,
including:

- `futoshiki_solverprojection256_epoch1_flashreinforce750_b128_20261011`,
  commit `856fc743f33774ee46fccd51405264ec6f64a776`.
- `futoshiki_solverprojection256_epoch1_flashreinforce1000_b128_weighted8k_20261011`,
  commit `6288e020e4856d9347868a53f5713d6cd01db55f`.
- `futoshiki5_two_expert_projection_sft_prewarmup_20261015`, selected epoch2
  policy hash `868872d834cb6fd6fe58501f15ee3798035ede416acad1753cdb9e695d1dc031`.
- `futoshiki5_g9i5_flash_mis_sc_async100_from_projection_sft_epoch2_20261017_v3`,
  commit `b0f03d2c19ce8d016bd04ec52658b3d6a8f283f6`.
- `futoshiki5_g8i7_continue100_flash_mis_sc_async_from_g9_expert100_20261017`,
  commit `2c04bb322c358d0d4e6cd7b1fe4809aeed388c64`.
- `futoshiki5_g6i8_continue100_flash_mis_sc_async_from_g8_expert100_20261017`,
  commit `897d3584e2e0dabbc93cad463a69a22b55f6e068`.

Original run-relative paths and input-bank SHA256 values are retained in
`data/manifest.json`; narrative/policy provenance is separate in
`provenance.json`. Original checkouts were unchanged, and no historical weights
were copied. The selected 280-update peak policy was historically deleted; the
recipe retains the final 300-update policy rather than pretending the peak is
available. Reproduction does not promise bitwise-identical stochastic outputs
across changed native kernels, hardware, or library versions.

The last three RL stages retain response-length mean MIS+SC normalization,
strict terminal Futoshiki grading, behavior top-32 probabilities, mask [0.5,5],
BF16 learning, and FP32 master/moment arithmetic. A native 256-response cohort
feeds two ordered 128-response updates with lags zero/one. This differs from the
generic package's default single-update/fixed-denominator recipe. Generated
`train.toml` files describe collection settings and must be run through the
example's `rl.py`, which applies the declared learner-batch geometry.

The native raw-session sampling loop and request seed stride are ported from
`run_fp8_futoshiki_comparison.py`/`run_futoshiki_expert_optimized.py`. The two
later tier transitions preserve the explicit 12,800/25,600 sampling offsets.
GSM8K/Countdown prompt and strict joint reward functions are ported into
`tasks.py`; they use the supplied policy tokenizer and no original runtime
imports. The minimal package change registers the existing strict Futoshiki
reward and recognizes it as binary for grouped methods.

## Preparation and continuation

`prepare.py` expands bundled historical puzzle banks into per-stage ordered
JSONL, checks gzip/uncompressed SHA256 values, and slices one shared 128,000-row
4x4 bank into the first 750 updates and final weighted 250 updates. Data files
contain no rollout probabilities or source snapshots. GSM8K attribution and
upstream MIT declaration are in `data/SOURCES.md`. Projection/SFT data use the
separate projection manifest owned by the projection implementation.

Fresh `--data-mode regenerate` banks are explicitly method replications.
Held-out target grids are reserved first, rejection has a finite bound, and the
two 4x4 phases slice the same generated bank. They do not claim historical
byte identity. Default historical replay preserves the actual recorded
cross-stage exclusions.

Every cross-stage optimizer continuation is explicit. It restores master
weights/moments/RNG while starting a new ordered dataset, as required for a
curriculum transition. Same-stage `--resume` also restores the exact pending
cohort, dataset index, model/tokenizer bytes, and resolved configuration.
Changing MIS bounds, a tokenizer, the saved policy, dataset, or recipe rejects
continuation. Current and historical optimizer checkpoint layouts are supported;
missing/pruned state never silently becomes a fresh optimizer.

Checkpoint publication is atomic after policy, tokenizer, optimizer, cohort,
and identity files are complete. Saved policy exports remain available; latest
two optimizer/cohort states and explicitly retained intermediate ancestors are
kept, alongside the latest twelve raw cohorts. `status` lists pending stages,
prerequisites, retained policies, and exact-resume points. A policy-only export
is an inference/fresh-branch artifact, not an exact-resume point.

## Evaluation and validation

Preflight is an explicit command before any training. It benchmarks every
distinct configured output budget/context, including the 8,192-token endpoint
test of the 4,096-token 4x4 phase, with intended native concurrency/logprobs.
It records comparison evidence and reserves learner-equivalent GPU space.
Training logs rollout throughput, actual cohort lag, learner time, and full-step
time without speed probes or automatic speed-based stops. This task did not
run GPU training or evaluations, so it makes no new full-model capacity claim.

Periodic evaluation uses the recipe's panel-specific counts, repeats, budgets,
and contexts; it emits pass@1/4/8 as applicable, mean lengths, cap rates, all
complete responses as JSONL, and tier/outcome Markdown trace samples. Separate
`evaluate --checkpoint` requires preflight for that actual checkpoint/panel.

Validation used the mandated `/home/lukas928/SimplePT/.venv/bin/python`:

- All 18 stages prepared successfully through the CLI without model loading;
  all 16 generated RL TOMLs passed configuration/native validation.
- `tests/test_hill_climb_runtime.py`: 11 CPU tests, including a real MIS+SC
  update, bitwise-identical interrupted/resumed second cohort half, no
  recollection, strict precursor grading, recorded dataset SHA, bounded fresh
  generation, shared 4x4 bank, evaluation profiles/pass@8, status, and missing
  optimizer rejection.
- Combined runtime/projection example tests: 24 CPU passes at handoff.
- Ruff formatting/checks pass for the new runtime modules and their tests.

Projection generation/SFT implementation and tests are documented separately in
`docs/porting/futoshiki_example_projection.md`; independent review adds its own
resume-identity and memory-lifecycle checks.
