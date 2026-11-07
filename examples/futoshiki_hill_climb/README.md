# Futoshiki: a complete hill climb

This example records the experiments behind **“Hill climbing with a 1.2B language model”** (research draft, November 5, 2026). It builds a reasoning checkpoint from LFM2.5-1.2B-Instruct, finds a first Futoshiki foothold with verified projection SFT, climbs 4×4 puzzles, then restarts and climbs three 5×5 tiers.

It includes the stages and their data/configuration lineage. Archived measurements are in [historical_results.json](historical_results.json); source artifact hashes, checkpoint identities, seeds, and dataset hashes are in [provenance.json](provenance.json). Those are historical results, not accuracy guarantees for a new run. The recipe's data and configurations are included; supply a compatible local model checkpoint as described below.

## The route

| Stage | Starts from | Work and retained output |
| --- | --- | --- |
| `gsm8k_16` | Released LFM2.5-1.2B-Instruct | FlashREINFORCE, batch 128, learning rate 1e-6. Historical run completed 16 updates; retain update 10. |
| `countdown3_13` | GSM8K update 10 | 13 updates; each number used once, positive integer intermediates, exact division. |
| `countdown4_50` | Three-number update 13 | 50 updates. |
| `countdown4_continue50` | Four-number update 50 | Another 50 FlashREINFORCE updates; this becomes the pre-Futoshiki model. |
| `solver_projection_256` | Post-Countdown checkpoint | Algorithmic solver guidance, one model proposal and ten greedy random-cut refinements. Train 256 verified examples for one SFT epoch: 64 optimizer updates. |
| `futoshiki4_rl750` | Selected projection epoch 1 | FlashREINFORCE, batch 128, 750 updates, balanced four-tier 4×4 data, 4,096-token training cap. |
| `futoshiki4_rl1000` | RL750 with its optimizer | Another 250 updates, 8,192-token cap, harder-tier counts 48/48/16/16 per batch. Total: 1,000 updates and 128,000 distinct prompts. |
| `mixed5_warmup75` | RL1000 | 75 updates on a balanced 5×5 mixture. |
| `mixed5_to250` | Mixed warmup with its optimizer | Continue to cumulative update 250. |
| `specialist_easy250` | Mixed update 250 | Another 250 updates on 11/5: the historical easy teacher at cumulative step 500. |
| `specialist_easy90` | Mixed update 75 | Train 100 updates on 11/5; retain local update 90, cumulative 165. |
| `specialist_tier2` | Retained cumulative-165 easy checkpoint | Train 100 updates on 9/5; retain local update 70, cumulative 235: the second teacher. |
| `projection_reset` | **RL1000 before its 5×5 warmup** | Frozen student projects 1,024 verified specialist outputs per tier. Response-only SFT, batch 32, learning rate 1e-5, two epochs: 128 updates. |
| `tier2_rl100` | Restart SFT epoch 2 | 100 updates on 5×5, 9 givens / 5 inequalities. |
| `tier3_rl100` | Previous tier with its optimizer | 100 updates on 5×5, 8 givens / 7 inequalities. |
| `tier4_rl100` | Previous tier with its optimizer | 100 updates on 5×5, 6 givens / 8 inequalities. Retain the final checkpoint and track the validation-selected update-280 peak separately. |

The final three stages carry one student forward. The two specialists are supervision sources within the same model lineage; no stronger outside model is needed. The reset student goes back to RL1000 rather than inheriting the specialists' or the regressing mixed run's weights.

The historical specialist choices were the easy specialist's step 500 and the next-tier curriculum specialist's step 235. These are different from the later optimized easy-expert experiments. Their identities and recorded weight hashes are preserved in `provenance.json`.

## What is reproduced

The bundled supervised examples preserve the selected original puzzle IDs and projected responses. They allow replay of the recorded SFT data. Procedural projection can regenerate responses on those puzzles using your own checkpoints. It keeps the historical verifier and greedy likelihood acceptance rule; stochastic generations need not match the archive byte for byte.

RL banks/configurations distinguish verified historical data from deterministic fresh generation. Inspect the generated data manifest before comparing a run with an archived score. A seed alone does not establish equality of datasets or generations: model weights, prompt template, native precision, batching, request limits, and token windows also matter.

The projection procedure accepts a correct, EOS-terminated refinement only when its mean log probability under the frozen student, scored on the ordinary task prompt, improves. The two-expert reset limits refinements to the retained suffix length after the cut. The earlier solver projection uses `min(2560, 3200 - prompt_length) - cut`, so its revised trajectory can grow within that budget. This is greedy likelihood ascent, not exact Metropolis–Hastings. Final-answer verification does not certify every intermediate deduction.

The two-expert restart also records a historical initialization adaptation: after repeated failed guided initialization, 21 of the full 2,048 chains (including seven in the first 512-chain block) were initialized from verified specialist responses and then refined by the frozen student. The recorded corpus identifies these cases. They must not be described as wholly student-generated initializations.

## Prepare a run

Run commands from the repository root after the [CUDA installation steps](../../README.md#start-locally). Supply a local HF-format checkpoint with its tokenizer; weights are not downloaded or bundled.

```bash
export RUN_ROOT="$PWD/runs/futoshiki-hill-climb"
export INITIAL_MODEL=/absolute/path/to/LFM2.5-1.2B-Instruct
uv run wandb login
uv run python examples/futoshiki_hill_climb/prepare.py \
  --root "$RUN_ROOT" --initial-model "$INITIAL_MODEL"
uv run python examples/futoshiki_hill_climb/rl.py status --root "$RUN_ROOT"
```

Preparation materializes stage data and configurations beneath `stages/STAGE/`. It checks bundled data and writes the run manifest. It performs no training or inference. Inspect the stages and input checkpoint lineage before launching. Historical source-data import is a separate preparation option; the ordinary workflow uses the standalone example's bundled artifacts and generators.

The arithmetic IDs are `gsm8k_16`, `countdown3_13`, `countdown4_50`, and `countdown4_continue50`, in that order. Follow them with the first projection/SFT stage, then `futoshiki4_rl750` and `futoshiki4_rl1000`. For the specialist branch run `mixed5_warmup75`, `mixed5_to250`, and `specialist_easy250`; branch again from the warmup using `specialist_easy90` and `specialist_tier2`. After `projection_reset`, finish with `tier2_rl100`, `tier3_rl100`, and `tier4_rl100`.

To enter at an existing checkpoint, preparation accepts an explicit stage import, for example `--artifact countdown4_continue50=/absolute/path/to/post-countdown/policy`. The manifest records the import rather than silently treating those weights as the released Instruct model. Imported weights must include compatible configuration/tokenizer files.

For an RL stage, benchmark the actual stage checkpoint, inspect the report, then train:

```bash
uv run python examples/futoshiki_hill_climb/rl.py preflight \
  --root "$RUN_ROOT" --stage futoshiki4_rl750
uv run python examples/futoshiki_hill_climb/rl.py train \
  --root "$RUN_ROOT" --stage futoshiki4_rl750
```

Run each dependent stage only after its parent checkpoint is complete. An interrupted stage resumes its own optimizer, RNG, and dataset cursor; moving to another tier preserves the selected optimizer state while initializing the new tier's dataset state. Use the actual completed checkpoint directory:

```bash
uv run python examples/futoshiki_hill_climb/rl.py train \
  --root "$RUN_ROOT" --stage tier3_rl100 \
  --resume "$RUN_ROOT/stages/tier3_rl100/checkpoints/step_19"
uv run python examples/futoshiki_hill_climb/rl.py evaluate \
  --root "$RUN_ROOT" --stage tier4_rl100 \
  --checkpoint "$RUN_ROOT/stages/tier4_rl100/checkpoints/step_99" --panel test
```

Example RL checkpoint names use zero-based completed update indices: `step_99` means 100 local updates. Historical curriculum archives used `step_100`; `historical_results.json` records those historical names independently. SFT checkpoints use `epoch_1/policy` and `epoch_2/policy`. Use the run manifest's resolved outputs when crossing a stage boundary.

## Run the projection and SFT stages

For `solver_projection_256` or `projection_reset`, first expand and verify the recorded data. Set `STUDENT_MODEL` to the stage's student: post-Countdown for the first foothold, pre-5×5 RL1000 for the reset.

```bash
export STAGE=solver_projection_256
export STUDENT_MODEL=/absolute/path/to/the/stage/student/policy
uv run python examples/futoshiki_hill_climb/projection.py prepare \
  --root "$RUN_ROOT" --stage "$STAGE"
uv run python examples/futoshiki_hill_climb/projection.py preflight \
  --root "$RUN_ROOT" --stage "$STAGE" --student "$STUDENT_MODEL" \
  --purpose training
```

Inspect the saved preflight, then replay the recorded response-only SFT data with native held-out evaluation and complete trace reports at the retained epoch:

```bash
uv run python examples/futoshiki_hill_climb/projection.py train \
  --root "$RUN_ROOT" --stage "$STAGE" --student "$STUDENT_MODEL" \
  --data recorded \
  --preflight "$RUN_ROOT/stages/$STAGE/preflight/training.json"
```

To regenerate projections, use `preflight --purpose projection` before `project`. Choose `--guides recorded` for verified recorded guidance, or recollect specialist guidance from your same-lineage checkpoints first. Regenerated output remains separate from the recorded SFT corpus:

```bash
uv run python examples/futoshiki_hill_climb/projection.py preflight \
  --root "$RUN_ROOT" --stage "$STAGE" --student "$STUDENT_MODEL" \
  --purpose projection
uv run python examples/futoshiki_hill_climb/projection.py project \
  --root "$RUN_ROOT" --stage "$STAGE" --student "$STUDENT_MODEL" \
  --guides recorded
```

Repeat with `STAGE=projection_reset` after its student and specialist dependencies are ready. Training with `--data recollected` explicitly selects regenerated projections. Replaying recorded data reproduces the supervised examples; it does not promise the same model hash or accuracy.

## Reading the measurements

The 4×4 climb ends at 77.44% pass@1 and 92.97% pass@8 on 128 puzzles, eight attempts each, at 8,192 tokens. The matched RL750 endpoint is 57.13% pass@1 under that **same 8K evaluation**. The RL750 training cap and its fixed 32-puzzle monitor were 4K; that monitor ends at 52.34%. Keep its curve separate from the 8K test comparison.

The restart's BF16 development scores are 36.04% and 11.33% pass@1. The later curriculum's native FP8 starting measurements are 35.55% and 12.30%. They are separate evaluation records for the same retained SFT checkpoint, not two successive training updates.

| Native FP8 validation panel | Before curriculum | After 9/5 | After 8/7 | Selected peak: update 280 | Final: update 300 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 11 givens / 5 inequalities | 35.55% | 58.50% | 70.51% | 70.02% | 67.58% |
| 9 givens / 5 inequalities | 12.30% | 29.39% | 41.89% | 40.14% | 37.40% |
| 8 givens / 7 inequalities | — | — | 29.69% | 32.13% | 31.25% |
| 6 givens / 8 inequalities | — | — | 9.28% | 11.72% | 12.11% |
| Four-tier pooled | — | — | 37.84% | 38.50% | 37.08% |

Each tier has 256 validation puzzles and four attempts per puzzle at 8,192 tokens. The missing early cells were not measured; they are not zeros. Update 280 is selected using these monitored panels. The completed budget is update 300. Historical update-280 weights were deleted, although their responses remain; the update-300 weights were retained. The example does not ship either checkpoint.

The final stage improved its hardest target while losing some earlier-tier performance. Its separate 128-puzzle, eight-attempt mixed test fell from 37.70% to 36.82%. There was no final 4×4 reevaluation, so the records do not establish retention of that earlier skill.

## Historical side branches

`mixed5_to350`, the balanced 5×5 continuation through update 350, is an optional historical arm. On its separate matched test, pass@1 fell from 12.89% at update 100 to 8.30% at update 350. It motivated the restart but is not the restart student's direct parent. Replaying update 251 implicated negative credit in that update; this was not a complete explanation or a demonstrated lasting fix.

`specialist_tier2_independent`, the earlier FlashREINFORCE/CISPO experiment, and the later grouped-MIS comparison are comparison arms, not required ancestors. They differ in update counts, grouping, retained responses, or token work. Their archived outcomes do not establish equal-work superiority of a loss function. The solver-grounded 512-example × four-epoch SFT comparison is also distinct from the selected 256-example × one-epoch projected foothold.

## Outputs and inspection

Keep each stage's policy, optimizer/RNG/cursor checkpoint, resolved config, dataset hashes, local metrics, native rollout log probabilities, and timing logs. New training runs log online to W&B project `climbing-gym` under the current W&B account (optional entity override); authenticate before launching. Every evaluation saves a Markdown report containing actual prompts, complete responses, expected assignments, and outcomes.

Run native setup validation before each stage on the actual checkpoint, intended prompt lengths, token budget, and probability metadata. Inspect batch size, all request/concurrency limits, tokens/second, and peak GPU memory, comparing a relevant prior report. A prior stage with a different workload is context rather than an interchangeable speed baseline. Training retains rollout and full-step timings; throughput fluctuations do not trigger automatic probes or stops.

The 5×5 MIS/score-centering stages retain their historical objective reduction and cohort schedule. Check the generated configuration for response-mean normalization, top-32 behavior probabilities, ratio bounds [0.5, 5], and a 256-response collection split into two 128-response updates with policy lag zero/one. Do not substitute a different current trainer reduction while calling it an exact historical recipe.

## Provenance

`provenance.json` contains compact identities and hashes for original manifests, protocols, configurations, and results. Artifact locations are relative to the archived source workspace; execution reads the files prepared under your chosen run root. Original source commits that were not available on GitHub are recorded as commit IDs without invented permalink URLs.

The selected original source report and two-expert experiment record were independently matched to SHA256 hashes in the research evidence manifest. The data manifests retain these identities. Supervision uses algorithmic verification and same-lineage model outputs throughout.

The RL driver keeps all saved policy exports, the latest two optimizer/cohort
states, and explicitly selected intermediate ancestors (including GSM8K update
10 and specialist selections). It retains the latest twelve raw cohort archives.
A policy whose optimizer state was pruned remains usable for inference or an
explicit fresh-optimizer branch; it is **not** an exact-resume point. `rl.py status`
lists the remaining exact-resume checkpoints and checks optimizer availability
for transitions. Missing optimizer state never silently starts fresh moments.

For an independent evaluation of a saved policy, benchmark that checkpoint and
panel first, then run the evaluation with the same arguments:

```bash
uv run python examples/futoshiki_hill_climb/rl.py preflight --root "$RUN_ROOT" \
  --stage tier4_rl100 --checkpoint "$RUN_ROOT/stages/tier4_rl100/checkpoints/step_99" --panel test
uv run python examples/futoshiki_hill_climb/rl.py evaluate --root "$RUN_ROOT" \
  --stage tier4_rl100 --checkpoint "$RUN_ROOT/stages/tier4_rl100/checkpoints/step_99" --panel test
```

Training startup validates every distinct configured evaluation budget and
context, including the 8,192-token endpoint test of the 4,096-token 4x4 training
stage. Periodic evaluations reuse this startup validation; they never launch
throughput probes during training. Every evaluation writes complete response
JSONL records plus a tier/outcome sample in a Markdown trace report.
