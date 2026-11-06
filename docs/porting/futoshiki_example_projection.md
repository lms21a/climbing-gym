# Historical Futoshiki projection and SFT example

The executable example is in `examples/futoshiki_hill_climb/`. It adds no model weights and has no runtime dependency on a SimplePT checkout. Preparation and tests performed for this port used CPU only. No model inference, evaluation or full training was launched.

## Sources and fidelity

The first foothold uses the recorded `futoshiki_epochs_solver_projection_20261010` experiment: its frozen `prepare_solver.py`, `projection.py`, `run.py`, source solver, selected original puzzle order, and `data/projected_256.jsonl`. The frozen original actor is the post-Countdown model, SHA256 `5cd9a1450cb1cf5a304760d03964f3bcb1429fd0e3755a550d1970c4e29c8439`. The original projection operated on 1,024 nested source examples; the retained SFT stage takes the first 256, balanced 64 per 4x4 tier. It does not substitute the separate 512-example comparison branch.

The later reset uses `futoshiki5_two_expert_projection_sft_prewarmup_20261015` and its frozen `source_snapshot/scripts/futoshiki_two_expert_projection.py`, `project_rl500_futoshiki_traces.py`, and `src/simple_pt_2/easyppo_critic.py`. Its student is the pre-5x5 RL1000 checkpoint, SHA256 `c27249979db28a1725f1b8df01e96a851774a8298b0dfcb83b9330a74109f8c8`. The easy and next-tier specialist hashes are respectively `f47ea3c1e6d373eb46a0cc0b0e829c110546384a436fddb8b6ac8a1ddab0d273` and `cfc611e2b1c4b2a6ea1e2587393a24b0142e66ae07b36c24f586ebd3ee015936`. These are the same LFM model lineage, not stronger outside teachers. The historically retained second-epoch SFT checkpoint hash is `868872d8...`; a new run is not promised identical weights or metrics.

`historical.py` preserves the specialized first-person proposal text, ordinary task scoring prompt, two original format/meta-commentary audits, zero-inclusive deterministic cuts, original learning-rate arithmetic, balanced tier shuffling, and original `FP32MasterAdamW` implementation. `native_projection.py` provides actual native continuous-admission proposal/scoring flow and atomic operation archives for deterministic restart. It does not call the generic block-growing package sampler or the immutable-answer adapter, because both differ from these historical experiments.

| Rule | Solver foothold | Two-specialist reset |
| --- | --- | --- |
| Guidance | Algorithmic detailed solver search on 1,024 selected puzzles | 1,024 verified outputs from each same-lineage specialist |
| Initialization | Native guided complete response | Native guided complete response |
| Reasoning/answer envelope | Generated thinking/answer/EOS, independently final-answer verified | Same; answer is generated, not automatically fixed |
| Native projection capacity | 128 live slots, context 8,192 | 64 live slots, context 18,432; queue blocks of 512 |
| Proposal | temperature .6, repetition penalty 1.1, top-p 1 | Same |
| Revision | 10 random cuts, suffix budget `min(2560,3200-prompt_length)-cut` | 10 random cuts, suffix budget `len(retained_completion)-cut`, 8,192 on empty initialization |
| Invalid initialization recovery | Up to 64 retries, no reference-copy fallback | Up to 20 retries after initial+10 refinement attempts, then explicit specialist seed and 10 additional refinements |
| Acceptance | Verified EOS-complete candidate and strictly greater mean raw ordinary-prompt base log probability | Same |
| SFT | FP32 parameters/AdamW; BF16 autocast forward; batch 4, head chunk 128, pack 3,200 | BF16 parameters; FP32 masters **and both moments**; batch 32, head chunk 512, pack 9,216 |
| LR and retained point | lr1e-5, cosine across two epochs, retain first epoch/64 updates | lr1e-5, four-epoch schedule with eight warmup steps and .2 floor, retain second epoch/128 updates |

Both use weight decay .01, gradient clip 1, exact prompt/completion IDs, response-token loss including EOS, and normalization across the whole packed update. Calling the revisions “MCMC” does not make them Metropolis–Hastings: the original rule is verifier-constrained greedy likelihood ascent.

The solver preflight uses the original evaluation seed 20261001 and seed offset 600000 for the cached first 128 initial proposals. The remaining initial proposals use seed 20261010/offset 600128; the subsequent refinements preserve the historical offsets. Reset uses native seed 20261012, deterministic per-block offsets, and source string-seeded cuts 2026101503. Native batching/precision changes can still alter stochastic samples.

## Portable data

`data/projection_manifest.json` identifies each source run/file by logical name and original file SHA256, and records separate curated/uncompressed/gzip SHA256, ordered-ID SHA256, row counts, compressed lengths and tier counts. `build_curated.py` is the maintainer exporter; consumers need only the committed compressed files. Score vectors, machine paths and weights are excluded. Tokenizer identity and file hashes are recorded; the runnable path verifies `tokenizer.json` and checks that historical prompts and complete responses match their exact IDs.

The bundled training records preserve all 256/2,048 original puzzle rows, ordinary rendered prompts, full responses, exact IDs, lengths, expected answer, and relevant specialist/projector/seed provenance. Solver guides preserve the original 1,024-puzzle order. Specialist guides, original collection pools and all original calibration/development/test splits are also included. Total projection assets are about 10 MiB compressed.

The audited 256-example foothold contains 195 solution grids. The 2,048-example reset contains 2,037 solution grids and **21 specialist-seeded chains**; the often quoted 7/512 refers only to its first block. Duplicate solution grids within training are preserved, rather than silently changing the archive. Held-out puzzle IDs and solution grids remain disjoint from training. `prepare` audits these properties, verifies the independent final-answer checker and checks all hashes before extraction. The `export` command writes readable archived full model traces; `recorded_projection_samples.md` includes examples from both tiers and specialist-seeded cases.

## Commands

All stage files live under `ROOT/stages/STAGE/`. `prepare --root ROOT --stage STAGE` is CPU only and extracts immutable `data/train_recorded.jsonl` plus guide/pool/heldout files. It refuses to overwrite changed files. `audit` repeats validation; `export --output FILE.md` writes complete original traces. `solver-guides` uses only a supplied compatible tokenizer and the algorithmic solver to reconstruct all 1,024 guidance traces exactly, verifying them against the archive.

`preflight --purpose training|projection --student MODEL` runs representative native requests with intended budgets and log probabilities, records actual concurrency and peak memory, rescoring time and full-step timing where applicable, compares to the bundled measured RTX4090 baseline (or `--prior-benchmark`), and saves actual prompt/response/expected/outcome Markdown traces. Projection and training preflights are separate: reset projection's long prompts use64 live slots, whereas ordinary short-prompt evaluation and specialist collection use 128. An explicit `--batch-size` permits new capacity comparisons; do not carry the projection's 64-slot cap into ordinary collection.

`train --data recorded` replays the exact original supervised records. `project --guides recorded` runs new stochastic projections conditioned on the original verified guides. `collect --stage projection_reset --tier TIER --student SPECIALIST --preflight REPORT` generates new single-sample specialist guides in original 512-puzzle chunks and chooses the first 1,024 valid unique puzzles per tier. Each specialist needs its own weight-matched ordinary preflight. When both are complete, they interleave into `data/guides_recollected.jsonl`. `preflight --guides recollected` then validates that new guide workload; `project --guides recollected` produces new projections; `train --data recollected` explicitly selects them. Pool exhaustion raises a clear error rather than introducing duplicate/easier examples.

Both collection and projection archives bind weight/tokenizer hashes, ordered input data and protocol before any reuse. An updated preflight never authorizes reuse of archives from another policy. Solver setup proposals are reused only for the original matching guide order. Local generated token scores and full candidate archives are retained for inspection, but are not bundled in the repository. Specialist-seed proposal log probabilities missing from recorded guides are marked unavailable; acceptance always uses separately computed frozen-student raw scores, so missing provenance does not change the objective.

Training writes online W&B to `lukas928/simple-pt`, durable local config/metrics, rolling exact optimizer/RNG/data-cursor resume state, immutable `checkpoints/epoch_1/policy` or `epoch_2/policy`, and native heldout reports after each epoch. By default the source solver test uses 128 puzzles with 8 attempts, and reset development uses 512 puzzles with 4 attempts. Reset final-test data is never used to select/retrain the checkpoint. An optional standalone `--eval-config` overrides the evaluation setup; its own matched native preflight is validated by the package's eval loader.

The learner and optimizer are moved to CPU before native evaluation; the native engine is closed before learning resumes. This preserves the original 24 GiB lifecycle rather than keeping a second actor resident beside full FP32 Adam state. Throughput, peak memory and full-step timing are logged without training-time speed probes or automatic speed stops.

Pending epoch evaluation on resume uses that same offload order and restores learner residency before the next update. Immutable epoch policies publish through a temporary sibling and atomic rename; a missing snapshot after an interrupted copy is reconstructed from the complete rolling checkpoint before evaluation. Rolling checkpoints bind every saved model/config/tokenizer file, and preflight/projection/collection identities bind the frozen model configuration separately from the original weight SHA256. Changed assets or numerical recipes require a separate run rather than relabeling an exact continuation.

## Validation

CPU tests cover corpus hashes/counts/order, verifier correctness, masks/EOS, grid leakage, complete trace export, exact algorithmic guide reconstruction, distinct historical suffix budgets and LR schedules, source balanced ordering, native-interface projection/recovery using explicit test doubles, changed-policy archive rejection, explicit specialist fallback versus solver no-copy behavior, BF16 optimizer FP32-state restoration, and bitwise tiny-model SFT parity across interrupted epoch evaluation/resume. The tiny models are unit-test fixtures, not model evaluations. No new accuracy or throughput numbers were measured.
