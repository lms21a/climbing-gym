# Knights: the recent full-response Value MIS+SC run

This is the method used by `knights_value_mis_sc_xor3_fullresponse100_20261104`, the recent Knights value run sometimes remembered as “BPO” or “BPOC.” Its recorded algorithm was **`full_response_value_mis_sc`**, exposed here as **`value_mis_sc`**. It combines a BPCO-style privileged bounded critic and length-adaptive advantages with the existing token MIS plus sparse score-centering actor objective. It does not introduce a separate BPO algorithm or substitute PPO's clipped actor objective.

The archived run completed **300 actor updates after 22 critic-only warmup batches**. [historical_summary.json](historical_summary.json) records the original readiness measurements; [provenance.json](provenance.json) identifies its source commit, exact data hashes, initial checkpoint weight hash, and historical W&B run. These are archived observations, not results from this example or guarantees for a new model.

## The production method

[production.toml](production.toml) preserves these settings:

| Component | Historical profile |
| --- | --- |
| Collection | 256 different prompts; one fresh full response each; native batch/request/concurrency limits 256; no dynamic filtering or policy lag |
| Response | One 4,096-token window, ordinary user chat with thinking disabled in the template and `<thinking>\n` assistant prefill; stop after `</answer>`; no tools, continuation, compaction, or truncation bootstrap |
| Sampler | Temperature 1, full support, sampled log probabilities and top-32 behavior head; native FP8 MLP/KV/head and adaptive batching; BF16 weight packing **disabled**, matching the source |
| Actor | Token MIS bounds [0.5, 5], sparse score centering, frozen pre-fit token advantages, one global `256 × 4096` divisor, no advantage or response-length normalization; learning rate 1e-6; gradient clip 1 |
| Critic | Independent cloned backbone; embeddings and token mixers frozen; feed-forward layers, FFN norms and FP32 scalar head trainable; arctangent output in [0,1], head initialization std `1/(hidden_size+1)` |
| Targets | Gamma 1, raw Monte Carlo outcomes at every pre-action state; length-adaptive policy GAE alpha 0.4; no stronger teacher; verified reference answer appears only in the critic prompt |
| Critic fitting | Two epochs per collection; learning rate 5e-6; gradient clip 10; exact FP32 Adam masters/moments, local mmap backing, bounded GPU staging; actor optimizer likewise keeps exact FP32 state on the host |
| Monitoring | Online W&B `lukas928/simple-pt`, local metrics and complete rollout text; dev evaluation every 10 actor updates; full actor/critic/dataset/RNG checkpoints every 10 updates |

Capped responses stay in training. The 4,096-token cap is the task horizon: incomplete answers receive the verifier's terminal outcome, with zero terminal bootstrap. Sparse score centering corrects the recorded sampling distribution; it does not insert a second PPO importance ratio.

## Critic readiness is part of the runnable route

`climbing_gym.train` first collects fresh responses while leaving actor weights and actor optimizer untouched. On **each collection, before critic fitting**, the gate checks Monte Carlo explained variance **> 0**, MSE **below the constant-outcome baseline**, and absolute calibration bias **< 0.08**. The critic is then fitted for two epochs. Three consecutive good pre-fit collections and at least ten completed warmup batches are required. At thirty unsuccessful batches the run saves its failed phase and raises; actor training never starts.

The gate uses training rollouts, not the dev/test panels or post-fit predictions. Warmup gradients flow only through the independent critic. During actor training, advantages are frozen before either model is fitted on that collection. Actor and critic accelerator ownership alternate, preserving the memory behavior of the source method.

Readiness counters, objective/sampler settings, both optimizers, RNG and dataset state are saved at every completed warmup boundary. Resuming from the middle of warmup continues the streak and next data/seed position. Resuming a ready boundary begins actor update zero without repeating warmup. The phase and method settings must match; missing readiness state and failed gates fail closed. The older PPO `critic_warmup_steps` setting is rejected here; use the explicit `value_warmup_*` fields.

## Prepare the standalone data and profile

Run from the repository root after installing the package and CUDA dependencies. Supply a compatible local HF-format LFM2.5 checkpoint with its tokenizer; model weights are not bundled.

```bash
export RUN_ROOT="$PWD/runs/knights-value-mis-sc"
export INITIAL_MODEL=/absolute/path/to/your/starting/policy
uv run wandb login
uv run python examples/knights_value_mis_sc/prepare.py \
  --root "$RUN_ROOT" --model "$INITIAL_MODEL"
```

Preparation performs exact solver/public-prompt checks and expands the bundled compressed banks. It performs no model inference, evaluation, or training. It writes `manifest.json`, `train.toml`, `eval_dev.toml`, and `eval_test.toml` beneath the chosen run directory and refuses to overwrite an existing run.

The production training bank has **512 unique XOR/IFF three-person puzzles**. The archived dev bank contains 64 puzzles per cell; preparation selects the **same first 16 per cell** used by the historical evaluator. The test panel uses **64 per cell**. Both span six familiar/XOR-IFF × two/three/four-person cells and request **four independent samples per puzzle**. Training, dev and test IDs and structural groups are disjoint. Each bundled public prompt is checked against the current renderer, and each row preserves a hash of its original prompt token IDs. Before benchmarking, the runner verifies CPU tokenizer/template parity against those hashes.

The historical starting weights were paragraph-no-tools XOR3 **step 40**, with SHA256 recorded in `provenance.json`. Preparation records whether a supplied `pytorch_model.bin` matches. Other compatible starting weights run the same method but do not reproduce the historical initialization. Exact archived puzzle identities and numerical objectives are preserved; the generic native adapter's request scheduling and random-seed stream differ from the old bespoke script. New generations and checkpoints are therefore not claimed to be bitwise historical reproductions. Within the new implementation, exact checkpoint resumption preserves the next sampler counter, dataset state, RNG, optimizer state and readiness phase.

## Benchmark, inspect, then train

Benchmark the actual checkpoint using representative training, dev and test prompts, the intended token budget and stored log probabilities/top-32 heads:

```bash
uv run python examples/knights_value_mis_sc/run.py preflight --root "$RUN_ROOT"
# When a relevant prior report exists, add --prior /absolute/path/to/prior-native-report.json.
uv run python examples/knights_value_mis_sc/run.py train --root "$RUN_ROOT"
```

The first command explicitly launches the native speed/memory benchmark. Its JSON records throughput, peak memory, batch/concurrency limits and comparison to the supplied prior report; inspect those reports before training. Use `--panel train`, `--panel dev`, or `--panel test` to benchmark only one panel. Training requires matching train/dev preflight selections. Setup probes are not rerun during training and speed fluctuations do not automatically stop a run. Rollout throughput and full-step timing remain logged throughout warmup and actor training.

Dev evaluations run every ten actor updates. Complete prompts, model responses, expected answers, outcomes, and available difficulty/success/failure strata are written to `eval/dev/traces.md` and collision-numbered successors. Full rollout JSON and metrics remain local as well as online training metrics.

## Resume and retained checkpoints

`checkpoints/latest_value_warmup.json` points to the latest complete warmup boundary. Warmup paths are `checkpoints/critic_warmup/step_N`; `N` is the zero-based **critic collection index**, while their metadata keeps actor step `-1`. Only the latest two complete warmup states are retained, including a ready boundary when reached.

The example runner publishes `latest_actor_checkpoint.json` in the run root and retains the latest **two** full actor checkpoints. A path such as `checkpoints/step_99` means 100 actor updates and contains the policy/tokenizer, critic, both exact optimizers, RNG, dataset cursor, and readiness state. Resume only from a complete checkpoint directory:

```bash
export CHECKPOINT=/absolute/path/to/the/complete/checkpoint-directory
uv run python examples/knights_value_mis_sc/run.py preflight \
  --root "$RUN_ROOT" --checkpoint "$CHECKPOINT"
uv run python examples/knights_value_mis_sc/run.py train \
  --root "$RUN_ROOT" --resume "$CHECKPOINT"
```

The runner computes the remaining actor updates toward the profile's configured total. It never turns 300 requested total updates into 300 additional updates on resume. New checkpoint-specific preflight is required before loading a resumed learner. A run stopped inside an incomplete collection resumes its last complete boundary; incomplete/unpublished state is not treated as completed work.

Every 20 actor updates and at the final update, `policy_exports/update_NNN` retains an ordinary HF policy/tokenizer for inspection or evaluation. These exports **do not contain exact training continuation state**. Checkpoint pruning occurs only after a complete new state and pointer are published; it does not delete metrics, complete rollout text, or trace reports. The retention hook belongs to this example and leaves unrelated generic trainers unchanged. Full FP32 optimizer checkpoints are still large; budget disk for two actor states, two warmup states, the mmap cache, and selected HF policy exports.

For held-out evaluation, benchmark and evaluate the actual selected policy:

```bash
export SELECTED_POLICY="$RUN_ROOT/policy_exports/update_300"
uv run python examples/knights_value_mis_sc/run.py preflight \
  --root "$RUN_ROOT" --checkpoint "$SELECTED_POLICY" --panel test
uv run python examples/knights_value_mis_sc/run.py evaluate \
  --root "$RUN_ROOT" --checkpoint "$SELECTED_POLICY" --panel test
```

The evaluator saves its summary, all rollout records, and complete trace report under `eval/CHECKPOINT_KEY/test/`. Held-out test results do not select training settings.

## A separate smoke profile

```bash
uv run python examples/knights_value_mis_sc/prepare.py \
  --root "$PWD/runs/knights-value-smoke" --model "$INITIAL_MODEL" --profile smoke
```

[smoke.toml](smoke.toml) uses batch/concurrency 2, a 256-token horizon, eight training puzzles, one puzzle per held-out cell, one sample per puzzle, and one actor update. Its readiness gate requires one successful collection within three warmup batches. It exercises the same real method with a smaller workload; it is not the historical experiment. A poorly calibrated critic can fail this small gate, in which case the actor intentionally remains unchanged. Use the same explicit preflight/train commands with the smoke root.

The port's CPU toy-model tests verify untouched actor weights/optimizer during warmup, pre-fit gate/streak/failure behavior, tokenizer export, bounded retention, and bitwise continuation from both an intermediate and a ready warmup checkpoint. No new model run or accuracy measurement is claimed by this example.
