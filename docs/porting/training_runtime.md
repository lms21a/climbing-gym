# Training runtime port

Source of truth: `/home/lukas928/SimplePT/simple-pt-memory-defaults`, commit `6edae8f` (`src/simple_pt_2/config`, `trainers/{common,packing,checkpoints}.py`, `grpo.py`, `training_memory.py`, `optimizer_resume.py`, `dataset_checkpoint.py`, and `countdown_dynamic.py`). Algorithm implementations are documented separately; their value/MIS lineage includes `simple-pt-knights-value-mis-sc` at `07b6762`. Source trees were read only. Imports now use `climbing_gym`; runtime paths never resolve against the installed package or original checkout.

## Public execution contract

`python -m climbing_gym.train CONFIG.toml` dispatches complete training loops:

| trainer.algorithm | trainer.advantage_estimator | Other required settings |
| --- | --- | --- |
| `ppo` | `gae` | Original actor/critic PPO trainer; historical `grpo` estimator spelling remains an alias for GAE |
| `grpo` | `grpo` or `sa_mrpo` | Original packed GRPO trainer |
| `cispo` | `grpo` | `loss_aggregation='seq_mean_token_sum'`, one policy epoch, no entropy/format objective |
| `flashreinforce` | `batch_centered` | `sample_mean`, one policy epoch, direct, one rollout per distinct prompt |
| `mis`, `mis_sc` | `batch_centered` | Flash collection contract, exact MIS correction; SC requires `behavior_topk` |
| `grouped_mis`, `grouped_mis_sc` | `grpo` | `sample_mean`, mixed binary groups, dynamic sampling, at least two rollouts per prompt, complete group request/concurrency limits |
| `mis_ppo` | `value` with `mis_use_value=true`; `grpo` otherwise | Eager whole-collection dual gates; one accumulated actor/critic step; terminal returns |
| `value_mis_sc` | `length_adaptive_gae` | Direct, one rollout per distinct prompt, SC metadata, fixed configured token denominator; frozen critic advantages precede both fits |

The unified loop invokes real algorithm updaters, not just exported losses. PPO and CISPO/GRPO retain their original complete loops and reductions. MIS aliases resolve the objective explicitly into persisted configuration. Unsupported estimator, sampling, reward, and collection combinations fail before allocating models. Raw native behavior probabilities require temperature 1, full support, n=1, and enabled sampled log probabilities.

The supported environment registry is GSM8K (`gsm8k`/`math_qa`), Countdown, Knights and Knaves, and Futoshiki. Harnesses are direct, infinity-think, and numbered-notebook where the algorithm permits them. Unsupported historical environment/harness imports were removed. Dataset loading supports local JSON/JSONL/CSV and optional Hugging Face datasets, including dataset directories and Parquet.

## Memory and continuation

Native collection offloads training models and optimizer state before syncing and collecting. Completed native buffers are released before reloading learner state by default. Optimizer-specific CPU residency hooks are preserved, including optional mapped-state page eviction. `release_inference_buffers_before_update=false` is an explicit opt-out; bounded optimizer workspace still applies.

`cpu_master_adamw` preserves FP32 master/moment arithmetic with host-resident states and a reusable bounded GPU workspace. Its default maximum is 48 MiB (three FP32 buffers); an older whole-parameter chunk setting cannot bypass the workspace bound. Both PPO optimizer builders support it. `master_adamw` and ordinary `adamw` remain available. Value-MIS also supports `mapped_master_adamw` for the critic, with disposable backing at `trainer.value_optimizer_backing_path` (resolved relative to the config). The durable checkpoint remains a regular torch file.

Full-model PPO learns with both actor and critic resident; CPU optimizer state reduces GPU usage but does not prove that every configured packing budget fits a particular GPU. No real full-model PPO capacity claim follows from CPU tests. Value-MIS alternates critic and actor residency during its frozen-target, critic-fit, and actor-update phases, retaining the critic's FP32 value head.

Checkpoints retain the original policy HF export, optional `value_model.pt`, and `metadata.json`. Optional `training_state.pt` adds exact optimizer tensor dtypes and Python/Torch/CUDA RNG states. Restoring bypasses PyTorch's dtype-coercing optimizer loader, preserving integer compressed-master bits and FP32 masters. Mapped optimizers restore into new disposable backing. Missing historical optimizer state requires explicit `resume.allow_weights_only=true`; that option permits a fresh optimizer rather than forcing one when exact state exists.

`dataset_state.json` preserves the fingerprinted cursor, epoch, shuffled order settings, queued dataset indices, next group ID, dynamic update index, and, when present, the drained native generation seed counter. Old checkpoints without the counter remain readable but cannot reproduce its exact continuation. Matching native batching is still necessary for identical stochastic trajectories.

Generic partial groups are never silently represented by a cursor alone: saving emits a warning and skips dataset state if unfinished groups remain. Direct dynamic multi-rollout training uses the source full-wave sampler (`NativeDynamicRunner`): each wave finishes complete groups, chooses variable-reward groups in issue order, and discards unused completed groups rather than carrying stale partial generations into the next update. Group/sampling audits are retained.

## Run records and preflight

All TOML file paths are expanded and resolved relative to the config file's directory (standalone helper calls default to cwd). W&B defaults are enabled, online, entity `lukas928`, project `simple-pt`. Full resolved config, validated training/evaluation benchmark metadata, local metrics, training trajectories, and checkpoints remain available locally. The logger/evaluation agent owns those facilities.

Before model loading, training validates a saved representative native benchmark using the intended output budget, sampled log probabilities, effective native batch size, request caps, throughput, peak memory, and any prior comparison. `[inference] preflight_report_path`/`preflight_prior_report_path` configure training evidence. Periodic evaluation is validated once at setup against its actual coordinator limits; `[eval]` can override those report paths when its budget differs. No live speed probes or automatic throughput stops run during training. Each step records rollout tokens/second and full-step timing. Evaluation trace paths/max samples are available through `[eval.logging] trace_path` and `max_traces`; evaluator writes complete actual responses separately.

## Verification

All code/tests used `/home/lukas928/SimplePT/.venv/bin/python`; no GPU training or model evaluation was launched by this task.

- `tests/test_training_runtime.py`: 15 CPU integration tests, including one real optimizer update through all ten algorithm routes (full original PPO/CISPO/GRPO loops, unified Flash/MIS/grouped/MIS-PPO, value critic freeze/fit/actor cycle), native-only dispatch, offload/reload ordering, exact master/moment/RNG continuation, bounded PPO optimizer builders, and portable config/W&B defaults.
- `tests/test_dataset_checkpoint.py`: 10 cases covering pending dynamic examples, shuffle across epochs, changed-bank rejection, partial-state refusal, native seed continuation, hard request budgets, and whole-group concurrency constraints.
- Source optimizer/memory tests preserve compressed integer master bits, bitwise resumed master updates, buffer release before learning, bounded workspace, and native release/session locking.
- Final combined runtime/shared/algorithm/preflight/logger suite: **185 passed** in 5.06s. This includes the focused runtime/preflight cases (63 passed) and Flash/MIS/grouped/normalization contracts (99 passed).

These are implementation/CPU contract checks, not environment accuracy evaluations. Real native inference/evaluation and full-package compatibility are assigned to the release verification agent, which records its own artifacts and traces.
