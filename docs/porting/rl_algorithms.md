# RL algorithm port

Source trees were read only. Destination imports use `climbing_gym`; no source checkout paths are embedded in the implementation.

## Selected implementations

| Destination | Source worktree / path | Source commit |
| --- | --- | --- |
| `cispo.py`, `ppo.py` | `simple-pt-knights-value-mis-sc/src/simple_pt_2/` | `07b67622aeab814512a5c5c551d663de65863b21` |
| `value_mis_sc.py`, `grouped_mis.py`, `grouped_sampling.py` | same | same |
| `flashreinforce.py`, `offpolicy.py`, `cpu_master_adamw.py` | `simple-pt-memory-defaults/src/simple_pt_2/` | `6edae8f74da3559a86ed5a6ac7c1a7782fb0ea91` |
| `mis_ppo.py` numerical functions | `simple-pt-knights-value-mis-sc/src/simple_pt/multiturn_mis_ppo.py` and `multiturn_grpo_icepop.py::compute_grpo_advantages` | `07b67622aeab814512a5c5c551d663de65863b21` |

Comparison candidates included `simple-pt-futoshiki-drgrpo-mis-sc` (`11520e4bf2259f5ba9a575dd8c600a15248a4a75`), `simple-pt-faster-offpolicy` (`7f8e4dc8b66ff5854da2775debcbaa7d3985e608`), and `simple-pt-flashreinforce` (`76bd5b8f9f41a5e0d91e961c53a876563ab3c0b6`). The selected later variants retain sparse fused heads, explicit generated-token spans, fixed MIS normalization, and bounded optimizer workspaces added after those candidates. The value worktree contributes its additional critic implementation absent from memory-defaults.

## Preserved numerical behavior

- CISPO clips **detached importance weights**, retaining signed actor gradients beyond a clip boundary. It averages trajectory token sums; memory pack scaling must use the logical collection's sequence count.
- FlashREINFORCE uses one batch outcome mean, no reward standard deviation, the sampled-action Bernoulli `KL(behavior || learner)` in FP64, and a complete-response gate. Its importance ratios remain unbounded on admitted samples. Equation 11 divides by original batch size and each response length; rejecting responses does not change the baseline or denominator.
- Token MIS masks ratios outside configured bounds before exponentiation. Sparse score centering uses exact sampler head probabilities and the learner-proportional omitted tail; importance weights and centering coefficients are detached. This is token correction, not exact trajectory density correction.
- Ungrouped MIS defaults to a **fixed token normalizer**; historical response means require `legacy_response_mean`. Grouped MIS uses group-centered raw rewards without standard deviation scaling and one global `B * max_tokens` denominator, retaining only complete mixed binary-outcome groups. Surplus queue entries remain immutable disk references with an explicit policy-lag bound.
- PPO retains clipped policy surrogate, token masks and pack weighting, full critic clone and independent optimizer, terminal outcome GAE across turn boundaries, checkpoint handling, warmup, periodic evaluation, and the complete training loop. `ppo.load_run_from_toml` accepts PPO only; legacy JustRL2 entrypoints and generic inference fallback were removed. Bounded critic/length-adaptive-target math remains available because value MIS relies on it; internal BPCO numerical helpers are retained but no BPCO CLI is exposed.
- Value MIS freezes actor advantages **before** fitting either model, uses pre-action critic positions, raw Monte Carlo outcome targets, full-response token advantages and the existing single MIS+SC score. Critic prompt overrides contain the reference answer and never the current response/future/outcome. The clone trains feed-forward/norm parameters and FP32 bounded value projection; actor and critic parameters remain independent. Mmap critic Adam state is disposable working storage; durable checkpoint states remain exact FP32 tensors.
- MIS-PPO preserves both complete original policy-only and actor-critic losses: separate token and geometric-mean trajectory gates, optional explicit fallback, active-token actor means, all-valid-token critic MSE/KL/entropy, and detached actor value baselines. `mis_ppo.update_policy` replaces its former vLLM trainer with actual native packed collection updates. The global trajectory gate spans all turns and packs; each packed loss is weighted by its fraction of original valid tokens, preserving the original trainer convention. Consequently its active-token reduction can depend on packing when token retention differs across packs; it is intentionally not silently changed to the other algorithms' fixed denominator.

## Runtime API / dependencies

Shared runtime owns `config`, `trainers/{common,packing,checkpoints}`, `training_memory`, `grpo`, `eval`, and the unified driver. In particular:

- `cispo.main(argv)` delegates to the real GRPO collection/update/checkpoint runtime, which dispatches `packed_cispo_policy_loss_from_model`.
- `flashreinforce.validate_config(config)` and `update_policy(run, trajectories)` return metrics and audit records. `flashreinforce_objective` selects `sequence_kl`, `mis`, or `mis_sc`.
- `offpolicy.update_grouped_corrected_policy(run, trajectories)` uses the same run contract for Dr. GRPO.
- `ppo.load_run_from_toml`, `ppo.main` provide the full separate actor/critic PPO trainer. Evaluation imports require `EvalConfig`, `EvalLoggingConfig`, `ResolvedEvalRun`, `_trajectory_record`, `build_eval_run`, `run_eval`.
- Value lifecycle: `build_critic(policy)`; `freeze_targets(run, data, alpha=.4, policy_lambda=None)` returns packs, advantages, targets, preupdate values, and diagnostics; `fit_critic(run, data, packs, targets, epochs=2)`; `update_actor(run, data, advantages)`. Run requires policy/optimizer, value model/optimizer, tokenizer, config, and optional `training_progress` callback. Fixed positive normalizer and native top-k behavior metadata are required.
- `mis_ppo.update_policy(run, trajectories)` uses policy/optimizer and optionally separate value model/optimizer. Config fields: `mis_token_ratio_min=.95`, `mis_token_ratio_max=1.05`, `mis_trajectory_ratio_min=.95`, `mis_trajectory_ratio_max=1.05`, `mis_fallback_to_valid=False`, `mis_normalize_advantages=True`, `mis_adv_norm_eps=1e-6`, `mis_value_loss_coef=.5`, `mis_kl_coef=0`, `mis_bootstrap_truncated_trajectories=False`, `mis_outcome_reward_weight=1`. Actor-only mode uses group-normalized advantages; with a critic it uses original token returns minus detached values. The critic mode is needed for single-rollout prompts to learn without a group baseline.
- Optimizer dependencies: `training_memory` workspace constants, `optimizer_chunk_elements`, `prepare_policy_update`; native model kernel modules `dppo_loss`, `offpolicy_loss`, `fused_cispo_head`, `master_weight_adamw`, `lfm2_shortconv`, `lfm2_fused`.

## Validation

Run with the mandated interpreter and CUDA hidden for CPU-only checks:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src uv run python -m pytest tests/test_cispo.py tests/test_mis_ppo_algorithm_api.py tests/test_offpolicy_math.py tests/test_offpolicy_loss.py -q
```

At initial handoff: **27 passed, 15 skipped** across those suites (run in two batches). This covers CISPO signed/clipped gradients and accumulation, all extracted MIS-PPO numerical API regressions, exact expected score-centering gradients, masked extreme ratios, and fused sparse custom-autograd value/gradient parity over FP32/BF16 vocabularies up to 65,536 entries. GPU cases were skipped deliberately. All owned modules/tests pass byte compilation.

Additional production-update suites are ready in `test_flashreinforce.py`, `test_offpolicy.py`, `test_offpolicy_normalization.py`, `test_grouped_mis.py`, `test_value_mis_sc.py`, `test_ppo_math.py`, and `test_mis_ppo_update.py`; these require the shared config/trainers/eval files being ported in the next integration wave. Historical script-only experiment tests were excluded. Tests use tiny synthetic policies and real packed trajectory contracts. A CPU mmap optimizer checkpoint round-trip checks exact state restoration without launching GPU work.

No model inference, training, or benchmark evaluation was launched; no model trace report is claimed. GPU speed/peak-memory preflight, actual rollout traces and training W&B logging remain release integration work under the repository instructions.

## Independent integration review follow-up

Reviewed the completed unified driver, PPO/GRPO runtimes, optimizer/checkpoint restoration, dynamic group collection, and value-MIS phase transitions. Frozen value targets are collected before critic fitting, the actor remains offloaded during critic work, and the critic is offloaded before actor work. MIS-PPO retains its original per-pack active-token actor reduction weighted by each pack's share of the global valid tokens; it is not silently converted to a fixed-token MIS reduction.

Fixed concrete boundary defects found during review:

- MIS-PPO's optional truncated-response bootstrap now recognizes a native direct response whose actual `finish_reason` is `length`, even though `DirectHarness` sets `terminal_reason='single_turn'`. EOS and string-stop responses remain terminal outcomes. Historical multi-turn `budget_exhausted`/`max_turns_reached` semantics are preserved.
- Mapped critic and shared optimizer construction exclude frozen parameters before eager FP32 state allocation. This matches the original value runner's trainable-only critic parameter list and avoids masters/moments/checkpoint bytes for frozen embeddings, mixers, and the unused language head.
- A zero `value_mis_alpha` is rejected during prelaunch validation, consistent with the numerical target function's strictly positive requirement.
- The MIS-PPO ratio prepass explicitly releases its last dense vocabulary logits reference before backward, avoiding an otherwise unnecessary full-pack logits allocation throughout the learner update.
- Actor-only MIS-PPO rejects singleton groups (which yield identically zero group advantages), nonzero entropy/KL coefficients, truncated bootstrap without a critic, and disabling its mandatory group advantage normalization. Its existing zero-coefficient grouped objective is unchanged.

`tests/test_algorithm_review.py` exercises these boundaries, including actual native-updater outcome replacement for capped responses and exact mmap byte coverage. No GPU work was used for this review. The focused integration regression run includes that file, `test_training_runtime.py`, `test_mis_ppo_update.py`, and the original `test_mis_ppo_algorithm_api.py`.
