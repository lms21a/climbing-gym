# Follow-up verification

The initial CPU-only review covered the standalone Futoshiki hill-climb example, readiness-gated Knights value training, optional CFOP integration, and removal of GitHub workflows. That phase used `/home/lukas928/SimplePT/.venv/bin/python` with `CUDA_VISIBLE_DEVICES=''` and launched no pretrained model inference/evaluation, training run, or GPU work. Tiny policies and explicit inference doubles were unit-test fixtures. The separately requested CUDA-enabled follow-up is recorded below; source SimplePT worktrees remained unchanged throughout.

The [original GPU release verification](release_verification.md) remains evidence for the shared native engine and LFM checkpoint compatibility. It does not establish throughput, capacity, convergence, or historical accuracy reproduction for these new example workloads. Their launch commands require their own matched native preflight.

## Concrete fixes from independent review

- **Pending SFT evaluation:** a resumed policy and its Adam tensors were restored to CUDA before the pending-epoch evaluator. That branch bypassed the usual checkpoint offload. It now moves both to CPU before evaluation and restores learner residency before another update. An evaluator failure leaves the pending marker intact and does not reload the learner.
- **Interrupted immutable SFT export:** copying directly into an epoch's final policy directory could expose a partial checkpoint. Export now uses a temporary sibling and atomic rename. Pending evaluation reconstructs a missing snapshot from the complete rolling checkpoint; completed immutable snapshots are checked and never overwritten.
- **Continuation identity:** SFT rolling state binds the saved model, configuration, and tokenizer files. Native projection, specialist collection, and preflight identities additionally bind `config.json`, so changing model semantics such as RoPE cannot silently reuse old artifacts. Historical weight hashes retain their original meaning. The runtime owner also bound Futoshiki RL checkpoints to the resolved configuration and saved policy/tokenizer bytes; independent regressions reject changed weights, tokenizer, and numerical settings.
- **Proposal probabilities and trace outcomes:** native proposal output rejects nonfinite or positive log probabilities. Projection preflight verifies the reconstructed prefix plus generated suffix, while the trace keeps the actual generated suffix verbatim. Previously, cutting inside an answer could produce a false failure by grading the suffix alone.
- **CFOP documentation:** executing the minimal solved-cube program against the real Rust binding exposed missing pair declarations. The example now declares all four already-solved pairs at the Cross boundary. A documentation execution test checks the actual published Python example; CFOP grading semantics were preserved.

These changes are covered in `tests/test_followup_verification.py`, alongside the existing example and environment suites.

## Futoshiki workflow and identities

The actual preparation/status CLIs materialized **all 18 recipe stages: 16 RL stages and two projection/SFT stages** using a declared placeholder model path, without loading a model. All 18 appeared in status with prerequisite availability. Script help succeeds with the standalone package on `PYTHONPATH`; documented `uv run` commands supply the installed project normally.

The scoped tests verify the original ordered banks, hashes, independent puzzle checks, source-derived prompt/reward behavior, distinct historical projection rules, supervised masks/EOS, full archived trace export, and selected checkpoint lineage. They also exercise interruption after the first half of a 256-response cohort: exact optimizer continuation consumes the saved second 128-response half without recollection, preserving policy lags zero and one and bitwise next-update parity. Changed dataset, pending-cohort bytes, configuration, or policy/tokenizer assets fail closed. Cross-stage optimizer transfer remains an explicit separate transition with a new stage dataset.

The 4,096-token training and 8,192-token endpoint-test budgets have distinct context/preflight reports. CPU tests check endpoint startup validation, panel repeats, and pass@4/pass@8 calculations. Full actual-response trace formatting and expected answers are exercised with explicit fixtures; [recorded projection samples](../../examples/futoshiki_hill_climb/recorded_projection_samples.md) are clearly identified historical model outputs, not new evaluation measurements.

## Knights value phase and CFOP

The Knights gate uses fresh-rollout **pre-fit** critic diagnostics, a minimum warmup count, consecutive readiness, and a hard maximum. Critic fitting during warmup does not advance the actor or its optimizer. Failed readiness preserves a failed-phase checkpoint and prevents actor training. The reviewed phase sidecar binds the readiness rule and numerical/collection method; exact continuation restores critic/actor optimizer state, RNG, dataset cursor, counters, and phase. Warmup directories are distinct from actor update checkpoints. Tokenizers are now persisted at both warmup and actor checkpoint boundaries after review feedback.

The dedicated tiny-policy tests compare uninterrupted training with both mid-warmup and readiness-boundary continuation, checking actor/critic parameters and collection order exactly. The example wrapper subtracts completed actor updates from its configured target on resume, so warmup batches do not consume the actor update budget. Held-out panels are separate from the readiness gate.

CFOP remains optional. The focused suite checks useful missing-binding errors, version/schema rejection, other environments working without the binding, malformed states/programs, phase fidelity, grounded reasoning, stage-local Cross, and binary-objective restrictions. With the installed historical `cfop_solver` 0.4.0 extension, the real Rust verification and documented minimal program pass. The minimal solved-cube example earns strict success with reward 1.6; it is a verifier fixture, not a model result.

AST inspection found no `simple_pt` imports in the runnable Futoshiki or CFOP modules. The filesystem contains no GitHub workflow files, and packaging no longer includes `.github` in the source-distribution selection.

## Scoped checks and reproduction

- **92 passed** across independent follow-up, Futoshiki projection/runtime, and CFOP suites in 7.00 seconds.
- **6 passed** in the reviewed Knights warmup suite in 5.41 seconds, including both warmup continuation boundaries.
- Actual CPU-only preparation/status and help commands completed; [record](../../artifacts/verification/followup_static.json), [preparation log](../../artifacts/verification/followup_prepare.log), [status output](../../artifacts/verification/followup_status.log).
- Test output: [Futoshiki/CFOP/follow-up log](../../artifacts/verification/followup_focused.log), [Knights warmup log](../../artifacts/verification/followup_knights.log).

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/lukas928/SimplePT/.venv/bin/python -m pytest -q \
  tests/test_followup_verification.py tests/test_futoshiki_hill_climb_projection.py \
  tests/test_hill_climb_runtime.py tests/test_cfop_support.py \
  tests/test_cfop_reasoning.py tests/test_cfop_integration.py
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/lukas928/SimplePT/.venv/bin/python -m pytest -q \
  tests/test_value_warmup.py
```

These are scoped review snapshots. Final whole-repository CPU and distribution checks are owned by the [packaging verification](followup_packaging.md); subsequent example-only tests may increase the final counts. Raw local review artifacts remain ignored and are not distributed. No new model accuracy or throughput measurement is claimed.

## CUDA-enabled follow-up — November 6, 2026

The complete repository suite was rerun with **`CUDA_VISIBLE_DEVICES=0`** against clean source commit `3868e4c7ab6522ff0b9f09f22e676e1ddd9a576f`. Before pytest, an explicit PyTorch assertion confirmed CUDA availability and exactly one selected GPU. The RTX 4090 had no compute processes and 0% utilization; this run owned the GPU exclusively.

**Result: 803 passed, zero skips, zero failures/errors, 42.35 seconds, exit code 0.** All 130 CUDA cases that were skipped in the preceding CPU-only whole-repository run executed successfully. The 14 warnings are the existing PyTorch TorchScript deprecations from `tests/test_models.py`. No code fixes, relaxed correctness assertions, or additional skips were needed.

Hardware/runtime: NVIDIA GeForce RTX 4090, 24,564 MiB, compute capability 8.9, driver 610.88; PyTorch `2.10.0+cu128`, CUDA `12.8`, Triton `3.6.0`, FlashAttention `2.8.3`, pytest `9.0.2`. The suite includes native CUDA/FP8 kernels, sampler probabilities, adaptive/session behavior, the 64-iteration graph recapture regression, and actual CUDA optimizer offload/resume checks, alongside the new example and CFOP contracts. These are existing unit/integration tests, not a pretrained model training or evaluation campaign, and do not establish new model quality or full-training capacity.

Full evidence: [pytest log](../../artifacts/verification/cuda_followup/pytest.log), [JUnit XML](../../artifacts/verification/cuda_followup/results.xml), [hardware/environment](../../artifacts/verification/cuda_followup/hardware.json), and [machine-readable summary](../../artifacts/verification/cuda_followup/summary.json). The summary explicitly records empty skip, failure, and error lists. No compute process remained after completion.

```bash
mkdir -p artifacts/verification/cuda_followup
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=src \
  /home/lukas928/SimplePT/.venv/bin/python -m pytest -ra \
  --junitxml=artifacts/verification/cuda_followup/results.xml \
  > artifacts/verification/cuda_followup/pytest.log 2>&1
```
