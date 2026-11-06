# Release packaging and documentation

This is a local-only release preparation. No remote repository was created,
no git push or package publication occurred, and no source SimplePT checkout was
changed. The root coordinator owns the final local commit.

## Attribution and package boundaries

The distribution is `climbing-gym` 0.1.0, imported as `climbing_gym`, requiring
Python >=3.12 (validated on 3.12.8). The user selected MIT; `LICENSE` credits
Lukas and contributors (2026). Original attribution/comments remain in ported
source. The original package metadata names Lukas; its read-only provenance is
`/home/lukas928/SimplePT/simple-pt/pyproject.toml`. Implementation source paths
and exact selected revisions are recorded in the other owner notes and native
manifest, particularly memory-defaults `6edae8f` and value/MIS `07b6762`.
Model weights/external datasets retain their own licenses and are not bundled.

The root package lazily exposes configuration, LFM model/native engine contracts,
and rollout coordination. CLI dispatches `train`, `evaluate`, `sft`, `project`,
`preflight`, and `benchmark` to the actual implementation modules. Top-level
import/help/version avoid Torch/CUDA/W&B initialization. Both a console script
and `python -m climbing_gym` are available. The benchmark now accepts optional
`argv` while preserving existing `main()` behavior, and its trace text correctly
distinguishes representative task prompts from synthetic throughput prompts.

Base dependencies are justified by runtime imports: Torch, Transformers,
Datasets, Safetensors, Hugging Face Hub, NumPy, and W&B. W&B is installed by
default because online training logging is the default. Optional `cuda` adds
Triton/FlashAttention and the latter's setuptools/wheel build prerequisites.
The two-stage CUDA installation in the README first installs prerequisites,
then builds FlashAttention against installed Torch without build isolation.
No acceleration install was attempted in the already validated shared environment.
The dev group contains pytest. `uv.lock` records the validated primary dependency
versions (Torch 2.10.0, Transformers 4.57.1, Datasets 4.4.2, Safetensors 0.7.0,
Hub 0.36.2, W&B 0.23.1, Triton 3.6.0, FlashAttention 2.8.3, NumPy 2.2.6,
pytest 9.0.2); transitive resolver results were not reinstalled into the shared
migration environment, so clean-environment CI remains a separate validation.

## Portable examples and docs

Ten `train_*.toml` examples cover all supported algorithm routes; four
`eval_*.toml` examples cover GSM8K, Countdown, Knights and Knaves, and Futoshiki.
`configs/sft.toml` consumes verified projection output. Examples reference local
JSONL data and `${CLIMBING_GYM_MODEL}`, never a source checkout or downloaded
weight path. RL optimizer states are CPU resident with a bounded 48 MiB workspace;
value MIS+SC uses mapped critic storage. Standard SFT retains its original AdamW
and full-head memory profile, explicitly documented. No example claims proven
1.2B full-model capacity or long-run convergence.

README gives executable local setup/evaluation and a package map. User docs
cover all estimator/reduction distinctions, native preflight using actual prompts,
4096-token reasoning budgets and sampled log probabilities, all concurrency
limits, BF16 packing/compile defaults, sparse probability timing, prior/baseline
comparisons, online W&B customization, local durable records, memory/resume,
complete evaluation traces, and verified greedy likelihood projection. The
projection documentation explicitly rejects exact-MH claims and stronger-teacher
references. CPU CI only tests and builds; it has no publishing step.

A shared path helper now expands `~`, `$VAR`, and `${VAR}`, reports unset
variables clearly, and resolves relative values against the TOML directory.
Training, standalone evaluation and SFT use the same helper; SFT now honors
`wandb.dir` consistently. CLI shell arguments retain ordinary cwd semantics.

## Validation

Every command used the mandated `/home/lukas928/SimplePT/.venv/bin/python` for
Python code/tests and `uv` for dependency resolution/build/install.
No GPU inference, model training or environment model evaluation was launched
by this packaging assignment. Unit-test fixtures are not model-quality results.

- `tests/test_release_packaging.py`: **24 passed**. Real method validators cover
  all ten training examples, build/verify all local environment schemas, and
  exercise all four actual standalone evaluation loaders without loading weights
  or collecting trajectories. Also covers all CLI routes, dependency-free
  top-level help/version in isolated Python, SFT configuration/log paths, and
  config-relative environment-variable expansion/unset-variable errors.
- Combined packaging/native-benchmark/preflight scope: **45 passed** in 2.74s
  (the final lazy-import assertions were rerun separately: 24 passed in 2.53s).
  Native benchmark testing uses a tiny CPU fixture; no 1.2B performance claim.
- `uv lock --python /home/lukas928/SimplePT/.venv/bin/python` succeeds.
- `uv build --python /home/lukas928/SimplePT/.venv/bin/python` creates wheel and
  sdist. Wheel metadata, MIT license, entrypoint and content boundaries were
  inspected. Sdist includes source, examples/configs, docs and lockfile. Weights,
  experiment artifacts and Python caches are excluded.
- `uv pip install --python SHARED_PYTHON --target TEMP_SITE --no-deps WHEEL`
  installed into a separate temporary target without modifying the shared env.
  Isolated shared Python (`-I`, cwd `/tmp`) imported 17 API/workflow modules from
  that wheel. All six actual workflow help commands and the generated console
  script's `--version` succeeded. Root import was checked before loading Torch.
  This validates wheel contents against installed dependencies, not installation
  of all dependencies into a clean environment.

Local machine-readable packaging evidence: `artifacts/packaging/validation.json`.
Build outputs are `dist/climbing_gym-0.1.0-py3-none-any.whl` and
`dist/climbing_gym-0.1.0.tar.gz`; these remain ignored local artifacts.
The independent release verifier owns final CPU/CUDA counts, native speed
comparison and actual-model evaluation traces in `release_verification.md`.

## Final release handoff

The independent verifier finalized **568 passed, 130 CUDA skips** on CPU and
**698 passed, no skips or failures** with CUDA (25.46s). The final wheel/sdist
were refreshed after all code, packaging and QA report changes. The isolated
validator confirmed current source/config/docs bytes match the distributions,
17 wheel API/workflow imports, all six workflow help commands, the console
version, MIT metadata, and exclusion of weights/artifacts. The sdist also includes
the required `tests/helpers/native_graph_recapture.py`; documentation links and
literal commands were checked. `uv lock --check` succeeds. Distribution hashes
and the temporary install location are recorded in local
`artifacts/packaging/validation.json`. No publication or remote change occurred.
