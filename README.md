# Climbing Gym

Climbing Gym is a local, native-engine toolkit for LFM post-training. It brings
policy and actor–critic training, supervised fine-tuning, verified reasoning
environments, and projection sampling into one installable Python package.

The primary model family is **LFM 2.5 1.2B Base, Instruct, and Thinking**.
Checkpoints and tokenizers stay compatible with their Hugging Face format.
Model weights are user supplied and are never bundled.

## Start locally

Python 3.12 is the tested interpreter. Install with [uv](https://docs.astral.sh/uv/):

```bash
uv sync --python 3.12 --group dev
uv run climbing-gym --help
uv run climbing-gym --version
uv run pytest -q
```

Native CPU inference and CPU contract tests work without FlashAttention. CUDA
training and fast decoding require a compatible NVIDIA driver, CUDA toolkit,
PyTorch, Triton, and FlashAttention. The migration used Torch 2.10.0,
Transformers 4.57.1, Triton 3.6.0, and FlashAttention 2.8.3. For a compatible
Linux CUDA environment, install acceleration after the base environment:

```bash
uv sync --extra cuda --no-install-package flash-attn
uv sync --extra cuda --no-build-isolation-package flash-attn
```

FlashAttention may compile locally; its build requires the matching CUDA toolkit
and a C++ compiler. CPU CI excludes this optional extra. See the
[run guide](docs/running.md) for configuration, memory, resume, and logging.

Point to an existing local checkpoint, authenticate W&B for training, then
prepare and benchmark the actual evaluation workload before running it:

```bash
export CLIMBING_GYM_MODEL=/absolute/path/to/LFM2.5-1.2B-Instruct
uv run climbing-gym preflight --config configs/eval_knights_knaves.toml \
  --evaluation --output artifacts/eval_knights_knaves/prompts.json
uv run climbing-gym benchmark --model "$CLIMBING_GYM_MODEL" \
  --prompts-json artifacts/eval_knights_knaves/prompts.json \
  --batches 3 --output-tokens 4096 --max-model-len 8192 \
  --device cuda --dtype bfloat16 --temperature 1 --top-k -1 --top-p 1 \
  --cuda-graphs --bf16-weight-packing --max-prefill-tokens 4096 \
  --output artifacts/eval_knights_knaves/speed.json
uv run climbing-gym evaluate configs/eval_knights_knaves.toml
```

Inspect the benchmark speed/memory and evaluation's complete
`evaluations/knights_knaves/traces.md` report. These three generated example rows
exercise the format; they do not constitute a benchmark accuracy claim. A first
benchmark is recorded as an initial baseline. Supply a relevant prior report on
later runs and investigate unexplained slowdowns. [Preflight details](docs/running.md#native-preflight)
cover exact flags, concurrency, sparse log probabilities, and training startup.

## Workflows

| Command | Purpose |
| --- | --- |
| `climbing-gym train CONFIG.toml` | Complete native collection/update/checkpoint loops |
| `climbing-gym evaluate CONFIG.toml` | Environment rewards, summaries, complete actual traces |
| `climbing-gym sft CONFIG.toml` | Packed response-only or full-document supervised training |
| `climbing-gym project --help` | Verified greedy likelihood projection into SFT data |
| `climbing-gym preflight --help` | Actual tokenized workload preparation without loading weights |
| `climbing-gym benchmark --help` | Fixed-budget native throughput and peak GPU memory |

`python -m climbing_gym` exposes the same commands. Workflow modules also remain
executable directly. Top-level package import, help, and version do not import
Torch or initialize CUDA or W&B.

The four environment families are **GSM8K, Countdown, Knights and Knaves, and
Futoshiki**. Examples cover all four and every supported training method. The
GSM8K format examples are handwritten arithmetic, not redistributed GSM8K data.

| Method | Convention retained |
| --- | --- |
| PPO | Clipped actor objective and a separate critic with terminal-outcome GAE |
| GRPO / CISPO | Group-relative outcomes; CISPO clips detached importance weights |
| FlashREINFORCE | Batch outcome centering and complete-response sampled KL gating |
| MIS / MIS+SC | Token correction, fixed denominator, optional sparse score centering |
| Grouped MIS / MIS+SC | Complete mixed binary-outcome groups and global fixed denominator |
| MIS-PPO | Separate token/trajectory gates with optional learned critic |
| Value MIS+SC | Frozen critic advantages, critic fit, then actor update |

These reductions are intentionally distinct. See [algorithm conventions](docs/algorithms.md)
for estimator/configuration contracts and limitations. Projection's historical
“MCMC” name describes a **greedy base-likelihood ascent heuristic**, not exact
Metropolis–Hastings. References must be algorithmic or independently verified
outputs of the same model; stronger-model distillation is outside this project.

## Package map

| Module | Responsibility |
| --- | --- |
| `climbing_gym.models`, `.inference` | LFM models, native generation/scoring, sessions and kernels |
| `.envs`, `.rollout` | Four verifiers, task prompts, bounded collection and harnesses |
| `.train`, `.ppo`, `.grpo`, `.cispo` | Unified entry point and retained training implementations |
| `.flashreinforce`, `.offpolicy`, `.mis_ppo`, `.value_mis_sc` | Algorithm objectives and updates |
| `.sft`, `.sampling` | Supervised training and verified native projection |
| `.config`, `.trainers`, `.eval`, `.preflight` | Portable TOML, checkpoints, logs, evaluation and startup checks |

## Scope and provenance

This is an initial local release. CPU tests cover numerical contracts, checkpoint
continuation, and integrated training routes. Native CUDA verification and actual
model traces are documented in the [release verification report](docs/porting/release_verification.md).
Long training, convergence, and full-model capacity across all methods are not
established by smoke tests. LFM3 is an additional inference-only implementation;
it has no release accuracy/performance guarantee.

Source code is MIT licensed; attribution and source revisions are recorded in
[porting notes](docs/porting/coordination.md). The port originates in Lukas's
SimplePT implementations and has no runtime dependency on that checkout.
Model weights and external datasets keep their own licenses. This repository
contains no weights and no publishing automation.
