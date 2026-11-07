# CFOP Rubik's Cube environment

CFOP adds exact Rust-backed cube verification through the optional `cfop_solver` extension. The environment ports the existing SimplePT contracts: full-cube `strict-cfop-trace/v2`, grounded structured reasoning, and the stage-local Cross task. It uses the same native LFM rollout and training paths as the other environments.

## Install the optional solver

The required `cfop-solver` 0.4.0 source is currently in the private
[`lms21a/cfopsolver`](https://github.com/lms21a/cfopsolver) repository. This adapter
is not a self-contained public CFOP installation: you need existing access to
that repository or a compatible wheel supplied separately by its maintainer.
The normal Climbing Gym install and lockfile do not fetch or require it. All
other environments remain usable without the solver.

If you already have access, install the pinned solver into the same environment
as Climbing Gym after its base dependencies are ready:


```bash
uv pip install --python .venv/bin/python \
  'cfop-solver @ git+https://github.com/lms21a/cfopsolver.git@8d7d395c5993a83d6378949750682c47e23d818f'
```

Alternatively, use your existing clean checkout at that revision:

```bash
git -C /absolute/path/to/CFOPSolver rev-parse HEAD
uv pip install --python .venv/bin/python /absolute/path/to/CFOPSolver
```

Use the pinned revision above; the adapter's version/schema checks also reject
incompatible builds. A later exact `uv sync` may remove this separately installed
package; reinstall it afterward, or use `uv sync --inexact` when intentionally
preserving external packages.

Building from source requires Rust/Cargo and the maturin build backend. A compatible prebuilt wheel can also be installed with `uv pip install --python .venv/bin/python /path/to/cfop_solver-0.4.0-...whl`. The historical Linux wheel SHA256 is recorded in [the porting note](porting/cfop.json); a locally rebuilt wheel may have a different container hash.

The adapter checks version **0.4.0**, `facelets-urfdlb-v1` states, and `cfop-verification/v1` verifier output. Missing bindings produce an installation error when constructing CFOP, while other environments remain available. Incompatible bindings fail explicitly rather than silently changing grading.

## Full-cube task

```python
from climbing_gym.config import Env

env = Env(name="cfop_cube", reward="cfop_dense_v2").build()
example = env.build_example({
    "state": "UUUUUUUUURRRRRRRRRFFFFFFFFFDDDDDDDDDLLLLLLLLLBBBBBBBBB",
    "completion": "CROSS[FR,FL,BL,BR]:\nOLL:\nPLL:\nEND",
})
result = env.evaluate_response(example, "CROSS[FR,FL,BL,BR]:\nOLL:\nPLL:\nEND")
assert result.feedback.success
```

A state contains exactly 54 uppercase facelets in `U R F D L B` order. The binding validates cubies, parity, and orientation as well as sticker counts. Prompts show only the cube state and public protocol. Cached target responses, scrambles, and solver traces are not inserted into the model prompt.

The direct response is an exact line-based program:

```text
SETUP: y
CROSS: ...moves...
F2L FR: ...moves...
F2L FL,BL: ...moves...
OLL: ...moves...
PLL: ...moves...
END
```

This is a grammar illustration, not an executable solution. `SETUP` is optional and contains whole-cube rotations. Cross pair declarations can use `CROSS[FR,BL]: ...`. F2L lines declare pairs actually solved at that boundary and require nonempty moves. Accepted move notation includes face, slice, wide, and rotation moves with prime/double suffixes. Extra prose, malformed spacing, duplicate declarations, or additional content after `END` is rejected.

Solving the final cube alone is insufficient. The verifier replays phase boundaries, checks declared pairs, preserves earlier milestones, and enforces phase fidelity. It rejects moving downstream work into an earlier phase. After the first physically solved prefix, subsequent moves must preserve physical solve; the first canonical solve must be the final atomic move. The preserved protocol includes explicit terminal and OLL-skip exceptions.

`cfop_dense_v2` ranges from **0 to 1.8**. It combines format/notation, verified progress, phase invariants, strict solve, and bounded efficiency terms. Reference solutions support move-count comparison; answers are not graded by matching the reference string. `feedback.success` and `reward.info["strict_success"]` identify strict success separately from partial reward.

The dense reward works with FlashREINFORCE, PPO, and compatible scalar-reward objectives. Binary-only grouped MIS and Value MIS+SC reject this reward. In particular, the bounded value critic must not be fitted to targets above one.

## Grounded reasoning and stage-local Cross

Set `[env] response_format="reasoning_v1"` to use the preserved `CFOP_PLAN_V1` envelope. It contains a closed `<think>...</think>` block with one structured `STEP` per answer phase, followed by the phase program. Move counts, milestones, and gained/preserved/broken pair claims are checked against Rust-derived transitions. This is grounded structured supervision, not unrestricted natural-language process verification. The CFOP tags differ from the other puzzle families' `<thinking>`/`<answer>` convention; do not prepend an incompatible opening tag.

`Env(name="cfop_phase", reward="cfop_phase_v1", args={"requested_stage": "cross"})` exposes the original stage-local task. It requires rows with `task_type="phase"` and `stage="cross"`. A submission contains one `CROSS...` line and `END`, optionally preceded by the grounded reasoning envelope. Its dense reward ranges from zero to one; strict phase success remains separate. Only Cross is supported by this environment. Auxiliary F2L/OLL/PLL training rows must not be passed off as full-cube episodes.

Each request submits one full program or one stage-local transition through `DirectHarness`. The adapter does not invent a new interactive cube protocol. Native token IDs, sampled log probabilities, and ordinary chat templates follow the shared rollout contract.

## Run the examples

[Evaluation configuration](../configs/eval_cfop_cube.toml) and [one-step FlashREINFORCE configuration](../configs/train_cfop_cube.toml) use two small, Rust-verified original example states. They are wiring examples, not a benchmark panel or throughput recommendation. Supply a local model and use the [native preflight workflow](running.md#native-preflight) on the actual configuration before training or evaluation:

```bash
export CLIMBING_GYM_MODEL=/absolute/path/to/LFM2.5-1.2B-Instruct
uv run climbing-gym preflight --config configs/eval_cfop_cube.toml \
  --evaluation --output artifacts/eval_cfop_cube/prompts.json
```

Benchmark those actual prompts with the matching token budget, chat settings, native options, sampled log probabilities, and batch/request limits. Save the report at the configuration's `inference.preflight_report_path`. After inspecting throughput and peak memory, launch:

```bash
uv run climbing-gym evaluate configs/eval_cfop_cube.toml
```

Evaluation writes `evaluations/cfop_cube/traces.md` with actual prompts, complete responses, expected programs, and outcomes. Use a separate matched preflight for `configs/train_cfop_cube.toml`, authenticate W&B, then run `uv run climbing-gym train configs/train_cfop_cube.toml`. Runtime timing remains logged; speed guards do not stop active training for fluctuations.

## Port and verification

The source files had remained untracked in the historical checkout and were no longer on disk. They were recovered from archived source reads and successful patches in the July 17, 2026 development session, then renamed into `climbing_gym`. An intermediate recovered module exactly matched its archived SHA256, providing an independent reconstruction check. Only the code and compact [provenance metadata](porting/cfop.json) are included here; no conversation transcript is bundled.

Focused CPU tests cover optional dependency absence, version/schema mismatch, malformed states and programs, real Rust replay, hidden downstream solves, solve-then-disturb traces, grounded reasoning, stage-local validation, and binary-objective rejection. No model training or inference was run while adding this environment.
