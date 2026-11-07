# Public release readiness review

The November 7, 2026 review covered native inference boundaries and ownership, training and configuration validation, SFT, checkpoint continuation, projection examples, optional CFOP integration, and distribution portability. Review started from `a180fad`; the final validation records a source fingerprint for the reviewed changes. Repository visibility and publishing were not changed by the review.

## Runtime fixes

- **Native actor retirement:** closing a constructor-owned adapter now drains its worker lock, releases inference buffers, drops actor and stop-table references, and collects unreachable native decoder/compiled-callable cycles before another actor loads. Setup failures clean up owned resources. Engines supplied through `from_components` remain caller-owned; closing an adapter does not invalidate their graphs or weights. Work waiting on close fails explicitly rather than dereferencing a retired engine.
- **Sampling boundaries:** invalid nonfinite controls, fractional/bool token counts, malformed string stops, and unsupported `extra_params` now fail clearly. Supported extras remain `repetition_penalty`, `ignore_eos`, and `behavior_topk`. Sampling-attribute objects retain supported numeric conversions, stop strings and extras instead of silently losing them.
- **Failed updates:** SFT rejects nonfinite clipping limits before forward execution and nonfinite gradients before optimizer mutation, including when clipping is disabled. Finite nonpositive limits retain the existing no-clipping behavior. MIS-PPO validates both actor and critic gradients before either optimizer step, so a failed critic backward cannot leave a partially updated actor.
- **Configuration fidelity:** invalid ratio bounds, optimizer settings and clipping limits are rejected. PPO settings that the retained implementation does not support now fail explicitly rather than being accepted without their advertised effect. Valid numerical reductions and supported clipping conventions are preserved.

CPU failure-injection tests check native reference retirement, in-flight worker ordering, shared-engine preservation, setup cleanup, sampling conversions, and unchanged model/optimizer state after nonfinite backward. The runtime/SFT review suite passed **118 focused CPU checks**; trainer validation has its own focused regressions. Checkpoint/RNG/data continuation, projection identities, pending SFT evaluation recovery, and example lifecycle contracts were also reviewed against their existing tests.

## Complete CUDA validation

The final repository suite passed **879 tests, with zero skips, failures or errors**, in **41.16 seconds**. The 14 warnings are existing PyTorch TorchScript deprecations. CUDA was explicitly enabled and asserted available before the run.

The environment was Python 3.12.8, PyTorch `2.10.0+cu128`, CUDA `12.8`, Triton `3.6.0`, FlashAttention `2.8.3`, and an RTX 4090 with compute capability 8.9. The suite includes actual native BF16/FP8 graph generation, adaptive/session and sampled-probability checks, graph recapture, optimizer offload/resume, and two new constructor-owned actor close/reload regressions. Those lifecycle cases verify raw-engine/decoder/weight reclamation, reduced allocated CUDA memory, and successful generation after loading another actor. Their sampling explicitly ignores EOS to exercise a fixed output budget.

Raw evidence is retained locally under the ignored `artifacts/release_review/` directory: `hardware.json`, `summary.json`, `cuda.log`, and `cuda.xml`. The summary records the tested source fingerprint and independently parsed JUnit counts. The focused runtime log is `artifacts/verification/public_native_review.log`. These local artifacts are deliberately excluded from distributions.

This pass used numerical/kernel/native-engine unit and integration tests, including tiny policies and explicit doubles. It did not launch pretrained model training or a quality evaluation, and it does not establish convergence, full-model training capacity for every method, or reproduction of historical scores. Earlier pretrained LFM compatibility evidence is described in the [original verification note](porting/release_verification.md).

## Reproduction and optional dependencies

After the documented [development setup](../CONTRIBUTING.md), run from the repository root:

```bash
CUDA_VISIBLE_DEVICES='' uv run python -m pytest -ra
CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 uv run python -m pytest -ra
```

The reported zero-skip CUDA run had the required CUDA libraries and the optional `cfop_solver` 0.4.0 binding installed. Reproducing that exact count requires those dependencies and compatible hardware. CFOP's pinned Rust solver is currently obtained separately from its private repository; it is not bundled or needed for the other environments. See [CFOP installation](cfop.md). Without that binding, CFOP-specific checks skip and constructing CFOP gives installation guidance.

Packaging checks cover wheel/source contents, imports, command help, portable files, and dependency declarations. They include isolated checks of the locked lightweight dependencies while reusing the validated Torch/Triton/NVIDIA installation; this is not a claim of a clean full CUDA bootstrap on another machine. No GitHub workflows or hosted publication were introduced.
