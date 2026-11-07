# Independent release verification

Verified locally on the RTX 4090 (24 GiB), using only `uv run python`. Source SimplePT worktrees remained read-only. No remote repository, publishing, long training, or real W&B training run was created. Full-model inference was exercised; tiny optimizer/unit updates do not establish full-model training capacity or convergence.

## Compatibility and numerical behavior

Real local LFM2.5-1.2B Base, Instruct, and Thinking checkpoints all load and generate through the standalone native engine. Base exercises its ordinary chat tokenizer/template; Instruct and Thinking exercise their own chat templates plus the established `<thinking>\n` prefill. Instruct additionally returns `4` for an ordinary chat request without that prefill. Each variant first completes a representative batch 32 × 4,096-token native rollout with sampled raw log probabilities. Native batch, maximum sequences and request cap are explicitly 32; all 12 quality requests are submitted concurrently. There is no continuation or compaction.

A separate-process deterministic original-versus-port comparison uses 12 prompts covering all four families, 64 tokens each, identical BF16/FP8 graph settings. **Every generated token ID and raw log probability is bitwise equal.** Independently recomputing first-token log probabilities from the actual native prefill distribution has maximum absolute error `9.33e-7`. Matching the actual prefill chunk/cache settings is necessary: a differently batched BF16 recomputation need not be bitwise identical. Raw results: `artifacts/verification/parity_comparison.json`, `artifacts/verification/parity_traces.md`.

The real `python -m climbing_gym evaluate --config ...` route also validates saved preflight evidence and completes 32 native requests with batch/sequence/inflight limits 32. It saves all 32 complete trajectories and trace records. These repeat three local Knights examples to test CLI/runtime wiring and are **not an accuracy benchmark**. Actual collection/evaluation wall time was 6.048 s for 11,084 completion tokens (1,832.7 output tokens/s including evaluator work). `artifacts/verification/cli_eval.toml`, `artifacts/verification/cli_eval_summary.json`, `artifacts/verification/cli_eval_traces.md`.

## Throughput and memory

Original and port run in separate processes, with `OMP_NUM_THREADS=8`, identical 128 saved prompt-ID rows spanning GSM8K-format arithmetic, Countdown, Knights and Knaves, and Futoshiki. Individual prompt lengths range 56–524 tokens. Each measured request forces 4096 output tokens, temperature 1, full support, sampled log probabilities, context 4864, prefill 4096, CUDA graphs, BF16 packing and FP8 MLP/KV/head plus adaptive batching. Model load and one full-budget warmup are excluded; medians use two measured runs. Tokens and requested log probabilities are transferred to CPU inside timing. Synchronous batch size is the request concurrency, without an additional semaphore. No optimizer memory is parked.

| Batch | Original output tokens/s | Port output tokens/s | Port/original | Original peak allocated GiB | Port peak allocated GiB |
| --- | ---: | ---: | ---: | ---: | ---: |
|32|11,950.6|11,938.1|0.99896|4.3244|4.3259|
|128|26,700.8|26,725.8|1.00094|7.0360|7.0429|

There is no material throughput regression. Small peak differences include benchmark output tensors retained by the timing harness. Raw measurements and reserved-memory counters: `artifacts/verification/original_speed.json`, `artifacts/verification/port_speed.json`, `artifacts/verification/speed_comparison.json`. Complete actual fixed-budget samples: `artifacts/verification/original_speed_traces.md`, `artifacts/verification/port_speed.md`.

Historical references remain workload-specific. Source `docs/native_adaptive_decode.md` reports 18,135→18,602 tokens/s for batch 128/8192 tokens/context 9216/prefill 4096, 6 GiB optimizer parking and a trained step 265 actor; ragged 128 reports 15,379→16,543, queue 512/live 128 reports 18,358→19,655. Source `docs/native_fp8_inference.md` reports an older MLP-only 11,601 tokens/s and 22.793 GiB peak versus BF16 10,889 and 22.041 GiB. Those trained checkpoints, archived prompts, parked memory and acceleration combinations differ from this release test. The matched original process above is the relevant regression baseline; the higher release number is not a speedup claim against those historical workloads.

Additional per-variant batch 32 preflights were 11,940.1 (Base), 11,930.5 (Instruct), and 11,935.5 (Thinking) tokens/s, with 4.3225–4.3244 GiB peak allocated. Actual EOS-terminated 12-request smoke throughput was 3,458.4, 1,595.7 and 3,094.7 tokens/s respectively. These smaller, shorter and variable-length batches naturally differ from full-capacity fixed-budget speed; full wall timing and output counts are retained in each `*_smoke.json`. No throughput watchdog or training-time speed probe was used.

## Actual model traces

Each model receives 12 examples spanning the available tiers in all four families. The arithmetic examples are handwritten **GSM8K-format smoke rows**, not a held-out GSM8K benchmark. Puzzle rows are generated/verified local examples. Outcomes use the real environment verifiers; strict formatting failures count as failures.

| Checkpoint | Verifier successes /12 | Full actual prompt, response, expected answer and outcome |
| --- | ---: | --- |
|Base|3|`artifacts/verification/base_traces.md`|
|Instruct|4|`artifacts/verification/instruct_traces.md`|
|Thinking|3|`artifacts/verification/thinking_traces.md`|

These reports retain successes and failures, all selected tiers, full 4,096-token preflight samples and complete EOS-terminated quality responses. They establish compatibility and inspectability, not an accuracy estimate or a quality guarantee on every puzzle. JSON response files retain exact IDs and raw per-token log probabilities.

## Automated and package verification

Final CPU-only suite after packaging, algorithm review and SFT failure-cleanup integration: **568 passed, 130 skipped**, 12.67 s. Skips are CUDA cases; 14 PyTorch TorchScript deprecation warnings remain. **Full CUDA suite: 698 passed, no skips or failures**, 25.46 s, including the 64-iteration recapture stress test and both new actual CUDA optimizer lifecycle cases. `artifacts/verification/cpu_suite_final.log`, `artifacts/verification/cuda_suite_final.log`, `artifacts/verification/recapture_regression.log`. CPU integration includes real tiny updates through all ten algorithm routes, exact checkpoint/RNG continuation, solver checks, SFT resume, projection scoring and logging/preflight contracts.

Release verification fixed six CUDA-only helper imports that referenced a nonexistent renamed test module. The first full CUDA pass then exposed one missing historical subprocess diagnostic; the generic 64-iteration graph recapture stress harness is now self-contained at `tests/helpers/native_graph_recapture.py`. New `tests/test_release_gpu_lifecycle.py` checks actual CUDA offload/reload and checkpoint restoration for resident/compressed masters and host-state Adam, bitwise next-update/FP32-master preservation, moment dtypes, and bounded host-state workspace.

AST import inspection and clean entry-module imports find no `simple_pt` dependency or loaded source module; `artifacts/verification/import_hygiene.json`. CUDA tests cover fused kernels, FP8 projections/KV/head, actual sampled log probabilities, asynchronous/session lifecycle, cache snapshots, synchronization, policy changes and optimizer transfers. See [packaging verification](release_packaging.md) for isolated wheel install/import/CLI and artifact contents. Recorded runtime: torch 2.10.0, Transformers 4.57.1, Triton 3.6.0, FlashAttention 2.8.3, Safetensors 0.7.0; test runner pytest 9.0.2.

## Reproduction

From the standalone checkout:

```bash
CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=8 PYTHONPATH=src uv run python -m pytest -q
OMP_NUM_THREADS=8 PYTHONPATH=src uv run python -m pytest -q
OMP_NUM_THREADS=8 PYTHONPATH=src uv run python -m climbing_gym.inference.benchmark \
  --model /path/to/LFM2.5-1.2B-Instruct --device cuda --dtype bfloat16 \
  --batches 32 128 --prompts-json artifacts/verification/prompts.json \
  --output-tokens 4096 --max-model-len 4864 --temperature 1 --warmups 1 --repeats 2 \
  --cuda-graphs --bf16-weight-packing --fp8-mlp --fp8-kv --fp8-lm-head --adaptive-batching \
  --output artifacts/verification/port_speed.json
```

`climbing_gym.preflight --config CONFIG --output prompts.json` prepares new real prompt IDs without loading weights. The original comparison uses the read-only `simple-pt-memory-defaults/scripts/benchmark_lfm25_inference.py` with `--backend native-cuda --logprobs --native-fp8-mlp --native-fp8-kv --native-fp8-lm-head --native-adaptive-batching` and otherwise the same model, prompts, budgets, dtype and repetitions. Its default BF16 packing is enabled. Provenance is source commit `6edae8f`; the standalone engine also contains the later session lineage documented in [native engine notes](native_engine.md).

Raw logs, QA scripts and model traces are saved locally under `artifacts/verification/`, deliberately excluded from distributions and git by the package ignore policy. Their links refer to this verified local workspace. No external checkpoint directory is a Python runtime dependency; checkpoint weights are user-provided data and are not redistributed.
