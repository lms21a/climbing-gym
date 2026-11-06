# Native inference and model port

## Sources and selection

- `/home/lukas928/SimplePT/simple-pt-knights-value-mis-sc`, commit `07b6762`: native LFM2.5 inference, advanced sessions, CPU generation/scoring, async adapter, and corresponding tests. This branch adds policy-version checks, immutable prefix snapshots, independent child forks, preserved RNG positions, and optional session GPU timing over the memory-defaults baseline.
- `/home/lukas928/SimplePT/simple-pt-memory-defaults`, commit `6edae8f`: model, training kernels, checkpoint/tokenizer compatibility; additive `release_inference_buffers()` and async `sleep()` lifecycle. Model-root contents match the value/MIS branch.
- `/home/lukas928/SimplePT/simple-pt-lfm3-fp8`, commit `639e958`: additive LFM3 inference, seven-tap convolution, gated/partial/sliding attention and optional FP8 operator projections. Its older LFM2/session implementation was deliberately not copied over the newer implementation.

All 155 existing worktrees were inspected via `git worktree list` and grouped by inference Python-file hashes. Targeted history/diffs compared adaptive-decode (`5232a21`), async-rollouts (`afd1083`), decode-optimizations (`f45e364`), fp8-inference (`ad0ee05`), lfm3-fp8, memory-defaults, and knights-value-mis-sc. The selected LFM2 implementation includes the mature FP8 MLP/KV/head, shared physical cache, grouped-query decode and adaptive compute-batch optimizations already integrated in the latest value/MIS branch. Exact original source commits and per-file SHA256 values are recorded in `native_source_manifest.json`.

## Preserved public contracts

`climbing_gym.models` exposes LFM2Config/LFM2Model and the registry. Models retain checkpoint key compatibility, FlashAttention, packed-sequence boundary handling, fused convolution, MLP recomputation and gradient/head/optimizer kernels. `climbing_gym.models.lfm2_compat` handles older and modern RoPE/config spellings and tokenizer exports without importing a SimplePT package.

`climbing_gym.inference` exposes the native LFM25Engine, LFM25AsyncInferenceEngine, sampling/result contracts, continuation sessions, and optional LFM3Engine. `AsyncInferenceEngine` is an alias of the native adapter. The obsolete vLLM implementation/config builders and vLLM benchmark were removed; torch compile model benchmarking remains under `models.benchmarks.lfm2`.

LFM2.5-1.2B Base, Instruct and Thinking use their own local tokenizer/chat templates. The async adapter preserves `enable_thinking`, ordinary chat formatting, assistant prefills, exact token-prompt bridging, text stops, sampled-token log probabilities, request-owned sparse behavior distributions, thread-safe policy synchronization, and sync/async generation. CUDA-only modules remain lazily imported by CPU paths. CPU fallback uses native PyTorch operations.

Explicit async `max_batch_size` and configured `max_num_seqs` now both bound the effective batch; nonpositive/noninteger limits fail early. This closes a source edge case where an explicit native batch limit silently overrode a lower configured request cap. The default remains 64 when neither is set. Native buffer release drains the generation lock and rejects active sessions before releasing captured graphs/cache/packed copies.

## Verification

Run with GPUs hidden and the shared migration virtual environment:

```bash
CUDA_VISIBLE_DEVICES='' PYTHONPATH=src /home/lukas928/SimplePT/.venv/bin/python -m pytest -q \
  tests/test_lfm25_engine.py tests/test_lfm25_async_engine.py tests/test_models.py \
  tests/test_lfm25_26b_compat.py tests/test_lfm2_checkpoint_layers.py \
  tests/test_lfm25_fp8_kv.py tests/test_lfm25_fp8.py tests/test_lfm25_cuda.py \
  tests/test_lfm25_session.py tests/test_lfm25_behavior_head.py tests/test_text_stops.py \
  tests/test_lfm25_adaptive.py tests/test_native_completion_scoring.py tests/test_lfm3_native.py
```

Result: **68 passed, 104 skipped** in 6.26s. Skips require CUDA. Fourteen PyTorch TorchScript deprecation warnings originate in torch.compile machinery. Coverage includes full-model/native prefill/decode parity on variable-length prompts, sync checkpoint shape validation, untempered completion scoring, checkpoint round trips, modern 30-layer Transformers parity, async batching, token/chat/thinking prefills, logprob alignment, stops inside a final token, and CPU packed-boundary correctness. Native benchmark has a separate real tiny-checkpoint CPU smoke test (`tests/test_native_benchmark.py`). No GPU work or trained-model accuracy evaluation was launched by this agent.

## Native speed check for release QA

The standalone native command writes JSON timing/config/peak-memory metrics and a Markdown file with complete actual sample prompts/responses alongside them. It forces the entire output budget while retaining temperature-1 sampling and sampled-token logprobs, so changing response lengths does not obscure execution speed.

Use a self-contained copied checkpoint for final release QA. Existing local checkpoints available during migration are `/home/lukas928/SimplePT/saved_models/LFM2.5-1.2B-Base`, `LFM2.5-1.2B-Instruct`, and `LFM2.5-1.2B-Thinking`. These are optional user-provided data paths, never runtime source dependencies.

```bash
PYTHONPATH=src /home/lukas928/SimplePT/.venv/bin/python -m climbing_gym.inference.benchmark \
  --model /path/to/LFM2.5-1.2B-Base --device cuda --dtype bfloat16 \
  --batches 1 32 128 --prompt-tokens 128 --output-tokens 4096 --max-model-len 4608 \
  --temperature 1 --warmups 1 --repeats 2 --cuda-graphs \
  --fp8-mlp --fp8-kv --fp8-lm-head --adaptive-batching --max-prefill-tokens 4096 \
  --output artifacts/native_speed.json
```

Repeat with the actual environment prompt IDs (`--prompts-json` accepts an object with a `prompts` list of token-ID rows) and intended training/evaluation token budget. For the historical Futoshiki workload use batch 128, 8192 output tokens, max model length 9216, and `--park-gib 6` to approximate resident optimizer memory. GPU ownership must be coordinated; run original and port in separate processes. This benchmark's synchronous concurrency equals each requested batch; also verify training's native batch, max_num_seqs, rollout max-inflight limit, and any higher-level semaphore agree before launch.

Prior RTX 4090 references in source `docs/native_adaptive_decode.md`: batch128 fixed8192, temperature1, logprobs, context9216, prefill4096, 6GiB optimizer parking: previous FP8 MLP/KV 18,135 output tok/s; plus FP8 head/adaptive 18,602 tok/s. Ragged128 improved 15,379 to 16,543 tok/s. Queue512/live128 improved 18,358 to 19,655 tok/s. These use a trained step265 actor and archived prompts, so synthetic/base-model results are not direct regressions against those numbers. Earlier MLP-only reference `docs/native_fp8_inference.md`: 11,601 tok/s, peak allocated22.793GiB, compared to BF16 10,889 tok/s and22.041GiB. FP8 KV changes the memory/compute profile substantially; compare matching flags, prompts, precision, budget, resident optimizer state and device.

After launch, inspect observed rollout throughput and full-step timing. Investigate unexpected slowdowns, especially mismatched concurrency or accidental eager decode; retain metrics throughout training and do not stop/probe training automatically due to speed fluctuations. Accuracy evaluations must use environment rewards and save complete Markdown traces spanning available tiers and successes/failures; this synthetic throughput command does not report environment accuracy.

## Remaining limits

CUDA numerical parity, FP8 paths, advanced session operations and real 1.2B throughput require the release QA GPU phase; CPU skips are not evidence for their GPU behavior. FP8 requires SM89+ and the native CUDA decoder needs compatible Triton/FlashAttention installations. FP8 remains opt-in and retains original BF16 weights for prefill/sync, rather than reducing all model residency.

LFM3 is an additive inference-only port from the working source. Its cached decode requires CUDA and it does not support learner weight synchronization or `from_model`; its prefill uses native CPU operations when FlashAttention is unavailable. LFM3 accuracy/performance was not revalidated during this CPU migration. The required LFM2.5 path is independently tested.
