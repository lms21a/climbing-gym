"""Benchmark native prefill/decode with fixed output budgets and sampled logprobs.

Timings include device-to-host transfer and exclude load/warmup. This is a
throughput check, not an environment accuracy evaluation.
"""
from __future__ import annotations

import argparse
import hashlib
from datetime import datetime, timezone
from dataclasses import asdict
import json
from pathlib import Path
import statistics
import time

import torch

from climbing_gym.inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig, flatten_token_batches
from climbing_gym.models.lfm2_compat import load_lfm2_tokenizer


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, help='Local checkpoint directory')
    parser.add_argument('--batches', type=int, nargs='+', default=[1, 32, 128])
    parser.add_argument('--prompt-tokens', type=int, default=128)
    parser.add_argument('--prompts-json', type=Path, help='JSON containing a prompts list of token-ID rows')
    parser.add_argument('--output-tokens', type=int, default=4096)
    parser.add_argument('--max-model-len', type=int)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--dtype', default='auto', choices=['auto', 'bfloat16', 'float16', 'float32'])
    parser.add_argument('--temperature', type=float, default=1.0)
    parser.add_argument('--top-k', type=int, default=-1, help='-1 disables top-k, matching rollout sampling')
    parser.add_argument('--top-p', type=float, default=1.0)
    parser.add_argument('--repetition-penalty', type=float, default=1.0)
    parser.add_argument('--behavior-topk', type=int, default=0)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--warmups', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=2)
    parser.add_argument('--cuda-graphs', action='store_true')
    parser.add_argument('--compile-decode', action='store_true')
    parser.add_argument('--fp8-mlp', action='store_true')
    parser.add_argument('--fp8-kv', action='store_true')
    parser.add_argument('--fp8-lm-head', action='store_true')
    parser.add_argument('--adaptive-batching', action='store_true')
    parser.add_argument('--bf16-weight-packing', action='store_true')
    parser.add_argument('--max-prefill-tokens', type=int, default=4096)
    parser.add_argument('--park-gib', type=float, default=0, help='Reserve optimizer-equivalent GPU memory')
    parser.add_argument('--output', type=Path, required=True)
    return parser


def _prompts(tokenizer, batch: int, length: int) -> list[list[int]]:
    rows = []
    for index in range(batch):
        ids = tokenizer.encode(f'Document {index}: Check each arithmetic operation carefully before continuing. ',
                               add_special_tokens=False)
        rows.append((ids * (length // len(ids) + 1))[:length])
    return rows


def main(argv=None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if min(*args.batches, args.prompt_tokens, args.output_tokens, args.warmups,
           args.repeats, args.max_prefill_tokens) <= 0 or args.park_gib < 0:
        parser.error('batch sizes, token counts and repetitions must be positive; parked memory must be nonnegative')
    if any((args.fp8_mlp, args.fp8_kv, args.fp8_lm_head, args.adaptive_batching)) and not args.cuda_graphs:
        parser.error('FP8 and adaptive batching require --cuda-graphs')
    device = torch.device(args.device)
    if (args.cuda_graphs or args.park_gib) and device.type != 'cuda':
        parser.error('CUDA graphs and parked GPU memory require --device cuda')
    tokenizer = load_lfm2_tokenizer(args.model)
    saved = json.loads(args.prompts_json.read_text())['prompts'] if args.prompts_json else None
    if saved is not None and (not saved or any(not row or any(type(token) is not int or token < 0 for token in row) for row in saved)):
        parser.error('--prompts-json must contain nonempty rows of nonnegative integer token IDs')
    if args.top_k == 0 or args.top_k < -1:
        parser.error('--top-k must be -1 or positive')
    context = args.max_model_len or max(args.prompt_tokens, max(map(len, saved)) if saved else 0) + args.output_tokens
    engine = LFM25Engine.from_pretrained(args.model, device=device, dtype=args.dtype, max_model_len=context)
    parked = torch.empty(int(args.park_gib * 1024 ** 3), device=device, dtype=torch.uint8) if args.park_gib else None
    if args.cuda_graphs:
        engine.enable_cuda_graphs(max_graphs=1, max_prefill_tokens=args.max_prefill_tokens,
            bf16_weight_packing=args.bf16_weight_packing, fp8_mlp=args.fp8_mlp,
            fp8_kv=args.fp8_kv, fp8_lm_head=args.fp8_lm_head, adaptive_batching=args.adaptive_batching)
    elif args.compile_decode:
        engine.compile_decode()
    sampling = LFM25SamplingConfig(max_new_tokens=args.output_tokens, temperature=args.temperature,
                                  seed=args.seed, ignore_eos=True, return_logprobs=True,
                                  top_k=0 if args.top_k == -1 else args.top_k, top_p=args.top_p,
                                  repetition_penalty=args.repetition_penalty, behavior_topk=args.behavior_topk)
    report = {'engine': 'native_lfm25', 'created_at': datetime.now(timezone.utc).isoformat(),
              'hardware': {'device': str(device), 'name': torch.cuda.get_device_name(device) if device.type == 'cuda' else 'cpu',
                           'torch_version': torch.__version__},
              'workload': {'kind': 'representative' if saved else 'synthetic',
                           'prompts_sha256': hashlib.sha256(args.prompts_json.read_bytes()).hexdigest() if saved else None,
                           'prompt_lengths': list(map(len, saved)) if saved else [args.prompt_tokens]},
              'config': {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
              'sampling': asdict(sampling), 'max_model_len': context,
              'scheduler': 'synchronous fixed batches; no hidden request concurrency cap', 'batches': []}
    traces = ['# Native throughput traces', '',
              ('Fixed-budget representative task prompts; environment answers are not scored.' if saved else
               'Fixed-budget synthetic prompts; expected answer: not applicable (throughput validation).'), '']
    sync = (lambda: torch.cuda.synchronize(device)) if device.type == 'cuda' else (lambda: None)
    for batch in args.batches:
        rows = [saved[i % len(saved)] for i in range(batch)] if saved else _prompts(tokenizer, batch, args.prompt_tokens)
        if any(len(row) + args.output_tokens > context for row in rows):
            parser.error('prompt plus output token budget exceeds --max-model-len')
        flat = flatten_token_batches(rows, device=device)
        for _ in range(args.warmups):
            engine.generate(*flat, repeats=1, sampling=sampling)
        sync()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        runs = []
        for _ in range(args.repeats):
            sync()
            start = time.perf_counter()
            output = engine.generate(*flat, repeats=1, sampling=sampling)
            tokens = output.token_ids.cpu()
            logprobs = output.logprobs.cpu()
            lengths = output.lengths.cpu()
            if output.topk_token_ids is not None:
                output.topk_token_ids.cpu()
                sparse_logprobs = output.topk_logprobs.cpu()
                if not bool(torch.isfinite(sparse_logprobs).all()):
                    raise RuntimeError('Native rollout returned non-finite sparse behavior logprobs')
            sync()
            elapsed = time.perf_counter() - start
            count = int(lengths.sum())
            if count != batch * args.output_tokens or not bool(torch.isfinite(logprobs).all()):
                raise RuntimeError('Native rollout returned an incomplete budget or non-finite logprobs')
            runs.append({'seconds': elapsed, 'output_tokens': count, 'tokens_per_second': count / elapsed})
        result = {'batch_size': batch, 'max_num_seqs': batch, 'max_inflight_requests': batch,
                  'runs': runs, 'median_tokens_per_second': statistics.median(r['tokens_per_second'] for r in runs),
                  'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if device.type == 'cuda' else None,
                  'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if device.type == 'cuda' else None}
        report['batches'].append(result)
        for row in range(min(3, batch)):
            completion = tokens[row, 0, :int(lengths[row, 0])].tolist()
            traces.extend([f'## Batch {batch}, row {row}', '', 'Prompt:', '',
                '```text', tokenizer.decode(rows[row], skip_special_tokens=False), '```', '',
                'Complete model response:', '', '```text', tokenizer.decode(completion, skip_special_tokens=False), '```', '',
                ('Expected answer: not scored (representative throughput workload).' if saved else
                 'Expected answer: not applicable (synthetic throughput workload).'),
                f'Outcome: {len(completion)} tokens; sampled log probabilities finite.', ''])
        print(json.dumps(result), flush=True)
        engine.release_inference_buffers()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + '\n')
    args.output.with_suffix('.md').write_text('\n'.join(traces) + '\n')
    del parked


if __name__ == '__main__':
    main()
