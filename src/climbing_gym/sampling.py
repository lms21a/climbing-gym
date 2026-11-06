"""Native greedy projection sampling, with block-growing and fixed-answer modes.

The historical command called its revision count ``mcmc_steps``. Acceptance is
strict ascent in mean raw base log probability, not Metropolis-Hastings.
References must be algorithmic or verified responses from this same policy.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
import time
from typing import Callable, Sequence

import torch

from . import constrained_projection as constrained
from .inference.lfm25_engine import LFM25Engine, LFM25SamplingConfig, flatten_token_batches
from .projection_sampling import ProjectionState, dump, format_audit, proposal_prompt, read_rows, target_prompt


@dataclass(frozen=True)
class ProjectionConfig:
    max_tokens: int = 4096
    blocks: int = 8
    revision_steps: int = 4
    batch_size: int = 128
    seed: int = 0
    temperature: float = 1.0
    top_p: float = 1.0
    top_k: int = 0
    fixed_answer: bool = False
    score_prefill_tokens: int = 4096
    score_head_chunk: int = 256

    def __post_init__(self):
        if min(self.max_tokens, self.blocks, self.batch_size, self.score_prefill_tokens, self.score_head_chunk) <= 0:
            raise ValueError('Token budgets, block count, batch size and scoring chunks must be positive')
        if self.revision_steps < 0 or (not self.fixed_answer and self.max_tokens % self.blocks):
            raise ValueError('Revision count must be nonnegative; block budget must divide max_tokens')
        LFM25SamplingConfig(temperature=self.temperature, top_p=self.top_p, top_k=self.top_k)


class NativeProjectionSampler:
    """One frozen native policy for both proposal and expert-free scoring.

    Synchronous batches have one explicit batch limit and no hidden queue or
    semaphore. ``verify`` independently checks each supplied reference and
    every final candidate. It is required; answer equality alone is not a
    verifier of reference correctness. Generation never uses another model.
    """

    def __init__(self, engine: LFM25Engine, tokenizer, config: ProjectionConfig,
                 *, verify: Callable[[dict, str], bool]):
        if not callable(verify):
            raise TypeError('An independent reference verifier is required')
        self.engine, self.tokenizer, self.config, self.verify = engine, tokenizer, config, verify
        self.timings: list[dict] = []

    def _sync(self):
        if self.engine.device.type == 'cuda':
            torch.cuda.synchronize(self.engine.device)

    def prepare_rows(self, rows: Sequence[dict]) -> list[dict]:
        prepared = []
        required = {'id', 'messages', 'expert', 'reference_answer', 'reference_source'}
        for row in rows:
            if not required.issubset(row):
                raise ValueError('Projection row schema requires id, messages, expert, reference_answer and reference_source')
        if len({str(r['id']) for r in rows}) != len(rows):
            raise ValueError('Projection row IDs must be unique')
        for source in rows:
            row = dict(source)
            if row.get('reference_source') not in {'algorithmic', 'same_model_verified'}:
                raise ValueError('Projection row schema: reference_source must be algorithmic or same_model_verified')
            audit = format_audit(row['expert'].removeprefix('<thinking>\n'))
            if not audit['valid'] or audit['answer'] != row['reference_answer'] or not self.verify(row, row['expert']):
                raise ValueError(f"Reference failed independent verification: {row['id']}")
            row['prompt'] = target_prompt(self.tokenizer, row['messages'])
            row['prompt_token_ids'] = list(self.tokenizer.encode(row['prompt'], add_special_tokens=False))
            prepared.append(row)
        return prepared

    def _generate(self, requests, *, seed: int):
        # Budget buckets preserve exact prefix IDs and avoid over-generation.
        results = [None] * len(requests)
        groups: dict[int, list[int]] = {}
        for index, (ids, budget) in enumerate(requests):
            if budget <= 0 or len(ids) + budget > self.engine.max_model_len:
                raise ValueError('Proposal context/budget overflow; refusing silent truncation')
            groups.setdefault(budget, []).append(index)
        for budget, indices in sorted(groups.items()):
            for offset in range(0, len(indices), self.config.batch_size):
                selected = indices[offset:offset+self.config.batch_size]
                flat = flatten_token_batches([requests[i][0] for i in selected], device=self.engine.device)
                sampling = LFM25SamplingConfig(max_new_tokens=budget, temperature=self.config.temperature,
                    top_p=self.config.top_p, top_k=self.config.top_k, seed=seed+offset,
                    return_logprobs=True, stop_token_sequences=(tuple(self.tokenizer.encode('</thinking>', add_special_tokens=False)),)
                    if self.config.fixed_answer else ())
                with torch.inference_mode():
                    out = self.engine.generate(*flat, repeats=1, sampling=sampling)
                if out.logprobs is None:
                    raise RuntimeError('Native projection requires sampled-token log probabilities')
                for j, index in enumerate(selected):
                    n = int(out.lengths[j])
                    tokens = out.token_ids[j, :n].tolist()
                    logprobs = out.logprobs[j, :n].tolist()
                    if not all(math.isfinite(x) for x in logprobs):
                        raise ValueError('Nonfinite proposal likelihood')
                    results[index] = (tokens, logprobs)
        return results

    def _score(self, rows, candidates):
        with torch.inference_mode():
            return self.engine.score_completions([r['prompt_token_ids'] for r in rows], candidates,
                max_prefill_tokens=self.config.score_prefill_tokens, head_chunk_size=self.config.score_head_chunk)

    def _record(self, started, generated_tokens, generation_seconds, scoring_seconds, **extra):
        self._sync()
        metrics = dict(generated_tokens=generated_tokens, generation_seconds=generation_seconds,
            scoring_seconds=scoring_seconds, full_step_seconds=time.perf_counter()-started,
            tokens_per_second=generated_tokens/max(generation_seconds, 1e-9), batch_size=self.config.batch_size,
            request_concurrency_limit=None, peak_allocated_bytes=torch.cuda.max_memory_allocated(self.engine.device)
            if self.engine.device.type == 'cuda' else 0, **extra)
        self.timings.append(metrics)
        return metrics

    def preflight(self, rows, *, prior: dict | None = None) -> dict:
        """Representative full-budget native rollout plus raw rescoring.

        Caller reviews the comparison; no timing guard runs during sampling or
        training. An available prior contains tokens_per_second and peak_allocated_bytes;
        without one this records an explicitly labeled initial baseline.
        """
        if not rows:
            raise ValueError('Representative rows are required')
        if prior is not None and min(float(prior['tokens_per_second']), float(prior['peak_allocated_bytes'])) <= 0:
            raise ValueError('Prior benchmark requires positive tokens_per_second and peak_allocated_bytes')
        rows = self.prepare_rows(rows[:self.config.batch_size])
        requests = []
        for row in rows:
            _, ids = (constrained.proposal_prompt if self.config.fixed_answer else proposal_prompt)(self.tokenizer, row, [])
            budget = constrained.reasoning_budget(self.tokenizer, row['reference_answer'], self.config.max_tokens) if self.config.fixed_answer else self.config.max_tokens
            requests.append((ids, budget))
        # Warm up the same budget and logprob path before timing.
        self._generate(requests, seed=self.config.seed)
        if self.engine.device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(self.engine.device)
        self._sync(); started = time.perf_counter()
        outputs = self._generate(requests, seed=self.config.seed)
        self._sync(); generation_seconds = time.perf_counter()-started
        score_started = time.perf_counter()
        nonempty = []
        for row, (tokens, _) in zip(rows, outputs):
            if self.config.fixed_answer:
                candidate, _ = constrained.proposal_candidate(self.tokenizer, row, [], tokens, self.config.max_tokens)
                if candidate:
                    nonempty.append((row, candidate['completion_token_ids']))
            elif tokens:
                nonempty.append((row, tokens))
        if nonempty:
            self._score([r for r, _ in nonempty], [t for _, t in nonempty])
        self._sync()
        report = self._record(started, sum(len(t) for t, _ in outputs), generation_seconds,
            time.perf_counter()-score_started, token_budget=self.config.max_tokens, logprobs=True,
            warmups=1, actual_request_count=len(rows), comparison=dict(status='initial_baseline'))
        if prior is not None:
            report['comparison'] = dict(status='compared',
                throughput_ratio=report['tokens_per_second']/prior['tokens_per_second'],
                peak_memory_ratio=report['peak_allocated_bytes']/prior['peak_allocated_bytes'])
        report['traces'] = [dict(id=r['id'], prompt=self.tokenizer.decode(request[0], skip_special_tokens=False),
            response=self.tokenizer.decode(tokens, skip_special_tokens=False), expected=r['reference_answer'],
            outcome='preflight proposal; '+str(bool(self.verify(r, self.tokenizer.decode(tokens, skip_special_tokens=False)))))
            for r, request, (tokens, _) in zip(rows, requests, outputs)]
        return report

    def run(self, rows: Sequence[dict], *, checkpoint: Path | None = None,
            on_metrics: Callable[[dict], None] | None = None) -> list[dict]:
        rows = self.prepare_rows(rows)
        signature = hashlib.sha256(json.dumps(dict(rows=rows, config=asdict(self.config)), sort_keys=True).encode()).hexdigest()
        saved = json.loads(checkpoint.read_text()) if checkpoint and checkpoint.exists() else None
        if saved and saved['signature'] != signature:
            raise ValueError('Projection resume data/config mismatch')
        states = [ProjectionState(**s) for s in saved['states']] if saved else [ProjectionState(r, self.config.seed) for r in rows]
        candidates = saved.get('candidates', [None]*len(rows)) if saved else [None]*len(rows)
        first = saved['next_block'] if saved else 0
        count = self.config.revision_steps+1 if self.config.fixed_answer else self.config.blocks
        for block in range(first, count):
            started = time.perf_counter()
            if self.config.fixed_answer:
                # Step zero starts from a verified algorithmic/same-policy reference.
                if block == 0:
                    candidates = [constrained.expert_completion(self.tokenizer, r) for r in rows]
                    score_start = time.perf_counter()
                    scores = self._score(rows, [c['completion_token_ids'] for c in candidates])
                    self._sync()
                    initial_score_seconds = time.perf_counter()-score_start
                    for candidate, logps in zip(candidates, scores):
                        constrained.add_score(candidate, logps)
                    for state, c in zip(states, candidates):
                        state.tokens, state.score = list(c['completion_token_ids']), c['base_mean_logprob']
                    steps = []
                else:
                    steps = [block]
            else:
                steps = range(-1, self.config.revision_steps)
            generated, generation_seconds, scoring_seconds = 0, 0., (initial_score_seconds if self.config.fixed_answer and block == 0 else 0.)
            active = [i for i, state in enumerate(states) if self.config.fixed_answer or self.tokenizer.eos_token_id not in state.tokens]
            for iteration in steps:
                requests, cuts = [], []
                for i in active:
                    state = states[i]
                    if self.config.fixed_answer:
                        budget = constrained.reasoning_budget(self.tokenizer, rows[i]['reference_answer'], self.config.max_tokens)
                        reasoning = candidates[i]['reasoning_token_ids']
                        cut = constrained.random_cut(self.config.seed, rows[i]['id'], block, len(reasoning), budget)
                        prefix = reasoning[:cut]
                        _, ids = constrained.proposal_prompt(self.tokenizer, rows[i], prefix)
                        budget -= cut
                    else:
                        cut = len(state.tokens) if iteration == -1 else state.cut(block, iteration)
                        prefix = state.tokens[:cut]
                        budget = self.config.max_tokens//self.config.blocks if iteration == -1 else len(state.tokens)-cut
                        _, ids = proposal_prompt(self.tokenizer, rows[i], prefix)
                    requests.append((ids, budget)); cuts.append((cut, prefix))
                self._sync(); before = time.perf_counter()
                outputs = self._generate(requests, seed=self.config.seed+10000+block*100+iteration+1)
                self._sync(); generation_seconds += time.perf_counter()-before
                generated += sum(len(t) for t, _ in outputs)
                pending = []
                for i, (cut, prefix), (tokens, logps) in zip(active, cuts, outputs):
                    if self.config.fixed_answer:
                        candidate, error = constrained.proposal_candidate(self.tokenizer, rows[i], prefix, tokens, self.config.max_tokens)
                        if error:
                            states[i].attempts += 1
                            continue
                        ids = candidate['completion_token_ids']
                    else:
                        candidate, ids = None, prefix+tokens
                    if ids:
                        pending.append((i, cut, logps, candidate, ids))
                before = time.perf_counter()
                scores = self._score([rows[i] for i, *_ in pending], [p[-1] for p in pending]) if pending else []
                self._sync(); scoring_seconds += time.perf_counter()-before
                for (i, cut, logps, candidate, ids), score in zip(pending, scores):
                    state = states[i]
                    if self.config.fixed_answer:
                        constrained.add_score(candidate, score)
                        state.attempts += 1
                        if candidate['base_mean_logprob'] > state.score:
                            state.tokens, state.score = list(ids), candidate['base_mean_logprob']
                            state.acceptances += 1
                            candidates[i] = candidate
                    else:
                        state.revise(ids, logps, list(score), cut=cut, initialize=iteration == -1)
                for i in active:
                    states[i].blocks = block+1
            metrics = self._record(started, generated, generation_seconds, scoring_seconds, block=block)
            if on_metrics:
                on_metrics(metrics)
            if checkpoint:
                # JSON never writes infinities, including an uninitialized empty state.
                state_dicts = [dict(asdict(s), score=s.score if math.isfinite(s.score) else None) for s in states]
                if any(s['score'] is None for s in state_dicts):
                    raise RuntimeError('Native generation produced an empty trajectory')
                dump(checkpoint, dict(signature=signature, next_block=block+1, states=state_dicts, candidates=candidates))
        records = []
        for row, state in zip(rows, states):
            response = self.tokenizer.decode(state.tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False)
            audit = format_audit(response)
            records.append(dict(id=row['id'], tier=row.get('tier'), prompt=row['prompt'],
                prompt_token_ids=row['prompt_token_ids'], completion=response, completion_token_ids=state.tokens,
                expected_answer=row['reference_answer'], verified=bool(audit['valid'] and self.verify(row, response)),
                format=audit, base_mean_logprob=state.score, attempts=state.attempts, acceptances=state.acceptances))
        return records


def environment_verifier(environment: str):
    """Use the package's independent task checker; row['task'] is the task row."""
    if environment == 'knights_knaves':
        from .knights_knaves import score
        return lambda row, text: bool(score(text, row['task'])['exact_match'])
    if environment == 'futoshiki':
        from .envs.futoshiki import score
        return lambda row, text: bool(score(text, row['task'])['exact_match'])
    if environment == 'countdown':
        from .countdown import score_response
        return lambda row, text: bool(score_response(text, row['task'])['exact'])
    if environment == 'gsm8k':
        from .envs.math_qa import normalize_math_answer
        return lambda row, text: bool(format_audit(text.removeprefix('<thinking>\n'))['valid'] and
            normalize_math_answer(format_audit(text.removeprefix('<thinking>\n'))['answer']) ==
            normalize_math_answer(row['task']['answer'].split('####')[-1].strip()))
    raise ValueError(f'Unsupported environment: {environment}')


def write_traces(path: Path, records: Sequence[dict]):
    import re
    def literal(value):
        value = str(value)
        fence = '`' * max(3, 1 + max((len(m.group()) for m in re.finditer(r'`+', value)), default=0))
        return fence+'text\n'+value+'\n'+fence
    lines = ['# Native projection traces', '']
    for row in records:
        lines += [f"## {row['id']} ({row.get('tier', 'unspecified')})", '', 'Prompt:', '', literal(row['prompt']), '',
            'Complete model response:', '', literal(row['response'] if 'response' in row else row['completion']), '',
            'Expected answer:', '', literal(row.get('expected', row.get('expected_answer', ''))), '',
            'Outcome: '+str(row.get('outcome', row.get('verified'))), '']
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text('\n'.join(lines), encoding='utf-8')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--rows', type=Path, required=True)
    parser.add_argument('--environment', choices=['gsm8k', 'countdown', 'knights_knaves', 'futoshiki'], required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--max-tokens', type=int, default=4096)
    parser.add_argument('--blocks', type=int, default=8)
    parser.add_argument('--revision-steps', type=int, default=4)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--fixed-answer', action='store_true')
    parser.add_argument('--max-model-len', type=int, default=12288)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    parser.add_argument('--cuda-graphs', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--prior-benchmark', type=Path)
    args = parser.parse_args(argv)
    from .models.lfm2_compat import load_lfm2_tokenizer
    config = ProjectionConfig(max_tokens=args.max_tokens, blocks=args.blocks, revision_steps=args.revision_steps,
        batch_size=args.batch_size, seed=args.seed, fixed_answer=args.fixed_answer)
    approval = args.output/'preflight.json'
    if not args.preflight and not approval.exists():
        parser.error('Run --preflight with --prior-benchmark and review the measured comparison first')
    engine = LFM25Engine.from_pretrained(args.model, device=args.device, max_model_len=args.max_model_len)
    if args.cuda_graphs:
        engine.enable_cuda_graphs(max_graphs=1, max_prefill_tokens=config.score_prefill_tokens)
    sampler = NativeProjectionSampler(engine, load_lfm2_tokenizer(args.model), config,
        verify=environment_verifier(args.environment))
    rows = read_rows(args.rows)
    identity = dict(config=asdict(config), model=str(Path(args.model).resolve()), rows_sha256=hashlib.sha256(args.rows.read_bytes()).hexdigest(),
                    max_model_len=args.max_model_len, device=args.device, cuda_graphs=args.cuda_graphs)
    if args.preflight:
        report = sampler.preflight(rows, prior=json.loads(args.prior_benchmark.read_text()) if args.prior_benchmark else None)
        dump(approval, dict(identity=identity, report=report))
        write_traces(args.output/'preflight_traces.md', report['traces'])
    else:
        if json.loads(approval.read_text())['identity'] != identity:
            parser.error('Preflight model, rows or execution configuration differs')
        def metrics(values):
            with (args.output/'timings.jsonl').open('a') as stream:
                stream.write(json.dumps(values)+'\n')
            print(json.dumps(values), flush=True)
        records = sampler.run(rows, checkpoint=args.output/'sampling_state.json', on_metrics=metrics)
        dump(args.output/'projected_all.json', records)
        with (args.output/'projected_train.jsonl').open('w') as stream:
            for record in records:
                if record['verified']:
                    stream.write(json.dumps(record)+'\n')
        write_traces(args.output/'sampling_traces.md', records)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
