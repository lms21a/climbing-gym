from dataclasses import replace
import math
from types import SimpleNamespace

import pytest
import torch

from climbing_gym import constrained_projection as fixed
from climbing_gym.projection_sampling import ProjectionState, backward_completion_loss, format_audit, proposal_prompt
from climbing_gym.sampling import NativeProjectionSampler, ProjectionConfig, environment_verifier


class Tokenizer:
    eos_token_id = 999
    eos_token = '<|im_end|>'

    def encode(self, text, add_special_tokens=False):
        parts = text.split(self.eos_token)
        ids = []
        for i, part in enumerate(parts):
            if i:
                ids.append(self.eos_token_id)
            ids.extend(ord(c)+1 for c in part)
        return ids

    def decode(self, ids, skip_special_tokens=False, clean_up_tokenization_spaces=False):
        return ''.join(self.eos_token if i == 999 else chr(i-1) for i in ids)

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return 'USER: '+messages[0]['content']+'\nASSISTANT: '


class NativeStub:
    """Exercise the actual sampler's native tensor interface without a model."""
    device = torch.device('cpu')
    max_model_len = 10000

    def __init__(self, tokenizer, fixed_answer=False):
        self.tokenizer = tokenizer
        self.fixed_answer = fixed_answer
        self.generated = []
        self.scored = []

    def generate(self, input_ids, cu_seqlens, max_seqlen, repeats, sampling):
        rows = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:]):
            prompt_ids = input_ids[int(start):int(end)].tolist()
            text = self.tokenizer.decode(prompt_ids)
            prefix = text.rsplit('<thinking>\n', 1)[-1]
            # Exact continuation prefix is present once at the end of the prompt.
            trajectory = 'better\n</thinking>' if self.fixed_answer else 'proof\n</thinking>\n<answer>2</answer>'+self.tokenizer.eos_token
            tokens = self.tokenizer.encode(trajectory)[len(self.tokenizer.encode(prefix)):]
            tokens = tokens[:sampling.max_new_tokens]
            if not tokens:
                tokens = self.tokenizer.encode('</thinking>')[:sampling.max_new_tokens]
            rows.append(tokens)
            self.generated.append((prompt_ids, sampling))
        size = max(map(len, rows))
        return SimpleNamespace(token_ids=torch.tensor([r+[0]*(size-len(r)) for r in rows]),
            lengths=torch.tensor(list(map(len, rows))), logprobs=torch.full((len(rows), size), -3.))

    def score_completions(self, prompts, completions, **kwargs):
        self.scored.extend(zip(prompts, completions))
        return [tuple([-1. if 'better' in self.tokenizer.decode(c) else -2.]*len(c)) for c in completions]


def row():
    return dict(id='a', tier='easy', messages=[dict(role='user', content='1+1?')],
        reference_source='algorithmic', reference_answer='2',
        expert='<thinking>\nproof\n</thinking>\n<answer>2</answer>')


def verify(row, text):
    return '<answer>2</answer>' in text


def test_greedy_mean_complete_trajectory_strict_acceptance():
    state = ProjectionState(row(), 7)
    assert state.revise([1, 2], [-4., -4.], [-2., -2.], cut=0, initialize=True)
    assert not state.revise([1, 3], [-1.], [-2., -2.], cut=1)
    assert state.revise([1, 4, 5], [-9., -9.], [-1., -1., -1.], cut=1)
    assert state.score == -1.
    assert state.proposal_logprobs == [-4., -9., -9.]
    assert (state.attempts, state.acceptances) == (2, 1)


def test_cuts_deterministic_and_include_zero():
    state = ProjectionState(row(), 9, tokens=list(range(7)))
    assert state.cut(4, 8) == ProjectionState(row(), 9, tokens=list(range(7))).cut(4, 8)
    assert 0 in {state.cut(0, i) for i in range(50)}
    with pytest.raises(ValueError):
        ProjectionState(row(), 9).cut(0, 0)


@pytest.mark.parametrize('base,suffix', [([], [-2.]), ([-2.], []), ([math.nan], [-1.])])
def test_state_rejects_score_misalignment_and_nonfinite(base, suffix):
    with pytest.raises(ValueError):
        ProjectionState(row(), 0).revise([1], suffix, base, cut=0)


def test_proposal_exact_prefix_never_reencoded():
    tok = Tokenizer()
    prefix = [999, 37, 555]
    text, ids = proposal_prompt(tok, row(), prefix)
    assert ids[-3:] == tuple(prefix)
    assert ids == tuple(tok.encode(text)) + tuple(prefix)


def test_immutable_answer_and_eos_scored_and_roundtrips():
    tok = Tokenizer()
    candidate = fixed.compose(tok, 'proof', '2')
    assert candidate['completion_token_ids'][-1] == tok.eos_token_id
    assert candidate['reference_answer'] == '2'
    assert format_audit(candidate['completion'])['valid']
    scores = [-1.]*len(candidate['completion_token_ids'])
    scores[-1] = -5.
    fixed.add_score(candidate, scores)
    assert candidate['base_total_logprob'] == sum(scores)
    with pytest.raises(ValueError):
        fixed.add_score(candidate, scores[:-1])
    assert fixed.proposal_candidate(tok, row(), [], tok.encode('open'), 128)[0] is None


def test_native_block_driver_raw_scoring_resume_and_no_reference_in_target(tmp_path):
    tok = Tokenizer(); engine = NativeStub(tok)
    config = ProjectionConfig(max_tokens=128, blocks=1, revision_steps=2, batch_size=8)
    sampler = NativeProjectionSampler(engine, tok, config, verify=verify)
    path = tmp_path/'sampling.json'
    result = sampler.run([row()], checkpoint=path)
    assert result[0]['verified']
    assert result[0]['attempts'] == 2
    assert result[0]['acceptances'] == 0
    for prompt, completion in engine.scored:
        assert 'Expert solution' not in tok.decode(prompt)
        assert completion[-1] == tok.eos_token_id
    resumed = NativeProjectionSampler(engine, tok, config, verify=verify).run([row()], checkpoint=path)
    assert result == resumed
    with pytest.raises(ValueError, match='resume'):
        NativeProjectionSampler(engine, tok, replace(config, seed=2), verify=verify).run([row()], checkpoint=path)


def test_native_fixed_answer_driver_scores_full_candidate_and_preserves_answer(tmp_path):
    tok = Tokenizer(); engine = NativeStub(tok, fixed_answer=True)
    sampler = NativeProjectionSampler(engine, tok, ProjectionConfig(max_tokens=128, fixed_answer=True,
        revision_steps=1), verify=verify)
    records = sampler.run([row()], checkpoint=tmp_path/'fixed.json')
    assert records[0]['verified']
    assert records[0]['format']['answer'] == '2'
    assert all(ids[-1] == tok.eos_token_id for _, ids in engine.scored)
    assert records[0]['attempts'] == 1


def test_references_require_source_schema_and_independent_verification():
    tok = Tokenizer(); engine = NativeStub(tok)
    sampler = NativeProjectionSampler(engine, tok, ProjectionConfig(), verify=verify)
    with pytest.raises(ValueError, match='reference_source'):
        sampler.prepare_rows([{**row(), 'reference_source': 'stronger_teacher'}])
    with pytest.raises(ValueError, match='independent'):
        NativeProjectionSampler(engine, tok, ProjectionConfig(), verify=lambda r,t: False).prepare_rows([row()])
    with pytest.raises(ValueError, match='unique'):
        sampler.prepare_rows([row(), row()])


def test_native_context_overflow_is_explicit():
    tok = Tokenizer(); engine = NativeStub(tok)
    sampler = NativeProjectionSampler(engine, tok, ProjectionConfig(), verify=verify)
    with pytest.raises(ValueError, match='overflow'):
        sampler._generate([([1]*9999, 2)], seed=0)


def test_countdown_and_gsm8k_verifier_interfaces():
    assert environment_verifier('countdown')({'task': {'numbers': [1, 2], 'target': 3, 'operators': '+-*/'}},
                                           '<thinking>\nproof</thinking><answer>1+2</answer>')
    assert environment_verifier('gsm8k')({'task': {'answer': 'Work\n#### 2'}},
                                       '<thinking>\nproof</thinking><answer>2</answer>')


def test_preflight_initial_baseline_and_measured_comparison(tmp_path):
    tok = Tokenizer(); engine = NativeStub(tok)
    sampler = NativeProjectionSampler(engine, tok, ProjectionConfig(max_tokens=128, blocks=1), verify=verify)
    report = sampler.preflight([row()])
    assert report['comparison']['status'] == 'initial_baseline'
    assert report['logprobs'] is True
    assert report['actual_request_count'] == 1
    assert report['batch_size'] == 128
    assert report['request_concurrency_limit'] is None
    assert report['traces'][0]['response'].endswith(tok.eos_token)
    assert report['traces'][0]['expected'] == '2'
    compared = sampler.preflight([row()], prior=dict(tokens_per_second=100., peak_allocated_bytes=1000))
    assert compared['comparison']['status'] == 'compared'
    assert compared['comparison']['throughput_ratio'] > 0


def test_trace_report_preserves_control_tags_and_fences(tmp_path):
    from climbing_gym.sampling import write_traces
    path = tmp_path/'traces.md'
    write_traces(path, [dict(id='a', prompt='```\n<thinking>full', response='</thinking>\n<answer>2</answer>',
                            expected='2', outcome='verified')])
    text = path.read_text()
    assert '````text\n```\n<thinking>full\n````' in text
    assert '</thinking>\n<answer>2</answer>' in text
