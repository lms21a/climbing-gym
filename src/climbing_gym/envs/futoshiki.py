"""Dense terminal rewards with explicit balanced and accuracy-first profiles."""
from dataclasses import dataclass
import math
import re

from climbing_gym.futoshiki import Puzzle, groups, render_prompt, variable_name
from climbing_gym.envs.contracts import BaseEnv, EnvFeedback, RewardRequest, RewardResult


@dataclass(frozen=True)
class ParsedFutoshikiResponse:
    assignments: dict[str, int]
    valid: bool
    answer_blocks: int


def strict_tag_format_valid(response: str) -> bool:
    """Validate generated text after the prompt's existing <thinking> opener.

    Only whitespace may separate the reasoning close and the single terminal
    answer. The native decoder may retain one terminal im_end token.
    """
    match = re.fullmatch(
        r'(?P<reasoning>.*?)</thinking>\s*<answer>(?P<answer>.*?)</answer>'
        r'\s*(?:<\|im_end\|>)?\s*', str(response), re.S,
    )
    if match is None:
        return False
    # Reject duplicate/nested tags, misspelled closers, truncated control tags,
    # attributes, and alternate control-tag spellings in either block. Ordinary
    # puzzle inequalities (A < B, C > D) are not markup.
    unexpected = re.compile(
        r'<\s*(?:/|!|\||(?:thinking|think|answer|analysis|final)\b)'
        r'|<[A-Za-z][\w:-]*(?:\s+[^<>\n]*|/?)>', re.I,
    )
    return not any(unexpected.search(match.group(part)) for part in ('reasoning', 'answer'))


def parse_response(text: str) -> ParsedFutoshikiResponse:
    blocks=re.findall(r'<answer>\s*(.*?)\s*</answer>',str(text),re.S|re.I)
    # Multiple, nested, or truncated answers never get best-of-many credit.
    if len(blocks)!=1 or len(re.findall(r'<answer>',str(text),re.I))!=1:
        return ParsedFutoshikiResponse({},False,len(blocks))
    body=blocks[0].strip()
    pairs=re.findall(r'\b([A-Z]+)\s*=\s*(-?\d+)\b',body)
    remainder=re.sub(r'\b[A-Z]+\s*=\s*-?\d+\b','',body)
    if not pairs or remainder.strip(' \t\r\n,;') or len(pairs)!=len({name for name,_ in pairs}):
        return ParsedFutoshikiResponse({},False,len(blocks))
    try:
        assignments={name:int(value) for name,value in pairs}
    except ValueError:
        return ParsedFutoshikiResponse({},False,1)
    return ParsedFutoshikiResponse(assignments,True,1)


def score(response: str, row: dict, *, reward_profile: str = 'dense', reasoning_tokens=None,
          length_alpha=0.2, length_tau=200.0) -> dict[str, float]:
    if reward_profile not in ('dense', 'dense_accuracy', 'dense_length', 'exact_match', 'exact_match_strict'):
        raise ValueError(f'Unknown Futoshiki reward profile: {reward_profile!r}')
    puzzle=Puzzle.from_row(row)
    solution=tuple(row['solution'])
    if not puzzle.accepts(solution):
        raise ValueError('Invalid target solution')
    n=puzzle.size
    fixed=dict(puzzle.givens)
    unknown=[i for i in range(n*n) if i not in fixed]
    if not unknown or not puzzle.inequalities:
        raise ValueError('Dense reward requires unknowns and inequalities')
    names={variable_name(i):i for i in range(n*n)}
    parsed=parse_response(response)
    valid=parsed.valid and all(name in names for name in parsed.assignments)
    submitted={names[name]:value for name,value in parsed.assignments.items()} if valid else {}
    changed_givens=[i for i,value in submitted.items() if i in fixed and value!=fixed[i]]
    # Environment-owned givens cannot be edited. Correctly repeated givens are
    # tolerated for compatibility, but earn no credit. Reject contradictions.
    valid=valid and not changed_givens
    usable={i:v for i,v in submitted.items() if i in unknown and 1<=v<=n} if valid else {}
    values={**fixed,**usable}
    correct=sum(usable.get(i)==solution[i] for i in unknown)
    group_checks=[all(i in values for i in unit) and len({values[i] for i in unit})==n
                  for unit in groups(n)] if valid else [False]*(2*n)
    ineq_checks=[a in values and b in values and values[a]<values[b]
                 for a,b in puzzle.inequalities] if valid else [False]*len(puzzle.inequalities)
    u=correct/len(unknown); g=sum(group_checks)/(2*n); q=sum(ineq_checks)/len(ineq_checks)
    exact=bool(valid and len(usable)==len(unknown) and correct==len(unknown) and all(group_checks) and all(ineq_checks))
    partial=(.6*u+.3*g+.1*q if reward_profile in {'dense_accuracy','dense_length'} else (u+g+q)/3) if valid else 0.0
    info={'reward':partial+float(exact),'partial_reward':partial,'exact_match':float(exact),
        'base_reward':float(exact),'unknown_accuracy':u,'group_accuracy':g,'inequality_accuracy':q,
        'correct_unknowns':float(correct),'unknown_count':float(len(unknown)),
        'groups_satisfied':float(sum(group_checks)),'group_count':float(2*n),
        'inequalities_satisfied':float(sum(ineq_checks)),'inequality_count':float(len(ineq_checks)),
        'answer_valid':float(valid),'answer_complete':float(valid and len(usable)==len(unknown)),
        'missing_unknowns':float(len(unknown)-len(usable)), 'changed_givens':float(len(changed_givens)),
        'invalid_value_count':float(sum(not 1<=v<=n for i,v in submitted.items() if i in unknown))}
    if reward_profile=='dense_length':
        if (isinstance(reasoning_tokens,bool) or not isinstance(reasoning_tokens,int) or reasoning_tokens<0
                or not math.isfinite(length_alpha) or length_alpha<=0
                or not math.isfinite(length_tau) or length_tau<=0):
            raise ValueError('Length reward requires nonnegative integer reasoning_tokens and finite positive alpha/tau')
        penalty=length_alpha*(1-partial)*math.exp(-reasoning_tokens/length_tau)
        info.update(unshaped_reward=info['reward'],length_penalty=penalty,
                    length_reward_recovered=length_alpha*(1-partial)-penalty,
                    reasoning_tokens=float(reasoning_tokens),reward=info['reward']-penalty)
    if reward_profile == 'exact_match_strict':
        format_valid = strict_tag_format_valid(response)
        gated_exact = float(exact and format_valid)
        info.update(solution_correct=float(exact), strict_format_valid=float(format_valid),
                    exact_match=gated_exact, base_reward=gated_exact, reward=gated_exact)
    return info


def dense_futoshiki_reward(request: RewardRequest) -> RewardResult:
    info=score(request.response,request.example.row)
    return RewardResult(info['reward'],info)


def accuracy_futoshiki_reward(request: RewardRequest) -> RewardResult:
    info=score(request.response,request.example.row,reward_profile='dense_accuracy')
    return RewardResult(info['reward'],info)


def exact_futoshiki_reward(request: RewardRequest) -> RewardResult:
    info=score(request.response,request.example.row)
    return RewardResult(info['exact_match'],info)


def strict_exact_futoshiki_reward(request: RewardRequest) -> RewardResult:
    info=score(request.response,request.example.row,reward_profile='exact_match_strict')
    return RewardResult(info['reward'],info)


@dataclass(slots=True)
class FutoshikiEnv(BaseEnv):
    reward_fn: object = dense_futoshiki_reward
    task_name: str = 'futoshiki'
    prompt_key: str = 'prompt'
    target_key: str = 'solution'
    reward_profile: str = 'dense'

    def build_prompt(self,row):
        # Re-render from public constraints so stale/user-supplied prompt fields
        # cannot accidentally reveal the solution used for grading.
        return render_prompt(Puzzle.from_row(row))

    def build_target(self,row):
        solution=tuple(row[self.target_key])
        if not Puzzle.from_row(row).accepts(solution):
            raise ValueError('Invalid Futoshiki target')
        return solution

    def parse_response(self,response):
        return parse_response(response)

    def build_feedback(self,example,response,*,parsed_response=None):
        info=(self.reward_fn.grade(response,example.row) if self.reward_profile=='dense_length'
              else score(response,example.row,reward_profile=self.reward_profile))
        return EnvFeedback(done=True,success=bool(info['exact_match']),metadata=info)


def futoshiki_metrics(trajectories):
    if not trajectories:
        return {}
    keys=('unknown_accuracy','group_accuracy','inequality_accuracy','partial_reward',
          'answer_valid','answer_complete','missing_unknowns','changed_givens','exact_match')
    if all('strict_format_valid' in t.evaluation.reward.info for t in trajectories):
        keys+=('strict_format_valid','solution_correct')
    if all('length_penalty' in t.evaluation.reward.info for t in trajectories):
        keys+=('unshaped_reward','length_penalty','length_reward_recovered','reasoning_tokens',
               'reasoning_repeat4_fraction','post_reasoning_tokens')
    return {f'futoshiki/{key}':sum(float(t.evaluation.reward.info[key]) for t in trajectories)/len(trajectories)
            for key in keys}
