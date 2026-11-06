import itertools
import random

import pytest

from climbing_gym.envs.futoshiki import FutoshikiEnv, score
from climbing_gym.futoshiki import Puzzle, build_dataset, generate, groups, latin_squares, reduced_constraints, render_prompt, solve, variable_name


def example():
    return {'size':4,'givens':{0:2,5:2,11:2,12:4,13:3},
        'inequalities':[(11,7),(13,9),(8,4),(6,10)],
        'solution':[2,1,4,3,3,2,1,4,1,4,3,2,4,3,2,1]}


def answer(values):
    return '<answer>'+' '.join(f'{variable_name(i)}={v}' for i,v in values.items())+'</answer>'


def test_solver_matches_independent_latin_enumeration():
    perms=list(itertools.permutations(range(1,5)))
    independent={sum(rows,()) for rows in itertools.product(perms,repeat=4)
                 if all(len({rows[r][c] for r in range(4)})==4 for c in range(4))}
    assert len(independent)==576
    assert set(latin_squares(4))==independent
    assert len(latin_squares(3))==12
    p=Puzzle.from_row(example())
    assert solve(p)==[tuple(example()['solution'])]
    assert len(solve(Puzzle(4,p.givens,()),limit=None))==4
    assert not solve(Puzzle(3,((0,1),(1,1)),()))
    assert not solve(Puzzle(3,(),((0,1),(1,0))))
    assert len(solve(Puzzle(5,(),()),limit=1))==1


def test_generator_uniqueness_determinism_and_needed_inequalities():
    a=generate(random.Random(42)); b=generate(random.Random(42))
    assert a==b
    for seed in range(12):
        row=generate(random.Random(seed))
        p=Puzzle.from_row(row)
        assert solve(p)==[tuple(row['solution'])]
        assert len(solve(Puzzle(p.size,p.givens,())))==2
        assert all(any(i not in dict(p.givens) for i in unit) for unit in groups(p.size))


def test_split_targets_and_puzzles_are_disjoint():
    data=build_dataset(train_count=12,validation_count=8,test_count=8)
    names=list(data)
    for a,b in itertools.combinations(names,2):
        assert not {tuple(r['solution']) for r in data[a]} & {tuple(r['solution']) for r in data[b]}
        assert not {r['id'] for r in data[a]} & {r['id'] for r in data[b]}


def test_substitution_equivalence_and_prompt_has_no_target_lookup():
    row=example(); p=Puzzle.from_row(row); fixed=dict(p.givens)
    reduced=reduced_constraints(p)
    for unit,simple in zip(groups(4),reduced['groups']):
        for values in itertools.product(range(1,5),repeat=len(simple['variables'])):
            state={**fixed,**dict(zip(simple['variables'],values))}
            assert (sorted(state[i] for i in unit)==[1,2,3,4])==(sorted(values)==simple['values'])
    prompt=render_prompt(p)
    assert 'E, I must be 1, 3' in prompt and '- 3 < J' in prompt
    assert 'Only output the unknowns' in prompt
    env=FutoshikiEnv()
    assert env.build_prompt({**row,'prompt':'SOLUTION LEAK','solution':[]})==prompt


def test_reward_exact_bonus_and_partial_credit_without_format_gate():
    row=example(); fixed=row['givens']; target=row['solution']
    unknown={i:v for i,v in enumerate(target) if i not in fixed}
    perfect=score(answer(unknown),row)
    assert perfect['reward']==2 and perfect['base_reward']==1
    assert perfect['unknown_accuracy']==perfect['group_accuracy']==perfect['inequality_accuracy']==1
    assert score(answer(dict(enumerate(target))),row)['reward']==2
    assert score(answer(unknown).replace(' ', ', '),row)['reward']==2
    partial=score(answer({1:1}),row)
    assert partial['reward']==pytest.approx((1/11)/3)
    assert partial['missing_unknowns']==10 and not partial['exact_match']
    row1={1:1,2:3,3:4,4:1,6:4,7:3,8:3,9:4,10:1,14:2,15:1}
    info=score(answer(row1),row)
    assert info['correct_unknowns']==4 and info['groups_satisfied']==8 and info['inequalities_satisfied']==2
    assert info['reward']==pytest.approx((4/11+1+.5)/3)
    # Every Latin square except the unique solution scores strictly below one.
    for square in latin_squares(4):
        candidate={i:v for i,v in enumerate(square) if i not in fixed}
        info=score(answer(candidate),row)
        assert info['reward']==2 if list(square)==target else info['reward']<1


@pytest.mark.parametrize('text', ['', 'B=1', '<answer>B=1', '<answer>B=1 B=4</answer>',
    '<answer>B=1</answer><answer>B=4</answer>', '<answer>Z=1</answer>',
    '<answer>B=1 or B=4</answer>', '<answer>A=3 B=1</answer>'])
def test_ambiguous_or_changed_given_answers_do_not_earn_reward(text):
    assert score(text,example())['reward']==0


def test_out_of_range_cannot_satisfy_inequality_and_missing_givens_are_owned():
    from climbing_gym.config import Env
    assert score('<answer>J=999</answer>',example())['reward']==0
    assert score('<answer>J=4</answer>',example())['reward']==pytest.approx((1/11+1/4)/3)
    env=Env(name='futoshiki',reward='dense').build()
    row=example(); ex=env.build_example(row)
    result=env.evaluate_response(ex,'<answer>J=4</answer>')
    assert result.reward.reward>0 and result.feedback.success is False
    assert env.evaluate_response(ex,answer({i:v for i,v in enumerate(row['solution']) if i not in row['givens']})).feedback.success


def test_five_generator_varied_unique_targets_and_reward():
    from climbing_gym.futoshiki import reduced_latin_five
    reduced = reduced_latin_five()
    assert len(reduced)==56 and len(set(reduced))==56
    assert all(Puzzle(5,(),()).accepts(s) for s in reduced)
    targets=set()
    for seed,(g,i) in enumerate([(6,8),(8,7),(9,5),(11,5)]):
        row=generate(random.Random(seed),size=5,givens_count=g,inequality_count=i)
        assert row==generate(random.Random(seed),size=5,givens_count=g,inequality_count=i)
        p=Puzzle.from_row(row)
        assert len(p.givens)==g and len(p.inequalities)==i
        assert solve(p)==[tuple(row['solution'])]
        assert len(solve(Puzzle(5,p.givens,())))==2
        assert all(any(j not in row['givens'] for j in unit) for unit in groups(5))
        assert all(a not in row['givens'] or b not in row['givens'] for a,b in p.inequalities)
        assert score(answer({j:v for j,v in enumerate(row['solution']) if j not in row['givens']}),row)['reward']==2
        assert score('',row)['reward']==0
        assert 'integer from 1 to 5' in row['prompt']
        targets.add(tuple(row['solution']))
    assert len(targets)==4




def test_five_dataset_preserves_panel_and_excludes_heldout_targets():
    from climbing_gym.futoshiki import build_five_dataset
    panel=build_dataset(size=5,train_count=0,validation_count=4,test_count=0,seed=101)['validation']
    kwargs=dict(train_count=16,validation_count=8,test_count=8,seed=102,validation_rows=panel)
    data=build_five_dataset(**kwargs)
    assert data==build_five_dataset(**kwargs)
    assert data['validation'][:4]==panel
    assert len({r['id'] for rows in data.values() for r in rows})==32
    for a,b in itertools.combinations(data,2):
        assert not {tuple(r['solution']) for r in data[a]} & {tuple(r['solution']) for r in data[b]}
    for rows in data.values():
        assert {r['tier'] for r in rows}=={'5x5_g6_i8','5x5_g8_i7','5x5_g9_i5','5x5_g11_i5'}
        assert all(solve(Puzzle.from_row(r))==[tuple(r['solution'])] for r in rows)
    with pytest.raises(ValueError):
        build_five_dataset(validation_count=3,validation_rows=panel)


def test_accuracy_profile_rewards_correct_values_and_preserves_contract():
    from climbing_gym.config import Env
    row=example()
    correct_cell='<answer>B=1</answer>'
    wrong_cell_inequality='<answer>H=3</answer>'
    assert score(correct_cell,row)['reward'] < score(wrong_cell_inequality,row)['reward']
    weighted=lambda text:score(text,row,reward_profile='dense_accuracy')
    assert weighted(correct_cell)['reward']==pytest.approx(.6/11)
    assert weighted(wrong_cell_inequality)['reward']==pytest.approx(.1/4)
    assert weighted(correct_cell)['reward'] > weighted(wrong_cell_inequality)['reward']
    for text in [correct_cell,wrong_cell_inequality,'<answer>B=1 B=1</answer>','<answer>A=3</answer>']:
        old=score(text,row);new=weighted(text)
        assert all(old[k]==v for k,v in new.items() if k not in {'reward','partial_reward'})
    env=Env(name='futoshiki',reward='dense_accuracy').build()
    result=env.evaluate_response(env.build_example(row),correct_cell)
    assert result.reward.reward==weighted(correct_cell)['reward']
    assert result.feedback.metadata==result.reward.info and not result.feedback.success
    perfect=answer({i:v for i,v in enumerate(row['solution']) if i not in row['givens']})
    assert weighted(perfect)['reward']==2
    assert env.evaluate_response(env.build_example(row),perfect).feedback.success
    with pytest.raises(ValueError):score(perfect,row,reward_profile='typo')
