"""Verifiable Futoshiki puzzles with fixed-substituted, non-spatial prompts.

The solver is an independent finite-domain search. Dataset generation for sizes
2-4 samples from all Latin squares, never from a teacher model.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import hashlib
import itertools
import json
import random
from typing import Mapping, Sequence


def groups(size: int) -> tuple[tuple[int, ...], ...]:
    return tuple(tuple(r*size+c for c in range(size)) for r in range(size)) + tuple(
        tuple(r*size+c for r in range(size)) for c in range(size))


def variable_name(index: int) -> str:
    # A..Z, AA..; stable labels are never values.
    result = ''
    index += 1
    while index:
        index, digit = divmod(index-1, 26)
        result = chr(65+digit)+result
    return result


@dataclass(frozen=True)
class Puzzle:
    size: int
    givens: tuple[tuple[int, int], ...]
    inequalities: tuple[tuple[int, int], ...]  # (smaller cell, larger cell)

    def __post_init__(self):
        if not 2 <= self.size <= 6:
            raise ValueError('Supported solver sizes are 2 through 6')
        if len(dict(self.givens)) != len(self.givens):
            raise ValueError('Duplicate given cells')
        if any(not 0 <= i < self.size**2 or not 1 <= v <= self.size for i,v in self.givens):
            raise ValueError('Invalid given cell or value')
        if len(set(self.inequalities)) != len(self.inequalities):
            raise ValueError('Duplicate inequalities')
        for a,b in self.inequalities:
            if not (0 <= a < self.size**2 and 0 <= b < self.size**2):
                raise ValueError('Invalid inequality cell')
            if abs(a//self.size-b//self.size)+abs(a%self.size-b%self.size) != 1:
                raise ValueError('Futoshiki inequalities must join adjacent cells')

    @classmethod
    def from_row(cls, row: Mapping) -> Puzzle:
        return cls(int(row['size']), tuple(sorted((int(i),int(v)) for i,v in dict(row['givens']).items())),
                   tuple((int(a),int(b)) for a,b in row['inequalities']))

    def accepts(self, values: Sequence[int]) -> bool:
        n = self.size
        return (len(values)==n*n and all(type(v) is int and 1 <= v <= n for v in values)
            and all(values[i]==v for i,v in self.givens)
            and all(len({values[i] for i in group})==n for group in groups(n))
            and all(values[a]<values[b] for a,b in self.inequalities))

    @property
    def identity(self) -> str:
        data = [self.size, sorted(self.givens), sorted(self.inequalities)]
        return hashlib.sha256(json.dumps(data,separators=(',',':')).encode()).hexdigest()


def solve(puzzle: Puzzle, *, limit: int | None = 2) -> list[tuple[int, ...]]:
    """Enumerate solutions with MRV search, all-different and inequality propagation.

    limit=2 suffices to reject ambiguous puzzles; None enumerates all solutions.
    No supplied solution/target is read by this solver.
    """
    if limit is not None and limit < 1:
        raise ValueError('limit must be positive or None')
    n = puzzle.size
    units = groups(n)
    domains = [set(range(1,n+1)) for _ in range(n*n)]
    for i,v in puzzle.givens:
        domains[i] = {v}
    found = []

    def propagate(ds):
        changed = True
        while changed:
            changed = False
            if any(not d for d in ds):
                return False
            for unit in units:
                singles = [next(iter(ds[i])) for i in unit if len(ds[i])==1]
                if len(singles)!=len(set(singles)):
                    return False
                for i in unit:
                    if len(ds[i])>1:
                        narrowed = ds[i]-set(singles)
                        if not narrowed:
                            return False
                        if narrowed != ds[i]:
                            ds[i]=narrowed; changed=True
                for v in range(1,n+1):
                    places = [i for i in unit if v in ds[i]]
                    if not places:
                        return False
                    if len(places)==1 and ds[places[0]]!={v}:
                        ds[places[0]]={v}; changed=True
            for a,b in puzzle.inequalities:
                da = {v for v in ds[a] if any(v<w for w in ds[b])}
                db = {w for w in ds[b] if any(v<w for v in ds[a])}
                if not da or not db:
                    return False
                if da!=ds[a] or db!=ds[b]:
                    ds[a],ds[b]=da,db; changed=True
        return True

    def visit(ds):
        if limit is not None and len(found)>=limit:
            return
        if not propagate(ds):
            return
        undecided = [i for i,d in enumerate(ds) if len(d)>1]
        if not undecided:
            values=tuple(next(iter(d)) for d in ds)
            assert puzzle.accepts(values)
            found.append(values)
            return
        i=min(undecided,key=lambda i:len(ds[i]))
        for v in sorted(ds[i]):
            child=[d.copy() for d in ds]; child[i]={v}; visit(child)
            if limit is not None and len(found)>=limit:
                break
    visit(domains)
    return found


@lru_cache(maxsize=3)
def latin_squares(size: int) -> tuple[tuple[int, ...], ...]:
    if not 2 <= size <= 4:
        raise ValueError('Exhaustive generator supports sizes 2-4')
    return tuple(solve(Puzzle(size,(),()),limit=None))


def reduced_constraints(puzzle: Puzzle) -> dict:
    fixed=dict(puzzle.givens)
    reduced=[]
    for unit in groups(puzzle.size):
        known=[fixed[i] for i in unit if i in fixed]
        if len(known)!=len(set(known)):
            raise ValueError('Contradictory givens')
        unknown=[i for i in unit if i not in fixed]
        if unknown:
            reduced.append({'variables':unknown,'values':sorted(set(range(1,puzzle.size+1))-set(known))})
    inequalities=[[{'constant':fixed[i]} if i in fixed else {'variable':i} for i in edge]
                  for edge in puzzle.inequalities]
    return {'groups':reduced,'inequalities':inequalities}


def render_prompt(puzzle: Puzzle) -> str:
    fixed=dict(puzzle.givens)
    unknown=[variable_name(i) for i in range(puzzle.size**2) if i not in fixed]
    if not unknown or not puzzle.inequalities:
        raise ValueError('Training puzzles need unknowns and inequalities')
    reduced=reduced_constraints(puzzle)
    lines=['Solve this number-constraint puzzle.', '',
        'Unknown variables: '+', '.join(unknown)+'.', '', 'Rules:',
        f'- Each unknown is an integer from 1 to {puzzle.size}.',
        '- Satisfy every remaining-value constraint below.',
        '- A variable has the same value wherever it appears.',
        '- Numbers may be reused in different groups; within a listed group each listed number appears exactly once.',
        '', 'Remaining-value constraints (fixed values have already been removed):']
    for group in reduced['groups']:
        names=', '.join(variable_name(i) for i in group['variables'])
        values=', '.join(map(str,group['values']))
        lines.append(f'- {names} must be {values}, in some order (each listed number exactly once).')
    lines += ['', 'Inequalities (fixed values have already been substituted):']
    def term(t):
        return str(t['constant']) if 'constant' in t else variable_name(t['variable'])
    lines += ['- '+term(a)+' < '+term(b) for a,b in reduced['inequalities']]
    lines += ['', 'The puzzle has exactly one solution. Reason through the constraints and check your assignment.',
        'Finish your reasoning with </thinking>, then give the unknown values inside <answer> and </answer>.',
        'Write one NAME=number assignment for each unknown, in the order listed above. Separate assignments with spaces.',
        'Only output the unknowns. Fixed values are supplied by the environment.']
    return '\n'.join(lines)


def generate(rng: random.Random, *, size: int = 4, givens_count: int = 5,
             inequality_count: int = 4, solution_pool=None, max_attempts: int = 10000) -> dict:
    """Generate a unique puzzle whose inequalities are needed for uniqueness."""
    if size == 5:
        if solution_pool is not None:
            raise ValueError('The 5x5 generator samples its own Latin squares')
        return generate_five(rng, givens_count=givens_count,
                             inequality_count=inequality_count, max_attempts=max_attempts)
    pool=latin_squares(size)
    targets=tuple(solution_pool) if solution_pool is not None else pool
    if not targets or not 0 <= givens_count < size*size or inequality_count<1:
        raise ValueError('Invalid generation settings')
    pool_set=set(pool)
    if any(tuple(s) not in pool_set for s in targets):
        raise ValueError('Invalid solution pool')
    edges=[(r*size+c,r*size+c+1) for r in range(size) for c in range(size-1)]
    edges += [(r*size+c,(r+1)*size+c) for r in range(size-1) for c in range(size)]
    for _ in range(max_attempts):
        target=rng.choice(targets)
        givens=tuple(sorted((i,target[i]) for i in rng.sample(range(size*size),givens_count)))
        base=[s for s in pool if all(s[i]==v for i,v in givens)]
        if len(base)<2:
            continue
        fixed=dict(givens)
        if any(all(i in fixed for i in unit) for unit in groups(size)):
            continue
        eligible=[edge for edge in edges if not all(i in fixed for i in edge)]
        if len(eligible)<inequality_count:
            continue
        clues=tuple((a,b) if target[a]<target[b] else (b,a) for a,b in rng.sample(eligible,inequality_count))
        remaining=[s for s in base if all(s[a]<s[b] for a,b in clues)]
        if len(remaining)!=1:
            continue
        puzzle=Puzzle(size,givens,clues)
        # Independently solve; filtering and propagation must agree.
        assert solve(puzzle)==[tuple(target)]
        return {'id':puzzle.identity, 'size':size, 'givens':dict(givens), 'inequalities':clues,
                'solution':target, 'prompt':render_prompt(puzzle),
                'solutions_without_inequalities':len(base), 'solution_count':1,
                'tier':f'{size}x{size}_g{givens_count}_i{inequality_count}'}
    raise RuntimeError('Generation exhausted; relax clue counts')


@lru_cache(maxsize=1)
def reduced_latin_five() -> tuple[tuple[int, ...], ...]:
    """Enumerate just the 5x5 squares with ordered first row and column."""
    fixed = tuple(sorted({**{i:i+1 for i in range(5)},
                          **{5*i:i+1 for i in range(5)}}.items()))
    return tuple(solve(Puzzle(5, fixed, ()), limit=None))


def generate_five(rng: random.Random, *, givens_count: int,
                  inequality_count: int, max_attempts: int = 10000) -> dict:
    """Solver-backed generation; no exhaustive full Latin-square bank.

    A reduced square, symbol permutation and permutation of rows 1..4
    uniquely parameterize every labeled 5x5 Latin square. Sampling these
    uniformly avoids limiting targets to cyclic squares. Puzzle rejection
    still changes the distribution of accepted targets.
    """
    if not 0 <= givens_count < 25 or not 1 <= inequality_count <= 40:
        raise ValueError('Invalid 5x5 clue counts')
    edges = [(5*r+c,5*r+c+1) for r in range(5) for c in range(4)]
    edges += [(5*r+c,5*(r+1)+c) for r in range(4) for c in range(5)]
    for attempt in range(1,max_attempts+1):
        reduced = rng.choice(reduced_latin_five())
        symbols = list(range(1,6)); rng.shuffle(symbols)
        rows = list(range(1,5)); rng.shuffle(rows)
        target = tuple(symbols[reduced[r*5+c]-1] for r in [0]+rows for c in range(5))
        givens = tuple(sorted((i,target[i]) for i in rng.sample(range(25),givens_count)))
        fixed = dict(givens)
        if any(all(i in fixed for i in unit) for unit in groups(5)):
            continue
        eligible = [(a,b) for a,b in edges if a not in fixed or b not in fixed]
        if len(eligible) < inequality_count:
            continue
        clues = tuple((a,b) if target[a]<target[b] else (b,a)
                      for a,b in rng.sample(eligible,inequality_count))
        puzzle = Puzzle(5,givens,clues)
        if solve(puzzle) != [target]:
            continue
        if len(solve(Puzzle(5,givens,()))) < 2:
            continue
        return {'id':puzzle.identity,'size':5,'givens':fixed,'inequalities':clues,
                'solution':target,'prompt':render_prompt(puzzle),'solution_count':1,
                'solutions_without_inequalities_lower_bound':2,
                'generation_attempts':attempt,'tier':f'5x5_g{givens_count}_i{inequality_count}'}
    raise RuntimeError('5x5 generation exhausted; relax clue counts')


def build_dataset(*, train_count=2048, validation_count=128, test_count=128, size=4, seed=20260912):
    """Target-disjoint deterministic splits; no exact puzzle overlap.

    Row/column/value symmetries may recur. This is not a claim of structural OOD.
    """
    if size == 5:
        return build_five_dataset(train_count=train_count, validation_count=validation_count,
                                  test_count=test_count, seed=seed)
    if size not in (3,4):
        raise ValueError('Target-disjoint datasets support sizes 3 through 5')
    rng=random.Random(seed)
    pool=list(latin_squares(size)); rng.shuffle(pool)
    cut1,cut2=int(.8*len(pool)),int(.9*len(pool))
    pools={'train':pool[:cut1],'validation':pool[cut1:cut2],'test':pool[cut2:]}
    splits={}; seen=set()
    settings=[(4,5),(5,4),(6,3),(7,3)] if size==4 else [(2,3),(3,2)]
    for name,count in [('train',train_count),('validation',validation_count),('test',test_count)]:
        if count<0:
            raise ValueError('Negative dataset count')
        rows=[]; attempts=0
        while len(rows)<count:
            attempts+=1
            if attempts>max(100,count*100):
                raise RuntimeError('Requested too many distinct puzzles')
            g,i=settings[len(rows)%len(settings)]
            row=generate(rng,size=size,givens_count=g,inequality_count=i,solution_pool=pools[name])
            if row['id'] in seen:
                continue
            seen.add(row['id']); rows.append(row)
        splits[name]=rows
    return splits


def build_five_dataset(*, train_count=8192, validation_count=128, test_count=128,
                       seed=20260920, validation_rows=()):
    """Distinct puzzles and disjoint target grids; preserve an optional eval panel.

    Targets are assigned to a split on first acceptance. Rejection sampling
    can affect their distribution. Symmetry-equivalent grids may recur.
    """
    if min(train_count,validation_count,test_count) < 0:
        raise ValueError('Negative dataset count')
    prefix = list(validation_rows)
    if len(prefix)>validation_count:
        raise ValueError('Validation panel exceeds requested split size')
    seen=set(); owners={}
    for row in prefix:
        puzzle=Puzzle.from_row(row)
        if puzzle.size!=5 or row['id']!=puzzle.identity or puzzle.identity in seen:
            raise ValueError('Invalid or duplicate validation puzzle')
        target=tuple(row['solution'])
        if solve(puzzle)!=[target] or len(solve(Puzzle(5,puzzle.givens,())))<2:
            raise ValueError('Validation puzzle must be unique and need inequalities')
        seen.add(puzzle.identity); owners[target]='validation'
    rng=random.Random(seed); splits={}; settings=[(6,8),(8,7),(9,5),(11,5)]
    for name,count in [('train',train_count),('validation',validation_count),('test',test_count)]:
        rows=list(prefix) if name=='validation' else []
        attempts=0
        while len(rows)<count:
            attempts+=1
            if attempts>max(100,count*100):
                raise RuntimeError('Requested too many distinct 5x5 puzzles')
            g,i=settings[len(rows)%4]
            row=generate_five(rng,givens_count=g,inequality_count=i)
            target=tuple(row['solution'])
            if row['id'] in seen or owners.get(target,name)!=name:
                continue
            seen.add(row['id']); owners[target]=name; rows.append(row)
        splits[name]=rows
    return splits
