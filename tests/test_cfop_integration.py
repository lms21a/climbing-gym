"""Real Rust verifier checks plus standalone configuration/rollout integration."""
from pathlib import Path
from types import SimpleNamespace
import builtins
import json

import pytest

from climbing_gym.config import Env
from climbing_gym.envs.cfop_cube import CFOPCompatibilityError, CFOPCubeEnv, cfop_dense_v2_reward, _validate_cfop_solver_module
from climbing_gym.envs.cfop_support import parse_cfop_response

ROOT = Path(__file__).parents[1]


def test_optional_binding_absence_has_actionable_error(monkeypatch):
    original = builtins.__import__
    def without_cfop(name, *args, **kwargs):
        if name == 'cfop_solver':
            raise ImportError('deliberately unavailable optional binding')
        return original(name, *args, **kwargs)
    monkeypatch.setattr(builtins, '__import__', without_cfop)
    # Other environments remain constructible without the optional extension.
    assert Env(name='countdown').build().task_name == 'countdown'
    with pytest.raises(RuntimeError, match=r'cfop-solver==0.4.0.*docs/cfop.md'):
        Env(name='cfop_cube', reward='cfop_dense_v2').build()


@pytest.mark.parametrize('change', [dict(__version__='0.3.0'), dict(STATE_FORMAT='other'), dict(VERIFICATION_SCHEMA='other')])
def test_incompatible_binding_is_rejected(change):
    compatible = dict(Cube=object, SOLVED_STATE='state', STATE_FORMAT='facelets-urfdlb-v1',
                      TrainingVerifier=object, VERIFICATION_SCHEMA='cfop-verification/v1', __version__='0.4.0')
    with pytest.raises(CFOPCompatibilityError):
        _validate_cfop_solver_module(SimpleNamespace(**(compatible | change)))


@pytest.mark.parametrize('response', ['CROSS:\nOLL:\nPLL:\nEND\n\n', 'CROSS:\r\nOLL:\nPLL:\nEND',
                                      'CROSS:\nF2L FR:\nOLL:\nPLL:\nEND', 'CROSS: U\tR\nOLL:\nPLL:\nEND'])
def test_strict_grammar_rejects_ambiguous_whitespace_and_empty_f2l(response):
    assert not parse_cfop_response(response).valid


@pytest.fixture(scope='module')
def rust():
    return pytest.importorskip('cfop_solver')


@pytest.fixture
def env(rust):
    return Env(name='cfop_cube', reward='cfop_dense_v2', args={'verifier_threads': 1}).build()


@pytest.fixture
def row():
    return json.loads((ROOT / 'examples/data/cfop_smoke.jsonl').read_text().splitlines()[0])


def test_actual_rust_verifier_accepts_bundled_programs_and_ignores_poisoned_prompt(env, rust):
    rows=[json.loads(line) for line in (ROOT / 'examples/data/cfop_smoke.jsonl').read_text().splitlines()]
    for row in rows:
        example=env.build_example(dict(row, prompt='ANSWER LEAK', completion='wrong cached target'))
        result=env.evaluate_response(example, row['completion'])
        assert 'ANSWER LEAK' not in example.prompt
        assert result.feedback.success is True
        assert result.reward.reward == pytest.approx(1.8)
        assert rust.verify_solution(row['state'], result.parsed_response.moves)


def test_solving_cube_in_cross_does_not_pass_strict_phase_fidelity(env, row):
    parsed=parse_cfop_response(row['completion'])
    hidden=f'CROSS: {parsed.moves}\nOLL:\nPLL:\nEND'
    result=env.evaluate_response(env.build_example(row), hidden)
    assert result.feedback.success is False
    assert result.reward.info['solved'] == 1
    assert result.reward.info['strict_success'] == 0
    assert result.reward.reward < 1


def test_solve_then_disturb_is_rejected(env, row):
    poisoned=row['completion'].replace('\nEND', " U U'\nEND")
    result=env.evaluate_response(env.build_example(row), poisoned)
    assert result.feedback.success is False
    assert result.reward.info['strict_success'] == 0


def test_auxiliary_sft_rows_are_not_full_environment_episodes(env, row):
    for change in [{'sft_only': True}, {'environment_eligible': False}]:
        with pytest.raises(ValueError):
            env.build_example(dict(row, **change))


def test_invalid_cube_state_fails_before_rollout(env):
    from climbing_gym.envs.cfop_support import FaceletStateError
    with pytest.raises(FaceletStateError):
        env.build_example({'state': 'U'*54, 'completion': ''})


def test_only_supported_stage_local_protocol_is_registered(rust):
    assert Env(name='cfop_phase',reward='cfop_phase_v1').build().requested_stage == 'cross'
    with pytest.raises(ValueError, match='requested_stage'):
        Env(name='cfop_phase',reward='cfop_phase_v1',args={'requested_stage':'oll'}).build()
    with pytest.raises(ValueError, match='requires reward'):
        Env(name='cfop_cube',reward='exact_match').build()


def test_dense_cfop_cannot_enter_binary_value_objective(monkeypatch):
    from dataclasses import replace
    from climbing_gym.config import ConfigManager
    from climbing_gym.train import validate_config
    monkeypatch.setenv('CLIMBING_GYM_MODEL', '/tmp/user-supplied-model')
    config=ConfigManager.from_toml(ROOT/'configs/train_cfop_cube.toml').config
    config=replace(config, trainer=replace(config.trainer, algorithm='value_mis_sc'))
    with pytest.raises(ValueError, match='binary.*CFOP'):
        validate_config(config)
