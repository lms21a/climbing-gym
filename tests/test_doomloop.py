from types import SimpleNamespace as NS
import hashlib
import json
from pathlib import Path

import pytest

from climbing_gym.doomloop import arithmetic_candidates, detect_doom_loop


NUMBERS = [2, 3, 4]
ROW = {"numbers": NUMBERS, "target": 14, "operators": "+-*/", "tier": "3_multiply", "id": "x", "curriculum_step": 0}
ATTEMPT = "Let's try (2 * 3) + 4. This makes ten, not the target."
REVIEWED = json.loads((Path(__file__).parent / 'fixtures/countdown_doomloop_reviewed.json').read_text())['cases']


@pytest.mark.parametrize('case', REVIEWED, ids=lambda case: case['id'])
def test_reviewed_saved_rollout(case):
    assert hashlib.sha256(case['response'].encode()).hexdigest() == case['response_sha256']
    result = detect_doom_loop(case['response'], case['numbers'])
    assert result['detected'] == case['expected_detected'], (case['id'], result)


@pytest.mark.parametrize("separator", [" ", "\n"])
def test_repeated_arithmetic_attempts(separator):
    found = detect_doom_loop(separator.join([ATTEMPT] * 30), NUMBERS)
    assert found['detected'] and found['score'] == 1
    assert found['signals']['arithmetic_cycle'] == 1


def test_cycles_and_commuted_candidates_with_different_wording():
    attempts = ["Maybe (2 * 3) + 4? No.", "How about (3 * 2) + 4? Still wrong.",
                "What if 4 + (2 * 3)? Not the target.", "Let's try (2 + 3) * 4? Too large."]
    result = detect_doom_loop(' '.join(attempts * 10), NUMBERS)
    assert result['signals']['arithmetic_cycle'] == 1


def test_prose_without_equations_and_short_tandem_loops():
    text = "I should reconsider this approach because the answer still does not satisfy the conditions of the problem. " * 16
    assert detect_doom_loop(text)['signals']['repeated_text'] == 1
    assert detect_doom_loop('wait ' * 60)['signals']['short_cycle'] > 0
    assert detect_doom_loop('no progress yet ' * 32)['detected']


@pytest.mark.parametrize('text', [
    '', 'I will check this once.', ATTEMPT * 2,
    'First (2 * 3) + 4.\n(2 * 3) + 4 = 10.\nLet me instead use 2 + 3 * 4.\n<answer>2+3*4</answer>',
    ' '.join(f'Candidate {i}: (2 + 3) * {i}. This alternative changes the calculation.' for i in range(1, 100)),
])
def test_normal_verification_and_long_nonrepeating_search_are_not_loops(text):
    assert detect_doom_loop(text, NUMBERS)['score'] == 0


def test_repeating_a_paragraph_once_is_not_sustained_looping():
    paragraph = ' '.join(f'The distinct constraint number {i} has been checked carefully.' for i in range(20))
    assert detect_doom_loop(paragraph + '\n' + paragraph)['score'] == 0


def test_loop_cannot_be_hidden_by_correct_answer_or_new_tail():
    loop = ATTEMPT * 30
    score = detect_doom_loop(loop, NUMBERS)['score']
    assert score > 0
    for suffix in ['\n<answer>2+3*4</answer>', ' '.join(f'Novel calculation {i}.' for i in range(1000))]:
        assert detect_doom_loop(loop + suffix, NUMBERS)['score'] >= score


def test_large_pool_loop_requires_stagnation():
    pool = [f'({i} * 2) + 3' for i in range(4, 20)]
    text = ' '.join(f'Maybe {candidate}? No.' for candidate in pool * 5)
    assert detect_doom_loop(text, NUMBERS)['signals']['arithmetic_cycle'] > 0
    # A single review of all 16 alternatives is allowed.
    text = ' '.join(f'Maybe {candidate}? No.' for candidate in pool * 2)
    assert detect_doom_loop(text, NUMBERS)['score'] == 0


def test_search_keeps_exploring_a_larger_pool_without_penalty():
    pool = ['(2*3)+4', '(2+3)*4', '(4-2)*3', '4-(3-2)', '(4+2)*3', '(4*3)-2', '(3+4)*2', '(4+3)-2']
    text = ' '.join(' '.join(f'Maybe {candidate}? Not it.' for candidate in pool) +
                    f' What if (2+3)*{i}? A new proposal.' for i in range(10, 45))
    assert detect_doom_loop(text, NUMBERS)['score'] == 0


def test_unicode_arithmetic_and_associative_reordering():
    text = 'Maybe (2 × 3) + 4? No. How about 4 + (3 * 2)? Still no. ' * 20
    assert detect_doom_loop(text, NUMBERS)['signals']['arithmetic_cycle'] == 1
    candidates = arithmetic_candidates('Maybe (2+3)+4? What if 4+(3+2)?', NUMBERS)
    assert candidates[0]['canonical'] == candidates[1]['canonical']
    distinct = arithmetic_candidates('Maybe (4-3)+2? What if (4-2)+3?', NUMBERS)
    assert distinct[0]['canonical'] != distinct[1]['canonical']


def test_invalid_expression_cannot_execute_and_large_expression_is_bounded():
    text = "Maybe __import__('os').system('false')? " + 'Maybe (' * 2000
    result = detect_doom_loop(text, NUMBERS)
    assert 0 <= result['score'] <= 1


def test_shaped_reward_preserves_correctness_and_failure_ordering():
    from climbing_gym.config import Env
    env = Env(name='countdown', reward='terminal_expression_doomloop').build()
    example = env.build_example(ROW)
    clean = env.evaluate_response(example, '<answer>2+3*4</answer>')
    loop_success = env.evaluate_response(example, ATTEMPT * 30 + '\n<answer>2+3*4</answer>')
    loop_failure = env.evaluate_response(example, ATTEMPT * 30)
    ordinary_failure = env.evaluate_response(example, '<answer>2+3+4</answer>')
    assert clean.reward.reward == 1
    assert loop_success.reward.reward == .75 and loop_success.feedback.success
    assert loop_success.reward.info['exact'] and loop_success.reward.info['base_reward'] == 1
    assert loop_success.reward.info['doomloop_penalty'] == .25
    assert ordinary_failure.reward.reward == 0
    assert loop_failure.reward.reward == -.25 and not loop_failure.feedback.success
    assert clean.reward.reward > loop_success.reward.reward > ordinary_failure.reward.reward > loop_failure.reward.reward


@pytest.mark.parametrize('weight', [-.1, 1, 2, float('inf'), float('nan')])
def test_invalid_penalty_configuration_fails_early(weight):
    from climbing_gym.config import Env
    with pytest.raises(ValueError, match='doomloop_penalty_weight'):
        Env(name='countdown', reward='terminal_expression_doomloop', args={'doomloop_penalty_weight': weight}).build()


def test_zero_penalty_and_original_reward_are_unchanged():
    from climbing_gym.config import Env
    text = ATTEMPT * 30 + '\n<answer>2+3*4</answer>'
    for reward, args in [('terminal_expression', {}), ('terminal_expression_doomloop', {'doomloop_penalty_weight': 0})]:
        env = Env(name='countdown', reward=reward, args=args)
        built = env.build()
        result = built.evaluate_response(built.build_example(ROW), text)
        assert result.reward.reward == 1 and result.feedback.success
        if reward == 'terminal_expression':
            assert 'doomloop' not in result.reward.info




def test_toml_configuration_reaches_reward(tmp_path):
    from climbing_gym.config import ConfigManager
    config = tmp_path / 'reward.toml'
    config.write_text('[env]\nname="countdown"\nreward="terminal_expression_doomloop"\ndoomloop_penalty_weight=0.4\n')
    env = ConfigManager.from_toml(config).config.env.build()
    result = env.evaluate_response(env.build_example(ROW), ATTEMPT * 30)
    assert result.reward.reward == -.4
