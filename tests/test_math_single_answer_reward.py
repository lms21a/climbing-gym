import pytest

from climbing_gym.envs.math_qa import MathQAEnv, exact_match_math_reward


@pytest.mark.parametrize("response, expected", [
    ("2 + 2 = 4.\n</thinking>\n<answer>4</answer>", 1),
    ("<answer>4</answer>", 1),
    ("Reasoning. <ANSWER> 4 </ANSWER>\n<|im_end|>\n", 1),
    (r"An intermediate result is \boxed{3}. <answer>4</answer>", 1),
    ("<answer>3</answer>", 0),
    ("<answer>4</answer><answer>4</answer>", 0),
    ("<answer>3</answer><answer>4</answer>", 0),
    ("<answer>4</answer><answer>", 0),
    ("<answer>4</answer><answer", 0),
    ("<answer>4</answer></answer>", 0),
    ("<answer><answer>4</answer></answer>", 0),
    ("<answer>4</answer> More reasoning", 0),
    ("<answer>4</answer><|im_end|><answer>4</answer>", 0),
    ("<answer>4", 0),
    ("<answer></answer>", 0),
    ("<answer junk>4</answer>", 0),
    (r"\boxed{4}", 0),
    (r"\boxed{4} <answer>3</answer>", 0),
])
def test_single_answer_reward_and_eval_feedback_agree(response, expected):
    from climbing_gym.config import Env
    env = Env(name="math_qa", reward="exact_match_single_answer").build()
    example = env.build_example({"question": "2+2?", "parsed_answer": "4"})
    result = env.evaluate_response(example, response)
    assert result.reward.reward == expected
    assert result.reward.info["exact_match"] == expected
    assert result.feedback.success == bool(expected)


def test_legacy_exact_match_is_unchanged():
    env = MathQAEnv(reward_fn=exact_match_math_reward)
    example = env.build_example({"question": "2+2?", "parsed_answer": "4"})
    assert env.evaluate_response(example, "<answer>4</answer><answer>4</answer>").reward.reward == 1


def test_single_answer_reward_does_not_require_or_reward_reasoning_length():
    from climbing_gym.config import Env
    env = Env(name="math_qa", reward="exact_match_single_answer").build()
    example = env.build_example({"question": "2+2?", "parsed_answer": "4"})
    for prefix in ("", "Some reasoning.\n", "Some reasoning.\n" * 1000):
        assert env.evaluate_response(example, prefix + "<answer>4</answer>").reward.reward == 1
