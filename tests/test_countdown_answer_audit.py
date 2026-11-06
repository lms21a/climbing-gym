from climbing_gym.countdown_answer_audit import score_countdown_answer


ROW = {"numbers": [2, 3, 4], "target": 14, "operators": "+-*/"}


def test_terminal_boxed_math_and_latex_are_recognized():
    for text in [r"Therefore, $\boxed{2+3\times 4}$;", r"\boxed{2+3*4 = 14}",
                 "<answer>2+3*4</answer>", "Final answer: 2+3*4"]:
        assert score_countdown_answer(text, ROW)["exact"]


def test_a_correct_intermediate_box_cannot_rescue_wrong_final():
    for text in [r"\boxed{2+3*4}" + "\nI cannot solve it.",
                 r"\boxed{2+3*4}" + "\n" + r"\boxed{2+3+4}",
                 r"\boxed{2+3*4 = 15}", r"\boxed{14}",
                 "<answer>2+3*4</answer> I cannot solve it."]:
        assert not score_countdown_answer(text, ROW)["exact"]


def test_optional_terminal_expression_reward_and_feedback_agree():
    from climbing_gym.config import Env
    env = Env(name="countdown", reward="terminal_expression").build()
    result = env.evaluate_response(env.build_example(ROW), r"\boxed{2+3*4}")
    assert result.feedback.success and result.reward.reward == 1
    assert not result.reward.info["strict_format_exact"]
    strict = Env(name="countdown", reward="exact_match").build()
    result = strict.evaluate_response(strict.build_example(ROW), r"\boxed{2+3*4}")
    assert not result.feedback.success and result.reward.reward == 0


def test_critic_reference_expression_does_not_change_countdown_policy_or_reward():
    from climbing_gym.config import Env
    from climbing_gym.ppo import build_privileged_critic_prompt

    row = {"numbers": [8, 4, 3, 2], "target": 12, "operators": "+-*/",
           "solution": "(8/4)*3*2", "critic_reference": "(8/4)*3*2 = 12"}
    standard = Env(name="countdown", reward="terminal_expression").build()
    privileged = Env(name="countdown", reward="terminal_expression", args={"target_key": "critic_reference"}).build()
    example = privileged.build_example(row)
    assert example.prompt == standard.build_example(row).prompt
    assert row["solution"] not in example.prompt
    assert example.target == row["critic_reference"]
    critic_prompt = build_privileged_critic_prompt(example.prompt, example.target, template="Verified solution: {answer}")
    assert "Verified solution: (8/4)*3*2 = 12" in critic_prompt
    # An alternate exact answer succeeds; the critic's witness is not the reward target string.
    assert privileged.evaluate_response(example, "<answer>8*3/(4/2)</answer>").reward.reward == 1
    assert privileged.evaluate_response(example, "<answer>8+4+3+2</answer>").reward.reward == 0
