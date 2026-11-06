from __future__ import annotations

import pytest

from climbing_gym.envs import (
    BaseEnv,
    MathQAEnv,
    exact_match_math_bounded_summary_2_turns_reward,
    exact_match_math_hybrid_2_turns_reward,
    exact_match_math_min_2_turns_reward,
    RewardRequest,
    RewardResult,
    exact_match_math_reward,
    exact_match_math_summary_bootstrap_2_turns_reward,
    exact_match_math_soft_target_2_turns_reward,
    exact_match_math_underturn_penalty_2_turns_reward,
    exact_match_with_format_math_reward,
)


class _MinimalEchoEnv(BaseEnv):
    def __init__(self) -> None:
        self.reward_fn = lambda request: 1.0 if request.parsed_response == request.example.target else 0.0




def test_base_env_can_stream_loaded_examples() -> None:
    env = _MinimalEchoEnv()
    rows = [
        {"prompt": "say hi", "target": "hi"},
        {"prompt": "say bye", "target": "bye"},
    ]

    env.load_rows(rows)

    first = env.next_example()
    second = env.next_example()

    assert first == env.build_example(rows[0])
    assert second == env.build_example(rows[1])
    assert env.next_example() is None


def test_base_env_can_batch_dataset_rows_across_epochs() -> None:
    env = _MinimalEchoEnv()
    rows = [
        {"prompt": "say hi", "target": "hi"},
        {"prompt": "say bye", "target": "bye"},
        {"prompt": "say later", "target": "later"},
    ]

    env.configure_dataset(rows, seed=11, shuffle=False)

    first = env.next_batch(2)
    second = env.next_batch(2)
    third = env.next_batch(2)

    assert first.indices == (0, 1)
    assert [example.target for example in first.examples] == ["hi", "bye"]
    assert second.indices == (2,)
    assert [example.target for example in second.examples] == ["later"]
    assert third.indices == (0, 1)
    assert [example.target for example in third.examples] == ["hi", "bye"]


def test_base_env_requires_rows_to_be_loaded_before_iteration() -> None:
    env = _MinimalEchoEnv()

    with pytest.raises(RuntimeError, match="before rows were loaded"):
        env.next_example()


def test_base_env_does_not_require_feedback() -> None:
    env = _MinimalEchoEnv()
    example = env.build_example({"prompt": "say hi", "target": "hi"})

    evaluation = env.evaluate_response(example, "hi")

    assert evaluation.parsed_response == "hi"
    assert evaluation.feedback is None
    assert evaluation.reward.reward == 1.0


def test_math_qa_env_extracts_boxed_answer_and_scores_exact_match() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    evaluation = env.evaluate_response(
        example,
        "We add the numbers and get the result.\n\\boxed{4}",
    )

    assert "\\boxed" in example.prompt
    assert evaluation.parsed_response.final_answer == "4"
    assert evaluation.parsed_response.has_boxed_answer is True
    assert evaluation.feedback.success is True
    assert evaluation.reward.reward == 1.0


def test_math_qa_env_supports_canonical_gsm8k_answer_rows() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_reward, target_key="answer")
    example = env.build_example(
        {
            "question": "Natalia sold 48 clips, then 24 more. How many total?",
            "answer": (
                "Natalia sold 48 + 24 = <<48+24=72>>72 clips altogether.\n"
                "#### 72"
            ),
        }
    )

    evaluation = env.evaluate_response(example, "</thinking>\n\\boxed{72}")

    assert example.target == "72"
    assert evaluation.feedback.success is True
    assert evaluation.reward.reward == 1.0


def test_math_qa_strict_thinking_answer_format_reward_counts_prefilled_open_tag() -> None:
    env = MathQAEnv(
        reward_fn=lambda request: exact_match_with_format_math_reward(
            request,
            format_weight=0.1,
        ),
        require_thinking_answer_format=True,
    )
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    formatted_correct = env.evaluate_response(
        example,
        "I add two and two.</thinking><answer>4</answer><|im_end|>",
    )
    malformed_correct = env.evaluate_response(example, "\\boxed{4}")
    formatted_wrong = env.evaluate_response(
        example,
        "I made a mistake.</thinking><answer>5</answer>",
    )
    repeated_answer = env.evaluate_response(
        example,
        "Reasoning.</thinking><answer>4</answer><answer>4</answer>",
    )

    assert "prefilled with an opening <thinking> tag" in example.prompt
    assert "<answer>final answer</answer>" in example.prompt
    assert formatted_correct.reward.reward == pytest.approx(1.1)
    assert formatted_correct.reward.info["reward_format"] == pytest.approx(1.0)
    assert malformed_correct.reward.reward == pytest.approx(1.0)
    assert malformed_correct.reward.info["reward_format"] == pytest.approx(0.0)
    assert formatted_wrong.reward.reward == pytest.approx(0.1)
    assert repeated_answer.reward.info["answer_tag_count"] == pytest.approx(2.0)
    assert repeated_answer.reward.info["reward_format"] == pytest.approx(0.0)


def test_math_qa_env_supports_answer_tag_fallback_and_float_rewards() -> None:
    captured_requests: list[RewardRequest] = []

    def reward_fn(request: RewardRequest) -> float:
        captured_requests.append(request)
        return 0.25

    env = MathQAEnv(reward_fn=reward_fn)
    example = env.build_example({"question": "What is one half?", "parsed_answer": r"\frac{1}{2}"})

    evaluation = env.evaluate_response(
        example,
        "Final answer: <answer>\\frac{1}{2}</answer>",
    )

    assert evaluation.parsed_response.final_answer == r"\frac{1}{2}"
    assert evaluation.parsed_response.has_answer_tag is True
    assert evaluation.reward.reward == 0.25
    assert captured_requests[0].env_name == "math_qa"
    assert captured_requests[0].feedback.success is True
    assert captured_requests[0].turn_count == 1


def test_math_qa_soft_target_2_turns_reward_peaks_at_two_turns() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_soft_target_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    one_turn = env.evaluate_response(example, "\\boxed{4}", turn_count=1)
    two_turn = env.evaluate_response(example, "\\boxed{4}", turn_count=2)
    three_turn = env.evaluate_response(example, "\\boxed{4}", turn_count=3)
    four_turn = env.evaluate_response(example, "\\boxed{4}", turn_count=4)

    assert one_turn.reward.reward == pytest.approx(1.0)
    assert two_turn.reward.reward == pytest.approx(1.1)
    assert three_turn.reward.reward == pytest.approx(1.0)
    assert four_turn.reward.reward == pytest.approx(0.9)
    assert two_turn.reward.info["turn_count"] == pytest.approx(2.0)
    assert two_turn.reward.info["turn_target_alignment"] == pytest.approx(1.0)


def test_math_qa_soft_target_2_turns_reward_gives_no_bonus_for_incorrect_answers() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_soft_target_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    one_turn = env.evaluate_response(example, "\\boxed{5}", turn_count=1)
    two_turn = env.evaluate_response(example, "\\boxed{5}", turn_count=2)
    four_turn = env.evaluate_response(example, "\\boxed{5}", turn_count=4)

    assert one_turn.reward.reward == pytest.approx(0.0)
    assert two_turn.reward.reward == pytest.approx(0.0)
    assert four_turn.reward.reward == pytest.approx(0.0)
    assert two_turn.reward.info["exact_match"] == pytest.approx(0.0)
    assert two_turn.reward.info["turn_bonus"] == pytest.approx(0.0)
    assert two_turn.reward.info["turn_target_alignment"] == pytest.approx(1.0)


def test_math_qa_summary_bootstrap_reward_prefers_summary_then_finalize() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_summary_bootstrap_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    single_turn = env.evaluate_response(example, "\\boxed{4}", turn_count=1)
    two_turn = env.evaluate_response(
        example,
        "\\boxed{4}",
        turn_count=2,
        turn_metadata=(
            {"summary": "Reduce the problem to 2 + 2 = 4."},
            {"summary": ""},
        ),
    )

    assert single_turn.reward.reward == pytest.approx(0.8)
    assert two_turn.reward.reward == pytest.approx(1.4)
    assert two_turn.reward.info["perfect_summary_chain"] == pytest.approx(1.0)
    assert two_turn.reward.info["intermediate_summary_coverage"] == pytest.approx(1.0)
    assert two_turn.reward.info["shaped_structure_reward"] == pytest.approx(0.4)


def test_math_qa_summary_bootstrap_reward_penalizes_terminal_summary_and_wrong_answers() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_summary_bootstrap_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    terminal_summary = env.evaluate_response(
        example,
        "\\boxed{4}",
        turn_count=2,
        turn_metadata=(
            {"summary": "We still need to answer."},
            {"summary": "This should have been omitted on the final turn."},
        ),
    )
    structured_but_wrong = env.evaluate_response(
        example,
        "\\boxed{5}",
        turn_count=2,
        turn_metadata=(
            {"summary": "Track the arithmetic state."},
            {"summary": ""},
        ),
    )

    assert terminal_summary.reward.reward == pytest.approx(1.14)
    assert terminal_summary.reward.info["final_turn_has_summary"] == pytest.approx(1.0)
    assert structured_but_wrong.reward.reward == pytest.approx(0.1)
    assert structured_but_wrong.reward.info["exact_match"] == pytest.approx(0.0)
    assert structured_but_wrong.reward.info["shaped_structure_reward"] == pytest.approx(0.1)


def test_math_qa_min_2_turns_reward_requires_correct_answer_and_two_turns() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_min_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    one_turn_correct = env.evaluate_response(example, "\\boxed{4}", turn_count=1)
    two_turn_correct = env.evaluate_response(example, "\\boxed{4}", turn_count=2)
    three_turn_wrong = env.evaluate_response(example, "\\boxed{5}", turn_count=3)

    assert one_turn_correct.reward.reward == pytest.approx(0.0)
    assert one_turn_correct.reward.info["eligible_turn_count"] == pytest.approx(0.0)
    assert two_turn_correct.reward.reward == pytest.approx(1.0)
    assert two_turn_correct.reward.info["eligible_turn_count"] == pytest.approx(1.0)
    assert three_turn_wrong.reward.reward == pytest.approx(0.0)
    assert three_turn_wrong.reward.info["exact_match"] == pytest.approx(0.0)


def test_math_qa_underturn_penalty_reward_penalizes_finishing_too_early() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_underturn_penalty_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    one_turn_correct = env.evaluate_response(example, "\\boxed{4}", turn_count=1)
    one_turn_wrong = env.evaluate_response(example, "\\boxed{5}", turn_count=1)
    two_turn_correct = env.evaluate_response(example, "\\boxed{4}", turn_count=2)
    three_turn_wrong = env.evaluate_response(example, "\\boxed{5}", turn_count=3)

    assert one_turn_correct.reward.reward == pytest.approx(-1.0)
    assert one_turn_correct.reward.info["under_turn_gap"] == pytest.approx(1.0)
    assert one_turn_correct.reward.info["under_turn_penalty"] == pytest.approx(1.0)
    assert one_turn_wrong.reward.reward == pytest.approx(-1.0)
    assert two_turn_correct.reward.reward == pytest.approx(1.0)
    assert two_turn_correct.reward.info["eligible_turn_count"] == pytest.approx(1.0)
    assert three_turn_wrong.reward.reward == pytest.approx(0.0)
    assert three_turn_wrong.reward.info["eligible_turn_count"] == pytest.approx(1.0)


def test_math_qa_hybrid_2_turns_reward_requires_correct_answer_and_strict_format() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_hybrid_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})
    metadata = (
        {"summary": "2 + 2 = 4.", "strict_intermediate_format": True},
        {"summary": "", "strict_final_format": True},
    )
    correct = env.evaluate_response(example, "\\boxed{4}", turn_count=2, turn_metadata=metadata)
    wrong = env.evaluate_response(example, "\\boxed{5}", turn_count=2, turn_metadata=metadata)
    missing_format = env.evaluate_response(example, "\\boxed{4}", turn_count=2)
    assert correct.reward.reward == 1
    assert correct.reward.info["strict_format_ok"] == 1
    assert wrong.reward.reward == missing_format.reward.reward == 0
    assert missing_format.reward.info["strict_format_ok"] == 0


def test_math_qa_bounded_summary_2_turns_reward_keeps_correctness_dominant() -> None:
    env = MathQAEnv(reward_fn=exact_match_math_bounded_summary_2_turns_reward)
    example = env.build_example({"question": "What is 2 + 2?", "parsed_answer": "4"})

    one_turn_correct = env.evaluate_response(example, "\\boxed{4}", turn_count=1)
    two_turn_structured = env.evaluate_response(
        example,
        "\\boxed{4}",
        turn_count=2,
        turn_metadata=(
            {"summary": "Reduce the arithmetic to 2 + 2 = 4."},
            {"summary": ""},
        ),
    )
    wrong_two_turn_structured = env.evaluate_response(
        example,
        "\\boxed{5}",
        turn_count=2,
        turn_metadata=(
            {"summary": "Track the arithmetic state."},
            {"summary": ""},
        ),
    )

    assert one_turn_correct.reward.reward == pytest.approx(0.95)
    assert two_turn_structured.reward.reward == pytest.approx(1.2)
    assert two_turn_structured.reward.info["perfect_summary_chain"] == pytest.approx(1.0)
    assert wrong_two_turn_structured.reward.reward == pytest.approx(0.02)
    assert wrong_two_turn_structured.reward.info["exact_match"] == pytest.approx(0.0)
