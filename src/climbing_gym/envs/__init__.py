from climbing_gym.envs.countdown import CountdownEnv, exact_countdown_reward
from climbing_gym.envs.futoshiki import FutoshikiEnv, dense_futoshiki_reward
from climbing_gym.envs.knights_knaves import KnightsKnavesEnv, exact_knights_knaves_reward
from climbing_gym.envs.contracts import (
    BaseEnv,
    EnvBatch,
    EnvEvaluation,
    EnvExample,
    EnvFeedback,
    RewardFn,
    RewardRequest,
    RewardResult,
)
from climbing_gym.envs.math_qa import (
    MathQAEnv,
    exact_match_math_bounded_summary_2_turns_reward,
    ParsedMathQAResponse,
    exact_match_math_hybrid_2_turns_reward,
    exact_match_math_min_2_turns_reward,
    exact_match_math_reward,
    exact_match_single_answer_math_reward,
    exact_match_math_summary_bootstrap_2_turns_reward,
    exact_match_math_soft_target_2_turns_reward,
    exact_match_math_underturn_penalty_2_turns_reward,
    exact_match_with_format_math_reward,
    normalize_math_answer,
)

__all__ = ['CountdownEnv', 'exact_countdown_reward', 'FutoshikiEnv', 'dense_futoshiki_reward', 'KnightsKnavesEnv', 'exact_knights_knaves_reward', 'BaseEnv', 'EnvBatch', 'EnvEvaluation', 'EnvExample', 'EnvFeedback', 'RewardFn', 'RewardRequest', 'RewardResult', 'MathQAEnv', 'exact_match_math_bounded_summary_2_turns_reward', 'ParsedMathQAResponse', 'exact_match_math_hybrid_2_turns_reward', 'exact_match_math_min_2_turns_reward', 'exact_match_math_reward', 'exact_match_single_answer_math_reward', 'exact_match_math_summary_bootstrap_2_turns_reward', 'exact_match_math_soft_target_2_turns_reward', 'exact_match_math_underturn_penalty_2_turns_reward', 'exact_match_with_format_math_reward', 'normalize_math_answer']

# CFOP's Rust binding is imported lazily when constructing its verifier.
from climbing_gym.envs.cfop_cube import (
    CFOPCompatibilityError, CFOPCubeEnv, CFOPPhaseEnv,
    build_cfop_prompt, build_cfop_reasoning_prompt,
    build_cfop_phase_prompt, build_cfop_phase_reasoning_prompt,
    cfop_dense_v2_reward, cfop_phase_v1_reward,
)
__all__ += [
    "CFOPCompatibilityError", "CFOPCubeEnv", "CFOPPhaseEnv",
    "build_cfop_prompt", "build_cfop_reasoning_prompt",
    "build_cfop_phase_prompt", "build_cfop_phase_reasoning_prompt",
    "cfop_dense_v2_reward", "cfop_phase_v1_reward",
]
