from climbing_gym.inference.config import (
    InferenceEngineConfig,
    SamplingConfig,
)
from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine
from climbing_gym.inference.lfm25_engine import (
    LFM25DecodeBatch,
    LFM25Engine,
    LFM25GenerationOutput,
    LFM25HybridCache,
    LFM25PackedWeights,
    LFM25PrefixBatch,
    LFM25SamplingConfig,
    LFM25StepOutput,
    describe_lfm25_checkpoint,
    flatten_token_batches,
)
from climbing_gym.inference.types import GenerationOutput, GenerationResult


def __getattr__(name: str):
    if name in {"LFM25GenerationSession", "LFM25ContinuationHandle", "LFM25SessionResult"}:
        from climbing_gym.inference import lfm25_session

        return getattr(lfm25_session, name)
    if name in {"LFM3Config", "LFM3Engine", "LFM3PackedWeights"}:
        from climbing_gym.inference import lfm3_engine

        return getattr(lfm3_engine, name)
    if name == "AsyncInferenceEngine":
        from climbing_gym.inference.engine import AsyncInferenceEngine

        return AsyncInferenceEngine
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

__all__ = [
    "AsyncInferenceEngine",
    "InferenceEngineConfig",
    "LFM25AsyncInferenceEngine",
    "LFM3Config",
    "LFM3Engine",
    "LFM3PackedWeights",
    "LFM25DecodeBatch",
    "LFM25Engine",
    "LFM25GenerationOutput",
    "LFM25GenerationSession",
    "LFM25ContinuationHandle",
    "LFM25SessionResult",
    "LFM25HybridCache",
    "LFM25PackedWeights",
    "LFM25PrefixBatch",
    "LFM25SamplingConfig",
    "LFM25StepOutput",
    "GenerationOutput",
    "GenerationResult",
    "SamplingConfig",
    "describe_lfm25_checkpoint",
    "flatten_token_batches",
]
