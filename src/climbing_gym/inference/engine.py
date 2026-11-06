"""Public async inference entry point backed by the native LFM engine."""

from climbing_gym.inference.lfm25_async_engine import LFM25AsyncInferenceEngine

AsyncInferenceEngine = LFM25AsyncInferenceEngine

__all__ = ["AsyncInferenceEngine"]
