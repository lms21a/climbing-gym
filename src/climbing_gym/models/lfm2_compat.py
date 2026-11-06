"""Checkpoint compatibility for the native LFM2 runtimes."""

from __future__ import annotations

from typing import Any, Mapping


def normalize_lfm2_config(payload: Mapping[str, Any]) -> dict[str, Any]:
    """Read both the Transformers 4 and 5 spellings without changing the input.

    In particular, LFM2.5-2.6B uses nested RoPE parameters with theta=1e7.
    Silently dropping that object would instead select the 1.2B default (1e6).
    Only unscaled RoPE is implemented by the native models.
    """
    normalized = dict(payload)
    rope = normalized.get("rope_parameters") or normalized.get("rope_scaling")
    if rope is not None:
        if not isinstance(rope, Mapping):
            raise ValueError("LFM2 rope_parameters must be an object")
        if any(isinstance(value, Mapping) for value in rope.values()):
            raise ValueError("Unsupported LFM2 per-layer rope_parameters")
        rope_type = rope.get("rope_type", rope.get("type", "default"))
        if rope_type != "default":
            raise ValueError(f"Unsupported LFM2 rope_type: {rope_type!r}; expected 'default'")
        if rope.get("rope_theta") is not None:
            # Transformers 4 may populate the legacy default alongside a modern
            # rope_parameters object, so the explicit nested value takes priority.
            normalized["rope_theta"] = float(rope["rope_theta"])
    if normalized.get("tie_embedding") is None and normalized.get("tie_word_embeddings") is not None:
        normalized["tie_embedding"] = bool(normalized["tie_word_embeddings"])
    if normalized.get("block_ff_dim") is None and normalized.get("intermediate_size") is not None:
        normalized["block_ff_dim"] = int(normalized["intermediate_size"])
    return normalized


def load_lfm2_tokenizer(model_path: str, **kwargs: Any) -> Any:
    """Load modern Rust tokenizer exports on Transformers 4 as well as 5."""
    from transformers import AutoTokenizer, PreTrainedTokenizerFast

    try:
        return AutoTokenizer.from_pretrained(model_path, **kwargs)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend does not exist" not in str(exc):
            raise
        # TokenizersBackend is the Transformers 5 name for the generic fast
        # tokenizer. Its tokenizer.json, special tokens and Jinja template are
        # already understood by the Transformers 4 implementation.
        return PreTrainedTokenizerFast.from_pretrained(model_path, **kwargs)
