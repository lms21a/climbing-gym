"""Response formatting ported from SimplePT; no inference dependencies."""
from dataclasses import dataclass
from typing import Any
import re

def prompt_prefills_open_think(prompt_text: str) -> bool:
    """Return True when prompt text currently ends inside an open <think> block."""
    lower = prompt_text.lower()
    return any(lower.rfind(f"<{tag}>") > lower.rfind(f"</{tag}>")
               for tag in ("think", "thinking"))


_THINK_RE = re.compile(r"<(?P<tag>think|thinking)>\s*(?P<body>.*?)\s*</(?P=tag)>", re.DOTALL | re.IGNORECASE)
_RESPONSE_XML_RE = re.compile(r"<response>\s*([\s\S]*?)\s*</response>", re.IGNORECASE)


@dataclass(frozen=True)
class SFTEnvBatch:
    """Canonical SFT batch: prompt/completion rows before tokenization."""

    sample_ids: list[str]
    prompts: list[str]
    metadata: list[dict[str, Any]]
    completions: list[str]


@dataclass(frozen=True)
class SFTSequenceConfig:
    max_seq_len: int
    enable_thinking: bool = True
    force_think_prefix: bool = False
    drop_response_xml: bool = False
    require_think_close: bool = False
    think_close_token_id: int = -1
    append_eos_token: bool = True


def adapt_completion_to_thinking_template(
    completion: str,
    continue_from_open_think: bool,
) -> str:
    """Adapt target text when prompt already opened `<think>`."""
    text = completion.strip()
    if (not continue_from_open_think) or (not text):
        return text

    think_match = _THINK_RE.search(text)
    if not think_match:
        return text

    think_content = think_match.group("body").strip()
    tag = think_match.group("tag")
    after_think = text[think_match.end() :].strip()

    response_matches = list(_RESPONSE_XML_RE.finditer(after_think))
    if response_matches:
        response_payload = response_matches[-1].group(1).strip()
        return f"{think_content}\n</{tag}>\n<response>\n{response_payload}\n</response>"

    if after_think:
        return f"{think_content}\n</{tag}>\n{after_think}"
    return f"{think_content}\n</{tag}>"


def strip_response_xml_envelope(text: str) -> str:
    """Remove `<response>...</response>` envelope while preserving payload."""
    cleaned = text.strip()
    if not cleaned:
        return cleaned

    response_matches = list(_RESPONSE_XML_RE.finditer(cleaned))
    if not response_matches:
        return cleaned

    payload = response_matches[-1].group(1).strip()
    think_match = _THINK_RE.search(cleaned)
    if think_match:
        think_content = think_match.group("body").strip()
        tag = think_match.group("tag")
        if payload:
            return f"<{tag}>\n{think_content}\n</{tag}>\n{payload}"
        return f"<{tag}>\n{think_content}\n</{tag}>"

    prefix = cleaned[: response_matches[-1].start()].strip()
    suffix = cleaned[response_matches[-1].end() :].strip()
    parts = [part for part in (prefix, payload, suffix) if part]
    return "\n".join(parts).strip()
