"""Projection sampling matching the released Finetuning with Sampling code.

The authors' practical rule (Appendix C.2, boost_math.py) is greedy ascent in
mean base-model token log probability. It is NOT exact Metropolis-Hastings.
Generation is conditioned on an expert; scoring never sees that expert.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re
from typing import Any

PREFILL = "<thinking>\n"
# Identical to the frozen October 2 LFM2.5-Instruct IF baseline.
FORMAT_INSTRUCTION = "Think first before providing the answer. Inside <thinking>...</thinking>, first fully plan out all the steps needed to solve the problem, then execute that plan and check your work. Your thinking inside <thinking>...</thinking> is not visible to the user. When you have finished thinking, close the tag with </thinking>. Only after closing </thinking>, provide the requested result inside <answer>...</answer> tags. Apply the task's content and formatting requirements to the text inside the answer tags. Do not write anything after the answer block."


def dump(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n")
    tmp.replace(path)


def read_rows(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def normalize_prompt(text: str) -> str:
    return " ".join(text.casefold().split())


def target_prompt(tokenizer, messages: list[dict]) -> str:
    messages = [dict(role=m["role"], content=m["content"]) for m in messages]
    if not messages or messages[-1]["role"] != "user":
        raise ValueError("Expected conversation ending in a user request")
    messages[-1]["content"] += "\n\n" + FORMAT_INSTRUCTION
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + PREFILL


def proposal_prompt(tokenizer, row: dict, prefix: list[int]) -> tuple[str, tuple[int, ...]]:
    partial = PREFILL + tokenizer.decode(prefix, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    conversation = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in row["messages"])
    prompt = (
        "You are given an instruction-following task, an expert solution, and an initial, partial response. "
        "Carefully study the solution, identifying the reasoning and information it provides. "
        "Continue the partial response in your own words. Preserve the solution's factual content and "
        "satisfy every instruction in the task.\n\n"
        f"Task:\n{conversation}\n\nExpert solution:\n{row['expert']}\n\n"
        f"Here is a partial response:\n{partial}\n\n"
        "Starting with the partial response, continue the response in your own words, including the thinking process. "
        "Ensure the final answer exactly matches that of the provided solution. "
        "Put that exact final answer inside <answer>...</answer> tags. "
        "Continue exactly where the partial response ends; do not repeat it.\n\n" + FORMAT_INSTRUCTION
    )
    text = tokenizer.apply_chat_template([dict(role="user", content=prompt)], tokenize=False,
                                         add_generation_prompt=True) + PREFILL
    # Preserve exact prefix token IDs; decoding/re-encoding partial BPEs corrupts continuation.
    ids = tuple(tokenizer.encode(text, add_special_tokens=False)) + tuple(prefix)
    return text, ids


def format_audit(completion: str) -> dict:
    text = completion.removesuffix("<|endoftext|>").removesuffix("<|im_end|>").rstrip()
    tags = re.findall(r"</?(?:thinking|answer)>", text)
    match = re.fullmatch(r"(.*?)</thinking>\s*<answer>(.*?)</answer>", text, flags=re.S)
    valid = bool(match and tags == ["</thinking>", "<answer>", "</answer>"])
    reasoning, answer = (match.group(1).strip(), match.group(2).strip()) if match else ("", "")
    return dict(valid=valid and bool(reasoning) and bool(answer), reasoning=reasoning, answer=answer,
                nonempty_thinking=bool(reasoning))


def backward_completion_loss(model, batch, *, denominator: int, head_chunk: int,
                             autocast_dtype=None) -> float:
    """Accumulate response-only SFT gradients, normalized over the whole update.

    Backbone/head checkpointing avoids retaining sequence-by-vocabulary logits.
    The packed completion mask includes EOS and excludes every prompt token.
    """
    import torch
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint

    if denominator <= 0 or head_chunk <= 0:
        raise ValueError("SFT denominator and head chunk must be positive")
    with torch.autocast(batch.inputs.device.type, dtype=autocast_dtype,
                        enabled=autocast_dtype is not None):
        hidden = checkpoint(model.get_hidden_states, batch.inputs, batch.position_ids,
                            use_reentrant=False)
        mask = batch.valid_completion_mask().reshape(-1).bool()
        selected = hidden.reshape(-1, hidden.shape[-1])[mask]
        targets = batch.targets.reshape(-1)[mask]
        if not len(targets):
            raise ValueError("SFT batch has no completion targets")

        def head_loss(h, y):
            return F.cross_entropy(model.lm_head(h).float(), y, reduction="sum")

        loss = sum(checkpoint(head_loss, selected[start:start+head_chunk],
                              targets[start:start+head_chunk], use_reentrant=False)
                   for start in range(0, len(targets), head_chunk)) / denominator
    loss.backward()
    return float(loss.detach())


@dataclass
class ProjectionState:
    """One chain; random cuts include index 0, matching the reference code."""
    row: dict
    seed: int
    tokens: list[int] = field(default_factory=list)
    proposal_logprobs: list[float] = field(default_factory=list)
    score: float = -math.inf
    attempts: int = 0
    acceptances: int = 0
    blocks: int = 0

    def cut(self, block: int, iteration: int) -> int:
        if not self.tokens:
            raise ValueError("Cannot cut an empty trajectory")
        # Independent of queue order and of unrelated samples/acceptance decisions.
        rng = random.Random(f"{self.seed}:{self.row['id']}:{block}:{iteration}")
        return rng.randrange(len(self.tokens))

    def revise(self, candidate: list[int], logprobs: list[float], base_logprobs: list[float],
               *, cut: int, initialize: bool = False) -> bool:
        if not candidate or len(base_logprobs) != len(candidate):
            raise ValueError("Missing candidate token scores")
        if len(logprobs) != len(candidate) - cut:
            raise ValueError("Proposal suffix/logprob mismatch")
        score = math.fsum(base_logprobs) / len(base_logprobs)
        if not math.isfinite(score):
            raise ValueError("Nonfinite candidate likelihood")
        if not initialize:
            self.attempts += 1
        accept = initialize or score > self.score
        if accept:
            self.tokens = list(candidate)
            self.proposal_logprobs = self.proposal_logprobs[:cut] + list(logprobs)
            self.score = score
            self.acceptances += int(not initialize)
        return accept
