"""Greedy projection of reasoning with an immutable expert answer.

This is a constrained complete-trajectory adaptation of the paper, not exact
Metropolis-Hastings or the released block-growing sampler. Its objective is
mean raw base-policy log probability, including the fixed answer and EOS.
"""
from __future__ import annotations

import math
import random

from .projection_sampling import PREFILL, format_audit

STOPS = ("</thinking>", "<answer>", "</answer>", "<thinking>", "<|endoftext|>")


def compose(tokenizer, reasoning: str, answer: str) -> dict:
    """Canonicalize only the envelope; preserve the reference answer verbatim."""
    reasoning = reasoning.strip()
    if not reasoning or not answer or answer != answer.strip():
        raise ValueError("Nonempty reasoning and a normalized reference answer are required")
    if any(tag in reasoning + answer for tag in (*STOPS, "<|im_end|>")) or "\ufffd" in reasoning:
        raise ValueError("Control tag or invalid Unicode in reasoning/answer")
    text = reasoning + "\n</thinking>\n<answer>" + answer + "</answer>"
    ids = tokenizer.encode(text, add_special_tokens=False) + [tokenizer.eos_token_id]
    decoded = tokenizer.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=False)
    if decoded != text + tokenizer.eos_token:
        raise ValueError("Completion tokenization did not round trip")
    audit = format_audit(decoded)
    if not audit["valid"] or audit["answer"] != answer:
        raise ValueError("Fixed-answer invariant failed")
    return dict(completion=decoded, completion_token_ids=ids,
                reasoning_token_ids=tokenizer.encode(reasoning, add_special_tokens=False),
                reference_answer=answer, format=audit)


def expert_completion(tokenizer, row):
    audit = format_audit(row["expert"].removeprefix(PREFILL))
    if not audit["valid"] or audit["answer"] != row["reference_answer"]:
        raise ValueError("Invalid source expert")
    return compose(tokenizer, audit["reasoning"], row["reference_answer"])


def proposal_prompt(tokenizer, row, prefix):
    task = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in row["messages"])
    instruction = (
        "Study this instruction-following task and its expert solution. Continue the reasoning "
        "in your own words, preserving the solution's facts and checking every task requirement. "
        "The expert's final answer is fixed and will be appended automatically. "
        "Write only the continuation of the reasoning, then close it with </thinking>. "
        "Do not repeat the supplied reasoning prefix or write an answer block.\n\n"
        f"Task:\n{task}\n\nExpert solution:\n{row['expert']}"
    )
    text = tokenizer.apply_chat_template([dict(role="user", content=instruction)],
                                         tokenize=False, add_generation_prompt=True) + PREFILL
    # No decoded prefix in the user message: partial BPEs must stay as token IDs.
    return text, tuple(tokenizer.encode(text, add_special_tokens=False)) + tuple(prefix)


def reasoning_budget(tokenizer, answer, max_tokens):
    suffix = tokenizer.encode("\n</thinking>\n<answer>" + answer + "</answer>", add_special_tokens=False)
    budget = max_tokens - len(suffix) - 8  # EOS and boundary-token safety margin.
    if budget <= 0:
        raise ValueError("Answer exceeds the complete-trajectory budget")
    return budget


def proposal_candidate(tokenizer, row, prefix, generated, max_tokens):
    decoded = tokenizer.decode(list(prefix) + list(generated), skip_special_tokens=False,
                               clean_up_tokenization_spaces=False)
    if "</thinking>" not in decoded:
        return None, "no_natural_reasoning_close"
    reasoning = decoded.split("</thinking>", 1)[0]
    try:
        candidate = compose(tokenizer, reasoning, row["reference_answer"])
    except ValueError as exc:
        return None, str(exc)
    if len(candidate["completion_token_ids"]) > max_tokens:
        return None, "complete_trajectory_exceeds_budget"
    return candidate, None


def random_cut(seed, row_id, step, reasoning_length, budget):
    if step == 0:
        return 0
    return random.Random(f"{seed}:{row_id}:{step}").randrange(min(reasoning_length, budget))


def add_score(candidate, logprobs):
    if len(logprobs) != len(candidate["completion_token_ids"]) or not logprobs:
        raise ValueError("Completion and scoring lengths differ")
    if not all(math.isfinite(x) for x in logprobs):
        raise ValueError("Nonfinite base likelihood")
    candidate["base_total_logprob"] = math.fsum(logprobs)
    candidate["base_mean_logprob"] = math.fsum(logprobs) / len(logprobs)
    return candidate
