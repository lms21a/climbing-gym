"""Historical Futoshiki projection/SFT rules, extracted from the run snapshots.

These complete-trajectory samplers are verifier-constrained greedy ascent,
not the package's block-growing sampler and not Metropolis-Hastings.
"""

from collections import defaultdict
from dataclasses import dataclass
import math
import random
import re

import torch

from climbing_gym.envs.futoshiki import score
from climbing_gym.projection_sampling import FORMAT_INSTRUCTION, format_audit

TIERS = ("5x5_g11_i5", "5x5_g9_i5")


@dataclass(frozen=True)
class Protocol:
    stage: str
    seed: int
    inference_seed: int
    projection_budget: int
    context: int
    native_batch: int
    packed_tokens: int
    batch_size: int
    head_chunk: int
    schedule_epochs: int
    retained_epoch: int
    retries: int
    block_size: int
    teacher_fallback: bool
    refinements: int = 10
    seed_refinements: int = 10
    learning_rate: float = 1e-5
    weight_decay: float = 0.01
    grad_clip: float = 1.0


PROTOCOLS = {
    "solver_projection_256": Protocol(
        "solver_projection_256",
        20261010,
        20261010,
        2560,
        8192,
        128,
        3200,
        4,
        128,
        2,
        1,
        64,
        1024,
        False,
    ),
    "projection_reset": Protocol(
        "projection_reset",
        2026101503,
        20261012,
        8192,
        18432,
        64,
        9216,
        32,
        512,
        4,
        2,
        20,
        512,
        True,
    ),
}


def ordinary_prompt(tokenizer, row):
    return (
        tokenizer.apply_chat_template(
            [dict(role="user", content=row["prompt"])],
            tokenize=False,
            add_generation_prompt=True,
        )
        + "<thinking>\n"
    )


def proposal_prompt(tok, row, prefix):
    # This is the original project_rl500_futoshiki_traces.py prompt, including
    # its first-person instruction and exact assistant-prefix token placement.
    expert = (
        row["expert"].removesuffix("<|endoftext|>").removesuffix("<|im_end|>").rstrip()
    )
    instruction = (
        "Solve the task below in your own words, using the expert response as guidance. "
        "Continue the assistant's partial solution directly, in the first person. "
        "Work through the puzzle constraints and explain the deductions and checks. "
        "Do not talk about the expert, a passage, a rewrite, or the instructions. "
        "Do not repeat the partial solution. Preserve established assignments. "
        "Finish with exactly the expert's final answer.\n\n"
        f"Task:\n{row['row']['prompt']}\n\nExpert response:\n<thinking>\n{expert}\n\n"
        f"Final answer to reproduce exactly:\n{row['reference_answer']}\n\n"
        "Write only the continuation of the assistant response.\n\n"
        + FORMAT_INSTRUCTION
    )
    text = (
        tok.apply_chat_template(
            [dict(role="user", content=instruction)],
            tokenize=False,
            add_generation_prompt=True,
        )
        + "<thinking>\n"
    )
    return text, tuple(tok.encode(text, add_special_tokens=False)) + tuple(prefix)


def candidate_audit(tokenizer, tokens, row, finish_reason, *, solver=False):
    text = tokenizer.decode(
        tokens, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    grade = score(text, row, reward_profile="exact_match_strict")
    formatting = format_audit(text)
    # Preserve the two historical audits, rather than silently broadening one.
    if solver:
        meta = bool(
            re.search(
                r"\b(?:expert|passage|provided (?:text|solution|trace)|paraphras\w*|rewriting)\b",
                formatting["reasoning"],
                re.I,
            )
        )
    else:
        meta = bool(
            re.search(
                r"\b(?:expert response|expert solution|provided passage|paraphras\w*|rewrit\w*)\b",
                text,
                re.I,
            )
        )
    accepted = bool(
        tokens
        and tokens[-1] == tokenizer.eos_token_id
        and finish_reason != "length"
        and grade["exact_match"]
        and formatting["valid"]
        and not meta
    )
    return dict(
        accepted=accepted, response=text, grade=grade, format=formatting, meta=meta
    )


def suffix_budget(protocol, guide, tokens, cut):
    if protocol.teacher_fallback:
        limit = len(tokens) if tokens else protocol.projection_budget
    else:
        limit = min(
            protocol.projection_budget,
            protocol.packed_tokens - len(guide["prompt_token_ids"]),
        )
    if not 0 <= cut < limit:
        raise ValueError("Cut must leave at least one generation token")
    return limit - cut


def balanced_batches(rows, *, batch_size, seed):
    if batch_size % len(TIERS):
        raise ValueError("Batch size must divide equally between tiers")
    buckets = {t: [r for r in rows if r["row"]["tier"] == t] for t in TIERS}
    if (
        len(rows) != sum(map(len, buckets.values()))
        or len(set(map(len, buckets.values()))) != 1
    ):
        raise ValueError("SFT requires equal counts from the two known tiers")
    for tier, values in buckets.items():
        random.Random(f"{seed}:{tier}").shuffle(values)
    per_tier = batch_size // len(TIERS)
    for offset in range(0, len(buckets[TIERS[0]]), per_tier):
        yield [
            r
            for i in range(per_tier)
            for t in TIERS
            if offset + i < len(buckets[t])
            for r in [buckets[t][offset + i]]
        ]


def training_batches(rows, protocol, epoch):
    if protocol.teacher_fallback:
        return list(
            balanced_batches(
                rows, batch_size=protocol.batch_size, seed=protocol.seed + epoch
            )
        )
    order = list(range(len(rows)))
    random.Random(protocol.seed + epoch).shuffle(order)
    return [
        [rows[i] for i in order[o : o + protocol.batch_size]]
        for o in range(0, len(rows), protocol.batch_size)
    ]


def learning_rate(protocol, step, examples):
    total = math.ceil(examples / protocol.batch_size) * protocol.schedule_epochs
    if not protocol.teacher_fallback:
        return protocol.learning_rate * 0.5 * (1 + math.cos(math.pi * step / total))
    warmup = min(8, max(1, total // 20))
    progress = max(0.0, (step - warmup) / max(1, total - warmup))
    return (
        protocol.learning_rate
        * min(1.0, (step + 1) / warmup)
        * (0.2 + 0.8 * 0.5 * (1 + math.cos(math.pi * progress)))
    )


class FP32MasterAdamW(torch.optim.Optimizer):
    """BF16 forward weights with FP32 masters AND moments, avoiding beta2 rounding.

    Update one parameter at a time to bound temporary GPU memory. State is
    explicitly movable for native evaluation and exact optimizer restoration.
    """

    def __init__(
        self, params, *, lr=2e-6, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01
    ):
        super().__init__(
            params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        )

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            raise ValueError("Closures are unsupported")
        for group in self.param_groups:
            beta1, beta2 = group["betas"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                state = self.state[parameter]
                if not state:
                    state.update(
                        step=0,
                        master=parameter.detach().float().clone(),
                        exp_avg=torch.zeros_like(parameter, dtype=torch.float32),
                        exp_avg_sq=torch.zeros_like(parameter, dtype=torch.float32),
                    )
                if state["master"].device != parameter.device:
                    raise RuntimeError("Restore optimizer state to the training device")
                state["step"] += 1
                gradient = parameter.grad.float()
                state["exp_avg"].lerp_(gradient, 1 - beta1)
                state["exp_avg_sq"].mul_(beta2).addcmul_(
                    gradient, gradient, value=1 - beta2
                )
                denominator = (
                    state["exp_avg_sq"]
                    .sqrt()
                    .div_((1 - beta2 ** state["step"]) ** 0.5)
                    .add_(group["eps"])
                )
                state["master"].mul_(1 - group["lr"] * group["weight_decay"])
                state["master"].addcdiv_(
                    state["exp_avg"],
                    denominator,
                    value=-group["lr"] / (1 - beta1 ** state["step"]),
                )
                parameter.copy_(state["master"])

    def move_state(self, device):
        for state in self.state.values():
            for key, value in list(state.items()):
                if isinstance(value, torch.Tensor):
                    state[key] = value.to(device)

    def load_state_dict(self, saved):
        # Optimizer.load_state_dict normally casts floating state to the BF16
        # parameter dtype. Preserve the FP32 master/moments on exact resume.
        if len(saved["param_groups"]) != len(self.param_groups):
            raise ValueError("Optimizer parameter groups differ")
        mapping = {}
        for group, incoming in zip(self.param_groups, saved["param_groups"]):
            if len(group["params"]) != len(incoming["params"]):
                raise ValueError("Optimizer parameter counts differ")
            mapping.update(zip(incoming["params"], group["params"]))
            group.update({k: v for k, v in incoming.items() if k != "params"})
        self.state = defaultdict(dict)
        for key, state in saved["state"].items():
            parameter = mapping[key]
            self.state[parameter] = {
                k: (
                    v.to(device=parameter.device, dtype=torch.float32)
                    if isinstance(v, torch.Tensor)
                    else v
                )
                for k, v in state.items()
            }
