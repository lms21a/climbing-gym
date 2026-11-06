"""Restore native master-weight optimizer state without dtype coercion."""
from copy import deepcopy

import torch


def restore_master_optimizer(optimizer, payload, *, require_all=True):
    groups = payload["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer parameter-group count differs")
    mapping = []
    for current, saved in zip(optimizer.param_groups, groups):
        if len(current["params"]) != len(saved["params"]):
            raise ValueError("Optimizer parameter count differs")
        for key in ("lr", "weight_decay", "betas", "eps"):
            if key not in current and key not in saved:
                continue
            if current[key] != saved[key]:
                raise ValueError(f"Optimizer setting differs: {key}")
        mapping.extend(zip(current["params"], saved["params"]))
    expected = {index for _, index in mapping}
    if not set(payload["state"]).issubset(expected) or (require_all and set(payload["state"]) != expected):
        raise ValueError("Optimizer checkpoint does not cover every parameter")
    optimizer.state.clear()
    for parameter, index in mapping:
        if index not in payload["state"]:
            continue
        # Optimizer.load_state_dict casts tensors to parameter dtype, which
        # corrupts compressed integer master bits and FP32 master copies.
        optimizer.state[parameter] = {key: value.to("cpu") if isinstance(value, torch.Tensor) else deepcopy(value)
                                      for key, value in payload["state"][index].items()}
    return {"parameters": len(mapping), "steps": sorted({int(state["step"]) for state in optimizer.state.values()})}
