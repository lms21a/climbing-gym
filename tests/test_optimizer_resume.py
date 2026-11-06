import copy

import pytest
import torch

from climbing_gym.optimizer_resume import restore_master_optimizer
from climbing_gym.ppo import MasterWeightAdamW


def test_restored_master_optimizer_matches_uninterrupted_next_update():
    p = torch.nn.Parameter(torch.tensor([1.1, -2.3], dtype=torch.bfloat16))
    opt = MasterWeightAdamW([p], lr=1e-5, weight_decay=.01)
    p.grad = torch.tensor([.12, -.23], dtype=p.dtype)
    opt.step()
    saved = copy.deepcopy(opt.state_dict())
    q = torch.nn.Parameter(p.detach().clone())
    resumed = MasterWeightAdamW([q], lr=1e-5, weight_decay=.01)
    assert restore_master_optimizer(resumed, saved)["steps"] == [1]
    assert resumed.state[q]["master_param"].dtype == torch.float32
    p.grad = torch.tensor([-.37, .45], dtype=p.dtype)
    q.grad = p.grad.clone()
    opt.step()
    resumed.step()
    assert torch.equal(p, q)
    assert torch.equal(opt.state[p]["master_param"], resumed.state[q]["master_param"])


def test_restore_preserves_compressed_integer_master_bits():
    p = torch.nn.Parameter(torch.zeros(2, dtype=torch.bfloat16))
    opt = MasterWeightAdamW([p], lr=1e-5, weight_decay=.01)
    payload = opt.state_dict()
    payload["state"][0] = {"step": 50, "master_param_low_bits": torch.tensor([32767, -12345], dtype=torch.int16),
                           "master_param_tie_round_down_indices": torch.tensor([1], dtype=torch.int32),
                           "exp_avg": torch.zeros(2, dtype=torch.bfloat16), "exp_avg_sq": torch.ones(2, dtype=torch.bfloat16)}
    restore_master_optimizer(opt, payload)
    assert opt.state[p]["master_param_low_bits"].tolist() == [32767, -12345]
    assert opt.state[p]["master_param_low_bits"].dtype == torch.int16
    assert opt.state[p]["master_param_tie_round_down_indices"].dtype == torch.int32
    payload["param_groups"][0]["lr"] = 1e-4
    with pytest.raises(ValueError, match="lr"):
        restore_master_optimizer(opt, payload)
