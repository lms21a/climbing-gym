from dataclasses import replace

import pytest
import torch

from climbing_gym.models.lfm2 import LFM2Config, LFM2Model


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA required')
@pytest.mark.parametrize('recompute,fused', [(False,False),(False,True),(True,True)])
def test_checkpointed_eval_mode_layers_preserve_packed_gradients_and_boundaries(recompute,fused):
    pytest.importorskip('flash_attn')
    torch.manual_seed(529)
    cfg = LFM2Config(vocab_size=257, hidden_size=128, mlp_hidden_size=192, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, layer_types=('conv','full_attention','conv','full_attention'),
        max_position_embeddings=256, attn_backend='flash', shortconv_backend='triton_k3',
        recompute_mlp=recompute, fused_backbone_kernels=fused)
    reference = LFM2Model(cfg).to(device='cuda', dtype=torch.bfloat16).eval()
    actual = LFM2Model(replace(cfg, checkpoint_layers=True)).to(device='cuda', dtype=torch.bfloat16).eval()
    actual.load_state_dict(reference.state_dict())
    # Several backward passes over differently packed examples exercise the
    # ContextVar restoration during recomputation as well as accumulation.
    for lengths in ((5,9), (7,3,8)):
        ids = torch.randint(0, 257, (1,sum(lengths)), device='cuda')
        positions = torch.cat([torch.arange(n, device='cuda') for n in lengths])[None]
        target = torch.randint(0,257,ids.shape,device='cuda')
        for model in (reference, actual):
            logits = model(ids,positions)
            torch.nn.functional.cross_entropy(logits.reshape(-1,257).float(), target.reshape(-1)).backward()
    for (name, first), (_, second) in zip(reference.named_parameters(), actual.named_parameters()):
        assert first.grad is not None and second.grad is not None, name
        torch.testing.assert_close(first.grad, second.grad, atol=0, rtol=0, msg=name)
