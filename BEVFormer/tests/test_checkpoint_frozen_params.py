"""ldm CheckpointFunction with frozen parameters (warm-started, partly frozen UNet)."""

import unittest

import torch

from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5_4 import SelfAttnBlock


def _block():
    torch.manual_seed(0)
    return SelfAttnBlock(dim=32, n_heads=4, d_head=8)  # checkpoint=True by default


class CheckpointFrozenParamsTest(unittest.TestCase):

    def test_frozen_block_backpropagates_to_input(self):
        block = _block()
        block.requires_grad_(False)
        x = torch.randn(2, 16, 32, requires_grad=True)
        block(x).sum().backward()
        self.assertIsNotNone(x.grad)
        self.assertTrue(torch.isfinite(x.grad).all())
        self.assertTrue(all(p.grad is None for p in block.parameters()))

    def test_matches_no_checkpoint_when_trainable(self):
        ref, ckpt = _block(), _block()
        ref.checkpoint = False
        x_ref = torch.randn(2, 16, 32, requires_grad=True)
        x_ckpt = x_ref.detach().clone().requires_grad_(True)
        ref(x_ref).sum().backward()
        ckpt(x_ckpt).sum().backward()
        torch.testing.assert_close(x_ckpt.grad, x_ref.grad)
        for p_ref, p_ckpt in zip(ref.parameters(), ckpt.parameters()):
            torch.testing.assert_close(p_ckpt.grad, p_ref.grad)

    def test_partially_frozen_block(self):
        block = _block()
        block.attn1.requires_grad_(False)
        x = torch.randn(2, 16, 32, requires_grad=True)
        block(x).sum().backward()
        self.assertTrue(all(p.grad is None for p in block.attn1.parameters()))
        self.assertTrue(all(p.grad is not None for p in block.ff.parameters()))


if __name__ == '__main__':
    unittest.main()
