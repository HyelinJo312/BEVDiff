"""Global semantic conditioning: opt-out, common CFG path and gradients."""

import copy
import unittest
from contextlib import ExitStack, redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch

from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5 import (
    DiffusionUNetModel, ResBlock, SBAM, SBAMResBlock, timestep_embedding,
)


def make_model(use_global_seg=True):
    with redirect_stdout(StringIO()):
        return DiffusionUNetModel(
            seg_bev_aligner=dict(bev_h=18, bev_w=18, num_classes=3,
                                 emb_channels=32, channel_mult=[1, 1, 1]),
            in_channels=32, out_channels=32, model_channels=32,
            seg_channels=[32, 32, 32], num_res_blocks=1,
            attention_ds=[1, 2, 4], channel_mult=[1, 2, 4],
            image_size=18, num_head_channels=32, resblock_updown=True,
            use_scale_shift_norm=True, return_multiscale=False,
            use_global_seg=use_global_seg,
        )


class GlobalSegEmbeddingTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.model = make_model()
        self.x = torch.randn(1, 32, 18, 18)
        self.t = torch.tensor([100])
        self.p = torch.rand(1, 4, 18, 18)

    def test_disabled_has_no_global_parameters_and_zero_init_matches(self):
        disabled = make_model(False)
        self.assertIsNone(disabled.seg_global_embed)
        self.assertFalse(any('seg_global_embed' in key for key in disabled.state_dict()))
        # Wake the final output so output equality is not just comparing zeros.
        with torch.no_grad():
            self.model.out[-1].weight.normal_(std=.01)
        disabled.load_state_dict({k: v for k, v in self.model.state_dict().items()
                                 if not k.startswith('seg_global_embed.')}, strict=True)
        with torch.no_grad():
            encoded = self.model.encode_seg(None, [{}], semantic_prob=self.p)
            self.assertEqual(torch.count_nonzero(self.model.seg_global_embed(encoded[1])), 0)
            expected = disabled(self.x, self.t, [{}], semantic_prob=self.p)
            actual = self.model(self.x, self.t, [{}], semantic_prob=self.p)
        self.assertGreater(actual.abs().sum().item(), 0)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_false_flag_skips_existing_global_module(self):
        self.model.use_global_seg = False
        with patch.object(self.model.seg_global_embed, 'forward', side_effect=AssertionError('Global path executed')):
            self.model(self.x, self.t, [{}], semantic_prob=self.p)

    def test_combined_embedding_reaches_encoder_middle_and_every_gate(self):
        with torch.no_grad():
            self.model.seg_global_embed[-1].weight.normal_(std=.02)
        encoded = self.model.encode_seg(None, [{}], semantic_prob=self.p)
        time = self.model.time_embed(timestep_embedding(self.t, self.model.model_channels))
        global_emb = self.model.seg_global_embed(encoded[1])
        self.assertGreater(global_emb.abs().sum().item(), 0)
        expected = (time + global_emb).detach()
        seen = []
        def block_hook(module, inputs):
            seen.append(inputs[1].detach())
        def gate_hook(module, inputs):
            seen.append(inputs[0].detach())
        gates = [m.time_gate for m in self.model.modules() if isinstance(m, SBAM)]
        with ExitStack() as stack:
            for block in (self.model.input_blocks[1], self.model.middle_block):
                stack.enter_context(block.register_forward_pre_hook(block_hook))
            for gate in gates:
                stack.enter_context(gate.register_forward_pre_hook(gate_hook))
            self.model(self.x, self.t, [{}], seg_bev_maps=encoded)
        self.assertEqual(len(seen), 2 + len(gates))
        for value in seen:
            torch.testing.assert_close(value, expected)

    def test_raw_and_preencoded_cfg_conditions_match(self):
        with torch.no_grad():
            self.model.seg_global_embed[-1].weight.normal_(std=.02)
            self.model.out[-1].weight.normal_(std=.01)
        self.model.eval()
        globals_seen = []
        with torch.no_grad(), patch.object(self.model.seg_aligner, 'project_semantics', side_effect=AssertionError('Live projection')):
            for probability in (self.p, torch.zeros_like(self.p)):
                encoded = self.model.encode_seg(None, [{}], semantic_prob=probability)
                globals_seen.append(self.model.seg_global_embed(encoded[1]))
                direct = self.model(self.x, self.t, [{}], semantic_prob=probability)
                preencoded = self.model(self.x, self.t, [{}], seg_bev_maps=encoded)
                torch.testing.assert_close(direct, preencoded)
        self.assertGreater((globals_seen[0] - globals_seen[1]).abs().sum().item(), 0)

    def test_global_mlp_and_condition_encoder_learn_after_zero_init(self):
        optimizer = torch.optim.SGD(self.model.parameters(), lr=.01)
        target = torch.randn(1, 128)
        for _ in range(2):
            optimizer.zero_grad()
            encoded = self.model.encode_seg(None, [{}], semantic_prob=self.p)
            z = self.model.seg_global_embed(encoded[1])
            (z - target).square().mean().backward()
            optimizer.step()
        for parameter in (self.model.seg_global_embed[3].weight,
                          self.model.seg_global_embed[-1].weight,
                          self.model.seg_aligner.prob_to_emb[0].weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_checkpointing_preserves_global_branch_gradients(self):
        with torch.no_grad():
            self.model.out[-1].weight.normal_(std=.01)
            self.model.seg_global_embed[-1].weight.normal_(std=.02)
            # Emulate nonzero residual outputs after warmup; at initialization
            # those zero convolutions block embedding gradients by design.
            for module in self.model.modules():
                if isinstance(module, (ResBlock, SBAMResBlock)):
                    module.out_layers[-1].weight.normal_(std=.01)
        checkpointed = copy.deepcopy(self.model)
        checkpointed.enable_gradient_checkpointing()
        outputs = []
        for model in (self.model, checkpointed):
            out = model(self.x, self.t, [{}], semantic_prob=self.p)
            out.square().mean().backward()
            outputs.append(out.detach())
        torch.testing.assert_close(*outputs)
        for a, b in zip(self.model.seg_global_embed.parameters(), checkpointed.seg_global_embed.parameters()):
            self.assertIsNotNone(a.grad)
            self.assertIsNotNone(b.grad)
            torch.testing.assert_close(a.grad, b.grad)
        self.assertGreater(checkpointed.seg_global_embed[-1].weight.grad.abs().sum().item(), 0)


if __name__ == '__main__':
    unittest.main()
