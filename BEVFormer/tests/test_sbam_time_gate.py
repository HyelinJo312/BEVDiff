"""Timestep-gated SBAM initialization, conditioning and checkpoint gradients."""

import copy
import unittest

import torch

from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5 import (
    SBAM, SBAMResBlock, timestep_embedding,
)


class SBAMTimeGateTest(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(42)
        self.x = torch.randn(2, 32, 8, 8)
        self.condition = torch.randn(2, 16, 8, 8)
        self.emb = timestep_embedding(torch.tensor([10, 900]), 128)

    def test_initial_output_and_input_gradients_match_original_sbam(self):
        sbam = SBAM(32, 16, 128)
        x = self.x.clone().requires_grad_()
        condition = self.condition.clone().requires_grad_()
        original = sbam.param_free_norm(x) * (1 + sbam.conv_gamma(condition)) + sbam.conv_beta(condition)
        gated = sbam(x, condition, self.emb)
        torch.testing.assert_close(gated, original, rtol=0, atol=0)
        expected_grads = torch.autograd.grad(original.square().mean(), (x, condition), retain_graph=True)
        actual_grads = torch.autograd.grad(gated.square().mean(), (x, condition))
        for actual, expected in zip(actual_grads, expected_grads):
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_gate_projection_and_semantic_branch_receive_gradients(self):
        sbam = SBAM(32, 16, 128)
        sbam(self.x, self.condition, self.emb).square().mean().backward()
        for parameter in (sbam.time_gate[-1].weight, sbam.time_gate[-1].bias,
                          sbam.conv_gamma.weight, sbam.conv_beta.weight):
            self.assertIsNotNone(parameter.grad)
            self.assertTrue(torch.isfinite(parameter.grad).all())
            self.assertGreater(parameter.grad.abs().sum().item(), 0)

    def test_timestep_changes_modulation_for_identical_spatial_inputs(self):
        sbam = SBAM(32, 16, 128)
        with torch.no_grad():
            # Choose opposite projections along the difference of two time
            # embeddings; both gates must respond without changing x or S.
            activated = sbam.time_gate[0](self.emb)
            delta = activated[1] - activated[0]
            sbam.time_gate[-1].weight[0].copy_(delta)
            sbam.time_gate[-1].weight[1].copy_(-delta)
        x = self.x[:1].repeat(2, 1, 1, 1)
        condition = self.condition[:1].repeat(2, 1, 1, 1)
        output = sbam(x, condition, self.emb)
        self.assertGreater((output[0] - output[1]).abs().max().item(), 0)
        emb = self.emb.clone().requires_grad_()
        sbam(x, condition, emb).square().mean().backward()
        self.assertGreater(emb.grad.abs().sum().item(), 0)

    def test_activation_checkpoint_preserves_outputs_and_gradients(self):
        plain = SBAMResBlock(32, 128, 0, out_channels=64, seg_channels=16)
        with torch.no_grad():
            # Wake the initially zero residual output to exercise gradients
            # through both SBAMs in a single backward pass.
            plain.out_layers[-1].weight.normal_(std=.01)
            plain.norm_0.time_gate[-1].weight.normal_(std=.01)
            plain.norm_1.time_gate[-1].weight.normal_(std=.01)
        checkpointed = copy.deepcopy(plain)
        checkpointed.use_checkpoint = True
        outputs, gradients = [], []
        for model in (plain, checkpointed):
            inputs = [value.clone().requires_grad_() for value in
                      (self.x, self.emb, self.condition)]
            output = model(*inputs)
            output.square().mean().backward()
            outputs.append(output.detach())
            gradients.append([value.grad for value in inputs] +
                             [p.grad for p in model.parameters()])
        torch.testing.assert_close(outputs[0], outputs[1])
        for actual, expected in zip(gradients[0], gradients[1]):
            self.assertIsNotNone(actual)
            self.assertIsNotNone(expected)
            torch.testing.assert_close(actual, expected)
        for module in (checkpointed.norm_0, checkpointed.norm_1):
            self.assertGreater(module.time_gate[-1].weight.grad.abs().sum().item(), 0)

    def test_gate_weights_round_trip_with_model_state(self):
        sbam = SBAM(32, 16, 128)
        with torch.no_grad():
            sbam.time_gate[-1].weight.normal_(std=.05)
        restored = SBAM(32, 16, 128)
        restored.load_state_dict(sbam.state_dict(), strict=True)
        torch.testing.assert_close(restored(self.x, self.condition, self.emb),
                                   sbam(self.x, self.condition, self.emb), rtol=0, atol=0)


if __name__ == '__main__':
    unittest.main()
