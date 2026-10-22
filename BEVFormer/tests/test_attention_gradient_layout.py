"""The self-attention output convolution receives contiguous BCHW activations."""

import copy

import pytest
import torch
from einops import rearrange

from projects.bevdiffuser.layout_diffusion import seg_diffusion_unet_v5 as v5
from projects.bevdiffuser.layout_diffusion import seg_diffusion_unet_v5_3 as v5_3


def original_forward(module, value):
    """SelfAttnTransformer forward before the BCHW layout correction."""
    height, width = value.shape[-2:]
    hidden = module.proj_in(module.norm(value))
    hidden = rearrange(hidden, 'b c h w -> b (h w) c')
    for block in module.transformer_blocks:
        hidden = block(hidden)
    hidden = rearrange(hidden, 'b (h w) c -> b c h w', h=height, w=width)
    return module.proj_out(hidden) + value


@pytest.mark.parametrize('variant', [v5, v5_3], ids=['v5', 'v5_3'])
@pytest.mark.parametrize('shape', [(4, 4), (3, 5)], ids=['square', 'rectangular'])
def test_contiguous_projection_preserves_forward_and_gradients(variant, shape):
    torch.manual_seed(23)
    fixed = variant.SelfAttnTransformer(32, n_heads=4, d_head=8)
    with torch.no_grad():
        fixed.proj_out.weight.normal_(std=0.05)
        fixed.proj_out.bias.normal_(std=0.05)
    original = copy.deepcopy(fixed)
    assert set(fixed.state_dict()) == set(original.state_dict())

    source = torch.randn(2, 32, *shape)
    reference_input = source.clone().requires_grad_()
    fixed_input = source.clone().requires_grad_()
    target = torch.randn_like(source)
    layouts = {'original': [], 'fixed': []}
    old_hook = original.proj_out.register_forward_pre_hook(
        lambda module, inputs: layouts['original'].append((inputs[0].stride(), inputs[0].is_contiguous())))
    new_hook = fixed.proj_out.register_forward_pre_hook(
        lambda module, inputs: layouts['fixed'].append((inputs[0].stride(), inputs[0].is_contiguous())))
    try:
        reference_output = original_forward(original, reference_input)
        fixed_output = fixed(fixed_input)
    finally:
        old_hook.remove()
        new_hook.remove()

    assert not layouts['original'][0][1]
    assert layouts['fixed'][0][1]
    channels = fixed.proj_out.in_channels
    height, width = shape
    assert layouts['fixed'][0][0] == (channels * height * width, height * width, width, 1)
    assert (fixed_output - fixed_input).abs().sum() > 0
    torch.testing.assert_close(fixed_output, reference_output, rtol=1e-5, atol=1e-6)

    (reference_output * target).sum().backward()
    (fixed_output * target).sum().backward()
    torch.testing.assert_close(fixed_input.grad, reference_input.grad, rtol=1e-5, atol=1e-6)
    for (name, old_parameter), (fixed_name, new_parameter) in zip(
            original.named_parameters(), fixed.named_parameters()):
        assert name == fixed_name
        assert old_parameter.grad is not None and new_parameter.grad is not None
        torch.testing.assert_close(new_parameter.grad, old_parameter.grad, rtol=1e-5, atol=1e-6)
    assert fixed.proj_out.weight.grad.stride() == fixed.proj_out.weight.stride()


@pytest.mark.parametrize('variant', [v5, v5_3], ids=['v5', 'v5_3'])
def test_zero_initialized_projection_remains_residual_identity(variant):
    module = variant.SelfAttnTransformer(32, n_heads=4, d_head=8).eval()
    source = torch.randn(1, 32, 3, 5)
    with torch.no_grad():
        output = module(source)
    torch.testing.assert_close(output, source, rtol=0, atol=0)
