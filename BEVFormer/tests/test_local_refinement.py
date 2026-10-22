"""Focused CPU checks for v5_3 full-resolution local refinement."""

from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

import torch

from projects.bevdiffuser.layout_diffusion.seg_diffusion_unet_v5_3 import (
    DiffusionUNetModel, LocalRefinement,
)


def make_model(use_local_refinement=True):
    with redirect_stdout(StringIO()):
        return DiffusionUNetModel(
            seg_bev_aligner=dict(bev_h=6, bev_w=6, num_classes=3,
                                 emb_channels=32, channel_mult=[1, 1, 1]),
            in_channels=32, out_channels=32, model_channels=32,
            seg_channels=[32, 32, 32], num_res_blocks=1,
            attention_ds=[], channel_mult=[1, 1, 1],
            image_size=24, num_head_channels=32, resblock_updown=True,
            num_pre_downsample=2, use_global_seg=True,
            use_local_refinement=use_local_refinement,
            local_refinement_channels=64,
        )


def test_zero_initialization_preserves_nonzero_baseline_output():
    torch.manual_seed(17)
    baseline = make_model(False).eval()
    with torch.no_grad():
        baseline.out[-1].weight.normal_(std=.01)
    model = make_model(True).eval()
    result = model.load_state_dict(baseline.state_dict(), strict=False)
    assert not result.unexpected_keys
    assert set(result.missing_keys) == {key for key in model.state_dict()
                                        if key.startswith('local_refinement.')}
    assert baseline.local_refinement is None
    assert not any(key.startswith('local_refinement.') for key in baseline.state_dict())
    assert torch.count_nonzero(model.local_refinement.out.weight) == 0
    x = torch.randn(2, 32, 24, 24)
    probability = torch.rand(2, 4, 6, 6)
    with torch.no_grad():
        expected = baseline(x, torch.tensor([100, 100]), [{}, {}], semantic_prob=probability)
        actual = model(x, torch.tensor([100, 100]), [{}, {}], semantic_prob=probability)
    assert expected.abs().sum() > 0
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_full_resolution_shape_and_two_step_gradient_flow():
    torch.manual_seed(18)
    branch = LocalRefinement(32, 32, 128, channels=64)
    with torch.no_grad():
        full = branch(torch.randn(1, 32, 200, 200),
                      torch.randn(1, 32, 200, 200), torch.randn(1, 128))
    assert full.shape == (1, 32, 200, 200)
    optimizer = torch.optim.SGD(branch.parameters(), lr=.1)
    x = torch.randn(1, 32, 8, 8)
    coarse = torch.randn_like(x)
    emb = torch.randn(1, 128)
    optimizer.zero_grad()
    branch(x, coarse, emb).square().mean().backward()
    assert branch.out.weight.grad.abs().sum() > 0
    assert torch.count_nonzero(branch.noisy_proj.weight.grad) == 0
    optimizer.step()
    optimizer.zero_grad()
    x = x.detach().requires_grad_()
    emb = emb.detach().requires_grad_()
    branch(x, coarse, emb).square().mean().backward()
    for gradient in (x.grad, emb.grad, branch.noisy_proj.weight.grad,
                     branch.coarse_proj.weight.grad, branch.fuse.weight.grad,
                     branch.time_film[-1].weight.grad):
        assert gradient is not None
        assert torch.isfinite(gradient).all()
        assert gradient.abs().sum() > 0


def test_raw_and_preencoded_condition_match_after_strict_checkpoint_roundtrip(tmp_path):
    torch.manual_seed(19)
    model = make_model(True).eval()
    with torch.no_grad():
        model.out[-1].weight.normal_(std=.01)
        model.local_refinement.out.weight.normal_(std=.01)
        model.seg_global_embed[-1].weight.normal_(std=.01)
    checkpoint = tmp_path / 'v5_3_local_refinement.pth'
    torch.save(model.state_dict(), checkpoint)
    loaded = make_model(True).eval()
    loaded.load_state_dict(torch.load(checkpoint, map_location='cpu'), strict=True)
    x = torch.randn(2, 32, 24, 24)
    timestep = torch.tensor([100, 100])
    probability = torch.rand(2, 4, 6, 6)
    with torch.no_grad(), patch.object(loaded.seg_aligner, 'project_semantics',
                                       side_effect=AssertionError('Live projection')):
        for condition in (probability, torch.zeros_like(probability)):
            encoded = loaded.encode_seg(None, [{}, {}], semantic_prob=condition)
            raw = loaded(x, timestep, [{}, {}], semantic_prob=condition)
            preencoded = loaded(x, timestep, [{}, {}], seg_bev_maps=encoded)
            original = model(x, timestep, [{}, {}], semantic_prob=condition)
            assert raw.abs().sum() > 0
            torch.testing.assert_close(raw, preencoded, rtol=0, atol=0)
            torch.testing.assert_close(raw, original, rtol=0, atol=0)
