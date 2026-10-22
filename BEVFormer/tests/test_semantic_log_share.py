"""Optional log-share transform at the learned semantic encoder boundary."""

import math
from unittest.mock import patch

import torch

from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner


def make_aligner(use_log_share=False):
    return SegBEVAligner(bev_h=8, bev_w=8, num_classes=16,
                         emb_channels=32, channel_mult=[1, 1, 1],
                         use_log_share=use_log_share, log_share_eps=0.01)


def test_disabled_encoder_matches_legacy_path_bit_exact():
    torch.manual_seed(9)
    aligner = make_aligner(False).eval()
    probability = torch.rand(2, 17, 8, 8)
    with torch.no_grad():
        expected = aligner.seg_bev_encoder(
            aligner.prob_to_emb(probability.to(aligner.prob_to_emb[0].weight.dtype))
            + aligner.bev_pos_embed)
        actual = aligner.encode_prob(probability)
    assert expected.keys() == actual.keys()
    for scale in expected:
        torch.testing.assert_close(actual[scale], expected[scale], rtol=0, atol=0)


def test_enabled_formula_endpoints_shape_no_mutation_and_gradient():
    torch.manual_seed(10)
    aligner = make_aligner(True)
    probability = torch.zeros(2, 17, 8, 8)
    probability[:, 0] = 1.0
    probability[:, 1] = 0.01
    original = probability.clone()
    captured = []
    handle = aligner.prob_to_emb[0].register_forward_pre_hook(
        lambda module, inputs: captured.append(inputs[0].detach().clone()))
    try:
        output = aligner.encode_prob(probability)
    finally:
        handle.remove()
    expected = torch.log1p(probability.float() / 0.01) / math.log1p(1.0 / 0.01)
    torch.testing.assert_close(captured[0], expected)
    torch.testing.assert_close(captured[0][:, 0], torch.ones_like(captured[0][:, 0]))
    assert torch.count_nonzero(captured[0][:, 2:]) == 0
    assert captured[0].shape == probability.shape
    assert output[1].shape[:2] == (2, 32)
    torch.testing.assert_close(probability, original, rtol=0, atol=0)

    differentiable = torch.rand(2, 17, 8, 8, requires_grad=True)
    loss = sum(value.square().mean() for value in aligner.encode_prob(differentiable).values())
    loss.backward()
    assert differentiable.grad is not None
    assert torch.isfinite(differentiable.grad).all()
    assert differentiable.grad.abs().sum() > 0


def test_cached_and_live_share_path_and_checkpoint_keys():
    torch.manual_seed(11)
    aligner = make_aligner(True).eval()
    legacy = make_aligner(False).eval()
    assert {key: value.shape for key, value in aligner.state_dict().items()} == {
        key: value.shape for key, value in legacy.state_dict().items()}
    probability = torch.rand(2, 17, 8, 8)
    support = torch.ones(2, 1, 8, 8)
    with torch.no_grad(), patch.object(aligner, 'project_semantics',
                                      return_value=(probability.clone(), support)) as project:
        cached = aligner(None, [{}, {}], semantic_prob=probability)
        live = aligner(None, [{}, {}], depth_maps=torch.ones(2, 1, 1, 1))
    project.assert_called_once()
    for scale in cached:
        torch.testing.assert_close(cached[scale], live[scale], rtol=0, atol=0)
