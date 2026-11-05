"""Tests for the BEVDet4D port of the semantic-only Stage 1 teacher (CPU only).

Run from BEVDiff/BEVDet with the jhl-bevdet env:
    python -m pytest projects/bevdiffuser/tests/test_bevdet_stage1.py -q
Data-dependent tests are skipped when the nuScenes infos / semantic cache / flagship
checkpoint are not present.
"""
import copy
import importlib
import os
import os.path as osp
import sys

import numpy as np
import pytest
import torch

BEVDET_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
if BEVDET_ROOT not in sys.path:
    sys.path.insert(0, BEVDET_ROOT)
os.chdir(BEVDET_ROOT)  # config paths are relative to BEVDiff/BEVDet

from mmcv import Config  # noqa: E402

from projects.bevdiffuser.bevdet_semantic_prior import (  # noqa: E402
    LoadSemanticBEVPrior, lidar_to_key_ego, warp_bev)
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v3 import SegBEVEncoder  # noqa: E402

CFG = 'projects/configs/bevdiffuser/bevdet4d_r50_onlyseg_sam_v2_metric3d.py'
BASE_CFG = 'configs/bevdet/bevdet-r50-4d-cbgs.py'
FLAGSHIP_CFG = 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py'
FLAGSHIP_UNET = ('../results/bevdiffuser/det/BEVDiffuser_tiny_onlyseg_sam3_metric3d/'
                 'checkpoint-50000/unet/diffusion_pytorch_model.safetensors')
CACHE_ROOT = '../BEVFormer/data/semantic_bev'
TRAIN_INFO = 'data/nuscenes/bevdetv3-nuscenes_infos_train.pkl'
R = 51.2
CACHE_RANGE = (-R, -R, R, R)
VEHICLE_CH = 1  # cache channel with the largest lift at GT car centres (measured on val)


def _build_unet(params):
    mod, cls = 'layout_diffusion.seg_diffusion_unet_v5', 'DiffusionUNetModel'
    module = importlib.import_module('projects.bevdiffuser.' + mod)
    return getattr(module, cls)(**copy.deepcopy(dict(params)))


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


# ---------------------------------------------------------------- F3: ds4 padding flag
def test_segbev_encoder_ds4_padding():
    torch.manual_seed(0)
    enc0 = SegBEVEncoder(256, channel_mult=(1, 1, 1))
    enc1 = SegBEVEncoder(256, channel_mult=(1, 1, 1), ds4_padding=1)
    assert enc0.down2[0].padding == (0, 0) and enc1.down2[0].padding == (1, 1)
    sd0, sd1 = enc0.state_dict(), enc1.state_dict()
    assert list(sd0) == list(sd1) and all(sd0[k].shape == sd1[k].shape for k in sd0)
    with torch.no_grad():
        flagship = {k: tuple(v.shape[-2:]) for k, v in enc0(torch.zeros(1, 256, 50, 50)).items()}
        bevdet = {k: tuple(v.shape[-2:]) for k, v in enc1(torch.zeros(1, 256, 64, 64)).items()}
    assert flagship == {1: (50, 50), 2: (25, 25), 4: (12, 12)}
    assert bevdet == {1: (64, 64), 2: (32, 32), 4: (16, 16)}
    with pytest.raises(ValueError):
        SegBEVEncoder(256, ds4_padding=2)


@pytest.mark.skipif(not osp.isfile(FLAGSHIP_UNET), reason='flagship checkpoint not found')
def test_flagship_checkpoint_still_loads_strictly():
    from safetensors.torch import load_file
    net = _build_unet(Config.fromfile(FLAGSHIP_CFG).unet.parameters)
    assert net.seg_aligner.seg_bev_encoder.down2[0].padding == (0, 0)
    missing, unexpected = net.load_state_dict(load_file(FLAGSHIP_UNET), strict=True)
    assert not missing and not unexpected


def test_bevdet_unet_shapes():
    cfg = Config.fromfile(CFG)
    net = _build_unet(cfg.unet.parameters).eval()
    prior = torch.rand(1, 17, cfg.prior_h_, cfg.prior_w_)
    with torch.no_grad():
        scales = net.encode_seg(None, [{}], semantic_prob=prior)
        out = net(torch.randn(1, 256, cfg.bev_h_, cfg.bev_w_), torch.tensor([100]), [{}],
                  semantic_prob=prior)
    assert {k: tuple(v.shape[-2:]) for k, v in scales.items()} == {1: (64, 64), 2: (32, 32), 4: (16, 16)}
    assert tuple(out.shape) == (1, 256, 128, 128)


# ---------------------------------------------------------------- config wiring
def test_unet_config_matches_flagship_except_resolution():
    ours = _norm(dict(Config.fromfile(CFG).unet))
    flag = _norm(dict(Config.fromfile(FLAGSHIP_CFG).unet))
    assert ours['type'] == flag['type']
    p, q = ours['parameters'], flag['parameters']
    a, b = p.pop('seg_bev_aligner'), q.pop('seg_bev_aligner')
    allowed = {'image_size', 'num_pre_downsample'}
    assert {k for k in set(p) | set(q) if p.get(k) != q.get(k)} == allowed
    assert (p['image_size'], p['num_pre_downsample']) == (128, 1)
    assert {k for k in set(a) | set(b) if a.get(k) != b.get(k)} == {'bev_h', 'bev_w', 'ds4_padding'}
    assert (a['bev_h'], a['bev_w'], a['ds4_padding']) == (64, 64, 1)


def test_train_pipeline_is_base_plus_prior_without_cbgs():
    ours, base = Config.fromfile(CFG), Config.fromfile(BASE_CFG)
    assert base.data.train.type == 'CBGSDataset'
    assert ours.data.train.type == 'NuScenesDataset' and 'dataset' not in ours.data.train
    pipe = _norm(list(ours.data.train.pipeline))
    names = [t['type'] for t in pipe]
    assert names.index('LoadSemanticBEVPrior') == names.index('BEVAug') + 1
    collect = pipe[-1]
    assert collect['type'] == 'Collect3D' and collect['keys'][-1] == 'semantic_bev_probabilities'
    stripped = [t for t in pipe if t['type'] != 'LoadSemanticBEVPrior']
    stripped[-1] = dict(stripped[-1], keys=collect['keys'][:-1])
    assert stripped == _norm(list(base.data.train.dataset.pipeline))
    ours_args = {k: v for k, v in _norm(dict(ours.data.train)).items() if k != 'pipeline'}
    base_args = {k: v for k, v in _norm(dict(base.data.train.dataset)).items() if k != 'pipeline'}
    assert ours_args == base_args
    assert _norm(dict(ours.model)) == _norm(dict(base.model))


# ---------------------------------------------------------------- F2: warp
def test_warp_identity_is_exact():
    prob = torch.rand(17, 50, 50)
    out = warp_bev(prob, np.eye(4), CACHE_RANGE, CACHE_RANGE, (50, 50))
    assert torch.allclose(out, prob, atol=1e-5)


def test_warp_moves_hot_cell_to_transformed_position():
    cell = 2 * R / 50
    ix, iy = 37, 12
    src_xy = np.array([-R + (ix + 0.5) * cell, -R + (iy + 0.5) * cell])
    prob = torch.zeros(1, 50, 50)
    prob[0, iy, ix] = 1.0
    a = np.deg2rad(30.0)
    t = np.eye(4)
    t[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    t[:2, 3] = [1.3, -2.1]
    flip = np.diag([-1.0, 1.0, 1.0, 1.0])  # flip_dx as in BEVAug
    src2dst = flip @ t
    dst_xy = (src2dst @ np.r_[src_xy, 0.0, 1.0])[:2]
    out = warp_bev(prob, src2dst, CACHE_RANGE, CACHE_RANGE, (64, 64))[0]
    r, c = np.unravel_index(int(out.argmax()), out.shape)
    centre = np.array([-R + (c + 0.5) * 1.6, -R + (r + 0.5) * 1.6])
    assert np.linalg.norm(centre - dst_xy) <= 1.6
    assert out.sum() > 0.3  # mass is spread but not lost (target well inside the grid)


def test_warp_preserves_share_bounds():
    prob = torch.rand(17, 50, 50)
    prob = prob / prob.sum(0, keepdim=True)
    a = np.deg2rad(-17.0)
    t = np.eye(4)
    t[:2, :2] = [[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]]
    out = warp_bev(prob, 1.03 * t, CACHE_RANGE, CACHE_RANGE, (64, 64))
    assert out.min() >= 0 and out.sum(0).max() <= 1 + 1e-5


# ---------------------------------------------------------------- real data
needs_data = pytest.mark.skipif(
    not (osp.isfile(TRAIN_INFO) and osp.isfile(osp.join(CACHE_ROOT, 'manifest.json'))),
    reason='nuScenes infos or semantic cache not found')


@needs_data
def test_pipeline_prior_aligns_with_augmented_gt_and_collates():
    """End-to-end on the real train pipeline (random image aug + BEV aug): GT car centres
    (BEVDet's own targets, after BEVAug) must land on the cached vehicle channel."""
    from mmcv.parallel import collate
    from mmdet3d.datasets import build_dataset
    cfg = Config.fromfile(CFG)
    ds = build_dataset(cfg.data.train)
    rng = np.random.default_rng(0)
    idx = [int(i) for i in rng.choice(len(ds), 24, replace=False)]
    loader = torch.utils.data.DataLoader(torch.utils.data.Subset(ds, idx), batch_size=2,
                                         num_workers=8, collate_fn=lambda b: collate(b, 2))
    hits, flipped, n_cars, coverage = [], [], 0, []
    for batch in loader:
        imgs = batch['img_inputs'][0]
        assert len(batch['img_inputs']) == 7 and tuple(imgs.shape[1:]) == (12, 3, 256, 704)
        priors = batch['semantic_bev_probabilities'].data[0]
        assert len(priors) == 2 and all(tuple(p.shape) == (17, 64, 64) for p in priors)
        for prior, boxes, labels in zip(priors, batch['gt_bboxes_3d'].data[0],
                                        batch['gt_labels_3d'].data[0]):
            coverage.append(float((prior.sum(0) > 0).float().mean()))
            xy = boxes.gravity_center[labels == 0, :2].numpy()
            xy = xy[np.abs(xy).max(1) < R - 2]
            if len(xy) == 0:
                continue
            g = torch.as_tensor(xy / R, dtype=torch.float32).view(1, 1, -1, 2)
            for p, store in ((prior, hits), (prior.flip(-1), flipped)):
                v = torch.nn.functional.grid_sample(p[None], g, align_corners=False)[0, VEHICLE_CH, 0]
                store.append(v.numpy())
            n_cars += len(xy)
    hit = float((np.concatenate(hits) > 0.1).mean())
    neg = float((np.concatenate(flipped) > 0.1).mean())
    print(f'cars={n_cars} hit={hit:.3f} x-flipped control={neg:.3f} '
          f'observed coverage mean={np.mean(coverage):.3f} min={np.min(coverage):.3f}')
    assert n_cars >= 50
    assert hit >= 0.7 and hit > neg + 0.3


@needs_data
def test_lidar_to_key_ego_uses_first_camera_ego():
    import pickle
    infos = pickle.load(open('data/nuscenes/bevdetv3-nuscenes_infos_val.pkl', 'rb'))['infos']
    info = infos[100]
    t = lidar_to_key_ego(info, 'CAM_FRONT_LEFT')
    # LIDAR_TOP sits ~1.8 m above the ego origin and is yawed ~90 deg relative to it.
    assert 1.0 < t[2, 3] < 2.5
    assert abs(abs(np.degrees(np.arctan2(t[1, 0], t[0, 0]))) - 90) < 5
    prior = LoadSemanticBEVPrior(CACHE_ROOT)
    assert prior.layout == 'share' and prior.cache_range == CACHE_RANGE
