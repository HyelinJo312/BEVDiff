"""Tests for the Stage 2 DiffBEVDet4D port.

Run from BEVDiff/BEVDet with jhl-bevdet (GPU tests need one CUDA device; bev_pool_v2 is CUDA-only):
    CUDA_VISIBLE_DEVICES=<id> python -m pytest projects/bevdet_plugin/tests/test_diff_bevdet4d.py -q
Weights are random (no checkpoint needed); equivalence tests copy one state dict into both models.
"""
import copy
import importlib.util
import itertools
import os
import os.path as osp
import sys

import numpy as np
import pytest
import torch

BEVDET_ROOT = osp.abspath(osp.join(osp.dirname(__file__), '../../..'))
for p in (BEVDET_ROOT, osp.join(BEVDET_ROOT, 'tools')):
    if p not in sys.path:
        sys.path.insert(0, p)
os.chdir(BEVDET_ROOT)

from mmcv import Config  # noqa: E402

STAGE2_CFG = 'projects/configs/diff_bevdet/diff_bevdet4d_r50_cbgs_onlyseg_mgd.py'
STAGE1_CFG = 'projects/configs/bevdiffuser/bevdet4d_r50_onlyseg_sam_v2_metric3d.py'
BASE_CFG = 'configs/bevdet/bevdet-r50-4d-cbgs.py'
# C0 recipes and the Stage 2 config / target / teacher that must match each of them
RECIPES = {
    'cbgs': dict(stage2=STAGE2_CFG, base=BASE_CFG, wrapper='CBGSDataset', epochs=20,
                 c0_run='bevdet4d-r50-cbgs_seed0',
                 stage1_run='BEVDiffuser_bevdet4d_r50_cbgs_onlyseg_sam3_metric3d'),
    'nocbgs': dict(stage2='projects/configs/diff_bevdet/diff_bevdet4d_r50_nocbgs_onlyseg_mgd.py',
                   base='configs/bevdet/bevdet-r50-4d.py', wrapper='NuScenesDataset', epochs=24,
                   c0_run='bevdet4d-r50-nocbgs_seed0',
                   stage1_run='BEVDiffuser_bevdet4d_r50_nocbgs_onlyseg_sam3_metric3d'),
}
TRAIN_INFO = 'data/nuscenes/bevdetv3-nuscenes_infos_train.pkl'
STUDENT_ONLY = ('mgd_generation.',)

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a CUDA device')
needs_data = pytest.mark.skipif(not osp.isfile(TRAIN_INFO), reason='nuScenes infos not found')


def _norm(v):
    if isinstance(v, (list, tuple)):
        return [_norm(x) for x in v]
    if isinstance(v, dict):
        return {k: _norm(x) for k, x in v.items()}
    return v


def _fingerprint(module):
    with torch.no_grad():
        return float(sum(t.double().sum() for t in itertools.chain(module.parameters(), module.buffers())
                         if t.is_floating_point()))


@pytest.fixture(scope='module')
def cfgs():
    return Config.fromfile(STAGE2_CFG), Config.fromfile(BASE_CFG), Config.fromfile(STAGE1_CFG)


@pytest.fixture(scope='module')
def models(cfgs):
    """(BEVDet4D, DiffBEVDet4D) with identical weights on the GPU, eval mode."""
    from mmdet3d.models import build_model
    stage2, base, _ = cfgs
    torch.manual_seed(0)
    ref = build_model(base.model)
    ref.init_weights()
    diff = build_model(stage2.model)
    missing, unexpected = diff.load_state_dict(ref.state_dict(), strict=False)
    assert not unexpected and missing and all(k.startswith(STUDENT_ONLY) for k in missing)
    return ref.cuda().eval(), diff.cuda().eval()


@pytest.fixture(scope='module')
def batch(cfgs):
    """Two real training samples (C0 augmentation + semantic prior), scattered to the GPU."""
    from mmcv.parallel import collate, scatter
    from mmdet3d.datasets import build_dataset
    stage2 = cfgs[0]
    ds = build_dataset(stage2.data.train.dataset)  # inner NuScenesDataset (no CBGS indexing needed)
    np.random.seed(0)
    data = collate([ds[i] for i in (11, 4021)], samples_per_gpu=2)
    return scatter(data, [torch.cuda.current_device()])[0]


def _eq_losses(a, b, rtol=1e-4, atol=1e-5):
    assert set(a) == set(b)
    for k in a:
        assert torch.allclose(torch.as_tensor(a[k]).float(), torch.as_tensor(b[k]).float(),
                              rtol=rtol, atol=atol), k


def _train_inner(cfg):
    """data.train with the CBGS wrapper (if any) unwrapped."""
    train = cfg.data.train
    return train.dataset if train.type == 'CBGSDataset' else train


def _cfg_dict(cfg):
    return _norm(cfg._cfg_dict.to_dict())


# ---------------------------------------------------------------- config wiring (CPU)
def test_c0_nocbgs_config_differs_only_in_sampling_and_schedule():
    cbgs = Config.fromfile(RECIPES['cbgs']['base'])
    nocbgs = Config.fromfile(RECIPES['nocbgs']['base'])
    a, b = _cfg_dict(cbgs), _cfg_dict(nocbgs)
    assert a['data']['train']['type'] == 'CBGSDataset' and b['data']['train']['type'] == 'NuScenesDataset'
    assert b['data']['train'] == a['data']['train']['dataset']  # wrapper removed, inner dataset unchanged
    assert (a['runner']['max_epochs'], a['lr_config']['step']) == (20, [20])
    assert (b['runner']['max_epochs'], b['lr_config']['step']) == (24, [24])
    for d in (a, b):
        del d['data']['train'], d['runner']['max_epochs'], d['lr_config']['step']
    assert a == b


def test_stage2_recipes_differ_only_in_c0_recipe():
    a = _cfg_dict(Config.fromfile(RECIPES['cbgs']['stage2']))
    b = _cfg_dict(Config.fromfile(RECIPES['nocbgs']['stage2']))
    assert b['data']['train'] == a['data']['train']['dataset']
    for d in (a, b):
        del (d['data']['train'], d['runner']['max_epochs'], d['lr_config']['step'], d['load_from'],
             d['bev_diffuser_cfg']['unet_checkpoint_dir'])
    assert a == b


@pytest.mark.parametrize('recipe', sorted(RECIPES))
def test_stage2_config_matches_c0_and_stage1(recipe):
    r = RECIPES[recipe]
    stage2, base, stage1 = (Config.fromfile(r['stage2']), Config.fromfile(r['base']),
                            Config.fromfile(STAGE1_CFG))
    # UNet = Stage 1 UNet (only the import path of `type` differs)
    u2, u1 = _norm(dict(stage2.unet)), _norm(dict(stage1.unet))
    assert u2['type'] == 'projects.bevdiffuser.' + u1['type'] and u2['parameters'] == u1['parameters']
    # C0 recipe: same dataset wrapper, dataset args, model (+ student-only keys), schedule, hooks
    assert stage2.data.train.type == base.data.train.type == r['wrapper']
    inner2 = {k: v for k, v in _norm(dict(_train_inner(stage2))).items() if k != 'pipeline'}
    inner0 = {k: v for k, v in _norm(dict(_train_inner(base))).items() if k != 'pipeline'}
    assert inner2 == inner0
    assert _norm(list(_train_inner(stage2).pipeline)) == _norm(list(stage1.data.train.pipeline))
    m2, m0 = _norm(dict(stage2.model)), _norm(dict(base.model))
    assert m2.pop('type') == 'DiffBEVDet4D' and m0.pop('type') == 'BEVDet4D'
    extra = {k: m2.pop(k) for k in ('use_mgd', 'mgd_alpha', 'mgd_lambda', 'loss_mix_weight')}
    assert extra == dict(use_mgd=True, mgd_alpha=100, mgd_lambda=0.55, loss_mix_weight=0.5)
    assert m2 == m0
    for key in ('optimizer', 'optimizer_config', 'lr_config', 'custom_hooks', 'checkpoint_config'):
        assert _norm(stage2[key]) == _norm(base[key]), key
    assert (stage2.data.samples_per_gpu, stage2.data.workers_per_gpu) == (base.data.samples_per_gpu,
                                                                          base.data.workers_per_gpu)
    assert base.runner.max_epochs == r['epochs']
    assert stage2.runner == dict(type='DiffEpochBasedRunner', max_epochs=r['epochs'])
    assert _norm(stage2.data.val) == _norm(base.data.val) and _norm(stage2.data.test) == _norm(base.data.test)
    # flagship teacher settings
    t = stage2.bev_diffuser_cfg
    assert (t.noise_timesteps, t.denoise_timesteps, t.num_inference_steps, t.cfg_scale,
            t.use_classifier_guidence, t.prediction_type) == (100, 100, 5, 2, False, 'sample')
    # target = this recipe's C0 EMA; teacher = the Stage 1 run on that target
    assert stage2.load_from.endswith(f"{r['c0_run']}/epoch_{r['epochs']}_ema.pth")
    assert t.unet_checkpoint_dir.endswith(f"{r['stage1_run']}/checkpoint-50000")


def test_mgd_mask_ratio():
    from projects.bevdet_plugin.detectors.diff_bevdet4d import sample_mgd_keep_mask
    torch.manual_seed(0)
    keep = sample_mgd_keep_mask(torch.zeros(4, 256, 64, 64), 0.55)
    assert tuple(keep.shape) == (4, 1, 64, 64)  # one mask per cell, shared over channels
    assert abs(float(1 - keep.mean()) - 0.55) < 0.02


# ---------------------------------------------------------------- GPU: no-teacher equivalence
@needs_gpu
@needs_data
def test_without_teacher_matches_bevdet4d(models, batch):
    ref, diff = models
    with torch.no_grad():
        _eq_losses(ref.forward_train(**batch), diff.forward_train(**batch))


@needs_gpu
@needs_data
def test_only_bev_and_given_bev_flags(models, batch):
    ref, diff = models
    with torch.no_grad():
        bev = diff(return_loss=False, only_bev=True, img_inputs=batch['img_inputs'],
                   img_metas=batch['img_metas'])
        ref_bev = ref.extract_img_feat(batch['img_inputs'], batch['img_metas'])[0][0]
        assert tuple(bev.shape) == (2, 256, 128, 128) and torch.allclose(bev, ref_bev, atol=1e-5)
        _eq_losses(diff.forward_train(**batch), diff.forward_train(**batch, given_bev=bev))
        # test path: identical to BEVDet4D, and given_bev reproduces it without the image branch
        test_kwargs = dict(img_inputs=[batch['img_inputs']], img_metas=[batch['img_metas']])
        out_ref = ref(return_loss=False, **test_kwargs)
        out_diff = diff(return_loss=False, **test_kwargs)
        out_given = diff.simple_test(None, batch['img_metas'], img=batch['img_inputs'], given_bev=bev)
    for a, b, c in zip(out_ref, out_diff, out_given):
        for o in (b, c):
            assert torch.allclose(a['pts_bbox']['scores_3d'], o['pts_bbox']['scores_3d'], atol=1e-5)
            assert torch.allclose(a['pts_bbox']['boxes_3d'].tensor, o['pts_bbox']['boxes_3d'].tensor, atol=1e-4)


# ---------------------------------------------------------------- GPU: teacher path
@needs_gpu
@needs_data
def test_train_step_with_teacher(cfgs, models, batch, tmp_path):
    from safetensors.torch import save_file
    from bevdiffuser_onlyseg import BEVDiffuser
    from projects.bevdiffuser.model_utils import build_unet_v2
    stage2 = cfgs[0]
    _, diff = models
    student = copy.deepcopy(diff).train()
    target = copy.deepcopy(diff).eval().requires_grad_(False)
    torch.manual_seed(0)
    unet = build_unet_v2(stage2.unet)
    (tmp_path / 'unet').mkdir()
    save_file(unet.state_dict(), str(tmp_path / 'unet' / 'diffusion_pytorch_model.safetensors'))
    teacher_cfg = dict(stage2.bev_diffuser_cfg, unet_cfg=stage2.unet, unet_checkpoint_dir=str(tmp_path))
    teacher = BEVDiffuser(**teacher_cfg).cuda().eval()
    teacher.requires_grad_(False)
    fp_target, fp_teacher = _fingerprint(target), _fingerprint(teacher)

    out = student.train_step(batch, None, model_target=target, bev_diffuser=teacher)
    log = out['log_vars']
    assert {'loss_bev', 'task_loss', 'loss'} <= set(log) and out['num_samples'] == 2
    assert abs(log['loss'] - (0.5 * log['task_loss'] + 0.5 * log['loss_bev'])) < 1e-4 * max(1, log['loss'])
    task_keys = [k for k in log if 'loss' in k and k not in ('loss', 'task_loss', 'loss_bev')]
    assert abs(sum(log[k] for k in task_keys) - log['task_loss']) < 1e-3 * max(1, log['task_loss'])

    out['loss'].backward()
    assert student.img_backbone.conv1.weight.grad is not None
    assert student.mgd_generation[0].weight.grad is not None
    neck_grads = [p.grad for p in student.img_bev_encoder_neck.parameters() if p.requires_grad]
    assert neck_grads and all(g is not None for g in neck_grads) and sum(float(g.abs().sum()) for g in neck_grads) > 0
    assert all(p.grad is None for p in itertools.chain(target.parameters(), teacher.parameters()))
    assert _fingerprint(target) == fp_target and _fingerprint(teacher) == fp_teacher


# ---------------------------------------------------------------- frozen teacher builder (CPU)
def test_build_frozen_teacher_loads_target_only(cfgs, tmp_path):
    import logging
    from safetensors.torch import save_file
    from mmdet3d.models import build_model
    from projects.bevdiffuser.model_utils import build_unet_v2
    spec = importlib.util.spec_from_file_location('train_onlyseg', osp.join(BEVDET_ROOT, 'tools/train_onlyseg.py'))
    train_onlyseg = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(train_onlyseg)

    stage2 = copy.deepcopy(cfgs[0])
    torch.manual_seed(0)
    student = build_model(stage2.model)
    target_state = {k: v + 1.0 if v.is_floating_point() else v for k, v in student.state_dict().items()
                    if not k.startswith(STUDENT_ONLY)}
    ckpt = tmp_path / 'target_ema.pth'
    torch.save({'epoch': 19, 'state_dict': target_state, 'updates': 0}, ckpt)
    unet_dir = tmp_path / 'stage1'
    (unet_dir / 'unet').mkdir(parents=True)
    save_file(build_unet_v2(stage2.unet).state_dict(), str(unet_dir / 'unet' / 'diffusion_pytorch_model.safetensors'))
    stage2.load_from = str(ckpt)
    stage2.bev_diffuser_cfg.unet_cfg = stage2.unet
    stage2.bev_diffuser_cfg.unet_checkpoint_dir = str(unet_dir)
    stage2.bev_diffuser_cfg.pretrained_model_name_or_path = 'projects/bevdiffuser/sd21_scheduler'
    before = copy.deepcopy(student.state_dict())

    target, teacher = train_onlyseg.build_frozen_teacher(stage2, student, logging.getLogger('test'))
    assert stage2.load_from is None  # the runner must not load it into the student
    k = 'img_backbone.conv1.weight'
    assert torch.equal(target.state_dict()[k], target_state[k])
    assert torch.equal(student.state_dict()[k], before[k])
    assert not any(p.requires_grad for p in target.parameters()) and not target.training
    assert target.with_prev and not any(p.requires_grad for p in teacher.parameters())

    bad = dict(target_state)
    bad.pop(k)
    torch.save({'state_dict': bad}, ckpt)
    stage2.load_from = str(ckpt)
    with pytest.raises(RuntimeError):
        train_onlyseg.build_frozen_teacher(stage2, student, logging.getLogger('test'))
