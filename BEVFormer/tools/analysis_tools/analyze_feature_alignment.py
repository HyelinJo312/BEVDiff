"""Global BEV feature alignment analysis for Experiment 4.

This script compares the feature that is actually optimized by each
distillation objective against its denoised diffusion teacher feature on paired
validation frames. For direct MSE setups this is the raw student BEV feature;
for MGD setups this is the learned MGD reconstruction output. It also computes
the before-distillation gap using the initial BEVFormer checkpoint.

Region-wise metrics and visualization are intentionally left out for a later
analysis pass.
"""

import argparse
import copy
import csv
import json
import os
import random
import sys
import time
from collections import OrderedDict

import numpy as np
import torch
import torch.nn.functional as F

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_TOOLS_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_BEVFORMER_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.append(_BEVFORMER_DIR)
sys.path.append(_TOOLS_DIR)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.core.bbox import LiDARInstance3DBoxes
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.bevdiffuser  # noqa: F401
import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


DEFAULT_SETUPS = OrderedDict([
    ('baseline_mse', dict(
        label='BEVDiffuser teacher',
        distill_loss='MSE',
        mode='baseline',
        config='../results/version2/stage2/'
               'DiffBEVFormer_tiny_original_24epoch/layout_tiny.py',
        ckpt='../results/version2/stage2/'
             'DiffBEVFormer_tiny_original_24epoch/epoch_24.pth',
        map=0.28837804496354535,
        nds=0.3870849757916887,
    )),
    ('semantic_mse', dict(
        label='Semantic-only teacher',
        distill_loss='MSE',
        mode='semantic',
        config='../results/version2/stage2/'
               'DiffBEVFormer_tiny_onlyseg_sam_no-mgd_t100/'
               'bev_tiny_onlyseg_sam_v2.py',
        ckpt='../results/version2/stage2/'
             'DiffBEVFormer_tiny_onlyseg_sam_no-mgd_t100/epoch_24.pth',
        map=0.2801796288733448,
        nds=0.3846135432556416,
    )),
    ('semantic_mgd', dict(
        label='Semantic-only teacher',
        distill_loss='MGD',
        mode='semantic',
        config='../results/version2/stage2/'
               'DiffBEVFormer_tiny_onlyseg_sam_mgd_v4_2_t100_alpha100_lambda_0.6/'
               'bev_tiny_onlyseg_sam_v2.py',
        ckpt='../results/version2/stage2/'
             'DiffBEVFormer_tiny_onlyseg_sam_mgd_v4_2_t100_alpha100_lambda_0.6/'
             'epoch_24.pth',
        map=0.29184324074836254,
        nds=0.3892576162403079,
    )),
])


DETERMINISTIC_PIPELINE_TEMPLATE = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter',
         point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
]


def make_pipeline(class_names):
    pipe = []
    for step in DETERMINISTIC_PIPELINE_TEMPLATE:
        pipe.append(copy.deepcopy(step))
    pipe.insert(3, dict(type='ObjectNameFilter', classes=class_names))
    pipe.insert(-1, dict(type='DefaultFormatBundle3D',
                         class_names=class_names))
    return pipe


def resolve_path(path, base_dir):
    if path is None:
        return None
    if os.path.isabs(path):
        return path
    return os.path.abspath(os.path.join(base_dir, path))


def set_reproducible_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class PairAlignmentMeter:
    """Streaming MSE, cosine, and centered linear CKA for paired features."""

    def __init__(self, channels):
        self.channels = channels
        self.n = 0
        self.sse = 0.0
        self.cos_sum = 0.0
        self.sum_x = torch.zeros(channels, dtype=torch.float64)
        self.sum_y = torch.zeros(channels, dtype=torch.float64)
        self.xtx = torch.zeros(channels, channels, dtype=torch.float64)
        self.yty = torch.zeros(channels, channels, dtype=torch.float64)
        self.xty = torch.zeros(channels, channels, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x, y):
        """Update with x/y shaped (B, HW, C), on any device."""
        x = x.detach().reshape(-1, self.channels).float().cpu()
        y = y.detach().reshape(-1, self.channels).float().cpu()
        if x.shape != y.shape:
            raise ValueError(f'feature shape mismatch: {x.shape} vs {y.shape}')

        n = x.shape[0]
        self.n += n
        diff = x - y
        self.sse += float((diff * diff).sum().item())
        self.cos_sum += float(F.cosine_similarity(x, y, dim=1, eps=1e-8)
                              .sum().item())

        xd = x.double()
        yd = y.double()
        self.sum_x += xd.sum(dim=0)
        self.sum_y += yd.sum(dim=0)
        self.xtx += xd.t().matmul(xd)
        self.yty += yd.t().matmul(yd)
        self.xty += xd.t().matmul(yd)

    def finalize(self):
        if self.n == 0:
            return dict(n=0, mse=float('nan'), cosine=float('nan'),
                        cka=float('nan'))

        n = float(self.n)
        cov_xx = self.xtx - torch.outer(self.sum_x, self.sum_x) / n
        cov_yy = self.yty - torch.outer(self.sum_y, self.sum_y) / n
        cov_xy = self.xty - torch.outer(self.sum_x, self.sum_y) / n

        hsic = (cov_xy * cov_xy).sum()
        var_x = (cov_xx * cov_xx).sum()
        var_y = (cov_yy * cov_yy).sum()
        cka = hsic / (torch.sqrt(var_x * var_y) + 1e-12)

        return dict(
            n=int(self.n),
            mse=self.sse / (self.n * self.channels),
            cosine=self.cos_sum / self.n,
            cka=float(cka.item()),
        )


def dc_data(value):
    return value.data[0] if hasattr(value, 'data') else value


def to_cuda(value):
    value = dc_data(value)
    if isinstance(value, list):
        return [v.cuda() if torch.is_tensor(v) else v for v in value]
    if torch.is_tensor(value):
        return value.cuda()
    return value


def stack_batch_tensor(data, key, required=False):
    if key not in data:
        if required:
            raise KeyError(f'missing required batch key: {key}')
        return None
    value = dc_data(data[key])
    if isinstance(value, list):
        return torch.stack([v.cuda() if torch.is_tensor(v) else v
                            for v in value], dim=0)
    if torch.is_tensor(value):
        return value.cuda()
    raise TypeError(f'unsupported tensor container for {key}: {type(value)}')


def first_sample(value):
    value = dc_data(value)
    while isinstance(value, list) and len(value) == 1:
        value = value[0]
    return value


def normalize_bbox(bbox, pc_range):
    x, y = torch.tensor_split(bbox[..., :2], 2, dim=-1)
    x = (x - pc_range[0]) / (pc_range[3] - pc_range[0])
    y = (y - pc_range[1]) / (pc_range[4] - pc_range[1])
    if bbox.shape[-1] > 2:
        z, x_size, y_size, z_size, yaw, vx, vy = torch.tensor_split(
            bbox[..., 2:], 7, dim=-1)
        z = (z - pc_range[2]) / (pc_range[5] - pc_range[2])
        x_size = x_size / (pc_range[3] - pc_range[0])
        y_size = y_size / (pc_range[4] - pc_range[1])
        z_size = z_size / (pc_range[5] - pc_range[2])
        return torch.cat((x, y, z, x_size, y_size, z_size, yaw, vx, vy),
                         dim=-1)
    return torch.cat((x, y), dim=-1)


def synthesize_layout_condition(data, cfg):
    """Build BEVDiffuser layout condition from GT boxes/labels in a shared batch.

    This mirrors CustomNuScenesDiffusionDataset_layout for samples_per_gpu=1.
    It lets us use one semantic shared dataloader while still feeding the
    baseline layout teacher with the same frame.
    """
    if cfg is None:
        raise KeyError('layout keys are missing and cfg was not provided')
    if 'gt_labels_3d' not in data or 'gt_bboxes_3d' not in data:
        raise KeyError('layout keys are missing and GT boxes/labels are absent')

    num_bboxes = int(cfg.num_bboxes)
    num_classes = int(cfg.num_classes)
    class_names = getattr(cfg, 'class_names', [])
    pc_range = torch.tensor(cfg.point_cloud_range, dtype=torch.float32)

    labels = first_sample(data['gt_labels_3d'])
    boxes = first_sample(data['gt_bboxes_3d'])

    layout_obj_classes = torch.LongTensor(num_bboxes).fill_(num_classes - 1)
    layout_obj_classes[0] = num_classes - 2
    layout_is_valid = torch.zeros(num_bboxes)
    layout_is_valid[0] = 1.0

    if torch.is_tensor(labels):
        class_ids = (labels.cpu().long() + len(class_names)) % len(class_names)
        num_valid = min(len(class_ids), num_bboxes - 1)
        layout_obj_classes[1: 1 + num_valid] = class_ids[:num_valid]
        layout_is_valid[1: 1 + num_valid] = 1.0

    layout_obj_bboxes = torch.zeros(num_bboxes, 9)
    layout_obj_bboxes[0] = torch.FloatTensor(
        [0, 0, 0, 1, 1, 1, 0, 0, 0])
    if isinstance(boxes, LiDARInstance3DBoxes):
        norm_boxes = normalize_bbox(boxes.tensor.cpu(), pc_range)
        norm_boxes[..., :2] = norm_boxes[..., :2] - 0.5
        num_valid = min(len(norm_boxes), num_bboxes - 1)
        layout_obj_bboxes[1: 1 + num_valid] = norm_boxes[:num_valid]

    return {
        'obj_class': layout_obj_classes.unsqueeze(0).cuda(),
        'obj_bbox': layout_obj_bboxes.unsqueeze(0).cuda(),
        'is_valid_obj': layout_is_valid.unsqueeze(0).cuda(),
    }


def build_condition(data, cfg=None):
    cond = {}
    for src, dst in [
        ('layout_obj_classes', 'obj_class'),
        ('layout_obj_bboxes', 'obj_bbox'),
        ('layout_obj_is_valid', 'is_valid_obj'),
        ('layout_obj_names', 'obj_name'),
        ('default_obj_names', 'default_obj_names'),
    ]:
        if src not in data:
            continue
        value = dc_data(data[src])
        if isinstance(value, list):
            value = torch.stack(value, dim=0)
        cond[dst] = value.cuda() if torch.is_tensor(value) else value
    if not cond:
        cond = synthesize_layout_condition(data, cfg)
    return cond


def build_setup(cfg_path, ckpt_path, init_ckpt_path, mode):
    cfg = Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    for legacy in ('use_adapter',):
        cfg.model.pop(legacy, None)

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, ckpt_path, map_location='cpu', strict=False)
    model.eval().cuda()

    init_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(init_model, init_ckpt_path, map_location='cpu',
                    strict=False)
    init_model.eval().cuda()
    init_model.requires_grad_(False)

    teacher_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(teacher_model, init_ckpt_path, map_location='cpu',
                    strict=False)
    teacher_model.eval().cuda()
    teacher_model.requires_grad_(False)

    if mode == 'baseline':
        from bevdiffuser import BEVDiffuser
    elif mode == 'semantic':
        from bevdiffuser_onlyseg import BEVDiffuser
    else:
        raise ValueError(f'unknown setup mode: {mode}')

    bev_diffuser = BEVDiffuser(**cfg.bev_diffuser_cfg).eval().cuda()
    bev_diffuser.requires_grad_(False)

    return dict(cfg=cfg, model=model, init_model=init_model,
                teacher_model=teacher_model, bev_diffuser=bev_diffuser)


def build_shared_loader(cfg_path, workers_per_gpu):
    """Build one shared semantic dataloader for all setups."""
    cfg = Config.fromfile(cfg_path)
    cfg.data.val.test_mode = False
    cfg.data.val.pipeline = make_pipeline(cfg.class_names)
    cfg.data.val.pop('samples_per_gpu', None)
    dataset_default_args = {
        'pc_range': cfg.point_cloud_range,
        'use_3d_bbox': cfg.use_3d_bbox,
        'num_classes': cfg.num_classes,
        'num_bboxes': cfg.num_bboxes,
    }
    dataset = build_dataset(cfg.data.val, default_args=dataset_default_args)
    loader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=workers_per_gpu,
        dist=False,
        shuffle=False,
        num_gpus=1,
    )
    return loader


@torch.no_grad()
def extract_student_bev(model, data):
    img = to_cuda(data['img'])
    img_metas = dc_data(data['img_metas'])

    len_queue = img.size(1)
    prev_img = img[:, :-1]
    cur_img = img[:, -1]
    prev_img_metas = copy.deepcopy(img_metas)

    prev_bev = None
    if prev_img.size(1) > 0:
        prev_bev = model.obtain_history_bev(prev_img, prev_img_metas)

    cur_img_metas = [each[len_queue - 1] for each in img_metas]
    if not cur_img_metas[0].get('prev_bev_exists', False):
        prev_bev = None

    img_feats = model.extract_feat(img=cur_img, img_metas=cur_img_metas)
    student_bev = model.pts_bbox_head(img_feats, cur_img_metas, prev_bev,
                                      only_bev=True)
    return student_bev.detach(), cur_img_metas


@torch.no_grad()
def extract_teacher_bev(setup, data, mode, seed):
    set_reproducible_seed(seed)
    img = to_cuda(data['img'])
    img_metas = dc_data(data['img_metas'])
    len_queue = img.size(1)
    cur_img_metas = [each[len_queue - 1] for each in img_metas]

    teacher_raw = setup['teacher_model'](
        return_loss=False, only_bev=True, img=img, img_metas=img_metas
    ).detach()

    B, _, C = teacher_raw.shape
    H = setup['model'].pts_bbox_head.bev_h
    W = setup['model'].pts_bbox_head.bev_w
    teacher_2d = teacher_raw.reshape(B, H, W, C).permute(
        0, 3, 1, 2).contiguous()

    if mode == 'baseline':
        cond = build_condition(data, setup['cfg'])
        teacher_2d = setup['bev_diffuser'](teacher_2d, cond)
    else:
        segmaps = stack_batch_tensor(data, 'seg_maps', required=True)
        depth_maps = stack_batch_tensor(data, 'depth_maps', required=False)
        teacher_2d = setup['bev_diffuser'](
            teacher_2d, cur_img_metas, segmaps, depth_maps)

    teacher_flat = teacher_2d.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return teacher_flat.detach()


def model_uses_mgd(model):
    return bool(getattr(model, 'use_mgd', False)
                and hasattr(model, 'mgd_generation'))


@torch.no_grad()
def feature_for_alignment(model, student_bev, seed):
    """Return the feature compared with T for the setup's distillation loss.

    Direct-MSE models compare raw student_bev with T. MGD models compare the
    trained reconstruction branch, mgd_generation(mask * student_bev), with T.
    """
    if not model_uses_mgd(model):
        return student_bev.detach(), 'student_bev'

    set_reproducible_seed(seed)
    B, HW, C = student_bev.shape
    H = model.pts_bbox_head.bev_h
    W = model.pts_bbox_head.bev_w
    if HW != H * W:
        raise ValueError(f'BEV spatial shape mismatch: HW={HW}, H*W={H * W}')

    student_2d = student_bev.permute(0, 2, 1).reshape(
        B, C, H, W).contiguous()
    if getattr(model, 'mgd_align', None) is not None:
        preds = model.mgd_align(student_2d)
    else:
        preds = student_2d

    mgd_lambda = float(getattr(model, 'mgd_lambda', 0.65))
    mask = torch.rand((B, 1, H, W), device=preds.device)
    mask = torch.where(mask > 1 - mgd_lambda,
                       torch.zeros_like(mask),
                       torch.ones_like(mask))
    mgd_out = model.mgd_generation(preds * mask)
    mgd_flat = mgd_out.permute(0, 2, 3, 1).reshape(B, H * W, C)
    return mgd_flat.detach(), 'mgd_output'


def sample_idx_from_data(data):
    img = to_cuda(data['img'])
    img_metas = dc_data(data['img_metas'])
    len_queue = img.size(1)
    cur_img_metas = [each[len_queue - 1] for each in img_metas]
    return cur_img_metas[0].get('sample_idx', None)


def row_from_metrics(name, setup_spec, align_metrics, gap_metrics,
                     aligned_feature):
    return OrderedDict([
        ('name', name),
        ('teacher', setup_spec['label']),
        ('distill_loss', setup_spec['distill_loss']),
        ('aligned_feature', aligned_feature),
        ('mAP', setup_spec['map']),
        ('NDS', setup_spec['nds']),
        ('MSE(A,T)', align_metrics['mse']),
        ('Cos(A,T)', align_metrics['cosine']),
        ('CKA(A,T)', align_metrics['cka']),
        ('MSE(S0,T)', gap_metrics['mse']),
        ('Cos(S0,T)', gap_metrics['cosine']),
        ('CKA(S0,T)', gap_metrics['cka']),
        ('num_cells', align_metrics['n']),
    ])


def write_outputs(output_dir, rows, args, n_frames, sample_mismatches):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'global_alignment.csv')
    with open(csv_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    json_path = os.path.join(output_dir, 'summary.json')
    payload = OrderedDict([
        ('num_frames', n_frames),
        ('sample_mismatches', sample_mismatches),
        ('teacher_init_ckpt', args.teacher_init_ckpt),
        ('rows', rows),
    ])
    with open(json_path, 'w') as f:
        json.dump(payload, f, indent=2)

    txt_path = os.path.join(output_dir, 'summary.txt')
    with open(txt_path, 'w') as f:
        f.write('=== Experiment 4 Global Feature Alignment ===\n')
        f.write(f'Frames: {n_frames}\n')
        f.write(f'Sample mismatches: {sample_mismatches}\n')
        f.write(f'Teacher init ckpt: {args.teacher_init_ckpt}\n\n')
        for row in rows:
            f.write(f'[{row["name"]}] {row["teacher"]} / '
                    f'{row["distill_loss"]}\n')
            f.write(f'  aligned_feature={row["aligned_feature"]}\n')
            f.write(f'  mAP={row["mAP"]:.6f}  NDS={row["NDS"]:.6f}\n')
            f.write('  A  vs T : '
                    f'MSE={row["MSE(A,T)"]:.6f}  '
                    f'Cos={row["Cos(A,T)"]:.6f}  '
                    f'CKA={row["CKA(A,T)"]:.6f}\n')
            f.write('  S0 vs T : '
                    f'MSE={row["MSE(S0,T)"]:.6f}  '
                    f'Cos={row["Cos(S0,T)"]:.6f}  '
                    f'CKA={row["CKA(S0,T)"]:.6f}\n\n')

    return csv_path, json_path, txt_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='../results/feature_alignment/'
                                'exp4_global_no_region')
    parser.add_argument('--teacher_init_ckpt',
                        default='./ckpts/bevformer_tiny_epoch_24.pth')
    parser.add_argument('--num_frames', type=int, default=200)
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260823)
    parser.add_argument('--baseline_config',
                        default=DEFAULT_SETUPS['baseline_mse']['config'])
    parser.add_argument('--baseline_ckpt',
                        default=DEFAULT_SETUPS['baseline_mse']['ckpt'])
    parser.add_argument('--semantic_mse_config',
                        default=DEFAULT_SETUPS['semantic_mse']['config'])
    parser.add_argument('--semantic_mse_ckpt',
                        default=DEFAULT_SETUPS['semantic_mse']['ckpt'])
    parser.add_argument('--semantic_mgd_config',
                        default=DEFAULT_SETUPS['semantic_mgd']['config'])
    parser.add_argument('--semantic_mgd_ckpt',
                        default=DEFAULT_SETUPS['semantic_mgd']['ckpt'])
    return parser.parse_args()


def main():
    args = parse_args()
    set_reproducible_seed(args.seed)

    base_dir = _BEVFORMER_DIR
    args.output_dir = resolve_path(args.output_dir, base_dir)
    args.teacher_init_ckpt = resolve_path(args.teacher_init_ckpt, base_dir)

    setup_specs = copy.deepcopy(DEFAULT_SETUPS)
    setup_specs['baseline_mse']['config'] = args.baseline_config
    setup_specs['baseline_mse']['ckpt'] = args.baseline_ckpt
    setup_specs['semantic_mse']['config'] = args.semantic_mse_config
    setup_specs['semantic_mse']['ckpt'] = args.semantic_mse_ckpt
    setup_specs['semantic_mgd']['config'] = args.semantic_mgd_config
    setup_specs['semantic_mgd']['ckpt'] = args.semantic_mgd_ckpt

    for spec in setup_specs.values():
        spec['config'] = resolve_path(spec['config'], base_dir)
        spec['ckpt'] = resolve_path(spec['ckpt'], base_dir)

    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    setups = OrderedDict()
    for name, spec in setup_specs.items():
        print(f'=== Building {name}: {spec["config"]} ===')
        setups[name] = build_setup(
            spec['config'],
            spec['ckpt'],
            args.teacher_init_ckpt,
            spec['mode'],
        )

    shared_loader_config = setup_specs['semantic_mse']['config']
    print(f'=== Building shared dataloader: {shared_loader_config} ===')
    shared_loader = build_shared_loader(
        shared_loader_config, args.workers_per_gpu)

    channels = setups['baseline_mse']['cfg'].model.pts_bbox_head.in_channels
    align_meters = {
        name: PairAlignmentMeter(channels) for name in setup_specs.keys()
    }
    gap_meters = {
        name: PairAlignmentMeter(channels) for name in setup_specs.keys()
    }
    aligned_feature_names = {
        name: None for name in setup_specs.keys()
    }

    data_iter = iter(shared_loader)
    n_done = 0
    sample_mismatches = 0

    for frame_idx in range(args.num_frames):
        try:
            data = next(data_iter)
        except StopIteration:
            print(f'Dataloader exhausted at frame {frame_idx}; stopping.')
            break

        for name, spec in setup_specs.items():
            setup = setups[name]
            student_bev, _ = extract_student_bev(setup['model'], data)
            init_bev, _ = extract_student_bev(setup['init_model'], data)
            teacher_bev = extract_teacher_bev(
                setup, data, spec['mode'], args.seed + frame_idx)
            aligned_bev, aligned_feature = feature_for_alignment(
                setup['model'], student_bev, args.seed + 100000 + frame_idx)

            if aligned_feature_names[name] is None:
                aligned_feature_names[name] = aligned_feature
            elif aligned_feature_names[name] != aligned_feature:
                raise RuntimeError(
                    f'aligned feature changed for {name}: '
                    f'{aligned_feature_names[name]} vs {aligned_feature}')

            align_meters[name].update(aligned_bev, teacher_bev)
            gap_meters[name].update(init_bev, teacher_bev)

        n_done += 1
        if n_done == 1 or n_done % 10 == 0:
            elapsed = time.time() - t0
            print(f'[{n_done}/{args.num_frames}] elapsed={elapsed:.1f}s')

    rows = []
    for name, spec in setup_specs.items():
        rows.append(row_from_metrics(
            name,
            spec,
            align_meters[name].finalize(),
            gap_meters[name].finalize(),
            aligned_feature_names[name] or 'unknown',
        ))

    csv_path, json_path, txt_path = write_outputs(
        args.output_dir, rows, args, n_done, sample_mismatches)
    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')
    print(f'Wrote {txt_path}')


if __name__ == '__main__':
    main()
