"""Task feature preservation analysis for semantic teacher distillation.

This analysis compares each trained Stage-2 student's deployed BEV feature
against the initial BEVFormer feature S0. Unlike feature-teacher alignment, no
diffusion teacher is built here.

The main question is:
  Does MGD preserve the task-compatible student feature manifold better than
  direct MSE while still improving downstream detection?
"""

import argparse
import copy
import csv
import json
import os
import sys
import time
from collections import OrderedDict

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BEVFORMER_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
sys.path.append(_BEVFORMER_DIR)
sys.path.append(_THIS_DIR)

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

import projects.bevdiffuser  # noqa: F401
import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader

from analyze_feature_alignment import (  # noqa: E402
    DEFAULT_SETUPS,
    PairAlignmentMeter,
    extract_student_bev,
    make_pipeline,
    resolve_path,
    sample_idx_from_data,
    set_reproducible_seed,
)


def build_student_pair_setup(cfg_path, student_ckpt, init_ckpt,
                             workers_per_gpu):
    cfg = Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    for legacy in ('use_adapter',):
        cfg.model.pop(legacy, None)

    student = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(student, student_ckpt, map_location='cpu', strict=False)
    student.eval().cuda()

    init_model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(init_model, init_ckpt, map_location='cpu', strict=False)
    init_model.eval().cuda()
    init_model.requires_grad_(False)

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

    return dict(cfg=cfg, student=student, init_model=init_model,
                loader=loader)


def row_from_metrics(name, spec, metrics):
    return OrderedDict([
        ('name', name),
        ('teacher', spec['label']),
        ('distill_loss', spec['distill_loss']),
        ('mAP', spec['map']),
        ('NDS', spec['nds']),
        ('MSE(S_after,S0)', metrics['mse']),
        ('Cos(S_after,S0)', metrics['cosine']),
        ('CKA(S_after,S0)', metrics['cka']),
        ('num_cells', metrics['n']),
    ])


def write_outputs(output_dir, rows, args, n_frames, sample_mismatches):
    os.makedirs(output_dir, exist_ok=True)

    csv_path = os.path.join(output_dir, 'preservation.csv')
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
        f.write('=== Task Feature Preservation Analysis ===\n')
        f.write(f'Frames: {n_frames}\n')
        f.write(f'Sample mismatches: {sample_mismatches}\n')
        f.write(f'S0 checkpoint: {args.teacher_init_ckpt}\n\n')
        for row in rows:
            f.write(f'[{row["name"]}] {row["teacher"]} / '
                    f'{row["distill_loss"]}\n')
            f.write(f'  mAP={row["mAP"]:.6f}  NDS={row["NDS"]:.6f}\n')
            f.write('  S_after vs S0 : '
                    f'MSE={row["MSE(S_after,S0)"]:.6f}  '
                    f'Cos={row["Cos(S_after,S0)"]:.6f}  '
                    f'CKA={row["CKA(S_after,S0)"]:.6f}\n\n')

    return csv_path, json_path, txt_path


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir',
                        default='../results/feature_alignment/'
                                'task_feature_preservation_200_frames')
    parser.add_argument('--teacher_init_ckpt',
                        default='./ckpts/bevformer_tiny_epoch_24.pth')
    parser.add_argument('--num_frames', type=int, default=200)
    parser.add_argument('--workers_per_gpu', type=int, default=2)
    parser.add_argument('--seed', type=int, default=20260824)
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
        setups[name] = build_student_pair_setup(
            spec['config'],
            spec['ckpt'],
            args.teacher_init_ckpt,
            args.workers_per_gpu,
        )

    channels = setups['baseline_mse']['cfg'].model.pts_bbox_head.in_channels
    meters = {name: PairAlignmentMeter(channels)
              for name in setup_specs.keys()}

    iters = {name: iter(setup['loader']) for name, setup in setups.items()}
    n_done = 0
    sample_mismatches = 0

    for frame_idx in range(args.num_frames):
        try:
            batches = {name: next(iterator)
                       for name, iterator in iters.items()}
        except StopIteration:
            print(f'Dataloader exhausted at frame {frame_idx}; stopping.')
            break

        sample_indices = {name: sample_idx_from_data(data)
                          for name, data in batches.items()}
        if len(set(sample_indices.values())) > 1:
            sample_mismatches += 1
            if sample_mismatches <= 5:
                print(f'WARN sample mismatch at frame {frame_idx}: '
                      f'{sample_indices}')

        for name, setup in setups.items():
            data = batches[name]
            student_bev, _ = extract_student_bev(setup['student'], data)
            init_bev, _ = extract_student_bev(setup['init_model'], data)
            meters[name].update(student_bev, init_bev)

        n_done += 1
        if n_done == 1 or n_done % 10 == 0:
            elapsed = time.time() - t0
            print(f'[{n_done}/{args.num_frames}] elapsed={elapsed:.1f}s')

    rows = []
    for name, spec in setup_specs.items():
        rows.append(row_from_metrics(name, spec, meters[name].finalize()))

    csv_path, json_path, txt_path = write_outputs(
        args.output_dir, rows, args, n_done, sample_mismatches)
    print(f'Wrote {csv_path}')
    print(f'Wrote {json_path}')
    print(f'Wrote {txt_path}')


if __name__ == '__main__':
    main()
