"""Same-frame LiDAR / v5 Metric3D / v3 DA3 dominant-class comparison."""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mmcv import Config

from debug_seg_bev_v5 import (
    BEVFORMER, WORKSPACE, COLORS, UNKNOWN, SEG_CLASS_NAMES,
    build_dataset, class_rgb, draw_bev, draw_gt_boxes,
)
from projects.bevdiffuser.data_utils import CustomNuScenesDiffusionDataset_seg_depth_v2
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v3 import SegBEVAligner as AlignerV3
from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner as AlignerV5


def make_dataset(config):
    settings = config.data.test.copy()
    # Legacy DA3 paths are relative to projects/bevdiffuser, its launch directory.
    for key in ('data_root', 'ann_file', 'semantic_path', 'depth_path'):
        value = Path(settings[key])
        if not value.is_absolute():
            settings[key] = str((Path(__file__).resolve().parent / value).resolve())
    settings.update(test_mode=True, load_annos=True)
    if 'use_semantic_bev_cache' in settings:
        settings['use_semantic_bev_cache'] = False
    return build_dataset(settings, default_args=dict(
        pc_range=config.point_cloud_range, use_3d_bbox=config.use_3d_bbox,
        num_classes=config.num_classes, num_bboxes=config.num_bboxes))


@torch.no_grad()
def extract_probabilities(aligner, item, meta, device):
    captured = []

    def capture(module, inputs):
        captured.append(inputs[0].detach().cpu().numpy().copy())

    handle = aligner.prob_to_emb.register_forward_pre_hook(capture)
    try:
        aligner(item['seg_maps'].data.unsqueeze(0).to(device), [meta],
                depth_maps=item['depth_maps'].data.unsqueeze(0).to(device))
    finally:
        handle.remove()
    if len(captured) != 1 or not np.isfinite(captured[0]).all():
        raise ValueError('Expected one finite pre-embedding semantic tensor')
    return captured[0][0]


def draw_row(axes, record, extent):
    ax = axes[0]
    points = record['points']
    ax.set_facecolor('white')
    ax.set_axisbelow(True)
    ax.grid(True, color='#dddddd', linewidth=.5)
    ax.scatter(points[:, 0], points[:, 1], c=points[:, 2], cmap='viridis',
               s=.35, alpha=1., linewidths=0, vmin=-3, vmax=3, rasterized=True)
    count = draw_gt_boxes(ax, record['boxes'], extent)
    ax.plot(0, 0, '+', color='black', zorder=5)
    ax.set(xlim=extent[:2], ylim=extent[2:], xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)',
           title=f'LiDAR + GT ({count}) | sample {record["index"]}')
    ax.set_aspect('equal')
    draw_bev(axes[1], class_rgb(record['v5'])[0], extent, 'v5 | Metric3D | dominant class')
    draw_bev(axes[2], class_rgb(record['v3'])[0], extent, 'v3 | DA3 | dominant class')


def add_legend(fig, records):
    present = set()
    for record in records:
        for key in ('v5', 'v3'):
            p = record[key]
            present.update(np.unique(p.argmax(0)[p.sum(0) > 0]).tolist())
    handles = [mpatches.Patch(color=COLORS[c], label=f'{c}: {SEG_CLASS_NAMES[c]}') for c in sorted(present)]
    handles.extend([mpatches.Patch(color=UNKNOWN, label='No semantic mass'),
                    mpatches.Patch(facecolor='none', edgecolor='#d62728', label='GT bbox + heading')])
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(.5, 0), ncol=7, fontsize=8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    configs = BEVFORMER / 'projects/configs/bevdiffuser'
    parser.add_argument('--v5_config', default=str(configs / 'bev_tiny_onlyseg_sam_v2_metric3d.py'))
    parser.add_argument('--v3_config', default=str(configs / 'bev_tiny_onlyseg_sam_v2_da3.py'))
    parser.add_argument('--samples_manifest', default=str(WORKSPACE / 'visualize/metric3d_v5/summary.json'))
    parser.add_argument('--output_dir', default=str(WORKSPACE / 'visualize/metric3d_v5_vs_da3_v3'))
    parser.add_argument('--device', default='cpu')
    args = parser.parse_args()
    cfg5, cfg3 = Config.fromfile(args.v5_config), Config.fromfile(args.v3_config)
    np.testing.assert_allclose(cfg5.point_cloud_range, cfg3.point_cloud_range)
    ds5, ds3 = make_dataset(cfg5), make_dataset(cfg3)
    a5 = AlignerV5(**cfg5.unet.parameters.seg_bev_aligner).to(args.device).eval()
    a3 = AlignerV3(**cfg3.unet.parameters.seg_bev_aligner).to(args.device).eval()
    index5 = {info['token']: index for index, info in enumerate(ds5.data_infos)}
    index3 = {info['token']: index for index, info in enumerate(ds3.data_infos)}
    samples = json.loads(Path(args.samples_manifest).read_text())['samples']
    if not samples:
        raise ValueError('No samples in manifest')
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    extent = tuple(cfg5.point_cloud_range[i] for i in (0, 3, 1, 4))
    records, summary = [], []
    for sample in samples:
        token = sample['sample_token']
        i5, i3 = index5[token], index3[token]
        for filename in ds3.get_data_info(i3)['img_filename']:
            relative = Path(filename.split('samples/')[-1]).with_suffix('.npy')
            depth_path = Path(ds3.depth_path) / 'samples' / relative
            if not depth_path.is_file():
                raise FileNotFoundError(f'DA3 depth required for comparison: {depth_path}')
        item5, item3 = ds5[i5], ds3[i3]
        meta5 = CustomNuScenesDiffusionDataset_seg_depth_v2._current_metas(item5['img_metas'])
        meta3 = CustomNuScenesDiffusionDataset_seg_depth_v2._current_metas(item3['img_metas'])
        assert meta5['sample_idx'] == meta3['sample_idx'] == token
        assert meta5['scene_token'] == meta3['scene_token']
        assert [Path(p).name for p in meta5['filename']] == [Path(p).name for p in meta3['filename']]
        np.testing.assert_allclose(meta5['lidar2img'], meta3['lidar2img'])
        p5 = extract_probabilities(a5, item5, meta5, args.device)
        p3 = extract_probabilities(a3, item3, meta3, args.device)
        assert p5.shape == p3.shape
        points = np.fromfile(ds5.get_data_info(i5)['pts_filename'], dtype=np.float32).reshape(-1, 5)
        inside = ((points[:, 0] >= extent[0]) & (points[:, 0] < extent[1])
                  & (points[:, 1] >= extent[2]) & (points[:, 1] < extent[3]))
        ann = ds5.get_ann_info(i5)
        record = dict(index=i5, v5=p5, v3=p3, points=points[inside],
                      boxes=ann['gt_bboxes_3d'][ann['gt_labels_3d'] >= 0])
        records.append(record)
        prefix = f'sample_{i5:05d}_{token[:8]}'
        fig, axes = plt.subplots(1, 3, figsize=(18, 6), constrained_layout=True)
        draw_row(axes, record, extent)
        fig.suptitle(f'Same frame: {token} | pre-embedding semantic BEV', fontsize=12)
        add_legend(fig, [record])
        fig.savefig(output / (prefix + '.png'), dpi=160, bbox_inches='tight')
        plt.close(fig)
        np.savez_compressed(output / (prefix + '.npz'), v5_probabilities=p5, v3_probabilities=p3,
                            pc_range=np.asarray(cfg5.point_cloud_range), sample_token=token,
                            scene_token=meta5['scene_token'])
        entry = dict(index_v5=i5, index_v3=i3, sample_token=token, scene_token=meta5['scene_token'],
                     image=prefix + '.png', v5_depth_shape=list(item5['depth_maps'].data.shape),
                     v3_depth_shape=list(item3['depth_maps'].data.shape),
                     v5_nonzero_fraction=float((p5.sum(0) > 0).mean()),
                     v3_nonzero_fraction=float((p3.sum(0) > 0).mean()))
        summary.append(entry)
        print(f'Saved {prefix}: tokens / cameras / calibration match; '
              f'v5 {entry["v5_nonzero_fraction"]:.1%}, v3 {entry["v3_nonzero_fraction"]:.1%} nonzero', flush=True)
    fig, axes = plt.subplots(len(records), 3, figsize=(18, len(records) * 5.5),
                             squeeze=False, constrained_layout=True)
    for row, record in zip(axes, records):
        draw_row(row, record, extent)
    add_legend(fig, records)
    fig.savefig(output / 'overview.png', dpi=120, bbox_inches='tight')
    plt.close(fig)
    report = dict(v5_config=args.v5_config, v3_config=args.v3_config,
                  v5_aligner=dict(cfg5.unet.parameters.seg_bev_aligner),
                  v3_aligner=dict(cfg3.unet.parameters.seg_bev_aligner),
                  v3_effective_seg_downsample_factor=a3.seg_downsample_factor,
                  note='Separate config-specific dataset pipelines. v5 uses Metric3D; v3 uses DA3. '
                       'Not an isolated projection-only ablation. Both are actual prob_to_emb inputs. '
                       'Black is class 0 (v3 also moves sky here); gray means zero semantic mass. '
                       'LiDAR: all in-range points, size 0.35, alpha 1, white background, GT boxes.',
                  samples=summary)
    (output / 'summary.json').write_text(json.dumps(report, indent=2) + '\n')
    print(f'Comparison saved to {output}', flush=True)


if __name__ == '__main__':
    main()
