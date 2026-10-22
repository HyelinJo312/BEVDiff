# SPDX-License-Identifier: AGPL-3.0
"""Visualize frozen pretrained DINOv3 BEV targets from real nuScenes frames.

Uses exactly the Stage 1 extractor and batch unpacking. A single PCA is fitted
on observed cells across all selected frames, for visualization only. Neither
training targets nor checkpoints are modified by this RGB projection.
"""

import argparse
import copy
import json
import os
from pathlib import Path
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import numpy as np
from PIL import Image
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BEVFORMER = SCRIPT_DIR.parents[1]
sys.path.insert(0, str(BEVFORMER))
sys.path.insert(0, str(SCRIPT_DIR))

from dinov3_bev_extractor import DINOv3BEVExtractor, _content_shape
from train_bev_diffuser_seg_dino import unpack_batch


UNKNOWN_RGB = np.array([0.055, 0.065, 0.085], dtype=np.float32)


def fit_shared_pca(features, masks):
    """Fit on raw observed vectors; exclude unknown zeros and fix PC signs."""
    vectors = np.concatenate([f[:, m].T for f, m in zip(features, masks)], axis=0).astype(np.float64)
    if len(vectors) < 3 or not np.isfinite(vectors).all():
        raise ValueError('PCA requires at least 3 finite observed BEV cells')
    mean = vectors.mean(0)
    centered = vectors - mean
    covariance = centered.T @ centered / (len(vectors) - 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:3]].T
    for component in components:
        if component[np.argmax(np.abs(component))] < 0:
            component *= -1
    scores = centered @ components.T
    lower, upper = np.percentile(scores, [2, 98], axis=0)
    variance = np.maximum(eigenvalues[order], 0)
    if variance.sum() <= 0:
        raise ValueError('Observed features have zero variance; PCA colors are undefined')
    return dict(mean=mean.astype(np.float32), components=components.astype(np.float32),
                lower=lower.astype(np.float32), upper=upper.astype(np.float32),
                explained_variance_ratio=(variance[:3] / variance.sum()).astype(np.float32),
                fitted_cells=np.asarray(len(vectors)))


def colorize(features, mask, pca):
    rgb = np.empty((*mask.shape, 3), dtype=np.float32)
    rgb[:] = UNKNOWN_RGB
    scores = (features[:, mask].T - pca['mean']) @ pca['components'].T
    rgb[mask] = np.clip((scores - pca['lower']) / np.maximum(pca['upper'] - pca['lower'], 1e-8), 0, 1)
    return rgb


def camera_images(images, meta):
    norm = meta['img_norm_cfg']
    pixels = images.detach().cpu().float().numpy()
    pixels = pixels * np.asarray(norm['std'])[None, :, None, None] + np.asarray(norm['mean'])[None, :, None, None]
    if not norm.get('to_rgb', True):
        pixels = pixels[:, [2, 1, 0]]
    result = {}
    for view, filename in enumerate(meta['filename']):
        height, width = map(int, _content_shape(meta, view))
        name = Path(filename).parent.name
        result[name] = np.clip(pixels[view, :, :height, :width].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return result


def plot_bev(ax, value, extent, title, **kwargs):
    im = ax.imshow(value, origin='lower', extent=extent, interpolation='nearest', **kwargs)
    ax.plot(0, 0, '+', color='white', markersize=8, markeredgewidth=1.5)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', aspect='equal')
    ax.set_facecolor(UNKNOWN_RGB)
    return im



def load_gt_annotations(dataset, index, bounds):
    """Use dataset box conversion/filtering so GT and LIDAR_TOP share coordinates."""
    annotations = dataset.get_ann_info(index)
    boxes = annotations['gt_bboxes_3d']
    labels = np.asarray(annotations['gt_labels_3d'])
    centers = boxes.gravity_center.cpu().numpy()
    valid = (labels >= 0) & np.isfinite(centers).all(1)
    valid &= (centers >= bounds[:3]).all(1) & (centers < bounds[3:]).all(1)
    selected = boxes[torch.from_numpy(valid)]
    # Native LiDARInstance3DBoxes bottom face, cyclic order. Do not reinterpret yaw.
    corners = (selected.corners[:, [0, 3, 7, 4], :2].cpu().numpy()
               if len(selected) else np.empty((0, 4, 2), dtype=np.float32))
    return dict(gt_corners=corners, gt_boxes=selected.tensor.cpu().numpy(),
                gt_names=[dataset.CLASSES[int(label)] for label in labels[valid]])


def plot_lidar(ax, cloud, extent, title='Measured LiDAR + GT | top view',
               gt_corners=None, z_range=(-5, 3)):
    """Height-colored points and red GT footprints on a separate white panel."""
    ax.set_facecolor('white')
    ax.grid(False)
    if cloud is not None:
        artist = ax.scatter(cloud[:, 0], cloud[:, 1], c=cloud[:, 2], cmap='turbo',
                            vmin=z_range[0], vmax=z_range[1], s=.65, alpha=.9, linewidths=0)
        ax.figure.colorbar(artist, ax=ax, fraction=.046, pad=.025, shrink=.8, label='z (m)')
    else:
        title = 'LiDAR unavailable'
    if gt_corners is not None:
        for corners in gt_corners:
            closed = np.concatenate([corners, corners[:1]], axis=0)
            line, = ax.plot(closed[:, 0], closed[:, 1], color='#e31a1c', linewidth=1.2, zorder=4)
            line.set_path_effects([path_effects.Stroke(linewidth=2, foreground='white'), path_effects.Normal()])
    ax.plot(0, 0, '+', color='#d62728', markersize=8, markeredgewidth=1.5, zorder=5)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)',
           xlim=extent[:2], ylim=extent[2:], aspect='equal')


def render_overview(samples, pca, extent, output):
    pairs_per_row = min(2, len(samples))
    rows = (len(samples) + pairs_per_row - 1) // pairs_per_row
    fig, axes = plt.subplots(rows, pairs_per_row * 2, figsize=(10 * pairs_per_row, 5 * rows),
                             squeeze=False, constrained_layout=True)
    for i, sample in enumerate(samples):
        row, column = i // pairs_per_row, 2 * (i % pairs_per_row)
        record = sample['record']
        label = f"{record['split']} {record['index']} | {record['token'][:10]}"
        plot_bev(axes[row, column], colorize(sample['bev'], sample['mask'], pca), extent,
                 f"{label}\nDINOv3 BEV | observed {record['observed_fraction']:.1%}")
        plot_lidar(axes[row, column + 1], sample['cloud'], extent,
                   label + f"\nLiDAR height + GT ({len(sample['gt_corners'])} boxes)",
                   gt_corners=sample['gt_corners'], z_range=sample['z_range'])
    for i in range(len(samples), rows * pairs_per_row):
        for column in (2 * (i % pairs_per_row), 2 * (i % pairs_per_row) + 1):
            axes[i // pairs_per_row, column].axis('off')
    explained = float(pca['explained_variance_ratio'].sum())
    fig.suptitle(f'DINOv3 BEV and separate LiDAR | {len(samples)} frames | shared PCA variance: {explained:.1%}\n'
                 'Same XY limits | LiDAR color = height (z) | red outlines = GT boxes | white background', fontsize=16)
    fig.savefig(output / 'overview.png', dpi=160)
    plt.close(fig)


def render_frame(sample, pca, extent, norm_max, support_max, output):
    rgb = colorize(sample['bev'], sample['mask'], pca)
    fig = plt.figure(figsize=(18, 12), constrained_layout=True)
    grid = fig.add_gridspec(3, 12, height_ratios=[1, 1, 1.6])
    camera_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    for i, name in enumerate(camera_order):
        ax = fig.add_subplot(grid[i // 3, (i % 3) * 4:(i % 3 + 1) * 4])
        ax.imshow(sample['cameras'][name])
        ax.set_title(name, fontsize=11)
        ax.axis('off')
    axes = [fig.add_subplot(grid[2, i * 3:(i + 1) * 3]) for i in range(4)]
    plot_bev(axes[0], rgb, extent, 'DINOv3 BEV | shared PCA RGB')
    plot_lidar(axes[1], sample['cloud'], extent,
               f"LiDAR height + GT ({len(sample['gt_corners'])} boxes)",
               gt_corners=sample['gt_corners'], z_range=sample['z_range'])
    cmap = plt.get_cmap('viridis').copy()
    cmap.set_bad(UNKNOWN_RGB)
    norm = np.linalg.norm(sample['bev'], axis=0)
    artist = plot_bev(axes[2], np.ma.masked_where(~sample['mask'], norm), extent,
                      'Feature L2 norm', cmap=cmap, vmin=0, vmax=norm_max)
    fig.colorbar(artist, ax=axes[2], fraction=.046, pad=.025, shrink=.8)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad(UNKNOWN_RGB)
    artist = plot_bev(axes[3], np.ma.masked_where(~sample['mask'], np.log10(1 + sample['support'])), extent,
                      'Observation mass | log10(1 + mass)', cmap=cmap, vmin=0, vmax=support_max)
    fig.colorbar(artist, ax=axes[3], fraction=.046, pad=.025, shrink=.8)
    record = sample['record']
    fig.suptitle(f"DINOv3 ViT-S/16 | {record['split']} index {record['index']} | {record['token']}\n"
                 f"50 x 50 x 384 | observed {record['observed_fraction']:.1%} | "
                 'PCA colors represent feature similarity, not semantic classes', fontsize=14)
    filename = f"frame_{record['index']:05d}_{record['token']}.png"
    fig.savefig(output / filename, dpi=140)
    plt.close(fig)
    record['figure'] = filename
    # Raw grid without axes, nearest-neighbor enlargement only; positive Y is up.
    image = Image.fromarray((np.flipud(rgb) * 255).round().astype(np.uint8))
    image.resize((600, 600), resample=Image.Resampling.NEAREST).save(output / 'maps' / filename)
    return rgb


def main():
    from mmcv import Config, DictAction
    from mmcv.parallel import collate
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_dino.py'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--num-frames', type=int, default=6)
    parser.add_argument('--indices', nargs='+', type=int, help='Overrides evenly spaced frame selection')
    parser.add_argument('--output-dir', default=str(BEVFORMER.parent / 'results/visualize/dinov3_bev'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--model-path', help='Optional HF snapshot directory containing pretrained model and processor')
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error('--num-frames must be positive')
    config_path, output = Path(args.config).resolve(), Path(args.output_dir).resolve()
    model_path = str(Path(args.model_path).resolve()) if args.model_path else None
    os.chdir(SCRIPT_DIR)  # Use the training launcher's relative dataset paths.
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)
    cfg = Config.fromfile(str(config_path))
    if args.cfg_options:
        cfg.merge_from_dict(args.cfg_options)
    if model_path:
        cfg.dino_bev_extractor.encoder.update(model_name=model_path, local_files_only=True)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    print('Loading pretrained DINOv3...', flush=True)
    extractor = DINOv3BEVExtractor(**cfg.dino_bev_extractor).to(device).eval()
    if extractor.encoder.hidden_size != 384:
        raise ValueError('Expected 384-channel DINOv3 ViT-S features')
    print('Loaded model revision:', extractor.encoder.revision, flush=True)

    import projects.mmdet3d_plugin  # Register the configured dataset/pipelines.
    from mmdet3d.datasets import build_dataset
    settings = copy.deepcopy(cfg.data[args.split])
    settings.pop('samples_per_gpu', None)
    dataset = build_dataset(settings, default_args=dict(pc_range=cfg.point_cloud_range))
    indices = args.indices or np.linspace(0, len(dataset) - 1, min(args.num_frames, len(dataset)), dtype=int).tolist()
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError('Frame indices must be unique and inside the dataset')
    for name in ('arrays', 'maps'):
        (output / name).mkdir(parents=True, exist_ok=True)
    samples = []
    bounds = np.asarray(cfg.point_cloud_range)
    for index in indices:
        images, metas, depth, _ = unpack_batch(collate([dataset[index]], samples_per_gpu=1), device)
        meta = metas[0]
        info = dataset.data_infos[index]
        if meta['sample_idx'] != info['token']:
            raise ValueError('Returned dataset sample differs from the requested frame')
        with torch.no_grad():
            result = extractor(images, metas, depth)
        bev = result['bev_features'][0].cpu().numpy()
        mask = result['observed_mask'][0, 0].cpu().numpy()
        support = result['support'][0, 0].cpu().numpy()
        cls = result['cls_tokens'][0].cpu().numpy()
        if bev.shape != (384, 50, 50) or not np.isfinite(bev).all() or not np.isfinite(cls).all():
            raise ValueError('Unexpected or nonfinite DINO BEV/CLS output')
        if not mask.any():
            raise ValueError(f'No observed BEV cells for frame {index}')
        if np.any(bev[:, ~mask] != 0):
            raise ValueError('Unobserved feature cells must remain zero')
        lidar_path = Path(info['lidar_path'])
        cloud = None
        if lidar_path.is_file():
            cloud = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 5)[:, :3]
            keep = np.isfinite(cloud).all(1) & (cloud >= bounds[:3]).all(1) & (cloud < bounds[3:]).all(1)
            cloud = cloud[keep]
        norms = np.linalg.norm(bev[:, mask], axis=0)
        array_name = f"arrays/frame_{index:05d}_{meta['sample_idx']}.npz"
        np.savez_compressed(output / array_name, bev_features=bev, cls_tokens=cls,
                            observed_mask=mask, support=support)
        record = dict(index=index, split=args.split, token=meta['sample_idx'], scene_token=info.get('scene_token'),
                      timestamp=info.get('timestamp'), shape=list(bev.shape), cls_shape=list(cls.shape),
                      observed_fraction=float(mask.mean()), feature_norm_mean=float(norms.mean()),
                      feature_norm_percentiles=np.percentile(norms, [0, 2, 50, 98, 100]).tolist(),
                      camera_files=list(meta['filename']), lidar_path=str(lidar_path.resolve()),
                      lidar_points_in_range=len(cloud) if cloud is not None else None,
                      arrays=array_name)
        gt = load_gt_annotations(dataset, index, bounds)
        record.update(gt_box_count=len(gt['gt_corners']), gt_names=gt['gt_names'],
                      gt_boxes_lidar=gt['gt_boxes'].tolist())
        samples.append(dict(bev=bev, mask=mask, support=support, cameras=camera_images(images[0], meta),
                            cloud=cloud, record=record, z_range=bounds[[2, 5]].tolist(), **gt))
        print(f"Extracted {len(samples)}/{len(indices)}: index={index}, observed={mask.mean():.1%}", flush=True)
        del result, images, depth
    pca = fit_shared_pca([s['bev'] for s in samples], [s['mask'] for s in samples])
    np.savez(output / 'shared_pca.npz', **pca)
    norm_max = float(np.percentile(np.concatenate([np.linalg.norm(s['bev'][:, s['mask']], axis=0) for s in samples]), 98))
    support_max = float(np.percentile(np.concatenate([np.log10(1 + s['support'][s['mask']]) for s in samples]), 98))
    extent = bounds[[0, 3, 1, 4]].tolist()
    for sample in samples:
        render_frame(sample, pca, extent, norm_max, support_max, output)
    render_overview(samples, pca, extent, output)
    cfg.dump(str(output / 'config.py'))
    summary = dict(config=str(config_path), model_name=extractor.encoder.model_name,
                   resolved_revision=extractor.encoder.revision, device=str(device), extractor=cfg.dino_bev_extractor,
                   pca_explained_variance_ratio=pca['explained_variance_ratio'].tolist(),
                   pca_fitted_cells=int(pca['fitted_cells']), rgb_percentiles=[2, 98],
                   feature_norm_display_max=norm_max, log_support_display_max=support_max,
                   lidar_visualization=dict(color='height_z', cmap='turbo',
                                            z_range=bounds[[2, 5]].tolist(), background='white',
                                            gt_outline='red',
                                            gt_source='dataset.get_ann_info; mapped classes and centers inside XYZ range'),
                   samples=[s['record'] for s in samples])
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2))
    (output / 'README.md').write_text(
        '# DINOv3 BEV 시각화\n\n'
        '- `overview.png`: 전체 프레임의 DINOv3 BEV와 흰색 배경 LiDAR를 나란히 비교.\n'
        '- `frame_*.png`: 카메라 6개, PCA, 높이(z) 색상과 빨간색 GT box를 표시한 흰색 배경 LiDAR 패널, feature L2 norm, 관측량.\n'
        '- `maps/`: BEV RGB만 저장한 이미지(최근접 확대).\n'
        '- `arrays/`: 원본 384×50×50 feature, 카메라별 CLS, 관측 mask, support.\n'
        '- `shared_pca.npz`: 모든 선택 프레임의 관측 셀로 계산한 공통 PCA와 색상 범위.\n'
        '- `statistics.json`: 실제 모델 revision, 프레임 token, 관측 비율, PCA 설명 분산.\n\n'
        '시각화 대상은 diffusion 입력 이전의 frozen pretrained DINOv3 BEV입니다. '
        '색상은 의미 클래스가 아니라 384채널 feature의 상위 3개 주성분입니다. '
        '관측되지 않은 0 벡터는 PCA 계산에서 제외했고 어두운 색으로 표시했습니다. '
        '이는 free space 판정이 아닙니다. 각 RGB 채널은 전체 관측 셀의 2~98 백분위로 공통 정규화했습니다.\n\n'
        '가로는 LiDAR X, 세로는 LiDAR Y이며 BEV의 흰색 십자와 LiDAR의 빨간색 십자는 원점입니다. '
        'LiDAR 색상은 높이(z, m)이며 모든 프레임에 동일한 point_cloud_range의 Z 범위를 사용합니다. '
        '빨간 윤곽은 dataset.get_ann_info로 읽은 GT 3D box의 바닥면 XY 투영입니다. '
        '데이터셋의 유효 GT 필터를 적용하고 detection 클래스와 XYZ 범위 내 중심을 가진 box만 표시합니다. '
        'LiDAR와 GT는 비교용 시각화에만 사용하며 DINO BEV 생성에는 사용하지 않습니다. '
        'Support는 bilinear splat의 누적 가중치이며 confidence 확률이 아닙니다. '
        'PCA는 시각화에만 사용하며 원본 저장 feature와 Stage 1 학습 목표를 변경하지 않습니다.\n', encoding='utf-8')
    print('Saved visualization to', output, flush=True)


if __name__ == '__main__':
    main()
