# SPDX-License-Identifier: AGPL-3.0
"""Compare the two Stage 1 BEV feature sources against the LiDAR top view.

Panel 0: Frozen DINOv3 2D patch features per camera, in the augmented image
         frame. This is the input the parameter-free lift-and-splat consumes,
         shown so the BEV panel can be read against its own source.
Panel 1: BEV feature from the detection-pretrained BEVFormer encoder
         (``bevformer_tiny_epoch_24.pth``, ``only_bev=True``), i.e. the clean
         diffusion target used by ``train_bev_diffuser_only_seg.py``.
Panel 2: BEV feature from frozen DINOv3 patch tokens lifted with Metric3D depth
         and splatted without any learnable parameter
         (``dinov3_bev_extractor.DINOv3BEVExtractor``), i.e. the clean diffusion
         target used by ``train_bev_diffuser_seg_dino.py``.
Panel 3: Measured LiDAR top view in the nuScenes devkit style, white background,
         points colored by radial distance, GT 3D boxes drawn per class.

Every feature map is projected to RGB with a PCA that is fitted per source over
all selected frames. The sources have different channel counts (256 vs 384) and
different scales, and splatting changes the covariance of the DINOv3 tokens, so
the three bases are NOT shared: colors are comparable across frames and views
within one source only, never between sources. Colors encode feature
similarity, not semantic classes.

Example:
    cd BEVFormer/projects/bevdiffuser
    python visualize_bev_feature_compare.py --split val --num-frames 6
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
from torch.nn import functional as F
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BEVFORMER = SCRIPT_DIR.parents[1]
REPO = BEVFORMER.parent
sys.path.insert(0, str(BEVFORMER))
sys.path.insert(0, str(SCRIPT_DIR))

from dinov3_bev_extractor import DINOv3BEVExtractor, _content_shape  # noqa: E402

# Light neutral gray marks cells no camera observed. It is not a free-space label.
UNOBSERVED_RGB = np.array([0.88, 0.88, 0.90], dtype=np.float32)

# nuScenes detection classes; colors follow the devkit's category grouping
# (vehicles orange/amber, VRU blue-red, static objects neutral).
CLASS_COLORS = {
    'car': '#ff9e00', 'truck': '#ff6f00', 'construction_vehicle': '#b35806',
    'bus': '#ffcc00', 'trailer': '#d98c00', 'barrier': '#8c8c8c',
    'motorcycle': '#e31a1c', 'bicycle': '#d4006a', 'pedestrian': '#1f78ff',
    'traffic_cone': '#00a0a0',
}


# ----------------------------------------------------------------------------
# Feature extraction
# ----------------------------------------------------------------------------

def build_bevformer(config_path, checkpoint, device, use_temporal_test):
    """Load the frozen detection-pretrained BEVFormer used as the clean target."""
    from mmcv import Config
    from mmcv.parallel import MMDataParallel
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model

    cfg = Config.fromfile(str(config_path))
    if cfg.get('plugin', False):
        import importlib
        importlib.import_module(os.path.dirname(cfg.plugin_dir).replace('/', '.'))
    cfg.model.pretrained = None
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, str(checkpoint), map_location='cpu')
    model.CLASSES = cfg.class_names
    # Frames are sampled far apart, so a carried-over prev_bev would come from an
    # unrelated frame. Single-frame BEV keeps the comparison with DINO honest.
    model.video_test_mode = bool(use_temporal_test)
    model = MMDataParallel(model.to(device), device_ids=[device.index or 0])
    model.eval()
    return model, cfg


@torch.no_grad()
def bevformer_bev(model, batch, bev_h, bev_w, dim):
    """Run ``only_bev=True`` exactly as Stage 1 does and return [C, H, W]."""
    inputs = {key: batch[key] for key in ('img', 'img_metas') if key in batch}
    bev = model(return_loss=False, only_bev=True, **inputs).detach()
    if bev.numel() != bev_h * bev_w * dim:
        raise ValueError(f'Unexpected BEVFormer output with {bev.numel()} elements')
    bev = bev.reshape(-1, bev_h, bev_w, dim).permute(0, 3, 1, 2).contiguous()
    return bev[0].float().cpu().numpy()


def apply_input_scale(encoder, scale):
    """Resize the preprocessed frame before DINOv3, for a denser patch grid.

    DINOv3 uses RoPE position embeddings, so it accepts any input size. The
    resize is a uniform scale of the whole padded frame, so a pixel keeps its
    relative position and the projector's geometry is untouched; only the
    extent the patch grid covers must be converted back for grid_sample.
    """
    if scale == 1.0:
        return
    base, patch = encoder.preprocess, encoder.patch_size

    def preprocess(images, img_metas):
        pixels = base(images, img_metas)
        height, width = pixels.shape[-2:]
        target = (int(round(height * scale)), int(round(width * scale)))
        pixels = F.interpolate(pixels, size=target, mode='bicubic',
                               align_corners=False, antialias=scale < 1)
        encoder.scaled_frame = (height, width, *target)
        return F.pad(pixels, (0, -target[1] % patch, 0, -target[0] % patch))

    encoder.preprocess = preprocess


def source_frame_shape(encoder, feature_image_shape):
    """Express the padded feature frame in the unscaled image coordinates."""
    scaled = getattr(encoder, 'scaled_frame', None)
    if scaled is None:
        return feature_image_shape
    height, width, target_h, target_w = scaled
    return (feature_image_shape[0] * height / target_h,
            feature_image_shape[1] * width / target_w)


@torch.no_grad()
def dino_bev(extractor, batch, device, input_scale=1.0):
    """Run the frozen DINOv3 target provider and keep its 2D patch tokens.

    The encoder and projector are called separately, in the same order as
    ``DINOv3BEVExtractor.forward``, because the 2D patch grid that feeds the
    lift-and-splat is what the image-space panels display.
    """
    from train_bev_diffuser_seg_dino import unpack_batch
    unpacked = unpack_batch(batch, device)
    images, metas, depth = unpacked[0], unpacked[1], unpacked[2]
    encoded = extractor.encoder(images, metas)
    projected = extractor.projector(
        encoded['patch_features'], depth, metas,
        source_frame_shape(extractor.encoder, encoded['feature_image_shape']))
    return (projected['bev_features'][0].float().cpu().numpy(),
            projected['observed_mask'][0, 0].cpu().numpy(),
            projected['support'][0, 0].float().cpu().numpy(),
            patch_feature_maps(encoded['patch_features'][0], metas[0],
                               extractor.encoder.patch_size, input_scale=input_scale),
            images[0], metas[0])


def patch_feature_maps(patches, meta, patch_size, input_scale=1.0):
    """Crop each view's patch grid to the image content and key it by camera.

    ``preprocess`` zeroes the padded border before the forward pass, so the
    patch rows and columns that only cover padding carry no image evidence and
    are dropped here instead of being colored. The content is measured in the
    frame DINOv3 actually saw, so ``input_scale`` must be applied before the
    patch division or the crop would keep only part of a rescaled grid.
    """
    maps = {}
    for view, filename in enumerate(meta['filename']):
        height, width = (value * input_scale for value in _content_shape(meta, view))
        rows = min(patches.shape[-2], max(1, int(np.ceil(height / patch_size))))
        columns = min(patches.shape[-1], max(1, int(np.ceil(width / patch_size))))
        grid = patches[view, :, :rows, :columns]
        maps[Path(filename).parent.name] = grid.float().cpu().numpy()
    return maps


# ----------------------------------------------------------------------------
# PCA projection to RGB
# ----------------------------------------------------------------------------

def fit_pca(features, masks):
    """Fit one PCA per feature source on observed cells of all frames."""
    vectors = np.concatenate([f[:, m].T for f, m in zip(features, masks)], 0).astype(np.float64)
    if len(vectors) < 3 or not np.isfinite(vectors).all():
        raise ValueError('PCA requires at least 3 finite observed BEV cells')
    mean = vectors.mean(0)
    centered = vectors - mean
    eigenvalues, eigenvectors = np.linalg.eigh(centered.T @ centered / (len(vectors) - 1))
    order = np.argsort(eigenvalues)[::-1]
    components = eigenvectors[:, order[:3]].T
    for component in components:  # Fix the sign so repeated runs give equal colors.
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
    rgb[:] = UNOBSERVED_RGB
    scores = (features[:, mask].T - pca['mean']) @ pca['components'].T
    rgb[mask] = np.clip((scores - pca['lower']) / np.maximum(pca['upper'] - pca['lower'], 1e-8), 0, 1)
    return rgb


# ----------------------------------------------------------------------------
# Rendering
# ----------------------------------------------------------------------------

def plot_bev(ax, rgb, extent, title):
    ax.grid(False)
    ax.imshow(rgb, origin='lower', extent=extent, interpolation='nearest')
    ax.plot(0, 0, '+', color='black', markersize=8, markeredgewidth=1.4, zorder=5)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)', aspect='equal')
    ax.set_facecolor(UNOBSERVED_RGB)


def plot_lidar(ax, cloud, extent, title, boxes=None, axes_limit=51.2, point_size=0.6):
    """nuScenes devkit style: white background, distance-colored points, GT boxes."""
    ax.set_facecolor('white')
    ax.grid(False)
    if cloud is not None and len(cloud):
        # Devkit render_sample_data: color = min(1, radial distance / limit / sqrt(2)).
        distance = np.linalg.norm(cloud[:, :2], axis=1)
        colors = np.minimum(1.0, distance / axes_limit / np.sqrt(2))
        ax.scatter(cloud[:, 0], cloud[:, 1], c=colors, cmap='viridis',
                   vmin=0, vmax=1, s=point_size, linewidths=0)
    else:
        title = 'LiDAR unavailable'
    for box in boxes or []:
        corners = box['corners']
        closed = np.concatenate([corners, corners[:1]], 0)
        line, = ax.plot(closed[:, 0], closed[:, 1], color=box['color'], linewidth=1.3, zorder=4)
        line.set_path_effects([path_effects.Stroke(linewidth=2.3, foreground='white'),
                               path_effects.Normal()])
        # Devkit marks heading by emphasizing the front face; draw center -> front.
        front = corners[[0, 1]].mean(0)
        ax.plot([corners.mean(0)[0], front[0]], [corners.mean(0)[1], front[1]],
                color=box['color'], linewidth=1.0, zorder=4)
    ax.plot(0, 0, '+', color='#d62728', markersize=8, markeredgewidth=1.5, zorder=5)
    ax.set(title=title, xlabel='LiDAR X (m)', ylabel='LiDAR Y (m)',
           xlim=extent[:2], ylim=extent[2:], aspect='equal')


def camera_images(images, meta):
    """Undo the dataset normalization for display only."""
    norm = meta['img_norm_cfg']
    pixels = images.detach().cpu().float().numpy()
    pixels = pixels * np.asarray(norm['std'])[None, :, None, None] + \
        np.asarray(norm['mean'])[None, :, None, None]
    if not norm.get('to_rgb', True):
        pixels = pixels[:, [2, 1, 0]]
    result = {}
    for view, filename in enumerate(meta['filename']):
        height, width = map(int, _content_shape(meta, view))
        name = Path(filename).parent.name
        result[name] = np.clip(pixels[view, :, :height, :width].transpose(1, 2, 0), 0, 255).astype(np.uint8)
    return result


def plot_patch_map(ax, rgb, title):
    """Image-space panel: rows/columns are patch indices, not pixels."""
    ax.grid(False)
    ax.imshow(rgb, interpolation='nearest')
    ax.set_title(title, fontsize=10)
    ax.axis('off')


def tile_patch_maps(rgb_maps, camera_order, columns=3, gap=2):
    """Stitch the per-view 2D PCA maps into one 2 x 3 image for the overview."""
    tiles = [rgb_maps[name] for name in camera_order if name in rgb_maps]
    if not tiles:
        return None
    height = max(tile.shape[0] for tile in tiles)
    width = max(tile.shape[1] for tile in tiles)
    rows = int(np.ceil(len(tiles) / columns))
    canvas = np.ones((rows * height + (rows - 1) * gap,
                      columns * width + (columns - 1) * gap, 3), dtype=np.float32)
    for i, tile in enumerate(tiles):
        top, left = (i // columns) * (height + gap), (i % columns) * (width + gap)
        canvas[top:top + tile.shape[0], left:left + tile.shape[1]] = tile
    return canvas


def render_frame(sample, pcas, extent, output, with_cameras=True, point_size=0.6):
    record = sample['record']
    rgb_bevformer = colorize(sample['bevformer'], sample['bevformer_mask'], pcas['bevformer'])
    rgb_dino = colorize(sample['dino'], sample['dino_mask'], pcas['dino'])
    camera_order = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT',
                    'CAM_BACK_LEFT', 'CAM_BACK', 'CAM_BACK_RIGHT']
    has_cameras = with_cameras and all(name in sample['cameras'] for name in camera_order)
    patch_rgb = sample['patch_rgb']
    if has_cameras:
        # Rows: raw cameras, DINOv3 2D patch features, then the BEV panels. The
        # 2D row sits directly under its camera so the lift input is readable.
        fig = plt.figure(figsize=(18, 17), constrained_layout=True)
        grid = fig.add_gridspec(5, 6, height_ratios=[1, 1, 1, 1, 2.3])
        for i, name in enumerate(camera_order):
            block, column = (i // 3) * 2, (i % 3) * 2
            ax = fig.add_subplot(grid[block, column:column + 2])
            ax.imshow(sample['cameras'][name])
            ax.set_title(name, fontsize=11)
            ax.axis('off')
            if name in patch_rgb:
                rows, columns_count = patch_rgb[name].shape[:2]
                plot_patch_map(fig.add_subplot(grid[block + 1, column:column + 2]),
                               patch_rgb[name],
                               f'{name} | DINOv3 2D {rows} x {columns_count} patches')
        axes = [fig.add_subplot(grid[4, i * 2:(i + 1) * 2]) for i in range(3)]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6.2), constrained_layout=True)
        axes = list(axes)
    plot_bev(axes[0], rgb_bevformer, extent,
             f"BEVFormer encoder BEV (detection-pretrained)\n{sample['bevformer'].shape[0]} ch | PCA RGB")
    plot_bev(axes[1], rgb_dino, extent,
             f"DINOv3 + Metric3D splat BEV (no learnable params)\n"
             f"{sample['dino'].shape[0]} ch | observed {record['dino_observed_fraction']:.1%}")
    plot_lidar(axes[2], sample['cloud'], extent,
               f"LiDAR top view + GT ({len(sample['boxes'])} boxes)",
               boxes=sample['boxes'], point_size=point_size)
    fig.suptitle(
        f"DINOv3 2D features, DINOv3 BEV and BEVFormer BEV | {record['split']} index {record['index']} "
        f"| {record['token']}\n"
        'PCA colors are fitted per source over all frames and encode feature similarity, not classes',
        fontsize=14)
    filename = f"frame_{record['index']:05d}_{record['token']}.png"
    fig.savefig(output / filename, dpi=140)
    plt.close(fig)
    record['figure'] = filename
    return rgb_bevformer, rgb_dino


def render_overview(samples, pcas, extent, output, camera_order, point_size=0.6):
    rows = len(samples)
    # The 6-view tile is much wider than tall, so give it extra column width.
    fig, axes = plt.subplots(rows, 4, figsize=(24, 5.2 * rows), squeeze=False,
                             constrained_layout=True,
                             gridspec_kw=dict(width_ratios=[1.6, 1, 1, 1]))
    for row, sample in enumerate(samples):
        record = sample['record']
        label = f"{record['split']} {record['index']} | {record['token'][:10]}"
        tiled = tile_patch_maps(sample['patch_rgb'], camera_order)
        if tiled is None:
            axes[row][0].axis('off')
        else:
            plot_patch_map(axes[row][0], tiled, f'{label}\nDINOv3 2D features | 6 views')
        plot_bev(axes[row][1], colorize(sample['dino'], sample['dino_mask'], pcas['dino']),
                 extent, f'{label}\nDINOv3 splat BEV')
        plot_bev(axes[row][2], colorize(sample['bevformer'], sample['bevformer_mask'], pcas['bevformer']),
                 extent, f'{label}\nBEVFormer encoder BEV')
        plot_lidar(axes[row][3], sample['cloud'], extent,
                   f"{label}\nLiDAR + GT ({len(sample['boxes'])} boxes)",
                   boxes=sample['boxes'], point_size=point_size)
    fig.suptitle(
        'DINOv3 2D features -> parameter-free lift-and-splat BEV, next to the detection-pretrained BEVFormer BEV\n'
        'BEV panels share XY limits | LiDAR color = radial distance (nuScenes style) | box color = GT class',
        fontsize=16)
    fig.savefig(output / 'overview.png', dpi=150)
    plt.close(fig)


# ----------------------------------------------------------------------------
# Data helpers
# ----------------------------------------------------------------------------

def load_gt_boxes(dataset, index, bounds):
    """Use the dataset's own box conversion so GT shares the LiDAR frame."""
    annotations = dataset.get_ann_info(index)
    boxes = annotations['gt_bboxes_3d']
    labels = np.asarray(annotations['gt_labels_3d'])
    centers = boxes.gravity_center.cpu().numpy()
    valid = (labels >= 0) & np.isfinite(centers).all(1)
    valid &= (centers >= bounds[:3]).all(1) & (centers < bounds[3:]).all(1)
    selected = boxes[torch.from_numpy(valid)]
    if not len(selected):
        return []
    # Native LiDARInstance3DBoxes bottom face in cyclic order. Do not reinterpret yaw.
    corners = selected.corners[:, [0, 3, 7, 4], :2].cpu().numpy()
    names = [dataset.CLASSES[int(label)] for label in labels[valid]]
    return [dict(corners=corner, name=name, color=CLASS_COLORS.get(name, '#e31a1c'),
                 box=box.tolist())
            for corner, name, box in zip(corners, names, selected.tensor.cpu().numpy())]


def load_cloud(info, bounds):
    path = Path(info['lidar_path'])
    if not path.is_file():
        return None, str(path)
    cloud = np.fromfile(path, dtype=np.float32).reshape(-1, 5)[:, :3]
    keep = np.isfinite(cloud).all(1) & (cloud >= bounds[:3]).all(1) & (cloud < bounds[3:]).all(1)
    return cloud[keep], str(path)


def build_split_dataset(cfg, split, point_cloud_range, load_annos=False):
    """Build one split; ``load_annos`` is needed when the pipeline collects GT."""
    from mmdet3d.datasets import build_dataset
    settings = copy.deepcopy(cfg.data[split])
    settings.pop('samples_per_gpu', None)
    if load_annos and settings.get('test_mode', False):
        settings['load_annos'] = True
    return build_dataset(settings, default_args=dict(pc_range=point_cloud_range))


# ----------------------------------------------------------------------------

def main():
    from mmcv import Config, DictAction
    from mmcv.parallel import collate

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--bev-config',
                        default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_v2.py'),
                        help='Stage 1 config whose model section defines the BEVFormer encoder')
    parser.add_argument('--bev-checkpoint', default=str(BEVFORMER / 'ckpts/bevformer_tiny_epoch_24.pth'))
    parser.add_argument('--dino-config',
                        default=str(BEVFORMER / 'projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_dino.py'))
    parser.add_argument('--cfg-options', nargs='+', action=DictAction,
                        help='Overrides applied to BOTH configs')
    parser.add_argument('--split', choices=['train', 'val'], default='val')
    parser.add_argument('--num-frames', type=int, default=6)
    parser.add_argument('--indices', nargs='+', type=int, help='Overrides evenly spaced selection')
    parser.add_argument('--output-dir', default=str(REPO / 'results/visualize/bev_feature_compare'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--dino-pipeline-config',
                        help='Config whose train/test pipeline replaces the DINO dataset pipeline, '
                             'e.g. projects/configs/bevformer/bevformer_base.py for full-resolution '
                             'images. Only the image preprocessing changes; geometry comes from the '
                             'pipeline itself, so lidar2img and the aligned depth follow it.')
    parser.add_argument('--dino2d-pca', choices=['per-view', 'shared'], default='per-view',
                        help='Color basis for the 2D panels. per-view maximizes contrast inside '
                             'each image; shared makes colors comparable between image panels.')
    parser.add_argument('--dino-input-scale', type=float, default=1.0,
                        help='Resize the image fed to DINOv3 by this factor. 2.0 doubles the '
                             'patch grid; the projector geometry is unchanged.')
    parser.add_argument('--dino-model', help='DINOv3 checkpoint id, e.g. '
                        'facebook/dinov3-vitb16-pretrain-lvd1689m. Overrides the config.')
    parser.add_argument('--model-path', help='Local HF snapshot directory for DINOv3')
    parser.add_argument('--point-size', type=float, default=0.6, help='LiDAR scatter marker size')
    parser.add_argument('--no-cameras', action='store_true', help='Omit the camera row in per-frame figures')
    parser.add_argument('--temporal-test-bev', action='store_true',
                        help='Keep BEVFormer video_test_mode; only meaningful for consecutive frames')
    parser.add_argument('--save-arrays', action='store_true', help='Also store the raw feature maps as npz')
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error('--num-frames must be positive')

    output = Path(args.output_dir).resolve()
    bev_config, dino_config = Path(args.bev_config).resolve(), Path(args.dino_config).resolve()
    checkpoint = Path(args.bev_checkpoint).resolve()
    model_path = str(Path(args.model_path).resolve()) if args.model_path else None
    os.chdir(SCRIPT_DIR)  # Both configs use dataset paths relative to this directory.
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)

    import projects.mmdet3d_plugin  # noqa: F401  Register datasets, pipelines and models.
    import projects.bevdiffuser.data_utils  # noqa: F401  Register the diffusion datasets.

    dino_cfg = Config.fromfile(str(dino_config))
    bev_cfg_raw = Config.fromfile(str(bev_config))
    if args.cfg_options:
        dino_cfg.merge_from_dict(args.cfg_options)
        bev_cfg_raw.merge_from_dict(args.cfg_options)
    pipeline_source = None
    if args.dino_pipeline_config:
        # The dataset derives lidar2img and the depth warp from the pipeline it
        # runs, so swapping the pipeline moves the whole image frame coherently.
        pipeline_source = Path(args.dino_pipeline_config).resolve()
        source_cfg = Config.fromfile(str(pipeline_source))
        key = 'train_pipeline' if args.split == 'train' else 'test_pipeline'
        dino_cfg.data[args.split].pipeline = copy.deepcopy(source_cfg[key])
    if args.dino_model:
        # Only the 2D/BEV feature width changes; the projector geometry is shared.
        dino_cfg.dino_bev_extractor.encoder.update(model_name=args.dino_model)
    if model_path:
        dino_cfg.dino_bev_extractor.encoder.update(model_name=model_path, local_files_only=True)

    print('Loading BEVFormer encoder...', flush=True)
    bev_model, bev_cfg = build_bevformer(bev_config, checkpoint, device, args.temporal_test_bev)
    if args.cfg_options:
        bev_cfg = bev_cfg_raw
    print('Loading frozen DINOv3...', flush=True)
    extractor = DINOv3BEVExtractor(**dino_cfg.dino_bev_extractor).to(device).eval()
    if args.dino_input_scale <= 0:
        parser.error('--dino-input-scale must be positive')
    apply_input_scale(extractor.encoder, args.dino_input_scale)

    if (bev_cfg.bev_h_, bev_cfg.bev_w_) != (dino_cfg.bev_h_, dino_cfg.bev_w_):
        raise ValueError('Both configs must use the same BEV grid for a side-by-side comparison')
    if tuple(bev_cfg.point_cloud_range) != tuple(dino_cfg.point_cloud_range):
        raise ValueError('Both configs must use the same point cloud range')
    bev_h, bev_w = int(bev_cfg.bev_h_), int(bev_cfg.bev_w_)
    bounds = np.asarray(bev_cfg.point_cloud_range, dtype=np.float64)
    extent = bounds[[0, 3, 1, 4]].tolist()

    # The Stage 1 BEVFormer pipeline collects gt_bboxes_3d, so test mode must
    # still read annotations; the DINO pipeline collects images only.
    bev_dataset = build_split_dataset(bev_cfg, args.split, bev_cfg.point_cloud_range, load_annos=True)
    dino_dataset = build_split_dataset(dino_cfg, args.split, dino_cfg.point_cloud_range)
    dino_index = {info['token']: i for i, info in enumerate(dino_dataset.data_infos)}

    indices = args.indices or np.linspace(0, len(bev_dataset) - 1,
                                          min(args.num_frames, len(bev_dataset)), dtype=int).tolist()
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(bev_dataset) for i in indices):
        raise ValueError('Frame indices must be unique and inside the dataset')
    output.mkdir(parents=True, exist_ok=True)
    if args.save_arrays:
        (output / 'arrays').mkdir(exist_ok=True)

    samples = []
    for index in indices:
        info = bev_dataset.data_infos[index]
        token = info['token']
        if token not in dino_index:
            raise KeyError(f'Sample {token} is missing from the DINO dataset split')

        bev_batch = collate([bev_dataset[index]], samples_per_gpu=1)
        bevformer = bevformer_bev(bev_model, bev_batch, bev_h, bev_w, int(bev_cfg._dim_))

        dino_batch = collate([dino_dataset[dino_index[token]]], samples_per_gpu=1)
        dino, dino_mask, support, patch_maps, images, meta = dino_bev(
            extractor, dino_batch, device, input_scale=args.dino_input_scale)
        if meta['sample_idx'] != token:
            raise ValueError('The DINO dataset returned a different frame than requested')
        if not np.isfinite(bevformer).all() or not np.isfinite(dino).all():
            raise ValueError(f'Non-finite BEV feature for frame {index}')
        if not all(np.isfinite(patch).all() for patch in patch_maps.values()):
            raise ValueError(f'Non-finite DINOv3 2D feature for frame {index}')
        if not dino_mask.any():
            raise ValueError(f'No observed DINO BEV cells for frame {index}')

        cloud, lidar_path = load_cloud(info, bounds)
        boxes = load_gt_boxes(bev_dataset, index, bounds)
        record = dict(
            index=index, split=args.split, token=token, scene_token=info.get('scene_token'),
            timestamp=info.get('timestamp'),
            bevformer_shape=list(bevformer.shape), dino_shape=list(dino.shape),
            dino_observed_fraction=float(dino_mask.mean()),
            dino_patch_grid={name: list(patch.shape[1:]) for name, patch in patch_maps.items()},
            bevformer_norm_mean=float(np.linalg.norm(bevformer, axis=0).mean()),
            dino_norm_mean=float(np.linalg.norm(dino[:, dino_mask], axis=0).mean()),
            lidar_path=lidar_path,
            lidar_points_in_range=int(len(cloud)) if cloud is not None else None,
            gt_box_count=len(boxes), gt_names=[box['name'] for box in boxes],
            camera_files=list(meta['filename']))
        if args.save_arrays:
            name = f"arrays/frame_{index:05d}_{token}.npz"
            np.savez_compressed(output / name, bevformer_bev=bevformer, dino_bev=dino,
                                dino_observed_mask=dino_mask, dino_support=support,
                                **{f'dino_2d_{view}': patch for view, patch in patch_maps.items()})
            record['arrays'] = name
        samples.append(dict(
            bevformer=bevformer, bevformer_mask=np.ones((bev_h, bev_w), dtype=bool),
            dino=dino, dino_mask=dino_mask, support=support, patch_maps=patch_maps,
            cloud=cloud, boxes=boxes,
            cameras=camera_images(images, meta) if not args.no_cameras else {},
            record=record))
        print(f'Extracted {len(samples)}/{len(indices)}: index={index}, '
              f'DINO observed={dino_mask.mean():.1%}, GT boxes={len(boxes)}', flush=True)

    # BEV panels keep one basis per source across frames, so a color means the
    # same thing in every frame. The 2D panels default to a per-view basis: a
    # shared basis spends its three components on scene-level contrast between
    # views and washes out the structure inside each image.
    pcas = dict(
        bevformer=fit_pca([s['bevformer'] for s in samples], [s['bevformer_mask'] for s in samples]),
        dino=fit_pca([s['dino'] for s in samples], [s['dino_mask'] for s in samples]))
    np.savez(output / 'pca_bevformer.npz', **pcas['bevformer'])
    np.savez(output / 'pca_dino.npz', **pcas['dino'])
    patch_variance = []
    if args.dino2d_pca == 'shared':
        patches = [patch for sample in samples for patch in sample['patch_maps'].values()]
        pcas['dino2d'] = fit_pca(patches, [np.ones(p.shape[1:], dtype=bool) for p in patches])
        np.savez(output / 'pca_dino2d.npz', **pcas['dino2d'])
        patch_variance.append(pcas['dino2d']['explained_variance_ratio'].sum())
    for sample in samples:
        sample['patch_rgb'] = {}
        for name, patch in sample['patch_maps'].items():
            mask = np.ones(patch.shape[1:], dtype=bool)
            pca = pcas['dino2d'] if args.dino2d_pca == 'shared' else fit_pca([patch], [mask])
            if args.dino2d_pca != 'shared':
                patch_variance.append(pca['explained_variance_ratio'].sum())
            sample['patch_rgb'][name] = colorize(patch, mask, pca)
    for sample in samples:
        render_frame(sample, pcas, extent, output, with_cameras=not args.no_cameras,
                     point_size=args.point_size)
    render_overview(samples, pcas, extent, output, camera_order=list(dino_cfg.camera_types),
                    point_size=args.point_size)

    summary = dict(
        bev_config=str(bev_config), bev_checkpoint=str(checkpoint), dino_config=str(dino_config),
        split=args.split, device=str(device),
        bevformer=dict(source='BEVFormer encoder, only_bev=True, detection-pretrained',
                       channels=int(bev_cfg._dim_), grid=[bev_h, bev_w],
                       video_test_mode=bool(args.temporal_test_bev),
                       explained_variance_ratio=pcas['bevformer']['explained_variance_ratio'].tolist(),
                       pca_fitted_cells=int(pcas['bevformer']['fitted_cells'])),
        dino_image_pipeline=dict(
            source=str(pipeline_source) if pipeline_source else str(dino_config),
            transforms=[step['type'] for step in dino_cfg.data[args.split].pipeline]),
        dino_2d=dict(source='frozen DINOv3 final-layer patch tokens in the augmented image frame',
                     model_name=extractor.encoder.model_name,
                     patch_size=int(extractor.encoder.patch_size),
                     channels=int(extractor.encoder.hidden_size),
                     pca_mode=args.dino2d_pca,
                     mean_top3_explained_variance=float(np.mean(patch_variance))),
        dino=dict(source='frozen DINOv3 patch tokens lifted by Metric3D depth, no learnable parameters',
                  model_name=extractor.encoder.model_name, revision=extractor.encoder.revision,
                  channels=int(extractor.encoder.hidden_size), grid=[bev_h, bev_w],
                  input_scale=float(args.dino_input_scale),
                  projector=dino_cfg.dino_bev_extractor['projector'],
                  explained_variance_ratio=pcas['dino']['explained_variance_ratio'].tolist(),
                  pca_fitted_cells=int(pcas['dino']['fitted_cells'])),
        lidar_visualization=dict(style='nuscenes devkit render_sample_data',
                                 color='radial distance, viridis, min(1, d / 51.2 / sqrt(2))',
                                 background='white', gt_outline='per detection class',
                                 gt_source='dataset.get_ann_info; mapped classes with centers inside XYZ range'),
        rgb_percentiles=[2, 98], samples=[s['record'] for s in samples])
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2))
    (output / 'README.md').write_text(
        '# BEV feature source comparison\n\n'
        '- `overview.png`: 선택한 모든 프레임을 한 장에 비교. 행마다 DINOv3 2D feature(6개 뷰 타일), '
        'DINOv3 BEV, BEVFormer BEV, LiDAR.\n'
        '- `frame_*.png`: 카메라 6장과 그 바로 아래의 DINOv3 2D feature 6장, 그리고 세 패널'
        '(BEVFormer BEV, DINOv3 BEV, LiDAR + GT box).\n'
        '- `pca_dino2d.npz`, `pca_dino.npz`, `pca_bevformer.npz`: 소스별 PCA 기저와 색상 정규화 범위.\n'
        '- `statistics.json`: 설정, 프레임 token, patch grid 크기, 관측 비율, PCA 설명 분산.\n'
        '- `arrays/`: `--save-arrays`를 준 경우의 원본 feature map(두 BEV와 뷰별 2D patch feature).\n\n'
        'DINOv3 2D 패널은 frozen DINOv3 마지막 layer의 patch token입니다. 해상도는 픽셀이 아니라 '
        'patch 격자(patch_size=16)이며, padding만 덮는 행과 열은 잘라냈습니다. 이 2D feature를 '
        'Metric3D metric depth로 역투영해 BEV에 splat한 결과가 DINOv3 BEV 패널이고, 이 경로에는 '
        '학습 파라미터가 없습니다. BEVFormer 패널은 3D detection으로 학습된 BEVFormer encoder가 '
        '만든 BEV feature입니다.\n\n'
        'PCA는 세 소스에 대해 따로 적합했습니다. 2D PCA는 모든 뷰와 프레임에 공통이므로 이미지 '
        '패널끼리는 색을 비교할 수 있지만, 2D와 BEV 패널 사이의 색 일치는 의미가 없습니다. '
        'Splatting이 patch vector를 평균하므로 공분산 자체가 달라집니다. 색상은 의미 클래스가 '
        '아니라 상위 3개 주성분입니다.\n\n'
        'DINOv3 BEV에서 회색 셀은 어떤 카메라도 관측하지 못한 0 벡터이며 free space 판정이 아닙니다. '
        'BEVFormer BEV는 attention으로 모든 셀이 채워지므로 관측 mask가 없습니다.\n\n'
        'LiDAR 패널은 nuScenes devkit 스타일입니다. 흰 배경에 점 색상은 원점에서의 수평 거리이고, '
        '빨간 십자가 ego 원점입니다. 박스는 detection 클래스별 색으로 그린 GT 3D box의 바닥면 '
        '투영이며, 중심에서 앞면으로 향하는 선이 heading입니다. LiDAR와 GT는 비교용이며 어떤 '
        'feature 생성에도 사용하지 않습니다.\n', encoding='utf-8')
    print('Saved visualization to', output, flush=True)


if __name__ == '__main__':
    main()
