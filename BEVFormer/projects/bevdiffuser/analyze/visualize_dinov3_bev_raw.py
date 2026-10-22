# SPDX-License-Identifier: AGPL-3.0
"""Lift DINOv3 features to BEV from the raw image frame, beside the LiDAR view.

The counterpart of ``SAM3/visualize_sam3_bev.py``: same frames, same BEV grid,
same LiDAR panel, same structure metrics, so the two encoders can be compared
directly. Both run on the ORIGINAL image frame rather than the dataset's
augmented padded one:

  * Metric3D depth is read at its native 900x1600 and paired with
    ``metric3d_lidar2img_raw``, so nothing is resampled and the depth pixel grid
    is the original image grid.
  * The image is resized only to the nearest multiple of the patch size, a
    uniform per-axis scale of the whole frame, so the patch grid still covers
    exactly the image content.
  * ``sam3_bev_extractor.SAM3FeatureProjector`` normalizes sample coordinates by
    each view's content shape, which is what both of those facts require. It is
    encoder-agnostic: all it assumes is that the feature map spans the content.

Unlike ``dinov3_bev_extractor.FrozenDINOv3Encoder``, there is no padding to undo
and no dataset normalization to invert, because the JPEG is read directly.

    cd BEVFormer/projects/bevdiffuser
    python visualize_dinov3_bev_raw.py --num-frames 6 --bev-grid 200 200
"""

import argparse
import copy
import json
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

SCRIPT_DIR = Path(__file__).resolve().parent
BEVFORMER = SCRIPT_DIR.parents[1]
REPO = BEVFORMER.parent
sys.path.insert(0, str(BEVFORMER))
sys.path.insert(0, str(SCRIPT_DIR))

from sam3_bev_extractor import SAM3FeatureProjector  # noqa: E402  encoder-agnostic
from BEVFormer.projects.bevdiffuser.analyze.visualize_bev_feature_compare import (colorize, fit_pca, load_cloud,  # noqa: E402
                                           load_gt_boxes, plot_bev, plot_lidar)


class RawFrameDINOv3:
    """Frozen DINOv3 on the untouched JPEG, with its own normalization."""

    def __init__(self, model_name, device, dtype='float32', view_batch_size=2):
        from transformers import AutoImageProcessor, AutoModel

        processor = AutoImageProcessor.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device).eval().requires_grad_(False)
        self.patch_size = int(self.model.config.patch_size)
        self.hidden_size = int(self.model.config.hidden_size)
        self.num_register_tokens = int(self.model.config.num_register_tokens)
        self.mean = torch.tensor(processor.image_mean, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(processor.image_std, device=device).view(1, 3, 1, 1)
        self.num_layers = int(self.model.config.num_hidden_layers)
        self.device, self.dtype = device, dtype
        self.view_batch_size = int(view_batch_size)
        self.model_name = model_name

    def _preprocess(self, images):
        """Resize to a multiple of the patch size; this is a uniform scale."""
        pixels = torch.stack([torch.from_numpy(np.asarray(image)).permute(2, 0, 1)
                              for image in images]).to(self.device).float() / 255.0
        height, width = pixels.shape[-2:]
        target = (max(self.patch_size, round(height / self.patch_size) * self.patch_size),
                  max(self.patch_size, round(width / self.patch_size) * self.patch_size))
        if target != (height, width):
            pixels = F.interpolate(pixels, size=target, mode='bicubic',
                                   align_corners=False, antialias=True).clamp(0, 1)
        return (pixels - self.mean) / self.std

    @torch.no_grad()
    def __call__(self, images, blocks):
        """Patch features per transformer block, each ``[V, C, Hp, Wp]``.

        ``blocks`` are 1-based block indices; ``num_layers`` is the final block,
        for which this reproduces ``last_hidden_state`` exactly.

        HF stores ``hidden_states[-1]`` ALREADY normed - it is literally
        ``last_hidden_state`` - while entries 1..L-1 are raw block outputs. So
        the final LayerNorm is applied to the intermediate taps only; applying
        it to the last one as well would normalize twice and give a map that
        matches nothing the model produces. Normalizing the mid blocks is the
        convention DINO's own ``get_intermediate_layers(norm=True)`` uses, and
        puts every tap on a common scale.
        """
        pixels = self._preprocess(images)
        height, width = pixels.shape[-2:]
        rows, columns = height // self.patch_size, width // self.patch_size
        use_amp = self.dtype != 'float32'
        parts = {index: [] for index in blocks}
        with torch.autocast(device_type=self.device.type,
                            dtype=getattr(torch, self.dtype) if use_amp else torch.bfloat16,
                            enabled=use_amp):
            for chunk in pixels.split(self.view_batch_size):
                states = self.model(pixel_values=chunk, return_dict=True,
                                    output_hidden_states=True).hidden_states
                expected = (chunk.shape[0], 1 + self.num_register_tokens + rows * columns,
                            self.hidden_size)
                for index in blocks:
                    # hidden_states[0] is the embedding output, so block i is at i.
                    tokens = (states[index] if index == self.num_layers
                              else self.model.norm(states[index]))
                    if tuple(tokens.shape) != expected:
                        raise ValueError(f'Unexpected DINOv3 token shape {tuple(tokens.shape)}; '
                                         f'expected {expected}')
                    parts[index].append(tokens[:, 1 + self.num_register_tokens:].float())
        return {index: torch.cat(chunks).transpose(1, 2).reshape(-1, self.hidden_size, rows, columns)
                for index, chunks in parts.items()}


def feature_statistics(bev, mask):
    """Cell-level structure over observed cells; matches the SAM3 BEV script."""
    vectors = bev[:, mask].T.astype(np.float64)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    upper = np.triu_indices(len(unit), 1)
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov((vectors - vectors.mean(0)).T)), 0, None)
    direction = vectors.mean(0)
    direction /= np.maximum(np.linalg.norm(direction), 1e-9)
    return dict(observed_cells=int(mask.sum()),
                mean_abs_cosine=float(np.abs(unit @ unit.T)[upper].mean()),
                effective_dim=float(eigenvalues.sum() ** 2 / max((eigenvalues ** 2).sum(), 1e-30)),
                mean_direction_alignment=float(np.abs(unit @ direction).mean()))


def layer_label(block, total):
    return f'block {block}/{total}' + (' (final)' if block == total else ' (mid)')


def render(samples, blocks, extent, output, point_size, label, total_blocks, dpi=150):
    pcas = {}
    for block in blocks:
        pcas[block] = fit_pca([s['bev'][block] for s in samples],
                              [s['mask'][block] for s in samples])
        np.savez(output / f'pca_block{block:02d}.npz', **pcas[block])

    columns = len(blocks) + 1
    for sample in samples:
        record = sample['record']
        fig, axes = plt.subplots(1, columns, squeeze=False,
                                 figsize=(6.3 * columns, 6.4), constrained_layout=True)
        axes = axes[0]
        for axis, block in zip(axes, blocks):
            entry = record['blocks'][str(block)]
            plot_bev(axis, colorize(sample['bev'][block], sample['mask'][block], pcas[block]),
                     extent, f"{label} | {layer_label(block, total_blocks)}\n"
                             f"observed {entry['observed_fraction']:.1%} | "
                             f"eff.dim {entry['structure']['effective_dim']:.1f}")
        plot_lidar(axes[len(blocks)], sample['cloud'], extent,
                   f"LiDAR top view + GT ({len(sample['boxes'])} boxes)",
                   boxes=sample['boxes'], point_size=point_size)
        fig.suptitle(f"DINOv3 features lifted to BEV by depth | val index {record['index']} | "
                     f"{record['token']}\nRaw 900x1600 depth frame, identical geometry for "
                     'every block; PCA fitted per block over all frames', fontsize=14)
        filename = f"frame_{record['index']:05d}_{record['token']}.png"
        fig.savefig(output / filename, dpi=dpi)
        plt.close(fig)
        record['figure'] = filename

    rows = len(samples)
    fig, axes = plt.subplots(rows, columns, squeeze=False,
                             figsize=(5.5 * columns, 5.4 * rows), constrained_layout=True)
    for row, sample in enumerate(samples):
        tag = f"val {sample['record']['index']} | {sample['record']['token'][:10]}"
        for column, block in enumerate(blocks):
            plot_bev(axes[row][column],
                     colorize(sample['bev'][block], sample['mask'][block], pcas[block]),
                     extent, f'{tag}\n{layer_label(block, total_blocks)}')
        plot_lidar(axes[row][len(blocks)], sample['cloud'], extent,
                   f"{tag}\nLiDAR + GT ({len(sample['boxes'])} boxes)",
                   boxes=sample['boxes'], point_size=point_size)
    fig.suptitle(f'{label} BEV targets by transformer depth, against the measured LiDAR\n'
                 'Gray cells are unobserved by every camera, not free space | '
                 'LiDAR color = radial distance (nuScenes style) | box color = GT class',
                 fontsize=16)
    fig.savefig(output / 'overview.png', dpi=dpi)
    plt.close(fig)
    return pcas


def main():
    from mmcv import Config
    from mmcv.parallel import collate
    from mmdet3d.datasets import build_dataset

    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--config', default=str(BEVFORMER / 'projects/configs/bevdiffuser/'
                                                'bev_tiny_onlyseg_sam_v2_metric3d_dino.py'))
    parser.add_argument('--model', default='facebook/dinov3-vitb16-pretrain-lvd1689m')
    parser.add_argument('--num-frames', type=int, default=6)
    parser.add_argument('--indices', nargs='+', type=int)
    parser.add_argument('--blocks', nargs='+', type=int, default=[6, 9, 12],
                        help='1-based transformer blocks to tap; the last one reproduces '
                             'last_hidden_state. Default taps two mid blocks and the final one.')
    parser.add_argument('--bev-grid', nargs=2, type=int, default=(200, 200))
    parser.add_argument('--depth-range', nargs=2, type=float, default=(2.5, 170.0))
    parser.add_argument('--pixel-stride', type=int, default=1)
    parser.add_argument('--point-chunk-size', type=int, default=65536)
    parser.add_argument('--output-dir',
                        default=str(REPO / 'results/visualize/dinov3B_feature_bev_200x200'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--inference-dtype', default='float32',
                        choices=['float32', 'bfloat16', 'float16'])
    parser.add_argument('--point-size', type=float, default=0.6)
    parser.add_argument('--save-arrays', action='store_true')
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error('--num-frames must be positive')

    os.chdir(SCRIPT_DIR)  # The config's dataset paths are relative to it.
    torch.manual_seed(0)
    np.random.seed(0)
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.save_arrays:
        (output / 'arrays').mkdir(exist_ok=True)

    import projects.mmdet3d_plugin  # noqa: F401
    import projects.bevdiffuser.data_utils  # noqa: F401

    cfg = Config.fromfile(args.config)
    settings = copy.deepcopy(cfg.data.val)
    settings.pop('samples_per_gpu', None)
    dataset = build_dataset(settings, default_args=dict(pc_range=cfg.point_cloud_range))
    dataset.CLASSES = cfg.class_names
    bounds = np.asarray(cfg.point_cloud_range, dtype=np.float64)
    bev_h, bev_w = args.bev_grid

    indices = args.indices or np.linspace(0, len(dataset) - 1,
                                          min(args.num_frames, len(dataset)), dtype=int).tolist()
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError('Frame indices must be unique and inside the dataset')

    print(f'Loading {args.model} ...', flush=True)
    encoder = RawFrameDINOv3(args.model, device, dtype=args.inference_dtype)
    blocks = sorted(set(args.blocks))
    if any(b < 1 or b > encoder.num_layers for b in blocks):
        raise ValueError(f'Blocks must be within 1..{encoder.num_layers}')
    projector = SAM3FeatureProjector(
        bev_h=bev_h, bev_w=bev_w, pc_range=tuple(float(v) for v in bounds),
        depth_range=tuple(args.depth_range), depth_type='camera_z', splat_mode='bilinear',
        pixel_stride=args.pixel_stride, point_chunk_size=args.point_chunk_size).to(device)
    print(f'BEV grid {bev_h}x{bev_w} | cell {(bounds[3] - bounds[0]) / bev_w:.3f} m | '
          f'raw depth frame | {encoder.hidden_size} channels | blocks {blocks} '
          f'of {encoder.num_layers}', flush=True)

    samples = []
    for index in indices:
        batch = collate([dataset[index]], samples_per_gpu=1)

        def payload(key):
            value = batch[key]
            if isinstance(value, (list, tuple)):
                value = value[0]
            return value.data[0]

        metas = payload('img_metas')
        if isinstance(metas[0], (list, tuple)):
            metas = metas[0]
        meta = metas[0]
        info = dataset.data_infos[index]
        if meta['sample_idx'] != info['token']:
            raise ValueError('The dataset returned a different frame than requested')

        # Native-resolution depth with the un-augmented projection: no warp.
        depth = dataset.load_depth_from_filenames(meta['filename'])
        frame = [tuple(int(v) for v in depth.shape[-2:])] * len(meta['filename'])
        geometry = dict(lidar2img=list(np.asarray(meta['metric3d_lidar2img_raw'], dtype=np.float32)),
                        pad_shape=frame, ori_shape=frame, img_shape=frame)

        images = [Image.open(filename).convert('RGB') for filename in meta['filename']]
        if any(image.size[::-1] != frame[0] for image in images):
            raise ValueError('Images and depth maps must share the original frame')
        features = encoder(images, blocks)
        depth_batch = depth[None].to(device)

        bev, mask, block_records = {}, {}, {}
        for block in blocks:
            projected = projector(features[block][None].to(device), depth_batch, [geometry])
            bev[block] = projected['bev_features'][0].float().cpu().numpy()
            mask[block] = projected['observed_mask'][0, 0].cpu().numpy()
            if not np.isfinite(bev[block]).all() or not mask[block].any():
                raise ValueError(f'Unusable BEV feature for block {block} on frame {index}')
            block_records[str(block)] = dict(
                observed_fraction=float(mask[block].mean()),
                structure=feature_statistics(bev[block], mask[block]))
            if args.save_arrays:
                np.savez_compressed(output / f'arrays/{index:05d}_block{block:02d}.npz',
                                    bev=bev[block], observed_mask=mask[block],
                                    support=projected['support'][0, 0].cpu().numpy())
        cloud, lidar_path = load_cloud(info, bounds)
        boxes = load_gt_boxes(dataset, index, bounds)
        record = dict(index=int(index), token=meta['sample_idx'],
                      channels=int(encoder.hidden_size), bev_grid=[bev_h, bev_w],
                      feature_grid=[int(v) for v in features[blocks[0]].shape[-2:]],
                      blocks=block_records, lidar_path=lidar_path,
                      lidar_points_in_range=int(len(cloud)) if cloud is not None else None,
                      gt_box_count=len(boxes))
        samples.append(dict(bev=bev, mask=mask, cloud=cloud, boxes=boxes, record=record))
        print(f"Frame {len(samples)}/{len(indices)}: index={index} | " +
              ' | '.join(f"b{b} obs {block_records[str(b)]['observed_fraction']:.1%} "
                         f"eff.dim {block_records[str(b)]['structure']['effective_dim']:.1f}"
                         for b in blocks), flush=True)

    extent = bounds[[0, 3, 1, 4]].tolist()
    label = f"DINOv3 {args.model.split('-')[1]}"
    pcas = render(samples, blocks, extent, output, args.point_size, label, encoder.num_layers)

    def mean_over(block, field):
        return float(np.mean([s['record']['blocks'][str(block)]['structure'][field]
                              for s in samples]))

    summary = dict(
        model=args.model, channels=encoder.hidden_size, patch_size=encoder.patch_size,
        feature_grid=samples[0]['record']['feature_grid'],
        dataset_config=str(Path(args.config).resolve()),
        bev_grid=[bev_h, bev_w], pc_range=[float(v) for v in bounds],
        bev_cell_metres=round((bounds[3] - bounds[0]) / bev_w, 4),
        depth_frame='raw',
        depth_note='Metric3D kept at its native 900x1600 with metric3d_lidar2img_raw, and the '
                   'image resized only to a multiple of the patch size, so depth pixels and '
                   'feature coordinates share the original image frame',
        projector='sam3_bev_extractor.SAM3FeatureProjector (content-frame normalization; '
                  'encoder-agnostic)',
        num_layers=encoder.num_layers,
        layer_note='The final LayerNorm is applied to every tapped block, as in DINO\'s own '
                   'get_intermediate_layers(norm=True), so the blocks are on a common scale. '
                   'The last block reproduces last_hidden_state.',
        blocks={str(block): dict(
            block=block, is_final=block == encoder.num_layers,
            mean_observed_fraction=float(np.mean(
                [s['record']['blocks'][str(block)]['observed_fraction'] for s in samples])),
            mean_abs_cosine=mean_over(block, 'mean_abs_cosine'),
            mean_effective_dim=mean_over(block, 'effective_dim'),
            mean_direction_alignment=mean_over(block, 'mean_direction_alignment'),
            pca_explained_variance_ratio=pcas[block]['explained_variance_ratio'].tolist())
            for block in blocks},
        samples=[s['record'] for s in samples])
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2))
    (output / 'README.md').write_text(
        '# DINOv3 ViT-B feature BEV\n\n'
        '- `overview.png`: 프레임(행) × [DINOv3 BEV, LiDAR + GT].\n'
        '- `frame_*.png`: 프레임별 BEV와 LiDAR.\n'
        '- `pca_dinov3.npz`: PCA 기저와 색상 범위.\n'
        '- `statistics.json`: 관측 비율과 구조 지표.\n'
        '- `arrays/`: `--save-arrays`를 준 경우의 BEV feature, observed mask, support.\n\n'
        '`results/visualize/sam3_feature_bev_200x200`과 같은 프레임, 같은 200x200 격자, '
        '같은 raw depth 경로, 같은 지표를 사용하므로 두 encoder를 직접 비교할 수 있습니다.\n\n'
        'Metric3D depth는 원본 900x1600 그대로 쓰고 `metric3d_lidar2img_raw`와 짝지으므로 '
        'resampling이 없습니다. 이미지는 patch size 배수로만 resize하며 이는 전체 프레임의 '
        '균일한 scale이라 content와 feature 격자의 대응이 정확히 유지됩니다.\n\n'
        'PCA는 이 run 안에서만 적합했으므로 SAM3 결과와 색을 비교하지 마세요. 구조만 '
        '비교하면 됩니다. 색상은 의미 클래스가 아니라 상위 3개 주성분입니다.\n', encoding='utf-8')
    print('Saved visualization to', output, flush=True)


if __name__ == '__main__':
    main()
