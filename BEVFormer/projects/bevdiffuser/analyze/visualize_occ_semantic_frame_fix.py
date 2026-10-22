"""Compare cached LiDAR-frame and freshly projected ego-frame occupancy priors."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

PROJECT_DIR = Path(__file__).resolve().parent
BEVFORMER_DIR = PROJECT_DIR.parents[1]
REPO_DIR = BEVFORMER_DIR.parent
sys.path.insert(0, str(BEVFORMER_DIR))


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def counts(prediction, target, valid):
    import numpy as np
    return (int(np.count_nonzero(prediction & target & valid)),
            int(np.count_nonzero(prediction & ~target & valid)),
            int(np.count_nonzero(~prediction & target & valid)))


def scores(tp, fp, fn):
    return dict(precision=tp / (tp + fp) if tp + fp else None,
                recall=tp / (tp + fn) if tp + fn else None,
                iou=tp / (tp + fp + fn) if tp + fp + fn else None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-csv', type=Path, default=REPO_DIR / 'results/analysis/occ_semantic_frame_alignment_20261015/selected_frames.csv')
    parser.add_argument('--config', type=Path, default=BEVFORMER_DIR / 'projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py')
    parser.add_argument('--device', default='cpu')
    parser.add_argument('--output', type=Path, default=REPO_DIR / 'results/analysis/occ_semantic_frame_fix_20261015')
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args()
    if args.threads not in (2, 3, 4):
        parser.error('--threads must be 2, 3, or 4')
    selected_path, config_path, output = (path.resolve() for path in (args.selected_csv, args.config, args.output))
    with selected_path.open(newline='') as stream:
        selected = list(csv.DictReader(stream))
    tokens = [row['token'] for row in selected]
    if len(selected) != 24 or len(set(tokens)) != 24 or [int(row['order']) for row in selected] != list(range(1, 25)):
        parser.error('Selected CSV must contain exactly 24 uniquely ordered frames')
    if digest(selected_path) != '007dd43afae27aa153e0f58552c23ed019d4d2f70c2f2ff8781a73bb02e243a5':
        parser.error('Only the approved selected_frames.csv is permitted')
    if output.exists() and any(path.is_file() for path in output.rglob('*')):
        parser.error(f'Output already contains artifacts: {output}')
    if any(not Path(row['cache']).is_file() or not Path(row['gt']).is_file() for row in selected):
        parser.error('A selected cached map or GT file is missing')
    os.environ['OMP_NUM_THREADS'] = str(args.threads)
    os.environ['MKL_NUM_THREADS'] = str(args.threads)
    os.environ['OPENBLAS_NUM_THREADS'] = str(args.threads)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    import mmcv
    from mmcv import Config
    from projects.bevdiffuser.build_semantic_bev_cache import offline_dataset
    from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner
    from projects.bevdiffuser.semantic_bev_cache import SemanticBEVCache

    torch.set_num_threads(args.threads)
    device = torch.device(args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        parser.error('CUDA unavailable')
    os.chdir(PROJECT_DIR)
    cfg = Config.fromfile(str(config_path))
    projection = dict(cfg.unet.parameters.seg_bev_aligner)
    if projection.get('output_frame') != 'ego' or tuple(projection['pc_range']) != (-40, -40, -1.0, 40, 40, 5.4):
        parser.error('Expected corrected occupancy ego-frame 50x50 projection config')
    if projection['bev_h'] != 50 or projection['bev_w'] != 50 or projection['num_classes'] != 16:
        parser.error('Expected 17-channel, 50x50 occupancy semantic projection')
    dataset = offline_dataset(cfg, 'val')
    lookup = {info['token']: index for index, info in enumerate(dataset.data_infos)}
    if any(token not in lookup for token in tokens):
        parser.error('Selected token is not in the configured validation annotations')
    lidar = SegBEVAligner(**{**projection, 'output_frame': 'lidar'}).to(device).eval()
    ego = SegBEVAligner(**projection).to(device).eval()
    cache_roots = {str(Path(row['cache']).parent.resolve()) for row in selected}
    if len(cache_roots) != 1:
        parser.error('Selected CSV spans multiple before-cache roots')
    manifest_path = Path(next(iter(cache_roots))) / 'manifest.json'
    manifest = json.loads(manifest_path.read_text())
    if manifest['contract']['projection'].get('output_frame', 'lidar') != 'lidar':
        parser.error('Before cache must be LiDAR-frame')
    before_cache = SemanticBEVCache(manifest_path.parent, manifest['contract'], required_tokens=tokens)
    output.mkdir(parents=True, exist_ok=True)
    (output / 'after_probabilities').mkdir(exist_ok=True)
    classes = {'cone': (15, 8), 'car': (1, 4), 'barrier': (9, 1), 'pedestrian': (10, 7)}
    rows = []
    runtimes = []
    for ordinal, selected_row in enumerate(selected, 1):
        start = time.perf_counter()
        token = selected_row['token']
        index = lookup[token]
        item = dataset.get_unconditioned_item(index)
        meta = dataset._current_metas(item['img_metas'])
        if meta['sample_idx'] != token or dataset.data_infos[index]['scene_token'] != selected_row['scene_token']:
            raise ValueError(f'Metadata identity mismatch for {token}')
        if tuple(meta['pad_shape'][0][:2]) != (928, 1600) or any(tuple(shape[:2]) != (928, 1600) for shape in meta['pad_shape']):
            raise ValueError(f'Unexpected padded image geometry for {token}')
        before = before_cache.load(token, meta)
        if Path(selected_row['cache']).resolve() != before_cache._path(token).resolve():
            raise ValueError(f'Before cache path mismatch for {token}')
        seg = dataset.load_segmaps(meta['filename'], meta, dataset.semantic_path)
        depth = dataset.load_depth_from_filenames(meta['filename'], meta)
        if seg.shape != depth.shape or tuple(seg.shape[1:]) != (928, 1600):
            raise ValueError(f'Raw-map alignment mismatch for {token}')
        with torch.no_grad():
            seg_input = seg.unsqueeze(0).to(device)
            depth_input = depth.unsqueeze(0).to(device)
            current_lidar, _ = lidar.project_semantics(seg_input, [meta], depth_input)
            after, _ = ego.project_semantics(seg_input, [meta], depth_input)
        current_lidar = current_lidar[0].cpu().numpy()
        after = after[0].cpu().numpy()
        if after.shape != (17, 50, 50) or after.dtype != np.float32 or not np.isfinite(after).all():
            raise ValueError(f'Invalid corrected probability array for {token}')
        with (output / 'after_probabilities' / f'{token}.npy').open('xb') as stream:
            np.save(stream, after, allow_pickle=False)
        with np.load(selected_row['gt'], allow_pickle=False) as gt_file:
            semantics = gt_file['semantics']
            mask = gt_file['mask_camera'].astype(bool)
        if semantics.shape != (200, 200, 16) or mask.shape != semantics.shape:
            raise ValueError(f'Unexpected GT shape for {token}')
        valid = mask.reshape(50, 4, 50, 4, 16).any(axis=(1, 3, 4)).T
        if int(valid.sum()) != int(selected_row['valid_cells']):
            raise ValueError(f'GT footprint mismatch for {token}')
        fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
        for row_index, name in enumerate(('cone', 'car')):
            sam_class, gt_class = classes[name]
            gt_map = ((semantics == gt_class) & mask).reshape(50, 4, 50, 4, 16).any(axis=(1, 3, 4)).T
            for column, (label, array) in enumerate((('GT', gt_map.astype(np.float32)),
                                                       ('Before', before[sam_class]), ('After', after[sam_class]))):
                axis = axes[row_index, column]
                masked = np.ma.masked_where(~valid, array)
                image = axis.imshow(masked, origin='lower', extent=(-40, 40, -40, 40),
                                    vmin=0, vmax=1, cmap='magma', interpolation='nearest')
                image.cmap.set_bad('white')
                axis.set_facecolor('white')
                axis.set_title(f'{name} | {label}')
                axis.set_xlabel('ego x (m)')
                axis.set_ylabel('ego y (m)')
                fig.colorbar(image, ax=axis, fraction=.046, pad=.04, label='GT occupancy / semantic score')
        fig.suptitle(f'{token} | camera-visible GT footprint; white invalid, black zero')
        with (output / f'{token}.png').open('xb') as stream:
            fig.savefig(stream, format='png', dpi=140)
        plt.close(fig)
        for name, (sam_class, gt_class) in classes.items():
            target = ((semantics == gt_class) & mask).reshape(50, 4, 50, 4, 16).any(axis=(1, 3, 4)).T
            for variant, probability in (('before_cache', before), ('current_lidar_cpu' if device.type == 'cpu' else 'current_lidar', current_lidar), ('after_ego_cpu' if device.type == 'cpu' else 'after_ego', after)):
                score = probability[sam_class]
                tp, fp, fn = counts(score >= .1, target, valid)
                rows.append(dict(token=token, scene_token=selected_row['scene_token'], class_name=name,
                                 variant=variant, valid_cells=int(valid.sum()), gt_positive_cells=int(target.sum()),
                                 observed_valid_cells=int(np.count_nonzero((probability.sum(0) > 1e-8) & valid)),
                                 tp=tp, fp=fp, fn=fn, **scores(tp, fp, fn)))
        runtimes.append(dict(token=token, seconds=round(time.perf_counter() - start, 3),
                             before_sha256=digest(selected_row['cache']),
                             after_sha256=hashlib.sha256(after.tobytes()).hexdigest(),
                             current_lidar_sha256=hashlib.sha256(current_lidar.tobytes()).hexdigest(),
                             current_lidar_max_abs_error=float(np.max(np.abs(before - current_lidar)))))
        print(f'{ordinal}/24 {token} {runtimes[-1]["seconds"]}s', flush=True)
    pooled = []
    for name in classes:
        for variant in ('before_cache', 'current_lidar_cpu' if device.type == 'cpu' else 'current_lidar', 'after_ego_cpu' if device.type == 'cpu' else 'after_ego'):
            subset = [row for row in rows if row['class_name'] == name and row['variant'] == variant]
            tp, fp, fn = (sum(row[key] for row in subset) for key in ('tp', 'fp', 'fn'))
            pooled.append(dict(class_name=name, variant=variant, frames=len(subset),
                               valid_cells=sum(row['valid_cells'] for row in subset),
                               gt_positive_cells=sum(row['gt_positive_cells'] for row in subset),
                               tp=tp, fp=fp, fn=fn, **scores(tp, fp, fn)))
    write_csv(output / 'frame_metrics.csv', rows)
    write_csv(output / 'pooled_metrics.csv', pooled)
    write_csv(output / 'runtime_hashes.csv', runtimes)
    summary = dict(selected_csv=str(selected_path), selected_sha256=digest(selected_path),
                   config=str(config_path), config_sha256=digest(config_path),
                   projector_sha256=digest(PROJECT_DIR / 'layout_diffusion/seg_bev_aligner_one_hot_v5.py'),
                   before_manifest_sha256=digest(manifest_path), before_source_version=manifest['contract']['source_version'],
                   device=str(device), cpu_caveat='CPU splatting may differ numerically from historical CUDA cache' if device.type == 'cpu' else None,
                   torch_version=torch.__version__, numpy_version=np.__version__, mmcv_version=mmcv.__version__,
                   threads=args.threads, tokens=tokens, threshold=.1,
                   gt='any camera-visible class voxel and valid mask per 4x4x16 bin, transposed to [Y,X]',
                   image_alignment='production offline_dataset test pipeline and production SAM3/Metric3D readers',
                   pooled=pooled)
    (output / 'summary.json').open('x').write(json.dumps(summary, indent=2, allow_nan=False) + '\n')


if __name__ == '__main__':
    main()
