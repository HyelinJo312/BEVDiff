"""Bounded, paired fixed-head diagnostic for the v1 and v3 occupancy teachers.

Run from BEVFormer/. This is not a Stage 2 student evaluation.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

import mmcv
import numpy as np
import torch
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import projects.mmdet3d_plugin  # noqa: F401
from projects.bevdiffuser.occ_diffusion_utils import decode_occ_bev, preserve_occ_state
from projects.mmdet3d_plugin.datasets.occ3d_nuscenes_dataset import OCC3D_CLASS_NAMES
from tools.bevdiffuser_onlyseg import BEVDiffuser, resolve_unet_checkpoint


ROOT = Path(__file__).resolve().parents[3]
BEV = ROOT / 'BEVFormer'
OLD_LOG = ROOT / 'results/diff_bevformer/occ/DiffBEVFormer_occ_onlyseg_metric3d_alpha0.1_lambda_0.55_t300_cfg4/20261014_102443.log'
NEW_DUMP = ROOT / 'results/diff_bevformer/occ/DiffBEVFormer_occ_onlyseg_metric3d_v3_alpha0.1_lambda_0.55_t300_cfg4/bev_onlyseg_sam_metric3d_occ.py'
STAGE1_CFG = BEV / 'projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py'
BASE_CFG = BEV / 'projects/configs/occ3d/bevformer_base_occ.py'
BASE_WEIGHTS = ROOT / 'results/occ3d/bevformer_base_occ/epoch_24.pth'
SELECTED = ROOT / 'results/analysis/occ_semantic_frame_alignment_20261015/selected_frames.csv'
SCHEDULER = Path('/home/korea_sdv01/.cache/huggingface/hub/models--stabilityai--stable-diffusion-2-1/snapshots/5cae40e6a2745ae2b01ad92ae5043f95f23644d6')
DEFAULT_OUTPUT = ROOT / 'results/analysis/occ_teacher_paired_5step_20261018'
SYSTEMS = ('v1', 'v3')


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(8 << 20), b''):
            digest.update(block)
    return digest.hexdigest()


def old_config():
    log = OLD_LOG.read_text()
    match = re.search(r' - INFO - Config:\n(.*?)(?=\n\d{4}-\d\d-\d\d .*? - INFO - Set random seed)', log, re.S)
    if not match:
        raise ValueError('Cannot isolate exact v1 config dump from run log')
    snapshot = match.group(1) + '\n'
    return Config.fromstring(snapshot, '.py'), hashlib.sha256(snapshot.encode()).hexdigest()


def selected_frames(limit):
    with SELECTED.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 24 or len({row['token'] for row in rows}) != 24:
        raise ValueError('Expected exactly 24 distinct fixed tokens')
    if limit not in (1, 24):
        raise ValueError('Only one-frame smoke or the exact 24-frame diagnostic is allowed')
    return rows[:limit]


def predecessor_indices(infos, index, count=4):
    scene = infos[index]['scene_token']
    preceding = [item for item in range(index + 1) if infos[item]['scene_token'] == scene]
    selected = preceding[-count:]
    indices = [selected[0]] * (count - len(selected)) + selected
    if any(infos[item]['scene_token'] != scene for item in indices):
        raise AssertionError('Temporal queue crossed a scene boundary')
    if any(infos[left]['timestamp'] > infos[right]['timestamp'] for left, right in zip(indices, indices[1:])):
        raise AssertionError('Temporal queue timestamps are not ordered')
    return indices, count - len(selected)


def gt_footprint(volume, visible, class_id):
    """Occ3D voxels are [X,Y,Z]; native feature maps are [Y,X]."""
    if volume.shape != visible.shape or volume.ndim != 3:
        raise ValueError('Expected matching [X,Y,Z] volumes')
    return ((volume == class_id) & visible).any(axis=2).T


def confusion(pred, target, visible):
    if pred.shape != target.shape or visible.shape != target.shape:
        raise ValueError('Prediction, GT, and mask shapes differ')
    valid = visible & (target < 18) & (pred < 18)
    return np.bincount(18 * target[valid].astype(np.int64) + pred[valid].astype(np.int64),
                       minlength=18 * 18).reshape(18, 18)


def class_rows(hist, name):
    rows = []
    for index, label in enumerate(OCC3D_CLASS_NAMES):
        true_positive = int(hist[index, index])
        gt_count = int(hist[index].sum())
        pred_count = int(hist[:, index].sum())
        false_positive = pred_count - true_positive
        false_negative = gt_count - true_positive
        union = gt_count + false_positive
        rows.append(dict(system=name, class_id=index, class_name=label,
                         tp=true_positive, fp=false_positive, fn=false_negative,
                         gt_count=gt_count, pred_count=pred_count,
                         iou=true_positive / union if union else None,
                         precision=true_positive / pred_count if pred_count else None,
                         recall=true_positive / gt_count if gt_count else None))
    return rows


def bev_map(tokens):
    if tokens.ndim == 3 and tokens.shape[1:] == (40000, 256):
        return tokens.reshape(1, 200, 200, 256).permute(0, 3, 1, 2).contiguous()
    if tokens.ndim == 3 and tokens.shape == (40000, 1, 256):
        return tokens.permute(1, 0, 2).reshape(1, 200, 200, 256).permute(0, 3, 1, 2).contiguous()
    raise ValueError(f'Unexpected native BEV token shape: {tuple(tokens.shape)}')


def feature_stats(feature, clean, mask=None):
    if feature.shape != clean.shape or feature.shape[0:2] != (1, 256):
        raise ValueError('Expected matching FP32 [1,256,200,200] BEV maps')
    if mask is not None:
        if not mask.any():
            return dict(mse=None, cosine=None, norm=None, cells=0)
        first = feature[0, :, mask]
        second = clean[0, :, mask]
    else:
        first = feature.flatten(2)[0]
        second = clean.flatten(2)[0]
    mse = (first - second).square().mean()
    cosine = torch.nn.functional.cosine_similarity(first, second, dim=0).mean()
    return dict(mse=float(mse), cosine=float(cosine), norm=float(first.norm(dim=0).mean()), cells=first.shape[1])


def linear_cka(feature, clean):
    first = torch.nn.functional.adaptive_avg_pool2d(feature, (50, 50))[0].flatten(1).T.float()
    second = torch.nn.functional.adaptive_avg_pool2d(clean, (50, 50))[0].flatten(1).T.float()
    first -= first.mean(0, keepdim=True)
    second -= second.mean(0, keepdim=True)
    cross = first.T @ second
    left = first.T @ first
    right = second.T @ second
    denominator = left.norm() * right.norm()
    return float(cross.square().sum() / denominator) if denominator > 0 else None


def stable_seed(token):
    return int.from_bytes(hashlib.sha256(token.encode()).digest()[:4], 'big')


def paired_summary(frame_rows):
    old = {row['token']: row for row in frame_rows if row['system'] == 'v1'}
    new = {row['token']: row for row in frame_rows if row['system'] == 'v3'}
    if old.keys() != new.keys():
        raise AssertionError('Unpaired teacher observations')
    differences = np.array([new[token]['global_mse'] - old[token]['global_mse'] for token in old])
    scenes = sorted({row['scene_token'] for row in frame_rows})
    scene_means = np.array([np.mean([new[token]['global_mse'] - old[token]['global_mse']
                                     for token in old if old[token]['scene_token'] == scene]) for scene in scenes])
    rng = np.random.default_rng(20261018)
    bootstrap = np.mean(rng.choice(scene_means, size=(10000, len(scenes)), replace=True), axis=1)
    return dict(v1_scene_mean_mse=float(np.mean([np.mean([row['global_mse'] for row in old.values()
                                                        if row['scene_token'] == scene]) for scene in scenes])),
                v3_scene_mean_mse=float(np.mean([np.mean([row['global_mse'] for row in new.values()
                                                        if row['scene_token'] == scene]) for scene in scenes])),
                delta_v3_minus_v1_scene_mean=float(scene_means.mean()),
                delta_v3_minus_v1_scene_median=float(np.median(scene_means)),
                v3_lower_mse_scenes=int((scene_means < 0).sum()), scene_count=len(scenes),
                scene_bootstrap_95_ci=np.percentile(bootstrap, [2.5, 97.5]).tolist(),
                frame_mean_delta=float(differences.mean()))


def write_csv(path, rows):
    if not rows:
        return
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, choices=(1, 24), default=24)
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    os.chdir(BEV)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    start = time.monotonic()
    old_cfg, old_snapshot_hash = old_config()
    new_cfg = Config.fromfile(str(NEW_DUMP))
    stage1_cfg = Config.fromfile(str(STAGE1_CFG))
    base_cfg = Config.fromfile(str(BASE_CFG))
    if stage1_cfg.model != base_cfg.model:
        raise ValueError('Stage 1 native BEV/occupancy model diverges from baseline config')
    if old_cfg.model.pts_bbox_head != new_cfg.model.pts_bbox_head:
        raise ValueError('Old and new Stage 2 head configs differ')
    if any(cfg.bev_diffuser_cfg[key] != new_cfg.bev_diffuser_cfg[key]
           for cfg in (old_cfg,) for key in ('prediction_type', 'noise_timesteps', 'denoise_timesteps',
                                               'num_inference_steps', 'cfg_scale', 'use_classifier_guidence')):
        raise ValueError('Stage 2 sampler settings differ')
    if new_cfg.bev_diffuser_cfg.prediction_type != 'sample' or new_cfg.bev_diffuser_cfg.noise_timesteps != 300:
        raise ValueError('Unexpected sampler spec')
    if not (SCHEDULER / 'scheduler/scheduler_config.json').is_file():
        raise FileNotFoundError(SCHEDULER)
    systems = {}
    inputs = [OLD_LOG, NEW_DUMP, STAGE1_CFG, BASE_CFG, BASE_WEIGHTS, SELECTED,
              SCHEDULER / 'scheduler/scheduler_config.json', Path(__file__),
              BEV / 'tools/bevdiffuser_onlyseg.py', BEV / 'projects/bevdiffuser/occ_diffusion_utils.py',
              BEV / 'projects/mmdet3d_plugin/bevformer/detectors/bevformer_occ.py',
              BEV / 'projects/mmdet3d_plugin/datasets/nuscenes_occ.py']
    for name, cfg, cache in (('v1', old_cfg, BEV / 'data/semantic_bev_occ'),
                             ('v3', new_cfg, BEV / 'data/semantic_bev_occ_v2')):
        ckpt = ROOT / 'results/bevdiffuser/occ' / ('BEVDiffuser_occ_onlyseg_sam3_metric3d' + ('_v3' if name == 'v3' else '')) / 'checkpoint-50000'
        weight = Path(resolve_unet_checkpoint(ckpt))
        manifest = cache / 'manifest.json'
        with manifest.open() as stream:
            contract = json.load(stream)
        expected_frame = 'lidar' if name == 'v1' else 'ego'
        if contract['contract']['projection'].get('output_frame', 'lidar') != expected_frame:
            raise ValueError(f'{name} cache frame differs from historical system')
        if cfg.semantic_bev_cache_root != f'data/{cache.name}':
            raise ValueError(f'{name} cache differs from exact run dump')
        systems[name] = dict(config=cfg, cache=cache, weight=weight, manifest=manifest,
                             output_frame=expected_frame)
        inputs.extend([weight, manifest])
    rows = selected_frames(args.limit)
    for row in rows:
        for name in SYSTEMS:
            cache_path = systems[name]['cache'] / (row['token'] + '.npy')
            if not cache_path.is_file():
                raise FileNotFoundError(cache_path)
            probability = np.load(cache_path, mmap_mode='r', allow_pickle=False)
            if probability.shape != (17, 50, 50) or probability.dtype != np.float32:
                raise ValueError(f'Invalid cache array {cache_path}')
            if not np.isfinite(probability).all():
                raise ValueError(f'Non-finite cache array {cache_path}')
            if row['token'] not in json.loads(systems[name]['manifest'].read_text())['entries']:
                raise ValueError(f'Missing manifest entry {name}: {row["token"]}')
            inputs.append(cache_path)
        inputs.append(Path(row['gt']))
    if args.preflight:
        print(json.dumps(dict(status='preflight_ok', frames=len(rows), systems={name: value['output_frame'] for name, value in systems.items()})))
        return
    output = args.output_dir
    if output.exists():
        suffix = 1
        while output.with_name(output.name + f'_{suffix}').exists():
            suffix += 1
        output = output.with_name(output.name + f'_{suffix}')
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one approved GPU with CUDA_VISIBLE_DEVICES')
    device = torch.device('cuda:0')
    provenance = dict(command=' '.join(sys.argv), selected_tokens=[row['token'] for row in rows],
                      old_log_config_snapshot_sha256=old_snapshot_hash,
                      subset_not_full_validation=True, cache_frames={name: systems[name]['output_frame'] for name in SYSTEMS},
                      files_sha256={str(path): sha256(path) for path in inputs},
                      gpu=dict(name=torch.cuda.get_device_name(0), uuid=os.popen('nvidia-smi --query-gpu=uuid --format=csv,noheader -i 7').read().strip()),
                      torch_version=torch.__version__, queue_length=4, dtype='float32',
                      feature_shape=[1, 256, 200, 200], seed_rule='SHA256(token) first 32 bits',
                      scheduler_snapshot=str(SCHEDULER))
    source_val = new_cfg.data.val
    dataset_cfg = dict(type='NuSceneOcc', **{key: copy.deepcopy(source_val[key]) for key in
        ('data_root', 'ann_file', 'classes', 'modality', 'bev_size', 'box_type_3d', 'test_mode')})
    pipeline = copy.deepcopy(new_cfg.test_pipeline)
    pipeline = pipeline[:-1] + pipeline[-1]['transforms']
    dataset_cfg['pipeline'] = pipeline
    dataset_cfg['queue_length'] = 4
    dataset = build_dataset(dataset_cfg)
    index_by_token = {info['token']: index for index, info in enumerate(dataset.data_infos)}
    if len(index_by_token) != len(dataset.data_infos):
        raise ValueError('Duplicate val tokens')
    baseline = build_model(copy.deepcopy(stage1_cfg.model))
    load_checkpoint(baseline, str(BASE_WEIGHTS), map_location='cpu', strict=True)
    baseline.to(device).eval().requires_grad_(False)
    teachers = {}
    for name in SYSTEMS:
        cfg = systems[name]['config'].bev_diffuser_cfg
        teacher = BEVDiffuser(cfg.unet_cfg, str(systems[name]['weight']), str(SCHEDULER),
                              prediction_type='sample', noise_timesteps=300, denoise_timesteps=300,
                              num_inference_steps=5, cfg_scale=4.0, use_classifier_guidence=False)
        teacher.to(device).eval().requires_grad_(False)
        if any(parameter.requires_grad for parameter in teacher.parameters()):
            raise AssertionError('Teacher is not frozen')
        teachers[name] = teacher
    frame_rows, feature_rows, class_frame_rows = [], [], []
    totals = {name: np.zeros((18, 18), dtype=np.int64) for name in ('clean', *SYSTEMS)}
    seen_scenes = set()
    for rank, row in enumerate(rows):
        token = row['token']
        index = index_by_token[token]
        info = dataset.data_infos[index]
        if info['scene_token'] != row['scene_token'] or str(info['timestamp']) != row['timestamp']:
            raise ValueError(f'Selected frame metadata mismatch: {token}')
        indices, padded = predecessor_indices(dataset.data_infos, index)
        queue = []
        for queue_index in indices:
            item = dataset.get_data_info(queue_index)
            dataset.pre_pipeline(item)
            example = dataset.pipeline(item)
            if example is None:
                raise ValueError('Deterministic val pipeline returned None')
            queue.append(example)
        merged = dataset.union2one(queue)
        metas = merged['img_metas'].data
        if metas[3]['sample_idx'] != token or metas[3]['scene_token'] != row['scene_token']:
            raise AssertionError('Queue current frame was substituted')
        images = merged['img'].data.unsqueeze(0).to(device, dtype=torch.float32)
        if images.shape[:3] != (1, 4, 6):
            raise ValueError(f'Unexpected image queue shape: {images.shape}')
        with torch.no_grad(), preserve_occ_state(baseline, reset_temporal=True):
            clean = bev_map(baseline.obtain_history_bev(images, [copy.deepcopy(metas)])).float()
            if not torch.isfinite(clean).all():
                raise ValueError('Non-finite clean BEV')
            baseline.eval()
            if any(module.training for module in baseline.modules()):
                raise AssertionError('Baseline decoder must remain eval after history extraction')
            predictions = {}
            outputs = decode_occ_bev(baseline, clean)
            predictions['clean'] = baseline.pts_bbox_head.get_occ(outputs, [metas[3]])[0].cpu().numpy().astype(np.uint8)
            noisy_reference = None
            noise_hash = None
            actual_grid = None
            for name in SYSTEMS:
                probability = np.load(systems[name]['cache'] / (token + '.npy'), allow_pickle=False)
                semantic = torch.from_numpy(probability.copy()).unsqueeze(0).to(device)
                teacher = teachers[name]
                if any(module.training for module in teacher.modules()):
                    raise AssertionError('Teacher must remain eval')
                capture = {}
                original_add_noise = teacher.noise_scheduler.add_noise
                def capture_noise(sample, noise, timesteps):
                    capture['noise'] = hashlib.sha256(noise.cpu().numpy().tobytes()).hexdigest()
                    result = original_add_noise(sample, noise, timesteps)
                    capture['noisy'] = hashlib.sha256(result.cpu().numpy().tobytes()).hexdigest()
                    capture['tensor'] = result.detach().clone()
                    return result
                teacher.noise_scheduler.add_noise = capture_noise
                try:
                    with torch.random.fork_rng(devices=[0]):
                        torch.manual_seed(stable_seed(token))
                        denoised = teacher(clean, [metas[3]], semantic_prob=semantic)
                finally:
                    teacher.noise_scheduler.add_noise = original_add_noise
                if denoised.shape != clean.shape or denoised.dtype != torch.float32:
                    raise ValueError('Teacher output tensor shape/dtype mismatch')
                if not torch.isfinite(denoised).all():
                    raise ValueError('Non-finite denoised BEV')
                grid = [int(item) for item in teacher.noise_scheduler.timesteps]
                if grid != [241, 181, 121, 61, 1]:
                    raise ValueError(f'Unexpected actual DDIM grid: {grid}')
                actual_grid = grid
                if noise_hash is None:
                    noise_hash = capture['noise']
                    noisy_hash = capture['noisy']
                    noisy_reference = capture['tensor']
                elif (noise_hash, noisy_hash) != (capture['noise'], capture['noisy']):
                    raise AssertionError('Paired calls did not receive the same noise and x_t')
                baseline.eval()
                if any(module.training for module in baseline.modules()):
                    raise AssertionError('Fixed head is not eval')
                decoded = decode_occ_bev(baseline, denoised)
                predictions[name] = baseline.pts_bbox_head.get_occ(decoded, [metas[3]])[0].cpu().numpy().astype(np.uint8)
                global_stats = feature_stats(denoised, clean)
                frame_rows.append(dict(token=token, scene_token=row['scene_token'], system=name,
                                       global_mse=global_stats['mse'], global_cosine=global_stats['cosine'],
                                       global_norm=global_stats['norm'], cka50=linear_cka(denoised, clean),
                                       queue_tokens='|'.join(dataset.data_infos[item]['token'] for item in indices),
                                       queue_padding=padded, seed=stable_seed(token), noise_sha256=noise_hash,
                                       x_t_sha256=noisy_hash))
                feature_rows.append(dict(token=token, system=name, region='global', **global_stats))
                if name == 'v1':
                    feature_rows.append(dict(token=token, system='noisy', region='global', **feature_stats(noisy_reference, clean)))
                gt_file = np.load(row['gt'])
                target = gt_file['semantics']
                visible = gt_file['mask_camera'].astype(bool)
                if target.shape != (200, 200, 16):
                    raise ValueError('Unexpected native GT voxel dimensions')
                for class_id in range(17):
                    footprint = gt_footprint(target, visible, class_id)
                    feature_rows.append(dict(token=token, system=name, region=OCC3D_CLASS_NAMES[class_id],
                                             **feature_stats(denoised, clean, torch.from_numpy(footprint).to(device))))
                foreground = ((target >= 1) & (target <= 10) & visible).any(axis=2).T
                static = ((target >= 11) & (target <= 16) & visible).any(axis=2).T
                for region, footprint in (('foreground_1_10', foreground), ('static_11_16', static)):
                    feature_rows.append(dict(token=token, system=name, region=region,
                                             **feature_stats(denoised, clean, torch.from_numpy(footprint).to(device))))
            for name, prediction in predictions.items():
                hist = confusion(prediction, target, visible)
                totals[name] += hist
                class_frame_rows.extend(dict(token=token, **entry) for entry in class_rows(hist, name))
            np.savez_compressed(output / f'pred_{rank + 1:02d}_{token}.npz',
                                clean=predictions['clean'], v1=predictions['v1'], v3=predictions['v3'])
            if rank < 4:
                import matplotlib
                matplotlib.use('Agg')
                import matplotlib.pyplot as plt
                fig, axes = plt.subplots(2, 4, figsize=(16, 8), constrained_layout=True)
                for line, class_id in enumerate((8, 4)):
                    for column, name in enumerate(('GT', 'clean', 'v1', 'v3')):
                        volume = target if name == 'GT' else predictions[name]
                        footprint = gt_footprint(volume, visible, class_id)
                        footprint = np.ma.masked_where(~visible.any(axis=2).T, footprint)
                        axes[line, column].set_facecolor('0.5')
                        axes[line, column].imshow(footprint, origin='lower', extent=(-40, 40, -40, 40),
                                                  vmin=0, vmax=1, cmap='gray')
                        axes[line, column].set_title(f'{OCC3D_CLASS_NAMES[class_id]}: {name}')
                fig.savefig(output / f'visual_{rank + 1:02d}_{token}.png', dpi=120)
                plt.close(fig)
        seen_scenes.add(row['scene_token'])
        print(f'{rank + 1}/{len(rows)} {token} paired; scenes={len(seen_scenes)}', flush=True)
    write_csv(output / 'per_frame.csv', frame_rows)
    write_csv(output / 'per_region_feature.csv', feature_rows)
    write_csv(output / 'per_frame_class.csv', class_frame_rows)
    scene_rows = []
    for scene in sorted(seen_scenes):
        for name in ('v1', 'v3', 'noisy'):
            region_rows = [entry for entry in feature_rows if entry['region'] == 'global'
                           and entry['system'] == name
                           and any(frame['token'] == entry['token'] and frame['scene_token'] == scene
                                   for frame in frame_rows)]
            if region_rows:
                scene_rows.append(dict(scene_token=scene, system=name, frames=len(region_rows),
                                       mse=float(np.mean([entry['mse'] for entry in region_rows])),
                                       cosine=float(np.mean([entry['cosine'] for entry in region_rows])),
                                       norm=float(np.mean([entry['norm'] for entry in region_rows]))))
    write_csv(output / 'per_scene_feature.csv', scene_rows)
    pooled = [entry for name, hist in totals.items() for entry in class_rows(hist, name)]
    write_csv(output / 'pooled_class.csv', pooled)
    summary = dict(frames=len(rows), scenes=len(seen_scenes), paired_feature_mse=paired_summary(frame_rows),
                   pooled_miou_excluding_free={name: float(np.mean([entry['iou'] for entry in pooled
                       if entry['system'] == name and entry['class_id'] != 17 and entry['iou'] is not None]))
                       for name in totals},
                   native_visible_class_support={entry['class_name']: entry['gt_count']
                       for entry in pooled if entry['system'] == 'clean'},
                   actual_ddim_grid=actual_grid, elapsed_seconds=time.monotonic() - start)
    provenance['sampler'] = dict(noise_timestep=300, denoise_timesteps=300, steps=5, cfg=4.0,
                                 prediction_type='sample', eta=0, actual_grid=actual_grid,
                                 initial_add_noise_timestep=300)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2, allow_nan=False))
    (output / 'README.md').write_text('Paired v1 occupancy semantic teacher + LiDAR-frame cache versus v3 teacher + ego-frame cache. '
                                      'A complete-system contrast, not a same-cache checkpoint ablation. Fixed baseline occupancy head '
                                      'measures direct feature compatibility, not Stage 2 predictor quality, an upper bound, or a veto. '
                                      'Clean-reference MSE/cosine/CKA are complementary mechanisms, not proof of transfer. '
                                      'This is a fixed 24-frame subset, not full validation.\n')
    print(json.dumps(dict(output=str(output), summary=summary), indent=2))


if __name__ == '__main__':
    main()
