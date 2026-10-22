"""Paired clean/CFG0/CFG1/CFG2 fixed-head occupancy diagnostic.

Run from BEVFormer with one visible GPU. This is a selected 24-frame probe,
not a full validation run or a Stage 2 student evaluation.
"""

import argparse
import copy
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F
from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import projects.mmdet3d_plugin  # noqa: F401
from BEVFormer.projects.bevdiffuser.analyze.analyze_occ_teacher_pair import (
    BASE_CFG, BASE_WEIGHTS, BEV, ROOT, SCHEDULER, SELECTED, bev_map,
    class_rows, confusion, feature_stats, linear_cka, predecessor_indices,
    selected_frames, sha256, stable_seed,
)
from projects.bevdiffuser.occ_diffusion_utils import decode_occ_bev, preserve_occ_state
from tools.bevdiffuser_onlyseg import BEVDiffuser, resolve_unet_checkpoint


RUN_CONFIG = ROOT / ('results/diff_bevformer/occ/'
    'DiffBEVFormer_occ_onlyseg_metric3d_v3_alpha0.1_lambda_0.55_t100_cfg2/'
    'bev_onlyseg_sam_metric3d_occ.py')
STAGE1_CONFIG = BEV / 'projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py'
DEFAULT_OUTPUT = ROOT / 'results/analysis/occ_condition_ablation_t100_20261019'
ARMS = {'unconditional': 0.0, 'conditional': 1.0, 'guided_cfg2': 2.0}


def selection_rows(path, limit, shard_index=0, num_shards=1):
    if limit < 1 or num_shards < 1 or not 0 <= shard_index < num_shards:
        raise ValueError('Invalid limit or shard index/count')
    if path is None:
        rows = selected_frames(limit)
    else:
        with Path(path).open(newline='') as stream:
            rows = list(csv.DictReader(stream))
        required = {'token', 'scene_token', 'timestamp', 'gt'}
        if not rows or not required <= rows[0].keys() or limit > len(rows):
            raise ValueError('Selection CSV lacks required columns or requested rows')
        rows = rows[:limit]
        if any(not all(row.get(key) for key in required) for row in rows):
            raise ValueError('Incomplete selection row')
        if len({row['token'] for row in rows}) != len(rows):
            raise ValueError('Duplicate selection token')
    return [dict(row, _selection_index=index) for index, row in enumerate(rows)
            if index % num_shards == shard_index]


def visible_cross_entropy(logits, target, visible):
    """Unweighted CE over camera-visible voxels; exclude free only in second view."""
    if logits.ndim != 5 or logits.shape[0] != 1 or logits.shape[-1] != 18:
        raise ValueError(f'Expected [1,X,Y,Z,18] logits, got {tuple(logits.shape)}')
    if tuple(logits.shape[1:-1]) != tuple(target.shape) or target.shape != visible.shape:
        raise ValueError('Logit, GT, and camera mask dimensions differ')
    mask = visible & (target >= 0) & (target < 18)
    nonfree = mask & (target < 17)
    labels = torch.as_tensor(target, device=logits.device, dtype=torch.long)
    mask_t = torch.as_tensor(mask, device=logits.device, dtype=torch.bool)
    nonfree_t = torch.as_tensor(nonfree, device=logits.device, dtype=torch.bool)
    per_voxel = F.cross_entropy(logits[0].float().reshape(-1, 18), labels.reshape(-1), reduction='none')
    def aggregate(region):
        count = int(region.sum())
        return (float(per_voxel[region.reshape(-1)].sum()), count) if count else (0.0, 0)
    return {'all': aggregate(mask_t), 'nonfree': aggregate(nonfree_t)}


def paired_teacher_forward(teacher, clean, meta, semantic, token, arms=None):
    """Use production forward for selected CFG scales with identical token noise."""
    arms = ARMS if arms is None else arms
    if not arms:
        raise ValueError('At least one diffusion arm is required')
    original_scale = teacher.cfg_scale
    original_add_noise = teacher.noise_scheduler.add_noise
    outputs, hashes = {}, {}
    try:
        for arm, scale in arms.items():
            capture = {}
            def capture_noise(sample, noise, timesteps):
                capture['noise'] = hashlib.sha256(noise.detach().cpu().numpy().tobytes()).hexdigest()
                result = original_add_noise(sample, noise, timesteps)
                capture['x_t'] = hashlib.sha256(result.detach().cpu().numpy().tobytes()).hexdigest()
                return result
            teacher.cfg_scale = scale
            teacher.noise_scheduler.add_noise = capture_noise
            with torch.random.fork_rng(devices=[clean.device.index or 0] if clean.is_cuda else []):
                torch.manual_seed(stable_seed(token))
                outputs[arm] = teacher(clean, [meta], semantic_prob=semantic)
            if not {'noise', 'x_t'} <= capture.keys():
                raise AssertionError('Production sampler did not add noise')
            hashes[arm] = capture
    finally:
        teacher.cfg_scale = original_scale
        teacher.noise_scheduler.add_noise = original_add_noise
    if len({item['noise'] for item in hashes.values()}) != 1 or len({item['x_t'] for item in hashes.values()}) != 1:
        raise AssertionError('Condition arms received different noise or x_t')
    for arm, output in outputs.items():
        if output.shape != clean.shape or output.dtype != clean.dtype or not torch.isfinite(output).all():
            raise ValueError(f'Invalid output for {arm}: {tuple(output.shape)}, {output.dtype}')
    return outputs, next(iter(hashes.values()))


def summarize(totals, ce_totals):
    pooled = [item for arm, hist in totals.items() for item in class_rows(hist, arm)]
    summary = {}
    for arm in totals:
        valid_iou = [item['iou'] for item in pooled if item['system'] == arm
                     and item['class_id'] < 17 and item['iou'] is not None]
        summary[arm] = {
            'miou_nonfree_defined_classes': float(np.mean(valid_iou)) if valid_iou else None,
            'defined_classes': len(valid_iou),
            'visible_ce_all': ce_totals[arm]['all'][0] / ce_totals[arm]['all'][1]
                              if ce_totals[arm]['all'][1] else None,
            'visible_ce_nonfree': ce_totals[arm]['nonfree'][0] / ce_totals[arm]['nonfree'][1]
                                  if ce_totals[arm]['nonfree'][1] else None,
            'visible_voxels_all': ce_totals[arm]['all'][1],
            'visible_voxels_nonfree': ce_totals[arm]['nonfree'][1],
        }
    return summary, pooled


def unique_output(path):
    if not path.exists():
        return path
    suffix = 1
    while path.with_name(f'{path.name}_{suffix}').exists():
        suffix += 1
    return path.with_name(f'{path.name}_{suffix}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit', type=int, default=24)
    parser.add_argument('--selection-csv', type=Path, default=None)
    parser.add_argument('--shard-index', type=int, default=0)
    parser.add_argument('--num-shards', type=int, default=1)
    parser.add_argument('--include-t300-cfg4', action='store_true')
    parser.add_argument('--include-t100-cfg4', action='store_true')
    parser.add_argument('--preflight', action='store_true')
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--run-config', type=Path, default=RUN_CONFIG)
    parser.add_argument('--scheduler', type=Path, default=SCHEDULER)
    parser.add_argument('--cfg-scale', type=float, default=None)
    parser.add_argument('--noise-timestep', type=int, default=None)
    parser.add_argument('--denoise-timestep', type=int, default=None)
    parser.add_argument('--conditional-only', action='store_true')
    args = parser.parse_args()
    os.chdir(BEV)
    os.environ['HF_HUB_OFFLINE'] = '1'
    os.environ['TRANSFORMERS_OFFLINE'] = '1'
    start = time.monotonic()
    run_path = args.run_config.resolve()
    cfg = Config.fromfile(str(run_path))
    base_cfg = Config.fromfile(str(BASE_CFG))
    stage1_cfg = Config.fromfile(str(STAGE1_CONFIG))
    if stage1_cfg.model != base_cfg.model:
        raise ValueError('Stage 1 clean BEV model differs from frozen baseline')
    spec = cfg.bev_diffuser_cfg
    if (spec.prediction_type, spec.noise_timesteps, spec.denoise_timesteps,
            spec.num_inference_steps, spec.cfg_scale, spec.use_classifier_guidence) != (
            'sample', 100, 100, 5, 2.0, False):
        raise ValueError('Run sampler differs from intended t100/5step/CFG2 diagnostic')
    noise_timestep = spec.noise_timesteps if args.noise_timestep is None else args.noise_timestep
    denoise_timestep = spec.denoise_timesteps if args.denoise_timestep is None else args.denoise_timestep
    cfg_scale = spec.cfg_scale if args.cfg_scale is None else args.cfg_scale
    if not np.isfinite(cfg_scale) or cfg_scale < 0:
        raise ValueError('CFG scale must be finite and nonnegative')
    if args.include_t300_cfg4 and (args.conditional_only or args.cfg_scale is not None or
            args.noise_timestep is not None or args.denoise_timestep is not None):
        raise ValueError('Combined t100/t300 comparison requires original t100 CFG0/1/2 settings')
    if args.include_t100_cfg4 and (args.conditional_only or args.cfg_scale is not None or
            args.noise_timestep is not None or args.denoise_timestep is not None):
        raise ValueError('t100 CFG4 extension requires original t100 CFG0/1/2 settings')
    scheduler_config = json.loads((args.scheduler / 'scheduler/scheduler_config.json').read_text())
    train_steps = scheduler_config['num_train_timesteps']
    if not (1 <= noise_timestep <= denoise_timestep <= train_steps and
            denoise_timestep >= spec.num_inference_steps):
        raise ValueError('Noise/denoise timesteps must fit scheduler training range and inference steps')
    primary_arms = ({f'conditional_cfg{cfg_scale:g}': cfg_scale} if args.conditional_only
            else {**ARMS, **({'guided_override': cfg_scale} if args.cfg_scale is not None else {})})
    if args.include_t100_cfg4:
        primary_arms['conditional_t100_cfg4'] = 4.0
    arms = {**primary_arms, **({'conditional_t300_cfg4': 4.0} if args.include_t300_cfg4 else {})}
    checkpoint = Path(spec.unet_checkpoint_dir)
    if not checkpoint.is_absolute():
        checkpoint = (BEV / checkpoint).resolve()
    weight = Path(resolve_unet_checkpoint(checkpoint))
    cache = Path(cfg.semantic_bev_cache_root)
    if not cache.is_absolute():
        cache = (BEV / cache).resolve()
    manifest = cache / 'manifest.json'
    contract = json.loads(manifest.read_text())
    if contract['contract']['projection'].get('output_frame') != 'ego':
        raise ValueError('Expected ego-frame semantic prior')
    scheduler_file = args.scheduler / 'scheduler/scheduler_config.json'
    if not scheduler_file.is_file():
        raise FileNotFoundError(scheduler_file)
    if args.include_t300_cfg4 and 300 > train_steps:
        raise ValueError('Scheduler cannot support t300')
    selection = args.selection_csv.resolve() if args.selection_csv else SELECTED
    rows = selection_rows(args.selection_csv, args.limit, args.shard_index, args.num_shards)
    if not rows:
        raise ValueError('Empty worker shard')
    for row in rows:
        path = cache / f"{row['token']}.npy"
        if not path.is_file() or row['token'] not in contract['entries']:
            raise FileNotFoundError(path)
        probability = np.load(path, mmap_mode='r', allow_pickle=False)
        if probability.shape != (17, 50, 50) or probability.dtype != np.float32 or not np.isfinite(probability).all():
            raise ValueError(f'Invalid semantic probability cache: {path}')
        if not Path(row['gt']).is_file():
            raise FileNotFoundError(row['gt'])
    if args.preflight:
        print(json.dumps({'status': 'preflight_ok', 'frames': len(rows),
                          'teacher_weight': str(weight), 'cache': str(cache),
                          'shard_index': args.shard_index, 'num_shards': args.num_shards,
                          'sampler': {'noise_timesteps': noise_timestep,
                                      'denoise_timesteps': denoise_timestep,
                                      'num_inference_steps': spec.num_inference_steps,
                                      'cfg_scale': cfg_scale}, 'arms': arms}))
        return
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError('Expose exactly one GPU with CUDA_VISIBLE_DEVICES')
    output = unique_output(args.output_dir.resolve())
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device('cuda:0')
    source_val = cfg.data.val
    dataset_cfg = dict(type='NuSceneOcc', **{key: copy.deepcopy(source_val[key]) for key in
        ('data_root', 'ann_file', 'classes', 'modality', 'bev_size', 'box_type_3d', 'test_mode')})
    pipeline = copy.deepcopy(cfg.test_pipeline)
    dataset_cfg['pipeline'] = pipeline[:-1] + pipeline[-1]['transforms']
    dataset_cfg['queue_length'] = 4
    dataset = build_dataset(dataset_cfg)
    index_by_token = {info['token']: index for index, info in enumerate(dataset.data_infos)}
    if len(index_by_token) != len(dataset.data_infos):
        raise ValueError('Duplicate validation tokens')
    baseline = build_model(copy.deepcopy(stage1_cfg.model))
    load_checkpoint(baseline, str(BASE_WEIGHTS), map_location='cpu', strict=True)
    baseline.to(device).eval().requires_grad_(False)
    teacher = BEVDiffuser(spec.unet_cfg, str(weight), str(args.scheduler),
                          prediction_type=spec.prediction_type,
                          noise_timesteps=noise_timestep,
                          denoise_timesteps=denoise_timestep,
                          num_inference_steps=spec.num_inference_steps,
                          cfg_scale=cfg_scale,
                          use_classifier_guidence=spec.use_classifier_guidence)
    teacher.to(device).eval().requires_grad_(False)
    if any(p.requires_grad for p in baseline.parameters()) or any(p.requires_grad for p in teacher.parameters()):
        raise AssertionError('Model or head is not frozen')
    inputs = [run_path, BASE_CFG, BASE_WEIGHTS, STAGE1_CONFIG, weight, manifest, selection,
              scheduler_file, Path(__file__), BEV / 'tools/bevdiffuser_onlyseg.py',
              BEV / 'projects/bevdiffuser/occ_diffusion_utils.py']
    inputs += [cache / f"{row['token']}.npy" for row in rows]
    inputs += [Path(row['gt']) for row in rows]
    provenance = {'command': ' '.join(sys.argv), 'files_sha256': {str(p): sha256(p) for p in inputs},
                  'selected_tokens': [row['token'] for row in rows], 'cache_frame': 'ego',
                  'seed_rule': 'SHA256(token) first 32 bits', 'queue_length': 4,
                  'torch_version': torch.__version__, 'gpu': torch.cuda.get_device_name(0),
                  'subset_not_full_validation': True, 'arms': arms,
                  'selection_csv': str(selection), 'selection_sha256': sha256(selection),
                  'selection_limit': args.limit, 'shard_index': args.shard_index,
                  'num_shards': args.num_shards, 'include_t300_cfg4': args.include_t300_cfg4,
                  'include_t100_cfg4': args.include_t100_cfg4,
                  'sampler_config': {key: spec[key] for key in ('prediction_type', 'noise_timesteps',
                        'denoise_timesteps', 'num_inference_steps', 'cfg_scale', 'use_classifier_guidence')},
                  'effective_sampler': {'prediction_type': spec.prediction_type,
                        'noise_timesteps': noise_timestep, 'denoise_timesteps': denoise_timestep,
                        'num_inference_steps': spec.num_inference_steps, 'cfg_scale': cfg_scale,
                        'use_classifier_guidence': spec.use_classifier_guidence},
                  'ddim_eta': 0.0}
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    frame_rows = []
    totals = {arm: np.zeros((18, 18), dtype=np.int64) for arm in ('clean', *arms)}
    ce_totals = {arm: {'all': [0.0, 0], 'nonfree': [0.0, 0]} for arm in totals}
    actual_grid = None
    actual_grid_300 = None
    seen_scenes = set()
    with (output / 'per_frame.jsonl').open('x') as progress:
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
                    raise ValueError('Validation pipeline returned None')
                queue.append(example)
            merged = dataset.union2one(queue)
            metas = merged['img_metas'].data
            if metas[3]['sample_idx'] != token or metas[3]['scene_token'] != row['scene_token']:
                raise AssertionError('Temporal queue current frame mismatch')
            images = merged['img'].data.unsqueeze(0).to(device, dtype=torch.float32)
            if images.shape[:3] != (1, 4, 6):
                raise ValueError(f'Unexpected image queue {tuple(images.shape)}')
            target_file = np.load(row['gt'])
            target = target_file['semantics']
            visible = target_file['mask_camera'].astype(bool)
            if target.shape != (200, 200, 16) or visible.shape != target.shape:
                raise ValueError('Unexpected occupancy GT shape')
            semantic = torch.from_numpy(np.load(cache / f'{token}.npy', allow_pickle=False).copy()).unsqueeze(0).to(device)
            with torch.no_grad(), preserve_occ_state(baseline, reset_temporal=True):
                clean = bev_map(baseline.obtain_history_bev(images, [copy.deepcopy(metas)])).float()
                baseline.eval()
                if any(module.training for module in baseline.modules()):
                    raise AssertionError('Frozen occupancy head is in train mode')
                denoised, noise_hash = paired_teacher_forward(teacher, clean, metas[3], semantic, token, primary_arms)
                grid = [int(item) for item in teacher.noise_scheduler.timesteps]
                if len(grid) != spec.num_inference_steps or grid != sorted(grid, reverse=True):
                    raise ValueError(f'Unexpected DDIM grid {grid}')
                if actual_grid is None:
                    actual_grid = grid
                elif actual_grid != grid:
                    raise AssertionError('DDIM grid changed between frames')
                hash_300 = None
                if args.include_t300_cfg4:
                    original_noise_timestep = teacher.noise_timesteps
                    original_denoise_timestep = teacher.denoise_timesteps
                    try:
                        teacher.noise_timesteps = 300
                        teacher.denoise_timesteps = 300
                        extra, hash_300 = paired_teacher_forward(
                            teacher, clean, metas[3], semantic, token,
                            {'conditional_t300_cfg4': 4.0})
                        grid_300 = [int(item) for item in teacher.noise_scheduler.timesteps]
                    finally:
                        teacher.noise_timesteps = original_noise_timestep
                        teacher.denoise_timesteps = original_denoise_timestep
                    if hash_300['noise'] != noise_hash['noise'] or hash_300['x_t'] == noise_hash['x_t']:
                        raise AssertionError('t100/t300 must share epsilon but differ in x_t')
                    if len(grid_300) != spec.num_inference_steps or grid_300 != sorted(grid_300, reverse=True):
                        raise ValueError(f'Unexpected t300 DDIM grid {grid_300}')
                    if actual_grid_300 is None:
                        actual_grid_300 = grid_300
                    elif actual_grid_300 != grid_300:
                        raise AssertionError('t300 DDIM grid changed between frames')
                    denoised.update(extra)
                features = {'clean': clean, **denoised}
                frame = {'token': token, 'scene_token': row['scene_token'],
                         'selection_index': row['_selection_index'],
                         'queue_tokens': [dataset.data_infos[item]['token'] for item in indices],
                         'queue_padding': padded, 'seed': stable_seed(token),
                         'noise_sha256': noise_hash['noise'], 'x_t_sha256': noise_hash['x_t'],
                         'arms': {}}
                if hash_300 is not None:
                    frame['noise_t300_sha256'] = hash_300['noise']
                    frame['x_t_t300_sha256'] = hash_300['x_t']
                for arm, feature in features.items():
                    decoded = decode_occ_bev(baseline, feature)
                    logits = decoded['occ']
                    ce = visible_cross_entropy(logits, target, visible)
                    prediction = baseline.pts_bbox_head.get_occ(decoded, [metas[3]])[0].cpu().numpy().astype(np.uint8)
                    hist = confusion(prediction, target, visible)
                    totals[arm] += hist
                    for region in ('all', 'nonfree'):
                        ce_totals[arm][region][0] += ce[region][0]
                        ce_totals[arm][region][1] += ce[region][1]
                    stats = feature_stats(feature, clean)
                    frame['arms'][arm] = {'mse_clean': stats['mse'], 'cosine_clean': stats['cosine'],
                                          'cka50_clean': linear_cka(feature, clean),
                                          'confusion': hist.tolist(),
                                          'ce_all_sum': ce['all'][0], 'ce_all_count': ce['all'][1],
                                          'ce_nonfree_sum': ce['nonfree'][0], 'ce_nonfree_count': ce['nonfree'][1],
                                          'visible_ce_all': ce['all'][0] / ce['all'][1] if ce['all'][1] else None,
                                          'visible_ce_nonfree': ce['nonfree'][0] / ce['nonfree'][1] if ce['nonfree'][1] else None,
                                          'class_counts': [{k: value for k, value in item.items()
                                                            if k in ('class_id', 'tp', 'fp', 'fn', 'iou')}
                                                           for item in class_rows(hist, arm)]}
                if 'conditional' in denoised and 'unconditional' in denoised:
                    change = feature_stats(denoised['conditional'], denoised['unconditional'])
                    frame['conditional_minus_unconditional'] = {
                        'mse': change['mse'], 'cosine': change['cosine']}
            progress.write(json.dumps(frame, allow_nan=False) + '\n')
            progress.flush()
            frame_rows.append(frame)
            seen_scenes.add(row['scene_token'])
            print(f'{rank + 1}/{len(rows)} {token} complete; scenes={len(seen_scenes)}', flush=True)
    arm_summary, pooled = summarize(totals, ce_totals)
    with (output / 'pooled_class.csv').open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=pooled[0].keys())
        writer.writeheader()
        writer.writerows(pooled)
    summary = {'frames': len(rows), 'scenes': len(seen_scenes), 'arms': arm_summary,
               'actual_ddim_grid': actual_grid, 'actual_ddim_grid_t300': actual_grid_300,
               'elapsed_seconds': time.monotonic() - start,
               'interpretation': 'Fixed-head compatibility on selected frames; not Stage 2 transfer.'}
    (output / 'summary.json').write_text(json.dumps(summary, indent=2, allow_nan=False))
    provenance['actual_ddim_grid'] = actual_grid
    provenance['actual_ddim_grid_t300'] = actual_grid_300
    (output / 'provenance.json').write_text(json.dumps(provenance, indent=2))
    print(json.dumps({'output': str(output), 'summary': summary}, indent=2))


if __name__ == '__main__':
    main()
