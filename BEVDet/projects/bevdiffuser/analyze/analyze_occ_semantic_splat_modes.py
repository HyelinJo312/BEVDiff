"""Paired nearest-versus-bilinear raw ego-frame semantic projection analysis."""

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import sys
import time

import numpy as np


PROJECT_DIR = Path(__file__).resolve().parent
BEVFORMER_DIR = PROJECT_DIR.parents[1]
REPO_DIR = BEVFORMER_DIR.parent
SELECTED_SHA256 = '007dd43afae27aa153e0f58552c23ed019d4d2f70c2f2ff8781a73bb02e243a5'
MODES = ('nearest', 'bilinear')
THRESHOLDS = (.05, .1, .2, .3, .5)
PRIMARY_THRESHOLD = .1
BOOTSTRAP_SEED = 20261015
BOOTSTRAP_DRAWS = 10000
CLASSES = (
    ('car', 1, 4), ('road', 2, 11), ('bus', 3, 3), ('truck', 4, 10),
    ('terrain', 5, 14), ('vegetation', 6, 16), ('sidewalk', 7, 13),
    ('bicycle', 8, 2), ('barrier', 9, 1), ('pedestrian', 10, 7),
    ('manmade', 11, 15), ('motorcycle', 12, 6), ('construction', 13, 5),
    ('trailer', 14, 9), ('cone', 15, 8),
)
CORE = ('cone', 'car', 'barrier', 'pedestrian')
PARTIAL_MAPPINGS = {'road', 'terrain', 'vegetation', 'sidewalk', 'manmade', 'construction'}
EXPECTED_GT = {
    'cone': (92, 10), 'car': (1367, 24), 'barrier': (172, 8), 'pedestrian': (192, 17),
    'road': (15506, 24), 'bus': (108, 6), 'truck': (321, 14), 'terrain': (7266, 20),
    'vegetation': (6842, 22), 'sidewalk': (6741, 23), 'bicycle': (6, 3),
    'manmade': (6106, 24), 'motorcycle': (29, 8), 'construction': (29, 3),
    'trailer': (91, 5),
}


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()


def mapping_flag(name):
    return 'partial/proxy' if name in PARTIAL_MAPPINGS else 'direct_name'


def write_csv(path, rows):
    if not rows:
        raise ValueError(f'No rows for {path}')
    with path.open('x', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def ratios(counts):
    counts = np.asarray(counts)
    tp, fp, fn = np.moveaxis(counts, -1, 0)
    values = {}
    for name, denominator in (('precision', tp + fp), ('recall', tp + fn),
                              ('iou', tp + fp + fn)):
        values[name] = np.divide(tp, denominator, out=np.full(tp.shape, np.nan, dtype=float),
                                 where=denominator != 0)
    return values


def count_cells(score, target, valid, threshold):
    prediction = score >= threshold
    return np.array((np.count_nonzero(prediction & target & valid),
                     np.count_nonzero(prediction & ~target & valid),
                     np.count_nonzero(~prediction & target & valid)), dtype=np.int64)


def paired_bootstrap(frame_counts, core_indices, draws=BOOTSTRAP_DRAWS, seed=BOOTSTRAP_SEED):
    counts = np.asarray(frame_counts, dtype=np.int64)
    if counts.ndim != 4 or counts.shape[1:] != (2, len(CLASSES), 3):
        raise ValueError('Expected [frames, modes, 15 classes, TP/FP/FN]')
    sampled_ids = np.random.default_rng(seed).integers(0, len(counts), (draws, len(counts)))
    sampled = counts[sampled_ids].sum(axis=1)
    sampled_iou = ratios(sampled)['iou']
    observed_iou = ratios(counts.sum(axis=0))['iou']
    rows = []
    for class_index, (name, _, _) in enumerate(CLASSES):
        paired = sampled_iou[:, :, class_index]
        valid = np.isfinite(paired).all(axis=1)
        delta = paired[valid, 1] - paired[valid, 0]
        observed = observed_iou[1, class_index] - observed_iou[0, class_index]
        low, high = np.percentile(delta, (2.5, 97.5)) if len(delta) else (np.nan, np.nan)
        rows.append(dict(endpoint=name, family='primary_core' if name in CORE else 'exploratory_all15',
                         nearest_iou=observed_iou[0, class_index],
                         bilinear_iou=observed_iou[1, class_index], delta_iou=observed,
                         ci_low=low, ci_high=high, valid_draws=len(delta), draws=draws,
                         seed=seed, adjustment='none'))
    paired_core = sampled_iou[:, :, core_indices]
    if paired_core.shape != (draws, 2, len(core_indices)):
        raise ValueError('Unexpected macro shape')
    valid = np.isfinite(paired_core).all(axis=(1, 2))
    core_delta = (paired_core[valid, 1].mean(axis=1) - paired_core[valid, 0].mean(axis=1))
    observed_core = observed_iou[:, core_indices]
    observed_delta = (observed_core[1].mean() - observed_core[0].mean()
                      if np.isfinite(observed_core).all() else np.nan)
    low, high = np.percentile(core_delta, (2.5, 97.5)) if len(core_delta) else (np.nan, np.nan)
    rows.insert(0, dict(endpoint='macro4', family='primary',
                        nearest_iou=observed_core[0].mean() if np.isfinite(observed_core[0]).all() else np.nan,
                        bilinear_iou=observed_core[1].mean() if np.isfinite(observed_core[1]).all() else np.nan,
                        delta_iou=observed_delta, ci_low=low, ci_high=high,
                        valid_draws=len(core_delta), draws=draws, seed=seed, adjustment='none'))
    return rows


def display_number(value, digits=4):
    return 'NA' if not np.isfinite(value) else f'{value:.{digits}f}'


def plot_modes(path, token, valid, targets, probabilities, plt):
    fig, axes = plt.subplots(2, 3, figsize=(12, 8), constrained_layout=True)
    cmap = plt.get_cmap('magma').copy()
    cmap.set_bad('white')
    for row_index, name in enumerate(('cone', 'car')):
        class_index = next(index for index, item in enumerate(CLASSES) if item[0] == name)
        semantic_id = CLASSES[class_index][1]
        arrays = (targets[class_index].astype(float), probabilities['nearest'][semantic_id],
                  probabilities['bilinear'][semantic_id])
        for column, (label, array) in enumerate(zip(('GT', 'Nearest', 'Bilinear'), arrays)):
            axis = axes[row_index, column]
            image = axis.imshow(np.ma.masked_where(~valid, array), origin='lower',
                                extent=(-40, 40, -40, 40), vmin=0, vmax=1, cmap=cmap,
                                interpolation='nearest')
            axis.set_facecolor('white')
            axis.set_title(f'{name} | {label}')
            axis.set_xlabel('ego x (m)')
            axis.set_ylabel('ego y (m)')
    fig.colorbar(image, ax=axes, shrink=.7, label='GT occupancy / semantic score')
    fig.suptitle(f'{token} | camera-visible GT; white invalid, black zero')
    with path.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=140)
    plt.close(fig)


def plot_errors(path, token, valid, targets, probabilities, plt):
    from matplotlib.colors import BoundaryNorm, ListedColormap

    cmap = ListedColormap(('black', '#2ca02c', '#d62728', '#1f77b4'))
    cmap.set_bad('white')
    norm = BoundaryNorm((-.5, .5, 1.5, 2.5, 3.5), cmap.N)
    fig, axes = plt.subplots(2, 2, figsize=(9, 8), constrained_layout=True)
    for row_index, name in enumerate(('cone', 'car')):
        class_index = next(index for index, item in enumerate(CLASSES) if item[0] == name)
        semantic_id = CLASSES[class_index][1]
        target = targets[class_index]
        for column, mode in enumerate(MODES):
            prediction = probabilities[mode][semantic_id] >= PRIMARY_THRESHOLD
            error = np.zeros(valid.shape, dtype=np.uint8)
            error[prediction & target & valid] = 1
            error[prediction & ~target & valid] = 2
            error[~prediction & target & valid] = 3
            axis = axes[row_index, column]
            axis.imshow(np.ma.masked_where(~valid, error), origin='lower',
                        extent=(-40, 40, -40, 40), cmap=cmap, norm=norm,
                        interpolation='nearest')
            axis.set_title(f'{name} | {mode}')
            axis.set_xlabel('ego x (m)')
            axis.set_ylabel('ego y (m)')
    fig.suptitle(f'{token} | threshold 0.1; green TP, red FP, blue FN, black TN, white invalid')
    with path.open('xb') as stream:
        fig.savefig(stream, format='png', dpi=140)
    plt.close(fig)


def plot_summary(output, bootstrap_rows, threshold_rows, plt):
    rows = bootstrap_rows
    fig, axis = plt.subplots(figsize=(9, 7), constrained_layout=True)
    for index, row in enumerate(rows):
        if np.isfinite(row['delta_iou']) and np.isfinite(row['ci_low']):
            color = 'tab:blue' if row['family'] != 'exploratory_all15' else 'tab:gray'
            axis.hlines(index, row['ci_low'], row['ci_high'], color=color)
            axis.plot(row['delta_iou'], index, 'o', color=color)
    axis.axvline(0, color='black', linewidth=.8)
    axis.set_yticks(range(len(rows)), [row['endpoint'] for row in rows])
    axis.invert_yaxis()
    axis.set_xlabel('Bilinear − nearest pooled IoU (paired scene bootstrap 95% CI)')
    with (output / 'paired_delta_ci.png').open('xb') as stream:
        fig.savefig(stream, format='png', dpi=140)
    plt.close(fig)
    fig, axis = plt.subplots(figsize=(7, 4), constrained_layout=True)
    for mode in MODES:
        subset = [row for row in threshold_rows if row['endpoint'] == 'macro4' and row['mode'] == mode]
        axis.plot([row['threshold'] for row in subset], [row['iou'] for row in subset], marker='o', label=mode)
    axis.set_xlabel('Score threshold (descriptive; not calibrated)')
    axis.set_ylabel('Macro4 pooled class IoU')
    axis.legend()
    with (output / 'threshold_sensitivity.png').open('xb') as stream:
        fig.savefig(stream, format='png', dpi=140)
    plt.close(fig)


def report_text(bootstrap_rows, pooled_rows, wins_rows, coverage_rows, runtime, args, selected_path, config_path):
    by_endpoint = {row['endpoint']: row for row in bootstrap_rows}
    lines = [
        '# Occ3D semantic BEV splat mode: paired 24-frame analysis', '',
        '## 설계', '',
        '- 사전 고정된 24개 서로 다른 scene/token에서 현재 production offline dataset으로 원본 SAM3/Metric3D를 한 번씩 읽고, 동일 tensor·metadata를 두 모드에 전달했다.',
        '- 유일한 투영 설정 차이는 `splat_mode=nearest` 대 `bilinear`이다. 두 모드 모두 ego frame, 동일 XYZ 범위와 50×50 격자다. 이전 캐시 및 2D warp는 사용하지 않았다.',
        '- GT는 Occ3D 200×200×16의 camera-visible voxel을 4×4×16 `any`로 모은 뒤 `[Y,X]`로 전치했다. GT 음성 frame의 FP도 pooled metric에 포함했다.',
        '- 사전 지정 primary는 0.1 threshold에서 cone/car/barrier/pedestrian의 각 pooled IoU를 동일 가중 평균한 macro4다. 점수는 calibrated probability가 아니다.',
        '- 24개 scene을 쌍으로 10,000번 재표본추출(seed 20261015), 매 draw마다 TP/FP/FN을 합산해 IoU를 다시 계산했다. 분모 0인 draw는 NA로 제외한다. CI는 percentile 95%다.',
        '', '## Primary 결과', '',
        '| Endpoint | Nearest IoU | Bilinear IoU | Δ | 95% CI | 유효 draw |',
        '| --- | ---: | ---: | ---: | ---: | ---: |',
    ]
    for name in ('macro4',) + CORE:
        row = by_endpoint[name]
        lines.append(f"| {name} | {display_number(row['nearest_iou'])} | {display_number(row['bilinear_iou'])} | {display_number(row['delta_iou'])} | [{display_number(row['ci_low'])}, {display_number(row['ci_high'])}] | {row['valid_draws']}/{row['draws']} |")
    primary = by_endpoint['macro4']
    if primary['ci_low'] > 0:
        conclusion = '이 표본의 primary macro4 IoU는 bilinear가 높고, paired delta의 95% CI는 0보다 크다.'
    elif primary['ci_high'] < 0:
        conclusion = '이 표본의 primary macro4 IoU는 nearest가 높고, bilinear−nearest delta의 95% CI는 0보다 작다.'
    else:
        conclusion = 'Primary delta의 95% CI가 0을 포함하므로 이 표본에서 명확한 우열을 확인하기 어렵다.'
    lines += ['', conclusion, '', '## Precision, recall와 장면별 변화', '',
              '| 클래스 | Nearest P / R | Bilinear P / R | Bilinear 승 / 무 / 패 | GT-positive scenes |',
              '| --- | ---: | ---: | ---: | ---: |']
    for name in CORE:
        metrics = {row['mode']: row for row in pooled_rows if row['class_name'] == name}
        nearest, bilinear = metrics['nearest'], metrics['bilinear']
        wins = next(row for row in wins_rows if row['class_name'] == name)
        lines.append(f"| {name} | {display_number(nearest['precision'])} / {display_number(nearest['recall'])} | {display_number(bilinear['precision'])} / {display_number(bilinear['recall'])} | {wins['wins']} / {wins['ties']} / {wins['losses']} | {wins['positive_scenes']} |")
    lines += ['', '## Supplementary: exploratory 15-class mappings', '',
              '| Mapped class | Nearest IoU | Bilinear IoU | Δ | 95% CI | 유효 draw |',
              '| --- | ---: | ---: | ---: | ---: | ---: |']
    for name, _, _ in CLASSES:
        row = by_endpoint[name]
        lines.append(f"| {name} | {display_number(row['nearest_iou'])} | {display_number(row['bilinear_iou'])} | {display_number(row['delta_iou'])} | [{display_number(row['ci_low'])}, {display_number(row['ci_high'])}] | {row['valid_draws']}/{row['draws']} |")
    lines += ['',
              '- `pooled_metrics.csv`는 15개 mapped class의 TP/FP/FN, precision, recall, IoU를 제공한다. `direct_name`은 클래스 이름 대응이며, `partial/proxy`는 taxonomy 범위가 다른 대응이다. 모든 지표는 semantic surface 대 occupancy footprint의 진단이며 GT free/others/otherflat은 평가하지 않는다.',
              '- `paired_bootstrap.csv`의 all15 classwise CI는 탐색적·다중검정 미보정이다. `wins.csv`의 승/무/패는 GT-positive scene만 사용하고 GT-negative scene FP는 별도 열로 보고한다.',
              '- 0.05/0.2/0.3/0.5 threshold는 `threshold_sensitivity.csv`의 기술통계다. 혼합된 threshold 효과에서 새 threshold를 선택하거나 downstream 이득을 주장하지 않는다.',
              '- 그림은 각 frame의 GT/nearest/bilinear와 첫 4 frame의 0.1 threshold TP/FP/FN을 보여준다. 흰색은 invalid, 검정은 유효한 0/TN이다.',
              '', '## 한계와 재현', '',
              '- 선택된 24 scene의 투영 정렬 진단일 뿐 전체 데이터, 학습 성능, detection/segmentation transfer를 입증하지 않는다.',
              '- CPU float32 재투영은 과거 CUDA cache와 bitwise 동일성을 가정하지 않는다. 실행 시점의 source/config/input 해시는 `provenance.json`과 `runtime_hashes.csv`에 있다.',
              f'- 실행: `{sys.executable} {Path(__file__).resolve()} --selected-csv {selected_path} --config {config_path} --device cpu --threads {args.threads} --output {args.output.resolve()}`',
              f'- 실행 시간: {runtime:.3f}초 (초기화 포함).', '']
    lines += ['## Coverage', '']
    for mode in MODES:
        subset = [row for row in coverage_rows if row['mode'] == mode]
        observed = sum(row['observed_valid_cells'] for row in subset)
        valid = sum(row['valid_cells'] for row in subset)
        lines.append(f'- {mode}: normalized semantic mass가 비영인 유효 셀 {observed}/{valid}. 비영 coverage 증가는 그 자체로 정확도 개선을 뜻하지 않는다.')
    lines += ['', '0.1은 고정된 primary threshold이고 나머지는 sensitivity다. 예측 점수의 분산과 셀 점유 면적이 달라질 수 있으므로 precision과 recall을 함께 해석해야 한다.', '']
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selected-csv', type=Path, default=REPO_DIR / 'results/analysis/occ_semantic_frame_alignment_20261015/selected_frames.csv')
    parser.add_argument('--config', type=Path, default=BEVFORMER_DIR / 'projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py')
    parser.add_argument('--device', choices=('cpu',), default='cpu')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--output', type=Path, default=REPO_DIR / 'results/analysis/occ_semantic_splat_mode_20261015')
    args = parser.parse_args()
    if args.threads != 2:
        parser.error('This bounded analysis requires --threads 2')
    selected_path, config_path, output = (path.resolve() for path in (args.selected_csv, args.config, args.output))
    args.output = output
    if selected_path != (REPO_DIR / 'results/analysis/occ_semantic_frame_alignment_20261015/selected_frames.csv').resolve():
        parser.error('Only the locked selected_frames.csv path is permitted')
    if config_path != (BEVFORMER_DIR / 'projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py').resolve():
        parser.error('Only the current occupancy config path is permitted')
    if output != (REPO_DIR / 'results/analysis/occ_semantic_splat_mode_20261015').resolve():
        parser.error('Only the approved new analysis output path is permitted')
    if digest(selected_path) != SELECTED_SHA256:
        parser.error('Only the approved selected_frames.csv is permitted')
    with selected_path.open(newline='') as stream:
        selected = list(csv.DictReader(stream))
    tokens = [row['token'] for row in selected]
    scenes = [row['scene_token'] for row in selected]
    if (len(selected) != 24 or len(set(tokens)) != 24 or len(set(scenes)) != 24
            or [int(row['order']) for row in selected] != list(range(1, 25))):
        parser.error('Selection must contain 24 uniquely ordered tokens from distinct scenes')
    if output.exists() and any(output.iterdir()):
        parser.error(f'Output already contains artifacts: {output}')
    if any(not Path(row['gt']).is_file() for row in selected):
        parser.error('A selected GT file is missing')
    source_paths = {
        'runner': Path(__file__), 'config': config_path,
        'projector': PROJECT_DIR / 'layout_diffusion/seg_bev_aligner_one_hot_v5.py',
        'dataset': PROJECT_DIR / 'data_utils.py',
        'offline_dataset': PROJECT_DIR / 'build_semantic_bev_cache.py',
    }
    sources = {name: digest(path) for name, path in source_paths.items()}
    for name in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS'):
        os.environ[name] = str(args.threads)
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import torch
    import mmcv
    from mmcv import Config

    sys.path.insert(0, str(BEVFORMER_DIR))
    from projects.bevdiffuser.build_semantic_bev_cache import offline_dataset
    from projects.bevdiffuser.layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner

    started = time.perf_counter()
    torch.set_num_threads(args.threads)
    os.chdir(PROJECT_DIR)
    cfg = Config.fromfile(str(config_path))
    projection = dict(cfg.unet.parameters.seg_bev_aligner)
    if (projection.get('output_frame') != 'ego' or tuple(projection['pc_range']) != (-40, -40, -1.0, 40, 40, 5.4)
            or projection['bev_h'] != 50 or projection['bev_w'] != 50 or projection['num_classes'] != 16):
        parser.error('Expected current 17-channel, 50x50 ego-frame occupancy config')
    effective = {mode: {**projection, 'splat_mode': mode} for mode in MODES}
    if ({key: value for key, value in effective['nearest'].items() if key != 'splat_mode'}
            != {key: value for key, value in effective['bilinear'].items() if key != 'splat_mode'}):
        raise AssertionError('Effective projection configs differ beyond splat_mode')
    dataset = offline_dataset(cfg, 'val')
    lookup = {info['token']: index for index, info in enumerate(dataset.data_infos)}
    if any(token not in lookup for token in tokens):
        parser.error('Selected token is absent from validation annotations')
    aligners = {mode: SegBEVAligner(**effective[mode]).to('cpu').eval() for mode in MODES}
    output.mkdir(parents=True, exist_ok=True)
    for subdir in ('probabilities/nearest', 'probabilities/bilinear', 'support/nearest',
                   'support/bilinear', 'frames', 'errors'):
        (output / subdir).mkdir(parents=True, exist_ok=True)
    core_indices = [index for index, item in enumerate(CLASSES) if item[0] in CORE]
    frame_counts = np.zeros((len(selected), len(MODES), len(CLASSES), 3), dtype=np.int64)
    threshold_counts = np.zeros((len(THRESHOLDS), len(selected), len(MODES), len(CLASSES), 3), dtype=np.int64)
    frame_rows, coverage_rows, runtime_rows = [], [], []
    for frame_index, selected_row in enumerate(selected):
        frame_start = time.perf_counter()
        token = selected_row['token']
        item = dataset.get_unconditioned_item(lookup[token])
        meta = dataset._current_metas(item['img_metas'])
        if meta['sample_idx'] != token or dataset.data_infos[lookup[token]]['scene_token'] != selected_row['scene_token']:
            raise ValueError(f'Metadata identity mismatch for {token}')
        if any(tuple(shape[:2]) != (928, 1600) for shape in meta['pad_shape']):
            raise ValueError(f'Unexpected image geometry for {token}')
        seg = dataset.load_segmaps(meta['filename'], meta, dataset.semantic_path)
        depth = dataset.load_depth_from_filenames(meta['filename'], meta)
        if seg.shape != depth.shape or tuple(seg.shape[1:]) != (928, 1600):
            raise ValueError(f'Raw-map alignment mismatch for {token}')
        seg_input, depth_input = seg.unsqueeze(0), depth.unsqueeze(0)
        probabilities, supports = {}, {}
        with torch.no_grad():
            for mode in MODES:
                probability, support = aligners[mode].project_semantics(seg_input, [meta], depth_input)
                probabilities[mode] = probability[0].numpy()
                supports[mode] = support[0].numpy()
        for mode in MODES:
            probability, support = probabilities[mode], supports[mode]
            if (probability.shape != (17, 50, 50) or probability.dtype != np.float32
                    or support.shape != (1, 50, 50) or support.dtype != np.float32
                    or not np.isfinite(probability).all() or not np.isfinite(support).all()):
                raise ValueError(f'Invalid {mode} output for {token}')
            with (output / 'probabilities' / mode / f'{token}.npy').open('xb') as stream:
                np.save(stream, probability, allow_pickle=False)
            with (output / 'support' / mode / f'{token}.npy').open('xb') as stream:
                np.save(stream, support, allow_pickle=False)
        with np.load(selected_row['gt'], allow_pickle=False) as gt_file:
            semantics = gt_file['semantics']
            mask = gt_file['mask_camera'].astype(bool)
        if semantics.shape != (200, 200, 16) or mask.shape != semantics.shape:
            raise ValueError(f'Unexpected GT shape for {token}')
        valid = mask.reshape(50, 4, 50, 4, 16).any(axis=(1, 3, 4)).T
        if int(valid.sum()) != int(selected_row['valid_cells']):
            raise ValueError(f'GT footprint mismatch for {token}')
        targets = [((semantics == gt_id) & mask).reshape(50, 4, 50, 4, 16).any(axis=(1, 3, 4)).T
                   for _, _, gt_id in CLASSES]
        plot_modes(output / 'frames' / f'{token}.png', token, valid, targets, probabilities, plt)
        if frame_index < 4:
            plot_errors(output / 'errors' / f'{token}.png', token, valid, targets, probabilities, plt)
        for mode_index, mode in enumerate(MODES):
            probability = probabilities[mode]
            coverage_rows.append(dict(order=frame_index + 1, token=token, scene_token=selected_row['scene_token'],
                                      mode=mode, valid_cells=int(valid.sum()),
                                      observed_valid_cells=int(np.count_nonzero((probability.sum(0) > 1e-8) & valid)),
                                      support_valid_sum=float(supports[mode][0][valid].sum())))
            for class_index, (name, semantic_id, gt_id) in enumerate(CLASSES):
                for threshold_index, threshold in enumerate(THRESHOLDS):
                    threshold_counts[threshold_index, frame_index, mode_index, class_index] = count_cells(
                        probability[semantic_id], targets[class_index], valid, threshold)
                counts = threshold_counts[THRESHOLDS.index(PRIMARY_THRESHOLD), frame_index, mode_index, class_index]
                frame_counts[frame_index, mode_index, class_index] = counts
                score = ratios(counts)
                frame_rows.append(dict(order=frame_index + 1, token=token, scene_token=selected_row['scene_token'],
                                       mode=mode, class_name=name, semantic_id=semantic_id, gt_id=gt_id,
                                       mapping_flag=mapping_flag(name), family='primary_core' if name in CORE else 'exploratory_all15',
                                       threshold=PRIMARY_THRESHOLD, valid_cells=int(valid.sum()),
                                       gt_positive_cells=int(targets[class_index].sum()),
                                       tp=int(counts[0]), fp=int(counts[1]), fn=int(counts[2]),
                                       **{key: float(value) for key, value in score.items()}))
        runtime_rows.append(dict(order=frame_index + 1, token=token,
                                 seconds=round(time.perf_counter() - frame_start, 3),
                                 seg_sha256=array_digest(seg.numpy()), depth_sha256=array_digest(depth.numpy()),
                                 gt_sha256=digest(selected_row['gt']),
                                 nearest_prob_sha256=array_digest(probabilities['nearest']),
                                 bilinear_prob_sha256=array_digest(probabilities['bilinear']),
                                 nearest_support_sha256=array_digest(supports['nearest']),
                                 bilinear_support_sha256=array_digest(supports['bilinear'])))
        print(f'{frame_index + 1}/24 {token} {runtime_rows[-1]["seconds"]}s', flush=True)
    pooled_rows = []
    for mode_index, mode in enumerate(MODES):
        for class_index, (name, semantic_id, gt_id) in enumerate(CLASSES):
            counts = frame_counts[:, mode_index, class_index].sum(axis=0)
            score = ratios(counts)
            pooled_rows.append(dict(mode=mode, class_name=name, semantic_id=semantic_id, gt_id=gt_id,
                                    mapping_flag=mapping_flag(name), family='primary_core' if name in CORE else 'exploratory_all15',
                                    frames=24, threshold=PRIMARY_THRESHOLD,
                                    gt_positive_cells=int(counts[0] + counts[2]),
                                    tp=int(counts[0]), fp=int(counts[1]), fn=int(counts[2]),
                                    **{key: float(value) for key, value in score.items()}))
    bootstrap_rows = paired_bootstrap(frame_counts, core_indices)
    wins_rows = []
    for class_index, (name, _, _) in enumerate(CLASSES):
        positive = (frame_counts[:, 0, class_index, 0] + frame_counts[:, 0, class_index, 2]) > 0
        scene_iou = ratios(frame_counts[:, :, class_index])['iou']
        valid_positive = positive & np.isfinite(scene_iou).all(axis=1)
        deltas = scene_iou[valid_positive, 1] - scene_iou[valid_positive, 0]
        wins_rows.append(dict(class_name=name, positive_scenes=int(positive.sum()),
                              valid_positive_scenes=int(valid_positive.sum()),
                              wins=int(np.count_nonzero(deltas > 0)), ties=int(np.count_nonzero(deltas == 0)),
                              losses=int(np.count_nonzero(deltas < 0)),
                              gt_negative_scenes=int((~positive).sum()),
                              nearest_negative_fp=int(frame_counts[~positive, 0, class_index, 1].sum()),
                              bilinear_negative_fp=int(frame_counts[~positive, 1, class_index, 1].sum())))
        if (int((frame_counts[:, 0, class_index, 0] + frame_counts[:, 0, class_index, 2]).sum()),
                int(positive.sum())) != EXPECTED_GT[name]:
            raise ValueError(f'GT footprint total or positive-scene count mismatch for {name}')
    threshold_rows = []
    for threshold_index, threshold in enumerate(THRESHOLDS):
        for mode_index, mode in enumerate(MODES):
            pooled = threshold_counts[threshold_index, :, mode_index].sum(axis=0)
            pooled_iou = ratios(pooled)['iou']
            for class_index, (name, _, _) in enumerate(CLASSES):
                score = ratios(pooled[class_index])
                threshold_rows.append(dict(threshold=threshold, mode=mode, endpoint=name,
                                           family='primary_core' if name in CORE else 'exploratory_all15',
                                           tp=int(pooled[class_index, 0]), fp=int(pooled[class_index, 1]),
                                           fn=int(pooled[class_index, 2]),
                                           **{key: float(value) for key, value in score.items()}))
            core_iou = pooled_iou[core_indices]
            threshold_rows.append(dict(threshold=threshold, mode=mode, endpoint='macro4', family='primary' if threshold == PRIMARY_THRESHOLD else 'descriptive',
                                       tp='', fp='', fn='', precision=np.nan, recall=np.nan,
                                       iou=float(core_iou.mean()) if np.isfinite(core_iou).all() else np.nan))
    plot_summary(output, bootstrap_rows, threshold_rows, plt)
    for filename, rows in (('perframe_metrics.csv', frame_rows), ('pooled_metrics.csv', pooled_rows),
                           ('paired_bootstrap.csv', bootstrap_rows), ('wins.csv', wins_rows),
                           ('threshold_sensitivity.csv', threshold_rows), ('coverage.csv', coverage_rows),
                           ('runtime_hashes.csv', runtime_rows)):
        write_csv(output / filename, rows)
    if sources != {name: digest(path) for name, path in source_paths.items()}:
        raise RuntimeError('Analysis source or config changed during execution')
    provenance = dict(selected_csv=str(selected_path), selected_sha256=digest(selected_path),
                      selected_tokens=tokens, selected_scenes=scenes, config=str(config_path),
                      source_sha256=sources, effective_projection_configs=effective,
                      class_mapping=[dict(class_name=name, semantic_id=semantic_id, gt_id=gt_id,
                                          mapping_flag=mapping_flag(name), family='primary_core' if name in CORE else 'exploratory_all15')
                                     for name, semantic_id, gt_id in CLASSES],
                      primary=dict(threshold=PRIMARY_THRESHOLD, endpoint='macro4', classes=CORE),
                      thresholds=THRESHOLDS, bootstrap=dict(seed=BOOTSTRAP_SEED, draws=BOOTSTRAP_DRAWS,
                                                            unit='paired_scene', interval='percentile_95',
                                                            undefined='omit_draw'),
                      input_pairing='one dataset item, one seg/depth tensor and metadata per token, fed to both modes',
                      gt='camera-visible 4x4x16 any pooled from [X,Y,Z] to [Y,X]',
                      runtime=dict(seconds=round(time.perf_counter() - started, 3), device='cpu', threads=args.threads,
                                   python=sys.version, torch=torch.__version__, numpy=np.__version__, mmcv=mmcv.__version__))
    with (output / 'provenance.json').open('x') as stream:
        json.dump(provenance, stream, indent=2, allow_nan=False)
        stream.write('\n')
    (output / 'report.md').open('x').write(report_text(bootstrap_rows, pooled_rows, wins_rows,
                                                      coverage_rows, provenance['runtime']['seconds'],
                                                      args, selected_path, config_path))


if __name__ == '__main__':
    main()
