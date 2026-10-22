# SPDX-License-Identifier: AGPL-3.0
"""Compare ViDAR BEV features with detection-pretrained BEVFormer-base features.

Panel 1: ViDAR BEV. ``pretrain-ViDAR-RN101-nus-full-1future.pth``, pretrained by
         visual point cloud forecasting (CVPR 2024). Its detection head is never
         trained; only the BEV encoder carries the pretraining signal.
Panel 2: BEVFormer-base BEV. ``bevformer_r101_dcn_24ep.pth``, trained with the
         3D detection head.
Panel 3: Measured LiDAR top view, nuScenes devkit style, with GT 3D boxes.

Both models are ResNet101-DCN with a 200x200 x 256 BEV grid and both read the
same frames through the same ``bevformer_base.py`` test pipeline, so the only
difference between the two feature panels is the pretraining objective.

The two repositories register colliding mmdet3d module names (``PerceptionTrans-
former``, ``CustomBEVFormerEncoder``, ...), so ViDAR runs in a separate process
inside its own tree and caches its BEV as npz; this process renders both.

Example:
    cd BEVFormer/projects/bevdiffuser
    python visualize_vidar_vs_bevformer.py --num-frames 6
"""

import argparse
import contextlib
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

SCRIPT_DIR = Path(__file__).resolve().parent
BEVFORMER = SCRIPT_DIR.parents[1]
REPO = BEVFORMER.parent


# ----------------------------------------------------------------------------
# Shared model-side helpers. Imported by both processes, so nothing here may
# touch a specific repository's plugin at module level.
# ----------------------------------------------------------------------------

def build_frame_dataset(dataset_config, data_root, split='val'):
    """Build the shared evaluation dataset, with absolute data paths.

    Both processes read the same config so the two models see byte-identical
    images; only the registry backing the pipeline classes differs.
    """
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    cfg = Config.fromfile(str(dataset_config))
    settings = copy.deepcopy(cfg.data[split])
    settings.pop('samples_per_gpu', None)
    settings['data_root'] = str(data_root) + os.sep
    settings['ann_file'] = str(Path(data_root) / f'nuscenes_infos_temporal_{split}.pkl')
    return cfg, build_dataset(settings)


def collate_frame(dataset, index, device):
    """Return one frame's images and metadata on ``device``."""
    from mmcv.parallel import collate

    batch = collate([dataset[index]], samples_per_gpu=1)

    def unwrap(key):
        value = batch[key]
        if isinstance(value, (list, tuple)):
            value = value[0]
        return value.data[0]

    images = unwrap('img')
    if isinstance(images, (list, tuple)):
        images = torch.stack(images)
    metas = unwrap('img_metas')
    if isinstance(metas[0], (list, tuple)):
        metas = metas[0]
    return images.to(device), metas


def as_scene_start(metas):
    """Apply the no-previous-BEV convention both encoders were trained with.

    ``BEVFormer.forward_test`` zeroes the ego translation and yaw delta in
    can_bus whenever ``prev_bev`` is None. Frames here are sampled far apart and
    run single-frame, so that branch is the correct one for both models; leaving
    absolute global translation in can_bus would feed the can_bus MLP values it
    never sees at inference.
    """
    for meta in metas:
        meta['can_bus'][:3] = 0
        meta['can_bus'][-1] = 0
    return metas


@torch.no_grad()
def encode_bev(model, images, metas, bev_h, bev_w, dim):
    """Run backbone + BEV encoder only, and return ``[C, H, W]``.

    Both heads expose ``only_bev=True`` and return ``[bs, bev_h * bev_w, dim]``;
    ViDAR's head asserts on it, since its detection branch is unused.
    """
    features = model.extract_feat(img=images, img_metas=metas)
    bev = model.pts_bbox_head(features, metas, None, only_bev=True)
    if bev.numel() != bev_h * bev_w * dim:
        raise ValueError(f'Unexpected BEV output with {bev.numel()} elements')
    bev = bev.reshape(-1, bev_h, bev_w, dim).permute(0, 3, 1, 2).contiguous()
    return bev[0].float().cpu().numpy()


# Optional backends ViDAR's plugin imports at module scope that this
# environment does not provide. None is reachable from the ResNet101 BEV encoder
# path: dvxlr and chamferdist belong to the point cloud forecasting loss, DCNv3
# to the InternImage backbone this config does not use.
VIDAR_UNUSED_BACKENDS = ('chamferdist', 'DCNv3')


@contextlib.contextmanager
def vidar_import_shims():
    """Make ViDAR's plugin importable in this environment, for inference only.

    Three things block the import, none of which touches the BEV encoder:

    1. The plugin re-registers ``ExpMomentumEMAHook`` and friends that mmdet
       2.28 already ships. These are training hooks, so overwriting is safe.
    2. ``e2e_predictor_utils`` JIT-builds the ``dvxlr`` CUDA extension at import
       time, and its kernels pass ``Tensor.type()`` to ``AT_DISPATCH_*``, which
       no longer compiles under torch 2.x. ``dvxlr`` is referenced only by the
       differentiable voxel rendering autograd Functions.
    3. ``chamferdist`` and ``DCNv3`` are not installed; see
       ``VIDAR_UNUSED_BACKENDS``.

    Each missing piece becomes a stub that is constructible, so module-level
    instantiation still works, but fatal on use. A silently wrong BEV feature is
    therefore impossible: any real dependence would raise instead.

    All shims are scoped to the plugin import so genuine problems still surface.
    """
    import types

    from mmcv.utils.registry import Registry
    from torch.utils import cpp_extension

    original_register, original_load = Registry._register_module, cpp_extension.load

    def permissive(self, module, module_name=None, force=False):
        return original_register(self, module, module_name=module_name, force=True)

    def make_stub(origin, attribute):
        message = (f'{origin}.{attribute} was stubbed out for BEV feature extraction '
                   'but is actually being used. This code path needs the real '
                   f'{origin}; install or rebuild it for this environment.')

        class Stub:
            def __init__(self, *_, **__):
                pass

            def __call__(self, *_, **__):
                raise RuntimeError(message)

            def __getattr__(self, name):
                # Let introspection (inspect, copy, pickle) probe dunders normally.
                if name.startswith('__') and name.endswith('__'):
                    raise AttributeError(name)
                raise RuntimeError(message)

            def __iter__(self):
                raise RuntimeError(message)

        return Stub

    class StubModule(types.ModuleType):
        def __getattr__(self, name):
            if name.startswith('__') and name.endswith('__'):
                raise AttributeError(name)
            return make_stub(self.__name__, name)

    def stub_load(name, *_, **__):
        return StubModule(name)

    restore = [name for name in VIDAR_UNUSED_BACKENDS if name not in sys.modules]
    for name in restore:
        sys.modules[name] = StubModule(name)
    Registry._register_module, cpp_extension.load = permissive, stub_load
    try:
        yield
    finally:
        Registry._register_module, cpp_extension.load = original_register, original_load
        for name in restore:
            sys.modules.pop(name, None)


def load_detector(config_path, checkpoint, device, plugin_module, allow_duplicates=False):
    """Build one detector from its own config and load its weights."""
    import importlib
    from mmcv import Config
    from mmcv.runner import load_checkpoint, wrap_fp16_model
    from mmdet3d.models import build_model

    if allow_duplicates:
        with vidar_import_shims():
            importlib.import_module(plugin_module)
    else:
        importlib.import_module(plugin_module)
    cfg = Config.fromfile(str(config_path))
    cfg.model.pretrained = None
    model = build_model(cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    load_checkpoint(model, str(checkpoint), map_location='cpu')
    return model.to(device).eval(), cfg


def feature_statistics(bev):
    """Summarize how much cell-specific signal a BEV feature map carries.

    ``mean_abs_cosine`` near 1 and ``mean_direction_alignment`` near 1 mean the
    map is close to rank one: every cell points the same way, so a diffusion
    target built from it carries little per-cell information.
    """
    vectors = bev.reshape(bev.shape[0], -1).T.astype(np.float64)
    unit = vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-9)
    upper = np.triu_indices(len(unit), 1)
    cosine = np.abs(unit @ unit.T)[upper]
    eigenvalues = np.clip(np.linalg.eigvalsh(np.cov((vectors - vectors.mean(0)).T)), 0, None)
    mean_direction = vectors.mean(0)
    mean_direction /= np.maximum(np.linalg.norm(mean_direction), 1e-9)
    return dict(
        mean_abs_cosine=float(cosine.mean()),
        effective_dim=float(eigenvalues.sum() ** 2 / max((eigenvalues ** 2).sum(), 1e-30)),
        channels=int(bev.shape[0]),
        mean_direction_alignment=float(np.abs(unit @ mean_direction).mean()),
        feature_norm_mean=float(np.linalg.norm(bev, axis=0).mean()))


# ----------------------------------------------------------------------------
# Stage A: ViDAR extraction, executed inside the ViDAR repository
# ----------------------------------------------------------------------------

def extract_vidar(args):
    """Cache ViDAR BEV features for the requested sample tokens."""
    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    # ViDAR's own plugin, resolved through the cwd this process was launched in.
    model, cfg = load_detector(args.vidar_config, args.vidar_checkpoint,
                               device, 'projects.mmdet3d_plugin', allow_duplicates=True)
    _, dataset = build_frame_dataset(args.dataset_config, args.data_root)
    index_of = {info['token']: i for i, info in enumerate(dataset.data_infos)}

    cache = Path(args.cache_dir)
    cache.mkdir(parents=True, exist_ok=True)
    for token in args.tokens:
        if token not in index_of:
            raise KeyError(f'Sample {token} is not in the ViDAR-side dataset')
        images, metas = collate_frame(dataset, index_of[token], device)
        if metas[0]['sample_idx'] != token:
            raise ValueError('The dataset returned a different frame than requested')
        bev = encode_bev(model, images, as_scene_start(metas),
                         int(cfg.bev_h_), int(cfg.bev_w_), int(cfg._dim_))
        np.save(cache / f'{token}.npy', bev)
        print(f'ViDAR BEV cached: {token} {bev.shape}', flush=True)
    (cache / 'vidar.json').write_text(json.dumps(dict(
        config=str(args.vidar_config), checkpoint=str(args.vidar_checkpoint),
        bev_grid=[int(cfg.bev_h_), int(cfg.bev_w_)], channels=int(cfg._dim_),
        tokens=list(args.tokens)), indent=2))


# ----------------------------------------------------------------------------
# Stage B: BEVFormer extraction and rendering
# ----------------------------------------------------------------------------

def render(samples, extent, output, point_size, dpi=150):
    """Three panels per frame plus one overview, in the house style."""
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from BEVFormer.projects.bevdiffuser.analyze.visualize_bev_feature_compare import colorize, fit_pca, plot_bev, plot_lidar

    masks = [np.ones(sample['vidar'].shape[1:], dtype=bool) for sample in samples]
    pcas = dict(vidar=fit_pca([s['vidar'] for s in samples], masks),
                bevformer=fit_pca([s['bevformer'] for s in samples], masks))
    for name, pca in pcas.items():
        np.savez(output / f'pca_{name}.npz', **pca)

    for sample in samples:
        record = sample['record']
        fig, axes = plt.subplots(1, 3, figsize=(19, 6.4), constrained_layout=True)
        plot_bev(axes[0], colorize(sample['vidar'], masks[0], pcas['vidar']), extent,
                 'ViDAR BEV | point cloud forecasting pretraining\n'
                 f"256 ch | eff.dim {record['vidar_stats']['effective_dim']:.1f}")
        plot_bev(axes[1], colorize(sample['bevformer'], masks[0], pcas['bevformer']), extent,
                 'BEVFormer-base BEV | 3D detection supervision\n'
                 f"256 ch | eff.dim {record['bevformer_stats']['effective_dim']:.1f}")
        plot_lidar(axes[2], sample['cloud'], extent,
                   f"LiDAR top view + GT ({len(sample['boxes'])} boxes)",
                   boxes=sample['boxes'], point_size=point_size)
        fig.suptitle(f"ViDAR vs BEVFormer-base BEV | val index {record['index']} | {record['token']}\n"
                     'PCA colors are fitted per model over all frames and encode feature '
                     'similarity, not semantic classes', fontsize=14)
        filename = f"frame_{record['index']:05d}_{record['token']}.png"
        fig.savefig(output / filename, dpi=dpi)
        plt.close(fig)
        record['figure'] = filename

    rows = len(samples)
    fig, axes = plt.subplots(rows, 3, figsize=(17, 5.4 * rows), squeeze=False,
                             constrained_layout=True)
    for row, sample in enumerate(samples):
        record = sample['record']
        label = f"val {record['index']} | {record['token'][:10]}"
        plot_bev(axes[row][0], colorize(sample['vidar'], masks[0], pcas['vidar']), extent,
                 f'{label}\nViDAR BEV')
        plot_bev(axes[row][1], colorize(sample['bevformer'], masks[0], pcas['bevformer']), extent,
                 f'{label}\nBEVFormer-base BEV')
        plot_lidar(axes[row][2], sample['cloud'], extent,
                   f"{label}\nLiDAR + GT ({len(sample['boxes'])} boxes)",
                   boxes=sample['boxes'], point_size=point_size)
    fig.suptitle('BEV features by pretraining objective: point cloud forecasting vs 3D detection\n'
                 'Same frames, same images, same 200 x 200 x 256 grid | '
                 'LiDAR color = radial distance (nuScenes style) | box color = GT class',
                 fontsize=16)
    fig.savefig(output / 'overview.png', dpi=dpi)
    plt.close(fig)
    return pcas


def main(args):
    sys.path.insert(0, str(BEVFORMER))
    sys.path.insert(0, str(SCRIPT_DIR))
    os.chdir(BEVFORMER)
    torch.set_num_threads(4)
    torch.manual_seed(0)
    np.random.seed(0)

    from BEVFormer.projects.bevdiffuser.analyze.visualize_bev_feature_compare import load_cloud, load_gt_boxes

    device = torch.device(args.device)
    if device.type == 'cuda':
        torch.cuda.set_device(device)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    data_root = Path(args.data_root).resolve()

    model, cfg = load_detector(args.bevformer_config, args.bevformer_checkpoint,
                               device, 'projects.mmdet3d_plugin')
    model.CLASSES = cfg.class_names
    _, dataset = build_frame_dataset(args.bevformer_config, data_root)
    dataset.CLASSES = cfg.class_names

    indices = args.indices or np.linspace(0, len(dataset) - 1,
                                          min(args.num_frames, len(dataset)), dtype=int).tolist()
    if len(set(indices)) != len(indices) or any(i < 0 or i >= len(dataset) for i in indices):
        raise ValueError('Frame indices must be unique and inside the dataset')
    tokens = [dataset.data_infos[i]['token'] for i in indices]

    # ViDAR first: a failure there should not cost a BEVFormer forward pass.
    cache = Path(args.cache_dir) if args.cache_dir else Path(tempfile.mkdtemp(prefix='vidar_bev_'))
    command = [sys.executable, str(Path(__file__).resolve()), '--extract-vidar',
               '--vidar-config', str(Path(args.vidar_config).resolve()),
               '--vidar-checkpoint', str(Path(args.vidar_checkpoint).resolve()),
               '--dataset-config', str(Path(args.bevformer_config).resolve()),
               '--data-root', str(data_root), '--device', args.device,
               '--cache-dir', str(cache), '--tokens', *tokens]
    vidar_root = Path(args.vidar_root).resolve()
    print(f'Extracting ViDAR BEV with the plugin in {vidar_root} ...', flush=True)
    # cwd stays at the BEVFormer root because nuScenes infos store image paths
    # relative to it; PYTHONPATH makes `projects.mmdet3d_plugin` resolve to
    # ViDAR's plugin instead, which is the only thing this subprocess needs.
    subprocess.run(command, cwd=BEVFORMER, check=True,
                   env=dict(os.environ, PYTHONPATH=str(vidar_root)))

    bounds = np.asarray(cfg.point_cloud_range, dtype=np.float64)
    bev_h, bev_w, dim = int(cfg.bev_h_), int(cfg.bev_w_), int(cfg._dim_)
    samples = []
    if args.save_arrays:
        (output / 'arrays').mkdir(exist_ok=True)
    for index, token in zip(indices, tokens):
        images, metas = collate_frame(dataset, index, device)
        if metas[0]['sample_idx'] != token:
            raise ValueError('The dataset returned a different frame than requested')
        bevformer = encode_bev(model, images, as_scene_start(metas), bev_h, bev_w, dim)
        vidar = np.load(cache / f'{token}.npy')
        if vidar.shape != bevformer.shape:
            raise ValueError(f'ViDAR BEV {vidar.shape} does not match BEVFormer {bevformer.shape}')
        if not np.isfinite(vidar).all() or not np.isfinite(bevformer).all():
            raise ValueError(f'Non-finite BEV feature for frame {index}')

        info = dataset.data_infos[index]
        cloud, lidar_path = load_cloud(info, bounds)
        boxes = load_gt_boxes(dataset, index, bounds)
        record = dict(index=int(index), token=token, scene_token=info.get('scene_token'),
                      shape=list(bevformer.shape), lidar_path=lidar_path,
                      lidar_points_in_range=int(len(cloud)) if cloud is not None else None,
                      gt_box_count=len(boxes), gt_names=[b['name'] for b in boxes],
                      vidar_stats=feature_statistics(vidar),
                      bevformer_stats=feature_statistics(bevformer))
        if args.save_arrays:
            name = f'arrays/frame_{index:05d}_{token}.npz'
            np.savez_compressed(output / name, vidar_bev=vidar, bevformer_bev=bevformer)
            record['arrays'] = name
        samples.append(dict(vidar=vidar, bevformer=bevformer, cloud=cloud,
                            boxes=boxes, record=record))
        print(f"Frame {len(samples)}/{len(indices)}: index={index} "
              f"eff.dim ViDAR {record['vidar_stats']['effective_dim']:.1f} vs "
              f"BEVFormer {record['bevformer_stats']['effective_dim']:.1f}", flush=True)

    extent = bounds[[0, 3, 1, 4]].tolist()
    pcas = render(samples, extent, output, args.point_size)

    def mean_stat(key, field):
        return float(np.mean([s['record'][key][field] for s in samples]))

    summary = dict(
        dataset_config=str(Path(args.bevformer_config).resolve()), data_root=str(data_root),
        frames=len(samples), bev_grid=[bev_h, bev_w], channels=dim,
        can_bus='translation and yaw delta zeroed; single frame, prev_bev=None for both models',
        vidar=dict(config=str(Path(args.vidar_config).resolve()),
                   checkpoint=str(Path(args.vidar_checkpoint).resolve()),
                   pretraining='visual point cloud forecasting (CVPR 2024), detection head untrained',
                   mean_abs_cosine=mean_stat('vidar_stats', 'mean_abs_cosine'),
                   mean_effective_dim=mean_stat('vidar_stats', 'effective_dim'),
                   mean_direction_alignment=mean_stat('vidar_stats', 'mean_direction_alignment'),
                   explained_variance_ratio=pcas['vidar']['explained_variance_ratio'].tolist()),
        bevformer=dict(config=str(Path(args.bevformer_config).resolve()),
                       checkpoint=str(Path(args.bevformer_checkpoint).resolve()),
                       pretraining='3D object detection supervision',
                       mean_abs_cosine=mean_stat('bevformer_stats', 'mean_abs_cosine'),
                       mean_effective_dim=mean_stat('bevformer_stats', 'effective_dim'),
                       mean_direction_alignment=mean_stat('bevformer_stats', 'mean_direction_alignment'),
                       explained_variance_ratio=pcas['bevformer']['explained_variance_ratio'].tolist()),
        lidar_visualization=dict(style='nuscenes devkit render_sample_data',
                                 color='radial distance, viridis', background='white',
                                 gt_outline='per detection class'),
        samples=[s['record'] for s in samples])
    (output / 'statistics.json').write_text(json.dumps(summary, indent=2))
    (output / 'README.md').write_text(
        '# ViDAR vs BEVFormer-base BEV feature 비교\n\n'
        '- `overview.png`: 선택한 모든 프레임. 행마다 ViDAR BEV, BEVFormer-base BEV, LiDAR + GT.\n'
        '- `frame_*.png`: 프레임별 세 패널.\n'
        '- `pca_vidar.npz`, `pca_bevformer.npz`: 모델별 PCA 기저와 색상 정규화 범위.\n'
        '- `statistics.json`: 설정, 프레임 token, feature 구조 지표.\n'
        '- `arrays/`: `--save-arrays`를 준 경우의 원본 200x200x256 feature.\n\n'
        '두 모델 모두 ResNet101-DCN 백본과 200x200x256 BEV 격자를 쓰고, 같은 프레임을 같은 '
        '`bevformer_base.py` test pipeline으로 읽습니다. 따라서 두 feature 패널의 차이는 '
        '사전학습 목적함수 차이입니다. ViDAR는 visual point cloud forecasting으로 학습했고 '
        'detection head는 학습되지 않았습니다. BEVFormer-base는 3D detection supervision으로 '
        '학습했습니다.\n\n'
        '두 모델 모두 prev_bev 없이 단일 프레임으로 돌리고, can_bus의 ego 평행이동과 yaw delta를 '
        '0으로 둡니다. 이는 BEVFormer가 scene 첫 프레임에서 쓰는 규약이며, 멀리 떨어진 프레임을 '
        '추출할 때 올바른 분기입니다.\n\n'
        'PCA는 모델별로 따로 적합했으므로 같은 패널 안에서 프레임 간 비교만 유효합니다. '
        '색상은 의미 클래스가 아니라 상위 3개 주성분입니다.\n\n'
        '`statistics.json`의 `mean_abs_cosine`, `effective_dim`, `mean_direction_alignment`은 '
        'BEV feature가 셀마다 고유한 정보를 얼마나 담는지를 나타냅니다. cosine과 alignment가 1에 '
        '가까우면 모든 셀이 같은 방향을 가리키는 rank-1에 가까운 상태이며, 그런 feature는 '
        'diffusion target으로 쓸 때 셀별 정보가 거의 없습니다.\n', encoding='utf-8')
    print('Saved visualization to', output, flush=True)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--extract-vidar', action='store_true',
                        help=argparse.SUPPRESS)  # internal subprocess entry
    parser.add_argument('--bevformer-config',
                        default=str(BEVFORMER / 'projects/configs/bevformer/bevformer_base.py'))
    parser.add_argument('--bevformer-checkpoint',
                        default=str(BEVFORMER / 'ckpts/bevformer_r101_dcn_24ep.pth'))
    parser.add_argument('--vidar-root', default=str(REPO / 'ViDAR'))
    parser.add_argument('--vidar-config',
                        default=str(REPO / 'ViDAR/projects/configs/vidar_pretrain/nusc_fullset/'
                                           'vidar_full_nusc_1future.py'))
    parser.add_argument('--vidar-checkpoint',
                        default=str(BEVFORMER / 'ckpts/pretrain-ViDAR-RN101-nus-full-1future.pth'))
    parser.add_argument('--dataset-config', help=argparse.SUPPRESS)
    parser.add_argument('--data-root', default=str(BEVFORMER / 'data/nuscenes'))
    parser.add_argument('--tokens', nargs='+', default=[], help=argparse.SUPPRESS)
    parser.add_argument('--cache-dir', help='Where ViDAR BEV npy files are cached')
    parser.add_argument('--num-frames', type=int, default=6)
    parser.add_argument('--indices', nargs='+', type=int, help='Overrides evenly spaced selection')
    parser.add_argument('--output-dir', default=str(REPO / 'results/visualize/bev_vidar_vs_bevformer'))
    parser.add_argument('--device', default='cuda:0')
    parser.add_argument('--point-size', type=float, default=0.6)
    parser.add_argument('--save-arrays', action='store_true')
    args = parser.parse_args()
    if args.num_frames < 1:
        parser.error('--num-frames must be positive')
    return args


if __name__ == '__main__':
    arguments = parse_args()
    if arguments.extract_vidar:
        extract_vidar(arguments)
    else:
        main(arguments)
