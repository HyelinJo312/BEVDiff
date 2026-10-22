"""Run native BEVFormer checkpoint inference and format nuScenes predictions.

Launch from BEVFormer with torchrun. Evaluation metrics are computed separately.
"""

import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from mmcv import Config
from mmcv.parallel import MMDistributedDataParallel
from mmcv.runner import get_dist_info, init_dist, load_checkpoint, wrap_fp16_model
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet.utils import setup_multi_processes

from projects.bevdiffuser.eval_by_condition import load_plugin
from projects.mmdet3d_plugin.bevformer.apis.test import custom_multi_gpu_test
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config', help='Native BEVFormer config reconstructed from checkpoint')
    parser.add_argument('checkpoint', help='BEVFormer checkpoint')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--local-rank', '--local_rank', type=int, default=0)
    args = parser.parse_args()
    os.environ.setdefault('LOCAL_RANK', str(args.local_rank))

    cfg = Config.fromfile(args.config)
    load_plugin(cfg)
    cfg.model.pretrained = None
    cfg.model.train_cfg = None
    cfg.data.test.test_mode = True
    cfg.data.test.pop('samples_per_gpu', None)
    setup_multi_processes(cfg)
    init_dist('pytorch', **cfg.dist_params)
    rank, world_size = get_dist_info()
    cfg.gpu_ids = range(world_size)

    dataset = build_dataset(cfg.data.test)
    data_loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2,
        dist=True, shuffle=False,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler,
    )

    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    if cfg.get('fp16') is not None:
        wrap_fp16_model(model)
    checkpoint = load_checkpoint(model, args.checkpoint, map_location='cpu', strict=True)
    model.CLASSES = checkpoint.get('meta', {}).get('CLASSES', dataset.CLASSES)
    model = MMDistributedDataParallel(
        model.cuda(), device_ids=[int(os.environ['LOCAL_RANK'])],
        broadcast_buffers=False,
    )
    if not hasattr(model, '_use_replicated_tensor_module'):
        model._use_replicated_tensor_module = False

    outputs = custom_multi_gpu_test(model, data_loader)
    if rank == 0:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        cfg.dump(str(args.output_dir / 'config.py'))
        result_files, _ = dataset.format_results(
            outputs, jsonfile_prefix=str(args.output_dir / 'predictions'))
        print(f'Formatted nuScenes predictions: {result_files}', flush=True)


if __name__ == '__main__':
    main()
