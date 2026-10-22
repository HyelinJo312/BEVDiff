"""Stage 2 Occ3D training with a frozen GT-layout diffusion teacher."""

from __future__ import division

import argparse
import copy
import os
import sys
import time
import warnings
from os import path as osp

import mmcv
import torch
from mmcv import Config, DictAction
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (HOOKS, DistSamplerSeedHook, EpochBasedRunner,
                         Fp16OptimizerHook, OptimizerHook, build_optimizer,
                         build_runner, get_dist_info, init_dist,
                         load_checkpoint)
from mmcv.utils import TORCH_VERSION, build_from_cfg, digit_version
from mmdet import __version__ as mmdet_version
from mmdet.apis import set_random_seed
from mmdet.core import EvalHook
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import setup_multi_processes
from mmdet3d import __version__ as mmdet3d_version
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model
from mmdet3d.utils import collect_env, get_root_logger
from mmseg import __version__ as mmseg_version

sys.path.append(osp.dirname(osp.abspath(__file__)) + '/..')

from projects.mmdet3d_plugin.core.evaluation.eval_hooks import CustomDistEvalHook
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.mmdet3d_plugin.models.hooks.occ_gradient_monitor import OccGradientMonitorHook
from bevdiffuser import BEVDiffuser


def parse_args():
    parser = argparse.ArgumentParser(description='Train BEVFormer Occ3D')
    parser.add_argument('config', help='train config file path')
    parser.add_argument('--work-dir', '--work_dir', dest='work_dir')
    parser.add_argument('--load-from', '--load_from', dest='load_from')
    parser.add_argument('--model-target-checkpoint', '--model_target_checkpoint',
                        dest='model_target_checkpoint')
    parser.add_argument('--resume-from', '--resume_from', dest='resume_from')
    parser.add_argument('--no-validate', action='store_true')
    group_gpus = parser.add_mutually_exclusive_group()
    group_gpus.add_argument('--gpus', type=int)
    group_gpus.add_argument('--gpu-ids', type=int, nargs='+')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--deterministic', action='store_true')
    parser.add_argument('--options', nargs='+', action=DictAction)
    parser.add_argument('--cfg-options', nargs='+', action=DictAction)
    parser.add_argument('--launcher', choices=['none', 'pytorch', 'slurm', 'mpi'],
                        default='none')
    parser.add_argument('--local-rank', '--local_rank', dest='local_rank',
                        type=int, default=0)
    parser.add_argument('--autoscale-lr', action='store_true')
    parser.add_argument('--unet-checkpoint-dir', '--unet_checkpoint_dir',
                        dest='unet_checkpoint_dir')
    parser.add_argument('--report-to', '--report_to', dest='report_to',
                        choices=['tensorboard', 'wandb'], default=None)
    parser.add_argument('--tracker-project-name', '--tracker_project_name',
                        dest='tracker_project_name', default='DiffBEVFormer')
    parser.add_argument('--tracker-run-name', '--tracker_run_name',
                        dest='tracker_run_name', default=None)
    parser.add_argument('--gradient-monitor-enabled', choices=('True', 'False'), default=None)
    parser.add_argument('--gradient-monitor-interval', type=int, default=None)
    args = parser.parse_args()

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    if args.options and args.cfg_options:
        raise ValueError('--options and --cfg-options cannot be both specified.')
    if args.options:
        warnings.warn('--options is deprecated; use --cfg-options instead.')
        args.cfg_options = args.options
    return args


def apply_gradient_monitor_options(cfg, args):
    if args.gradient_monitor_interval is not None and args.gradient_monitor_interval <= 0:
        raise ValueError('gradient_monitor.interval must be a positive integer')
    overrides = {}
    if args.gradient_monitor_enabled is not None:
        overrides['gradient_monitor.enabled'] = args.gradient_monitor_enabled == 'True'
    if args.gradient_monitor_interval is not None:
        overrides['gradient_monitor.interval'] = args.gradient_monitor_interval
    if overrides:
        cfg.merge_from_dict(overrides)


def import_custom_modules(cfg, config_path):
    if cfg.get('custom_imports', None):
        from mmcv.utils import import_modules_from_strings
        import_modules_from_strings(**cfg.custom_imports)
    if not cfg.get('plugin', False):
        return

    import importlib

    module_dir = cfg.get('plugin_dir', osp.dirname(config_path))
    module_parts = osp.dirname(module_dir).split('/')
    module_path = module_parts[0]
    for part in module_parts[1:]:
        module_path += f'.{part}'
    print(module_path)
    importlib.import_module(module_path)


def get_logger_name(cfg):
    return 'mmseg' if cfg.model.type == 'EncoderDecoder3D' else 'mmdet'


def add_logger_hook(cfg, hook_type, init_kwargs=None):
    hooks = cfg.log_config.setdefault('hooks', [])
    if any(hook.get('type') == hook_type for hook in hooks):
        return
    hook_cfg = dict(type=hook_type)
    if init_kwargs is not None:
        hook_cfg['init_kwargs'] = init_kwargs
    hooks.append(hook_cfg)


def configure_runtime(cfg, args):
    if args.tracker_run_name is None:
        args.tracker_run_name = osp.splitext(osp.basename(args.config))[0]
    if args.work_dir is not None:
        cfg.work_dir = osp.join(args.work_dir, args.tracker_run_name)
    elif cfg.get('work_dir', None) is None:
        cfg.work_dir = osp.join(
            './work_dirs', osp.splitext(osp.basename(args.config))[0])

    if (args.unet_checkpoint_dir is not None
            and (osp.isfile(args.unet_checkpoint_dir)
                 or osp.isdir(args.unet_checkpoint_dir))
            and cfg.get('bev_diffuser_cfg') is not None):
        cfg.bev_diffuser_cfg.unet_checkpoint_dir = args.unet_checkpoint_dir
    if args.resume_from is not None and osp.isfile(args.resume_from):
        cfg.resume_from = args.resume_from
    if args.load_from is not None and osp.isfile(args.load_from):
        cfg.load_from = args.load_from
    if args.model_target_checkpoint is not None:
        cfg.model_target_checkpoint = args.model_target_checkpoint
    cfg.gpu_ids = args.gpu_ids if args.gpu_ids is not None else range(
        1 if args.gpus is None else args.gpus)

    if (digit_version(TORCH_VERSION) == digit_version('1.8.1')
            and cfg.optimizer['type'] == 'AdamW'):
        cfg.optimizer['type'] = 'AdamW2'
    if args.autoscale_lr:
        cfg.optimizer['lr'] = cfg.optimizer['lr'] * len(cfg.gpu_ids) / 8
    if args.report_to == 'tensorboard':
        add_logger_hook(cfg, 'TensorboardLoggerHook')
    elif args.report_to == 'wandb':
        add_logger_hook(
            cfg, 'WandbLoggerHook',
            init_kwargs=dict(
                project=args.tracker_project_name,
                name=args.tracker_run_name,
                id=args.tracker_run_name))


def validate_teacher_alignment_config(cfg):
    alignment = cfg.model.get('teacher_bev_alignment')
    if alignment is None:
        return
    from projects.bevdiffuser.bev_frame_alignment import TeacherBEVAlignment
    profile = TeacherBEVAlignment(**alignment)
    if list(profile.source_pc_range) != list(cfg.model.pts_bbox_head.pc_range):
        raise ValueError('Alignment source range must match the occupancy head range')
    target = cfg.get('model_target')
    if target is None or target.type != 'BEVFormer':
        raise ValueError('model_target must be the native detection BEVFormer')
    target_head = target.pts_bbox_head
    if target_head.type != 'BEVFormerHead':
        raise ValueError('model_target must use BEVFormerHead')
    if (target_head.transformer.type != 'PerceptionTransformer' or
            target_head.transformer.encoder.type != 'BEVFormerEncoder'):
        raise ValueError('model_target must use the native detection transformer')
    if list(target_head.transformer.encoder.pc_range) != list(profile.teacher_pc_range):
        raise ValueError('Detection target range must match alignment teacher range')
    if list(target_head.bbox_coder.pc_range) != list(profile.teacher_pc_range):
        raise ValueError('Detection target bbox coder range must match teacher range')
    if (target_head.bev_h, target_head.bev_w) != profile.teacher_bev_size:
        raise ValueError('Detection target BEV size must match alignment teacher size')
    unet_parameters = cfg.bev_diffuser_cfg.unet_cfg.parameters
    student_head = cfg.model.pts_bbox_head
    if (target_head.in_channels != student_head.in_channels or
            target_head.in_channels != target_head.transformer.embed_dims or
            target_head.in_channels != unet_parameters.in_channels or
            target_head.in_channels != unet_parameters.out_channels):
        raise ValueError('Detection target, student and UNet channels must match')
    if (profile.teacher_bev_size[0] != profile.teacher_bev_size[1] or
            unet_parameters.image_size != profile.teacher_bev_size[0]):
        raise ValueError('Teacher BEV size must match the square UNet image_size')
    divisor = 2 ** unet_parameters.num_pre_downsample
    if any(size % divisor for size in profile.teacher_bev_size):
        raise ValueError('Teacher BEV size is incompatible with UNet pre-downsampling')
    for split in ('train', 'val', 'test'):
        dataset = cfg.data[split]
        if list(dataset.pc_range) != list(profile.source_pc_range):
            raise ValueError(f'{split} dataset range differs from alignment source range')
        if list(dataset.layout_pc_range) != list(profile.teacher_pc_range):
            raise ValueError(f'{split} layout range differs from teacher range')
    if list(cfg.layout_dataset_args.layout_pc_range) != list(profile.teacher_pc_range):
        raise ValueError('Layout dataset args differ from teacher range')
    range_filters = [step for step in cfg.data.train.pipeline
                     if step['type'] == 'ObjectRangeFilter']
    if len(range_filters) != 1 or list(range_filters[0].point_cloud_range) != list(
            profile.teacher_pc_range):
        raise ValueError('Training object filter must use the teacher LiDAR range')


def build_train_datasets(cfg):
    datasets = [build_dataset(cfg.data.train)]
    if len(cfg.workflow) == 2:
        val_dataset = copy.deepcopy(cfg.data.val)
        if 'dataset' in cfg.data.train:
            val_dataset.pipeline = cfg.data.train.dataset.pipeline
        else:
            val_dataset.pipeline = cfg.data.train.pipeline
        val_dataset.test_mode = False
        datasets.append(build_dataset(val_dataset))
    return datasets


def build_frozen_teacher(cfg):
    if cfg.get('bev_diffuser_cfg') is None:
        return None, None
    checkpoint = cfg.get('model_target_checkpoint')
    if not checkpoint or not osp.isfile(checkpoint):
        raise FileNotFoundError(f'Model target checkpoint not found: {checkpoint}')
    model_target = build_model(
        cfg.model_target, train_cfg=cfg.get('train_cfg'),
        test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model_target, checkpoint, strict=True, map_location='cpu')
    model_target.requires_grad_(False)
    model_target.eval()
    bev_diffuser = BEVDiffuser(**cfg.bev_diffuser_cfg)
    bev_diffuser.requires_grad_(False)
    bev_diffuser.eval()
    return model_target, bev_diffuser


def train_occ_model(model, datasets, cfg, distributed=False, validate=False,
                    timestamp=None, meta=None, model_target=None,
                    bev_diffuser=None, target_checkpoint=None):
    logger = get_root_logger(log_level=cfg.log_level,
                             name=get_logger_name(cfg))
    logger.propagate = False
    datasets = datasets if isinstance(datasets, (list, tuple)) else [datasets]
    data_loaders = [
        build_dataloader(
            dataset, cfg.data.samples_per_gpu, cfg.data.workers_per_gpu,
            len(cfg.gpu_ids), dist=distributed, seed=cfg.seed,
            shuffler_sampler=cfg.data.shuffler_sampler,
            nonshuffler_sampler=cfg.data.nonshuffler_sampler)
        for dataset in datasets
    ]

    if distributed:
        model = MMDistributedDataParallel(
            model.cuda(), device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=cfg.get('find_unused_parameters', False))
        if not hasattr(model, '_use_replicated_tensor_module'):
            model._use_replicated_tensor_module = False
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)
    if model_target is not None:
        model_target = model_target.cuda()
    if bev_diffuser is not None:
        bev_diffuser = bev_diffuser.cuda()

    optimizer = build_optimizer(model, cfg.optimizer)
    if 'runner' not in cfg:
        cfg.runner = dict(type='EpochBasedRunner', max_epochs=cfg.total_epochs)
        warnings.warn('Please set `runner` in the config.', UserWarning)
    elif 'total_epochs' in cfg:
        assert cfg.total_epochs == cfg.runner.max_epochs

    runner_args = dict(model=model, optimizer=optimizer,
                       work_dir=cfg.work_dir, logger=logger, meta=meta)
    if model_target is not None or bev_diffuser is not None:
        runner_args['model_target'] = model_target
        runner_args['bev_diffuser'] = bev_diffuser
    runner = build_runner(cfg.runner, default_args=runner_args)
    runner.timestamp = timestamp

    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config
    runner.register_training_hooks(
        cfg.lr_config, optimizer_config, cfg.checkpoint_config,
        cfg.log_config, cfg.get('momentum_config', None))
    monitor_cfg = cfg.get('gradient_monitor', {})
    if monitor_cfg.get('enabled', False):
        core = getattr(model, 'module', model)
        diffuser_cfg = cfg.get('bev_diffuser_cfg', {})
        metadata = dict(
            definition='Weighted pre-clipping rank-local original BEV gradients, pooled across ranks; not encoder parameter gradients',
            kind='layout',
            distillation_loss=(
                'valid_cell_mse' if getattr(core, 'teacher_bev_alignment', None)
                is not None and core.teacher_bev_alignment.mask_invalid else 'mse'),
            distill_loss_weight=core.distill_loss_weight,
            task_weight=1.0 - core.loss_mix_weight,
            kd_weight=core.loss_mix_weight,
            use_mgd=False,
            teacher_bev_alignment=cfg.model.get('teacher_bev_alignment'),
            layout_pc_range=cfg.data.train.get('layout_pc_range'),
            effective_task_weight=1.0 - core.loss_mix_weight,
            effective_kd_weight=core.loss_mix_weight * core.distill_loss_weight,
            teacher_checkpoint=diffuser_cfg.get('unet_checkpoint_dir'),
            target_checkpoint=target_checkpoint,
            target_config=cfg.get('model_target'),
            teacher_transport=(
                'output_only' if getattr(core, 'teacher_bev_alignment', None)
                is not None else 'none'),
            student_load_from=cfg.get('load_from'),
            noise_timesteps=diffuser_cfg.get('noise_timesteps'),
            denoise_timesteps=diffuser_cfg.get('denoise_timesteps'),
            num_inference_steps=diffuser_cfg.get('num_inference_steps'),
            cfg_scale=2.0,
            seed=cfg.get('seed'),
            resume_from=cfg.get('resume_from'))
        runner.register_hook(OccGradientMonitorHook(
            interval=monitor_cfg.get('interval', 200),
            work_dir=cfg.work_dir, metadata=metadata), priority='NORMAL')
    if distributed and isinstance(runner, EpochBasedRunner):
        runner.register_hook(DistSamplerSeedHook())
    if validate:
        register_eval_hook(runner, cfg, distributed)
    if cfg.get('custom_hooks', None):
        for hook_cfg in cfg.custom_hooks:
            if hook_cfg['type'] == 'UpdateTarget':
                raise ValueError('UpdateTarget cannot overwrite the frozen detection target')
            hook_cfg = hook_cfg.copy()
            priority = hook_cfg.pop('priority', 'NORMAL')
            runner.register_hook(build_from_cfg(hook_cfg, HOOKS),
                                 priority=priority)

    if cfg.get('resume_from', None):
        runner.resume(cfg.resume_from)
    elif cfg.get('load_from', None):
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def register_eval_hook(runner, cfg, distributed):
    val_samples_per_gpu = cfg.data.val.get('samples_per_gpu', 1)
    val_dataset_cfg = copy.deepcopy(cfg.data.test)
    if val_samples_per_gpu > 1:
        val_dataset_cfg.pipeline = replace_ImageToTensor(
            val_dataset_cfg.pipeline)
    val_dataset = build_dataset(val_dataset_cfg, dict(test_mode=True))
    val_dataloader = build_dataloader(
        val_dataset, samples_per_gpu=val_samples_per_gpu,
        workers_per_gpu=cfg.data.workers_per_gpu, dist=distributed,
        shuffle=False, shuffler_sampler=cfg.data.shuffler_sampler,
        nonshuffler_sampler=cfg.data.nonshuffler_sampler)
    eval_cfg = copy.deepcopy(cfg.get('evaluation', {}))
    eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
    eval_cfg['jsonfile_prefix'] = osp.join(
        cfg.work_dir, 'val',
        time.ctime().replace(' ', '_').replace(':', '_'))
    eval_hook = CustomDistEvalHook if distributed else EvalHook
    runner.register_hook(eval_hook(val_dataloader, **eval_cfg))


def main():
    args = parse_args()
    cfg = Config.fromfile(args.config)
    if args.cfg_options is not None:
        cfg.merge_from_dict(args.cfg_options)
    apply_gradient_monitor_options(cfg, args)
    import_custom_modules(cfg, args.config)
    if cfg.get('cudnn_benchmark', False):
        torch.backends.cudnn.benchmark = True
    if cfg.get('close_tf32', False):
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    configure_runtime(cfg, args)
    validate_teacher_alignment_config(cfg)
    setup_multi_processes(cfg)

    if args.launcher == 'none':
        distributed = False
    else:
        distributed = True
        init_dist(args.launcher, **cfg.dist_params)
        _, world_size = get_dist_info()
        cfg.gpu_ids = range(world_size)
    mmcv.mkdir_or_exist(osp.abspath(cfg.work_dir))
    cfg.dump(osp.join(cfg.work_dir, osp.basename(args.config)))

    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    log_file = osp.join(cfg.work_dir, f'{timestamp}.log')
    logger = get_root_logger(
        log_file=log_file, log_level=cfg.log_level, name=get_logger_name(cfg))
    logger.propagate = False
    meta = dict()
    env_info = '\n'.join([f'{key}: {value}'
                          for key, value in collect_env().items()])
    logger.info('Environment info:\n' + '-' * 60 + '\n' + env_info + '\n' +
                '-' * 60)
    logger.info(f'Distributed training: {distributed}')
    if distributed:
        logger.info(f'num of gpus: {world_size}')
    logger.info(f'Config:\n{cfg.pretty_text}')
    if args.seed is not None:
        logger.info(f'Set random seed to {args.seed}, '
                    f'deterministic: {args.deterministic}')
        set_random_seed(args.seed, deterministic=args.deterministic)
    cfg.seed = args.seed
    meta['env_info'] = env_info
    meta['config'] = cfg.pretty_text
    meta['seed'] = args.seed
    meta['exp_name'] = osp.basename(args.config)

    model = build_model(
        cfg.model, train_cfg=cfg.get('train_cfg'), test_cfg=cfg.get('test_cfg'))
    model.init_weights()
    logger.info(f'Model:\n{model}')
    datasets = build_train_datasets(cfg)
    if cfg.checkpoint_config is not None:
        cfg.checkpoint_config.meta = dict(
            mmdet_version=mmdet_version,
            mmseg_version=mmseg_version,
            mmdet3d_version=mmdet3d_version,
            config=cfg.pretty_text,
            CLASSES=datasets[0].CLASSES,
            PALETTE=datasets[0].PALETTE
            if hasattr(datasets[0], 'PALETTE') else None)
    model.CLASSES = datasets[0].CLASSES

    target_checkpoint = cfg.get('model_target_checkpoint')
    model_target, bev_diffuser = build_frozen_teacher(cfg)
    train_occ_model(
        model, datasets, cfg, distributed=distributed,
        validate=(not args.no_validate), timestamp=timestamp, meta=meta,
        model_target=model_target, bev_diffuser=bev_diffuser,
        target_checkpoint=target_checkpoint)


if __name__ == '__main__':
    os.environ['TORCH_DISTRIBUTED_DEBUG'] = 'INFO'
    main()
