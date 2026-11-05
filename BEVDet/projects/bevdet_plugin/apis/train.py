# Stage 2 training loop for DiffBEVDet4D.
#
# Copy of BEVDet's mmdet3d/apis/train.py::train_detector (hooks, EMA / custom hooks, optimizer,
# resume) with the additions of projects/mmdet3d_plugin/bevformer/apis/mmdet_train.py:
# the frozen model_target and bev_diffuser are moved to the GPU (not wrapped: they hold no
# trainable parameters) and handed to the runner (DiffEpochBasedRunner), which passes them to
# DiffBEVDet4D.train_step. mmdet3d/apis/train.py itself is left unchanged.
import warnings

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import (DistSamplerSeedHook, EpochBasedRunner, Fp16OptimizerHook,
                         OptimizerHook, build_optimizer, build_runner)

from mmdet3d.datasets import build_dataset
from mmdet3d.utils import find_latest_checkpoint
from mmdet.core import DistEvalHook as MMDET_DistEvalHook
from mmdet.core import EvalHook as MMDET_EvalHook
from mmdet.datasets import build_dataloader as build_mmdet_dataloader
from mmdet.datasets import replace_ImageToTensor
from mmdet.utils import get_root_logger as get_mmdet_root_logger


def custom_train_detector(model,
                          dataset,
                          cfg,
                          distributed=False,
                          validate=False,
                          timestamp=None,
                          meta=None,
                          model_target=None,
                          bev_diffuser=None):
    logger = get_mmdet_root_logger(log_level=cfg.log_level)

    # prepare data loaders
    dataset = dataset if isinstance(dataset, (list, tuple)) else [dataset]
    runner_type = 'EpochBasedRunner' if 'runner' not in cfg else cfg.runner['type']
    data_loaders = [
        build_mmdet_dataloader(
            ds,
            cfg.data.samples_per_gpu,
            cfg.data.workers_per_gpu,
            # `num_gpus` will be ignored if distributed
            num_gpus=len(cfg.gpu_ids),
            dist=distributed,
            seed=cfg.seed,
            runner_type=runner_type,
            persistent_workers=cfg.data.get('persistent_workers', False))
        for ds in dataset
    ]

    # put model on gpus
    if distributed:
        find_unused_parameters = cfg.get('find_unused_parameters', False)
        model = MMDistributedDataParallel(
            model.cuda(),
            device_ids=[torch.cuda.current_device()],
            broadcast_buffers=False,
            find_unused_parameters=find_unused_parameters)
        # MMCV 1.x forward() reads this private flag, absent in newer PyTorch.
        if not hasattr(model, '_use_replicated_tensor_module'):
            model._use_replicated_tensor_module = False
    else:
        model = MMDataParallel(
            model.cuda(cfg.gpu_ids[0]), device_ids=cfg.gpu_ids)

    # frozen BEV target and diffusion teacher: on this rank's GPU, not wrapped / not optimised
    if model_target is not None:
        model_target = model_target.cuda()
    if bev_diffuser is not None:
        bev_diffuser = bev_diffuser.cuda()

    # build runner
    optimizer = build_optimizer(model, cfg.optimizer)

    if 'runner' not in cfg:
        cfg.runner = {
            'type': 'EpochBasedRunner',
            'max_epochs': cfg.total_epochs
        }
        warnings.warn(
            'config is now expected to have a `runner` section, '
            'please set `runner` in your config.', UserWarning)
    else:
        if 'total_epochs' in cfg:
            assert cfg.total_epochs == cfg.runner.max_epochs

    default_args = dict(
        model=model,
        optimizer=optimizer,
        work_dir=cfg.work_dir,
        logger=logger,
        meta=meta)
    if model_target is not None or bev_diffuser is not None:
        if cfg.runner['type'] != 'DiffEpochBasedRunner':
            raise ValueError('model_target / bev_diffuser need runner type DiffEpochBasedRunner')
        default_args.update(model_target=model_target,    # frozen BEVDet4D (clean BEV x0)
                            bev_diffuser=bev_diffuser)    # frozen semantic-only teacher
    runner = build_runner(cfg.runner, default_args=default_args)

    # an ugly workaround to make .log and .log.json filenames the same
    runner.timestamp = timestamp

    # fp16 setting
    fp16_cfg = cfg.get('fp16', None)
    if fp16_cfg is not None:
        optimizer_config = Fp16OptimizerHook(
            **cfg.optimizer_config, **fp16_cfg, distributed=distributed)
    elif distributed and 'type' not in cfg.optimizer_config:
        optimizer_config = OptimizerHook(**cfg.optimizer_config)
    else:
        optimizer_config = cfg.optimizer_config

    # register hooks (MEGVIIEMAHook / SequentialControlHook come from cfg.custom_hooks)
    runner.register_training_hooks(
        cfg.lr_config,
        optimizer_config,
        cfg.checkpoint_config,
        cfg.log_config,
        cfg.get('momentum_config', None),
        custom_hooks_config=cfg.get('custom_hooks', None))

    if distributed:
        if isinstance(runner, EpochBasedRunner):
            runner.register_hook(DistSamplerSeedHook())

    # register eval hooks
    if validate:
        val_samples_per_gpu = cfg.data.val.pop('samples_per_gpu', 1)
        if val_samples_per_gpu > 1:
            cfg.data.val.pipeline = replace_ImageToTensor(cfg.data.val.pipeline)
        val_dataset = build_dataset(cfg.data.val, dict(test_mode=True))
        val_dataloader = build_mmdet_dataloader(
            val_dataset,
            samples_per_gpu=val_samples_per_gpu,
            workers_per_gpu=cfg.data.workers_per_gpu,
            dist=distributed,
            shuffle=False)
        eval_cfg = cfg.get('evaluation', {})
        eval_cfg['by_epoch'] = cfg.runner['type'] != 'IterBasedRunner'
        eval_hook = MMDET_DistEvalHook if distributed else MMDET_EvalHook
        runner.register_hook(eval_hook(val_dataloader, **eval_cfg), priority='LOW')

    resume_from = None
    if cfg.resume_from is None and cfg.get('auto_resume'):
        resume_from = find_latest_checkpoint(cfg.work_dir)

    if resume_from is not None:
        cfg.resume_from = resume_from

    if cfg.resume_from:
        runner.resume(cfg.resume_from)
    elif cfg.load_from:
        # train_onlyseg.py loads load_from into model_target and clears it, so the student
        # starts from init_weights (flagship protocol); refuse a silent student load.
        if model_target is not None:
            raise RuntimeError('cfg.load_from must be consumed by model_target, not the student')
        runner.load_checkpoint(cfg.load_from)
    runner.run(data_loaders, cfg.workflow)


def custom_train_model(model,
                       dataset,
                       cfg,
                       distributed=False,
                       validate=False,
                       timestamp=None,
                       meta=None,
                       model_target=None,
                       bev_diffuser=None):
    """Same role as projects/mmdet3d_plugin/bevformer/apis/train.py::custom_train_model."""
    custom_train_detector(
        model,
        dataset,
        cfg,
        distributed=distributed,
        validate=validate,
        timestamp=timestamp,
        meta=meta,
        model_target=model_target,
        bev_diffuser=bev_diffuser)
