"""Native occupancy baseline interfaces shared by the two diffusion stages."""
import copy
from contextlib import contextmanager
from pathlib import Path

import torch
from mmcv.parallel import MMDataParallel, MMDistributedDataParallel
from mmcv.runner import load_checkpoint
from mmdet3d.models import build_model
from safetensors.torch import load_file
from diffusers.utils.constants import SAFETENSORS_WEIGHTS_NAME


def unwrap(model):
    return getattr(model, 'module', model)


@contextmanager
def preserve_occ_state(model, reset_temporal=False):
    modes = {module: module.training for module in model.modules()}
    core = unwrap(model)
    history = core.prev_frame_info
    try:
        model.eval()
        if reset_temporal:
            core.prev_frame_info = dict(prev_bev=None, scene_token=None, prev_pos=0, prev_angle=0)
        yield
    finally:
        for module, training in modes.items():
            module.training = training
        core.prev_frame_info = history


def decode_occ_bev(model, bev):
    """Decode a supplied BEV through the native occupancy head."""
    return unwrap(model).pts_bbox_head(
        None, None, given_bev=bev)


def occ_task_modules(model):
    transformer = unwrap(model).pts_bbox_head.transformer
    return (transformer.decoder, transformer.predicter)


def occ_task_parameters(model):
    return [p for module in occ_task_modules(model) for p in module.parameters()]


def set_occ_task_train(model, training):
    for module in occ_task_modules(model):
        module.train(training)


def load_diffusion_weights(model, path):
    root = Path(path)
    paths = [root] if root.is_file() else [root / SAFETENSORS_WEIGHTS_NAME, root / 'unet' / SAFETENSORS_WEIGHTS_NAME]
    checkpoint = next((candidate for candidate in paths if candidate.is_file()), None)
    if checkpoint is None:
        raise FileNotFoundError(f'UNet safetensors checkpoint not found: {path}')
    model.load_state_dict(load_file(str(checkpoint), device='cpu'), strict=True)


def get_occ_model(args, cfg):
    """Strictly load all baseline weights, including the native occupancy decoder."""
    model = build_model(copy.deepcopy(cfg.model))
    load_checkpoint(model, args.bev_checkpoint, map_location='cpu', strict=True)
    model.cuda()
    if args.launcher == 'none':
        model = MMDataParallel(model, device_ids=[torch.cuda.current_device()])
    else:
        model = MMDistributedDataParallel(model, device_ids=[torch.cuda.current_device()],
                                         broadcast_buffers=False, find_unused_parameters=True)
        if not hasattr(model, '_use_replicated_tensor_module'):
            model._use_replicated_tensor_module = False
    model.eval()
    return model
