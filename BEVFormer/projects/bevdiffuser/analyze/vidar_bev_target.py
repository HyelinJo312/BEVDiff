# SPDX-License-Identifier: AGPL-3.0
"""Frozen ViDAR BEV target for semantic-only Stage 1.

ViDAR and BEVFormer register conflicting plugin names, so ViDAR runs in a
persistent subprocess. Requests contain sample tokens; replies contain raw
float32 BEV tensors. Worker logs go to stderr, leaving stdout for the protocol.
"""

import argparse
import atexit
import copy
import hashlib
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import traceback

import numpy as np
import torch

SCRIPT = Path(__file__).resolve()
REPO = SCRIPT.parents[3]
VIDAR_CONFIG = REPO / 'ViDAR/projects/configs/vidar_pretrain/nusc_fullset/vidar_full_nusc_1future.py'
DATASET_CONFIG = REPO / 'BEVFormer/projects/configs/bevformer/bevformer_base.py'
DATA_ROOT = REPO / 'BEVFormer/data/nuscenes'
EXPECTED_SHAPE = (256, 200, 200)
EXPECTED_RANGE = (-51.2, -51.2, -5.0, 51.2, 51.2, 3.0)
PREPROCESSING = 'vidar_native_test_camera_only_bgr_std1_full_resolution'
EXPECTED_MISSING_KEYS = {
    'pts_bbox_head.transformer.reference_points.weight',
    'pts_bbox_head.transformer.reference_points.bias',
}


def sha256(path):
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def _send(stream, header, payload=b''):
    stream.write(json.dumps(header, separators=(',', ':')).encode() + b'\n')
    stream.write(payload)
    stream.flush()


def _read_header(stream):
    line = stream.readline()
    if not line:
        raise RuntimeError('ViDAR worker closed its output pipe')
    return json.loads(line)


def _read_bytes(stream, count):
    payload = bytearray(count)
    view = memoryview(payload)
    while view:
        received = stream.readinto(view)
        if not received:
            raise RuntimeError('ViDAR worker returned an incomplete BEV tensor')
        view = view[received:]
    return payload


def _build_dataset(split, vidar_cfg):
    from mmcv import Config
    from mmdet3d.datasets import build_dataset

    settings = copy.deepcopy(Config.fromfile(str(DATASET_CONFIG)).data.test)
    settings.pop('samples_per_gpu', None)
    settings['data_root'] = str(DATA_ROOT) + os.sep
    settings['ann_file'] = str(DATA_ROOT / f'nuscenes_infos_temporal_{split}.pkl')

    # ViDAR's test pipeline, without LiDAR inputs unused by its BEV encoder.
    pipeline = [copy.deepcopy(transform) for transform in vidar_cfg.test_pipeline
                if transform['type'] != 'LoadPointsFromFile']
    if [transform['type'] for transform in pipeline] != [
            'LoadMultiViewImageFromFiles', 'NormalizeMultiviewImage',
            'PadMultiViewImage', 'DefaultFormatBundle3D', 'CustomCollect3D']:
        raise ValueError('Unexpected ViDAR camera test pipeline')
    norm = pipeline[1]
    if (tuple(norm['mean']), tuple(norm['std']), norm['to_rgb']) != (
            (103.53, 116.28, 123.675), (1.0, 1.0, 1.0), False):
        raise ValueError('ViDAR image normalization differs from the checkpoint')
    pipeline[-1]['keys'] = ['img']
    settings['pipeline'] = pipeline
    dataset = build_dataset(settings)

    # The nuScenes info file stores camera paths relative to BEVFormer, while
    # this process runs from ViDAR to load its plugin.
    for info in dataset.data_infos:
        for camera in info['cams'].values():
            path = Path(camera['data_path'])
            if path.is_absolute():
                continue
            if path.parts[:2] == ('data', 'nuscenes'):
                path = Path(*path.parts[2:])
            elif not path.parts or path.parts[0] not in ('samples', 'sweeps'):
                raise ValueError(f'Unexpected nuScenes camera path: {path}')
            camera['data_path'] = str(DATA_ROOT / path)
    return dataset


def _load_model(stage1_config, checkpoint, device):
    from mmcv import Config
    from mmcv.runner import load_checkpoint
    from mmdet3d.models import build_model
    from BEVFormer.projects.bevdiffuser.analyze.visualize_vidar_vs_bevformer import vidar_import_shims

    with vidar_import_shims():
        importlib.import_module('projects.mmdet3d_plugin')
    native_cfg = Config.fromfile(str(VIDAR_CONFIG))
    stage1_cfg = Config.fromfile(str(stage1_config))
    if stage1_cfg.model != native_cfg.model:
        raise ValueError('Stage 1 ViDAR model differs from the pretrained model config')
    if tuple(float(x) for x in native_cfg.point_cloud_range) != EXPECTED_RANGE:
        raise ValueError('ViDAR point_cloud_range differs from Stage 1')
    if (int(native_cfg.bev_h_), int(native_cfg.bev_w_), int(native_cfg._dim_)) != (
            EXPECTED_SHAPE[1], EXPECTED_SHAPE[2], EXPECTED_SHAPE[0]):
        raise ValueError('ViDAR BEV shape differs from Stage 1')

    stage1_cfg.model.pretrained = None
    model = build_model(stage1_cfg.model, train_cfg=stage1_cfg.get('train_cfg'),
                        test_cfg=stage1_cfg.get('test_cfg'))
    loaded = load_checkpoint(model, str(checkpoint), map_location='cpu')
    missing = set(model.state_dict()) - set(loaded['state_dict'])
    unexpected = set(loaded['state_dict']) - set(model.state_dict())
    if missing - EXPECTED_MISSING_KEYS or unexpected:
        raise ValueError(f'ViDAR checkpoint mismatch: missing={sorted(missing)}, '
                         f'unexpected={sorted(unexpected)}')
    model.to(device).eval().requires_grad_(False)
    return model, native_cfg, sorted(missing), sorted(unexpected)


def _extract_bev(model, dataset, index_of, token, device):
    from BEVFormer.projects.bevdiffuser.analyze.visualize_vidar_vs_bevformer import as_scene_start, collate_frame, encode_bev

    images, metas = collate_frame(dataset, index_of[token], device)
    if len(metas) != 1 or metas[0]['sample_idx'] != token:
        raise ValueError(f'ViDAR dataset returned the wrong sample for {token}')
    channels, height, width = EXPECTED_SHAPE
    bev = encode_bev(model, images, as_scene_start(metas), height, width, channels)
    if bev.shape != EXPECTED_SHAPE or bev.dtype != np.float32 or not np.isfinite(bev).all():
        raise ValueError(f'Invalid ViDAR BEV for {token}: {bev.shape}, {bev.dtype}')
    return bev


def worker(args):
    protocol = os.fdopen(os.dup(sys.stdout.fileno()), 'wb', buffering=0)
    os.dup2(sys.stderr.fileno(), sys.stdout.fileno())
    sys.stdout = sys.stderr

    stage1_config = Path(args.stage1_config).resolve()
    checkpoint = Path(args.vidar_checkpoint).resolve()
    for path in (VIDAR_CONFIG, DATASET_CONFIG, stage1_config, checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    device = torch.device(args.device)
    if device.type != 'cuda':
        raise ValueError('ViDAR worker requires CUDA')
    torch.cuda.set_device(device)

    model, native_cfg, missing, unexpected = _load_model(stage1_config, checkpoint, device)
    dataset = _build_dataset(args.split, native_cfg)
    index_of = {info['token']: index for index, info in enumerate(dataset.data_infos)}
    if len(index_of) != len(dataset.data_infos):
        raise ValueError('Duplicate sample tokens in ViDAR dataset')

    provenance = dict(
        config=str(VIDAR_CONFIG), config_sha256=sha256(VIDAR_CONFIG),
        stage1_config=str(stage1_config), stage1_config_sha256=sha256(stage1_config),
        checkpoint=str(checkpoint), checkpoint_sha256=sha256(checkpoint),
        checkpoint_missing_keys=missing, checkpoint_unexpected_keys=unexpected,
        dataset_config=str(DATASET_CONFIG), dataset_config_sha256=sha256(DATASET_CONFIG),
        data_root=str(DATA_ROOT), split=args.split, preprocessing=PREPROCESSING,
        temporal_mode='single_frame_no_prev_bev', point_cloud_range=list(EXPECTED_RANGE),
        shape=list(EXPECTED_SHAPE))
    _send(protocol, dict(status='ready', provenance=provenance, num_samples=len(index_of)))

    for line in sys.stdin.buffer:
        try:
            request = json.loads(line)
            if request.get('op') == 'close':
                _send(protocol, dict(status='closed'))
                return
            if request.get('op') != 'get':
                raise ValueError('Unsupported ViDAR worker request')
            tokens = request.get('tokens')
            if not isinstance(tokens, list) or not tokens or not all(isinstance(t, str) for t in tokens):
                raise ValueError('Expected a nonempty list of sample tokens')
            with torch.inference_mode():
                bevs = [_extract_bev(model, dataset, index_of, token, device) for token in tokens]
            value = np.ascontiguousarray(np.stack(bevs))
            _send(protocol, dict(status='ok', tokens=tokens, shape=list(value.shape),
                                 dtype='float32', nbytes=value.nbytes), value.tobytes())
        except Exception as exc:
            traceback.print_exc(file=sys.stderr)
            _send(protocol, dict(status='error', message=repr(exc)))


class ViDARBEVTarget(torch.nn.Module):
    """Proxy for one frozen ViDAR worker on this rank's CUDA device."""

    is_vidar_target = True

    def __init__(self, stage1_config, checkpoint, split, device):
        super().__init__()
        self.shape = EXPECTED_SHAPE
        command = [sys.executable, str(SCRIPT), '--split', split, '--device', str(device),
                   '--stage1-config', str(Path(stage1_config).expanduser().resolve()),
                   '--vidar-checkpoint', str(Path(checkpoint).expanduser().resolve())]
        self.process = subprocess.Popen(command, cwd=str(REPO / 'ViDAR'),
                                        env=dict(os.environ, PYTHONPATH=str(REPO / 'ViDAR')),
                                        stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        try:
            ready = _read_header(self.process.stdout)
            if ready.get('status') != 'ready':
                raise RuntimeError(f'ViDAR worker startup failed: {ready}')
            self.provenance = ready['provenance']
            if tuple(self.provenance['shape']) != self.shape:
                raise ValueError('Worker BEV shape differs from Stage 1')
            atexit.register(self.close)
        except Exception:
            self.close()
            raise

    def get(self, tokens, device):
        tokens = list(tokens)
        self.process.stdin.write(json.dumps(dict(op='get', tokens=tokens)).encode() + b'\n')
        self.process.stdin.flush()
        header = _read_header(self.process.stdout)
        if header.get('status') != 'ok':
            raise RuntimeError(f'ViDAR BEV request failed: {header}')
        expected_shape = (len(tokens), *self.shape)
        expected_bytes = int(np.prod(expected_shape)) * np.dtype('float32').itemsize
        if (header.get('tokens') != tokens or tuple(header['shape']) != expected_shape
                or header['dtype'] != 'float32' or header['nbytes'] != expected_bytes):
            raise ValueError(f'ViDAR worker response does not match request: {header}')
        value = np.frombuffer(_read_bytes(self.process.stdout, expected_bytes),
                              dtype=np.float32).reshape(expected_shape)
        if not np.isfinite(value).all():
            raise ValueError('ViDAR BEV contains nonfinite values')
        return torch.from_numpy(value.copy()).to(device=device, dtype=torch.float32)

    def close(self):
        process = getattr(self, 'process', None)
        if process is None:
            return
        if process.poll() is None:
            try:
                process.stdin.write(b'{"op":"close"}\n')
                process.stdin.flush()
                _read_header(process.stdout)
                process.wait(timeout=15)
            except (OSError, RuntimeError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        process.stdin.close()
        process.stdout.close()
        self.process = None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--split', choices=('train', 'val'), required=True)
    parser.add_argument('--device', required=True)
    parser.add_argument('--stage1-config', required=True)
    parser.add_argument('--vidar-checkpoint', required=True)
    worker(parser.parse_args())
