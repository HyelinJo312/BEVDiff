"""Versioned float32 pre-embedding semantic BEV cache; no source I/O on reads."""

import hashlib
import inspect
import json
import os
from pathlib import Path
import tempfile

import numpy as np


def _json(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


# 'share': [C, H, W] per-class share (legacy, the only layout before counts were stored).
# 'share+counts+support': [2C+1, H, W] = share | raw per-class pixel mass | total support.
CACHE_LAYOUTS = ('share', 'share+counts+support')


def cache_channels(num_classes, layout):
    """Stored channel count for a cache layout (num_classes excludes background)."""
    classes = num_classes + 1
    if layout == 'share':
        return classes
    if layout == 'share+counts+support':
        return 2 * classes + 1
    raise ValueError(f'cache layout must be one of {CACHE_LAYOUTS}, got {layout!r}')


def split_cache_array(value, classes, layout):
    """Return (share, counts or None, support or None) from a stored cache array."""
    expected = classes if layout == 'share' else 2 * classes + 1
    if value.ndim != 3 or value.shape[0] != expected:
        raise ValueError(f'cache layout {layout!r} expects {expected} channels, got {value.shape}')
    if layout == 'share':
        return value, None, None
    return value[:classes], value[classes:2 * classes], value[2 * classes:]


def read_cache_manifest(root):
    """Return the stored contract of a cache root, or None if it has no manifest."""
    path = Path(root) / 'manifest.json'
    if not path.is_file():
        return None
    return json.loads(path.read_text()).get('contract') or {}


def cache_contract(projection, raw_shape, depth_range, remap, source_version, layout='share'):
    import torch
    from .layout_diffusion.seg_bev_aligner_one_hot_v5 import SegBEVAligner

    if not source_version:
        raise ValueError('An explicit immutable SAM3/Metric3D source version is required')
    cache_channels(0, layout)  # validate
    fields = ('bev_h', 'bev_w', 'pc_range', 'num_classes', 'sky_as_ignore',
              'depth_range', 'depth_type', 'splat_mode', 'pixel_stride', 'eps', 'output_frame')
    defaults = inspect.signature(SegBEVAligner.__init__).parameters
    parameters = {key: projection.get(key, defaults[key].default) for key in fields}
    root = Path(__file__).resolve().parent
    # Conservative invalidation, checked once at dataset initialization, not per sample.
    code_hash = hashlib.sha256()
    for name in ('data_utils.py', 'layout_diffusion/seg_bev_aligner_one_hot_v5.py'):
        code_hash.update((root / name).read_bytes())
    contract = dict(format_version=2, projection=parameters,
        projection_runtime=dict(device_type='cuda', torch=str(torch.__version__),
            cuda=torch.version.cuda, allow_tf32=torch.backends.cuda.matmul.allow_tf32),
        raw_shape=list(raw_shape), dataset_depth_range=list(depth_range),
        remap={str(k): v for k, v in (remap or {}).items()},
        source_version=source_version, code_sha256=code_hash.hexdigest())
    # Legacy share caches carry no layout field, so their contracts stay unchanged.
    if layout != 'share':
        contract['layout'] = layout
    return json.loads(_json(contract))


def geometry_digest(meta, output_frame='lidar'):
    from .data_utils import get_content_shapes

    geometry = dict(
        cameras=[str(Path(name).parent.name + '/' + Path(name).name) for name in meta['filename']],
        lidar2img=np.asarray(meta['lidar2img'], dtype=np.float64).tolist(),
        raw_lidar2img=np.asarray(meta['metric3d_lidar2img_raw'], dtype=np.float64).tolist(),
        content_shapes=[list(shape[:2]) for shape in get_content_shapes(meta)],
        pad_shapes=[list(shape[:2]) for shape in meta['pad_shape']])
    if output_frame == 'ego':
        calibration = np.asarray(meta['ego2lidar'], dtype=np.float64)
        if calibration.shape != (4, 4) or not np.isfinite(calibration).all():
            raise ValueError('ego2lidar must be a finite 4x4 matrix')
        geometry['ego2lidar'] = calibration.tolist()
    elif output_frame != 'lidar':
        raise ValueError('output_frame must be lidar or ego')
    return _digest(geometry)


def validate_cache_config(cfg):
    """Catch CLI overrides that changed the model but not the dataset cache contract."""
    for split in ('train', 'val', 'test'):
        settings = cfg.data.get(split, {})
        if not settings.get('use_semantic_bev_cache', False):
            continue
        arguments = (settings['depth_raw_shape'], settings['depth_range'],
                     settings.get('seg_id_remap'), settings['semantic_bev_cache_source_version'])
        expected = cache_contract(cfg.unet.parameters.seg_bev_aligner, *arguments)
        declared = cache_contract(settings['semantic_bev_cache_projection'], *arguments)
        if expected != declared:
            raise ValueError(f'{split} cache projection settings differ from the current UNet aligner')


def _atomic_write(path, writer):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name, dir=str(path.parent))
    try:
        with os.fdopen(fd, 'wb') as stream:
            writer(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


class SemanticBEVCache:
    # Reads only np.load stored arrays, so the runtime that produced them cannot change
    # what is read; these fields are enforced only when this process may write entries,
    # where mixing runtimes inside one cache would make its contents inconsistent.
    WRITE_ONLY_FIELDS = ('projection_runtime',)

    def __init__(self, root, contract, required_tokens=None, create=False):
        self.root = Path(root)
        self.contract = contract
        self.manifest_path = self.root / 'manifest.json'
        self.runtime_differs = False
        if self.manifest_path.is_file():
            self.manifest = json.loads(self.manifest_path.read_text())
            stored = self.manifest.get('contract') or {}
            relaxed = () if create else self.WRITE_ONLY_FIELDS
            differing = {key: dict(cache=stored.get(key), requested=contract.get(key))
                         for key in set(stored) | set(contract)
                         if stored.get(key) != contract.get(key)}
            if (not create and stored.get('projection', {}).get('output_frame', 'lidar') == 'lidar'
                    and contract['projection'].get('output_frame', 'lidar') == 'lidar'):
                old_projection = {**stored.get('projection', {}), 'output_frame': 'lidar'}
                new_projection = {**contract['projection'], 'output_frame': 'lidar'}
                if old_projection == new_projection:
                    differing.pop('projection', None)
            self.runtime_differs = any(key in differing for key in self.WRITE_ONLY_FIELDS)
            blocking = {key: value for key, value in differing.items() if key not in relaxed}
            if blocking:
                raise ValueError('Semantic BEV cache contract mismatch; rebuild into a new '
                                 f'directory. Differing fields: {_json(blocking)}')
        elif create:
            self.manifest = dict(contract=contract, entries={})
        else:
            raise FileNotFoundError(f'Missing semantic BEV cache manifest: {self.manifest_path}')
        parameters = contract['projection']
        self.output_frame = parameters.get('output_frame', 'lidar')
        self.layout = contract.get('layout', 'share')
        self.shape = (cache_channels(parameters['num_classes'], self.layout),
                      parameters['bev_h'], parameters['bev_w'])
        if required_tokens is not None:
            missing = set(required_tokens) - self.manifest['entries'].keys()
            if missing:
                raise ValueError(f'Semantic BEV cache is incomplete: {len(missing)} missing samples; '
                                 f'example {sorted(missing)[0]}')

    def _path(self, token):
        if not token or Path(token).name != token or token in ('.', '..'):
            raise ValueError('Invalid cache sample token')
        return self.root / (token + '.npy')

    def _validate(self, value):
        if value.shape != self.shape or value.dtype != np.float32:
            raise ValueError(f'Expected float32 semantic cache ({self.layout}) with shape {self.shape}')
        if not np.isfinite(value).all() or (value < 0).any():
            raise ValueError('Invalid semantic probability values in cache')

    def load(self, token, meta):
        entry = self.manifest['entries'].get(token)
        if entry is None:
            raise KeyError(f'Sample {token} is not in semantic BEV cache')
        if entry['geometry'] != geometry_digest(meta, self.output_frame):
            raise ValueError(f'Semantic BEV cache geometry mismatch for {token}')
        value = np.load(self._path(token), allow_pickle=False)
        self._validate(value)
        if hashlib.sha256(value.tobytes()).hexdigest() != entry['sha256']:
            raise ValueError(f'Corrupt semantic BEV cache for {token}')
        return value

    def save(self, token, meta, value):
        if self.runtime_differs:
            raise ValueError('Refusing to add entries to a cache built under a different '
                             f'projection runtime: {_json(self.contract["projection_runtime"])} '
                             'differs from the manifest; rebuild into a new directory')
        value = np.ascontiguousarray(value)
        self._validate(value)
        _atomic_write(self._path(token), lambda stream: np.save(stream, value, allow_pickle=False))
        self.manifest['entries'][token] = dict(geometry=geometry_digest(meta, self.output_frame),
            sha256=hashlib.sha256(value.tobytes()).hexdigest())

    def flush(self):
        payload = (_json(self.manifest) + '\n').encode('utf-8')
        _atomic_write(self.manifest_path, lambda stream: stream.write(payload))
