"""Semantic BEV prior for BEVDet / BEVDet4D pipelines (semantic-only BEV diffusion teacher).

The semantic BEV cache (``<cache_root>/<sample_token>.npy``, built by
``build_semantic_bev_cache.py`` for BEVFormer) is stored in each sample's LiDAR frame on a
50x50 grid over the x/y extent of ``pc_range`` (rows = y, cols = x, cell centres,
i.e. ``align_corners=False``). BEVDet's BEV feature lives in a different frame:

    key ego  = ego pose of the first camera of the current frame
               (``ego2globals[:, 0, 0]`` in ``BEVDet4D.prepare_inputs``; CAM_FRONT_LEFT), then
    BEV aug  = ``bda_mat`` (4x4) from ``BEVAug``, applied to every frame's BEV.

``LoadSemanticBEVPrior`` warps the cached probabilities lidar -> global -> key ego -> bda
and resamples them to the UNet prior grid with one bilinear ``grid_sample``. Bilinear
weights are convex, so 'share' vectors stay valid (sum <= 1). Points outside the cache
extent get 0, the cache's own value for unobserved cells. Image-view augmentation needs no
handling: BEVDet undoes it through post_rot/post_tran, so it does not move BEV content.
"""
import os.path as osp

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.parallel import DataContainer as DC
from pyquaternion import Quaternion

from mmdet3d.datasets.builder import PIPELINES

from .layout_diffusion.seg_bev_aligner_one_hot_v5 import SEMANTIC_TYPES, semantic_input
from .semantic_bev_cache import read_cache_manifest, split_cache_array


def pose_matrix(rotation, translation):
    """4x4 float64 transform from a (w, x, y, z) quaternion and an xyz translation."""
    mat = np.eye(4)
    mat[:3, :3] = Quaternion(rotation).rotation_matrix
    mat[:3, 3] = translation
    return mat


def lidar_to_key_ego(info, key_cam):
    """4x4 transform from the sample's LiDAR frame to BEVDet's key-ego frame (no bda)."""
    lidar2ego = pose_matrix(info['lidar2ego_rotation'], info['lidar2ego_translation'])
    ego2global = pose_matrix(info['ego2global_rotation'], info['ego2global_translation'])
    cam = info['cams'][key_cam]
    keyego2global = pose_matrix(cam['ego2global_rotation'], cam['ego2global_translation'])
    return np.linalg.inv(keyego2global) @ ego2global @ lidar2ego


def warp_bev(prob, src2dst, src_range, dst_range, out_hw):
    """Resample a BEV map into another metric frame and grid.

    Args:
        prob (Tensor): [C, Hs, Ws] on the source grid (rows = y, cols = x).
        src2dst (ndarray): 4x4 metric transform from the source to the destination frame.
        src_range, dst_range: (x_min, y_min, x_max, y_max) covered by each grid.
        out_hw: (H, W) of the destination grid.

    Destination cells are treated as points at z = 0 of the destination frame; the
    x/y-to-z coupling of the inverse transform (roll/pitch only) is ignored.
    Returns a [C, H, W] float32 tensor; values outside the source extent are 0.
    """
    dst2src = np.linalg.inv(np.asarray(src2dst, dtype=np.float64))
    rot, trans = dst2src[:2, :2], dst2src[:2, 3]
    src_range = np.asarray(src_range, dtype=np.float64)
    dst_range = np.asarray(dst_range, dtype=np.float64)
    c_s, h_s = (src_range[:2] + src_range[2:]) / 2, (src_range[2:] - src_range[:2]) / 2
    c_d, h_d = (dst_range[:2] + dst_range[2:]) / 2, (dst_range[2:] - dst_range[:2]) / 2
    # normalised dst n' -> metric p' = c_d + h_d * n' -> source p = rot p' + trans
    # -> normalised source n = (p - c_s) / h_s
    theta = np.zeros((2, 3))
    theta[:, :2] = rot * h_d[None, :] / h_s[:, None]
    theta[:, 2] = (rot @ c_d + trans - c_s) / h_s
    prob = prob.float()
    grid = F.affine_grid(torch.from_numpy(theta).float()[None],
                         (1, prob.shape[0], int(out_hw[0]), int(out_hw[1])), align_corners=False)
    return F.grid_sample(prob[None], grid, mode='bilinear', padding_mode='zeros',
                         align_corners=False)[0]


@PIPELINES.register_module()
class LoadSemanticBEVPrior(object):
    """Load the cached semantic BEV of the current frame in BEVDet's (augmented) BEV frame.

    Place it after ``BEVAug`` (it reads ``bda_mat`` from ``results['img_inputs'][-1]`` and the
    key camera from ``results['cam_names'][0]``) and add ``key`` to ``Collect3D.keys``.

    Args:
        cache_root (str): directory with ``<token>.npy`` and ``manifest.json``.
        cache_shape (tuple): stored share shape (C, H, W), C = num_classes + 1.
        out_size (tuple): (H, W) of the emitted prior (the UNet aligner's bev_h, bev_w).
        out_range (tuple | None): x/y extent of the emitted grid; defaults to the cache extent.
        semantic_type (str): 'share' (flagship) or 'share+logcount'.
        logcount_scale (float): scale of the log-count channels for 'share+logcount'.
        key (str): output key; the Stage 1 loop reads 'semantic_bev_probabilities'.
        check_manifest (bool): require a manifest whose projection matches cache_shape and
            whose output_frame is 'lidar'.
    """

    def __init__(self, cache_root, cache_shape=(17, 50, 50), out_size=(64, 64),
                 out_range=None, semantic_type='share', logcount_scale=1e5,
                 key='semantic_bev_probabilities', check_manifest=True):
        if semantic_type not in SEMANTIC_TYPES:
            raise ValueError(f'semantic_type must be one of {SEMANTIC_TYPES}, got {semantic_type!r}')
        self.cache_root = cache_root
        self.cache_shape = tuple(int(v) for v in cache_shape)
        self.out_size = tuple(int(v) for v in out_size)
        self.semantic_type = semantic_type
        self.logcount_scale = float(logcount_scale)
        self.key = key
        self.layout = 'share'
        cache_range = None
        contract = read_cache_manifest(cache_root)
        if contract is None:
            if check_manifest:
                raise FileNotFoundError(f'no manifest.json in semantic BEV cache {cache_root}')
        else:
            proj = contract.get('projection', {})
            if proj.get('output_frame') != 'lidar':
                raise ValueError(f"cache output_frame must be 'lidar', got {proj.get('output_frame')!r}")
            stored = (int(proj['num_classes']) + 1, int(proj['bev_h']), int(proj['bev_w']))
            if stored != self.cache_shape:
                raise ValueError(f'cache_shape {self.cache_shape} != manifest projection {stored}')
            pc = proj['pc_range']
            cache_range = (pc[0], pc[1], pc[3], pc[4])
            self.layout = contract.get('layout', 'share')
        if semantic_type == 'share+logcount' and self.layout != 'share+counts+support':
            raise ValueError("semantic_type='share+logcount' needs a 'share+counts+support' cache")
        if cache_range is None:
            cache_range = (-51.2, -51.2, 51.2, 51.2)
        self.cache_range = tuple(float(v) for v in cache_range)
        self.out_range = tuple(float(v) for v in (out_range if out_range is not None else self.cache_range))

    def load_cache(self, token):
        path = osp.join(self.cache_root, f'{token}.npy')
        value = np.load(path, allow_pickle=False)
        classes = self.cache_shape[0]
        stored = self.cache_shape if self.layout == 'share' else (2 * classes + 1, *self.cache_shape[1:])
        if value.shape != stored or not np.issubdtype(value.dtype, np.floating):
            raise ValueError(f'expected floating {stored} at {path}, got {value.shape} {value.dtype}')
        share, counts, _ = split_cache_array(value, classes, self.layout)
        sem = semantic_input(share, counts, self.semantic_type, self.logcount_scale)
        return torch.from_numpy(np.ascontiguousarray(sem, dtype=np.float32))

    def __call__(self, results):
        info = results['curr']
        key_cam = results['cam_names'][0]
        bda = results['img_inputs'][-1]
        bda = bda.numpy() if isinstance(bda, torch.Tensor) else np.asarray(bda)
        bda4 = np.eye(4)
        bda4[:bda.shape[0], :bda.shape[1]] = bda
        src2dst = bda4 @ lidar_to_key_ego(info, key_cam)
        prior = warp_bev(self.load_cache(info['token']), src2dst,
                         self.cache_range, self.out_range, self.out_size)
        results[self.key] = DC(prior.contiguous())
        return results

    def __repr__(self):
        return (f'{self.__class__.__name__}(cache_root={self.cache_root!r}, '
                f'cache_shape={self.cache_shape}, out_size={self.out_size}, '
                f'semantic_type={self.semantic_type!r})')
