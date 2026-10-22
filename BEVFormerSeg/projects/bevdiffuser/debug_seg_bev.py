"""
debug_seg_bev.py
----------------
SegBEVAligner 시각화 스크립트. 두 가지 모드 지원:

  class_proj  seg_id를 IPM으로 BEV에 직접 투영 → "어느 셀에 어느 class가 있는가"
  pca         학습된 seg_bev feature를 PCA로 시각화 → "embedding이 semantics를 인코딩하는가"
  both        두 모드 모두 실행

사용법:
    # class_proj 모드 (checkpoint 불필요)
    python debug_seg_bev.py \
        --bev_config ../../projects/configs/bevdiffuser/layout_tiny_seg_v2.py \
        --num_samples 4 --mode class_proj

    # pca 모드 (checkpoint 필요)
    python debug_seg_bev.py \
        --bev_config ../../projects/configs/bevdiffuser/layout_tiny_seg_v2.py \
        --checkpoint_dir /path/to/checkpoint \
        --num_samples 4 --pca_samples 20 --mode pca
"""

import argparse
import os
import sys

# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.normpath(os.path.join(SCRIPT_DIR, "../.."))
# if PROJECT_ROOT not in sys.path:
#     sys.path.insert(0, PROJECT_ROOT)
# if os.environ.get("PYTHONPATH"):
#     os.environ["PYTHONPATH"] = f"{PROJECT_ROOT}:{os.environ['PYTHONPATH']}"
# else:
#     os.environ["PYTHONPATH"] = PROJECT_ROOT

import numpy as np
import cv2
import torch
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn.decomposition import PCA
from nuscenes.nuscenes import NuScenes
from nuscenes.utils.data_classes import LidarPointCloud
from nuscenes.utils.geometry_utils import BoxVisibility
from mmcv import Config
from mmdet3d.datasets import build_dataset
from mmdet.apis import set_random_seed
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))+"/..")
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from model_utils import build_unet


# ─── Class 정의 ──────────────────────────────────────────────────────────────

SEG_CLASS_NAMES = {
    0:  'background',
    1:  'sedan',
    2:  'highway',
    3:  'bus',
    4:  'truck',
    5:  'terrain',
    6:  'tree',
    7:  'sidewalk',
    8:  'bicycle',
    9:  'barrier',
    10: 'pedestrian',
    11: 'manmade',
    12: 'motorcycle',
    13: 'crane',
    14: 'trailer',
    15: 'cone',
    16: 'sky',
}

# 프로젝트 컬러맵 (RGBA uint8 → RGB float [0,1])
_COLORS_UINT8 = np.array([
    [  0,   0,   0, 153],  # 0 background
    [255, 120,  50, 153],  # 1 sedan
    [255, 192, 203, 153],  # 2 highway
    [  0,   0, 255, 153],  # 3 bus (Blue)
    [255, 255,   0, 153],  # 4 truck (Yellow)
    [  0, 255, 255, 153],  # 5 terrain
    [  0, 175,   0, 153],  # 6 tree
    [255,   0,   0, 153],  # 7 sidewalk
    [255,  20, 147, 153],  # 8 bicycle/bicyclist (Deep Pink)
    [135,  60,   0, 153],  # 9 barrier/barricade
    [160,  32, 240, 153],  # 10 person/pedestrian
    [255,   0, 255, 153],  # 11 manmade/building
    [139, 137, 137, 153],  # 12 motorcycle
    [ 75,   0,  75, 153],  # 13 crane
    [150, 240,  80, 153],  # 14 trailer
    [184, 134,  11, 153],  # 15 cone
    [135, 206, 235, 153],  # 16 sky
], dtype=np.uint8)
CLASS_COLORS = [tuple(c[:3] / 255.0) for c in _COLORS_UINT8]  # list of (R,G,B) float tuples


# ─── Class-Projection BEV ────────────────────────────────────────────────────

@torch.no_grad()
def class_projection_bev(seg_aligner, seg_id, img_metas, num_classes=17):
    """
    학습된 embedding을 우회하여 class ID를 IPM으로 BEV에 직접 투영.

    Args:
        seg_aligner: SegBEVAligner 인스턴스 (unet.seg_aligner)
        seg_id:      (B, V, H, W) int tensor, 0-16
        img_metas:   list[dict] (len=B), 각 dict에 'lidar2img': (V,4,4)
        num_classes: 17

    Returns:
        bev_class:  (B, bev_h, bev_w) — 각 셀의 dominant class ID
        bev_count:  (B, bev_h, bev_w, num_classes) — 셀별 class 투표 수
        bev_valid:  (B, bev_h, bev_w) — valid projection이 있는 셀 마스크
    """
    B, V, H, W = seg_id.shape
    device = seg_id.device
    bev_h = seg_aligner.bev_h
    bev_w = seg_aligner.bev_w
    Q = bev_h * bev_w
    D = seg_aligner.num_points_in_pillar
    QD = Q * D
    ds = seg_aligner.seg_encoder.downsample_factor
    min_v_ratio=0.0
    # ── (1) one-hot 생성 및 downsample (SegEmbedEncoder와 동일한 순서/방식) ────
    # -1(invalid)은 0(background)으로 변환, 나머지는 [0, num_classes-1] clamp
    seg_clean = seg_id.clamp(min=-1, max=num_classes - 1).long()
    seg_clean = torch.where(seg_clean == -1, torch.zeros_like(seg_clean), seg_clean)

    # nearest downsample 먼저 → one-hot (이산 ID에 수학적으로 유효, 메모리 효율)
    seg_ds = F.interpolate(
                seg_clean.view(B * V, H, W).unsqueeze(1).float(),
                scale_factor=0.5, mode='nearest',
            ).squeeze(1).long()                                         # (B*V, H//2, W//2)
    oh_ds = F.one_hot(seg_ds, num_classes).float()              # (B*V, H//2, W//2, C)
    oh_ds = oh_ds.permute(0, 3, 1, 2).contiguous()             # (B*V, C, H//2, W//2)

    # ── (2) IPM projection (seg_aligner 내부 로직 재사용) ─────────────────────
    # seg_aligner.pc_range 사용 — point_sampling이 self.pc_range로 denormalize하므로 반드시 일치
    Z_bins = int(round(seg_aligner.pc_range[5] - seg_aligner.pc_range[2]))
    ref_3d = seg_aligner._get_reference_points(
        bev_h, bev_w, Z=Z_bins,
        num_points_in_pillar=D,
        dim='3d', bs=B, device=device, dtype=torch.float32,
    )  # (B, D, Q, 3)

    uv, bev_mask = seg_aligner.point_sampling(ref_3d, img_metas)
    # uv: (V, B, Q, D, 2),  bev_mask: (V, B, Q, D)

    imgH, imgW = seg_aligner.final_dim  # lidar2img가 calibrated된 원본 이미지 크기

    uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)
    u_px = uv_flat[..., 0]   # (B, V, QD) — 원본 이미지 픽셀 좌표
    v_px = uv_flat[..., 1]

    # valid_in = (u_px >= 0) & (u_px <= imgW - 1) & (v_px >= 0) & (v_px <= imgH - 1)
    v_min_px = imgH * min_v_ratio
    valid_in = (
        (u_px >= 0) & (u_px <= (imgW - 1)) &
        (v_px >= v_min_px) & (v_px <= (imgH - 1))   # min_v_ratio=0.0 이면 기존과 동일
    )
    mask_bv  = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD) & valid_in

    # 원본 이미지 픽셀 → [-1, 1] 정규화 (align_corners=True 기준)
    # u ∈ [0, imgW-1] → [-1, +1] → grid_sample 내부에서 feature pixel ≈ u/ds로 자동 매핑
    gx = 2.0 * (u_px / (imgW - 1.0)) - 1.0
    gy = 2.0 * (v_px / (imgH - 1.0)) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (B, V, QD, 2)

    # ── (3) one-hot sampling ──────────────────────────────────────────────────
    grid_v = grid.view(B * V, QD, 1, 2)
    sampled = F.grid_sample(
        oh_ds, grid_v,
        mode='nearest', padding_mode='zeros', align_corners=True,
    )                                                     # (B*V, C, QD, 1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)        # (B*V, QD, C)
    sampled = sampled.view(B, V, Q, D, num_classes)

    # ── (4) 마스크 적용 후 pillar+view 합산 → argmax ──────────────────────────
    mask = mask_bv.view(B, V, Q, D).unsqueeze(-1).float()
    sampled = sampled * mask                              # invalid 셀 0 처리

    # (B, Q, C): pillar(D) 방향 합산
    count_D = sampled.sum(dim=3)                          # (B, V, Q, C)
    # (B, Q, C): view(V) 방향 합산
    count = count_D.sum(dim=1)                            # (B, Q, C)

    bev_count = count.view(B, bev_h, bev_w, num_classes).cpu()
    bev_valid = (mask_bv.view(B, V, Q, D).sum(dim=(1, 3)) > 0)  # (B, Q)
    bev_valid = bev_valid.view(B, bev_h, bev_w).cpu()

    # argmax — projection이 없는 셀은 0(background)으로
    bev_class = bev_count.argmax(dim=-1)                  # (B, bev_h, bev_w)
    bev_class[~bev_valid] = 0

    return bev_class, bev_count, bev_valid


def class_projection_bev_v2(seg_aligner, seg_id, img_metas, depth_maps=None, num_classes=16, eps=1e-6):
    """
    SegBEVAligner.forward의 seg_bev_prob과 동일한 메커니즘으로 BEV 확률 맵 생성.
    Args:
        seg_aligner: SegBEVAligner 인스턴스 (unet.seg_aligner)
        seg_id:      (B, V, H, W) int tensor, 0~(num_classes-1), -1=invalid
        img_metas:   list[dict] (len=B), 각 dict에 'lidar2img': (V,4,4)
        depth_maps:  (B, V, dH, dW) float tensor, DA3 예측 깊이. None이면 가중치 미적용
        num_classes: class 수 (background 제외). default=16
        eps:         확률 정규화 분모 clamp 값

    Returns:
        seg_bev_prob: (B, C, bev_h, bev_w) — C=num_classes+1, 확률 분포
        bev_valid:    (B, bev_h, bev_w) — valid projection이 있는 셀 마스크
    """
    B, V, H, W = seg_id.shape
    device = seg_id.device
    bev_h = seg_aligner.bev_h
    bev_w = seg_aligner.bev_w
    Q = bev_h * bev_w
    D = seg_aligner.num_points_in_pillar
    QD = Q * D
    C = num_classes + 1  # SegEmbedEncoder와 동일: num_classes + 1 bins

    # ── (1) one-hot 생성 (SegEmbedEncoder와 동일한 순서/방식) ─────────────────
    # -1(invalid) → 0(background), 나머지 [0, num_classes-1] clamp
    seg_clean = seg_id.clamp(min=-1, max=num_classes - 1).long()
    seg_clean = torch.where(seg_clean == -1, torch.zeros_like(seg_clean), seg_clean)

    # nearest downsample 먼저 → one-hot (이산 ID에 수학적으로 유효, 메모리 효율)
    seg_ds = F.interpolate(
        seg_clean.view(B * V, H, W).unsqueeze(1).float(),
        scale_factor=0.5, mode='nearest',
    ).squeeze(1).long()                                         # (B*V, H//2, W//2)
    oh_ds = F.one_hot(seg_ds, num_classes=C).float()           # (B*V, H//2, W//2, C)
    oh_ds = oh_ds.permute(0, 3, 1, 2).contiguous()             # (B*V, C, H//2, W//2)

    # ── (2) IPM projection (SegBEVAligner.forward와 동일) ────────────────────
    Z_bins = int(round(seg_aligner.pc_range[5] - seg_aligner.pc_range[2]))
    z_sampling = getattr(seg_aligner, 'z_sampling', 'uniform')
    ref_3d = seg_aligner._get_reference_points(
        bev_h, bev_w, Z=Z_bins,
        num_points_in_pillar=D,
        dim='3d', bs=B, device=device, dtype=torch.float32,
        # z_sampling=z_sampling,
    )  # (B, D, Q, 3)

    uv, bev_mask, proj_depth = seg_aligner.point_sampling(ref_3d, img_metas)
    # uv: (V, B, Q, D, 2),  bev_mask: (V, B, Q, D),  proj_depth: (V, B, Q, D)

    imgH, imgW = seg_aligner.final_dim

    uv_flat = uv.permute(1, 0, 2, 3, 4).contiguous().view(B, V, QD, 2)
    u_px = uv_flat[..., 0]   # (B, V, QD)
    v_px = uv_flat[..., 1]

    # seg_aligner.v_min_frac 사용 (SegBEVAligner.forward와 동일)
    v_sky_thresh = seg_aligner.v_min_frac * imgH
    valid_in = (
        (u_px >= 0) & (u_px <= (imgW - 1)) &
        (v_px >= v_sky_thresh) & (v_px <= (imgH - 1))
    )
    mask_bv = bev_mask.permute(1, 0, 2, 3).contiguous().view(B, V, QD) & valid_in

    # 원본 이미지 픽셀 → [-1, 1] 정규화 (align_corners=True 기준)
    gx = 2.0 * (u_px / (imgW - 1.0)) - 1.0
    gy = 2.0 * (v_px / (imgH - 1.0)) - 1.0
    grid = torch.stack([gx, gy], dim=-1)  # (B, V, QD, 2)

    # ── (3) one-hot sampling ──────────────────────────────────────────────────
    grid_v = grid.view(B * V, QD, 1, 2)
    sampled = F.grid_sample(
        oh_ds, grid_v,
        mode='nearest', padding_mode='zeros', align_corners=True,
    )                                                     # (B*V, C, QD, 1)
    sampled = sampled.squeeze(-1).permute(0, 2, 1)        # (B*V, QD, C)
    sampled = sampled.view(B, V, Q, D, C)

    # ── (4) Depth Consistency Weighting + Masking (SegBEVAligner.forward와 동일) ──
    mask_bvqd = mask_bv.view(B, V, Q, D)                                      # [B, V, Q, D]
    depth_consistency_mode = getattr(seg_aligner, 'depth_consistency_mode', None)

    if depth_maps is not None and depth_consistency_mode is not None:
        dH, dW = depth_maps.shape[-2:]                                         # e.g. (448, 798)

        # (4a) DA3 depth → final_dim으로 bilinear resize (해상도 정합)
        da3 = F.interpolate(
            depth_maps.view(B * V, 1, dH, dW),
            size=(imgH, imgW), mode='bilinear', align_corners=True
        )                                                                      # [B*V, 1, imgH, imgW]

        # (4b) 투영된 (u,v) 좌표에서 DA3 depth 샘플링 (seg와 동일한 grid 사용)
        da3_sampled = F.grid_sample(
            da3, grid_v, mode='bilinear',
            padding_mode='zeros', align_corners=True
        )                                                                      # [B*V, 1, QD, 1]
        da3_sampled = da3_sampled.squeeze(1).squeeze(-1)                       # [B*V, QD]
        da3_sampled = da3_sampled.view(B, V, Q, D)                             # [B, V, Q, D]

        # (4c) proj_depth 차원 재배치: (V,B,Q,D) → (B,V,Q,D)
        proj_depth_bvqd = proj_depth.permute(1, 0, 2, 3).contiguous()         # [B, V, Q, D]

        # (4d) Depth consistency 가중치 계산
        w_c = seg_aligner._compute_depth_consistency(proj_depth_bvqd, da3_sampled)  # [B, V, Q, D]

        # (4e) 마스크 × depth consistency 가중치 동시 적용
        sampled = sampled * (mask_bvqd * w_c).unsqueeze(-1)                    # [B, V, Q, D, C]
    else:
        # depth_maps 미제공 시 기존 이진 마스크만 적용
        sampled = sampled * mask_bvqd.float().unsqueeze(-1)

    count_D = sampled.sum(dim=3)     # (B, V, Q, C) : Z축(Pillar) 방향 가중합
    count_map = count_D.sum(dim=1)   # (B, Q, C)    : View(카메라) 방향 합산

    bev_count = count_map.view(B, bev_h, bev_w, C).cpu()
    # ── (5) 확률 분포로 정규화 (SegBEVAligner.forward와 동일) ─────────────────
    denom = count_map.sum(dim=-1, keepdim=True).clamp_min(eps)
    f_bev_prob = count_map / denom  # (B, Q, C)

    # (B, C, bev_h, bev_w)
    seg_bev_prob = f_bev_prob.permute(0, 2, 1).contiguous().view(B, C, bev_h, bev_w)

    bev_valid = (mask_bv.view(B, V, Q, D).sum(dim=(1, 3)) > 0)  # (B, Q)
    bev_valid = bev_valid.view(B, bev_h, bev_w)

    return seg_bev_prob, bev_count, bev_valid


# ─── 시각화 ──────────────────────────────────────────────────────────────────

def visualize_class_bev(bev_class, bev_valid, save_path, scene_token='',
                        nusc=None, sample_token=None, bev_extent=None):
    """
    bev_class:    (bev_h, bev_w) int tensor (single sample)
    bev_valid:    (bev_h, bev_w) bool tensor
    nusc:         NuScenes 인스턴스 (None이면 LiDAR 패널 생략)
    sample_token: 해당 샘플의 nuScenes token
    bev_extent:   (x_min, x_max, y_min, y_max) — LiDAR BEV 범위
    """
    H, W = bev_class.shape
    cls_np = bev_class.cpu().numpy()
    valid_np = bev_valid.cpu().numpy()

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls_id, color in enumerate(CLASS_COLORS):
        rgb[cls_np == cls_id] = color[:3]
    rgb[~valid_np] = [0.15, 0.15, 0.15]

    has_lidar = (nusc is not None) and (sample_token is not None)
    ncols = 2 if has_lidar else 1
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 7))
    if ncols == 1:
        axes = [axes]

    col = 0
    if has_lidar:
        _draw_lidar_top_on_axes(
            nusc, sample_token, axes[col],
            lidar_render_mode="scatter",
            lidar_cmap="viridis",
            pts_size=2.0,
            pts_stride=1,
            pts_alpha=0.9,
            box_lw=0.6,
            show_boxes=True,
            bev_extent=bev_extent,
        )
        axes[col].set_title('LiDAR Top-Down BEV', fontsize=11)
        col += 1

    axes[col].imshow(rgb, origin='lower', interpolation='nearest')
    axes[col].set_title(f'Class-Projected BEV  {scene_token}', fontsize=11)
    # axes[col].set_xlabel('BEV X (→)')
    # axes[col].set_ylabel('BEV Y (↑)')
    # axes[col].set_xticks([0, W // 2, W - 1])
    # axes[col].set_yticks([0, H // 2, H - 1])

    present = sorted(set(cls_np[valid_np].flatten().tolist()))
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=f'{i}: {SEG_CLASS_NAMES[i]}')
        for i in present
    ]
    axes[col].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left',
                     fontsize=8, framealpha=0.8)

    fig.suptitle(scene_token, fontsize=10)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {save_path}')


# nuScenes 카메라 순서: 0=FRONT, 1=FRONT_RIGHT, 2=FRONT_LEFT,
#                       3=BACK,  4=BACK_LEFT,    5=BACK_RIGHT
# 시각화 배치: FRONT/BACK이 각 행 중앙에 오도록 정렬
_VIEW_ORDER = [
    (2, 'CAM_FRONT_LEFT'),  (0, 'CAM_FRONT'),  (1, 'CAM_FRONT_RIGHT'),
    (4, 'CAM_BACK_LEFT'),   (3, 'CAM_BACK'),   (5, 'CAM_BACK_RIGHT'),
]


def visualize_class_per_view(seg_id_batch, sample_idx, save_path):
    """
    원본 seg_id map (카메라 뷰별) 시각화 — IPM 입력 확인용.
    seg_id_batch: (B, V, H, W)
    FRONT/BACK이 각 행의 중앙에 위치하도록 정렬.
    """
    _, V, H, W = seg_id_batch.shape
    seg = seg_id_batch[sample_idx].cpu().numpy()  # (V, H, W)

    fig, axes = plt.subplots(2, 3, figsize=(15, 6))
    for pos, (v_idx, cam_name) in enumerate(_VIEW_ORDER):
        ax = axes[pos // 3, pos % 3]
        if v_idx >= V:
            ax.axis('off')
            continue
        rgb = np.zeros((H, W, 3), dtype=np.float32)
        for cls_id, color in enumerate(CLASS_COLORS):
            rgb[seg[v_idx] == cls_id] = color[:3]
        ax.imshow(rgb, aspect='auto')
        ax.set_title(cam_name, fontsize=9)
        ax.axis('off')

    fig.suptitle(f'Segmentation Maps (sample {sample_idx})', fontsize=11)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {save_path}')


def colorize_mask_to_rgba(mask_hw: np.ndarray, colors_rgba: np.ndarray,
                          unknown_to_bg: bool = True,
                          unknown_rgba=(0, 0, 0, 0)) -> np.ndarray:
    m = mask_hw.copy()
    if unknown_to_bg:
        m[m < 0] = 0
    else:
        pass
    max_id = colors_rgba.shape[0] - 1
    valid = (m >= 0) & (m <= max_id)

    out = np.zeros((m.shape[0], m.shape[1], 4), dtype=np.uint8)
    if not unknown_to_bg:
        out[:] = np.array(unknown_rgba, dtype=np.uint8)
    out[valid] = colors_rgba[m[valid].astype(np.int32)]
    return out

def overlay_rgba_on_bgr(rgb_bgr: np.ndarray, mask_rgba: np.ndarray) -> np.ndarray:
    rgb = rgb_bgr.astype(np.float32)
    mask_rgb = mask_rgba[..., :3].astype(np.float32)      # RGB
    mask_bgr = mask_rgb[..., ::-1]                        # BGR
    alpha = (mask_rgba[..., 3:4].astype(np.float32) / 255.0)

    out = rgb * (1.0 - alpha) + mask_bgr * alpha
    return np.clip(out, 0, 255).astype(np.uint8)

def visualize_seg_overlay_per_view(
    seg_id_batch,
    img_metas,
    len_queue,
    sample_idx,
    save_path,
    alpha=0.5,
):
    """
    원본 카메라 이미지 위에 seg_id map을 overlay하여 시각화.
    데이터 증강(padding, scale)으로 인한 화질 저하 및 불일치를 막기 위해 img_metas의 원본 파일 경로를 사용하여 이미지를 읽어옵니다.

    Args:
        seg_id_batch: (B, V, H_seg, W_seg) int tensor
        img_metas:    배치의 img_metas (e.g. data['img_metas'].data[0] 등에서 추출된 리스트)
        len_queue:    temporal queue 길이
        sample_idx:   배치 내 샘플 인덱스
        save_path:    저장 경로
        alpha:        seg overlay 불투명도 (동작하지 않음: color palette 기반 우선)
    """
    V = seg_id_batch.shape[1]
    seg = seg_id_batch[sample_idx].cpu().numpy()  # (V, H_seg, W_seg)

    # 원본 이미지 파일 경로 배열 (V개)
    img_paths = img_metas[sample_idx]['filename']

    fig, axes = plt.subplots(2, 3, figsize=(18, 8))
    for pos, (v_idx, cam_name) in enumerate(_VIEW_ORDER):
        ax = axes[pos // 3, pos % 3]
        if v_idx >= V:
            ax.axis('off')
            continue

        H_seg, W_seg = seg[v_idx].shape

        mask = seg[v_idx].copy()
        
        # 1) 원본 이미지 읽기
        img_path = img_paths[v_idx]
        rgb_bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
        if rgb_bgr is None:
            print(f"[WARN] Failed to read image: {img_path}")
            ax.axis('off')
            continue

        # 2) colorize RGBA
        mask_rgba = colorize_mask_to_rgba(mask, _COLORS_UINT8, unknown_to_bg=True)

        # 3) Resize mask to original image size
        if rgb_bgr.shape[0] != mask_rgba.shape[0] or rgb_bgr.shape[1] != mask_rgba.shape[1]:
            mask_rgba_rs = cv2.resize(mask_rgba, (rgb_bgr.shape[1], rgb_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
        else:
            mask_rgba_rs = mask_rgba

        # 4) BGR overlay 수행 (viz_semantic.py 함수 사용을 위해)
        over_bgr = overlay_rgba_on_bgr(rgb_bgr, mask_rgba_rs)
        
        # 5) matplotlib 시각화를 위해 다시 BGR -> RGB 변환
        overlay = over_bgr[..., ::-1]

        ax.imshow(overlay, aspect='auto')
        ax.set_title(cam_name, fontsize=9)
        ax.axis('off')

    # legend: seg map에 실제로 등장하는 class만 표시
    present = sorted(set(int(c) for c in seg.flatten()))
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=f'{i}: {SEG_CLASS_NAMES[i]}')
        for i in present if i in SEG_CLASS_NAMES
    ]
    fig.legend(handles=patches, loc='upper center',
               ncol=min(len(patches), 9), fontsize=7,
               framealpha=0.8, bbox_to_anchor=(0.5, 0.06))

    fig.suptitle(f'Segmentation Overlay  (sample {sample_idx})', fontsize=11)
    
    # 하단에 범례가 들어갈 수 있도록 여백(bottom=0.06)을 추가합니다.
    plt.tight_layout(rect=[0, 0.06, 1, 1])
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {save_path}')


# ─── LiDAR Top-Down ──────────────────────────────────────────────────────────

def _get_color(nusc, category_name: str):
    if category_name == 'bicycle':
        return np.array(nusc.colormap['vehicle.bicycle']) / 255.0
    if category_name == 'construction_vehicle':
        return np.array(nusc.colormap['vehicle.construction']) / 255.0
    if category_name == 'traffic_cone':
        return np.array(nusc.colormap['movable_object.trafficcone']) / 255.0
    for key in nusc.colormap.keys():
        if category_name in key:
            return np.array(nusc.colormap[key]) / 255.0
    return np.array([0, 0, 0], dtype=np.float32)

def _draw_lidar_top_on_axes(nusc, sample_token, ax,
                            view=np.eye(4),
                            box_vis_level=BoxVisibility.ANY,
                            axes_limit=50.0,
                            show_boxes=True,
                            lidar_render_mode="scatter",   # 'scatter' | 'height'
                            lidar_cmap="viridis",
                            pts_size=2.0,                 # smaller = thinner
                            pts_stride=1,                  # take every Nth point
                            pts_alpha=0.9,                 # 0~1
                            box_lw=0.6,                    # GT box line width
                            bev_extent=None):                   
    sample_record = nusc.get('sample', sample_token)
    assert 'LIDAR_TOP' in sample_record['data'], "No LIDAR_TOP for this sample."
    lidar_token = sample_record['data']['LIDAR_TOP']
    data_path, boxes, _ = nusc.get_sample_data(lidar_token, box_vis_level=box_vis_level)

    if lidar_render_mode == "height":
        import matplotlib.pyplot as plt
        curr = plt.get_cmap()
        plt.set_cmap(lidar_cmap)
        LidarPointCloud.from_file(data_path).render_height(ax, view=view)
        plt.set_cmap(curr)
    else:
        # Fine-grained scatter: controllable size/stride/alpha
        pc = LidarPointCloud.from_file(data_path).points  # (4, N)
        if pts_stride > 1:
            pc = pc[:, ::pts_stride]
        # apply view (homogeneous)
        P = np.vstack([pc[:3], np.ones(pc.shape[1], dtype=np.float32)])  # (4,N)
        PV = view @ P
        x, y = PV[0], PV[1]
        z = pc[2]  # color by original height
        # robust normalization for color
        zmin, zmax = np.percentile(z, [2.0, 98.0])
        z = np.clip((z - zmin) / (zmax - zmin + 1e-6), 0.0, 1.0)
        ax.scatter(x, y, c=z, s=pts_size, alpha=pts_alpha,
                   cmap=lidar_cmap, marker='.', linewidths=0, rasterized=True)
    
    if show_boxes:
        for box in boxes:
            c = _get_color(nusc, box.name)
            box.render(ax, view=view, colors=(c, c, c), linewidth=box_lw)

    # ★ apply bounds: prefer BEV extent if provided
    if bev_extent is not None:
        xmin, xmax, ymin, ymax = bev_extent
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
    else:
        ax.set_xlim(-axes_limit, axes_limit)
        ax.set_ylim(-axes_limit, axes_limit)

    ax.axis('off')
    ax.set_aspect('equal')


# ─── PCA 시각화 ──────────────────────────────────────────────────────────────

def visualize_pca_bev(seg_bev, bev_class, bev_valid, save_path, scene_token='', pca_global=None, nusc=None, sample_token=None, bev_extent=None):
    """
    seg_bev:   (C, H, W) float tensor — 단일 샘플의 50×50 BEV feature
    bev_class: (H, W) int tensor — class_projection_bev 결과 (class color용)
    bev_valid: (H, W) bool tensor
    pca_global: 이미 fit된 PCA 객체 (None이면 이 샘플로 fit)

    저장 파일:
      {save_path}_rgb.png     — PCA 256→3 공간 이미지
      {save_path}_scatter.png — PCA 256→2 scatter + class 색상
      {save_path}_compare.png — class proj BEV vs PCA RGB side-by-side
    """
    C, H, W = seg_bev.shape
    flat = seg_bev.permute(1, 2, 0).reshape(-1, C).numpy()  # (H*W, C)

    # ── PCA fit ──────────────────────────────────────────────────────────────
    pca3 = pca_global if pca_global is not None else PCA(n_components=3).fit(flat)
    pca2 = PCA(n_components=2).fit(flat)

    # ── 시각화 1: PCA RGB map ─────────────────────────────────────────────────
    rgb_flat = pca3.transform(flat)                          # (H*W, 3)
    lo, hi = rgb_flat.min(0), rgb_flat.max(0)
    rgb_flat = (rgb_flat - lo) / np.clip(hi - lo, 1e-6, None)
    rgb_img = rgb_flat.reshape(H, W, 3).astype(np.float32)
    rgb_img[~bev_valid.numpy()] = 0.15  # invalid 셀 회색

    fig, ax = plt.subplots(figsize=(6, 6))
    ax.imshow(rgb_img, origin='lower', interpolation='nearest')
    ax.set_title(f'seg_bev PCA RGB  {scene_token}', fontsize=11)
    ax.set_xlabel('BEV X (→)')
    ax.set_ylabel('BEV Y (↑)')
    rgb_path = save_path + '_rgb.png'
    plt.tight_layout()
    plt.savefig(rgb_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {rgb_path}')

    # ── 시각화 2: PCA scatter (class 색상) ───────────────────────────────────
    xy = pca2.transform(flat)                                # (H*W, 2)
    cls_flat = bev_class.numpy().reshape(-1)                 # (H*W,)
    valid_flat = bev_valid.numpy().reshape(-1)               # (H*W,)

    fig, ax = plt.subplots(figsize=(7, 6))
    present_classes = sorted(set(cls_flat[valid_flat].tolist()))
    for cls_id in present_classes:
        mask = valid_flat & (cls_flat == cls_id)
        if mask.sum() == 0:
            continue
        ax.scatter(xy[mask, 0], xy[mask, 1],
                   c=[CLASS_COLORS[cls_id][:3]],
                   label=f'{cls_id}: {SEG_CLASS_NAMES[cls_id]}',
                   s=8, alpha=0.7, linewidths=0)
    ax.set_title(f'seg_bev PCA Scatter (class colored)  {scene_token}', fontsize=10)
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=7, framealpha=0.8)
    scatter_path = save_path + '_scatter.png'
    plt.tight_layout()
    plt.savefig(scatter_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {scatter_path}')

    # ── 시각화 3: side-by-side (LiDAR | PCA RGB | class proj) + legend ──────────
    cls_np = bev_class.numpy()
    cls_rgb = np.zeros((H, W, 3), dtype=np.float32)
    for cls_id, color in enumerate(CLASS_COLORS):
        cls_rgb[cls_np == cls_id] = color[:3]
    cls_rgb[~bev_valid.numpy()] = 0.15

    has_lidar = (nusc is not None) and (sample_token is not None)
    ncols = 3 if has_lidar else 2
    fig, axes = plt.subplots(1, ncols, figsize=(7 * ncols, 6))

    col = 0
    if has_lidar:
        _draw_lidar_top_on_axes(
            nusc, sample_token, axes[col],
            lidar_render_mode="scatter",
            lidar_cmap="viridis",
            pts_size=2.0,
            pts_stride=1,
            pts_alpha=0.9,
            box_lw=0.6,
            show_boxes=True,
            bev_extent=bev_extent,
        )
        axes[col].set_title('LiDAR Top-Down BEV', fontsize=11)
        col += 1

    axes[col].imshow(rgb_img, origin='lower', interpolation='nearest')
    axes[col].set_title('seg_bev PCA RGB', fontsize=11)
    axes[col].axis('off')
    
    col += 1
    axes[col].imshow(cls_rgb, origin='lower', interpolation='nearest')
    axes[col].set_title('Class Projection BEV', fontsize=11)
    axes[col].axis('off')

    # class legend
    present = sorted(set(cls_np[bev_valid.numpy()].flatten().tolist()))
    patches = [
        mpatches.Patch(color=CLASS_COLORS[i], label=f'{i}: {SEG_CLASS_NAMES[i]}')
        for i in present
    ]
    axes[col].legend(handles=patches, bbox_to_anchor=(1.02, 1), loc='upper left',
                     fontsize=7, framealpha=0.8)

    fig.suptitle(scene_token, fontsize=10)
    compare_path = save_path + '_compare.png'
    plt.tight_layout()
    plt.savefig(compare_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  saved → {compare_path}')


def print_bev_stats(bev_class, bev_count, bev_valid, sample_idx=0):
    """BEV 셀 class 분포 텍스트 출력."""
    cls = bev_class[sample_idx].cpu().numpy()
    valid = bev_valid[sample_idx].cpu().numpy()
    total_valid = valid.sum()

    print(f'\n[BEV Stats] sample={sample_idx}, valid cells={total_valid}/{cls.size}')
    print(f'  {"class":<12} {"cells":>6} {"ratio":>7}')
    print(f'  {"-"*28}')
    cnt = bev_count[sample_idx]  # (H, W, C)
    for cls_id in range(17):
        n = (cls == cls_id).sum()
        if n == 0:
            continue
        avg_votes = cnt[:, :, cls_id][cls == cls_id].mean().item()
        print(f'  {SEG_CLASS_NAMES[cls_id]:<12} {n:>6}  ({n/total_valid*100:5.1f}%)  avg_votes={avg_votes:.1f}')


# ─── Main ────────────────────────────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(description='Debug SegBEVAligner visualization')
    parser.add_argument('--bev_config', required=True, help='config file path')
    parser.add_argument('--checkpoint_dir', default=None,
                        help='unet checkpoint dir (pca 모드에서 필요)')
    parser.add_argument('--mode', default='both',
                        choices=['class_proj', 'pca', 'both'],
                        help='시각화 모드')
    parser.add_argument('--num_samples', type=int, default=4,
                        help='시각화할 샘플 수')
    parser.add_argument('--pca_samples', type=int, default=20,
                        help='글로벌 PCA fit에 사용할 샘플 수 (pca 모드)')
    parser.add_argument('--output_dir', default='./debug_seg_bev_out',
                        help='출력 디렉토리')
    parser.add_argument('--scene_token', default=None, type=str,
                        help='특정 scene만 필터링하여 시각화 (scene_token 지정)')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cuda:0')
    return parser.parse_args()


def main():
    args = parse_args()
    set_random_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device(args.device)

    do_class_proj = args.mode in ('class_proj', 'both')
    do_pca        = args.mode in ('pca', 'both')

    if do_pca and args.checkpoint_dir is None:
        print('WARNING: pca 모드는 checkpoint가 필요합니다. random weights로 진행합니다.')

    # ── Config & UNet ─────────────────────────────────────────────────────────
    bev_cfg = Config.fromfile(args.bev_config)
    unet = build_unet(bev_cfg.unet)

    if args.checkpoint_dir is not None:
        unet.from_pretrained(args.checkpoint_dir, subfolder='unet')
        print(f'Loaded unet from {args.checkpoint_dir}')
    else:
        print('No checkpoint provided — using random weights')

    unet.to(device)
    unet.eval()
    seg_aligner = unet.seg_aligner

    # ── forward hook으로 seg_bev 캡처 (pca 모드) ─────────────────────────────
    captured = {}
    if do_pca:
        def _hook(module, input, output):
            # output: dict {1: (B, C, H, W), 2: ..., 4: ...}
            captured['seg_bev'] = output[1].detach().cpu()  # 50×50 원본 해상도
        seg_aligner.register_forward_hook(_hook)

    # ── Dataset ───────────────────────────────────────────────────────────────
    dataset = build_dataset(
        bev_cfg.data.train,
        default_args={
            'pc_range':      bev_cfg.point_cloud_range,
            'use_3d_bbox':   bev_cfg.use_3d_bbox,
            'num_classes':   bev_cfg.num_classes,
            'num_bboxes':    bev_cfg.num_bboxes,
            'class_names': bev_cfg.total_class,
            'seg_class': bev_cfg.seg_class
        },
    )

    if hasattr(dataset, 'version') and hasattr(dataset, 'data_root') and NuScenes is not None:
        nusc = NuScenes(version=dataset.version, dataroot=dataset.data_root, verbose=False)
    else:
        nusc = None


    # ── Scene 필터링 ──────────────────────────────────────────────────────────
    if args.scene_token is not None:
        orig_len = len(dataset.data_infos)
        dataset.data_infos = [
            info for info in dataset.data_infos
            if info['scene_token'] == args.scene_token
        ]
        if len(dataset.data_infos) == 0:
            print(f'ERROR: scene_token "{args.scene_token}" 에 해당하는 '
                  f'샘플이 없습니다 (전체 {orig_len}개 중).\n'
                  f'올바른 scene_token을 확인하세요.')
            sys.exit(1)
        # data_infos 변경 후 내부 상태 갱신
        # (1) flag 배열: sampler, _rand_another에서 사용
        dataset.flag = np.zeros(len(dataset.data_infos), dtype=np.uint8)
        # (2) queue_length=0: prepare_train_data에서 인접 프레임 접근 방지
        dataset.queue_length = 0
        print(f'Scene 필터링: {orig_len} → {len(dataset.data_infos)} samples '
              f'(scene_token={args.scene_token})')

    dataloader = build_dataloader(
        dataset,
        samples_per_gpu=1,
        workers_per_gpu=2,
        dist=False,
        shuffle=False,
        shuffler_sampler=bev_cfg.data.shuffler_sampler,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )
    print(f'Dataset size: {len(dataset)}')

    # ── Phase 1: PCA fit용 feature 누적 ──────────────────────────────────────
    global_pca = None
    if do_pca:
        print(f'\n[PCA Phase 1] {args.pca_samples}개 샘플로 글로벌 PCA fit 중...')
        all_feats = []
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                if len(all_feats) >= args.pca_samples:
                    break
                seg_maps = torch.stack(batch['seg_maps'].data[0], dim=0).to(device)
                img = batch['img'].data[0]
                len_queue = img.size(1)
                img_metas = [each[len_queue - 1] for each in batch['img_metas'].data[0]]

                depth_maps_pca = None
                if 'depth_maps' in batch.keys():
                    depth_maps_pca = torch.stack(batch['depth_maps'].data[0], dim=0).to(device)

                seg_aligner(seg_maps, img_metas, depth_maps=depth_maps_pca)   # hook이 captured['seg_bev'] 채움
                bev = captured['seg_bev']          # (B, C, H, W)
                B, C, H, W = bev.shape
                flat = bev.permute(0, 2, 3, 1).reshape(B * H * W, C).numpy()
                all_feats.append(flat)

        all_feats = np.concatenate(all_feats, axis=0)  # (N*H*W, C)
        print(f'  총 {all_feats.shape[0]}개 BEV 셀로 PCA fit (C={all_feats.shape[1]})')
        global_pca = PCA(n_components=3).fit(all_feats)
        print(f'  PCA 완료. 설명 분산: {global_pca.explained_variance_ratio_.sum():.3f}')

    # ── Phase 2: 샘플별 시각화 ────────────────────────────────────────────────
    print(f'\n[Phase 2] {args.num_samples}개 샘플 시각화 중...')
    collected = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if collected >= args.num_samples:
                break

            seg_maps = torch.stack(batch['seg_maps'].data[0], dim=0).to(device)
            img = batch['img'].data[0]
            len_queue = img.size(1)
            img_metas = [each[len_queue - 1] for each in batch['img_metas'].data[0]]

            depth_maps = None
            if 'depth_maps' in batch.keys():
                depth_maps = torch.stack(batch['depth_maps'].data[0], dim=0).to(device)

            print(f'\n[Step {step}] seg_maps {tuple(seg_maps.shape)}, '
                  f'unique ids={seg_maps.unique().tolist()}')

            # class-projection BEV (항상 계산 — PCA scatter의 class color에도 필요)
            # bev_class, bev_count, bev_valid = class_projection_bev(
            #     seg_aligner, seg_maps, img_metas
            # )

            seg_bev_prob, bev_count, bev_valid = class_projection_bev_v2(seg_aligner, seg_maps, img_metas, depth_maps=depth_maps)
            bev_class = seg_bev_prob.argmax(dim=1).cpu()  # (B, bev_h, bev_w)
            bev_valid = bev_valid.cpu()
            
            # PCA 모드: seg_aligner 재실행해서 hook 갱신
            if do_pca:
                seg_aligner(seg_maps, img_metas, depth_maps=depth_maps)
                seg_bev_batch = captured['seg_bev']  # (B, C, H, W)

            for b in range(seg_maps.shape[0]):
                if collected >= args.num_samples:
                    break

                scene = img_metas[b].get('scene_token', f'step{step}_b{b}')[:12]

                # # multi-view seg map은 모든 모드에서 시각화
                # view_png = os.path.join(
                #     args.output_dir, f'seg_views_{collected:03d}_{scene}.png')
                # visualize_class_per_view(seg_maps.cpu(), b, view_png)

                # seg map + 카메라 이미지 overlay 시각화
                overlay_png = os.path.join(
                    args.output_dir, f'seg_overlay_{collected:03d}_{scene}.png')
                visualize_seg_overlay_per_view(
                    seg_maps.cpu(), img_metas, len_queue, b, overlay_png)

                if do_class_proj:
                    print_bev_stats(bev_class, bev_count, bev_valid, sample_idx=b)

                    bev_png = os.path.join(
                        args.output_dir, f'bev_class_{collected:03d}_{scene}.png')
                    pc_range = bev_cfg.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
                    x_min, y_min, _, x_max, y_max, _ = pc_range
                    extent = (x_min, x_max, y_min, y_max)
                    sample_token_b = img_metas[b].get('sample_idx', None)
                    visualize_class_bev(
                        bev_class[b], bev_valid[b], bev_png, scene,
                        nusc=nusc,
                        sample_token=sample_token_b,
                        bev_extent=extent,
                    )

                if do_pca:
                    print_bev_stats(bev_class, bev_count, bev_valid, sample_idx=b)

                    pca_prefix = os.path.join(
                        args.output_dir, f'pca_{collected:03d}_{scene}')
                        
                    # Calculate bev_extent from pc_range
                    pc_range = bev_cfg.get('point_cloud_range', [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0])
                    x_min, y_min, _, x_max, y_max, _ = pc_range
                    extent = (x_min, x_max, y_min, y_max)
                    
                    sample_token = img_metas[b].get('sample_idx', None)
                    
                    visualize_pca_bev(
                        seg_bev=seg_bev_batch[b],       # (C, H, W)
                        bev_class=bev_class[b],          # (H, W)
                        bev_valid=bev_valid[b],          # (H, W)
                        save_path=pca_prefix,
                        scene_token=scene,
                        pca_global=global_pca,
                        nusc=nusc,
                        sample_token=sample_token,
                        bev_extent=extent,
                    )

                collected += 1

    print(f'\nDone. {collected} samples saved to {args.output_dir}/')


if __name__ == '__main__':
    main()
