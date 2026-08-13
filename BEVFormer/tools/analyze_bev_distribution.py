"""Analyze BEV feature distribution between two Stage-2 setups.

Compares baseline (BEVDiffuser, layout-only Stage-1) vs ours (Semantic
Guidance Stage-1) on a paired set of val frames. Computes the following
without dumping intermediate features to disk:

  D1 — per-channel statistics (mean, std, sparsity, kurtosis)
  D2 — per-cell L2 norm distribution (histogram + spatial heatmap)
  D3 — 2D FFT power spectrum (heatmap + radial profile)
  D4 — PCA cluster of subsampled cells (colored by dominant seg class)
  D5 — paired cell-wise difference (overall / FG / BG)

Outputs PNG plots and a summary.txt to --output_dir.
"""

import argparse
import copy
import os
import sys
import time

# Make `projects.*` importable when invoked from BEVFormer/ (mirrors train.py:28),
# and `tools/` siblings (bevdiffuser, bevdiffuser_seg) bare-importable.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(_THIS_DIR + '/..')
sys.path.append(_THIS_DIR)

import matplotlib.pyplot as plt
import numpy as np
import torch

from mmcv import Config
from mmcv.runner import load_checkpoint
from mmdet3d.datasets import build_dataset
from mmdet3d.models import build_model

# Project plugins — auto-register custom datasets / detectors / pipelines.
import projects.bevdiffuser  # noqa: F401
import projects.mmdet3d_plugin  # noqa: F401
from projects.mmdet3d_plugin.datasets.builder import build_dataloader


# ---------------------------------------------------------------------------
# Online statistics accumulator
# ---------------------------------------------------------------------------

class RunningStats:
    """Streaming per-channel sums for moments up to 4th order + sparsity."""

    def __init__(self, C):
        self.n = 0
        self.S = torch.zeros(C, dtype=torch.float64)
        self.S2 = torch.zeros(C, dtype=torch.float64)
        self.S3 = torch.zeros(C, dtype=torch.float64)
        self.S4 = torch.zeros(C, dtype=torch.float64)
        self.sparse_count = torch.zeros(C, dtype=torch.float64)

    @torch.no_grad()
    def update(self, x):  # x: (N, C) on CPU
        x = x.double()
        self.n += x.shape[0]
        self.S += x.sum(0)
        self.S2 += (x ** 2).sum(0)
        self.S3 += (x ** 3).sum(0)
        self.S4 += (x ** 4).sum(0)
        self.sparse_count += (x.abs() < 0.01).double().sum(0)

    def finalize(self):
        n = self.n
        mean = self.S / n
        var = (self.S2 / n - mean ** 2).clamp_min(1e-12)
        std = var.sqrt()
        # Central moments via raw-moment relations.
        m3 = self.S3 / n - 3 * mean * self.S2 / n + 2 * mean ** 3
        m4 = (self.S4 / n - 4 * mean * self.S3 / n
              + 6 * mean ** 2 * self.S2 / n - 3 * mean ** 4)
        kurt = m4 / (var ** 2 + 1e-12) - 3
        sparsity = self.sparse_count / n
        return dict(mean=mean.float(), std=std.float(),
                    kurtosis=kurt.float(), sparsity=sparsity.float(), n=n)


class FeatureAggregator:
    """Per-feature-key accumulator: channel stats, FFT, norms, PCA samples."""

    def __init__(self, C, H, W, pca_sample_size=200):
        self.C, self.H, self.W = C, H, W
        self.pca_sample_size = pca_sample_size
        self.rstats = RunningStats(C)
        self.power_sum = torch.zeros(H, W, dtype=torch.float64)
        self.norm_samples = []  # list of (HW,) np arrays per frame
        self.pca_samples = []   # list of (pca_sample_size, C) cpu tensors
        self.pca_labels = []    # list of (pca_sample_size,) int tensors (or None)
        self.frame_count = 0

    @torch.no_grad()
    def update(self, feat, dominant_labels=None):
        """feat: (1, HW, C) on CPU.
        dominant_labels: (HW,) long tensor of class id per cell, or None.
        """
        x = feat.squeeze(0).float()  # (HW, C)
        self.rstats.update(x)

        # FFT power (per-channel 2D FFT, average over channels then sum frames)
        x_2d = x.reshape(self.H, self.W, self.C).permute(2, 0, 1)  # (C, H, W)
        fft = torch.fft.fft2(x_2d, norm='ortho')
        power = (fft.abs() ** 2).mean(dim=0)  # (H, W)
        self.power_sum += power.double()

        # Per-cell L2 norm
        self.norm_samples.append(x.norm(dim=-1).numpy())

        # PCA subsample
        idx = torch.randperm(x.shape[0])[:self.pca_sample_size]
        self.pca_samples.append(x[idx])
        if dominant_labels is not None:
            self.pca_labels.append(dominant_labels[idx])
        self.frame_count += 1


# ---------------------------------------------------------------------------
# Setup builders
# ---------------------------------------------------------------------------

DETERMINISTIC_PIPELINE_TEMPLATE = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True,
         with_attr_label=False),
    dict(type='ObjectRangeFilter',
         point_cloud_range=[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]),
    # ObjectNameFilter populated per-setup
    dict(type='NormalizeMultiviewImage',
         mean=[123.675, 116.28, 103.53],
         std=[58.395, 57.12, 57.375], to_rgb=True),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    # DefaultFormatBundle3D populated per-setup
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img']),
]


def make_pipeline(class_names):
    """Train-like pipeline minus PhotoMetricDistortion (deterministic)."""
    pipe = []
    for step in DETERMINISTIC_PIPELINE_TEMPLATE:
        s = copy.deepcopy(step)
        if s['type'] == 'ObjectRangeFilter':
            s['point_cloud_range'] = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
        pipe.append(s)
    # Insert ObjectNameFilter + DefaultFormatBundle3D per setup
    pipe.insert(3, dict(type='ObjectNameFilter', classes=class_names))
    pipe.insert(-1, dict(type='DefaultFormatBundle3D', class_names=class_names))
    return pipe


def build_setup(cfg_path, student_ckpt, teacher_init_ckpt, setup):
    """Build (cfg, student, teacher, diffuser, dataloader) for one setup."""
    cfg = Config.fromfile(cfg_path)
    cfg.model.train_cfg = None
    # Strip legacy kwargs not present in current detector signatures (e.g.,
    # v3_2's config carries `use_adapter` which has been renamed to `use_proj`).
    for legacy in ('use_adapter',):
        cfg.model.pop(legacy, None)

    # Build student detector and load trained Stage-2 weights.
    model = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model, student_ckpt, map_location='cpu', strict=False)
    model.eval().cuda()

    # Build teacher (model_target) — same arch, init from common pretrain so
    # teacher_bev_raw is identical across baseline/ours setups (sanity check).
    model_target = build_model(cfg.model, test_cfg=cfg.get('test_cfg'))
    load_checkpoint(model_target, teacher_init_ckpt, map_location='cpu',
                    strict=False)
    model_target.eval().cuda()
    for p in model_target.parameters():
        p.requires_grad_(False)

    # Build BEV diffuser — setup-specific wrapper (located under BEVFormer/tools/).
    # train.py / train_seg.py use bare imports of these (tools/ on sys.path).
    if setup == 'ours':
        from bevdiffuser_seg import BEVDiffuser
    else:
        from bevdiffuser import BEVDiffuser
    bev_diffuser = BEVDiffuser(**cfg.bev_diffuser_cfg).eval().cuda()
    for p in bev_diffuser.parameters():
        p.requires_grad_(False)

    # Build val dataloader with deterministic pipeline (LoadAnnotations3D
    # enabled so layout/seg dataset hooks can produce diffuser conditions).
    cfg.data.val.test_mode = False
    cfg.data.val.pipeline = make_pipeline(cfg.class_names)
    # `samples_per_gpu` lives in the dict but is not a dataset constructor kwarg;
    # test.py pops it before build_dataset (see tools/test.py:178).
    cfg.data.val.pop('samples_per_gpu', None)
    dataset_default_args = {
        'pc_range': cfg.point_cloud_range,
        'use_3d_bbox': cfg.use_3d_bbox,
        'num_classes': cfg.num_classes,
        'num_bboxes': cfg.num_bboxes,
    }
    dataset = build_dataset(cfg.data.val, default_args=dataset_default_args)
    loader = build_dataloader(
        dataset, samples_per_gpu=1, workers_per_gpu=2,
        dist=False, shuffle=False, num_gpus=1)
    return cfg, model, model_target, bev_diffuser, loader


# ---------------------------------------------------------------------------
# Forward (single frame, captures three BEV features)
# ---------------------------------------------------------------------------

def _build_condition(data):
    """Replicate diff_bevformer_seg.get_condition() from a dataloader batch."""
    cond = {}
    keys = [
        ('layout_obj_classes', 'obj_class'),
        ('layout_obj_bboxes', 'obj_bbox'),
        ('layout_obj_is_valid', 'is_valid_obj'),
        ('layout_obj_names', 'obj_name'),
        ('default_obj_names', 'default_obj_names'),
    ]
    for src, dst in keys:
        if src in data:
            v = data[src]
            v = v.data[0] if hasattr(v, 'data') else v
            cond[dst] = torch.stack(v).cuda() if isinstance(v, list) else v.cuda()
    return cond


def _to_cuda(x):
    if hasattr(x, 'data'):  # DataContainer
        x = x.data[0]
    if isinstance(x, list):
        return [t.cuda() if torch.is_tensor(t) else t for t in x]
    return x.cuda() if torch.is_tensor(x) else x


@torch.no_grad()
def forward_capture(model, model_target, bev_diffuser, data, setup):
    """One-frame forward that mirrors DiffBEVFormer(Seg).forward_train BEV
    extraction, returning student/teacher/denoised BEV in (1, HW, C) form."""
    img_dc = data['img']
    img_metas_dc = data['img_metas']
    img = _to_cuda(img_dc)              # (1, queue, V, C, H, W)
    img_metas = img_metas_dc.data[0]    # list of dicts (len=1, B=1)

    len_queue = img.size(1)
    prev_img = img[:, :-1]
    cur_img = img[:, -1]
    prev_img_metas = copy.deepcopy(img_metas)

    prev_bev = None
    if prev_img.size(1) > 0:
        prev_bev = model.obtain_history_bev(prev_img, prev_img_metas)
    img_metas_cur = [each[len_queue - 1] for each in img_metas]
    if not img_metas_cur[0].get('prev_bev_exists', False):
        prev_bev = None

    # Student BEV (image-only)
    img_feats = model.extract_feat(img=cur_img, img_metas=img_metas_cur)
    student_bev = model.pts_bbox_head(img_feats, img_metas_cur, prev_bev,
                                      only_bev=True)

    # Teacher raw BEV (image-only, same architecture, init from common pretrain)
    teacher_bev_raw = model_target(return_loss=False, only_bev=True,
                                   img=img, img_metas=img_metas).detach()

    B, _, C = teacher_bev_raw.shape
    H, W = model.pts_bbox_head.bev_h, model.pts_bbox_head.bev_w
    bev_2d = teacher_bev_raw.reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()

    cond = _build_condition(data)
    seg_bev_prob = None
    if setup == 'ours':
        segmaps = _to_cuda(data['seg_maps'])
        if isinstance(segmaps, list):
            segmaps = torch.stack(segmaps)
        depth_maps = None
        denoised, seg_bev_prob = bev_diffuser(bev_2d, img_metas_cur, cond,
                                              segmaps, depth_maps)
    else:
        denoised = bev_diffuser(bev_2d, cond)

    denoised = denoised.permute(0, 2, 3, 1).reshape(B, H * W, C)

    return dict(
        student_bev=student_bev.cpu(),
        teacher_bev_raw=teacher_bev_raw.cpu(),
        teacher_bev_denoised=denoised.cpu(),
        seg_bev_prob=seg_bev_prob.cpu() if seg_bev_prob is not None else None,
        sample_idx=img_metas_cur[0].get('sample_idx', None),
        H=H, W=W,
    )


# ---------------------------------------------------------------------------
# FG mask helper (from ours seg_bev_prob)
# ---------------------------------------------------------------------------

# det-relevant seg labels from SEG_LABEL_DIC (diff_bevformer_seg.py:21-28).
FG_SEG_IDX = torch.tensor([1, 3, 4, 8, 9, 10, 12, 13, 14, 15])


def fg_mask_from_seg_prob(seg_bev_prob, threshold=0.3):
    """seg_bev_prob: (1, 17, H, W) -> fg_mask (HW,) bool, dominant_label (HW,) long."""
    if seg_bev_prob is None:
        return None, None
    fg_prob = seg_bev_prob[0, FG_SEG_IDX].sum(dim=0)  # (H, W)
    fg_mask = (fg_prob > threshold).reshape(-1)
    dominant = seg_bev_prob[0].argmax(dim=0).reshape(-1)  # (HW,) in [0..16]
    return fg_mask, dominant


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_D1_channel_stats(agg_b, agg_o, key, output_dir):
    """4-panel: std, sparsity, kurtosis per channel + std histogram."""
    sb = agg_b.rstats.finalize()
    so = agg_o.rstats.finalize()
    C = sb['std'].shape[0]
    xch = np.arange(C)

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    # (a) std per channel — sorted by baseline
    order = torch.argsort(sb['std']).numpy()
    axes[0, 0].plot(xch, sb['std'][order].numpy(), label='baseline')
    axes[0, 0].plot(xch, so['std'][order].numpy(), label='ours', alpha=0.7)
    axes[0, 0].set_title(f'[{key}] per-channel std (sorted by baseline)')
    axes[0, 0].set_xlabel('channel rank')
    axes[0, 0].set_ylabel('std')
    axes[0, 0].legend()

    # (b) sparsity
    axes[0, 1].plot(xch, sb['sparsity'][order].numpy(), label='baseline')
    axes[0, 1].plot(xch, so['sparsity'][order].numpy(), label='ours', alpha=0.7)
    axes[0, 1].set_title(f'[{key}] per-channel sparsity (|x|<0.01 fraction)')
    axes[0, 1].set_xlabel('channel rank')
    axes[0, 1].set_ylabel('sparsity')
    axes[0, 1].legend()

    # (c) kurtosis
    axes[1, 0].plot(xch, sb['kurtosis'][order].numpy(), label='baseline')
    axes[1, 0].plot(xch, so['kurtosis'][order].numpy(), label='ours', alpha=0.7)
    axes[1, 0].set_title(f'[{key}] per-channel excess kurtosis')
    axes[1, 0].set_xlabel('channel rank')
    axes[1, 0].set_ylabel('kurtosis')
    axes[1, 0].legend()

    # (d) std distribution
    axes[1, 1].hist(sb['std'].numpy(), bins=40, alpha=0.5, label='baseline')
    axes[1, 1].hist(so['std'].numpy(), bins=40, alpha=0.5, label='ours')
    axes[1, 1].set_title(f'[{key}] std value histogram')
    axes[1, 1].set_xlabel('std')
    axes[1, 1].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, f'D1_channel_stats_{key}.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_D2_norm_distribution(agg_b, agg_o, key, H, W, output_dir):
    norms_b = np.concatenate(agg_b.norm_samples)
    norms_o = np.concatenate(agg_o.norm_samples)
    # Spatial avg norm
    spat_b = np.stack([n.reshape(H, W) for n in agg_b.norm_samples]).mean(0)
    spat_o = np.stack([n.reshape(H, W) for n in agg_o.norm_samples]).mean(0)

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    axes[0].hist(np.log10(norms_b + 1e-9), bins=60, alpha=0.5, label='baseline')
    axes[0].hist(np.log10(norms_o + 1e-9), bins=60, alpha=0.5, label='ours')
    axes[0].set_title(f'[{key}] log10(cell L2 norm)')
    axes[0].legend()

    vmax = max(spat_b.max(), spat_o.max())
    im1 = axes[1].imshow(spat_b, vmin=0, vmax=vmax, cmap='viridis')
    axes[1].set_title(f'[{key}] baseline avg norm')
    plt.colorbar(im1, ax=axes[1])
    im2 = axes[2].imshow(spat_o, vmin=0, vmax=vmax, cmap='viridis')
    axes[2].set_title(f'[{key}] ours avg norm')
    plt.colorbar(im2, ax=axes[2])
    diff = spat_o - spat_b
    mx = max(abs(diff.min()), abs(diff.max())) + 1e-9
    im3 = axes[3].imshow(diff, vmin=-mx, vmax=mx, cmap='RdBu_r')
    axes[3].set_title(f'[{key}] ours - baseline')
    plt.colorbar(im3, ax=axes[3])

    plt.tight_layout()
    out = os.path.join(output_dir, f'D2_norm_{key}.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_D3_power_spectrum(agg_b, agg_o, key, output_dir):
    pb = agg_b.power_sum.numpy() / max(agg_b.frame_count, 1)
    po = agg_o.power_sum.numpy() / max(agg_o.frame_count, 1)
    # fftshift to center DC
    pb_s = np.fft.fftshift(pb)
    po_s = np.fft.fftshift(po)
    ratio = np.log10((po_s + 1e-12) / (pb_s + 1e-12))

    fig, axes = plt.subplots(1, 4, figsize=(18, 4))
    im1 = axes[0].imshow(np.log10(pb_s + 1e-12), cmap='magma')
    axes[0].set_title(f'[{key}] log power baseline')
    plt.colorbar(im1, ax=axes[0])
    im2 = axes[1].imshow(np.log10(po_s + 1e-12), cmap='magma')
    axes[1].set_title(f'[{key}] log power ours')
    plt.colorbar(im2, ax=axes[1])
    mx = max(abs(ratio.min()), abs(ratio.max())) + 1e-9
    im3 = axes[2].imshow(ratio, vmin=-mx, vmax=mx, cmap='RdBu_r')
    axes[2].set_title(f'[{key}] log10(ours/baseline)')
    plt.colorbar(im3, ax=axes[2])

    # Radial profile
    H, W = pb_s.shape
    cy, cx = H // 2, W // 2
    y, x = np.indices(pb_s.shape)
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)
    rb = np.bincount(r.ravel(), weights=pb_s.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    ro = np.bincount(r.ravel(), weights=po_s.ravel()) / np.maximum(np.bincount(r.ravel()), 1)
    axes[3].plot(rb, label='baseline')
    axes[3].plot(ro, label='ours', alpha=0.7)
    axes[3].set_yscale('log')
    axes[3].set_xlabel('radial freq')
    axes[3].set_ylabel('power')
    axes[3].set_title(f'[{key}] radial profile')
    axes[3].legend()

    plt.tight_layout()
    out = os.path.join(output_dir, f'D3_freq_{key}.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    # Return high-freq ratio (radius > 30% nyquist) for summary
    rmax = max(rb.shape[0], 1)
    cutoff = int(0.3 * rmax)
    def hf_ratio(rad):
        denom = rad.sum() + 1e-12
        return rad[cutoff:].sum() / denom
    return out, hf_ratio(rb), hf_ratio(ro)


def plot_D4_pca_clusters(agg, seg_label_iter, key, tag, output_dir):
    """PCA on subsampled cells, scatter colored by dominant class id."""
    try:
        from sklearn.decomposition import PCA
    except ImportError:
        print('[D4] sklearn not available — skipping PCA plot')
        return None

    feats = torch.cat(agg.pca_samples, dim=0).numpy()  # (N, C)
    if len(agg.pca_labels) > 0:
        labels = torch.cat(agg.pca_labels, dim=0).numpy()
    else:
        labels = np.zeros(feats.shape[0], dtype=np.int64)

    pca = PCA(n_components=2)
    z = pca.fit_transform(feats)
    fig, ax = plt.subplots(figsize=(7, 6))
    uniq = np.unique(labels)
    cmap = plt.get_cmap('tab20', max(uniq.max() + 1, 17))
    for lab in uniq:
        m = labels == lab
        ax.scatter(z[m, 0], z[m, 1], s=4, color=cmap(int(lab)),
                   alpha=0.5, label=f'cls {lab}')
    ax.set_title(f'[{tag}/{key}] PCA cells (colored by dominant seg cls)\n'
                 f'explained var: {pca.explained_variance_ratio_.sum():.3f}')
    ax.legend(markerscale=2, fontsize=6, loc='best')
    plt.tight_layout()
    out = os.path.join(output_dir, f'D4_pca_{tag}_{key}.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)
    return out


def plot_D5_paired_diff(paired_all, paired_fg, paired_bg, key, H, W,
                        output_dir):
    diffs = np.concatenate(paired_all)
    diffs_fg = (np.concatenate(paired_fg) if len(paired_fg) > 0
                else np.array([]))
    diffs_bg = (np.concatenate(paired_bg) if len(paired_bg) > 0
                else np.array([]))
    spatial = np.stack([d.reshape(H, W) for d in paired_all]).mean(0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].hist(diffs, bins=60, alpha=0.5, label='all')
    if diffs_fg.size > 0:
        axes[0].hist(diffs_fg, bins=60, alpha=0.5, label='FG only')
    if diffs_bg.size > 0:
        axes[0].hist(diffs_bg, bins=60, alpha=0.5, label='BG only')
    axes[0].set_title(f'[{key}] paired cell L2 diff')
    axes[0].set_xlabel('||baseline - ours||')
    axes[0].legend()

    cats = ['all']
    means = [float(diffs.mean())]
    if diffs_fg.size > 0:
        cats.append('FG'); means.append(float(diffs_fg.mean()))
    if diffs_bg.size > 0:
        cats.append('BG'); means.append(float(diffs_bg.mean()))
    axes[1].bar(cats, means)
    axes[1].set_title(f'[{key}] mean paired diff')

    im = axes[2].imshow(spatial, cmap='viridis')
    axes[2].set_title(f'[{key}] spatial avg paired diff')
    plt.colorbar(im, ax=axes[2])

    plt.tight_layout()
    out = os.path.join(output_dir, f'D5_paired_{key}.png')
    plt.savefig(out, dpi=120)
    plt.close(fig)

    fg_mean = float(diffs_fg.mean()) if diffs_fg.size > 0 else float('nan')
    bg_mean = float(diffs_bg.mean()) if diffs_bg.size > 0 else float('nan')
    return out, float(diffs.mean()), fg_mean, bg_mean


# ---------------------------------------------------------------------------
# Summary writer
# ---------------------------------------------------------------------------

def write_summary(out_path, args, agg_b, agg_o, paired_summary, freq_summary,
                  keys, n_frames):
    with open(out_path, 'w') as f:
        f.write('=== BEV Feature Distribution Analysis Summary ===\n')
        f.write(f'Frames: {n_frames}\n')
        f.write(f'Baseline ckpt: {args.ckpt_baseline}\n')
        f.write(f'Ours     ckpt: {args.ckpt_ours}\n')
        f.write(f'Teacher init:  {args.teacher_init_ckpt}\n\n')
        for k in keys:
            sb = agg_b[k].rstats.finalize()
            so = agg_o[k].rstats.finalize()
            f.write(f'--- {k} ---\n')
            f.write(f'  Channel stats (mean across channels):\n')
            f.write(f'    std         baseline={sb["std"].mean():.4f}  '
                    f'ours={so["std"].mean():.4f}\n')
            f.write(f'    sparsity    baseline={sb["sparsity"].mean()*100:.2f}%  '
                    f'ours={so["sparsity"].mean()*100:.2f}%\n')
            f.write(f'    kurtosis    baseline={sb["kurtosis"].mean():.3f}  '
                    f'ours={so["kurtosis"].mean():.3f}\n')
            # Norm CV
            nb = np.concatenate(agg_b[k].norm_samples)
            no = np.concatenate(agg_o[k].norm_samples)
            f.write(f'  Norm CV (std/mean): baseline={nb.std()/(nb.mean()+1e-9):.3f}  '
                    f'ours={no.std()/(no.mean()+1e-9):.3f}\n')
            # FFT high-freq ratio
            if k in freq_summary:
                hb, ho = freq_summary[k]
                f.write(f'  High-freq energy ratio (>0.3 nyquist): '
                        f'baseline={hb:.4f}  ours={ho:.4f}\n')
            # Paired diff
            if k in paired_summary:
                m, fg, bg = paired_summary[k]
                f.write(f'  Paired cell L2: avg={m:.4f}  '
                        f'FG={fg:.4f}  BG={bg:.4f}  '
                        f'ratio(FG/BG)={fg/(bg+1e-9):.3f}\n')
            f.write('\n')


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--config_baseline', required=True)
    p.add_argument('--ckpt_baseline', required=True)
    p.add_argument('--config_ours', required=True)
    p.add_argument('--ckpt_ours', required=True)
    p.add_argument('--teacher_init_ckpt',
                   default='./ckpts/bevformer_tiny_epoch_24.pth',
                   help='Pretrained BEVFormer checkpoint used for both '
                        'setups\' model_target (so teacher_bev_raw is paired).')
    p.add_argument('--num_frames', type=int, default=200)
    p.add_argument('--pca_sample_size', type=int, default=200)
    p.add_argument('--output_dir', required=True)
    return p.parse_args()


def run_analysis(args):
    os.makedirs(args.output_dir, exist_ok=True)
    t0 = time.time()

    print('=== Building baseline setup ===')
    cfg_b, model_b, mt_b, diff_b, loader_b = build_setup(
        args.config_baseline, args.ckpt_baseline,
        args.teacher_init_ckpt, 'baseline')
    print('=== Building ours setup ===')
    cfg_o, model_o, mt_o, diff_o, loader_o = build_setup(
        args.config_ours, args.ckpt_ours,
        args.teacher_init_ckpt, 'ours')

    C = cfg_b.model.pts_bbox_head.in_channels
    H = cfg_b.model.pts_bbox_head.bev_h
    W = cfg_b.model.pts_bbox_head.bev_w
    print(f'BEV shape (H, W, C) = ({H}, {W}, {C})')

    keys = ['student_bev', 'teacher_bev_raw', 'teacher_bev_denoised']
    agg_b = {k: FeatureAggregator(C, H, W, args.pca_sample_size) for k in keys}
    agg_o = {k: FeatureAggregator(C, H, W, args.pca_sample_size) for k in keys}
    paired_all = {k: [] for k in keys}
    paired_fg = {k: [] for k in keys}
    paired_bg = {k: [] for k in keys}
    sample_mismatch = 0
    sanity_teacher_raw_l2 = []

    it_b = iter(loader_b)
    it_o = iter(loader_o)
    for i in range(args.num_frames):
        try:
            data_b = next(it_b)
            data_o = next(it_o)
        except StopIteration:
            print(f'Dataloader exhausted at frame {i}; stopping early.')
            break
        feats_b = forward_capture(model_b, mt_b, diff_b, data_b, 'baseline')
        feats_o = forward_capture(model_o, mt_o, diff_o, data_o, 'ours')

        if feats_b['sample_idx'] != feats_o['sample_idx']:
            sample_mismatch += 1
            if sample_mismatch <= 3:
                print(f'WARN frame {i}: sample idx mismatch '
                      f'{feats_b["sample_idx"]} vs {feats_o["sample_idx"]}')

        # Sanity: teacher_bev_raw should match closely (same init ckpt)
        tr_l2 = (feats_b['teacher_bev_raw'] - feats_o['teacher_bev_raw']
                 ).norm().item()
        sanity_teacher_raw_l2.append(tr_l2)

        # FG mask from ours seg_bev_prob (shared because same frame)
        fg_mask, dominant = fg_mask_from_seg_prob(feats_o['seg_bev_prob'])

        for k in keys:
            agg_b[k].update(feats_b[k], dominant)
            agg_o[k].update(feats_o[k], dominant)
            diff = (feats_b[k].squeeze(0) - feats_o[k].squeeze(0)
                    ).norm(dim=-1)  # (HW,)
            paired_all[k].append(diff.numpy())
            if fg_mask is not None:
                paired_fg[k].append(diff[fg_mask].numpy())
                paired_bg[k].append(diff[~fg_mask].numpy())

        if (i + 1) % 10 == 0 or i == 0:
            elapsed = time.time() - t0
            print(f'[{i + 1}/{args.num_frames}] elapsed={elapsed:.1f}s '
                  f'teacher_raw_l2_avg={np.mean(sanity_teacher_raw_l2):.4f}')

    n_done = sum(1 for _ in paired_all['student_bev'])
    print(f'Forward complete. {n_done} frames; sample mismatches: '
          f'{sample_mismatch}.')

    print('=== Generating plots ===')
    freq_summary = {}
    paired_summary = {}
    for k in keys:
        plot_D1_channel_stats(agg_b[k], agg_o[k], k, args.output_dir)
        plot_D2_norm_distribution(agg_b[k], agg_o[k], k, H, W, args.output_dir)
        _, hb, ho = plot_D3_power_spectrum(agg_b[k], agg_o[k], k,
                                           args.output_dir)
        freq_summary[k] = (hb, ho)
        _, m, fg, bg = plot_D5_paired_diff(paired_all[k], paired_fg[k],
                                           paired_bg[k], k, H, W,
                                           args.output_dir)
        paired_summary[k] = (m, fg, bg)

    plot_D4_pca_clusters(agg_b['teacher_bev_denoised'], None,
                         'teacher_bev_denoised', 'baseline', args.output_dir)
    plot_D4_pca_clusters(agg_o['teacher_bev_denoised'], None,
                         'teacher_bev_denoised', 'ours', args.output_dir)
    plot_D4_pca_clusters(agg_b['student_bev'], None, 'student_bev', 'baseline',
                         args.output_dir)
    plot_D4_pca_clusters(agg_o['student_bev'], None, 'student_bev', 'ours',
                         args.output_dir)

    summary_path = os.path.join(args.output_dir, 'summary.txt')
    write_summary(summary_path, args, agg_b, agg_o, paired_summary,
                  freq_summary, keys, n_done)
    print(f'Summary written to {summary_path}')


if __name__ == '__main__':
    args = parse_args()
    run_analysis(args)
