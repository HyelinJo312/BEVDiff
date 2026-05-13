# fit_pca_sckit.py
"""
Offline PCA fitting for DINOv2 features (multi-GPU support).

Extracts 768-dim DINOv2 features from nuScenes training images,
fits sklearn PCA to reduce to C_REDUCED dims, and saves mu/P to .npz.

Usage:
    Single GPU:
        python fit_pca_sckit.py

    Multi-GPU (e.g. 4 GPUs):
        torchrun --nproc_per_node=4 dino_pca.py
"""
import os
import sys

# Allow imports from BEVFormer root (e.g. projects.mmdet3d_plugin, projects.bevdiffuser)
_BEVFORMER_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))
if _BEVFORMER_ROOT not in sys.path:
    sys.path.insert(0, _BEVFORMER_ROOT)

import numpy as np
import torch
import torch.distributed as dist
from mmcv import Config
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize as sklearn_normalize
from mmcv.runner import get_dist_info
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from mmdet3d.datasets import build_dataset
from projects.mmdet3d_plugin.datasets.builder import build_dataloader
from projects.bevdiffuser.fm_feature import GetDINOV2Feat


# =======================================================================
# 1. PCA setup
# =======================================================================

C_IN = 768
C_REDUCED = 128
MAX_SAMPLES = 500_000

DATA_ROOT = '/rhome/hyelin/projects/BEVDiffV2/BEVFormer/data'
TRAIN_INFO_PATH = os.path.join(DATA_ROOT, 'nuscenes_infos_train.pkl')
BEV_CONFIG = "/rhome/hyelin/projects/BEVDiffV2/BEVFormer/projects/configs/bevdiffuser/layout_tiny_dino_v3.py"


# =======================================================================
# 2. Distributed helpers
# =======================================================================

def setup_distributed():
    """Initialize distributed process group if launched via torchrun.

    Returns:
        rank: Process rank (0 if single-GPU)
        world_size: Total number of processes (1 if single-GPU)
        device: torch.device for this rank
    """
    if "RANK" in os.environ:
        # Launched via torchrun
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        device = torch.device(f"cuda:{rank}")
        torch.cuda.set_device(device)
        print(f"[dist] Rank {rank}/{world_size}, device: {device}")
    else:
        # Single-GPU fallback
        rank = 0
        world_size = 1
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[dist] Single-GPU mode, device: {device}")
    return rank, world_size, device


def cleanup_distributed():
    if dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return not dist.is_initialized() or dist.get_rank() == 0


# =======================================================================
# 3. NuScenes train dataloader
# =======================================================================

def build_train_dataloader(bev_cfg, batch_size=4, num_workers=4, distributed=False):
    train_dataset = build_dataset(bev_cfg.data.train,
                                  default_args={
                                      'pc_range': bev_cfg.point_cloud_range,
                                      'use_3d_bbox': bev_cfg.use_3d_bbox,
                                      'num_classes': bev_cfg.num_classes,
                                      'num_bboxes': bev_cfg.num_bboxes,
                                      'use_layout': True,
                                      'use_semantics': False,
                                  })

    train_dataloader = build_dataloader(
        train_dataset,
        samples_per_gpu=batch_size,
        workers_per_gpu=num_workers,
        num_gpus=get_dist_info()[1],
        dist=distributed,
        seed=0,
        shuffler_sampler=bev_cfg.data.shuffler_sampler,
        nonshuffler_sampler=bev_cfg.data.nonshuffler_sampler,
    )
    return train_dataloader


# =======================================================================
# 4. Feature extraction
# =======================================================================

@torch.no_grad()
def extract_features(dino_model, batch, device) -> torch.Tensor:
    """Extract DINOv2 patch features from a batch.

    Args:
        dino_model: Pre-loaded DINOv2 feature extractor.
        batch: mmdet3d-style dict batch.
        device: torch.device for this rank.

    Returns:
        img_feats: (B, V, Hp, Wp, C_IN) — spatial-last, ready to reshape(-1, C_IN)
    """
    img = batch['img'].data[0]
    len_queue = img.size(1)
    img = img[:, -1, ...].to(device, non_blocking=True)  # (B, V, C, H, W)
    img_metas = [each[len_queue - 1] for each in batch['img_metas'].data[0]]

    dino_outputs = dino_model(img, img_metas)

    img_feats = dino_outputs['last_tokens']          # (B, V, C_dino, Hp, Wp)
    img_feats = img_feats.permute(0, 1, 3, 4, 2).contiguous()  # (B, V, Hp, Wp, C_dino)

    assert img_feats.shape[-1] == C_IN, (
        f"Expected last dim = {C_IN}, got {img_feats.shape[-1]}"
    )
    return img_feats


# =======================================================================
# 5. Feature collection + PCA fit + save
# =======================================================================

def collect_feature_samples(
    dataloader,
    dino_model,
    device,
    max_samples_per_rank: int,
    batch_size_per_device: int = 4,
):
    """Collect flattened pixel-level features for PCA fitting.

    Each rank collects up to max_samples_per_rank features from its
    portion of the data (split by DistributedSampler).

    Returns:
        feature_bank: (N_samples, C_IN) numpy array, L2-normalized per sample
    """
    feature_list = []
    n_collected = 0

    for batch_idx, batch in enumerate(dataloader):
        feats = extract_features(dino_model, batch, device)  # (B, V, N, C_IN)
        feats = feats.reshape(-1, C_IN)  # (B*V*N, C_IN)
        feats_np = feats.cpu().float().numpy()  # ensure float32

        # L2 normalize each feature vector → focuses PCA on direction, not magnitude
        # feats_np = sklearn_normalize(feats_np, norm='l2', axis=1)

        n_batch = feats_np.shape[0]

        # Subsample if exceeding max_samples
        if n_collected + n_batch > max_samples_per_rank:
            remain = max_samples_per_rank - n_collected
            if remain <= 0:
                break
            idx = np.random.choice(n_batch, remain, replace=False)
            feats_np = feats_np[idx]
            n_batch = remain

        feature_list.append(feats_np)
        n_collected += n_batch

        if is_main_process() and (batch_idx + 1) % 10 == 0:
            print(
                f"[collect] batch {batch_idx + 1}, "
                f"frames {(batch_idx + 1) * batch_size_per_device}, "
                f"collected {n_collected}/{max_samples_per_rank}"
            )

        if n_collected >= max_samples_per_rank:
            break

    feature_bank = np.concatenate(feature_list, axis=0)  # (N_local, C_IN)
    if is_main_process():
        print(f"[collect] Local feature bank shape: {feature_bank.shape}")
        print(f"[collect] L2-normalized: False")
    return feature_bank


def gather_features_to_rank0(local_features, device):
    """Gather numpy feature arrays from all ranks to rank 0.

    Args:
        local_features: (N_local, C_IN) numpy array on each rank
        device: torch.device for this rank

    Returns:
        On rank 0: (N_total, C_IN) numpy array with all features.
        On other ranks: None
    """
    if not dist.is_initialized():
        return local_features

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # Communicate sizes first
    local_size = torch.tensor([local_features.shape[0]], dtype=torch.long, device=device)
    all_sizes = [torch.zeros(1, dtype=torch.long, device=device) for _ in range(world_size)]
    dist.all_gather(all_sizes, local_size)
    all_sizes = [s.item() for s in all_sizes]

    if is_main_process():
        print(f"[gather] Features per rank: {all_sizes}, total: {sum(all_sizes)}")

    # all_gather requires same-size tensors, so pad to max size
    max_size = max(all_sizes)
    local_tensor = torch.from_numpy(local_features).to(device)
    padded = torch.zeros(max_size, C_IN, dtype=torch.float32, device=device)
    padded[:local_tensor.shape[0]] = local_tensor

    gathered = [torch.zeros(max_size, C_IN, dtype=torch.float32, device=device)
                for _ in range(world_size)]
    dist.all_gather(gathered, padded)

    if rank == 0:
        # Trim each rank's tensor to its actual size
        trimmed = [g[:s].cpu().numpy() for g, s in zip(gathered, all_sizes)]
        merged = np.concatenate(trimmed, axis=0)
        print(f"[gather] Merged feature bank shape: {merged.shape}")
        return merged
    else:
        return None


def fit_and_save_pca(feature_bank, n_components, out_dir, tag):
    """Fit sklearn PCA and save projection matrix P and mean mu to .npz.

    Args:
        feature_bank: (N_samples, C_IN) numpy array (L2-normalized)
        n_components: Target dimensionality (e.g. 80)
        out_dir: Output directory
        tag: Filename tag

    Returns:
        out_path: Path to saved .npz file
    """
    os.makedirs(out_dir, exist_ok=True)

    print(f"[PCA] Feature bank stats — "
          f"shape: {feature_bank.shape}, "
          f"mean: {feature_bank.mean():.6f}, "
          f"std: {feature_bank.std():.6f}, "
          f"row-norm mean: {np.linalg.norm(feature_bank, axis=1).mean():.4f}")

    print("[PCA] Fitting PCA ...")
    pca = PCA(
        n_components=n_components,
        svd_solver="full",
        whiten=False,
    )
    pca.fit(feature_bank)

    P = pca.components_.astype(np.float32)   # (C_REDUCED, C_IN)
    mu = pca.mean_.astype(np.float32)        # (C_IN,)
    evr = pca.explained_variance_ratio_      # (n_components,)
    cumulative = np.cumsum(evr)

    # ── Per-component explained variance ──
    print(f"[PCA] P shape: {P.shape}, mu shape: {mu.shape}")
    print(f"[PCA] Total explained variance (top {n_components}): {cumulative[-1]:.4f}")
    print(f"[PCA] Per-component explained variance ratio:")
    for i in range(n_components):
        bar = '█' * int(evr[i] * 200)  # visual bar
        print(f"  PC{i+1:3d}: {evr[i]:.6f}  (cumul: {cumulative[i]:.4f})  {bar}")

    # ── Milestones ──
    for threshold in [0.50, 0.70, 0.80, 0.90, 0.95, 0.99]:
        idx = np.searchsorted(cumulative, threshold)
        if idx < n_components:
            print(f"  → {threshold*100:.0f}% variance at PC{idx+1}")
        else:
            print(f"  → {threshold*100:.0f}% variance NOT reached within {n_components} components")

    out_path = os.path.join(out_dir, f"pca_{tag}_{C_IN}_to_{n_components}_bevformer.npz")
    np.savez(out_path, mu=mu, P=P, l2_normalized=False)
    print(f"[PCA] Saved to {out_path}")
    return out_path


# =======================================================================
# 6. Main
# =======================================================================

def main():
    # 1) Distributed setup
    rank, world_size, device = setup_distributed()
    distributed = world_size > 1
    max_samples_per_rank = MAX_SAMPLES // world_size

    if is_main_process():
        print(f"[main] world_size={world_size}, "
              f"max_samples_per_rank={max_samples_per_rank}")

    # 2) Load BEV config
    bev_cfg = Config.fromfile(BEV_CONFIG)

    # 3) Build dataloader (with DistributedSampler if multi-GPU)
    train_loader = build_train_dataloader(
        bev_cfg,
        batch_size=4,
        num_workers=4,
        distributed=distributed,
    )

    # 4) Load DINOv2 model (once per rank)
    if is_main_process():
        print("[main] Loading DINOv2 model ...")
    dino_model = GetDINOV2Feat(device=device)
    dino_model.to(device)
    dino_model.eval()

    # 5) Collect features (each rank processes its shard)
    local_features = collect_feature_samples(
        dataloader=train_loader,
        dino_model=dino_model,
        device=device,
        max_samples_per_rank=max_samples_per_rank,
        batch_size_per_device=4,
    )


    # 6) Gather all features to rank 0
    feature_bank = gather_features_to_rank0(local_features, device)

    # 7) Fit PCA on rank 0 only
    if is_main_process():
        pca_path = fit_and_save_pca(
            feature_bank=feature_bank,
            n_components=C_REDUCED,
            out_dir="/rhome/hyelin/projects/BEVDiffV2/BEVFormer/pca_ckpts",
            tag="sckit",
        )
        print(f"[main] Done. PCA saved at: {pca_path}")

    # Cleanup
    cleanup_distributed()


if __name__ == "__main__":
    main()