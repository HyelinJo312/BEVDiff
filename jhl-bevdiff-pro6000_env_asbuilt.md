# jhl-bevdiff-pro6000 — As-Built Environment

Companion to `jhl-bevdiff-pro6000_env_upgrade.md`, recording what was actually
installed on the RTX PRO 6000 Blackwell server and where reality differed from
the plan.

Built: 2026-09-02. Env path: `/home/user/miniconda3/envs/jhl-bevdiff-pro6000`.
Full pip freeze: `requirements-pro6000.txt`.

## Server facts

```text
GPU:            8x NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition
VRAM:           97887 MiB each, 300 W cap
compute cap:    (12, 0)  ->  sm_120
NVIDIA driver:  580.126.20
nvidia-smi CUDA: 13.0
system nvcc:    12.8.61  (/usr/local/cuda)
gcc:            13.3.0
CPU / RAM:      256 cores / 503 GB
repo root:      /home/user/data/hyelin/BEVDiff   (not /rhome/hyelin/projects/BEVDiffV2)
```

## How the env was created

The plan called for a source build of `mmcv-full==1.7.2`. That was unnecessary:
`hiwon-nvidia`, `shh-spd`, and `shh-spd2` on this server already carry the exact
target stack, with `sm_120` present in `mmcv._ext`. So:

```bash
conda create -y -n jhl-bevdiff-pro6000 --clone hiwon-nvidia
# then drop a stray easy-install.pth pointing at another project's ops:
rm /home/user/miniconda3/envs/jhl-bevdiff-pro6000/lib/python3.10/site-packages/easy-install.pth
```

That yields python 3.10.20, torch 2.7.1+cu128, torchvision 0.22.1+cu128,
mmcv-full 1.7.2 (sm_120), mmdet 2.28.2, mmsegmentation 0.30.0, numpy 1.23.5 —
all matching the plan.

## Submodules

`mmdetection3d/` and `TPVFormer/` were empty. They are gitlinks (mode 160000)
but `.gitmodules` was missing, so `git submodule update` could not resolve them.
`.gitmodules` has been restored; both were cloned at their pinned commits:

```text
mmdetection3d  f1107977dfd26155fc1f83779ee6535d2468f449   (v0.17.1)
TPVFormer      459bc060901c9c4920f802252f04b290a449e4a1
```

TPVFormer is referenced nowhere in the code — it is a placeholder. mmdet3d is
required.

## Native builds (all sm_120)

```bash
export CUDA_HOME=/usr/local/cuda FORCE_CUDA=1 MAX_JOBS=48
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
```

- `mmcv._ext` — inherited from the cloned env, already sm_120.
- `mmdet3d` — 11 extensions built from `mmdetection3d/`, editable install.
  Apply `mmdet3d-0.17.1-pro6000.patch` to a fresh v0.17.1 checkout first.
- `detectron2` 0.6 — cloned to `/home/user/data/hyelin/detectron2`, editable
  install, `_C` built for sm_120. Required because
  `projects/mmdet3d_plugin/__init__.py` does `from .dd3d import *` and dd3d
  imports detectron2 unconditionally.

BEVFormer itself contains no `.cu`/`.cpp`. The `voxel_pooling_train` /
`voxel_pooling_inference` ops named in the upgrade plan do not exist in this
checkout, so mmcv, mmdet3d, and detectron2 are the complete set of native builds.

## Code patches (`mmdet3d-0.17.1-pro6000.patch`)

1. Torch 2.x C++ port of the ops: add `#include <ATen/cuda/CUDAContext.h>`,
   drop `THC/THC.h` and `extern THCState *state`, `-std=c++14` -> `c++17`.
2. Comment out the `mmcv<=1.4.0` version gate in `mmdet3d/__init__.py`.
3. Unpin numpy/numba in `requirements/runtime.txt`.
4. `numba.errors` -> `numba.core.errors` in
   `mmdet3d/datasets/pipelines/data_augment_utils.py`.
5. `@CONV_LAYERS.register_module(force=True)` for the 10 spconv classes in
   `mmdet3d/ops/spconv/conv.py`. mmcv 1.7.2 vendored mmdet3d's spconv and
   registers the same names first, so without `force` the import dies with
   `KeyError: 'SparseConv2d is already registered in conv layer'`.

Items 1-3 came from `/home/user/data/sihwan/mmdetection3d`, which is the same
upstream commit already forward-ported on this server. Items 4-5 were found here.

## Deviation from the plan: the HF stack

The plan pins `transformers==4.29.2` / `diffusers==0.20.0`. **Those do not run
this branch.** `projects/bevdiffuser/fm_feature.py` imports `Dinov2Model`, added
in transformers 4.31, so the plan's pin fails at import. The plan's numbers came
from upstream BEVDiffuser; the environment that actually ran this code
(`requirements`, `jhl-bevdiff_env_summary.txt`) used much newer versions.
Installed instead, matching the real source env:

```text
transformers==4.46.3      (plan said 4.29.2 — too old for DINOv2)
diffusers==0.35.1         (plan said 0.20.0)
tokenizers==0.20.3
huggingface-hub==0.34.4
safetensors==0.5.3
datasets==3.1.0
accelerate==0.20.3        (as planned)
wandb==0.15.8             (as planned)
```

Other pins follow the plan: numpy 1.23.5, scipy 1.10.1, numba 0.57.1,
scikit-image 0.19.3, timm 0.6.13, trimesh 2.35.39, opencv 4.8.1.78.
`opencv-python` was kept rather than swapped for `-headless`; `libGL.so.1` is
present on this host and `import cv2` works.

Dead dependencies deliberately skipped: `pytorch_lightning`, `taming`, `clip`,
`kornia`, `albumentations`. These appear only in unused `ldm/` files and were
absent from the original working env too.

## NCCL: P2P must be disabled

```console
$ nvidia-smi topo -p2p n
        GPU0  GPU1  GPU2  ...
 GPU0    X    NS    NS
 GPU1   NS     X    NS
```

Every pair reports **NS (Not Supported)**. With P2P left on, a two-rank
`torchrun` all-reduce hangs indefinitely at init. With `NCCL_P2P_DISABLE=1` it
completes immediately.

The `projects/bevdiffuser/*.sh` launchers were written for the old A6000 server
and several set `NCCL_P2P_DISABLE=0` explicitly, or leave it commented out
(P2P on by default). **Every one of them will deadlock here.** Required in each
launcher:

```bash
export NCCL_IB_DISABLE=1
export NCCL_P2P_DISABLE=1
```

`train_only_seg.sh` has been updated. The others still need it:
`train_seg_da3.sh`, `train_original.sh`, `train_seg_v2.sh` (explicit `=0`), and
`train_seg.sh`, `train_seg_v3.sh`, `train_dino.sh`, `train_only_seg_v2.sh`, plus
the `test_*.sh` scripts (commented out).

`TCNN_CUDA_ARCHITECTURES=86` in those scripts is stale for Blackwell (should be
120), though nothing in this repo imports tiny-cuda-nn.

## Verified working

```text
all packages import at the pinned versions
mmcv MultiScaleDeformableAttention + nms run on GPU
mmdet3d Voxelization / furthest_point_sample / iou3d nms_gpu run on GPU
sm_120 confirmed in mmcv._ext, all 11 mmdet3d ops, detectron2._C
full plugin import incl. dd3d/detectron2
BEVFormer built from bev_tiny_onlyseg_sam_v2_da3.py: 33.5M params, moved to GPU
DiffusionUNetModel built from the same config: 449.2M params, moved to GPU
train_bev_diffuser_only_seg.py imports cleanly
torchrun 2-rank NCCL all-reduce (with NCCL_P2P_DISABLE=1)
```

## Not yet available: data and checkpoints

The environment is complete; a training run is still blocked on assets that do
not exist on this server.

- `data/nuscenes/nuscenes_infos_temporal_{train,val}.pkl` — BEVFormer's temporal
  infos. `/home/user/data/Dataset/nuscenes` has raw nuScenes and the standard
  mmdet3d `nuscenes_infos_*.pkl`, but not the temporal variants. Regenerate with
  `tools/create_data.py`.
- `data/nuscenes_depth_da3` — DA3 depth maps, referenced by
  `bev_tiny_onlyseg_sam_v2_da3.py`. Not present anywhere; must be generated.
- SAM3 semantic maps (`nuscenes_semantic_sam3`) — likewise absent.
- `ckpts/bevformer_tiny_epoch_24.pth` — the BEV backbone checkpoint the launcher
  loads.

`data/` and `ckpts/` are gitignored, so these were never going to arrive with
the clone.
