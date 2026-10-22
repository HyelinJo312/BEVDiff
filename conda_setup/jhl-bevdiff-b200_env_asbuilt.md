# jhl-bevdiff (B200) — As-Built Environment

What was actually built on the 8x B200 server, and where reality differed from
`jhl-bevdiff-b200_build_runbook.md`.

Built: 2026-09-03. Env path: `/NHNHOME/data/miniconda3/envs/jhl-bevdiff` (14 GB).
Full pip freeze: `requirements-b200.txt`.

## Server facts

```text
GPU:             8x NVIDIA B200, 183359 MiB each, 1000 W cap
compute cap:     (10, 0)  ->  sm_100
NVIDIA driver:   580.95.05
nvidia-smi CUDA: 13.0
system nvcc:     13.1.80  (/usr/local/cuda -> cuda-13.1; no CUDA 12.x on the host)
gcc:             13.3.0
CPU / RAM:       72 cores (2x AMD EPYC 9365, NUMA 0-35 / 36-71) / 2263 GB
conda:           26.7.1 at /NHNHOME/data/miniconda3
repo root:       /NHNHOME/data/hyelin/BEVDiff
build dir:       /NHNHOME/data/hyelin/build
hostname:        DCTN-kaist
```

## The stack

```text
python            3.10.21          (conda-forge)
cuda-toolkit      12.8.1           (in-env, nvcc 12.8.93)
torch             2.7.1+cu128      arch list includes sm_100
torchvision       0.22.1+cu128
torchaudio        2.7.1+cu128
mmcv-full         1.7.2            source-built, sm_100 only
mmdet             2.28.2
mmsegmentation    0.30.0
mmdet3d           0.17.1           editable, 11 ops, sm_100 only
detectron2        0.6              editable @ a2f4a87, sm_100 only
transformers      4.46.3
diffusers         0.35.1
numpy             1.23.5
setuptools        69.5.1
```

## The CUDA 12.8 problem, and how it was solved

This host has only nvcc 13.1. `torch.utils.cpp_extension` requires an nvcc whose
major version matches `torch.version.cuda`, so nothing would build against
`torch 2.7.1+cu128`. Installing the CUDA 12.8 toolkit *into the conda env* and
pointing `CUDA_HOME` at `$CONDA_PREFIX` solved it, and kept the whole pro6000
recipe valid with only the arch list changed:

```bash
conda install -y --override-channels \
  -c nvidia/label/cuda-12.8.1 -c conda-forge cuda-toolkit=12.8.1
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export TORCH_CUDA_ARCH_LIST="10.0+PTX"
```

Build variables live in `/NHNHOME/data/hyelin/build/env.sh`; source it before
any rebuild.

## Deviations from the runbook

Five things the runbook did not anticipate. All are fixed; listed so a rebuild
does not rediscover them.

**1. `--override-channels -c nvidia/...` alone cannot solve.**
`cuda-nvml-dev` needs `libgcc-ng`, which lives on conda-forge. Both channels
must be passed together, or the solve fails with
`nothing provides libgcc-ng >=11.2.0`.

**2. conda `defaults` Terms of Service are unaccepted on this install.**
Any plain `conda create` / `conda search` dies with `CondaToSNonInteractiveError`.
Sidestepped with `--override-channels -c conda-forge` throughout, which is
preferable anyway. The alternative is
`conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
(and `.../pkgs/r`).

**3. setuptools 84 has no `pkg_resources`; mmcv 1.7.2 imports it.**
`mmcv-1.7.2/setup.py` line 6 does `from pkg_resources import ...`, so the build
died at metadata generation with `ModuleNotFoundError: No module named
'pkg_resources'`. Fixed by pinning `setuptools==69.5.1` (+ `wheel<0.46`) before
any source build. This pin also covers the mmdet3d editable install.

**4. The pro6000 patch is missing its own item 4.**
`jhl-bevdiff-pro6000_env_asbuilt.md` lists `numba.errors` → `numba.core.errors`
in `mmdet3d/datasets/pipelines/data_augment_utils.py` as part of
`mmdet3d-0.17.1-pro6000.patch`. It is not in that file — `grep -c
data_augment_utils` returns 0. The fix was evidently applied by hand on the
pro6000 box and never captured. Without it the plugin import dies at
`ModuleNotFoundError: No module named 'numba.errors'`.

Also unpinned `networkx>=2.2,<2.3` in `requirements/runtime.txt`: the mmdet3d
editable install silently downgraded networkx 3.4.2 → 2.2. networkx is imported
nowhere in mmdet3d or BEVFormer, and scikit-image only needs `>=2.2`.

Both fixes plus everything from the pro6000 patch are captured in
**`mmdet3d-0.17.1-b200.patch`** (34 files). Use that one, not the pro6000 patch.

**5. `seaborn` is an undeclared dd3d dependency.**
`projects/mmdet3d_plugin/dd3d/datasets/nuscenes.py` does `import seaborn as sns`.
It appears in `requirements-pro6000.txt` as `seaborn==0.12.2` only because that
env was cloned from a donor; no step in either plan installs it. Added
`seaborn==0.12.2`.

## numpy gets clobbered three times

`numpy==1.23.5` was re-pinned after torch (which pulled 2.2.6), after the mmcv
install (2.2.6 again, along with opencv 5.0.0.93 and yapf 0.43.0), and checked
again after the HF stack. Any rebuild should re-assert the pin after each of
those three steps rather than assuming it held.

## NCCL: P2P must stay ENABLED — the opposite of the pro6000 box

```console
$ nvidia-smi topo -p2p n     # OK for all 28 pairs
$ torchrun --nproc_per_node 8 nccl_check.py
all_reduce scalar -> 36.0    # 1+2+...+8, correct
all_reduce 256MB  -> ok
real 0m40.5s
```

NCCL connects `via P2P/CUMEM` over 32 channels. 2-rank run: 33 s. Init is ~21 s,
almost entirely CUDA kernel loading.

**A first cold run timed out at 300 s.** Every run after that completed in
33-41 s. This is a cold-cache artifact, not a P2P fault — do not "fix" it by
copying the pro6000 workaround.

This host also has **real InfiniBand**: 10 mlx5 HCAs (400/200/100 Gb) with SHARP
and GPUDirect RDMA (`nvidia-peermem` and DMABUF both enabled). So the
`NCCL_IB_DISABLE=1` carried in the old launchers is wrong here too.

## Verified working

```text
torch 2.7.1+cu128, arch list ['sm_75','sm_80','sm_86','sm_90','sm_100','sm_120','compute_120']
capability (10,0), 8 devices, GPU matmul
sm_100 in mmcv._ext (55 entries, no other arch)
mmcv MultiScaleDeformableAttention + nms on GPU
sm_100 in all 11 mmdet3d ops and detectron2._C (no other arch)
mmdet3d Voxelization / furthest_point_sample / iou3d nms_gpu on GPU
transformers Dinov2Model importable
full plugin import incl. dd3d/detectron2
BEVFormer from bev_tiny_onlyseg_sam_v2_da3.py:  33.5M params on GPU
DiffusionUNetModel from the same config:       449.2M params on GPU
train_bev_diffuser_only_seg.py imports cleanly
torchrun 2-rank and 8-rank NCCL all-reduce, P2P enabled
```

Both parameter counts match the pro6000 as-built exactly (33.5M / 449.2M), which
is the strongest available evidence that this env reproduces that one.

## How to run the entrypoints

`layout_diffusion` is a top-level module living in `projects/bevdiffuser/`, and
the configs use paths relative to that directory (`../configs/...`,
`../../data/nuscenes/`). So the launchers must run **with cwd =
`BEVFormer/projects/bevdiffuser`**, with both the BEVFormer root and that
directory on `PYTHONPATH`. `torchrun` supplies the second one implicitly via
`sys.path[0]`; a manual `python -c` does not.

## Still to do

- **Launcher fixes** (runbook step 9), not yet applied: remove
  `NCCL_P2P_DISABLE=1` (currently forced in `train_only_seg.sh` and committed),
  reconsider `NCCL_IB_DISABLE=1` given the IB hardware above, retune
  `taskset -c 16-31,48-63` for 72 cores / 2 NUMA nodes, and
  `TCNN_CUDA_ARCHITECTURES=120` → `100`.
- **Data and checkpoints** — being copied separately; out of scope for the env.
  Still absent at build time: `nuscenes_infos_temporal_{train,val}.pkl`,
  `nuscenes_depth_da3` (empty dir), `ckpts/bevformer_tiny_epoch_24.pth`, and
  the `BEVFormer/data/` symlinks.

## Reproducing

```bash
git submodule update --init --recursive
git -C mmdetection3d apply mmdet3d-0.17.1-b200.patch
# then runbook steps 1-8, with the five deviations above already folded in
```

`requirements-b200.txt` records `mmcv-full @ file:///NHNHOME/data/hyelin/build/mmcv-1.7.2`.
That path is durable on this host (unlike the pro6000 freeze's `/tmp/mmcvbuild`),
but the freeze is still not portable to another machine — mmcv must be rebuilt
from source there.
