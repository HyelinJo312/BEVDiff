# jhl-bevdiff (B200) — Environment Build Plan

Target: build a working env for `BEVDiff/` on the **8x NVIDIA B200** server.

Predecessors:
- `jhl-bevdiff_env_summary.txt` — the original A6000 env (torch 1.10 / mmcv 1.4.0), the semantic baseline.
- `jhl-bevdiff-pro6000_env_upgrade.md` — the upgrade plan written for an RTX PRO 6000 (sm_120) box.
- `jhl-bevdiff-pro6000_env_asbuilt.md` — what actually got built there, and the plan's errors.

**The pro6000 as-built is the recipe to follow, not the upgrade plan.** But this
is a different machine, and four facts differ in ways that change the steps.

## Server facts (measured 2026-09-03)

```text
GPU:             8x NVIDIA B200, 183359 MiB each, 1000 W cap
compute cap:     (10, 0)  ->  sm_100          [pro6000 was sm_120]
NVIDIA driver:   580.95.05
nvidia-smi CUDA: 13.0
system nvcc:     13.1.80  (/usr/local/cuda -> cuda-13.1; NO 12.x toolkit)
gcc:             13.3.0
CPU / RAM:       72 cores (2x AMD EPYC 9365, 2 NUMA nodes 0-35 / 36-71) / 2263 GB
conda:           26.7.1 at /NHNHOME/data/miniconda3   (envs/ is EMPTY)
repo root:       /NHNHOME/data/hyelin/BEVDiff
scratch/data FS: /NHNHOME/data  (lustre, 160T free)
network:         pypi / download.pytorch.org / github / huggingface / anaconda.org all reachable
```

## The four deltas from the pro6000 build

| | pro6000 | this B200 box | consequence |
|---|---|---|---|
| GPU arch | sm_120 | **sm_100** | `TORCH_CUDA_ARCH_LIST="10.0+PTX"` everywhere |
| system nvcc | 12.8.61 | **13.1 only** | must supply CUDA 12.8 ourselves (see below) |
| donor env | `hiwon-nvidia` clonable | **no envs exist** | full from-scratch build, incl. source mmcv |
| `nvidia-smi topo -p2p n` | NS for every pair | **OK for every pair** | `NCCL_P2P_DISABLE=1` must be **removed**, not added |

The P2P one is the easy mistake: `train_only_seg.sh` was already edited on the
pro6000 box to force `NCCL_P2P_DISABLE=1`, and that edit is committed. On B200
(NVLink/NVSwitch) leaving it set does not hang — it silently routes all-reduce
through host memory and throws away most of the interconnect. It has to be
reverted here.

## The CUDA 12.8 problem

`torch==2.7.1+cu128` is the proven torch for this stack. Building any C++/CUDA
extension against it requires an `nvcc` whose major version matches
`torch.version.cuda`; `torch.utils.cpp_extension` errors out on 13.1 vs 12.8.
This box has only CUDA 13.1.

Two ways out:

**A (recommended).** Install a CUDA 12.8 toolkit *into the conda env* and point
`CUDA_HOME` at it. Keeps the entire pro6000 recipe — torch 2.7.1+cu128,
mmcv-full 1.7.2, mmdet 2.28.2, and `mmdet3d-0.17.1-pro6000.patch` — valid
verbatim, with only the arch list changed. All the C++ porting work in that
patch was validated against exactly this toolchain.

```bash
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit=12.8
export CUDA_HOME="$CONDA_PREFIX"
```

**B.** Jump to `torch 2.9+cu130` and use the system nvcc 13.1. Cleaner on paper,
but CUDA 13 dropped headers and default-arch behaviour that mmcv 1.7.2 and
mmdet3d 0.17.1 still assume, and the existing patch is unvalidated there. That
is a second porting project on top of this one — not worth it unless A fails.

Go with A. Fall back to B only if the 12.8 toolkit cannot be made to build.

## Phases

Each phase ends in a check that must pass before moving on.

### Phase 0 — repo prerequisites

`mmdetection3d/` and `TPVFormer/` are gitlinks with empty working trees.
`.gitmodules` is present, so:

```bash
git submodule update --init --recursive
# expected: mmdetection3d f1107977..., TPVFormer 459bc060...
```

`TPVFormer` is referenced nowhere in the code — a placeholder. `mmdet3d` is
required and must end up editable-installed from this checkout.

Then apply the port patch to the fresh v0.17.1 tree:

```bash
git -C mmdetection3d apply ../mmdet3d-0.17.1-pro6000.patch
```

The patch is arch-agnostic — it is a torch-1.x → torch-2.x C++ port plus three
Python fixes (mmcv version gate, `numba.errors` → `numba.core.errors`,
`force=True` on the spconv `CONV_LAYERS` registrations, which mmcv 1.7.2 now
also registers). Nothing in it is sm_120-specific, so it carries over unchanged.

**Check:** both submodule dirs non-empty; `git -C mmdetection3d diff --stat`
shows the 33 patched files.

### Phase 1 — env skeleton + torch

```bash
conda create -y -n jhl-bevdiff-b200 python=3.10
conda activate jhl-bevdiff-b200
conda install -y -c nvidia/label/cuda-12.8.1 cuda-toolkit=12.8
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
```

**Check — this is the gate for the whole plan:**

```bash
python -c "import torch; print(torch.__version__, torch.version.cuda); \
           print(torch.cuda.get_arch_list()); \
           print(torch.cuda.get_device_capability())"
```

`get_arch_list()` must contain `sm_100`, and capability must read `(10, 0)`. If
`sm_100` is missing from the 2.7.1 wheel, move up to `torch 2.8.x+cu128` — later
cu128 wheels definitely carry it — and re-verify before building anything.
Also confirm `nvcc --version` now reports 12.8 with `CUDA_HOME=$CONDA_PREFIX`.

### Phase 2 — build environment variables

Every native build in phases 3-5 uses:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="10.0+PTX"     # sm_100, NOT the pro6000's 12.0
export MAX_JOBS=32                          # 72 cores; leave headroom
```

### Phase 3 — mmcv-full 1.7.2 from source

No donor env here, so unlike the pro6000 build this really must be compiled.
This is the long pole — expect a substantial build with `MAX_JOBS=32`.

```bash
git clone --depth 1 -b v1.7.2 https://github.com/open-mmlab/mmcv.git /NHNHOME/data/hyelin/build/mmcv-1.7.2
pip install /NHNHOME/data/hyelin/build/mmcv-1.7.2 --no-build-isolation
```

Build under `/NHNHOME/data/hyelin/build`, not `/tmp` — the pro6000 build used
`/tmp/mmcvbuild` and the artifact path leaked into `requirements-pro6000.txt`.

**Check:**

```bash
cuobjdump --list-elf "$CONDA_PREFIX"/lib/python3.10/site-packages/mmcv/_ext*.so | grep -c sm_100
python -c "from mmcv.ops import MultiScaleDeformableAttention, nms; print('ok')"
```

`sm_100` count must be non-zero. This is the op BEVFormer cannot run without.

### Phase 4 — OpenMMLab python layer

```bash
pip install mmdet==2.28.2 mmsegmentation==0.30.0
pip install numpy==1.23.5 scipy==1.10.1 numba==0.57.1 \
            scikit-image==0.19.3 networkx==3.4.2 trimesh==2.35.39 \
            opencv-python==4.8.1.78 nuscenes-devkit==1.1.10 timm==0.6.13 \
            yapf==0.40.1
```

Install these **before** mmdet3d: mmdet3d's own `requirements/runtime.txt`
still asks for `numpy<1.20` / `numba==0.48.0` on some paths, and while the patch
unpins them, having the correct versions already resolved avoids a downgrade.
Keep `numpy<2` throughout.

### Phase 5 — mmdet3d editable + detectron2

```bash
pip install -e /NHNHOME/data/hyelin/BEVDiff/mmdetection3d --no-build-isolation
```

11 extensions get compiled (spconv, iou3d, voxel, roiaware_pool3d, ball_query,
knn, paconv, group_points, interpolate, furthest_point_sample, gather_points).

detectron2 is not optional: `projects/mmdet3d_plugin/__init__.py` ends with
`from .dd3d import *`, and dd3d imports `detectron2.layers` unconditionally.

```bash
git clone https://github.com/facebookresearch/detectron2 /NHNHOME/data/hyelin/detectron2
git -C /NHNHOME/data/hyelin/detectron2 checkout a2f4a8771ab77e8411c26b27f24f9489a28a2453
pip install -e /NHNHOME/data/hyelin/detectron2 --no-build-isolation
```

BEVFormer itself contains no `.cu`/`.cpp` — verified again on this checkout. The
`voxel_pooling_train` / `voxel_pooling_inference` ops the upgrade plan lists do
not exist in this branch. mmcv + mmdet3d + detectron2 is the complete set.

**Check:** every one of the 11 `.so` files plus `detectron2._C` reports `sm_100`
under `cuobjdump`, and each op runs on GPU (Voxelization, `furthest_point_sample`,
`iou3d.nms_gpu`).

### Phase 6 — HF / diffusion stack

Take the as-built numbers, **not** the upgrade plan's. The plan pins
`transformers==4.29.2` / `diffusers==0.20.0`; `projects/bevdiffuser/fm_feature.py`
imports `Dinov2Model`, which lands in transformers 4.31, so the plan's pin fails
at import. Those numbers were copied from upstream BEVDiffuser and never ran
this branch.

```bash
pip install transformers==4.46.3 diffusers==0.35.1 tokenizers==0.20.3 \
            huggingface-hub==0.34.4 safetensors==0.5.3 datasets==3.1.0 \
            accelerate==0.20.3 wandb==0.15.8 tensorboard==2.14.0 \
            einops ftfy Jinja2 tabulate
```

Skip `flash_attn` even though it appears in `requirements-pro6000.txt` — grep
confirms nothing in `BEVFormer/` imports it (nor xformers, deepspeed, or apex).
It rode in from the cloned donor env, and building it for sm_100 is a long
compile for nothing. Same for the dead `ldm/` deps the as-built already dropped:
`pytorch_lightning`, `taming`, `clip`, `kornia`, `albumentations`.

### Phase 7 — launcher fixes for this machine

`projects/bevdiffuser/*.sh` are written for the A6000 box and then partly
re-tuned for the pro6000. Three things are wrong here in every one of them:

1. **NCCL.** Remove `NCCL_P2P_DISABLE=1` (currently forced in
   `train_only_seg.sh`). P2P works on all 28 pairs here; disabling it costs
   most of the interconnect. `NCCL_IB_DISABLE=1` — re-check whether this box has
   IB before keeping it.
2. **`taskset -c 16-31,48-63`.** Sized for a 256-core box. This one has 72 cores
   across two NUMA nodes (0-35 / 36-71); that mask straddles both nodes, which is
   the worst case for a dataloader. Either drop the `taskset` or pin to one node.
3. **`TCNN_CUDA_ARCHITECTURES=120`** → `100` (cosmetic; nothing imports
   tiny-cuda-nn).

Also revisit `CUDA_VISIBLE_DEVICES=4,5,6,7` and `GPUS=4` — hardcoded because
GPUs 0-3 were busy on the old server. All 8 B200s are idle here. And with 183 GB
per GPU vs 48 GB, `TRAIN_BATCH_SIZE=2` is leaving most of the card unused; worth
re-tuning once a run is stable, keeping in mind that changing effective batch
size changes the optimisation, not just the throughput.

Scripts still needing the NCCL/taskset pass: `train_seg_da3.sh`,
`train_original.sh`, `train_seg_v2.sh`, `train_seg.sh`, `train_seg_v3.sh`,
`train_dino.sh`, `train_only_seg_v2.sh`, and the `test_*.sh` set.

### Phase 8 — smoke tests

Mirror the pro6000 as-built's verified list:

```text
all packages import at pinned versions
mmcv MultiScaleDeformableAttention + nms on GPU
mmdet3d Voxelization / furthest_point_sample / iou3d nms_gpu on GPU
sm_100 in mmcv._ext, all 11 mmdet3d ops, detectron2._C
full plugin import incl. dd3d/detectron2
BEVFormer built from bev_tiny_onlyseg_sam_v2_da3.py  (expect 33.5M params)
DiffusionUNetModel from the same config                (expect 449.2M params)
train_bev_diffuser_only_seg.py imports cleanly
torchrun 2-rank NCCL all-reduce — WITH P2P ENABLED this time
```

Then freeze: `pip freeze > requirements-b200.txt`.

## Data and checkpoints — the real blocker

The env is maybe a day of work. The assets are the longer pole, and two of them
have to be *generated*, not copied.

Config `bev_tiny_onlyseg_sam_v2_da3.py` resolves paths relative to
`projects/bevdiffuser/`, i.e. it wants `BEVFormer/data/{nuscenes,nuscenes_sam3,nuscenes_depth_da3}`.
`BEVFormer/data/nuscenes/` exists but is **empty**, and `BEVFormer/ckpts/` does
not exist at all. Both are gitignored, so nothing arrives with the clone.

| asset | on this server | action |
|---|---|---|
| raw nuScenes | `/NHNHOME/data/Dataset/nuscenes` — samples, sweeps, maps, can_bus, v1.0-trainval/test/mini all present | symlink into `BEVFormer/data/nuscenes` |
| SAM3 semantic maps | `/NHNHOME/data/Dataset/nuscenes_sam3` — 148 GB, all 6 CAM dirs | symlink; **but a stray `.…_mask.bin.eKzReh` temp file suggests a copy still in flight — confirm it finished before trusting it** |
| DA3 depth maps | `/NHNHOME/data/Dataset/nuscenes_depth_da3` — **empty directory** | must be generated (`load_depth.py` / the DA3 pipeline) |
| `nuscenes_infos_temporal_{train,val}.pkl` | **absent** — no `.pkl` anywhere under the dataset root | regenerate with `tools/create_data.py`; needs the env from phase 5 |
| `ckpts/bevformer_tiny_epoch_24.pth` | **absent** | fetch from BEVFormer releases or copy from the old server |

The temporal infos depend on a working mmdet3d, so they cannot be produced
before phase 5 — but they are independent of phases 6-7 and can run in parallel
with those. DA3 depth generation is the one item with no stated recipe in any of
the three source documents; it needs its own scoping.

## Order of work

```text
0. submodules + patch          — minutes, no dependencies
1. env + torch, VERIFY sm_100  — gate; everything downstream assumes it
2-5. native builds             — mmcv is the long compile
6. HF stack                    — quick, pip only
7. launcher fixes              — minutes; do NOT skip the NCCL revert
8. smoke tests                 — freeze to requirements-b200.txt

in parallel from phase 5:
   temporal infos via create_data.py
   confirm the sam3 copy completed
   locate bevformer_tiny_epoch_24.pth
   scope DA3 depth generation      <- the one genuinely open item
```
