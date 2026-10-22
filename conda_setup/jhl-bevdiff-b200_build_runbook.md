# jhl-bevdiff (B200) — Build Runbook

Step-by-step build procedure. Companion to `jhl-bevdiff-b200_env_plan.md`
(which holds the analysis and the reasoning behind these choices).

Everything below assumes:

```text
env name:   jhl-bevdiff
env path:   /NHNHOME/data/miniconda3/envs/jhl-bevdiff
repo:       /NHNHOME/data/hyelin/BEVDiff
build dir:  /NHNHOME/data/hyelin/build      (created in step 1)
```

Data assets are deliberately out of scope here — the env is buildable and fully
verifiable without them.

## Pre-flight facts (verified 2026-09-03)

```text
8x B200, sm_100, driver 580.95.05, nvidia-smi CUDA 13.0
system nvcc 13.1.80 only — no CUDA 12.x on this box
gcc 13.3.0  (CUDA 12.8 supports up to gcc 14 — fine)
72 cores / 2263 GB RAM
conda 26.7.1 at /NHNHOME/data/miniconda3, envs/ empty
nvidia-smi topo -p2p n: OK for all pairs
torch-2.7.1+cu128-cp310 wheel:      confirmed available
cuda-toolkit 12.8.1 / cuda-nvcc 12.8.93 on nvidia/label/cuda-12.8.1: confirmed
```

**Two gotchas found during pre-flight:**

1. `/usr/local/cuda/bin` is already on `PATH` and points at CUDA **13.1**. Conda
   activation prepends `$CONDA_PREFIX/bin`, so the env's nvcc wins — but every
   build step below sets `PATH="$CUDA_HOME/bin:$PATH"` explicitly rather than
   relying on that.
2. Anaconda's `defaults` channels have an unaccepted Terms of Service on this
   install, so any plain `conda create` / `conda search` fails with
   `CondaToSNonInteractiveError`. The runbook sidesteps it with
   `--override-channels -c conda-forge`, which is preferable anyway (no Anaconda
   defaults licensing question for org use). If you would rather accept it:
   `conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main`
   (and `.../pkgs/r`).

---

## Step 0 — repo prerequisites

No env needed; can run now.

```bash
cd /NHNHOME/data/hyelin/BEVDiff
git submodule update --init --recursive
git -C mmdetection3d apply /NHNHOME/data/hyelin/BEVDiff/mmdet3d-0.17.1-pro6000.patch
mkdir -p /NHNHOME/data/hyelin/build
```

`mmdetection3d/` and `TPVFormer/` are gitlinks with empty working trees.
`TPVFormer` is a placeholder referenced nowhere in the code; `mmdet3d` is
required. The patch is a torch-1.x → torch-2.x C++ port plus three Python fixes
(mmcv version gate, `numba.errors` → `numba.core.errors`, `force=True` on the
spconv `CONV_LAYERS` registrations) — nothing in it is sm_120-specific, so it
applies unchanged for sm_100.

**GATE 0**

```bash
git -C mmdetection3d rev-parse HEAD     # f1107977dfd26155fc1f83779ee6535d2468f449
git -C mmdetection3d diff --stat | tail -1   # 33 files changed
```

---

## Step 1 — env skeleton + CUDA 12.8 toolkit

```bash
source /NHNHOME/data/miniconda3/etc/profile.d/conda.sh
conda create -y -n jhl-bevdiff --override-channels -c conda-forge python=3.10
conda activate jhl-bevdiff
conda install -y --override-channels -c nvidia/label/cuda-12.8.1 cuda-toolkit=12.8.1
```

This is the crux of the whole build. `torch 2.7.1+cu128` needs an nvcc whose
major version matches `torch.version.cuda`; `torch.utils.cpp_extension` refuses
13.1 against 12.8. Putting the 12.8 toolkit inside the env is what keeps the
entire proven pro6000 recipe — mmcv 1.7.2, mmdet 2.28.2, and the mmdet3d patch —
valid here with only the arch list changed.

**GATE 1**

```bash
"$CONDA_PREFIX/bin/nvcc" --version | tail -2     # must say release 12.8
```

---

## Step 2 — torch, and the sm_100 gate

```bash
pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 \
    --index-url https://download.pytorch.org/whl/cu128
pip install numpy==1.23.5
```

Pin numpy immediately: torch pulls numpy as a dependency and will take 2.x if
left alone, which breaks ABI against everything downstream. Keep `numpy<2` for
the rest of the build.

**GATE 2 — the gate for the whole plan. Do not proceed if this fails.**

```bash
python - <<'PY'
import torch
print(torch.__version__, torch.version.cuda)
print(torch.cuda.get_arch_list())
print(torch.cuda.is_available(), torch.cuda.device_count())
print(torch.cuda.get_device_capability(0), torch.cuda.get_device_name(0))
PY
```

Required: `sm_100` present in `get_arch_list()`, capability `(10, 0)`,
8 devices visible.

If `sm_100` is absent from the 2.7.1 wheel, move up to the newest `2.8.x+cu128`
and re-run this gate before building anything. Do **not** switch to a cu130
wheel — that invalidates the 12.8 toolkit, the patch, and the whole recipe.

---

## Step 3 — build environment

Every native build in steps 4-6 needs these. Written to a file so the values
cannot drift between steps:

```bash
cat > /NHNHOME/data/hyelin/build/env.sh <<'SH'
source /NHNHOME/data/miniconda3/etc/profile.d/conda.sh
conda activate jhl-bevdiff
export CUDA_HOME="$CONDA_PREFIX"
export PATH="$CUDA_HOME/bin:$PATH"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="10.0+PTX"   # B200 = sm_100, NOT the pro6000's 12.0
export MAX_JOBS=32
SH
source /NHNHOME/data/hyelin/build/env.sh
pip install ninja psutil
```

`10.0+PTX` is the single most important line — building with the pro6000's
`12.0` produces `.so` files that load but have no kernel for this GPU.

---

## Step 4 — mmcv-full 1.7.2 from source

Unlike the pro6000 build, there is no donor env to clone here (`envs/` is
empty), so this really must compile. It is the long pole of the build.

```bash
source /NHNHOME/data/hyelin/build/env.sh
git clone --depth 1 -b v1.7.2 https://github.com/open-mmlab/mmcv.git \
    /NHNHOME/data/hyelin/build/mmcv-1.7.2
pip install /NHNHOME/data/hyelin/build/mmcv-1.7.2 --no-build-isolation
```

Build under `/NHNHOME/data/hyelin/build`, not `/tmp` — the pro6000 build used
`/tmp/mmcvbuild` and the path leaked into `requirements-pro6000.txt` as
`mmcv-full @ file:///tmp/mmcvbuild/mmcv-full-1.7.2`, which makes that freeze
non-reinstallable.

`--no-build-isolation` is required so the build sees the already-installed
torch rather than fetching a fresh one.

**GATE 4**

```bash
cuobjdump --list-elf "$CONDA_PREFIX"/lib/python3.10/site-packages/mmcv/_ext*.so | grep -c sm_100
python -c "import mmcv; from mmcv.ops import MultiScaleDeformableAttention, nms; print(mmcv.__version__)"
```

`sm_100` count must be non-zero, version `1.7.2`. `MultiScaleDeformableAttention`
is the op BEVFormer cannot run without — if only this one is missing, the whole
build is worthless, so check it explicitly rather than trusting the import of
`mmcv` alone.

---

## Step 5 — OpenMMLab python layer + numeric pins

```bash
source /NHNHOME/data/hyelin/build/env.sh
pip install mmdet==2.28.2 mmsegmentation==0.30.0
pip install scipy==1.10.1 numba==0.57.1 scikit-image==0.19.3 \
            networkx==3.4.2 trimesh==2.35.39 opencv-python==4.8.1.78 \
            nuscenes-devkit==1.1.10 timm==0.6.13 yapf==0.40.1
```

Install these **before** mmdet3d. mmdet3d 0.17.1's `requirements/runtime.txt`
asks for `numpy<1.20` / `numba==0.48.0`; the patch unpins them, but having the
right versions already resolved avoids a downgrade attempt.

`opencv-python` (not `-headless`): the as-built notes `libGL.so.1` was present on
the pro6000 host and `import cv2` worked. Verify it here too — if `import cv2`
raises `ImportError: libGL.so.1`, swap to
`opencv-python-headless==4.8.1.78`.

**GATE 5**

```bash
python -c "import cv2, mmdet, mmseg, numpy, numba; print(cv2.__version__, mmdet.__version__, mmseg.__version__, numpy.__version__)"
```

Expect `4.8.1 2.28.2 0.30.0 1.23.5`. If numpy is no longer 1.23.5, something
above pulled 2.x — fix before continuing.

---

## Step 6 — mmdet3d editable + detectron2

```bash
source /NHNHOME/data/hyelin/build/env.sh
pip install -e /NHNHOME/data/hyelin/BEVDiff/mmdetection3d --no-build-isolation
```

Compiles 11 extensions: spconv, iou3d, voxel, roiaware_pool3d, ball_query, knn,
paconv, group_points, interpolate, furthest_point_sample, gather_points.

*If the editable install fails on setuptools:* pin `setuptools==69.5.1` and
retry with `--config-settings editable_mode=compat`. Old-style `setup.py`
editable installs are the usual casualty of a modern setuptools.

detectron2 is **not** optional — `projects/mmdet3d_plugin/__init__.py` ends with
`from .dd3d import *`, and dd3d does an unconditional `from detectron2.layers
import ...`:

```bash
git clone https://github.com/facebookresearch/detectron2 /NHNHOME/data/hyelin/detectron2
git -C /NHNHOME/data/hyelin/detectron2 checkout a2f4a8771ab77e8411c26b27f24f9489a28a2453
pip install -e /NHNHOME/data/hyelin/detectron2 --no-build-isolation
```

BEVFormer itself contains no `.cu`/`.cpp` — re-verified on this checkout. The
`voxel_pooling_train` / `voxel_pooling_inference` ops listed in the old upgrade
plan do not exist in this branch. mmcv + mmdet3d + detectron2 is the complete
set of native builds.

**GATE 6**

```bash
source /NHNHOME/data/hyelin/build/env.sh
for so in $(python -c "import mmdet3d,os;print(os.path.dirname(mmdet3d.__file__))")/ops/*/*.so \
          $(python -c "import detectron2,os;print(os.path.dirname(detectron2.__file__))")/*.so; do
  printf '%-60s %s\n' "$(basename $so)" "$(cuobjdump --list-elf $so | grep -c sm_100)"
done
```

Every line must show a non-zero count — 11 mmdet3d ops plus `detectron2._C`.
Then run them, not just import them:

```bash
python - <<'PY'
import torch
from mmdet3d.ops import Voxelization, furthest_point_sample
from mmdet3d.ops.iou3d.iou3d_utils import nms_gpu
pts = torch.rand(1000, 4, device='cuda')
print('voxel  ', [t.shape for t in Voxelization([0.1]*3, [0,0,0,10,10,10], 32)(pts)])
print('fps    ', furthest_point_sample(torch.rand(1,1000,3,device='cuda'), 64).shape)
boxes = torch.tensor([[0.,0,0,2,2,2,0],[0.1,0,0,2,2,2,0]], device='cuda')
print('nms    ', nms_gpu(boxes, torch.tensor([0.9,0.8],device='cuda'), 0.5))
PY
```

---

## Step 7 — HF / diffusion stack

Use the **as-built** numbers, not the old upgrade plan's. That plan pins
`transformers==4.29.2` / `diffusers==0.20.0`, but `projects/bevdiffuser/fm_feature.py`
imports `Dinov2Model`, which only lands in transformers 4.31 — the plan's pin
fails at import. Those numbers came from upstream BEVDiffuser and never ran this
branch.

```bash
source /NHNHOME/data/hyelin/build/env.sh
pip install transformers==4.46.3 diffusers==0.35.1 tokenizers==0.20.3 \
            huggingface-hub==0.34.4 safetensors==0.5.3 datasets==3.1.0 \
            accelerate==0.20.3 wandb==0.15.8 tensorboard==2.14.0 \
            einops ftfy Jinja2 tabulate
python -c "import numpy; assert numpy.__version__=='1.23.5', numpy.__version__"
```

Deliberately skipped:

- `flash_attn` — appears in `requirements-pro6000.txt` but grep confirms nothing
  under `BEVFormer/` imports it (nor xformers, deepspeed, apex). It rode in from
  the cloned donor env; building it for sm_100 is a long compile for nothing.
- `pytorch_lightning`, `taming`, `clip`, `kornia`, `albumentations` — dead deps
  reachable only from unused `ldm/` files, absent from the original A6000 env too.

The trailing numpy assert is there because `datasets`/`transformers` are the most
likely packages to drag numpy 2.x back in.

---

## Step 8 — full verification

Mirror of the pro6000 as-built's verified list, with the P2P expectation
inverted for this machine.

```bash
source /NHNHOME/data/hyelin/build/env.sh
cd /NHNHOME/data/hyelin/BEVDiff/BEVFormer
export PYTHONPATH="$PWD:$PYTHONPATH"

# 1. full plugin import, incl. dd3d/detectron2
python -c "import projects.mmdet3d_plugin; print('plugin ok')"

# 2. build both models from the target config
python - <<'PY'
from mmcv import Config
from mmdet3d.models import build_model
cfg = Config.fromfile('projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_da3.py')
import projects.mmdet3d_plugin  # noqa
m = build_model(cfg.model).cuda()
print('BEVFormer params %.1fM' % (sum(p.numel() for p in m.parameters())/1e6))
PY
```

Expected from the pro6000 as-built: BEVFormer **33.5M** params,
DiffusionUNetModel **449.2M** params. A different number means a config or
version drift, not just a build difference — worth chasing.

```bash
# 3. NCCL — 2-rank all-reduce WITH P2P ENABLED (opposite of the pro6000 box)
cd /NHNHOME/data/hyelin/build
cat > nccl_check.py <<'PY'
import os, torch, torch.distributed as dist
dist.init_process_group('nccl')
r = dist.get_rank(); torch.cuda.set_device(r)
t = torch.ones(1, device='cuda') * (r + 1)
dist.all_reduce(t)
print('rank', r, '->', t.item(), '(expect 3.0)')
PY
torchrun --nproc_per_node 2 nccl_check.py
```

`nvidia-smi topo -p2p n` reports OK for all 28 pairs here, so this must complete
immediately **without** `NCCL_P2P_DISABLE`. If it hangs, that is a real finding
about this machine, not something to paper over with the pro6000 workaround.

```bash
# 4. training entrypoint imports
cd /NHNHOME/data/hyelin/BEVDiff/BEVFormer
python -c "import ast,sys; ast.parse(open('projects/bevdiffuser/train_bev_diffuser_only_seg.py').read()); print('parsed')"
PYTHONPATH="$PWD:$PYTHONPATH" python -c "
import importlib.util as u
s=u.spec_from_file_location('t','projects/bevdiffuser/train_bev_diffuser_only_seg.py')
m=u.module_from_spec(s); s.loader.exec_module(m)" 2>&1 | tail -5
```

The entrypoint import is the last thing that can be checked without data — it
exercises the diffusers/transformers stack including `Dinov2Model`.

```bash
# 5. freeze
pip freeze > /NHNHOME/data/hyelin/BEVDiff/requirements-b200.txt
```

---

## Step 9 — launcher fixes (env-adjacent, no data needed)

`projects/bevdiffuser/*.sh` were written for the A6000 box, then partly re-tuned
for the pro6000. Three things are wrong on B200 in every one of them:

1. **`NCCL_P2P_DISABLE=1` must be removed.** It is currently forced in
   `train_only_seg.sh` and committed. On this machine P2P works on all pairs;
   leaving it set does not hang, it silently routes all-reduce through host
   memory and throws away most of the interconnect. This is the easiest fix to
   miss because nothing visibly breaks.
2. **`taskset -c 16-31,48-63`** — sized for a 256-core box. This one has 72 cores
   over two NUMA nodes (0-35 / 36-71), so that mask straddles both nodes, the
   worst case for a dataloader. Drop the `taskset` or pin within one node.
3. **`TCNN_CUDA_ARCHITECTURES=120`** → `100`. Cosmetic; nothing imports
   tiny-cuda-nn.

Also revisit `CUDA_VISIBLE_DEVICES=4,5,6,7` / `GPUS=4` (hardcoded because GPUs
0-3 were busy on the old server; all 8 B200s are idle here), and note that
`NCCL_IB_DISABLE=1` may now be wrong — `/opt/mellanox/doca/tools` is on `PATH`,
so this host may actually have InfiniBand. Worth checking before keeping it.

`TRAIN_BATCH_SIZE=2` against 183 GB/GPU (vs 48 GB on the A6000) leaves most of
the card unused, but changing it changes the optimisation and not just
throughput — that is a training decision for after the first run reproduces.

Scripts still needing this pass: `train_seg_da3.sh`, `train_original.sh`,
`train_seg_v2.sh`, `train_seg.sh`, `train_seg_v3.sh`, `train_dino.sh`,
`train_only_seg_v2.sh`, and the `test_*.sh` set.

---

## Summary of gates

| step | gate | non-negotiable? |
|---|---|---|
| 0 | submodules populated, 33 files patched | yes |
| 1 | `$CONDA_PREFIX/bin/nvcc` reports 12.8 | yes |
| 2 | `sm_100` in `torch.cuda.get_arch_list()` | **hard stop** |
| 4 | `sm_100` in `mmcv/_ext*.so`, MSDeformAttn imports | **hard stop** |
| 5 | numpy still 1.23.5, `import cv2` works | yes |
| 6 | `sm_100` in 11 ops + `detectron2._C`, ops run on GPU | **hard stop** |
| 7 | numpy still 1.23.5 after HF stack | yes |
| 8 | plugin import, 33.5M/449.2M params, NCCL without P2P disable | yes |
