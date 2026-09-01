# jhl-bevdiff-pro6000 Environment Upgrade Plan

This document summarizes the target virtual environment for running BEVDiffV2 on
the NVIDIA RTX PRO 6000 Blackwell server.

The plan is based on:

- Current BEVDiffV2 environment: `jhl-bevdiff`
- Target server GPU: NVIDIA RTX PRO 6000 Blackwell
- Target server driver: `580.126.20`
- Target server `nvidia-smi` CUDA version: `13.0`
- Confirmed GPU compute capability:

```bash
python -c "import torch; print(torch.cuda.get_device_capability())"
# (12, 0)
```

Therefore the target GPU architecture is:

```text
sm_120
```

## Overall Strategy

BEVDiffV2 is based on the BEVFormer/OpenMMLab 1.x stack. The code depends heavily
on APIs such as:

- `mmcv.runner`
- `mmcv.parallel`
- `mmcv.ops`
- `mmdet.core`
- BEVFormer `MultiScaleDeformableAttention`

Because of this, the recommended strategy is not to migrate to OpenMMLab 2.x
(`mmcv 2.x`, `mmengine`, `mmdet 3.x`, `mmdet3d 1.x`) as the first step.

Instead, follow the same style as the ViDAR/H200 forward-port:

```text
Keep OpenMMLab 1.x.
Upgrade PyTorch/CUDA for the new GPU.
Source-build CUDA extensions for sm_120.
Patch version gates where needed.
```

## Target Environment

```text
env name: jhl-bevdiff-pro6000

python==3.10.20

GPU: NVIDIA RTX PRO 6000 Blackwell
GPU arch: sm_120
NVIDIA Driver: 580.126.20
nvidia-smi CUDA: 13.0
CUDA Toolkit / nvcc: 12.8 series recommended
```

Recommended CUDA architecture list:

```bash
export TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0+PTX"
```

If the environment will be used only on the RTX PRO 6000 Blackwell server, this
can be simplified to:

```bash
export TORCH_CUDA_ARCH_LIST="12.0+PTX"
```

## PyTorch Stack

The old environment uses `torch==1.10.0+cu111`, which is not suitable for
Blackwell.

Target versions:

```text
torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128
```

Reason:

- PyTorch CUDA 12.8 wheels are available from `torch==2.7.0+cu128`.
- `torch==2.7.1+cu128` is a practical minimum target for Blackwell support.
- The server driver supports CUDA 13.0, so CUDA 12.8 runtime wheels are usable.
- CUDA 13.0 PyTorch wheels require newer PyTorch versions and would increase
  OpenMMLab 1.x compatibility risk.

## OpenMMLab Stack

Target versions:

```text
mmcv-full==1.7.2
mmdet==2.28.2
mmsegmentation==0.30.0
mmdet3d==0.17.1 editable, using the local repo
```

Important notes:

- `mmcv-full==1.7.2` must be source-built.
- Do not use a prebuilt `mmcv-full` wheel for the target GPU.
- BEVFormer depends on CUDA ops inside `mmcv._ext`, especially
  `MultiScaleDeformableAttention`.
- The built `mmcv._ext` must contain `sm_120` support.

The local `mmdet3d` package should continue to come from:

```text
/rhome/hyelin/projects/BEVDiffV2/mmdetection3d
```

The current `mmdet3d==0.17.1` code has an MMCV version gate around
`mmcv<=1.4.0`. To use `mmcv-full==1.7.2`, the version check needs to be raised,
similar to the ViDAR/H200 forward-port.

## Diffusion, HF, and Logging Packages

Requested package set:

```bash
pip install accelerate==0.20.3 diffusers==0.20.0 transformers==4.29.2 wandb==0.15.8 datasets ftfy tensorboard Jinja2 tabulate scipy yapf==0.40.1
```

Recommended pinned versions:

```text
accelerate==0.20.3
diffusers==0.20.0
transformers==4.29.2
wandb==0.15.8
datasets==2.14.7
ftfy
tensorboard
Jinja2
tabulate
yapf==0.40.1
```

Additional compatibility pins:

```text
huggingface-hub==0.20.3
tokenizers<0.14
safetensors>=0.3.1
```

Reason:

- `datasets` without a version pin may pull newer `huggingface-hub`.
- Very new `huggingface-hub` versions may not match older
  `diffusers==0.20.0` and `transformers==4.29.2` cleanly.

## Numeric, Image, and Utility Packages

Target versions:

```text
numpy==1.23.5
scipy==1.10.1
opencv-python-headless==4.8.1.78
numba==0.57.x
yapf==0.40.1
```

Notes:

- Use `opencv-python-headless` instead of `opencv-python` on the server.
- This avoids `ImportError: libGL.so.1` on headless systems.
- Keep `numpy<2`.
- `numpy==1.23.5` is a safe target for the PyTorch/OpenMMLab/numba stack.
- `numba==0.48.0` from the old environment is not suitable for Python 3.10.

## CUDA Extensions That Must Be Rebuilt

The old environment contains Python 3.8 and CUDA 11.1 build artifacts. These
cannot be reused in the Python 3.10 / PyTorch 2.7 / CUDA 12.8 / sm_120
environment.

Rebuild the following:

```text
mmcv._ext
mmdet3d.ops.*
BEVFormer/projects/mmdet3d_plugin/bevdepth/ops/voxel_pooling_train
BEVFormer/projects/mmdet3d_plugin/bevdepth/ops/voxel_pooling_inference
detectron2._C, if dd3d is used
```

Known existing BEVDiffV2 custom ops:

```text
mmdetection3d/mmdet3d/ops/spconv
mmdetection3d/mmdet3d/ops/iou3d
mmdetection3d/mmdet3d/ops/voxel
mmdetection3d/mmdet3d/ops/roiaware_pool3d
mmdetection3d/mmdet3d/ops/ball_query
mmdetection3d/mmdet3d/ops/knn
mmdetection3d/mmdet3d/ops/paconv
mmdetection3d/mmdet3d/ops/group_points
mmdetection3d/mmdet3d/ops/interpolate
mmdetection3d/mmdet3d/ops/furthest_point_sample
mmdetection3d/mmdet3d/ops/gather_points
BEVFormer/projects/mmdet3d_plugin/bevdepth/ops/voxel_pooling_train
BEVFormer/projects/mmdet3d_plugin/bevdepth/ops/voxel_pooling_inference
```

## Current to Target Version Changes

| Package | Current `jhl-bevdiff` | Target `jhl-bevdiff-pro6000` |
|---|---:|---:|
| Python | `3.8.20` | `3.10.20` |
| torch | `1.10.0+cu111` | `2.7.1+cu128` |
| torchvision | `0.11.1+cu111` | `0.22.1+cu128` |
| torchaudio | `0.10.0+cu111` | `2.7.1+cu128` |
| mmcv-full | `1.4.0` | `1.7.2`, source-built |
| mmdet | `2.14.0` | `2.28.2` |
| mmdet3d | `0.17.1`, editable | `0.17.1`, editable |
| mmsegmentation | `0.14.1` | `0.30.0` |
| numpy | `1.19.5` | `1.23.5` |
| numba | `0.48.0` | `0.57.x` |
| scipy | `1.10.1` | `1.10.1` |
| opencv | `opencv-python==4.13.0.92` | `opencv-python-headless==4.8.1.78` |
| accelerate | `0.20.3` | `0.20.3` |
| diffusers | `0.35.1` | `0.20.0` |
| transformers | `4.46.3` | `4.29.2` |
| wandb | `0.15.8` | `0.15.8` |
| yapf | `0.40.1` | `0.40.1` |

## Source Build Requirements

Recommended build environment variables:

```bash
export CUDA_HOME="$CONDA_PREFIX"
export FORCE_CUDA=1
export MMCV_WITH_OPS=1
export TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0+PTX"
export MAX_JOBS=32
```

For `mmcv-full==1.7.2`, build from source:

```bash
git clone --depth 1 -b v1.7.2 https://github.com/open-mmlab/mmcv.git ~/workspace/mmcv-1.7.2
cd ~/workspace/mmcv-1.7.2
MMCV_WITH_OPS=1 FORCE_CUDA=1 TORCH_CUDA_ARCH_LIST="8.6;9.0;12.0+PTX" MAX_JOBS=32 \
  pip install . --no-build-isolation
```

After building, verify that `sm_120` code is present:

```bash
cuobjdump --list-elf "$CONDA_PREFIX/lib/python3.10/site-packages/mmcv/_ext"*.so | grep sm_120
```

## Expected Code-Side Compatibility Notes

This document is focused on the environment, but the following code-side patches
are likely required after the environment is created:

- Raise the local `mmdet3d` MMCV version gate to allow `mmcv-full==1.7.2`.
- Update `numba.errors` imports to `numba.core.errors` if present.
- Ensure scripts accept both `--local_rank` and `--local-rank`.
- Prefer `torchrun` over `python -m torch.distributed.launch`.
- Rebuild all CUDA extensions with `sm_120`.
- Check any Torch 2.x C++ extension issues:
  - old `.data<T>()` usage
  - old `.type().is_cuda()` usage
  - deprecated THC includes
  - C++ standard lower than C++17

## Final Recommended Package Set

```text
python==3.10.20

torch==2.7.1+cu128
torchvision==0.22.1+cu128
torchaudio==2.7.1+cu128

mmcv-full==1.7.2
mmdet==2.28.2
mmsegmentation==0.30.0
mmdet3d==0.17.1 editable

accelerate==0.20.3
diffusers==0.20.0
transformers==4.29.2
wandb==0.15.8
datasets==2.14.7
huggingface-hub==0.20.3
tokenizers<0.14
safetensors>=0.3.1
ftfy
tensorboard
Jinja2
tabulate

numpy==1.23.5
scipy==1.10.1
opencv-python-headless==4.8.1.78
numba==0.57.x
yapf==0.40.1
```

## Short Summary

Use `torch==2.7.1+cu128` for Blackwell support, keep the OpenMMLab 1.x style
stack, upgrade to `mmcv-full==1.7.2`, `mmdet==2.28.2`, and
`mmsegmentation==0.30.0`, keep local `mmdet3d==0.17.1` editable, and source-build
all CUDA extensions with `sm_120` included.
