# Semantic BEV Probability Cache

The standard Metric3D/v5 detection config reads pre-embedding semantic BEV
probabilities directly with `np.load(path, allow_pickle=False)`; that legacy
path does not require a manifest. The Occ3D training dataset checks the
manifest's output frame when its explicit expected-frame setting is provided;
the cache root must still be kept distinct.


## Occ3D cache

Occ3D must use a separate cache because its projector covers
`[-40, -40, -1.0, 40, 40, 5.4]`, while the detection cache covers
`[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]`. Both arrays have shape
`[17, 50, 50]`, but their cells represent different physical locations and
the Z filtering performed before XY splatting is irreversible.

The Stage 1 occupancy config
`projects/configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py`
projects raw LiDAR XYZ into ego coordinates before XYZ clipping and splatting.
Its new cache is distinct from the historical LiDAR-frame occupancy cache:

```text
data/semantic_bev_occ_v2
```

Its immutable source identifier is
`semantic_bev_occ_ego`. The resulting manifest must report
`projection.output_frame=ego`, an ego-calibration geometry digest,
`projection.pc_range=[-40,-40,-1.0,40,40,5.4]`,
`projection.bev_h=50`, `projection.bev_w=50`, and
`projection.num_classes=16`.

The prepared launcher reuses the generic builder. From
`BEVFormer/projects/bevdiffuser/`:

```bash
bash build_semantic_bev_cache_occ.sh
```

The launcher defaults to eight visible GPUs (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7`)
and maps them to logical `cuda:0` through `cuda:7`. Override
`CUDA_VISIBLE_DEVICES` with a comma-separated list for fewer physical GPUs;
the builder always uses the logical indices inside that visibility mask.
For example, one or two GPUs and a bounded subset:

```bash
CUDA_VISIBLE_DEVICES=3 bash build_semantic_bev_cache_occ.sh --split train --indices 0 1
CUDA_VISIBLE_DEVICES=2,5 bash build_semantic_bev_cache_occ.sh --split train --indices 0 1
bash build_semantic_bev_cache_occ.sh --wait-for-lock
```

The launcher accepts `PYTHON`, `CONFIG`, `CACHE_ROOT`, `SOURCE_VERSION`,
`SPLIT`, and `FLUSH_EVERY` environment overrides and forwards additional
arguments such as `--indices` or `--help`. The coordinator owns the sole root
lock and cache writer; each worker projects a disjoint assigned index on one
GPU and returns a CPU array. No `torchrun` or distributed process group is used.

Equivalent direct command from `BEVFormer/projects/bevdiffuser/`:

```bash
CUDA_VISIBLE_DEVICES=0 python build_semantic_bev_cache.py \
  --config ../configs/bevdiffuser/bev_onlyseg_sam_v2_metric3d_occ.py \
  --cache-root ../../data/semantic_bev_occ_v2 \
  --source-version semantic_bev_occ_ego \
  --split all \
  --device cuda:0 \
  --flush-every 50
```

For two or eight GPUs use `--devices cuda:0 cuda:1` or
`--devices cuda:0 cuda:1 cuda:2 cuda:3 cuda:4 cuda:5 cuda:6 cuda:7`
instead of `--device cuda:0`, with matching `CUDA_VISIBLE_DEVICES`.

The builder is resumable. It verifies existing entries through the manifest,
uses an exclusive `.builder.lock`, and refuses a populated cache whose contract
does not match the occupancy projector. Existing manifest entries skip raw SAM3
and Metric3D reads. Unmanifested `.npy` files are rejected rather than
overwritten; inspect/reconcile them manually or use a new root. A changed
projection/runtime contract or corrupted array also stops the build. The
builder rejects duplicate dataset tokens or duplicate requested indices rather
than risk replacing geometry from another frame. The
coordinator flushes committed entries on failure or interruption after stopping
workers. Do not point the occupancy config at
`data/semantic_bev` and do not copy that detection manifest into the new root.
The manifest is builder provenance and resume bookkeeping. The occupancy config
checks `projection.output_frame=ego` once at dataset initialization, then uses
the lightweight NumPy read path with shape and floating-dtype checks. Detection
reads remain unchanged.

This cache contains float32 `[17, 50, 50]` bilinear-splat semantic mass in the
ego frame over `[-40, -40, -1.0, 40, 40, 5.4]`. Validation here covers
manifest contract, geometry and checksums for requested entries; it does not
establish CUDA parity between GPUs, cache/live numerical equivalence, or a
downstream occupancy score. The Stage 2 config still points at the historical
`data/semantic_bev_occ` root and must not be used as evidence for this new
teacher cache until explicitly updated and validated.

The bounded comparison runner reads exactly the 24 tokens in the selected CSV;
it never builds the full cache. From any directory:

```bash
/NHNHOME/data/miniconda3/envs/jhl-bevdiff/bin/python \
  BEVFormer/projects/bevdiffuser/visualize_occ_semantic_frame_fix.py \
  --device cpu --threads 2
```

Run this command from the repository root, or pass absolute script and input
paths. It writes new token PNGs, After probability arrays, per-frame and pooled
CSV scores, and runtime/hash provenance under
`results/analysis/occ_semantic_frame_fix_20261015/`; it rejects pre-existing
output artifacts. CPU results can differ numerically from the historical CUDA
cache. The existing Stage 2 teacher and detection/DA3 configs are unchanged.

## Train from existing arrays


The default cache root is `BEVFormer/data/semantic_bev`. Each file is named
`<sample_token>.npy` and contains floating `[17, 50, 50]` probabilities.
From the repository root:

```bash
conda activate jhl-bevdiff
cd BEVFormer/projects/bevdiffuser
bash train_only_seg_cache.sh
```

Cache settings come from
`projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py`:
`use_semantic_bev_cache` and `semantic_bev_cache_root` are propagated by the
config to train/val/test. To select another directory, edit
`semantic_bev_cache_root` in that config. The script does not override it with
command-line options or environment variables.

The config uses `../../data/...` paths relative to
`BEVFormer/projects/bevdiffuser/`. Run the script and direct Python training,
evaluation or cache-builder commands using this config from that directory.

## Data flow

`CustomNuScenesDiffusionDataset_seg_depth_v2` selects the token from the actual
returned current-frame metadata (including temporal queues and resampled
training items). It loads the array, checks the configured shape and floating
dtype, and returns a contiguous float32 tensor as
`data['semantic_bev_probabilities']`. Missing files and invalid shapes/types
raise with the offending path; there is no automatic live fallback.

MMCV collates these tensors into the batch. Training and evaluation pass
`semantic_prob` with shape `[B, 17, 50, 50]` to the v5 UNet on its input device.
The cached branch skips raw SAM3/depth loading and geometric projection.
`prob_to_emb`, learned position embeddings, the condition encoder and the UNet
continue to train. CFG dropout still supplies zero probabilities for the
unconditional input. No argmax or extra normalization is applied, so fractional
semantic mass and zero-valued unknown cells are preserved.

RGB images, annotations and the BEVFormer model are still required. Arrays must
represent the semantic taxonomy and BEV coordinates used by the experiment;
loading does not convert caches between different projection methods or apply
new BEV rotations/flips to them.

## Config and optional offline tools

Dataset settings are `use_semantic_bev_cache`, `semantic_bev_cache_root`, and
`semantic_bev_cache_shape`. The current config derives the expected shape from
the aligner class count and BEV dimensions. Older configs with
`semantic_bev_cache_projection` remain supported for deriving the shape;
`semantic_bev_cache_source_version` is accepted but unused by dataset reads.

For live projection, set `use_semantic_bev_cache=False` in the config. Live mode retains the SAM3/Metric3D
source-path and raw-calibration requirements.

Existing `.npy` files need no regeneration or manifest edits. The optional
`build_semantic_bev_cache.py` writer still uses its own manifest, source version,
checksums and exclusive lock for build/resume bookkeeping. Those checks are
separate from training reads. Set `--source-version` on the builder when the
source maps change; the default is the legacy config value, if present, or
`sam3_metric3d_v1`. Training configs do not need this metadata.
`validate_semantic_bev_cache.py` can still compare
live projection with directly loaded probabilities when raw maps are available.

Regression checks (from `BEVFormer` in the training environment):

```bash
PYTHONPATH=. python -m unittest discover -s tests -p 'test_semantic_bev_npy_dataset.py' -v
```
