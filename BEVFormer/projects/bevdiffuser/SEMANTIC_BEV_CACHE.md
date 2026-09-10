# Local Semantic BEV Probability Cache

The v5 Metric3D path supports both live projection and a local pre-embedding
cache. Production defaults remain live until all required train/validation
tokens have been built and verified.

## What Is Cached

One float32 `[17, 50, 50]` NumPy array per sample token, directly from
`SegBEVAligner.project_semantics`. No argmax, float16 conversion or additional
normalization is applied. Unknown cells remain zero; fractional semantic mass
below one is preserved. Support maps are not needed by training and are not
cached. Approximately 6 GB is needed for train plus validation, excluding
filesystem overhead.

`prob_to_emb`, the learned position embedding, the condition encoder and the
UNet still run and receive gradients on every training step. No model parameter
names or shapes are changed. Existing CFG dropout uses zero probabilities for
the unconditional input; it does not zero the learned embedding.

## Build and Switch

The configured persistent location is now
`BEVFormer/data/semantic_bev_cache/metric3d_v5`, as requested. This directory is
currently on NFS: it avoids raw-map reads and splatting, but not NFS cache reads.
The earlier local-SSD timing results used `/tmp` and are not directly transferable
to this location. For fully local cache I/O, use a local SSD mount or a symlink
to one; `/tmp` itself may be cleaned on reboot.

```bash
cd /rhome/hyelin/projects/BEVDiffV2
conda activate jhl-bevdiff
export CACHE_ROOT="$PWD/BEVFormer/data/semantic_bev_cache/metric3d_v5"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
CUDA_VISIBLE_DEVICES=4 python -u BEVFormer/projects/bevdiffuser/build_semantic_bev_cache.py \
  --cache-root "$CACHE_ROOT" --split all --device cuda
```

GPU 4 is an example; choose an available GPU. The builder creates the directory.
Wait for both train and validation to complete successfully before enabling cache
mode. Re-run the same build command after interruption to verify/resume entries.

The builder uses the current Metric3D config by default. `--config` selects
another config. CUDA is required: CPU/GPU projection arithmetic can differ,
especially with TF32. Use the training environment/GPU type; the manifest records
PyTorch/CUDA versions and the matmul TF32 setting, including `close_tf32`.
`--indices 0 1 2 3 4 5 6 7 --split train` builds a small subset;
it is intentionally insufficient for full training. Re-running the builder
verifies completed entries and builds missing ones. Only one writer is allowed.

If another builder owns `.builder.lock`, the command exits with an explanation
before importing the model or initializing CUDA. A different GPU does not allow
concurrent writes to the same root. Prefer letting the existing builder finish.
To queue a verification/resume run explicitly, add `--wait-for-lock` to the build
command. The waiting process does not allocate GPU memory and can be cancelled
with Ctrl+C without stopping the existing builder. After acquiring the lock, it
verifies completed entries and generates only missing entries.

On the host, `lslocks -o PID,COMMAND,TYPE,MODE,PATH` can identify the lock owner.
Do not delete `.builder.lock` while a builder is running: replacing its inode can
allow two writers to corrupt the manifest. The file can remain after a normal
exit; the OS releases the lock when the owning process closes it or exits.

After both splits are complete:

```bash
USE_SEMANTIC_BEV_CACHE=1 SEMANTIC_BEV_CACHE_ROOT="$CACHE_ROOT" \
  bash BEVFormer/projects/bevdiffuser/train_only_seg.sh
```

Return to live SAM3/Metric3D loading:

```bash
USE_SEMANTIC_BEV_CACHE=0 bash BEVFormer/projects/bevdiffuser/train_only_seg.sh
```

With neither environment variable set, the script uses the config's
`use_semantic_bev_cache` and `semantic_bev_cache_root`. The config propagates
these values to train/val/test. Environment overrides apply to all three splits.
Do not change the flag mid-process: stop/restart or resume at a checkpoint.
The current shell script has its resume option commented out, so these training
commands start a fresh run. To continue an existing run, set its `RESUME_FROM`
checkpoint and enable the script's `--resume_from_checkpoint` argument first.
For a fresh run, choose a distinct `RUN_NAME` to avoid mixing existing outputs.
Visualization tools intentionally keep raw-map mode because they render depth
and support as well as probabilities.

## Invalidation and Failure Behavior

- The manifest records projection settings, raw depth shape/range, semantic ID
  remapping, an explicit source version and conservative preprocessing/projector
  code hashes. Changing these requires a new cache directory.
- Treat `semantic_bev_cache_source_version` as immutable. Increment it if SAM3 or
  Metric3D source contents change, even when file names remain the same. Cache
  readers deliberately do not stat/open the raw maps to check freshness.
- Each read verifies sample geometry, float32 shape, finite nonnegative values
  and the array checksum. Model/dataset projection overrides are checked too.
- Training initialization rejects missing manifest coverage. Missing/corrupt
  files and changed per-sample geometry raise errors; there is no live fallback.
- The current scale/padding geometry is fixed. New stochastic crops/flips/scales
  need a matching augmentation-aware cache strategy; they must not reuse this
  single-geometry cache silently.
- RGB frames, annotations and the frozen BEVFormer teacher are still required.
  This removes raw SAM3/Metric3D loading and splatting, not all data/model costs.

## Sequential Validation

`validate_semantic_bev_cache.py` supports stages 1, 2 and 4. Stage 1 writes tiny
live/cached configs for the same samples; stage 2 checks real UNet gradients and
Accelerator checkpoint resume; stage 4 forbids both raw-map loaders and works
with unavailable source roots. Run stage 4 under `strace -f -e trace=openat` to
audit file opens independently.

Use `profile_stage1.py` with `PROFILE_CONFIG` pointing to each tiny config for
stage 3. It invokes the actual training shell script with a short step limit,
fresh output directory, fixed seed and no checkpoint writes. Per-rank JSONL
records include sample tokens, DataLoader wait and synchronized step timings.
Compare matching tokens, exclude warmup, and report shared-GPU/tiny-hot-subset
limitations. The user-requested CFG/encoder output comparison is omitted.
