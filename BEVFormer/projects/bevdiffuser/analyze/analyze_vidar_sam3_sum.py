"""Parameter-free ViDAR + observed SAM3 BEV target and descriptive diagnostics.

This script uses saved visualization arrays. It does not run either encoder, and
does not change a Stage 1 or Stage 2 training path.
"""

from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


REPO = Path(__file__).resolve().parents[3]
DEFAULT_VIDAR = REPO / "results/visualize/bev_vidar_vs_bevformer"
DEFAULT_SAM = REPO / "results/visualize/sam3_feature_bev_200x200"
DEFAULT_OUT = REPO / "results/visualize/vidar_sam3_sum_200x200"
EPS = 1e-8


def config_grid(path: Path) -> tuple[list[float], list[int]]:
    source = path.read_text()
    match = re.search(r"^point_cloud_range\s*=\s*(\[[^\n]+\])", source, re.M)
    if match is None:
        raise ValueError(f"point_cloud_range missing: {path}")
    pc_range = [float(v) for v in ast.literal_eval(match.group(1))]
    grid = []
    for name in ("bev_h_", "bev_w_"):
        found = re.search(rf"^{name}\s*=\s*(\d+)", source, re.M)
        if found is None:
            raise ValueError(f"{name} missing: {path}")
        grid.append(int(found.group(1)))
    return pc_range, grid


def pair_samples(vidar_dir: Path, sam_dir: Path):
    vm = json.loads((vidar_dir / "statistics.json").read_text())
    sm = json.loads((sam_dir / "statistics.json").read_text())
    vi = {int(s["index"]): s for s in vm["samples"]}
    si = {int(s["index"]): s for s in sm["samples"]}
    if len(vi) != len(vm["samples"]) or len(si) != len(sm["samples"]):
        raise ValueError("Duplicate frame index in source metadata")
    if vi.keys() != si.keys() or not vi:
        raise ValueError("ViDAR and SAM3 frame indices differ")
    vrange, vgrid = config_grid(Path(vm["vidar"]["config"]))
    drange, dgrid = config_grid(Path(vm["dataset_config"]))
    srange = [float(x) for x in sm["pc_range"]]
    if vrange != drange or vrange != srange:
        raise ValueError(f"Source pc_range mismatch: {vrange}, {drange}, {srange}")
    if vgrid != dgrid or vgrid != vm["bev_grid"] or vgrid != sm["bev_grid"]:
        raise ValueError("Source BEV grid mismatch")
    if int(vm["channels"]) != 256 or int(sm["taps"]["neck_fpn2"]["channels"]) != 256:
        raise ValueError("This experiment requires two 256-channel features")
    pairs = []
    for index in sorted(vi):
        v, s = vi[index], si[index]
        if v["token"] != s["token"]:
            raise ValueError(f"Frame token mismatch at {index}")
        vp = vidar_dir / v["arrays"]
        sp = sam_dir / "arrays" / f"{index:05d}_neck_fpn2.npz"
        if not vp.is_file() or not sp.is_file():
            raise FileNotFoundError(f"Missing matched arrays: {vp}, {sp}")
        pairs.append((index, v["token"], vp, sp))
    return pairs, vrange, vgrid, vm, sm


def fuse(v: np.ndarray, s: np.ndarray, mask: np.ndarray, alpha: float):
    if v.shape != s.shape or v.ndim != 3 or mask.shape != v.shape[1:]:
        raise ValueError(f"Incompatible shapes: {v.shape}, {s.shape}, {mask.shape}")
    if v.dtype != np.float32 or s.dtype != np.float32:
        raise ValueError("Expected float32 source features")
    if not np.isfinite(v).all() or not np.isfinite(s).all():
        raise ValueError("Non-finite source feature")
    mask = mask.astype(bool, copy=False)
    if not mask.any():
        raise ValueError("SAM3 observation mask is empty; observed RMS undefined")
    # float64 accumulators prevent RMS reduction error without changing target dtype.
    rv = float(np.sqrt(np.mean(np.square(v.astype(np.float64)))))
    rs = float(np.sqrt(np.mean(np.square(s[:, mask].astype(np.float64)))))
    if rv <= EPS or rs <= EPS:
        raise ValueError(f"Degenerate source RMS: ViDAR={rv}, SAM3={rs}")
    vn = (v / np.float32(rv)).astype(np.float32)
    sn = np.zeros_like(s)
    sn[:, mask] = s[:, mask] / np.float32(rs)
    target = (vn + np.float32(alpha) * sn).astype(np.float32)
    if not np.isfinite(target).all():
        raise ValueError("Non-finite fused target")
    outside_error = float(np.max(np.abs(target[:, ~mask] - vn[:, ~mask]))) if (~mask).any() else 0.0
    if outside_error != 0.0:
        raise AssertionError("SAM3 changed unobserved cells")
    return vn, sn, target, rv, rs, outside_error


def cell_norm(x: np.ndarray):
    return np.sqrt(np.sum(np.square(x.astype(np.float64)), axis=0))


def masked_rms(x: np.ndarray, mask: np.ndarray):
    return float(np.sqrt(np.mean(np.square(x[:, mask].astype(np.float64)))))


def seam_stats(x: np.ndarray, mask: np.ndarray):
    """Mean channel L2 jump across mask edge versus observed-observed edge."""
    boundary, internal = [], []
    for axis in (0, 1):
        if axis == 0:
            delta = x[:, 1:, :] - x[:, :-1, :]
            a, b = mask[1:, :], mask[:-1, :]
        else:
            delta = x[:, :, 1:] - x[:, :, :-1]
            a, b = mask[:, 1:], mask[:, :-1]
        jump = cell_norm(delta)
        boundary.append(jump[a ^ b])
        internal.append(jump[a & b])
    boundary = np.concatenate(boundary)
    internal = np.concatenate(internal)
    bmean = float(np.mean(boundary)) if boundary.size else None
    imean = float(np.mean(internal)) if internal.size else None
    return {
        "boundary_edges": int(boundary.size),
        "observed_internal_edges": int(internal.size),
        "boundary_mean_l2": bmean,
        "observed_internal_mean_l2": imean,
        "boundary_to_internal_ratio": bmean / imean if bmean is not None and imean and imean > EPS else None,
    }


def feature_stats(vn, sn, target, mask, alpha, rv, rs, outside_error):
    vnorm, snorm, tnorm = cell_norm(vn), cell_norm(sn), cell_norm(target)
    dot_vs = np.sum(vn[:, mask].astype(np.float64) * sn[:, mask].astype(np.float64), axis=0)
    cos_vs = dot_vs / np.maximum(vnorm[mask] * snorm[mask], EPS)
    dot_vt = np.sum(vn[:, mask].astype(np.float64) * target[:, mask].astype(np.float64), axis=0)
    cos_vt = dot_vt / np.maximum(vnorm[mask] * tnorm[mask], EPS)
    base_seam, target_seam = seam_stats(vn, mask), seam_stats(target, mask)
    return {
        "alpha": alpha,
        "observed_cells": int(mask.sum()),
        "coverage": float(mask.mean()),
        "source_rms": {"vidar_all": rv, "sam3_observed": rs},
        "normalized_rms": {
            "vidar_all": float(np.sqrt(np.mean(np.square(vn.astype(np.float64))))),
            "vidar_observed": masked_rms(vn, mask),
            "sam3_observed": masked_rms(sn, mask),
            "target_all": float(np.sqrt(np.mean(np.square(target.astype(np.float64))))),
            "target_observed": masked_rms(target, mask),
        },
        "mean_observed_cosine_vidar_sam3": float(cos_vs.mean()),
        "mean_observed_cosine_target_vidar": float(cos_vt.mean()),
        "median_observed_cosine_target_vidar": float(np.median(cos_vt)),
        "cancellation_fraction_target_norm_below_vidar": float(np.mean(tnorm[mask] < vnorm[mask])),
        "observed_delta_l2_mean": float(np.mean(cell_norm(target - vn)[mask])),
        "outside_mask_max_abs_error": outside_error,
        "seam_vidar": base_seam,
        "seam_target": target_seam,
        "seam_boundary_jump_delta": target_seam["boundary_mean_l2"] - base_seam["boundary_mean_l2"]
        if target_seam["boundary_mean_l2"] is not None else None,
    }


def pooled_pca(samples, seed=0, cells_per_map=400):
    rng = np.random.default_rng(seed)
    vectors = []
    for maps in samples:
        for x in maps:
            flat = x.reshape(x.shape[0], -1).T
            idx = rng.choice(len(flat), min(cells_per_map, len(flat)), replace=False)
            vectors.append(flat[idx])
    a = np.concatenate(vectors, axis=0).astype(np.float64)
    mean = a.mean(axis=0)
    centered = a - mean
    cov = centered.T @ centered / max(len(centered) - 1, 1)
    values, basis = np.linalg.eigh(cov)
    basis = basis[:, -3:][:, ::-1]
    for i in range(3):
        pivot = np.argmax(np.abs(basis[:, i]))
        if basis[pivot, i] < 0:
            basis[:, i] *= -1
    projections = centered @ basis
    lo = np.percentile(projections, 1, axis=0)
    hi = np.percentile(projections, 99, axis=0)
    return mean, basis, lo, hi, values[-3:][::-1].tolist()


def pca_rgb(x, pca):
    mean, basis, lo, hi, _ = pca
    projected = (x.reshape(x.shape[0], -1).T.astype(np.float64) - mean) @ basis
    rgb = np.clip((projected - lo) / np.maximum(hi - lo, EPS), 0, 1)
    return rgb.reshape(x.shape[1], x.shape[2], 3)


def plot_overview(rows, pca, out):
    fig, axes = plt.subplots(len(rows), 5, figsize=(17, 3.3 * len(rows)), squeeze=False)
    for ri, row in enumerate(rows):
        index, mask, maps, delta = row
        for ci, (label, feature) in enumerate(zip(("ViDAR", "sum 0.25", "sum 0.5", "sum 1.0"), maps)):
            axes[ri, ci].imshow(pca_rgb(feature, pca), origin="lower")
            axes[ri, ci].set_title(f"{index:05d}: {label}")
            axes[ri, ci].axis("off")
        ax = axes[ri, 4]
        ax.imshow(delta, origin="lower", cmap="magma", vmin=0, vmax=16)
        ax.contour(mask.astype(float), levels=[0.5], colors="cyan", linewidths=0.4, origin="lower")
        ax.set_title(f"|sum 0.5 - ViDAR|; observed {mask.mean():.1%}")
        ax.axis("off")
    fig.suptitle("Shared PCA basis and RGB scale across frames; final heatmap is channel L2", fontsize=13, y=0.997)
    fig.tight_layout(rect=(0, 0, 1, 0.975))
    fig.savefig(out, dpi=150)
    plt.close(fig)


def self_test():
    rng = np.random.default_rng(3)
    v = rng.standard_normal((4, 5, 6)).astype(np.float32)
    s = rng.standard_normal((4, 5, 6)).astype(np.float32)
    mask = rng.random((5, 6)) > 0.4
    vn, sn, t, rv, rs, err = fuse(v, s, mask, 0.5)
    assert t.shape == v.shape and t.dtype == np.float32 and err == 0.0
    np.testing.assert_array_equal(t[:, ~mask], vn[:, ~mask])
    np.testing.assert_allclose(t[:, mask], vn[:, mask] + 0.5 * sn[:, mask], rtol=1e-6)
    assert np.isclose(masked_rms(sn, mask), 1) and rv > 0 and rs > 0
    assert seam_stats(t, mask)["boundary_edges"] > 0
    try:
        fuse(v, s, np.zeros_like(mask), 0.5)
    except ValueError:
        pass
    else:
        raise AssertionError("Empty mask was accepted")
    print("self-test passed")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--vidar-dir", type=Path, default=DEFAULT_VIDAR)
    parser.add_argument("--sam-dir", type=Path, default=DEFAULT_SAM)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--main-alpha", type=float, default=0.5)
    parser.add_argument("--alphas", type=float, nargs="+", default=[0.0, 0.25, 0.5, 1.0])
    parser.add_argument("--pca-seed", type=int, default=0)
    parser.add_argument("--self-test", action="store_true")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    if args.output_dir.exists():
        raise FileExistsError(f"Output exists; choose a new directory: {args.output_dir}")
    if args.main_alpha not in args.alphas or min(args.alphas) < 0:
        raise ValueError("main-alpha must appear in alphas, and all alphas must be nonnegative")
    pairs, pc_range, grid, vm, sm = pair_samples(args.vidar_dir, args.sam_dir)
    args.output_dir.mkdir(parents=True)
    (args.output_dir / "arrays").mkdir()
    rows, records = [], []
    for index, token, vp, sp in pairs:
        with np.load(vp) as vfile, np.load(sp) as sfile:
            v = vfile["vidar_bev"]
            s = sfile["bev"]
            mask = sfile["observed_mask"].astype(bool)
        if v.shape != (256, *grid) or s.shape != v.shape or mask.shape != tuple(grid):
            raise ValueError(f"Source array grid/channel mismatch: {index}")
        if not np.all(s[:, ~mask] == 0):
            raise ValueError(f"Nonzero SAM3 values outside observed mask: {index}")
        maps, by_alpha = [], {}
        for alpha in args.alphas:
            vn, sn, target, rv, rs, outside = fuse(v, s, mask, alpha)
            by_alpha[str(alpha)] = feature_stats(vn, sn, target, mask, alpha, rv, rs, outside)
            maps.append(target)
            if alpha == args.main_alpha:
                path = args.output_dir / "arrays" / f"frame_{index:05d}_{token}_alpha{alpha:g}.npz"
                np.savez_compressed(path, target=target, observed_mask=mask,
                                    alpha=np.float32(alpha), r_vidar=np.float32(rv),
                                    r_sam3=np.float32(rs), token=token,
                                    pc_range=np.asarray(pc_range, dtype=np.float32))
                main_target = target
                saved = str(path.relative_to(args.output_dir))
        rows.append((index, mask, maps, cell_norm(main_target - vn)))
        records.append({"index": index, "token": token, "vidar_input": str(vp),
                        "sam3_input": str(sp), "target": saved, "by_alpha": by_alpha})
    pca = pooled_pca([r[2] for r in rows], args.pca_seed)
    # The overview expects the four standard variants in order.
    if args.alphas == [0.0, 0.25, 0.5, 1.0]:
        plot_overview(rows, pca, args.output_dir / "overview_readable.png")
    summary = {}
    keys = ("coverage", "mean_observed_cosine_vidar_sam3",
            "mean_observed_cosine_target_vidar", "cancellation_fraction_target_norm_below_vidar",
            "observed_delta_l2_mean", "seam_boundary_jump_delta")
    for alpha in args.alphas:
        values = [r["by_alpha"][str(alpha)] for r in records]
        summary[str(alpha)] = {key: {"mean": float(np.mean([v[key] for v in values])),
                                     "min": float(np.min([v[key] for v in values])),
                                     "max": float(np.max([v[key] for v in values]))}
                               for key in keys}
    statistics = {
        "formula": "T = V / RMS_all(V) + alpha * M * S / RMS_observed(S)",
        "main_alpha": args.main_alpha, "alphas": args.alphas,
        "shape_chw": [256, *grid], "dtype": "float32", "pc_range": pc_range,
        "cell_metres_xy": [(pc_range[3] - pc_range[0]) / grid[1],
                            (pc_range[4] - pc_range[1]) / grid[0]],
        "frame_count": len(records), "pca_seed": args.pca_seed,
        "pca_explained_top3_eigenvalues": pca[4],
        "source_vidar_stats": str(args.vidar_dir / "statistics.json"),
        "source_sam3_stats": str(args.sam_dir / "statistics.json"),
        "vidar_single_frame": vm.get("can_bus"),
        "sam3_prompt_fused": sm["taps"]["neck_fpn2"]["prompt_fused"],
        "aggregate_across_six_frames": summary, "frames": records,
    }
    (args.output_dir / "statistics.json").write_text(json.dumps(statistics, indent=2) + "\n")
    readme = f"""# ViDAR + observed SAM3 neck BEV, parameter-free sum

This folder contains a **descriptive six-frame pilot**, not a trained diffusion
model or evidence of downstream improvement. Inputs are saved ViDAR and SAM3
`neck_fpn2` feature arrays for exactly matched frame tokens. The neck tap is
prompt-independent (`prompt_fused=false`); the visual ViDAR extraction used one
frame with `prev_bev=None`, so motion evidence is not established here.

Target per frame: `T = V / rV + {args.main_alpha:g} * M * S / rS`, where `rV` is
the scalar RMS of all ViDAR channels/cells in that frame, `rS` is the scalar
RMS of SAM3 channels in observed cells, and `M` is the binary observed mask.
There are no trained parameters. The saved `target` is float32 CHW
`[256, 200, 200]` (`[H,W,C]` after transpose), plus mask/scales/token metadata.
The grid is `{pc_range}`, 0.512 m per XY cell. Unobserved cells are exactly
the normalized ViDAR feature. An empty mask or degenerate RMS is rejected.

`statistics.json` reports each frame and alpha in `{args.alphas}`. Cosine and
cancellation are **same-channel arithmetic diagnostics**, not semantic
alignment or task performance. The seam diagnostic compares mean channel L2
neighbor jumps across the SAM3 observed/unobserved boundary to jumps within
observed regions; its increase from ViDAR alone can flag mask artifacts.
Coverage and ratios are summarized by frame mean/min/max, without treating
pixels as independent samples. `overview_readable.png` uses one pooled PCA basis and
one RGB scale for all six frames and variants; its last column is the channel
L2 change for alpha {args.main_alpha:g} (fixed 0–16 color scale), with the mask
boundary in cyan. PCA RGB is an appearance diagnostic, not semantic labels.

Generated with:

```bash
python BEVFormer/projects/bevdiffuser/analyze_vidar_sam3_sum.py
```

Source arrays and configs are referenced in `statistics.json`. No SAM3
inference, environment changes, or training were performed.
"""
    (args.output_dir / "README.md").write_text(readme)
    print(f"Wrote {len(records)} matched targets and statistics to {args.output_dir}")
    print(f"PCA overview: {(args.output_dir / 'overview_readable.png').exists()}")


if __name__ == "__main__":
    main()
