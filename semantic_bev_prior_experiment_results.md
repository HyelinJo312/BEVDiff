# Semantic BEV Prior Experiment Results

## 0. Result Source

All numeric experiment results are tracked in this file. The experiment plan should describe table structure and experimental intent only.

| Method | Result Directory | Log Used | Eval Epoch |
|---|---|---|---:|
| Ours (MGD) | `results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_mgd_v9_t100_alpha100_lambda_0.6` | `20260826_115011.log.json` | 24 |
| Ours (MSE) | `results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_da3_no-mgd` | `20260831_110247.log.json` | 24 |

Note: `DiffBEVFormer_tiny_onlyseg_sam_da3_no-mgd` also contains `20260829_153048`, but that log only has an epoch-12 validation result and is not used as the final Ours (MSE) number.

## 1. Main Detection Result

| Method | Teacher Signal | Stage 1 Task Head | Distillation | NDS ↑ | mAP ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
|---|---|---:|---|---:|---:|---:|---:|---:|---:|---:|
| BEVFormer 24e | none | no | none | 35.56 | 25.47 |  |  |  |  |  |
| BEVDiffuser | GT layout | yes | MSE | 38.71 | 28.84 | 0.8707 |  | 0.5801 |  |  |
| BEVDiffuser | GT layout | yes | MGD | 39.42 | 29.26 |  |  |  |  |  |
| Ours | Semantic BEV + SB-FDN | no | MSE | 38.48 | 27.98 | 0.8632 | 0.2841 | 0.5897 | 0.6103 | 0.2035 |
| Ours | Semantic BEV + SB-FDN | no | MGD | 38.90 | 29.26 | 0.8526 | 0.2831 | 0.6127 | 0.6125 | 0.2125 |

## 2. Core Comparisons

### 2-A. Ours vs Baseline

| Comparison | ΔNDS | ΔmAP | Interpretation |
|---|---:|---:|---|
| Ours (MSE) vs BEVFormer 24e | +2.92 | +2.51 | Semantic BEV teacher is useful even with direct MSE |
| Ours (MGD) vs BEVFormer 24e | +3.34 | +3.79 | Full semantic teacher + MGD gives clear downstream transfer gain |

### 2-B. Ours vs Original BEVDiffuser

The main paper comparison should be Ours (MGD) against the original BEVDiffuser setting, not against BEVDiffuser + MGD.

| Comparison | ΔNDS | ΔmAP | ΔmATE | ΔmAOE | Interpretation |
|---|---:|---:|---:|---:|---|
| Ours (MGD) vs BEVDiffuser (MSE) | +0.19 | +0.42 | -0.0181 | +0.0326 | Comparable transfer without GT layout or Stage 1 task head |

Ours improves mATE but has worse mAOE than BEVDiffuser (MSE). The safest claim is therefore BEVDiffuser-level detection transfer, not uniformly better detection quality on every metric.

### 2-C. Strong Control

| Comparison | ΔNDS | ΔmAP | Interpretation |
|---|---:|---:|---|
| Ours (MGD) vs BEVDiffuser (MGD) | -0.52 | +0.00 | Privileged GT-layout teacher remains stronger on NDS when also given MGD |

`BEVDiffuser + MGD` should be presented as a strong loss-control or privileged-condition control. It is not the main baseline for the claim about removing GT layout.

## 3. MGD vs Direct MSE

| Teacher | Teacher Condition | MSE NDS ↑ | MSE mAP ↑ | MGD NDS ↑ | MGD mAP ↑ | MGD Gain |
|---|---|---:|---:|---:|---:|---|
| BEVDiffuser | GT layout | 38.71 | 28.84 | 39.42 | 29.26 | +0.71 NDS / +0.42 mAP |
| Ours | Semantic BEV + SB-FDN | 38.48 | 27.98 | 38.90 | 29.26 | +0.42 NDS / +1.28 mAP |

Interpretation:

- MGD improves both BEVDiffuser and Ours, so it should be framed as a generally useful BEV distillation strategy.
- The mAP gain is much larger for Ours, which supports the claim that MGD is especially helpful when transferring semantic-only teacher features.
- Ours (MSE) is already above the BEVFormer baseline, so the semantic teacher signal is not weak; it is simply harder to exploit with point-wise copying.

## 4. Detailed Ours Results

### 4-A. Ours (MGD)

Source: `results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_mgd_v9_t100_alpha100_lambda_0.6/20260826_115011.log.json`

| Metric | Value |
|---|---:|
| NDS | 0.38897 |
| mAP | 0.29262 |
| mATE | 0.8526 |
| mASE | 0.2831 |
| mAOE | 0.6127 |
| mAVE | 0.6125 |
| mAAE | 0.2125 |

### 4-B. Ours (MSE)

Source: `results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_da3_no-mgd/20260831_110247.log.json`

| Metric | Value |
|---|---:|
| NDS | 0.38482 |
| mAP | 0.27979 |
| mATE | 0.8632 |
| mASE | 0.2841 |
| mAOE | 0.5897 |
| mAVE | 0.6103 |
| mAAE | 0.2035 |

## 5. Stage 1 Diagnostic Result

This result should remain an appendix diagnostic, not a main comparison.

| Method | NDS ↑ | mAP ↑ | Setting |
|---|---:|---:|---|
| BEVDiffuser-tiny | 48.61 | 34.56 | Frozen detection head probe, noise_T=5 |
| onlyseg_sam3_v4 | 35.22 | 25.09 | Frozen detection head probe, noise_T=5 |

Interpretation:

- BEVDiffuser is favored by the frozen detection head probe because it uses detection-specific GT layout and task-head adaptation.
- Ours is a task-head-free semantic teacher, so a frozen detection head can under-read the feature.
- This diagnostic should not be used as the main evidence against or for the proposed teacher formulation.

## 6. Safe Paper Interpretation

The safest conclusion is:

> Ours achieves original BEVDiffuser-level downstream detection transfer while removing GT layout conditioning and Stage 1 task-head supervision from the teacher.

The contribution should be framed as:

- Task-head-free semantic BEV diffusion teacher.
- Removal of privileged GT layout conditioning.
- Robust semantic teacher transfer through MGD.
- Task-agnostic semantic conditioning formulation that can be extended beyond 3D detection.

Avoid claiming:

- New SOTA 3D detection.
- Uniform improvement over BEVDiffuser on every metric.
- Superiority over BEVDiffuser + MGD as the main result.
