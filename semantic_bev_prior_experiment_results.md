# Semantic BEV Prior Experiment Results

## Main Result Table

| Method | Teacher Signal | Distillation | NDS ↑ | mAP ↑ |
|---|---|---|---:|---:|
| BEVFormer 24e | - | - | 35.56 | 25.47 |
| BEVDiffuser | GT layout | MSE | 38.71 | 28.84 |
| BEVDiffuser | GT layout | MGD | 39.42 | 29.26 |
| Ours | Semantic BEV | MSE | 38.46 | 28.02 |
| Ours | Semantic BEV | MGD | 39.03 | 29.09 |

## Key Takeaways

### 1. Semantic BEV teacher provides useful transfer signal

`Ours + MSE` improves clearly over the BEVFormer 24e baseline:

| Comparison | ΔNDS | ΔmAP |
|---|---:|---:|
| Ours + MSE vs BEVFormer 24e | +2.90 | +2.55 |

This indicates that the semantic BEV teacher is not a weak or irrelevant signal. Even with direct MSE, it transfers useful BEV information to the student.

### 2. Direct MSE is not sufficient for semantic-only teacher transfer

Compared with the original BEVDiffuser setting, `Ours + MSE` is lower:

| Comparison | ΔNDS | ΔmAP |
|---|---:|---:|
| Ours + MSE vs BEVDiffuser + MSE | -0.25 | -0.82 |

This suggests that directly copying semantic-only teacher features with point-wise MSE is suboptimal. The semantic BEV teacher is useful, but it is harder to exploit with direct feature matching than the GT-layout BEVDiffuser teacher.

### 3. MGD is important for semantic-only teacher transfer

For our semantic BEV teacher, replacing MSE with MGD gives a large gain:

| Comparison | ΔNDS | ΔmAP |
|---|---:|---:|
| Ours + MGD vs Ours + MSE | +0.57 | +1.07 |

The mAP improvement is especially large. This supports using MGD as the distillation method in the full semantic-only formulation.

### 4. MGD is a general robust distillation method, not a semantic-only trick

BEVDiffuser also benefits from MGD:

| Comparison | ΔNDS | ΔmAP |
|---|---:|---:|
| BEVDiffuser + MGD vs BEVDiffuser + MSE | +0.72 | +0.42 |

Therefore, MGD should be interpreted as a generally useful BEV distillation method. However, the mAP gain is larger for the semantic-only teacher, suggesting that MGD is particularly helpful when transferring semantic BEV teacher features.

### 5. Ours reaches original BEVDiffuser-level transfer without GT layout

The main comparison is the original BEVDiffuser setting against our full method:

| Comparison | ΔNDS | ΔmAP |
|---|---:|---:|
| Ours + MGD vs BEVDiffuser + MSE | +0.32 | +0.25 |

Ours achieves comparable and slightly better performance than the original BEVDiffuser while using semantic BEV teacher signal instead of GT layout conditioning.

### 6. BEVDiffuser + MGD is a strong control, not the main baseline

`BEVDiffuser + MGD` gives the highest NDS and mAP in this table:

| Comparison | ΔNDS | ΔmAP |
|---|---:|---:|
| Ours + MGD vs BEVDiffuser + MGD | -0.39 | -0.17 |

This control shows that GT-layout BEVDiffuser also benefits from MGD. It should be used as a loss-control or privileged-condition control, not as the main baseline for the original BEVDiffuser comparison.

## Final Interpretation

The safe conclusion is not that semantic BEV conditioning alone outperforms BEVDiffuser. Instead, the result supports the following claim:

> Ours achieves comparable and slightly better performance than the original BEVDiffuser while removing GT layout conditioning and task-head supervision from the teacher.

The contribution is therefore:

- A task-head-free semantic BEV diffusion teacher.
- Removal of privileged GT layout conditioning.
- Robust transfer of semantic BEV teacher features through MGD.
- BEVDiffuser-level downstream detection transfer without relying on detection-specific layout teacher information.

This should be framed as a teacher formulation and transfer robustness contribution, not as a new SOTA detection result.
