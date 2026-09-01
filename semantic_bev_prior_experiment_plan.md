# Semantic BEV Prior Ablation Experiment Plan

## 0. Paper-level Claim

이 실험 섹션의 목표는 SOTA 3D object detection 성능을 주장하는 것이 아니다. 논문에서 안전하게 밀 수 있는 핵심 주장은 다음과 같다.

> BEVDiffuser는 GT layout과 task-supervised adaptation에 의존하는 detection-specific teacher이다. 우리는 dense semantic BEV prior와 SB-FDN을 사용해 GT layout 없이 task-head-free generative BEV diffusion teacher를 학습하고, 이 semantic conditioning formulation을 detection뿐 아니라 BEV segmentation 같은 BEV task에도 적용할 수 있음을 보인다.

두 번째 축은 distillation 방법이다.

> Semantic-only teacher는 student feature space와 분포 차이가 더 클 수 있으므로, point-wise direct MSE보다 MGD가 semantic structure를 더 안정적으로 전달한다.

All numeric results and result-specific interpretation are maintained in `semantic_bev_prior_experiment_results.md`.

## 1. Experiment Priority

| Priority | Experiment | Role in Paper |
|---:|---|---|
| P0 | Main detection transfer | Main claim: remove GT layout and task head while keeping BEVDiffuser-level transfer |
| P0 | MGD vs direct MSE | Explains why semantic-only teacher needs robust distillation |
| P1 | Conditioning and SB-FDN ablation | Shows the semantic prior and injection design are necessary |
| P1 | Feature alignment analysis | Mechanistic evidence for feature mismatch and MGD |
| P1 | BEV segmentation transfer | Supports task-agnostic semantic conditioning |
| P2 | Visualizations | Qualitative support for denoising, modulation, and alignment |
| Appendix | Stage 1 frozen-head probe | Diagnostic only; not a main comparison |

## 2. Main Detection Transfer

### Purpose

GT layout과 Stage 1 task head를 제거해도 Stage 2 detection transfer가 BEVDiffuser와 competitive한지 검증한다.

### Table Template

| Method | Teacher Condition | Stage 1 Task Head | Distillation | NDS ↑ | mAP ↑ | mATE ↓ | mAOE ↓ |
|---|---|---:|---|---:|---:|---:|---:|
| BEVFormer 24e | none | no | none | see results | see results | see results | see results |
| BEVDiffuser | GT layout | yes | MSE | see results | see results | see results | see results |
| Ours | Semantic BEV + SB-FDN | no | MGD | see results | see results | see results | see results |

### Interpretation Plan

- Main comparison: `Ours + MGD` vs original `BEVDiffuser + MSE`.
- Claim to defend: Ours reaches BEVDiffuser-level transfer while removing GT layout and Stage 1 task-head supervision.
- Keep `BEVDiffuser + MGD` as a strong control, not the main baseline.

## 3. MGD vs Direct MSE Distillation

### Purpose

Semantic-only teacher를 student로 transfer할 때 direct MSE보다 MGD가 적합한지 검증한다. 이 실험은 feature alignment analysis와 연결해서 설명한다.

### Table Template

| Teacher | Teacher Condition | Distillation | NDS ↑ | mAP ↑ | Delta vs MSE |
|---|---|---|---:|---:|---|
| BEVDiffuser | GT layout | MSE | see results | see results | - |
| BEVDiffuser | GT layout | MGD | see results | see results | see results |
| Ours | Semantic BEV + SB-FDN | MSE | see results | see results | - |
| Ours | Semantic BEV + SB-FDN | MGD | see results | see results | see results |

### Interpretation Plan

- If Ours (MSE) is above the BEVFormer baseline, semantic BEV teacher signal is useful.
- If Ours gains more from MGD than BEVDiffuser does, MGD is especially helpful under semantic-teacher feature mismatch.
- Do not frame MGD as a semantic-only trick; it can also improve the GT-layout teacher.

## 4. Conditioning and SB-FDN Ablation

### Purpose

Semantic BEV prior가 필요한지, 그리고 단순 concat/add보다 SB-FDN 방식의 injection이 필요한지 분리해서 확인한다. 평가는 Stage 1 probe가 아니라 Stage 2 downstream transfer metric으로 한다.

### Minimal Detection Ablation Template

| Variant | GT Layout | Semantic BEV | SB-FDN | Stage 1 Task Head | Distill Loss | NDS ↑ | mAP ↑ |
|---|---:|---:|---:|---:|---|---:|---:|
| No diffusion | no | no | no | no | none | see results | see results |
| BEVDiffuser | yes | no | no | yes | MSE | see results | see results |
| Semantic direct injection | no | yes | no | no | MGD | TBD | TBD |
| Semantic + SB-FDN | no | yes | yes | no | MGD | see results | see results |

### Optional Variants

| Variant | Purpose |
|---|---|
| Layout + Semantic | Upper-bound with privileged GT layout; not the main method |
| Semantic + SB-FDN without DA3 depth consistency | Separates semantic projection from depth-consistency weighting, if DA3 is used in the final model |
| Semantic map quality variants | Tests robustness to SAM/semantic source quality |

## 5. BEV Segmentation Transfer

### Purpose

제안한 semantic conditioning formulation이 detection-specific GT layout에 묶이지 않고 BEV segmentation에도 적용 가능한지 보인다.

### Important Clarification

동일한 teacher checkpoint를 detection과 segmentation에 그대로 재사용한다고 주장하지 않는다. Task별 clean BEV target feature space는 다를 수 있으므로, diffusion teacher checkpoint는 task별로 학습될 수 있다.

재사용되는 것은 checkpoint가 아니라 formulation이다.

- Semantic BEV prior condition
- SegBEVAligner
- SB-FDN injection
- Task-head-free denoising objective
- Stage 2 distillation framework

### Cross-task Setup

| Task | Clean BEV Target | Teacher Condition | Stage 1 Task Head | Stage 2 Student | Metric |
|---|---|---|---:|---|---|
| 3D Detection | detection-pretrained BEV encoder feature | Semantic BEV + SB-FDN | no | BEVFormer Det | NDS/mAP |
| BEV Segmentation | segmentation-pretrained BEV encoder feature | Semantic BEV + SB-FDN | no | BEV segmentation model | mIoU |

### Segmentation Result Template

| Method | Teacher Condition | Distillation | mIoU ↑ | Drivable ↑ | Lane ↑ | Vehicle ↑ |
|---|---|---|---:|---:|---:|---:|
| Seg baseline | none | none | TBD | TBD | TBD | TBD |
| Ours | Semantic BEV + SB-FDN | MGD | TBD | TBD | TBD | TBD |

## 6. Feature Alignment Analysis

### Purpose

Semantic-only teacher가 student feature space와 더 큰 mismatch를 가질 수 있고, 이 때문에 direct MSE보다 MGD가 적합하다는 설명을 정량적으로 뒷받침한다.

### Notation

- `S0`: distillation 전 baseline student BEV feature
- `S_mse`: direct MSE로 distill된 student BEV feature
- `S_mgd`: MGD로 distill된 student BEV feature
- `T_base`: BEVDiffuser denoised teacher BEV feature
- `T_sem`: semantic-only denoised teacher BEV feature

### 6-A. Pre-distillation Teacher-student Gap

| Teacher | MSE(S0,T) ↓ | Cos(S0,T) ↑ | CKA(S0,T) ↑ | Interpretation |
|---|---:|---:|---:|---|
| BEVDiffuser teacher | TBD | TBD | TBD | closer to detection student |
| Semantic-only teacher | TBD | TBD | TBD | larger semantic modulation gap |

### 6-B. Post-distillation Alignment and Performance

| Teacher | Student | Distill Loss | NDS ↑ | mAP ↑ | MSE(S,T) ↓ | Cos(S,T) ↑ | CKA(S,T) ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| BEVDiffuser | student | MSE | see results | see results | TBD | TBD | TBD |
| Semantic-only | student | MSE | see results | see results | TBD | TBD | TBD |
| Semantic-only | student | MGD | see results | see results | TBD | TBD | TBD |

### 6-C. Region-wise Alignment

| Teacher | Distill Loss | Region | MSE(S,T) ↓ | CKA(S,T) ↑ | Activation Corr ↑ |
|---|---|---|---:|---:|---:|
| BEVDiffuser | MSE | object foreground | TBD | TBD | TBD |
| Semantic-only | MSE | object foreground | TBD | TBD | TBD |
| Semantic-only | MGD | object foreground | TBD | TBD | TBD |
| Semantic-only | MGD | semantic boundary | TBD | TBD | TBD |
| Semantic-only | MGD | road/drivable | TBD | TBD | TBD |
| Semantic-only | MGD | static/background | TBD | TBD | TBD |

Recommended regions:

- Object foreground
- Semantic boundary
- Road/drivable area
- Static objects such as barrier, cone, sign, and divider
- Background

## 7. Stage 1 Diagnostics

### Purpose

Stage 1 teacher가 denoising teacher로 학습되었는지 확인한다. Frozen detection head probe는 main evidence가 아니라 appendix diagnostic으로 둔다.

### Diagnostics Template

| Diagnostic | What it Shows |
|---|---|
| Validation denoising loss | Teacher is learning clean BEV feature reconstruction |
| Noisy-to-denoised reconstruction error | Diffusion model moves noisy feature toward clean target |
| Trainable lightweight probe | Whether denoised feature contains task-relevant information without frozen-head bias |
| PCA visualization | Qualitative sanity check of denoising trajectory |
| Frozen detection head probe | Appendix-only diagnostic; see results file |

## 8. Visualizations

### 8-A. Stage 1 Denoising

| Semantic BEV prior | Noisy BEV PCA | Denoised BEV PCA | Clean BEV PCA | Error Reduction |
|---|---|---|---|---|

### 8-B. Feature Alignment

| Student BEV PCA | BEVDiffuser Teacher PCA | Semantic Teacher PCA | Error S-T_base | Error S-T_sem |
|---|---|---|---|---|

| Student w/ Direct MSE | Student w/ MGD | Semantic Teacher | Region Zoom |
|---|---|---|---|

Recommended visualizations:

- SB-FDN modulation magnitude map
- Object/semantic boundary zoom-in
- Foreground/background error heatmap
- Feature activation heatmap
- PCA RGB map

## 9. Recommended Paper Order

1. Main detection transfer
2. MGD vs direct MSE distillation
3. Conditioning and SB-FDN ablation
4. BEV segmentation transfer
5. Feature alignment analysis
6. Visualizations
7. Stage 1 diagnostics in appendix

## 10. Final Story to Defend

1. BEVDiffuser relies on detection-specific GT layout and task-head adaptation.
2. Ours removes both, replacing GT layout with dense semantic BEV prior and SB-FDN.
3. Ours reaches original BEVDiffuser-level detection transfer without privileged GT layout.
4. MGD is important because semantic-only teacher features are useful but harder to copy directly with MSE.
5. The same semantic conditioning formulation can be applied to BEV segmentation.
6. The contribution is **task-agnostic semantic conditioning + robust distillation under feature mismatch**, not SOTA detection.
