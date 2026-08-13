# Semantic BEV Prior Only Experiment Plan

## Core Claim

이 논문은 SOTA 3D object detection 성능을 주장하는 것이 아니다. 핵심 주장은 다음과 같다.

> BEVDiffuser는 GT layout과 task-supervised adaptation에 의존하는 detection-specific teacher이다. 반면, 우리는 dense semantic BEV prior와 SB-FDN을 이용해 task-head-free generative BEV diffusion teacher를 학습하고, 이 semantic conditioning formulation을 3D detection뿐 아니라 BEV segmentation 같은 다른 BEV task에도 적용할 수 있다.

또 다른 중요한 분석 축은 다음이다.

> Semantic-only teacher는 student feature space와 mismatch가 커질 수 있으므로, direct MSE보다 MGD가 더 적합하다.

즉, 실험은 단순 성능 향상보다 다음 질문에 답해야 한다.

- GT layout 없이도 BEVDiffuser와 competitive한 downstream transfer가 가능한가?
- Stage 1에서 task head 없이도 유효한 generative teacher를 만들 수 있는가?
- 같은 semantic conditioning interface를 detection과 segmentation에 모두 적용할 수 있는가?
- semantic-only teacher와 student 사이의 feature mismatch를 MGD가 더 잘 흡수하는가?

## Experiment 1: Main Detection Transfer

### Purpose

GT layout과 Stage 1 task head 없이도 BEVDiffuser와 competitive한 Stage 2 detection transfer를 보이는지 검증한다.

### Table

| Method | Stage 1 Condition | Stage 1 Task Head | Distill Loss | mAP ↑ | NDS ↑ | mATE ↓ | mAOE ↓ |
|---|---|---:|---|---:|---:|---:|---:|
| BEVDiffuser | GT layout | yes | MSE/MGD | 0.2884 | 0.3871 | 0.8707 | 0.5801 |
| Ours | Semantic BEV + SB-FDN | no | MGD | 0.2909 | 0.3903 | 0.8562 | 0.5610 |

### Interpretation

- 향상폭은 작지만, 더 적은 task-specific information으로 BEVDiffuser와 competitive하다.
- 핵심 메시지는 SOTA detection이 아니라 **GT layout 없이도 BEVDiffuser-level transfer가 가능하다**는 점이다.
- mATE와 mAOE 개선은 semantic BEV prior가 geometry/localization에 도움을 준다는 보조 근거가 된다.

## Experiment 2: BEV Segmentation Transfer

### Purpose

제안한 conditioning formulation이 detection-specific GT layout에 묶이지 않고 BEV segmentation에도 적용 가능한지 검증한다.

### Important Clarification

여기서 같은 teacher checkpoint를 모든 task에 재사용한다고 주장하지 않는다. Task별 clean BEV target feature space는 달라질 수 있으므로, diffusion teacher checkpoint는 task별로 학습될 수 있다.

재사용되는 것은 다음이다.

- Semantic BEV prior condition
- SegBEVAligner
- SB-FDN injection
- Task-head-free denoising objective
- Stage 2 distillation framework

### Cross-task Setup Table

| Task | Clean BEV Target | Stage 1 Condition | Stage 1 Task Head | Stage 2 Student | Metric |
|---|---|---|---:|---|---|
| 3D Detection | detection-pretrained BEV encoder feature | Semantic BEV + SB-FDN | no | BEVFormer Det | mAP/NDS |
| BEV Segmentation | segmentation-pretrained BEV encoder feature | Semantic BEV + SB-FDN | no | BEV Seg model | mIoU |

### Segmentation Result Table

| Method | Stage 1 Condition | Stage 1 Task Head | Distillation | mIoU ↑ | Drivable ↑ | Lane ↑ | Vehicle ↑ |
|---|---|---:|---|---:|---:|---:|---:|
| Seg baseline | none | no | none |  |  |  |  |
| Ours | Semantic BEV + SB-FDN | no | MGD |  |  |  |  |

### Interpretation

- BEVDiffuser는 object box layout condition이라 segmentation에 부자연스럽다.
- 우리는 task가 바뀌어도 condition design을 유지할 수 있다.
- 이 실험이 성공하면 contribution은 detection accuracy가 아니라 **task-agnostic semantic conditioning formulation**으로 정리된다.

## Experiment 3: Conditioning and SB-FDN Ablation

### Purpose

Semantic BEV prior와 SB-FDN이 각각 필요한지 확인한다.

### Metric

Stage 1 mAP/NDS가 아니라 Stage 2 transfer metric으로 평가한다.

### Table

| Variant | GT Layout | Semantic BEV | SB-FDN | Stage 1 Task Head | Det mAP ↑ | Det NDS ↑ | Seg mIoU ↑ |
|---|---:|---:|---:|---:|---:|---:|---:|
| No diffusion | no | no | no | no |  |  |  |
| BEVDiffuser | yes | no | no | yes |  |  | N.A./adapted |
| Semantic concat/add | no | yes | no | no |  |  |  |
| Semantic + SB-FDN | no | yes | yes | no |  |  |  |
| Layout + Semantic | yes | yes | yes | optional |  |  | optional |

### Interpretation

- `Semantic + SB-FDN`이 concat/add보다 좋으면 SB-FDN 주입 방식의 필요성을 보일 수 있다.
- `Layout + Semantic`은 main method가 아니라 upper-bound 또는 stronger supervised variant로 둔다.

## Experiment 4: Feature Alignment Analysis

### Purpose

Feature alignment analysis는 MGD 사용 이유를 설명하기 위한 핵심 분석이다.

확인할 내용은 다음이다.

- Semantic-only teacher가 baseline teacher보다 student feature와 더 큰 mismatch를 가지는가?
- Direct MSE가 그 mismatch 때문에 불리한가?
- MGD가 semantic teacher의 정보를 더 안정적으로 흡수하는가?

### Notation

- `S`: student BEV feature
- `T`: denoised teacher BEV feature
- `T_base`: BEVDiffuser teacher feature
- `T_sem`: semantic-only teacher feature
- `S0`: distillation 전 또는 baseline student BEV feature

### 4-A. Global Alignment

| Teacher | Distill Loss | mAP ↑ | NDS ↑ | MSE(S,T) ↓ | Cos(S,T) ↑ | CKA(S,T) ↑ |
|---|---|---:|---:|---:|---:|---:|
| BEVDiffuser teacher | MSE |  |  | low |  |  |
| Semantic-only teacher | MSE |  |  | higher |  |  |
| Semantic-only teacher | MGD |  |  | not necessarily lowest |  |  |

Expected pattern:

- `Semantic-only teacher + MSE`의 `MSE(S,T)`가 `BEVDiffuser teacher + MSE`보다 높으면 semantic teacher가 student와 더 큰 feature-space mismatch를 가진다는 근거가 된다.
- `Semantic-only teacher + MGD`는 MSE가 최저가 아니어도 mAP/NDS가 더 좋을 수 있다.
- 이 경우 MGD는 point-wise feature copying이 아니라 semantic structure를 더 유연하게 전달하는 방식으로 해석할 수 있다.

### 4-B. Teacher-student Gap Before Distillation

Stage 2 초반 또는 distillation 전 student feature와 teacher feature의 gap을 비교한다.

| Teacher | MSE(S0,T) ↓ | Cos(S0,T) ↑ | CKA(S0,T) ↑ | Interpretation |
|---|---:|---:|---:|---|
| BEVDiffuser teacher |  |  |  | closer to student |
| Semantic-only teacher |  |  |  | larger semantic modulation gap |

Interpretation:

- 이 표는 semantic-only teacher가 애초에 student feature space와 더 멀 수 있음을 보여준다.
- 이 결과는 direct MSE보다 MGD가 필요한 이유를 뒷받침한다.

### 4-C. Region-wise Alignment

Global similarity는 semantic conditioning이 어디에 효과를 주는지 숨길 수 있다. 따라서 region-wise 분석을 추가한다.

| Teacher | Distill Loss | Region | MSE(S,T) ↓ | CKA(S,T) ↑ | Activation Corr ↑ |
|---|---|---|---:|---:|---:|
| BEVDiffuser | MSE | foreground |  |  |  |
| Semantic-only | MSE | foreground |  |  |  |
| Semantic-only | MGD | foreground |  |  |  |
| Semantic-only | MGD | semantic boundary |  |  |  |
| Semantic-only | MGD | static/background |  |  |  |

Recommended regions:

- Object foreground
- Semantic boundary
- Road/drivable area
- Static objects such as barrier and traffic cone
- Background

Interpretation:

- Semantic boundary나 static object region에서 MGD가 더 좋은 structural alignment를 보이면 SB-FDN과 semantic BEV prior의 역할을 설명하기 좋다.
- 전체 MSE보다 region-wise CKA나 activation correlation이 더 설득력 있을 수 있다.

## Experiment 5: MGD vs Direct MSE Distillation

### Purpose

Feature alignment 분석과 연결되는 성능 ablation이다. Semantic-only teacher에 direct MSE와 MGD를 각각 적용해 downstream 성능을 비교한다.

### Main Table

| Teacher | Distill Loss | mAP ↑ | NDS ↑ | mATE ↓ | mAOE ↓ |
|---|---|---:|---:|---:|---:|
| Semantic-only teacher | Direct MSE |  |  |  |  |
| Semantic-only teacher | MGD |  |  |  |  |

### Extended Table

| Teacher | Distill Loss | mAP ↑ | NDS ↑ |
|---|---|---:|---:|
| BEVDiffuser teacher | Direct MSE |  |  |
| Semantic-only teacher | Direct MSE |  |  |
| Semantic-only teacher | MGD |  |  |

### Interpretation

- Semantic-only teacher는 direct MSE로는 feature gap 때문에 성능이 제한될 수 있다.
- MGD가 semantic teacher의 정보를 더 안정적으로 전달하면, Feature Alignment 분석과 성능 ablation이 서로 맞물린다.

## Experiment 6: Stage 1 Diagnostic Evaluation

### Purpose

Stage 1 teacher의 직접 평가를 위한 diagnostic이다. 단, frozen detection head probe는 main evidence로 두기보다 appendix diagnostic으로 두는 것이 안전하다.

### Current Frozen Probe Result

| Method | NDS ↑ | mAP ↑ | Setting |
|---|---:|---:|---|
| BEVDiffuser-tiny | 48.61 | 34.56 | noise_T=5 |
| onlyseg_sam3_v4 | 35.22 | 25.09 | noise_T=5 |

### Interpretation

- 이 결과는 BEVDiffuser가 detection-specific head alignment에 유리하다는 진단으로 볼 수 있다.
- Ours는 task-head-free teacher라 frozen detection head가 바로 읽기 어렵다.
- 따라서 이 실험을 main comparison으로 두면 semantic-only teacher가 불리하게 보일 수 있다.

### Optional Stage 1 Diagnostics

- Validation denoising loss
- Noisy-to-denoised reconstruction error
- Trainable lightweight probe
- PCA sanity visualization

## Experiment 7: Visualization

### 7-A. Stage 1 Denoising Visualization

Purpose: denoising 과정이 clean target feature 쪽으로 복원되는지 sanity check한다.

| Semantic BEV prior | Noisy BEV PCA | Denoised BEV PCA | Clean BEV PCA | Error Reduction |
|---|---|---|---|---|

Notes:

- Noisy, denoised, clean feature를 같은 PCA basis로 project한다.
- 이 그림은 main evidence가 아니라 Stage 1 denoising sanity check로 둔다.

### 7-B. Feature Alignment Visualization

Purpose: semantic teacher mismatch와 MGD 필요성을 시각적으로 보여준다.

| Student BEV PCA | BEVDiffuser Teacher PCA | Semantic Teacher PCA | Error S-T_base | Error S-T_sem |
|---|---|---|---|---|

MSE vs MGD comparison:

| Student w/ Direct MSE | Student w/ MGD | Semantic Teacher | Region Zoom |
|---|---|---|---|

Recommended visualizations:

- SB-FDN modulation magnitude map
- Object/semantic boundary zoom-in
- Foreground/background error heatmap
- Feature activation heatmap
- PCA RGB map

## Final Experiment Section Order

Recommended paper order:

1. Main detection transfer
2. BEV segmentation transfer
3. Conditioning and SB-FDN ablation
4. Feature alignment analysis
5. MGD vs direct MSE ablation
6. Visualizations
7. Stage 1 frozen-head diagnostic in appendix

## Final Story

The experiments should support the following story:

1. BEVDiffuser relies on detection-specific GT layout and task-head adaptation.
2. Ours is a task-head-free semantic-only generative teacher.
3. Detection transfer is competitive with BEVDiffuser despite removing GT layout conditioning.
4. The same semantic conditioning formulation is applicable to BEV segmentation.
5. Semantic-only teacher features can be distributionally farther from student features.
6. Feature alignment analysis and MGD ablation explain why MGD is more suitable than direct MSE.
7. The contribution is **task-agnostic semantic conditioning + robust distillation under feature mismatch**, not SOTA detection.
