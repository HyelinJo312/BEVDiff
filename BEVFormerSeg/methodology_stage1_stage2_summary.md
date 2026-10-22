# Semantic BEV Prior Only Methodology: Stage 1 and Stage 2

## Method Positioning

`semantic_conditioning_comparison.md` 기준으로 방법론의 중심은 **Semantic BEV prior only**로 둔다. 즉, semantic 정보를 기존 GT layout condition에 추가하는 것이 아니라, BEVDiffuser의 GT-layout 기반 conditioning을 **dense semantic BEV prior로 대체**하는 방향이다.

정리하면 실험 구도는 다음처럼 가져가는 것이 가장 깔끔하다.

- **Baseline**: Original BEVDiffuser with GT layout conditioning.
- **Main method**: Semantic BEV prior only, without GT layout object tokens.
- **Comparison / upper bound**: Semantic-augmented Layout, where GT layout is kept and semantic information is added.

이렇게 두면 논문의 핵심 주장은 단순히 "semantic을 추가해서 성능이 올랐다"가 아니라, **task-specific GT layout conditioning 없이도 competitive transfer performance를 얻는 더 일반적인 BEV diffusion conditioning**이 된다.

## Stage 1: Semantic-only Generative BEV Pre-training

Stage 1에서는 downstream detection head를 붙이지 않는다. 즉, task head 없이 semantic BEV prior만을 condition으로 사용하는 diffusion teacher를 **generative pre-training** 방식으로 학습한다.

### Goal

기존 BEVDiffuser는 GT object layout token을 diffusion condition으로 사용한다. 내 방법의 Stage 1은 이 sparse하고 task-specific한 layout condition을 제거하고, segmentation에서 얻은 scene-level dense semantic BEV prior를 diffusion condition으로 사용한다.

### Input, Condition, and Objective

- Clean BEV feature target은 frozen BEVFormer 계열 모델에서 얻는다.
- Clean BEV feature에 DDPM noise를 추가해 noisy BEV feature를 만든다.
- Diffusion UNet은 noisy BEV feature와 timestep을 입력받고, semantic BEV prior만을 condition으로 사용해 denoising을 학습한다.
- Main method에서는 GT layout object tokens를 사용하지 않는다.
- Stage 1에는 detection/classification/regression task head가 없으므로, Stage 1 자체를 mAP/NDS로 직접 평가하지 않는다.
- Stage 1의 목적은 downstream metric 최적화가 아니라, semantic BEV prior를 통해 clean BEV feature distribution을 복원하는 generative teacher를 만드는 것이다.

### Semantic BEV Prior Injection

- Multi-view semantic segmentation map을 `SegBEVAligner`로 BEV 공간에 정렬한다.
- 정렬된 semantic BEV feature는 multi-scale BEV prior로 사용된다.
- UNet의 attention block은 외부 layout context 없이 self-attention으로 동작한다.
- Semantic prior는 decoder의 BFDN/FDN path를 통해 spatially adaptive normalization 형태로 주입된다.
- 따라서 semantic condition은 object token을 attention으로 참조하는 수준이 아니라, BEV feature의 spatial statistics 자체를 semantic-dependent하게 조절한다.

### Output

Stage 1의 결과물은 semantic-only condition을 받은 denoised BEV feature를 생성할 수 있는 frozen diffusion teacher이다. 이 teacher는 Stage 2에서 student BEVFormer의 feature distillation target으로 사용된다. 따라서 Stage 1의 가치는 standalone detection score가 아니라, Stage 2 transfer 성능으로 검증한다.

### Key Implementation Files

- `BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg.py`
- `BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v3.py`
- `BEVFormer/projects/bevdiffuser/train_bev_diffuser_only_seg.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_diffusion_unet.py`

### How to Describe Stage 1 in the Paper

Stage 1은 "GT layout을 semantic으로 보강한 diffusion teacher"가 아니라, **task head 없이 GT layout conditioning을 dense semantic BEV prior로 대체해 학습한 generative BEV diffusion teacher**로 설명한다. 이 점이 novelty의 중심이다.

다만 clean BEV target 자체는 detection-trained BEVFormer에서 오기 때문에, claim은 "fully task-agnostic representation learning"보다 **task-agnostic conditioning interface**가 더 안전하다.

## Stage 2: BEVFormer Training with Semantic Diffusion Distillation

Stage 2에서는 student BEVFormer를 학습하면서, Stage 1에서 학습한 frozen semantic-only diffusion teacher의 denoised BEV feature를 distillation target으로 사용한다.

### Goal

Student BEVFormer가 일반적인 detection loss만으로 학습할 때보다 더 semantic-aware한 BEV representation을 갖도록, Stage 1 teacher가 생성한 denoised BEV feature를 feature-level supervision으로 제공한다.

### Training Flow

- Student BEVFormer는 multi-view camera image만 입력받아 BEV feature와 detection prediction을 만든다.
- Frozen semantic-only diffusion teacher는 noisy/reference BEV feature와 semantic BEV prior를 받아 denoised teacher BEV feature를 생성한다.
- Student는 기존 detection loss로 3D detection task를 학습한다.
- 동시에 student BEV feature가 denoised teacher BEV feature를 따라가도록 feature distillation loss를 추가한다.
- 학습 후 배포되는 모델은 student BEVFormer이며, diffusion teacher와 semantic condition branch는 inference model에 포함되지 않는다.

### Distillation Mechanism

현재 Semantic BEV Prior Only Stage 2의 핵심 distillation은 MGD-style feature distillation로 정리할 수 있다.

- `noise_timesteps=100`
- `denoise_timesteps=100`
- `mgd_alpha=100`
- `mgd_lambda=0.6`

MGD를 쓰는 이유는 teacher feature가 semantic FDN modulation을 거치면서 student의 원래 BEVFormer feature 분포와 달라지기 때문이다. 단순 MSE보다 masked generation 방식이 teacher-student feature gap을 더 안정적으로 흡수한다는 논리로 설명할 수 있다.

### Key Implementation Files

- `BEVFormer/projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_v3.py`
- `BEVFormer/tools/train_onlyseg.py`
- `BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer_onlyseg.py`

## Semantic-augmented Layout as Comparison / Upper Bound

기존 `layout_tiny_seg_v4_2` 계열은 main method라기보다 **Semantic-augmented Layout** variant로 위치시키는 것이 좋다. 이 variant는 BEVDiffuser의 GT layout conditioning을 유지하면서 semantic 정보를 추가하므로, raw performance 측면에서는 강할 수 있지만 novelty는 main method보다 약하다.

### Related Files

- `BEVFormer/projects/configs/bevdiffuser/layout_tiny_seg_v4_2.py`
- `BEVFormer/projects/bevdiffuser/train_bev_diffuser_seg_v2.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/layout_seg_diffusion_unet_v4_2.py`
- `BEVFormer/projects/configs/diff_bevformer/layout_tiny_seg_v4_2_mgd.py`
- `BEVFormer/tools/train_seg.py`
- `BEVFormer/projects/mmdet3d_plugin/bevformer/detectors/diff_bevformer_seg.py`

## Main Stage 2 Results

Baseline:

- Stage 2: `results/version2/stage2/DiffBEVFormer_tiny_original_24epoch`

Semantic BEV Prior Only:

- Stage 1 teacher: `results/version2/stage1/BEVDiffuser_tiny_onlyseg_sam3_v2/checkpoint-50000`
- Stage 2: `results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_mgd_v4_t100_alpha100_lambda_0.6`

Only the matched Stage 2 comparison point is used:

- Stage 2: epoch 24

| Stage | Comparison Point | Metric | Baseline | Semantic BEV Prior Only | Delta |
|---|---:|---:|---:|---:|---:|
| Stage 2 | epoch 24 | mAP | 0.28838 | 0.29088 | +0.00250 |
| Stage 2 | epoch 24 | NDS | 0.38708 | 0.39031 | +0.00323 |
| Stage 2 | epoch 24 | mATE | 0.8707 | 0.8562 | -0.0145 |
| Stage 2 | epoch 24 | mASE | 0.2858 | 0.2899 | +0.0041 |
| Stage 2 | epoch 24 | mAOE | 0.5801 | 0.5610 | -0.0191 |
| Stage 2 | epoch 24 | mAVE | 0.6286 | 0.6309 | +0.0023 |
| Stage 2 | epoch 24 | mAAE | 0.2057 | 0.2134 | +0.0077 |

Class-wise mean AP:

| Class | Baseline | Semantic BEV Prior Only | Delta |
|---|---:|---:|---:|
| car | 0.4815 | 0.4858 | +0.0043 |
| truck | 0.2276 | 0.2343 | +0.0068 |
| construction_vehicle | 0.0784 | 0.0682 | -0.0102 |
| bus | 0.3214 | 0.3125 | -0.0089 |
| trailer | 0.1033 | 0.1037 | +0.0004 |
| barrier | 0.3877 | 0.3980 | +0.0103 |
| motorcycle | 0.2692 | 0.2830 | +0.0138 |
| bicycle | 0.2577 | 0.2485 | -0.0093 |
| pedestrian | 0.3538 | 0.3496 | -0.0042 |
| traffic_cone | 0.4033 | 0.4253 | +0.0220 |

## Result Analysis

The main result is positive but modest. Semantic BEV Prior Only improves the BEVDiffuser baseline by **+0.25 mAP** and **+0.323 NDS** points at epoch 24. The strongest gains appear in traffic cone, motorcycle, barrier, truck, and car AP, while construction vehicle, bus, bicycle, and pedestrian decrease slightly.

The error terms suggest that the method mainly improves geometric quality rather than all detection attributes uniformly:

- mATE improves by 0.0145, so object center localization becomes better.
- mAOE improves by 0.0191, so orientation estimation becomes better.
- mASE, mAVE, and mAAE become slightly worse.

This pattern is consistent with the method's motivation. Dense semantic BEV prior provides spatial scene structure, so improvements are more visible in localization/orientation and static or geometry-sensitive categories. It does not necessarily improve velocity or attribute prediction, because those signals are not directly represented by the semantic prior.

## Competitiveness for Overseas Conferences

With the current numbers alone, the result is probably **not strong enough as a pure performance paper** for top-tier CV conferences. The gain over BEVDiffuser is real but small, and reviewers may ask whether +0.25 mAP / +0.323 NDS is within run-to-run variance unless repeated seeds or stronger ablations are provided.

However, the method can still be competitive if the paper is framed as a **methodology/annotation-dependence contribution** rather than a leaderboard improvement:

- The main novelty is removing task-specific GT layout conditioning from BEV diffusion.
- Stage 1 is task-head-free generative BEV pre-training.
- Stage 2 shows that this layout-free teacher still gives competitive or slightly better downstream detection performance.
- The key message is **competitive transfer without GT layout conditioning**, not SOTA detection accuracy.

For a stronger submission, the current result should be supported with:

- repeated runs or variance analysis,
- ablation against layout-only, semantic-only without FDN, MSE vs MGD, and semantic-augmented layout,
- annotation-cost or condition-dependence comparison,
- qualitative BEV feature / detection visualizations showing semantic prior effects,
- ideally transfer to another BEV task or dataset split to support the task-agnostic conditioning claim.

Current competitiveness:

- **Top-tier CVPR/ICCV/ECCV main track**: risky unless the story is sharpened and ablations are strong.
- **IROS/ICRA/IV/ITSC or domain-focused autonomous driving venues**: plausible if experiments are clean and the reduced GT-layout dependency is emphasized.
- **Workshop submission**: quite viable with the current conceptual angle and result.

## Final Method Story

1. Existing BEV diffusion teachers rely on task-specific GT layout conditioning.
2. This layout condition is tied to a detection annotation schema and is hard to reuse for other BEV tasks.
3. Stage 1 performs task-head-free generative BEV pre-training by replacing GT layout conditioning with a dense semantic BEV prior.
4. Stage 2 transfers the semantic diffusion teacher's denoised BEV feature to a student BEVFormer through MGD-based feature distillation.
5. The final contribution is a more reusable semantic conditioning interface with reduced dependence on GT layout annotations, while preserving competitive downstream transfer performance.
