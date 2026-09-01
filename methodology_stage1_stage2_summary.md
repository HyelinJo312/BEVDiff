# Semantic BEV Prior Only Methodology: Stage 1 and Stage 2

## Method Positioning

`semantic_conditioning_comparison.md` 기준으로 방법론의 중심은 **Semantic BEV prior only**로 둔다. 즉, semantic 정보를 기존 GT layout condition에 추가하는 것이 아니라, BEVDiffuser의 GT-layout 기반 conditioning을 **dense semantic BEV prior로 대체**하는 방향이다.

이 방법론의 목적은 SOTA 3D object detection 성능을 밀어붙이는 것이 아니다. 핵심 문제의식은 **BEVDiffuser가 GT layout이라는 3D object detection 전용 condition에 묶여 있다**는 점이다. 반면 내 방법은 dense semantic BEV prior를 condition으로 사용해, downstream task가 3D detection에서 BEV segmentation 등으로 바뀌어도 같은 conditioning design을 유지할 수 있는 **task-agnostic conditioning interface**를 제안한다.

정리하면 실험 구도는 다음처럼 가져가는 것이 가장 깔끔하다.

- **Baseline**: Original BEVDiffuser with GT layout conditioning.
- **Main method**: Semantic BEV prior only, without GT layout object tokens.
- **Comparison / upper bound**: Semantic-augmented Layout, where GT layout is kept and semantic information is added.

이렇게 두면 논문의 핵심 주장은 단순히 "semantic을 추가해서 성능이 올랐다"가 아니라, **task-specific GT layout conditioning 없이도 downstream BEV task에 재사용 가능한 conditioning formulation을 만들 수 있다**가 된다. 3D detection 결과는 이 주장의 첫 번째 검증이며, 최종적으로는 BEV segmentation 같은 dense prediction task로 확장해 task-agnostic conditioning의 장점을 보여주는 것이 중요하다.

## Stage 1: Semantic-only Generative BEV Pre-training

Stage 1에서는 downstream task head를 붙이지 않는다. 즉, detection head나 segmentation head 없이 semantic BEV prior만을 condition으로 사용하는 diffusion teacher를 **generative pre-training** 방식으로 학습한다.

### Goal

기존 BEVDiffuser는 GT object layout token을 diffusion condition으로 사용한다. 이 layout token은 3D object detection annotation schema에 직접 묶여 있기 때문에, BEV segmentation 같은 다른 downstream task로 확장할 때 condition 자체를 다시 설계해야 한다. 내 방법의 Stage 1은 이 sparse하고 task-specific한 layout condition을 제거하고, segmentation에서 얻은 scene-level dense semantic BEV prior를 diffusion condition으로 사용한다.

### Input, Condition, and Objective

- Clean BEV feature target은 frozen BEVFormer 계열 모델에서 얻는다.
- Clean BEV feature에 DDPM noise를 추가해 noisy BEV feature를 만든다.
- Diffusion UNet은 noisy BEV feature와 timestep을 입력받고, semantic BEV prior만을 condition으로 사용해 denoising을 학습한다.
- Main method에서는 GT layout object tokens를 사용하지 않는다.
- Stage 1에는 detection/classification/regression/segmentation task head가 없으므로, Stage 1 자체를 mAP/NDS 또는 mIoU로 직접 평가하지 않는다.
- Stage 1의 목적은 downstream metric 최적화가 아니라, semantic BEV prior를 통해 clean BEV feature distribution을 복원하는 generative teacher를 만드는 것이다.

### Semantic BEV Prior Injection

- Multi-view semantic segmentation map을 `SegBEVAligner`로 BEV 공간에 정렬한다.
- 정렬된 semantic BEV feature는 multi-scale BEV prior로 사용된다.
- UNet의 attention block은 외부 layout context 없이 self-attention으로 동작한다.
- Semantic prior는 decoder의 SBAM path를 통해 spatially adaptive normalization 형태로 주입된다.
- 따라서 semantic condition은 object token을 attention으로 참조하는 수준이 아니라, BEV feature의 spatial statistics 자체를 semantic-dependent하게 조절한다.

### Semantic BEV Prior Construction Details

`BEVFormer/projects/bevdiffuser/layout_diffusion/seg_bev_aligner_one_hot_v3.py`
기준으로 semantic BEV prior는 다음 순서로 만들어진다.

1. Dataset은 각 camera image에 대응하는 SAM3 semantic mask(`*_mask.bin`)를
읽고, image pipeline과 동일하게 nearest-neighbor resize 및 padding을 적용한다.
따라서 `seg_maps`는 padded image frame 기준의 multi-view semantic ID map
`[V, H_pad, W_pad]`로 dataloader에 들어간다.
2. Config의 `seg_id_remap={16:2, 17:2, 18:2, 19:16}`에 따라 SAM3 raw class
id를 model taxonomy로 정리한다. 현재 설정에서는 crosswalk, lane, road arrow를
road class로 합치고, sky는 별도 sky id로 둔다.
3. `SegEmbedEncoder`는 batch semantic ID map `[B,V,H,W]`를 `[B*V,H,W]`로
펼친 뒤, invalid label `-1`을 0으로 치환하고, configurable nearest downsampling을
적용한다. 현재 config의 `seg_downsample_factor=2`는 기존 half-resolution setting을
재현하며, `seg_downsample_factor=1`로 두면 full-resolution semantic one-hot을 사용할 수 있다.
이후 `F.one_hot`으로 각 pixel의 semantic class id를 one-hot vector로 변환한다.
현재 `num_classes=16`이므로 one-hot channel은 background/ignore를 포함해 `C=17`이다.
4. `SegBEVAligner`는 BEVFormer-style 3D reference points를 BEV grid마다
생성한다. 현재 BEV grid는 `50x50`이고, 각 BEV cell에 대해
`num_points_in_pillar=4`개의 height sample을 사용한다. Config에서는
`pillar_z_range=(-1.84,1.16)`을 사용해 semantic projection에 쓰는 z range를 제한한다.
5. 각 3D BEV reference point를 `lidar2img` calibration으로 multi-view image
plane에 투영한다. 투영된 image coordinate가 image boundary 안에 있고 positive
depth를 갖는 경우에만 valid semantic observation으로 사용한다.
6. 투영 위치에서 semantic one-hot map을 `grid_sample(..., mode='nearest')`로
샘플링한다. 이 결과는 `[B,V,Q,D,C]` 형태의 semantic vote가 된다. 여기서
`Q=bev_h*bev_w`, `D=num_points_in_pillar`, `C=semantic classes`이다.
7. DA3 depth map이 있는 경우, depth consistency weight를 semantic vote에 곱한다.
이 weight는 diffusion model에 직접 들어가는 별도 condition이 아니라,
semantic-to-BEV projection 과정에서 각 semantic observation의 신뢰도를 조절하는
voting weight이다.
8. 구체적으로 projected BEV point의 camera-frame depth `d_proj`와 같은 image
coordinate에서 sample한 DA3 depth `d_DA3`를 비교하고, Gaussian kernel
`w_depth = exp(-(d_proj-d_DA3)^2/(2*sigma^2))`를 계산한다. 현재 config에서는
`depth_consistency_mode='gaussian'`, `depth_consistency_sigma=4.0`이다.
9. 최종 semantic vote는 projection validity mask와 depth consistency weight를
곱한 뒤 pillar 방향과 camera-view 방향으로 합산된다. 즉 BEV cell `q`의 semantic
histogram은 개념적으로 `H_q(c)=sum_v sum_d m_qvd w_qvd 1[s_qvd=c]`와 같이 계산된다.
10. `sky_as_ignore=True`인 경우 sky channel의 count는 ignore/background channel로
이동되고 sky channel은 0으로 지워진다. 이 처리는 sky가 BEV ground-plane semantic
prior를 지배하지 않도록 하는 projection 후처리로 볼 수 있다.
11. Semantic histogram은 class probability distribution으로 normalize되고,
convolutional embedding(`prob_to_emb`)과 learnable BEV positional embedding을 거쳐
256-channel semantic BEV feature가 된다.
12. 마지막으로 `SegBEVEncoder`가 이 semantic BEV feature를 multi-scale prior
`{1,2,4}`로 변환한다. 현재 `channel_mult=[1,1,1]`이므로 각 scale의 semantic prior
channel은 모두 256으로 유지되어 UNet의 `seg_channels=[256,256,256]`과 맞춰진다.

이 과정을 논문에서 설명할 때 중요한 점은, DA3 depth가 diffusion denoising의
독립적인 condition이 아니라는 것이다. DA3 depth는 semantic mask를 BEV로 정렬할 때
occlusion이나 projection mismatch가 의심되는 semantic observation을 약하게 만드는
**depth-consistency weighted semantic voting**으로 사용된다. 최종적으로 diffusion
UNet에 주입되는 것은 DA3 depth map 자체가 아니라, DA3 depth로 projection confidence가
보정된 dense semantic BEV prior이다.

### Output

Stage 1의 결과물은 semantic-only condition을 받은 denoised BEV feature를 생성할 수 있는 frozen diffusion teacher이다. 이 teacher는 Stage 2에서 downstream BEV model의 feature distillation target으로 사용된다. 따라서 Stage 1의 가치는 standalone detection score가 아니라, 여러 downstream BEV task에서의 transfer 성능으로 검증한다.

### Key Implementation Files

- `BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg.py`
- `BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v3.py`
- `BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_da3.py`
- `BEVFormer/projects/bevdiffuser/train_bev_diffuser_only_seg.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_diffusion_unet.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_diffusion_unet_v2.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_bev_aligner_one_hot_v3.py`
- `BEVFormer/projects/bevdiffuser/data_utils.py`

### How to Describe Stage 1 in the Paper

Stage 1은 "GT layout을 semantic으로 보강한 diffusion teacher"가 아니라, **task head 없이 GT layout conditioning을 dense semantic BEV prior로 대체해 학습한 generative BEV diffusion teacher**로 설명한다. 이 점이 novelty의 중심이다.

다만 clean BEV target 자체는 detection-trained BEVFormer에서 오기 때문에, claim은 "fully task-agnostic representation learning"보다 **task-agnostic conditioning interface** 또는 **task-transferable BEV diffusion teacher**가 더 안전하다.

## Stage 2: Downstream BEV Task Training with Semantic Diffusion Distillation

Stage 2에서는 downstream BEV model을 학습하면서, Stage 1에서 학습한 frozen semantic-only diffusion teacher의 denoised BEV feature를 distillation target으로 사용한다. 현재 3D object detection 실험에서는 student BEVFormer를 사용하지만, 같은 구조는 BEV segmentation student에도 적용할 수 있다.

### Goal

Downstream model이 task loss만으로 학습할 때보다 더 semantic-aware한 BEV representation을 갖도록, Stage 1 teacher가 생성한 denoised BEV feature를 feature-level supervision으로 제공한다.

### Training Flow

- Student model은 multi-view camera image만 입력받아 BEV feature와 task prediction을 만든다.
- Frozen semantic-only diffusion teacher는 noisy/reference BEV feature와 semantic BEV prior를 받아 denoised teacher BEV feature를 생성한다.
- Student는 각 downstream task에 맞는 task loss로 학습한다. 현재 detection 실험에서는 3D detection loss를 사용하고, BEV segmentation 실험에서는 segmentation loss를 사용할 수 있다.
- 동시에 student BEV feature가 denoised teacher BEV feature를 따라가도록 feature distillation loss를 추가한다.
- 학습 후 배포되는 모델은 student model이며, diffusion teacher와 semantic condition branch는 inference model에 포함되지 않는다.

### Distillation Mechanism

현재 Semantic BEV Prior Only Stage 2의 핵심 distillation은 MGD-style feature distillation로 정리할 수 있다.

- `noise_timesteps=100`
- `denoise_timesteps=100`
- `mgd_alpha=100`
- `mgd_lambda=0.6`

MGD를 쓰는 이유는 teacher feature가 SBAM modulation을 거치면서 student의 원래 BEVFormer feature 분포와 달라지기 때문이다. 단순 MSE보다 masked generation 방식이 teacher-student feature gap을 더 안정적으로 흡수한다는 논리로 설명할 수 있다.

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

현재 결과는 Semantic BEV Prior Only의 **3D object detection downstream transfer** 결과이다. 이 결과의 역할은 SOTA detection 성능을 주장하는 것이 아니라, GT layout condition 없이 학습한 semantic-only generative teacher가 BEVDiffuser의 layout-conditioned teacher와 비교해 competitive한 transfer 성능을 낸다는 것을 보여주는 것이다.

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

The main result is positive but modest. Semantic BEV Prior Only improves the BEVDiffuser baseline by **+0.25 mAP** and **+0.323 NDS** points at epoch 24. This should not be presented as a SOTA detection gain. The important point is that the method removes GT layout conditioning and still preserves, or slightly improves, BEVDiffuser-level downstream detection transfer. The strongest gains appear in traffic cone, motorcycle, barrier, truck, and car AP, while construction vehicle, bus, bicycle, and pedestrian decrease slightly.

The error terms suggest that the method mainly improves geometric quality rather than all detection attributes uniformly:

- mATE improves by 0.0145, so object center localization becomes better.
- mAOE improves by 0.0191, so orientation estimation becomes better.
- mASE, mAVE, and mAAE become slightly worse.

This pattern is consistent with the method's motivation. Dense semantic BEV prior provides spatial scene structure, so improvements are more visible in localization/orientation and static or geometry-sensitive categories. It does not necessarily improve velocity or attribute prediction, because those signals are not directly represented by the semantic prior.

## Feature Alignment Analysis

Feature alignment analysis is needed to explain why direct MSE distillation is not ideal for the semantic-only teacher and why MGD is a better Stage 2 distillation objective.

The key hypothesis is:

- The baseline BEVDiffuser teacher is conditioned by GT layout and remains relatively close to the student BEVFormer feature space.
- The semantic-only teacher injects dense semantic BEV priors through SBAM, which can shift feature statistics more strongly.
- Therefore, the semantic-only teacher may be more informative but farther from the camera-only student feature distribution.
- Direct MSE can over-constrain the student to match teacher features point-wise, while MGD can absorb the teacher's semantic structure more flexibly.

### Global Teacher-student Alignment

Here, `S` denotes the student BEV feature and `T` denotes the denoised teacher BEV feature. `MSE(S,T)` measures point-wise feature distance, while `CKA(S,T)` measures structural representation similarity.

| Teacher | Distill Loss | mAP | NDS | MSE(S,T) | Cos(S,T) | CKA(S,T) | Expected Interpretation |
|---|---|---:|---:|---:|---:|---:|---|
| BEVDiffuser teacher | MSE |  |  | lower |  |  | layout teacher is closer to student feature space |
| Semantic-only teacher | MSE |  |  | higher |  |  | semantic teacher has larger feature-space mismatch |
| Semantic-only teacher | MGD |  |  | not necessarily lowest |  | higher or task-better | MGD absorbs semantic structure without forcing point-wise copying |

The most useful result pattern is not necessarily that MGD gives the lowest `MSE(S,T)`. Direct MSE may reduce point-wise distance more aggressively. The stronger claim is that semantic-only teacher features are harder to align with direct MSE, and MGD gives better downstream performance and/or better structural alignment despite not minimizing raw feature distance.

### Region-wise Alignment

Global feature similarity can hide where semantic conditioning helps. A region-wise analysis should compare student-teacher alignment on foreground, semantic boundary, static-object, and background regions.

| Teacher | Distill Loss | Region | MSE(S,T) | CKA(S,T) | Activation Corr. | Expected Interpretation |
|---|---|---|---:|---:|---:|---|
| BEVDiffuser teacher | MSE | foreground |  |  |  | baseline alignment under layout conditioning |
| Semantic-only teacher | MSE | foreground |  |  |  | larger mismatch under direct MSE |
| Semantic-only teacher | MGD | foreground |  |  |  | better task-relevant structural absorption |
| Semantic-only teacher | MGD | semantic boundary |  |  |  | semantic prior affects geometry-sensitive regions |
| Semantic-only teacher | MGD | background/static |  |  |  | scene-level semantic context is preserved |

This analysis supports the message that **semantic-only teacher features are useful but distributionally different**, so MGD is used not merely as another loss but as a better mechanism for transferring semantic structure under feature-space mismatch.

### Visualization for Alignment

The visualization should focus on feature mismatch and semantic structure transfer rather than small box-level differences.

Recommended figures:

- Student BEV PCA, BEVDiffuser teacher PCA, semantic-only teacher PCA.
- `|S - T|` error heatmaps for baseline teacher and semantic-only teacher.
- Student trained with direct MSE vs student trained with MGD.
- Semantic boundary or object-region zoom-ins.
- SBAM modulation magnitude map to show where semantic prior changes the diffusion feature.

For PCA visualization, noisy BEV, denoised BEV, and clean target BEV should be projected using the same PCA basis. This is useful as a Stage 1 denoising sanity check, but the stronger paper figure is the Stage 2 feature alignment visualization above.

## Extension to BEV Segmentation

BEVDiffuser's GT layout condition is naturally aligned with 3D object detection because it uses object categories and object boxes as privileged conditioning information. This makes the baseline less natural for dense BEV tasks such as BEV segmentation, where the downstream target is not a sparse object layout.

Semantic BEV Prior Only is better positioned for cross-task transfer:

- Stage 1 teacher is trained without a downstream task head.
- The condition is dense scene semantics, not detection-specific object layout.
- The same conditioning interface, SegBEVAligner, SBAM injection, and task-head-free denoising recipe can be used for both 3D detection and BEV segmentation.
- The actual diffusion teacher checkpoint may be task-specific because each task can use a different pretrained BEV encoder feature as the clean target.
- For BEV segmentation, Stage 1 can use a segmentation-pretrained BEV encoder feature as the clean target, and Stage 2 can replace the detection head/loss with a segmentation head/loss while keeping the semantic conditioning design unchanged.

This is the strongest experimental direction for the paper. If detection and BEV segmentation both benefit from the same semantic conditioning interface, the paper can claim that the contribution is not detection accuracy itself, but a **reusable BEV diffusion conditioning formulation**.

## Competitiveness for Overseas Conferences

With the current numbers alone, the result is not meant to be a pure performance paper. The gain over BEVDiffuser is real but small, and reviewers may ask whether +0.25 mAP / +0.323 NDS is within run-to-run variance unless repeated seeds or stronger ablations are provided.

However, the method can still be competitive if the paper is framed as a **methodology/annotation-dependence contribution** rather than a leaderboard improvement:

- The main novelty is removing 3D-detection-specific GT layout conditioning from BEV diffusion.
- Stage 1 is task-head-free generative BEV pre-training.
- Stage 2 shows that this layout-free teacher still gives competitive or slightly better downstream detection performance.
- The same semantic conditioning interface can be extended to BEV segmentation, where BEVDiffuser's object-layout condition is less natural.
- Feature alignment analysis can explain why semantic-only teacher features require MGD rather than naive direct MSE.
- The key message is **reusable task-agnostic conditioning without GT layout**, not SOTA detection accuracy.

For a stronger submission, the current result should be supported with:

- repeated runs or variance analysis,
- ablation against layout-only, semantic-only without SBAM, MSE vs MGD, and semantic-augmented layout,
- feature alignment analysis with MSE, cosine similarity, CKA, and region-wise alignment,
- annotation-cost or condition-dependence comparison,
- qualitative BEV feature / detection visualizations showing semantic prior effects,
- BEV segmentation transfer using the same semantic conditioning interface to support the task-agnostic conditioning claim.

Current competitiveness:

- **Top-tier CVPR/ICCV/ECCV main track**: still risky with detection-only results, but much more defensible if the same semantic conditioning interface works for both detection and BEV segmentation.
- **IROS/ICRA/IV/ITSC or domain-focused autonomous driving venues**: plausible if experiments are clean and the reduced GT-layout dependency is emphasized.
- **Workshop submission**: quite viable with the current conceptual angle and detection result, stronger with segmentation transfer.

## Final Method Story

1. Existing BEV diffusion teachers rely on task-specific GT layout conditioning.
2. This layout condition is tied to a 3D object detection annotation schema and is hard to reuse for other BEV tasks such as BEV segmentation.
3. Stage 1 performs task-head-free generative BEV pre-training by replacing GT layout conditioning with a dense semantic BEV prior.
4. Stage 2 transfers the semantic diffusion teacher's denoised BEV feature to downstream BEV students through MGD-based feature distillation.
5. Feature alignment analysis shows that semantic-only teacher features can be distributionally farther from student features, motivating MGD over direct MSE.
6. Detection results show competitive transfer without GT layout conditioning, and BEV segmentation is the key extension for demonstrating cross-task applicability of the conditioning interface.
7. The final contribution is a reusable semantic conditioning interface with reduced dependence on task-specific GT layout annotations, rather than a SOTA detection claim.
