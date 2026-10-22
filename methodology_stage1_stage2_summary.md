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

**핵심 경로:** `SAM3 semantic map + Metric3D depth → 관측 표면의 semantic BEV prior → multi-scale semantic encoding → timestep-gated SBAM을 통한 diffusion UNet decoder 조건 주입 → clean BEV feature 복원 loss`. 대표 semantic-only run은 `task_loss_scale=0`으로 학습한다. 따라서 여기서 **task-agnostic인 것은 conditioning interface와 학습 objective**이며, clean BEV target의 출처까지 task-agnostic이라고 주장하지 않는다.

### Goal

기존 BEVDiffuser는 GT object layout token을 diffusion condition으로 사용한다. 이 layout token은 3D object detection annotation schema에 직접 묶여 있기 때문에, BEV segmentation 같은 다른 downstream task로 확장할 때 condition 자체를 다시 설계해야 한다. 내 방법의 Stage 1은 이 sparse하고 task-specific한 layout condition을 제거하고, segmentation에서 얻은 scene-level dense semantic BEV prior를 diffusion condition으로 사용한다.

### Input, Condition, and Objective

- Clean BEV feature target은 frozen BEVFormer 계열 모델에서 얻는다.
- Clean BEV feature에 DDPM noise를 추가해 noisy BEV feature를 만든다.
- Diffusion UNet은 noisy BEV feature와 timestep을 입력받고, semantic BEV prior만을 condition으로 사용해 denoising을 학습한다.
- Main method에서는 GT layout object tokens를 사용하지 않는다.
- Stage 1에는 detection/classification/regression/segmentation task head가 없으므로, Stage 1 자체를 mAP/NDS 또는 mIoU로 직접 평가하지 않는다.
- Stage 1의 목적은 downstream metric 최적화가 아니라, semantic BEV prior를 통해 clean BEV feature distribution을 복원하는 generative teacher를 만드는 것이다.

### Semantic BEV Prior Construction Details

현재 semantic BEV 생성은
`BEVFormer/projects/bevdiffuser/layout_diffusion/seg_bev_aligner_one_hot_v5.py`와
`BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py`를
기준으로 한다. 핵심은 **metric depth로 관측된 semantic surface를 3D 공간으로
역투영하고, BEV 격자에 splatting하는 방식**이다. 이전 v3의 BEV pillar reference
point를 이미지에 투영해 semantic vote를 모으고 DA3 depth-consistency Gaussian
weight를 적용하던 설명은 현재 v5 생성 과정에 해당하지 않는다.

| 항목 | 현재 Metric3D/v5 설정 |
| --- | --- |
| Semantic 입력 | SAM3 semantic ID map |
| Depth 입력 | Metric3D metric depth, `depth_type='camera_z'` |
| 원본 map 크기 | `900 × 1600` |
| 유효 depth 범위 | `2.5 ≤ d ≤ 170.0` m |
| BEV 격자 | `50 × 50` |
| 3D 공간 범위 | `[-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]` m |
| Pixel sampling | `pixel_stride=1`, 정렬된 map의 모든 픽셀을 검사 |
| BEV 누적 | `splat_mode='bilinear'` |
| 투영 결과 | `[B, 17, 50, 50]` semantic mass/probabilities |
| 학습 가능한 prior | scale `{1,2,4}`, 각 scale 256 channels |

1. **Semantic과 depth를 이미지 augmentation 좌표계에 정렬한다.**
   `CustomNuScenesDiffusionDataset_seg_depth_v2`는 SAM3 mask와 Metric3D depth를
   읽고, 원본 및 augmentation 후 `lidar2img`로 이미지 평면 변환을 복원한다.
   이 변환을 `cv2.warpPerspective(..., flags=cv2.INTER_NEAREST)`로 적용하므로
   현재 구현은 semantic과 depth 모두 nearest-neighbor 정렬을 사용한다.
   Image content 영역으로 변환한 뒤 padding하며, semantic은 `-1`, depth는 `0`으로
   채운다. Batch 입력은 동일한 `[B,V,H_pad,W_pad]` shape을 갖는다.
   뒤에서 사용하는 **bilinear splatting은 BEV 격자에 값을 분배하는 연산**이며,
   이 단계의 이미지 보간 방식과 구분된다.

2. **Semantic taxonomy와 유효 관측을 결정한다.**
   Dataset은 SAM3 raw ID에 `seg_id_remap={16:2, 17:2, 18:2, 19:16}`을 적용한다.
   Crosswalk, lane, road arrow는 road ID 2로 합치고 sky는 ID 16으로 옮긴다.
   Depth의 NaN/Inf와 범위 밖 값은 dataset에서 0으로 처리하고, projector에서도
   유한한 depth와 유효 범위를 검사한다. Semantic ID `0`, `-1`, 범위 밖 label은
   제외한다. `sky_as_ignore=True`이므로 sky ID 16도 **splatting 전에 제외**한다.
   따라서 현재 실제 관측을 누적하는 채널은 ID 1–15이며, 출력의 17개 채널 중
   0과 16에는 투영 질량을 누적하지 않는다.

3. **각 유효 픽셀을 depth로 3D 공간에 역투영한다.**
   Augmentation이 반영된 카메라별 `lidar2img` 행렬을 역변환하여 픽셀 `(u,v)`에
   대응하는 ray와 카메라 원점을 얻는다. `camera_z` 설정에서 구현식은 다음과 같다.

   ```text
   A = inverse(lidar2img)
   ray = A[:3, :3] @ [u, v, 1]
   xyz = depth * ray + A[:3, 3]
   ```

   여기서 depth는 optical-axis depth이므로 ray를 단위 길이로 정규화하지 않는다.
   선택 가능한 `ray_distance` 모드는 ray를 정규화하지만 현재 config에서는 사용하지
   않는다. 이렇게 얻은 점 중 유한하고 `pc_range`의 XYZ 하한 이상·상한 미만인 점만
   유지한다. 높이 z는 이 공간 필터에 사용하며, 최종 출력은 XY 평면의 2D BEV이다.

4. **Semantic surface를 BEV에 bilinear splatting한다.**
   3D 점의 XY를 BEV 연속 좌표로 변환한다. 현재 cell 크기는 양 축 모두 2.048 m이며,
   bilinear 모드에서는 cell 중심에 정수 좌표가 오도록 0.5를 뺀다.

   ```text
   gx = (x - xmin) / (xmax - xmin) * bev_w - 0.5
   gy = (y - ymin) / (ymax - ymin) * bev_h - 0.5
   ```

   `fx`, `fy`를 각 좌표의 소수 부분이라고 하면 주변 네 cell에 각각
   `(1-fx)(1-fy)`, `fx(1-fy)`, `(1-fx)fy`, `fx*fy`의 가중치를 분배한다.
   각 가중치는 해당 픽셀의 **semantic class 채널**에 `scatter_add_`로 누적한다.
   격자 밖 이웃은 버리며, 경계에서 버린 가중치를 다시 정규화하지 않는다.
   픽셀별 one-hot tensor를 명시적으로 생성하거나 BEV pillar의 여러 높이에서
   semantic map을 샘플링하는 과정은 없다.

5. **카메라별 누적 질량을 조정하고 여러 카메라의 관측을 합친다.**
   카메라 `v`, BEV cell `q`, class `c`의 누적값을 `H_v(q,c)`라고 하면,
   실제 구현의 집계는 다음과 같다. `eps` 기본값은 `1e-6`이다.

   ```text
   m_v(q) = sum_c H_v(q,c)
   P_v(q,c) = H_v(q,c) / max(m_v(q), 1.0)
   N(q) = sum_v 1[m_v(q) > eps]
   P(q,c) = sum_v P_v(q,c) / max(N(q), 1.0)
   ```

   한 카메라의 누적 질량이 1 이상이면 class histogram을 합 1로 정규화한다.
   1 미만이면 작은 관측 질량을 보존한다. 예를 들어 두 class의 누적값이
   `[0.125, 0.25]`이면 `[1/3, 2/3]`으로 키우지 않고 그대로 유지한다.
   이후 cell별 관측 카메라 수로 나눈다. 따라서 코드가 `probabilities`라고 부르는
   결과는 **모든 cell에서 채널 합이 반드시 1인 확률분포는 아니다**.
   관측이 없는 cell은 모든 채널이 0이며, 이를 free-space 또는 background의
   확정 label로 해석하지 않는다. 별도로 반환하는 `support=sum_v m_v`는 sampling
   mass이며 calibrated depth confidence가 아니다. `support`는 현재 UNet 조건에
   포함되지 않는다.

6. **Semantic BEV를 학습 가능한 multi-scale prior로 변환한다.**
   `project_semantics()`는 `@torch.no_grad()`로 실행되며 기하 투영에는 학습 가능한
   파라미터가 없다. 출력 `P`는 `prob_to_emb`의
   `Conv3×3(17→256) → SiLU → Conv1×1(256→256)`를 거치고,
   `[1,256,50,50]`의 학습 가능한 BEV positional embedding을 더한다.
   이어서 v3 파일에서 재사용하는 `SegBEVEncoder`가 scale `{1,2,4}`의 feature를
   생성한다. 현재 `channel_mult=[1,1,1]`에 따른 shape은 각각
   `[B,256,50,50]`, `[B,256,25,25]`, `[B,256,12,12]`이다.
   Embedding, positional embedding, condition encoder는 Stage 1에서 UNet과 함께
   학습되고, 이 multi-scale prior가 v5 UNet decoder의 SBAM 경로에 주입된다.

### Semantic BEV Cache and Training Path

Reference run `BEVDiffuser_tiny_onlyseg_sam3_metric3d`는 `use_semantic_bev_cache=True`로
학습되었다. 즉 위의 1-5단계(이미지 좌표 정렬, 3D 역투영, BEV splatting, 카메라 집계)는
**학습 루프가 아니라 오프라인에서 한 번 수행**되고, 학습 중에는 그 결과 `P`를 읽어
6단계의 학습 가능한 embedding/encoder부터 실행한다. Cache 모드에서는 위의 기하 투영 결과 `P`를
샘플마다 `[17,50,50]` float32 `<sample_token>.npy`로 미리 저장해 사용한다.
기본 위치는 `BEVFormer/data/semantic_bev`이다. Config의
`semantic_bev_cache_root='../../data/semantic_bev'`는
`BEVFormer/projects/bevdiffuser`를 실행 디렉터리로 하는 상대경로이고,
`train_only_seg_cache.sh`도 이 디렉터리에서 실행한다.

Dataset은 실제 반환된 현재 프레임의 token으로 `np.load(..., allow_pickle=False)`를
호출하고, shape와 floating dtype을 확인해 float32 tensor로 전달한다. Manifest,
소스 해시 또는 PyTorch/CUDA 버전 일치 검사는 학습 로딩에 필요하지 않다.
배치에서는 `semantic_bev_probabilities`라는 key를 사용하고, UNet에는
`semantic_prob=`로 전달한다. Cache 모드에서는 위 1–5단계의 raw map 로딩 및
투영을 건너뛰고 6단계의 `encode_probabilities()`부터 실행한다.

```text
Live:  SAM3 + Metric3D → 이미지 좌표 정렬 → 3D 역투영 → BEV splatting/집계 ─┐
                                                                          ├→ P → 학습 가능한 embedding/encoder → SBAM
Cache: <sample_token>.npy → np.load() ──────────────────────────────────────┘
```

Cache에는 argmax label, 학습된 embedding 또는 `support`를 저장하지 않는다.
따라서 cache를 사용해도 semantic embedding과 condition encoder의 학습은 유지된다.
현재 CFG dropout은 unconditional 입력으로 `P`와 같은 shape의 0 tensor를 사용하며,
이 역시 학습 가능한 embedding/encoder를 거친다. 확률 입력이 0이라는 것이 최종
condition feature도 0이라는 뜻은 아니다.

논문에서는 이 생성 방식을 **metric-depth-guided semantic surface lift-and-splat**로
설명할 수 있다. Metric3D depth는 픽셀의 3D 위치를 결정하는 기하 입력이며,
UNet에 독립적인 depth condition으로 주입되는 것이 아니다. 최종 diffusion condition은
이 관측 표면에서 구성한 dense semantic BEV의 학습 가능한 multi-scale 표현이다.

### Semantic BEV Prior Injection

- Multi-view SAM3 semantic map을 Metric3D depth와 함께 `SegBEVAligner`의
  역투영·splatting으로 BEV에 집계한다. Cache 모드에서는 이 투영 결과를 직접 읽는다.
- Semantic BEV probabilities를 학습 가능한 embedding과 encoder로 변환해
  multi-scale BEV prior로 사용한다.
- UNet의 attention block은 외부 layout context 없이 self-attention으로 동작한다.
- Semantic prior는 decoder의 SBAM path를 통해 spatially adaptive normalization 형태로 주입된다.
- 따라서 semantic condition은 object token을 attention으로 참조하는 수준이 아니라, BEV feature의 spatial statistics 자체를 semantic-dependent하게 조절한다.

### Condition Injection Path in the v5 UNet

`seg_diffusion_unet_v5.py` 기준으로 semantic prior가 UNet에 들어가는 경로는 두 가지다.

**1) Timestep-gated SBAM: decoder의 공간 조건 주입**

SBAM은 정규화된 UNet feature에 semantic BEV를 반영하는 SPADE 계열 연산이다. 각
`SBAM` 모듈은 현재 해상도의 semantic feature `P_r`로부터 **서로 다른** 3×3 convolution을
통해 `γ`(scale)와 `β`(shift)를 만든다. 한편 timestep embedding에는 50×50 semantic
feature의 global average pooling 결과가 더해진다(`use_global_seg=True`). 이 합성
embedding `e`가 현재 샘플과 timestep에 대한 gate 입력이다. 따라서 이름은
*timestep gate*지만 실제 출력은 timestep과 장면의 전역 semantic 정보에 함께 의존한다.

```text
e = time_embed(t) + global_semantic_embed(P_50)
γ = Conv3×3_γ(P_r),  β = Conv3×3_β(P_r)       # [B, C, H, W]
(g_γ, g_β) = 2 · sigmoid(Linear_zero(SiLU(e))) # 각각 [B, 1, 1, 1]
SBAM(x, P_r, e) = GroupNorm(x) · (1 + g_γ · γ) + g_β · β
```

**공간적 위치와 채널별 조절은 `γ/β`가 담당**하고, gate는 각 샘플에서 `γ`와 `β`의
전체 효과를 따로 조절하는 두 scalar다. Gate 자체는 cell별·채널별 attention map이
아니다. `time_gate`의 마지막 linear layer는 0으로 초기화되어 처음에는
`g_γ=g_β=2·sigmoid(0)=1`이다. 따라서 초기에는 ungated SBAM과 같은 크기의
modulation을 허용하고, 학습을 통해 각 gate가 `(0, 2)` 범위에서 변한다. 1보다 큰
값은 semantic modulation의 증폭을 뜻하며, gate의 값만으로 특정 위치의 semantic
정보를 얼마나 사용했는지는 알 수 없다.

Gate는 UNet 전체에 공유되는 하나의 값이 아니다. 각 해상도에는 decoder block이
3개씩 있으며 그중 첫째와 셋째(`i=0, 2`)가 `SBAMResBlock`, 둘째(`i=1`)가 일반
`ResBlock`이다. 각 `SBAMResBlock`에는
`norm_0`과 `norm_1`이 있고, **각 norm 지점마다 독립된 `time_gate`, `γ/β` convolution**이
있다. 현재 config(`image_size=50`, `channel_mult=[1,2,4]`, `num_res_blocks=2`)에서
해상도별 배치는 다음과 같다.

| Decoder 해상도 | 선택되는 semantic prior | SBAMResBlock 수 | 독립된 gate 쌍(`norm_0`/`norm_1`) |
| --- | --- | ---: | ---: |
| 12×12 | `P_12` | 2 | 4 |
| 25×25 | `P_25` | 2 | 4 |
| 50×50 | `P_50` | 2 | 4 |

즉 총 6개 `SBAMResBlock`의 12개 norm 지점이 각각 `g_γ`와 `g_β`를 계산한다.
해상도·블록·`norm_0`/`norm_1`에 따라 gate 값이 달라질 수 있는 이유는 **모듈마다
별도의 파라미터**를 갖기 때문이다. 같은 지점에서도 `e`에 timestep이 들어가므로
denoising step에 따라 값이 바뀔 수 있다. 다만 특정 timestep이나 해상도에서 gate가
반드시 커져야 한다는 제약은 없다.

Stage 2의 DDIM sampling에서는 semantic prior를 조건부·무조건부 각각 한 번 encoding해
여러 denoising step에 재사용한다. 같은 샘플의 `γ/β` spatial map은 그 동안 유지되고,
timestep이 바뀔 때 `e`를 다시 계산하므로 gate를 통한 modulation 강도는 달라질 수 있다.
이것이 gate가 수행하는 **denoising 단계별 조건 강도 조절**이며, 새로운 semantic BEV를
매 step 다시 만드는 과정은 아니다.

`SBAMResBlock`에서는 `norm_0`이 첫 convolution 전에, `norm_1`이 두 번째 convolution
전에 적용된다. Gate는 이 두 정규화 지점의 semantic scale/shift를 조절하며,
residual skip connection 전체를 직접 곱하는 계수는 아니다. 현재 config의 일반
`use_scale_shift_norm=True`는 다른 `ResBlock`의 timestep 처리 설정이고,
SBAM용 `sbam_use_scale_shift_norm`은 지정되지 않아 기본값 `False`다. 따라서
`norm_1` 뒤에 별도의 timestep scale/shift를 추가하는 경로는 현재 SBAM 블록에서
사용하지 않는다.

```python
for module in self.input_blocks:   h = module(h, emb)            # spatial condition 없음
h = self.middle_block(h, emb)                                    # spatial condition 없음
for module in self.output_blocks:  h = module(h, emb, seg_bev)   # SBAM 주입
```

따라서 encoder와 middle block은 semantic BEV를 **공간적으로 보지 못한다.**
`_select_ctx`가 각 decoder block의 `ctx_ds`(1/2/4)에 맞는 scale을 골라 전달한다.

**2) Global semantic embedding (`use_global_seg=True`)**

```python
global_emb = seg_global_embed(seg_bev_maps[1])  # GAP -> LayerNorm -> Linear -> SiLU -> zero_module(Linear)
emb = emb + global_emb                          # timestep embedding에 가산
```

가장 fine한 scale(`50 × 50`)을 global average pooling해 timestep embedding에 더한다.
마지막 layer가 `zero_module` 초기화이므로 학습 초기에는 no-op이며, **encoder가 semantic
정보를 받는 유일한 경로**가 이것이다.

**Cross-attention은 없다.** `context=None`일 때 cross-attention branch가 두 번째
self-attention으로 퇴화한다는 이유로 제거되었고, attention block은
`SelfAttnTransformer`다. 결과적으로 모든 conditioning은 SBAM과 timestep embedding
두 경로로만 이루어진다. 이 설계는 "object token을 attention으로 참조"하는 original
BEVDiffuser와 구조적으로 다른 지점이므로 논문에서 대비해 기술할 수 있다.

### Diffusion Formulation and Training Objective

아래는 reference run `results/bevdiffuser/BEVDiffuser_tiny_onlyseg_sam3_metric3d`의 실제
설정이며, `train_bev_diffuser_only_seg.py`의 학습 루프와 해당 run의 `hparams.yml`을
기준으로 한다.

**Latent space에 VAE가 없다.** Diffusion은 BEV feature space에서 직접 동작한다.

```python
latents = bev_model(return_loss=False, only_bev=True, **batch).detach()
latents = latents.reshape(-1, 50, 50, 256).permute(0, 3, 1, 2)   # [B, 256, 50, 50]
```

Clean target은 frozen `ckpts/bevformer_tiny_epoch_24.pth`의 BEV 출력이다. 이 모델은
`eval()` + `requires_grad_(False)`로 Stage 1 내내 고정되며, 별도의 latent
encoder/decoder를 두지 않는다. 즉 diffusion의 state space가 곧 256채널 `50 × 50`
BEV feature map이다.

**Forward process**는 SD2.1의 `DDPMScheduler` 설정을 그대로 사용한다
(`num_train_timesteps=1000`). 샘플마다 `t ~ U{0,...,999}`를 뽑아
`add_noise(latents, noise, t)`로 noisy BEV feature를 만든다.

**Objective는 diffusion loss 단독이다.**

| 항목 | 값 |
| --- | --- |
| `prediction_type` | `sample` (x0-prediction) |
| Loss | `F.mse_loss(model_pred, target)`, 가중치 없음 |
| `task_loss_scale` | `0.0` |
| Timestep 재가중 | 없음 (min-SNR-gamma 등 미적용) |
| Loss masking | 없음 (관측/미관측 cell 구분 없이 모든 channel·cell 동일 가중) |

```python
prediction_type = 'sample'          # target = 깨끗한 latents
denoise_loss = F.mse_loss(model_pred.float(), target.float(), reduction="mean")
total_loss   = denoise_loss + args.task_loss_scale * task_loss   # task_loss_scale = 0.0
```

`sample` 예측이므로 UNet은 noise `eps`나 velocity `v`가 아니라 **clean BEV feature를
직접 회귀**한다. `get_task_loss`(예측 BEV를 frozen task head에 다시 통과시키는 경로)는
코드에 존재하지만 계수가 `0.0`이라 gradient에 기여하지 않는다. 따라서 Stage 1의 학습
신호는 "주어진 semantic BEV prior 아래에서 frozen BEV feature를 얼마나 잘 복원하는가"
하나뿐이며, downstream task 신호는 Stage 1에 전혀 들어오지 않는다. 이는 앞 절에서
"task head가 없다"고 기술한 내용의 구현 수준 근거이기도 하다.

비교 대상인 original BEVDiffuser teacher(`BEVDiffuser_BEVFormer_tiny_original_bs4`)는
`task_loss_scale=0.1`로 학습되었다. 즉 두 teacher는 conditioning뿐 아니라
**training objective에서도 다르다.** Ablation과 논문 기술에서 이 축을 명시해야 하며,
두 teacher를 비교할 때 이 차이를 통제하지 않으면 conditioning 효과만 분리되지 않는다.

**Classifier-free guidance dropout**은 `uncond_prob=0.2`이고, 구현이 배치 단위다.

```python
def get_segmaps_cond(segmaps):
    if np.random.rand() < args.uncond_prob:   # 스칼라 난수 1개
        cond = torch.zeros_like(segmaps)
    else:
        cond = segmaps
    return cond
```

`np.random.rand()`를 샘플별이 아니라 step마다 한 번만 뽑으므로, dropout이 발생하면
해당 rank의 배치 전체가 동시에 unconditional이 된다. Dropout이 적용되는 대상은 경로에
따라 다르다. Cache 모드(reference run)에서는 load한 `[B,17,50,50]` 확률 tensor
`semantic_prob`을 직접 0으로 만들고, live 모드에서는 semantic ID map을 0으로 채워
`labels > 0` 조건이 모두 실패하게 만든다. 어느 쪽이든 `P`가 전부 0이 되며, `prob_to_emb`와
BEV positional embedding을 거쳐 **학습된 상수 BEV**가 unconditional 표현으로 작동한다.

### Output

Stage 1의 결과물은 semantic-only condition을 받은 denoised BEV feature를 생성할 수 있는 frozen diffusion teacher이다. 이 teacher는 Stage 2에서 downstream BEV model의 feature distillation target으로 사용된다. 따라서 Stage 1의 가치는 standalone detection score가 아니라, 여러 downstream BEV task에서의 transfer 성능으로 검증한다.

### Key Implementation Files

- `BEVFormer/projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d.py`
- `BEVFormer/projects/bevdiffuser/train_only_seg_cache.sh` (실행 스크립트. 현재 `RUN_NAME`과 task loss 설정은 reference run과 다르다)
- `BEVFormer/projects/bevdiffuser/build_semantic_bev_cache.py` (semantic BEV 사전 추출)
- `BEVFormer/projects/bevdiffuser/train_bev_diffuser_only_seg.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_diffusion_unet_v5.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_bev_aligner_one_hot_v5.py`
- `BEVFormer/projects/bevdiffuser/layout_diffusion/seg_bev_aligner_one_hot_v3.py` (`SegBEVEncoder` 재사용)
- `BEVFormer/projects/bevdiffuser/data_utils.py` (`CustomNuScenesDiffusionDataset_seg_depth_v2`)

### Reference Run: `BEVDiffuser_tiny_onlyseg_sam3_metric3d`

이 run이 현재 Stage 1의 기준 구성이며, Stage 2에서
`DiffBEVFormer_tiny_onlyseg_metric3d_t100_alpha100_lambda_0.55`의 teacher로 사용된다.

| 항목 | 값 |
| --- | --- |
| Stage-1 config | `bev_tiny_onlyseg_sam_v2_metric3d.py` |
| UNet | `seg_diffusion_unet_v5.DiffusionUNetModel` |
| Semantic BEV aligner | `seg_bev_aligner_one_hot_v5.SegBEVAligner` |
| Frozen BEV target | `ckpts/bevformer_tiny_epoch_24.pth` |
| `use_semantic_bev_cache` | `True` (사전 추출된 semantic BEV를 load) |
| `prediction_type` | `sample` |
| `task_loss_scale` | `0.0` |
| `uncond_prob` | `0.2` (배치 단위 dropout) |
| Optimizer | AdamW, betas `(0.9, 0.999)`, wd `0.01`, eps `1e-8` |
| Learning rate | `1e-4`, `constant` scheduler |
| Gradient clipping | `1.0` |
| Steps / batch | 50,000 steps, `4/GPU x 4 GPU = 16` |
| Precision / seed | fp32, seed `0` (`deterministic: false`) |
| Checkpointing | 10,000 step 간격, 최대 3개 보관 |
| Stage 2에서 사용 | `checkpoint-50000` |

#### Reproducibility 주의사항

**1) Flagship run은 gated SBAM으로 학습되었으며, 현재 소스에도 gate가 활성화되어 있다.**

`checkpoint-50000/unet/diffusion_pytorch_model.safetensors`의 key를 비교하면 다음과 같다.

| run | 총 tensor 수 | `time_gate` key 수 |
| --- | --- | --- |
| `BEVDiffuser_tiny_onlyseg_sam3_metric3d` | 635 | **24** |
| `BEVDiffuser_tiny_onlyseg_sam3_metric3d_no-gate` | 611 | 0 |

즉 flagship은 `gate = 2 * sigmoid(time_gate(emb))`를 `gamma`/`beta`에 곱하는
timestep-gated SBAM으로 학습되었다. **현재 워킹트리**의
`seg_diffusion_unet_v5.py` 활성 `SBAM`에도 이 gate가 있다. 다만 두 과거 run이
**같은 config 파일**을 사용한 사실은 변하지 않는다. 당시 no-gate 비교는 소스 교체로
구분되었으므로, run 재현 시에는 덤프 config와 checkpoint key를 함께 확인해야 한다.

> 재현성 개선 과제: gate 사용 여부를 명시적 config flag로 분리해 앞으로의 ablation을 기록한다.

**2) Ablation 축이 config에 드러나지 않거나 두 개 이상 섞여 있다.**

| Ablation | 의도한 축 | 실제로 함께 바뀌는 것 |
| --- | --- | --- |
| `no-gate` | SBAM timestep gate 제거 | config 동일, 소스 차이로만 구분 |
| `concat` | SBAM 대신 input concat | UNet 모듈이 `seg_diffusion_unet_v5_2`로 교체 (`use_semantic_bev_cache`는 양쪽 모두 `True`로 동일) |

Semantic BEV 생성 경로는 두 run 모두 cache(`True`)로 동일하므로, `concat` 비교에서
바뀌는 것은 UNet의 condition 주입 방식뿐이다.

**3) Cache 설정은 과거 run의 덤프 config를 기준으로 확인한다.**

`bev_tiny_onlyseg_sam_v2_metric3d.py`의 현재 워킹트리는 `True`이고,
run 시점에도 `True`였다. 근거는 mmdet이 Stage 2 실행 시각(2026-09-14 01:01)에 work_dir로
덤프한 config로, 거기에는 `use_semantic_bev_cache = True`,
`semantic_bev_cache_shape = (17, 50, 50)`이 기록되어 있다. Cache `.npy` 파일이
2026-09-09~11에 생성된 것과도 일치한다. 과거에는 이 값이 달랐던 워킹트리 상태가
존재했으므로 **config 파일의 현재 값만 보고 run 설정을 추정하면 안 되며,
각 run의 work_dir에 덤프된 config가 정확한 기록이다.**

**4) `train_only_seg_cache.sh`는 현재 reference run과 다른 실험을 가리킨다.**

현재 `RUN_NAME=BEVDiffuser_tiny_onlyseg_sam3_metric3d_task-loss`,
`TASK_LOSS_SCALE=0.1`이다. Reference run의 diffusion-only objective를 재현하려면
실행 명령에서 `task_loss_scale=0.0`과 별도 run 이름을 명시해야 한다.

### How to Describe Stage 1 in the Paper

Stage 1은 "GT layout을 semantic으로 보강한 diffusion teacher"가 아니라, **task head 없이 GT layout conditioning을 dense semantic BEV prior로 대체해 학습한 generative BEV diffusion teacher**로 설명한다. 이 점이 novelty의 중심이다.

다만 clean BEV target 자체는 detection-trained BEVFormer에서 오기 때문에, claim은 "fully task-agnostic representation learning"보다 **task-agnostic conditioning interface** 또는 **task-transferable BEV diffusion teacher**가 더 안전하다.

## Stage 2: Downstream BEV Task Training with Semantic Diffusion Distillation

Stage 2에서는 Stage 1의 diffusion teacher를 고정하고 downstream BEV student를 학습한다. 현재 구현은 BEVFormer-tiny **3D detection** student에 대한 것이며, BEV segmentation으로의 전이는 별도 실험으로 검증해야 한다.

### Goal

Student가 task loss를 받으면서 teacher가 생성한 BEV feature를 보조 supervision으로 사용하도록 한다. 생성 feature의 낮은 MSE만으로 downstream 성능 향상을 주장하지 않고, 최종 task metric으로 transfer를 검증한다.

### Training Flow

- Student BEVFormer는 multi-view image에서 BEV feature `S`와 detection 결과를 만든다. **Detection task loss는 마스킹하지 않은 `S`**로 계산한다.
- 고정된 target BEV model의 BEV feature에 매 호출 새로운 Gaussian noise를 더하고, Stage 1 UNet에 semantic BEV prior를 조건으로 주어 DDIM으로 복원한다. 생성 결과 `T`는 detach하여 distillation target으로 사용한다.
- Student는 detection task loss와 `MGD(S,T)`를 함께 받는다. MGD generator와 teacher는 추론 경로에 사용하지 않는다.

### Distillation Mechanism

현재 detection 경로의 MGD(Masked Generative Distillation)는 **spatial cell mask**를 쓴다. 각 위치 `(h,w)`를 확률 `λ`로 선택해 그 셀의 **256개 채널 전체를 0**으로 만든다(`use_spatialchannel=False`). 마스크는 student BEV의 MGD 경로에만 적용된다. Masked feature는 `3×3 Conv → ReLU → 3×3 Conv` generator `G`를 거치고, 복원 결과를 teacher의 전체 BEV와 비교한다. 정렬용 1×1 convolution은 현재 사용하지 않는다(`mgd_align=None`).

```text
T = stop_grad(DDIM(noise(target_BEV), semantic_BEV; t, CFG))
M[b, 1, h, w] ∈ {0, 1},  P(M = 0) = λ
L_MGD = α · mean((G(S ⊙ M) − T)²)
L_train = 0.5 · L_detection + 0.5 · L_MGD
```

`L_train`의 0.5/0.5 비율은 detector의 `_parse_losses_mix(weight=0.5)`에 따른다. 따라서 `α`는 MSE에 곱한 계수이고, 실제 optimizer loss에서는 그 값에 다시 0.5가 곱해진다. 마스킹된 셀의 student 입력에는 MGD 경로의 직접 gradient가 없지만, generator가 주변 셀을 활용해 복원하며 detection task loss는 원래 `S` 전체에 전달된다.

| 하이퍼파라미터 | 역할 | 대표 detection run | 현재 Stage 2 config |
| --- | --- | ---: | ---: |
| `noise_timesteps` | target BEV에 noise를 추가하는 timestep | 100 | 300 |
| `denoise_timesteps` | DDIM schedule의 시작 범위 | 100 | 300 |
| `num_inference_steps` | DDIM 복원 step 수 | 5 | 5 |
| `cfg_scale` | DDIM 각 step에서 `p_uncond + s(p_cond − p_uncond)`의 `s` | 덤프 config에 없음 (현재 sampler 기본값 2.0; 당시 값 미확인) | 4.0 |
| `mgd_alpha` (`α`) | 복원 MSE의 크기 | 100 | 100 |
| `mgd_lambda` (`λ`) | spatial cell의 마스킹 확률 | 0.55 | 0.55 |

`t`와 CFG는 생성 teacher BEV의 내용과 변동성을 바꾸고, `α`와 `λ`는 student로 전달되는 KD 신호의 크기와 위치를 바꾼다. 따라서 이 값들은 Stage 2의 핵심 실험 축이다. 대표 run의 `t=100, α=100, λ=0.55`와 **현재 워킹트리 config의 `t=300, CFG=4`를 같은 설정으로 취급하지 않는다**. 대표 run의 CFG는 덤프 config에 없어 당시 sampler 소스나 실행 기록으로 추가 확인해야 한다. MGD가 단순 MSE보다 좋은지는 같은 teacher와 sampling 설정을 맞춘 ablation으로 판단한다.

### Key Implementation Files

- `BEVFormer/projects/configs/diff_bevformer/bev_tiny_onlyseg_sam_metric3d.py`
- `BEVFormer/tools/train_onlyseg.py`
- `BEVFormer/tools/bevdiffuser_onlyseg.py` (noise, DDIM, CFG)
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

모든 수치는 각 run의 `*.log.json`에 기록된 **epoch 24** nuScenes val 평가값이며, 설정은 각 work dir에 덤프된 config에서 확인했다. 모든 run은 run당 1회(단일 seed)이다.

### Runs

| ID | Run (`results/diff_bevformer/...`) | Stage-1 teacher (`results/bevdiffuser/.../checkpoint-50000`) | Teacher conditioning / injection | Distill loss | Noise / denoise t |
|---|---|---|---|---|---|
| B | `baseline/DiffBEVFormer_tiny_original/DiffBEVFormer_tiny_original_rerun` | `BEVDiffuser_BEVFormer_tiny_original_bs4` | GT layout tokens (`LayoutDiffusionUNetModel`) | MSE | 5 / 5 |
| B-MGD | `baseline/DiffBEVFormer_tiny_original_mgd` | `BEVDiffuser_BEVFormer_tiny_original_bs4` | GT layout tokens | MGD (α=100, λ=0.55) | 5 / 5 |
| **Main** | `DiffBEVFormer_tiny_onlyseg_metric3d_t100_alpha100_lambda_0.55` | `BEVDiffuser_tiny_onlyseg_sam3_metric3d` | Semantic BEV, gated SBAM (`seg_diffusion_unet_v5`) | MGD (α=100, λ=0.55) | 100 / 100 |
| A1 no-gate | `det_ablation/DiffBEVFormer_tiny_onlyseg_metric3d_no-gate_t100_alpha100_lambda_0.55` | `BEVDiffuser_tiny_onlyseg_sam3_metric3d_no-gate` | Semantic BEV, SBAM without timestep gate | MGD (α=100, λ=0.55) | 100 / 100 |
| A2 concat | `det_ablation/DiffBEVFormer_tiny_onlyseg_metric3d_concat_t100_alpha100_lambda_0.55` | `BEVDiffuser_tiny_onlyseg_sam3_metric3d_concat` | Semantic BEV, input concat (`seg_diffusion_unet_v5_2`) | MGD (α=100, λ=0.55) | 100 / 100 |
| A3 no-MGD | `det_ablation/DiffBEVFormer_tiny_onlyseg_metric3d_t100_no-mgd` | `BEVDiffuser_tiny_onlyseg_sam3_metric3d` (Main와 동일) | Semantic BEV, gated SBAM | MSE (`use_mgd=False`) | 100 / 100 |

공통: student `DiffBEVFormer`(B, B-MGD) / `DiffBEVFormerSegV3`(semantic runs), `samples_per_gpu=8`, 24 epochs, 4 GPU. Semantic runs는 모두 `use_semantic_bev_cache=True`이며, 덤프된 config에 `use_spatialchannel` 설정은 없다(현재 워킹트리 config에는 `use_spatialchannel=True`가 추가되어 있으므로 이 run들과 다르다).

### Overall Detection Metrics (epoch 24)

| ID | mAP ↑ | NDS ↑ | mATE ↓ | mASE ↓ | mAOE ↓ | mAVE ↓ | mAAE ↓ |
|---|---:|---:|---:|---:|---:|---:|---:|
| B (layout, MSE, t5) | 0.27850 | 0.38263 | 0.8667 | 0.2909 | 0.5788 | **0.6089** | 0.2208 |
| B-MGD (layout, MGD, t5) | **0.28724** | 0.38432 | 0.8588 | 0.2888 | 0.5936 | 0.6354 | 0.2165 |
| **Main** (semantic, gated SBAM, MGD, t100) | 0.28507 | **0.38732** | **0.8495** | 0.2871 | **0.5691** | 0.6284 | 0.2179 |
| A1 no-gate | 0.28399 | 0.38402 | 0.8666 | **0.2869** | 0.5693 | 0.6461 | **0.2109** |
| A2 concat | 0.28408 | 0.38047 | 0.8580 | 0.2887 | 0.6026 | 0.6550 | 0.2113 |
| A3 no-MGD (MSE) | 0.28118 | 0.38378 | 0.8549 | 0.2874 | 0.5983 | 0.6148 | 0.2127 |

### Pairwise Deltas

| Comparison | 바뀌는 축 | ΔmAP | ΔNDS | 비고 |
|---|---|---:|---:|---|
| Main − B | conditioning + injection + Stage-1 task loss + distill loss + t | +0.00657 | +0.00469 | 여러 축이 동시에 바뀜 |
| Main − B-MGD | conditioning + injection + Stage-1 task loss + t (distill loss 동일) | −0.00217 | +0.00300 | 가장 가까운 baseline 비교 |
| A3 − B | conditioning + injection + Stage-1 task loss + t (둘 다 MSE) | +0.00268 | +0.00115 | |
| B-MGD − B | MSE → MGD (layout teacher) | +0.00874 | +0.00169 | |
| Main − A3 | MSE → MGD (semantic teacher) | +0.00389 | +0.00354 | |
| A1 − Main | timestep gate 제거 | −0.00108 | −0.00330 | |
| A2 − Main | SBAM → input concat | −0.00099 | −0.00685 | |

### Class-wise AP (epoch 24, 4개 distance threshold 평균)

| Class | B | B-MGD | Main | A1 no-gate | A2 concat | A3 no-MGD |
|---|---:|---:|---:|---:|---:|---:|
| car | 0.4752 | 0.4779 | 0.4788 | **0.4823** | 0.4790 | 0.4769 |
| truck | 0.2244 | **0.2337** | 0.2210 | 0.2206 | 0.2242 | 0.2246 |
| construction_vehicle | 0.0629 | **0.0776** | 0.0631 | 0.0659 | 0.0674 | 0.0728 |
| bus | 0.2905 | **0.3322** | 0.3189 | 0.3211 | 0.3288 | 0.3123 |
| trailer | 0.1091 | 0.0827 | **0.1098** | 0.0939 | 0.0920 | 0.0859 |
| barrier | 0.3851 | 0.3968 | 0.3962 | 0.3903 | **0.4065** | 0.3987 |
| motorcycle | 0.2533 | **0.2767** | 0.2674 | 0.2686 | 0.2652 | 0.2506 |
| bicycle | 0.2295 | 0.2425 | **0.2540** | 0.2401 | 0.2277 | 0.2364 |
| pedestrian | 0.3484 | 0.3457 | **0.3510** | 0.3459 | 0.3502 | 0.3480 |
| traffic_cone | 0.4067 | 0.4066 | 0.3906 | **0.4109** | 0.3998 | 0.4055 |

### Previous Result (version2, 참고용)

이전 문서 버전은 `results/version2/stage2/DiffBEVFormer_tiny_original_24epoch`(baseline)와
`results/version2/stage2/DiffBEVFormer_tiny_onlyseg_sam_mgd_v4_t100_alpha100_lambda_0.6`
(teacher `results/version2/stage1/BEVDiffuser_tiny_onlyseg_sam3_v2/checkpoint-50000`)를 비교했다.
Epoch 24에서 baseline mAP 0.28838 / NDS 0.38708, semantic-only mAP 0.29088 / NDS 0.39031
(+0.00250 / +0.00323)이었다. 이 teacher는 위 Main run의 Metric3D teacher와 다르므로 위 표와
섞어서 비교하지 않는다. 또한 version2 baseline(mAP 0.28838)과 현재 baseline rerun B(mAP 0.27850)는
약 1.0 mAP 차이가 나는데, 두 run의 설정이 동일한지는 아직 확인하지 않았다. 동일하다면 이는
seed 간 변동만으로도 ~1 mAP 차이가 날 수 있다는 근거가 된다.

## Result Analysis

**1) Semantic-only teacher는 GT layout 없이 baseline 수준의 transfer를 유지한다.**
Main은 기본 BEVDiffuser(B) 대비 +0.66 mAP / +0.47 NDS이다. 그러나 가장 가까운 비교인 B-MGD(같은
distillation loss) 대비로는 −0.22 mAP / +0.30 NDS로, mAP는 낮고 NDS는 높다. 따라서 현재 근거로
지지되는 claim은 "GT layout conditioning을 제거해도 detection transfer가 **competitive하다**"이며,
"semantic conditioning이 layout보다 낫다"는 아니다. Main − B의 이득 중 상당 부분은 MGD에서 온다
(layout teacher에서도 MSE → MGD가 +0.87 mAP).

**2) 개선은 주로 geometry 쪽 error term에서 나타난다.**
Main은 6개 run 중 mATE(0.8495)와 mAOE(0.5691)가 가장 낮다. B-MGD 대비 mATE −0.0093, mAOE −0.0245이고,
mAVE는 B 대비 나빠진다(0.6089 → 0.6284). 이는 dense semantic BEV prior가 장면의 공간 구조를 제공하지만
velocity/attribute 신호는 직접 담지 않는다는 동기와 일관된다. NDS 우위는 이 localization/orientation
개선에서 온다.

**3) Ablation은 방향은 일관되지만 효과 크기가 작다.**
- Timestep gate(A1): 제거 시 −0.11 mAP / −0.33 NDS. mATE가 0.8495 → 0.8666으로 나빠지는 것이 NDS 하락의 주 원인이다.
- SBAM vs input concat(A2): concat 시 −0.10 mAP / −0.69 NDS로 NDS 하락이 가장 크며, mAOE(0.6026)와 mAVE(0.6550)가 크게 나빠진다. Spatially adaptive normalization 주입이 단순 concat보다 낫다는 근거이다.
- MGD vs MSE(A3): MSE로 바꾸면 −0.39 mAP / −0.35 NDS. Semantic teacher에서 MGD가 유효하다는 근거이며, Feature Alignment Analysis의 가설과 일치한다. 다만 layout teacher에서도 MGD 이득(+0.87 mAP)이 있으므로 "semantic teacher이기 때문에 MGD가 필요하다"는 주장은 이 결과만으로는 성립하지 않는다.

**4) Class-wise 결과는 혼재되어 있다.**
B 대비 Main은 bus(+0.028), bicycle(+0.025), motorcycle(+0.014), barrier(+0.011)에서 높고
traffic_cone(−0.016)에서 낮다. B-MGD 대비로는 truck, construction_vehicle, bus, motorcycle에서 낮고
trailer, bicycle에서 높다. 단일 seed이므로 class 단위 차이를 해석하는 데는 주의가 필요하다.

**5) 통계적 주의.**
모든 비교는 단일 seed이고, 대부분의 delta가 1 mAP 미만이다. Version2 baseline과 현재 baseline rerun
사이의 ~1 mAP 차이를 고려하면, 위 차이들은 run 간 변동 범위 안에 있을 가능성이 높다. 논문 claim에는
최소한 Main, B-MGD, B에 대한 반복 seed(예: 3 seeds)와 평균 ± std가 필요하다.

**6) 남은 confound.**
Main과 B-MGD는 distillation loss는 같지만 Stage 2 noise/denoise timestep(100 vs 5)과 Stage 1
`task_loss_scale`(0.0 vs 0.1)이 다르다. Conditioning 효과만 분리하려면 layout teacher를 t=100으로
distill한 run, 또는 semantic teacher를 t=5로 distill한 run이 필요하다.

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
