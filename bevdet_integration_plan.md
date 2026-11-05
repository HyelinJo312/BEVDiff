# BEVDet 적용 설계: Semantic BEV Teacher (Stage 1/2) 연결

작성: 2026-11-04. 범위는 설계와 사전 검증이며, 이 문서를 쓰는 동안 `BEVDet/` 아래 파일은 수정하지 않았다.
BEVDet 환경 셋업(info pkl, 1-iteration 학습·평가 시험)은 별도 세션이 진행 중이다.

## 0. 이 실험이 지지하는 claim

- BEVFormer(backward projection, 50×50)에 이어 **LSS 계열 forward projection(BEVDet, 128×128)**에서도
  같은 semantic conditioning interface와 task-head-free Stage 1 recipe가 동작하는지 검증한다.
- 이는 **backbone/view-transform 일반화** 근거이지 cross-task 근거가 아니다. Cross-task 주장은 여전히 BEV segmentation이 담당한다.
- BEVDiffuser 논문은 BEVDet 결과를 보고하지 않는다(BEVFormer-tiny/base, BEVFormerV2, BEVFusion만 보고).
  따라서 BEVDet에서 "BEVDiffuser 대비" 비교를 하려면 layout teacher를 직접 학습해야 한다(§5 C3).

## 1. 검증된 사실

### 1.1 BEVDet 쪽 (dev3.0, commit 26144be, `configs/bevdet/bevdet-r50.py`)

| 항목 | 값 / 근거 |
| --- | --- |
| Distillation 지점 | `extract_img_feat()`가 반환하는 `img_feats[0]` = `bev_encoder` 출력, `[B,256,128,128]`. CenterHead 입력과 동일 |
| BEV 격자 | x, y ∈ [-51.2, 51.2), 0.8 m/cell → 128×128 |
| 축 순서 | `ranks_bev = ... + y*X + x` (`view_transformer.py:269-272`) → **H=y, W=x** |
| 좌표계 | **keyego** = `ego2globals[:, 0]` (`bevdet.py prepare_inputs`). `cams[0]`=`CAM_FRONT_LEFT`이고 `Ncams=6`이면 순서 고정(`choose_cams`) |
| BEV augmentation | `BEVAug`: `bda = Flip @ (Scale @ Rot)`, rot ±22.5°, scale 0.95–1.05, flip_dx/dy 0.5. `tran_lim` 미설정 → t=0. Test에서는 identity |
| bda 적용 위치 | view transformer가 frustum 점에 `bda`를 곱함(`view_transformer.py:177-179`) → BEV feature 자체가 augmented frame |
| 이미지 augmentation | resize/crop/flip/rot은 `post_rots/post_trans`로 되돌리므로 BEV 좌표에는 영향 없음 |
| 학습 설정 | `samples_per_gpu=8`, AdamW lr 2e-4, 24 epochs |

참고: BEVDet의 GT box는 `CAM_FRONT` 시각의 ego pose로 만들고(`get_gt`), BEV feature는 `CAM_FRONT_LEFT` 시각의 ego pose를 쓴다. BEVDet 원본의 내부 불일치이며 여기서는 건드리지 않는다.

### 1.2 Semantic BEV cache 쪽 (BEVFormer)

| 항목 | 값 / 근거 |
| --- | --- |
| 위치 / 수 | `BEVFormer/data/semantic_bev/<sample_token>.npy`, 34,150 files, `[17,50,50]` float32 |
| 좌표계 | **LIDAR_TOP frame.** `lidar2img`를 `sensor2lidar`로 만든다(`BEVFormer/projects/mmdet3d_plugin/datasets/nuscenes_dataset.py:173-182`). aligner가 이 행렬로 역투영한다(`seg_bev_aligner_one_hot_v5.py:179`) |
| 축 순서 | `index = label*cells + iy*W + ix` (`seg_bev_aligner_one_hot_v5.py:147`) → **H=y, W=x**, 2.048 m/cell |
| 미관측 cell | 모든 채널 0 → resampling의 zero padding과 의미가 일치 |

> 이전 handoff 메모의 "semantic prior는 ego 좌표 기준 cache"는 **틀렸다.** 아래 1.3의 실측으로 확인했다.

### 1.3 실측 검증 (CPU only, nuScenes val에서 무작위 300 samples, car center 2,949개)

LIDAR_TOP→ego 변환은 yaw −90.0°(±0.1°), translation (0.97, 0, 1.84) m다.
LiDAR 시각 ego와 keyego(CAM_FRONT_LEFT 시각) 사이의 xy 거리는 median 0.22 m, max 0.51 m다.

Vehicle 채널(채널 1, GT car 위치에서 lift가 가장 큼)의 mass를 GT car center에서 샘플링한 결과:

| 가설 / 변환 | mean mass@car | frac(mass > 0.1) |
| --- | ---: | ---: |
| cache = LiDAR frame (원 격자 50×50) | 0.562 | **0.811** |
| cache = ego frame이라고 가정 | 0.064 | 0.117 |
| **제안 변환** (lidar→global→keyego→bda, 64×64로 resampling) | 0.535 | **0.818** |
| 대조: lidar→ego 변환 생략 (cache를 ego로 취급) | 0.063 | 0.123 |
| 대조: bda 생략 | 0.124 | 0.253 |

결론: cache는 LiDAR frame이고, 제안 변환을 쓰면 정렬이 원래 수준으로 돌아온다. lidar→ego 변환이나 bda를 빼먹으면 **에러 없이 조용히** prior가 틀린 위치에 놓인다.

### 1.4 UNet 이식 가능성 (CPU, dummy input, `jhl-bevdet` env)

- `seg_diffusion_unet_v5`와 `seg_bev_aligner_one_hot_v5`는 mmdet3d plugin을 import하지 않는다. `BEVFormer`를 `sys.path`에 넣으면 jhl-bevdet env에서 생성된다.
- `image_size=128`, `num_pre_downsample=1`, aligner `bev_h=bev_w=64`로 만들고 `[1,256,128,128]`를 넣으면 `[1,256,128,128]`가 나온다. 파라미터는 451.1M, `time_gate` tensor는 24개로 flagship과 같다.
- **버그 (수정 필요):** `SegBEVEncoder`(`seg_bev_aligner_one_hot_v3.py`)의 ds=4 conv가 `Conv2d(k=3, stride=2)`이고 **padding=0**이다.
  - 64×64 prior에서는 64/32/**15**가 나오지만 UNet decoder는 64/32/**16**을 기대한다 → `SBAM`의 `assert seg_cond.size()[2:] == x.size()[2:]`에서 실패한다.
  - 50×50에서는 25→12로 우연히 맞았다. 격자 크기가 4로 나눠 2가 남을 때(50, 54, 58, …)만 맞는다.
  - 테스트에서는 padding=1로 바꿔 64/32/16을 확인했다.
  - Flagship 출력이 바뀌면 안 되므로 **기본값 padding=0을 유지하는 config flag**로 노출해야 한다. padding은 파라미터가 아니라서 checkpoint key는 그대로다.

### 1.5 기존 Stage 2 프로토콜 (BEVFormer, dumped config 기준)

- Main(`DiffBEVFormer_tiny_onlyseg_metric3d_t100_alpha100_lambda_0.55`)과 B(`DiffBEVFormer_tiny_original_rerun`)는
  모두 `load_from='./ckpts/bevformer_tiny_epoch_24.pth'`에서 시작해 24 epochs를 더 학습한다.
- `baseline/bevformer_tiny`는 `load_from=None`으로 처음부터 24 epochs를 학습한다.
- 따라서 "BEVFormer-tiny 대비" 이득에는 **추가 24 epochs 효과가 섞여 있다.** BEVDet에서는 처음부터 이 대조군(C1)을 둔다.

## 2. Prior 정렬 설계

목표: cache `P_lidar [17,50,50]`(LiDAR frame) → augmented keyego frame의 `P' [17,G,G]`.

- augmented keyego frame의 target 점 `p'`에 대해 source 좌표는
  `p_lidar = T_keyego→lidar · bda⁻¹ · p'`로 구한다.
- `T_lidar→keyego = inv(keyego2global) @ ego2global(lidar ts) @ lidar2ego`.
- 2D로 근사한다(z 결합 무시). lidar2ego의 pitch(약 1.4°)와 z 범위 [-5, 3] m 때문에 생기는 오차는 0.2 m 이하로, cell(1.6 m)보다 훨씬 작다.
- 두 격자 모두 ±51.2 m를 덮으므로 normalized 좌표는 `metric/51.2`이다. `affine_grid` + `grid_sample`로 한 번에 계산한다.
- `grid_sample` 옵션: `align_corners=False`(cache의 bilinear splat에서 cell center가 `(i+0.5)·cell`인 정의와 일치), `padding_mode='zeros'`(미관측 = 0).
- P는 cell 안의 정규화된 class 분포라서 보간해도 되며, scale에 대한 Jacobian 보정은 필요 없다.

검증에 사용한 핵심 코드:

```python
def resample(P, M2, b2, G, R=51.2):        # output p' samples source at M2 @ p' + b2 (metric)
    theta = torch.zeros(1, 2, 3)
    theta[0, :, :2] = torch.as_tensor(M2, dtype=torch.float32)
    theta[0, :, 2] = torch.as_tensor(b2 / R, dtype=torch.float32)
    grid = F.affine_grid(theta, (1, P.shape[0], G, G), align_corners=False)
    return F.grid_sample(P[None], grid, mode='bilinear', padding_mode='zeros', align_corners=False)[0]

keyego2lidar = np.linalg.inv(inv(keyego2global) @ lidarego2global @ lidar2lidarego)
Binv = np.linalg.inv(bda[:3, :3])
P_new = resample(P_lidar, keyego2lidar[:2, :2] @ Binv[:2, :2], keyego2lidar[:2, 3], G=64)
```

구현 위치 제안: BEVDet dataset pipeline에 `BEVAug` **다음** 단계로 `LoadSemanticBEVPrior`를 추가한다(CPU, token으로 cache를 load하고 info의 calibration과 `bda_mat`을 사용).
출력 key는 `semantic_bev_probabilities` `[17,64,64]`이다. Stage 1과 Stage 2가 같은 구현을 쓰도록 해서 정렬 코드가 한 곳에만 있게 한다.

## 3. 해상도 설계

| 선택지 | UNet core | Prior 격자 | 비고 |
| --- | --- | --- | --- |
| **A (권장)** `image_size=128`, `num_pre_downsample=1` | 64×64 (1.6 m) | 64×64 | source 2.048 m와 비슷한 해상도. core 연산량은 tiny(50²)의 약 1.6배. BEVFormer-base 선례(`num_pre_downsample=2`, 200→50)와 같은 패턴 |
| B `num_pre_downsample=0` | 128×128 (0.8 m) | 128×128 | 연산량 약 6.5배. prior는 2.048 m 원본을 늘린 것이라 정보가 늘지 않음 |
| C cache를 0.8 m로 재생성 | — | 128×128 | 34k samples × 17×128×128 (fp16 약 19 GB, fp32 약 38 GB). 대규모 cache 생성이므로 승인이 필요하고 1차 실험에는 불필요 |

A와 B 모두 §1.4의 `SegBEVEncoder` padding flag가 필요하다(128에서도 64→**31**로 어긋남).

## 4. Stage 1 / Stage 2 설계

### Stage 1 (BEVDet target)

- env: `jhl-bevdet`. 기존 `train_bev_diffuser_only_seg.py`는 BEVFormer 전용 API(`only_bev=True`, plugin dataset builder)에 묶여 있으므로 **BEVDet용 학습 script를 새로** 만든다.
- latents: frozen BEVDet(`eval()`, `requires_grad_(False)`)의 `extract_img_feat(img_inputs, img_metas)[0][0]`, `[B,256,128,128]`.
- data: BEVDet **train pipeline을 bda 포함 그대로** 사용하고, 그 뒤에 `LoadSemanticBEVPrior`를 붙인다.
  - Stage 2에서 teacher가 bda가 적용된 BEV를 받으므로 Stage 1도 같은 분포로 학습한다.
  - BEVFormer Stage 1에는 BEV augmentation이 없었다 → 새로 생기는 차이이므로 문서에 명시한다.
- objective는 flagship과 같다: x0-prediction MSE, `task_loss_scale=0`, `uncond_prob=0.2`(batch 단위), 50k steps, bs 16, lr 1e-4 constant, fp32, seed 0.
- frozen target: **공식 BEVDet-R50 가중치 `BEVDet/ckpts/bevdet-r50.pth`**(2026-11-04 결정, EMA epoch_24, 출처·검증은 `BEVDet/ckpts/PROVENANCE.md`).
- target scale 점검: BEVFormer Stage 1은 BEV feature를 정규화 없이 x0로 썼다. BEVDet `img_feats[0]`의 분포(mean/std/abs-max)가 BEVFormer-tiny와 다르면 t=100의 SNR이 달라져 "t=100"이 같은 조건이 아니게 된다. 학습 전에 train ~200 samples로 통계를 재고, 필요하면 `target_scale` config flag(기본 1.0)를 둔다.

### Stage 2 (DiffBEVDet)

- student: 공식 가중치(C0)에서 시작해 24 epochs를 더 학습한다(§1.5 프로토콜과 동일). frozen target은 별도 copy로 둔다.
  - schedule은 `bevdet-r50.py`와 같다: AdamW lr 2e-4, wd 1e-7, grad_clip max_norm 5, MEGVIIEMAHook. C1과 C2에서 EMA 설정을 똑같이 둔다.
  - grad_clip 때문에 loss 배율이 결과에 영향을 준다. C1의 det loss 가중치(1.0 vs 0.5)는 BEVFormer A3(no-MGD) dumped config와 같은 규칙으로 정한다.
- teacher feature: target BEV에 noise(t=100) → DDIM 5 steps + CFG로 복원. **CFG 값은 명시적으로 고정하고 기록한다.** Flagship run의 CFG는 dumped config에 없어 미확인이다.
- MGD: spatial cell mask λ=0.55(채널 공유), 3×3 Conv–ReLU–3×3 Conv generator, α=100. `L = 0.5·L_det + 0.5·L_MGD`.
  - detection loss는 마스킹하지 않은 `img_feats[0]`로 계산한다.
- 추론에는 원래 BEVDet만 쓴다.

## 5. 비교군 (matched)

| ID | 구성 | 목적 |
| --- | --- | --- |
| C0 | 공식 BEVDet-R50(`ckpts/bevdet-r50.pth`), 학습 없음. val 평가로 README 28.3/35.0 재현 확인 | baseline 겸 Stage 1/2 target |
| C1 | C0 + 24ep 계속 학습, distillation 없음 | 추가 epoch 효과 분리. **필수** |
| C2 | C0 + Ours (semantic teacher, MGD) 24ep | main |
| C3 (선택) | C0 + BEVDiffuser layout teacher(BEVDet feature로 학습, 원래 설정 task_loss 0.1) 24ep | BEVDet에서 "vs BEVDiffuser"를 주장하려면 필요. 비용 큼 |

delta가 1 mAP 미만일 가능성이 높으므로 C1/C2는 가능하면 seed 2–3개로 돌린다.

## 6. 구현 순서와 검증 (bev_engineer용)

1. `SegBEVEncoder` ds=4 padding을 config flag로 노출한다(기본 0).
   - test: 50 격자 출력이 bit-identical, checkpoint key 동일 / 64·128 격자에서 scale이 UNet 크기와 일치.
2. `LoadSemanticBEVPrior` pipeline 단계. tests:
   - identity(bda=I, T=I)이면 bilinear upsample과 같음
   - synthetic hot cell이 예상 위치로 이동
   - flip/rotation이 GT box 변환과 일치
   - real-data GT-car 정렬 재현(§1.3, frac≥0.8)
3. BEVDet Stage 1 script → 1-batch smoke test(shape, gradient가 UNet/encoder에만 흐르는지, target이 frozen인지).
4. DiffBEVDet detector + config → 1-iteration smoke test(loss key, 0.5/0.5 가중치, teacher·target frozen).
5. ~~`tools/test.py` distributed 경로 패치~~ → 셋업 세션이 적용 완료(`tools/test.py:245-247`).

코드는 모두 `BEVDet/projects`(BEVFormer/projects 복사본, `BEVDet/projects/COPY_PROVENANCE.md`)에서 수정한다. BEVFormer 원본은 건드리지 않는다.
`LoadSemanticBEVPrior`는 `img_inputs[-1]`의 4×4 `bda_mat` 전체(회전·scale·flip·translation)를 써야 한다. 현재 config의 translation은 0이지만 코드에서 가정하지 않는다.

## 7. 결정이 필요한 사항

- ~~Stage 1/2 target~~ → 공식 BEVDet-R50 가중치로 결정(2026-11-04).
- Stage 2 CFG 값(flagship 값 미확인).
- C3(BEVDet용 layout teacher)를 돌릴지.
- GPU: 현재 8장 모두 다른 run이 사용 중이다(약 98% util). C0(8 GPU, 24ep), Stage 1(50k steps), C1/C2 모두 승인이 필요한 full training이다.

## 8. 셋업 세션 변경 사항 검토 (02:22–02:32 기준)

- `mmdet3d/__init__.py`: mmcv 상한을 1.7.2로 변경. 문제 없음.
- `mmdet3d/apis/train.py`: DDP wrapper에 `_use_replicated_tensor_module=False`를 설정해 mmcv 1.7.2와 PyTorch 2.7 조합을 맞춤. BEVFormer `tools/train_*`와 같은 패치로 문제 없음.
  - **단 `tools/test.py:241`의 `MMDistributedDataParallel` 경로에는 없어서** `dist_test.sh`로 평가하면 `forward`에서 AttributeError가 난다(mmcv `parallel/distributed.py:159-160`). 단일 GPU test(`MMDataParallel`)는 영향 없음.
- `mmdet3d/models/utils/spconv_voxelize.py`: spconv import를 선택적으로 바꿈. camera-only라 문제 없음.
- `tools/create_data_bevdet_infos_only.py`: 원본 `__main__`에서 `create_groundtruth_database`만 뺀 것과 같다. 출력은 `BEVDet/data/nuscenes/bevdetv3-nuscenes_infos_{train,val}.pkl`(로컬 디렉터리, 공유 dataset에는 쓰지 않음).
- `tools/dist_train_bevdet_r50.sh`: 기본값이 `CUDA_VISIBLE_DEVICES=0-7`(8 GPU)이다. 현재 GPU가 모두 사용 중이므로 실행 전 확인이 필요하다. `--seed 0`, `--deterministic` 없음.
- `bev_pool_v2_ext` 빌드: `cuobjdump`로 `sm_100` cubin 확인.
- `jhl-bevdet`: mmdet3d 1.0.0rc4(BEVDet editable). `jhl-bevdiff`는 0.17.1 그대로 유지.

## 9. 2026-11-04 업데이트: 복사·가중치 완료와 작업 순서

### 9.1 완료
- `BEVDet/projects` ← `BEVFormer/projects` working tree 복사(`__pycache__` 제외, 394 files, `diff -rq` 동일). 출처 commit과 uncommitted 상태: `BEVDet/projects/COPY_PROVENANCE.md`.
- `BEVDet/ckpts/bevdet-r50.pth`: 공식 dev3.0 BEVDet-R50. Qualcomm AI Hub가 Baidu 공식 파일을 재호스팅한 public S3에서 받았다. 검증 내용은 `BEVDet/ckpts/PROVENANCE.md`.
  - md5가 S3 ETag와 같다.
  - `bevdet-r50.py` 모델과 478/478 key·shape가 일치한다.
  - EMA update 수가 학습 schedule과 맞다.
  - **mAP/NDS 재현은 아직 하지 않았다.**
- `jhl-bevdet`에서 `BEVDet/projects`의 v5 UNet을 import해 CPU forward로 확인했다. 128×128×256 입력, 64×64 prior, ds4 padding을 런타임 patch한 상태에서 출력 (1,256,128,128).
- Semantic cache는 BEVDet info token을 train 28130/28130, val 6019/6019 모두 포함한다. info에 lidar2ego, ego2global, `cams[*].ego2global`이 모두 있다.

### 9.2 작업 순서
| # | 작업 | 자원 | 승인 |
| --- | --- | --- | --- |
| W1 | 공식 가중치 val 평가 (기대 28.3/35.0). 결과는 `results/bevdet/det/bevdet-r50_official/`에 쓴다. `dist_test_bevdet.sh`는 JSON을 checkpoint 폴더(`ckpts/`)에 쓰므로 prefix를 바꿔 실행한다 | GPU 1장, 수십 분 | 필요 |
| W2 | ds4 padding flag (`BEVDet/projects`만, 기본 0) + 단위 테스트 | CPU | - |
| W3 | `LoadSemanticBEVPrior` + 단위/실데이터 정렬 테스트(frac≥0.8) | CPU | - |
| W4 | BEVDet feature 통계(`img_feats[0]`) vs BEVFormer-tiny feature 통계 → `target_scale` 결정 | GPU 1장, 수 분 | 필요 |
| W5 | BEVDet Stage 1 script + config, 1-batch smoke test | GPU 1장, 수 분 | 필요 |
| W6 | Stage 1 full (50k steps, bs16) | multi-GPU, 장시간 | 필요 |
| W7 | DiffBEVDet + C1/C2 config, 1-iter smoke test | GPU 1장 | 필요 |
| W8 | C1·C2 full 24ep (seed 2–3개) + 평가 | 8 GPU × 2+ run | 필요 |

W2·W3는 GPU 없이 바로 할 수 있다. W1은 env, bev_pool_v2 sm_100 build, info pkl, 가중치를 한 번에 확인하는 첫 GPU 작업이다.

## 10. 2026-11-04 결정 변경: BEVDet4D-R50-CBGS를 직접 학습

§1–§9의 plain BEVDet-R50 내용은 이 절로 대체된다. 단, prior 정렬·해상도·UNet 설계는 그대로 유효하다.

### 10.1 결정
- 대상: **BEVDet4D-R50-CBGS** (`configs/bevdet/bevdet-r50-4d-cbgs.py`, 수정 없이 사용).
- 공식 4D 가중치는 Baidu에만 있어 구할 수 없다. 그래서 공식 recipe(CBGS 20 epoch, 8 GPU × 8, seed 0)로 직접 학습한 EMA `epoch_20_ema.pth`를 C0로 쓴다. C0가 clean target, Stage 1 source, Stage 2 student 초기값을 모두 맡는다.
- 재현 기준: config header의 `align_after_view_transfromation=False` 수치 mAP 0.3157 / NDS 0.4485.
- 4D 출력 `img_feats[0]`는 현재 frame의 keyego 좌표 + bda 기준 `[B,256,128,128]`이다. 따라서 §2 prior 정렬과 §3 해상도 설계는 바뀌지 않는다.
- Stage 2 CBGS 처리: 권장 기본값은 CBGS 유지 + 샘플 수를 맞춘 짧은 학습(≈5 epoch ≈ 일반 24 epoch)이다. C1과 C2는 같은 설정을 쓴다. 이 부분은 미확정이다.

### 10.2 jhl-bevdet 검증 (1 GPU smoke, 공식 config + 작은 subset, 2026-11-04)
- 데이터:
  - info는 scene별 시간순이다(간격 중앙값 0.5 s, 역순 0건). 이전 frame도 실제 인접 frame이다.
  - CBGS 확장 배율은 4.39배(28130→123580)이다.
  - 입력 크기는 `(12,3,256,704)`이다.
- 학습: DDP(torchrun) 1 epoch(temporal off)와 resume 후 1 epoch(temporal on)가 모두 정상 종료했다.
  - loss는 정상적으로 감소했다(heatmap 1490 → 2.8).
  - GPU 메모리는 약 9 GB이고, checkpoint와 EMA 저장·복원도 정상이다.
- 평가 버그:
  - `tools/test.py:211-213`이 4D 모델의 `align_after_view_transfromation`을 True로 바꾼다.
  - 이 경로의 `bevdet.py:322` `gen_grid`가 3×3 bda를 가정해서, dev3.0의 4×4 bda에서 RuntimeError가 난다.
  - 해결: 항상 `--no-aavt`로 평가한다. 학습과 같은 경로다.
- EMA resume: `resume_from`만 주면 EMA가 다시 시작된다. `custom_hooks.0.resume=<epoch_N_ema.pth>`를 함께 줘야 하며, 이렇게 주면 정상 동작함을 확인했다.
- 수치 sanity: 공식 BEVDet-R50 가중치를 val 80 sample(2 scenes)에서 돌렸다. mAP 0.158, car AP 0.447이 나와 bev_pool_v2 sm_100 추론은 정상으로 보인다. 전체 val 평가는 아니다.

### 10.3 스크립트 (BEVDet/tools, 새 파일)
- `dist_train_bevdet4d_r50_cbgs.sh`:
  - 8 GPU만 허용한다(batch 64가 recipe의 일부).
  - 기본 출력은 `results/bevdet/det/bevdet4d-r50-cbgs_seed{SEED}`이다.
  - 기존 checkpoint가 있는 폴더에 새 run을 시작하지 않는다.
  - `RESUME_EPOCH=n`이면 model·optimizer·EMA를 함께 resume한다.
  - launch 정보(명령, git HEAD, working tree 상태)를 `launch_*.txt`에 남긴다.
  - `DRY_RUN=1`이면 명령만 출력한다.
- `dist_test_bevdet4d.sh`:
  - `--no-aavt --eval bbox`로 실행한다.
  - 결과는 checkpoint 옆 `eval_<ckpt>/`에 쓰고 `test_log.txt`를 남긴다.
  - 같은 출력 폴더가 이미 있으면 거부한다.

### 10.4 작업 순서 (갱신)
| # | 작업 | 자원 | 승인 |
| --- | --- | --- | --- |
| W0 | C0 학습: `bash tools/dist_train_bevdet4d_r50_cbgs.sh` (38,620 iter) → `dist_test_bevdet4d.sh`로 epoch_20_ema 평가 | 8 GPU, 단독 사용 시 약 6–11 h(추정) | 필요 |
| W2·W3 | ds4 padding flag, `LoadSemanticBEVPrior` + 테스트 (C0와 무관, 병행 가능) | CPU | - |
| W4–W8 | §9.2와 같다. target만 C0(4D)로 바뀐다. Stage 1 data는 4D pipeline(2 frame)을 쓰고, latents는 frozen C0의 `extract_img_feat(...)[0][0]`이다 | §9.2 | 필요 |

## 11. Stage 1 (BEVDet4D) 코드 수정 계획 (2026-11-04)

### 11.1 근거
- flagship Stage 1 `BEVDiffuser_tiny_onlyseg_sam3_metric3d`의 실제 인자는 tensorboard hparams에서 확인했다.
  - config는 `bev_tiny_onlyseg_sam_v2_metric3d.py`이고 `pretrained_unet_checkpoint=None`(scratch)이다.
  - batch 4×4 GPU, lr 1e-4 constant, 50k step이다.
  - `prediction_type=sample`, uncond 0.2, task loss 0, seed 0, clip 1.0, wd 0.01이다.
- 참고 스크립트 `train_only_seg_cache_det.sh`는 현재 **base run** 설정이다. base config에 tiny ckpt로 초기화하고, pre/post-downsample만 학습한다. BEVDet은 flagship(tiny, scratch)을 따른다.
- 현재 v5 UNet 소스는 flagship ckpt와 strict 일치한다(635/635, `time_gate` 24).
- `jhl-bevdet`에서 `projects.mmdet3d_plugin`, `data_utils`, `test_bev_diffuser_only_seg`는 import에 실패한다(`HungarianAssigner3D` registry 충돌). `semantic_bev_cache`, `model_utils`, `scheduler_utils`는 OK다.
- `stabilityai/stable-diffusion-2-1` scheduler는 HF가 401을 반환하고 로컬 캐시도 없다. 이 노드에서는 BEVFormer 쪽 스크립트도 시작 단계에서 실패한다.
- 학습 스크립트는 flagship 이후 cudnn off → cudnn on + TF32 on으로 바뀌었다.
- `BEVFormer/data/semantic_bev`는 2026-11-02에 재생성됐다(flagship 이후). [17,50,50] share, `output_frame=lidar`이며, flagship 당시 cache와 같은지는 미확인이다.

### 11.2 파일 (모두 `BEVDet/projects` 안, BEVFormer 원본 무수정)
| # | 파일 | 내용 |
| --- | --- | --- |
| F1 | `bevdiffuser/train_bev_diffuser_only_seg_bevdet.py` (새로 복사·수정) | (1) plugin, `data_utils`, BEVFormer evaluate import를 제거하고 `mmdet3d.datasets.build_dataloader`를 쓴다. (2) frozen BEVDet4D는 C0 EMA를 strict load한 뒤 `eval()`·`requires_grad_(False)`로 두고 DDP로 감싸지 않는다. (3) latents는 `extract_img_feat(img_inputs, img_metas)[0][0]` `[B,256,128,128]`이고, reshape/permute를 제거한다. (4) `task_loss_scale≠0`이면 에러를 낸다. (5) 학습 중 BEVFormer 평가를 제거한다. (6) TF32는 flag로 둔다. 나머지 루프(noise, x0 MSE, 배치 단위 CFG dropout, optimizer, checkpoint)는 그대로 둔다 |
| F2 | `bevdiffuser/bevdet_semantic_prior.py` (새 파일) | `LoadSemanticBEVPrior` pipeline: BEVAug 다음에 둔다. cache를 load해 `split_cache_array`/`semantic_input`을 적용하고, lidar→global→keyego(현재 frame CAM_FRONT_LEFT)→`bda_mat`(4×4) 순서로 warp한 뒤 64×64로 출력한다. 출력 key는 `semantic_bev_probabilities`(DC)다 |
| F3 | `layout_diffusion/seg_bev_aligner_one_hot_v3.py`, `_v5.py` | `SegBEVEncoder` ds4 conv padding flag를 추가한다(기본 0 = flagship과 동일) |
| F4 | `configs/bevdiffuser/bevdet4d_r50_onlyseg_sam_v2_metric3d.py` (새 파일) | `_base_`는 `bevdet-r50-4d-cbgs.py`다. `custom_imports`로 F2를 등록한다. train은 CBGS 없이 `NuScenesDataset`을 쓴다. UNet은 flagship과 같고 `image_size=128`, `num_pre_downsample=1`, aligner 64×64, ds4 padding 1만 바꾼다 |
| F5 | `bevdiffuser/train_only_seg_cache_det_bevdet4d.sh` (새 파일) | `train_only_seg_cache_det.sh` 기반이다. 4 GPU×4, flagship 하이퍼파라미터, UNet scratch를 쓴다. ckpt는 `results/bevdet/det/bevdet4d-r50-cbgs_seed0/epoch_20_ema.pth`다. 로컬 scheduler 경로, env 검사, 기존 출력 보호, launch 기록을 넣는다 |
| F6 | `bevdiffuser/sd21_scheduler/scheduler/scheduler_config.json` (새 파일) | SD-2.1 DDPM/DDIM 설정을 로컬에 둔다(scaled_linear 0.00085–0.012, 1000 step, `steps_offset` 1, `clip_sample` false, `set_alpha_to_one` false) |

### 11.3 검증 순서
1. (CPU) F3: 기본값에서 flagship ckpt strict load와 50×50 출력이 bit-identical인지, padding 1에서 64 입력의 scale이 {64,32,16}인지 확인한다.
2. (CPU) F2:
   - identity, synthetic hot cell, flip/rot을 GT box 변환과 비교하는 단위 테스트를 한다.
   - 실데이터 car-center 정렬이 frac≥0.8인지 확인한다(bda 포함).
3. (CPU) F4: config parse와 dataset build를 하고, sample 1개의 key와 shape(`[17,64,64]`, `(12,3,256,704)`)를 확인한다.
4. (GPU 1장, C0 전에도 가능) F1+F5를 smoke ckpt로 20 iter 돌린다.
   - loss가 finite인지 본다.
   - grad가 UNet에만 흐르고 frozen BEVDet은 변하지 않는지 확인한다.
   - CFG zero batch와 checkpoint save/resume을 확인한다.
5. (C0 후, GPU) W4 feature 통계를 내고 `target_scale`을 결정한다.
6. (승인 후) full Stage 1을 돌린다.

### 11.4 결정 (2026-11-04, 사용자)
- Stage 1 UNet은 scratch로 학습한다(flagship과 동일).
- Stage 1 데이터에는 CBGS를 쓰지 않는다(`NuScenesDataset` 자연 분포).
- augmentation과 TF32는 미정이다.

### 11.5 구현 완료 (2026-11-04, `BEVDet/projects` 안, BEVFormer 원본 무수정)
- 결정: augmentation A(BEVDet4D 학습 설정 그대로), TF32 기본 off(`--allow_tf32`로 켬), `--target_scale` 기본 1.0.
- 파일:
  - F1 `bevdiffuser/train_bev_diffuser_only_seg_bevdet.py`
  - F2 `bevdiffuser/bevdet_semantic_prior.py`
  - F3 `layout_diffusion/seg_bev_aligner_one_hot_v3.py`·`_v5.py`(`ds4_padding`, 기본 0)
  - F4 `configs/bevdiffuser/bevdet4d_r50_onlyseg_sam_v2_metric3d.py`
  - F5 `bevdiffuser/train_only_seg_cache_det_bevdet4d.sh`
  - F6 `bevdiffuser/sd21_scheduler/`(공개 mirror 2곳과 값 일치)
  - 테스트 `bevdiffuser/tests/test_bevdet_stage1.py`
- CPU 테스트 10/10 통과:
  - flagship ckpt strict load
  - 64 입력에서 prior scale 64/32/16
  - config가 base·flagship과 허용 키만 다름
  - warp identity·hot cell·share 보존
  - 실제 학습 pipeline에서 GT 차량 중심 254개 중 86.6% 적중(좌우 flip 대조 37.8%)
- warp 보존율(train 300개): semantic 질량이 BEV aug 없이 99.3%, BEV aug 포함 평균 97.1%(p5 89%). 관측 셀 비율이 낮은 것(평균 약 50%)은 cache 자체의 성질이다.
- GPU 1장 smoke(4D smoke ckpt를 target으로 사용):
  - 10 step 후 resume해 20 step까지 정상, 첫 step에서 UNet 전 파라미터에 grad, frozen target 불변 검사 통과
  - uncond_prob=1.0 경로 정상
  - `stage1_run_info.json`과 config dump 생성 확인
  - checkpoint 1개는 약 5.1 GB다(UNet + optimizer, limit 3 ⇒ 약 15 GB).
- 예비 W4: 공식 BEVDet-R50 `img_feats[0]`은 E[x²]=0.38(std 0.62)이고, flagship BEVFormer-tiny 초기 x0-MSE는 약 0.56으로 같은 order다. smoke 4D ckpt는 0.021(미학습)이다. 최종 판단은 C0 후에 한다.

## 12. Stage 2 (DiffBEVDet4D) 구현 계획 (2026-11-04)

### 12.1 정정 (flagship Stage 2 run 기록 기준)
근거는 `results/diff_bevformer/det/DiffBEVFormer_tiny_onlyseg_metric3d_t100_alpha100_lambda_0.55/`의 dumped config, log, 그리고 `BEVFormer/tools/train_onlyseg.py`다.
- **student는 scratch에서 시작했다.** `load_from`(bevformer_tiny_epoch_24)은 `build_frozen_teacher`에서 `model_target`에만 load되고, 그 뒤 `cfg.load_from=None`이 된다.
  - log에 student checkpoint load 기록이 없다.
  - epoch 1 iter 50의 값이 loss_cls 1.35 / loss_bbox 1.71로 scratch 수준이다.
  - 따라서 §1.5·§5의 "C0 + 24ep 추가 학습(C1 필수)"과 §10.1의 "CBGS ≈5 epoch" 권장은 철회한다.
  - BEVFormer 비교(scratch 24ep vs scratch 24ep + distill)는 이미 matched다.
- **CFG는 사용했다.** flagship 당시 코드(`d800397`)에서 `noise_pred = uncond + 2·(cond − uncond)`로 2가 고정돼 있었다. 현재 `cfg_scale` 기본값도 2다. `use_classifier_guidence=False`는 task-gradient classifier guidance가 꺼져 있다는 뜻이다.
- **teacher 설정:** noise t=100, `denoise_timesteps=100`, DDIM 5 step, CFG 2.
- **MGD:** cell mask λ=0.55(채널 공유), 3×3-ReLU-3×3 generator, α=100. loss = 0.5·task + 0.5·bev(`_parse_losses_mix`).
- **target 갱신:** `UpdateTarget(epoch_interval=0)`은 실행되지 않으므로 target은 끝까지 frozen이다.

### 12.2 비교 설계 (갱신)
| ID | 구성 |
| --- | --- |
| C0 | BEVDet4D-R50-CBGS scratch, 공식 recipe(CBGS 20ep, 8×8), seed 0 |
| C2 | C0과 같은 recipe·scratch + teacher(C0 frozen target → Stage 1 UNet) + MGD 0.5/0.5 |
| (선택) | C2와 같지만 distill 없이 task×0.5만 적용. grad_clip 5 아래에서 loss 배율 효과를 분리한다 |

C0와 C2는 seed 2–3개로 돌리는 것이 좋다.

### 12.3 파일 (새 파일만, BEVDet core·`tools/train.py`·`bevdet.py` 무수정)
- 위치는 `BEVDet/projects/bevdet_plugin/`(새 plugin 패키지)로 한다.
  - `projects/mmdet3d_plugin`은 `jhl-bevdet`에서 import에 실패한다(registry 충돌).
  - `mmdet3d/models/detectors/`는 vendor 코드라 baseline과 섞인다.
- `bevdet_plugin/detectors/diff_bevdet4d.py`: `DiffBEVDet4D(BEVDet4D)`
  - `forward(only_bev, given_bev)`
  - `obtain_bev`
  - `train_step(model_target, bev_diffuser)`
  - `forward_train`
  - `forward_pts_train(given_bev, ...)`
  - MGD
  - `_parse_losses_mix`
  - `forward_test(given_bev)`
- `bevdet_plugin/runner/diff_epoch_based_runner.py`: BEVFormer 것을 복사한다.
- `bevdet_plugin/apis/train.py`: BEVDet `train_detector`를 바탕으로, model_target·bev_diffuser를 cuda로 옮기고 runner `default_args`에 넣는다. EMA·custom hook 처리는 그대로 둔다.
- `tools/bevdiffuser_onlyseg.py`: BEVFormer 것을 그대로 복사한다.
- `tools/train_onlyseg.py`: BEVFormer `train_onlyseg.py`와 BEVDet `tools/train.py`를 합친다. `load_from`은 target 전용이고, student는 `init_weights`에서 시작한다.
- `configs/diff_bevdet/diff_bevdet4d_r50_cbgs_onlyseg_mgd.py`: C0 recipe에 prior pipeline·`bev_diffuser_cfg`·`DiffEpochBasedRunner`를 더한다.
- `tools/dist_train_diff_bevdet4d.sh`: 8 GPU 실행 스크립트. 평가는 기존 `dist_test_bevdet4d.sh`에 `CONFIG`를 지정해 쓴다.
- 테스트: `bevdet_plugin/tests/test_diff_bevdet4d.py`.

### 12.4 구현 완료 (2026-11-04)
- 결정: Stage 2 학습 규모 = C0(CBGS 20ep, 8×8), 선택 대조군 없음, `load_from`은 target 전용(flagship과 동일).
- 새 파일:
  - `projects/bevdet_plugin/{__init__,detectors/diff_bevdet4d,runner/diff_epoch_based_runner,apis/train}.py`
  - `tools/train_onlyseg.py`
  - `tools/bevdiffuser_onlyseg.py`(BEVFormer 것과 바이트 동일)
  - `projects/configs/diff_bevdet/diff_bevdet4d_r50_cbgs_onlyseg_mgd.py`
  - `tools/dist_train_diff_bevdet4d.sh`
  - `projects/bevdet_plugin/tests/test_diff_bevdet4d.py`
- 기존 파일은 수정하지 않았다(`bevdet.py`, `tools/train.py`, `mmdet3d/apis/train.py`).
- 테스트 6/6 통과(GPU 1장):
  - config가 C0 recipe·Stage 1 UNet·flagship teacher 설정과 일치
  - MGD mask 비율 55%
  - teacher 없이 loss·추론이 BEVDet4D와 같음
  - `only_bev`/`given_bev` 동작
  - teacher 경로: loss=0.5·task+0.5·bev, grad는 student에만, target·UNet 불변
  - target만 load되고 student는 그대로이며, key 불일치 시 에러
- GPU 1장 smoke(subset, smoke ckpt target, random UNet teacher):
  - 1 epoch 후 resume(EMA 포함)해 2번째 epoch까지 정상
  - `--no-aavt` 추론으로 결과 80/80 작성
- iteration 시간(공유 GPU, 1장 기준): Stage 2 median 3.47 s, C0 smoke 1.56 s(약 2.2배). 메모리는 GPU당 20.5 GB(bs 8). 실제 시간은 본 학습의 첫 로그로 확인한다.
- 실행 순서: C0 → Stage 1 → `bash tools/dist_train_diff_bevdet4d.sh` → `CONFIG=<stage2 config> bash tools/dist_test_bevdet4d.sh <work_dir>/epoch_20_ema.pth`.

### 12.5 CBGS 없는 recipe 준비 (2026-11-05)
- 배경: C0(CBGS)가 4 GPU × 8 = batch 32로 실행 중이다(사용자가 `dist_train_bevdet4d_r50_cbgs.sh`를 `GPUS=4`, `CUDA_VISIBLE_DEVICES=4,5,6,7`로 바꿨다). epoch당 3,862 iter이다. epoch 1–2는 GPU 공유로 iter당 약 4 s였고, epoch 3부터 0.46 s이다. 시간을 줄이려고 CBGS 없는 recipe를 함께 준비했다. 실제로 쓸지는 사용자가 정한다.
- 원칙: C0, Stage 1 target, Stage 2는 같은 recipe를 쓴다. CBGS C0를 target으로 쓰면서 student와 baseline만 CBGS 없이 학습하는 조합은 쓰지 않는다(더 강한 recipe의 distill 효과가 섞인다).
- 새 파일:
  - `configs/bevdet/bevdet-r50-4d.py`: cbgs config에서 `CBGSDataset` wrapper만 제거하고, 24 epoch / lr step [24]로 바꿨다(upstream `bevdet-r50.py` 일정). 사용자가 추가한 `evaluation`, `checkpoint_config` 줄도 포함한다.
  - `tools/dist_train_bevdet4d_r50_nocbgs.sh`: 4 GPU, run `bevdet4d-r50-nocbgs_seed0`, port 28621.
  - `projects/configs/diff_bevdet/diff_bevdet4d_r50_nocbgs_onlyseg_mgd.py`: cbgs Stage 2 config와 base, `data.train`, epoch 24, target/teacher 경로만 다르다.
- 수정:
  - Stage 1 launcher와 Stage 2 launcher는 `RECIPE=nocbgs|cbgs`를 필수로 받는다. target, teacher, config, run name이 recipe에서 정해진다.
  - Stage 1 run 이름: `BEVDiffuser_bevdet4d_r50_{nocbgs,cbgs}_onlyseg_sam3_metric3d`(기존 `..._r50_onlyseg_...`에서 변경, 아직 실행된 적 없음).
  - Stage 2 launcher는 8 GPU에서 4 GPU로 바꿨다(C0와 같은 batch 32). 기본 `CUDA_VISIBLE_DEVICES=4,5,6,7`이다.
  - cbgs Stage 2 config의 teacher 경로를 새 Stage 1 이름으로 바꿨다.
- 검증: CPU 테스트 16 passed / 3 skipped(GPU 전용). 새 테스트로 다음을 확인한다.
  - C0 두 config는 wrapper, epoch, lr step만 다르다.
  - Stage 2 두 config는 그 외에 target과 teacher 경로만 다르다.
  - 각 recipe의 Stage 2가 자기 C0와 Stage 1에 맞는다.
  - 데이터셋: 길이 28,130, prior [17,64,64]. launcher DRY_RUN과 guard(RECIPE 누락, ckpt 없음, GPU 수)도 확인했다.
- 실행(CBGS 제외 시):
  1. `bash tools/dist_train_bevdet4d_r50_nocbgs.sh`
  2. `RECIPE=nocbgs bash projects/bevdiffuser/train_only_seg_cache_det_bevdet4d.sh`
  3. `RECIPE=nocbgs bash tools/dist_train_diff_bevdet4d.sh`
  4. 평가: `CONFIG=<config> bash tools/dist_test_bevdet4d.sh <work_dir>/epoch_24_ema.pth`
