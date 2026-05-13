---
name: Stage 2 distillation 진단 (재구성 2026-05-11)
description: Stage 2 transfer 실패의 본질은 distillation 메커니즘. teacher 우위 +8.43 mAP → student +0.01. MSE-only / projector 부재 / aux_seg 채널 단절이 주범.
type: project
---

⚠️ **이전 버전의 "noise_T=100 mismatch" D1 가설은 폐기**. 정정된 측정값(`project_stage2_measurements.md`)에서 ours teacher 는 noise_T=100 에서 +8.43 mAP 우위가 가장 큼. Stage 2 ours 가 noise_T=100 을 고른 건 합리적 선택.

## 재구성된 진단

### D1 (new). **Distillation channel poverty — MSE × 100 단일 신호 의존**
- `detectors/diff_bevformer_seg.py:304-315`: `loss_bev = MSE(proj(bev_s), bev_t) × 100`
- teacher BEV 는 condition-rich (GT layout + GT seg + 5-step DDIM denoised) — 정보가 channel/spatial 에 분산됨.
- L2 는 등방성 (channel-blind, isotropic). teacher 의 dense semantic structure(클래스별 channel subspace, 객체-경계 spatial pattern) 가 student 로 흐르려면 **structural / relational / attention-style loss** 가 필수.
- V1 의 `loss_bev_rel`(cosine self-similarity matrix MSE, line 324-327) 와 `loss_bev_aux_seg`(KL, line 339-344) 가 이 역할인데 가중치가 작거나 (`bev_rel_weight` default 10.0 vs MSE 100, `aux_seg_weight` default 1.0~5.0 vs MSE 100), V2 는 relational 을 아예 제거.

### D2 (new). **Representation gap → student manifold 도달불가 target**
- student: ResNet50 + 3-layer encoder + 1-step BEV (no GT). teacher: 같은 backbone 위에 +GT layout encoder + GT seg→FDN modulation + 5-step DDIM denoise. feature space 의 분산·구조가 크게 다름.
- MSE 직접 정합 시 student 는 도달불가 점을 향해 학습 → gradient 의 평균 방향으로 평탄화 → mAP 미증.
- V1 의 `BEVDistillProjector` (line 21-36 1×1, V2 는 3×3+residual line 39-59) 가 이 gap 을 brige 하는 학습가능 adapter. config `_aux.py:144` `use_proj=True` 이지만 V2 detector 자체가 projector 모듈을 제거했다는 점(line 407-427) 이 신호 약화의 가장 직접적 원인일 수 있음.
- **단방향 만으로 부족할 수 있음**: student→teacher basis projector 외에 teacher→student basis 의 inverse projector 도입 시 양방향 정합 가능.

### D3 (new). **Aux seg KL 의 channel 단절 (V2 의 결정적 약화)**
- V1 (line 287, 339-344): KL target = `seg_bev_prob` (= teacher diffuser 가 forward 중 실제로 사용한 dense seg conditioning 출력). **ours-특유 학습된 모듈 출력**.
- V2 (line 629-635): KL target = `bev_diffuser.unet.seg_aligner.compute_prob_only(...)` (= seg_aligner 모듈을 GT seg 에 직접 적용한 분포). **학습된 conditioning pathway 우회**, 사실상 GT seg projection 그 자체.
- → V2 의 KL 은 teacher 의 학습된 우위가 아니라 GT seg 의 BEV projection 만 흘림. baseline 도 GT seg 를 access 하면 가능한 신호 → ours 차별 채널 아님.

### D4. **Loss bucket scale imbalance**
- `_parse_losses_mix(weight=0.5)`: 'bev' 이름의 loss 합산 후 0.5 곱.
- V1+aux: MSE×100 + KL×1.0 (또는 5.0) + (옵션) sim_MSE×10. KL 이 두 자릿수 차이로 압도당함.
- 보조 채널이 절대 scale 에서 mute.

### D5 (verified). **Teacher 사실상 frozen**
- 모든 stage-2 config `custom_hooks=[dict(type='UpdateTarget', epoch_interval=0)]` → no-op. student 진화 ↔ teacher 정체. distill gap 누적.

### D6 (code-reviewer 분석으로 부정). **학습/평가 setup mismatch 가설 폐기**
- Stage 2 학습도 5-step DDIM, CFG=2.0, noise_T=100 고정 — Stage 1 평가 setup 과 완전 일치 (code-reviewer 보고).
- 따라서 "학습이 single step 이라 ours 강점 못 본다"·"매 iter random t mismatch" 가설은 부정.

### D7 (code-reviewer 가 짚은 최강 후보, 정량적 정황 추가). **Projector shortcut 으로 student backbone 으로 가는 gradient 가 약화**
- `forward_pts_train` (line 240-243) 의 task loss 는 raw `bev` 사용. distill MSE (line 289-304) 는 `bev_distill_proj(bev)` 통과한 `bev_s_2d` 사용.
- → projector 가 학습 가능한 1×1 conv 이라 student backbone 의 분포가 teacher 와 크게 달라도 **projector 자체가 distribution shift 를 자기 weight 로 흡수**.
- 결과: `loss_bev` 는 빠르게 떨어지지만 student backbone 으로 흐르는 useful gradient 는 약함. task loss path 에는 distill 의 영향이 거의 안 박힘.

**정량적 강화 근거 (code-reviewer 2026-05-11)**:
- `v4_adapter.py:361-369` paramwise_cfg: `bev_distill_proj` 의 custom_keys 가 주석 처리 → projector LR = **3e-4 (default)**, backbone LR = **1.5e-4 (lr_mult=0.5)**. **projector 가 backbone 의 2× 속도로 학습됨**.
- teacher 는 학습 내내 완전 frozen (D5 dead-code 확인 — UpdateTarget hook 이 epoch_interval=0 로 매번 False, before_run 분기도 주석).
- → 학습 다이내믹스가 명확히 "projector 가 빠르게 student 분포 → frozen teacher 분포로 매핑 → loss_bev 빠르게 만족 → backbone gradient 약화" 방향으로 편향됨.
- **이것이 +8.43 → +0.01 의 메커니즘적 설명 + 정량적 정황**. D1(채널 빈곤) 과 D2(projector 단방향) 의 진짜 원인 후보.

### D3 보정 (code-reviewer 분석으로 일부 수정)
- V1 의 `seg_bev_prob` 와 V2 의 `compute_prob_only` 는 사실상 **거의 같은 값**. seg_aligner 의 prob 계산이 parameter-free 이기 때문 (`seg_bev_aligner_one_hot_da3.py:264-273` docstring 명시).
- 따라서 "V1 의 target 은 학습된 표현, V2 는 GT projection" 이라는 이전 진단은 **틀림**.
- 두 분포의 진짜 차이는 **KL normalization 만**: V1 = validity mask (FoV 무효 cell 제외, sum/valid_count), V2 = 전 픽셀 평균 (FoV 무효의 degenerate 분포가 KL 평탄화).
- → V1 이 numerically 더 안전. V1 의 aux_seg 강점은 "ours-특유 학습 표현" 이 아니라 "FoV-aware masking 으로 noise 제거".

## 진단의 한 줄
**Teacher 우위는 풍부한데 distillation pipe 가 좁다.** noise_T 는 합리적 선택. 진짜 병목은 (i) L2 단일채널, (ii) projector 약화, (iii) aux 채널이 ours 학습된 표현이 아닌 GT projection 으로 우회.

## How to apply (개선 아이디어 작성 시 비중)
- 우선: (a) **V1 detector(projector+relational+aux_seg with seg_bev_prob) 복귀** — V2 가 V1 의 핵심 채널을 뺀 게 의심.
- 그다음: (b) **attention transfer / FDN 중간 feature distill** (ours-특유 모듈 출력을 직접 정합)
- 보조: (c) **foreground-weighted MSE**, (d) **multi-noise sampling**, (e) **EMA teacher**.
- 공정 비교: baseline 도 noise_T=100 으로 Stage 2 학습 / V1 vs V2 같은 setup 비교 / aux_seg target 을 seg_bev_prob vs compute_prob_only 직접 비교.
