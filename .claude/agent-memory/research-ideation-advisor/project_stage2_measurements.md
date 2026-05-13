---
name: Stage 1/2 측정값 (최종 정정 2026-05-11)
description: noise_T=100 에서 ours teacher 가 큰 우위(+6.61 NDS/+8.43 mAP) — Stage 2 전이 실패의 본질은 distillation 메커니즘. noise_T 부정합 가설 폐기.
type: project
---

## 최종 정정 측정값 (NDS / mAP)

### Stage 1
- noise_T=5  : Ours **+1.18 NDS / +3.25 mAP**  (작은 우위)
- noise_T=100: Ours **+6.61 NDS / +8.43 mAP**  (큰 우위 — ours 의 강점 noise level)

### Stage 2 (둘 다 자기 최적 noise_T)
- BEVDiffuser(noise_T=5):  38.71 / 28.84
- Ours(noise_T=100):       39.75 / **28.85**  → mAP gap +0.01 만 잔존
- **+8.43 mAP teacher 우위 → +0.01 mAP student 우위로 전이 실패**

## 폐기된 가설
- 이전 메모리(`project_stage2_diagnosis.md` D1)의 "noise_T=100 이 conditioning 을 죽인다" 가설 — **틀렸음**. noise_T=100 에서 ours teacher 는 dense semantic 우위가 가장 큼.
- Stage 2 에서 ours 가 noise_T=100 을 쓴 건 자기 강점을 정확히 고른 합리적 선택.

## 재구성된 핵심 문제
**Teacher 의 +8.43 mAP 우위가 student 로 전이되지 않는 distillation 메커니즘 자체의 문제.**

진단 축 (team-lead 지시):
1. **MSE × 100 만으로 부족**: teacher 의 풍부한 semantic 정보를 단순 L2 로는 평탄화/희석. 보조 채널 필요.
2. **Teacher↔Student representation gap 과대**: student 는 image-only tiny BEVFormer, teacher 는 GT layout+seg conditioning + multi-step denoised BEV. feature space 차이가 너무 커서 MSE 직접 정합이 student 의 도달불가 점을 향한 학습 → 평균에 머무름. V1 의 projector 를 V2 가 뺀 게 결정타였을 가능성.
3. **Aux seg KL 의 신호 형태가 약함**: V2 는 `seg_aligner.compute_prob_only` 로 GT 분포를 새로 만들어 KL → teacher diffuser 출력과 분리된 신호. ours 만의 학습된 표현이 흐르는 채널이 아님. V1 은 `seg_bev_prob`(diffuser 직접 출력) 사용.
4. **학습 중 noise scheduling 의 실제 동작**: Stage 1 평가는 multi-step denoising. Stage 2 학습 매 iter 호출 시 noise level / DDIM step 수가 어떻게 적용되는지 code-reviewer 확인 대기.

## How to apply
Stage 2 개선 아이디어 작성 시 비중:
- (a) Loss 보강: attention/relational distill, foreground-weighted MSE, masked distill
- (b) Projector/Adapter 복귀·강화: V1 projector + inverse projector (양방향)
- (c) Multi-target distill: diffuser UNet 중간 feature (특히 FDN 직후) 정합
- (d) Curriculum / multi-noise sampling
- (e) 공정 비교 실험: 동일 noise_T(=100) 에서 baseline 도 Stage 2, V1 vs V2 같은 setup 비교
