---
name: Stage 2 distillation loss form
description: Stage 2 ours에서 활성화된 distillation 신호 구성 — projector + MSE×100 단일 (aux/relational 비활성)
type: project
---

Stage 2 ours가 Stage 1 teacher 우위(+8.43 mAP at noise_T=100)를 student로 전이 못 하는 구조적 후보 (조사 결과).

**Why:** 사용자 측정에서 Stage 1 ours teacher는 noise_T=100에서 baseline 대비 +8.43 mAP인데 Stage 2 student는 +0.01 mAP. 학습 코드의 distill 신호가 정보 손실 지점.

**How to apply:** Stage 2 distillation 관련 디버깅/실험 설계 시 아래 순으로 의심.

활성 코드 경로 (확정):
- sh: `BEVFormer/tools/dist_train_seg_v2.sh` → train_seg.py
- config: `projects/configs/diff_bevformer/layout_tiny_seg_v4_adapter.py`
  - `bev_diffuser_cfg`(`:128-136`): noise_T=100, denoise_T=100, infer=5, CFG=2.0, prediction_type="sample"
  - model(`:142-144`): `DiffBEVFormerSeg` (V1), `use_proj=True`. use_aux_seg/use_bev_rel 명시 안 함 → 기본 False로 비활성
- detector: `diff_bevformer_seg.py:62-401` (V1). V2는 `:407-682`에 있지만 미사용.

비대칭 핵심 (가장 강한 의심):
- task head는 raw `bev` 사용 (`:240-243`)
- distill MSE는 학습 가능한 `bev_distill_proj` 통과한 `bev_s_2d`로 계산 (`:289-304`)
- → projector(1x1-GN-GELU-1x1, `:21-36`)가 distribution mismatch를 자기 가중치로 흡수, loss_bev 빠르게 하락하지만 student backbone에는 teacher 구조 신호 약하게만 전달

기타 의심 (우선순위 순):
1. MSE 단일 신호 — per-pixel L2는 GT-segmap-lifted 구조성 정보 손실
2. loss balance: bev_loss×100 vs task_loss → distill 절대 magnitude가 task 1/5~1/10일 가능성
3. CFG=2.0 + seg_uncond=zeros(`bevdiffuser_seg.py:61-63, 83`)로 teacher feature가 student와 통계적으로 멀어짐 → projector absorber 환경 강화
4. normalize/whitening 부재 (`bev_distill_gn` 주석 `:91, :306-309`)

부정된 가설:
- Stage 1 평가 vs Stage 2 학습 setup gap: 둘 다 multi-step DDIM 5 step (`bevdiffuser_seg.py:65-86`). noise_T 적용 위치/방식 동일.
- gradient 단절: detach 위치 모두 정상 (teacher requires_grad_(False)).
- "Stage 2 ours가 noise_T=100을 써서 자기 강점이 안 살았다" → 코드/실험표 모두 부정 (ours 강점은 noise_T=100에서 더 큼).

가용 후크 (config/코드 한두 줄로 토글):
- `use_proj=False` → projector shortcut 제거 (raw MSE)
- `use_aux_seg=True` → KL distill 추가 (V1: validity mask 사용)
- `use_bev_rel=True` → cosine self-sim distill
- `bev_distill_gn` + `[MSE-B]` 주석 해제 → affine-free GroupNorm 정합
- FoV-가중 MSE (`:298-301` 주석) 활성
- progress weight 스케줄 (`:131-133` 주석) 활성
- `type='DiffBEVFormerSegV2'` 교체 (projector 빠짐 + aux_seg GT prob target)

V1 vs V2 aux_seg target 차이의 본질:
- V1: diffuser forward에서 받은 `seg_bev_prob` (`:285,287`) — CFG된 conditional pass 결과
- V2: `bev_diffuser.unet.seg_aligner.compute_prob_only(...)` 직접 호출 (`:628-631`) — diffuser 우회
- seg_aligner의 prob 계산은 parameter-free (`seg_bev_aligner_one_hot_da3.py:264-273` docstring) → 두 target 값 거의 동일
- 실질 차이: KL normalization — V1은 validity mask, V2는 전 픽셀 평균
