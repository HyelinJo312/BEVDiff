# Feature Distillation에서 Projector가 필요한 이유

## 1. 출발점: Student와 Teacher의 BEV feature는 본질적으로 다른 분포에서 나온다

| | **Student (BEVFormer)** | **Teacher (BEV Diffuser)** |
|---|---|---|
| 입력 | Multi-view camera images | BEVFormer feature + denoising step |
| 추가 conditioning | 없음 | **GT layout + Grounded SAM segmap** (CFG 적용) |
| 출력 feature 분포 | 카메라 lifting 분포 | Conditioning에 의해 reshape된 분포 |

Student는 카메라 이미지만으로 BEV를 lifting하므로, 그 feature는 BEVFormer가 학습 과정에서 자연스럽게 형성한 분포 안에 있습니다. 반면 teacher는 BEVFormer의 출력 위에 **diffusion denoising + 외부 conditioning(layout, segmap)** 으로 구조화된 보정을 추가합니다. 이 보정은 student의 카메라 lifting 만으로는 자연스럽게 형성되지 않는 방향의 feature를 생성합니다.

---

## 2. Baseline 대비 Gap이 더 커진 이유

CFG (classifier-free guidance)는 **baseline (BEVDiffuser, layout only)과 내 방법 둘 다에서 동일하게 사용**됩니다. 따라서 CFG 자체는 두 방법의 차이를 만들지 않습니다.

차이를 만드는 것은 **conditioning 신호의 종류와 그 신호가 teacher BEV feature에 주입되는 방식**입니다.

| | Baseline | 내 방법 |
|---|---|---|
| Conditioning 단위 | Object token (sparse) | Object + per-pixel semantic (dense) |
| 주입 메커니즘 | Cross-attention | **GatedFDN (FiLM/SPADE-style normalization 변조)** |
| 적용 layer | Attention block | **모든 FDN block, multi-scale로 누적** |
| Feature에 미치는 영향 | Object-level token interaction | **모든 spatial 위치의 정규화 통계가 semantic-dependent** |

Baseline은 BEV feature 위에 객체 정보를 attention으로 "참조"하는 방식이라, feature 분포 자체의 형상은 BEVFormer가 만든 분포와 가깝게 유지됩니다. 내 방법은 GatedFDN을 통해 **feature의 정규화 통계 자체를 segmap-dependent하게 다시 정의**하므로, teacher의 BEV feature는 BEVFormer의 자연스러운 분포로부터 구조적으로 더 멀리 이동한 곳에 위치합니다.

CFG는 양쪽에서 동일하게 conditional 방향을 amplify하지만, **amplify되는 방향 자체가** 내 방법에서는 GatedFDN 누적 modulation의 결과여서 student의 카메라 lifting 분포로부터 훨씬 더 멀리 위치합니다.

---

## 3. Vanilla MSE의 구조적 문제

`MSE(student_bev, teacher_bev)`는 **두 feature가 같은 좌표계 안의 같은 분포에서 나왔다**는 가정을 깔고 있습니다. 그래야 pointwise 차이가 의미 있는 신호가 됩니다.

내 setup에서는 이 가정이 깨집니다:

- Student의 feature 좌표계: BEVFormer가 카메라로부터 자연스럽게 만드는 분포
- Teacher의 feature 좌표계: GatedFDN이 모든 layer에서 segmap에 의해 정규화 통계를 spatial-adaptive하게 재정의한 분포

이 상태에서 vanilla MSE를 적용하면:

- Student가 자기 좌표계 안에서 최선을 다해도 GatedFDN이 만든 spatial-semantic 변형을 카메라 lifting만으로 재현 불가능
- 손실은 줄지 않고, gradient는 student를 **자연스러운 카메라 lifting 분포에서 벗어난 방향**으로 끌어당김
- 결과적으로 detection task에 도움 안 되는 방향으로 BEV feature가 왜곡됨

---

## 4. Baseline은 왜 vanilla MSE로도 잘 됐나

Baseline은 conditioning을 cross-attention으로만 주입하므로, teacher BEV feature의 분포 형상이 BEVFormer 본래의 분포에서 크게 벗어나지 않습니다:

- Layout 정보(객체 위치, 클래스)는 student도 카메라 lifting + detection을 통해 어느 정도 도달 가능
- Cross-attention은 token-level 참조만 추가할 뿐, normalization 통계 자체는 변형하지 않음
- Student와 teacher의 feature 좌표계가 거의 일치 → MSE 가정이 근사적으로 성립

따라서 baseline에서는 vanilla MSE만으로 충분히 distillation이 작동했습니다.

내 방법은 GatedFDN의 spatially-adaptive normalization이 **모든 denoising layer에서 누적적으로 feature 분포를 reshape**하므로 좌표계 불일치가 구조적으로 커졌고, **vanilla MSE 가정이 더 이상 성립하지 않는 영역**으로 진입했습니다.

---

## 5. Projector가 해결하는 것

Projector는 student-side에 붙는 학습 가능한 변환기로, 다음 역할을 합니다:

```
loss = MSE( projector(student_bev),  teacher_bev )
       └────────────┬────────────┘
        teacher 좌표계로 옮긴 student
```

- **Basis 변환**: Student의 좌표계 → teacher의 GatedFDN-modulated 좌표계로의 매핑을 학습
- **Scale 정규화**: GroupNorm이 두 분포의 magnitude 차이를 흡수
- **비선형 표현**: GELU가 GatedFDN이 만드는 비선형 (sigmoid gate × affine modulation) 분포 변화를 따라감

핵심은 **MSE의 "같은 좌표계" 가정을 projector가 학습으로 만들어준다**는 점입니다. Student는 자기 좌표계에서 자유롭게 task에 최적화하고, projector가 그 결과를 teacher 좌표계로 옮긴 뒤에 비교하므로:

- Student feature: detection-optimal 방향으로 자유롭게 학습
- Distillation gradient: teacher 좌표계에서의 mismatch만 잡아내 student로 흘려보냄
- 두 목적이 충돌 없이 병행

---

## 6. Student-side에만 두는 이유

Teacher는 frozen이므로 teacher feature를 변환하는 것은 의미가 없습니다. 또한 teacher 측을 변환하면 teacher의 좌표계 자체를 바꾸게 되어, 우리가 transfer하려는 "teacher의 표현"의 정의가 흐려집니다. Student-side projector는 **teacher 좌표계를 기준으로 두고, student가 그쪽으로 매핑하는 방법을 학습**하는 비대칭 구조를 만듭니다.

---

## 7. Inference 비용은 0

Projector는 distillation loss 계산에만 쓰이고, student의 detection forward path에는 전혀 개입하지 않습니다. 학습이 끝나면 projector는 버려지고, 배포되는 모델은 순수 BEVFormer입니다. 따라서 **모델 크기, 추론 속도, 메모리에 어떠한 추가 비용도 발생하지 않으면서** distillation 효과만 얻습니다.

---

## 한 문장 요약

> **Baseline과 내 방법 모두 CFG를 동일하게 사용하지만, 내 방법은 segmap conditioning을 GatedFDN을 통해 모든 denoising layer에서 spatially-adaptive normalization으로 주입하여 teacher BEV feature의 정규화 통계 자체를 segmap-dependent하게 재정의한다. 이로 인해 teacher 분포가 student의 카메라 lifting 좌표계로부터 구조적으로 더 멀리 이동하고, "두 feature가 같은 좌표계에 있다"는 vanilla MSE의 가정이 깨진다. Projector는 이 좌표계 어긋남을 student-side에서 학습 가능한 매핑으로 흡수하여 MSE의 가정을 복구하고 distillation gradient가 detection-optimal 방향으로 흘러가게 만든다.**
