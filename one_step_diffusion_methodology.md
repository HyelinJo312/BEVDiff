# One-Step Diffusion 기반 BEV Representation Learning 방법론 정리

## 1. 전체 방법론 개요

본 방법론은 두 단계로 구성된다.

- **Stage 1:** Frozen BEV encoder 기반의 generative BEV diffusion pre-training
- **Stage 2:** Scratch BEV encoder와 Stage 1 pretrained diffusion을 활용한 one-step diffusion fine-tuning 및 downstream task learning

핵심 아이디어는 Stage 1에서 학습된 diffusion model을 Stage 2에서 두 가지 방식으로 활용하는 것이다.

1. **Frozen multi-step diffusion teacher**로 사용
2. **Stage 2 one-step diffusion student의 initialization**으로 사용

즉, Stage 2의 student diffusion은 random initialization으로 학습하지 않고, 반드시 Stage 1 pretrained diffusion weight으로 초기화한다.

\[
D_S^{\mathrm{init}} \leftarrow D_{\mathrm{pre}}
\]

따라서 Stage 2는 단순히 teacher의 출력을 distillation하는 것이 아니라, Stage 1에서 학습된 multi-step diffusion 자체를 downstream task에 적합한 one-step diffusion으로 fine-tuning하는 과정으로 볼 수 있다.

\[
\boxed{
\text{Multi-step Generative Diffusion Pre-training}
\rightarrow
\text{One-step Diffusion Fine-tuning + Task Learning}
}
\]

---

# 2. Stage 1: Generative BEV Diffusion Pre-training

## 2.1 Frozen BEV Encoder

Stage 1에서는 사전에 detection task로 학습된 BEV encoder를 사용한다.

\[
F_0^T = E_T(I)
\]

여기서

- \(I\): multi-view camera images
- \(E_T\): detection-pretrained BEV encoder
- \(F_0^T\): clean BEV feature

Stage 1에서 \(E_T\)는 전체 학습 동안 **freeze**한다.

즉, Stage 1에서 새롭게 학습되는 핵심 모듈은 diffusion model이다.

---

## 2.2 Forward Diffusion

Clean BEV feature \(F_0^T\)에 임의의 timestep \(t\)에 해당하는 Gaussian noise를 추가한다.

\[
F_t^T
=
\sqrt{\bar{\alpha}_t}F_0^T
+
\sqrt{1-\bar{\alpha}_t}\epsilon,
\qquad
\epsilon \sim \mathcal{N}(0,I)
\]

Stage 1에서는 다양한 noise level을 학습할 수 있도록 timestep을 sampling한다.

\[
t \sim p(t)
\]

예를 들어 전체 diffusion timestep이 \(T=1000\)이라면, 학습 중 여러 \(t\in[0,T]\)에 대해 noisy BEV feature를 구성한다.

---

## 2.3 Diffusion Pre-training

Diffusion model \(D_{\mathrm{pre}}\)은 noisy BEV feature와 timestep을 입력으로 받아 denoising을 학습한다.

\[
D_{\mathrm{pre}}(F_t^T,t)
\]

Stage 1의 diffusion parameterization이 \(\epsilon\)-prediction인 경우,

\[
\hat{\epsilon}
=
D_{\mathrm{pre}}(F_t^T,t)
\]

이며 기본적인 diffusion loss는 다음과 같이 구성할 수 있다.

\[
\mathcal{L}_{\mathrm{diff}}
=
\|\epsilon-\hat{\epsilon}\|_2^2
\]

Stage 1 종료 후, 다양한 noise level에서 BEV feature를 denoise할 수 있는 pretrained diffusion model

\[
\boxed{D_{\mathrm{pre}}}
\]

을 획득한다.

---

# 3. Stage 2: One-Step Diffusion Fine-tuning + Downstream Task Learning

## 3.1 Stage 2 Student 구조

Stage 2의 student branch는 다음과 같이 구성한다.

\[
I
\rightarrow
E_S
\rightarrow
D_S
\rightarrow
H_S
\rightarrow
\hat{y}
\]

각 모듈의 초기화는 다음과 같다.

| Module | Initialization | Training |
|---|---|---|
| BEV Encoder \(E_S\) | **Scratch** | Trainable |
| Diffusion Student \(D_S\) | **Stage 1 pretrained weight** | Trainable |
| Task Head \(H_S\) | Scratch | Trainable |

특히 diffusion student는 반드시 다음과 같이 초기화한다.

\[
\boxed{
D_S \leftarrow D_{\mathrm{pre}}
}
\]

따라서 Stage 2 diffusion은 새로운 diffusion model을 처음부터 학습하는 것이 아니라, Stage 1 pretrained diffusion을 **one-step inference에 적합하도록 fine-tuning**하는 역할을 한다.

또한 Stage 2의 **student one-step diffusion은 condition-free / null-condition 방식으로 학습 및 추론한다.**

즉, student diffusion \(D_S\)는 noisy BEV feature와 현재 noise level에 해당하는 timestep \(t\)를 입력으로 사용하지만, segmentation map, depth map 등 diffusion condition은 모두 null로 대체한다.

\[
\boxed{
c_S = \varnothing
}
\]

구현 관점에서는 segmentation condition은 zero map으로 구성하고, depth condition은 사용하지 않는다.

\[
\mathrm{seg}_S = 0,
\qquad
\mathrm{depth}_S = \varnothing
\]

따라서 student one-step denoising은 다음과 같이 표현할 수 있다.

\[
\boxed{
D_S(F_t, t, c_S=\varnothing)
}
\]

---

## 3.2 Stage 2 Teacher 구성

Stage 1 pretrained diffusion을 복사하여 frozen teacher를 구성한다.

\[
\boxed{
D_T \leftarrow D_{\mathrm{pre}}
}
\]

Teacher \(D_T\)는 Stage 2 전체 학습 동안 freeze한다.

Teacher와 student는 기본적으로 동일한 diffusion architecture를 사용하되 역할이 다르다.

- **Teacher:** multi-step denoising
- **Student:** null-condition one-step denoising

즉, 모델 크기나 구조를 줄이는 것이 아니라 **sampling step 수를 줄이는 distillation**을 수행한다.

단, condition 사용 방식은 teacher와 student가 다르다. Teacher는 Stage 1 diffusion과 동일하게 condition을 사용할 수 있지만, student는 학습과 추론 모두에서 condition을 null로 고정한다.

---

# 4. Stage 2의 Timestep 설정

## 4.1 Training Timestep Sampling

Stage 2 training에서는 하나의 고정 timestep만 사용하지 않고, moderate noise range에서 timestep을 sampling한다.

\[
\boxed{
t_{\mathrm{train}} \sim p(t), \qquad 0 < t_{\mathrm{train}} \leq t_{\max} < T
}
\]

예를 들어 전체 diffusion schedule이 \(T=1000\)일 때, 현재 구현에서는 다음과 같은 범위를 사용할 수 있다.

\[
t_{\mathrm{train}} \sim \mathrm{Uniform}\{1,\dots,200\}
\]

이는 student one-step diffusion이 teacher와 동일하게 다양한 moderate noise level에서 denoising/refinement를 학습하도록 하기 위함이다.

각 training iteration에서 sampling된 \(t_{\mathrm{train}}\)은 teacher와 student가 공유한다.

\[
\boxed{
t_{\mathrm{teacher}} = t_{\mathrm{student}} = t_{\mathrm{train}}
}
\]

따라서 teacher와 student의 차이는 noisy BEV input이나 noise level이 아니라, denoising trajectory의 길이와 condition 사용 여부에 집중된다.

## 4.2 Inference / Evaluation Starting Timestep \(t^*\)

Inference 또는 대표 evaluation 설정에서는 하나의 fixed starting timestep \(t^*\)를 사용할 수 있다.

\[
\boxed{
0 < t^* < T
}
\]

예를 들어,

\[
t^*=100
\]

과 같은 설정을 사용할 수 있다.

본 방법론에서는 pure Gaussian noise에 가까운 \(t=T\)에서 시작하는 것보다, 원래 BEV feature의 semantic/spatial information이 충분히 남아 있는 **moderate noise level**을 사용하는 것을 기본 방향으로 한다.

이는 Stage 2 diffusion의 역할이 pure noise에서 새로운 BEV feature를 생성하는 것이 아니라,

\[
\boxed{
\text{BEV Feature Denoising / Refinement}
}
\]

이기 때문이다.

---

# 5. Stage 2 Training

## 5.1 Scratch BEV Feature 생성

Stage 2의 scratch BEV encoder를 통해 feature를 생성한다.

\[
F_0^S
=
E_S(I)
\]

Stage 1의 detection-pretrained BEV encoder feature를 student encoder와 직접 alignment하지 않는다.

즉,

\[
F_0^S \approx F_0^T
\]

와 같은 direct feature distillation loss는 사용하지 않는다.

---

## 5.2 Noise Injection

Scratch BEV encoder가 생성한 \(F_0^S\)에 training timestep \(t_{\mathrm{train}}\)에 해당하는 noise를 추가한다.

\[
\boxed{
F_{t_{\mathrm{train}}}
=
\sqrt{\bar{\alpha}_{t_{\mathrm{train}}}}F_0^S
+
\sqrt{1-\bar{\alpha}_{t_{\mathrm{train}}}}\epsilon
}
\]

여기서

\[
\epsilon \sim \mathcal{N}(0,I)
\]

이다.

Teacher와 student는 **동일한 noisy feature \(F_{t_{\mathrm{train}}}\)** 를 입력으로 사용한다.

즉,

\[
F_{t_{\mathrm{train}}}^{Teacher}
=
F_{t_{\mathrm{train}}}^{Student}
\]

가 되도록 하며, 동일한 Gaussian noise realization을 공유한다.

이를 통해 teacher와 student의 차이는 noisy BEV input이 아니라 **denoising trajectory의 길이**와 **condition 사용 여부**에 집중되도록 한다.

Student에 대해서는 diffusion condition을 모두 null로 대체한다.

\[
c_S=\varnothing
\]

반면 teacher는 multi-step target 생성을 위해 기존 condition을 사용할 수 있다.

---

# 6. Teacher: Multi-Step Sampling

Teacher는 동일한 \(F_{t_{\mathrm{train}}}\)에서 시작하여 여러 denoising step을 통해 \(F_0^T\)를 생성한다.

예를 들어

\[
t_{\mathrm{train}}=100
\]

이고 5-step sampling을 사용할 경우,

\[
100
\rightarrow
80
\rightarrow
60
\rightarrow
40
\rightarrow
20
\rightarrow
0
\]

과 같은 timestep sequence를 사용할 수 있다.

따라서 teacher의 denoising 과정은

\[
F_{100}
\rightarrow
F_{80}
\rightarrow
F_{60}
\rightarrow
F_{40}
\rightarrow
F_{20}
\rightarrow
F_0^T
\]

가 된다.

Teacher diffusion은 각 sampling step마다 현재 noise level에 대응하는 timestep을 입력받는다.

예를 들어,

\[
D_T(F_{100},100)
\]

\[
D_T(F_{80},80)
\]

\[
D_T(F_{60},60)
\]

과 같이 동작한다.

DDIM과 같이 timestep skipping을 지원하는 sampler를 사용하면 모든 intermediate timestep을 방문하지 않고도 few-step multi-step sampling을 수행할 수 있다.

Teacher output은 stop-gradient 처리한다.

\[
F_0^T
=
\mathrm{sg}
\left(
D_T^{multi-step}(F_{t_{\mathrm{train}}})
\right)
\]

---

# 7. Student: One-Step Sampling

Student는 teacher와 동일한 \(F_{t_{\mathrm{train}}}\)에서 시작하지만 diffusion U-Net을 단 한 번만 호출한다.

\[
\boxed{
F_{t_{\mathrm{train}}}
\xrightarrow[\mathrm{NFE}=1]{D_S(\cdot,t_{\mathrm{train}},c_S=\varnothing)}
F_0^S
}
\]

예를 들어 \(t_{\mathrm{train}}=100\)이면,

\[
D_S(F_{100},100,c_S=\varnothing)
\]

을 한 번 수행하여 바로 clean/denoised BEV feature를 예측한다.

중요한 점은 student에 입력되는 timestep이 \(0\)이 아니라 **현재 입력 feature의 noise level인 \(t_{\mathrm{train}}\)** 라는 것이다.

즉,

\[
\boxed{
t_{\mathrm{student}} = t_{\mathrm{train}}
}
\]

이다.

Timestep은 목표 timestep을 의미하는 것이 아니라, 현재 input feature가 위치한 noise level을 나타낸다.

또 하나의 중요한 점은 student의 condition 입력은 training과 inference 모두에서 null이라는 것이다.

\[
\boxed{
c_S^{train}=c_S^{infer}=\varnothing
}
\]

따라서 student는 segmentation/depth condition에 의존하지 않고, scratch BEV encoder가 만든 noisy BEV feature와 pretrained diffusion prior만으로 one-step refinement를 학습한다.

---

# 8. One-Step Distillation

Teacher의 multi-step denoised output을 student의 one-step output target으로 사용한다.

가장 단순한 형태의 distillation loss는 다음과 같다.

\[
\boxed{
\mathcal{L}_{\mathrm{distill}}
=
\|
F_0^S
-
\mathrm{sg}(F_0^T)
\|
}
\]

예를 들어 \(L_1\), \(L_2\), cosine distance 등을 사용할 수 있다.

본 방법론에서는 BEV encoder feature 자체의 direct alignment는 사용하지 않는다.

즉,

\[
\|E_S(I)-E_T(I)\|
\]

형태의 loss는 사용하지 않는다.

이는 Stage 1의 detection-pretrained BEV encoder knowledge가 직접 student encoder로 전달되는 것을 방지하고, Stage 2 성능 향상이 pretrained diffusion 및 one-step diffusion fine-tuning에서 기인하도록 하기 위함이다.

---

# 9. Downstream Task Learning

Student의 one-step denoised BEV feature를 downstream task head에 입력한다.

\[
F_0^S
\rightarrow
H_S
\rightarrow
\hat{y}
\]

Task loss를

\[
\mathcal{L}_{\mathrm{task}}
\]

라고 하면 Stage 2의 기본 objective는 다음과 같이 구성한다.

\[
\boxed{
\mathcal{L}_{\mathrm{Stage2}}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{distill}}
\mathcal{L}_{\mathrm{distill}}
}
\]

Stage 2에서 최종적으로 학습되는 모듈은

\[
E_S,\quad D_S,\quad H_S
\]

이며,

\[
D_T
\]

는 frozen teacher로만 사용한다.

---

# 10. Feature Distribution Mismatch에 대한 처리

Stage 1 diffusion은 detection-pretrained BEV encoder의 feature distribution을 이용하여 학습되었다.

반면 Stage 2에서는 BEV encoder가 scratch부터 학습되므로 초기에는

\[
p(F_0^S)
\neq
p(F_0^T)
\]

와 같은 feature distribution mismatch가 발생할 수 있다.

그러나 이를 해결하기 위해 Stage 1 BEV encoder feature와 Stage 2 BEV encoder feature를 직접 alignment하지 않는다.

대신 다음과 같은 optimization 전략을 활용한다.

### 10.1 Pretrained Diffusion Initialization

Stage 2 student diffusion은 반드시 Stage 1 pretrained diffusion으로 초기화한다.

\[
D_S^{init}=D_{\mathrm{pre}}
\]

이를 통해 Stage 1에서 학습된 BEV denoising prior를 Stage 2에 직접 전달한다.

### 10.2 Conservative Diffusion Fine-tuning

Stage 2 초반에는 pretrained diffusion representation이 급격히 손실되지 않도록 diffusion learning rate를 BEV encoder보다 작게 설정하는 것을 고려할 수 있다.

예:

\[
LR_D < LR_E
\]

### 10.3 Distillation Loss Warm-up

Stage 2 초기에는 scratch BEV encoder의 feature가 충분히 안정화되지 않았으므로,

\[
\lambda_{\mathrm{distill}}
\]

을 작은 값에서 시작하여 점진적으로 증가시키는 방식을 고려할 수 있다.

### 10.4 Normalization

필요한 경우 diffusion input에 normalization을 적용하여 activation scale 차이를 완화할 수 있다.

중요한 점은 이러한 방법들이 Stage 1 encoder feature를 직접 student에게 전달하지 않는다는 것이다.

---

# 11. Stage 2 Inference

Inference에서는 teacher를 완전히 제거한다.

최종 inference pipeline은 다음과 같다.

\[
I
\rightarrow
E_S
\rightarrow
F_0
\]

Inference에서는 대표 starting timestep \(t^*\)를 고정하여 noisy BEV feature를 구성한다.

\[
F_{t^*}
=
\sqrt{\bar{\alpha}_{t^*}}F_0
+
\sqrt{1-\bar{\alpha}_{t^*}}\epsilon
\]

그리고 student diffusion을 한 번만 호출한다.

\[
\boxed{
F_{t^*}
\xrightarrow[\mathrm{NFE}=1]{D_S(\cdot,t^*,c_S=\varnothing)}
\hat{F}_0
}
\]

마지막으로 task head를 적용한다.

\[
\hat{F}_0
\rightarrow
H_S
\rightarrow
\hat{y}
\]

따라서 최종 inference architecture는

\[
\boxed{
\text{Images}
\rightarrow
\text{Scratch-trained BEV Encoder}
\rightarrow
\text{One-step Diffusion}
\rightarrow
\text{Task Head}
}
\]

이며 multi-step teacher는 inference에 사용되지 않는다.

---

# 12. 전체 Training / Inference 구조 요약

## Stage 1

\[
I
\rightarrow
E_T^{det-pretrained,\ frozen}
\rightarrow
F_0^T
\rightarrow
F_t^T
\rightarrow
D_{\mathrm{pre}}
\]

- BEV encoder: detection-pretrained, frozen
- Diffusion: trainable
- 다양한 timestep에서 generative denoising 학습
- 결과: pretrained multi-timestep diffusion model

---

## Stage 2 Training

### Initialization

\[
\boxed{
D_T \leftarrow D_{\mathrm{pre}}
}
\]

\[
D_T:\ frozen
\]

\[
\boxed{
D_S \leftarrow D_{\mathrm{pre}}
}
\]

\[
D_S:\ trainable
\]

\[
E_S:\ scratch,\ trainable
\]

\[
H_S:\ scratch,\ trainable
\]

### Forward

\[
I
\rightarrow
E_S
\rightarrow
F_0^S
\rightarrow
F_{t_{\mathrm{train}}}
\]

이후 동일한 \(F_{t_{\mathrm{train}}}\)를 teacher와 student에 입력한다.

Teacher:

\[
F_{t_{\mathrm{train}}}
\xrightarrow[\mathrm{multi-step}]{D_T}
F_0^T
\]

Student:

\[
F_{t_{\mathrm{train}}}
\xrightarrow[\mathrm{one-step}]{D_S(c_S=\varnothing)}
F_0^S
\rightarrow
H_S
\rightarrow
\hat{y}
\]

Loss:

\[
\boxed{
\mathcal{L}
=
\mathcal{L}_{\mathrm{task}}
+
\lambda_{\mathrm{distill}}
\mathcal{L}_{\mathrm{distill}}
}
\]

---

## Stage 2 Inference

\[
I
\rightarrow
E_S
\rightarrow
F_0
\rightarrow
F_{t^*}
\xrightarrow[\mathrm{NFE}=1]{D_S(c_S=\varnothing)}
\hat F_0
\rightarrow
H_S
\rightarrow
\hat y
\]

- Teacher 제거
- Student diffusion만 사용
- Fixed inference timestep \(t^*\) 사용
- Student condition은 모두 null
- Diffusion U-Net NFE = 1

---

# 13. 방법론의 핵심 포인트

## 13.1 Stage 1 Diffusion Pre-training의 직접적 재사용

Stage 1 diffusion은 Stage 2에서 단순 teacher로만 사용하지 않는다.

\[
\boxed{
\text{Teacher Target}
+
\text{Student Initialization}
}
\]

두 방식으로 활용한다.

---

## 13.2 Multi-step Diffusion → One-step Diffusion Fine-tuning

Stage 2에서는

\[
\boxed{
D_{\mathrm{pre}}
\rightarrow
D_S^{one-step}
}
\]

으로 fine-tuning한다.

즉 Stage 1에서 학습한 generative BEV diffusion prior를 보존하면서 downstream task에 적합한 one-step denoising behavior로 변환하는 것을 목표로 한다.

---

## 13.3 Scratch BEV Encoder 유지

Stage 2 BEV encoder는 Stage 1 encoder weight을 사용하지 않고 scratch부터 학습한다.

이를 통해 downstream 성능 향상이 detection-pretrained BEV encoder의 직접적인 weight transfer나 feature distillation 때문이 아니라,

\[
\boxed{
\text{Pretrained Diffusion Prior}
+
\text{One-step Diffusion Fine-tuning}
}
\]

에서 기인하도록 설계한다.

---

## 13.4 Training-only Multi-step Teacher

Teacher diffusion은 Stage 2 training에서만 사용된다.

Inference에서는

\[
D_T
\]

를 완전히 제거하므로 최종 model complexity에는 포함되지 않는다.

---

# 14. 초기 실험 권장 설정

방법론의 feasibility를 확인하기 위한 가장 단순한 baseline은 다음과 같다.

| 항목 | 초기 설정 예시 |
|---|---|
| Total diffusion timestep | \(T=1000\) |
| Stage 2 train timestep | \(t_{\mathrm{train}}\sim\mathrm{Uniform}\{1,\dots,200\}\) |
| Stage 2 inference timestep | \(t^*=100\) |
| Teacher sampler | DDIM |
| Teacher sampling steps | 5 |
| Teacher schedule | \(t_{\mathrm{train}}\rightarrow\cdots\rightarrow0\), e.g. \(100\rightarrow80\rightarrow60\rightarrow40\rightarrow20\rightarrow0\) |
| Student sampling steps | 1 |
| Student timestep input | \(t=t_{\mathrm{train}}\) during training, \(t=t^*\) during inference |
| Student condition | Null condition \((c_S=\varnothing)\) |
| Teacher init | Stage 1 pretrained diffusion |
| Student init | **Stage 1 pretrained diffusion** |
| BEV encoder init | Scratch |
| Task head init | Scratch |
| Stage 2 loss | Task loss + one-step distillation loss |
| Stage 2 inference | Student NFE = 1 |

단, inference timestep \(t^*=100\)은 고정된 최종값이라기보다 초기 실험값으로 보고, 실제 noise schedule의 SNR과 downstream 성능을 기준으로 추가 검증하는 것이 바람직하다.

예를 들어

\[
t^* \in \{50,100,200,300\}
\]

에 대한 inference ablation을 고려할 수 있다.

Training timestep range 역시

\[
t_{\mathrm{train}} \sim \mathrm{Uniform}\{1,\dots,t_{\max}\},
\qquad
t_{\max}\in\{100,200,300\}
\]

와 같이 비교할 수 있다.

또한 teacher sampling step 역시

\[
5,\;10,\;20
\]

step을 비교하여 teacher target 품질과 training cost 사이의 trade-off를 확인할 수 있다.
