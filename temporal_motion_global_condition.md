# Temporal Motion BEV as a Global Diffusion Condition

Stage 1의 semantic BEV conditioning에 temporal 정보를 **global embedding**으로 추가하는
설계 문서. 대상 실험은 `results/bevdiffuser/BEVDiffuser_tiny_onlyseg_sam3_metric3d`이며,
Stage 1/2 전반은 `methodology_stage1_stage2_summary.md`를 따른다.

---

## 1. 문제: mAVE만 악화된다

`DiffBEVFormer_tiny_onlyseg_metric3d_t100_alpha100_lambda_0.55` vs baseline
(`baseline/DiffBEVFormer_tiny_original/..._rerun`), nuScenes val, epoch 24:

| TP error | baseline | metric3d | |
| --- | --- | --- | --- |
| trans_err | 0.8667 | **0.8495** | 개선 |
| scale_err | 0.2909 | **0.2871** | 개선 |
| orient_err | 0.5788 | **0.5691** | 개선 |
| attr_err | 0.2208 | **0.2179** | 개선 |
| **vel_err** | **0.6089** | 0.6284 | **악화 (유일)** |

단일 프레임에서 추론 가능한 속성은 전부 좋아지고, 운동량만 나빠진다.

### 이것은 semantic condition 고유의 문제가 아니다

epoch 24까지 완주한 distillation run 전부를 mAVE로 정렬하면:

| run | teacher condition | mAP | NDS | mAVE |
| --- | --- | --- | --- | --- |
| **baseline (MGD 없음)** | — | 0.2785 | 0.3826 | **0.6089** |
| metric3d t100 | semantic BEV | 0.2851 | 0.3873 | 0.6284 |
| original_mgd | **GT layout** | 0.2872 | 0.3843 | 0.6354 |
| no-gate | semantic BEV | 0.2840 | 0.3840 | 0.6461 |
| sam_mgd_v1 (da3) | semantic BEV | 0.2830 | 0.3800 | 0.6503 |
| concat | semantic BEV | 0.2841 | 0.3805 | 0.6550 |
| t150 | semantic BEV | 0.2850 | 0.3817 | 0.6642 |
| t10 | semantic BEV | 0.2823 | 0.3772 | 0.6659 |
| dino t100 | semantic+DINO | 0.2756 | 0.3645 | 0.7100 |

**8개 run 전부 baseline보다 나쁘다.** GT layout을 조건으로 쓰는 `original_mgd`도 예외가
아니다. 즉 conditioning 종류의 문제가 아니라 distillation 경로 전반의 문제다.

(`t50`, `dino_alpha1000`, `obsmask`는 epoch 12에서 멈춰 있어 제외했다. epoch 24 run과
비교 불가능하다.)

### 메커니즘: 조건부 평균이 motion을 지운다

Stage 1의 objective는 `prediction_type='sample'` + 가중치 없는 MSE다. x0-regression에서
MSE의 최적해는 조건부 평균이다.

```
E[ ||f(x_t, c) - x_0||^2 ]  ->  f*(x_t, c) = E[x_0 | x_t, c]
```

BEVFormer-tiny의 속도는 temporal self-attention이 `prev_bev`와 현재 BEV 사이에 실어
나르는 **잔차**에서 나온다. 어떤 단일 프레임 조건도 이 잔차를 설명할 수 없으므로, 조건부
평균에서 평균으로 회귀 = 소거된다. teacher의 x0 예측에 motion residual이 체계적으로 빠지고,
`loss_bev * alpha=100`이 student를 그쪽으로 끌어당긴다.

이 설명은 위 표와 정확히 맞는다 — 단일 프레임에서 추론 가능한 것은 보존/개선되고,
불가능한 것만 악화된다.

### teacher 자체의 mAVE

teacher BEV를 직접 디코딩한 값:

| teacher | mAP | mAVE |
| --- | --- | --- |
| original (GT layout) @5_5_5 | 0.3943 | **0.2513** |
| metric3d (semantic) @100_100_5 | 0.2477 | **0.6662** |
| *(참고) baseline student* | *0.2785* | *0.6089* |

metric3d teacher는 **그 자체로 baseline student보다 속도가 나쁘다**. 조건이 motion을 담지
못한다는 직접 증거다.

### 기대치 조정 (중요)

GT layout teacher는 mAVE 0.2513으로 압도적인데도 그 student는 0.6354로 오히려 더 나쁘다.
**teacher의 속도 품질이 student로 전이되지 않는다.** MGD의 masked reconstruction이 student
BEV를 "부분 관측으로부터 복원 가능"하게 강제하고, 이는 공간적으로 매끄러운 BEV를 선호하게
만들어 temporal residual을 깎는다. 이 압력은 teacher 내용과 무관하게 1차적으로 작동한다.

따라서 **temporal conditioning은 teacher 쪽 병목을 풀지만 MGD 쪽 병목은 남는다.**
직접적인 NDS 이득은 크지 않다 — mAVE를 baseline 수준(0.6089)까지 회복해도 `delta/10 = +0.002`,
0.55까지 낮춰야 `+0.008`이다. 더 큰 가치는 속도 손실 없이 alpha를 올릴 여지가 생긴다는 쪽,
그리고 "단일 프레임 conditioning은 x0-regression 하에서 원리적으로 motion을 보존할 수 없다"는
original BEVDiffuser가 할 수 없는 주장을 확보한다는 쪽이다.

---

## 2. 왜 SBAM이 아니라 global embedding인가

### SBAM은 motion을 표현할 수 없다

```python
out = GroupNorm(x) * (1 + gamma(seg_bev)) + beta(seg_bev)
```

위치별 affine 변조다. 각 cell에서 채널 통계를 조절할 뿐 **아무것도 옮기지 못한다.**
motion은 본질적으로 transport 연산 — t-1에 `(x,y)`에 있던 feature가 t에 `(x+u, y+v)`에
있다는 관계다. 연산자로 보면 SBAM은 대각 연산자이고 transport는 비대각 연산자가 필요하다.
용량 문제가 아니라 **표현 불가능**이다.

두 번째로 SBAM은 **decoder 전용**이다. motion residual은 저에너지/고주파라 encoder를
통과하며 이미 소실된 뒤다.

> 설계 원칙: **static semantics는 SBAM으로, motion은 다른 경로로.**

### 주입 슬롯 지도

| # | 슬롯 | 현재 | encoder 도달 | 공간적 | transport |
| --- | --- | --- | --- | --- | --- |
| 1 | input concat | v5_2만 | O | O | X |
| 2 | **`emb` (timestep)** | `seg_global_embed` | **O** | X | X |
| 3 | SBAM gamma/beta | O | X (decoder만) | O | X |
| 4 | SBAM gate | O (스칼라 2) | X | X | X |
| 5 | cross-attn `attn2` | **제거됨** | O | O | **O** |
| 6 | encoder ResBlock | `emb`만 | O | X | X |
| 7 | skip `hs` | X | — | O | X |

본 문서는 **2번(`emb`)** 을 택한다. 근거:

- SBAM을 완전히 우회하면서 **encoder/middle/decoder 전부에 도달**한다.
- `seg_global_embed`라는 동일 구조의 선례가 이미 있어 일관적이다.
- 구현량이 가장 작고, 기존 경로를 한 줄도 바꾸지 않는다.

한계: 공간적으로 균일하다. "장면이 얼마나 동적인가"는 전달하지만 "어디가 동적인가"는
전달하지 못한다. 이것으로 부족하면 순서는 4번(spatial gate) -> 5번(cross-attention)이다.

### 참고: channel stacking(Option A)을 먼저 하지 않는 이유

K개 프레임을 채널로 쌓아 conv로 읽게 하는 방식은 **도달거리가 부족하다.**
조건 인코더의 receptive field (cell = 102.4/50 = 2.048 m):

| 단계 | RF | 실거리 |
| --- | --- | --- |
| `prob_to_emb` | 3 cell | 6.1 m |
| `down0` -> **scale 1** | **5 cell** | **10.2 m (반경 ±4.1 m)** |
| `down1` -> scale 2 | 7 cell | 14.3 m |
| `down2` -> scale 4 | 11 cell | 22.5 m |

ego-warp 이후 남는 객체 자체의 지면 이동량:

| 속도 | dt=0.5s | dt=1.0s | dt=1.5s |
| --- | --- | --- | --- |
| 15 m/s | 3.7 cell | 7.3 cell | 11.0 cell |
| 30 m/s | 7.3 cell | 14.6 cell | 22.0 cell |

가장 짧은 간격에 54 km/h 차량도 3.7 cell 움직이는데 scale 1의 반경은 ±2 cell이다.
SBAM이 가장 조밀하게 주입하는 스케일에서 프레임 간 연관 자체가 불가능하다.
dilated stack으로 RF를 늘릴 수는 있으나(1->2->4->8이면 RF 33, 반경 16 cell), global
embedding보다 구현량이 크고 검증할 것이 많다.

---

## 3. Motion BEV 생성

### 3.1 데이터 경로 (구현 완료)

`data_utils.py`의 `CustomNuScenesDiffusionDataset_seg_depth_v2`에
`use_semantic_temporal` 플래그를 추가했다. **기본 False이며 그때는 기존 동작과 완전히
동일하다.**

켜면 큐 프레임별로 다음을 방출한다:

| key | shape | 내용 |
| --- | --- | --- |
| `temporal_semantic_bev` | `[K, C, H, W]` | 프레임별 캐시된 semantic BEV |
| `temporal_lidar2global` | `[K, 4, 4]` | 그 프레임의 강체 변환 |
| `temporal_timestamp` | `[K]` | 초 단위, dt 정규화용 |
| `temporal_semantic_valid` | `[K]` | 큐가 프레임을 반복한 자리는 False |

인덱스는 학습 큐 순서를 따른다 — **`-1`이 현재 프레임**, `0..K-2`가 과거. 현재 프레임을
스택에 남기는 이유는 상대 변환이 양 끝을 모두 필요로 하기 때문이다:

```
T_rel = inv(lidar2global[i]) @ lidar2global[-1]
```

현재 프레임의 확률은 `semantic_bev_probabilities`로 **변경 없이** 따로 나간다. 기존 SBAM
경로가 그대로 돌아가므로 두 경로를 독립적으로 ablation할 수 있다.

`use_semantic_bev_cache=True`가 전제다 (history를 per-token 캐시에서 읽는다). 아니면
`__init__`에서 즉시 에러를 낸다.

### 3.2 변환을 `can_bus`나 `meta_keys`에서 가져오지 않는 이유

semantic BEV는 각 프레임 자신의 lidar 좌표계에 캐시되어 있으므로 t-1을 t에 융합하려면
`T_rel = inv(lidar2global(t)) @ lidar2global(t-1)`이 필요하다.

**`can_bus`로는 불가능하다.** `can_bus[-2]`와 `can_bus[-1]`(ego yaw)이
`nuscenes_infos_temporal_{train,val}.pkl` 전체에서 항상 0이다 (28130 / 6019 샘플, 전수 확인).
ego yaw = 0이면 `PerceptionTransformer`가 유도하는 shift가 global 좌표계에 남고,
`rotate()` 호출은 no-op이 된다.

**`meta_keys`로도 불가능하다.** `lidar2ego_{translation,rotation}`은 `input_dict`에 아예
들어가지 않고, 기본 `meta_keys`에 있는 `ego2lidar`는 occupancy 데이터셋에서만 채워져
여기서는 `<<MISSING>>`으로 도착한다. 상수로 하드코딩할 수도 없다 — train+val 전체에
**3종의 서로 다른 센서 캘리브레이션**이 존재한다.

그래서 변환은 전체 annotation 레코드를 가진 유일한 위치인 `data_utils.py`의
`self.data_infos`에서 만든다. 큐 프레임이 무작위로 뽑히므로 인덱스가 아니라 토큰으로
조회한다 (`self._info_by_token`).

### 3.3 검증: 정적 클래스 soft-IoU

warp가 맞았는지는 **정적 클래스의 정합도**로 잰다. 정적 클래스는
`2 highway, 5 terrain, 6 tree, 7 sidewalk, 11 building`.

```python
def soft_iou(a, b):                 # 임계값 없음
    return np.minimum(a, b).sum() / np.maximum(a, b).sum()
```

val 연속 쌍 600개:

| warp | static soft-IoU | |
| --- | --- | --- |
| none (warp 안 함) | 0.7467 | |
| **can_bus warp (현재 BEVFormer 구현)** | **0.7043** | **안 하느니 못함** |
| **올바른 강체 변환** | **0.7658** | 72.0%의 쌍에서 개선 |
| 부호 반전 대조군 | 0.6902 | |

`lidar2ego`를 생략하면 안 된다 (400쌍): none 0.7496 / 포함 0.7676 / **생략 0.7015**.
lidar 장착 오프셋 `(0.944, 0, 1.84) m`이 프레임 간 회전에서 상쇄되지 않는다.

데이터셋이 실제로 방출한 `temporal_lidar2global`로 end-to-end 재검증 (120쌍):

```
none = 0.6397   emitted-warp = 0.6791   delta = +0.0393   개선 74.2%   PASS
```

**통과 기준**: `correct > none` **그리고** `flipped < none`. 두 조건이 동시에 만족되면
부호와 축 순서가 맞은 것이다.

### 3.4 warp는 raw semantic BEV에 한다

`prob_to_emb` **이전**이다. 이유 넷:

1. **보간의 의미가 성립한다.** `P`는 물리 격자 위의 확률 히스토그램이고 bilinear 보간은
   분포의 볼록결합이라 결과도 유효한 분포다. `prob_to_emb`(Conv3x3 -> SiLU -> Conv1x1)
   이후에는 각 cell이 3x3 이웃의 비선형 함수라 그 보장이 없다.
2. **zero = 미관측 의미가 깨진다.** `P`에서 정확히 0은 "관측 없음"이고
   `grid_sample(padding_mode='zeros')`가 이를 보존한다. `prob_to_emb` 이후에는 conv bias
   때문에 0 입력이 0 출력이 아니다 — feature 공간의 zero padding은 학습된 "미관측" 표현이
   아니라 무의미한 리터럴 0이다.
3. **`bev_pos_embed`를 같이 옮기면 안 된다.** 이는 현재 ego 기준 격자 위치를 인코딩하는
   학습 파라미터다. `encode_prob` 이후에 warp하면 positional embedding이 내용물과 함께
   움직여 "ego로부터 어디인가"가 틀어진다.
4. **비용.** `[B,17,50,50]` vs `[B,256,50,50]`, 15배 차이.

결과적으로 **Option C는 `encode_prob`과 `project_semantics`를 전혀 건드리지 않는다.**

```
P_t, P_{t-1}  (raw 17ch, cache)
     |
     +--> encode_prob(P_t) ---------------> SBAM        (기존 경로, 무변경)
     |
     +--> warp(P_{t-1} -> t)                            (raw에서 warp)
            dP = (P_t - warped) / dt
            |dP| 또는 relu 분할
            motion_global_embed ----------> emb         (신규 경로)
```

---

## 4. 구현 계획 (S4 ~ S6)

### S4 — `SemanticMotionEncoder`

**위치**: `layout_diffusion/seg_bev_aligner_one_hot_v5.py`에 신규 클래스.
`SegBEVAligner.encode_prob` / `project_semantics`는 손대지 않는다.

```python
class SemanticMotionEncoder(nn.Module):
    def __init__(self, num_classes, bev_h, bev_w, pc_range,
                 history=1, mode='abs'):        # mode: 'abs' | 'relu_split'
        ...
        self.register_buffer('_cell_xy', ..., persistent=False)   # 셀 중심 1회 캐시

    @torch.no_grad()
    def forward(self, probs, mats, stamps, valid):
        # probs [B,K,C,H,W], mats [B,K,4,4], stamps [B,K], valid [B,K]
        return motion                                    # [B, C 또는 2C, H, W]
```

처리 순서:

1. **history 프레임 선택** — 뒤에서부터 가장 가까운 `valid[j]=True`인 `j < K-1`.
   없으면 motion = 0. `history=1`(K=2 상당)로 시작한다. K=3의 다단계 변환 합성은
   불필요한 위험이다.
2. **상대 변환** `T_rel = inv(mats[j]) @ mats[-1]` (lidar_t -> lidar_j)
3. **샘플링 그리드** — 셀 중심 `(X,Y)`를 `T_rel`로 보내고 `[-1,1]` 정규화.
   셀 중심은 버퍼로 캐시 (`project_semantics`의 `_pixel_grid` 방식).
4. **`grid_sample`** `mode='bilinear'`, `padding_mode='zeros'`, `align_corners=True`
5. **dt 정규화** `dP = (probs[-1] - warped) / clamp(dt, 1e-3)`
6. **부호 처리** — 아래 참조

구현 주의:

- `@torch.no_grad()` + autocast 비활성화로 fp32 고정 (`project_semantics`와 동일)
- 배치 전체를 한 번에 `grid_sample` (프레임 루프 금지)
- 씬 시작 / 중복 프레임 / `dt ~ 0` -> 전부 0 텐서

#### GAP 부호 상쇄 (Option C의 핵심 함정)

움직이는 차량은 **새 위치에서 `dP > 0`, 옛 위치에서 `dP < 0`** 이다. global embedding은
GAP를 거치므로 `dP`를 그대로 쓰면 **둘이 상쇄되어 0에 수렴한다.**

Option C의 목적은 "이 장면이 얼마나 동적인가"이므로 부호보다 크기가 중요하다:

```python
motion = dP.abs()                                       # mode='abs'
motion = torch.cat([F.relu(dP), F.relu(-dP)], dim=1)    # mode='relu_split'
```

(나중에 spatial gate나 cross-attention으로 확장할 때는 부호 있는 `dP`가 필요하다.
dataset 출력을 raw stack으로 둔 이유가 이것이다.)

#### dt 정규화가 필요한 이유

`prepare_train_data`가 큐를 셔플한다:

```python
index_list = list(range(index - queue_length, index))
random.shuffle(index_list)
index_list = sorted(index_list[1:])
index_list.append(index)
```

`queue_length=3`이면 `{t-3, t-2, t-1}` 중 2개를 무작위로 골라 현재와 합친다. 따라서
dt가 샘플마다 0.5 / 1.0 / 1.5초로 달라진다 (실측: 한 샘플에서 `dt = [0.499, 1.0]`).
정규화하지 않으면 같은 속도가 샘플마다 다른 `dP` 크기로 나타난다.

### S5 — `motion_global_embed`

**위치**: `layout_diffusion/seg_diffusion_unet_v5.py`. `seg_global_embed`를 미러링한다.

```python
# __init__  (기본 False -> 기존 체크포인트/동작 불변)
self.use_motion_emb = use_motion_emb
if use_motion_emb:
    self.motion_encoder = SemanticMotionEncoder(**motion_encoder)
    self.motion_global_embed = nn.Sequential(
        nn.Conv2d(motion_ch, C, 3, padding=1), SiLU(),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(1),
        nn.LayerNorm(C), linear(C, C), SiLU(),
        zero_module(linear(C, time_embed_dim)),      # 필수
    )

# forward  (kwarg 추가, 기본 None)
def forward(self, x, timesteps, img_metas, seg_cond=None, depth_maps=None,
            seg_bev_maps=None, semantic_prob=None,
            temporal_bev=None, temporal_mats=None,
            temporal_time=None, temporal_valid=None):
    ...
    if self.use_global_seg:
        emb = emb + self.seg_global_embed(seg_bev_maps[1].to(emb))
    if self.use_motion_emb and temporal_bev is not None:
        motion = self.motion_encoder(temporal_bev, temporal_mats,
                                     temporal_time, temporal_valid)
        emb = emb + self.motion_global_embed(motion.to(emb))
```

- `zero_module` 초기화 -> 학습 시작 시 정확히 현재 모델과 동일
- `temporal_bev=None`이면 아무 일도 없음 -> 기존 config 그대로 동작
- SBAM / `output_blocks` / `encode_prob` 전혀 무변경

**`motion_encoder`는 UNet이 소유한다.** `self.seg_aligner`와 `encode_seg()`가 이미
확립한 패턴이고, 학습 스크립트에 두면 `tools/bevdiffuser_onlyseg.py` 래퍼와
`test_bev_diffuser_only_seg.py`에 같은 코드를 세 번 복제해야 한다.

### S6 — 학습 루프 배선

**위치**: `train_bev_diffuser_only_seg.py`. 인코더를 호출하지 않고 배치 키만 넘긴다.

```python
cond_kwargs = {}
if 'semantic_bev_probabilities' in batch:
    semantic_prob = stack(batch['semantic_bev_probabilities'])
    temporal = None
    if 'temporal_semantic_bev' in batch:
        temporal = dict(
            temporal_bev   = stack(batch['temporal_semantic_bev']),
            temporal_mats  = stack(batch['temporal_lidar2global']),
            temporal_time  = stack(batch['temporal_timestamp']),
            temporal_valid = stack(batch['temporal_semantic_valid']),
        )
    # CFG dropout: 같은 draw 하나로 둘 다 끈다
    if np.random.rand() < args.uncond_prob:
        semantic_prob = torch.zeros_like(semantic_prob)
        if temporal is not None:
            temporal['temporal_bev'] = torch.zeros_like(temporal['temporal_bev'])
    cond_kwargs['semantic_prob'] = semantic_prob
    if temporal: cond_kwargs.update(temporal)
    seg_cond = None
```

**dropout은 한 번만 뽑아 둘 다 적용해야 한다.** 현재 `get_segmaps_cond`는 호출할 때마다
새로 뽑으므로 그대로 두 번 부르면 uncond일 때 motion만 살아남아 조건이 누수된다.

### S6b — `BEVDiffuser` 래퍼 통과

**위치**: `tools/bevdiffuser_onlyseg.py`

```python
def forward(self, x, img_metas, segmaps=None, depth_maps=None,
            grad_fn=None, semantic_prob=None,
            temporal_bev=None, temporal_mats=None,
            temporal_time=None, temporal_valid=None):
```

이것이 있어야 **stage-2 학습에서 teacher가 실제로 motion 조건을 받는다.** 빠뜨리면
stage-1만 temporal이고 stage-2 teacher는 조건 없이 도는, 조용히 어긋난 상태가 된다.

### Config

```python
# bev_tiny_onlyseg_sam_v2_metric3d_temporal.py
unet = dict(parameters=dict(
    use_motion_emb=True,
    motion_encoder=dict(history=1, mode='abs'),
))
```

`_base_` 상속이므로 dict 병합으로 들어간다.

---

## 5. 범위 밖으로 둔 것

### BEVFormer의 temporal 경로는 건드리지 않는다

`PerceptionTransformer`의 `prev_bev` 정렬은 위 3.2/3.3에서 본 대로 기하학적으로 부정확하다.
그럼에도 손대지 않는 이유:

- **일관된 계통 오차는 학습으로 흡수된다.** shift가 잘못된 프레임에서 계산되어도 그것은
  ego 운동의 결정론적 함수이고, TSA의 학습된 offset과 `can_bus_mlp`(`use_can_bus=True`)가
  매 스텝 같은 패턴을 보며 보상하도록 학습된다. 실제로 baseline은 0.2785 / 0.3826으로
  정상 범위다.
- 바꾸면 baseline, `original_mgd`, 완주한 8개 run 전부와 비교가 깨지고 재학습해야 한다.
- 하나의 변경만 검증한다는 원칙 — BEVFormer 내부 수정과 신규 motion 조건을 섞으면
  축이 두 개인 비교가 된다.

motion 분기는 신규 코드라 적응할 기존 동작이 없으므로 처음부터 올바른 변환을 쓰는 비용이 0이다.
좌표계 충돌도 없다 — 잘못된 정렬은 `prev_bev`를 섞는 방식에만 영향을 줬고 출력 BEV는 어느
경우든 현재 프레임 좌표계다.

### 추론 시 과거 semantic BEV는 쓰지 않는다

| 단계 | diffuser | 과거 semantic BEV |
| --- | --- | --- |
| stage-2 **inference** | **전혀 사용 안 함** (`forward_train`에만 등장) | 해당 없음 |
| stage-2 **training** | teacher로 사용 | `data.train`에 queue 있음 -> 사용 |
| stage-1 **eval** | 진단용 | queue 없음 -> 0 |

diffuser는 **train-time-only teacher**다. 최종 시스템의 추론 경로에는 semantic BEV가
현재 프레임조차 들어가지 않는다 — student는 그냥 BEVFormer다.

teacher가 실제로 쓰이는 stage-2 학습에서는 queue가 있으므로 history가 정상적으로 들어간다.

### stage-1 eval의 한계

`data.val` / `data.test`는 `queue_length`를 설정하지 않고 `test_pipeline`을 쓴다.
`_temporal_metas`가 test 증강에 대해 단일 프레임을 반환하므로 K=1이고 motion은 0이 된다.

이는 **배포 경로가 아니라 진단 도구의 한계**이며 S4~S6를 막지 않는다. 다만 "teacher mAVE
게이트"를 쓰려면 진단이 학습과 같은 조건이어야 하므로, 필요해질 때 다음 중 하나로 해결한다:

| 방법 | 내용 |
| --- | --- |
| A | `data.val/test`에 `queue_length` 부여 (config) |
| B | `__getitem__`에서 `prev_idx`로 과거 토큰 조회 |
| **C** | `_info_by_token` + `info['prev']` 체인 — **학습/평가가 같은 코드 경로가 되고 dt 분포도 안정** |

---

## 6. 검증 게이트

| # | 작업 | 게이트 |
| --- | --- | --- |
| 1 | `SemanticMotionEncoder` (S4) | 정적 IoU: `correct > none` **그리고** `flipped < none` |
| 2 | `use_motion_emb` (S5) | off일 때 기존과 파라미터/loss 동일. on + motion=0일 때도 동일 (zero_module 확인) |
| 3 | 학습 루프 배선 (S6) | uncond step에서 motion이 실제로 0인지 |
| 4 | 래퍼 kwarg 통과 (S6b) | stage-2에서 teacher가 motion을 받는지 |
| 5 | stage-1 50k 학습 | — |
| 6 | teacher mAVE (현재 0.6662) | 의미 있게 내려가지 않으면 stage-2 중단 (단, 5절의 eval 한계 선결) |

---

## 7. 현재 상태

**완료**

- `data_utils.py`: `use_semantic_temporal` 플래그, `lidar2global()`,
  `_temporal_semantic_bev()`, 네 개 배치 키 방출. 교체된 줄 2개, 나머지는 전부 추가.
- `projects/configs/bevdiffuser/bev_tiny_onlyseg_sam_v2_metric3d_temporal.py`:
  `_base_` 상속, reference config 대비 차이 8개
  (`use_semantic_bev_cache`, `use_semantic_temporal` x 3 데이터셋 + 헬퍼 dict).
- 검증: flag OFF에서 `temporal_*` 키 누출 없음. flag ON에서 shape/dtype/dt/valid 정상,
  `temporal_semantic_bev[-1] == semantic_bev_probabilities` 확인,
  end-to-end 정적 IoU PASS.

**미착수**: S4, S5, S6, S6b, config의 `use_motion_emb`.

**참고 구현**: warp 검증 스크립트는 `warp_correct()` / `soft_iou()` 형태로 별도 보관 중이며,
S4를 배치 경로로 구현한 뒤 같은 수치가 재현되는지 대조하는 데 쓴다.
