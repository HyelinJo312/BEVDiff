#!/usr/bin/env bash
# Stage1(projects/bevdiffuser/train_only_seg_cache.sh) 종료 → GPU 4-7 반환 확인
#   → Stage2(tools/dist_train_onlyseg_cache_det_v2.sh) 자동 실행
#
# 사용법 (BEVFormer/ 에서 실행):
#   bash tools/chain_stage1_to_stage2.sh
#   nohup bash tools/chain_stage1_to_stage2.sh > chain.out 2>&1 &   # 백그라운드 실행 시 로그 저장
#
# 환경변수로 조정 가능:
#   WATCH_PID      : 감시할 Stage1 PID (기본: 패턴으로 자동 탐지, 공백 구분 다중 지정 가능)
#   GONE_N         : Stage1 프로세스가 연속 N회 안 보이면 종료로 판정 (기본 2)
#   STAGE1_PATTERN : Stage1 탐지용 pgrep -f 패턴
#   STAGE2_SCRIPT  : 이어서 실행할 스크립트 (기본: tools/dist_train_onlyseg_cache_det_v2.sh)
#   GPU_IDS        : 비워지는지 확인할 GPU (기본 "4 5 6 7" — Stage1/Stage2가 쓰는 GPU)
#   MEM_FREE_MB    : 이 값 미만이면 "빈 GPU"로 판정 (기본 1000)
#   STABLE_N       : 연속 N회 비어야 통과 (기본 3)
#   GPU_TIMEOUT    : GPU 대기 최대 초 (기본 0: 제한 없음)
#   REQUIRE_CKPT   : Stage2가 요구하는 체크포인트 경로 (기본: Stage2 스크립트에서 파싱)
#   ALLOW_MISSING_CKPT=1 : 체크포인트 없어도 강행

set -uo pipefail

# 실행 중인 train_only_seg_cache.sh의 Python 명령과 출력 디렉터리를 함께 매칭한다.
# 다른 GPU에서 실행 중인 같은 학습 스크립트는 감시 대상에서 제외한다.
STAGE1_PATTERN="${STAGE1_PATTERN:-train_bev_diffuser_only_seg[.]py.*--output_dir .*/BEVDiffuser_tiny_onlyseg_sam3_metric3d_v2([[:space:]]|$)}"
GPU_IDS="${GPU_IDS:-4 5 6 7}"
MEM_FREE_MB="${MEM_FREE_MB:-1000}"
STABLE_N="${STABLE_N:-3}"
POLL_SEC="${POLL_SEC:-60}"
GPU_TIMEOUT="${GPU_TIMEOUT:-0}"

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEVFORMER_DIR="$(dirname "$TOOLS_DIR")"
STAGE2_SCRIPT="${STAGE2_SCRIPT:-$TOOLS_DIR/dist_train_onlyseg_cache_det_v2.sh}"

LOG_DIR="${LOG_DIR:-$BEVFORMER_DIR/../results/chain_logs}"
mkdir -p "$LOG_DIR"
CHAIN_LOG="$LOG_DIR/chain_$(date +%Y%m%d_%H%M%S).log"
STAGE2_LOG="$LOG_DIR/stage2_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

log "chain script 시작 (log: $CHAIN_LOG)"
log "  Stage1 패턴 : $STAGE1_PATTERN"
log "  Stage2      : $STAGE2_SCRIPT"
log "  감시 GPU    : [$GPU_IDS]"

[[ -f "$STAGE2_SCRIPT" ]] || { log "ERROR: Stage2 스크립트가 없습니다: $STAGE2_SCRIPT"; exit 1; }

# ── 1. Stage1 종료 대기 ─────────────────────────────────────────────────────
# dataloader worker 는 epoch 마다 새로 뜨므로 PID 목록을 고정해두면 "옛 worker 가
# 전부 죽음 = 학습 종료" 로 오판할 수 있다. 그래서 매 폴링마다 다시 탐색하고,
# 연속 GONE_N 회 아무것도 잡히지 않을 때만 종료로 판정한다.
GONE_N="${GONE_N:-2}"

stage1_procs() {
    if [[ -n "${WATCH_PID:-}" ]]; then
        for p in $WATCH_PID; do kill -0 "$p" 2>/dev/null && echo "$p"; done
    else
        # 자기 자신/부모 셸이 패턴을 포함해 잡히는 경우 제외
        pgrep -f "$STAGE1_PATTERN" | grep -vx -e "$$" -e "$PPID"
    fi
}

first="$(stage1_procs | tr '\n' ' ')"
if [[ -z "${first// /}" ]]; then
    log "ERROR: Stage1 프로세스('${WATCH_PID:-$STAGE1_PATTERN}')를 찾지 못했습니다. 감시 대상을 확인한 뒤 다시 실행하세요."
    exit 1
else
    log "Stage1 감시 시작 (현재 프로세스 $(echo $first | wc -w)개: $(echo $first | cut -c1-120)...)"
    gone=0
    while (( gone < GONE_N )); do
        alive="$(stage1_procs | tr '\n' ' ')"
        if [[ -z "${alive// /}" ]]; then
            gone=$((gone + 1))
            log "Stage1 프로세스 없음 ($gone/$GONE_N)"
            (( gone < GONE_N )) && sleep 15
        else
            gone=0
            log "Stage1 실행 중 (프로세스 $(echo $alive | wc -w)개, 대기 $((SECONDS/60))분) — ${POLL_SEC}s 후 재확인"
            sleep "$POLL_SEC"
        fi
    done
    log "Stage1 프로세스 모두 종료됨."
fi

# ── 2. GPU 반환 확인 ────────────────────────────────────────────────────────
log "GPU [$GPU_IDS] 해제 대기 (임계값 ${MEM_FREE_MB}MiB, 연속 ${STABLE_N}회)"
gpu_free_once() {
    for g in $GPU_IDS; do
        used="$(nvidia-smi --id="$g" --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null)"
        if [[ -z "$used" ]]; then
            log "  GPU $g 조회 실패"
            return 1
        fi
        if (( used >= MEM_FREE_MB )); then
            log "  GPU $g 사용 중: ${used}MiB"
            return 1
        fi
        # 메모리가 적어도 compute process가 남아 있으면 아직 반환 전
        apps="$(nvidia-smi --id="$g" --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d '[:space:]')"
        if [[ -n "$apps" ]]; then
            log "  GPU $g 에 compute process 잔존: $apps"
            return 1
        fi
    done
    return 0
}

stable=0
waited=0
while (( stable < STABLE_N )); do
    if gpu_free_once; then
        stable=$((stable + 1))
        log "  GPU 전부 비어있음 ($stable/$STABLE_N)"
        (( stable < STABLE_N )) && sleep 10 && waited=$((waited + 10))
    else
        stable=0
        sleep 30
        waited=$((waited + 30))
    fi
    if (( GPU_TIMEOUT > 0 && waited >= GPU_TIMEOUT )); then
        log "ERROR: ${GPU_TIMEOUT}s 내에 GPU가 비워지지 않아 중단합니다."
        exit 1
    fi
done
log "GPU [$GPU_IDS] 반환 확인 완료."

# ── 3. Stage2가 필요로 하는 입력 파일 확인 ──────────────────────────────────
# Stage2 스크립트의 경로들은 BEVFormer/ 기준 상대경로이므로 여기서 먼저 이동한다.
cd "$BEVFORMER_DIR" || exit 1

parse_var() {  # parse_var VARNAME → Stage2 스크립트 안의 VARNAME="..." 값
    grep -m1 "^$1=" "$STAGE2_SCRIPT" | cut -d'"' -f2
}

if [[ -z "${REQUIRE_CKPT:-}" ]]; then
    REQUIRE_CKPT="$(parse_var UNET_CHECKPOINT_DIR)"
fi
if [[ -d "$REQUIRE_CKPT" ]]; then
    log "UNet 체크포인트 확인: $REQUIRE_CKPT"
else
    log "ERROR: Stage2가 요구하는 UNet 체크포인트가 없습니다: $REQUIRE_CKPT"
    log "  (Stage1이 도중에 죽었을 수 있음. 실제 존재하는 checkpoint:)"
    ls -d "$(dirname "$REQUIRE_CKPT")"/checkpoint-* 2>/dev/null | tee -a "$CHAIN_LOG"
    if [[ "${ALLOW_MISSING_CKPT:-0}" != "1" ]]; then
        log "중단합니다. (강행하려면 ALLOW_MISSING_CKPT=1)"
        exit 1
    fi
    log "ALLOW_MISSING_CKPT=1 — 그대로 진행합니다."
fi

STAGE2_CONFIG="$(parse_var CONFIG)"
STAGE2_LOAD_FROM="$(parse_var LOAD_FROM)"
for f in "$STAGE2_CONFIG" "$STAGE2_LOAD_FROM"; do
    if [[ -n "$f" && ! -e "$f" ]]; then
        log "ERROR: Stage2 입력 파일이 없습니다: $f"
        exit 1
    fi
done
log "Stage2 config/pretrained 확인: $STAGE2_CONFIG , $STAGE2_LOAD_FROM"

# ── 4. Stage2 실행 ──────────────────────────────────────────────────────────
log "Stage2 시작: $STAGE2_SCRIPT (log: $STAGE2_LOG)"
bash "$STAGE2_SCRIPT" 2>&1 | tee "$STAGE2_LOG"
rc=${PIPESTATUS[0]}
log "Stage2 종료 (exit code=$rc)"
exit "$rc"
