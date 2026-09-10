#!/usr/bin/env bash
# Stage1(train_only_seg.sh) 종료 → GPU 반환 확인 → Stage2(dist_train_onlyseg.sh) 자동 실행
#
# 사용법 (BEVFormer/ 에서 실행):
#   bash tools/chain_stage1_to_stage2.sh
#   nohup bash tools/chain_stage1_to_stage2.sh > /dev/null 2>&1 &   # 세션 끊겨도 유지
#
# 환경변수로 조정 가능:
#   WATCH_PID   : 감시할 Stage1 PID (기본: 패턴으로 자동 탐지)
#   GPU_IDS     : 비워지는지 확인할 GPU (기본 "4 5 6 7")
#   MEM_FREE_MB : 이 값 미만이면 "빈 GPU"로 판정 (기본 1000)
#   STABLE_N    : 연속 N회 비어야 통과 (기본 3)
#   GPU_TIMEOUT : GPU 대기 최대 초 (기본 1800)
#   REQUIRE_CKPT: Stage2가 요구하는 체크포인트 경로 (기본: dist_train_onlyseg.sh에서 파싱)
#   ALLOW_MISSING_CKPT=1 : 체크포인트 없어도 강행

set -uo pipefail

STAGE1_PATTERN="${STAGE1_PATTERN:-train_bev_diffuser_only_seg.py}"
GPU_IDS="${GPU_IDS:-4 5 6 7}"
MEM_FREE_MB="${MEM_FREE_MB:-1000}"
STABLE_N="${STABLE_N:-3}"
POLL_SEC="${POLL_SEC:-60}"
GPU_TIMEOUT="${GPU_TIMEOUT:-1800}"

TOOLS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BEVFORMER_DIR="$(dirname "$TOOLS_DIR")"
STAGE2_SCRIPT="$TOOLS_DIR/dist_train_onlyseg.sh"

LOG_DIR="${LOG_DIR:-$BEVFORMER_DIR/../results/version2/chain_logs}"
mkdir -p "$LOG_DIR"
CHAIN_LOG="$LOG_DIR/chain_$(date +%Y%m%d_%H%M%S).log"
STAGE2_LOG="$LOG_DIR/stage2_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%F %T')] $*" | tee -a "$CHAIN_LOG"; }

log "chain script 시작 (log: $CHAIN_LOG)"

# ── 1. Stage1 PID 확인 ───────────────────────────────────────────────────────
if [[ -n "${WATCH_PID:-}" ]]; then
    PIDS="$WATCH_PID"
else
    # 자기 자신/부모 셸이 패턴을 포함해 잡히는 경우 제외
    PIDS="$(pgrep -f "$STAGE1_PATTERN" | grep -vx -e "$$" -e "$PPID" | tr '\n' ' ')"
fi

if [[ -z "${PIDS// /}" ]]; then
    log "WARN: Stage1 프로세스('$STAGE1_PATTERN')를 찾지 못했습니다. 이미 종료된 것으로 보고 진행합니다."
else
    log "Stage1 감시 대상 PID: $PIDS"
    while :; do
        alive=""
        for p in $PIDS; do
            kill -0 "$p" 2>/dev/null && alive="$alive $p"
        done
        [[ -z "${alive// /}" ]] && break
        n=$(echo $alive | wc -w)
        log "Stage1 실행 중 (프로세스 ${n}개, 대기 $((SECONDS/60))분) — ${POLL_SEC}s 후 재확인"
        sleep "$POLL_SEC"
    done
    log "Stage1 프로세스 모두 종료됨."
fi

# ── 2. GPU 반환 확인 ────────────────────────────────────────────────────────
log "GPU [$GPU_IDS] 메모리 해제 대기 (임계값 ${MEM_FREE_MB}MiB, 연속 ${STABLE_N}회)"
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
    if (( waited > GPU_TIMEOUT )); then
        log "ERROR: ${GPU_TIMEOUT}s 내에 GPU가 비워지지 않아 중단합니다."
        exit 1
    fi
done
log "GPU [$GPU_IDS] 반환 확인 완료."

# ── 3. Stage2가 필요로 하는 체크포인트 확인 ─────────────────────────────────
if [[ -z "${REQUIRE_CKPT:-}" ]]; then
    ckpt_rel="$(grep -m1 '^UNET_CHECKPOINT_DIR=' "$STAGE2_SCRIPT" | cut -d'"' -f2)"
    REQUIRE_CKPT="$BEVFORMER_DIR/$ckpt_rel"
fi
if [[ -d "$REQUIRE_CKPT" ]]; then
    log "체크포인트 확인: $REQUIRE_CKPT"
else
    log "ERROR: Stage2가 요구하는 체크포인트가 없습니다: $REQUIRE_CKPT"
    log "  (Stage1이 도중에 죽었을 수 있음. 실제 존재하는 checkpoint:)"
    ls -d "$(dirname "$REQUIRE_CKPT")"/checkpoint-* 2>/dev/null | tee -a "$CHAIN_LOG"
    if [[ "${ALLOW_MISSING_CKPT:-0}" != "1" ]]; then
        log "중단합니다. (강행하려면 ALLOW_MISSING_CKPT=1)"
        exit 1
    fi
    log "ALLOW_MISSING_CKPT=1 — 그대로 진행합니다."
fi

# ── 4. Stage2 실행 ──────────────────────────────────────────────────────────
log "Stage2 시작: $STAGE2_SCRIPT (log: $STAGE2_LOG)"
cd "$BEVFORMER_DIR" || exit 1
bash "$STAGE2_SCRIPT" 2>&1 | tee "$STAGE2_LOG"
rc=${PIPESTATUS[0]}
log "Stage2 종료 (exit code=$rc)"
exit "$rc"
