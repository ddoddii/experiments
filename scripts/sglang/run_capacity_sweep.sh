#!/bin/bash
# Phase 2 B — capacity sweep: 고정 GPU pool에서 concurrency를 올리며 각 티어가
# 언제까지 버티는지(TTFT 폭발 전) 측정. "sessions-per-GPU at fixed memory".
#
# 질문: park(spare GPU, host RAM 0)가 hicache(host DRAM)의 capacity를 따라잡나?
#   - radix   : GPU-only. 압박↑ → 축출 스래싱 → 낮은 C에서 TTFT 폭발 (capacity 바닥).
#   - hicache : + host DRAM L2. 축출분을 host로 → 높은 C까지 버팀 (incumbent).
#   - park    : + spare GPU2 풀. 같은 capacity 확장을 host RAM 0으로 하는가?
#
# 핵심 최적화: arm 설정은 concurrency와 무관 → arm당 서버 1회 시작 후 C를 루프
# (naive 15회 재시작 대신 3회). 각 (arm,C)마다 reuse delta도 수집.
#
# 사용:
#   ./scripts/sglang/run_capacity_sweep.sh
#   POOL=40000 CVALUES="8 12 16 24 32" ARMS="radix hicache park" ./scripts/sglang/run_capacity_sweep.sh
set -e
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

POOL=${POOL:-40000}
CVALUES=${CVALUES:-"8 12 16 24 32"}
ARMS=${ARMS:-"radix hicache park"}
TOOL_DELAY=${TOOL_DELAY:-3}
MAX_ITEMS=${MAX_ITEMS:-0}          # 0 = 전체(BFCL 200)
HICACHE_RATIO=${HICACHE_RATIO:-2.0}
PARK_GPU=${PARK_GPU:-2}
PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-200000}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
ROUTER_PORT=${ROUTER_PORT:-8000}
BENCH="${BENCH:-benchmark/sglang_BFCL_multi_turn_concurrent.py}"

TAG="cap_pool${POOL}_d${TOOL_DELAY}"
OUTDIR=${OUTDIR:-results/capacity/${TAG}}
mkdir -p "$OUTDIR"
SCRAPER="benchmark/phase0_metrics_scraper.py"

echo "================================================================"
echo " CAPACITY SWEEP  pool=${POOL}  C=[${CVALUES}]  arms=[${ARMS}]"
echo " bench=${BENCH}  -> ${OUTDIR}"
echo "================================================================"

wait_router() {
  echo -n "  waiting router health..."
  for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${ROUTER_PORT}/v1/models" >/dev/null 2>&1; then
      echo " OK"; return 0
    fi
    sleep 2; echo -n "."
  done
  echo " TIMEOUT"; return 1
}

start_arm() {
  local arm=$1
  case "$arm" in
    radix)   PREFILL_MAX_TOTAL_TOKENS=${POOL} CACHE_MODE=radix bash scripts/sglang/start_1P_1D.sh ;;
    hicache) PREFILL_MAX_TOTAL_TOKENS=${POOL} CACHE_MODE=hicache_host bash scripts/sglang/start_1P_1D.sh ;;
    park)    PREFILL_MAX_TOTAL_TOKENS=${POOL} CACHE_MODE=radix IDLE_KV_PARKING=1 \
               PARK_GPU=${PARK_GPU} PARK_POOL_TOKENS=${PARK_POOL_TOKENS} bash scripts/sglang/start_1P_1D.sh ;;
    *) echo "ERROR: unknown arm '${arm}'" >&2; return 1 ;;
  esac
}

for ARM in $ARMS; do
  echo ""
  echo "############################################################"
  echo "# arm: ${ARM}  (start server once, sweep C)"
  echo "############################################################"
  if ! start_arm "$ARM"; then
    echo ">> start failed for ${ARM}, skipping"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue
  fi
  wait_router || { echo "router not ready, skip ${ARM}"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue; }
  sleep 5
  HOSTMEM=$(free -m | awk '/^Mem:/{print $3}' 2>/dev/null || echo "?")
  echo "$HOSTMEM" > "$OUTDIR/hostmem_${ARM}.txt"
  echo "  host RAM used at ready: ${HOSTMEM} MB"

  for C in $CVALUES; do
    echo ""
    echo ">>> arm=${ARM} C=${C} =========================================="
    BEFORE="$OUTDIR/metrics_${ARM}_c${C}_before.json"
    AFTER="$OUTDIR/metrics_${ARM}_c${C}_after.json"
    DELTA="$OUTDIR/metrics_${ARM}_c${C}_delta.json"
    python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
        --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
        --tag before --out "$BEFORE" >/dev/null 2>&1 || true
    CONFIG="${TAG}_${ARM}_c${C}" CONCURRENCY="$C" TOOL_DELAY="$TOOL_DELAY" MAX_ITEMS="$MAX_ITEMS" \
      python "$BENCH" || echo "  (benchmark non-zero for ${ARM} C=${C})"
    python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
        --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
        --tag after --out "$AFTER" >/dev/null 2>&1 || true
    python "$SCRAPER" --delta "$BEFORE" "$AFTER" --out "$DELTA" >/dev/null 2>&1 || true
  done

  if [ "${ARM#park}" != "$ARM" ]; then
    grep -iE "DIAG|GPU[0-9]+-(park|fetch)" logs/p1.log 2>/dev/null | tail -40 > "$OUTDIR/park_diag_${ARM}.txt" || true
  fi
  bash scripts/sglang/stop_1P_1D.sh || true
  sleep 5
done

echo ""
echo "############################################################"
echo "# Capacity analysis"
echo "############################################################"
python benchmark/plot_capacity.py --outdir "$OUTDIR" --tag "$TAG" \
    --arms "$ARMS" --cvalues "$CVALUES" || true
echo "Results in $OUTDIR/"
