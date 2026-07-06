#!/bin/bash
# Phase 0 "압박(pressure)" 스윕.
#
# 목적: baseline(C=1, 큰 pool)에서는 GPU radix가 모든 prefix를 잡아 계층 캐시
#       (L2/L3)가 발동조차 안 했다. 여기서는 두 압박 knob으로 radix eviction을
#       강제해 계층 캐시가 "실제로 도는 지점"을 잡는다:
#         - PREFILL_MAX_TOTAL_TOKENS : prefill KV pool 축소 → radix eviction
#         - CONCURRENCY              : 동시 대화 → tool-call 유휴 중 타세션이 prefix evict
#
# 기대 결과:
#   radix         → 압박으로 reuse_ratio 하락 (evict된 prefix 재계산)
#   hicache_host  → evict된 prefix가 L2(host)로 → 다음 턴 load-back → reuse 회복, L2_host_used>0
#   hicache_file  → + L3(storage) prefetch → L3_prefetched>0
#
# 사용:
#   ./scripts/sglang/run_phase0_pressure.sh
#   PREFILL_MAX_TOTAL_TOKENS=20000 CONCURRENCY=24 ./scripts/sglang/run_phase0_pressure.sh
#   MODES="radix hicache_file" ./scripts/sglang/run_phase0_pressure.sh
set -e

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

# --- 압박 파라미터 (A6000 x2 = 1P1D 기준 기본값) ---
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-30000}  # 기본 pool(~245k)보다 훨씬 작게 → eviction
export CONCURRENCY=${CONCURRENCY:-16}
export MAX_ITEMS=${MAX_ITEMS:-0}     # 0=전체 200
export TOOL_DELAY=${TOOL_DELAY:-0}   # tool-call 유휴시간(초). >0이면 유휴 중 eviction을 낮은 C로도 유발

MODES=${MODES:-"radix hicache_host hicache_file"}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
ROUTER_PORT=${ROUTER_PORT:-8000}
# 결과를 (pool, C, items)별로 자동 분리 저장해 knee 스윕이 서로 덮어쓰지 않게 한다.
TAG="p${PREFILL_MAX_TOTAL_TOKENS}_c${CONCURRENCY}"
[ "$TOOL_DELAY" != "0" ] && TAG="${TAG}_d${TOOL_DELAY}"
[ "$MAX_ITEMS" != "0" ] && TAG="${TAG}_n${MAX_ITEMS}"
OUTDIR=${OUTDIR:-results/phase0_pressure/${TAG}}
CONFIG_PREFIX=${CONFIG_PREFIX:-phase0p_${TAG}}
mkdir -p "$OUTDIR"

SCRAPER="benchmark/phase0_metrics_scraper.py"
BENCH="benchmark/sglang_BFCL_multi_turn_concurrent.py"

echo "================================================================"
echo " Phase 0 PRESSURE  PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS}  CONCURRENCY=${CONCURRENCY}"
echo " modes: ${MODES}"
echo "================================================================"

wait_router() {
  echo -n "  waiting router health..."
  for _ in $(seq 1 60); do
    if curl -sf "http://127.0.0.1:${ROUTER_PORT}/health" >/dev/null 2>&1 \
       || curl -sf "http://127.0.0.1:${ROUTER_PORT}/v1/models" >/dev/null 2>&1; then
      echo " OK"; return 0
    fi
    sleep 2; echo -n "."
  done
  echo " TIMEOUT"; return 1
}

for MODE in $MODES; do
  echo ""
  echo "############################################################"
  echo "# pressure mode: ${MODE}"
  echo "############################################################"

  if ! CACHE_MODE=$MODE bash scripts/sglang/start_1P_1D.sh; then
    echo ">> start failed for ${MODE}, skipping"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue
  fi
  wait_router || { echo "router not ready, skip ${MODE}"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue; }
  sleep 5

  BEFORE="$OUTDIR/metrics_${MODE}_before.json"
  AFTER="$OUTDIR/metrics_${MODE}_after.json"
  DELTA="$OUTDIR/metrics_${MODE}_delta.json"

  python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
      --tag before --out "$BEFORE" >/dev/null

  echo ">> running CONCURRENT benchmark (C=${CONCURRENCY})..."
  CONFIG="${CONFIG_PREFIX}_${MODE}" python "$BENCH" || echo "  (benchmark non-zero for ${MODE})"

  python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
      --tag after --out "$AFTER" >/dev/null

  echo ">> recompute-vs-fetch summary (${MODE}, pressure):"
  python "$SCRAPER" --delta "$BEFORE" "$AFTER" --out "$DELTA" || true

  bash scripts/sglang/stop_1P_1D.sh || true
  sleep 5
done

echo ""
echo "############################################################"
echo "# Aggregating pressure results..."
echo "############################################################"
python benchmark/phase0_analyze.py --outdir "$OUTDIR" --config-prefix "$CONFIG_PREFIX" || true
echo "Results in $OUTDIR/"
