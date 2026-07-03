#!/bin/bash
# Phase 0 baseline 오케스트레이터.
#
# 각 CACHE_MODE에 대해:
#   1) 1P1D 서버 시작
#   2) 벤치마크 직전 /metrics 스냅샷 (before)
#   3) BFCL multi-turn 벤치마크 실행 (순차 C=1)
#   4) 벤치마크 직후 /metrics 스냅샷 (after)
#   5) delta 요약 계산 (재계산 vs fetch)
#   6) 서버 종료
#
# 목표: 기존 hierarchical cache 계층별로 multi-turn 재사용률(reuse_ratio)과
#       TTFT/throughput이 어떻게 달라지는지 비교하여, "fetch가 recompute를
#       실제로 얼마나 대체하는지" baseline을 확보한다.
#
# 사용:
#   ./scripts/sglang/run_phase0_baseline.sh
#   MODES="radix hicache_file" ./scripts/sglang/run_phase0_baseline.sh   # 일부만
set -e

cd "$(dirname "$0")/../.."   # -> experiments/
ROOT=$(pwd)

MODES=${MODES:-"none radix hicache_host hicache_file hicache_file_decode_offload"}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
ROUTER_PORT=${ROUTER_PORT:-8000}
OUTDIR=${OUTDIR:-results/phase0}
mkdir -p "$OUTDIR"

SCRAPER="benchmark/phase0_metrics_scraper.py"
BENCH="benchmark/sglang_BFCL_v3_multi_turn_base.py"

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
  echo "# Phase 0 mode: ${MODE}"
  echo "############################################################"

  CACHE_MODE=$MODE bash scripts/sglang/start_1P_1D.sh
  wait_router || { echo "router not ready, skip ${MODE}"; continue; }
  sleep 5   # 안정화

  BEFORE="$OUTDIR/metrics_${MODE}_before.json"
  AFTER="$OUTDIR/metrics_${MODE}_after.json"
  DELTA="$OUTDIR/metrics_${MODE}_delta.json"

  python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
      --tag before --out "$BEFORE" >/dev/null

  echo ">> running benchmark..."
  CONFIG="phase0_${MODE}" python "$BENCH" || echo "  (benchmark returned non-zero for ${MODE})"

  python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
      --tag after --out "$AFTER" >/dev/null

  echo ">> recompute-vs-fetch summary (${MODE}):"
  python "$SCRAPER" --delta "$BEFORE" "$AFTER" --out "$DELTA" || true

  bash scripts/sglang/stop_1P_1D.sh || true
  sleep 5
done

echo ""
echo "############################################################"
echo "# All modes done. Aggregating..."
echo "############################################################"
python benchmark/phase0_analyze.py --outdir "$OUTDIR" || true
echo "Results in $OUTDIR/"
