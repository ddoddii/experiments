#!/bin/bash
# Phase 1 마무리: idle-KV-parking(4a) vs host-DRAM hicache 정면(head-to-head) 비교.
#
# 배경:
#   - 4a(전용 유휴 GPU park 풀)는 capacity/survival을 검증했다 (drop 0, survival 100%).
#   - 하지만 이 2xA6000 토폴로지는 NVLink가 (0-1),(2-3) 쌍에만 있어
#     P(GPU0)-park(GPU2)는 PCIe다. 즉 park 티어의 대역폭 상한 = host-DRAM 티어와 동일.
#   - 남은 질문: "그렇다면 park 경로가 기존 host-DRAM hicache 대비
#     end-to-end BFCL 워크로드에서 실제로 이득이 있는가?"를 수치로 결론낸다.
#
# 세 팔(arm)을 동일 압박(pressure)에서 back-to-back 측정한다 (cross-run drift 제거):
#   radix     GPU-only prefix cache            = park의 현실적 base (fetch 통합 없음)
#   hicache   + host-DRAM L2 (통합된 fetch 경로) = 이겨야 할 incumbent
#   park      radix + 전용 유휴 GPU park 풀(4a)  = fetch 통합 없는 저장 티어
#
# 해석 축:
#   - avg_ttft_s        : 낮을수록 좋음 (prefix 재사용이 재계산을 줄임)
#   - reuse_ratio       : cached/prompt. hicache는 높고, park는 fetch 미통합이라 radix와 같아야 함
#   - overall throughput: tok/s
# 예상: park ≈ radix < hicache. 즉 저장만으로는 이득 없음 → fetch 통합이 병목이지 대역폭이 아님.
#
# 사용:
#   ./scripts/sglang/run_head_to_head.sh
#   PREFILL_MAX_TOTAL_TOKENS=40000 CONCURRENCY=8 TOOL_DELAY=3 ./scripts/sglang/run_head_to_head.sh
#   ARMS="radix hicache park" ./scripts/sglang/run_head_to_head.sh
set -e

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

# --- 압박/워크로드 파라미터 (기존 phase0p_p40000_c8_d3 와 동일하게 잡아 비교 가능) ---
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-40000}
export CONCURRENCY=${CONCURRENCY:-8}
export TOOL_DELAY=${TOOL_DELAY:-3}
export MAX_ITEMS=${MAX_ITEMS:-0}     # 0 = 전체 200
export HICACHE_RATIO=${HICACHE_RATIO:-2.0}

# park 팔 설정 (4a: 전용 유휴 GPU2 풀)
PARK_GPU=${PARK_GPU:-2}
PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-200000}

ARMS=${ARMS:-"radix hicache park"}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
ROUTER_PORT=${ROUTER_PORT:-8000}

TAG="h2h_p${PREFILL_MAX_TOTAL_TOKENS}_c${CONCURRENCY}_d${TOOL_DELAY}"
[ "$MAX_ITEMS" != "0" ] && TAG="${TAG}_n${MAX_ITEMS}"
OUTDIR=${OUTDIR:-results/head_to_head/${TAG}}
mkdir -p "$OUTDIR"

SCRAPER="benchmark/phase0_metrics_scraper.py"
BENCH="benchmark/sglang_BFCL_multi_turn_concurrent.py"

echo "================================================================"
echo " HEAD-TO-HEAD  pool=${PREFILL_MAX_TOTAL_TOKENS} C=${CONCURRENCY} delay=${TOOL_DELAY}s"
echo " arms: ${ARMS}   -> ${OUTDIR}"
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

# arm 이름 -> start_1P_1D.sh 에 넘길 환경변수 셋
start_arm() {
  local arm=$1
  case "$arm" in
    radix)
      CACHE_MODE=radix bash scripts/sglang/start_1P_1D.sh
      ;;
    hicache)
      CACHE_MODE=hicache_host bash scripts/sglang/start_1P_1D.sh
      ;;
    park)
      # radix base + 전용 유휴 GPU park 풀 (4a). start 스크립트가 --base-gpu-id +
      # 가시성(0,1,PARK_GPU) + SGLANG_KV_PARK_GPU 전달을 처리한다.
      CACHE_MODE=radix IDLE_KV_PARKING=1 PARK_GPU=${PARK_GPU} \
        PARK_POOL_TOKENS=${PARK_POOL_TOKENS} bash scripts/sglang/start_1P_1D.sh
      ;;
    *)
      echo "ERROR: unknown arm '${arm}'" >&2; return 1
      ;;
  esac
}

for ARM in $ARMS; do
  echo ""
  echo "############################################################"
  echo "# arm: ${ARM}"
  echo "############################################################"

  if ! start_arm "$ARM"; then
    echo ">> start failed for ${ARM}, skipping"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue
  fi
  wait_router || { echo "router not ready, skip ${ARM}"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue; }
  sleep 5

  BEFORE="$OUTDIR/metrics_${ARM}_before.json"
  AFTER="$OUTDIR/metrics_${ARM}_after.json"
  DELTA="$OUTDIR/metrics_${ARM}_delta.json"

  python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
      --tag before --out "$BEFORE" >/dev/null

  echo ">> running BFCL benchmark (arm=${ARM}, C=${CONCURRENCY}, delay=${TOOL_DELAY}s)..."
  CONFIG="${TAG}_${ARM}" python "$BENCH" || echo "  (benchmark non-zero for ${ARM})"

  python "$SCRAPER" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" \
      --tag after --out "$AFTER" >/dev/null

  echo ">> reuse-vs-recompute summary (${ARM}):"
  python "$SCRAPER" --delta "$BEFORE" "$AFTER" --out "$DELTA" || true

  # park 팔이면 P 로그에서 park diag(recv/processed/survival)도 남긴다.
  if [ "$ARM" = "park" ]; then
    grep -E "idle_kv_park|park-diag|survival" logs/p1.log 2>/dev/null | tail -20 \
      > "$OUTDIR/park_diag_${ARM}.txt" || true
  fi

  bash scripts/sglang/stop_1P_1D.sh || true
  sleep 5
done

echo ""
echo "############################################################"
echo "# Head-to-head analysis"
echo "############################################################"
python benchmark/head_to_head_analyze.py --outdir "$OUTDIR" --tag "$TAG" --arms "$ARMS" || true
echo "Results in $OUTDIR/"
