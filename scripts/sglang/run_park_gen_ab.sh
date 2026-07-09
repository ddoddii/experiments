#!/bin/bash
# park 생성-KV 기여 A/B: SGLANG_KV_PARK_GEN=0(prefix-only) vs 1(prefix+생성) 을 같은 pool에서
# 돌려 reuse_ratio를 비교한다. 생성 KV 재사용이 되면 GEN=1의 reuse가 GEN=0보다 높아야 한다.
#
# 저압박(pool 120k)에서 재는 게 깨끗하다: 축출이 없어 GEN=0은 prompt만 재사용(reuse≈0.74),
# GEN=1이 0.74를 넘으면 그 초과분이 순수 "생성 KV 재사용" 기여다(다른 요인 없음).
#
# 각 arm: start(park, GEN=$g) → /metrics before → BFCL 벤치 → /metrics after → reuse delta → stop.
#
# 사용:
#   ./scripts/sglang/run_park_gen_ab.sh
#   PREFILL_MAX_TOTAL_TOKENS=40000 CONCURRENCY=8 TOOL_DELAY=3 ./scripts/sglang/run_park_gen_ab.sh
set -e

cd "$(dirname "$0")/../.."

export CONCURRENCY=${CONCURRENCY:-8}
export TOOL_DELAY=${TOOL_DELAY:-3}
export MAX_ITEMS=${MAX_ITEMS:-0}
POOL=${PREFILL_MAX_TOTAL_TOKENS:-120000}   # 기본 저압박(생성 KV 기여 isolate)
PARK_GPU=${PARK_GPU:-2}
PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-200000}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
ROUTER_PORT=${ROUTER_PORT:-8000}

TAG="p${POOL}_c${CONCURRENCY}_d${TOOL_DELAY}"
OUTDIR=${OUTDIR:-results/park_gen_ab/${TAG}}
mkdir -p "$OUTDIR"
SCR="benchmark/phase0_metrics_scraper.py"
BENCH="benchmark/sglang_BFCL_multi_turn_concurrent.py"

echo "================================================================"
echo " PARK GEN A/B  pool=${POOL} C=${CONCURRENCY} delay=${TOOL_DELAY}s  -> ${OUTDIR}"
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

for GEN in 0 1; do
  echo ""
  echo "############  SGLANG_KV_PARK_GEN=${GEN}  ############"
  if ! CACHE_MODE=radix IDLE_KV_PARKING=1 PARK_GPU=${PARK_GPU} \
        PARK_POOL_TOKENS=${PARK_POOL_TOKENS} PREFILL_MAX_TOTAL_TOKENS=${POOL} \
        SGLANG_KV_PARK_GEN=${GEN} bash scripts/sglang/start_1P_1D.sh; then
    echo ">> start failed (GEN=${GEN}), skipping"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue
  fi
  wait_router || { echo "router not ready (GEN=${GEN})"; bash scripts/sglang/stop_1P_1D.sh || true; sleep 5; continue; }
  sleep 5

  BEFORE="$OUTDIR/gen${GEN}_before.json"; AFTER="$OUTDIR/gen${GEN}_after.json"; DELTA="$OUTDIR/gen${GEN}_delta.json"
  python "$SCR" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" --tag before --out "$BEFORE" >/dev/null

  echo ">> running BFCL (GEN=${GEN})..."
  CONFIG="park_gen${GEN}_${TAG}" python "$BENCH" || echo "  (benchmark non-zero)"

  python "$SCR" --prefill-url "http://127.0.0.1:${PREFILL_PORT}/metrics" \
      --decode-url "http://127.0.0.1:${DECODE_PORT}/metrics" --tag after --out "$AFTER" >/dev/null
  echo ">> reuse (GEN=${GEN}):"
  python "$SCR" --delta "$BEFORE" "$AFTER" --out "$DELTA" || true
  grep -iE "GPU2-park|GPU2-fetch|DIAG" logs/p1.log 2>/dev/null | tail -10 > "$OUTDIR/gen${GEN}_parklog.txt" || true

  bash scripts/sglang/stop_1P_1D.sh || true
  sleep 5
done

echo ""
echo "############  A/B 요약 (pool ${POOL})  ############"
python3 - "$OUTDIR" <<'PY'
import json, sys, os
outdir = sys.argv[1]
print(f"{'':10}{'reuse':>8}{'cached':>12}{'uncached':>12}{'TTFT(s)':>9}")
for g in (0, 1):
    p = os.path.join(outdir, f"gen{g}_delta.json")
    try:
        d = json.load(open(p))["summary"]
    except Exception:
        print(f"GEN={g}  (no delta)"); continue
    print(f"GEN={g:<6}{d.get('reuse_ratio'):>8}{d.get('cached_tokens (reuse/fetch)',0):>12,.0f}"
          f"{d.get('uncached_tokens (recompute)',0):>12,.0f}{str(d.get('ttft_avg_s'))[:7]:>9}")
print("\nGEN=1 reuse > GEN=0 reuse 이면 생성 KV 재사용이 실제로 recompute를 줄인 것.")
PY
echo "Results in $OUTDIR/"
