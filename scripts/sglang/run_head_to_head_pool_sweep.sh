#!/bin/bash
# Head-to-head PRESSURE sweep: run the 3-arm (radix/hicache/park) head-to-head at
# several P-GPU pool sizes to find where fetch-on-hit (4b) has room to land.
#
# 동기: pool 40000에서 park fetch의 DIAG가 nospace=32%였다 — P GPU 풀이 빡빡해
#   fetch한 KV를 stage할 자리가 없어 3분의 1이 alloc 실패. 압박을 낮추면(풀↑)
#   restore 자리가 생겨 fetch가 더 걸린다. 단 너무 키우면 radix도 evict를 덜 해
#   (reuse↑) hicache와의 격차가 닫혀 되찾을 게 없어진다 → 스윗스팟을 찾는다.
#
# 각 pool 값마다 run_head_to_head.sh 를 호출(arm 3개 back-to-back). 결과는
# results/head_to_head/h2h_p<POOL>_c<C>_d<DELAY>/ 로 pool별 분리 저장된다.
#
# 사용:
#   ./scripts/sglang/run_head_to_head_pool_sweep.sh
#   POOLS="60000 80000 120000" CONCURRENCY=8 TOOL_DELAY=3 ./scripts/sglang/run_head_to_head_pool_sweep.sh
set -e

cd "$(dirname "$0")/../.."

# pool 40000은 이미 측정됨(nospace-bound). 완화 구간을 스윕한다.
POOLS=${POOLS:-"60000 80000 120000"}
export CONCURRENCY=${CONCURRENCY:-8}
export TOOL_DELAY=${TOOL_DELAY:-3}
export MAX_ITEMS=${MAX_ITEMS:-0}

echo "================================================================"
echo " HEAD-TO-HEAD POOL SWEEP  pools=[${POOLS}]  C=${CONCURRENCY} delay=${TOOL_DELAY}s"
echo "================================================================"

for POOL in $POOLS; do
  echo ""
  echo "################  POOL=${POOL}  ################"
  PREFILL_MAX_TOTAL_TOKENS=$POOL bash scripts/sglang/run_head_to_head.sh || \
    echo ">> pool ${POOL} run failed, continuing"
done

echo ""
echo "################  CROSS-POOL SUMMARY  ################"
python benchmark/head_to_head_pool_compare.py \
  --pools "$POOLS" --c "$CONCURRENCY" --delay "$TOOL_DELAY" || true
