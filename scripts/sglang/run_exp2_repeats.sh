#!/bin/bash
# Exp 2, repeated: is the p95 result reproducible, or was it one lucky run?
#
# The single uncapped run reported p95 TTFT falling from 6.58 s to 2.01 s. p95 is a tail
# statistic over ~1400 turns and is volatile; a 3x move from one measurement is not
# something to put in a paper. This runs the same comparison N times and reports the
# PAIRED difference per repeat, so the question becomes "does the sign hold on every
# independent workload" rather than "how big was it once".
#
# Each repeat draws a DIFFERENT slice of ShareGPT (ITEM_OFFSET). Both arms inside a
# repeat see the identical slice, so the comparison stays paired; across repeats the
# workload changes, so a result that only holds for one particular set of conversations
# will show up as a sign flip instead of hiding inside the average.
#
# 사용:
#   ./scripts/sglang/run_exp2_repeats.sh            # 3 repeats, ~2h30m
#   REPEATS=5 ./scripts/sglang/run_exp2_repeats.sh
set -e
cd "$(dirname "$0")/../.."

REPEATS=${REPEATS:-3}
BASE_OUT=${BASE_OUT:-results/exp2/repeat_nocap_c32}
ITEMS=${MAX_ITEMS:-288}

echo "================================================================"
echo " EXP2 repeats: $REPEATS x (park_local, park_pd), ~50 min each"
echo " each repeat uses ShareGPT items [offset, offset+$ITEMS)"
echo "================================================================"

for i in $(seq 1 "$REPEATS"); do
  off=$(( (i - 1) * ITEMS ))
  out="${BASE_OUT}_r${i}"
  echo
  echo "############ repeat $i/$REPEATS   ITEM_OFFSET=$off -> $out ############"
  if [ -f "$out/imbalance.json" ]; then
    echo "  already present, skipping (delete $out to redo)"
    continue
  fi
  OUTDIR="$out" \
  CONCURRENCY=${CONCURRENCY:-32} MAX_ITEMS="$ITEMS" MAX_TURNS=${MAX_TURNS:-12} \
  ITEM_OFFSET="$off" \
  PREFILL_MAX_TOTAL_TOKENS= PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.85} \
  PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-10000} \
  ARMS="park_local park_pd" \
    ./scripts/sglang/run_exp2_pd_imbalance.sh
done

echo
python benchmark/collect_repeats.py --dirs ${BASE_OUT}_r* \
  --out "${BASE_OUT}_summary.json" | tee "${BASE_OUT}_summary.md"
