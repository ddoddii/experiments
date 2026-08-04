#!/bin/bash
# Exp 2 -- prefill/decode HBM imbalance. Does reusable KV end up where there is room?
#
# THIS EXPERIMENT IS ABOUT MEMORY, NOT LATENCY. The headline is per-GPU KV saturation and
# the free capacity that goes unused while another pool evicts; TTFT is recorded so the
# run can be shown not to have paid for balance with latency, not as the result.
#
# WHY NO ROUTER SKEW (unlike run_exp2_imbalance.sh, which this supersedes)
#   That script forced a P0-vs-P1 imbalance with two routers because two prefills under a
#   balanced router fill at the same rate. The P/D imbalance needs no such forcing: a
#   decode node releases a session's KV when the turn ends (pd_hbm_occupancy.py: "turn
#   종료 후 D는 token usage→0으로 복귀"), so during the 3 s tool think-time the decode
#   pools sit near empty while the prefill pools are still holding cached prefixes.
#   Demand on the two sides is ANTI-CORRELATED, structurally, which is both the ideal
#   condition for borrowing and a property of PD disaggregation rather than of a
#   workload trick. Dropping the second router also removes the failure that invalidated
#   the previous Exp 2 (503 "No available servers" on :8001).
#
# LAYOUT
#   PD_LAYOUT=b puts one prefill and one decode on each NVLink island (P0=gpu0 D0=gpu1,
#   P1=gpu2 D1=gpu3), because server17 has exactly two bridges and a PCIe-only park
#   target (3.3 GB/s) is ranked below CPU DRAM (26 GB/s) by the bandwidth-aware selector.
#   Under the old layout every P->D link was PCIe and the park_pd arm would have quietly
#   degenerated into park_local.
#
# ARMS
#   hicache        no parking. Establishes M1: is there stranded headroom at all?
#   park_local     park onto own GPU only -> separates parking from PLACEMENT
#   park_pd        own + both decode GPUs, pressure-aware       <- the claim
#   park_pd_blind  same candidates, PRESSURE_AWARE=0            <- see below
#   park_slowlink  own + peer prefill (PCIe), BW_AWARE=0        <- validates the demotion
#
#   park_pd_blind exists because "saturation became uniform" is trivially achievable by
#   any policy that writes into idle HBM, so uniformity on its own proves nothing. The
#   blind arm should reach a similar M2 while converting far less of what it parks into
#   fetch hits. If it does not, pressure-aware placement is not earning its place.
#
# 사용:
#   ARMS=hicache ./scripts/sglang/run_exp2_pd_imbalance.sh    # M1 probe first (~25 min)
#   ./scripts/sglang/run_exp2_pd_imbalance.sh                 # all five (~2h10m)
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

# --- workload (matches Exp 1 / the models sweep for Llama-3.1-8B, at higher concurrency
# so the prefill pools actually saturate; without saturation M1 is 0 by construction) ---
export CONCURRENCY=${CONCURRENCY:-16}
export MAX_TOKENS=${MAX_TOKENS:-1024}
export MAX_TURNS=${MAX_TURNS:-10}
export MIN_TURNS=${MIN_TURNS:-3}
export MAX_ITEMS=${MAX_ITEMS:-144}
# Which slice of the corpus. Repeats use disjoint slices so a result that only holds for
# one particular set of conversations shows up as a sign flip rather than as an average.
export ITEM_OFFSET=${ITEM_OFFSET:-0}
export TOOL_DELAY=${TOOL_DELAY:-3}
export TIMEOUT=${TIMEOUT:-600}

# --- server ---
export PD_LAYOUT=b
# ${VAR-default}, NOT ${VAR:-default}. With the colon an explicitly EMPTY value is
# replaced by the default, so `PREFILL_MAX_TOTAL_TOKENS= ./run...` -- the natural way to
# ask for "no cap" -- silently ran with the 60000 cap still in place and produced a probe
# that looked like a valid uncapped measurement. Without the colon, empty means empty,
# and start_2P_2D.sh already omits --max-total-tokens when the value is empty.
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS-60000}
export DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS-}
export PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70}
# Decode keeps as much of its pool as the park pools allow: the decode headroom is the
# resource under study, so trimming it hard would erase the effect being measured. The
# park pools need ~2.44 GB per decode GPU (2 prefills x 10k tokens x 128 KiB).
export PARK_MEM_FRACTION_D=${PARK_MEM_FRACTION_D:-0.80}
export PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-30000}
# Set PARK_POOL_TOKENS_PER_GPU to fix the allocation on each GPU rather than the total
# across them. That is the comparison the paper's premise calls for: idle HBM is claimed
# to be free WHERE IT EXISTS, so an arm reaching three GPUs may use three GPUs' worth,
# and the arms then differ only in reach. Equalising the total instead asks a different
# and already-answered question -- Exp 2 v1..v4 measured that one, and concentrating a
# fixed budget in one pool beat spreading it, because small pools evict independently.
#
# SIZING IT: there are two different "free" numbers and only one of them is usable.
#   free INSIDE the serving KV pool  -- what sglang:cache_occupancy reports, ~35 GB across
#     the two decode GPUs here. This is the waste Exp 2 is ABOUT, but it is already
#     allocated to the serving pool and a park pool cannot be carved from it.
#   free UNALLOCATED HBM             -- what a park pool actually needs. On a 47.4 GiB card
#     at mem-fraction 0.85 the prefill has 6.05 GB left, at 0.80 a decode has ~9.5 GB, and
#     each decode GPU hosts TWO pools (one per prefill).
# Sizing from the first number produced a 60000-token request that OOM'd the scheduler at
# startup. At ~119 KiB/token the ceiling with today's fractions is ~35k tokens/GPU
# (decode-bound). To go past that, lower PARK_MEM_FRACTION_D: the decode serving pool runs
# at 14% occupancy, so converting it into park pools leaves the decode GPU's TOTAL
# KV-capable memory roughly unchanged and merely re-partitions it.
export PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-}
export ROUTER_MODE=balanced

ARMS=${ARMS:-"hicache park_local park_pd park_pd_blind park_slowlink"}
OUTDIR=${OUTDIR:-results/exp2/pd_layoutb_p${PREFILL_MAX_TOTAL_TOKENS}_c${CONCURRENCY}}
mkdir -p "$OUTDIR"

echo "================================================================"
echo " EXP2 P/D imbalance   layout=b pool=$PREFILL_MAX_TOTAL_TOKENS C=$CONCURRENCY"
echo " decode mem-fraction=$PARK_MEM_FRACTION_D   arms: $ARMS"
# Echo what the pool knobs RESOLVED to, not what was requested. The uncapped probe was
# indistinguishable from the capped run in its own output until the pool sizes were read
# back out of imbalance.json afterwards.
echo " prefill pool : ${PREFILL_MAX_TOTAL_TOKENS:-UNCAPPED}  (mem-fraction $PARK_MEM_FRACTION)"
echo " decode  pool : ${DECODE_MAX_TOTAL_TOKENS:-UNCAPPED}  (mem-fraction $PARK_MEM_FRACTION_D)"
if [ -n "$PARK_POOL_TOKENS_PER_GPU" ]; then
  echo " park pool    : $PARK_POOL_TOKENS_PER_GPU tok on EACH candidate GPU"
else
  echo " park pool    : $PARK_POOL_TOKENS tok budget per prefill, split over candidates"
fi
echo " -> $OUTDIR"
echo "================================================================"

# Every arm sees the same decode mem-fraction, including the baselines. Otherwise the
# baseline is the only arm with full-size decode pools and every occupancy comparison
# confounds placement with capacity.
start_arm() {
  case "$1" in
    hicache)
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=file ./scripts/sglang/start_2P_2D.sh ;;
    park_local)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=0 PARK_GPUS_P1=2 \
        ./scripts/sglang/start_2P_2D.sh ;;
    park_pd)
      # own GPU + the NVLink-bridged decode + the far decode as a slow fallback.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=0,1,3 PARK_GPUS_P1=2,3,1 \
        SGLANG_KV_PARK_PRESSURE_AWARE=1 SGLANG_KV_PARK_BW_AWARE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    park_pd_blind)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=0,1,3 PARK_GPUS_P1=2,3,1 \
        SGLANG_KV_PARK_PRESSURE_AWARE=0 SGLANG_KV_PARK_BW_AWARE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    park_slowlink)
      # Peer prefill only, over PCIe. BW_AWARE must be OFF or the selector demotes the
      # peer below the local pool and this collapses into park_local.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=0,2 PARK_GPUS_P1=2,0 \
        SGLANG_KV_PARK_PRESSURE_AWARE=1 SGLANG_KV_PARK_BW_AWARE=0 \
        ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

for arm in $ARMS; do
  echo
  echo "──────────────── arm: $arm ────────────────"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
  # M3: every park records the candidates it did NOT pick. Only meaningful for the park
  # arms; harmless for hicache, which never calls _select_pool.
  rm -f "$OUTDIR/decisions_$arm".*.jsonl
  export SGLANG_KV_PARK_DECISION_LOG="$PWD/$OUTDIR/decisions_$arm"
  start_arm "$arm"

  # The occupancy series IS the result here, so it is sampled fast (0.5 s) and started
  # before the first request.
  python benchmark/kv_occupancy_timeseries.py --out "$OUTDIR/occ_$arm.csv" \
    --interval 0.5 > "$OUTDIR/sampler_occ_$arm.log" 2>&1 &
  OCC_PID=$!
  python benchmark/park_location_sampler.py --out "$OUTDIR/parked_$arm.csv" \
    --interval 1 > "$OUTDIR/sampler_park_$arm.log" 2>&1 &
  PARK_PID=$!
  python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_$arm.csv" --interval 1 \
    > "$OUTDIR/sampler_mem_$arm.log" 2>&1 &
  MEM_PID=$!
  trap 'kill $OCC_PID $PARK_PID $MEM_PID 2>/dev/null || true' EXIT

  CONFIG="exp2pd_$arm" \
    python benchmark/sglang_sharegpt_multi_turn_concurrent.py 2>&1 \
    | tee "$OUTDIR/bench_$arm.log" | tail -20

  kill $OCC_PID $PARK_PID $MEM_PID 2>/dev/null || true
  trap - EXIT
  sleep 2
  cp "results/exp2pd_$arm.json" "$OUTDIR/bench_$arm.json" 2>/dev/null \
    || echo "  [warn] results/exp2pd_$arm.json not found"
  for f in p1 p2 d1 d2; do
    cp "logs/sglang/$f.log" "$OUTDIR/${f}_$arm.log" 2>/dev/null || true
  done
done

./scripts/sglang/stop.sh > /dev/null 2>&1 || true

echo
python benchmark/collect_imbalance.py --dir "$OUTDIR" --arms $ARMS \
  --out "$OUTDIR/imbalance.json" | tee "$OUTDIR/imbalance.md"

echo
python benchmark/collect_decisions.py --dir "$OUTDIR" \
  --out "$OUTDIR/decisions.json" | tee "$OUTDIR/decisions.md"

cat <<'EOF'

────────────────────────────────────────────────────────────────
If this was the hicache-only M1 probe, read it before running anything else:

  M1 stranded headroom LARGE (tens of GB, saturated a good fraction of the run)
    -> the premise holds; run the remaining arms.

  M1 near zero
    -> STOP. Either nothing saturated (raise CONCURRENCY, or lower
       PREFILL_MAX_TOTAL_TOKENS) or the decode pools had no spare room either
       (check the per-GPU free(GB) column, and whether PARK_MEM_FRACTION_D
       trimmed decode too hard). Running the other four arms first would spend
       two hours measuring a placement policy that has nothing to place into.
────────────────────────────────────────────────────────────────
EOF
