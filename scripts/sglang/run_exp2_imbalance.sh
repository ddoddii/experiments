#!/bin/bash
# Exp 2 -- does peer-GPU placement pay off when load concentrates on ONE prefill?
#
# The question: when routing is uneven or bursty, P0's KV pool saturates while P1 still
# has headroom. Moving P0's reusable KV into P1's idle HBM should reduce local eviction,
# CPU overflow, re-prefill, and tail TTFT. This script measures whether it does.
#
# WHY THE SKEW IS FORCED, NOT HOPED FOR
#   Under the normal single router both prefills fill at the same rate, so P1 has no
#   headroom for P0 to borrow and the mechanism is invisible -- an experiment that would
#   report "no effect" for a reason that has nothing to do with the mechanism. So we run
#   one router per prefill (ROUTER_MODE=skew: :8000 -> P0, :8001 -> P1) and the client
#   assigns SESSIONS by weight (SKEW=0.9 -> 90% to P0). Sessions, not requests: a
#   conversation whose turns land on different prefills has no reusable prefix on either.
#   Decodes stay shared, so only the prefill side is skewed.
#
# ARMS (same server config except placement, so the delta is placement)
#   radix        evicted KV discarded            -> pays full re-prefill
#   hicache      evicted KV -> CPU DRAM          -> pays PCIe fetch, commits host RAM
#   park_local   parking on, PEER PLACEMENT OFF  -> P0 may only park on its own GPU,
#                                                   which is precisely the saturated one
#   park         parking on, peer placement on   -> P0 parks into P1's idle HBM
#
# park_local is the arm that isolates the contribution being claimed. Without it, "park
# beats hicache" could be entirely the local park pool, and the peer mechanism -- the
# thing the paper is about -- would be unmeasured.
#
# 사용:
#   ./scripts/sglang/run_exp2_imbalance.sh
#   SKEW=0.9 CONCURRENCY=16 ./scripts/sglang/run_exp2_imbalance.sh
#   SKEWS="0.5 0.75 0.9 1.0" ./scripts/sglang/run_exp2_imbalance.sh   # sweep
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-60000}
export CONCURRENCY=${CONCURRENCY:-16}
export TOOL_DELAY_SEC=${TOOL_DELAY:-3}
export TIMEOUT=${TIMEOUT:-600}
export ROUTER_MODE=skew

ARMS=${ARMS:-"radix hicache park_local park"}
SKEWS=${SKEWS:-${SKEW:-0.9}}
LOG_DIR=${LOG_DIR:-logs/sglang}
U0=http://127.0.0.1:8000/v1/chat/completions
U1=http://127.0.0.1:8001/v1/chat/completions

echo "================================================================"
echo " EXP2 imbalance   pool=$PREFILL_MAX_TOTAL_TOKENS C=$CONCURRENCY delay=${TOOL_DELAY_SEC}s"
echo " skews: $SKEWS   arms: $ARMS"
echo "================================================================"

start_arm() {
  case "$1" in
    radix)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 ./scripts/sglang/start_2P_2D.sh ;;
    hicache)
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=file ./scripts/sglang/start_2P_2D.sh ;;
    park_local)
      # PARK_PEER unset -> each prefill owns a pool on its OWN GPU only, so P0's only
      # park target is the GPU that is saturating. Host overflow stays ON: otherwise this
      # arm would lose for lack of ANY tier rather than for lack of a peer, and the
      # comparison would not isolate the peer mechanism.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=1 SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-8} \
        ./scripts/sglang/start_2P_2D.sh ;;
    park)
      # PARK_PEER=1 -> pools on both P GPUs at HALF the tokens each, so total park HBM
      # per GPU matches park_local and the only difference is where placement may put it.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=1 \
        SGLANG_KV_PARK_BW_AWARE=1 SGLANG_KV_PARK_PRESSURE_AWARE=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=1 SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-8} \
        ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

for skew in $SKEWS; do
  TAG="skew${skew}_p${PREFILL_MAX_TOTAL_TOKENS}_c${CONCURRENCY}"
  OUTDIR=${OUTDIR_BASE:-results/exp2}/$TAG
  mkdir -p "$OUTDIR"
  echo
  echo "############ SKEW = $skew  -> $OUTDIR ############"

  for arm in $ARMS; do
    echo
    echo "──────────────── arm: $arm (skew $skew) ────────────────"
    ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
    start_arm "$arm"

    python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_$arm.csv" --interval 1 \
      > "$OUTDIR/sampler_mem_$arm.log" 2>&1 &
    MEM_PID=$!
    python benchmark/park_location_sampler.py --out "$OUTDIR/parked_$arm.csv" --interval 1 \
      > "$OUTDIR/sampler_park_$arm.log" 2>&1 &
    PARK_PID=$!
    # Per-GPU KV pool occupancy: this is what shows the imbalance actually happened.
    # Without it a null result cannot be distinguished from "the skew never materialised".
    python benchmark/kv_occupancy_timeseries.py --out "$OUTDIR/occ_$arm.csv" \
      --interval 1 > "$OUTDIR/sampler_occ_$arm.log" 2>&1 &
    OCC_PID=$!
    trap 'kill $MEM_PID $PARK_PID $OCC_PID 2>/dev/null || true' EXIT

    python benchmark/phase0_metrics_scraper.py --tag "${arm}_before" \
      --out "$OUTDIR/metrics_${arm}_before.json" > /dev/null 2>&1 || true

    CONFIG="exp2_${TAG}_${arm}" SGLANG_URLS="$U0,$U1" SKEW="$skew" \
      python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py 2>&1 \
      | tee "$OUTDIR/bench_$arm.log" | tail -25

    python benchmark/phase0_metrics_scraper.py --tag "${arm}_after" \
      --out "$OUTDIR/metrics_${arm}_after.json" > /dev/null 2>&1 || true
    python benchmark/phase0_metrics_scraper.py --delta \
      "$OUTDIR/metrics_${arm}_before.json" "$OUTDIR/metrics_${arm}_after.json" \
      --out "$OUTDIR/metrics_${arm}_delta.json" > /dev/null 2>&1 || true

    kill $MEM_PID $PARK_PID $OCC_PID 2>/dev/null || true
    trap - EXIT
    sleep 2

    found=$(find results/sglang_hicache -name "*exp2_${TAG}_${arm}*.json" \
            -newermt '-2 hours' 2>/dev/null | head -1)
    [ -n "$found" ] && cp "$found" "$OUTDIR/bench_$arm.json" \
      || echo "  [warn] bench json not found for $arm"
    cp "$LOG_DIR"/p1.log "$OUTDIR/p1_$arm.log" 2>/dev/null || true
    cp "$LOG_DIR"/p2.log "$OUTDIR/p2_$arm.log" 2>/dev/null || true
  done

  echo
  python benchmark/collect_arm_metrics.py --dir "$OUTDIR" --arms $ARMS \
    --out "$OUTDIR/table.json" | tee "$OUTDIR/table.md"
done

./scripts/sglang/stop.sh > /dev/null 2>&1 || true
echo
echo "Read the table like this:"
echo "  park vs park_local  -> the PEER mechanism's contribution (the paper's claim)"
echo "  park vs hicache     -> host commitment removed at equal or better TTFT"
echo "  park vs radix       -> that reuse beats recompute at all"
echo "  occ_*.csv           -> proof the skew materialised (P0 saturated, P1 had headroom)"
