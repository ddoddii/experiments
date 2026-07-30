#!/bin/bash
# Exp 1 -- GPU-first unified placement vs the two incumbents, three arms back to back.
#
#   radix     GPU-only prefix cache. KV evicted from the local pool is DISCARDED.
#             The recompute baseline: whatever it loses to must be worth its cost.
#   hicache   SGLang HiCache. Evicted KV is offloaded to CPU DRAM (L2) and, with the
#             file backend, to disk (L3). The incumbent, and the arm whose host
#             commitment we claim to remove.
#   park      Proposed. Evicted KV goes to the peer GPU with the most headroom first,
#             CPU DRAM only on overflow, location tracked in a shared index, next turn
#             prefetched to the target GPU.
#
# Back to back in ONE invocation rather than three sessions because the host page cache,
# GPU fragmentation and the shared box's other tenants all drift between sessions, and
# the memory numbers are the result -- a cross-session comparison of peak host memory is
# not a comparison of the arms.
#
# Every arm is instrumented identically (mem breakdown + park telemetry + /metrics delta),
# then collect_arm_metrics.py reduces the four artifacts per arm into the paper table.
#
# 사용:
#   ./scripts/sglang/run_exp1_placement.sh
#   CONCURRENCY=16 TOOL_DELAY=3 PREFILL_MAX_TOTAL_TOKENS=60000 \
#     ./scripts/sglang/run_exp1_placement.sh
#   ARMS="hicache park" ./scripts/sglang/run_exp1_placement.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."
ROOT=$(pwd)

# --- workload / pressure -----------------------------------------------------------
# The pool must be small enough that KV is actually EVICTED -- with a pool that holds the
# whole working set all three arms are identical and the experiment measures nothing.
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-60000}
export CONCURRENCY=${CONCURRENCY:-16}
# Tool delay is the mechanism's premise, not a detail: it is the idle window during which
# a conversation's KV is cold and gets evicted. At 0 there is no window and parking has
# nothing to park.
export TOOL_DELAY_SEC=${TOOL_DELAY:-3}
# The KV-flush flood MUST be off here. _kv_flush() fires 30 concurrent 3000-token
# requests every time a tool delay elapses; at C=16 that is up to 480 concurrent
# requests / 1.44M tokens against a 60k-token prefill pool -- 24x oversubscription. It
# aborts the in-flight requests (finish_reason="abort"), the aborts fail the Mooncake KV
# transfers, the router opens every prefill circuit, and the rest of the run is 503.
# One sweep was lost exactly this way, in the radix arm, with parking and hicache both
# off. The flood exists for a different setup (a LARGE pool, where eviction has to be
# forced); here the pool is deliberately small and saturates on its own -- the failed run
# measured prefill token_usage at 0.9988 with the flood as the thing that killed it.
export FLOOD_DURING_DELAY=${FLOOD_DURING_DELAY:-0}
export MAX_ITEMS=${MAX_ITEMS:-0}
export TIMEOUT=${TIMEOUT:-600}

ARMS=${ARMS:-"radix hicache park"}
TAG=${TAG:-"p${PREFILL_MAX_TOTAL_TOKENS}_c${CONCURRENCY}_d${TOOL_DELAY_SEC}"}
OUTDIR=${OUTDIR:-results/exp1/$TAG}
LOG_DIR=${LOG_DIR:-logs/sglang}
ROUTER=${ROUTER:-http://127.0.0.1:8000}
mkdir -p "$OUTDIR"

BENCH=benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
SCRAPER=benchmark/phase0_metrics_scraper.py

echo "================================================================"
echo " EXP1 placement   pool=$PREFILL_MAX_TOTAL_TOKENS C=$CONCURRENCY delay=${TOOL_DELAY_SEC}s"
echo " arms: $ARMS   -> $OUTDIR"
echo "================================================================"

start_arm() {
  case "$1" in
    radix)
      # No hierarchical cache and no parking: eviction = discard. PARK_NO_HICACHE=1 is
      # what drops the hicache args (there is no ENABLE_HICACHE knob).
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 \
        ./scripts/sglang/start_2P_2D.sh ;;
    hicache)
      # SGLang's OWN default: write_through_selective + file backend. This is the
      # incumbent as shipped, which is what the claim has to beat; measured at 76.6 GB
      # AnonPages + 108.6 GB page cache.
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=file \
        ./scripts/sglang/start_2P_2D.sh ;;
    hicache_wb)
      # Conservative variant: write_back never populates the L3 file backend (measured
      # hicache_dir = 0), so this arm isolates the HOST TIER from the page cache. A
      # reviewer who objects that the default arm's win is partly page cache gets
      # answered by this row: AnonPages is still 76.0 GB.
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_back \
        HICACHE_STORAGE_BACKEND=none \
        ./scripts/sglang/start_2P_2D.sh ;;
    park)
      # PARK_PEER=1 is the proposal: pools on both P GPUs so placement can pick the
      # idler one. Without it each prefill can only park on its own GPU and the "peer
      # GPU first" claim is not what runs.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=${PARK_PEER:-1} \
        SGLANG_KV_PARK_BW_AWARE=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=${HOST_OVERFLOW:-1} \
        SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-8} \
        SGLANG_KV_PARK_REUSE_AWARE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

for arm in $ARMS; do
  echo
  echo "──────────────── arm: $arm ────────────────"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
  start_arm "$arm"

  # samplers AFTER the servers are up: the mem sampler fixes its per-GPU columns and its
  # pid set on the first probe, and the park sampler needs the telemetry dir to exist
  python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_$arm.csv" --interval 1 \
    > "$OUTDIR/sampler_mem_$arm.log" 2>&1 &
  MEM_PID=$!
  python benchmark/park_location_sampler.py --out "$OUTDIR/parked_$arm.csv" --interval 1 \
    > "$OUTDIR/sampler_park_$arm.log" 2>&1 &
  PARK_PID=$!
  # Kill the samplers even if the benchmark dies: a sampler left running past the run
  # appends a flat post-run tail that drags every time-weighted mean toward idle.
  trap 'kill $MEM_PID $PARK_PID 2>/dev/null || true' EXIT

  python $SCRAPER --tag "${arm}_before" --out "$OUTDIR/metrics_${arm}_before.json" \
    > /dev/null 2>&1 || echo "  [warn] scraper before failed"

  CONFIG="exp1_${TAG}_${arm}" SGLANG_URL="$ROUTER/v1/chat/completions" \
    python $BENCH 2>&1 | tee "$OUTDIR/bench_$arm.log" | tail -25

  python $SCRAPER --tag "${arm}_after" --out "$OUTDIR/metrics_${arm}_after.json" \
    > /dev/null 2>&1 || true
  python $SCRAPER --delta "$OUTDIR/metrics_${arm}_before.json" \
    "$OUTDIR/metrics_${arm}_after.json" \
    --out "$OUTDIR/metrics_${arm}_delta.json" > /dev/null 2>&1 || true

  kill $MEM_PID $PARK_PID 2>/dev/null || true
  trap - EXIT
  sleep 2

  # The benchmark writes under results/sglang_hicache/<model>/; copy it to the arm's
  # expected name so collect_arm_metrics can find all four artifacts in one directory.
  found=$(find results/sglang_hicache -name "*exp1_${TAG}_${arm}*.json" -newermt '-2 hours' \
          2>/dev/null | head -1)
  if [ -n "$found" ]; then
    cp "$found" "$OUTDIR/bench_$arm.json"
    echo "  bench -> $OUTDIR/bench_$arm.json"
  else
    echo "  [warn] could not locate the bench json for $arm; table row will be partial"
  fi
  cp "$LOG_DIR"/p1.log "$OUTDIR/p1_$arm.log" 2>/dev/null || true
  cp "$LOG_DIR"/p2.log "$OUTDIR/p2_$arm.log" 2>/dev/null || true
done

./scripts/sglang/stop.sh > "$OUTDIR/stop_final.log" 2>&1 || true

echo
echo "================================================================"
python benchmark/collect_arm_metrics.py --dir "$OUTDIR" --arms $ARMS \
  --out "$OUTDIR/table.json" | tee "$OUTDIR/table.md"
echo "================================================================"
echo "artifacts: $OUTDIR"
