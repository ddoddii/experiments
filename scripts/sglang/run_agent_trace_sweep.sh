#!/bin/bash
# Codex / SWE-bench Pro trace replay: 4 arms x concurrency sweep.
#
# Arms: recompute (no cache at all), radix (GPU prefix cache only), hicache (radix +
# host DRAM + disk), park (radix + idle-GPU victim cache + host overflow).
# radix is included here and not in the context-length sweep because it is the honest
# "what does SGLang already give you" baseline once the workload actually reuses
# prefixes -- hicache's extra tiers only pay when the GPU tier overflows.
#
# PARK SIZING -- the part that decides whether the park arm measures anything.
#   Sessions peak at ~48k tokens = 6.3GB of KV each (128 KiB/token, Llama-3.1-8B).
#   1. SESSION_KEYED=1 IS MANDATORY. In the default ring layout every turn parks its own
#      version, so one 12-turn session spanning 12k->48k parks ~360k tokens (47GB) and
#      the pool thrashes without a single reusable entry surviving. Session-keyed gives
#      each conversation ONE slab it overwrites in place.
#   2. The slab must hold the PEAK context, so SLAB_TOKENS >= 48000. The default 6000
#      would send every one of these sessions straight to the host tier (the
#      `if n > pool.slab` branch), and the GPU arm would measure host DRAM.
#   3. The park buffer lives OUTSIDE --mem-fraction-static: ~14.4GB at 0.70, ~19.2GB at
#      0.60, i.e. 2-3 slabs per GPU, 4-6 sessions parked across two prefills. That is
#      why the concurrency sweep is informative -- C=4 fits, C=8/16 overflow to host and
#      then to eviction, which is exactly the victim-cache gradient being measured.
#
# The serving KV pool is 1.4x (C=4) to 5.7x (C=16) oversubscribed at these context
# lengths, so eviction happens on its own. Unlike the context-length probe this workload
# needs no synthetic flood.
#
# 사용:
#   ./scripts/sglang/run_agent_trace_sweep.sh
#   CONCURRENCIES="4 8 16" ARMS="recompute radix hicache park" \
#     ./scripts/sglang/run_agent_trace_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

ARMS=${ARMS:-"recompute radix hicache park"}
CONCURRENCIES=${CONCURRENCIES:-"4 8 16"}
export SESSIONS=${SESSIONS:-data/codex_sessions_100.json}
export MODEL_PATH=${MODEL_PATH:-/home/uhmturks/hf_models/Llama-3.1-8B-Instruct}
export MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
export TOOL_DELAY=${TOOL_DELAY:-3}
export MAX_TOKENS=${MAX_TOKENS:-1024}
export TIMEOUT=${TIMEOUT:-600}
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-60000}
export PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.60}
export SGLANG_KV_PARK_SESSION_KEYED=${SGLANG_KV_PARK_SESSION_KEYED:-1}
export SGLANG_KV_PARK_SLAB_TOKENS=${SGLANG_KV_PARK_SLAB_TOKENS:-48000}
PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-144000}   # 3 slabs = 18.9GB
export PARK_POOL_TOKENS_PER_GPU
OUTDIR=${OUTDIR:-results/agent_trace}
PARK_DIR=${PARK_DIR:-/dev/shm/sglang_kv_parking}
mkdir -p "$OUTDIR"

if [ ! -f "$SESSIONS" ]; then
  echo "ERROR: $SESSIONS not found. Build it first:"
  echo "  MODEL_PATH=$MODEL_PATH python benchmark/agent_trace.py \\"
  echo "    --file data/codex_swebenchpro/codex_swebenchpro.json --select \\"
  echo "    --min-turns 8 --max-turns 15 --min-ctx 12000 --max-ctx 48000 \\"
  echo "    --n-sessions 100 --max-out-tokens 1024 --out $SESSIONS"
  exit 1
fi

# THE GATE. A park lookup hashes an exact token sequence, so if this model's chat
# template is not append-only over these traces every fetch misses, fetch_hits stays 0,
# and the park arm produces a healthy-looking figure that measured nothing -- which is
# how two Qwen3-14B runs were lost (results/models/qwen14b/INVALID.md). Checking costs
# seconds against hours of GPU time.
if [ "${SKIP_APPEND_ONLY_CHECK:-0}" != "1" ]; then
  echo "=== append-only gate ==="
  MODEL_PATH="$MODEL_PATH" python benchmark/agent_trace.py \
    --file "$SESSIONS" --verify-append-only --limit 20 || {
      echo "  >>> FAILED. Not starting any servers. Fix the template/trace handling"
      echo "  >>> first, or SKIP_APPEND_ONLY_CHECK=1 if you accept a park arm that"
      echo "  >>> cannot hit."
      exit 1
    }
fi

start_arm() {
  case "$1" in
    recompute)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 DISABLE_RADIX_CACHE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    radix)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 ./scripts/sglang/start_2P_2D.sh ;;
    hicache)
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=file ./scripts/sglang/start_2P_2D.sh ;;
    park)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=${PARK_PEER:-0} \
        SGLANG_KV_PARK_BW_AWARE=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=1 \
        SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-32} \
        SGLANG_KV_PARK_REUSE_AWARE=1 ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

echo "================================================================"
echo " Agent-trace replay   sessions=$SESSIONS"
echo " arms: $ARMS   concurrency: $CONCURRENCIES   tool_delay=${TOOL_DELAY}s"
echo " park: session-keyed, slab=${SGLANG_KV_PARK_SLAB_TOKENS} tok, "
echo "       pool=${PARK_POOL_TOKENS_PER_GPU} tok/GPU, mem-fraction=${PARK_MEM_FRACTION}"
echo " -> $OUTDIR"
echo "================================================================"

for arm in $ARMS; do
  for C in $CONCURRENCIES; do
    echo
    echo "════════════════ arm=$arm  C=$C ════════════════"
    ./scripts/sglang/stop.sh > "$OUTDIR/stop_${arm}_c${C}.log" 2>&1 || true
    # Cumulative counters keyed by GPU, not by run: a leftover file would make this
    # point's "before" snapshot nonzero and understate its delta.
    rm -f "$PARK_DIR"/parked_gpu*.json 2>/dev/null || true
    start_arm "$arm"

    python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_${arm}_c${C}.csv" \
      --interval 2 > "$OUTDIR/sampler_mem_${arm}_c${C}.log" 2>&1 &
    MEM_PID=$!
    python benchmark/park_location_sampler.py --out "$OUTDIR/parked_${arm}_c${C}.csv" \
      --interval 2 > "$OUTDIR/sampler_park_${arm}_c${C}.log" 2>&1 &
    PARK_PID=$!
    trap 'kill $MEM_PID $PARK_PID 2>/dev/null || true' EXIT

    python benchmark/phase0_metrics_scraper.py --tag "${arm}_c${C}_before" \
      --out "$OUTDIR/metrics_${arm}_c${C}_before.json" > /dev/null 2>&1 || true

    CONFIG="agent_${arm}_c${C}" CONCURRENCY="$C" \
      python benchmark/sglang_agent_trace_concurrent.py \
      2>&1 | tee "$OUTDIR/bench_${arm}_c${C}.log" | tail -30
    cp -f "results/agent_${arm}_c${C}.json" "$OUTDIR/bench_${arm}_c${C}.json" 2>/dev/null || true

    python benchmark/phase0_metrics_scraper.py --tag "${arm}_c${C}_after" \
      --out "$OUTDIR/metrics_${arm}_c${C}_after.json" > /dev/null 2>&1 || true
    python benchmark/phase0_metrics_scraper.py --delta \
      --before "$OUTDIR/metrics_${arm}_c${C}_before.json" \
      --after "$OUTDIR/metrics_${arm}_c${C}_after.json" \
      --out "$OUTDIR/metrics_${arm}_c${C}_delta.json" > /dev/null 2>&1 || true

    kill $MEM_PID $PARK_PID 2>/dev/null || true
    sleep 2
  done
done

./scripts/sglang/stop.sh > "$OUTDIR/stop_final.log" 2>&1 || true
echo
echo "[done] $OUTDIR"
echo "  next: python benchmark/collect_arm_metrics.py --dir $OUTDIR --arms $ARMS ..."
