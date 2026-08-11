#!/bin/bash
# Codex / SWE-bench Pro trace replay: 3 arms x concurrency sweep.
#
# Arms: recompute (no cache at all), hicache (radix + host DRAM + disk, i.e. SGLang as
# shipped), park (radix + idle-GPU victim cache + host overflow). A radix-only arm is
# available via ARMS if the GPU-tier-only baseline is ever wanted, but it is not run by
# default -- hicache already contains the radix tier, so radix mainly answers a
# different question (what the extra tiers cost) than the one this sweep asks.
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
#   CONCURRENCIES="4 8 16" ARMS="recompute hicache park" \
#     ./scripts/sglang/run_agent_trace_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

ARMS=${ARMS:-"recompute hicache park"}
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
      # L2-ONLY (host DRAM), NO L3 disk tier -- HICACHE_STORAGE_BACKEND=none.
      #
      # Practical: the file backend has no size cap, and this workload can write
      # 100 sessions x 48k tokens x 128 KiB = 0.63 TB of KV pages. It filled the disk
      # mid-run, and sglang does not fail loudly when that happens -- it logs
      # "Failed to save tensor ... 0 written" and degrades to a cache miss, so the
      # point would have completed with numbers that stopped measuring hicache. The
      # start-of-point df check cannot catch that; only removing the unbounded tier can.
      #
      # Methodological, and the reason this is the right default rather than a
      # workaround: park has exactly two tiers, idle GPU HBM and host DRAM. Leaving the
      # disk tier on gives hicache a THIRD tier park does not have, so a hicache win
      # could be the disk rather than the design, and a hicache loss would be against a
      # configuration nobody would deploy for latency. L2-only holds the tier count
      # equal and is bounded by hicache-ratio x GPU KV pool (~9.4GB per prefill here).
      # HICACHE_STORAGE_BACKEND=file to put the disk tier back for an explicit A/B.
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=${HICACHE_STORAGE_BACKEND:-none} \
        ./scripts/sglang/start_2P_2D.sh ;;
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

    # Disk check BEFORE the ~80 minutes, not after. hicache's file backend fails
    # SILENTLY when the disk fills -- sglang logs "Failed to save tensor ... 0 written"
    # and keeps serving degraded to a cache miss, so the point completes with numbers
    # that quietly stopped measuring hicache. Twice now that has cost a finished
    # measurement. stop.sh clears /tmp/hicache between points, so this is about one
    # point's worth: 100 sessions at up to 48k tokens is a lot of KV pages.
    # The bar depends on whether anything will actually write KV to disk. Only the
    # hicache arm WITH the L3 file backend does, and that is no longer the default
    # (HICACHE_STORAGE_BACKEND=none), so every other point needs room for results and
    # logs only -- a few GB. A flat 40G bar is left over from when the file tier was on
    # and would now block runs that cannot fill the disk at all, which is how the
    # recompute point got refused with 33G free and /tmp/hicache empty.
    _need=${NEED_FREE_GB:-}
    if [ -z "$_need" ]; then
      if [ "$arm" = "hicache" ] && [ "${HICACHE_STORAGE_BACKEND:-none}" = "file" ]; then
        _need=40
      else
        _need=10
      fi
    fi
    _free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    _hc_gb=$(du -sBG /tmp/hicache 2>/dev/null | cut -f1 | tr -dc '0-9')
    echo "  disk free: ${_free_gb}G   /tmp/hicache: ${_hc_gb:-0}G   (need ${_need}G for $arm)"
    if [ "${_free_gb:-0}" -lt "$_need" ]; then
      echo "  ERROR: only ${_free_gb}G free (need ${_need}G). Free space and"
      echo "         re-run just this point:"
      echo "           ARMS=\"$arm\" CONCURRENCIES=\"$C\" ./scripts/sglang/run_agent_trace_sweep.sh"
      exit 1
    fi

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
