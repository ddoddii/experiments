#!/bin/bash
# Host(DRAM)->GPU KV traffic vs context length, per arm -- RULER-style Figure 5(a).
#
# For each arm in ARMS: stop, start the arm's 2P2D config (same case block
# run_qps_sweep.sh uses, so this is directly comparable to those results), then run
# benchmark/host_gpu_traffic_probe.py across CONTEXT_LENS and save results/host_gpu_traffic/<arm>.json.
# At the end, plots all arms onto one figure via plot_host_gpu_traffic.py.
#
# CONTEXT_LENS must stay under the arm's serving pool (PREFILL_MAX_TOTAL_TOKENS) -- a
# single request's KV can't exceed the pool it's prefilling into. Default pool is
# whatever start_2P_2D.sh / run_models_sweep.sh's model_pool() picks for llama8b
# (~60000 tokens), so the default CONTEXT_LENS list (4096..32768) stays safely under it.
# Raise PREFILL_MAX_TOTAL_TOKENS explicitly (and expect higher host RAM/VRAM use -- see
# CLAUDE.md's hicache RAM warning) before adding 65536 or larger.
#
# 사용:
#   ARMS="recompute hicache park" CONTEXT_LENS="4096,8192,16384,32768" \
#     ./scripts/sglang/run_host_gpu_traffic_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

ARMS=${ARMS:-"recompute hicache park"}
CONTEXT_LENS=${CONTEXT_LENS:-"4096,8192,16384,32768"}
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-60000}
export MODEL_PATH=${MODEL_PATH:-/home/uhmturks/hf_models/Llama-3.1-8B-Instruct}
export MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
export PREFILL_PORTS=${PREFILL_PORTS:-30000,30001}
export PARK_DIR=${PARK_DIR:-/dev/shm/sglang_kv_parking}
OUTDIR=${OUTDIR:-results/host_gpu_traffic}
mkdir -p "$OUTDIR"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: no model at MODEL_PATH='$MODEL_PATH' (need config.json for the tokenizer"
  echo "       and bytes-per-token calc)."
  exit 1
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
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=${PARK_PEER:-1} \
        SGLANG_KV_PARK_BW_AWARE=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=${HOST_OVERFLOW:-1} \
        SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-8} \
        SGLANG_KV_PARK_REUSE_AWARE=1 ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

echo "================================================================"
echo " Host->GPU traffic sweep   model=$MODEL_PATH   pool=$PREFILL_MAX_TOTAL_TOKENS"
echo " context_lens: $CONTEXT_LENS   arms: $ARMS   -> $OUTDIR"
echo "================================================================"

for arm in $ARMS; do
  echo
  echo "════════════════ arm: $arm ════════════════"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
  # idle_kv_parking's telemetry files persist across restarts and are cumulative
  # counters -- a stale file from a PREVIOUS arm's run would make this arm's "before"
  # snapshot already nonzero and understate its true delta.
  rm -f "$PARK_DIR"/parked_gpu*.json 2>/dev/null || true
  start_arm "$arm"

  python benchmark/host_gpu_traffic_probe.py --arm "$arm" \
    --context-lens "$CONTEXT_LENS" \
    --pool-tokens "$PREFILL_MAX_TOTAL_TOKENS" \
    --out "$OUTDIR/$arm.json" \
    2>&1 | tee "$OUTDIR/probe_$arm.log"
done

./scripts/sglang/stop.sh > "$OUTDIR/stop_final.log" 2>&1 || true

echo
echo "================================================================"
echo " plotting"
echo "================================================================"
PLOT_ARGS=""
for arm in $ARMS; do
  [ -f "$OUTDIR/$arm.json" ] && PLOT_ARGS="$PLOT_ARGS --$arm $OUTDIR/$arm.json"
done
python benchmark/plot_host_gpu_traffic.py $PLOT_ARGS --out "$OUTDIR/fig_host_gpu_traffic"

echo
echo "[done] $OUTDIR/fig_host_gpu_traffic.png"
