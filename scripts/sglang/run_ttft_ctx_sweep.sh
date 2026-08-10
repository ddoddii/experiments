#!/bin/bash
# TTFT vs context length, per arm -- companion to run_host_gpu_traffic_sweep.sh, same
# arm-startup case block (so results are directly comparable) and same CONTEXT_LENS /
# PREFILL_MAX_TOTAL_TOKENS constraints -- see that script's header for the full
# explanation of why a context length must stay under the pool, and what raising it to
# 65536/131072 costs in host RAM / VRAM.
#
# For each arm in ARMS: stop, start the arm's 2P2D config, run
# benchmark/ttft_ctx_probe.py across CONTEXT_LENS, save results/ttft_ctx/<arm>.json.
# At the end, plots all arms onto one figure via plot_ttft_ctx.py.
#
# 사용:
#   ARMS="recompute hicache park" CONTEXT_LENS="4096,8192,16384,32768,65536,131072" \
#   PREFILL_MAX_TOTAL_TOKENS=170000 REPS=3 \
#     ./scripts/sglang/run_ttft_ctx_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

ARMS=${ARMS:-"recompute hicache park"}
CONTEXT_LENS=${CONTEXT_LENS:-"4096,8192,16384,32768,65536,131072"}
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-170000}
export MODEL_PATH=${MODEL_PATH:-/home/uhmturks/hf_models/Llama-3.1-8B-Instruct}
export MODEL=${MODEL:-meta-llama/Llama-3.1-8B-Instruct}
export PREFILL_PORTS=${PREFILL_PORTS:-30000,30001}
export PARK_DIR=${PARK_DIR:-/dev/shm/sglang_kv_parking}
OUTDIR=${OUTDIR:-results/ttft_ctx}
REPS=${REPS:-3}
mkdir -p "$OUTDIR"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: no model at MODEL_PATH='$MODEL_PATH' (need config.json for the tokenizer)."
  exit 1
fi

_max_L=$(echo "$CONTEXT_LENS" | tr ',' '\n' | sort -n | tail -1)
if [ "$_max_L" -ge "$PREFILL_MAX_TOTAL_TOKENS" ]; then
  echo "ERROR: largest CONTEXT_LENS ($_max_L) >= PREFILL_MAX_TOTAL_TOKENS ($PREFILL_MAX_TOTAL_TOKENS)."
  echo "       Raise PREFILL_MAX_TOTAL_TOKENS past $_max_L with margin,"
  echo "       e.g. PREFILL_MAX_TOTAL_TOKENS=$((_max_L * 13 / 10))."
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
echo " TTFT vs context-length sweep   model=$MODEL_PATH   pool=$PREFILL_MAX_TOTAL_TOKENS"
echo " context_lens: $CONTEXT_LENS   arms: $ARMS   reps=$REPS   -> $OUTDIR"
echo "================================================================"

for arm in $ARMS; do
  echo
  echo "════════════════ arm: $arm ════════════════"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
  rm -f "$PARK_DIR"/parked_gpu*.json 2>/dev/null || true
  start_arm "$arm"

  python benchmark/ttft_ctx_probe.py --arm "$arm" \
    --context-lens "$CONTEXT_LENS" \
    --pool-tokens "$PREFILL_MAX_TOTAL_TOKENS" \
    --reps "$REPS" \
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
python benchmark/plot_ttft_ctx.py $PLOT_ARGS --out "$OUTDIR/fig_ttft_ctx"

echo
echo "[done] $OUTDIR/fig_ttft_ctx.png"
