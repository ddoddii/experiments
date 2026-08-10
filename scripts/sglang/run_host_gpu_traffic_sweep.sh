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
#
# For 65536/131072: PREFILL_MAX_TOTAL_TOKENS MUST be raised past the largest value in
# CONTEXT_LENS (with margin -- try +10-15%), and MIN_DOCS should drop to 1. A single
# 131072-token doc alone is ~17.2GB of KV (131072 x 128 KiB/token for this model); the
# --min-docs 3 default that's fine at small L would mean ~52GB of documents at 128k,
# well past both a realistic GPU pool and hicache's host budget (hicache-ratio x device
# pool) or park's SGLANG_KV_PARK_HOST_MAX_GB (default 8GB) -- most of it would get
# evicted-before-write or dropped rather than actually round-tripping through host,
# which looks identical to "the mechanism doesn't work" (see host_gpu_traffic_probe.py's
# docstring for the same failure mode at the old --min-docs 6). Expect higher host
# RAM/VRAM use either way -- check `free -h` / `nvidia-smi` per CLAUDE.md's hicache RAM
# warning before running, and consider APPEND=1 to add 64k/128k to an already-collected
# 4k-32k run instead of re-measuring everything at the bigger pool size.
#
# 사용:
#   ARMS="recompute hicache park" CONTEXT_LENS="4096,8192,16384,32768" \
#     ./scripts/sglang/run_host_gpu_traffic_sweep.sh
#
#   # add 64k/128k to an existing run, merged into the same per-arm JSON:
#   CONTEXT_LENS="65536,131072" PREFILL_MAX_TOTAL_TOKENS=170000 MIN_DOCS=1 APPEND=1 \
#     ARMS="recompute hicache park" ./scripts/sglang/run_host_gpu_traffic_sweep.sh
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
WORK_POOL_TOKENS=${WORK_POOL_TOKENS:-}
MIN_DOCS=${MIN_DOCS:-}
MIN_FLOOD=${MIN_FLOOD:-}
APPEND=${APPEND:-0}
mkdir -p "$OUTDIR"

if [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: no model at MODEL_PATH='$MODEL_PATH' (need config.json for the tokenizer"
  echo "       and bytes-per-token calc)."
  exit 1
fi

# Preflight: a context length >= the pool can never be admitted, and the failure mode
# is a slow per-request timeout inside the probe rather than a clear error up front.
_max_L=$(echo "$CONTEXT_LENS" | tr ',' '\n' | sort -n | tail -1)
if [ "$_max_L" -ge "$PREFILL_MAX_TOTAL_TOKENS" ]; then
  echo "ERROR: largest CONTEXT_LENS ($_max_L) >= PREFILL_MAX_TOTAL_TOKENS ($PREFILL_MAX_TOTAL_TOKENS)."
  echo "       A single request that size can never be admitted into the pool it's"
  echo "       prefilling into. Raise PREFILL_MAX_TOTAL_TOKENS past $_max_L with margin,"
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

  PROBE_ARGS=(--arm "$arm" --context-lens "$CONTEXT_LENS"
              --pool-tokens "$PREFILL_MAX_TOTAL_TOKENS" --out "$OUTDIR/$arm.json")
  [ -n "$WORK_POOL_TOKENS" ] && PROBE_ARGS+=(--work-pool-tokens "$WORK_POOL_TOKENS")
  [ -n "$MIN_DOCS" ] && PROBE_ARGS+=(--min-docs "$MIN_DOCS")
  [ -n "$MIN_FLOOD" ] && PROBE_ARGS+=(--min-flood "$MIN_FLOOD")
  [ "$APPEND" = "1" ] && PROBE_ARGS+=(--append)

  python benchmark/host_gpu_traffic_probe.py "${PROBE_ARGS[@]}" \
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
