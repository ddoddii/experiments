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
MIN_FLOOD=${MIN_FLOOD:-}
FLOOD_MULTIPLE=${FLOOD_MULTIPLE:-}
FLOOD_TOKENS=${FLOOD_TOKENS:-}

# PARK POOL SIZING -- the park arm is only meaningful if the pool can actually hold the
# documents being swept, and the default cannot. start_2P_2D.sh defaults
# PARK_POOL_TOKENS=30000 split evenly over the candidate GPUs, so PARK_PEER=1 gives a
# 15000-token pool: from L=16384 up, ONE document does not fit and nothing is ever
# parked. Until the companion sglang fix, that failed silently (see that repo's
# "Guard oversize parks in the ring path" commit) and simply looked like the mechanism
# not helping -- park's TTFT tracked recompute exactly at 16k/32k/64k with zero hits.
#
# Two constraints, both per prefill GPU (~48GB, weights ~16GB, mem-fraction-static 0.70,
# KV bytes/token = 128 KiB for Llama-3.1-8B):
#   1. the park buffer lives OUTSIDE mem-fraction-static, so it has ~14.4GB =
#      ~110k tokens per GPU -- and PARK_PEER=1 puts TWO pools on each GPU, halving that
#      to ~55k tokens each. A single document must fit in ONE pool.
#   2. the flood parks too (every completed request is parked), so with same-length
#      flood documents the target only survives to phase 4 if the pool holds roughly
#      (1 + n_flood) documents -- ~26GB at 64k, more than any 48GB GPU can spare.
#
# FLOOD_TOKENS removes constraint 2. sglang skips parking any request shorter than
# SGLANG_KV_PARK_ON_EVICT_MIN_TOKENS (default 512), so short flood documents still
# evict from the KV pool -- total flood TOKENS is what evicts -- while never entering
# the park ring. The park pool then only has to hold the target itself, and 64k
# becomes reachable.
#
# 128k stays out of reach regardless: one 131072-token document is 17.2GB of KV and
# cannot sit beside a KV pool that must itself admit a 131072-token request, on top of
# ~16GB of weights.
#
# Working configuration for the park arm up to 64k (verified against the budgets above:
# weights+KV 25.2GB of 28.8GB, park buffer 11.8GB of 19.2GB):
#   PARK_PEER=0 PARK_POOL_TOKENS_PER_GPU=90000 PARK_MEM_FRACTION=0.60 \
#   FLOOD_TOKENS=384 FLOOD_MULTIPLE=1.5 PREFILL_MAX_TOTAL_TOKENS=70000 \
#   CONTEXT_LENS="4096,8192,16384,32768,65536"
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
    park_host)
      # THE BANDWIDTH ABLATION. Identical to `park` except SGLANG_KV_PARK_FORCE_HOST=1
      # sends every park to pinned CPU DRAM over PCIe instead of to a peer GPU. Same
      # index, same eviction, same reuse ranking, same fetch code, and the GPU park pool
      # is still ALLOCATED so the two arms are memory-matched -- only the medium moves.
      # That is what isolates link bandwidth; `park vs hicache` moves placement policy,
      # transfer software and medium all at once and cannot attribute the difference to
      # any one of them.
      #
      # HOST_MAX_GB must cover one document (8.6GB at 64k), well above the 8GB default,
      # or the host tier refuses the park and this arm silently measures nothing.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=${PARK_PEER:-1} \
        SGLANG_KV_PARK_BW_AWARE=1 \
        SGLANG_KV_PARK_FORCE_HOST=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=1 \
        SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-24} \
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

  # hicache's file backend (/tmp/hicache) failing a write is silent past this point --
  # sglang logs "Failed to save tensor ... 0 written" and keeps serving (degrading to
  # no-cache behavior) rather than crashing, so the run "succeeds" with numbers that
  # quietly stopped meaning what they're supposed to. Docs here are much bigger than
  # run_qps_sweep.sh's BFCL prompts (up to 17GB each at 128k), so the default free-space
  # bar is higher too.
  _free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
  _hc_gb=$(du -sBG /tmp/hicache 2>/dev/null | cut -f1 | tr -dc '0-9')
  echo "  disk free: ${_free_gb}G   /tmp/hicache: ${_hc_gb:-0}G"
  if [ "${_free_gb:-0}" -lt "${NEED_FREE_GB:-30}" ]; then
    echo "  ERROR: only ${_free_gb}G free (need ${NEED_FREE_GB:-30}G). Free space (rm -rf"
    echo "         /tmp/hicache, or check df -h more broadly) and re-run:"
    echo "           ARMS=\"$arm\" CONTEXT_LENS=\"$CONTEXT_LENS\" ./scripts/sglang/run_ttft_ctx_sweep.sh"
    exit 1
  fi

  PROBE_ARGS=(--arm "$arm" --context-lens "$CONTEXT_LENS"
              --pool-tokens "$PREFILL_MAX_TOTAL_TOKENS" --reps "$REPS"
              --out "$OUTDIR/$arm.json")
  [ -n "$MIN_FLOOD" ] && PROBE_ARGS+=(--min-flood "$MIN_FLOOD")
  [ -n "$FLOOD_MULTIPLE" ] && PROBE_ARGS+=(--flood-multiple "$FLOOD_MULTIPLE")
  [ -n "$FLOOD_TOKENS" ] && PROBE_ARGS+=(--flood-tokens "$FLOOD_TOKENS")

  python benchmark/ttft_ctx_probe.py "${PROBE_ARGS[@]}" \
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
