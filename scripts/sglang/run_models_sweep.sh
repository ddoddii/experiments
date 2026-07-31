#!/bin/bash
# The cross-model figure: run the same three arms on each model, then plot one row of
# panels with model on the x-axis.
#
# MODEL SIZE IS NOT A FREE PARAMETER HERE. 2P2D gives each of the four processes ONE
# A6000 (49 GB), so the weights must fit on a single card with room left for the KV pool
# and activations:
#
#   Llama-3.1-8B   BF16   16.0 GB   ->  33.0 GB free   fits comfortably
#   Qwen3-14B      BF16   29.6 GB   ->  19.4 GB free   fits
#   Qwen3-30B      BF16   61.0 GB   ->  DOES NOT FIT   (needs TP=2, i.e. 8 GPUs for 2P2D)
#   Qwen3-30B      FP8    30.5 GB   ->  18.5 GB free   fits
#   Qwen3-30B      AWQ-4  15.2 GB   ->  33.8 GB free   fits, closest to the 8B's headroom
#
# So the 30B row must be a quantised checkpoint. Running it TP=2 instead would use all
# four GPUs for 1P1D and leave NO peer prefill GPU to park onto -- which removes the
# mechanism under test, so that is not an option for this figure.
#
# WHAT SHRINKING HEADROOM DOES TO THE RESULT
#   Free HBM per GPU falls 33 -> 19 -> 18 GB across the three models, so the idle GPU HBM
#   the proposal exploits shrinks with model size, and HiCache's host pool -- sized as
#   GPU-KV x hicache-ratio -- shrinks with it. Expect the host-DRAM saving to NARROW on
#   the larger models. That is a real property of the mechanism on a fixed GPU budget,
#   not a measurement problem, and the figure should show it rather than hide it.
#
# PREFILL POOL IS SCALED PER MODEL, not held at 60k. A 60k-token pool is a different
# fraction of free HBM for each model, so holding it fixed would confound model size with
# pressure. PREFILL_POOL_FRAC keeps the pool at the same fraction of what is available.
#
# 사용:
#   ./scripts/sglang/run_models_sweep.sh                    # all three
#   MODELS="qwen14b" ./scripts/sglang/run_models_sweep.sh   # one
#   ARMS="radix park" ./scripts/sglang/run_models_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

HF=${HF_HOME_MODELS:-/home/uhmturks/hf_models}
MODELS=${MODELS:-"llama8b qwen14b qwen30b"}
ARMS=${ARMS:-"radix hicache park"}
OUTBASE=${OUTBASE:-results/models}
mkdir -p "$OUTBASE"

# model key -> path | pool tokens | extra server env
# Pools chosen to sit at a similar fraction of each model's free HBM (~35-40%), so the
# three models see comparable eviction pressure rather than comparable absolute capacity.
model_path() {
  case "$1" in
    llama8b)  echo "$HF/Llama-3.1-8B-Instruct" ;;
    qwen14b)  echo "$HF/Qwen3-14B" ;;
    qwen30b)  echo "${QWEN30B_PATH:-$HF/Qwen3-30B-A3B-Instruct-2507-FP8}" ;;
  esac
}
model_pool() {
  case "$1" in
    llama8b)  echo "${POOL_LLAMA8B:-60000}" ;;
    qwen14b)  echo "${POOL_QWEN14B:-30000}" ;;
    qwen30b)  echo "${POOL_QWEN30B:-24000}" ;;
  esac
}
model_parser() {
  case "$1" in
    llama8b)  echo "llama3" ;;
    *)        echo "qwen25" ;;   # Qwen3 uses the qwen25 tool-call parser in SGLang
  esac
}
model_label() {
  case "$1" in
    llama8b) echo "Llama-3.1-8B" ;;
    qwen14b) echo "Qwen3-14B" ;;
    qwen30b) echo "Qwen3-30B" ;;
  esac
}

for mk in $MODELS; do
  MP=$(model_path "$mk")
  if [ ! -d "$MP" ]; then
    echo "=========================================================="
    echo " SKIP $mk: $MP not found."
    echo " Fetch it once, then re-run:"
    echo "   huggingface-cli download Qwen/Qwen3-14B --local-dir $HF/Qwen3-14B"
    echo "   huggingface-cli download Qwen/Qwen3-30B-A3B-Instruct-2507-FP8 \\"
    echo "     --local-dir $HF/Qwen3-30B-A3B-Instruct-2507-FP8"
    echo " A BF16 30B will NOT fit on one A6000 (61 GB weights vs 49 GB) -- use FP8 or AWQ."
    echo "=========================================================="
    continue
  fi
  export MODEL_PATH="$MP"
  export TOOL_CALL_PARSER=$(model_parser "$mk")
  export PREFILL_MAX_TOTAL_TOKENS=$(model_pool "$mk")
  OUT="$OUTBASE/$mk"

  echo "=========================================================="
  echo " MODEL $mk  ($(model_label "$mk"))"
  echo "   path=$MODEL_PATH  parser=$TOOL_CALL_PARSER  pool=$PREFILL_MAX_TOTAL_TOKENS"
  echo "=========================================================="

  OUTDIR="$OUT" TAG="$mk" ARMS="$ARMS" \
    CONCURRENCY=${CONCURRENCY:-8} MAX_TOKENS=${MAX_TOKENS:-1024} \
    ./scripts/sglang/run_exp1_sharegpt.sh
done

echo
echo "=========================================================="
SPECS=""
for mk in $MODELS; do
  [ -f "$OUTBASE/$mk/table.json" ] && SPECS="$SPECS $(model_label "$mk")=$OUTBASE/$mk/table.json"
done
if [ -n "$SPECS" ]; then
  python benchmark/plot_models.py $SPECS --out "$OUTBASE/fig_models"
else
  echo "no table.json produced -- nothing to plot"
fi
