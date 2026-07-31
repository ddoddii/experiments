#!/bin/bash
# The cross-model figure: run the same arms on each model, then plot one row of panels
# with model on the x-axis.
#
# MODEL SIZE IS NOT A FREE PARAMETER HERE. 2P2D gives each of the four processes ONE
# A6000 (49 GB), so the weights must fit on a single card with room left for the KV pool
# and activations:
#
#   Llama-3.1-8B   BF16   16.0 GB  ->  33.0 GB free   fits comfortably
#   Llama-2-13B    BF16   26.0 GB  ->  23.0 GB free   fits, but see the two caveats below
#   Qwen3-14B      BF16   29.6 GB  ->  19.4 GB free   fits
#   Qwen3-30B      BF16   61.0 GB  ->  DOES NOT FIT   (needs TP=2, i.e. 8 GPUs for 2P2D)
#   Qwen3-30B      FP8    30.5 GB  ->  18.5 GB free   fits
#
# A 30B row must therefore be a quantised checkpoint. Running it TP=2 instead would use
# all four GPUs for 1P1D and leave NO peer prefill GPU to park onto -- which removes the
# mechanism under test, so that is not an option for this figure.
#
# TWO THINGS ABOUT LLAMA-2-13B ("Llama-13B") THAT CHANGE THE EXPERIMENT
#
#   1. It uses MHA, not GQA: 40 layers x 40 KV heads = 800 KiB per token, 6.25x the
#      Llama-3.1-8B's 128 KiB. The pools are configured in TOKENS, so a 60k-token pool
#      would be 49 GB on this model -- larger than the card. Pools are derived per model
#      below; they are not a shared constant.
#
#   2. It tops out at 4096 positions. The ShareGPT config the other models use reaches
#      ~11k tokens by the last turn, so the 13B would reject requests partway through
#      every conversation. Its workload is shortened automatically, which means ITS TTFT
#      IS NOT COMPARABLE ACROSS MODELS -- only the ratio between arms within its column
#      is. UNIFORM_WORKLOAD=1 caps every model to the 13B's shape instead, restoring
#      cross-model comparability at the cost of a smaller reuse prize for all of them.
#
# WHAT SHRINKING HEADROOM DOES TO THE RESULT
#   Free HBM per GPU falls 33 -> 23 -> 19 GB across the default three, so the idle GPU
#   HBM the proposal exploits shrinks with model size, and HiCache's host pool -- sized
#   as GPU-KV x hicache-ratio -- shrinks with it. Expect the host-DRAM saving to NARROW
#   on the larger models. That is a real property of the mechanism on a fixed GPU budget,
#   not a measurement problem, and the figure should show it rather than hide it.
#
# 사용:
#   ./scripts/sglang/run_models_sweep.sh                      # llama8b llama13b qwen14b
#   MODELS="qwen14b" ./scripts/sglang/run_models_sweep.sh     # one model
#   MODELS="llama8b llama13b qwen14b" UNIFORM_WORKLOAD=1 ./scripts/sglang/run_models_sweep.sh
#   ARMS="radix park" ./scripts/sglang/run_models_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

HF=${HF_HOME_MODELS:-/home/uhmturks/hf_models}
MODELS=${MODELS:-"llama8b llama13b qwen14b"}
ARMS=${ARMS:-"radix hicache park"}
OUTBASE=${OUTBASE:-results/models}
mkdir -p "$OUTBASE"

# ---------------------------------------------------------------- per-model geometry
#
# POOL SIZES ARE IN TOKENS, AND A TOKEN IS NOT THE SAME SIZE IN EVERY MODEL. Measured
# KV bytes per token (2 x layers x kv_heads x head_dim x 2 B):
#
#   Llama-3.1-8B   GQA  32L x  8kv    128 KiB/tok
#   Llama-2-13B    MHA  40L x 40kv    800 KiB/tok   <- 6.25x the 8B
#   Qwen3-14B      GQA  40L x  8kv    160 KiB/tok
#   Qwen3-30B-A3B  GQA  48L x  4kv     96 KiB/tok (FP8 weights, BF16 KV)
#
# So a 60k-token pool is 7.9 GB on the 8B and 49.2 GB on the 13B -- larger than the whole
# card. Holding the token count fixed across models would not be a fair comparison; it
# would simply fail to allocate. Both the serving pool and the PARK pool are therefore
# derived per model at a similar fraction of that model's free HBM (~40% and ~25%), so
# every model sees comparable eviction pressure.
model_path() {
  case "$1" in
    llama8b)  echo "$HF/Llama-3.1-8B-Instruct" ;;
    llama13b) echo "${LLAMA13B_PATH:-$HF/Llama-2-13b-chat-hf}" ;;
    qwen14b)  echo "$HF/Qwen3-14B" ;;
    qwen30b)  echo "${QWEN30B_PATH:-$HF/Qwen3-30B-A3B-Instruct-2507-FP8}" ;;
  esac
}
model_pool() {        # serving KV pool, tokens
  case "$1" in
    llama8b)  echo "${POOL_LLAMA8B:-60000}" ;;
    llama13b) echo "${POOL_LLAMA13B:-8000}" ;;
    qwen14b)  echo "${POOL_QWEN14B:-20000}" ;;
    qwen30b)  echo "${POOL_QWEN30B:-45000}" ;;
  esac
}
model_park_pool() {   # idle-GPU park buffer, tokens
  case "$1" in
    llama8b)  echo "${PARKPOOL_LLAMA8B:-30000}" ;;
    llama13b) echo "${PARKPOOL_LLAMA13B:-4000}" ;;
    qwen14b)  echo "${PARKPOOL_QWEN14B:-12000}" ;;
    qwen30b)  echo "${PARKPOOL_QWEN30B:-28000}" ;;
  esac
}
model_memfrac() {     # --mem-fraction-static for the PARK arm
  # This governs weights + serving KV pool. The park buffer is allocated OUTSIDE it, so
  # the fraction has to leave room on both sides. At the 0.70 default that works for the
  # 8B, the bigger models do not fit at all: Qwen3-14B needs 29.6 GB of weights plus a
  # 4.9 GB pool = 34.5 GB against a 34.3 GB budget, and Llama-2-13B lands at 35.0 GB.
  # Both would fail to allocate rather than run slowly. Raised per model, with the pools
  # trimmed to match.
  case "$1" in
    llama8b)  echo "${MEMFRAC_LLAMA8B:-0.70}" ;;
    llama13b) echo "${MEMFRAC_LLAMA13B:-0.80}" ;;
    qwen14b)  echo "${MEMFRAC_QWEN14B:-0.80}" ;;
    qwen30b)  echo "${MEMFRAC_QWEN30B:-0.80}" ;;
  esac
}
model_parser() {
  case "$1" in
    llama8b|llama13b) echo "llama3" ;;
    *)                echo "qwen25" ;;
  esac
}
model_label() {
  case "$1" in
    llama8b)  echo "Llama-3.1-8B" ;;
    llama13b) echo "Llama-2-13B" ;;
    qwen14b)  echo "Qwen3-14B" ;;
    qwen30b)  echo "Qwen3-30B" ;;
  esac
}
model_max_ctx() {     # the model's own position-embedding limit
  case "$1" in
    llama8b)  echo 131072 ;;
    llama13b) echo 4096 ;;
    qwen14b)  echo 32768 ;;
    qwen30b)  echo 262144 ;;
  esac
}

# Llama-2-13B tops out at 4096 positions. The ShareGPT config used for the other models
# (MAX_TOKENS=1024, MAX_TURNS=10) reaches ~11k tokens by the last turn, so that model
# would start rejecting requests partway through every conversation. CTX_CAP shrinks the
# workload for it. The consequence is stated rather than hidden: the 13B column is then
# measured on a SHORTER conversation than the others, and its TTFT is not directly
# comparable to theirs -- only the ratio between arms within that column is.
# Set UNIFORM_WORKLOAD=1 to instead cap every model to the 13B's shape, which restores
# cross-model comparability at the cost of a smaller reuse prize for all of them.
UNIFORM_WORKLOAD=${UNIFORM_WORKLOAD:-0}
workload_for() {
  local mk=$1
  if [ "$UNIFORM_WORKLOAD" = "1" ] || [ "$(model_max_ctx "$mk")" -le 8192 ]; then
    echo "512 4"        # MAX_TOKENS MAX_TURNS -> ~3k tokens by the last turn
  else
    echo "1024 10"
  fi
}

for mk in $MODELS; do
  MP=$(model_path "$mk")
  if [ ! -d "$MP" ]; then
    echo "=========================================================="
    echo " SKIP $mk: $MP not found."
    echo " Fetch it once, then re-run:"
    echo "   huggingface-cli download meta-llama/Llama-2-13b-chat-hf \\"
    echo "     --local-dir $HF/Llama-2-13b-chat-hf"
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
  export PARK_POOL_TOKENS=$(model_park_pool "$mk")
  export PARK_MEM_FRACTION=$(model_memfrac "$mk")
  read -r _MT _TURNS <<< "$(workload_for "$mk")"
  OUT="$OUTBASE/$mk"

  echo "=========================================================="
  echo " MODEL $mk  ($(model_label "$mk"))"
  echo "   path=$MODEL_PATH  parser=$TOOL_CALL_PARSER"
  echo "   serving pool=$PREFILL_MAX_TOTAL_TOKENS tok  park pool=$PARK_POOL_TOKENS tok"
  echo "   park arm --mem-fraction-static=$PARK_MEM_FRACTION"
  echo "   workload: MAX_TOKENS=$_MT MAX_TURNS=$_TURNS  (model max ctx $(model_max_ctx "$mk"))"
  if [ "$(model_max_ctx "$mk")" -le 8192 ]; then
    echo "   NOTE: short-context model -- this column is measured on a SHORTER"
    echo "         conversation than the others; compare arms WITHIN it, not across."
  fi
  echo "=========================================================="

  OUTDIR="$OUT" TAG="$mk" ARMS="$ARMS" \
    CONCURRENCY=${CONCURRENCY:-8} MAX_TOKENS="$_MT" MAX_TURNS="$_TURNS" \
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
