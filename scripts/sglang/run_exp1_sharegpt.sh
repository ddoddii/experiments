#!/bin/bash
# Exp 1 on ShareGPT -- the same three arms at a context length where KV reuse can pay.
#
# WHY THIS RUN EXISTS
#   On BFCL the median context is 408 tokens, so a FULL re-prefill costs ~66 ms. That is
#   the ceiling on what any amount of KV reuse can save per turn, and both non-radix arms
#   carry a larger constant overhead than that (park +562 ms on turn 0, where nothing can
#   be reused or fetched). The mechanism cannot show a TTFT benefit there no matter how
#   well it works -- and it did work: park had the highest reuse ratio (0.497 vs 0.446)
#   and the fewest recomputed tokens.
#
#   ShareGPT AT ITS DEFAULT SETTINGS DOES NOT FIX THIS. The six previous ShareGPT runs
#   have a median prompt of 840 tokens -> a ~135 ms prize, still well under the overhead.
#   Context grows ~500 tokens/turn there because responses hit MAX_TOKENS=512.
#
#   So this script raises the knobs that actually drive context length:
#     MAX_TOKENS 512 -> 1024        each turn appends ~2x more verbatim text
#     MAX_TURNS   6  -> 10          more turns to accumulate
#     MIN_TURNS   3  -> 6           only conversations long enough to get there
#     MAX_PROMPT_CHARS 6000 -> 16000  stop discarding long opening prompts
#   Expected: ~1.1k tokens/turn, so turn 5 is ~6k (prize ~900 ms) and turn 9 is ~11k
#   (prize ~1.6 s). Past turn ~4 the prize exceeds the overhead and the comparison
#   becomes meaningful. Verify with ttft_by_turn.py rather than assuming it.
#
#   CONCURRENCY drops to 8: at 11k tokens, 16 concurrent conversations need ~176k tokens
#   against a 60k pool, which saturates the prefill rather than merely evicting from it.
#   8 x 11k = 88k against 60k still forces real eviction, which is the point.
#
# 사용:
#   ./scripts/sglang/run_exp1_sharegpt.sh
#   MAX_TOKENS=2048 MAX_TURNS=12 ./scripts/sglang/run_exp1_sharegpt.sh   # longer still
#   ARMS="radix park" ./scripts/sglang/run_exp1_sharegpt.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-60000}
export CONCURRENCY=${CONCURRENCY:-8}
export TOOL_DELAY=${TOOL_DELAY:-3}
export MAX_TOKENS=${MAX_TOKENS:-1024}
export MAX_TURNS=${MAX_TURNS:-10}
export MIN_TURNS=${MIN_TURNS:-6}
export MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-16000}
export MAX_ITEMS=${MAX_ITEMS:-200}
export TIMEOUT=${TIMEOUT:-600}
export DATA_PATH=${DATA_PATH:-data/ShareGPT_V3_unfiltered_cleaned_split.json}

ARMS=${ARMS:-"recompute hicache park"}
TAG=${TAG:-"sharegpt_p${PREFILL_MAX_TOTAL_TOKENS}_c${CONCURRENCY}_m${MAX_TOKENS}"}
OUTDIR=${OUTDIR:-results/exp1/$TAG}
LOG_DIR=${LOG_DIR:-logs/sglang}
mkdir -p "$OUTDIR"

if [ ! -f "$DATA_PATH" ]; then
  echo "ERROR: $DATA_PATH not found. Fetch it once:"
  echo "  cd data && wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
  exit 1
fi

echo "================================================================"
echo " EXP1 ShareGPT   pool=$PREFILL_MAX_TOTAL_TOKENS C=$CONCURRENCY delay=${TOOL_DELAY}s"
echo " max_tokens=$MAX_TOKENS turns=$MIN_TURNS..$MAX_TURNS  -> $OUTDIR"
echo " arms: $ARMS"
echo "================================================================"

start_arm() {
  case "$1" in
    recompute)
      # No prefix reuse at all: --disable-radix-cache, which also forces hicache off
      # since hicache layers on top of the radix tree. Every turn re-prefills its whole
      # context. This is the "RE" baseline of the reference figures.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 DISABLE_RADIX_CACHE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    radix)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 ./scripts/sglang/start_2P_2D.sh ;;
    radix_memfrac)
      # radix at the park arm's memory budget -- attributes the constant overhead
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 \
        FORCE_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70} ./scripts/sglang/start_2P_2D.sh ;;
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

_n_arms=$(echo $ARMS | wc -w); _i_arm=0
_sweep_t0=$SECONDS
for arm in $ARMS; do
  _i_arm=$((_i_arm + 1))
  echo
  echo "──────────────── arm $_i_arm/$_n_arms: $arm ── elapsed $((  (SECONDS-_sweep_t0)/60 ))m ────────────────"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
  start_arm "$arm"

  python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_$arm.csv" --interval 1 \
    > "$OUTDIR/sampler_mem_$arm.log" 2>&1 &
  MEM_PID=$!
  python benchmark/park_location_sampler.py --out "$OUTDIR/parked_$arm.csv" --interval 1 \
    > "$OUTDIR/sampler_park_$arm.log" 2>&1 &
  PARK_PID=$!
  trap 'kill $MEM_PID $PARK_PID 2>/dev/null || true' EXIT

  python benchmark/phase0_metrics_scraper.py --tag "${arm}_before" \
    --out "$OUTDIR/metrics_${arm}_before.json" > /dev/null 2>&1 || true

  # NOTE: the ShareGPT runner takes ROUTER_URL (not SGLANG_URL) and writes to
  # results/<CONFIG>.json (not the model-slug tree the BFCL runner uses).
  # Stream the benchmark live. It used to end in `| tail -20`, which holds every line
  # until the process exits -- so a 45-minute arm showed nothing at all until it was
  # done, and a hung run was indistinguishable from a slow one. `python -u` and
  # `stdbuf -oL` are both needed: without them Python block-buffers stdout as soon as it
  # is a pipe rather than a terminal, so even without tail the output would arrive in
  # 4 KB bursts.
  _arm_t0=$SECONDS
  CONFIG="exp1_${TAG}_${arm}" ROUTER_URL="http://127.0.0.1:8000/v1/chat/completions" \
    stdbuf -oL -eL python -u benchmark/sglang_sharegpt_multi_turn_concurrent.py 2>&1 \
    | tee "$OUTDIR/bench_$arm.log"
  echo "  [arm $arm finished in $(( (SECONDS - _arm_t0) / 60 ))m $(( (SECONDS - _arm_t0) % 60 ))s]"

  python benchmark/phase0_metrics_scraper.py --tag "${arm}_after" \
    --out "$OUTDIR/metrics_${arm}_after.json" > /dev/null 2>&1 || true
  python benchmark/phase0_metrics_scraper.py --delta \
    "$OUTDIR/metrics_${arm}_before.json" "$OUTDIR/metrics_${arm}_after.json" \
    --out "$OUTDIR/metrics_${arm}_delta.json" > /dev/null 2>&1 || true

  kill $MEM_PID $PARK_PID 2>/dev/null || true
  trap - EXIT
  sleep 2

  if [ -f "results/exp1_${TAG}_${arm}.json" ]; then
    cp "results/exp1_${TAG}_${arm}.json" "$OUTDIR/bench_$arm.json"
    echo "  bench -> $OUTDIR/bench_$arm.json"
  else
    echo "  [warn] results/exp1_${TAG}_${arm}.json missing; table row will be partial"
  fi
  cp "$LOG_DIR"/p1.log "$OUTDIR/p1_$arm.log" 2>/dev/null || true
done

./scripts/sglang/stop.sh > "$OUTDIR/stop_final.log" 2>&1 || true

echo
echo "================================================================"
python benchmark/collect_arm_metrics.py --dir "$OUTDIR" --arms $ARMS \
  --out "$OUTDIR/table.json" | tee "$OUTDIR/table.md"
echo
echo "---- TTFT by turn (this is the one that separates mechanism from overhead) ----"
python benchmark/ttft_by_turn.py --dir "$OUTDIR" --arms $ARMS \
  | tee "$OUTDIR/ttft_by_turn.txt"
echo "================================================================"
echo "Check FIRST that the context actually grew: if the median prompt is still under"
echo "~4k tokens the prize is still smaller than the overhead and the TTFT comparison"
echo "says nothing about placement. Raise MAX_TOKENS / MAX_TURNS and re-run."
