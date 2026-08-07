#!/bin/bash
# OPEN-LOOP load sweep: throughput and TTFT vs offered session rate, per arm.
#
# WHY THIS RUN EXISTS
#   The closed-loop Exp1 run cannot produce a throughput claim, and that is a property of
#   its harness rather than of the arms. There, CONCURRENCY=8 workers each start a new
#   conversation only when their previous one finishes, so the CLIENT paces the server:
#   the same conversations, the same concurrency, the same wall time, and therefore the
#   same throughput no matter how fast the server is. Measured across the three arms it
#   is flat to within 1%:
#       Llama-3.1-8B   245.9 / 249.1 / 247.8 tok/s   (recompute / hicache / park)
#       Llama-2-13B    151.9 / 151.0 / 151.8
#       Qwen3-14B      165.5 / 165.8 / 165.7
#   Nothing about KV placement can move those numbers.
#
#   Here sessions arrive as a POISSON PROCESS at a fixed rate, whether or not the server
#   is keeping up, so the SERVER becomes the bottleneck. An arm that spends less GPU time
#   re-prefilling sustains a higher rate before its latency knee. That is the throughput
#   claim the paper wants: not "we are faster at fixed load" but "we serve more load at
#   the same latency".
#
# WHAT TO LOOK FOR
#   Delivered throughput rises with offered rate, then PLATEAUS -- that plateau is the
#   arm's capacity. Past it, TTFT climbs steeply and `sessions_unfinished_at_drain` goes
#   nonzero. The figure is the plateau height and the rate at which the knee occurs.
#   If no arm has plateaued by the top rate, the sweep did not reach saturation and the
#   throughput numbers are all still client-limited -- raise RATES and re-run.
#
# COST
#   arms x rates x (LOAD_DURATION + drain). At the defaults: 3 x 6 x ~(300 + 180)s
#   ~= 2.5 h plus 3 server restarts.
#
# 사용:
#   ./scripts/sglang/run_qps_sweep.sh
#   RATES="0.05 0.1 0.2 0.4" ARMS="recompute park" ./scripts/sglang/run_qps_sweep.sh
#   MODEL_KEY=qwen14b ./scripts/sglang/run_qps_sweep.sh
#   WORKLOAD=bfcl RATES="0.2 0.4 0.8 1.5 3" ./scripts/sglang/run_qps_sweep.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

MODEL_KEY=${MODEL_KEY:-llama8b}
# sharegpt (default) or bfcl. Both benchmark scripts already speak the same open-loop
# protocol (SESSION_RATE/ITEM_OFFSET in, {mode:open_loop, window_*, job_delay_*} out),
# so this sweep and collect_qps.py/plot_job_delay.py need no per-workload branch below
# this block -- only which script gets launched and which of its env knobs apply.
WORKLOAD=${WORKLOAD:-sharegpt}
# Reuse the sweep's per-model geometry rather than restating it, so this run cannot drift
# from the closed-loop numbers it is meant to sit beside.
eval "$(sed -n '/^model_path()/,/^}/p;/^model_pool()/,/^}/p;/^model_park_pool()/,/^}/p;/^model_memfrac()/,/^}/p;/^model_parser()/,/^}/p;/^model_max_ctx()/,/^}/p;/^model_kv_bytes()/,/^}/p;/^workload_for()/,/^}/p' scripts/sglang/run_models_sweep.sh)"

# The eval above lifts the FUNCTIONS out of run_models_sweep.sh but not the variables
# they close over, so these have to be restated here or model_path() expands "$HF/..."
# to "/Llama-3.1-8B-Instruct" and the server dies 40 frames deep in huggingface_hub
# trying to resolve that as a repo id.
HF=${HF_HOME_MODELS:-/home/uhmturks/hf_models}
UNIFORM_WORKLOAD=${UNIFORM_WORKLOAD:-0}

export MODEL_PATH=$(model_path "$MODEL_KEY")
if [ -z "$MODEL_PATH" ] || [ ! -f "$MODEL_PATH/config.json" ]; then
  echo "ERROR: no model at MODEL_PATH='$MODEL_PATH' (MODEL_KEY=$MODEL_KEY)"
  echo "       Expected a directory containing config.json."
  echo "       Point HF_HOME_MODELS at the model root, e.g."
  echo "         HF_HOME_MODELS=/home/uhmturks/hf_models ./scripts/sglang/run_qps_sweep.sh"
  exit 1
fi
export TOOL_CALL_PARSER=$(model_parser "$MODEL_KEY")
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-$(model_pool "$MODEL_KEY")}
export PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-$(model_park_pool "$MODEL_KEY")}
export PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-$(model_memfrac "$MODEL_KEY")}
read -r _MT _TURNS <<< "$(workload_for "$MODEL_KEY")"

export TIMEOUT=${TIMEOUT:-600}

case "$WORKLOAD" in
  sharegpt)
    BENCH=benchmark/sglang_sharegpt_multi_turn_concurrent.py
    export MAX_TOKENS=${MAX_TOKENS:-$_MT}
    export MAX_TURNS=${MAX_TURNS:-$_TURNS}
    export MIN_TURNS=${MIN_TURNS:-6}
    export MAX_PROMPT_CHARS=${MAX_PROMPT_CHARS:-16000}
    export TOOL_DELAY=${TOOL_DELAY:-3}
    export DATA_PATH=${DATA_PATH:-data/ShareGPT_V3_unfiltered_cleaned_split.json}
    if [ ! -f "$DATA_PATH" ]; then
      echo "ERROR: $DATA_PATH not found. Fetch it once:"
      echo "  cd data && wget https://huggingface.co/datasets/anon8231489123/ShareGPT_Vicuna_unfiltered/resolve/main/ShareGPT_V3_unfiltered_cleaned_split.json"
      exit 1
    fi
    ;;
  bfcl)
    # No DATA_PATH / MIN_TURNS / MAX_PROMPT_CHARS -- the BFCL runner loads its own fixed
    # data/BFCL_v3_multi_turn_base.json (200 conversations, ~4 turns each) and ignores
    # those knobs. TOOL_DELAY DEFAULTS TO 0 here for the same reason it does in
    # run_why_faster.sh: the harness replays recorded turns rather than executing tools,
    # so any nonzero delay is a number someone picked, and the 66% TTFT share that makes
    # BFCL a valid prefill-bound testbed was measured at 0.
    BENCH=benchmark/sglang_BFCL_multi_turn_concurrent.py
    export MAX_TOKENS=${MAX_TOKENS:-512}
    export TOOL_DELAY=${TOOL_DELAY:-0}
    ;;
  *) echo "unknown WORKLOAD: $WORKLOAD (want sharegpt | bfcl)"; exit 1 ;;
esac

# Offered SESSION rate (conversations/s). A session is however many turns the item has,
# with TOOL_DELAY between them, so the turn rate the server sees is roughly
# RATE x turns/session -- but only while it keeps up, which is what the sweep measures.
RATES=${RATES:-"0.05 0.1 0.2 0.35 0.5 0.75"}
ARMS=${ARMS:-"recompute hicache park"}
export LOAD_DURATION=${LOAD_DURATION:-300}
export WARMUP_S=${WARMUP_S:-60}
export DRAIN_TIMEOUT=${DRAIN_TIMEOUT:-300}

# One rate point must never replay a conversation another point already served: a repeat
# is a free cache hit for the two cached arms and no help to recompute. Each point gets a
# disjoint slice via ITEM_OFFSET, so the corpus must hold sum(rate)*duration sessions.
#
# DERIVED FROM RATES, not a constant: a hardcoded default disagrees with the default
# sweep the moment either changes, and the first version of this script shipped with
# exactly that bug (600 against a sweep needing 705). The preflight below stays as a
# guard for explicit overrides.
_need=0
for r in $RATES; do
  _need=$(python3 -c "print(int($_need + $r * $LOAD_DURATION + 20))")
done
export MAX_ITEMS=${MAX_ITEMS:-$_need}

TAG=${TAG:-"qps_${MODEL_KEY}_${WORKLOAD}"}
OUTDIR=${OUTDIR:-results/qps/$TAG}
LOG_DIR=${LOG_DIR:-logs/sglang}
mkdir -p "$OUTDIR"

# Preflight the corpus budget before spending hours discovering it mid-run.
#
# BFCL's corpus is a FIXED 200 conversations -- a sweep at realistic rates routinely
# needs more disjoint sessions than that exists (e.g. the default RATES already needs
# ~705). There the corpus WILL wrap and MAX_ITEMS cannot fix it, so this is a warning
# (each script's own sessions_repeated field marks and reports which points wrapped)
# rather than the hard stop ShareGPT gets, where a 90k-conversation corpus makes running
# out an actual misconfiguration worth failing loudly on.
if [ "$MAX_ITEMS" -lt "$_need" ]; then
  if [ "$WORKLOAD" = "bfcl" ]; then
    echo "[warn] MAX_ITEMS=$MAX_ITEMS < needed $_need, and BFCL only has 200 total --"
    echo "       every point past the corpus's own capacity will wrap and replay"
    echo "       conversations, biasing the cached arms. Check sessions_repeated in"
    echo "       each point's summary; qps.md flags it per-arm as WRAPPED."
  else
    echo "ERROR: MAX_ITEMS=$MAX_ITEMS but the sweep needs >= $_need disjoint conversations"
    echo "       (sum of rate x LOAD_DURATION over $(echo $RATES | wc -w) points, plus slack)."
    echo "       Re-run with MAX_ITEMS=$_need, or shorten RATES / LOAD_DURATION."
    exit 1
  fi
fi

echo "================================================================"
echo " OPEN-LOOP $WORKLOAD sweep   model=$MODEL_KEY  pool=$PREFILL_MAX_TOTAL_TOKENS"
echo " rates: $RATES  (sessions/s)   load=${LOAD_DURATION}s warmup=${WARMUP_S}s"
echo " max_tokens=$MAX_TOKENS tool_delay=${TOOL_DELAY}s  corpus_need=$MAX_ITEMS"
echo " arms: $ARMS  bench=$BENCH  -> $OUTDIR"
echo "================================================================"

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
    hicache_memfrac)
      # hicache at the PARK arm's --mem-fraction-static. Under load the two are not
      # otherwise comparable: park runs at 0.70 AND allocates its park buffer outside
      # that, so its SERVING KV pool is the smaller of the two. Pool size is what caps
      # resident sequences before queueing starts, so at high offered rates a difference
      # in TTFT confounds the placement mechanism with the memory budget. This arm holds
      # the budget equal and lets the mechanism answer on its own.
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=file \
        FORCE_MEM_FRACTION=${PARK_MEM_FRACTION} ./scripts/sglang/start_2P_2D.sh ;;
    park)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=${PARK_PEER:-1} \
        SGLANG_KV_PARK_BW_AWARE=1 \
        SGLANG_KV_PARK_HOST_OVERFLOW=${HOST_OVERFLOW:-1} \
        SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-8} \
        SGLANG_KV_PARK_REUSE_AWARE=1 ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

_sweep_t0=$SECONDS
for arm in $ARMS; do
  echo
  echo "════════════════ arm: $arm ── elapsed $(( (SECONDS-_sweep_t0)/60 ))m ════════════════"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$arm.log" 2>&1 || true
  start_arm "$arm"

  python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_$arm.csv" --interval 2 \
    > "$OUTDIR/sampler_mem_$arm.log" 2>&1 &
  MEM_PID=$!
  trap 'kill $MEM_PID 2>/dev/null || true' EXIT

  _off=0
  for rate in $RATES; do
    echo
    echo "──────── $arm @ ${rate} sessions/s  (offset=$_off) ────────"
    # Check the disk BEFORE the ten minutes, not after. The hicache file backend fills
    # it, and the failure lands on json.dump at the very end -- so the cost of finding
    # out late is the whole measurement, not a retry.
    _free_gb=$(df -BG --output=avail . | tail -1 | tr -dc '0-9')
    _hc_gb=$(du -sBG /tmp/hicache 2>/dev/null | cut -f1 | tr -dc '0-9')
    echo "  disk free: ${_free_gb}G   /tmp/hicache: ${_hc_gb:-0}G"
    if [ "${_free_gb:-0}" -lt "${NEED_FREE_GB:-10}" ]; then
      echo "  ERROR: only ${_free_gb}G free (need ${NEED_FREE_GB:-10}G). The hicache file"
      echo "         backend at /tmp/hicache is the usual culprit; stop.sh now clears it."
      echo "         Free space and re-run this arm:"
      echo "           ARMS=\"$arm\" RATES=\"$RATES\" ./scripts/sglang/run_qps_sweep.sh"
      exit 1
    fi
    _t0=$SECONDS
    CONFIG="qps_${TAG}_${arm}_r${rate}" \
      ROUTER_URL="http://127.0.0.1:8000/v1/chat/completions" \
      SESSION_RATE="$rate" ITEM_OFFSET="$_off" \
      stdbuf -oL -eL python -u "$BENCH" 2>&1 \
      | tee "$OUTDIR/bench_${arm}_r${rate}.log"
    echo "  [point done in $(( (SECONDS-_t0)/60 ))m $(( (SECONDS-_t0)%60 ))s]"

    if [ -f "results/qps_${TAG}_${arm}_r${rate}.json" ]; then
      cp "results/qps_${TAG}_${arm}_r${rate}.json" "$OUTDIR/bench_${arm}_r${rate}.json"
    elif [ -f "results/qps_${TAG}_${arm}_r${rate}.summary.json" ]; then
      # The full write hit ENOSPC but the summary survived; it carries every number the
      # collector reads, so the point is not lost -- only its per-turn detail.
      cp "results/qps_${TAG}_${arm}_r${rate}.summary.json" \
         "$OUTDIR/bench_${arm}_r${rate}.summary.json"
      echo "  [warn] only the summary survived for $arm r$rate (disk was full)"
    else
      echo "  [warn] results/qps_${TAG}_${arm}_r${rate}.json missing"
    fi

    # Advance past the conversations this point consumed, plus slack.
    _off=$(python3 -c "print(int($_off + $rate * $LOAD_DURATION + 20))")

    # Fold into the curve after EVERY point, not once at the end: a sweep that dies at
    # hour two must not lose the points that already completed.
    python benchmark/collect_qps.py --dir "$OUTDIR" --out "$OUTDIR/qps.json" \
      > /dev/null 2>&1 || true
  done

  kill $MEM_PID 2>/dev/null || true
  trap - EXIT
  cp "$LOG_DIR"/p1.log "$OUTDIR/p1_$arm.log" 2>/dev/null || true
done

./scripts/sglang/stop.sh > "$OUTDIR/stop_final.log" 2>&1 || true

echo
echo "================================================================"
python benchmark/collect_qps.py --dir "$OUTDIR" --out "$OUTDIR/qps.json" \
  | tee "$OUTDIR/qps.md"
python benchmark/plot_qps.py "$OUTDIR/qps.json" --out "$OUTDIR/fig_qps"
python benchmark/plot_job_delay.py "$OUTDIR/qps.json" --out "$OUTDIR/fig_job_delay"
python benchmark/plot_normalized_latency.py "$OUTDIR/qps.json" --out "$OUTDIR/fig_norm_latency"
echo "================================================================"
echo "READ THE PLATEAU, NOT THE TOP POINT. If delivered throughput is still rising at the"
echo "highest rate, no arm reached capacity and the sweep is still client-limited --"
echo "extend RATES upward and re-run. If sessions_unfinished_at_drain is large at the top"
echo "rate for every arm, that point is past saturation and its TTFT is a queue, not a"
echo "latency."
