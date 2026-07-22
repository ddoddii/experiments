#!/bin/bash
# Per-turn TTFT / throughput vs growing context length: park (session-keyed slab,
# host-RAM-free) vs hicache. Runs the two arms back-to-back on 2P2D and captures
# per-turn (ctx-len, ttft, tput) points, then draws the two-panel figure.
#
# Final park config = the finalized slab arm: PARK_POOL_TOKENS=96000, SLAB=6000
# (16 slabs), session-keyed, host-RAM-free (no hicache). Override via env.
#
# 사용: nohup ./scripts/sglang/run_perturn_compare.sh > logs/perturn.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."

# NEVER run this whole script under `sudo` -- start_2P_2D.sh does `conda activate
# sglang`, and sudo resets PATH/HOME and skips your shell rc, so conda either
# isn't found or resolves to a different install without the sglang env -> every
# server start fails ("hicache start failed r1", etc.), which is exactly what
# happens if you type `sudo WORKLOAD=... ./run_perturn_compare.sh`. Cache-dropping
# only needs privilege for one line inside hard_stop(), handled separately below
# via a scoped `sudo -n` -- run this script itself as your normal user.
if [ "${EUID:-$(id -u)}" = "0" ]; then
  echo "ERROR: do not run this script with sudo (breaks conda activation in" >&2
  echo "start_2P_2D.sh -> every server start will fail). Instead:" >&2
  echo "  1) sudo -v                          # cache sudo credentials (one password prompt)" >&2
  echo "  2) WORKLOAD=... ./run_perturn_compare.sh   # run normally; hard_stop() will use" >&2
  echo "     'sudo -n' automatically for just the cache-drop line while your credentials" >&2
  echo "     are cached. Or set DROP_CACHES=0 and skip cache-dropping entirely." >&2
  exit 1
fi

# Park pool is a raw torch.zeros buffer on EACH P's own GPU (P0->gpu0, P1->gpu1),
# competing with that P's weights. At 128KB/token it is ~7.7GB @ 60k, ~12.3GB @ 96k.
# So mem-fraction-static must be high enough that (weights + park + KV) fits the budget,
# or SGLang profiles negative KV. 60k @ 0.70 leaves ample KV; bump both to go larger.
REPS=${REPS:-1}
PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-60000}          # 10 slabs @ SLAB=6000 (best slab TTFT)
SLAB=${SGLANG_KV_PARK_SLAB_TOKENS:-6000}
PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70}          # was 0.50 -> negative KV with the park buffer
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-24000}
export DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS:-24000}
# workload switch: bfcl (tool-call multi-turn) | sharegpt (plain long-form multi-turn).
# each gets its own benchmark, CONFIG tag, and OUTDIR so results/figures don't collide.
WORKLOAD=${WORKLOAD:-bfcl}
case "$WORKLOAD" in
  sharegpt) BENCH=benchmark/sglang_perturn_sharegpt.py; TAG=perturn_sharegpt ;;
  bfcl)     BENCH=benchmark/sglang_perturn_ctxlen.py;   TAG=perturn ;;
  *) echo "unknown WORKLOAD=$WORKLOAD (use bfcl|sharegpt)"; exit 1 ;;
esac
CONC=${CONCURRENCY:-16}; DELAY=${TOOL_DELAY:-3}; ITEMS=${MAX_ITEMS:-200}
OUTDIR=${OUTDIR:-results/${TAG}}
mkdir -p "$OUTDIR" logs/sglang
echo "=== workload=$WORKLOAD  bench=$BENCH  tag=$TAG  outdir=$OUTDIR ==="

hard_stop() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "sglang_router"        2>/dev/null || true
  pkill -9 -f "mooncake"             2>/dev/null || true
  for p in 30000 30001 30002 30003 8000 8998 8999; do fuser -k ${p}/tcp 2>/dev/null || true; done
  sleep 12
  # Host page cache is OS-level and outlives our processes -- Linux keeps clean
  # file-backed pages around indefinitely under low memory pressure, so a prior
  # run's hicache file-tier cache can still be resident (seen: 92GB leftover from
  # a run 5 days earlier, inflating the next run's "starting" host RAM). Drop it
  # so every arm's host-RAM timeline starts from a comparable clean state.
  # DROP_CACHES=0 to disable. Requires root or passwordless sudo for this one
  # command -- `test -w` avoids the bash redirection error a plain non-root
  # `echo 3 > drop_caches` would print (redirect setup fails before the
  # command's own 2>/dev/null can apply).
  if [ "${DROP_CACHES:-1}" = "1" ]; then
    sync
    if [ -w /proc/sys/vm/drop_caches ]; then
      echo 3 > /proc/sys/vm/drop_caches && echo "(dropped page cache)"
    elif sudo -n true 2>/dev/null; then
      sudo -n sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches' \
        && echo "(dropped page cache via sudo)"
    else
      echo "(note: no permission to drop page cache -- host-RAM ABSOLUTE numbers may" \
           "carry leftover cache from earlier runs; the within-run hicache-vs-park" \
           "DELTA is still valid, since both arms of one run start seconds/minutes" \
           "apart under the same system state. To enable: run once as root/sudo, or" \
           "grant passwordless sudo for 'echo 3 > /proc/sys/vm/drop_caches', or set" \
           "DROP_CACHES=0 to silence this message.)"
    fi
  fi
}

run_bench() {  # config mem_csv
  local cfg=$1 mem_csv=$2
  # sample GPU HBM + host DRAM for the duration of THIS arm's benchmark
  python benchmark/sys_mem_sampler.py --out "$mem_csv" --interval 1 &
  local samp=$!
  env CONCURRENCY=$CONC TOOL_DELAY=$DELAY MAX_ITEMS=$ITEMS CONFIG="$cfg" \
      python "$BENCH" > "$OUTDIR/${cfg}.out" 2>&1 || true
  kill "$samp" 2>/dev/null || true
  echo ">> $cfg done -> results/${cfg}.json  (mem -> $mem_csv)"
}

for rep in $(seq 1 $REPS); do
  echo "=== rep ${rep}: hicache (parking explicitly OFF) ==="
  hard_stop
  # Explicitly force parking OFF and strip any leaked exports so this arm is a clean
  # hicache-only baseline (guards against IDLE_KV_PARKING lingering in the shell).
  env -u PARK_NO_HICACHE -u PARK_POOL_TOKENS -u PARK_MEM_FRACTION \
      -u SGLANG_KV_PARK_SESSION_KEYED -u SGLANG_KV_PARK_SLAB_TOKENS \
      IDLE_KV_PARKING=0 \
      ./scripts/sglang/start_2P_2D.sh > "$OUTDIR/start_hicache_r${rep}.log" 2>&1 \
    || { echo "hicache start failed r${rep}"; continue; }
  run_bench "${TAG}_hicache_r${rep}" "$OUTDIR/mem_hicache_r${rep}.csv"

  echo "=== rep ${rep}: park (session-keyed slab, host-RAM-free) ==="
  hard_stop
  IDLE_KV_PARKING=1 PARK_NO_HICACHE=1 \
    SGLANG_KV_PARK_SESSION_KEYED=1 SGLANG_KV_PARK_SLAB_TOKENS=$SLAB \
    PARK_POOL_TOKENS=$PARK_POOL_TOKENS PARK_MEM_FRACTION=$PARK_MEM_FRACTION \
    ./scripts/sglang/start_2P_2D.sh > "$OUTDIR/start_park_r${rep}.log" 2>&1 \
    || { echo "park start failed r${rep}"; continue; }
  run_bench "${TAG}_park_r${rep}" "$OUTDIR/mem_park_r${rep}.csv"

  # coexistence (KV victim caching): hicache host tier + GPU victim cache LAYERED.
  # Same as the park arm but WITHOUT PARK_NO_HICACHE, so both tiers are active. The
  # victim absorbs re-reference hits on GPU; measure whether that reduces host fetch /
  # page cache vs hicache-alone. Set RUN_BOTH=0 to skip.
  if [ "${RUN_BOTH:-1}" = "1" ]; then
    echo "=== rep ${rep}: hicache + GPU victim (layered) ==="
    hard_stop
    IDLE_KV_PARKING=1 \
      SGLANG_KV_PARK_SESSION_KEYED=1 SGLANG_KV_PARK_SLAB_TOKENS=$SLAB \
      PARK_POOL_TOKENS=$PARK_POOL_TOKENS PARK_MEM_FRACTION=$PARK_MEM_FRACTION \
      ./scripts/sglang/start_2P_2D.sh > "$OUTDIR/start_both_r${rep}.log" 2>&1 \
      || { echo "both start failed r${rep}"; continue; }
    run_bench "${TAG}_both_r${rep}" "$OUTDIR/mem_both_r${rep}.csv"
  fi
done
hard_stop

echo ""
echo "=== plotting ==="
# only include the coexistence arm when it was requested THIS run (RUN_BOTH=1) --
# otherwise stale perturn_both_*.json from a prior run would sneak back into the plot.
BOTH_ARG_J=""; BOTH_ARG_C=""
if [ "${RUN_BOTH:-1}" = "1" ]; then
  BOTH_JSON=$(ls results/${TAG}_both_r*.json 2>/dev/null) && BOTH_ARG_J="--both $BOTH_JSON" || true
  BOTH_CSV=$(ls "$OUTDIR"/mem_both_r*.csv 2>/dev/null) && BOTH_ARG_C="--both $BOTH_CSV" || true
fi
python benchmark/plot_perturn_ctxlen.py \
  --park  results/${TAG}_park_r*.json \
  --hicache results/${TAG}_hicache_r*.json $BOTH_ARG_J \
  --out "$OUTDIR/fig_perturn.png" || true
python benchmark/plot_mem_timeline.py \
  --park  "$OUTDIR"/mem_park_r*.csv \
  --hicache "$OUTDIR"/mem_hicache_r*.csv $BOTH_ARG_C \
  --out "$OUTDIR/fig_mem.png" || true
echo "done."
