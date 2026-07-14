#!/bin/bash
# Phase 2 slice-2 validation sweep: does park pool capacity drive survival -> TTFT?
# Sweeps PARK_POOL_TOKENS (radix+park, host-RAM-free) x reps + a hicache reference,
# capturing avg TTFT, throughput, park survival, and host RAM per run. Produces the
# survival->TTFT curve (the paper's core figure) via plot_park_sweep.py.
#
# ~10 runs x ~6min = ~1h. Run in background (nohup). OOM-safe: PARK_MEM_FRACTION=0.5
# leaves HBM for the largest (120k) park buffer; smaller pools have more headroom.
#
# 사용: nohup ./scripts/sglang/run_2p2d_park_sweep.sh > logs/park_sweep.out 2>&1 &
set -u
cd "$(dirname "$0")/../.."

POOLS=${POOLS:-"40000 60000 80000 120000"}
REPS=${REPS:-2}
PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.50}
export PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-24000}
export DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS:-24000}
BENCH=benchmark/sglang_BFCL_multi_turn_concurrent.py
CONC=${CONCURRENCY:-16}; DELAY=${TOOL_DELAY:-3}; ITEMS=${MAX_ITEMS:-200}
OUTDIR=${OUTDIR:-results/park_sweep}
mkdir -p "$OUTDIR" logs
SUMMARY="$OUTDIR/sweep_summary.csv"
echo "arm,pool,rep,ttft,tput,wall,host_mb,survival" > "$SUMMARY"

hard_stop() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "sglang_router"        2>/dev/null || true
  pkill -9 -f "mooncake"             2>/dev/null || true
  for p in 30000 30001 30002 30003 8000 8998 8999; do fuser -k ${p}/tcp 2>/dev/null || true; done
  sleep 12
}

_num() { python3 -c "import json,sys;print(json.load(open(sys.argv[1]))['summary'].get(sys.argv[2],''))" "$1" "$2" 2>/dev/null || echo ""; }

run_bench() {  # arm pool rep   (server already up)
  local arm=$1 pool=$2 rep=$3
  local host_mb; host_mb=$(free -m | awk '/^Mem:/{print $3}')
  local cfg="sweep_${arm}_p${pool}_r${rep}"
  : > logs/sglang/p1.log.mark 2>/dev/null || true
  env CONCURRENCY=$CONC TOOL_DELAY=$DELAY MAX_ITEMS=$ITEMS CONFIG="$cfg" \
      python "$BENCH" > "$OUTDIR/${cfg}.out" 2>&1 || true
  local rj="results/${cfg}.json"
  local ttft tput wall surv
  ttft=$(_num "$rj" avg_ttft_s); tput=$(_num "$rj" overall_throughput_tok_per_s); wall=$(_num "$rj" total_wall_time_s)
  surv=""
  [ "$arm" = "park" ] && surv=$(grep -ohE "survival=[0-9]+%" logs/sglang/p1.log logs/sglang/p2.log 2>/dev/null | tail -1 | grep -oE "[0-9]+")
  echo "$arm,$pool,$rep,$ttft,$tput,$wall,$host_mb,$surv" >> "$SUMMARY"
  echo ">> ${cfg}: TTFT=${ttft}s tput=${tput} host=${host_mb}MB survival=${surv}%"
}

echo "=== hicache reference (${REPS} reps) ==="
for rep in $(seq 1 $REPS); do
  hard_stop
  ./scripts/sglang/start_2P_2D.sh > "$OUTDIR/start_hicache_r${rep}.log" 2>&1 || { echo "hicache start failed r${rep}"; continue; }
  run_bench hicache 0 $rep
done

for pool in $POOLS; do
  echo "=== park pool=${pool} (${REPS} reps) ==="
  for rep in $(seq 1 $REPS); do
    hard_stop
    IDLE_KV_PARKING=1 PARK_NO_HICACHE=1 PARK_POOL_TOKENS=$pool PARK_MEM_FRACTION=$PARK_MEM_FRACTION \
      ./scripts/sglang/start_2P_2D.sh > "$OUTDIR/start_park_p${pool}_r${rep}.log" 2>&1 || { echo "park start failed p${pool} r${rep}"; continue; }
    run_bench park $pool $rep
  done
done
hard_stop

echo ""
echo "=== summary ($SUMMARY) ==="
cat "$SUMMARY"
python benchmark/plot_park_sweep.py --csv "$SUMMARY" || true
echo "done."
