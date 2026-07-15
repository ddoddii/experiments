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
BENCH=benchmark/sglang_perturn_ctxlen.py
CONC=${CONCURRENCY:-16}; DELAY=${TOOL_DELAY:-3}; ITEMS=${MAX_ITEMS:-200}
OUTDIR=${OUTDIR:-results/perturn}
mkdir -p "$OUTDIR" logs/sglang

hard_stop() {
  pkill -9 -f "sglang.launch_server" 2>/dev/null || true
  pkill -9 -f "sglang_router"        2>/dev/null || true
  pkill -9 -f "mooncake"             2>/dev/null || true
  for p in 30000 30001 30002 30003 8000 8998 8999; do fuser -k ${p}/tcp 2>/dev/null || true; done
  sleep 12
}

run_bench() {  # config
  local cfg=$1
  env CONCURRENCY=$CONC TOOL_DELAY=$DELAY MAX_ITEMS=$ITEMS CONFIG="$cfg" \
      python "$BENCH" > "$OUTDIR/${cfg}.out" 2>&1 || true
  echo ">> $cfg done -> results/${cfg}.json"
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
  run_bench "perturn_hicache_r${rep}"

  echo "=== rep ${rep}: park (session-keyed slab, host-RAM-free) ==="
  hard_stop
  IDLE_KV_PARKING=1 PARK_NO_HICACHE=1 \
    SGLANG_KV_PARK_SESSION_KEYED=1 SGLANG_KV_PARK_SLAB_TOKENS=$SLAB \
    PARK_POOL_TOKENS=$PARK_POOL_TOKENS PARK_MEM_FRACTION=$PARK_MEM_FRACTION \
    ./scripts/sglang/start_2P_2D.sh > "$OUTDIR/start_park_r${rep}.log" 2>&1 \
    || { echo "park start failed r${rep}"; continue; }
  run_bench "perturn_park_r${rep}"
done
hard_stop

echo ""
echo "=== plotting ==="
# merge reps by glob; plotter accepts multiple files per arm
python benchmark/plot_perturn_ctxlen.py \
  --park  results/perturn_park_r*.json \
  --hicache results/perturn_hicache_r*.json \
  --out "$OUTDIR/perturn_ctxlen.png" || true
echo "done."
