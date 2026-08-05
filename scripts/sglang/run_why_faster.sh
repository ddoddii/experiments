#!/bin/bash
# WHY is parking onto an idle decode GPU faster? Not "is it" -- "why".
#
# The head-to-head arms cannot answer this. Comparing our GPU parking against hicache
# moves THREE things at once, and the resulting win says nothing about which one paid:
#
#     placement policy   (what to keep, what to give up, when)
#     transfer software  (how the bytes actually move)
#     storage medium     (peer HBM over NVLink vs host DRAM over PCIe)
#
# So this runner adds one arm that moves exactly ONE of them. park_host is our code with
# SGLANG_KV_PARK_FORCE_HOST=1: same index, same reuse-value eviction, same fetch path,
# same park pool still allocated on the decode GPUs -- and every park lands in pinned host
# DRAM instead. That turns a 3-variable comparison into a chain of 1-variable ones:
#
#   radix      -> park_host    policy + software, medium held at DRAM
#   park_host  -> park_gpu     medium ONLY (this is "is it the link?")
#   hicache    -> park_host    same medium, both DRAM over the same PCIe bus, so any
#                              difference here CANNOT be bandwidth -- it is software
#   radix      -> park_gpu     the total, for the headline
#
# and CAP=<list> repeats park_gpu at several park-pool sizes, which is the capacity axis:
# if the gain were really "more cache", TTFT must track pool size. If it is flat, capacity
# is not the mechanism.
#
# WHAT THE PRIOR DATA ALREADY PREDICTS -- worth writing down before running, so the run
# can falsify it rather than be read to agree with it. results/nvlink_microbench.json
# measures peer-GPU 52.7 GB/s against host round-trip 26.3 GB/s each way. At 128 KiB per
# token that is 0.0025 ms/token versus 0.0050 ms/token, while re-prefill costs 0.132
# ms/token. So the medium can be worth at most ~2% of what a hit saves, and the
# park_gpu - park_host gap SHOULD BE NEARLY ZERO. If it is large, something other than
# bandwidth is different between the two and must be found before any of this is claimed.
#
# 사용:
#   ./scripts/sglang/run_why_faster.sh
#   ARMS="radix hicache park_host park_gpu" ./scripts/sglang/run_why_faster.sh
#   CAP="8000 16000 32000" ./scripts/sglang/run_why_faster.sh     # capacity axis
#   COSTMODEL=1 ./scripts/sglang/run_why_faster.sh                # + per-fetch trace run
set -e

cd "$(dirname "$0")/../.."
ROOT=$(pwd)

export PD_LAYOUT=${PD_LAYOUT:-b}
export CONCURRENCY=${CONCURRENCY:-32}
export TOOL_DELAY=${TOOL_DELAY:-0}
export HICACHE_RATIO=${HICACHE_RATIO:-1.2}
export PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70}
export PARK_MEM_FRACTION_D=${PARK_MEM_FRACTION_D:-0.88}
export PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-32000}

# Park onto the DECODE GPUs. Under PD_LAYOUT=b, P0=gpu0 D0=gpu1 P1=gpu2 D1=gpu3, so each
# prefill lists its NVLink-bridged decode first and the far decode as a fallback. This is
# the configuration the paper claims: the victim cache lives in decode's idle HBM.
PARK_GPUS_P0=${PARK_GPUS_P0:-0,1,3}
PARK_GPUS_P1=${PARK_GPUS_P1:-2,3,1}

ARMS=${ARMS:-"radix hicache park_host park_gpu"}
CAP=${CAP:-}
COSTMODEL=${COSTMODEL:-0}
BENCH=${BENCH:-benchmark/sglang_sharegpt_multi_turn_concurrent.py}

TAG=${TAG:-why_c${CONCURRENCY}_p${PARK_POOL_TOKENS_PER_GPU}}
OUTDIR=${OUTDIR:-results/why/${TAG}}
mkdir -p "$OUTDIR"

echo "================================================================"
echo " WHY IS IT FASTER   C=$CONCURRENCY  layout=$PD_LAYOUT"
echo " park pool : $PARK_POOL_TOKENS_PER_GPU tok on EACH of P0[$PARK_GPUS_P0] P1[$PARK_GPUS_P1]"
echo " arms      : $ARMS"
[ -n "$CAP" ] && echo " capacity  : $CAP"
echo " -> $OUTDIR"
echo "================================================================"

start_arm() {
  case "$1" in
    radix)
      # No offload tier of any kind: the recompute floor everything else is measured from.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 \
        ./scripts/sglang/start_2P_2D.sh ;;
    hicache)
      IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
        HICACHE_STORAGE_BACKEND=file ./scripts/sglang/start_2P_2D.sh ;;
    park_host)
      # THE CONTROL. Identical to park_gpu except SGLANG_KV_PARK_FORCE_HOST=1. The GPU
      # park pools are still allocated and still cost the same HBM -- they are simply
      # never written to -- so a difference between these two arms cannot be blamed on
      # decode having less memory in one of them.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=$PARK_GPUS_P0 PARK_GPUS_P1=$PARK_GPUS_P1 \
        SGLANG_KV_PARK_FORCE_HOST=1 SGLANG_KV_PARK_HOST_OVERFLOW=1 \
        SGLANG_KV_PARK_HOST_MAX_GB=${HOST_MAX_GB:-32} \
        SGLANG_KV_PARK_PRESSURE_AWARE=1 SGLANG_KV_PARK_BW_AWARE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    park_gpu)
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=$PARK_GPUS_P0 PARK_GPUS_P1=$PARK_GPUS_P1 \
        SGLANG_KV_PARK_PRESSURE_AWARE=1 SGLANG_KV_PARK_BW_AWARE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    *) echo "unknown arm: $1"; exit 1 ;;
  esac
}

run_one() {   # $1 = arm label (may carry a _capNNN suffix), $2 = arm kind
  local label=$1 kind=$2
  echo
  echo "──────────────── $label ────────────────"
  ./scripts/sglang/stop.sh > "$OUTDIR/stop_$label.log" 2>&1 || true
  sleep 3
  start_arm "$kind"

  python benchmark/kv_occupancy_timeseries.py --out "$OUTDIR/occ_$label.csv" \
    --interval 0.5 > "$OUTDIR/sampler_occ_$label.log" 2>&1 &
  local OCC_PID=$!
  python benchmark/sys_mem_breakdown.py --out "$OUTDIR/mem_$label.csv" --interval 1 \
    > "$OUTDIR/sampler_mem_$label.log" 2>&1 &
  local MEM_PID=$!
  trap 'kill '"$OCC_PID $MEM_PID"' 2>/dev/null || true' EXIT

  CONFIG="why_$label" python "$BENCH" 2>&1 \
    | tee "$OUTDIR/bench_$label.log" | tail -15

  kill $OCC_PID $MEM_PID 2>/dev/null || true
  trap - EXIT
  sleep 2
  cp "results/why_$label.json" "$OUTDIR/bench_$label.json" 2>/dev/null \
    || echo "  [warn] results/why_$label.json missing"
  # The park telemetry files carry fetch_ms_tier / fetch_tok_tier, which is where the
  # per-tier cost comes from. They live in /dev/shm and are gone once the server exits,
  # so they are copied BEFORE the next stop.sh.
  cp /dev/shm/sglang_kv_parking/parked_bytes_*.json "$OUTDIR/" 2>/dev/null || true
  for f in "$OUTDIR"/parked_bytes_*.json; do
    [ -e "$f" ] || continue
    mv "$f" "${f%.json}.$label.json" 2>/dev/null || true
  done
  # Digest, not the raw log: .gitignore has a blanket '*.log' and every raw log copied
  # here in earlier runners was silently dropped before it could be pushed.
  for f in p1 p2 d1 d2; do
    [ -f "logs/$f.log" ] || continue
    { grep -iE "park pool|clamp|ready to roll|max_total_num_tokens" "logs/$f.log" | head -20
      echo "---- errors ----"
      grep -iE "error|traceback|out of memory|SIGQUIT received" "logs/$f.log" | head -20
      echo "---- tail ----"; tail -15 "logs/$f.log"
    } > "$OUTDIR/$f.$label.digest.txt" 2>/dev/null || true
  done
}

for arm in $ARMS; do
  run_one "$arm" "$arm"
done

# Capacity axis: same arm, same code, same medium -- only the pool size moves.
for cap in $CAP; do
  PARK_POOL_TOKENS_PER_GPU=$cap run_one "park_gpu_cap${cap}" park_gpu
done

# Cost-model run: SYNC_FETCH=1 so the recorded ms is the transfer, not the enqueue, and
# a per-fetch trace so ms can be REGRESSED on bytes per tier. Kept separate from the
# perf arms on purpose -- synchronizing the fetch changes the very TTFT being measured,
# so these numbers are for the cost model only and must never be quoted as throughput.
if [ "$COSTMODEL" = "1" ]; then
  for kind in park_gpu park_host; do
    echo
    echo "──────────────── costmodel: $kind (SYNC_FETCH=1) ────────────────"
    ./scripts/sglang/stop.sh > /dev/null 2>&1 || true
    sleep 3
    rm -rf "$OUTDIR/trace_$kind"
    SGLANG_KV_PARK_SYNC_FETCH=1 \
      SGLANG_KV_PARK_FETCH_TRACE="$ROOT/$OUTDIR/trace_$kind" \
      start_arm "$kind"
    CONFIG="why_costmodel_$kind" python "$BENCH" 2>&1 | tail -8
    sleep 2
  done
fi

./scripts/sglang/stop.sh > /dev/null 2>&1 || true

echo
echo "############################################################"
python benchmark/decompose_speedup.py --dir "$OUTDIR" || true
if [ "$COSTMODEL" = "1" ]; then
  python benchmark/fit_fetch_cost.py --dir "$OUTDIR" \
      --out "$OUTDIR/fig_fetch_cost" || true
fi
echo
echo "Results in $OUTDIR/"
