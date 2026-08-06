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
# TOOL_DELAY is set by the WORKLOAD block below, not here: a default assigned at this
# point would make the block's ${TOOL_DELAY:-3} a no-op and silently drop the think-time
# gap, which is the whole window parking is supposed to exploit.
export HICACHE_RATIO=${HICACHE_RATIO:-1.2}
export PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70}
export PARK_MEM_FRACTION_D=${PARK_MEM_FRACTION_D:-0.88}
export PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-32000}

# HOLD DECODE KV CAPACITY CONSTANT ACROSS ARMS. This is not a tuning detail, it is the
# control that was missing, and without it every park-vs-hicache number in this directory
# is confounded.
#
# sglang sizes a KV pool from memory that is AVAILABLE, not from the whole card:
# rest = available - total * (1 - mem_fraction_static). A park pool allocated on a decode
# GPU is already-used memory when that decode sizes its own pool, so the decode pool
# SHRINKS by exactly the park pool. Measured: decode went from 216,374 tokens under
# hicache to 154,304 under park -- 29% less cache for the same work -- while total HBM on
# the card stayed identical, which is why this hid so well. Comparing those two is
# comparing a cache tier AND a 29% capacity cut at the same time.
#
# Pinning decode's capacity to the smaller value gives both arms the same decode pool, so
# the only thing left differing is where evicted prefill KV goes. Set it to 0 to opt out
# and let each arm size itself, but then do not compare the arms on latency.
export DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS:-154304}

# Park onto the DECODE GPUs. Under PD_LAYOUT=b, P0=gpu0 D0=gpu1 P1=gpu2 D1=gpu3, so each
# prefill lists its NVLink-bridged decode first and the far decode as a fallback. This is
# the configuration the paper claims: the victim cache lives in decode's idle HBM.
PARK_GPUS_P0=${PARK_GPUS_P0:-0,1,3}
PARK_GPUS_P1=${PARK_GPUS_P1:-2,3,1}

# The paper's three bars are recompute / hicache / ours. park_host is not one of them --
# it is the internal control that isolates the medium, and it stays in the default set
# because without it "the GPU link is why" is an assertion.
#
# radix is deliberately NOT in the default arms but IS still selectable, and that choice
# needs to be a conscious one rather than an omission: hicache and park BOTH run with the
# radix prefix cache underneath, so recompute->X measures the value of caching at all,
# while radix->X measures what the offload tier adds ON TOP of the GPU cache. In the
# c16_p32000 run radix alone beat both hicache and park, so if the paper reports only
# recompute as the floor, that result still exists and a reviewer can ask for it. Run
# ARMS="recompute radix hicache park_gpu" to keep it in view.
ARMS=${ARMS:-"recompute hicache park_host park_gpu"}
CAP=${CAP:-}
COSTMODEL=${COSTMODEL:-0}

# --- WORKLOAD: whether this run can show a prefill effect at all ----------------------
# This is not a flavour choice, it decides whether the experiment is capable of a result.
#
# sharegpt (the old default) generates ~1,900 tokens per request. At TPOT 0.0258 that is
# ~49 s of decode against a 0.25 s TTFT, so prefill is 0.5% of the request. Measured on
# it: parking raised the prefix hit rate from 37% to 61% and TTFT still got WORSE, because
# there was no prefill time to give back. Two runs were spent before that was noticed.
#
# longctx makes prefill the bottleneck on purpose: a long per-session document prefix
# (expensive to re-prefill, O(n^2)) with a short answer (cheap decode), accumulating over
# turns, with think-time gaps between them. Sized so the working set OVERSUBSCRIBES the
# prefill pool -- 16 sessions x ~10.6k tokens is ~170k against a 60k pool -- because a
# victim cache can only pay when the pool is actually evicting.
WORKLOAD=${WORKLOAD:-longctx}
case "$WORKLOAD" in
  longctx)
    BENCH=${BENCH:-benchmark/sglang_longctx_multi_turn_concurrent.py}
    export PREFIX_WORDS=${PREFIX_WORDS:-8000}   # ~10.6k tok: re-prefill ~1.7 s
    export MAX_TOKENS=${MAX_TOKENS:-16}         # ~0.4 s of decode -> prefill dominates 4:1
    export NUM_TURNS=${NUM_TURNS:-4}
    export MAX_ITEMS=${MAX_ITEMS:-128}
    export TOOL_DELAY=${TOOL_DELAY:-3}          # think-time: the gap parking exists to use
    ;;
  bfcl)
    # The workload the design is actually motivated by, and a valid testbed unlike the
    # other two: measured TTFT is 66% of a request (sharegpt 1%, longctx 80%).
    # It gets there differently from longctx -- not from a long document, but because the
    # TOOL DEFINITIONS are ~5,335 tokens and are re-sent on every turn, while a tool call
    # replies in a median of 9 tokens. Long prompt, tiny output, naturally.
    #
    # TOOL_DELAY DEFAULTS TO 0: the dataset as it is, with nothing injected. The delay is
    # a time.sleep in the harness -- the benchmark replays recorded turns and never
    # executes the tools -- so any non-zero value is a number someone picked, and a result
    # that depends on it is partly a result about the picking. The 66% TTFT share that
    # makes this a valid testbed was measured at 0, so nothing is lost by leaving it there.
    #
    # It also makes the test HARDER, which is the point. With no gap there is no idle time
    # for anyone to hide background work in, so a win here needs no caveat about tool
    # latency. GAPS= still sweeps it if the question ever becomes "how much idle time would
    # this need", but that is a separate question from whether it works at all.
    BENCH=${BENCH:-benchmark/sglang_BFCL_multi_turn_concurrent.py}
    export TOOL_DELAY=${TOOL_DELAY:-0}
    export MAX_TOKENS=${MAX_TOKENS:-512}
    ;;
  sharegpt)
    BENCH=${BENCH:-benchmark/sglang_sharegpt_multi_turn_concurrent.py}
    export TOOL_DELAY=${TOOL_DELAY:-0}
    ;;
  *) echo "unknown WORKLOAD: $WORKLOAD (want bfcl | longctx | sharegpt)"; exit 1 ;;
esac

# The workload is in the tag: a longctx run and a sharegpt run are not comparable, and
# without it the second would overwrite the first in place.
TAG=${TAG:-why_${WORKLOAD}_c${CONCURRENCY}_p${PARK_POOL_TOKENS_PER_GPU}}
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
    recompute)
      # No prefix reuse at all (--disable-radix-cache, which also forces hicache off since
      # hicache layers on the radix tree). Every turn re-prefills its whole context. This
      # is the floor the other two are measured against.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=0 DISABLE_RADIX_CACHE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    radix)
      # GPU prefix cache only, no offload tier. Not a default arm; see ARMS above.
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
    park_sync)
      # park_gpu with the OLD blocking copy. Not a straw man -- it is what every result
      # so far was measured with, so it is the only honest baseline for the async change.
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=$PARK_GPUS_P0 PARK_GPUS_P1=$PARK_GPUS_P1 \
        SGLANG_KV_PARK_ASYNC_PARK=0 \
        SGLANG_KV_PARK_PRESSURE_AWARE=1 SGLANG_KV_PARK_BW_AWARE=1 \
        ./scripts/sglang/start_2P_2D.sh ;;
    park_nvlink)
      # ONLY the NVLink-bridged decode GPU, and nothing else.
      #
      # `nvidia-smi topo -m` on this box: NV4 on (0,1) and (2,3), everything else NODE or
      # PHB -- i.e. across a PCIe host bridge. Under PD_LAYOUT=b that makes gpu1 the
      # NVLink decode partner of P0 and gpu3 that of P1, and every other candidate a PCIe
      # hop. The default park_gpu list (0,1,3 / 2,3,1) gives each prefill one NVLink
      # target AND one PCIe target and reports both as "peer", so its 10.2 GB/s aggregate
      # cannot be attributed to either.
      #
      # This arm is also exactly the paper's claim stated literally -- the victim cache
      # lives in the idle HBM of the decode node next door -- with no PCIe target and no
      # parking onto the prefill's own GPU (which would take HBM from the serving pool it
      # is meant to relieve).
      PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 \
        PARK_GPUS_P0=${PARK_NVLINK_P0:-1} PARK_GPUS_P1=${PARK_NVLINK_P1:-3} \
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
  # Stamp the sglang commit this arm ran against. Re-running one arm into an existing
  # OUTDIR overwrites only that arm, leaving the others from an older build under the
  # same names -- which already happened: an async-park r1 sat beside blocking-park r2
  # and r3 with nothing on disk to say so. The stamp makes that detectable.
  { echo "{\"arm\": \"$label\", \"ts\": $(date +%s),"
    echo " \"sglang_sha\": \"$(git -C "$(python -c 'import sglang,os;print(os.path.dirname(os.path.dirname(os.path.dirname(sglang.__file__))))' 2>/dev/null)" rev-parse --short HEAD 2>/dev/null || echo unknown)\","
    echo " \"workload\": \"$WORKLOAD\", \"concurrency\": $CONCURRENCY}"
  } > "$OUTDIR/meta_$label.json" 2>/dev/null || true
  # The park telemetry files carry fetch_ms_tier / fetch_tok_tier, which is where the
  # per-tier cost comes from. They live in /dev/shm and are gone once the server exits,
  # so they are copied BEFORE the next stop.sh.
  # The file is parked_gpu<N>.json (_parked_bytes_file in idle_kv_parking.py). The first
  # version of this line globbed parked_bytes_*.json, matched nothing, and -- because the
  # copy is best-effort -- said nothing, so the whole run came back with no per-tier fetch
  # telemetry and decompose_speedup.py reported "no park arm in this directory".
  _n=0
  for f in /dev/shm/sglang_kv_parking/parked_gpu*.json; do
    [ -e "$f" ] || continue
    cp "$f" "$OUTDIR/$(basename "${f%.json}").$label.json" 2>/dev/null && _n=$((_n + 1))
  done
  case "$label" in
    park*)
      if [ "$_n" -eq 0 ]; then
        # Capture WHY, not just THAT. Three runs have now come back with no park
        # telemetry and no way to tell whether the files were never written, written
        # somewhere else, or removed before the copy -- each round cost a full rerun to
        # not answer. Everything needed to tell them apart goes into a .txt (never .log,
        # which .gitignore drops) so it survives the push.
        {
          echo "no parked_gpu*.json found at copy time for $label"
          echo "--- SGLANG_KV_PARK_DIR=${SGLANG_KV_PARK_DIR:-/dev/shm/sglang_kv_parking}"
          ls -la "${SGLANG_KV_PARK_DIR:-/dev/shm/sglang_kv_parking}" 2>&1 | head -30
          echo "--- any park file anywhere under /dev/shm ---"
          find /dev/shm -name 'parked_gpu*' -o -name 'gpu*_usage' 2>/dev/null | head -10
          echo "--- server processes still alive? ---"
          pgrep -af "sglang.launch_server" 2>/dev/null | head -6
          echo "--- park lines from each prefill log (tail) ---"
          for f in p1 p2; do
            _L=${LOG_DIR:-logs/sglang}; [ -d "$_L" ] || _L=logs
            [ -f "$_L/$f.log" ] || continue
            echo "[$_L/$f]"
            grep -iE "idle kv parking|park pool|telemetry publish failed|DISABLED" \
              "$_L/$f.log" 2>/dev/null | tail -12
          done
        } > "$OUTDIR/telemetry_missing_$label.txt" 2>&1
        echo "  [warn] no parked_gpu*.json for $label -- diagnosis written to"
        echo "         $OUTDIR/telemetry_missing_$label.txt"
      fi ;;
  esac
  # Digest, not the raw log: .gitignore has a blanket '*.log' and every raw log copied
  # here in earlier runners was silently dropped before it could be pushed.
  #
  # TAIL, NOT HEAD. logs/*.log is not truncated between arms, so `head` returned the
  # FIRST server's startup lines for every arm -- the first run of this script produced
  # seven byte-identical digests reporting a pool size that belonged to a stale log from
  # a previous session. Config was unverifiable for the whole run. Taking the LAST
  # startup block gets the server this arm actually started.
  # logs/sglang/, not logs/ -- the 2P2D starter writes there. Reading the wrong path is
  # why every digest in every run so far carried a stale July startup block and the
  # "identical to the previous arm" warning fired on runs whose logs were fine.
  LOGD=${LOG_DIR:-logs/sglang}
  [ -d "$LOGD" ] || LOGD=logs
  for f in p1 p2 d1 d2; do
    [ -f "$LOGD/$f.log" ] || continue
    { echo "---- startup (LAST occurrence: the log is not truncated between arms) ----"
      grep -iE "park pool|clamp|ready to roll|max_total_num_tokens" "$LOGD/$f.log" | tail -8
      echo "---- errors (last) ----"
      grep -iE "error|traceback|out of memory|SIGQUIT received" "$LOGD/$f.log" | tail -15
      echo "---- tail ----"; tail -15 "$LOGD/$f.log"
    } > "$OUTDIR/$f.$label.digest.txt" 2>/dev/null || true
  done
  # Fail loudly rather than silently reporting a stale config next run.
  if [ -f "$OUTDIR/p1.$label.digest.txt" ] && [ -n "$_PREV_DIGEST" ] \
     && cmp -s "$OUTDIR/p1.$label.digest.txt" "$_PREV_DIGEST"; then
    echo "  [warn] p1 digest identical to the previous arm's -- logs/p1.log is probably"
    echo "         not being rotated, so per-arm config in these digests is unreliable."
  fi
  _PREV_DIGEST="$OUTDIR/p1.$label.digest.txt"
}
_PREV_DIGEST=""

# REPEATS interleaves the arms (a1 b1 a2 b2 ...) rather than running all of one then all
# of the other. Server17 drifts over a session -- the same config re-run gave TTFT p95
# from 1.45 to 6.58 s -- so grouping by arm confounds the arm with whenever it happened
# to run. Interleaving spreads that drift across both arms instead of loading it onto one.
REPEATS=${REPEATS:-1}

# GAPS turns the tool-call gap into an AXIS instead of a constant, and it has to be one.
# TOOL_DELAY is a time.sleep in the harness -- the benchmark replays recorded turns and
# does not execute the tools -- so any single value is a number someone chose. A result
# that only holds at "3 s" is a result about that choice.
#
# Swept, it becomes a finding instead: the gap is idle time background work can hide in,
# so the question is HOW MUCH gap the advantage needs, and that break-even can be compared
# against what real tool calls actually cost (file ops ~ms, HTTP APIs 100 ms-2 s, sandboxed
# code execution seconds). If the break-even is tens of milliseconds the claim covers
# essentially every agentic deployment; if it needs seconds, it covers few.
#
# Note the gap helps BOTH arms -- hicache's writeback thread uses it too -- so this is not
# a handicap being handed to one side, it is the axis on which the two differ.
GAPS=${GAPS:-}
for rep in $(seq 1 "$REPEATS"); do
  for gap in ${GAPS:-__none__}; do
    for arm in $ARMS; do
      lab="$arm"
      [ "$gap" != "__none__" ] && { export TOOL_DELAY="$gap"; lab="${arm}_g${gap}"; }
      [ "$REPEATS" = "1" ] || lab="${lab}_r${rep}"
      run_one "$lab" "$arm"
    done
  done
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
# The medium comparison, when both arms are present. Same code, same policy, same pool
# allocation -- only SGLANG_KV_PARK_FORCE_HOST differs -- so a difference between them is
# the medium and nothing else.
if [ -n "$(ls "$OUTDIR"/parked_gpu*.park_host*.json 2>/dev/null)" ] \
   && [ -n "$(ls "$OUTDIR"/parked_gpu*.park_gpu*.json 2>/dev/null)" ]; then
  python benchmark/why_gpu_beats_dram.py --dir "$OUTDIR" || true
fi
if [ "$COSTMODEL" = "1" ]; then
  python benchmark/fit_fetch_cost.py --dir "$OUTDIR" \
      --out "$OUTDIR/fig_fetch_cost" || true
fi
echo
echo "Results in $OUTDIR/"
