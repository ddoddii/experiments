#!/bin/bash
# Two-minute end-to-end check that the park tier is actually being read back, before
# committing hours to a sweep.
#
# WHY THIS EXISTS
#   check_prefix_continuity.py validates the assumption offline, but against SYNTHETIC
#   replies and the local apply_chat_template. The server does its own rendering, and the
#   real replies are not synthetic. Three separate configurations have now looked
#   completely healthy end-to-end -- full pools, zero errors, plausible bars -- while
#   fetch_hits sat at 0 and the park tier contributed nothing. Each cost about two hours
#   to discover. This runs 5 conversations and reads the one counter that distinguishes
#   "working" from "silently doing nothing".
#
# 사용:
#   MODEL_KEY=qwen14b ./scripts/sglang/smoke_park.sh
#   MODEL_KEY=llama8b ./scripts/sglang/smoke_park.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

MODEL_KEY=${MODEL_KEY:-qwen14b}
ITEMS=${ITEMS:-5}
OUT=${OUT:-results/smoke/$MODEL_KEY}
mkdir -p "$OUT"

# Reuse the sweep's per-model geometry rather than restating it, so the smoke test cannot
# drift from the run it is meant to predict.
eval "$(sed -n '/^model_path()/,/^}/p;/^model_pool()/,/^}/p;/^model_park_pool()/,/^}/p;/^model_memfrac()/,/^}/p;/^model_parser()/,/^}/p;/^model_max_ctx()/,/^}/p;/^model_kv_bytes()/,/^}/p;/^workload_for()/,/^}/p' scripts/sglang/run_models_sweep.sh)"
HF=${HF_HOME_MODELS:-/home/uhmturks/hf_models}
UNIFORM_WORKLOAD=${UNIFORM_WORKLOAD:-0}

export MODEL_PATH=$(model_path "$MODEL_KEY")
export TOOL_CALL_PARSER=$(model_parser "$MODEL_KEY")
export PREFILL_MAX_TOTAL_TOKENS=$(model_pool "$MODEL_KEY")
export PARK_POOL_TOKENS=$(model_park_pool "$MODEL_KEY")
export PARK_MEM_FRACTION=$(model_memfrac "$MODEL_KEY")
read -r MT TURNS <<< "$(workload_for "$MODEL_KEY")"

echo "=== smoke: $MODEL_KEY  ($ITEMS conversations, park arm only) ==="
echo "  model=$MODEL_PATH"
echo "  pool=$PREFILL_MAX_TOTAL_TOKENS park=$PARK_POOL_TOKENS memfrac=$PARK_MEM_FRACTION"
echo "  MAX_TOKENS=$MT MAX_TURNS=$TURNS"

./scripts/sglang/stop.sh > "$OUT/stop.log" 2>&1 || true
PARK_NO_HICACHE=1 IDLE_KV_PARKING=1 PARK_PEER=1 \
  SGLANG_KV_PARK_BW_AWARE=1 SGLANG_KV_PARK_HOST_OVERFLOW=1 \
  SGLANG_KV_PARK_REUSE_AWARE=1 \
  ./scripts/sglang/start_2P_2D.sh

python benchmark/park_location_sampler.py --out "$OUT/parked.csv" --interval 1 \
  > "$OUT/sampler.log" 2>&1 &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null || true' EXIT

CONFIG="smoke_$MODEL_KEY" ROUTER_URL="http://127.0.0.1:8000/v1/chat/completions" \
  MAX_ITEMS=$ITEMS MAX_TOKENS=$MT MAX_TURNS=$TURNS CONCURRENCY=${CONCURRENCY:-4} \
  stdbuf -oL -eL python -u benchmark/sglang_sharegpt_multi_turn_concurrent.py 2>&1 \
  | tee "$OUT/bench.log" | tail -6

sleep 2
kill $SAMPLER 2>/dev/null || true
trap - EXIT
./scripts/sglang/stop.sh > /dev/null 2>&1 || true

echo
python3 - "$OUT/parked.csv" <<'PY'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1])))
if not rows:
    print("VERDICT: no telemetry -- the park manager never published. Check the p*.log.")
    raise SystemExit(1)
last = rows[-1]
g = lambda k: int(float(last.get(k, 0) or 0))
hits, miss = g("fetch_hits"), g("fetch_miss")
print(f"  fetch_hits {hits}   (peer {g('fetch_peer_hits')}, local {g('fetch_local_hits')}, "
      f"host {g('fetch_host_hits')})")
print(f"  fetch_miss {miss}   nospace {g('fetch_nospace')}   already {g('fetch_already')}")
print(f"  parked: local {last.get('park_local_gb')} GB  peer {last.get('park_peer_gb')} GB")
print()
if hits > 0:
    print(f"VERDICT: PASS -- the park tier is being read back "
          f"({hits}/{hits + miss} lookups hit). Safe to run the full sweep.")
else:
    print("VERDICT: FAIL -- fetch_hits is 0. The park tier is doing nothing and the arm")
    print("         will score the same as no-park while looking healthy. Do NOT spend")
    print("         hours on the sweep. Check, in order:")
    print("           python benchmark/check_prefix_continuity.py --model $MODEL_PATH --try-fixes")
    print("           park pool per GPU (PARK_POOL_TOKENS/2) vs one conversation's tokens")
    raise SystemExit(1)
PY
