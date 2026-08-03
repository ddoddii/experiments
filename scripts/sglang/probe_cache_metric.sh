#!/bin/bash
# 4-minute probe: is sglang:cache_occupancy exported, and is the prefill radix cache
# actually holding KV on the GPU?
#
# WHY THIS EXISTS
#   The imbalance probe reported prefill pools at 2% occupancy. token_usage cannot settle
#   that question -- it is defined as (max_total - available - evictable)/max_total, so a
#   pool that is 100% full of prefix cache and a pool that is genuinely empty BOTH report
#   ~0.02. The two readings have opposite consequences for Exp 2, and the run that
#   produced them is 25 minutes long. This is the 4-minute version.
#
#   It also separates the two reasons occupancy could equal pressure:
#     - the source tree was never pulled, so the gauge does not exist        (fix and re-run)
#     - the gauge exists and evictable really is ~0 on the prefill GPU       (a FINDING:
#       HiCache has moved the cache off the GPU to host DRAM, which reframes Exp 2 rather
#       than blocking it)
#
# 사용:
#   ./scripts/sglang/probe_cache_metric.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/../.."

SRC=${SGLANG_SRC_DIR:-$HOME/sglang-source}

echo "=================== 1/4  source tree ==================="
echo "  $SRC"
if [ -d "$SRC/.git" ]; then
  echo "  branch : $(git -C "$SRC" rev-parse --abbrev-ref HEAD)"
  echo "  commit : $(git -C "$SRC" rev-parse --short HEAD)"
fi
_n=$(grep -c "cache_occupancy" "$SRC/python/sglang/srt/metrics/collector.py" 2>/dev/null || echo 0)
if [ "$_n" -lt 2 ]; then
  echo
  echo "  >>> collector.py does NOT contain sglang:cache_occupancy."
  echo "  >>> The source tree is stale. Fix and stop here:"
  echo "  >>>     git -C $SRC pull"
  echo "  >>> Nothing below would be meaningful, so not starting any servers."
  exit 1
fi
echo "  collector.py contains cache_occupancy ($_n references) -- patch is present."

echo
echo "=================== 2/4  starting 2P2D (hicache) ==================="
PD_LAYOUT=b PARK_MEM_FRACTION_D=${PARK_MEM_FRACTION_D:-0.80} \
  PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-60000} \
  IDLE_KV_PARKING=0 HICACHE_WRITE_POLICY=write_through_selective \
  HICACHE_STORAGE_BACKEND=file \
  ./scripts/sglang/start_2P_2D.sh > /tmp/probe_start.log 2>&1
echo "  up (log: /tmp/probe_start.log)"

echo
echo "=================== 3/4  building a cache ==================="
# Enough multi-turn traffic that a working prefix cache MUST be holding something:
# 8 sessions x 4 turns, each turn re-sending the whole conversation.
CONCURRENCY=8 MAX_ITEMS=8 MIN_TURNS=4 MAX_TURNS=4 MAX_TOKENS=256 TOOL_DELAY=0 \
  CONFIG=probe_cache_metric \
  python benchmark/sglang_sharegpt_multi_turn_concurrent.py 2>&1 | tail -4

echo
echo "=================== 4/4  what the prefills report ==================="
python3 - <<'PY'
import urllib.request
KEYS = ("sglang:cache_occupancy", "sglang:token_usage", "sglang:evictable_tokens",
        "sglang:num_used_tokens", "sglang:max_total_num_tokens", "sglang:cache_hit_rate")
for port, name in ((30000, "P0"), (30001, "P1"), (30002, "D0"), (30003, "D1")):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:
        print(f"{name}: unreachable ({e})")
        continue
    vals = {}
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2 and parts[0].split("{", 1)[0] in KEYS:
            try:
                vals[parts[0].split("{", 1)[0]] = float(parts[1])
            except ValueError:
                pass
    print(f"\n{name}:")
    for k in KEYS:
        print(f"   {k:32s} {vals.get(k, 'ABSENT')}")
    occ, ev, cap = (vals.get("sglang:cache_occupancy"), vals.get("sglang:evictable_tokens"),
                    vals.get("sglang:max_total_num_tokens"))
    if occ is None:
        print("   -> gauge ABSENT although the source tree has it. Either this server was"
              "\n      started before the pull, or it imports sglang from somewhere else."
              "\n      Run ./scripts/sglang/which_sglang.sh to find out which.")
    elif ev is not None and cap:
        print(f"   -> {ev:.0f} of {cap:.0f} tokens are prefix-cached on this GPU "
              f"({ev/cap*100:.1f}% of the pool)")
PY

echo
echo "Read it like this:"
echo "  P0/P1 evictable LARGE  -> the cache lives on the prefill GPU. Exp 2 proceeds as"
echo "                            designed; cache_occupancy is the right occupancy metric."
echo "  P0/P1 evictable ~0     -> HiCache has moved the cache OFF the GPU to host DRAM."
echo "                            The prefill GPU pool really is near-empty, and the"
echo "                            imbalance to report is 'cache in host DRAM while GPU HBM"
echo "                            sits idle' -- which is the paper's actual claim. Exp 2"
echo "                            gets reframed, not abandoned."
echo
echo "Servers left UP so you can poke further. ./scripts/sglang/stop.sh when done."
