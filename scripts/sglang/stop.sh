#!/bin/bash
# Stop everything a start_2P_2D.sh run left behind, and WAIT until it is really gone.
#
# This file used to be empty (0 bytes) while several runbooks called it, so "stop then
# start the next arm" silently did nothing and the next arm inherited the previous arm's
# servers -- the kind of failure that produces two arms with suspiciously identical
# numbers. Waiting for the processes to exit matters just as much as killing them: the
# next start_*.sh binds the same ports and allocates the same GPUs, and a still-dying
# scheduler holds both.
#
# Also clears the hicache file backend and the park telemetry, both of which are stale
# the moment the servers are gone and both of which have corrupted a later run.
#
# 사용:
#   ./scripts/sglang/stop.sh
#   KEEP_PARK_DIR=1 ./scripts/sglang/stop.sh      # leave /dev/shm telemetry to inspect
#   KEEP_HICACHE_DIR=1 ./scripts/sglang/stop.sh   # leave /tmp/hicache to inspect
set +e

PARK_DIR=${SGLANG_KV_PARK_DIR:-/dev/shm/sglang_kv_parking}
PATTERNS=(
  "sglang::scheduler"
  "sglang::detokenizer"
  "sglang.launch_server"
  "sglang_router.launch_router"
  "mooncake.http_metadata_server"
)
# 8000/8001: router(s) -- 8001 exists only in ROUTER_MODE=skew.
# 8998/8999: PD bootstrap. 30000-30003: servers. 8080: mooncake metadata.
PORTS=(8000 8001 8998 8999 30000 30001 30002 30003 8080)

echo "[1/3] Terminating (SIGTERM first, so schedulers can release GPU memory)..."
for p in "${PATTERNS[@]}"; do
  pkill -f "$p" 2>/dev/null && echo "  TERM $p"
done

# Give them a chance to exit cleanly; a SIGKILLed scheduler can leave the GPU context
# for several seconds longer than a SIGTERMed one.
for i in $(seq 1 15); do
  left=0
  for p in "${PATTERNS[@]}"; do
    n=$(pgrep -f "$p" 2>/dev/null | wc -l)
    left=$((left + n))
  done
  [ "$left" -eq 0 ] && break
  sleep 1
done

echo "[2/3] Force-killing whatever survived..."
for p in "${PATTERNS[@]}"; do
  pkill -9 -f "$p" 2>/dev/null && echo "  KILL $p"
done
for port in "${PORTS[@]}"; do
  fuser -k "${port}/tcp" 2>/dev/null && echo "  freed :$port"
done
sleep 2

echo "[3/3] Verifying..."
alive=0
for p in "${PATTERNS[@]}"; do
  n=$(pgrep -f "$p" 2>/dev/null | wc -l)
  if [ "$n" -gt 0 ]; then echo "  STILL ALIVE ($n): $p"; alive=$((alive + n)); fi
done
if [ "$alive" -gt 0 ]; then
  echo "  -> $alive process(es) refused to die; the next start WILL fail on port/GPU"
  echo "     conflicts. Inspect with: pgrep -af sglang"
else
  echo "  all sglang/mooncake processes gone"
fi

# The park telemetry is keyed by GPU id, not by run, so a stale file from the previous
# arm would be read by the next arm's sampler as live occupancy. The sampler defends
# with --stale-s, but removing them makes the next run's CSV unambiguous.
if [ -d "$PARK_DIR" ] && [ -z "$KEEP_PARK_DIR" ]; then
  rm -f "$PARK_DIR"/parked_gpu*.json "$PARK_DIR"/gpu*_usage 2>/dev/null
  echo "  cleared stale park telemetry in $PARK_DIR"
fi

# The hicache FILE backend writes KV pages to disk and never removes them: HiCacheFile
# has a clear() but nothing calls it on shutdown, so /tmp/hicache grows across every
# point of every hicache arm until the disk is full. That has now killed two completed
# measurements -- a 30-minute closed-loop arm and a 10-minute open-loop point -- and both
# times only the hicache arm, because it is the only arm that writes KV to disk.
# KEEP_HICACHE_DIR=1 to inspect it instead.
HICACHE_DIR=${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-/tmp/hicache}
if [ -d "$HICACHE_DIR" ] && [ -z "$KEEP_HICACHE_DIR" ]; then
  _sz=$(du -sh "$HICACHE_DIR" 2>/dev/null | cut -f1)
  rm -rf "${HICACHE_DIR:?}"/* 2>/dev/null
  echo "  cleared hicache file backend in $HICACHE_DIR (was ${_sz:-unknown})"
fi

if command -v nvidia-smi >/dev/null 2>&1; then
  echo
  echo "GPU memory now:"
  nvidia-smi --query-gpu=index,memory.used,memory.total --format=csv,noheader | sed 's/^/  /'
fi
