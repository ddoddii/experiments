#!/bin/bash
# Which sglang is the RUNNING server actually executing?
#
# probe_cache_metric.sh found sglang:cache_occupancy present in the source tree but
# absent from every live server. The gauge is registered in log_stats() on the line after
# sglang:token_usage, and token_usage IS exported, so the running process cannot be
# executing the patched file. This finds out which file it is executing instead.
#
# Run it while the servers are still up.
set +e
SRC=${SGLANG_SRC_DIR:-$HOME/sglang-source}
source "$(conda info --base)/etc/profile.d/conda.sh" 2>/dev/null
conda activate sglang 2>/dev/null

echo "=== 1. what the start script decided ==="
grep -E "^\[env\]" /tmp/probe_start.log 2>/dev/null | head -8 \
  || echo "  (no /tmp/probe_start.log -- re-run probe_cache_metric.sh)"

echo
echo "=== 2. what a fresh import resolves to, in this shell ==="
python - <<'PY'
import importlib.util, os
for mod in ("sglang", "sglang.srt.metrics.collector"):
    try:
        spec = importlib.util.find_spec(mod)
        print(f"  {mod:34s} {spec.origin}")
    except Exception as e:
        print(f"  {mod:34s} <{e}>")
PY
echo "  PYTHONPATH=${PYTHONPATH:-<unset>}"

echo
echo "=== 3. does THAT collector.py have the gauge? ==="
_f=$(python -c "import sglang.srt.metrics.collector as m; print(m.__file__)" 2>/dev/null)
if [ -n "$_f" ]; then
  echo "  file : $_f"
  echo "  cache_occurrence count : $(grep -c cache_occupancy "$_f" 2>/dev/null)"
  echo "  mtime: $(stat -c %y "$_f" 2>/dev/null)"
else
  echo "  could not import sglang here"
fi
echo "  source tree copy:"
echo "    file : $SRC/python/sglang/srt/metrics/collector.py"
echo "    count: $(grep -c cache_occupancy "$SRC/python/sglang/srt/metrics/collector.py" 2>/dev/null)"
echo "    git  : $(git -C "$SRC" rev-parse --short HEAD 2>/dev/null) on $(git -C "$SRC" rev-parse --abbrev-ref HEAD 2>/dev/null)"

echo
echo "=== 4. what the LIVE server process has on its path ==="
# The scheduler subprocesses inherit the launcher's environment, so read it from /proc
# rather than trusting that this shell looks like theirs.
_pid=$(pgrep -f "sglang.launch_server" | head -1)
if [ -n "$_pid" ]; then
  echo "  pid $_pid"
  echo "  PYTHONPATH : $(tr '\0' '\n' < /proc/$_pid/environ 2>/dev/null | grep '^PYTHONPATH=' | cut -d= -f2-)"
  echo "  cwd        : $(readlink -f /proc/$_pid/cwd 2>/dev/null)"
  echo "  exe        : $(readlink -f /proc/$_pid/exe 2>/dev/null)"
  # The definitive answer: which collector.py file the process has OPEN/mapped.
  echo "  sglang files this process actually loaded:"
  tr '\0' '\n' < /proc/$_pid/maps 2>/dev/null | grep -o '/[^ ]*sglang[^ ]*' | sort -u | head -3
  ls -l /proc/$_pid/cwd 2>/dev/null >/dev/null
  echo "  (python bytecode caches for the loaded tree:)"
  find "$SRC/python/sglang/srt/metrics" -name "collector*.pyc" -newer "$SRC/python/sglang/srt/metrics/collector.py" 2>/dev/null | head -3
else
  echo "  no sglang.launch_server running -- start the servers first"
fi

echo
echo "=== 5. verdict ==="
_c=$(grep -c cache_occupancy "$SRC/python/sglang/srt/metrics/collector.py" 2>/dev/null || true)
[ -z "$_c" ] && _c=0
if [ "$_c" -lt 2 ] 2>/dev/null; then
  echo "  The SOURCE TREE is stale: collector.py has $_c references to cache_occupancy."
  echo "  Nothing is wrong with the path or the servers. Pull, then restart:"
  echo "     git -C $SRC pull"
  echo "     ./scripts/sglang/stop.sh && ./scripts/sglang/probe_cache_metric.sh"
elif python -c "import sglang" 2>/dev/null && \
     ! python -c "import sglang, sys; sys.exit(0 if '"'"'$SRC'"'"' in sglang.__file__ else 1)" 2>/dev/null; then
  echo "  sglang imports from OUTSIDE $SRC. PYTHONPATH is not reaching the servers:"
  echo "     cd $SRC/python && pip install -e . --no-deps --no-build-isolation"
else
  echo "  Source is current and the path is right, so the running servers predate the"
  echo "  pull. Restart them:"
  echo "     ./scripts/sglang/stop.sh && ./scripts/sglang/probe_cache_metric.sh"
fi
