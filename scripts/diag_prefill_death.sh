#!/bin/bash
# Collect everything needed to explain why the prefill workers stopped serving mid-run.
#
# What we already know from the c4 run and repro_empty_output.py:
#   - failure is a clean CUTOFF, not per-tool-class: dataset idx 0-74 all succeeded,
#     idx 75-199 all produced HTTP 200 with zero tokens. The dataset happens to be
#     ordered by tool class, which is why it first looked class-dependent.
#   - the reproducer now gets HTTP 500 KVTransferError ("Failed to get kvcache from
#     prefill instance, it might be dead"), then HTTP 503 from the router
#     ("No available prefill workers (all circuits open or unhealthy)").
#   - Mooncake logged: TcpTransport::getConnection ... connect: Cannot assign requested
#     address, to <node public IP>:15119.
#
# So the question is narrow: did the prefill PROCESSES die, or are they alive but
# unreachable/circuit-broken -- and was the transport failure caused by ephemeral port
# exhaustion (EADDRNOTAVAIL is the classic symptom of that).
#
# 사용:  ./scripts/diag_prefill_death.sh 2>&1 | tee /tmp/diag_prefill.txt
set +e
cd "$(dirname "$0")/.."
LOG_DIR=${LOG_DIR:-logs/sglang}
line() { printf '\n===== %s =====\n' "$1"; }

line "1. are the servers still running?"
pgrep -af "sglang.launch_server" | sed 's/^/  /' || \
  echo "  NONE -- the prefill/decode processes are GONE (they crashed)"
echo "  routers:"; pgrep -af "sglang_router" | sed 's/^/    /'
echo "  mooncake:"; pgrep -af "mooncake" | sed 's/^/    /'

line "2. do the servers answer DIRECTLY (bypassing the router)?"
for port in 30000 30001 30002 30003; do
  code=$(curl -s -o /dev/null -w '%{http_code}' -m 5 "http://127.0.0.1:$port/get_model_info")
  echo "  :$port /get_model_info -> HTTP ${code:-timeout}"
done
echo "  router :8000 /health -> $(curl -s -o /dev/null -w '%{http_code}' -m 5 http://127.0.0.1:8000/health)"

line "3. EPHEMERAL PORT PRESSURE (prime suspect for EADDRNOTAVAIL)"
echo "  ip_local_port_range: $(cat /proc/sys/net/ipv4/ip_local_port_range)"
echo "  tw_reuse=$(cat /proc/sys/net/ipv4/tcp_tw_reuse 2>/dev/null) fin_timeout=$(cat /proc/sys/net/ipv4/tcp_fin_timeout 2>/dev/null)"
echo "  socket summary:"; ss -s 2>/dev/null | sed 's/^/    /'
echo "  connection states:"
ss -ant 2>/dev/null | awk 'NR>1{print $1}' | sort | uniq -c | sort -rn | sed 's/^/    /'
echo "  TIME-WAIT total: $(ss -ant state time-wait 2>/dev/null | wc -l)"
echo "  top peers by connection count:"
ss -ant 2>/dev/null | awk 'NR>1{print $5}' | sed 's/:[0-9]*$//' | sort | uniq -c | sort -rn | head -8 | sed 's/^/    /'

line "4. what is :15119 / which address is advertised for KV transfer?"
echo "  local addresses:"; hostname -I 2>/dev/null | sed 's/^/    /'
ss -antlp 2>/dev/null | grep -E ":(15[0-9]{3}|8998|8999|9000|9001)" | sed 's/^/    /' | head
echo "  mooncake/bootstrap env of live servers:"
for pid in $(pgrep -f "sglang.launch_server" | head -2); do
  echo "    pid $pid:"
  tr '\0' '\n' < /proc/$pid/environ 2>/dev/null | grep -iE "mooncake|bootstrap|MASTER|HOST" | sed 's/^/      /'
done

line "5. how the prefill logs END (crash vs quiet)"
for f in "$LOG_DIR"/p1.log "$LOG_DIR"/p2.log; do
  [ -f "$f" ] || { echo "  $f missing"; continue; }
  echo "  --- $f (last 25 lines, progress spam removed)"
  tr '\r' '\n' < "$f" | grep -vE "Prefill batch|Decode batch|GET /metrics|GET /server_info" | tail -25 | sed 's/^/    /'
done

line "6. first occurrence of each failure signature"
for pat in "Cannot assign requested address" "KVTransferError" "out of memory" "Traceback" "CUDA error" "Aborted"; do
  echo "  [$pat]"
  grep -m2 -H "$pat" "$LOG_DIR"/*.log 2>/dev/null | cut -c1-200 | sed 's/^/    /'
done

line "7. router's view: when did circuits open?"
grep -m20 -iE "circuit|unhealthy|remov|fail" "$LOG_DIR"/router.log 2>/dev/null | cut -c1-180 | sed 's/^/  /' | head -20

line "8. resources"
free -h | sed 's/^/  /'
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null | sed 's/^/  /'
echo "  OOM kills:"
{ dmesg 2>/dev/null || sudo -n dmesg 2>/dev/null; } | grep -iE "killed process|out of memory" | tail -5 | sed 's/^/    /'

line "done"
cat <<'EOT'
Interpretation:
  - processes GONE + Traceback/OOM in p*.log
      -> a real crash; section 6 names it.
  - processes ALIVE and :30000 answers 200
      -> the prefills were never dead. The ROUTER's circuit breaker is stuck open,
         so restarting only the router would recover the run.
  - TIME-WAIT in the tens of thousands, or a narrow ip_local_port_range
      -> ephemeral port exhaustion broke Mooncake's transport (EADDRNOTAVAIL),
         which made decode report the prefill as dead, which opened the circuits.
EOT
