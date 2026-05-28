#!/bin/bash
# vllm-ppd KV buffer × concurrency sweep
# ========================================
# P 노드 kv_buffer_size를 1→2→4GB로 늘리면서
# 각 buffer 크기에서 C=8, C=12, C=16 실행.
# OOM / 실패 여부와 latency를 비교해 동시성 상한 곡선 측정.
#
# 사용법:
#   cd ~/experiments
#   bash scripts/vllm-ppd/run_buffer_sweep.sh
#
# 결과:
#   results/vllm-ppd/bfcl_multiturn_results_vllm_ppd_buf{N}g_c{C}.json
#   results/vllm-ppd/buffer_sweep_summary.txt

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
BENCHMARK="$PROJECT_DIR/benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py"
RESULTS_DIR="$PROJECT_DIR/results/vllm-ppd"
SUMMARY="$RESULTS_DIR/buffer_sweep_summary.txt"

BUFFERS=(1 2 4)
CONCURRENCIES=(8 12 16)

mkdir -p "$RESULTS_DIR"

# ─── summary header ───────────────────────────────────────────────────────────
{
  echo "vllm-ppd KV buffer sweep — $(date '+%Y-%m-%d %H:%M')"
  echo "buffer_gb | concurrency | status | result_file"
  echo "----------|-------------|--------|------------"
} | tee "$SUMMARY"

# ─── sweep ────────────────────────────────────────────────────────────────────
for BUF in "${BUFFERS[@]}"; do
  MEM_POOL=$(python3 -c "print(max(2, ${BUF} * 2))")
  echo ""
  echo "════════════════════════════════════════════"
  echo "  KV_BUFFER_GB=${BUF}  mem_pool=${MEM_POOL}GB"
  echo "════════════════════════════════════════════"

  # Start servers — run in subshell so set -e doesn't exit sweep on failure
  (
    export KV_BUFFER_GB=$BUF
    bash "$SCRIPT_DIR/start_2P_2D.sh"
  )
  SERVER_EXIT=$?

  if [ $SERVER_EXIT -ne 0 ]; then
    echo "  ERROR: servers failed to start (exit $SERVER_EXIT)"
    for C in "${CONCURRENCIES[@]}"; do
      echo "${BUF}GB | C=${C} | server_start_failed | —" | tee -a "$SUMMARY"
    done
    bash "$SCRIPT_DIR/cleanup_all.sh" 2>/dev/null || true
    sleep 5
    continue
  fi

  # ─── concurrency loop (no server restart between C values) ────────────────
  for C in "${CONCURRENCIES[@]}"; do
    CFG="vllm_ppd_buf${BUF}g_c${C}"
    RESULT_FILE="$RESULTS_DIR/bfcl_multiturn_results_${CFG}.json"
    echo ""
    echo "  ── C=${C}  (config: ${CFG}) ──"

    (
      cd "$PROJECT_DIR"
      CONFIG="$CFG" CONCURRENCY=$C \
        python "$BENCHMARK"
    )
    BM_EXIT=$?

    if [ $BM_EXIT -eq 0 ]; then
      STATUS="ok"
    else
      # Try to distinguish OOM from other failures
      if grep -qiE "out of memory|cuda oom|killed" \
           "$PROJECT_DIR/logs/vllm-ppd/2P_2D/"*.log 2>/dev/null; then
        STATUS="OOM"
      else
        STATUS="failed_exit${BM_EXIT}"
      fi
      echo "  WARNING: benchmark exited $BM_EXIT → ${STATUS}"
    fi

    RFILE_DISPLAY="${RESULT_FILE##$PROJECT_DIR/}"
    echo "${BUF}GB | C=${C} | ${STATUS} | ${RFILE_DISPLAY}" | tee -a "$SUMMARY"

    # If OOM: servers may be crashed — restart before next concurrency level
    if [ "$STATUS" = "OOM" ]; then
      echo "  OOM detected — restarting servers..."
      bash "$SCRIPT_DIR/cleanup_all.sh" 2>/dev/null || true
      sleep 5
      (
        export KV_BUFFER_GB=$BUF
        bash "$SCRIPT_DIR/start_2P_2D.sh"
      ) || {
        echo "  ERROR: server restart failed, skipping remaining C values for buf=${BUF}GB"
        break
      }
    fi
  done

  # Stop servers before next buffer size
  echo ""
  echo "  Stopping servers (buf=${BUF}GB done)..."
  bash "$SCRIPT_DIR/cleanup_all.sh" 2>/dev/null || true
  sleep 5
done

# ─── final summary ────────────────────────────────────────────────────────────
echo ""
echo "════════════════════════════════════════════"
echo "SWEEP COMPLETE"
echo "════════════════════════════════════════════"
cat "$SUMMARY"
echo ""
echo "Full results: $RESULTS_DIR/"
echo "Summary:      $SUMMARY"
