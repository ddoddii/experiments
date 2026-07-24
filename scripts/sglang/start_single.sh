#!/bin/bash
# Single-GPU plain SGLang server (NO PD-disagg, NO router) for the Intro
# context-length TTFT microbenchmark. radix cache ON so that:
#   - a warmed L-token prefix HITS   -> only the small delta is prefilled ("cached")
#   - a fresh unique L-token prompt MISSES -> full prefill of L ("recompute", i.e.
#     exactly what you pay after the prefix cache is evicted)
# --max-running-requests 1 serializes requests so each TTFT is clean (no batching).
#
# 사용:
#   ./scripts/sglang/start_single.sh                 # gpu0, port 30010, ctx 40k
#   GPU=1 PORT=30011 CONTEXT_LENGTH=65536 ./scripts/sglang/start_single.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
GPU=${GPU:-0}
PORT=${PORT:-30010}
CTX=${CONTEXT_LENGTH:-40000}          # must exceed the largest sweep length (+delta)
MEMFRAC=${MEM_FRACTION:-0.85}
# MAX_RUNNING=1 serializes requests -> clean single-request timing (ttft_ctx_sweep).
# For the QPS load sweep (qps_sweep.py) set MAX_RUNNING high so the server BATCHES,
# e.g. MAX_RUNNING=256 ./scripts/sglang/start_single.sh
MAX_RUNNING=${MAX_RUNNING:-1}
LOG_DIR=${LOG_DIR:-logs/sglang}; mkdir -p "$LOG_DIR"

echo "[single] gpu=$GPU port=$PORT ctx=$CTX max_running=$MAX_RUNNING model=$MODEL_PATH"
pkill -9 -f "launch_server.*--port $PORT" 2>/dev/null || true
sleep 2

CUDA_VISIBLE_DEVICES=$GPU python -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --host 127.0.0.1 --port "$PORT" \
  --mem-fraction-static "$MEMFRAC" \
  --context-length "$CTX" \
  --max-running-requests "$MAX_RUNNING" \
  > "$LOG_DIR/single.log" 2>&1 &
echo "  PID $!"

echo -n "  waiting for ready"
for i in $(seq 1 180); do
  if curl -s "http://127.0.0.1:$PORT/get_model_info" >/dev/null 2>&1; then
    echo " OK"; exit 0
  fi
  echo -n "."; sleep 2
done
echo " TIMEOUT (see $LOG_DIR/single.log)"; exit 1
