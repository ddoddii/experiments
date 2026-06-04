#!/bin/bash
# SGLang 1P1D — KV Cache Tier Latency Test
# ==========================================
# KV cache tier 실험 전용 단일 P+D 구성.
# 모든 요청이 같은 P 노드에서 처리되어야 anchor eviction이 deterministic.
#
# 구성:
#   P1 (GPU 0, port 30000, bootstrap 8998) — hicache 활성화
#   D1 (GPU 2, port 30002)
#   Router (port 8001) — 2P2D router(8000)와 충돌 없음
#
# 사용법:
#   bash scripts/sglang/start_1P_1D_kv_tier.sh
#
# 벤치마크 실행:
#   SERVER_URL=http://127.0.0.1:8001 \
#     python benchmark/sglang_kv_tier_latency.py
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Qwen3-14B"}
QUANTIZATION=${QUANTIZATION:-""}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"hermes"}
HICACHE_RATIO=${HICACHE_RATIO:-"1.2"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/sglang_1p1d"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "SGLang 1P1D KV Tier Latency Test Server"
echo "=================================================="
echo "Model    : $(basename $MODEL_PATH)"
echo "Log dir  : $LOG_DIR"
echo "Router   : http://127.0.0.1:8001"
echo ""

# ─── 기존 프로세스 정리 ────────────────────────────────────────────────────
echo "[0/5] Stopping existing processes..."
pkill -9 -f "sglang.launch_server"  2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router"   2>/dev/null || true
for port in 8001 8080 8998 30000 30002; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 3

# /tmp/hicache 정리 (이전 실험 KV 캐시 제거 → 깨끗한 상태로 시작)
if [ -d "/tmp/hicache" ]; then
    SIZE=$(du -sh /tmp/hicache 2>/dev/null | cut -f1)
    echo "  Cleaning /tmp/hicache ($SIZE) ..."
    rm -rf /tmp/hicache
fi

# ─── Mooncake metadata server ─────────────────────────────────────────────
echo "[1/5] Starting Mooncake metadata server..."
python -m mooncake.http_metadata_server > "$LOG_DIR/mooncake.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/mooncake.log"
sleep 2

export MOONCAKE_MASTER_SERVER=127.0.0.1:8080

# ─── Prefill server P1 (GPU 0) ────────────────────────────────────────────
echo "[2/5] Starting Prefill P1 (GPU 0, port 30000)..."
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port 30000 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-ratio $HICACHE_RATIO \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  > "$LOG_DIR/p1.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/p1.log"
sleep 3

# ─── Decode server D1 (GPU 2) ─────────────────────────────────────────────
echo "[3/5] Starting Decode D1 (GPU 2, port 30002)..."
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port 30002 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend mooncake \
  --tool-call-parser $TOOL_CALL_PARSER \
  > "$LOG_DIR/d1.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/d1.log"

# ─── 서버 ready 대기 ──────────────────────────────────────────────────────
echo ""
echo "[4/5] Waiting for servers to be ready..."
for name_port in "p1:30000" "d1:30002"; do
  name="${name_port%%:*}"
  port="${name_port##*:}"
  echo -n "  Waiting for $name (port $port)..."
  while ! grep -q "ready to roll" "$LOG_DIR/${name}.log" 2>/dev/null; do
    sleep 3
    echo -n "."
    # 에러 감지
    if grep -q "Traceback\|Error\|CUDA out of memory" "$LOG_DIR/${name}.log" 2>/dev/null; then
      echo " ERROR"
      echo "  !! $name failed. Check: $LOG_DIR/${name}.log"
      tail -5 "$LOG_DIR/${name}.log"
      exit 1
    fi
  done
  echo " OK"
done

# ─── Router (port 8001) ────────────────────────────────────────────────────
echo ""
echo "[5/5] Starting Router (port 8001)..."
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 8998 \
  --decode  http://127.0.0.1:30002 \
  --host 0.0.0.0 --port 8001 \
  > "$LOG_DIR/router.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/router.log"
sleep 3

echo ""
echo "=================================================="
echo "Ready. Run the benchmark:"
echo ""
echo "  SERVER_URL=http://127.0.0.1:8001 \\"
echo "    python benchmark/sglang_kv_tier_latency.py"
echo ""
echo "Logs: $LOG_DIR/"
echo "  tail -f $LOG_DIR/p1.log"
echo "=================================================="
