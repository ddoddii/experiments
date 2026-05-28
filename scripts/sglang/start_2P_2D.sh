#!/bin/bash
set -e

# conda 초기화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
mkdir -p logs

# SGLang이 실제로 사용하는 hicache SSD 디렉토리 (/tmp/hicache)
HICACHE_DIR="/tmp/hicache"

echo "[0/6] Stopping any existing SGLang/Mooncake/exporter processes..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
pkill -9 -f "sglang_hicache_exporter" 2>/dev/null || true
# 포트 점유 프로세스도 강제 종료
for port in 8998 8999 9000 9001 30000 30001 30002 30003 8000 9199; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 5

# /tmp/hicache 정리 (이전 실험 잔여물 제거)
if [ -d "$HICACHE_DIR" ]; then
    SIZE=$(du -sh "$HICACHE_DIR" 2>/dev/null | cut -f1)
    echo "  Cleaning $HICACHE_DIR ($SIZE) ..."
    rm -rf "$HICACHE_DIR"
fi

echo "[1/6] Starting Mooncake metadata server..."
python -m mooncake.http_metadata_server > logs/mooncake.log 2>&1 &
MOONCAKE_PID=$!
echo "  PID: $MOONCAKE_PID"
sleep 2

export MOONCAKE_MASTER_SERVER=127.0.0.1:8080

echo "[2/6] Starting Prefill server 1 (GPU 0, port 30000, bootstrap 8998)..."
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30000 \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio 1.2 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  > logs/p1.log 2>&1 &
echo "  PID: $!"
sleep 3

echo "[3/6] Starting Prefill server 2 (GPU 1, port 30001, bootstrap 8999)..."
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30001 \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio 1.2 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8999 \
  > logs/p2.log 2>&1 &
echo "  PID: $!"
sleep 3

# hicache 파일 경로를 SGLang이 어떤 dir을 쓰는지 확인 (로그에서 grep)
# 실제 경로가 다른 경우: HICACHE_DIRS="p1:/actual/path,p2:/actual/path" 로 오버라이드
sleep 3

echo "[4/6] Starting Decode server 1 (GPU 2, port 30002)..."
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30002 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  > logs/d1.log 2>&1 &
echo "  PID: $!"
sleep 3

echo "[5/6] Starting Decode server 2 (GPU 3, port 30003)..."
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30003 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  > logs/d2.log 2>&1 &
echo "  PID: $!"
sleep 3

echo ""
echo "Waiting for all servers to be ready..."
for port in 30000 30001 30002 30003; do
  echo -n "  Waiting for port $port..."
  while ! grep -q "ready to roll" logs/$(case $port in 30000) echo p1;; 30001) echo p2;; 30002) echo d1;; 30003) echo d2;; esac).log 2>/dev/null; do
    sleep 3
    echo -n "."
  done
  echo " OK"
done

echo ""
echo "[6/6] Starting Router (port 8000)..."
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:30000 8998 \
  --prefill http://127.0.0.1:30001 8999 \
  --decode http://127.0.0.1:30002 \
  --decode http://127.0.0.1:30003 \
  --host 0.0.0.0 --port 8000 \
  > logs/router.log 2>&1 &
echo "  PID: $!"

# ─── Diagnostics: /metrics 엔드포인트 확인 ─────────────────────────────────
echo ""
echo "Checking SGLang /metrics endpoints..."
for port in 30000 30001 30002 30003; do
  METRIC_LINES=$(curl -s "http://localhost:${port}/metrics" | wc -l)
  if [ "$METRIC_LINES" -gt 0 ]; then
    echo "  Port $port: native /metrics OK ($METRIC_LINES lines)"
  else
    echo "  Port $port: /metrics EMPTY — use sglang_hicache_exporter.py instead"
  fi
done

# ─── Start hicache exporter (Prometheus custom exporter, port 9199) ─────────
echo ""
echo "Starting SGLang hicache Prometheus exporter (port 9199)..."
pkill -f "sglang_hicache_exporter" 2>/dev/null || true
sleep 1

HICACHE_DIRS="p1:${HICACHE_DIR},p2:${HICACHE_DIR}" \
SGLANG_INSTANCES="p1:30000,p2:30001,d1:30002,d2:30003" \
EXPORTER_PORT=9199 \
python3 "$PROJECT_DIR/benchmark/sglang_hicache_exporter.py" \
  > logs/hicache_exporter.log 2>&1 &
EXPORTER_PID=$!
echo "  Exporter PID: $EXPORTER_PID"
sleep 2

if curl -s "http://localhost:9199/metrics" | grep -q "sglang_scrape_ok"; then
  echo "  Exporter: READY (http://localhost:9199/metrics)"
else
  echo "  Exporter: not yet responding (check logs/hicache_exporter.log)"
fi

echo ""
echo "All done. Logs at logs/*.log"
echo "Router at http://127.0.0.1:8000"
echo ""
echo "To check hicache metrics:"
echo "  curl http://localhost:9199/metrics"
echo "  curl http://localhost:30000/get_server_info | python3 -m json.tool"