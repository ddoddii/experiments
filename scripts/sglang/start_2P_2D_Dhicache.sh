#!/bin/bash
# SGLang 2P2D — D-only hicache
# ==============================
# P 노드: hicache 없음 (순수 prefill, GPU-only KV)
# D 노드: --enable-hierarchical-cache (SSD KV offload)
#
# 목적: Decode 노드에 hicache를 적용해 동시 요청 처리 용량 확장.
#   - P 노드는 매번 full prefill → SSD prefix 재사용 없음
#   - D 노드는 active sequence KV를 SSD에 offload → 더 큰 배치 처리 가능
#
# RAM: D 2대 × 1.2 × 25.52 GB ≈ 61 GB DRAM 필요 → 125 GB 서버에서 안전
#
# 벤치마크:
#   CONFIG=sglang_2p2d_dhicache python benchmark/sglang_BFCL_v3_multi_turn_base.py
#   CONFIG=sglang_2p2d_dhicache_c4 CONCURRENCY=4 \
#     python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
mkdir -p logs

HICACHE_DIR="/tmp/hicache"

echo "[0/6] Stopping any existing SGLang/Mooncake/exporter processes..."
pkill -9 -f "sglang.launch_server"       2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
pkill -9 -f "sglang_hicache_exporter"    2>/dev/null || true
for port in 8998 8999 9000 9001 30000 30001 30002 30003 8000 9199; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 5

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

echo "[2/6] Starting Prefill server 1 (GPU 0, port 30000) — NO hicache..."
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30000 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  > logs/p1.log 2>&1 &
echo "  PID: $!"
sleep 3

echo "[3/6] Starting Prefill server 2 (GPU 1, port 30001) — NO hicache..."
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30001 \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8999 \
  > logs/p2.log 2>&1 &
echo "  PID: $!"
sleep 6

echo "[4/6] Starting Decode server 1 (GPU 2, port 30002) — hicache ON..."
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30002 \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio 1.2 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser llama3 \
  > logs/d1.log 2>&1 &
echo "  PID: $!"
sleep 3

echo "[5/6] Starting Decode server 2 (GPU 3, port 30003) — hicache ON..."
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30003 \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio 1.2 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser llama3 \
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

echo ""
echo "Starting SGLang hicache Prometheus exporter (port 9199)..."
pkill -f "sglang_hicache_exporter" 2>/dev/null || true
sleep 1

# D 노드만 hicache — HICACHE_DIRS는 d1/d2만 추적
HICACHE_DIRS="d1:${HICACHE_DIR},d2:${HICACHE_DIR}" \
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
echo "All done. Config: D-only hicache"
echo "  P1/P2: no hicache (GPU-only KV)"
echo "  D1/D2: hicache ON → /tmp/hicache/"
echo ""
echo "Benchmark:"
echo "  CONFIG=sglang_2p2d_dhicache python benchmark/sglang_BFCL_v3_multi_turn_base.py"
echo "  CONFIG=sglang_2p2d_dhicache_c4 CONCURRENCY=4 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py"
