#!/bin/bash
# SGLang 2P2D — KV Pressure 실험용 (작은 GPU KV pool)
# ======================================================
# P 노드의 GPU KV pool을 PREFILL_MEM_FRACTION으로 축소해
# 적은 동시성으로도 L2/L3 hicache eviction을 유발함.
#
# mem_fraction_static 의미:
#   GPU KV pool = (1 - mem_fraction_static) × GPU 전체 메모리
#   Qwen3-14B @ A6000 49GB 기준:
#     0.88 (기본값) → KV = 0.12 × 49 = 5.9GB ≈ 36,000 tokens
#     0.93          → KV = 0.07 × 49 = 3.4GB ≈ 21,000 tokens  ← 기본값
#     0.95          → KV = 0.05 × 49 = 2.5GB ≈ 15,000 tokens
#     0.97          → KV = 0.03 × 49 = 1.5GB ≈  9,000 tokens
#
# CONCURRENCY=8 (P1당 4개) × avg 6000 token ctx = 24,000 tokens
# → 0.93 설정 시 P1 GPU KV 초과 → L2 DRAM eviction 발생
#
# 사용법:
#   bash start_2P_2D_kv_pressure.sh                      # 기본값 (0.93)
#   PREFILL_MEM_FRACTION=0.95 bash start_2P_2D_kv_pressure.sh
#
# 벤치마크:
#   CONCURRENCY=8 TOOL_DELAY_SEC=10 \
#     python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Qwen3-14B"}
QUANTIZATION=${QUANTIZATION:-""}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"hermes"}
HICACHE_RATIO=${HICACHE_RATIO:-"1.2"}
PREFILL_MEM_FRACTION=${PREFILL_MEM_FRACTION:-"0.15"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/sglang"
mkdir -p "$LOG_DIR"

HICACHE_DIR="/tmp/hicache"

# mem_fraction_static 의미: KV pool = mem_fraction_static × available_after_model_load
#   available ≈ 49GB - 28GB(weights) - 2.5GB(CUDA) = ~18.5GB
#   default (0.88): KV = 0.88 × 18.5 = 16.3GB ≈ 99,600 tokens  (eviction 거의 없음)
#   0.15          : KV = 0.15 × 18.5 =  2.8GB ≈ 17,000 tokens  ← 기본값 (flood으로 확실히 evict)
#   0.10          : KV = 0.10 × 18.5 =  1.9GB ≈ 11,500 tokens  (더 공격적)
echo "Model               : $(basename $MODEL_PATH)"
echo "hicache-ratio       : $HICACHE_RATIO"
echo "prefill mem_fraction: $PREFILL_MEM_FRACTION"
echo "  → P1/P2 GPU KV pool ≈ $(python3 -c "print(f'{$PREFILL_MEM_FRACTION * 18.5:.1f}GB ≈ {int($PREFILL_MEM_FRACTION * 18.5 * 1024**3 / 163840):,} tokens (default=99,600)')")"
echo ""

echo "[0/6] Stopping existing processes..."
pkill -9 -f "sglang.launch_server"     2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router"   2>/dev/null || true
pkill -9 -f "sglang_hicache_exporter"       2>/dev/null || true
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
python -m mooncake.http_metadata_server > "$LOG_DIR/mooncake.log" 2>&1 &
MOONCAKE_PID=$!
echo "  PID: $MOONCAKE_PID"
sleep 2

export MOONCAKE_MASTER_SERVER=127.0.0.1:8080

# TCP 포트 고갈 방지 (고동시성 KV transfer로 TIME_WAIT 누적 방지)
sudo sysctl -w net.ipv4.tcp_tw_reuse=1      2>/dev/null || true
sudo sysctl -w net.ipv4.tcp_fin_timeout=15  2>/dev/null || true
sudo sysctl -w net.ipv4.ip_local_port_range="1024 65535" 2>/dev/null || true

echo "[2/6] Starting Prefill server 1 (GPU 0, port 30000)..."
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30000 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --mem-fraction-static $PREFILL_MEM_FRACTION \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio $HICACHE_RATIO \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  > "$LOG_DIR/p1.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo "[3/6] Starting Prefill server 2 (GPU 1, port 30001)..."
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30001 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --mem-fraction-static $PREFILL_MEM_FRACTION \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio $HICACHE_RATIO \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8999 \
  > "$LOG_DIR/p2.log" 2>&1 &
echo "  PID: $!"
sleep 6

echo "[4/6] Starting Decode server 1 (GPU 2, port 30002)..."
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30002 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser $TOOL_CALL_PARSER \
  > "$LOG_DIR/d1.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo "[5/6] Starting Decode server 2 (GPU 3, port 30003)..."
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30003 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser $TOOL_CALL_PARSER \
  > "$LOG_DIR/d2.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo ""
echo "Waiting for all servers to be ready..."
for port in 30000 30001 30002 30003; do
  log=$(case $port in 30000) echo p1;; 30001) echo p2;; 30002) echo d1;; 30003) echo d2;; esac)
  echo -n "  Waiting for port $port ($log)..."
  while ! grep -q "ready to roll" "$LOG_DIR/${log}.log" 2>/dev/null; do
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
  > "$LOG_DIR/router.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo ""
echo "Starting hicache exporter (port 9199)..."
pkill -f "sglang_hicache_exporter" 2>/dev/null || true
sleep 1
HICACHE_DIRS="p1:${HICACHE_DIR},p2:${HICACHE_DIR}" \
SGLANG_INSTANCES="p1:30000,p2:30001,d1:30002,d2:30003" \
EXPORTER_PORT=9199 \
LOG_DIR="$PROJECT_DIR/logs/sglang" \
python3 "$PROJECT_DIR/benchmark/sglang_hicache_exporter.py" \
  > "$LOG_DIR/hicache_exporter.log" 2>&1 &
sleep 2

echo ""
echo "All done."
echo "  Router     : http://127.0.0.1:8000"
echo "  Exporter   : http://127.0.0.1:9199/metrics"
echo "  GPU KV pool: $(python3 -c "print(f'{(1-$PREFILL_MEM_FRACTION)*49:.1f}GB per prefill GPU')")"
echo ""
echo "KV Pressure 실험 실행:"
echo "  CONCURRENCY=8 TOOL_DELAY_SEC=0  python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py"
echo "  CONCURRENCY=8 TOOL_DELAY_SEC=10 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py"
echo "  CONCURRENCY=8 TOOL_DELAY_SEC=30 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py"
