#!/bin/bash
# SGLang 2P2D — P-only hicache (가변 ratio)
# ==========================================
# P 노드: --enable-hierarchical-cache (prefix KV → SSD, cross-session 재사용)
# D 노드: hicache 없음 (decode 모드가 chunk cache 강제 → hicache 불가)
#
# 참고: SGLang 0.5.9에서 --disaggregation-mode decode는 내부적으로
#   disable-radix-cache를 강제 설정함 → --enable-hierarchical-cache와 충돌.
#   따라서 D 노드에는 hicache를 적용할 수 없음.
#
# start_2P_2D.sh와의 차이: hicache-ratio를 HICACHE_RATIO 환경변수로 조절 가능.
#   기본값 0.5 (P 2대 × 0.5 × 25.52 ≈ 26 GB DRAM, 안전)
#   start_2P_2D.sh는 ratio=1.2 고정.
#
# RAM 주의: P 2대 × hicache_ratio × GPU_KV_size GB
#   - ratio 0.5: 2 × 0.5 × 25.52 ≈ 26 GB DRAM  (권장)
#   - ratio 1.0: 2 × 1.0 × 25.52 ≈ 51 GB DRAM  (안전)
#   - ratio 1.2: 2 × 1.2 × 25.52 ≈ 61 GB DRAM  (start_2P_2D.sh 기본값)
#
# 벤치마크:
#   CONFIG=sglang_2p2d_pdhicache python benchmark/sglang_BFCL_v3_multi_turn_base.py
#   CONFIG=sglang_2p2d_pdhicache_c4 CONCURRENCY=4 \
#     python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
mkdir -p logs/sglang

HICACHE_DIR="/tmp/hicache"
# 4노드 모두 hicache → DRAM 부족 위험. ratio 낮춤.
# 실패 시: HICACHE_RATIO=1.0 bash start_2P_2D_PDhicache.sh
HICACHE_RATIO=${HICACHE_RATIO:-"0.5"}

echo "[0/6] Stopping any existing SGLang/Mooncake/exporter processes..."
pkill -9 -f "sglang.launch_server"          2>/dev/null || true
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

echo "  hicache-ratio = $HICACHE_RATIO (est. DRAM: $(python3 -c "print(f'{4*float(\"$HICACHE_RATIO\")*25.52:.0f} GB')") across 4 nodes)"
echo "  RAM check: $(free -h | awk 'NR==2{print $2" total, "$7" available"}')"

echo "[1/6] Starting Mooncake metadata server..."
python -m mooncake.http_metadata_server > logs/sglang/mooncake.log 2>&1 &
MOONCAKE_PID=$!
echo "  PID: $MOONCAKE_PID"
sleep 2

export MOONCAKE_MASTER_SERVER=127.0.0.1:8080

echo "[2/6] Starting Prefill server 1 (GPU 0, port 30000) — hicache ON..."
CUDA_VISIBLE_DEVICES=0 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30000 \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio $HICACHE_RATIO \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  > logs/sglang/p1.log 2>&1 &
echo "  PID: $!"
sleep 3

echo "[3/6] Starting Prefill server 2 (GPU 1, port 30001) — hicache ON..."
CUDA_VISIBLE_DEVICES=1 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30001 \
  --enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio $HICACHE_RATIO \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8999 \
  > logs/sglang/p2.log 2>&1 &
echo "  PID: $!"
sleep 6

echo "[4/6] Starting Decode server 1 (GPU 2, port 30002) — NO hicache (decode mode incompatible)..."
CUDA_VISIBLE_DEVICES=2 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30002 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser llama3 \
  > logs/sglang/d1.log 2>&1 &
echo "  PID: $!"
sleep 3

echo "[5/6] Starting Decode server 2 (GPU 3, port 30003) — NO hicache (decode mode incompatible)..."
CUDA_VISIBLE_DEVICES=3 python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30003 \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser llama3 \
  > logs/sglang/d2.log 2>&1 &
echo "  PID: $!"
sleep 3

echo ""
echo "Waiting for all servers to be ready..."
for port in 30000 30001 30002 30003; do
  echo -n "  Waiting for port $port..."
  while ! grep -q "ready to roll" logs/sglang/$(case $port in 30000) echo p1;; 30001) echo p2;; 30002) echo d1;; 30003) echo d2;; esac).log 2>/dev/null; do
    sleep 3
    echo -n "."
    # OOM 감지
    if grep -q "OOM\|out of memory\|Killed" logs/sglang/p1.log logs/sglang/p2.log logs/sglang/d1.log logs/sglang/d2.log 2>/dev/null; then
      echo ""
      echo "ERROR: OOM detected in logs! Try lowering HICACHE_RATIO:"
      echo "  HICACHE_RATIO=0.3 bash $0"
      exit 1
    fi
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
  > logs/sglang/router.log 2>&1 &
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

# P 노드만 hicache → p1/p2 SSD 추적
HICACHE_DIRS="p1:${HICACHE_DIR},p2:${HICACHE_DIR}" \
SGLANG_INSTANCES="p1:30000,p2:30001,d1:30002,d2:30003" \
EXPORTER_PORT=9199 \
LOG_DIR="$PROJECT_DIR/logs/sglang" \
python3 "$PROJECT_DIR/benchmark/sglang_hicache_exporter.py" \
  > logs/sglang/hicache_exporter.log 2>&1 &
EXPORTER_PID=$!
echo "  Exporter PID: $EXPORTER_PID"
sleep 2

if curl -s "http://localhost:9199/metrics" | grep -q "sglang_scrape_ok"; then
  echo "  Exporter: READY (http://localhost:9199/metrics)"
else
  echo "  Exporter: not yet responding (check logs/sglang/hicache_exporter.log)"
fi

echo ""
echo "All done. Config: P-only hicache (ratio=$HICACHE_RATIO)"
echo "  P1/P2: hicache ON (prefix KV → SSD, ratio=$HICACHE_RATIO)"
echo "  D1/D2: no hicache (decode mode forces chunk cache — hicache incompatible)"
echo "  SSD dir: /tmp/hicache/"
echo "  RAM usage: ~$(python3 -c "print(f'{2*float(\"$HICACHE_RATIO\")*25.52:.0f} GB')") DRAM (P nodes L2 tier)"
echo ""
echo "Benchmark:"
echo "  CONFIG=sglang_2p2d_pdhicache python benchmark/sglang_BFCL_v3_multi_turn_base.py"
echo "  CONFIG=sglang_2p2d_pdhicache_c4 CONCURRENCY=4 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py"
echo ""
echo "  RAM 모니터링: watch -n 2 'free -h | head -3'"
echo "  SSD 사용: du -sh /tmp/hicache"
echo "  Logs at logs/sglang/*.log"
