#!/bin/bash
# SGLang 2P2D — P-only hicache
# ==============================
# P 노드: --enable-hierarchical-cache (prefix KV → CPU DRAM → SSD, cross-turn 재사용)
# D 노드: hicache 없음 (decode 모드가 chunk cache 강제 → hicache 불가)
#
# hicache-ratio: DRAM 캐시 크기 = GPU KV 캐시 × ratio
#   Llama-3.1-8B BF16 A6000 기준 GPU KV ≈ 31GB (47GB - 16GB 모델 가중치)
#   ratio=1.2 → P 2대 × ~37GB ≈ 74GB DRAM (기본값; 125GB 서버라면 안전)
#   RAM 여유가 적으면 HICACHE_RATIO 낮춰라 (예: 0.5).
#
# 사용법:
#   bash start_2P_2D.sh                                  # 기본값 (Llama-3.1-8B, BF16, llama3 parser)
#   MODEL_PATH=/path/to/Qwen QUANTIZATION= TOOL_CALL_PARSER=hermes bash start_2P_2D.sh
#   HICACHE_RATIO=0.5 bash start_2P_2D.sh               # ratio=0.5
#
# 벤치마크:
#   CONFIG=sglang_2p2d python benchmark/sglang_BFCL_v3_multi_turn_base.py
set -e

# conda 초기화
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
QUANTIZATION=${QUANTIZATION:-""}           # 8B@BF16=16GB, quantization 불필요
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"llama3"}   # Llama-3.1 uses llama3 parser
HICACHE_RATIO=${HICACHE_RATIO:-"1.2"}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-"write_through_selective"}
# 압박(pressure) knob: KV 풀을 작게 잡아 점유율이 포화까지 오르게 (불균형/축출 측정용).
# 빈 값이면 기본 대용량 풀. 예: PREFILL_MAX_TOTAL_TOKENS=20000
PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-}
DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS:-}
P_MTT=""; [ -n "$PREFILL_MAX_TOTAL_TOKENS" ] && P_MTT="--max-total-tokens $PREFILL_MAX_TOTAL_TOKENS"
D_MTT=""; [ -n "$DECODE_MAX_TOTAL_TOKENS" ] && D_MTT="--max-total-tokens $DECODE_MAX_TOTAL_TOKENS"

# --- Phase 2 slice-2: idle-KV-parking in 2P2D (IDLE_KV_PARKING=1) --------------
# Each prefill parks decode-finished KV onto whichever P GPU is momentarily idle
# (pressure-aware), discoverable cross-node via the shared index. Requires: all 4
# procs SEE all 4 GPUs (CUDA IPC) -> full visibility + --base-gpu-id; park pool on
# each P's own GPU (P0->gpu0, P1->gpu1); reduced --mem-fraction-static to leave HBM
# for the park buffer. All nodes of one run share SGLANG_KV_PARK_EPOCH.
PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-30000}          # per-P park buffer (~4GB @ 30k)
PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70}         # leave room for the park buffer
SGLANG_KV_PARK_GEN=${SGLANG_KV_PARK_GEN:-1}
SGLANG_KV_PARK_PRESSURE_AWARE=${SGLANG_KV_PARK_PRESSURE_AWARE:-1}
# session-keyed parking (task 7): free superseded per-conversation versions -> small pool
# holds the live working set (higher survival, less variance). SGLANG_KV_PARK_SESSION_KEYED=1.
SGLANG_KV_PARK_SESSION_KEYED=${SGLANG_KV_PARK_SESSION_KEYED:-0}
SGLANG_KV_PARK_SLAB_TOKENS=${SGLANG_KV_PARK_SLAB_TOKENS:-6000}   # session-keyed slab size
# hicache args (prefill). PARK_NO_HICACHE=1 drops them so radix+park runs at host RAM 0
# (the clean "park as a host-RAM-free alternative to hicache" arm).
HICACHE_ARG="--enable-hierarchical-cache --hicache-storage-backend file --hicache-ratio ${HICACHE_RATIO} --hicache-write-policy ${HICACHE_WRITE_POLICY}"
[ "${PARK_NO_HICACHE:-0}" = "1" ] && HICACHE_ARG=""
# DISABLE_RADIX_CACHE=1 -> pure "recompute" baseline: no prefix reuse at all (radix off,
# so hicache -- which layers on top of radix -- is force-dropped too). Every turn
# re-prefills its full context from scratch. Motivation figure: recompute vs prefix-cache.
RADIX_ARG=""
if [ "${DISABLE_RADIX_CACHE:-0}" = "1" ]; then
  RADIX_ARG="--disable-radix-cache"
  HICACHE_ARG=""
fi
if [ "${IDLE_KV_PARKING:-0}" = "1" ]; then
  export SGLANG_KV_PARK_EPOCH="$(date +%s)"
  rm -rf /dev/shm/sglang_kv_parking 2>/dev/null || true
  PARK_ARG="--disaggregation-enable-idle-kv-parking"
  MEMFRAC_P="--mem-fraction-static ${PARK_MEM_FRACTION}"
  CVD_P0="0,1,2,3"; BASE_P0="--base-gpu-id 0"
  CVD_P1="0,1,2,3"; BASE_P1="--base-gpu-id 1"
  CVD_D0="0,1,2,3"; BASE_D0="--base-gpu-id 2"
  CVD_D1="0,1,2,3"; BASE_D1="--base-gpu-id 3"
  _PENV="SGLANG_KV_PARK_POOL_TOKENS=${PARK_POOL_TOKENS} SGLANG_KV_PARK_GEN=${SGLANG_KV_PARK_GEN} SGLANG_KV_PARK_PRESSURE_AWARE=${SGLANG_KV_PARK_PRESSURE_AWARE} SGLANG_KV_PARK_SESSION_KEYED=${SGLANG_KV_PARK_SESSION_KEYED} SGLANG_KV_PARK_SLAB_TOKENS=${SGLANG_KV_PARK_SLAB_TOKENS}"
  ENV_P0="SGLANG_KV_PARK_GPUS=0 ${_PENV}"
  ENV_P1="SGLANG_KV_PARK_GPUS=1 ${_PENV}"
  ENV_D0=""; ENV_D1=""
else
  PARK_ARG=""; MEMFRAC_P=""
  CVD_P0=0; BASE_P0=""; CVD_P1=1; BASE_P1=""; CVD_D0=2; BASE_D0=""; CVD_D1=3; BASE_D1=""
  ENV_P0=""; ENV_P1=""; ENV_D0=""; ENV_D1=""
fi
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/sglang"
mkdir -p "$LOG_DIR"

HICACHE_DIR="/tmp/hicache"

echo "Model     : $(basename $MODEL_PATH)  quantization: ${QUANTIZATION:-none}  parser: $TOOL_CALL_PARSER"
echo "hicache-ratio = $HICACHE_RATIO"
echo "hicache-write-policy = $HICACHE_WRITE_POLICY"
echo ""
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
python -m mooncake.http_metadata_server > "$PROJECT_DIR/logs/mooncake.log" 2>&1 &
MOONCAKE_PID=$!
echo "  PID: $MOONCAKE_PID"
sleep 2

export MOONCAKE_MASTER_SERVER=127.0.0.1:8080

echo "[2/6] Starting Prefill server 1 (GPU 0, port 30000, bootstrap 8998)  park=${IDLE_KV_PARKING:-0}..."
env CUDA_VISIBLE_DEVICES=${CVD_P0} ${ENV_P0} python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30000 ${BASE_P0} \
  --enable-metrics \
  ${P_MTT} ${PARK_ARG} ${MEMFRAC_P} ${RADIX_ARG} \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  ${HICACHE_ARG} \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8998 \
  > "$LOG_DIR/p1.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo "[3/6] Starting Prefill server 2 (GPU 1, port 30001, bootstrap 8999)..."
env CUDA_VISIBLE_DEVICES=${CVD_P1} ${ENV_P1} python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30001 ${BASE_P1} \
  --enable-metrics \
  ${P_MTT} ${PARK_ARG} ${MEMFRAC_P} ${RADIX_ARG} \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  ${HICACHE_ARG} \
  --disaggregation-mode prefill --disaggregation-transfer-backend mooncake \
  --disaggregation-bootstrap-port 8999 \
  > "$LOG_DIR/p2.log" 2>&1 &
echo "  PID: $!"
sleep 3

# hicache 파일 경로를 SGLang이 어떤 dir을 쓰는지 확인 (로그에서 grep)
# 실제 경로가 다른 경우: HICACHE_DIRS="p1:/actual/path,p2:/actual/path" 로 오버라이드
sleep 3

echo "[4/6] Starting Decode server 1 (GPU 2, port 30002)..."
env CUDA_VISIBLE_DEVICES=${CVD_D0} ${ENV_D0} python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30002 ${BASE_D0} \
  --enable-metrics \
  ${D_MTT} ${PARK_ARG} ${RADIX_ARG} \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser $TOOL_CALL_PARSER \
  > "$LOG_DIR/d1.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo "[5/6] Starting Decode server 2 (GPU 3, port 30003)..."
env CUDA_VISIBLE_DEVICES=${CVD_D1} ${ENV_D1} python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30003 ${BASE_D1} \
  --enable-metrics \
  ${D_MTT} ${PARK_ARG} ${RADIX_ARG} \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser $TOOL_CALL_PARSER \
  > "$LOG_DIR/d2.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo ""
echo "Waiting for all servers to be ready..."
for port in 30000 30001 30002 30003; do
  echo -n "  Waiting for port $port..."
  while ! grep -q "ready to roll" "$LOG_DIR/$(case $port in 30000) echo p1;; 30001) echo p2;; 30002) echo d1;; 30003) echo d2;; esac).log" 2>/dev/null; do
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
LOG_DIR="$PROJECT_DIR/logs/sglang" \
python3 "$PROJECT_DIR/benchmark/sglang_hicache_exporter.py" \
  > logs/sglang/hicache_exporter.log 2>&1 &
EXPORTER_PID=$!
echo "  Exporter PID: $EXPORTER_PID"
sleep 2

if curl -s "http://localhost:9199/metrics" | grep -q "sglang_scrape_ok"; then
  echo "  Exporter: READY (http://localhost:9199/metrics)"
else
  echo "  Exporter: not yet responding (check logs/hicache_exporter.log)"
fi

echo ""
echo "All done. Logs at logs/sglang/*.log"
echo "Router at http://127.0.0.1:8000"
echo ""
echo "To check hicache metrics:"
echo "  curl http://localhost:9199/metrics"
echo "  curl http://localhost:30000/get_server_info | python3 -m json.tool"