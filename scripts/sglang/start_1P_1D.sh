#!/bin/bash
# Phase 0 baseline: 1P1D SGLang 서버 시작 (캐시 계층별 비교용)
#
# 목적: 기존 hierarchical cache 경로가 multi-turn 워크로드에서
#       "재계산(recompute) vs fetch"를 어디까지 커버하는지 측정하기 위한
#       baseline 서버를 캐시 모드별로 띄운다.
#
# 사용법:
#   CACHE_MODE=hicache_file ./scripts/sglang/start_1P_1D.sh
#
# CACHE_MODE (기본: hicache_file):
#   none                        prefill radix cache 비활성 → 매 턴 full re-prefill (recompute 바닥값)
#   radix                       GPU radix cache만 (L1)      → GPU에 남아있으면 prefix hit
#   hicache_host                L1+L2 (GPU+CPU DRAM)        → storage backend 없이 host offload (host-only Tier)
#   hicache_file                L1+L2+L3 (file storage)     → 현재 실험 기본 구성
#   hicache_file_decode_offload 위 + decode offload 활성   → decode KV까지 host/storage로 offload
#
# 주의:
#   - --enable-metrics 필수 (phase0_metrics_scraper.py가 /metrics를 긁음)
#   - decode offload는 server_args.py:4346 제약상 storage backend가 반드시 필요
#   - RAM 125GB 제약: HICACHE_RATIO를 너무 크게 잡지 말 것

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
CACHE_MODE=${CACHE_MODE:-"hicache_file"}
HICACHE_RATIO=${HICACHE_RATIO:-2.0}   # host pool / device pool 비율 (>1). 1P1D는 2P2D보다 RAM 여유 있음
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
XFER=${XFER:-mooncake}                # disaggregation transfer backend

mkdir -p logs

echo "================================================================"
echo " Phase 0 baseline 1P1D  |  CACHE_MODE=${CACHE_MODE}  HICACHE_RATIO=${HICACHE_RATIO}"
echo "================================================================"

# --- 캐시 모드별 서버 인자 구성 ---------------------------------------------
PREFILL_CACHE_ARGS=""
DECODE_CACHE_ARGS=""
case "$CACHE_MODE" in
  none)
    PREFILL_CACHE_ARGS="--disable-radix-cache"
    ;;
  radix)
    PREFILL_CACHE_ARGS=""   # 기본 radix cache (GPU L1)
    ;;
  hicache_host)
    PREFILL_CACHE_ARGS="--enable-hierarchical-cache --hicache-ratio ${HICACHE_RATIO}"
    ;;
  hicache_file)
    PREFILL_CACHE_ARGS="--enable-hierarchical-cache --hicache-ratio ${HICACHE_RATIO} --hicache-storage-backend file"
    ;;
  hicache_file_decode_offload)
    PREFILL_CACHE_ARGS="--enable-hierarchical-cache --hicache-ratio ${HICACHE_RATIO} --hicache-storage-backend file"
    DECODE_CACHE_ARGS="--enable-hierarchical-cache --hicache-ratio ${HICACHE_RATIO} --hicache-storage-backend file --disaggregation-decode-enable-offload-kvcache"
    ;;
  *)
    echo "ERROR: unknown CACHE_MODE=${CACHE_MODE}" >&2
    exit 1
    ;;
esac

echo "[0/5] Stopping existing processes..."
pkill -9 -f "sglang.launch_server" 2>/dev/null || true
pkill -9 -f "mooncake.http_metadata_server" 2>/dev/null || true
pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
for port in $BOOTSTRAP_PORT $PREFILL_PORT $DECODE_PORT $ROUTER_PORT 8080; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
sleep 5

echo "[1/5] Starting Mooncake metadata server..."
python -m mooncake.http_metadata_server > logs/mooncake.log 2>&1 &
sleep 2
export MOONCAKE_MASTER_SERVER=127.0.0.1:8080

echo "[2/5] Prefill (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})  args: ${PREFILL_CACHE_ARGS}"
CUDA_VISIBLE_DEVICES=${PREFILL_GPU} python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port ${PREFILL_PORT} \
  --enable-metrics \
  ${PREFILL_CACHE_ARGS} \
  --disaggregation-mode prefill --disaggregation-transfer-backend ${XFER} \
  --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
  > logs/p1.log 2>&1 &
P1_PID=$!
sleep 3

echo "[3/5] Decode  (GPU ${DECODE_GPU}, port ${DECODE_PORT})  args: ${DECODE_CACHE_ARGS}"
CUDA_VISIBLE_DEVICES=${DECODE_GPU} python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port ${DECODE_PORT} \
  --enable-metrics \
  ${DECODE_CACHE_ARGS} \
  --disaggregation-mode decode --disaggregation-transfer-backend ${XFER} \
  > logs/d1.log 2>&1 &
D1_PID=$!
sleep 3

echo "[4/5] Waiting for servers to be ready..."
# 실패 판정은 로그 문자열이 아니라 "프로세스 생존 여부 + 타임아웃"으로 한다.
# (sglang은 기동 중 "Ignore import error when loading ..." 같은 양성 경고를 찍으므로
#  'error' 문자열 매칭은 오탐을 낸다. RDMA 미탐지 → mooncake TCP 폴백도 정상.)
READY_TIMEOUT=${READY_TIMEOUT:-900}
wait_ready() {
  local name=$1 pid=$2 elapsed=0
  echo -n "  ${name}..."
  while ! grep -q "ready to roll" logs/${name}.log 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo " FAILED (process exited) — see logs/${name}.log"; tail -30 logs/${name}.log; return 1
    fi
    if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
      echo " TIMEOUT after ${READY_TIMEOUT}s — see logs/${name}.log"; tail -30 logs/${name}.log; return 1
    fi
    sleep 3; elapsed=$((elapsed + 3)); echo -n "."
  done
  echo " OK (${elapsed}s)"
}
wait_ready p1 "$P1_PID"
wait_ready d1 "$D1_PID"

echo "[5/5] Starting Router (port ${ROUTER_PORT})..."
python -m sglang_router.launch_router \
  --pd-disaggregation \
  --prefill http://127.0.0.1:${PREFILL_PORT} ${BOOTSTRAP_PORT} \
  --decode http://127.0.0.1:${DECODE_PORT} \
  --host 0.0.0.0 --port ${ROUTER_PORT} \
  > logs/router.log 2>&1 &
sleep 3

echo ""
echo "Ready. CACHE_MODE=${CACHE_MODE}"
echo "  Router:  http://127.0.0.1:${ROUTER_PORT}"
echo "  Prefill /metrics: http://127.0.0.1:${PREFILL_PORT}/metrics"
echo "  Decode  /metrics: http://127.0.0.1:${DECODE_PORT}/metrics"
