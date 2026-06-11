#!/bin/bash
# SGLang 1P1D — KV Break-even Map Experiment (research.md §5.A)
# ==============================================================
# (tool duration × context length) → 최적 tier break-even 맵 측정용
# 단일 P+D 구성. 모든 요청이 같은 P 노드에서 처리되어야
# anchor eviction이 deterministic (start_1P_1D_kv_tier.sh와 동일 topology).
#
# 구성:
#   P1 (GPU 0, port 30000, bootstrap 8998) — hicache 활성화 (file backend → /tmp/hicache)
#   D1 (GPU 2, port 30002)
#   Router (port 8001) — 2P2D router(8000)와 충돌 없음
#
# 실행 순서:
#   1) bash scripts/sglang/start_1P_1D_breakeven.sh     # 서버 시작
#   2) SERVER_URL=http://127.0.0.1:8001 \
#        python benchmark/sglang_kv_breakeven_map.py    # 벤치마크
#
# 참고: /tmp 가 실제 디스크(ext4)인지 확인 — tmpfs면 L3(SSD) 측정 무의미
#   df -T /tmp
set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Qwen3-14B"}
QUANTIZATION=${QUANTIZATION:-""}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"hermes"}
HICACHE_RATIO=${HICACHE_RATIO:-"1.2"}

# ─── HiCache 정책 (1차 측정에서 L2/L3 무이득 → 기본값 함정 배제용 명시) ──────
# write_through        : 계산 즉시 host(DRAM)+SSD 기록 (selective는 재접근 prefix만 기록)
# wait_complete        : storage 로드를 끝까지 기다림 (best_effort는 포기 후 재계산 fallback
#                        → L2/L3 first-miss ≈ cold 패턴의 유력 용의자)
# HICACHE_IO_BACKEND   : 빈 값이면 서버 기본값. 'direct' 지원 여부는 버전에 따라 다름
# ※ 플래그 이름/지원 여부 확인: python3 -m sglang.launch_server --help | grep hicache
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-"write_through"}
HICACHE_PREFETCH_POLICY=${HICACHE_PREFETCH_POLICY:-"wait_complete"}
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-""}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/sglang_1p1d"
mkdir -p "$LOG_DIR"

echo "=================================================="
echo "SGLang 1P1D KV Break-even Map Server"
echo "=================================================="
echo "Model    : $(basename $MODEL_PATH)"
echo "Log dir  : $LOG_DIR"
echo "Router   : http://127.0.0.1:8001"
echo "HiCache  : write=$HICACHE_WRITE_POLICY  prefetch=$HICACHE_PREFETCH_POLICY"
echo "           io=${HICACHE_IO_BACKEND:-(server default)}  ratio=$HICACHE_RATIO"
echo ""

# ─── /tmp 디스크 타입 확인 ─────────────────────────────────────────────────
if df -T /tmp | grep -q tmpfs; then
  echo "⚠  WARNING: /tmp is tmpfs (RAM) — L3 'SSD' 측정이 무의미합니다."
  echo "   HICACHE 경로를 실제 디스크로 바꾸세요."
  echo ""
fi

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
  --hicache-write-policy $HICACHE_WRITE_POLICY \
  --hicache-storage-prefetch-policy $HICACHE_PREFETCH_POLICY \
  ${HICACHE_IO_BACKEND:+--hicache-io-backend $HICACHE_IO_BACKEND} \
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
echo "    python benchmark/sglang_kv_breakeven_map.py"
echo ""
echo "예상 소요: 길이당 3-5분 × 5 lengths ≈ 20-30분"
echo "결과: results/sglang_hicache/$(basename $MODEL_PATH)/kv_breakeven_map.{json,png}"
echo ""
echo "Logs: $LOG_DIR/"
echo "  tail -f $LOG_DIR/p1.log"
echo "=================================================="
