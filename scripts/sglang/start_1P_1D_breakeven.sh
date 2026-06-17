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

# ─── GPU 배치 / 전송 백엔드 (research.md §2.8 — 전송 세금의 정체) ────────────
# server17 토폴로지 (nvidia-smi topo -m):
#   GPU0↔GPU1 NV4, GPU2↔GPU3 NV4, 그 외 NODE(PCIe host bridge 경유)
#   → NVLink 경로 실험: P_GPU=0 D_GPU=1  /  PCIe 경로 실험: P_GPU=0 D_GPU=2
# TRANSFER_BACKEND:
#   mooncake : 이 머신은 RDMA NIC 없음 → TCP loopback fallback (~1.2 GB/s 실측,
#              GPU 배치와 무관하게 NVLink/P2P 미사용)
#   nixl     : UCX 기반 — same-node에서 CUDA IPC/P2P (NVLink/PCIe) 사용 가능.
#              사전 설치 필요: pip install nixl
P_GPU=${P_GPU:-"0"}
D_GPU=${D_GPU:-"1"}
TRANSFER_BACKEND=${TRANSFER_BACKEND:-"mooncake"}

# ─── HiCache 정책 ────────────────────────────────────────────────────────────
# write_through        : 계산 즉시 host(DRAM)+SSD 기록 (selective는 재접근 prefix만 기록)
# prefetch=best_effort : 기본값 복귀. ⚠ wait_complete는 PD에서 24–126s 스톨 실측
#                        (research.md §2.8) — 사용 금지
# HICACHE_MEM_LAYOUT   : layer_first(기본)는 I/O 비효율. page_first(kernel io 권장),
#                        page_first_direct(direct io 전용) 시도 가능
# HICACHE_IO_BACKEND   : 빈 값이면 서버 기본값(kernel). direct는 page_first_direct와 조합
# ※ 플래그 이름/지원 여부 확인: python3 -m sglang.launch_server --help | grep hicache
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-"write_through"}
HICACHE_PREFETCH_POLICY=${HICACHE_PREFETCH_POLICY:-"best_effort"}
HICACHE_MEM_LAYOUT=${HICACHE_MEM_LAYOUT:-""}
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-""}

# ─── Decode 노드 KV offload (sglang 0.5.9, research.md §2.9) ────────────────
# 1: decode 노드가 생성한 KV를 비동기로 storage에 기록 → P가 multi-turn에서 재사용
#    = D→하위 tier 경로.
# ⚠ decode 서버에 --hicache-storage-backend가 반드시 함께 필요:
#    DecodeKVCacheOffloadManager가 HiCacheController에 server_args.hicache_storage_backend를
#    넘기는데, 미설정이면 storage thread가 안 떠서 ack_backup_queue AttributeError로 즉사
#    (2026-06-11 실측). 같은 호스트라 file 백엔드(/tmp/hicache)를 P와 공유 → P가 D의
#    offload KV를 읽을 수 있음. 멀티노드로 가면 양쪽 다 mooncake store로 전환.
# ⚠ 동작해도 multi-turn 부하에서 CUDA illegal memory access 보고됨 (sglang #11016,
#    transfer_kv_all_layer_lf_pf 커널). 재현 시 D_HICACHE_MEM_LAYOUT으로 layout을 바꿔
#    다른 전송 커널 경로 시도.
D_OFFLOAD_KVCACHE=${D_OFFLOAD_KVCACHE:-"0"}
D_HICACHE_MEM_LAYOUT=${D_HICACHE_MEM_LAYOUT:-""}

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
echo "GPUs     : P=GPU$P_GPU  D=GPU$D_GPU  (0-1/2-3 = NVLink NV4, cross = PCIe)"
echo "Transfer : $TRANSFER_BACKEND"
echo "HiCache  : write=$HICACHE_WRITE_POLICY  prefetch=$HICACHE_PREFETCH_POLICY"
echo "           layout=${HICACHE_MEM_LAYOUT:-(server default)}  io=${HICACHE_IO_BACKEND:-(server default)}  ratio=$HICACHE_RATIO"
echo "D-offload: ${D_OFFLOAD_KVCACHE} (1=decode KV→storage, sglang #11016 주의)"
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

# ─── Transfer backend 준비 ────────────────────────────────────────────────
if [ "$TRANSFER_BACKEND" = "nixl" ] && ! python3 -c "import nixl" 2>/dev/null; then
  echo "✗ nixl python package 미설치 — 'pip install nixl' 후 재실행하세요."
  exit 1
fi

if [ "$TRANSFER_BACKEND" = "mooncake" ]; then
  echo "[1/5] Starting Mooncake metadata server..."
  python -m mooncake.http_metadata_server > "$LOG_DIR/mooncake.log" 2>&1 &
  echo "  PID: $!  log: $LOG_DIR/mooncake.log"
  sleep 2
  export MOONCAKE_MASTER_SERVER=127.0.0.1:8080
else
  echo "[1/5] Skipping Mooncake metadata server (backend=$TRANSFER_BACKEND)"
fi

# ─── Prefill server P1 (GPU 0) ────────────────────────────────────────────
echo "[2/5] Starting Prefill P1 (GPU $P_GPU, port 30000)..."
CUDA_VISIBLE_DEVICES=$P_GPU python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port 30000 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --enable-metrics \
  --enable-cache-report \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-ratio $HICACHE_RATIO \
  --hicache-write-policy $HICACHE_WRITE_POLICY \
  --hicache-storage-prefetch-policy $HICACHE_PREFETCH_POLICY \
  ${HICACHE_MEM_LAYOUT:+--hicache-mem-layout $HICACHE_MEM_LAYOUT} \
  ${HICACHE_IO_BACKEND:+--hicache-io-backend $HICACHE_IO_BACKEND} \
  --disaggregation-mode prefill \
  --disaggregation-transfer-backend $TRANSFER_BACKEND \
  --disaggregation-bootstrap-port 8998 \
  > "$LOG_DIR/p1.log" 2>&1 &
echo "  PID: $!  log: $LOG_DIR/p1.log"
sleep 3

# ─── Decode server D1 (GPU 2) ─────────────────────────────────────────────
echo "[3/5] Starting Decode D1 (GPU $D_GPU, port 30002)..."
CUDA_VISIBLE_DEVICES=$D_GPU python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port 30002 \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --enable-metrics \
  --enable-cache-report \
  --disaggregation-mode decode \
  --disaggregation-transfer-backend $TRANSFER_BACKEND \
  $([ "$D_OFFLOAD_KVCACHE" = "1" ] && echo "--disaggregation-decode-enable-offload-kvcache \
      --hicache-ratio $HICACHE_RATIO \
      --hicache-storage-backend file \
      ${D_HICACHE_MEM_LAYOUT:+--hicache-mem-layout $D_HICACHE_MEM_LAYOUT}") \
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
