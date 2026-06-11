#!/bin/bash
# SGLang single-server hicache 실험
# ====================================
# PD disaggregation 없이 단일 GPU 서버에서 hicache 3-tier 동작을 검증한다.
#
# ── disagg vs non-disagg ────────────────────────────────────────────────────
#   disagg:     P서버가 매 turn full recompute + D서버 transfer
#               → radix prefix cache 미동작 → hicache L2/L3 read 발생 안 함
#   non-disagg: 단일 서버, radix prefix cache 정상 동작
#               → 이전 turn KV를 GPU에 캐시 → GPU 꽉 차면 DRAM/SSD으로 eviction
#               → 다음 turn: L2 DRAM 또는 L3 SSD에서 KV load → TTFT 변화 관찰 가능
#
# ── KV pool 크기 (Qwen3-14B @ A6000 49GB) ──────────────────────────────────
#   model weights: ~28GB (BF16)  CUDA/graph: ~3GB  available: ~18GB
#
#   MEM_FRACTION  KV pool   token capacity   비고
#   0.88 (기본)   15.8 GB   ~97k tokens      eviction 유발 어려움
#   0.70 (실험)   12.6 GB   ~77k tokens      CONCURRENCY=8 × 10k tok → eviction
#   0.50          9.0 GB    ~55k tokens      더 공격적, CONCURRENCY 낮아도 eviction
#
#   ※ 실제 capacity는 서버 시작 후 /get_server_info 로 확인 (위 수치는 추정값)
#
# ── hicache 3-tier ──────────────────────────────────────────────────────────
#   L1 GPU HBM  : KV pool (mem_fraction_static 비율)
#   L2 CPU DRAM : KV pool × HICACHE_RATIO  (ratio=1.2 → ~15GB DRAM)
#   L3 SSD file : /tmp/hicache (무제한, 디스크 여유분까지)
#
#   write_through: KV 계산 즉시 L1+L2+L3 동시 write
#                  → L1 LRU eviction 발생 시 L2/L3에서 read → TTFT 증가
#   write_through_selective: 긴 prefix (system prompt 등)만 선택 write (쓰기 부하 절감)
#   write_back: L1 eviction 시에만 L2/L3 write (non-disagg에서만 안전, 권장)
#
# ── 동시성 vs eviction ───────────────────────────────────────────────────────
#   CONCURRENCY=4 × avg 10k tok = 40k < 77k → eviction 없음
#   CONCURRENCY=8 × avg 10k tok = 80k > 77k → LRU eviction 발생
#   TOOL_DELAY_SEC=10 + FLOOD_N=30 → delay 동안 flood으로 active KV 강제 eviction
#
# 사용법:
#   bash scripts/sglang/start_single_hicache.sh                        # 기본값
#   MEM_FRACTION=0.5 bash scripts/sglang/start_single_hicache.sh       # KV pool 더 작게
#   GPU=1 PORT=30001 bash scripts/sglang/start_single_hicache.sh       # GPU/port 지정
#   HICACHE_WRITE_POLICY=write_back bash scripts/sglang/start_single_hicache.sh
#
# 벤치마크 실행:
#   SGLANG_URL=http://127.0.0.1:30000/v1/chat/completions \
#   SGLANG_INSTANCES=server:30000 \
#   CONFIG=sglang_single_hicache \
#   CONCURRENCY=8 TOOL_DELAY_SEC=10 FLOOD_N=30 FLOOD_TOKENS=3000 \
#     python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
#
# 캐시 동작 빠른 확인:
#   python benchmark/test_cache_ttft.py   # t2/t1 << 1.0 이면 L1 cache hit 정상
#
# 모니터링:
#   watch -n 2 'grep -i "cache hit" logs/sglang_single/server.log | tail -3'
#   watch -n 2 'du -sh /tmp/hicache 2>/dev/null'

set -e

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Qwen3-14B"}
QUANTIZATION=${QUANTIZATION:-""}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"hermes"}
MEM_FRACTION=${MEM_FRACTION:-"0.70"}
HICACHE_RATIO=${HICACHE_RATIO:-"1.2"}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-"write_through"}
# ─── L2(DRAM) hit 살리기 노브 (research.md §2.8: L2 ≈ cold = hit 무효 관측) ──
# HICACHE_RATIO=3.0      : host pool을 GPU pool의 3배로 — ratio 1.2는 host가 GPU보다
#                          겨우 20% 커서 GPU flood가 host까지 같이 밀어내는 구조
# HICACHE_MEM_LAYOUT     : layer_first(기본)는 I/O 비효율. page_first(kernel io),
#                          page_first_direct(direct io 전용)
# HICACHE_IO_BACKEND     : kernel(기본) | direct (direct는 page_first_direct와 조합)
# HICACHE_PREFETCH_POLICY: best_effort(기본) | wait_complete (PD에서는 스톨 — 단일은 시험 가능)
# 재시도 권장 조합:
#   HICACHE_RATIO=3.0 HICACHE_MEM_LAYOUT=page_first bash scripts/sglang/start_single_hicache.sh
#   HICACHE_RATIO=3.0 HICACHE_MEM_LAYOUT=page_first_direct HICACHE_IO_BACKEND=direct bash ...
HICACHE_MEM_LAYOUT=${HICACHE_MEM_LAYOUT:-""}
HICACHE_IO_BACKEND=${HICACHE_IO_BACKEND:-""}
HICACHE_PREFETCH_POLICY=${HICACHE_PREFETCH_POLICY:-""}
PORT=${PORT:-"30000"}
GPU=${GPU:-"0"}

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(dirname "$(dirname "$SCRIPT_DIR")")"
LOG_DIR="$PROJECT_DIR/logs/sglang_single"
mkdir -p "$LOG_DIR"

HICACHE_DIR="/tmp/hicache"

# KV pool 추정치 출력 (실제값은 서버 시작 후 확인)
KV_EST=$(python3 -c "
avail = 18.0
kv = $MEM_FRACTION * avail
tok = int(kv * 1024**3 / 163840)
dram = kv * $HICACHE_RATIO
print(f'KV ≈ {kv:.1f}GB ≈ {tok:,} tokens  |  L2 DRAM ≈ {dram:.1f}GB')
")

echo "=================================================="
echo " SGLang single-server hicache 실험"
echo "=================================================="
echo " Model       : $(basename $MODEL_PATH)"
echo " GPU         : $GPU  →  port $PORT"
echo " mem_fraction: $MEM_FRACTION  →  $KV_EST"
echo " hicache     : ratio=$HICACHE_RATIO  policy=$HICACHE_WRITE_POLICY"
echo "               layout=${HICACHE_MEM_LAYOUT:-(default)}  io=${HICACHE_IO_BACKEND:-(default)}  prefetch=${HICACHE_PREFETCH_POLICY:-(default)}"
echo " hicache dir : $HICACHE_DIR"
echo " Log dir     : $LOG_DIR"
echo "=================================================="
echo ""

# ── 기존 프로세스 정리 ────────────────────────────────────────────────────
echo "[0/3] Stopping existing SGLang processes..."
pkill -9 -f "sglang.launch_server"        2>/dev/null || true
pkill -9 -f "sglang_router.launch_router" 2>/dev/null || true
pkill -9 -f "sglang_hicache_exporter"     2>/dev/null || true
for port in "$PORT" 9199; do
  fuser -k "${port}/tcp" 2>/dev/null || true
done
sleep 3

if [ -d "$HICACHE_DIR" ]; then
  SIZE=$(du -sh "$HICACHE_DIR" 2>/dev/null | cut -f1)
  echo "  Cleaning $HICACHE_DIR ($SIZE) ..."
  rm -rf "$HICACHE_DIR"
fi

# ── SGLang server (non-disagg) ────────────────────────────────────────────
echo "[1/3] Starting SGLang server (GPU $GPU, port $PORT)..."
CUDA_VISIBLE_DEVICES=$GPU python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" \
  --tp 1 \
  --port "$PORT" \
  ${QUANTIZATION:+--quantization "$QUANTIZATION"} \
  --mem-fraction-static "$MEM_FRACTION" \
  --enable-hierarchical-cache \
  --hicache-storage-backend file \
  --hicache-ratio "$HICACHE_RATIO" \
  --hicache-write-policy "$HICACHE_WRITE_POLICY" \
  ${HICACHE_MEM_LAYOUT:+--hicache-mem-layout "$HICACHE_MEM_LAYOUT"} \
  ${HICACHE_IO_BACKEND:+--hicache-io-backend "$HICACHE_IO_BACKEND"} \
  ${HICACHE_PREFETCH_POLICY:+--hicache-storage-prefetch-policy "$HICACHE_PREFETCH_POLICY"} \
  --tool-call-parser "$TOOL_CALL_PARSER" \
  > "$LOG_DIR/server.log" 2>&1 &
SERVER_PID=$!
echo "  PID: $SERVER_PID  log: $LOG_DIR/server.log"

# ── ready 대기 ────────────────────────────────────────────────────────────
echo ""
echo "[2/3] Waiting for server to be ready..."
echo -n "  port $PORT..."
while ! grep -q "ready to roll" "$LOG_DIR/server.log" 2>/dev/null; do
  sleep 3
  echo -n "."
  if grep -qE "Traceback|RuntimeError|CUDA out of memory|OOM|Killed" \
      "$LOG_DIR/server.log" 2>/dev/null; then
    echo " ERROR"
    echo "  Last log lines:"
    tail -10 "$LOG_DIR/server.log"
    exit 1
  fi
done
echo " OK"

# 실제 KV pool 크기 확인
ACTUAL=$(curl -s "http://localhost:$PORT/get_server_info" 2>/dev/null \
  | python3 -c "
import sys, json
try:
    d = json.load(sys.stdin)
    # internal_states 또는 최상위에서 memory_usage 찾기
    mem = {}
    for st in d.get('internal_states', []):
        if isinstance(st, dict) and 'memory_usage' in st:
            mem = st['memory_usage']
            break
    if not mem:
        mem = d.get('memory_usage', {})
    cap = mem.get('token_capacity', '?')
    kv  = mem.get('kvcache', '?')
    fmt_cap = f'{cap:,}' if isinstance(cap, int) else str(cap)
    print(f'kvcache={kv}GB  token_capacity={fmt_cap} tokens')
except Exception as e:
    print(f'(parse error: {e})')
" 2>/dev/null || echo "(unavailable)")
echo "  실제 KV pool: $ACTUAL"

# cache hit rate 메트릭 노출 여부 확인
METRICS_LINES=$(curl -s "http://localhost:$PORT/metrics" 2>/dev/null | wc -l)
if [ "$METRICS_LINES" -gt 0 ]; then
  echo "  /metrics: $METRICS_LINES lines"
  curl -s "http://localhost:$PORT/metrics" 2>/dev/null \
    | grep -E "cache_hit|num_running|num_used|token_usage" || echo "  (no matching metrics)"
else
  echo "  /metrics: empty"
fi

# ── hicache exporter (Prometheus, port 9199) ─────────────────────────────
echo ""
echo "[3/3] Starting hicache exporter (port 9199)..."
SGLANG_INSTANCES="server:$PORT" \
HICACHE_DIRS="server:$HICACHE_DIR" \
EXPORTER_PORT=9199 \
LOG_DIR="$LOG_DIR" \
python3 "$PROJECT_DIR/benchmark/sglang_hicache_exporter.py" \
  > "$LOG_DIR/hicache_exporter.log" 2>&1 &
sleep 2

if curl -s "http://localhost:9199/metrics" 2>/dev/null | grep -q "sglang_scrape_ok"; then
  echo "  Exporter: READY  →  http://localhost:9199/metrics"
  curl -s "http://localhost:9199/metrics" 2>/dev/null \
    | grep -E "kvcache_allocated|token_capacity" || true
else
  echo "  Exporter: not yet responding (check $LOG_DIR/hicache_exporter.log)"
fi

echo ""
echo "=================================================="
echo " Ready."
echo ""
echo " 서버    : http://127.0.0.1:$PORT"
echo " Exporter: http://localhost:9199/metrics"
echo " Logs    : $LOG_DIR/"
echo ""
echo " 1. 캐시 hit/miss 빠른 확인 (prefix cache 동작 검증):"
echo "    python benchmark/test_cache_ttft.py"
echo "    → t2/t1 << 1.0 이면 L1 GPU cache hit 정상"
echo "    → t2/t1 ≈ 1.0  이면 cache 미동작"
echo ""
echo " 2. 전체 벤치마크 (KV tier eviction 실험):"
echo "    SGLANG_URL=http://127.0.0.1:$PORT/v1/chat/completions \\"
echo "    SGLANG_INSTANCES=server:$PORT \\"
echo "    CONFIG=sglang_single_hicache_c8_d10s \\"
echo "    CONCURRENCY=8 TOOL_DELAY_SEC=10 FLOOD_N=30 FLOOD_TOKENS=3000 \\"
echo "      python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py"
echo ""
echo " 3. 실시간 모니터링:"
echo "    watch -n 2 'grep -i \"cache hit\" $LOG_DIR/server.log | tail -3'"
echo "    watch -n 2 'du -sh $HICACHE_DIR 2>/dev/null'"
echo "    curl -s http://localhost:9199/metrics | grep -E 'token_usage|running|l3_ssd'"
echo "=================================================="
