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

# SLURM 하위에서는 scripts/slurm/env.sh 가 이미 conda 를 activate 하고, 포트/GPU/로그
# 디렉터리를 job 단위로 잡아 둔 상태로 이 스크립트를 부른다. 그 위에 다시 activate 하면
# 클러스터에 따라 PATH 가 꼬이므로 건너뛴다.
if [ -z "${SLURM_JOB_ID:-}" ] || [ -z "${CONDA_PREFIX:-}" ]; then
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate sglang
fi
# 수정한 sglang 소스 트리를 쓰게 만들고, 설치본을 import 하고 있으면 loud 하게 죽는다.
# start_2P_2D.sh 와 start_single.sh 에는 처음부터 있었는데 이 파일에는 없었다. 그래서
# 1P1D 로 도는 동안에는 idle-KV-parking 패치가 적용되지 않은 채로 실험이 돌 수 있었고,
# 증상은 "park arm 과 baseline 이 이상하게 비슷하다" 뿐이다 -- 정확히 이 가드가
# 막으려고 존재하는 실패다. A100 은 1P1D 가 기본이므로 여기가 주 경로다.
source "$(dirname "${BASH_SOURCE[0]}")/_use_source.sh"
# 내 job 소속 프로세스만 죽이는 job_pkill (공유 노드 대비). 자세한 이유는 파일 주석 참고.
source "$(dirname "${BASH_SOURCE[0]}")/_job_scope.sh"

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
CACHE_MODE=${CACHE_MODE:-"hicache_file"}
HICACHE_RATIO=${HICACHE_RATIO:-2.0}   # host pool / device pool 비율 (>1). 1P1D는 2P2D보다 RAM 여유 있음
PREFILL_GPU=${PREFILL_GPU:-0}
DECODE_GPU=${DECODE_GPU:-1}
PREFILL_PORT=${PREFILL_PORT:-30000}
DECODE_PORT=${DECODE_PORT:-30001}
BOOTSTRAP_PORT=${BOOTSTRAP_PORT:-8998}
ROUTER_PORT=${ROUTER_PORT:-8000}
ROUTER_PROM_PORT=${ROUTER_PROM_PORT:-29000}   # sgl-router Prometheus exporter (기본 29000)
# Mooncake 메타데이터 서버 포트. 공유 노드에서 8080 고정은 곧 충돌이므로 job 마다 다른
# 값을 받을 수 있어야 한다 (scripts/slurm/env.sh 가 넣어준다). 기본값은 종전과 동일.
MOONCAKE_PORT=${MOONCAKE_PORT:-8080}
# 라우터 bind 주소. 전용 머신에서는 종전대로 0.0.0.0, 공유 클러스터에서는 루프백만.
ROUTER_HOST=${ROUTER_HOST:-0.0.0.0}
# 로그 위치. run_why_faster.sh 의 digest 수집이 LOG_DIR 을 먼저 보므로 여기와 같아야 한다.
LOG_DIR=${LOG_DIR:-logs}
XFER=${XFER:-mooncake}                # disaggregation transfer backend
# --disaggregation-ib-device. mooncake 는 이 값을 그대로 TransferEngine 에 넘긴다.
# RDMA 가 있는 노드에서 GPUDirect 가 없으면 KV pool 등록이 EFAULT 로 실패하는데,
# 쓸 수 있는 HCA 가 0개가 되면 mooncake 가 TCP 로 폴백한다. 어느 값이 그렇게 만드는지는
# scripts/slurm/probe_mooncake.py --sweep 이 찾아준다. 빈 값이면 종전대로 자동 탐색.
IB_DEVICE=${IB_DEVICE:-}
IB_ARG=""
[ -n "$IB_DEVICE" ] && IB_ARG="--disaggregation-ib-device ${IB_DEVICE}"
# 압박(pressure) knob: prefill KV pool을 작게 잡아 radix eviction을 유발한다.
# (빈 값이면 미적용 = 기본 대용량 pool). 값 예: 30000, 20000
PREFILL_MAX_TOTAL_TOKENS=${PREFILL_MAX_TOTAL_TOKENS:-}
DECODE_MAX_TOTAL_TOKENS=${DECODE_MAX_TOTAL_TOKENS:-}

mkdir -p "$LOG_DIR"

echo "================================================================"
echo " Phase 0 baseline 1P1D  |  CACHE_MODE=${CACHE_MODE}  HICACHE_RATIO=${HICACHE_RATIO}"
echo " P=gpu${PREFILL_GPU}:${PREFILL_PORT}  D=gpu${DECODE_GPU}:${DECODE_PORT}" \
     " router=${ROUTER_PORT}  logs=${LOG_DIR}"
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
  PREFILL_CACHE_ARGS="--enable-hierarchical-cache \
    --hicache-ratio ${HICACHE_RATIO} \
    --hicache-storage-backend file"

  DECODE_CACHE_ARGS="--hicache-ratio ${HICACHE_RATIO} \
    --hicache-storage-backend file \
    --disaggregation-decode-enable-offload-kvcache"
  ;;
  *)
    echo "ERROR: unknown CACHE_MODE=${CACHE_MODE}" >&2
    exit 1
    ;;
esac

# GPU 배치: 기본은 CUDA_VISIBLE_DEVICES로 격리.
PREFILL_CVD="${PREFILL_GPU}"; DECODE_CVD="${DECODE_GPU}"
PREFILL_GPU_ARG=""; DECODE_GPU_ARG=""
PREFILL_ENV=""

# Idle KV parking (Phase 1). IDLE_KV_PARKING=1이면 P/D 양쪽에 플래그 추가.
if [ "${IDLE_KV_PARKING:-0}" = "1" ]; then
  PREFILL_CACHE_ARGS="${PREFILL_CACHE_ARGS} --disaggregation-enable-idle-kv-parking"
  DECODE_CACHE_ARGS="${DECODE_CACHE_ARGS} --disaggregation-enable-idle-kv-parking"
  # CUDA IPC는 각 프로세스가 상대 GPU를 "볼 수" 있어야 한다. 격리(CUDA_VISIBLE_DEVICES=단일GPU)
  # 대신 두 GPU를 모두 보이게 하고 --base-gpu-id로 배치한다.
  PARK_VISIBLE=${PARK_VISIBLE:-"${PREFILL_GPU},${DECODE_GPU}"}
  # slice 4 / Phase 2 slice 1: 유휴 GPU park 풀.
  #   PARK_GPUS="2,3" (다중, 콤마) 지정 시 여러 유휴 GPU를 후보로 → prefill이 headroom 큰 풀 선택.
  #   PARK_GPU (단일) 은 back-compat.
  if [ -n "${PARK_GPUS:-}" ]; then
    PARK_VISIBLE="${PARK_VISIBLE},${PARK_GPUS}"
    PREFILL_ENV="SGLANG_KV_PARK_GPUS=${PARK_GPUS}"
    [ -n "${PARK_POOL_TOKENS:-}" ] && PREFILL_ENV="${PREFILL_ENV} SGLANG_KV_PARK_POOL_TOKENS=${PARK_POOL_TOKENS}"
  elif [ -n "${PARK_GPU:-}" ]; then
    PARK_VISIBLE="${PARK_VISIBLE},${PARK_GPU}"
    PREFILL_ENV="SGLANG_KV_PARK_GPU=${PARK_GPU}"
    [ -n "${PARK_POOL_TOKENS:-}" ] && PREFILL_ENV="${PREFILL_ENV} SGLANG_KV_PARK_POOL_TOKENS=${PARK_POOL_TOKENS}"
  fi
  PREFILL_CVD="$PARK_VISIBLE"; DECODE_CVD="$PARK_VISIBLE"
  PREFILL_GPU_ARG="--base-gpu-id ${PREFILL_GPU}"
  DECODE_GPU_ARG="--base-gpu-id ${DECODE_GPU}"
fi

# 압박 knob 적용 (prefill radix eviction 유발용)
if [ -n "$PREFILL_MAX_TOTAL_TOKENS" ]; then
  PREFILL_CACHE_ARGS="${PREFILL_CACHE_ARGS} --max-total-tokens ${PREFILL_MAX_TOTAL_TOKENS}"
fi
if [ -n "$DECODE_MAX_TOTAL_TOKENS" ]; then
  DECODE_CACHE_ARGS="${DECODE_CACHE_ARGS} --max-total-tokens ${DECODE_MAX_TOTAL_TOKENS}"
fi

echo "[0/5] Stopping existing processes..."
job_pkill 9 "sglang.launch_server"
job_pkill 9 "mooncake.http_metadata_server"
job_pkill 9 "sglang_router.launch_router"
for port in $BOOTSTRAP_PORT $PREFILL_PORT $DECODE_PORT $ROUTER_PORT $ROUTER_PROM_PORT $MOONCAKE_PORT; do
  fuser -k ${port}/tcp 2>/dev/null || true
done
# Idle KV parking rendezvous is shared via /dev/shm. Stale IPC handles from a prior
# run point at dead GPU memory (CUDA "invalid resource handle" -> parking setup fails),
# so wipe the rendezvous dir on every restart.
# SGLANG_KV_PARK_DIR 을 존중한다: SLURM 에서는 job 마다 다른 디렉터리라, 고정 경로를
# 지우면 같은 노드의 다른 job 의 telemetry 를 날린다.
PARK_DIR=${SGLANG_KV_PARK_DIR:-/dev/shm/sglang_kv_parking}
rm -rf "$PARK_DIR" 2>/dev/null || true
sleep 5

echo "[1/5] Starting Mooncake metadata server (port ${MOONCAKE_PORT})..."
if [ "$MOONCAKE_PORT" = "8080" ]; then
  python -m mooncake.http_metadata_server > "$LOG_DIR/mooncake.log" 2>&1 &
else
  python -m mooncake.http_metadata_server --port "$MOONCAKE_PORT" > "$LOG_DIR/mooncake.log" 2>&1 &
fi
MC_PID=$!
sleep 2
# 조용히 실패하면 KV 전송이 통째로 죽고, 증상은 벤치 쪽의 KVTransferError 로만 나타난다.
if ! kill -0 "$MC_PID" 2>/dev/null; then
  echo "  Mooncake metadata server FAILED — see $LOG_DIR/mooncake.log"
  tail -20 "$LOG_DIR/mooncake.log"
  echo "  (--port 를 지원하지 않는 버전이면 MOONCAKE_PORT=8080 으로 두고 노드를 독점하라.)"
  exit 1
fi
export MOONCAKE_MASTER_SERVER=127.0.0.1:${MOONCAKE_PORT}

echo "[2/5] Prefill (GPU ${PREFILL_GPU}, port ${PREFILL_PORT})  args: ${PREFILL_CACHE_ARGS}"
env CUDA_VISIBLE_DEVICES=${PREFILL_CVD} ${PREFILL_ENV} python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port ${PREFILL_PORT} ${PREFILL_GPU_ARG} \
  --enable-metrics \
  ${PREFILL_CACHE_ARGS} \
  --disaggregation-mode prefill --disaggregation-transfer-backend ${XFER} ${IB_ARG} \
  --disaggregation-bootstrap-port ${BOOTSTRAP_PORT} \
  > "$LOG_DIR/p1.log" 2>&1 &
P1_PID=$!
sleep 3

echo "[3/5] Decode  (GPU ${DECODE_GPU}, port ${DECODE_PORT})  args: ${DECODE_CACHE_ARGS}"
CUDA_VISIBLE_DEVICES=${DECODE_CVD} python3 -m sglang.launch_server \
  --model-path "$MODEL_PATH" --tp 1 --port ${DECODE_PORT} ${DECODE_GPU_ARG} \
  --enable-metrics \
  ${DECODE_CACHE_ARGS} \
  --disaggregation-mode decode --disaggregation-transfer-backend ${XFER} ${IB_ARG} \
  > "$LOG_DIR/d1.log" 2>&1 &
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
  while ! grep -q "ready to roll" "$LOG_DIR/${name}.log" 2>/dev/null; do
    if ! kill -0 "$pid" 2>/dev/null; then
      echo " FAILED (process exited) — see $LOG_DIR/${name}.log"; tail -30 "$LOG_DIR/${name}.log"; return 1
    fi
    if [ "$elapsed" -ge "$READY_TIMEOUT" ]; then
      echo " TIMEOUT after ${READY_TIMEOUT}s — see $LOG_DIR/${name}.log"; tail -30 "$LOG_DIR/${name}.log"; return 1
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
  --host ${ROUTER_HOST} --port ${ROUTER_PORT} \
  --prometheus-port ${ROUTER_PROM_PORT} \
  > "$LOG_DIR/router.log" 2>&1 &
ROUTER_PID=$!
sleep 5
if ! kill -0 "$ROUTER_PID" 2>/dev/null; then
  echo "  Router FAILED to start — see $LOG_DIR/router.log"; tail -20 "$LOG_DIR/router.log"; exit 1
fi

echo ""
echo "Ready. CACHE_MODE=${CACHE_MODE}"
echo "  Router:  http://127.0.0.1:${ROUTER_PORT}"
echo "  Prefill /metrics: http://127.0.0.1:${PREFILL_PORT}/metrics"
echo "  Decode  /metrics: http://127.0.0.1:${DECODE_PORT}/metrics"
