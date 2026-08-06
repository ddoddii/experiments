# shellcheck shell=bash
# A100 / SLURM 공통 환경. sbatch job 과 `srun --pty bash` 셸 양쪽에서 source 한다.
#
#     source "$EXP_ROOT/scripts/slurm/env.sh"
#
# server17 (A6000 x4, 전용 머신) 과 A100 클러스터의 차이는 GPU 세대만이 아니다.
# 노드를 남과 공유한다는 점이 더 크고, 기존 스크립트는 전부 전용 머신을 가정하고 있다:
#
#   1. 고정 포트    30000/30001/8000/8998/8080 을 하드코딩 → 같은 노드에 붙은 두 번째
#                   job 이 bind 에서 죽거나, 더 나쁘게는 남의 서버에 벤치를 쏜다.
#   2. 광역 pkill   start/stop 스크립트가 `pkill -f sglang.launch_server` 를 한다.
#                   내 job 두 개가 한 노드에 뜨면 늦게 시작한 쪽이 먼저 뜬 쪽을 죽인다.
#   3. GPU 번호     PREFILL_GPU=0 DECODE_GPU=1 을 하드코딩. SLURM 이 물리 GPU 4,5 를
#                   줬는데 cgroup 격리가 없는 사이트라면 0,1 은 남의 GPU 다.
#   4. 공유 경로    /dev/shm/sglang_kv_parking, /tmp/hicache 를 job 간에 공유 →
#                   앞 job 의 park telemetry 를 뒤 job 의 sampler 가 읽는다.
#
# 이 파일이 네 가지를 전부 job 단위로 격리한다. 여기서 export 하는 값들은 start_1P_1D.sh
# 가 `env` 로 launch 하며 상속시키므로, 별도로 넘길 필요는 없다.

# ─── 리포지토리 위치 ────────────────────────────────────────────────────────────
# sbatch 는 제출한 스크립트를 노드의 spool 디렉터리로 COPY 해서 실행한다. 그래서 batch
# job 안에서 $0 / ${BASH_SOURCE[0]} 는 /var/spool/slurmd/job12345/slurm_script 를 가리키고,
# "스크립트 위치에서 리포 루트를 유도"하는 흔한 패턴이 batch 에서는 통하지 않는다.
# EXP_ROOT 를 명시적으로 받고, 없으면 관례 경로를 쓴다.
EXP_ROOT=${EXP_ROOT:-"$HOME/experiments"}
if [ ! -d "$EXP_ROOT/scripts/sglang" ]; then
  echo "[env] ERROR: EXP_ROOT=$EXP_ROOT 에 scripts/sglang 이 없다." >&2
  echo "[env]        sbatch --export=ALL,EXP_ROOT=/path/to/experiments ... 로 넘겨라." >&2
  return 1 2>/dev/null || exit 1
fi
export EXP_ROOT

# ─── SLURM 파티션 (문서 및 submit wrapper 용) ──────────────────────────────────
export A100_PARTITION=${A100_PARTITION:-suma_a100}
export A100_QOS=${A100_QOS:-a100_qos}

# JOB_TAG: 로그/결과/공유 디렉터리를 job 단위로 가르는 열쇠. 대화형 srun 셸에서는
# SLURM_JOB_ID 가 있고, 아예 SLURM 밖에서 돌리면 PID 로 떨어진다.
export SLURM_JOB_ID=${SLURM_JOB_ID:-}
if [ -n "$SLURM_JOB_ID" ]; then
  export JOB_TAG=${JOB_TAG:-"j${SLURM_JOB_ID}"}
else
  export JOB_TAG=${JOB_TAG:-"local$$"}
fi

# ─── conda ────────────────────────────────────────────────────────────────────
# 클러스터 로그인 노드와 계산 노드의 rc 파일이 다를 수 있어서 `conda` 가 PATH 에 없는
# 경우가 흔하다. hook 파일을 직접 찾는다.
if ! command -v conda >/dev/null 2>&1; then
  for _c in "$HOME/miniconda3" "$HOME/anaconda3" "$HOME/miniforge3" \
            /opt/conda /usr/local/miniconda3 /usr/local/anaconda3; do
    if [ -f "$_c/etc/profile.d/conda.sh" ]; then
      # shellcheck disable=SC1090
      . "$_c/etc/profile.d/conda.sh"; break
    fi
  done
else
  # shellcheck disable=SC1091
  . "$(conda info --base)/etc/profile.d/conda.sh"
fi
if ! command -v conda >/dev/null 2>&1; then
  echo "[env] ERROR: conda 를 찾을 수 없다. CONDA_HOME 을 설정하거나 module load 하라." >&2
  return 1 2>/dev/null || exit 1
fi
export SGLANG_CONDA_ENV=${SGLANG_CONDA_ENV:-sglang}
conda activate "$SGLANG_CONDA_ENV" || {
  echo "[env] ERROR: conda activate $SGLANG_CONDA_ENV 실패" >&2
  return 1 2>/dev/null || exit 1
}

# ─── 수정한 sglang 소스가 실제로 실행되게 만든다 ──────────────────────────────
# 이게 안 되면 `import sglang` 은 conda env 에 pip 로 설치된 복사본을 집는다. 서버는
# 정상적으로 뜨고, 로그도 정상이고, 다만 park 패치가 없는 upstream 이 돈다. 증상은
# "park arm 이 baseline 과 비슷하다" 하나뿐이라 며칠을 잡아먹을 수 있다.
#
# 권장은 소스 트리를 editable 로 한 번 설치하는 것이다. 그러면 .py 를 고칠 때마다
# 아무것도 다시 할 필요가 없다 (서버 재시작만 하면 된다):
#
#     conda activate sglang
#     cd $SGLANG_SRC_DIR/python && pip install -e . --no-deps
#
#   --no-deps 가 중요하다. 빼면 pip 가 torch/flashinfer 를 다시 해결하려 들면서
#   멀쩡한 env 를 깨뜨릴 수 있다.
#
# 아래 PYTHONPATH 는 editable 설치가 없어도 동작하게 하는 안전망이다. 둘 중 하나만
# 되어 있으면 되고, preflight.sh 가 계산 노드에서 실제 import 경로를 확인해서
# 설치본을 집고 있으면 실험을 시작하지 않는다.
export SGLANG_SRC_DIR=${SGLANG_SRC_DIR:-"$HOME/sglang-source"}
if [ -d "$SGLANG_SRC_DIR/python/sglang" ]; then
  export PYTHONPATH="$SGLANG_SRC_DIR/python${PYTHONPATH:+:$PYTHONPATH}"
else
  echo "[env] WARNING: $SGLANG_SRC_DIR/python/sglang 이 없다."
  echo "[env]          SGLANG_SRC_DIR 을 클론 위치로 지정하라 (기본: \$HOME/sglang-source)."
fi

# ─── 모델 ─────────────────────────────────────────────────────────────────────
# 클러스터 홈이 server17 과 다른 파일시스템일 수 있으므로 후보를 훑는다.
# preflight.sh 가 최종적으로 존재 여부를 검증한다.
# 참고: 이 파일은 `set -e` 인 스크립트에서 source 된다. 그래서 아래처럼 실패해도 되는
# 검사는 전부 if 문으로 쓴다 -- `[ -d x ] && y` 형태는 for 루프의 마지막 명령일 때
# errexit 를 건드려서, "모델을 못 찾았다"가 아니라 "job 이 조용히 종료됐다"가 된다.
if [ -z "${MODEL_PATH:-}" ]; then
  for _m in "$HOME/hf_models/Llama-3.1-8B-Instruct" \
            "/home/uhmturks/hf_models/Llama-3.1-8B-Instruct" \
            "$HOME/models/Llama-3.1-8B-Instruct"; do
    if [ -d "$_m" ]; then MODEL_PATH="$_m"; break; fi
  done
fi
export MODEL_PATH=${MODEL_PATH:-"$HOME/hf_models/Llama-3.1-8B-Instruct"}

# ─── GPU: 논리 인덱스 → SLURM 이 실제로 할당한 물리 인덱스 ──────────────────────
# 사이트마다 두 가지 모드가 있고, 둘을 구분하지 않으면 조용히 남의 GPU 를 잡는다.
#
#   cgroup 격리 O : 할당된 GPU 만 보이고 0..N-1 로 재번호. CUDA_VISIBLE_DEVICES=0,1
#   cgroup 격리 X : 노드의 GPU 가 전부 보이고 CUDA_VISIBLE_DEVICES=4,5 같은 실제 인덱스
#
# 두 경우 모두 "할당 목록의 i 번째"를 쓰면 맞다. 아래 A100_GPU_LIST 가 그 목록이다.
_gpu_src="${CUDA_VISIBLE_DEVICES:-}"
[ -z "$_gpu_src" ] && _gpu_src="${SLURM_JOB_GPUS:-}"
[ -z "$_gpu_src" ] && _gpu_src="${GPU_DEVICE_ORDINAL:-}"
if [ -z "$_gpu_src" ] && command -v nvidia-smi >/dev/null 2>&1; then
  # SLURM 밖(전용 머신, 디버깅)에서 source 한 경우.
  _gpu_src=$(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | paste -sd, -)
fi
IFS=',' read -r -a A100_GPU_LIST <<< "$_gpu_src"
export A100_NGPU=${#A100_GPU_LIST[@]}

# MIG 인스턴스는 UUID 로 노출되고 CUDA_VISIBLE_DEVICES=MIG-xxxx 형태가 된다. 이 실험은
# peer HBM 으로의 CUDA IPC 에 의존하는데 MIG 는 그걸 막는다 — 조용히 park 가 꺼진 채로
# "GPU parking 이 hicache 와 같다"는 결과를 내는 것보다 여기서 죽는 게 낫다.
case "$_gpu_src" in
  *MIG-*|*GPU-*)
    echo "[env] ERROR: GPU 가 UUID/MIG 로 노출됐다 ($_gpu_src)." >&2
    echo "[env]        idle-KV-parking 은 peer GPU 간 CUDA IPC 를 쓰므로 MIG 에서 동작하지 않는다." >&2
    echo "[env]        --gres=gpu:2 로 온전한 카드를 요청하라." >&2
    return 1 2>/dev/null || exit 1 ;;
esac

# gpu_at <논리번호> → 물리번호
gpu_at() { echo "${A100_GPU_LIST[$1]}"; }

export A100_TOPOLOGY=${A100_TOPOLOGY:-1p1d}   # 1p1d (GPU 2장) | 2p2d (4장)
# run_why_faster.sh 가 읽는 이름. 그쪽 기본값은 2p2d 라, 이걸 export 하지 않으면
# srun 셸에서 손으로 러너를 부를 때 GPU 2장짜리 할당에 4장짜리 구성을 띄우려 든다.
export NODES=${NODES:-$A100_TOPOLOGY}
case "$A100_TOPOLOGY" in
  1p1d)
    [ "$A100_NGPU" -ge 2 ] || { echo "[env] ERROR: 1p1d 는 GPU 2장 필요, 받은 건 $A100_NGPU" >&2; return 1 2>/dev/null || exit 1; }
    export PREFILL_GPU=$(gpu_at 0)
    export DECODE_GPU=$(gpu_at 1)
    ;;
  2p2d)
    [ "$A100_NGPU" -ge 4 ] || { echo "[env] ERROR: 2p2d 는 GPU 4장 필요, 받은 건 $A100_NGPU" >&2; return 1 2>/dev/null || exit 1; }
    # PD_LAYOUT=b (각 NVLink 섬에 P 하나 D 하나) 기준 논리 배치.
    export GPU_P0=$(gpu_at 0) GPU_D0=$(gpu_at 1) GPU_P1=$(gpu_at 2) GPU_D1=$(gpu_at 3)
    # 정직하게 적어둔다: 2P2D 는 아직 job 격리가 되지 않는다.
    #
    #   포트  start_2P_2D.sh 는 30000-30003 / 8000 / 8998 / 8080 을 하드코딩한다.
    #         ready 대기 로직이 포트 번호로 로그 파일명을 고르는 case 문까지 포함해서
    #         박혀 있어서, 여기서 포트를 넘겨도 받지 않는다.
    #   GPU   같은 스크립트가 PD_LAYOUT 으로 GPU_P0..GPU_D1 을 스스로 다시 계산하므로
    #         위에서 export 한 물리 번호를 덮어쓴다. cgroup 격리가 되는 사이트라면
    #         (할당 GPU 가 0..3 으로 재번호) 우연히 맞지만, 아니면 남의 GPU 를 잡는다.
    #
    # 따라서 2P2D 는 --exclusive 로 노드를 독점해서 돌려라. 1P1D 는 이 제약이 없다.
    echo "[env] WARNING: 2p2d 는 포트/GPU 를 하드코딩하는 start_2P_2D.sh 를 쓴다."
    echo "[env]          노드를 공유하면 다른 job 과 충돌한다. sbatch --exclusive 를 붙여라."
    if [ "$(IFS=,; echo "${A100_GPU_LIST[*]}")" != "0,1,2,3" ]; then
      echo "[env] ERROR: 할당 GPU 가 [$(IFS=,; echo "${A100_GPU_LIST[*]}")] 인데 start_2P_2D.sh 는" >&2
      echo "[env]        0..3 을 가정한다. 이대로면 할당받지 않은 GPU 를 잡는다." >&2
      echo "[env]        --exclusive 로 노드 전체를 받거나, 1p1d 를 쓰라." >&2
      return 1 2>/dev/null || exit 1
    fi
    ;;
  *) echo "[env] ERROR: A100_TOPOLOGY 는 1p1d | 2p2d (받은 값: $A100_TOPOLOGY)" >&2
     return 1 2>/dev/null || exit 1 ;;
esac

# ─── 포트: job 마다 겹치지 않는 블록 ───────────────────────────────────────────
# 32768 미만에 둔다. 리눅스 기본 ephemeral 범위가 32768-60999 이므로 그 위의 고정 포트는
# 아무 클라이언트 소켓에게나 선점당할 수 있다 — 재현되지 않는 bind 실패의 전형적 원인.
#
# job id 모듈로는 충돌할 수 있으니(300 주기) 실제로 비어 있는지 확인하고 밀어낸다.
_port_free() {
  if command -v ss >/dev/null 2>&1; then
    [ -z "$(ss -Hltn "sport = :$1" 2>/dev/null)" ]
  elif command -v fuser >/dev/null 2>&1; then
    ! fuser "$1/tcp" >/dev/null 2>&1
  else
    return 0   # 확인할 방법이 없으면 비었다고 본다
  fi
}
_block_free() {
  local base=$1 off
  for off in 0 1 2 3 10 11 20 21 30; do
    _port_free $((base + off)) || return 1
  done
  return 0
}
if [ -z "${PORT_BASE:-}" ]; then
  _seed=${SLURM_JOB_ID:-$$}
  _slot=$(( _seed % 300 ))
  for _try in 0 1 2 3 4 5 6 7; do
    PORT_BASE=$(( 10240 + ((_slot + _try * 37) % 300) * 64 ))
    if _block_free "$PORT_BASE"; then break; fi
  done
fi
export PORT_BASE

export PREFILL_PORT=$((PORT_BASE + 0))
export DECODE_PORT=$((PORT_BASE + 1))
export PREFILL2_PORT=$((PORT_BASE + 2))     # 2p2d 전용
export DECODE2_PORT=$((PORT_BASE + 3))      # 2p2d 전용
export ROUTER_PORT=$((PORT_BASE + 10))
export ROUTER_PROM_PORT=$((PORT_BASE + 11))
export BOOTSTRAP_PORT=$((PORT_BASE + 20))
export BOOTSTRAP2_PORT=$((PORT_BASE + 21))
export MOONCAKE_PORT=$((PORT_BASE + 30))
export MOONCAKE_MASTER_SERVER="127.0.0.1:${MOONCAKE_PORT}"

# run_why_faster.sh 의 metrics scraper / occupancy sampler 가 읽는 값.
# 1p1d 에서는 decode 가 PREFILL_PORT+1 이지 30002 가 아니다.
export PREFILL_METRICS_PORT=$PREFILL_PORT
if [ "$A100_TOPOLOGY" = "1p1d" ]; then
  export DECODE_METRICS_PORT=$DECODE_PORT
  export OCC_TARGETS="${PREFILL_PORT}:P0:P ${DECODE_PORT}:D0:D"
else
  export DECODE_METRICS_PORT=$DECODE_PORT
  export OCC_TARGETS="${PREFILL_PORT}:P0:P ${PREFILL2_PORT}:P1:P ${DECODE_PORT}:D0:D ${DECODE2_PORT}:D1:D"
fi

# 벤치마크 클라이언트가 붙을 곳. 이걸 export 하지 않으면 벤치는 :8000 으로 쏘고,
# 운 나쁘면 같은 노드의 남의 라우터에 붙는다.
export ROUTER_URL=${ROUTER_URL:-"http://127.0.0.1:${ROUTER_PORT}/v1/chat/completions"}
# 라우터를 0.0.0.0 이 아니라 루프백에만 묶는다. 공유 노드에서 외부에 열어둘 이유가 없다.
export ROUTER_HOST=${ROUTER_HOST:-127.0.0.1}

# Mooncake KV 전송을 루프백으로. (start_2P_2D.sh 의 주석 참고: 라우팅 가능한 주소를 쓰면
# 고정 4-tuple 때문에 TIME-WAIT 고갈로 EADDRNOTAVAIL 이 난다.)
export SGLANG_HOST_IP=${SGLANG_HOST_IP:-127.0.0.1}

# ─── job 단위 디렉터리 ────────────────────────────────────────────────────────
# park telemetry 와 hicache 파일 백엔드는 GPU 번호로만 키가 잡혀 있어서 job 간에 섞인다.
# 둘 다 env 로 재지정 가능하고 stop 스크립트도 이 변수를 읽는다.
export SGLANG_KV_PARK_DIR=${SGLANG_KV_PARK_DIR:-"/dev/shm/sglang_kv_parking_${JOB_TAG}"}
export SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR=${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR:-"${TMPDIR:-/tmp}/hicache_${JOB_TAG}"}
export LOG_DIR=${LOG_DIR:-"$EXP_ROOT/logs/slurm/${JOB_TAG}"}
mkdir -p "$LOG_DIR" "$SGLANG_KV_PARK_DIR" "$SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR"

# ─── 요약 ─────────────────────────────────────────────────────────────────────
env_summary() {
  echo "  node        : $(hostname)   job=${SLURM_JOB_ID:-<none>}  tag=${JOB_TAG}"
  echo "  topology    : ${A100_TOPOLOGY}   GPUs=[$(IFS=,; echo "${A100_GPU_LIST[*]}")]"
  if [ "$A100_TOPOLOGY" = "1p1d" ]; then
    echo "  placement   : P=gpu${PREFILL_GPU}  D=gpu${DECODE_GPU}"
  else
    echo "  placement   : P0=gpu${GPU_P0} D0=gpu${GPU_D0} P1=gpu${GPU_P1} D1=gpu${GPU_D1}"
  fi
  echo "  ports       : base=${PORT_BASE}  P=${PREFILL_PORT} D=${DECODE_PORT}" \
       "router=${ROUTER_PORT} bootstrap=${BOOTSTRAP_PORT} mooncake=${MOONCAKE_PORT}"
  echo "  model       : ${MODEL_PATH}"
  echo "  sglang src  : ${SGLANG_SRC_DIR}"
  echo "  logs        : ${LOG_DIR}"
  echo "  park dir    : ${SGLANG_KV_PARK_DIR}"
  echo "  hicache dir : ${SGLANG_HICACHE_FILE_BACKEND_STORAGE_DIR}"
}
