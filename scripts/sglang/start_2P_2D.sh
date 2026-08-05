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
# run the modified source tree, and fail loudly if we'd be running the installed copy
source "$(dirname "${BASH_SOURCE[0]}")/_use_source.sh"

MODEL_PATH=${MODEL_PATH:-"/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"}
QUANTIZATION=${QUANTIZATION:-""}           # 8B@BF16=16GB, quantization 불필요
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-"llama3"}   # Llama-3.1 uses llama3 parser
HICACHE_RATIO=${HICACHE_RATIO:-"1.2"}
HICACHE_WRITE_POLICY=${HICACHE_WRITE_POLICY:-"write_through_selective"}
# 압박(pressure) knob: KV 풀을 작게 잡아 점유율이 포화까지 오르게 (불균형/축출 측정용).
# 빈 값이면 기본 대용량 풀. 예: PREFILL_MAX_TOTAL_TOKENS=20000
# ROUTER_MODE=skew starts one router per prefill (:8000 -> P0, :8001 -> P1) so the client
# can set the prefill-side load imbalance exactly. Default "balanced" = the usual single
# router on :8000. See the [6/6] block.
# ─── Mooncake transfer address ──────────────────────────────────────────────
# All four workers are on ONE host, but SGLang derives the KV-transfer address from
# get_local_ip_auto(), which picks the node's routable address (165.132.142.205 here).
# Every Mooncake transfer then opens a TCP connection from that address to that same
# address on a FIXED port, e.g. 165.132.142.205:16916.
#
# A fixed (src_ip, dst_ip, dst_port) triple leaves only the local port range to make the
# 4-tuple unique: 32768-60999 = 28,231 ports, each held ~60 s in TIME-WAIT. That caps
# sustained transfers at roughly 470/s, and past it connect() returns EADDRNOTAVAIL --
# "Cannot assign requested address", exactly the error in p1.log at :15119 and :16916.
# tcp_tw_reuse=2 on this kernel means TIME-WAIT reuse applies to LOOPBACK ONLY, so using
# the routable address opts out of the one mechanism that would have recycled the tuples.
#
# SGLANG_HOST_IP=127.0.0.1 makes the transfer loopback, which both enables tw_reuse and
# removes the NIC path. Override only for genuinely multi-node runs.
#
# (An earlier diagnosis of mine dismissed port exhaustion because `ss` showed 121
# TIME-WAIT sockets -- measured minutes after the failure, long past the 60 s drain. That
# measurement proved nothing.)
export SGLANG_HOST_IP=${SGLANG_HOST_IP:-127.0.0.1}

ROUTER_MODE=${ROUTER_MODE:-balanced}
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
PARK_POOL_TOKENS=${PARK_POOL_TOKENS:-30000}          # per-P park BUDGET, split over candidates
PARK_POOL_TOKENS_PER_GPU=${PARK_POOL_TOKENS_PER_GPU:-}  # if set, size EACH pool instead
PARK_MEM_FRACTION=${PARK_MEM_FRACTION:-0.70}         # leave room for the park buffer

# --- PD_LAYOUT: which GPU each role sits on ------------------------------------------
# server17's `nvidia-smi topo -m` has exactly TWO NVLink bridges, (0,1) and (2,3);
# every other pair is PCIe (NODE/PHB), measured at 3.3 GB/s against 27-53 GB/s for a
# bridged pair and 26 GB/s for pinned host memory. Two bridges over four GPUs means each
# GPU has exactly ONE fast partner, so "both prefills fast to each other" and "each
# prefill fast to a decode" are mutually exclusive. That is the hardware, not a choice.
#
#   a (default) P0=0 P1=1 D0=2 D1=3   -- P<->P is NVLink; every P->D link is PCIe.
#   b           P0=0 D0=1 P1=2 D1=3   -- each NVLink island holds one P and one D.
#
# Layout b is what Exp 2 (P/D imbalance) needs: a prefill's one fast park target becomes
# a decode GPU, so parking into idle decode HBM survives the bandwidth-aware demotion in
# _select_pool() instead of being ranked below the local pool.
#
# Layout b also puts HALF the Mooncake P->D KV transfers on NVLink (P0->D0, P1->D1),
# which moves baseline TTFT on its own. Run every arm of a comparison under the SAME
# layout, and do not place layout-b absolute latencies beside layout-a ones.
PD_LAYOUT=${PD_LAYOUT:-a}
case "$PD_LAYOUT" in
  a) GPU_P0=0; GPU_P1=1; GPU_D0=2; GPU_D1=3 ;;
  b) GPU_P0=0; GPU_D0=1; GPU_P1=2; GPU_D1=3 ;;
  *) echo "PD_LAYOUT must be 'a' or 'b', got '$PD_LAYOUT'" >&2; exit 1 ;;
esac

# Decode-side --mem-fraction-static. Until now only the PREFILLS got one, so decode took
# SGLang's default and held ~42.6 of 49 GB -- no room for a prefill to allocate a park
# pool on a decode GPU, which is the whole mechanism Exp 2 measures.
#
# Keep the cut SMALL. Lowering this shrinks decode's KV pool, and decode's spare KV pool
# is precisely the headroom the experiment claims to harvest: trim it too hard and the
# measurement destroys the thing being measured. 0.80 leaves ~10 GB, and the pools need
# 2.44 GB (2 prefills x 10k tokens x 128 KiB at PARK_POOL_TOKENS=30000 over 3 candidates).
#
# Set it on the BASELINE arms too. Otherwise the baseline is the only arm with full-size
# decode pools and every occupancy comparison confounds placement with capacity -- the
# same trap FORCE_MEM_FRACTION already exists to close on the prefill side.
PARK_MEM_FRACTION_D=${PARK_MEM_FRACTION_D:-}
MEMFRAC_D=""
[ -n "${PARK_MEM_FRACTION_D}" ] && MEMFRAC_D="--mem-fraction-static ${PARK_MEM_FRACTION_D}"
SGLANG_KV_PARK_GEN=${SGLANG_KV_PARK_GEN:-1}
SGLANG_KV_PARK_PRESSURE_AWARE=${SGLANG_KV_PARK_PRESSURE_AWARE:-1}
# session-keyed parking (task 7): free superseded per-conversation versions -> small pool
# holds the live working set (higher survival, less variance). SGLANG_KV_PARK_SESSION_KEYED=1.
SGLANG_KV_PARK_SESSION_KEYED=${SGLANG_KV_PARK_SESSION_KEYED:-0}
SGLANG_KV_PARK_SLAB_TOKENS=${SGLANG_KV_PARK_SLAB_TOKENS:-6000}   # session-keyed slab size
# Phase 3: bandwidth-ranked placement (measured link matrix, /dev/shm cached) and the
# CPU DRAM overflow tier (placement priority 3). HOST_OVERFLOW=1 lets a park that no GPU
# can take land in pinned host memory instead of being dropped; allocation is on demand
# at bucket granularity, so nothing is reserved when it is not used. HOST_MAX_GB is a
# LIMIT, not a reservation.
SGLANG_KV_PARK_BW_AWARE=${SGLANG_KV_PARK_BW_AWARE:-1}
SGLANG_KV_PARK_HOST_OVERFLOW=${SGLANG_KV_PARK_HOST_OVERFLOW:-0}
SGLANG_KV_PARK_HOST_BUCKET_TOKENS=${SGLANG_KV_PARK_HOST_BUCKET_TOKENS:-2048}
SGLANG_KV_PARK_HOST_MAX_GB=${SGLANG_KV_PARK_HOST_MAX_GB:-8}
# reuse-value eviction: give up the CHEAPEST-to-rebuild prefix, discounted by
# staleness, instead of the oldest. REUSE_AWARE=0 reproduces plain LRU (A/B arm).
# Placement decision log (Exp 2 / M3). Empty = off; set to a path PREFIX and each
# prefill appends <prefix>.gpu<N>.jsonl -- one record per park with every candidate's
# live usage at the instant of the choice. The rejected candidates are what make
# "placement followed headroom" checkable instead of a restatement of the selector.
SGLANG_KV_PARK_DECISION_LOG=${SGLANG_KV_PARK_DECISION_LOG:-}
SGLANG_KV_PARK_REUSE_AWARE=${SGLANG_KV_PARK_REUSE_AWARE:-1}
SGLANG_KV_PARK_REUSE_HALFLIFE_S=${SGLANG_KV_PARK_REUSE_HALFLIFE_S:-30}
# --- "why is it faster" ablation knobs (run_why_faster.sh) ---------------------------
# These MUST be listed in _PENV below. The park env is passed to each server as an
# explicit string, not inherited, so a knob that is exported but not listed there is
# silently ignored -- the arm would run as the default and the ablation would report
# "no difference" for a variable that was never actually moved.
#
# FORCE_HOST=1   every park goes to CPU DRAM instead of an idle GPU; everything else
#                (index, policy, fetch code, allocated park pool) unchanged -> isolates
#                the MEDIUM.
# SYNC_FETCH=1   wait for the fetch copy instead of enqueueing it. Required for the
#                cost-model run: the default is async, so the recorded ms is enqueue
#                time and a bandwidth fitted from it is fiction.
# FETCH_TRACE    directory for per-fetch rows (tier, tokens, bytes, ms).
SGLANG_KV_PARK_FORCE_HOST=${SGLANG_KV_PARK_FORCE_HOST:-0}
SGLANG_KV_PARK_ASYNC_PARK=${SGLANG_KV_PARK_ASYNC_PARK:-1}
SGLANG_KV_PARK_SYNC_FETCH=${SGLANG_KV_PARK_SYNC_FETCH:-0}
SGLANG_KV_PARK_FETCH_TRACE=${SGLANG_KV_PARK_FETCH_TRACE:-}
# hicache args (prefill). PARK_NO_HICACHE=1 drops them so radix+park runs at host RAM 0
# (the clean "park as a host-RAM-free alternative to hicache" arm).
#
# HICACHE_WRITE_POLICY  = write_through | write_through_selective | write_back
#   write_through           : offload every access to L2/L3 immediately (max host/L3 use)
#   write_through_selective : offload only hot data past a threshold (medium)
#   write_back              : offload only when evicted from the upper level (min host use)
# HICACHE_STORAGE_BACKEND = file | none
#   file : L1(GPU)+L2(host DRAM)+L3(disk /tmp/hicache). The L3 files show up as
#          RECLAIMABLE OS page cache -> the "94GB" a reviewer will question.
#   none : L2-only (no L3 disk). Host footprint = just the L2 host KV pool
#          (hicache-ratio x GPU KV), NO file page cache. Answers "turn the file
#          backend off and use only the L2 host cache -- what happens?"
HICACHE_STORAGE_BACKEND=${HICACHE_STORAGE_BACKEND:-file}
_BACKEND_ARG="--hicache-storage-backend file"
if [ "${HICACHE_STORAGE_BACKEND}" = "none" ] || [ -z "${HICACHE_STORAGE_BACKEND}" ]; then
  _BACKEND_ARG=""          # L2-only: hierarchical cache with host DRAM tier, no L3 disk
fi
HICACHE_ARG="--enable-hierarchical-cache ${_BACKEND_ARG} --hicache-ratio ${HICACHE_RATIO} --hicache-write-policy ${HICACHE_WRITE_POLICY}"
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
  CVD_P0="0,1,2,3"; BASE_P0="--base-gpu-id ${GPU_P0}"
  CVD_P1="0,1,2,3"; BASE_P1="--base-gpu-id ${GPU_P1}"
  CVD_D0="0,1,2,3"; BASE_D0="--base-gpu-id ${GPU_D0}"
  CVD_D1="0,1,2,3"; BASE_D1="--base-gpu-id ${GPU_D1}"
  _PENV="SGLANG_KV_PARK_POOL_TOKENS=${PARK_POOL_TOKENS} SGLANG_KV_PARK_GEN=${SGLANG_KV_PARK_GEN} SGLANG_KV_PARK_PRESSURE_AWARE=${SGLANG_KV_PARK_PRESSURE_AWARE} SGLANG_KV_PARK_SESSION_KEYED=${SGLANG_KV_PARK_SESSION_KEYED} SGLANG_KV_PARK_SLAB_TOKENS=${SGLANG_KV_PARK_SLAB_TOKENS} SGLANG_KV_PARK_BW_AWARE=${SGLANG_KV_PARK_BW_AWARE} SGLANG_KV_PARK_HOST_OVERFLOW=${SGLANG_KV_PARK_HOST_OVERFLOW} SGLANG_KV_PARK_HOST_BUCKET_TOKENS=${SGLANG_KV_PARK_HOST_BUCKET_TOKENS} SGLANG_KV_PARK_HOST_MAX_GB=${SGLANG_KV_PARK_HOST_MAX_GB} SGLANG_KV_PARK_REUSE_AWARE=${SGLANG_KV_PARK_REUSE_AWARE} SGLANG_KV_PARK_REUSE_HALFLIFE_S=${SGLANG_KV_PARK_REUSE_HALFLIFE_S} SGLANG_KV_PARK_DECISION_LOG=${SGLANG_KV_PARK_DECISION_LOG} SGLANG_KV_PARK_FORCE_HOST=${SGLANG_KV_PARK_FORCE_HOST} SGLANG_KV_PARK_ASYNC_PARK=${SGLANG_KV_PARK_ASYNC_PARK} SGLANG_KV_PARK_SYNC_FETCH=${SGLANG_KV_PARK_SYNC_FETCH} SGLANG_KV_PARK_FETCH_TRACE=${SGLANG_KV_PARK_FETCH_TRACE}"
  # Which GPUs each prefill may park ONTO.
  #
  # Default (0 / 1) gives each prefill a pool on its own GPU only. Cross-P sharing then
  # exists on the FETCH side alone: P0 can read a prefix P1 parked into P1's pool, but P0
  # cannot place into P1's idle HBM. That is the "park_local" configuration.
  #
  # PARK_PEER=1 gives each prefill a pool on BOTH P GPUs, so under uneven load P0's
  # placement policy can choose its GPU1 pool because GPU1's live serving usage is low --
  # the peer-placement mechanism Exp 2 measures. Pool tokens are HALVED so the total park
  # HBM per GPU is unchanged: otherwise the peer arm would win simply by having twice the
  # park capacity, which is not the claim.
  # PARK_POOL_TOKENS is the per-prefill park BUDGET, split evenly over that prefill's
  # candidate GPUs -- not a per-pool size. _init_park_gpu_pool() builds one pool of
  # SGLANG_KV_PARK_POOL_TOKENS on EVERY listed GPU, so passing the budget through
  # unchanged would give a 3-candidate arm three times the park cache of a 1-candidate
  # arm and "placement freedom won" would really be "three times the cache won".
  # The PARK_PEER branch has always halved for this reason; this generalises it so the
  # PARK_GPUS_* overrides get the same treatment.
  _park_env_for() {   # $1 = comma-separated GPU list -> echoes the env prefix
    local _list="$1"
    local _n; _n=$(awk -F, '{print NF}' <<< "$_list")
    local _per=$(( PARK_POOL_TOKENS / _n ))
    # PARK_POOL_TOKENS_PER_GPU fixes the allocation on EACH GPU instead of the total
    # across them, so arms differ only in how many GPUs they reach.
    #
    # Which of the two to hold constant is not a detail, it decides what the comparison
    # means. Equalising the TOTAL asks "given a fixed HBM budget, does spreading it help?"
    # -- and the answer measured in Exp 2 is no, because one big pool evicts less than
    # several small ones. Equalising PER GPU asks "given that each GPU can spare this
    # much, does reaching more of them help?", which is the question the paper's premise
    # actually poses: idle HBM is claimed to be free where it exists, so a policy that
    # reaches three GPUs should be allowed three GPUs' worth.
    [ -n "${PARK_POOL_TOKENS_PER_GPU}" ] && _per=${PARK_POOL_TOKENS_PER_GPU}
    echo "SGLANG_KV_PARK_GPUS=${_list} ${_PENV/SGLANG_KV_PARK_POOL_TOKENS=${PARK_POOL_TOKENS}/SGLANG_KV_PARK_POOL_TOKENS=${_per}}"
  }
  if [ "${PARK_PEER:-0}" = "1" ]; then
    # Peer = the OTHER PREFILL, which is layout-dependent: under PD_LAYOUT=b the GPU
    # numbered 1 is a decode node, so hardcoding "0,1" here would silently mean
    # something else entirely.
    _L0="${GPU_P0},${GPU_P1}"; _L1="${GPU_P1},${GPU_P0}"
  else
    _L0="${PARK_GPUS_P0:-${GPU_P0}}"; _L1="${PARK_GPUS_P1:-${GPU_P1}}"
  fi
  ENV_P0="$(_park_env_for "$_L0")"
  ENV_P1="$(_park_env_for "$_L1")"
  if [ -n "${PARK_POOL_TOKENS_PER_GPU}" ]; then
    echo "park: P0 -> GPU[$_L0]   P1 -> GPU[$_L1]   ${PARK_POOL_TOKENS_PER_GPU} tok on EACH pool"
  else
    echo "park: P0 -> GPU[$_L0]   P1 -> GPU[$_L1]   budget ${PARK_POOL_TOKENS} tok/prefill, split evenly"
  fi
  ENV_D0=""; ENV_D1=""
else
  PARK_ARG=""
  # FORCE_MEM_FRACTION lets a NON-park arm run at the park arm's --mem-fraction-static.
  # Without it the park arm is the only one at 0.70 while the others take SGLang's
  # default, so any TTFT difference confounds the placement mechanism with the memory
  # budget. Exp 1 measured park +0.56 s on TURN 0 -- where there is no prefix to reuse
  # and no fetch to perform -- which is a constant overhead, not the mechanism; this knob
  # is how that gets attributed.
  MEMFRAC_P=""
  [ -n "${FORCE_MEM_FRACTION}" ] && MEMFRAC_P="--mem-fraction-static ${FORCE_MEM_FRACTION}"
  CVD_P0=${GPU_P0}; BASE_P0=""; CVD_P1=${GPU_P1}; BASE_P1=""
  CVD_D0=${GPU_D0}; BASE_D0=""; CVD_D1=${GPU_D1}; BASE_D1=""
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
echo "hicache-storage-backend = ${HICACHE_STORAGE_BACKEND:-file}"
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

echo "[2/6] Starting Prefill server 1 (GPU ${GPU_P0}, port 30000, bootstrap 8998)  park=${IDLE_KV_PARKING:-0}..."
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

echo "[3/6] Starting Prefill server 2 (GPU ${GPU_P1}, port 30001, bootstrap 8999)..."
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

echo "[4/6] Starting Decode server 1 (GPU ${GPU_D0}, port 30002)..."
env CUDA_VISIBLE_DEVICES=${CVD_D0} ${ENV_D0} python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30002 ${BASE_D0} \
  --enable-metrics \
  ${D_MTT} ${PARK_ARG} ${MEMFRAC_D} ${RADIX_ARG} \
  ${QUANTIZATION:+--quantization $QUANTIZATION} \
  --disaggregation-mode decode --disaggregation-transfer-backend mooncake \
  --tool-call-parser $TOOL_CALL_PARSER \
  > "$LOG_DIR/d1.log" 2>&1 &
echo "  PID: $!"
sleep 3

echo "[5/6] Starting Decode server 2 (GPU ${GPU_D1}, port 30003)..."
env CUDA_VISIBLE_DEVICES=${CVD_D1} ${ENV_D1} python3 -m sglang.launch_server \
  --model-path $MODEL_PATH --tp 1 --port 30003 ${BASE_D1} \
  --enable-metrics \
  ${D_MTT} ${PARK_ARG} ${MEMFRAC_D} ${RADIX_ARG} \
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
if [ "$ROUTER_MODE" = "skew" ]; then
  # Exp 2 (controlled imbalance): ONE ROUTER PER PREFILL, both listing both decodes.
  # A single router balances the two prefills, so P0 and P1 fill up together and
  # peer-GPU placement has nothing to exploit -- the effect under test is invisible by
  # construction. Two routers move the routing decision to the client, which then sets
  # the skew exactly (SKEW=0.9 -> 90% of sessions to P0) instead of hoping the router's
  # policy produces an imbalance. Decodes stay shared so only the PREFILL side is skewed.
  echo "[6/6] Starting TWO routers (skew mode): :8000 -> P0 only, :8001 -> P1 only"
  python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30000 8998 \
    --decode http://127.0.0.1:30002 \
    --decode http://127.0.0.1:30003 \
    --host 0.0.0.0 --port 8000 \
    > "$LOG_DIR/router_p0.log" 2>&1 &
  echo "  :8000 (P0) PID: $!"
  python -m sglang_router.launch_router \
    --pd-disaggregation \
    --prefill http://127.0.0.1:30001 8999 \
    --decode http://127.0.0.1:30002 \
    --decode http://127.0.0.1:30003 \
    --host 0.0.0.0 --port 8001 \
    > "$LOG_DIR/router_p1.log" 2>&1 &
  echo "  :8001 (P1) PID: $!"
  echo "  benchmark with: SGLANG_URLS=http://127.0.0.1:8000/v1/chat/completions,\\"
  echo "                              http://127.0.0.1:8001/v1/chat/completions SKEW=0.9"
else
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
fi

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

# ─── Router readiness gate ──────────────────────────────────────────────────
# The router is launched with & and takes seconds to come up and register workers, but
# this script used to print "All done" right after. A benchmark started immediately got
# HTTP 503 "No available servers" for most of its run, and one full Exp1+Exp2 sweep was
# lost that way: every arm reported 199/220 turns failed, yet still produced a complete
# table of plausible-looking numbers.
#
# /health is NOT the check. In an earlier incident the router answered /health 200 while
# every circuit was open. The only check that means anything is a REAL request that comes
# back non-503.
wait_for_router() {
  local port=$1 label=$2 deadline=$((SECONDS + ${ROUTER_WAIT_S:-180}))
  echo -n "  waiting for router :$port ($label) to route a request"
  while [ $SECONDS -lt $deadline ]; do
    local body code
    body=$(curl -s -o /tmp/.router_probe_$port -w '%{http_code}' -m 10 \
      -X POST "http://127.0.0.1:$port/v1/chat/completions" \
      -H 'Content-Type: application/json' \
      -d '{"model":"x","messages":[{"role":"user","content":"hi"}],"max_tokens":1,"stream":false}' \
      2>/dev/null)
    code="$body"
    if [ "$code" = "200" ]; then
      echo " OK ($((SECONDS)) s)"
      rm -f /tmp/.router_probe_$port
      return 0
    fi
    echo -n "."
    sleep 3
  done
  echo " TIMEOUT"
  echo "  ERROR: router :$port never served a request in ${ROUTER_WAIT_S:-180}s."
  echo "         last status=$code body: $(head -c 300 /tmp/.router_probe_$port 2>/dev/null)"
  echo "         Do NOT benchmark against this: results would be ~all 503."
  rm -f /tmp/.router_probe_$port
  return 1
}

echo ""
echo "Router readiness:"
wait_for_router 8000 "P0/all" || exit 1
if [ "$ROUTER_MODE" = "skew" ]; then
  wait_for_router 8001 "P1" || exit 1
fi

echo ""
echo "All done. Logs at logs/sglang/*.log"
echo "Router at http://127.0.0.1:8000"
echo ""
echo "To check hicache metrics:"
echo "  curl http://localhost:9199/metrics"
echo "  curl http://localhost:30000/get_server_info | python3 -m json.tool"