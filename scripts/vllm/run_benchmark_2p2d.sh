#!/bin/bash
# =============================================================================
# 2P2D Disaggregated Prefill Benchmark — WildChat Multi-turn
# =============================================================================
#
# Architecture:
#
#   Client
#     └─► disagg_proxy_p2p_nccl_xpyd.py  (port 10001, HTTP service)
#               ├─► Prefill-1  (GPU 0, port 8100, P2pNcclConnector producer)
#               ├─► Prefill-2  (GPU 1, port 8101, P2pNcclConnector producer)
#               ├─► Decode-1   (GPU 2, port 8200, P2pNcclConnector consumer)
#               └─► Decode-2   (GPU 3, port 8201, P2pNcclConnector consumer)
#
#   disagg_proxy_p2p_nccl_xpyd.py also handles:
#     - ZMQ service-discovery (port 30001): P/D instances register here
#     - Injects decode server address into request ID → P2pNcclConnector routes KV
#
# Usage:
#   bash run_benchmark_2p2d.sh
#   MODEL_PATH=/other/model bash run_benchmark_2p2d.sh
#
# =============================================================================

set -euo pipefail

# =============================================================================
# CUDA library path fix
# =============================================================================
# nixl requires libcudart.so.13, but vllm-source conda env only has cu12.
# libcudart.so.13 is present in vllm-ppd env — add it to the search path.
# Permanent fix: conda run -n vllm-source pip install nvidia-cuda-runtime-cu13
_CU13_LIB=/home/uhmturks/miniconda3/envs/vllm-ppd/lib/python3.12/site-packages/nvidia/cu13/lib
if [[ -f "${_CU13_LIB}/libcudart.so.13" ]]; then
    export LD_LIBRARY_PATH="${_CU13_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"
else
    echo "WARNING: libcudart.so.13 not found at ${_CU13_LIB}"
    echo "         If vllm serve fails, install it with:"
    echo "         conda run -n vllm-source pip install nvidia-cuda-runtime-cu13"
fi

# =============================================================================
# Configuration
# =============================================================================
export MODEL_PATH=${MODEL_PATH:-/home/uhmturks/hf_models/Llama-3.1-8B-Instruct}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Llama}
HOST_IP=${HOST_IP:-127.0.0.1}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VLLM_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# disagg_proxy_p2p_nccl_xpyd.py: single proxy that handles both
#   - ZMQ service-discovery (port NCCL_PROXY_PORT=30001): P/D register themselves
#   - HTTP routing (port SERVICE_PORT=10001): injects D address into request ID
#     so P2pNcclConnector knows which D to transfer KV to
XPYD_PROXY="${VLLM_ROOT}/examples/online_serving/disaggregated_serving_p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py"

# Dataset
DATA_PATH="${SCRIPT_DIR}/../data/WildChat_1M.json"
CONVERTED_DATA="${SCRIPT_DIR}/wildchat_conv_512.json"

# Results
RESULTS_DIR="${SCRIPT_DIR}/results"
TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
RESULTS_JSON="${RESULTS_DIR}/results_2p2d_${TIMESTAMP}.json"

# GPU assignments  (A6000 x4)
P1_GPU=0;  P2_GPU=1
D1_GPU=2;  D2_GPU=3

# HTTP ports
P1_PORT=8100;  P2_PORT=8101
D1_PORT=8200;  D2_PORT=8201
# disagg_proxy_p2p_nccl_xpyd.py handles both roles:
#   - ZMQ registration listener (NCCL_PROXY_PORT): P/D instances register here
#   - HTTP service (SERVICE_PORT): benchmark client connects here
SERVICE_PORT=10001   # xpyd proxy HTTP service (benchmark connects here)
NCCL_PROXY_PORT=30001  # xpyd proxy ZMQ registration port

# Per-instance KV transfer ports (each instance needs its own)
P1_KV_PORT=21001;  P2_KV_PORT=21002
D1_KV_PORT=22001;  D2_KV_PORT=22002

# Model settings
MAX_MODEL_LEN=8192
GPU_MEM_UTIL_P=0.85   # prefill can use less memory
GPU_MEM_UTIL_D=0.85   # decode needs KV cache space

# Benchmark settings
NUM_ITEMS=512          # conversations to sample from WildChat
MIN_TURNS=4            # minimum turns per conversation
NUM_CLIENTS=4
MAX_ACTIVE_CONVERSATIONS=8
REQUEST_RATE=0         # 0 = no delay (max throughput)
SEED=99
SERVER_TIMEOUT=1200    # seconds to wait for each server to start

PYTHON="conda run -n vllm-source python"
# Use the env's Python directly for background daemons (conda run swallows output)
CONDA_PYTHON="/home/uhmturks/miniconda3/envs/vllm-source/bin/python"

# =============================================================================
# Setup
# =============================================================================
mkdir -p "${RESULTS_DIR}"

PIDS=()

log() {
    echo "[$(date '+%H:%M:%S')] $*"
}

cleanup() {
    echo ""
    echo "[cleanup] Stopping all background processes..."
    for pid in "${PIDS[@]:-}"; do
        kill "$pid" 2>/dev/null || true
    done
    wait 2>/dev/null || true
    echo "[cleanup] Done."
}
trap cleanup EXIT INT TERM

# Kill any leftover processes from a previous run holding our ports
# (prevents "Address already in use" on restart)
kill_port() {
    local port=$1
    local pids
    pids=$(lsof -ti:"${port}" 2>/dev/null) || true
    if [[ -n "${pids}" ]]; then
        log "Killing leftover process(es) on port ${port}: ${pids}"
        echo "${pids}" | xargs kill -9 2>/dev/null || true
    fi
}

for _port in "${NCCL_PROXY_PORT}" "${SERVICE_PORT}" \
             "${P1_PORT}" "${P2_PORT}" "${D1_PORT}" "${D2_PORT}" \
             "${P1_KV_PORT}" "${P2_KV_PORT}" "${D1_KV_PORT}" "${D2_KV_PORT}"; do
    kill_port "${_port}"
done
sleep 1

wait_for_server() {
    local port=$1
    local label=${2:-"server"}
    local check=${3:-"http"}   # "http" = /v1/models, "tcp" = port open
    log "Waiting for ${label} (port ${port})..."
    local deadline=$((SECONDS + SERVER_TIMEOUT))
    while (( SECONDS < deadline )); do
        if [[ "${check}" == "tcp" ]]; then
            if nc -z localhost "${port}" 2>/dev/null; then
                log "${label} is ready."
                return 0
            fi
        else
            if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
                log "${label} is ready."
                return 0
            fi
        fi
        sleep 3
    done
    log "ERROR: Timeout waiting for ${label} (port ${port})"
    log "Check logs in ${RESULTS_DIR}/"
    exit 1
}

# =============================================================================
# Preflight checks
# =============================================================================
log "=== 2P2D Disaggregated Prefill Benchmark ==="
log "Model:   ${MODEL_PATH}"
log "GPUs:    P1=GPU${P1_GPU} P2=GPU${P2_GPU} | D1=GPU${D1_GPU} D2=GPU${D2_GPU}"
log "Ports:   P1=${P1_PORT} P2=${P2_PORT} | D1=${D1_PORT} D2=${D2_PORT}"
log "Service: http://localhost:${SERVICE_PORT}"
log "Results: ${RESULTS_JSON}"
echo ""

if [[ ! -d "${MODEL_PATH}" ]]; then
    log "ERROR: MODEL_PATH not found: ${MODEL_PATH}"
    exit 1
fi

if [[ ! -f "${XPYD_PROXY}" ]]; then
    log "ERROR: xpyd proxy not found: ${XPYD_PROXY}"
    exit 1
fi

num_gpus=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
if (( num_gpus < 4 )); then
    log "ERROR: Need 4 GPUs, found ${num_gpus}"
    exit 1
fi
log "Found ${num_gpus} GPUs. OK."

# =============================================================================
# Step 1: Convert WildChat dataset
# =============================================================================
if [[ ! -f "${CONVERTED_DATA}" ]]; then
    if [[ ! -f "${DATA_PATH}" ]]; then
        log "ERROR: WildChat dataset not found: ${DATA_PATH}"
        log "Download with: python download_datasets.py"
        exit 1
    fi
    log "Converting WildChat → ${CONVERTED_DATA}"
    ${PYTHON} "${SCRIPT_DIR}/convert_wildchat_to_openai.py" \
        "${DATA_PATH}" \
        "${CONVERTED_DATA}" \
        --seed "${SEED}" \
        --max-items "${NUM_ITEMS}" \
        --min-turns "${MIN_TURNS}"
else
    log "Using existing converted dataset: ${CONVERTED_DATA}"
fi

# =============================================================================
# Step 2: Start xpyd proxy (ZMQ registration on 30001, HTTP service on 10001)
# =============================================================================
log "Starting xpyd proxy (ZMQ=${NCCL_PROXY_PORT}, HTTP=${SERVICE_PORT})..."
PROXY_PORT=${NCCL_PROXY_PORT} \
    "${CONDA_PYTHON}" "${XPYD_PROXY}" \
    > "${RESULTS_DIR}/xpyd_proxy.log" 2>&1 &
PIDS+=($!)
sleep 2  # let the ZMQ socket bind before P/D instances start

# =============================================================================
# Step 3: Start Prefill servers
# =============================================================================
# P2pNcclConnector config per instance:
#   - kv_role         : kv_producer (prefill)
#   - kv_port         : this instance's KV transfer port (unique per instance)
#   - proxy_ip/port   : NCCL coordinator address
#   - http_port       : this instance's HTTP port (for coordinator registration)
#   - send_type       : PUT_ASYNC for async non-blocking KV send

log "Starting Prefill-1 (GPU ${P1_GPU}, port ${P1_PORT}, kv_port ${P1_KV_PORT})..."
CUDA_VISIBLE_DEVICES=${P1_GPU} vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${P1_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL_P}" \
    --dtype float16 \
    --trust-remote-code \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"${P1_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${P1_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "${RESULTS_DIR}/prefill1.log" 2>&1 &
PIDS+=($!)

log "Starting Prefill-2 (GPU ${P2_GPU}, port ${P2_PORT}, kv_port ${P2_KV_PORT})..."
CUDA_VISIBLE_DEVICES=${P2_GPU} vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${P2_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL_P}" \
    --dtype float16 \
    --trust-remote-code \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"${P2_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${P2_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "${RESULTS_DIR}/prefill2.log" 2>&1 &
PIDS+=($!)

# =============================================================================
# Step 4: Start Decode servers
# =============================================================================
log "Starting Decode-1 (GPU ${D1_GPU}, port ${D1_PORT}, kv_port ${D1_KV_PORT})..."
CUDA_VISIBLE_DEVICES=${D1_GPU} vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${D1_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL_D}" \
    --dtype float16 \
    --trust-remote-code \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"${D1_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${D1_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "${RESULTS_DIR}/decode1.log" 2>&1 &
PIDS+=($!)

log "Starting Decode-2 (GPU ${D2_GPU}, port ${D2_PORT}, kv_port ${D2_KV_PORT})..."
CUDA_VISIBLE_DEVICES=${D2_GPU} vllm serve "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --host 0.0.0.0 \
    --port "${D2_PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL_D}" \
    --dtype float16 \
    --trust-remote-code \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"${D2_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${D2_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "${RESULTS_DIR}/decode2.log" 2>&1 &
PIDS+=($!)

# =============================================================================
# Step 5: Wait for all P/D servers
# =============================================================================
wait_for_server "${P1_PORT}" "Prefill-1"
wait_for_server "${P2_PORT}" "Prefill-2"
wait_for_server "${D1_PORT}" "Decode-1"
wait_for_server "${D2_PORT}" "Decode-2"
wait_for_server "${SERVICE_PORT}" "xpyd-proxy" "tcp"

log "All servers ready. Starting benchmark..."
echo ""

# =============================================================================
# Step 6: Run benchmark
# =============================================================================
${PYTHON} "${SCRIPT_DIR}/benchmark_serving_multi_turn.py" \
    --model "${MODEL_PATH}" \
    --served-model-name "${SERVED_MODEL_NAME}" \
    --url "http://localhost:${SERVICE_PORT}" \
    --input-file "${CONVERTED_DATA}" \
    --num-clients "${NUM_CLIENTS}" \
    --max-active-conversations "${MAX_ACTIVE_CONVERSATIONS}" \
    --request-rate "${REQUEST_RATE}" \
    --warmup-step \
    --stats-json-output "${RESULTS_JSON}" \
    --seed "${SEED}" \
    2>&1 | tee "${RESULTS_DIR}/benchmark_${TIMESTAMP}.log"

echo ""
log "=== Benchmark complete ==="
log "Results JSON : ${RESULTS_JSON}"
log "Logs dir     : ${RESULTS_DIR}/"
log "  prefill1.log / prefill2.log"
log "  decode1.log  / decode2.log"
log "  xpyd_proxy.log"
