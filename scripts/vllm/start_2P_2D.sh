#!/bin/bash
# =============================================================================
# Start Script for vLLM 2P2D Disaggregated Prefill
#
#   GPU0: Prefill-1 (port 8100)
#   GPU1: Prefill-2 (port 8101)
#   GPU2: Decode-1  (port 8200)
#   GPU3: Decode-2  (port 8201)
#   Proxy: xpyd (ZMQ 30001, HTTP 10001)
#
# Usage: bash scripts/vllm/start_2P_2D.sh
# =============================================================================

set -eo pipefail

source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate vllm-source

export CC=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-gcc
export CXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export CUDAHOSTCXX=$CONDA_PREFIX/bin/x86_64-conda-linux-gnu-g++
export LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:$CONDA_PREFIX/lib:$LIBRARY_PATH


# =============================================================================
# Configuration
# =============================================================================

export MODEL_PATH=${MODEL_PATH:-/home/uhmturks/hf_models/Qwen3.6-27B}
SERVED_MODEL_NAME=${SERVED_MODEL_NAME:-Qwen}
HOST_IP=${HOST_IP:-127.0.0.1}
TOOL_CALL_PARSER=${TOOL_CALL_PARSER:-hermes}
# TP=4 splits the model across 4 GPUs (~13.5GB/GPU for 27B BF16), no quantization needed
QUANTIZATION=${QUANTIZATION:-}
DTYPE=${DTYPE:-bfloat16}

XPYD_PROXY="/home/uhmturks/vllm-source/examples/disaggregated/p2p_nccl_xpyd/disagg_proxy_p2p_nccl_xpyd.py"

LOG_DIR="/home/uhmturks/experiments/logs/vllm/2P_2D"
mkdir -p "$LOG_DIR"
rm -f "$LOG_DIR"/*.log 2>/dev/null || true

P1_GPU=0; P2_GPU=1; D1_GPU=2; D2_GPU=3
P1_PORT=8100; P2_PORT=8101; D1_PORT=8200; D2_PORT=8201
SERVICE_PORT=10001
NCCL_PROXY_PORT=30001
P1_KV_PORT=21001; P2_KV_PORT=21002; D1_KV_PORT=22001; D2_KV_PORT=22002
MAX_MODEL_LEN=8192
GPU_MEM_UTIL=0.85
SERVER_TIMEOUT=600

CONDA_PYTHON="/home/uhmturks/miniconda3/envs/vllm-source/bin/python"

# Kill leftover processes on target ports
for _port in $NCCL_PROXY_PORT $SERVICE_PORT $P1_PORT $P2_PORT $D1_PORT $D2_PORT \
             $P1_KV_PORT $P2_KV_PORT $D1_KV_PORT $D2_KV_PORT; do
    pids=$(lsof -ti:"${_port}" 2>/dev/null) || true
    [[ -n "$pids" ]] && echo "$pids" | xargs kill -9 2>/dev/null || true
done
sleep 2

wait_for_server() {
    local port=$1 label=${2:-server} deadline=$((SECONDS + SERVER_TIMEOUT))
    echo -n "  Waiting for $label (port $port)..."
    while (( SECONDS < deadline )); do
        if curl -sf "http://localhost:${port}/v1/models" > /dev/null 2>&1; then
            echo " OK"; return 0
        fi
        sleep 3; echo -n "."
    done
    echo " TIMEOUT"; exit 1
}

echo "=============================================="
echo "Starting vLLM 2P2D Disaggregated"
echo "Model: $(basename $MODEL_PATH)  dtype: $DTYPE  quantization: ${QUANTIZATION:-none}  parser: $TOOL_CALL_PARSER"
echo "=============================================="

echo "[1/5] Starting xpyd proxy (ZMQ=$NCCL_PROXY_PORT, HTTP=$SERVICE_PORT)..."
PROXY_PORT=${NCCL_PROXY_PORT} \
    "${CONDA_PYTHON}" "${XPYD_PROXY}" \
    > "$LOG_DIR/proxy.log" 2>&1 &
sleep 2

echo "[2/5] Starting Prefill-1 (GPU $P1_GPU, port $P1_PORT)..."
CUDA_VISIBLE_DEVICES=$P1_GPU "$CONDA_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 --port $P1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --dtype $DTYPE --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER \
    ${QUANTIZATION:+--quantization $QUANTIZATION} \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"${P1_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${P1_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "$LOG_DIR/prefill1.log" 2>&1 &

echo "[3/5] Starting Prefill-2 (GPU $P2_GPU, port $P2_PORT)..."
CUDA_VISIBLE_DEVICES=$P2_GPU "$CONDA_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 --port $P2_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --dtype $DTYPE --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER \
    ${QUANTIZATION:+--quantization $QUANTIZATION} \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_producer\",\"kv_buffer_size\":\"1e9\",\"kv_port\":\"${P2_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${P2_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "$LOG_DIR/prefill2.log" 2>&1 &

echo "[4/5] Starting Decode-1 (GPU $D1_GPU, port $D1_PORT)..."
CUDA_VISIBLE_DEVICES=$D1_GPU "$CONDA_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 --port $D1_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --dtype $DTYPE --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER \
    ${QUANTIZATION:+--quantization $QUANTIZATION} \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"${D1_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${D1_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "$LOG_DIR/decode1.log" 2>&1 &

echo "[5/5] Starting Decode-2 (GPU $D2_GPU, port $D2_PORT)..."
CUDA_VISIBLE_DEVICES=$D2_GPU "$CONDA_PYTHON" -m vllm.entrypoints.cli.main serve "$MODEL_PATH" \
    --served-model-name "$SERVED_MODEL_NAME" \
    --host 0.0.0.0 --port $D2_PORT \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization $GPU_MEM_UTIL \
    --dtype $DTYPE --trust-remote-code --enforce-eager \
    --enable-auto-tool-choice --tool-call-parser $TOOL_CALL_PARSER \
    ${QUANTIZATION:+--quantization $QUANTIZATION} \
    --kv-transfer-config \
    "{\"kv_connector\":\"P2pNcclConnector\",\"kv_role\":\"kv_consumer\",\"kv_buffer_size\":\"8e9\",\"kv_port\":\"${D2_KV_PORT}\",\"kv_connector_extra_config\":{\"proxy_ip\":\"${HOST_IP}\",\"proxy_port\":\"${NCCL_PROXY_PORT}\",\"http_ip\":\"${HOST_IP}\",\"http_port\":\"${D2_PORT}\",\"send_type\":\"PUT_ASYNC\",\"nccl_num_channels\":\"16\",\"mem_pool_size_gb\":\"8\"}}" \
    > "$LOG_DIR/decode2.log" 2>&1 &

echo ""
echo "Waiting for all servers to be ready..."
wait_for_server $P1_PORT "Prefill-1"
wait_for_server $P2_PORT "Prefill-2"
wait_for_server $D1_PORT "Decode-1"
wait_for_server $D2_PORT "Decode-2"

echo ""
echo "=============================================="
echo "vLLM 2P2D ready!"
echo "  P1: http://localhost:$P1_PORT  (GPU $P1_GPU)"
echo "  P2: http://localhost:$P2_PORT  (GPU $P2_GPU)"
echo "  D1: http://localhost:$D1_PORT  (GPU $D1_GPU)"
echo "  D2: http://localhost:$D2_PORT  (GPU $D2_GPU)"
echo "  Proxy: http://localhost:$SERVICE_PORT"
echo "  Logs: $LOG_DIR/"
echo "=============================================="
