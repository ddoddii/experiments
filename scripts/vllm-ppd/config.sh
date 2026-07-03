#!/bin/bash
# ============================================================================
# Unified Configuration for vLLM PPD Servers
# ============================================================================
# All model and path configurations should be defined here.
# Other scripts should source this file.
#
# Environment variable overrides (for model scaling experiments):
# - MODEL_PATH: Override model path before sourcing this file
# - MAX_MODEL_LEN: Override max model length
# - GPU_MEMORY_UTILIZATION: Override GPU memory utilization
# - QUANTIZATION: vLLM quantization scheme (default: fp8 for 27B on A6000)
# - TOOL_CALL_PARSER: vLLM tool call parser (default: hermes for Qwen)
# ============================================================================

# Model path - set MODEL_PATH env var before sourcing this file
# Default: Qwen3.6-27B (requires fp8 quantization to fit in 49GB VRAM)
export MODEL_PATH="${MODEL_PATH:-/home/uhmturks/hf_models/Qwen3-14B}"

# Model name for pkill (derived from MODEL_PATH if not set)
export MODEL_NAME="${MODEL_NAME:-Qwen3-14B}"

# Model settings (support environment variable override)
export MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
export GPU_MEMORY_UTILIZATION="${GPU_MEMORY_UTILIZATION:-0.85}"

# Quantization: fp8 lets Qwen3.6-27B (~54GB BF16) fit in a single 49GB A6000
# Set to "" to disable (e.g., for smaller models that fit without quantization)
# export QUANTIZATION="${QUANTIZATION:-fp8}"

# Tool call parser: hermes for Qwen models; llama3_json for Llama-3.x
export TOOL_CALL_PARSER="${TOOL_CALL_PARSER:-hermes}"

# Reduce CUDA allocator fragmentation (prevents OOM on logits.sort() when KV cache is near-full)
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"

# Project directory
export PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

# Log directory
export LOG_DIR="${PROJECT_DIR}/logs"

# Standard ports
export PROXY_HTTP_PORT=10001
export PROXY_ZMQ_PORT=30001

# GPU server base ports
export PREFILL_BASE_PORT=8100
export DECODE_BASE_PORT=8200
export REPLICA_BASE_PORT=8300
