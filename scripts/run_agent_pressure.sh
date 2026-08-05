#!/bin/bash
# Agent-stack host-capacity experiment, ONE config per invocation.
#
# Measures how much host DRAM the agent-serving stack still has while SGLang serves
# LLM traffic -- i.e. the Motivation claim that host-side KV offloading eats the
# capacity the agent stack needs for orchestration / scheduling / tool execution.
#
# Per config it runs, with steady LLM load on the server:
#   1. sys_mem_breakdown.py       -> where the host RAM went (anon pool vs page cache)
#   2. agent_host_pressure.py ramp -> admitted agent concurrency K_max + tool p99 curve
#   3. agent_host_pressure.py rag  -> page-cache eviction damage (read-only probe)
#
# The server must ALREADY be up in the configuration you want to label. Examples:
#   HICACHE_WRITE_POLICY=write_through                   ./scripts/sglang/start_2P_2D.sh
#   HICACHE_WRITE_POLICY=write_back                      ./scripts/sglang/start_2P_2D.sh
#   HICACHE_STORAGE_BACKEND=none                         ./scripts/sglang/start_2P_2D.sh
#   PARK_NO_HICACHE=1 IDLE_KV_PARKING=1           ./scripts/sglang/start_2P_2D.sh
#
# 사용:
#   LABEL=write_back ./scripts/run_agent_pressure.sh
#   LABEL=park TOOL_MEM_MB=1024 MAX_WORKERS=110 ./scripts/run_agent_pressure.sh
set -e
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate sglang
cd "$(dirname "$0")/.."

LABEL=${LABEL:?set LABEL=<config name>, e.g. LABEL=write_back}
OUT_DIR=${OUT_DIR:-results/agent}
TOOL_MEM_MB=${TOOL_MEM_MB:-1024}      # anonymous working set of one tool execution
MAX_WORKERS=${MAX_WORKERS:-110}       # ramp ceiling (110GB of tool WS at 1GB each)
HOLD=${HOLD:-6}                       # seconds per ramp step
FLOOR_GB=${FLOOR_GB:-8}                # OOM guard: abort before the kernel kills anything
LOAD_QPS=${LOAD_QPS:-4}                # LLM load kept on the server during the test
LOAD_CTX=${LOAD_CTX:-8000}
ROUTER=${ROUTER:-http://127.0.0.1:8000}
RAG_FILE=${RAG_FILE:-/home/uhmturks/hf_models/Llama-3.1-8B-Instruct/model-00001-of-00004.safetensors}
mkdir -p "$OUT_DIR"

echo "=== agent host pressure: $LABEL ==="
echo "  tool_mem=${TOOL_MEM_MB}MB max_workers=$MAX_WORKERS floor=${FLOOR_GB}GB llm_qps=$LOAD_QPS"

# --- background: memory breakdown sampler
python benchmark/sys_mem_breakdown.py --out "$OUT_DIR/mem_$LABEL.csv" --interval 1 &
SAMPLER=$!
# --- background: steady LLM load so HiCache actually holds + offloads its pool
python benchmark/qps_sweep.py --url "$ROUTER" --ctx-len "$LOAD_CTX" \
  --max-new-tokens 128 --qps "$LOAD_QPS" --duration 100000 --arms cached \
  --out "$OUT_DIR/load_$LABEL.png" > "$OUT_DIR/load_$LABEL.out" 2>&1 &
LOAD=$!
cleanup() { kill $SAMPLER $LOAD 2>/dev/null || true; }
trap cleanup EXIT
sleep 20   # let the load reach steady state (pool populated, cache warm)

# --- 1. page-cache damage probe FIRST (read-only; before we add anon pressure)
python benchmark/agent_host_pressure.py --mode rag --label "$LABEL" \
  --rag-file "$RAG_FILE" --rag-reads 20000 \
  --out "$OUT_DIR/rag_$LABEL.json" || echo "[warn] rag probe skipped"

# --- 2. capacity ramp
python benchmark/agent_host_pressure.py --mode ramp --label "$LABEL" \
  --tool-mem-mb "$TOOL_MEM_MB" --max-workers "$MAX_WORKERS" \
  --hold "$HOLD" --floor-gb "$FLOOR_GB" \
  --out "$OUT_DIR/ramp_$LABEL.json"

cleanup
echo
echo "=== done: $LABEL -> $OUT_DIR/{mem,ramp,rag}_$LABEL.* ==="
echo "after all configs, plot with:"
echo "  python benchmark/plot_agent_pressure.py --ramp \\"
echo "    write_through=$OUT_DIR/ramp_write_through.json \\"
echo "    write_back=$OUT_DIR/ramp_write_back.json \\"
echo "    L2_only=$OUT_DIR/ramp_L2_only.json \\"
echo "    park=$OUT_DIR/ramp_park.json \\"
echo "    --out $OUT_DIR/fig_agent_host_capacity.png"
