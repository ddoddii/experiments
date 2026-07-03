# vllm serve Qwen/Qwen3.6-27B     --enable-auto-tool-choice     --tool-call-parser hermes  --tensor-parallel-size 4 --served-model-name Qwen3.6-27B

# MODEL="Qwen/Qwen3.6-27B" vllm_BFCL_v3_multi_turn_base.py


TOOL_DELAY=2.0 CONCURRENCY=1 python benchmark/vllm_4gpu_BFCL_v3_multi_turn_concurrent.py