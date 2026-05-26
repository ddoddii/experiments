# run_bfcl_direct.py (수정판)
import json, requests, time, csv
from pathlib import Path

VLLM_URL = "http://localhost:8000/v1/chat/completions"
MODEL = "/home/uhmturks/hf_models/Llama-3.1-8B-Instruct"

# v4 데이터 경로 (이미 존재)
DATA_FILE = Path("/home/uhmturks/experiments/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_base.json")
FUNC_DOC_DIR = Path("/home/uhmturks/experiments/gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/multi_turn_func_doc")

# tool schema 로드 (func_doc 디렉토리에서)
def load_tools(involved_classes):
    tools = []
    for cls in involved_classes:
        fname = FUNC_DOC_DIR / f"{cls}.json"
        if fname.exists():
            with open(fname) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        schema = json.loads(line)
                        tools.append({
                            "type": "function",
                            "function": {
                                "name": schema["name"],
                                "description": schema["description"],
                                "parameters": schema["parameters"],
                            }
                        })
    return tools

# vLLM 메트릭 수집
def get_vllm_metrics():
    try:
        r = requests.get("http://localhost:8000/metrics", timeout=2).text
        metrics = {}
        for line in r.splitlines():
            if line.startswith("#"): continue
            if "gpu_cache_usage_perc" in line:
                metrics["cache_usage"] = float(line.split()[-1])
            if "prefix_cache_hit_rate" in line:
                metrics["prefix_hit_rate"] = float(line.split()[-1])
            if "num_requests_waiting" in line:
                metrics["waiting"] = float(line.split()[-1])
        return metrics
    except:
        return {}

# 데이터 로드
print(f"Loading: {DATA_FILE}")
test_cases = []
with open(DATA_FILE) as f:
    for line in f:
        line = line.strip()
        if line:
            test_cases.append(json.loads(line))

print(f"Total test cases: {len(test_cases)}")
print(f"First case keys: {list(test_cases[0].keys())}")
print(f"Sample: {json.dumps(test_cases[0], indent=2)[:500]}")