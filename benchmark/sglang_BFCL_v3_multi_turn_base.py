import json
import os
import re
import time
import requests
from tqdm import tqdm

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

ROUTER_URL = os.environ.get("SGLANG_URL", "http://127.0.0.1:8000/v1/chat/completions")
# SGLang은 /v1/models 에서 모델명을 확인: curl http://localhost:30000/v1/models
# 보통 HF 모델 ID (meta-llama/Llama-3.1-8B-Instruct) 로 등록됨
MODEL  = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CONFIG = os.environ.get("CONFIG", "sglang_2p2d")
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TIMEOUT    = int(os.environ.get("TIMEOUT", "120"))

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "http://localhost:9091")

CLASS_TO_FILE = {
    "GorillaFileSystem": _p("data/multi_turn_func_doc/gorilla_file_system.json"),
    "TicketAPI":         _p("data/multi_turn_func_doc/ticket_api.json"),
    "MessageAPI":        _p("data/multi_turn_func_doc/message_api.json"),
    "MathAPI":           _p("data/multi_turn_func_doc/math_api.json"),
    "TradingBot":        _p("data/multi_turn_func_doc/trading_bot.json"),
    "TwitterAPI":        _p("data/multi_turn_func_doc/posting_api.json"),
    "TravelAPI":         _p("data/multi_turn_func_doc/travel_booking.json"),
    "VehicleControlAPI": _p("data/multi_turn_func_doc/vehicle_control.json"),
}

TYPE_MAP = {
    "float":   "number",
    "integer": "integer",
    "dict":    "object",
    "list":    "array",
    "tuple":   "array",
    "str":     "string",
    "bool":    "boolean",
    "none":    "null",
}

def dict_to_object(obj):
    if isinstance(obj, dict):
        if "type" in obj and isinstance(obj["type"], str):
            obj["type"] = TYPE_MAP.get(obj["type"], obj["type"])
        return {k: dict_to_object(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [dict_to_object(v) for v in obj]
    return obj

def load_func_doc(path):
    with open(path) as f:
        content = f.read().strip()
    docs = json.loads(content) if content.startswith("[") else [json.loads(l) for l in content.splitlines() if l.strip()]
    cleaned = []
    for doc in docs:
        doc.pop("response", None)
        doc = dict_to_object(doc)
        cleaned.append(doc)
    return cleaned

def push_metric(metric: str, value: float, labels: dict):
    """Pushgateway에 단일 gauge 메트릭 전송."""
    job = labels.pop("job", CONFIG)
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    body = f"# TYPE {metric} gauge\n{metric}{{{label_str}}} {value}\n"
    try:
        requests.post(f"{PUSHGATEWAY_URL}/metrics/job/{job}", data=body, timeout=2)
    except Exception:
        pass

_PYTHON_TAG_RE = re.compile(r'<\|python_tag\|>(.*)', re.DOTALL)

def _parse_python_tag(content: str) -> dict | None:
    """Parse Llama3 <|python_tag|> tool call from raw text content (fallback when server has no tool parser)."""
    m = _PYTHON_TAG_RE.search(content)
    if not m:
        return None
    try:
        raw = m.group(1).strip()
        data = json.loads(raw)
        name = data.get("name", "")
        params = data.get("parameters", data.get("arguments", {}))
        return {
            "id": f"call_{name}",
            "type": "function",
            "function": {"name": name, "arguments": json.dumps(params)},
        }
    except Exception:
        return None

# ── SGLang hicache 모니터링 (exporter가 떠있으면 자동 수집) ──────────────────
from sglang_hicache_exporter import SGLangHiCachePoller

poller = SGLangHiCachePoller()
poller.start()

# ── 데이터 로드 ──────────────────────────────────────────────────────────────
func_docs = {cls: load_func_doc(path) for cls, path in CLASS_TO_FILE.items()}
items = [json.loads(l) for l in open(_p("data/BFCL_v3_multi_turn_base.json"))]
results = []

t_experiment_start = time.perf_counter()
total_output_tokens = 0

for item in tqdm(items, desc="items"):
    tools = []
    for cls in item.get("involved_classes", []):
        for func in func_docs.get(cls, []):
            tools.append({"type": "function", "function": func})

    system_content = (
        "You are a helpful assistant with access to the following tools. "
        "Use them to complete the user's requests.\n\n"
        f"Environment state:\n{json.dumps(item['initial_config'], indent=2)}"
    )
    conversation = [{"role": "system", "content": system_content}]
    turn_metrics = []

    tqdm.write(f"\n{'='*60}")
    tqdm.write(f"[{item['id']}] turns={len(item['question'])} classes={item.get('involved_classes')}")

    for turn_idx, turn in enumerate(item["question"]):
        user_msg = turn[0]
        conversation.append(user_msg)
        ctx_chars = sum(len(str(m.get("content", ""))) for m in conversation)
        tqdm.write(f"  [turn {turn_idx}] ctx={ctx_chars}ch  user: {user_msg['content'][:80]}...")

        payload = {
            "model": MODEL,
            "messages": conversation,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": MAX_TOKENS,
            "temperature": 0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }

        try:
            t_request = time.perf_counter()
            resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
            resp.raise_for_status()

            t_first_token = None
            t_last_token = None
            token_count = 0
            assistant_content = ""
            tool_calls_map = {}
            server_completion_tokens = None

            for line in resp.iter_lines():
                if not line:
                    continue
                line = line.decode("utf-8")
                if not line.startswith("data: "):
                    continue
                data = line[len("data: "):]
                if data == "[DONE]":
                    break

                chunk = json.loads(data)
                if chunk.get("usage"):
                    server_completion_tokens = chunk["usage"].get("completion_tokens")
                if not chunk.get("choices"):
                    continue
                delta = chunk["choices"][0]["delta"]
                has_content = bool(delta.get("content") or delta.get("tool_calls"))

                if t_first_token is None and has_content:
                    t_first_token = time.perf_counter()
                if has_content:
                    token_count += 1
                    t_last_token = time.perf_counter()
                if delta.get("content"):
                    assistant_content += delta["content"]
                if delta.get("tool_calls"):
                    for tc in delta["tool_calls"]:
                        idx = tc.get("index", 0)
                        if idx not in tool_calls_map:
                            tool_calls_map[idx] = {
                                "id": tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name": tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", ""),
                                },
                            }
                        else:
                            tool_calls_map[idx]["function"]["arguments"] += tc.get("function", {}).get("arguments", "")

            tool_calls_result = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else []

            # Fallback: parse <|python_tag|> from text content if no structured tool_calls
            if not tool_calls_result and "<|python_tag|>" in assistant_content:
                tc = _parse_python_tag(assistant_content)
                if tc:
                    tool_calls_result = [tc]
                    assistant_content = ""

            actual_tokens = server_completion_tokens if server_completion_tokens is not None else token_count

            ttft = (t_first_token - t_request) if t_first_token else None
            e2e  = (t_last_token  - t_request) if t_last_token  else None
            decode_time = (t_last_token - t_first_token) if (t_last_token and token_count > 1) else None
            tpot = (decode_time / (actual_tokens - 1)) if (decode_time and actual_tokens > 1) else None
            turn_throughput = ((actual_tokens - 1) / decode_time) if (decode_time and actual_tokens > 1) else None

            total_output_tokens += actual_tokens

            tqdm.write(
                f"    → ttft={ttft:.3f}s  tpot={tpot:.4f}s  "
                f"tokens={actual_tokens}  throughput={turn_throughput:.1f} tok/s"
                if (ttft and tpot and turn_throughput) else
                f"    → ttft={ttft}  tokens={actual_tokens}"
            )

            # Pushgateway에 per-turn 지표 전송
            lbl = {"config": CONFIG, "turn": str(turn_idx), "item_id": item["id"]}
            if ttft:
                push_metric("bfcl_turn_ttft_seconds", ttft, {**lbl})
            if tpot:
                push_metric("bfcl_turn_tpot_seconds", tpot, {**lbl})
            push_metric("bfcl_turn_context_chars", ctx_chars, {**lbl})

            turn_metrics.append({
                "turn": turn_idx,
                "user": user_msg["content"][:120],
                "assistant": assistant_content[:200] if assistant_content else None,
                "tool_calls": tool_calls_result if tool_calls_result else None,
                "ttft_s": round(ttft, 4) if ttft else None,
                "tpot_s": round(tpot, 4) if tpot else None,
                "output_tokens": actual_tokens,
                "e2e_latency_s": round(e2e, 4) if e2e else None,
                "throughput_tok_per_s": round(turn_throughput, 2) if turn_throughput else None,
                "ctx_chars": ctx_chars,
            })

            conversation.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls_result[:1] if tool_calls_result else None,
            })

        except Exception as e:
            tqdm.write(f"    → ERROR: {e}")
            turn_metrics.append({"turn": turn_idx, "error": str(e)})
            break

    valid_turns = [t for t in turn_metrics if t.get("ttft_s")]
    results.append({
        "id": item["id"],
        "num_turns": len(item["question"]),
        "turns": turn_metrics,
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid_turns) / len(valid_turns), 4) if valid_turns else None,
        "avg_tpot_s": round(sum(t["tpot_s"] for t in valid_turns if t.get("tpot_s")) / max(1, sum(1 for t in valid_turns if t.get("tpot_s"))), 4) if valid_turns else None,
        "avg_throughput_tok_per_s": round(sum(t["throughput_tok_per_s"] for t in valid_turns if t.get("throughput_tok_per_s")) / max(1, sum(1 for t in valid_turns if t.get("throughput_tok_per_s"))), 2) if valid_turns else None,
        "total_output_tokens": sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens")),
        "ttft_by_turn": {str(t["turn"]): t["ttft_s"] for t in valid_turns},
    })

    push_metric("bfcl_items_completed", len(results), {"config": CONFIG})

t_experiment_end = time.perf_counter()
total_wall_time = t_experiment_end - t_experiment_start

poller.stop()
kv_stats = poller.stats()

valid = [r for r in results if r.get("avg_ttft_s")]
summary = {
    "config": CONFIG,
    "model": MODEL,
    "total_items": len(results),
    "success_items": len(valid),
    "error_items": len(results) - len(valid),
    "total_output_tokens": total_output_tokens,
    "total_wall_time_s": round(total_wall_time, 2),
    "overall_throughput_tok_per_s": round(total_output_tokens / total_wall_time, 2),
    "avg_ttft_s": round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4) if valid else None,
    "avg_tpot_s": round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / len(valid), 4) if valid else None,
    "avg_throughput_tok_per_s": round(sum(r["avg_throughput_tok_per_s"] for r in valid if r["avg_throughput_tok_per_s"]) / len(valid), 2) if valid else None,
    "kv_cache_stats": kv_stats,  # token_usage per instance
}

output = {"summary": summary, "results": results}

os.makedirs(_p("results"), exist_ok=True)
shm_path = f"/dev/shm/bfcl_multiturn_results_{CONFIG}.json"
out_path  = _p(f"results/bfcl_multiturn_results_{CONFIG}.json")
with open(shm_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)

import shutil
shutil.copy(shm_path, out_path)

print(f"\n{'='*60}")
print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개")
print(f"총 출력 토큰: {total_output_tokens}")
print(f"전체 소요시간: {total_wall_time:.2f}s")
print(f"전체 throughput: {summary['overall_throughput_tok_per_s']} tok/s")
print(f"평균 TTFT: {summary['avg_ttft_s']}s")
print(f"평균 TPOT: {summary['avg_tpot_s']}s")
print(f"평균 per-request throughput: {summary['avg_throughput_tok_per_s']} tok/s")
print(f"\n[SGLang hicache KV stats]")
for inst, s in kv_stats.items():
    if s:
        print(f"  {inst}: token_usage max={s['token_usage']['max']:.3f}  "
              f"mean={s['token_usage']['mean']:.3f}  samples={s['token_usage']['samples']}")
print(f"\n결과 저장: {out_path}")
