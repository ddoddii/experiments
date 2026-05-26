import json
import os
import time
import requests
from tqdm import tqdm

ROUTER_URL = "http://127.0.0.1:8000/v1/chat/completions"
MODEL = "meta-llama/Llama-3.1-8B-Instruct"
CONFIG = os.environ.get("CONFIG", "2P_2D_sglang")

CLASS_TO_FILE = {
    "GorillaFileSystem": "multi_turn_func_doc/gorilla_file_system.json",
    "TicketAPI":         "multi_turn_func_doc/ticket_api.json",
    "MessageAPI":        "multi_turn_func_doc/message_api.json",
    "MathAPI":           "multi_turn_func_doc/math_api.json",
    "TradingBot":        "multi_turn_func_doc/trading_bot.json",
    "TwitterAPI":        "multi_turn_func_doc/posting_api.json",
    "TravelAPI":         "multi_turn_func_doc/travel_booking.json",
    "VehicleControlAPI": "multi_turn_func_doc/vehicle_control.json",
}

TYPE_MAP = {
    "float": "number",
    "integer": "integer",  # 이건 유효하긴 한데 명시
    "dict": "object",
    "list": "array",
    "tuple": "array",
    "str": "string",
    "bool": "boolean",
    "none": "null",
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

func_docs = {cls: load_func_doc(path) for cls, path in CLASS_TO_FILE.items()}
items = [json.loads(l) for l in open("data/BFCL_v3_multi_turn_base.json")]
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
        tqdm.write(f"  [turn {turn_idx}] user: {user_msg['content'][:80]}...")

        payload = {
            "model": MODEL,
            "messages": conversation,
            "tools": tools,
            "tool_choice": "auto",
            "max_tokens": 512,
            "stream": True,
        }

        try:
            t_request = time.perf_counter()
            resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=120)
            resp.raise_for_status()

            t_first_token = None
            t_last_token = None
            token_count = 0
            assistant_content = ""
            tool_calls_map = {}  # index → accumulated tool_call

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

            ttft = (t_first_token - t_request) if t_first_token else None
            e2e = (t_last_token - t_request) if t_last_token else None
            decode_time = (t_last_token - t_first_token) if (t_last_token and token_count > 1) else None
            tpot = (decode_time / (token_count - 1)) if (decode_time and token_count > 1) else None
            # per-turn throughput: decode 구간 기준 (prefill 제외)
            turn_throughput = ((token_count - 1) / decode_time) if (decode_time and token_count > 1) else None

            total_output_tokens += token_count

            tqdm.write(
                f"    → ttft={ttft:.3f}s  tpot={tpot:.4f}s  "
                f"tokens={token_count}  throughput={turn_throughput:.1f} tok/s"
                if (ttft and tpot and turn_throughput) else
                f"    → ttft={ttft}  tokens={token_count}"
            )

            turn_metrics.append({
                "turn": turn_idx,
                "user": user_msg["content"][:120],
                "assistant": assistant_content[:200] if assistant_content else None,
                "tool_calls": tool_calls_result if tool_calls_result else None,
                "ttft_s": round(ttft, 4) if ttft else None,
                "tpot_s": round(tpot, 4) if tpot else None,
                "output_tokens": token_count,
                "e2e_latency_s": round(e2e, 4) if e2e else None,
                "throughput_tok_per_s": round(turn_throughput, 2) if turn_throughput else None,
            })

            # Llama3 chat template은 tool_call 1개만 허용
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
    })

t_experiment_end = time.perf_counter()
total_wall_time = t_experiment_end - t_experiment_start

# 전체 요약
valid = [r for r in results if r.get("avg_ttft_s")]
summary = {
    "config": CONFIG,
    "total_items": len(results),
    "success_items": len(valid),
    "error_items": len(results) - len(valid),
    "total_output_tokens": total_output_tokens,
    "total_wall_time_s": round(total_wall_time, 2),
    "overall_throughput_tok_per_s": round(total_output_tokens / total_wall_time, 2),
    "avg_ttft_s": round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4) if valid else None,
    "avg_tpot_s": round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / len(valid), 4) if valid else None,
    "avg_throughput_tok_per_s": round(sum(r["avg_throughput_tok_per_s"] for r in valid if r["avg_throughput_tok_per_s"]) / len(valid), 2) if valid else None,
}

output = {"summary": summary, "results": results}

shm_path = f"/dev/shm/bfcl_multiturn_results_{CONFIG}.json"
out_path = f"results/bfcl_multiturn_results_{CONFIG}.json"
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
print(f"결과 저장: {out_path}")