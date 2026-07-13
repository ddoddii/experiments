#!/usr/bin/env python3
"""
SGLang BFCL v3 multi-turn 벤치마크 — 동시성(concurrent) 버전.

Phase 0 "압박(pressure)" 실험용. 여러 대화(item)를 동시에 처리해 GPU radix
cache에 eviction 압박을 준다. 이때 tool-call 유휴시간 동안 다른 세션이 GPU를
채워 내 prefix를 밀어내는 상황 = idle P-node parking의 동기를 재현한다.

- 대화 내부(turn)는 순차(다음 턴이 이전 턴 출력·tool에 의존).
- 대화 간(item)은 CONCURRENCY개까지 동시 실행.

환경변수:
  CONFIG          결과 파일 태그 (기본 concurrent)
  CONCURRENCY     동시 실행 대화 수 (기본 16)
  MAX_ITEMS       처리할 item 수 제한 (기본 전체 200)
  ROUTER_URL      (기본 http://127.0.0.1:8000/v1/chat/completions)

출력 스키마는 순차 버전과 동일 → phase0_analyze.py가 그대로 처리.
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests
from tqdm import tqdm

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
CONFIG = os.environ.get("CONFIG", "concurrent")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "16"))
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))  # 0 = 전체
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TIMEOUT = int(os.environ.get("TIMEOUT", "120"))
# tool-call 유휴시간(think-time) 모사. 턴 사이에 이 시간만큼 쉰다.
# >0이면 낮은 동시성으로도 "유휴 중 타세션이 내 prefix를 evict"하는 상황이 생겨,
# radix(재계산) vs hicache(fetch)의 TTFT 차이가 queueing 없이 드러난다.
TOOL_DELAY = float(os.environ.get("TOOL_DELAY", "0"))

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
    "float": "number", "integer": "integer", "dict": "object", "list": "array",
    "tuple": "array", "str": "string", "bool": "boolean", "none": "null",
}


def dict_to_object(obj):
    if isinstance(obj, dict):
        if "type" in obj and isinstance(obj["type"], str):
            obj["type"] = TYPE_MAP.get(obj["type"], obj["type"])
        return {k: dict_to_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dict_to_object(v) for v in obj]
    return obj


def load_func_doc(path):
    with open(path) as f:
        content = f.read().strip()
    docs = json.loads(content) if content.startswith("[") else [
        json.loads(l) for l in content.splitlines() if l.strip()]
    cleaned = []
    for doc in docs:
        doc.pop("response", None)
        cleaned.append(dict_to_object(doc))
    return cleaned


func_docs = {cls: load_func_doc(path) for cls, path in CLASS_TO_FILE.items()}
items = [json.loads(l) for l in open("data/BFCL_v3_multi_turn_base.json")]
if MAX_ITEMS > 0:
    items = items[:MAX_ITEMS]


def run_turn(conversation, tools):
    """한 턴 요청 → (metrics dict, assistant_content, tool_calls)."""
    payload = {
        "model": MODEL, "messages": conversation, "tools": tools,
        "tool_choice": "auto", "max_tokens": MAX_TOKENS, "stream": True,
    }
    t_request = time.perf_counter()
    resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    t_first = t_last = None
    token_count = 0
    assistant_content = ""
    tool_calls_map = {}
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
        if t_first is None and has_content:
            t_first = time.perf_counter()
        if has_content:
            token_count += 1
            t_last = time.perf_counter()
        if delta.get("content"):
            assistant_content += delta["content"]
        for tc in delta.get("tool_calls") or []:
            idx = tc.get("index", 0)
            if idx not in tool_calls_map:
                tool_calls_map[idx] = {
                    "id": tc.get("id", ""), "type": tc.get("type", "function"),
                    "function": {"name": tc.get("function", {}).get("name", ""),
                                 "arguments": tc.get("function", {}).get("arguments", "")},
                }
            else:
                tool_calls_map[idx]["function"]["arguments"] += tc.get("function", {}).get("arguments", "")

    tool_calls = [tool_calls_map[i] for i in sorted(tool_calls_map)] if tool_calls_map else []
    ttft = (t_first - t_request) if t_first else None
    e2e = (t_last - t_request) if t_last else None
    decode_time = (t_last - t_first) if (t_last and token_count > 1) else None
    tpot = (decode_time / (token_count - 1)) if (decode_time and token_count > 1) else None
    tput = ((token_count - 1) / decode_time) if (decode_time and token_count > 1) else None
    return ({
        "ttft_s": round(ttft, 4) if ttft else None,
        "tpot_s": round(tpot, 4) if tpot else None,
        "output_tokens": token_count,
        "e2e_latency_s": round(e2e, 4) if e2e else None,
        "throughput_tok_per_s": round(tput, 2) if tput else None,
    }, assistant_content, tool_calls)


def process_item(item):
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
    for turn_idx, turn in enumerate(item["question"]):
        # tool-call 유휴시간 모사: 이전 턴(assistant/tool) 이후 다음 유저 턴까지의 gap.
        if TOOL_DELAY > 0 and turn_idx > 0:
            time.sleep(TOOL_DELAY)
        conversation.append(turn[0])
        try:
            m, assistant_content, tool_calls = run_turn(conversation, tools)
            m["turn"] = turn_idx
            turn_metrics.append(m)
            conversation.append({
                "role": "assistant",
                "content": assistant_content or None,
                "tool_calls": tool_calls[:1] if tool_calls else None,
            })
        except Exception as e:  # noqa: BLE001
            turn_metrics.append({"turn": turn_idx, "error": str(e)})
            break

    valid = [t for t in turn_metrics if t.get("ttft_s")]
    return {
        "id": item["id"],
        "num_turns": len(item["question"]),
        "turns": turn_metrics,
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(t["tpot_s"] for t in valid if t.get("tpot_s")) / max(1, sum(1 for t in valid if t.get("tpot_s"))), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(t["throughput_tok_per_s"] for t in valid if t.get("throughput_tok_per_s")) / max(1, sum(1 for t in valid if t.get("throughput_tok_per_s"))), 2) if valid else None,
        "total_output_tokens": sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens")),
    }


def main():
    print(f"[concurrent] CONFIG={CONFIG} CONCURRENCY={CONCURRENCY} items={len(items)} url={ROUTER_URL}")
    results = []
    lock = threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(process_item, it): it for it in items}
        for fut in tqdm(as_completed(futs), total=len(futs), desc="items"):
            r = fut.result()
            with lock:
                results.append(r)
    wall = time.perf_counter() - t0

    total_out = sum(r["total_output_tokens"] for r in results)
    valid = [r for r in results if r.get("avg_ttft_s")]
    summary = {
        "config": CONFIG,
        "concurrency": CONCURRENCY,
        "tool_delay_s": TOOL_DELAY,
        "total_items": len(results),
        "success_items": len(valid),
        "error_items": len(results) - len(valid),
        "total_output_tokens": total_out,
        "total_wall_time_s": round(wall, 2),
        "overall_throughput_tok_per_s": round(total_out / wall, 2) if wall else None,
        "avg_ttft_s": round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / len(valid), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(r["avg_throughput_tok_per_s"] for r in valid if r["avg_throughput_tok_per_s"]) / len(valid), 2) if valid else None,
    }
    output = {"summary": summary, "results": results}
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{CONFIG}.json"  # CONFIG가 실험 이름 (데이터/구성 반영)
    json.dump(output, open(out_path, "w"), indent=2, ensure_ascii=False)

    print(f"\n{'='*60}")
    print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개  (C={CONCURRENCY})")
    print(f"전체 소요시간: {wall:.2f}s  overall throughput: {summary['overall_throughput_tok_per_s']} tok/s")
    print(f"평균 TTFT: {summary['avg_ttft_s']}s  평균 TPOT: {summary['avg_tpot_s']}s")
    print(f"결과 저장: {out_path}")


if __name__ == "__main__":
    main()
