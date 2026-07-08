"""
BFCL v3 Multi-Turn Base Benchmark — vLLM-PPD 2P2D (Concurrent)
===============================================================
동시 요청 수(CONCURRENCY)를 조절해 KV cache 적재를 관찰하기 위한 병렬 버전.

  - 서로 다른 item들은 CONCURRENCY만큼 동시에 처리
  - 같은 item 내 turn들은 순서대로 처리 (앞 turn 결과가 다음 turn 입력)
  - KV cache, TPOT 등에서 동시성 효과를 관찰할 수 있음

Usage:
  # 동시 4개 item (기본)
  CONCURRENCY=4 python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py

  # 동시 8개 + Pushgateway
  CONCURRENCY=8 PUSHGATEWAY_URL=localhost:9091 \\
    CONFIG=vllm_ppd_c8 python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py

  # 다른 설정 재사용
  VLLM_URL=http://127.0.0.1:8000/v1/chat/completions \\
    MODEL=meta-llama/Llama-3.1-8B-Instruct-FC \\
    CONFIG=vllm_4gpu_c4 CONCURRENCY=4 \\
    python benchmark/vllmppd_BFCL_v3_multi_turn_concurrent.py
"""

import json
import os
import sys
import time
import threading
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from kv_cache_poller import KVCachePoller

# ─── Paths ─────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)

# ─── Config ────────────────────────────────────────────────────────────────
ROUTER_URL  = os.environ.get("VLLM_URL",  "http://127.0.0.1:10001/v1/chat/completions")
MODEL       = os.environ.get("MODEL",     "/home/uhmturks/hf_models/Qwen3.6-27B")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "4"))
CONFIG      = os.environ.get("CONFIG",    f"vllm_ppd_2p2d_c{CONCURRENCY}")
MAX_TOKENS  = int(os.environ.get("MAX_TOKENS",  "512"))
TIMEOUT     = int(os.environ.get("TIMEOUT",     "600"))
# Simulated tool execution delay between turns (seconds).
# Injects idle time to study KV tier placement under realistic agent workloads.
TOOL_DELAY  = float(os.environ.get("TOOL_DELAY", "0"))

PUSHGATEWAY_URL = os.environ.get("PUSHGATEWAY_URL", "")

# ─── Optional Pushgateway ───────────────────────────────────────────────────
HAS_PUSHGATEWAY = False
_pg_lock        = threading.Lock()   # serialize pushgateway calls across threads

if PUSHGATEWAY_URL:
    try:
        from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
        HAS_PUSHGATEWAY = True
        _pg_registry = CollectorRegistry()
        _pg_ttft     = Gauge("bfcl_turn_ttft_seconds",  "Client-side TTFT per turn",
                             ["config", "item_id", "turn"], registry=_pg_registry)
        _pg_tpot     = Gauge("bfcl_turn_tpot_seconds",  "Client-side TPOT per turn",
                             ["config", "item_id", "turn"], registry=_pg_registry)
        _pg_ctx      = Gauge("bfcl_turn_context_chars", "Conversation context length (chars)",
                             ["config", "item_id", "turn"], registry=_pg_registry)
        _pg_progress = Gauge("bfcl_items_completed",    "BFCL items completed so far",
                             ["config"],                  registry=_pg_registry)
        print(f"✓ Pushgateway enabled: {PUSHGATEWAY_URL}")
    except ImportError:
        print("⚠ prometheus_client not installed — Pushgateway disabled.")

def push_to_pg(item_id: str, turn_idx: int, ttft, tpot, ctx_chars: int, n_done: int):
    if not HAS_PUSHGATEWAY:
        return
    try:
        with _pg_lock:  # serialize: multiple threads must not push simultaneously
            if ttft is not None:
                _pg_ttft.labels(config=CONFIG, item_id=item_id, turn=str(turn_idx)).set(ttft)
            if tpot is not None:
                _pg_tpot.labels(config=CONFIG, item_id=item_id, turn=str(turn_idx)).set(tpot)
            _pg_ctx.labels(config=CONFIG, item_id=item_id, turn=str(turn_idx)).set(ctx_chars)
            _pg_progress.labels(config=CONFIG).set(n_done)
            push_to_gateway(PUSHGATEWAY_URL, job=f"bfcl_{CONFIG}", registry=_pg_registry)
    except Exception as e:
        tqdm.write(f"  [PG] push failed: {e}")

# ─── Data loading ───────────────────────────────────────────────────────────
def _p(rel):
    return os.path.join(PROJECT_ROOT, rel)

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
    "float": "number", "integer": "integer", "dict": "object",
    "list": "array",   "tuple": "array",     "str": "string",
    "bool": "boolean", "none": "null",
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
    docs = json.loads(content) if content.startswith("[") else \
           [json.loads(l) for l in content.splitlines() if l.strip()]
    return [dict_to_object({k: v for k, v in doc.items() if k != "response"}) for doc in docs]

func_docs = {cls: load_func_doc(path) for cls, path in CLASS_TO_FILE.items()}
items     = [json.loads(l) for l in open(_p("data/BFCL_v3_multi_turn_base.json"))]

print(f"Config      : {CONFIG}")
print(f"URL         : {ROUTER_URL}")
print(f"Model       : {MODEL}")
print(f"Items       : {len(items)}")
print(f"Concurrency : {CONCURRENCY}")
print(f"Tool delay  : {TOOL_DELAY}s per tool-call turn")
print(f"PushGW      : {PUSHGATEWAY_URL or 'disabled'}")
print("=" * 60)

# ─── Thread-safe shared state ───────────────────────────────────────────────
_results_lock       = threading.Lock()
_counter_lock       = threading.Lock()
_results            = []
_total_output_tokens = 0
_items_done          = 0

t_experiment_start = time.perf_counter()

# ─── KV cache background poller ────────────────────────────────────────────
kv_poller = KVCachePoller(interval=2.0)
kv_poller.start()

# ─── Per-item worker (runs in thread pool) ──────────────────────────────────
def process_item(item_idx: int, item: dict) -> dict:
    global _total_output_tokens, _items_done

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
    item_tokens  = 0

    tqdm.write(f"\n[{item['id']}] turns={len(item['question'])}  classes={item.get('involved_classes')}")

    for turn_idx, turn in enumerate(item["question"]):
        user_msg  = turn[0]
        conversation.append(user_msg)
        ctx_chars = sum(len(str(m.get("content", "") or "")) for m in conversation)

        tqdm.write(f"  [{item['id']} t{turn_idx}] ctx={ctx_chars}  user: {user_msg['content'][:60]}...")

        payload = {
            "model":       MODEL,
            "messages":    conversation,
            "tools":       tools,
            "tool_choice": "auto",
            "max_tokens":  MAX_TOKENS,
            "temperature": 0,
            "stream":      True,
            "stream_options": {"include_usage": True},
        }

        try:
            t_request     = time.perf_counter()
            resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
            resp.raise_for_status()

            t_first_token     = None
            t_last_token      = None
            token_count       = 0
            assistant_content = ""
            tool_calls_map    = {}
            server_completion_tokens = None

            for line in resp.iter_lines():
                if not line: continue
                line = line.decode("utf-8")
                if not line.startswith("data: "): continue
                data = line[len("data: "):]
                if data == "[DONE]": break

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
                                "id":   tc.get("id", ""),
                                "type": tc.get("type", "function"),
                                "function": {
                                    "name":      tc.get("function", {}).get("name", ""),
                                    "arguments": tc.get("function", {}).get("arguments", ""),
                                },
                            }
                        else:
                            tool_calls_map[idx]["function"]["arguments"] += \
                                tc.get("function", {}).get("arguments", "")

            tool_calls_result = [tool_calls_map[i] for i in sorted(tool_calls_map)] \
                                if tool_calls_map else []

            actual_tokens = server_completion_tokens if server_completion_tokens is not None else token_count

            ttft        = (t_first_token - t_request)           if t_first_token else None
            e2e         = (t_last_token  - t_request)           if t_last_token  else None
            decode_time = (t_last_token  - t_first_token)       if (t_last_token and token_count > 1) else None
            tpot        = (decode_time   / (actual_tokens - 1)) if (decode_time and actual_tokens > 1) else None
            turn_tput   = ((actual_tokens - 1) / decode_time)   if (decode_time and actual_tokens > 1) else None

            item_tokens += actual_tokens

            tqdm.write(
                f"    [{item['id']} t{turn_idx}] → ttft={ttft:.3f}s  tpot={tpot:.4f}s  "
                f"tokens={actual_tokens}  tput={turn_tput:.1f} tok/s"
                if (ttft and tpot and turn_tput)
                else f"    [{item['id']} t{turn_idx}] → ttft={ttft}  tokens={actual_tokens}"
            )

            turn_metrics.append({
                "turn":                 turn_idx,
                "user":                 user_msg["content"][:120],
                "assistant":            assistant_content[:200] if assistant_content else None,
                "tool_calls":           tool_calls_result if tool_calls_result else None,
                "ttft_s":               round(ttft,      4) if ttft      else None,
                "tpot_s":               round(tpot,      4) if tpot      else None,
                "output_tokens":        actual_tokens,
                "e2e_latency_s":        round(e2e,       4) if e2e       else None,
                "throughput_tok_per_s": round(turn_tput, 2) if turn_tput else None,
                "context_chars":        ctx_chars,
            })

            with _counter_lock:
                current_done = _items_done
            push_to_pg(item["id"], turn_idx, ttft, tpot, ctx_chars, current_done)

            conversation.append({
                "role":       "assistant",
                "content":    assistant_content or None,
                "tool_calls": tool_calls_result[:1] if tool_calls_result else None,
            })

            # Simulate tool execution time (studies KV idle across tiers)
            if TOOL_DELAY > 0 and tool_calls_result:
                time.sleep(TOOL_DELAY)

        except Exception as e:
            tqdm.write(f"    [{item['id']} t{turn_idx}] ERROR: {e}")
            turn_metrics.append({"turn": turn_idx, "error": str(e)})
            break

    # ─── Build result ────────────────────────────────────────────────────────
    valid_turns = [t for t in turn_metrics if t.get("ttft_s")]
    result = {
        "id":          item["id"],
        "num_turns":   len(item["question"]),
        "turns":       turn_metrics,
        "avg_ttft_s":  round(sum(t["ttft_s"] for t in valid_turns) / len(valid_turns), 4)
                       if valid_turns else None,
        "avg_tpot_s":  round(
                         sum(t["tpot_s"] for t in valid_turns if t.get("tpot_s")) /
                         max(1, sum(1 for t in valid_turns if t.get("tpot_s"))), 4)
                       if valid_turns else None,
        "avg_throughput": round(
                            sum(t["throughput_tok_per_s"] for t in valid_turns if t.get("throughput_tok_per_s")) /
                            max(1, sum(1 for t in valid_turns if t.get("throughput_tok_per_s"))), 2)
                          if valid_turns else None,
        "total_output_tokens": sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens")),
        "ttft_by_turn": {str(t["turn"]): t["ttft_s"] for t in valid_turns},
    }

    with _counter_lock:
        _total_output_tokens += item_tokens
        _items_done += 1

    with _results_lock:
        _results.append(result)

    return result


# ─── Run concurrent ─────────────────────────────────────────────────────────
pbar = tqdm(total=len(items), desc="items", position=0)

with ThreadPoolExecutor(max_workers=CONCURRENCY) as executor:
    futures = {
        executor.submit(process_item, idx, item): idx
        for idx, item in enumerate(items)
    }
    for future in as_completed(futures):
        try:
            future.result()
        except Exception as e:
            tqdm.write(f"[FATAL] item {futures[future]} failed: {e}")
        pbar.update(1)

pbar.close()

# ─── Stop KV cache poller ───────────────────────────────────────────────────
kv_poller.stop()
kv_stats = kv_poller.stats()

# ─── Final summary ──────────────────────────────────────────────────────────
t_experiment_end = time.perf_counter()
total_wall_time  = t_experiment_end - t_experiment_start

valid   = [r for r in _results if r.get("avg_ttft_s")]
summary = {
    "config":                       CONFIG,
    "model":                        MODEL,
    "concurrency":                  CONCURRENCY,
    "total_items":                  len(_results),
    "success_items":                len(valid),
    "error_items":                  len(_results) - len(valid),
    "total_output_tokens":          _total_output_tokens,
    "total_wall_time_s":            round(total_wall_time, 2),
    "overall_throughput_tok_per_s": round(_total_output_tokens / total_wall_time, 2),
    "avg_ttft_s":    round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4)             if valid else None,
    "avg_tpot_s":    round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / len(valid), 4) if valid else None,
    "avg_throughput_tok_per_s": round(
                        sum(r["avg_throughput"] for r in valid if r["avg_throughput"]) / len(valid), 2)
                     if valid else None,
    "tool_delay_s":             TOOL_DELAY,
    "kv_cache_per_gpu":         kv_stats,
}

output = {"summary": summary, "results": _results}

os.makedirs("results", exist_ok=True)
out_path = f"results/bfcl_multiturn_results_{CONFIG}.json"
shm_path = f"/dev/shm/bfcl_multiturn_results_{CONFIG}.json"
with open(shm_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
shutil.copy(shm_path, out_path)

print(f"\n{'='*60}")
print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개")
print(f"동시성       : {CONCURRENCY}")
print(f"총 출력 토큰  : {_total_output_tokens}")
print(f"전체 소요시간 : {total_wall_time:.2f}s")
print(f"전체 throughput: {summary['overall_throughput_tok_per_s']} tok/s")
print(f"평균 TTFT     : {summary['avg_ttft_s']}s")
print(f"평균 TPOT     : {summary['avg_tpot_s']}s")
print(f"평균 per-req throughput: {summary['avg_throughput_tok_per_s']} tok/s")
print(f"결과 저장     : {out_path}")
