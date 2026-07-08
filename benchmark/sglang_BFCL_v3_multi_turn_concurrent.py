"""
BFCL v3 Multi-Turn Base Benchmark — SGLang 2P2D (Concurrent)
=============================================================
동시 요청 수(CONCURRENCY)를 조절해 KV cache / hicache 적재를 관찰하기 위한 병렬 버전.

  - 서로 다른 item들은 CONCURRENCY만큼 동시에 처리
  - 같은 item 내 turn들은 순서대로 처리 (앞 turn 결과가 다음 turn 입력)
  - L1 GPU KV cache, L2 DRAM, L3 SSD 사용률이 명확하게 올라가는 것을 관찰 가능

Tool Delay + KV Flush 실험:
  서버 설정: --hicache-write-policy write_through (write_back은 disagg prefill SGLang 버그)
             --mem-fraction-static 0.5 (GPU KV pool ~9.25GB ≈ 56,500 tokens)
             write_through도 GPU LRU 유지함. flush 60k > 56.5k → eviction 보장.

  시나리오:
    1) tool call 반환 후 TOOL_DELAY_SEC sleep
    2) FLOOD_N × FLOOD_TOKENS flood request로 GPU KV pool 강제 overflow
       → sleeping conversation의 prefix가 LRU로 DRAM(L2)으로 evict
    3) 다음 turn TTFT에서 L2/L3 fetch overhead 측정

  기대 결과 (write_back 기준):
    delay=0s  → GPU hit      → TTFT ~0.5-0.8s
    delay+flush → DRAM fetch → TTFT ~1.5-2.0s  (+900ms overhead)

Usage:
  # 기본 (동시 4개, delay 없음)
  CONCURRENCY=4 python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py

  # tool delay 실험 (concurrent 8, 5초 delay)
  CONCURRENCY=8 TOOL_DELAY_SEC=5 \\
    python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py

  # delay sweep (별도 실행 3회)
  for D in 0 1 5; do
    CONCURRENCY=8 TOOL_DELAY_SEC=$D \\
      python benchmark/sglang_BFCL_v3_multi_turn_concurrent.py
  done

Environment variables:
  SGLANG_URL      SGLang router URL  (default: http://127.0.0.1:8000/v1/chat/completions)
  MODEL           model name         (default: /home/uhmturks/hf_models/Qwen3-14B)
  CONCURRENCY     parallel items     (default: 4)
  TOOL_DELAY_SEC  tool execution delay in seconds (default: 0)
                  applied after every model turn that contains a tool_call.
  CONFIG          result file tag    (default: sglang_2p2d_c{CONCURRENCY}_d{delay}s)
  MAX_TOKENS      max output tokens  (default: 512)
  TIMEOUT         per-request sec    (default: 600)
  PUSHGATEWAY_URL http://host:9091   (default: http://localhost:9091)
"""

import json
import os
import re
import sys
import time
import threading
import shutil
import requests
from concurrent.futures import ThreadPoolExecutor, as_completed
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sglang_hicache_exporter import SGLangHiCachePoller

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

# ─── Config ─────────────────────────────────────────────────────────────────
ROUTER_URL      = os.environ.get("SGLANG_URL",       "http://127.0.0.1:8000/v1/chat/completions")
MODEL           = os.environ.get("MODEL",             "/home/uhmturks/hf_models/Qwen3-14B")
CONCURRENCY     = int(os.environ.get("CONCURRENCY",   "4"))
TOOL_DELAY_SEC  = float(os.environ.get("TOOL_DELAY_SEC", "0"))
HICACHE_WRITE_POLICY = os.environ.get("HICACHE_WRITE_POLICY", "write_through")
_delay_tag      = f"_d{int(TOOL_DELAY_SEC)}s" if TOOL_DELAY_SEC > 0 else ""
_policy_tag     = "_wb" if HICACHE_WRITE_POLICY == "write_back" else ""
CONFIG          = os.environ.get("CONFIG",            f"sglang_2p2d_c{CONCURRENCY}{_delay_tag}{_policy_tag}")
MAX_TOKENS      = int(os.environ.get("MAX_TOKENS",    "512"))
TIMEOUT         = int(os.environ.get("TIMEOUT",       "600"))
PUSHGATEWAY_URL    = os.environ.get("PUSHGATEWAY_URL",    "http://localhost:9091")
# KV flush: tool delay 후 명시적으로 P1/P2 KV pool을 채워서 eviction을 강제 유도
# FLOOD_DURING_DELAY=0 으로 끌 수 있음
FLOOD_DURING_DELAY = os.environ.get("FLOOD_DURING_DELAY", "1") != "0"
FLOOD_N            = int(os.environ.get("FLOOD_N",     "30"))   # flood 요청 수 (30×3000=90k > 79k KV pool)
FLOOD_TOKENS       = int(os.environ.get("FLOOD_TOKENS", "3000"))  # 요청당 토큰 수 (≈ chars÷5)

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))

# ─── Pushgateway (simple HTTP POST, no prometheus_client dependency) ─────────
_pg_lock = threading.Lock()

def _kv_flush(flood_n: int = FLOOD_N, flood_tokens: int = FLOOD_TOKENS):
    """
    Prefill 서버의 GPU KV pool을 채워서 sleeping conversation의 KV를 L2/L3로 강제 evict.

    각 요청마다 고유한 seed prefix → prefix sharing 없음 → KV pool을 최대한 채움.
    max_tokens=4 로 decode 시간 최소화. flood_n × flood_tokens 토큰 → KV pool overflow.
    """
    filler = ("alpha beta gamma delta epsilon zeta eta theta iota kappa " * 200)
    chars_needed = flood_tokens * 5  # ~5 chars per token

    def _one(seed: int):
        text = f"[kv-flush seed={seed:08x}] " + filler[:chars_needed]
        try:
            requests.post(ROUTER_URL, json={
                "model":       MODEL,
                "messages":    [{"role": "user", "content": text}],
                "max_tokens":  4,
                "temperature": 0,
                "stream":      False,
            }, timeout=60)
        except Exception:
            pass

    threads = [threading.Thread(target=_one, args=(s,), daemon=True) for s in range(flood_n)]
    for t in threads: t.start()
    for t in threads: t.join(timeout=65)


def _push_metric(metric: str, value: float, labels: dict):
    job = labels.pop("job", CONFIG)
    label_str = ",".join(f'{k}="{v}"' for k, v in labels.items())
    body = f"# TYPE {metric} gauge\n{metric}{{{label_str}}} {value}\n"
    try:
        requests.post(f"{PUSHGATEWAY_URL}/metrics/job/{job}", data=body, timeout=2)
    except Exception:
        pass

def push_to_pg(item_id: str, turn_idx: int, ttft, tpot, ctx_chars: int, n_done: int):
    lbl = {"config": CONFIG, "turn": str(turn_idx), "item_id": item_id}
    with _pg_lock:
        if ttft is not None:
            _push_metric("bfcl_turn_ttft_seconds",  ttft,      {**lbl})
        if tpot is not None:
            _push_metric("bfcl_turn_tpot_seconds",  tpot,      {**lbl})
        _push_metric("bfcl_turn_context_chars",     ctx_chars, {**lbl})
        _push_metric("bfcl_items_completed",        n_done,    {"config": CONFIG})

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

# ─── Data loading ────────────────────────────────────────────────────────────
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
print(f"Model       : {MODEL_SLUG}")
print(f"Items       : {len(items)}")
print(f"Concurrency : {CONCURRENCY}")
print(f"Tool delay  : {TOOL_DELAY_SEC}s  {'(KV eviction experiment active)' if TOOL_DELAY_SEC > 0 else '(no delay)'}")
if TOOL_DELAY_SEC > 0:
    print(f"KV flush    : {'ON' if FLOOD_DURING_DELAY else 'OFF'}  "
          f"({FLOOD_N} reqs × {FLOOD_TOKENS} tok = {FLOOD_N*FLOOD_TOKENS:,} tokens/flush)")
    print(f"  ↑ flush으로 P1/P2 GPU KV 강제 overflow → L2/L3 eviction 보장")
print(f"PushGW      : {PUSHGATEWAY_URL}")
print("=" * 60)

# ─── Thread-safe shared state ────────────────────────────────────────────────
_results_lock        = threading.Lock()
_counter_lock        = threading.Lock()
_results             = []
_total_output_tokens = 0
_items_done          = 0

t_experiment_start = time.perf_counter()

# ─── SGLang hicache background poller ────────────────────────────────────────
hicache_poller = SGLangHiCachePoller(interval=2.0)
hicache_poller.start()

# ─── Per-item worker ─────────────────────────────────────────────────────────
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
    tool_delay_for_next_turn = 0.0  # delay applied before this turn (set by previous turn)

    tqdm.write(f"\n[{item['id']}] turns={len(item['question'])}  classes={item.get('involved_classes')}")

    for turn_idx, turn in enumerate(item["question"]):
        user_msg  = turn[0]
        conversation.append(user_msg)
        ctx_chars = sum(len(str(m.get("content", "") or "")) for m in conversation)

        delay_label = f"  [after {tool_delay_for_next_turn:.1f}s tool delay]" if tool_delay_for_next_turn > 0 else ""
        tqdm.write(f"  [{item['id']} t{turn_idx}]{delay_label} ctx={ctx_chars}ch  user: {user_msg['content'][:60]}...")
        turn_delay_applied = tool_delay_for_next_turn
        tool_delay_for_next_turn = 0.0  # reset for this turn

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
            t_request = time.perf_counter()
            t_wall_s  = t_request - t_experiment_start
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

            # Fallback: parse <|python_tag|> from text content if no structured tool_calls
            if not tool_calls_result and "<|python_tag|>" in assistant_content:
                tc = _parse_python_tag(assistant_content)
                if tc:
                    tool_calls_result = [tc]
                    assistant_content = ""

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
                "t_wall_s":             round(t_wall_s,  1),
                "user":                 user_msg["content"][:120],
                "assistant":            assistant_content[:200] if assistant_content else None,
                "tool_calls":           tool_calls_result if tool_calls_result else None,
                "had_tool_call":        bool(tool_calls_result),
                "tool_delay_before_s":  round(turn_delay_applied, 3),
                "kv_flush_applied":     (turn_delay_applied > 0 and FLOOD_DURING_DELAY),
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

            # ── Append assistant message ──────────────────────────────────────
            conversation.append({
                "role":       "assistant",
                "content":    assistant_content or None,
                "tool_calls": tool_calls_result[:1] if tool_calls_result else None,
            })

            # ── Tool delay + KV flush: force eviction to L2/L3 ───────────────
            # 1) Sleep TOOL_DELAY_SEC  (tool execution simulation)
            # 2) Send FLOOD_N × FLOOD_TOKENS flood requests to overflow P1/P2 GPU KV
            #    → sleeping conversation's prefix is LRU-evicted to DRAM (L2) or SSD (L3)
            # 3) Next turn TTFT reflects the tier: L2 ≈ +900ms, L3 ≈ +1000ms
            if tool_calls_result and TOOL_DELAY_SEC > 0:
                fn_name = tool_calls_result[0]['function']['name']
                tqdm.write(
                    f"    [{item['id']} t{turn_idx}] tool_call={fn_name}"
                    f" → sleeping {TOOL_DELAY_SEC}s ..."
                )
                time.sleep(TOOL_DELAY_SEC)

                if FLOOD_DURING_DELAY:
                    tqdm.write(
                        f"    [{item['id']} t{turn_idx}] KV flush: {FLOOD_N} reqs × {FLOOD_TOKENS} tok"
                        f" → forcing GPU KV eviction to L2/L3 ..."
                    )
                    _kv_flush()
                    tqdm.write(f"    [{item['id']} t{turn_idx}] flush done → next TTFT = L2/L3 fetch")

                tool_delay_for_next_turn = TOOL_DELAY_SEC

                # Add synthetic tool result so model context is well-formed
                tc = tool_calls_result[0]
                conversation.append({
                    "role":         "tool",
                    "tool_call_id": tc["id"],
                    "content":      json.dumps({"status": "success",
                                                "function": tc["function"]["name"]}),
                })

        except Exception as e:
            tqdm.write(f"    [{item['id']} t{turn_idx}] ERROR: {e}")
            turn_metrics.append({"turn": turn_idx, "error": str(e)})
            break

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


# ─── Run concurrent ──────────────────────────────────────────────────────────
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

# ─── Stop hicache poller ─────────────────────────────────────────────────────
hicache_poller.stop()
hicache_stats = hicache_poller.stats()

# ─── Final summary ───────────────────────────────────────────────────────────
t_experiment_end = time.perf_counter()
total_wall_time  = t_experiment_end - t_experiment_start

valid   = [r for r in _results if r.get("avg_ttft_s")]

# ── Tool delay TTFT breakdown ─────────────────────────────────────────────────
# turns_after_delay: the turn IMMEDIATELY following a tool call with delay.
# Its TTFT shows which KV tier the conversation landed on after sleeping.
all_turns = [t for r in _results for t in r["turns"] if t.get("ttft_s")]
turns_after_delay    = [t for t in all_turns if t.get("tool_delay_before_s", 0) > 0]
turns_no_delay       = [t for t in all_turns if t.get("tool_delay_before_s", 0) == 0]

def _avg(vals): return round(sum(vals) / len(vals), 4) if vals else None

def _percentile(vals, p):
    if not vals: return None
    s = sorted(vals)
    idx = int(len(s) * p / 100)
    return round(s[min(idx, len(s)-1)], 4)

ttft_all_vals     = [t["ttft_s"] for t in all_turns]
ttft_after_delay  = _avg([t["ttft_s"] for t in turns_after_delay])
ttft_no_delay     = _avg([t["ttft_s"] for t in turns_no_delay])
ttft_degradation  = round(ttft_after_delay - ttft_no_delay, 4) \
                    if (ttft_after_delay and ttft_no_delay) else None

# TTFT histogram buckets (KV tier 판별용)
# <0.5s = L1 GPU hit, 0.5-1.5s = 경쟁/중간, 1.5-2.5s = L2 DRAM, >2.5s = L3 SSD
def _hist(vals):
    if not vals: return {}
    n = len(vals)
    buckets = {"lt_0.5s": 0, "0.5_1.5s": 0, "1.5_2.5s": 0, "gt_2.5s": 0}
    for v in vals:
        if   v < 0.5:  buckets["lt_0.5s"]  += 1
        elif v < 1.5:  buckets["0.5_1.5s"] += 1
        elif v < 2.5:  buckets["1.5_2.5s"] += 1
        else:          buckets["gt_2.5s"]  += 1
    return {k: {"count": v, "pct": round(v/n*100, 1)} for k, v in buckets.items()}

summary = {
    "config":                       CONFIG,
    "model":                        MODEL,
    "concurrency":                  CONCURRENCY,
    "hicache_write_policy":         HICACHE_WRITE_POLICY,
    "tool_delay_sec":               TOOL_DELAY_SEC,
    "flood_during_delay":           FLOOD_DURING_DELAY,
    "flood_n":                      FLOOD_N if FLOOD_DURING_DELAY else 0,
    "flood_tokens":                 FLOOD_TOKENS if FLOOD_DURING_DELAY else 0,
    "total_items":                  len(_results),
    "success_items":                len(valid),
    "error_items":                  len(_results) - len(valid),
    "total_output_tokens":          _total_output_tokens,
    "total_wall_time_s":            round(total_wall_time, 2),
    "overall_throughput_tok_per_s": round(_total_output_tokens / total_wall_time, 2),
    "avg_ttft_s":        round(sum(r["avg_ttft_s"] for r in valid) / len(valid), 4)                   if valid else None,
    "avg_tpot_s":        round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / len(valid), 4) if valid else None,
    "avg_throughput_tok_per_s": round(
                            sum(r["avg_throughput"] for r in valid if r["avg_throughput"]) / len(valid), 2)
                         if valid else None,
    # KV tier degradation from tool delay
    "ttft_after_tool_delay_s":  ttft_after_delay,
    "ttft_no_delay_s":          ttft_no_delay,
    "ttft_degradation_s":       ttft_degradation,
    "turns_after_delay_count":  len(turns_after_delay),
    # TTFT distribution
    "ttft_p50_s":  _percentile(ttft_all_vals, 50),
    "ttft_p90_s":  _percentile(ttft_all_vals, 90),
    "ttft_p99_s":  _percentile(ttft_all_vals, 99),
    "ttft_histogram": _hist(ttft_all_vals),
    "ttft_histogram_after_delay": _hist([t["ttft_s"] for t in turns_after_delay]),
    "hicache_stats": hicache_stats,
}

output = {"summary": summary, "results": _results}

out_dir  = _p(f"results/sglang_hicache/{MODEL_SLUG}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, f"bfcl_multiturn_{CONFIG}.json")
shm_path = f"/dev/shm/bfcl_sglang_{MODEL_SLUG}_{CONFIG}.json"
with open(shm_path, "w") as f:
    json.dump(output, f, indent=2, ensure_ascii=False)
shutil.copy(shm_path, out_path)

print(f"\n{'='*60}")
print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개")
print(f"동시성       : {CONCURRENCY}")
print(f"Tool delay   : {TOOL_DELAY_SEC}s")
print(f"총 출력 토큰  : {_total_output_tokens}")
print(f"전체 소요시간 : {total_wall_time:.2f}s")
print(f"전체 throughput: {summary['overall_throughput_tok_per_s']} tok/s")
print(f"평균 TTFT     : {summary['avg_ttft_s']}s")
print(f"평균 TPOT     : {summary['avg_tpot_s']}s")
print(f"평균 per-req throughput: {summary['avg_throughput_tok_per_s']} tok/s")
print(f"\n── TTFT 분포 (전체 {len(ttft_all_vals)}개 turn) ──")
print(f"  p50={summary['ttft_p50_s']}s  p90={summary['ttft_p90_s']}s  p99={summary['ttft_p99_s']}s")
hist = summary["ttft_histogram"]
print(f"  <0.5s (L1 GPU)   : {hist['lt_0.5s']['count']:4d}턴  ({hist['lt_0.5s']['pct']}%)")
print(f"  0.5~1.5s (중간)   : {hist['0.5_1.5s']['count']:4d}턴  ({hist['0.5_1.5s']['pct']}%)")
print(f"  1.5~2.5s (L2 예상): {hist['1.5_2.5s']['count']:4d}턴  ({hist['1.5_2.5s']['pct']}%)")
print(f"  >2.5s    (L3 예상): {hist['gt_2.5s']['count']:4d}턴  ({hist['gt_2.5s']['pct']}%)")
if TOOL_DELAY_SEC > 0:
    print(f"\n── Tool Delay KV Tier 영향 ({TOOL_DELAY_SEC}s delay) ──")
    print(f"  delay 전 turn TTFT  : {ttft_no_delay}s  (KV GPU 재사용)")
    print(f"  delay 후 turn TTFT  : {ttft_after_delay}s  (KV 다른 tier 이동 후)")
    print(f"  TTFT 증가량         : {ttft_degradation}s  ({len(turns_after_delay)}개 turn 샘플)")
    if ttft_degradation and ttft_no_delay:
        pct = ttft_degradation / ttft_no_delay * 100
        verdict = "↑ KV eviction 발생!" if ttft_degradation > 0.2 else "△ 미미함 (eviction 없음)"
        print(f"  degradation         : {pct:+.0f}%  {verdict}")
    dh = summary["ttft_histogram_after_delay"]
    if dh:
        print(f"  delay 후 TTFT 분포  :")
        print(f"    <0.5s (GPU)  : {dh['lt_0.5s']['count']}턴 ({dh['lt_0.5s']['pct']}%)")
        print(f"    1.5~2.5s (L2): {dh['1.5_2.5s']['count']}턴 ({dh['1.5_2.5s']['pct']}%)")
        print(f"    >2.5s (L3)   : {dh['gt_2.5s']['count']}턴 ({dh['gt_2.5s']['pct']}%)")
print(f"\n[SGLang hicache KV stats (token_usage sampled during benchmark)]")
for inst, s in hicache_stats.items():
    if s and s.get("token_usage"):
        tu = s["token_usage"]
        print(f"  {inst}: max={tu['max']:.3f}  mean={tu['mean']:.3f}  samples={tu['samples']}")
print(f"\n결과 저장     : {out_path}")
