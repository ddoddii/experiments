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
  ITEM_OFFSET     시작 offset (open-loop sweep에서 rate point마다 서로 다른
                   corpus slice를 쓰기 위함; sharegpt 러너와 동일한 관례)
  BFCL_CATEGORIES 풀에 넣을 multi-turn 카테고리들, 공백 구분 (기본 "base" 단독,
                   기존 호출과 동일). "base miss_func miss_param long_context"로
                   4개를 합치면 코퍼스가 200 -> 600+ 로 늘어남 -- QPS sweep의
                   wrap 문제를 줄이려면 이걸 쓸 것. base 외 3개는 이 repo에
                   없으므로 gorilla/ 클론에서 먼저 복사해야 함 (아래 주석 참고).
  ROUTER_URL      (기본 http://127.0.0.1:8000/v1/chat/completions)
  SESSION_RATE    >0이면 open-loop: 대화가 SESSION_RATE/s로 Poisson 도착.
                   0(기본)이면 기존 closed-loop (CONCURRENCY개 워커 풀).

출력 스키마는 순차 버전과 동일 → phase0_analyze.py가 그대로 처리.
open-loop 모드는 sglang_sharegpt_multi_turn_concurrent.py의 run_open_loop()와
동일한 설계 -- "job delay" (세션 도착 -> 세션 완료) 정의를 그쪽과 맞추기 위함.
"""
import json
import os
import random
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
ITEM_OFFSET = int(os.environ.get("ITEM_OFFSET", "0"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "512"))
TIMEOUT = int(os.environ.get("TIMEOUT", "120"))
# tool-call 유휴시간(think-time) 모사. 턴 사이에 이 시간만큼 쉰다.
# >0이면 낮은 동시성으로도 "유휴 중 타세션이 내 prefix를 evict"하는 상황이 생겨,
# radix(재계산) vs hicache(fetch)의 TTFT 차이가 queueing 없이 드러난다.
TOOL_DELAY = float(os.environ.get("TOOL_DELAY", "0"))

# ---------------------------------------------------------------- open-loop mode
# Ported from sglang_sharegpt_multi_turn_concurrent.py -- see the comment there for why
# this exists (closed loop makes the CLIENT the bottleneck, so throughput/delay are flat
# across arms no matter how fast the server is; open loop makes the SERVER the
# bottleneck, which is what a JPS-vs-delay figure needs).
SESSION_RATE = float(os.environ.get("SESSION_RATE", "0"))      # sessions/s; >0 = open loop
LOAD_DURATION = float(os.environ.get("LOAD_DURATION", "240"))  # seconds of arrivals
WARMUP_S = float(os.environ.get("WARMUP_S", "45"))
DRAIN_TIMEOUT = float(os.environ.get("DRAIN_TIMEOUT", "420"))
OPEN_WORKERS = int(os.environ.get("OPEN_WORKERS", "3072"))

T0 = time.perf_counter()

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


# BFCL_CATEGORIES selects which multi-turn category files to pool into one corpus.
# Default is "base" alone, i.e. IDENTICAL to every caller that predates this -- nothing
# changes unless the env var is set.
#
# WHY MORE THAN BASE: base is a fixed 200 conversations, which an open-loop QPS sweep
# chews through fast (BFCL sessions are short: few turns, TOOL_DELAY=0), so most points
# wrap and bias the cached arms. gorilla/ (already cloned per CLAUDE.md) ships three more
# multi-turn categories with the IDENTICAL schema (id/question/initial_config/
# involved_classes) -- miss_func, miss_param, long_context -- so pooling them is free
# corpus, not a new dataset or a new harness. long_context also carries a deliberately
# bigger initial_config (deeper nested file-system/social state serialized into the
# system prompt), which pushes the prefill tail longer without changing tool-def cost.
#
# Not committed to this repo (gorilla/ isn't tracked here -- see CLAUDE.md). Copy them
# from the local gorilla clone once:
#   cd ~/experiments && cp gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/BFCL_v4_multi_turn_{miss_func,miss_param,long_context}.json data/
BFCL_CATEGORIES = os.environ.get("BFCL_CATEGORIES", "base").split()
_CATEGORY_FILES = {
    "base":         "data/BFCL_v3_multi_turn_base.json",     # already present (200 items)
    "miss_func":    "data/BFCL_v4_multi_turn_miss_func.json",
    "miss_param":   "data/BFCL_v4_multi_turn_miss_param.json",
    "long_context": "data/BFCL_v4_multi_turn_long_context.json",
}

func_docs = {cls: load_func_doc(path) for cls, path in CLASS_TO_FILE.items()}
_all_items = []
for _cat in BFCL_CATEGORIES:
    _path = _CATEGORY_FILES.get(_cat)
    if _path is None:
        raise SystemExit(f"unknown BFCL category '{_cat}' (want one of "
                          f"{sorted(_CATEGORY_FILES)})")
    if not os.path.exists(_path):
        raise SystemExit(
            f"BFCL category '{_cat}' needs {_path}, which isn't in this repo.\n"
            f"  Copy it from the gorilla clone once:\n"
            f"    cp gorilla/berkeley-function-call-leaderboard/bfcl_eval/data/"
            f"BFCL_v4_multi_turn_{_cat}.json {_path}")
    with open(_path) as f:
        _cat_items = [json.loads(l) for l in f]
    for _it in _cat_items:
        _it["_bfcl_category"] = _cat   # provenance, for later per-category breakdowns
    _all_items.extend(_cat_items)
# Every multi-turn category draws from the same environment/class family, but this is
# NOT verified upstream -- an item needing a class this harness has no func_doc for
# would silently get an empty tool list rather than fail loudly, so check once at load.
_unmapped = {cls for it in _all_items for cls in it.get("involved_classes", [])
             if cls not in CLASS_TO_FILE}
if _unmapped:
    print(f"[warn] classes with no func_doc mapping: {sorted(_unmapped)} -- items "
          f"needing them will get an empty tool list.")
_n_all = len(_all_items)
# ITEM_OFFSET slices out a DISJOINT range of the corpus so a sweep's rate points do not
# each replay the same conversations against a server the previous point already warmed
# -- same reasoning as the ShareGPT runner. Unlike ShareGPT's ~90k conversations, BFCL
# has a FIXED 200, so a sweep's cumulative ITEM_OFFSET routinely runs past the corpus
# length -- and a plain slice there returns an EMPTY list, which used to raise
# SystemExit and kill the whole `set -e` sweep at whatever point first ran past 200,
# not just warp that point's own numbers. Wrapping with modulo instead keeps every
# point non-empty; run_open_loop()'s own sessions_repeated counter is what actually
# flags and reports how much a given point wrapped, same as the ShareGPT path.
if _n_all == 0:
    items = []
elif MAX_ITEMS > 0:
    if ITEM_OFFSET + MAX_ITEMS <= _n_all:
        items = _all_items[ITEM_OFFSET:ITEM_OFFSET + MAX_ITEMS]   # unchanged in-range case
    else:
        _start = ITEM_OFFSET % _n_all
        items = [_all_items[(_start + i) % _n_all] for i in range(MAX_ITEMS)]
else:
    _start = ITEM_OFFSET % _n_all
    items = _all_items[_start:] + _all_items[:_start]


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
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        # sglang emits at least one chunk under concurrent load with an EMPTY choices
        # list (a keep-alive / usage-only-style chunk) -- chunk["choices"][0] on that
        # raises IndexError: list index out of range, caught by process_item's bare
        # except and misreported as a session error. Measured: 0% error at low offered
        # rate, 95-96% at one specific mid-sweep rate point, dropping again at the
        # highest rate -- a load-dependent crash, not organic saturation, and it silently
        # invalidated most of a rate point's throughput/TTFT/job-delay stats (computed
        # only over the handful of survivors) rather than failing loudly. Same guard the
        # ShareGPT runner's run_turn() already has for the identical chunk shape.
        if not chunk.get("choices"):
            continue
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
        # Absolute clock relative to T0, same convention as the ShareGPT runner: needed
        # to window a turn into an open-loop steady-state measurement and to compute
        # session-level job delay (arrival -> last turn done).
        "t_start_s": round(t_request - T0, 3),
        "t_done_s": round((t_last or t_request) - T0, 3),
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
        if not turn:
            # BFCL v4's miss_func/miss_param/long_context categories (pooled in via
            # BFCL_CATEGORIES) carry turns with an empty message list -- turn[0] on
            # that raised IndexError: list index out of range, caught by the except
            # below and misrecorded as a request-level error. Confirmed via item id:
            # multi_turn_miss_func items errored at ~100% wherever ITEM_OFFSET's
            # sweep window landed on them (e.g. 122/122 at one rate point), IDENTICAL
            # across recompute/hicache/park -- a corpus/parsing bug, not a placement
            # or caching one, so it wasn't caught by the earlier choices[0] fix.
            continue
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
    total_out = sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens"))
    # Normalized latency (ms/token): see the identical block in
    # sglang_sharegpt_multi_turn_concurrent.py -- serving time only (excludes the
    # TOOL_DELAY sleep between turns), divided by tokens generated.
    _serving_s = sum(t["e2e_latency_s"] for t in turn_metrics
                     if t.get("e2e_latency_s") and not t.get("error"))
    return {
        "id": item["id"],
        "num_turns": len(item["question"]),
        "turns": turn_metrics,
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(t["tpot_s"] for t in valid if t.get("tpot_s")) / max(1, sum(1 for t in valid if t.get("tpot_s"))), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(t["throughput_tok_per_s"] for t in valid if t.get("throughput_tok_per_s")) / max(1, sum(1 for t in valid if t.get("throughput_tok_per_s"))), 2) if valid else None,
        "total_output_tokens": total_out,
        "normalized_latency_ms_per_tok": round(_serving_s * 1000 / total_out, 3) if total_out else None,
    }


def run_open_loop(items):
    """Poisson session arrivals at SESSION_RATE for LOAD_DURATION seconds, then drain.

    Direct port of sglang_sharegpt_multi_turn_concurrent.py's run_open_loop -- same
    windowing, same job-delay definition (arrival -> last turn done) -- so a BFCL point
    and a ShareGPT point on the same JPS-vs-delay figure mean the same thing.
    """
    global T0
    results, lock = [], threading.Lock()
    inflight = [0]
    peak_inflight = [0]
    rng = random.Random(0xC0FFEE ^ int(SESSION_RATE * 1000))

    def task(item, t_arrival_s):
        with lock:
            inflight[0] += 1
            peak_inflight[0] = max(peak_inflight[0], inflight[0])
        try:
            r = process_item(item)
        except Exception as e:  # noqa: BLE001
            r = {"id": item.get("id"), "turns": [{"turn": 0, "error": str(e)}],
                 "total_output_tokens": 0, "num_turns": 0}
        r["t_arrival_s"] = round(t_arrival_s, 3)
        turns = [t for t in (r.get("turns") or []) if not t.get("error")]
        if turns and turns[-1].get("t_done_s") is not None:
            r["job_delay_s"] = round(turns[-1]["t_done_s"] - t_arrival_s, 3)
        else:
            r["job_delay_s"] = None
        with lock:
            inflight[0] -= 1
            results.append(r)

    pool = ThreadPoolExecutor(max_workers=OPEN_WORKERS)
    futs = []
    T0 = time.perf_counter()
    launched = 0
    next_report = WARMUP_S
    with pool:
        while True:
            elapsed = time.perf_counter() - T0
            if elapsed >= LOAD_DURATION:
                break
            it = dict(items[launched % len(items)])
            it["id"] = f"{it.get('id')}#{launched}"
            futs.append(pool.submit(task, it, elapsed))
            launched += 1
            time.sleep(rng.expovariate(SESSION_RATE))
            if elapsed >= next_report:
                with lock:
                    n_done, n_live = len(results), inflight[0]
                print(f"  [t={elapsed:6.0f}s] launched={launched:4d} done={n_done:4d} "
                      f"inflight={n_live:4d}", flush=True)
                next_report += 60
        t_load_end = time.perf_counter() - T0
        print(f"  [load window closed at {t_load_end:.0f}s] launched={launched}, "
              f"draining up to {DRAIN_TIMEOUT:.0f}s ...", flush=True)
        deadline = time.perf_counter() + DRAIN_TIMEOUT
        n_unfinished = 0
        for f in futs:
            try:
                f.result(timeout=max(0.0, deadline - time.perf_counter()))
            except Exception:  # noqa: BLE001
                n_unfinished += 1

    lo, hi = WARMUP_S, t_load_end
    win_tokens = win_turns = 0
    win_ttfts = []
    win_job_delays = []
    win_norm_lat = []
    for r in results:
        for t in (r.get("turns") or []):
            ts, td = t.get("t_start_s"), t.get("t_done_s")
            if ts is None or td is None or t.get("error"):
                continue
            if ts >= lo and td <= hi:
                win_tokens += t.get("output_tokens") or 0
                win_turns += 1
                if t.get("ttft_s") is not None:
                    win_ttfts.append(t["ttft_s"])
        ta, jd = r.get("t_arrival_s"), r.get("job_delay_s")
        if ta is not None and jd is not None and ta >= lo and ta + jd <= hi:
            win_job_delays.append(jd)
            if r.get("normalized_latency_ms_per_tok") is not None:
                win_norm_lat.append(r["normalized_latency_ms_per_tok"])
    win = max(1e-9, hi - lo)
    win_ttfts.sort()
    win_job_delays.sort()
    win_norm_lat.sort()

    def _p(seq, p):
        if not seq:
            return None
        k = max(0, min(len(seq) - 1, int(round((p / 100.0) * (len(seq) - 1)))))
        return round(seq[k], 4)

    stats = {
        "mode": "open_loop",
        "session_rate": SESSION_RATE,
        "load_duration_s": round(t_load_end, 2),
        "warmup_s": WARMUP_S,
        "window_s": round(win, 2),
        "sessions_launched": launched,
        "sessions_repeated": max(0, launched - len(items)),
        "sessions_completed": len(results),
        "sessions_unfinished_at_drain": n_unfinished,
        "peak_inflight_sessions": peak_inflight[0],
        "window_throughput_tok_s": round(win_tokens / win, 2),
        "window_turns": win_turns,
        "window_turn_rate_s": round(win_turns / win, 3),
        "window_ttft_p50_s": _p(win_ttfts, 50),
        "window_ttft_p95_s": _p(win_ttfts, 95),
        "window_ttft_p99_s": _p(win_ttfts, 99),
        # Average Job Delay: session arrival to session completion (all turns, all
        # TOOL_DELAY gaps included). This is the metric a JPS-vs-delay figure plots.
        "window_job_delay_n": len(win_job_delays),
        "window_job_delay_mean_s": round(sum(win_job_delays) / len(win_job_delays), 4)
                                    if win_job_delays else None,
        "window_job_delay_p50_s": _p(win_job_delays, 50),
        "window_job_delay_p95_s": _p(win_job_delays, 95),
        "window_job_delay_p99_s": _p(win_job_delays, 99),
        # Normalized latency (ms/token), serving-time-only (no TOOL_DELAY) -- the
        # throughput-vs-latency figure convention. See the identical field in the
        # ShareGPT runner.
        "window_norm_latency_mean_ms_tok": round(sum(win_norm_lat) / len(win_norm_lat), 4)
                                            if win_norm_lat else None,
        "window_norm_latency_p50_ms_tok": _p(win_norm_lat, 50),
        "window_norm_latency_p95_ms_tok": _p(win_norm_lat, 95),
    }
    if stats["sessions_repeated"] > 0:
        print(f"\n  *** WARNING: the corpus wrapped -- {stats['sessions_repeated']} of "
              f"{launched} sessions replayed a conversation already served (BFCL has "
              f"only 200 total). Repeats are free cache hits for the cached arms and no "
              f"help to recompute; this point OVERSTATES the cached arms.\n", flush=True)
    return results, stats


def _atomic_dump(obj, path):
    """Write to a sibling temp file, then rename -- json.dump straight onto the final
    path truncates it first, so a disk-full mid-write destroys the previous contents.
    Ported from the ShareGPT runner, which hit exactly this during a long sweep."""
    tmp = f"{path}.tmp.{os.getpid()}"
    try:
        with open(tmp, "w") as fh:
            json.dump(obj, fh, indent=2, ensure_ascii=False)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, path)
    except Exception:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _finish(results, wall, extra=None):
    """Summarise, write, print. Shared by the closed- and open-loop paths."""
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
    # Percentiles over EVERY turn, not over per-session means -- the same gap the
    # long-context bench had. Without them a single run reports one mean with no spread,
    # and on this host the same config re-run has moved TTFT p95 by more than the effects
    # being measured. Per-turn is the right population because the cache acts per turn.
    all_ttft = sorted(t["ttft_s"] for r in results for t in r.get("turns", [])
                      if t.get("ttft_s"))
    if all_ttft:
        def _pct(q):
            if len(all_ttft) == 1:
                return round(all_ttft[0], 4)
            i = q / 100 * (len(all_ttft) - 1)
            lo, hi = int(i), min(int(i) + 1, len(all_ttft) - 1)
            return round(all_ttft[lo] + (all_ttft[hi] - all_ttft[lo]) * (i - lo), 4)
        summary.update({
            "n_ttft_samples": len(all_ttft),
            "ttft_p50_s": _pct(50), "ttft_p90_s": _pct(90),
            "ttft_p95_s": _pct(95), "ttft_p99_s": _pct(99),
            "ttft_mean_per_turn_s": round(sum(all_ttft) / len(all_ttft), 4),
        })
    if extra:
        summary.update(extra)

    output = {"summary": summary, "results": results}
    os.makedirs("results", exist_ok=True)
    out_path = f"results/{CONFIG}.json"  # CONFIG가 실험 이름 (데이터/구성 반영)
    # Summary first (small, atomic), then the full file -- if disk fills on the big write
    # the summary survives and carries every field collect_qps.py reads.
    try:
        _atomic_dump({"summary": summary}, out_path.replace(".json", ".summary.json"))
    except OSError as e:
        print(f"[error] could not even write the summary: {e}")
    try:
        _atomic_dump(output, out_path)
    except OSError as e:
        print(f"[error] full results not written ({e}); the summary survived at "
              f"{out_path.replace('.json', '.summary.json')}.")

    print(f"\n{'='*60}")
    print(f"완료: {summary['total_items']}개 / 에러: {summary['error_items']}개  (C={CONCURRENCY})")
    print(f"전체 소요시간: {wall:.2f}s  overall throughput: {summary['overall_throughput_tok_per_s']} tok/s")
    print(f"평균 TTFT: {summary['avg_ttft_s']}s  평균 TPOT: {summary['avg_tpot_s']}s")
    if extra:
        print(f"[open-loop] offered {extra['session_rate']}/s -> "
              f"{extra['window_throughput_tok_s']} tok/s in window, "
              f"job delay mean {extra['window_job_delay_mean_s']}s "
              f"p50 {extra['window_job_delay_p50_s']}s p95 {extra['window_job_delay_p95_s']}s, "
              f"TTFT p50 {extra['window_ttft_p50_s']}s, "
              f"peak inflight {extra['peak_inflight_sessions']}, "
              f"unfinished {extra['sessions_unfinished_at_drain']}")
    print(f"결과 저장: {out_path}")


def main():
    print(f"[concurrent] CONFIG={CONFIG} CONCURRENCY={CONCURRENCY} items={len(items)} url={ROUTER_URL}")
    if not items:
        raise SystemExit(f"ITEM_OFFSET={ITEM_OFFSET} leaves no items (corpus has "
                          f"{len(_all_items)}).")

    if SESSION_RATE > 0:
        print(f"[open-loop] rate={SESSION_RATE}/s load={LOAD_DURATION}s "
              f"warmup={WARMUP_S}s drain<={DRAIN_TIMEOUT}s", flush=True)
        t0 = time.perf_counter()
        results, open_stats = run_open_loop(items)
        wall = time.perf_counter() - t0
        _finish(results, wall, extra=open_stats)
        return

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
    _finish(results, wall)


if __name__ == "__main__":
    main()
