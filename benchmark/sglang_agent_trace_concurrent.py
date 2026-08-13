#!/usr/bin/env python3
"""
Replay recorded agent sessions (Codex / SWE-bench Pro traces) as a concurrent
multi-turn serving workload.

WHY THIS AND NOT THE BFCL RUNNER
  BFCL grows only 3.6k -> 6k tokens across a session, so prefix reuse never pays for
  its own lookup and every cached arm loses to recompute there. These traces run 8-15
  turns from ~12k to ~48k tokens, which is the regime the design targets: eviction
  happens naturally under concurrency (no synthetic flood needed) and a restored prefix
  is worth seconds of re-prefill.

THE ASSISTANT TURN IS REPLAYED FROM THE TRACE, NOT FROM THE MODEL
  The existing BFCL runner appends the model's OWN generated reply to the conversation.
  Across arms that makes the conversations diverge after turn 1: different text, then
  different context lengths, then TTFTs that are not comparable because the arms were
  not asked the same question. Here the recorded reply is appended instead, so every arm
  sees a byte-identical prompt sequence and the only thing that differs is how the
  server served it. Generation still happens -- max_tokens is set to the recorded reply's
  length so the decode work is realistic -- its text is simply not fed back.
  REPLAY_GENERATED=1 restores the old behaviour for an explicit A/B.

Input: the selection written by agent_trace.py --select (already filtered to the
experiment's turn/context bounds, with per-turn recorded_out_tokens).

환경변수:
  SESSIONS      선별된 세션 JSON (기본 data/codex_sessions_100.json)
  CONFIG        결과 태그
  CONCURRENCY   동시 세션 수 (기본 8)
  MAX_ITEMS     세션 수 제한 (0 = 전체)
  TOOL_DELAY    턴 사이 유휴시간 초 (기본 3)
  MAX_TOKENS    출력 상한 (기본 1024; recorded length가 더 작으면 그쪽을 씀)
  ROUTER_URL / MODEL / TIMEOUT
"""
import json
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import requests

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
SESSIONS = os.environ.get("SESSIONS", "data/codex_sessions_100.json")
CONFIG = os.environ.get("CONFIG", "agent_trace")
CONCURRENCY = int(os.environ.get("CONCURRENCY", "8"))
MAX_ITEMS = int(os.environ.get("MAX_ITEMS", "0"))
TOOL_DELAY = float(os.environ.get("TOOL_DELAY", "3"))
MAX_TOKENS = int(os.environ.get("MAX_TOKENS", "1024"))
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))
REPLAY_GENERATED = os.environ.get("REPLAY_GENERATED", "0") == "1"


def run_turn(messages, max_tokens):
    """One streaming request -> per-turn metrics. Returns (metrics, generated_text)."""
    payload = {
        "model": MODEL, "messages": messages,
        "max_tokens": max(1, max_tokens), "stream": True,
        "stream_options": {"include_usage": True},
    }
    t_request = time.perf_counter()
    resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
    resp.raise_for_status()

    t_first = t_last = None
    token_count = 0
    text = []
    prompt_tokens = None
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
        # usage-only chunks carry an EMPTY choices list; indexing [0] on those raises
        # IndexError, which the per-session except would misreport as a failed session.
        # Same guard the BFCL runner needed after that exact incident.
        if chunk.get("usage"):
            prompt_tokens = chunk["usage"].get("prompt_tokens")
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0].get("delta") or {}
        if delta.get("content"):
            now = time.perf_counter()
            if t_first is None:
                t_first = now
            t_last = now
            token_count += 1
            text.append(delta["content"])

    # A stream that opened 200 and closed without a single content delta is a SERVER-SIDE
    # ABORT (a failed prefill->decode KV transfer looks exactly like this), not a turn that
    # legitimately had nothing to say. It raises no exception, so without an explicit flag
    # it lands in the results as ttft_s=None and is then skipped by every percentile --
    # 94 of them once passed as a clean run with error_items=0.
    aborted = token_count == 0
    ttft = (t_first - t_request) if t_first else None
    e2e = (t_last - t_request) if t_last else None
    decode = (t_last - t_first) if (t_first and t_last) else None
    tpot = (decode / (token_count - 1)) if (decode and token_count > 1) else None
    tput = ((token_count - 1) / decode) if (decode and token_count > 1) else None
    return ({
        "ttft_s": round(ttft, 4) if ttft else None,
        "tpot_s": round(tpot, 4) if tpot else None,
        "output_tokens": token_count,
        "prompt_tokens": prompt_tokens,
        "e2e_latency_s": round(e2e, 4) if e2e else None,
        "throughput_tok_per_s": round(tput, 2) if tput else None,
        **({"error": "empty stream (server aborted; no content delta)"} if aborted else {}),
    }, "".join(text))


def process_session(sess):
    turn_metrics = []
    # Start from the first turn's prompt and extend it turn by turn, so the growth is
    # exactly the trace's. Rebuilding from each turn's stored prompt_messages would be
    # equivalent for the recorded path but would silently ignore REPLAY_GENERATED.
    messages = list(sess["turns"][0]["prompt_messages"])
    for ti, t in enumerate(sess["turns"]):
        if TOOL_DELAY > 0 and ti > 0:
            time.sleep(TOOL_DELAY)
        try:
            cap = min(MAX_TOKENS, t.get("recorded_out_tokens") or MAX_TOKENS)
            m, generated = run_turn(messages, cap)
            m["turn"] = ti
            m["expected_prompt_tokens"] = t.get("prompt_tokens")
            turn_metrics.append(m)
            reply = t["recorded_reply"]["content"]
            messages.append({"role": "assistant",
                             "content": generated if REPLAY_GENERATED else reply})
            # Extend with whatever the trace had between this reply and the next prompt
            # (tool observations, the next user message) -- taken from the NEXT turn's
            # stored prompt so the reconstruction cannot drift from the trace.
            if ti + 1 < len(sess["turns"]):
                nxt = sess["turns"][ti + 1]["prompt_messages"]
                messages.extend(nxt[len(messages):])
        except Exception as e:  # noqa: BLE001
            turn_metrics.append({"turn": ti, "error": str(e)})
            break

    valid = [t for t in turn_metrics if t.get("ttft_s")]
    total_out = sum(t["output_tokens"] for t in turn_metrics if t.get("output_tokens"))
    n_tpot = sum(1 for t in valid if t.get("tpot_s"))
    n_tput = sum(1 for t in valid if t.get("throughput_tok_per_s"))
    return {
        "id": sess["id"],
        "num_turns": len(sess["turns"]),
        "turns": turn_metrics,
        "peak_prompt_tokens": sess.get("peak_prompt_tokens"),
        "avg_ttft_s": round(sum(t["ttft_s"] for t in valid) / len(valid), 4) if valid else None,
        "avg_tpot_s": round(sum(t["tpot_s"] for t in valid if t.get("tpot_s")) / n_tpot, 4) if n_tpot else None,
        "avg_throughput_tok_per_s": round(sum(t["throughput_tok_per_s"] for t in valid if t.get("throughput_tok_per_s")) / n_tput, 2) if n_tput else None,
        "total_output_tokens": total_out,
    }


def _atomic_dump(obj, path):
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        json.dump(obj, fh, indent=2)
    os.replace(tmp, path)


def main():
    with open(SESSIONS) as fh:
        sessions = json.load(fh)
    if MAX_ITEMS > 0:
        sessions = sessions[:MAX_ITEMS]
    print(f"[agent-trace] CONFIG={CONFIG} C={CONCURRENCY} sessions={len(sessions)} "
          f"tool_delay={TOOL_DELAY}s replay={'generated' if REPLAY_GENERATED else 'recorded'}")

    results, lock = [], threading.Lock()
    t0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as ex:
        futs = {ex.submit(process_session, s): s["id"] for s in sessions}
        for f in as_completed(futs):
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                r = {"id": futs[f], "num_turns": 0, "turns": [{"error": str(e)}],
                     "avg_ttft_s": None, "avg_tpot_s": None,
                     "avg_throughput_tok_per_s": None, "total_output_tokens": 0}
            with lock:
                results.append(r)
                if len(results) % 10 == 0:
                    print(f"  {len(results)}/{len(sessions)} sessions done", flush=True)
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
        "avg_tpot_s": round(sum(r["avg_tpot_s"] for r in valid if r["avg_tpot_s"]) / max(1, sum(1 for r in valid if r["avg_tpot_s"])), 4) if valid else None,
        "avg_throughput_tok_per_s": round(sum(r["avg_throughput_tok_per_s"] for r in valid if r["avg_throughput_tok_per_s"]) / max(1, sum(1 for r in valid if r["avg_throughput_tok_per_s"])), 2) if valid else None,
        "replay_mode": "generated" if REPLAY_GENERATED else "recorded",
    }
    # Turn-level failures, reported next to the turn-level percentiles they distort.
    # success_items/error_items are per SESSION and stay green while a tenth of the turns
    # in every session die, so they cannot be the only integrity signal in the summary.
    all_turns = [t for r in results for t in r.get("turns", [])]
    failed = [t for t in all_turns if t.get("error")]
    summary.update({
        "total_turns": len(all_turns),
        "failed_turns": len(failed),
        "failed_turn_rate": round(len(failed) / len(all_turns), 4) if all_turns else None,
    })
    if failed:
        print(f"[WARNING] {len(failed)}/{len(all_turns)} turns failed "
              f"({len(failed) / len(all_turns) * 100:.1f}%) -- the TTFT percentiles below "
              f"are over the SURVIVORS and are not comparable to a clean run.", flush=True)
    # Percentiles over EVERY turn, not per-session means: the cache acts per turn, and a
    # single mean hides a spread that has moved more than the effects being measured.
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

    os.makedirs("results", exist_ok=True)
    out_path = f"results/{CONFIG}.json"
    _atomic_dump({"summary": summary}, out_path.replace(".json", ".summary.json"))
    _atomic_dump({"summary": summary, "results": results}, out_path)
    print(json.dumps(summary, indent=2))
    print(f"[saved] {out_path}")


if __name__ == "__main__":
    main()
