#!/usr/bin/env python3
"""
Pinpoint why 125/200 BFCL items produced ZERO output tokens with HTTP 200.

What we know from results/sglang_hicache/Llama-3.1-8B-Instruct/bfcl_multiturn_sglang_2p2d_c4.json:
  - 125 items have EVERY turn with ttft_s=None, output_tokens=0, assistant='', error=None
  - only 7 turns recorded an actual 400
  - the split is deterministic BY TOOL CLASS, not random:
        TradingBot            100% fail        GorillaFileSystem      0% fail
        TravelAPI             100% fail        MathAPI                0% fail (alone)
        TicketAPI+TravelAPI   100% fail        TicketAPI              0% fail (alone)
        VehicleControlAPI      37-67% fail
  - context length is NOT the cause: TravelAPI fails at ~266 tokens while
    GorillaFileSystem+TwitterAPI succeeds at ~412
  - the schemas convert cleanly under TYPE_MAP, and function names are all valid

So the request is accepted and the model returns nothing, for particular tool sets. This
script sends ONE turn per class -- non-streamed, so the full response object is visible --
and prints finish_reason, usage, content, tool_calls and any error body. Comparing a
failing class against a passing one in the same run isolates the cause instead of
guessing.

Run with the 2P2D servers up (the same router the benchmark used).

사용:
  python benchmark/repro_empty_output.py
  python benchmark/repro_empty_output.py --classes TradingBot GorillaFileSystem
  python benchmark/repro_empty_output.py --no-tools     # same prompts, tools removed
"""
import argparse
import json
import os
import sys

import requests

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)


def _p(rel):
    return os.path.join(ROOT, rel)


CLASS_TO_FILE = {
    "GorillaFileSystem": "gorilla_file_system",
    "TicketAPI": "ticket_api",
    "MessageAPI": "message_api",
    "MathAPI": "math_api",
    "TradingBot": "trading_bot",
    "TwitterAPI": "posting_api",
    "TravelAPI": "travel_booking",
    "VehicleControlAPI": "vehicle_control",
}
TYPE_MAP = {"float": "number", "integer": "integer", "dict": "object", "list": "array",
            "tuple": "array", "str": "string", "bool": "boolean", "none": "null"}


def dict_to_object(obj):
    if isinstance(obj, dict):
        if "type" in obj and isinstance(obj["type"], str):
            obj["type"] = TYPE_MAP.get(obj["type"], obj["type"])
        return {k: dict_to_object(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [dict_to_object(v) for v in obj]
    return obj


def load_func_doc(stem):
    with open(_p(f"data/multi_turn_func_doc/{stem}.json")) as f:
        content = f.read().strip()
    docs = (json.loads(content) if content.startswith("[")
            else [json.loads(l) for l in content.splitlines() if l.strip()])
    return [dict_to_object({k: v for k, v in d.items() if k != "response"}) for d in docs]


def find_case(items, cls):
    """First dataset item whose involved_classes is exactly [cls], else any containing it."""
    exact = [it for it in items if (it.get("involved_classes") or []) == [cls]]
    return (exact or [it for it in items if cls in (it.get("involved_classes") or [])]
            or [None])[0]


def probe(url, model, tools, user_msg, max_tokens, label):
    payload = {"model": model,
               "messages": [{"role": "user", "content": user_msg}],
               "max_tokens": max_tokens, "temperature": 0, "stream": False}
    if tools:
        payload["tools"] = [{"type": "function", "function": d} for d in tools]
    try:
        r = requests.post(url, json=payload, timeout=180)
    except Exception as e:  # noqa: BLE001
        print(f"  {label}: REQUEST FAILED {type(e).__name__}: {e}")
        return
    if r.status_code >= 400:
        print(f"  {label}: HTTP {r.status_code}")
        print(f"      body: {r.text[:700]}")
        return
    try:
        d = r.json()
    except Exception:  # noqa: BLE001
        print(f"  {label}: HTTP 200 but unparsable body: {r.text[:300]}")
        return
    ch = (d.get("choices") or [{}])[0]
    msg = ch.get("message") or {}
    usage = d.get("usage") or {}
    content = (msg.get("content") or "")
    tcs = msg.get("tool_calls") or []
    verdict = "EMPTY" if (not content.strip() and not tcs) else "ok"
    print(f"  {label}: HTTP 200  [{verdict}]  finish_reason={ch.get('finish_reason')!r} "
          f"completion_tokens={usage.get('completion_tokens')} "
          f"prompt_tokens={usage.get('prompt_tokens')}")
    if tcs:
        f0 = (tcs[0].get("function") or {})
        print(f"      tool_call[0]: {f0.get('name')!r} args={str(f0.get('arguments'))[:120]}")
    if content.strip():
        print(f"      content[:200]: {content[:200]!r}")
    if verdict == "EMPTY":
        # the fields that distinguish "model emitted EOS immediately" from
        # "parser swallowed everything" from "max_tokens clamped to 0"
        print(f"      raw choice: {json.dumps(ch)[:400]}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default=os.environ.get(
        "SGLANG_URL", "http://127.0.0.1:8000/v1/chat/completions"))
    ap.add_argument("--classes", nargs="*", default=[
        "TradingBot", "TravelAPI", "VehicleControlAPI",   # failing
        "GorillaFileSystem", "MathAPI", "TicketAPI"])     # passing
    ap.add_argument("--max-tokens", type=int, default=512)
    ap.add_argument("--no-tools", action="store_true",
                    help="send the same user prompts WITHOUT tools -- if output appears, "
                         "the tool schema/parser is implicated, not the prompt")
    ap.add_argument("--one-tool", action="store_true",
                    help="send only the FIRST function of each class -- if that fixes a "
                         "failing class, the problem scales with the tool list")
    args = ap.parse_args()

    base = args.url.split("/v1/")[0].rstrip("/")
    model = None
    try:
        import urllib.request
        with urllib.request.urlopen(f"{base}/get_model_info", timeout=5) as r:
            model = json.loads(r.read().decode()).get("model_path")
    except Exception:  # noqa: BLE001
        pass
    if not model:
        print(f"[error] cannot reach {base}/get_model_info -- are the servers up?")
        return 1
    print(f"[repro] url={args.url}\n[repro] served model={model}")
    print(f"[repro] mode: {'NO TOOLS' if args.no_tools else ('ONE TOOL' if args.one_tool else 'full tool list')}"
          f"  max_tokens={args.max_tokens}\n")

    items = [json.loads(l) for l in open(_p("data/BFCL_v3_multi_turn_base.json"))]
    for cls in args.classes:
        stem = CLASS_TO_FILE.get(cls)
        if not stem:
            print(f"  {cls}: unknown class"); continue
        case = find_case(items, cls)
        if case is None:
            print(f"  {cls}: no dataset item found"); continue
        # first user turn of that case
        q = case.get("question") or []
        user = ""
        for turn in q:
            for m in (turn if isinstance(turn, list) else [turn]):
                if isinstance(m, dict) and m.get("role") == "user":
                    user = m.get("content", "")
                    break
            if user:
                break
        if not user:
            print(f"  {cls}: could not extract a user message"); continue
        docs = load_func_doc(stem)
        tools = None if args.no_tools else (docs[:1] if args.one_tool else docs)
        n = 0 if tools is None else len(tools)
        probe(args.url, model, tools, user, args.max_tokens,
              f"{cls:<20} (id={case['id']}, {n} tools, prompt {len(user)} chars)")
    print("\nRead it like this:")
    print("  - EMPTY with finish_reason='stop' and completion_tokens=0 -> the model")
    print("    emitted EOS immediately: chat-template / tool-prompt problem for that set.")
    print("  - EMPTY only with the full tool list but ok with --one-tool -> the tool")
    print("    list size or one specific schema is the trigger.")
    print("  - ok with --no-tools but EMPTY with tools -> tool schema or tool-call parser")
    print("    (--tool-call-parser llama3), not the conversation.")
    print("  - HTTP 400 -> the body now names the offending field.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
