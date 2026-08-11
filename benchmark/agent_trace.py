#!/usr/bin/env python3
"""
Load recorded agent trajectories (SWE-agent / OpenHands) into a normalised replay
format, and VERIFY THE ONE PROPERTY THE PARK MECHANISM DEPENDS ON before any run.

WHY THE VERIFIER EXISTS
  The parked-KV index hashes an exact token sequence, so a parked prefix is only
  findable if turn N's rendered prompt is a strict token-prefix of turn N+1's. Two full
  Qwen3-14B runs were thrown away because that silently did not hold -- the chat
  template rendered the generation prompt differently from history, every lookup missed,
  fetch_hits was 0, and the figures still looked healthy (see
  results/models/qwen14b/INVALID.md). Agent traces are exactly where this breaks: tool
  calls and observations are rendered by template branches that ordinary chat turns
  never touch. Verify per model, before spending hours of GPU time.

NORMALISED FORMAT
  {"id": str,
   "turns": [{"prompt_messages": [...],   # what to send at this turn
              "recorded_reply": {...},    # the assistant message the trace recorded
              "delay_s": float|None}],    # gap BEFORE this turn, from trace timestamps
   "source": "swe-agent"|"openhands"}

사용:
  # 1. inspect what a file actually contains (run this first on a new dataset)
  python benchmark/agent_trace.py --dump path/to/trajectory.json

  # 2. load a directory of traces and report turn/context statistics
  python benchmark/agent_trace.py --dir traces/ --stats

  # 3. THE GATE: does this model's chat template keep the prompt append-only?
  MODEL_PATH=/home/uhmturks/hf_models/Llama-3.1-8B-Instruct \
  python benchmark/agent_trace.py --dir traces/ --verify-append-only
"""
import argparse
import glob
import json
import os


# --------------------------------------------------------------------------- loading
# ShareGPT/Vicuna-style `from`/`value`, which is what Inferact/codex_swebenchpro_traces
# ships (610 rows, each a `conversations` list of 12-200 messages). No timestamps, so an
# inter-turn delay has to be imposed rather than measured -- that becomes a stated
# parameter of the experiment, not a property of the trace.
_FROM_ROLE = {
    "human": "user", "user": "user",
    "observation": "user", "tool": "user", "function_response": "user",
    "gpt": "assistant", "assistant": "assistant", "chatgpt": "assistant",
    # A function_call is an assistant EMISSION, so it is its own generation step and
    # must count as a turn; folding it into the following text reply would undercount
    # turns and, worse, merge two prompts the server actually saw separately.
    "function_call": "assistant",
    "system": "system",
}


def _messages_sharegpt(conv):
    if not isinstance(conv, list) or not conv:
        return None
    out = []
    for m in conv:
        if not isinstance(m, dict) or "value" not in m:
            return None
        src = str(m.get("from") or m.get("role") or "").lower()
        role = _FROM_ROLE.get(src)
        if role is None:
            return None
        out.append({"role": role, "content": m["value"] or ""})
    return out


def _messages_swe_agent(doc):
    """SWE-agent .traj: the `history` field is already an OpenAI-style message list.
    `trajectory` holds the parallel action/observation view, which is lossy for our
    purpose (it drops the system prompt and the exact rendering), so history wins."""
    hist = doc.get("history")
    if not isinstance(hist, list) or not hist:
        return None
    out = []
    for m in hist:
        if not isinstance(m, dict) or "role" not in m:
            return None
        msg = {"role": m["role"], "content": m.get("content") or ""}
        # SWE-agent carries thought/action alongside content; keep tool_calls if the
        # trace used the structured form, since that is what changes the rendering.
        if m.get("tool_calls"):
            msg["tool_calls"] = m["tool_calls"]
        if m.get("tool_call_id"):
            msg["tool_call_id"] = m["tool_call_id"]
        out.append(msg)
    return out


def _messages_openhands(doc):
    """OpenHands: a list of events with source/action/observation and a timestamp.
    Returns (messages, timestamps) -- timestamps are what make a REAL inter-turn delay
    distribution possible instead of an assumed one."""
    events = doc if isinstance(doc, list) else doc.get("events") or doc.get("history")
    if not isinstance(events, list) or not events:
        return None, None
    msgs, ts = [], []
    for e in events:
        if not isinstance(e, dict):
            return None, None
        src = e.get("source")
        if src is None and "role" not in e:
            return None, None
        role = e.get("role") or ("assistant" if src == "agent" else "user")
        content = (
            e.get("content")
            or e.get("message")
            or (e.get("args") or {}).get("content")
            or ""
        )
        msgs.append({"role": role, "content": content if isinstance(content, str)
                     else json.dumps(content)})
        ts.append(e.get("timestamp"))
    return msgs, ts


def _parse_ts(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    try:
        from datetime import datetime
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except Exception:  # noqa: BLE001
        return None


def _session_from_messages(sid, msgs, stamps=None, source="sharegpt"):
    """Split a message list at every assistant emission -- each is one generation step,
    so its prompt is everything before it and its recorded reply is that message."""
    turns, prev_reply_t = [], None
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i == 0:
            continue
        reply_t = _parse_ts(stamps[i]) if stamps else None
        trigger_t = _parse_ts(stamps[i - 1]) if stamps else None
        delay = (trigger_t - prev_reply_t) if (
            trigger_t is not None and prev_reply_t is not None) else None
        turns.append({
            "prompt_messages": msgs[:i],
            "recorded_reply": m,
            "delay_s": round(delay, 3) if delay and delay > 0 else None,
        })
        if reply_t is not None:
            prev_reply_t = reply_t
    if not turns:
        return None
    return {"id": sid, "turns": turns, "source": source}


def load_multi(path):
    """One file holding MANY sessions (the codex_swebenchpro.json layout: a list of rows,
    each with a `conversations` list). Also accepts JSONL."""
    with open(path) as fh:
        head = fh.read(1)
        fh.seek(0)
        rows = ([json.loads(ln) for ln in fh if ln.strip()] if head != "["
                else json.load(fh))
    if isinstance(rows, dict):
        rows = rows.get("data") or rows.get("train") or rows.get("rows") or []
    sessions = []
    for i, row in enumerate(rows):
        conv = row.get("conversations") if isinstance(row, dict) else None
        if conv is None and isinstance(row, list):
            conv = row
        msgs = _messages_sharegpt(conv) if conv is not None else None
        if not msgs:
            continue
        s = _session_from_messages(f"{os.path.basename(path)}#{i}", msgs)
        if s:
            sessions.append(s)
    return sessions


def load_trace(path):
    """One trajectory file -> normalised session, or None if the format is unrecognised.

    Deliberately returns None rather than guessing: a mis-parsed trace produces a
    plausible-looking session with the wrong context growth, which is the failure mode
    that is hardest to notice downstream."""
    with open(path) as fh:
        doc = json.load(fh)

    msgs = _messages_swe_agent(doc) if isinstance(doc, dict) else None
    source, stamps = "swe-agent", None
    if msgs is None:
        msgs, stamps = _messages_openhands(doc)
        source = "openhands"
    if not msgs:
        return None

    # A turn is a point where the agent was asked to generate: the prompt is everything
    # before an assistant message, and that assistant message is the recorded reply.
    turns, prev_reply_t = [], None
    for i, m in enumerate(msgs):
        if m["role"] != "assistant" or i == 0:
            continue
        reply_t = _parse_ts(stamps[i]) if stamps else None
        # The gap we want is THINK TIME: previous reply -> the input that triggers this
        # turn. Measuring previous reply -> THIS reply instead would fold the recorded
        # model's own generation time into the delay, which is (a) not think time and
        # (b) generation on someone else's hardware, so replaying it would inject a
        # sleep proportional to a latency we are trying to measure ourselves.
        trigger_t = _parse_ts(stamps[i - 1]) if stamps else None
        delay = (trigger_t - prev_reply_t) if (
            trigger_t is not None and prev_reply_t is not None) else None
        turns.append({
            "prompt_messages": msgs[:i],
            "recorded_reply": m,
            "delay_s": round(delay, 3) if delay and delay > 0 else None,
        })
        if reply_t is not None:
            prev_reply_t = reply_t
    if not turns:
        return None
    return {"id": os.path.basename(path), "turns": turns, "source": source}


def load_dir(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else sorted(
        glob.glob(os.path.join(pattern, "**", "*.json"), recursive=True)
        + glob.glob(os.path.join(pattern, "**", "*.traj"), recursive=True)
    )
    sessions, skipped = [], []
    for p in paths:
        try:
            s = load_trace(p)
        except Exception as e:  # noqa: BLE001
            skipped.append((p, repr(e)))
            continue
        (sessions.append(s) if s else skipped.append((p, "unrecognised format")))
    return sessions, skipped


# --------------------------------------------------------------------------- verify
def verify_append_only(sessions, tokenizer, limit=None):
    """Is turn N's rendered prompt a strict TOKEN prefix of turn N+1's?

    Renders through the model's own chat template, which is the only thing that matters
    -- comparing the message lists would pass even when the template breaks the property,
    and the template is where it actually broke last time."""
    bad, checked = [], 0
    for s in sessions[: limit or len(sessions)]:
        prev = None
        for ti, t in enumerate(s["turns"]):
            ids = tokenizer.apply_chat_template(
                t["prompt_messages"], tokenize=True, add_generation_prompt=True
            )
            if prev is not None:
                checked += 1
                n = min(len(prev), len(ids))
                if len(ids) < len(prev) or list(prev) != list(ids[: len(prev)]):
                    # Report where it diverged: a template that appends a few tokens in
                    # the wrong place looks identical to one that rewrites history unless
                    # you say which token index broke.
                    div = next((i for i in range(n) if prev[i] != ids[i]), n)
                    bad.append({
                        "session": s["id"], "turn": ti,
                        "prev_len": len(prev), "this_len": len(ids),
                        "diverged_at_token": div,
                    })
            prev = ids
    return checked, bad


def measure_and_filter(sessions, tokenizer, min_turns, max_turns, min_ctx, max_ctx,
                       n_want, max_out):
    """Keep sessions whose SHAPE matches the experiment spec, measured with the real
    tokenizer rather than a chars/4 estimate -- the context bounds are the whole point of
    the selection, and a 25% estimation error would silently admit sessions well outside
    them.

    Truncates each session at the last turn still under max_ctx instead of dropping it:
    an agent trace that eventually runs to 100k is a perfectly good 8-15 turn session up
    to the point it crosses the bound, and dropping it biases the set toward traces that
    happened to finish early."""
    kept, stats = [], []
    for s in sessions:
        lens, out_lens = [], []
        for t in s["turns"]:
            ids = tokenizer.apply_chat_template(
                t["prompt_messages"], tokenize=True, add_generation_prompt=True)
            lens.append(len(ids))
            out_lens.append(len(tokenizer(t["recorded_reply"]["content"],
                                          add_special_tokens=False)["input_ids"]))
        keep_n = sum(1 for L in lens if L <= max_ctx)
        if keep_n < min_turns or not lens:
            continue
        keep_n = min(keep_n, max_turns)
        if lens[keep_n - 1] < min_ctx:      # never gets long enough to be interesting
            continue
        s2 = dict(s)
        s2["turns"] = [
            dict(t, prompt_tokens=lens[i], recorded_out_tokens=min(out_lens[i], max_out))
            for i, t in enumerate(s["turns"][:keep_n])
        ]
        s2["peak_prompt_tokens"] = lens[keep_n - 1]
        kept.append(s2)
        stats.append((len(s2["turns"]), lens[0], lens[keep_n - 1]))
        if len(kept) >= n_want:
            break
    return kept, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", help="directory of traces, or a glob")
    ap.add_argument("--file", help="single file holding many sessions (codex_swebenchpro.json)")
    ap.add_argument("--select", action="store_true",
                     help="filter to the experiment's session set and write --out")
    ap.add_argument("--min-turns", type=int, default=8)
    ap.add_argument("--max-turns", type=int, default=15)
    ap.add_argument("--min-ctx", type=int, default=12000)
    ap.add_argument("--max-ctx", type=int, default=48000)
    ap.add_argument("--n-sessions", type=int, default=100)
    ap.add_argument("--max-out-tokens", type=int, default=1024)
    ap.add_argument("--dump", help="print the parsed structure of ONE file and exit")
    ap.add_argument(
        "--peek",
        help="report the SCHEMA of a large multi-session file without assuming a "
             "format. Run this first on a new dataset: a parser written against a "
             "guessed shape yields sessions that look plausible and have the wrong "
             "context growth, which nothing downstream can detect.",
    )
    ap.add_argument("--peek-depth", type=int, default=3)
    ap.add_argument("--stats", action="store_true")
    ap.add_argument("--verify-append-only", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--out", help="write the normalised sessions here")
    args = ap.parse_args()

    if args.dump:
        s = load_trace(args.dump)
        if s is None:
            with open(args.dump) as fh:
                doc = json.load(fh)
            print(f"[unrecognised] {args.dump}")
            print(f"  top-level type: {type(doc).__name__}")
            if isinstance(doc, dict):
                print(f"  keys: {list(doc)[:20]}")
            elif doc:
                print(f"  first element keys: "
                      f"{list(doc[0])[:20] if isinstance(doc[0], dict) else type(doc[0])}")
            print("  -> add a parser for this shape in agent_trace.py; do NOT guess "
                  "downstream, a mis-parsed trace yields the wrong context growth.")
            raise SystemExit(2)
        print(f"[{s['source']}] {s['id']}: {len(s['turns'])} turns")
        for i, t in enumerate(s["turns"][:3]):
            print(f"  turn {i}: {len(t['prompt_messages'])} msgs  delay={t['delay_s']}")
            print(f"    last prompt msg: {str(t['prompt_messages'][-1])[:160]}")
        return

    if args.file:
        sessions, skipped = load_multi(args.file), []
        print(f"[loaded] {len(sessions)} sessions from {args.file}")
    elif args.dir:
        sessions, skipped = load_dir(args.dir)
        print(f"[loaded] {len(sessions)} sessions, {len(skipped)} skipped")
        for p, why in skipped[:5]:
            print(f"  skipped {p}: {why}")
    else:
        raise SystemExit("pass --file, --dir, --dump or --peek")
    if not sessions:
        raise SystemExit("no sessions parsed -- run --dump/--peek to see the shape")

    if args.select:
        model_path = os.environ.get("MODEL_PATH", "")
        if not model_path:
            raise SystemExit("MODEL_PATH required: the context bounds are measured with "
                              "the real tokenizer, not estimated from characters")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.model_max_length = int(1e9)
        kept, stats = measure_and_filter(
            sessions, tok, args.min_turns, args.max_turns,
            args.min_ctx, args.max_ctx, args.n_sessions, args.max_out_tokens)
        print(f"\n[select] {len(kept)}/{args.n_sessions} sessions matched "
              f"turns {args.min_turns}-{args.max_turns}, context "
              f"{args.min_ctx:,}-{args.max_ctx:,} tokens")
        if kept:
            import statistics as st
            print(f"  turns/session : min {min(x[0] for x in stats)} "
                  f"median {st.median([x[0] for x in stats])} "
                  f"max {max(x[0] for x in stats)}")
            print(f"  first-turn ctx: median {st.median([x[1] for x in stats]):,.0f} tok")
            print(f"  peak ctx      : median {st.median([x[2] for x in stats]):,.0f} "
                  f"max {max(x[2] for x in stats):,} tok")
        if len(kept) < args.n_sessions:
            print(f"  [warn] only {len(kept)} sessions matched. Widen --max-turns or "
                  f"--max-ctx, or lower --n-sessions -- do NOT silently run with fewer "
                  f"than planned, the concurrency sweep divides this set across arms.")
        sessions = kept

    if args.stats:
        import statistics as st
        nt = [len(s["turns"]) for s in sessions]
        ch = [len(str(t["prompt_messages"])) for s in sessions for t in s["turns"]]
        dl = [t["delay_s"] for s in sessions for t in s["turns"] if t["delay_s"]]
        print(f"  turns/session : min {min(nt)} median {st.median(nt)} max {max(nt)}")
        print(f"  prompt chars  : median {st.median(ch):,.0f} max {max(ch):,}"
              f"   (~{st.median(ch)/4:,.0f} / {max(ch)/4:,.0f} tokens at 4 chars/token)")
        if dl:
            print(f"  inter-turn delay (s): median {st.median(dl):.1f} "
                  f"p90 {sorted(dl)[int(.9*len(dl))]:.1f} max {max(dl):.1f}  n={len(dl)}")
        else:
            print("  inter-turn delay: NOT PRESENT in these traces -- the replay will "
                  "need an assumed distribution, and that assumption becomes a stated "
                  "parameter of the experiment rather than a measured property.")

    if args.verify_append_only:
        model_path = os.environ.get("MODEL_PATH", "")
        if not model_path:
            raise SystemExit("MODEL_PATH required to render the chat template")
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained(model_path)
        tok.model_max_length = int(1e9)
        checked, bad = verify_append_only(sessions, tok, args.limit)
        print(f"\n[append-only] checked {checked} turn transitions")
        if not bad:
            print("  PASS -- every turn's prompt is a strict token prefix of the next.")
            print("  Parked prefixes will be findable; the park arm can be trusted.")
        else:
            print(f"  FAIL on {len(bad)} transitions. The park index hashes exact token")
            print("  sequences, so EVERY lookup will miss and the park arm will measure")
            print("  nothing while still producing a healthy-looking figure. Do not run.")
            for b in bad[:5]:
                print(f"    {b['session']} turn {b['turn']}: prev {b['prev_len']} tok, "
                      f"now {b['this_len']} tok, diverged at token {b['diverged_at_token']}")
            raise SystemExit(1)

    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w") as fh:
            json.dump(sessions, fh)
        print(f"[saved] {args.out}  ({len(sessions)} sessions)")


if __name__ == "__main__":
    main()
