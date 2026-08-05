#!/usr/bin/env python3
"""
Does turn N's prompt remain a TOKEN-EXACT PREFIX of turn N+1's prompt?

Every prefix-reuse mechanism in this work depends on that being true, and the parked-KV
index depends on it most strictly: it stores a block under a rolling hash of the exact
token sequence, so a single token inserted or removed anywhere in the history makes every
later lookup miss. The radix cache degrades gracefully in the same situation -- it simply
matches a shorter prefix -- which is why a broken template shows up as "the park arm
contributes nothing" while the baselines still look healthy.

That is exactly what Qwen3-14B did: park pools filled normally, and fetch_hits stayed at
0 across 1623 lookups while SGLang's own hierarchical cache reached a 50.6% hit rate.

Runs entirely offline against the tokenizer and chat template -- no server, no GPU,
seconds rather than the two hours a sweep costs.

WHAT TO SUSPECT WHEN IT FAILS
  Qwen3's template inserts an empty "<think>\\n\\n</think>" block into the assistant slot
  when generating, and strips thinking content when re-rendering history. If those two
  do not agree, turn N's prompt is not a prefix of turn N+1's. `--no-thinking-kwarg`
  reproduces what the benchmark actually sends.

사용:
  python benchmark/check_prefix_continuity.py --model /home/uhmturks/hf_models/Qwen3-14B
  python benchmark/check_prefix_continuity.py --model .../Qwen3-14B --no-thinking-kwarg
  python benchmark/check_prefix_continuity.py --model .../Llama-3.1-8B-Instruct
"""
import argparse
import sys


def render(tok, messages, add_generation_prompt, kwargs):
    return tok.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=add_generation_prompt, **kwargs)


def continuity(tok, kwargs, turns, system, echo_prefix=""):
    """Render `turns` turns and return (all_exact, [(len_a, len_b, common), ...]).

    echo_prefix is prepended to each stored assistant message. That is the candidate
    repair: if the template appends a block to the ASSISTANT SLOT when generating but
    drops it when re-rendering history, writing that block back into the stored content
    makes the two agree again and the prompt becomes append-only."""
    msgs = [{"role": "system", "content": system}] if system else []
    prompts = []
    for t in range(turns):
        msgs = msgs + [{"role": "user", "content": f"Question number {t} about a topic."}]
        prompts.append(render(tok, msgs, True, kwargs))
        msgs = msgs + [{"role": "assistant",
                        "content": echo_prefix + f"This is answer number {t}. " * 8}]
    out, ok = [], True
    for i in range(len(prompts) - 1):
        a, b = prompts[i], prompts[i + 1]
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        ok &= common == len(a)
        out.append((len(a), len(b), common))
    return ok, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--turns", type=int, default=4)
    ap.add_argument("--no-thinking-kwarg", action="store_true",
                    help="do NOT pass enable_thinking=False (the benchmark passes it)")
    ap.add_argument("--system", default="You are a helpful assistant.")
    ap.add_argument("--try-fixes", action="store_true",
                    help="also evaluate candidate repairs and name the one that works")
    args = ap.parse_args()

    try:
        from transformers import AutoTokenizer
    except ImportError:
        print("needs transformers: pip install transformers")
        return 2
    tok = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)

    kwargs = {} if args.no_thinking_kwarg else {"enable_thinking": False}
    # apply_chat_template rejects unknown kwargs on templates that do not declare them,
    # so fall back rather than crash on a model without a thinking mode.
    try:
        render(tok, [{"role": "user", "content": "hi"}], True, kwargs)
    except (TypeError, ValueError) as e:
        print(f"[note] template rejected {kwargs} ({type(e).__name__}); rendering without it")
        kwargs = {}

    msgs = [{"role": "system", "content": args.system}] if args.system else []
    prompts = []
    for t in range(args.turns):
        msgs = msgs + [{"role": "user", "content": f"Question number {t} about a topic."}]
        # The prompt the server prefills for this turn.
        prompts.append(render(tok, msgs, True, kwargs))
        # The assistant reply, appended verbatim -- what the ShareGPT runner does.
        msgs = msgs + [{"role": "assistant",
                        "content": f"This is answer number {t}. " * 8}]

    print(f"model: {args.model}")
    print(f"chat_template_kwargs: {kwargs or '(none)'}\n")
    ok = True
    for i in range(len(prompts) - 1):
        a, b = prompts[i], prompts[i + 1]
        common = 0
        for x, y in zip(a, b):
            if x != y:
                break
            common += 1
        is_prefix = common == len(a)
        ok &= is_prefix
        print(f"turn {i} -> {i+1}: prompt {len(a):>5} -> {len(b):>5} tokens, "
              f"common prefix {common:>5}  "
              f"{'EXACT PREFIX' if is_prefix else 'BREAKS at token ' + str(common)}")
        if not is_prefix:
            lo = max(0, common - 6)
            print(f"    turn {i}   ...{tok.decode(a[lo:common+10])!r}")
            print(f"    turn {i+1} ...{tok.decode(b[lo:common+10])!r}")

    if args.try_fixes:
        print("\n--- candidate repairs ---")
        # The last candidate is the one that matters in a real run. With thinking left
        # ON the model actually EMITS a "<think>...</think>" block, and if that raw
        # output is stored verbatim the template strips it again on re-render -- so a
        # candidate that passes with clean stored replies can still fail in practice.
        # Storing the reply with the block removed is what the runner must do.
        cands = [
            ("as the benchmark sends it", kwargs, ""),
            ("without enable_thinking", {}, ""),
            ("echo the empty think block into stored replies",
             kwargs, "<think>\n\n</think>\n\n"),
            ("echo it, without enable_thinking", {}, "<think>\n\n</think>\n\n"),
            ("thinking on, RAW reply stored (block left in)",
             {}, "<think>\nreasoning here\n</think>\n\n"),
        ]
        best = None
        for name, kw, pre in cands:
            try:
                good, rows = continuity(tok, kw, args.turns, args.system, pre)
            except (TypeError, ValueError) as e:
                print(f"  {name:<46} n/a ({type(e).__name__})")
                continue
            worst = min((c - a) for a, _, c in rows) if rows else 0
            print(f"  {name:<46} {'PASS' if good else f'FAIL (short by {-worst} tok)'}")
            if good and best is None:
                best = (name, kw, pre)
        print()
        if best:
            print(f"USE: {best[0]}")
            print(f"     chat_template_kwargs={best[1] or '(none)'}  "
                  f"assistant_prefix={best[2]!r}")
        else:
            print("No candidate makes the prompt append-only. This model cannot carry the")
            print("reuse comparison; pick one whose template is append-only (the check")
            print("passes on Llama-3.1-8B) rather than working around the template.")

    print()
    if ok:
        print("PASS: every turn's prompt is a token-exact prefix of the next.")
        print("      Prefix reuse and the parked-KV index can both match. If the park arm")
        print("      still reports fetch_hits=0, the cause is elsewhere.")
    else:
        print("FAIL: the prompt is NOT append-only across turns.")
        print("      The radix cache will still match the unchanged head and look")
        print("      healthy; the parked-KV index, which hashes the exact sequence, will")
        print("      miss every lookup. This model cannot be used for the reuse")
        print("      comparison until the rendering is made append-only.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
