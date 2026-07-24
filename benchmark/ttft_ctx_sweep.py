#!/usr/bin/env python3
"""
Intro motivation microbenchmark: TTFT vs CONTEXT LENGTH, cached-prefix-hit vs
recompute (full re-prefill). Controlled fixed lengths (1k/2k/4k/8k/16k/32k), NOT a
dataset -- so the recompute cost curve is clean and the "prefix cache eviction is
expensive" point is unambiguous.

Runs against ONE plain radix-cache-ON SGLang server (scripts/sglang/start_single.sh).
Uses the native /generate endpoint with explicit `input_ids` so context length is
exact and controllable. For each length L, per rep:
  1. warm the L-token base prefix (so it's resident)
  2. cached    = TTFT of (base_L + tiny delta)  -> hits the L-prefix, prefills only delta
  3. recompute = TTFT of a FRESH unique L-token prompt -> full prefill of L (a cold miss,
     which is exactly what you pay when the prefix cache has been evicted)
Ordering (warm -> cached -> recompute) each rep guarantees the cached measurement can't
be contaminated by the recompute prompt evicting the base.

Output: JSON {lengths, cached_ms, recompute_ms, raw} + a single-panel figure (log2 x).

사용:
  # the server's --context-length must exceed the LARGEST sweep length (+delta).
  # start_single.sh defaults to 40000, so 64k needs CONTEXT_LENGTH=70000.
  CONTEXT_LENGTH=70000 ./scripts/sglang/start_single.sh    # start the server first
  python benchmark/ttft_ctx_sweep.py \
    --url http://127.0.0.1:30010 \
    --lengths 1000 2000 4000 8000 16000 32000 64000 --reps 5 \
    --out results/intro/fig_ttft_ctx_sweep.png
"""
import argparse
import json
import os
import random
import statistics as st
import time

import requests


def gen_ids(n, rng, lo=10, hi=120000):
    """n random token ids well below the Llama-3.1 special-token range (>=128000)."""
    return [rng.randint(lo, hi) for _ in range(n)]


def ttft(url, input_ids, timeout=600):
    """Time-to-first-token via streaming /generate (max_new_tokens=1). The first
    streamed chunk arrives after prefill completes, so this is prefill-dominated."""
    t0 = time.time()
    r = requests.post(
        f"{url}/generate",
        json={"input_ids": input_ids,
              "sampling_params": {"max_new_tokens": 1, "temperature": 0.0},
              "stream": True},
        stream=True, timeout=timeout,
    )
    r.raise_for_status()
    for line in r.iter_lines():
        if line:                       # first non-empty chunk = first token
            dt = time.time() - t0
            r.close()
            return dt
    return None


def mean_ms(xs):
    xs = [x for x in xs if x is not None]
    return round(1000 * st.mean(xs), 1) if xs else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30010")
    ap.add_argument("--lengths", nargs="+", type=int,
                    default=[1000, 2000, 4000, 8000, 16000, 32000])
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--delta", type=int, default=8, help="new tokens on the cached turn")
    ap.add_argument("--warmup", type=int, default=2, help="throwaway reqs before timing")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/intro/fig_ttft_ctx_sweep.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)

    # server warmup (CUDA graph capture / first-call lag)
    print(f"warming up server at {args.url} ...")
    for _ in range(max(1, args.warmup)):
        try:
            ttft(args.url, gen_ids(512, rng))
        except Exception as e:  # noqa: BLE001
            print(f"[error] server not reachable: {e}")
            print("  start it first:  ./scripts/sglang/start_single.sh")
            return

    lengths, cached_ms, recompute_ms, raw = [], [], [], {}
    print(f"\n{'L (tok)':>8}  {'cached':>9}  {'recompute':>10}  {'ratio':>6}")
    for L in args.lengths:
        base = gen_ids(L, rng)             # fixed base prefix for this L
        cac, rec = [], []
        try:
            for _ in range(args.reps):
                ttft(args.url, base)                                   # 1) warm/refresh prefix
                cac.append(ttft(args.url, base + gen_ids(args.delta, rng)))  # 2) cached hit
                rec.append(ttft(args.url, gen_ids(L, rng)))            # 3) recompute (unique miss)
        except Exception as e:  # noqa: BLE001
            # one bad length (usually L > server --context-length, or KV OOM) must not
            # discard the lengths that already succeeded -> skip it and keep going.
            print(f"{L:>8}  [skipped: {type(e).__name__}: {e}]")
            print(f"           -> likely L exceeds the server's --context-length "
                  f"(restart with CONTEXT_LENGTH>{L}) or KV OOM (lower reps / free a GPU)")
            continue
        cm, rm = mean_ms(cac), mean_ms(rec)
        lengths.append(L); cached_ms.append(cm); recompute_ms.append(rm)
        raw[str(L)] = {"cached": cac, "recompute": rec}
        ratio = f"{rm / cm:.1f}x" if (cm and rm) else "-"
        print(f"{L:>8}  {str(cm)+'ms':>9}  {str(rm)+'ms':>10}  {ratio:>6}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    js = args.out.rsplit(".", 1)[0] + ".json"
    json.dump({"lengths": lengths, "cached_ms": cached_ms,
               "recompute_ms": recompute_ms, "reps": args.reps, "raw": raw},
              open(js, "w"), indent=2)
    print(f"\n[saved] {js}")
    render(lengths, cached_ms, recompute_ms, args.out)


def render(lengths, cached_ms, recompute_ms, out):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

    use_paper_style()
    fig, ax = plt.subplots(figsize=(3.7, 2.9))
    xs = [L / 1000.0 for L in lengths]
    for ys, color, label, key in (
        (recompute_ms, PALETTE["recompute"], "recompute", "recompute"),
        (cached_ms, PALETTE["park"], "cached", "park"),
    ):
        ax.plot(xs, ys, color=color, lw=1.7, ls=STYLE[key]["ls"], marker=STYLE[key]["marker"],
                ms=5, label=label, zorder=3, markeredgecolor="white", markeredgewidth=0.6)
    ax.set_xscale("log", base=2)
    ax.set_xticks(xs)
    ax.set_xticklabels([f"{int(x)}" if x >= 1 else f"{x:g}" for x in xs])
    ax.set_xlabel("context length (k tokens)")
    ax.set_ylabel("mean TTFT (ms)")
    ax.set_title("Recompute cost vs context length")
    style_axes(ax)
    ax.legend(frameon=True, fancybox=False, edgecolor=PALETTE["muted"], facecolor="white",
              handlelength=2.6, loc="upper left")
    fig.tight_layout()
    savefig(fig, out.rsplit(".", 1)[0])


if __name__ == "__main__":
    main()
