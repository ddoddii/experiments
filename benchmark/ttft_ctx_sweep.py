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


def request(url, input_ids, gen=1, timeout=900):
    """Stream /generate and return (ttft_s, total_s, n_out).

    ttft is the first non-empty chunk, which arrives once prefill is done.

    gen>1 keeps the stream open to the end so end-to-end throughput can be measured.
    That matters because throughput derived from a gen=1 run is not a measurement: with
    one output token, tokens/second is just context/TTFT, a deterministic transform of
    the TTFT panel, and plotting both would show the same number twice. Only a real
    decode phase makes the second panel independent of the first."""
    t0 = time.time()
    r = requests.post(
        f"{url}/generate",
        # ignore_eos is what makes the throughput panel a measurement. The prompts are
        # random token ids, so the model hits EOS at an unpredictable point: without this
        # the same nominal 128-token request produced ~28 tokens at L=1k and ~128 at
        # L=2k, and since decode dominates the wall time, the throughput curve became a
        # plot of how long each random prompt happened to babble -- both arms dipping at
        # the same lengths, non-monotonic, unrelated to context length or to caching.
        # Recording the true n_out (below) fixes the numerator but not the denominator;
        # the output length has to be held CONSTANT, not merely counted.
        json={"input_ids": input_ids,
              "sampling_params": {"max_new_tokens": gen, "temperature": 0.0,
                                  "ignore_eos": True},
              "stream": True},
        stream=True, timeout=timeout,
    )
    r.raise_for_status()
    first, n_out, last = None, None, None
    for line in r.iter_lines():
        if not line:
            continue
        if first is None:
            first = time.time() - t0
        last = line
    total = time.time() - t0
    r.close()
    # The server's own count. With ignore_eos it should always equal `gen`; it is
    # read back rather than assumed so the caller can verify that (see the warn).
    if last:
        try:
            txt = last.decode() if isinstance(last, bytes) else last
            obj = json.loads(txt[5:] if txt.startswith("data:") else txt)
            n_out = obj.get("meta_info", {}).get("completion_tokens")
        except Exception:  # noqa: BLE001
            pass
    return first, total, (n_out if n_out else gen)


def ttft(url, input_ids, timeout=600):
    """Backward-compatible TTFT-only helper."""
    return request(url, input_ids, gen=1, timeout=timeout)[0]


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
    ap.add_argument("--gen-tokens", type=int, default=128,
                    help="output tokens per request. >1 makes the throughput panel a "
                         "measurement rather than a restatement of TTFT; 1 reproduces "
                         "the original TTFT-only sweep.")
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

    G = max(1, args.gen_tokens)
    lengths, cached_ms, recompute_ms, raw = [], [], [], {}
    cached_tps, recompute_tps = [], []
    print(f"\n{'L (tok)':>8}  {'cached':>9}  {'recompute':>10}  {'ratio':>6}"
          f"  {'cached t/s':>11}  {'recomp t/s':>11}")
    for L in args.lengths:
        base = gen_ids(L, rng)             # fixed base prefix for this L
        cac, rec = [], []
        cac_tp, rec_tp = [], []
        cac_n, rec_n = [], []
        try:
            for _ in range(args.reps):
                ttft(args.url, base)                                   # 1) warm/refresh prefix
                # 2) cached hit: the L-token prefix is resident, only `delta` is prefilled
                f, tot, n = request(args.url, base + gen_ids(args.delta, rng), gen=G)
                cac.append(f)
                # End-to-end token rate: prompt + output over the whole request. Prompt
                # tokens are counted for BOTH arms even though the cached arm did not
                # recompute them -- the request still delivered a response conditioned on
                # all L of them, and charging only the 8 fresh ones would credit the
                # cached arm with doing less work rather than with doing it faster.
                cac_tp.append((L + args.delta + n) / tot if tot else None)
                cac_n.append(n)
                f, tot, n = request(args.url, gen_ids(L, rng), gen=G)   # 3) cold miss
                rec.append(f)
                rec_tp.append((L + n) / tot if tot else None)
                rec_n.append(n)
        except Exception as e:  # noqa: BLE001
            # one bad length (usually L > server --context-length, or KV OOM) must not
            # discard the lengths that already succeeded -> skip it and keep going.
            print(f"{L:>8}  [skipped: {type(e).__name__}: {e}]")
            print(f"           -> likely L exceeds the server's --context-length "
                  f"(restart with CONTEXT_LENGTH>{L}) or KV OOM (lower reps / free a GPU)")
            continue
        cm, rm = mean_ms(cac), mean_ms(rec)
        ct = round(st.mean([x for x in cac_tp if x]), 1) if any(cac_tp) else None
        rt = round(st.mean([x for x in rec_tp if x]), 1) if any(rec_tp) else None
        lengths.append(L); cached_ms.append(cm); recompute_ms.append(rm)
        cached_tps.append(ct); recompute_tps.append(rt)
        # n_out is recorded, not just used: when the throughput panel came out
        # non-monotonic the output length was the cause, and it had to be reverse-solved
        # from the token rate because nothing stored it.
        raw[str(L)] = {"cached": cac, "recompute": rec,
                       "cached_tps": cac_tp, "recompute_tps": rec_tp,
                       "cached_n_out": cac_n, "recompute_n_out": rec_n}
        # With ignore_eos every request must emit exactly G tokens. If it does not, the
        # decode time varies per request and the throughput panel is measuring output
        # length rather than prefill -- say so loudly instead of plotting it.
        odd = [x for x in cac_n + rec_n if x != G]
        if odd:
            print(f"  [warn] L={L}: {len(odd)}/{len(cac_n) + len(rec_n)} requests "
                  f"returned n_out != {G} (saw {sorted(set(odd))}). Decode time is not "
                  f"constant, so the throughput panel is NOT comparable across lengths.")
        ratio = f"{rm / cm:.1f}x" if (cm and rm) else "-"
        print(f"{L:>8}  {str(cm)+'ms':>9}  {str(rm)+'ms':>10}  {ratio:>6}"
              f"  {str(ct):>11}  {str(rt):>11}")

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    js = args.out.rsplit(".", 1)[0] + ".json"
    json.dump({"lengths": lengths, "cached_ms": cached_ms,
               "recompute_ms": recompute_ms, "cached_tps": cached_tps,
               "recompute_tps": recompute_tps, "gen_tokens": G,
               "reps": args.reps, "raw": raw},
              open(js, "w"), indent=2)
    print(f"\n[saved] {js}")
    render(lengths, cached_ms, recompute_ms, args.out,
           cached_tps if G > 1 else None, recompute_tps if G > 1 else None, G)


def render(lengths, cached_ms, recompute_ms, out,
           cached_tps=None, recompute_tps=None, gen_tokens=1):
    """One panel when only TTFT was measured, two when a decode phase was included.

    The throughput panel is drawn ONLY from a gen>1 run. With one output token the token
    rate is context/TTFT, an exact transform of panel (a), and two panels showing the
    same measurement twice would overstate how much evidence there is."""
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

    two = bool(cached_tps and recompute_tps and any(cached_tps) and any(recompute_tps))
    use_paper_style()
    fig, axes = plt.subplots(1, 2 if two else 1,
                             figsize=((7.0, 2.9) if two else (3.7, 2.9)))
    axes = list(axes) if two else [axes]
    xs = [L / 1000.0 for L in lengths]

    def draw(ax, cached, recompute, ylabel, title):
        for ys, color, label, key in (
            (recompute, PALETTE["recompute"], "recompute", "recompute"),
            (cached, PALETTE["park"], "cached", "park"),
        ):
            gx = [x for x, y in zip(xs, ys) if y is not None]
            gy = [y for y in ys if y is not None]
            ax.plot(gx, gy, color=color, lw=1.7, ls=STYLE[key]["ls"],
                    marker=STYLE[key]["marker"], ms=5, label=label, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.set_xscale("log", base=2)
        ax.set_xticks(xs)
        ax.set_xticklabels([f"{int(x)}" if x >= 1 else f"{x:g}" for x in xs])
        ax.set_xlabel("context length (k tokens)")
        ax.set_ylabel(ylabel)
        if title:
            ax.set_title(title)
        style_axes(ax)

    draw(axes[0], cached_ms, recompute_ms, "mean TTFT (ms)",
         "(a) Time to first token" if two else None)
    if two:
        draw(axes[1], cached_tps, recompute_tps, "throughput (tok/s)",
             f"(b) End-to-end throughput ({gen_tokens} output tokens)")
    axes[0].legend(frameon=True, fancybox=False, edgecolor=PALETTE["muted"],
                   facecolor="white", handlelength=2.6, loc="upper left")
    fig.tight_layout()
    savefig(fig, out.rsplit(".", 1)[0])


if __name__ == "__main__":
    main()
