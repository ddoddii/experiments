#!/usr/bin/env python3
"""
QPS load sweep: mean TTFT, mean TPOT, and total throughput vs request rate (QPS),
comparing cached-prefix-hit vs recompute (full re-prefill). Same spirit as the
scheduling-policy QPS figure, but the two curves are 'cached' vs 'recompute'.

Fixed context length L. Poisson (exponential inter-arrival) arrivals at each target
QPS, driven against ONE radix-cache-ON SGLang server started with BATCHING enabled
(MAX_RUNNING high). Two arms on the same server:
  - recompute : every request is a FRESH unique L-token prompt -> full prefill (miss).
  - cached    : every request is a shared warmed L-token prefix + a unique short delta
                -> the L-prefix HITS, only the delta is prefilled.
Each request decodes exactly --max-new-tokens tokens (ignore_eos) so TPOT/throughput
are clean. Because recompute pays L-prefill per request, it saturates the server at a
much lower QPS (TTFT knee earlier); cached sustains far higher load.

Per (arm, QPS): mean TTFT (ms), mean TPOT (ms/token), total throughput (tok/s).

IMPORTANT: start the server with batching ON, e.g.
  MAX_RUNNING=256 CONTEXT_LENGTH=40000 ./scripts/sglang/start_single.sh

사용:
  python benchmark/qps_sweep.py --url http://127.0.0.1:30010 \
    --ctx-len 8000 --max-new-tokens 128 \
    --qps 2 4 8 12 16 24 32 48 64 --duration 20 \
    --out results/intro/fig_qps_sweep.png
"""
import argparse
import json
import os
import random
import statistics as st
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import requests


def gen_ids(n, rng, lo=10, hi=120000):
    return [rng.randint(lo, hi) for _ in range(n)]


def one_request(url, input_ids, max_new, timeout=600):
    """Streamed /generate; returns (ttft_s, tpot_s, n_out) or None on failure.
    ttft = first-chunk latency; tpot = (t_last - t_first)/(n_out-1). ignore_eos so
    exactly max_new tokens are produced -> fixed n_out."""
    t0 = time.time()
    t_first = t_last = None
    r = requests.post(
        f"{url}/generate",
        json={"input_ids": input_ids,
              "sampling_params": {"max_new_tokens": max_new, "temperature": 0.0,
                                  "ignore_eos": True},
              "stream": True},
        stream=True, timeout=timeout,
    )
    r.raise_for_status()
    for line in r.iter_lines():
        if not line:
            continue
        now = time.time()
        if t_first is None:
            t_first = now
        t_last = now
    r.close()
    if t_first is None:
        return None
    tpot = (t_last - t_first) / (max_new - 1) if max_new > 1 else 0.0
    return (t_first - t0, tpot, max_new)


def run_point(url, qps, duration, make_ids, max_new, pool, drain_timeout=180):
    """Drive Poisson arrivals at `qps` for `duration` s, then drain. Returns dict of
    mean ttft_ms, mean tpot_ms, throughput_tok_s, n_ok, n_fail."""
    results = []
    lock = threading.Lock()
    rng_arr = random.Random(0xC0FFEE ^ int(qps * 1000))

    def task(ids):
        try:
            res = one_request(url, ids, max_new)
        except Exception:  # noqa: BLE001
            res = None
        with lock:
            results.append(res)

    t_start = time.time()
    futures = []
    while time.time() - t_start < duration:
        futures.append(pool.submit(task, make_ids()))
        time.sleep(rng_arr.expovariate(qps))     # exponential inter-arrival
    # drain outstanding
    for f in futures:
        try:
            f.result(timeout=drain_timeout)
        except Exception:  # noqa: BLE001
            with lock:
                results.append(None)
    t_end = time.time()

    ok = [r for r in results if r is not None]
    n_fail = len(results) - len(ok)
    if not ok:
        return {"qps": qps, "ttft_ms": None, "tpot_ms": None,
                "throughput_tok_s": 0.0, "n_ok": 0, "n_fail": n_fail}
    ttft_ms = 1000 * st.mean(r[0] for r in ok)
    tpot_ms = 1000 * st.mean(r[1] for r in ok)
    total_tokens = sum(r[2] for r in ok)
    throughput = total_tokens / (t_end - t_start)
    return {"qps": qps, "ttft_ms": round(ttft_ms, 1), "tpot_ms": round(tpot_ms, 3),
            "throughput_tok_s": round(throughput, 1), "n_ok": len(ok), "n_fail": n_fail}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:30010")
    ap.add_argument("--ctx-len", type=int, default=8000, help="fixed context length L")
    ap.add_argument("--delta", type=int, default=8, help="new tokens per cached request")
    ap.add_argument("--max-new-tokens", type=int, default=128, help="decode length per req")
    ap.add_argument("--qps", nargs="+", type=float,
                    default=[2, 4, 8, 12, 16, 24, 32, 48, 64])
    ap.add_argument("--duration", type=float, default=20, help="seconds of load per QPS point")
    ap.add_argument("--arms", nargs="+", default=["cached", "recompute"],
                    choices=["cached", "recompute"])
    ap.add_argument("--workers", type=int, default=1024, help="thread pool size")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/intro/fig_qps_sweep.png")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    L, M = args.ctx_len, args.max_new_tokens
    shared_base = gen_ids(L, rng)   # cached arm's shared prefix

    # id factories per arm
    id_lock = threading.Lock()
    def make_cached():
        with id_lock:
            d = gen_ids(args.delta, rng)
        return shared_base + d
    def make_recompute():
        with id_lock:
            return gen_ids(L, rng)
    factory = {"cached": make_cached, "recompute": make_recompute}

    pool = ThreadPoolExecutor(max_workers=args.workers)
    data = {}
    with pool:
        for arm in args.arms:
            print(f"\n=== arm: {arm}  (L={L}, decode={M}) ===")
            # warm: for cached, seat the shared prefix; for recompute, just wake the server
            try:
                one_request(args.url, shared_base if arm == "cached" else gen_ids(512, rng), M)
            except Exception as e:  # noqa: BLE001
                print(f"[error] server not reachable: {e}\n  start it with "
                      f"MAX_RUNNING=256 ./scripts/sglang/start_single.sh")
                return
            print(f"  {'QPS':>5}  {'TTFT(ms)':>9}  {'TPOT(ms)':>9}  {'tok/s':>8}  "
                  f"{'ok':>4}  {'fail':>4}")
            rows = []
            for q in args.qps:
                row = run_point(args.url, q, args.duration, factory[arm], M, pool)
                rows.append(row)
                print(f"  {q:>5g}  {str(row['ttft_ms']):>9}  {str(row['tpot_ms']):>9}  "
                      f"{row['throughput_tok_s']:>8}  {row['n_ok']:>4}  {row['n_fail']:>4}")
            data[arm] = rows

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    js = args.out.rsplit(".", 1)[0] + ".json"
    json.dump({"ctx_len": L, "max_new_tokens": M, "duration": args.duration, "arms": data},
              open(js, "w"), indent=2)
    print(f"\n[saved] {js}")
    render(data, args.out)


def render(data, out):
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from paperstyle import PALETTE, STYLE, use_paper_style, style_axes, savefig

    # arm -> (palette key, STYLE key). cached borrows the green 'park' style.
    style = {"cached": ("park", "park"), "recompute": ("recompute", "recompute")}
    panels = [("ttft_ms", "mean TTFT (ms)", "(a) TTFT vs QPS"),
              ("tpot_ms", "mean TPOT (ms/token)", "(b) TPOT vs QPS"),
              ("throughput_tok_s", "total throughput (tok/s)", "(c) Throughput vs QPS")]

    use_paper_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 2.9))
    for ax, (key, ylabel, title) in zip(axes, panels):
        for arm, rows in data.items():
            pkey, skey = style.get(arm, ("ink", "hicache"))
            xs = [r["qps"] for r in rows if r.get(key) is not None]
            ys = [r[key] for r in rows if r.get(key) is not None]
            ax.plot(xs, ys, color=PALETTE[pkey], lw=1.6, ls=STYLE[skey]["ls"],
                    marker=STYLE[skey]["marker"], ms=4, label=arm, zorder=3,
                    markeredgecolor="white", markeredgewidth=0.6)
        ax.set_xlabel("QPS (requested rate)")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        style_axes(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.tight_layout(rect=[0, 0.08, 1, 1])
    fig.legend(handles, labels, loc="lower center", ncol=len(labels), frameon=True,
               fancybox=False, edgecolor=PALETTE["muted"], handlelength=2.6,
               columnspacing=2.0, bbox_to_anchor=(0.5, 0.0))
    savefig(fig, out.rsplit(".", 1)[0])


if __name__ == "__main__":
    main()
