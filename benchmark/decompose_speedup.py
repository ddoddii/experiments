#!/usr/bin/env python3
"""
Split the measured speedup into the things that could have caused it, one variable at a
time -- link, capacity, transfer software -- instead of reporting a total and guessing.

WHY A CHAIN AND NOT A COMPARISON. "Ours beats hicache" moves placement policy, transfer
software and storage medium together, so the number cannot be attributed. run_why_faster.sh
adds park_host: our code with SGLANG_KV_PARK_FORCE_HOST=1, which is byte-for-byte the
park_gpu arm except that parks land in pinned host DRAM. That makes each step move one
thing:

    radix     -> park_host    policy + our transfer software  (medium fixed at DRAM)
    park_host -> park_gpu     MEDIUM ONLY                     (the "is it the link?" test)
    hicache   -> park_host    same medium, same PCIe bus      (so: software only)
    radix     -> park_gpu     the total

THE LINK BUDGET IS THE POINT OF THIS SCRIPT. A tier's cost per token is
bytes_per_token / bandwidth; the saving it buys is one token of re-prefill. On
Llama-3.1-8B (128 KiB/token) with the measured 52.7 GB/s peer-GPU and 26.3 GB/s host
figures from results/nvlink_microbench.json:

    peer GPU   0.0025 ms/token        host DRAM  0.0050 ms/token
    re-prefill 0.1317 ms/token

So switching medium changes the cost of a hit by 0.0025 ms/token against a 0.1317 ms/token
prize -- under 2%. The link CANNOT be the explanation for a large win, and this script
prints that budget next to the observed park_gpu - park_host gap so the claim is checked
rather than assumed. If the observed gap exceeds the budget, the two arms differ in
something else and that must be found before anything is claimed.

CAPACITY is the other candidate and gets its own axis: run_why_faster.sh CAP="..." repeats
park_gpu at several pool sizes. If capacity is the mechanism, TTFT tracks pool size. Flat
means it is not.

TPOT is reported against decode batch size (D*_running), because in PD disaggregation
per-token decode latency is a function of how many sequences share the forward pass. A
TPOT change that tracks batch size is a scheduling side effect, not a cache effect, and
should not be claimed as one.

사용:
  python benchmark/decompose_speedup.py --dir results/why/why_c32_p32000
"""
import argparse
import csv
import glob
import json
import os
import statistics as st

# Same fit the park module uses for reuse value (a*n + b*n^2, Llama-3.1-8B on an A6000).
# Re-fit per model and GPU; these are not universal constants.
PREFILL_MS_PER_TOK = 0.131656
PREFILL_MS_PER_TOK2 = 2.40638e-06
# results/nvlink_microbench.json, 50 iters, 512-4096 token transfers.
BW_GPU_GBPS = 52.7
BW_HOST_GBPS = 26.3      # one direction; a park+fetch round trip pays it twice
KIB_PER_TOKEN = 128.0

# recompute is the floor; radix is optional and reported when present (see FLOOR below).
ORDER = ["recompute", "radix", "hicache", "park_host", "park_sync", "park_gpu",
         "park_nvlink"]
# Fall back to radix only when an arm named recompute was not run, so an older result
# directory still analyses instead of printing "(missing arm)" on every row.
FLOOR = ("recompute", "radix")


def reprefill_ms(n):
    return PREFILL_MS_PER_TOK * n + PREFILL_MS_PER_TOK2 * n * n


def load_bench(d, arm):
    p = os.path.join(d, f"bench_{arm}.json")
    if not os.path.exists(p):
        return None
    return json.load(open(p))["summary"]


def load_park(d, arm):
    """Sum the per-prefill park telemetry for one arm. Returns None if the arm has none
    (radix and hicache never load the park manager)."""
    files = (glob.glob(os.path.join(d, f"parked_gpu*.{arm}.json"))
             or glob.glob(os.path.join(d, f"parked_bytes_*.{arm}.json")))  # older layout
    if not files:
        return None
    agg = {"fetch_hits": 0, "fetch_miss": 0, "fetch_already": 0, "fetched_tokens": 0,
           "parked_tokens": 0, "fetch_ms_sum": 0.0,
           "ms": {"local": 0.0, "peer": 0.0, "host": 0.0},
           "tok": {"local": 0, "peer": 0, "host": 0},
           "n": {"local": 0, "peer": 0, "host": 0}, "sync": set(), "files": len(files),
           "phase": {}, "miss_find_ms": 0.0, "src": {},
           "park_phase": {}, "park_bytes": 0, "park_n": 0}
    for f in files:
        try:
            j = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        for k in ("fetch_hits", "fetch_miss", "fetch_already", "fetched_tokens",
                  "parked_tokens", "fetch_ms_sum"):
            agg[k] += j.get(k, 0) or 0
        for tier in ("local", "peer", "host"):
            agg["ms"][tier] += (j.get("fetch_ms_tier") or {}).get(tier, 0.0)
            agg["tok"][tier] += (j.get("fetch_tok_tier") or {}).get(tier, 0)
            agg["n"][tier] += (j.get("fetch_n_tier") or {}).get(tier, 0)
        agg["sync"].add(j.get("sync_fetch", 0))
        for k, v in (j.get("fetch_phase_ms") or {}).items():
            agg["phase"][k] = agg["phase"].get(k, 0.0) + v
        agg["miss_find_ms"] += j.get("miss_find_ms_sum", 0.0) or 0.0
        for k, v in (j.get("park_phase_ms") or {}).items():
            agg["park_phase"][k] = agg["park_phase"].get(k, 0.0) + v
        agg["park_bytes"] += j.get("park_bytes_moved", 0) or 0
        agg["park_n"] += j.get("park_n", 0) or 0
        agg["park_async"] = j.get("park_async", agg.get("park_async"))
        agg["park_pending_peak"] = max(agg.get("park_pending_peak") or 0,
                                       j.get("park_pending_peak", 0) or 0)
        agg["park_lag_ms"] = agg.get("park_lag_ms", 0.0) + (j.get("park_publish_lag_ms") or 0)
        for g, v in (j.get("fetch_src_ms") or {}).items():
            e = agg["src"].setdefault(g, {"ms": 0.0, "tok": 0, "n": 0})
            e["ms"] += v
            e["tok"] += (j.get("fetch_src_tok") or {}).get(g, 0)
            e["n"] += (j.get("fetch_src_n") or {}).get(g, 0)
    return agg


def occ_stats(d, arm):
    """Decode batch size and prefill hit rate, averaged over the run. Both are needed to
    tell a cache effect from a scheduling one."""
    p = os.path.join(d, f"occ_{arm}.csv")
    if not os.path.exists(p):
        return None
    rows = list(csv.DictReader(l for l in open(p) if not l.startswith("#")))
    if not rows:
        return None
    def col(name):
        v = [float(r[name]) for r in rows if r.get(name) not in (None, "")]
        return v
    out = {}
    dr = [col(f"D{i}_running") for i in (0, 1)]
    dr = [v for v in dr if v]
    if dr:
        per = [st.mean(v) for v in dr]
        out["decode_running"] = sum(per)
        out["decode_running_max"] = max(max(v) for v in dr)
    ph = [col(f"P{i}_hit") for i in (0, 1)]
    ph = [v for v in ph if v]
    if ph:
        out["prefill_hit"] = st.mean([st.mean(v) for v in ph])
    return out


def fmt(v, w=8, p=4, dash="-"):
    return f"{dash:>{w}}" if v is None else f"{v:>{w}.{p}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--kib-per-token", type=float, default=KIB_PER_TOKEN)
    args = ap.parse_args()
    d = args.dir

    arms = [a for a in ORDER if load_bench(d, a)]
    caps = sorted(
        (int(os.path.basename(p).split("cap")[1].split(".json")[0]), os.path.basename(p))
        for p in glob.glob(os.path.join(d, "bench_park_gpu_cap*.json")))
    if not arms:
        raise SystemExit(f"no bench_*.json in {d}")

    bench = {a: load_bench(d, a) for a in arms}
    park = {a: load_park(d, a) for a in arms}
    occ = {a: occ_stats(d, a) for a in arms}

    # Arms in one directory are only comparable if they ran against the same build.
    # Re-running a single arm overwrites just that arm, and the stale ones keep their
    # names -- an async-park r1 once sat beside blocking-park r2/r3 with nothing on disk
    # saying so, and averaging them would have compared a change against itself.
    metas = {}
    for a in arms:
        mp = os.path.join(d, f"meta_{a}.json")
        if os.path.exists(mp):
            try:
                metas[a] = json.load(open(mp))
            except Exception:  # noqa: BLE001
                pass
    shas = {m.get("sglang_sha") for m in metas.values() if m.get("sglang_sha")}
    if len(shas) > 1:
        print("\n!!! ARMS RAN AGAINST DIFFERENT BUILDS -- not comparable !!!")
        for a in arms:
            m = metas.get(a) or {}
            print(f"    {a:>14}  sglang={m.get('sglang_sha', 'unstamped')}  "
                  f"ts={m.get('ts', '?')}")
        print("    Re-run the arms you intend to compare into a fresh OUTDIR.")
    elif metas and len(metas) < len(arms):
        print(f"\n  [note] {len(arms) - len(metas)} arm(s) predate build stamping; "
              f"check their mtimes before comparing.")

    # Decode KV capacity, per arm. A park pool on a decode GPU shrinks that decode's own
    # pool, because sglang sizes pools from AVAILABLE memory -- measured 216,374 -> 154,304
    # tokens, a 29% cut, with total card HBM unchanged so nothing else reveals it. If the
    # arms differ here they are not comparable on latency, whatever else is equal.
    dcaps = {}
    for a in arms:
        p_ = os.path.join(d, f"occ_{a}.csv")
        if not os.path.exists(p_):
            continue
        rows = [r for r in csv.DictReader(l for l in open(p_) if not l.startswith("#"))]
        v = [float(r["D0_cap_tok"]) for r in rows if r.get("D0_cap_tok")]
        if v:
            dcaps[a] = st.median(v)
    if len(set(round(v) for v in dcaps.values())) > 1:
        print("\n!!! DECODE KV CAPACITY DIFFERS BETWEEN ARMS -- not comparable !!!")
        for a, v in dcaps.items():
            print(f"    {a:>14}: {v:>10,.0f} tokens")
        print("    A park pool on a decode GPU shrinks that decode's own pool, so the")
        print("    park arm is running with less cache AND a different tier. Pin")
        print("    DECODE_MAX_TOTAL_TOKENS to the smallest value and re-run.")

    print(f"\n=== arms ({d}) ===")
    print(f"{'arm':>10} {'TTFT s':>8} {'TPOT s':>9} {'tok/s':>8} {'P hit':>7} "
          f"{'D batch':>8} {'errs':>5}")
    for a in arms:
        s, o = bench[a], occ[a] or {}
        print(f"{a:>10} {fmt(s.get('avg_ttft_s'))} {fmt(s.get('avg_tpot_s'), 9, 5)} "
              f"{fmt(s.get('overall_throughput_tok_per_s'), 8, 1)} "
              f"{fmt(o.get('prefill_hit'), 7, 3)} {fmt(o.get('decode_running'), 8, 1)} "
              f"{s.get('error_items', '?'):>5}")

    # --- can this workload show a prefill effect at all? -----------------------------
    # FIRST, because everything below is uninterpretable if the answer is no. A prefix
    # cache can only give back prefill time; if a request spends 49 s decoding and 0.25 s
    # on TTFT, then even a perfect cache moves 0.5% of it, and a null result says nothing
    # about the cache. Two runs were spent learning this on the sharegpt workload.
    print(f"\n=== is this workload prefill-bound? ===")
    print(f"{'arm':>10} {'out tok/req':>12} {'decode s/req':>13} {'TTFT s':>8} "
          f"{'TTFT share':>11}")
    shares = []
    for a in arms:
        s = bench[a]
        n_req = s.get("success_items") or s["total_items"]
        out_per = (s.get("total_output_tokens") or 0) / max(1, n_req)
        dec = out_per * (s.get("avg_tpot_s") or 0)
        ttft = s["avg_ttft_s"]
        share = 100 * ttft / (ttft + dec) if (ttft + dec) > 0 else 0
        shares.append(share)
        print(f"{a:>10} {out_per:>12.0f} {dec:>13.2f} {ttft:>8.4f} {share:>10.1f}%")
    if shares:
        m = max(shares)
        if m < 10:
            print(f"  => NO. TTFT is at most {m:.1f}% of a request. This workload is "
                  f"decode-bound and\n     CANNOT show a prefill-cache effect -- a null "
                  f"result here is about the workload,\n     not about the cache. Use "
                  f"WORKLOAD=longctx (long prefix, short output).")
        elif m < 30:
            print(f"  => WEAK. TTFT is {m:.1f}% of a request, so the most a perfect cache "
                  f"can win is\n     bounded by that. Raise PREFIX_WORDS or lower "
                  f"MAX_TOKENS to sharpen it.")
        else:
            print(f"  => YES. TTFT is {m:.1f}% of a request, so prefill work is worth "
                  f"enough to measure.")

    print(f"\n=== fetch tiers (where the bytes came from, and what they cost) ===")
    any_park = False
    for a in arms:
        pk = park[a]
        if not pk:
            continue
        any_park = True
        tot_n = sum(pk["n"].values())
        sync = "SYNC" if pk["sync"] == {1} else ("async" if pk["sync"] == {0} else "MIXED")
        print(f"  {a}  ({pk['files']} prefill(s), fetch timing = {sync})")
        if sync != "SYNC":
            # The default fetch is enqueue-and-continue, so these ms are scheduler time,
            # not transfer time. Saying so here stops the number being read as bandwidth.
            print("     note: async fetch -- ms below is ENQUEUE cost on the scheduler "
                  "thread, not transfer time.")
        for tier in ("local", "peer", "host"):
            n, tk, ms = pk["n"][tier], pk["tok"][tier], pk["ms"][tier]
            if not n:
                continue
            print(f"     {tier:>5}: {n:>5} fetches  {tk:>9} tok  {ms:>9.1f} ms  "
                  f"{ms / max(1, n):>7.2f} ms/fetch  {1000 * ms / max(1, tk):>7.3f} us/tok"
                  f"  ({100 * n / max(1, tot_n):.0f}% of fetches)")
        # Where the scheduler thread went. Published on every run, so this does not
        # need the COSTMODEL trace -- and it is the number that explains the result.
        if pk["phase"]:
            grand = sum(pk["phase"].values())
            nf = max(1, sum(pk["n"].values()))
            print(f"     scheduler-thread cost, {grand / nf:.1f} ms/fetch total:")
            for k, v in sorted(pk["phase"].items(), key=lambda kv: -kv[1]):
                if v <= 0:
                    continue
                print(f"       {k:>13} {v / nf:>7.2f} ms/fetch  "
                      f"{100 * v / grand if grand else 0:>5.1f}%")
            if pk["miss_find_ms"]:
                print(f"       {'(misses)':>13} {pk['miss_find_ms']:>7.0f} ms total spent "
                      f"on index scans that found nothing")
        # Achieved bandwidth PER SOURCE GPU. "peer" hides whether a fetch crossed NVLink
        # or PCIe, and a copy implementation that wastes the link looks identical to a
        # slow link until these are separated.
        if pk["src"]:
            bpt = 128 * 1024
            print(f"     per source GPU (achieved vs 52.7 GB/s peer ceiling):")
            for g, e in sorted(pk["src"].items()):
                gb = e["tok"] * bpt / 2 ** 30
                bw = gb / (e["ms"] / 1000) if e["ms"] > 0 else 0
                who = "local" if g == "-1" else f"gpu{g}"
                print(f"       {who:>7}: {e['n']:>4} fetches  {gb:>7.1f} GB  "
                      f"{e['ms'] / max(1, e['n']):>7.1f} ms/fetch  {bw:>6.1f} GB/s "
                      f"({100 * bw / 52.7:>3.0f}%)")
        # The WRITE side. Reported next to the fetch because the two are wildly
        # asymmetric -- parking moved 5.7x the bytes it ever read back -- and because
        # this path is synchronous on the scheduler thread while the fetch is not.
        if pk["park_phase"]:
            g = sum(pk["park_phase"].values())
            gb = pk["park_bytes"] / 2 ** 30
            print(f"     PARK (write) cost {g / 1000:.1f} s total over {pk['park_n']} parks, "
                  f"{gb:.0f} GB moved -> {gb / (g / 1000) if g else 0:.1f} GB/s:")
            for k, v in sorted(pk["park_phase"].items(), key=lambda kv: -kv[1]):
                if v <= 0:
                    continue
                print(f"       {k:>13} {v / 1000:>7.1f} s  {100 * v / g:>5.1f}%")
            fg = sum(pk["phase"].values()) / 1000 if pk["phase"] else 0
            if fg:
                print(f"       park is {g / 1000 / fg:.1f}x the total FETCH cost "
                      f"({fg:.1f} s)")
            if pk.get("park_async") is not None:
                mode = "ASYNC (event-deferred)" if pk["park_async"] else "BLOCKING"
                print(f"       mode: {mode}")
                if pk["park_async"]:
                    # Lag is the price of not blocking: a prefix is unfindable until its
                    # copy lands, so a next turn arriving inside that window still misses.
                    lag = pk["park_lag_ms"] / max(1, pk["park_n"])
                    print(f"       publish lag {lag:.1f} ms/park avg, "
                          f"max {pk['park_pending_peak']} parks in flight at once")
                    if pk["phase"].get("sync", 0) / max(1, g) > 0.5:
                        print("       WARNING: sync still dominates despite async mode -- "
                              "the event\n                is being waited on somewhere, "
                              "not just recorded.")
        hit = pk["fetch_hits"] / max(1, pk["fetch_hits"] + pk["fetch_miss"])
        print(f"     hit {100 * hit:.1f}%  fetched {pk['fetched_tokens']} tok  "
              f"parked {pk['parked_tokens']} tok  already-had {pk['fetch_already']}")
    if not any_park:
        print("  (none -- no park arm in this directory)")

    # --- the chain: each step moves exactly one thing --------------------------------
    floor = next((a for a in FLOOR if a in bench), None)
    STEPS = [
        (floor, "hicache", "value of SGLang's host tier"),
        (floor, "park_gpu", "value of ours    <- the two bars above are the headline"),
        ("hicache", "park_gpu", "OURS vs INCUMBENT"),
        ("park_host", "park_gpu", "MEDIUM ONLY  <- the 'is it the link?' control"),
        ("park_sync", "park_gpu", "ASYNC PARK  <- blocking copy -> event-deferred"),
        ("park_gpu", "park_nvlink", "PCIe targets removed: NVLink decode GPU only"),
        ("hicache", "park_nvlink", "OURS (NVLink only) vs INCUMBENT"),
        ("hicache", "park_host", "transfer software only (both arms are DRAM/PCIe)"),
    ]
    print(f"\n=== one variable at a time  (floor = {floor or 'NONE'}) ===")
    if floor is None:
        # A two-arm run (hicache vs park_gpu) is a legitimate quick check, so say what it
        # can and cannot answer rather than printing "(missing arm)" four times.
        print("  No recompute/radix arm: the absolute value of caching is not measurable")
        print("  here. The hicache -> park_gpu row below is, and it is the headline one.")
    if floor == "radix":
        print("  NOTE: no recompute arm here, so the floor is radix -- which already has "
              "the GPU\n  prefix cache. These deltas are what the offload tier adds ON TOP "
              "of it, not the\n  value of caching.")
    if "radix" in bench and "recompute" in bench:
        # Keep this visible rather than letting the arm selection bury it.
        print(f"  (radix, GPU cache only, is also present at TTFT "
              f"{bench['radix']['avg_ttft_s']:.4f})")
    for lo, hi, what in STEPS:
        if lo is None or lo not in bench or hi not in bench:
            continue   # arm not in this run; the header above already said which
        t0, t1 = bench[lo]["avg_ttft_s"], bench[hi]["avg_ttft_s"]
        p0, p1 = bench[lo].get("avg_tpot_s"), bench[hi].get("avg_tpot_s")
        dp = f"  TPOT {100 * (p1 - p0) / p0:+6.1f}%" if p0 and p1 else ""
        print(f"  {lo:>10} -> {hi:<10}  TTFT {t0:.4f} -> {t1:.4f}  "
              f"{100 * (t1 - t0) / t0:+6.1f}%{dp}    {what}")

    # --- link budget: the most the medium could possibly be worth ---------------------
    bpt = args.kib_per_token * 1024
    ms_per_tok_gpu = bpt / (BW_GPU_GBPS * 1e9) * 1e3
    ms_per_tok_host = bpt / (BW_HOST_GBPS * 1e9) * 1e3
    print(f"\n=== link budget (what the medium can explain, at most) ===")
    print(f"  {args.kib_per_token:.0f} KiB/token: peer GPU {ms_per_tok_gpu * 1000:.2f} us/tok "
          f"@ {BW_GPU_GBPS} GB/s   host {ms_per_tok_host * 1000:.2f} us/tok @ {BW_HOST_GBPS} GB/s")
    print(f"  re-prefill {PREFILL_MS_PER_TOK * 1000:.1f} us/tok  -> switching medium changes "
          f"the cost of a hit by {100 * (ms_per_tok_host - ms_per_tok_gpu) / PREFILL_MS_PER_TOK:.1f}% "
          f"of what the hit saves")
    pg, ph = park.get("park_gpu"), park.get("park_host")
    if pg and ph and bench.get("park_gpu") and bench.get("park_host"):
        n_req = bench["park_gpu"].get("success_items") or bench["park_gpu"]["total_items"]
        toks = pg["fetched_tokens"]
        budget_s = toks * (ms_per_tok_host - ms_per_tok_gpu) / 1e3 / max(1, n_req)
        obs_s = bench["park_host"]["avg_ttft_s"] - bench["park_gpu"]["avg_ttft_s"]
        print(f"  over {toks} fetched tokens / {n_req} requests:")
        print(f"     bandwidth can explain at most {budget_s * 1e3:+.1f} ms of TTFT per request")
        print(f"     observed park_host - park_gpu   {obs_s * 1e3:+.1f} ms per request")
        # Two independent questions, and conflating them is how "the link is why it is
        # faster" gets asserted from data that does not say so:
        #   (i)  is the observed gap WITHIN what bandwidth could produce?
        #   (ii) is the gap big enough to matter next to the total speedup?
        # A gap can be over budget (so not bandwidth) and still be tiny (so irrelevant).
        ratio = abs(obs_s) / abs(budget_s) if abs(budget_s) > 1e-9 else float("inf")
        if obs_s <= 0:
            print("     => the GPU arm is not faster here at all; the medium bought nothing.")
        elif ratio > 1.5:     # 1.5x leaves room for run-to-run TTFT noise
            print(f"     => {ratio:.1f}x OVER the bandwidth budget, so this gap is NOT the "
                  f"link. The arms differ in\n        something else -- host pinning cost, "
                  f"allocation, eviction pressure, sync. Find it\n        before attributing "
                  f"any of it to NVLink.")
        else:
            print("     => within the bandwidth budget: the medium plausibly accounts for "
                  "this gap.")
        if base := bench.get(floor):
            tot = base["avg_ttft_s"] - bench["park_gpu"]["avg_ttft_s"]
            if abs(tot) > 1e-9:
                print(f"     either way it is {100 * obs_s / tot:.1f}% of the total "
                      f"{tot * 1e3:.0f} ms radix -> park_gpu gain.")

    # --- what the avoided recompute is worth -----------------------------------------
    print(f"\n=== does the hit rate account for the TTFT change? ===")
    base = bench.get(floor)
    for a in ("park_host", "park_gpu"):
        pk, s = park.get(a), bench.get(a)
        if not pk or not s or not base:
            continue
        n_req = s.get("success_items") or s["total_items"]
        avg_fetch_tok = pk["fetched_tokens"] / max(1, n_req)
        saved_ms = reprefill_ms(avg_fetch_tok)
        cost_ms = pk["fetch_ms_sum"] / max(1, n_req)
        pred_s = (saved_ms - cost_ms) / 1e3
        obs_s = s["avg_ttft_s"] - base["avg_ttft_s"]
        print(f"  {a:>10}: {avg_fetch_tok:7.0f} tok/req fetched -> re-prefill avoided "
              f"{saved_ms:7.1f} ms, fetch cost {cost_ms:5.1f} ms")
        print(f"  {'':>10}  predicted TTFT change {-pred_s * 1e3:+7.1f} ms   "
              f"observed {obs_s * 1e3:+7.1f} ms   "
              f"unexplained {(obs_s + pred_s) * 1e3:+7.1f} ms")

    # --- capacity axis ---------------------------------------------------------------
    if caps:
        print(f"\n=== capacity axis (park_gpu, pool tokens per GPU) ===")
        print(f"{'pool tok':>10} {'TTFT s':>8} {'P hit':>7} {'fetched tok':>12}")
        for cap, fname in caps:
            arm = fname[len("bench_"):-len(".json")]
            s = json.load(open(os.path.join(d, fname)))["summary"]
            o = occ_stats(d, arm) or {}
            pk = load_park(d, arm)
            print(f"{cap:>10} {fmt(s['avg_ttft_s'])} {fmt(o.get('prefill_hit'), 7, 3)} "
                  f"{(pk['fetched_tokens'] if pk else 0):>12}")
        ts = [json.load(open(os.path.join(d, f)))["summary"]["avg_ttft_s"] for _, f in caps]
        spread = 100 * (max(ts) - min(ts)) / min(ts)
        print(f"  TTFT spread across a {caps[-1][0] / caps[0][0]:.0f}x capacity range: "
              f"{spread:.1f}%")
        print("  => capacity IS the mechanism" if spread > 10 else
              "  => flat: capacity is NOT the mechanism over this range")

    # --- TPOT ------------------------------------------------------------------------
    print(f"\n=== TPOT vs decode batch size ===")
    print("  TPOT in PD disaggregation is set by how many sequences share the decode")
    print("  forward pass. If TPOT tracks D batch, the change is scheduling, not caching.")
    pairs = [(bench[a].get("avg_tpot_s"), (occ[a] or {}).get("decode_running"), a)
             for a in arms if bench[a].get("avg_tpot_s") and (occ[a] or {}).get("decode_running")]
    for t, r, a in pairs:
        print(f"  {a:>10}  TPOT {t:.5f} s   decode batch {r:.1f}")
    if len(pairs) >= 3:
        xs = [p[1] for p in pairs]
        ys = [p[0] for p in pairs]
        mx, my = st.mean(xs), st.mean(ys)
        den = sum((x - mx) ** 2 for x in xs) * sum((y - my) ** 2 for y in ys)
        if den > 0:
            r = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / den ** 0.5
            print(f"  correlation TPOT vs decode batch: r = {r:+.2f}  "
                  f"({'scheduling effect' if abs(r) > 0.8 else 'not explained by batch size'})")
    print()


if __name__ == "__main__":
    main()
