#!/usr/bin/env python3
"""
Agent-stack host-capacity pressure test -- quantifies the Motivation claim:
"using host DRAM as KV offload storage shrinks the host capacity the
 agent-serving stack (orchestration / scheduling / tool execution) can use."

WHY this is needed: a reviewer will not accept "HiCache uses 61GB of host DRAM"
as harm by itself. Harm has to be shown in the units the agent stack cares about:
how many concurrent tool executions still fit, and how fast they run.

WHAT IT DOES
------------
Emulates the *agent side* of an agent-serving stack as a co-located sidecar while
SGLang serves LLM traffic on the same node, then ramps the sidecar up and finds
where it breaks:

  one "agent worker" == one concurrent tool execution / orchestration slot.
  Each iteration ("tool call") allocates --tool-mem-mb of ANONYMOUS memory,
  touches all of it, and runs one bandwidth-bound pass over it (sum + partial
  sort) -- i.e. the profile of a code-interpreter / dataframe / JSON-parsing tool.
  So a worker costs ~tool-mem-mb of non-reclaimable host DRAM plus host memory
  bandwidth, exactly the resources HiCache's L2 pool is holding.

  mode=ramp  : add one worker every --hold s, up to --max-workers. Records, per
               concurrency K: MemAvailable, tool-call p50/p99 latency, tool
               throughput, failures. Stops when MemAvailable drops below
               --floor-gb or a worker fails to allocate -> that K is the
               ADMITTED AGENT CONCURRENCY (K_max), the headline capacity number.
  mode=fixed : hold K=--workers for --duration s. Latency comparison at a
               concurrency every config can sustain -> shows the slowdown that
               exists even BELOW the capacity cliff (reclaim + bandwidth
               contention), not just the cliff itself.
  mode=rag   : no anonymous pressure. mmap a large read-only file (use a model
               weight shard -- do NOT write a new corpus, the disk is tight) and
               do random 4KB reads. Reports p50/p99 read latency and MAJOR fault
               count. This is the answer to "the L3 page cache is reclaimable, so
               it is free": reclaim is not free, it evicts the agent stack's own
               file working set, turning minor faults into disk reads.

SAFETY (server17 is a SHARED box)
---------------------------------
This test deliberately consumes host RAM, so it is written to stop BEFORE the
kernel OOM killer runs -- never to trigger it:
  * every worker sets oom_score_adj=1000, so if the kernel ever does have to
    kill something it kills THIS sidecar, not the SGLang server or another
    tenant's job;
  * the controller aborts the ramp as soon as MemAvailable < --floor-gb
    (default 8GB) and reports the last good K;
  * workers free their buffer between iterations, so an abort releases memory
    immediately.
Do not lower --floor-gb below ~6GB on a shared node.

USAGE (per config; run the *same* sidecar against each server config)
---------------------------------------------------------------------
  # 1. start the server config under test, e.g.
  #    HICACHE_WRITE_POLICY=write_back ./scripts/sglang/start_2P_2D.sh
  #    (or PARK_NO_HICACHE=1 ... for the victim-cache arm)
  # 2. put steady LLM load on it in another shell (so HiCache actually holds and
  #    offloads its pool), e.g. benchmark/qps_sweep.py or the BFCL script
  # 3. ramp the agent sidecar:
  python benchmark/agent_host_pressure.py --mode ramp \
      --label write_back --tool-mem-mb 1024 --max-workers 96 \
      --out results/agent/ramp_write_back.json
  # 4. page-cache damage probe (read-only, writes nothing):
  python benchmark/agent_host_pressure.py --mode rag \
      --label write_back --rag-file /home/uhmturks/hf_models/Llama-3.1-8B-Instruct/model-00001-of-00004.safetensors \
      --out results/agent/rag_write_back.json
"""
import argparse
import json
import mmap
import os
import random
import statistics as st
import time
from multiprocessing import Event, Process, Queue

try:
    import numpy as np
except ImportError:  # noqa: BLE001
    np = None


# ---------------------------------------------------------------- host counters

def meminfo_mb():
    """/proc/meminfo -> MB dict (only what we need)."""
    want = {"MemTotal", "MemFree", "MemAvailable", "AnonPages",
            "Active(file)", "Inactive(file)", "SwapFree", "SwapTotal"}
    out = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, _, rest = line.partition(":")
            if k in want:
                out[k] = float(rest.strip().split()[0]) / 1024.0
    return out


def self_faults():
    """(minor, major) fault counts for this process tree (self + reaped children)."""
    with open("/proc/self/stat") as f:
        p = f.read().rsplit(")", 1)[1].split()
    # after "(comm)": state(0) ppid(1) pgrp(2) session(3) tty(4) tpgid(5) flags(6)
    # minflt(7) cminflt(8) majflt(9) cmajflt(10)
    return int(p[7]) + int(p[8]), int(p[9]) + int(p[10])


# ------------------------------------------------------------ the agent worker

def _deprioritize_for_oom():
    """Make the kernel kill US first, never the SGLang server or another tenant."""
    try:
        with open(f"/proc/{os.getpid()}/oom_score_adj", "w") as f:
            f.write("1000")
    except Exception:  # noqa: BLE001
        pass


def worker(wid, tool_mem_mb, q, stop, seed=0):
    """One concurrent tool-execution slot. Each iteration = one 'tool call':
    allocate tool_mem_mb of anonymous memory, touch all of it, one bandwidth-bound
    pass, free. Reports (wid, latency_s) per iteration, or ('FAIL', msg)."""
    _deprioritize_for_oom()
    rng = random.Random(seed ^ wid)
    n = int(tool_mem_mb * 1024 * 1024 / 8)          # float64 elements
    _sink = []
    warm = True     # first iteration faults in a fresh heap -> a cold outlier that
                    # would otherwise dominate p99 at EVERY K. Not reported.
    while not stop.is_set():
        t0 = time.time()
        try:
            if np is not None:
                a = np.empty(n, dtype=np.float64)
                a.fill(1.0 + rng.random())           # touch every page
                s = float(a.sum())                   # bandwidth-bound pass
                k = min(1 << 20, n)
                a[:k].sort()                         # a little CPU on top
                del a
            else:                                    # numpy-free fallback
                b = bytearray(n * 8)
                for off in range(0, len(b), 4096):
                    b[off] = 1
                s = sum(b[:1 << 20])
                del b
            _sink.append(s); del _sink[:-1]           # keep `s` live (no dead-code elim)
            if warm:
                warm = False
            else:
                q.put((wid, time.time() - t0))
        except MemoryError as e:
            q.put(("FAIL", f"w{wid}: MemoryError {e}"))
            return
        except Exception as e:                        # noqa: BLE001
            q.put(("FAIL", f"w{wid}: {type(e).__name__} {e}"))
            return


# ------------------------------------------------------------------ ramp / fixed

def drain(q, lat, fails):
    while True:
        try:
            item = q.get_nowait()
        except Exception:  # noqa: BLE001
            return
        if item[0] == "FAIL":
            fails.append(item[1])
        else:
            lat.append(item[1])


def summarize(lat, window_s):
    if not lat:
        return {"p50_ms": None, "p99_ms": None, "n": 0, "tool_calls_per_s": 0.0}
    xs = sorted(lat)
    p = lambda f: 1000 * xs[min(len(xs) - 1, int(f * len(xs)))]  # noqa: E731
    return {"p50_ms": round(p(0.50), 1), "p99_ms": round(p(0.99), 1),
            "mean_ms": round(1000 * st.mean(xs), 1), "n": len(xs),
            "tool_calls_per_s": round(len(xs) / window_s, 2)}


def run_ramp(args):
    """Add one worker every --hold s until --max-workers, MemAvailable < floor, or
    a worker fails. Returns per-K rows; the last clean K is the admitted agent
    concurrency."""
    q, stop = Queue(), Event()
    procs, rows, fails = [], [], []
    mt = meminfo_mb().get("MemTotal", 0.0)
    print(f"[ramp] label={args.label} tool_mem={args.tool_mem_mb}MB hold={args.hold}s "
          f"floor={args.floor_gb}GB MemTotal={mt/1024:.1f}GB")
    print(f"  {'K':>4}  {'req_GB':>7}  {'avail_GB':>9}  {'p50_ms':>8}  {'p99_ms':>8}  "
          f"{'calls/s':>8}  {'fail':>4}")
    kmax = 0
    try:
        for k in range(1, args.max_workers + 1):
            p = Process(target=worker,
                        args=(k, args.tool_mem_mb, q, stop, args.seed), daemon=True)
            p.start()
            procs.append(p)

            # A step ends after --hold s, but is EXTENDED (up to 3x) until at least
            # one tool call has completed -- otherwise a step shorter than one tool
            # call reports p50/p99 = None and the row is useless.
            lat, avail = [], []
            t0 = time.time()
            hard_cap = args.hold * 3
            while True:
                drain(q, lat, fails)
                avail.append(meminfo_mb().get("MemAvailable", 0.0))
                el = time.time() - t0
                if avail[-1] < args.floor_gb * 1024 or fails:
                    break
                if el >= args.hold and lat:
                    break
                if el >= hard_cap:
                    break
                time.sleep(0.25)
            drain(q, lat, fails)
            window = max(1e-6, time.time() - t0)

            alive = sum(1 for p_ in procs if p_.is_alive())
            a_min = min(avail) if avail else 0.0
            s = summarize(lat, window)
            row = {"k": k, "requested_gb": round(k * args.tool_mem_mb / 1024.0, 2),
                   "mem_avail_gb": round(a_min / 1024.0, 2),
                   "alive": alive, "n_fail": len(fails), **s}
            rows.append(row)
            print(f"  {k:>4}  {row['requested_gb']:>7.1f}  {row['mem_avail_gb']:>9.2f}  "
                  f"{str(s['p50_ms']):>8}  {str(s['p99_ms']):>8}  "
                  f"{s['tool_calls_per_s']:>8}  {len(fails):>4}")

            if fails:
                print(f"  -> STOP: worker allocation failed ({fails[-1]})")
                break
            if a_min < args.floor_gb * 1024:
                print(f"  -> STOP: MemAvailable {a_min/1024:.2f}GB < floor "
                      f"{args.floor_gb}GB (aborting before the OOM killer runs)")
                break
            if alive < k:
                print("  -> STOP: a worker died (likely OOM-killed)")
                break
            kmax = k
    finally:
        stop.set()
        for p_ in procs:
            p_.join(timeout=5)
            if p_.is_alive():
                p_.terminate()

    admitted_gb = round(kmax * args.tool_mem_mb / 1024.0, 2)
    print(f"\n[result] admitted agent concurrency K_max = {kmax} "
          f"({admitted_gb} GB of tool-execution working set)")
    return {"mode": "ramp", "kmax": kmax, "admitted_gb": admitted_gb,
            "mem_total_gb": round(mt / 1024.0, 2), "rows": rows, "fails": fails}


def run_fixed(args):
    """Hold K=--workers for --duration s -- latency at a concurrency all configs
    sustain, isolating contention (bandwidth/reclaim) from the capacity cliff."""
    q, stop = Queue(), Event()
    procs = [Process(target=worker, args=(i + 1, args.tool_mem_mb, q, stop, args.seed),
                     daemon=True) for i in range(args.workers)]
    print(f"[fixed] label={args.label} K={args.workers} tool_mem={args.tool_mem_mb}MB "
          f"duration={args.duration}s")
    for p in procs:
        p.start()
    lat, fails, avail = [], [], []
    time.sleep(min(5.0, args.duration / 4))          # warm: skip first allocations
    drain(q, lat, fails); lat.clear()
    t0 = time.time()
    while time.time() - t0 < args.duration:
        drain(q, lat, fails)
        avail.append(meminfo_mb().get("MemAvailable", 0.0))
        time.sleep(0.25)
    drain(q, lat, fails)
    stop.set()
    for p in procs:
        p.join(timeout=5)
        if p.is_alive():
            p.terminate()
    s = summarize(lat, args.duration)
    out = {"mode": "fixed", "workers": args.workers,
           "mem_avail_gb": round(min(avail) / 1024.0, 2) if avail else None,
           "n_fail": len(fails), **s}
    print(f"[result] p50={s['p50_ms']}ms p99={s['p99_ms']}ms "
          f"calls/s={s['tool_calls_per_s']} avail_min={out['mem_avail_gb']}GB "
          f"fails={len(fails)}")
    return out


# ---------------------------------------------------------------- page-cache probe

def run_rag(args):
    """Random 4KB reads over a large read-only file (the agent stack's RAG corpus /
    index / tool data). Writes NOTHING. If HiCache's L3 file backend has evicted
    this file's pages, reads become MAJOR faults (real disk I/O) and p99 explodes
    -- which is why 'it is only reclaimable page cache' is not the same as free."""
    path = args.rag_file
    if not path or not os.path.exists(path):
        print(f"[error] --rag-file not found: {path}\n"
              "  point it at any large read-only file, e.g. a model weight shard "
              "(do NOT create a new corpus; the disk is nearly full)")
        return None
    size = os.path.getsize(path)
    mi0, (min0, maj0) = meminfo_mb(), self_faults()
    print(f"[rag] label={args.label} file={os.path.basename(path)} "
          f"{size/2**30:.1f}GiB reads={args.rag_reads} "
          f"file_cache={(mi0.get('Active(file)',0)+mi0.get('Inactive(file)',0))/1024:.1f}GB")
    rng = random.Random(args.seed)
    lat = []
    with open(path, "rb") as f:
        mm = mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ)
        for _ in range(args.rag_reads):
            off = rng.randrange(0, max(1, size - 4096)) & ~0xFFF
            t0 = time.perf_counter()
            _ = mm[off:off + 4096][0]                 # forces the page in
            lat.append(time.perf_counter() - t0)
        mm.close()
    min1, maj1 = self_faults()
    xs = sorted(lat)
    p = lambda fr: 1e3 * xs[min(len(xs) - 1, int(fr * len(xs)))]   # noqa: E731
    out = {"mode": "rag", "file": path, "file_gib": round(size / 2**30, 2),
           "reads": len(lat), "p50_ms": round(p(0.50), 4), "p99_ms": round(p(0.99), 4),
           "p999_ms": round(p(0.999), 4), "mean_ms": round(1e3 * st.mean(xs), 4),
           "minor_faults": min1 - min0, "major_faults": maj1 - maj0,
           "file_cache_gb": round((mi0.get("Active(file)", 0)
                                   + mi0.get("Inactive(file)", 0)) / 1024.0, 2)}
    print(f"[result] p50={out['p50_ms']}ms p99={out['p99_ms']}ms "
          f"p99.9={out['p999_ms']}ms  minor={out['minor_faults']} "
          f"MAJOR={out['major_faults']} ({100*out['major_faults']/max(1,len(lat)):.1f}% "
          "of reads hit disk)")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ramp", "fixed", "rag"], default="ramp")
    ap.add_argument("--label", default="config", help="server config under test")
    ap.add_argument("--tool-mem-mb", type=int, default=1024,
                    help="anonymous working set of one tool execution")
    ap.add_argument("--max-workers", type=int, default=96, help="ramp ceiling")
    ap.add_argument("--workers", type=int, default=16, help="fixed-mode concurrency")
    ap.add_argument("--hold", type=float, default=6.0, help="seconds per ramp step")
    ap.add_argument("--duration", type=float, default=60.0, help="fixed-mode seconds")
    ap.add_argument("--floor-gb", type=float, default=8.0,
                    help="abort when MemAvailable falls below this (OOM guard; do "
                         "not go below ~6 on a shared node)")
    ap.add_argument("--rag-file", default="", help="large read-only file to probe")
    ap.add_argument("--rag-reads", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    if np is None:
        print("[warn] numpy not importable -- using the slow bytearray fallback; "
              "run inside the 'sglang' conda env for the real bandwidth profile")

    res = {"ramp": run_ramp, "fixed": run_fixed, "rag": run_rag}[args.mode](args)
    if res is None:
        return
    res["label"] = args.label
    res["tool_mem_mb"] = args.tool_mem_mb
    if args.out:
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        json.dump(res, open(args.out, "w"), indent=2)
        print(f"[saved] {args.out}")


if __name__ == "__main__":
    main()
