#!/usr/bin/env python3
"""
Pinned-host-memory probe -- decides how the CPU DRAM overflow tier should be
allocated, and pins down the bandwidth numbers the paper will quote for the
CPU-resident path.

WHY THIS EXISTS
---------------
The design's priority-3 tier is "CPU DRAM overflow". How that memory is obtained
decides whether the paper's central claim survives:

  (a) on-demand: allocate pinned host memory per park, free it on eviction
      -> committed host DRAM == actually cached KV. The claim holds.
  (c) pre-reserved: one big pinned pool at startup
      -> committed regardless of occupancy. This is exactly HiCache's 61 GB
         problem, re-created. The claim collapses.

(a) is only viable if allocation is cheap relative to the transfer it precedes,
AND if freeing actually returns the memory to the OS. Both are measured here.

THE TRAP THIS PROBE IS LOOKING FOR
----------------------------------
PyTorch has a *caching* host allocator: `torch.empty(..., pin_memory=True)`
followed by `del` typically does NOT return the pages to the OS -- the allocator
keeps them for the next request. That makes (a) look fast while quietly behaving
like (c): host RSS stays high, so "committed == cached" would be false and a
reviewer measuring /proc/meminfo would catch it. So section A measures RSS and
Mlocked AFTER the free, and compares torch's allocator against raw
cudaHostAlloc/cudaFreeHost, which does return memory.

WHAT IT MEASURES
----------------
  A. allocation cost + whether memory comes back
     - cold vs warm (repeat) allocation time, per size
     - VmRSS / VmLck / meminfo Mlocked deltas after alloc and after free
     - torch pinned vs raw cudaHostAlloc
  B. transfer bandwidth, both directions
     - H2D (fetch: CPU DRAM -> GPU) and D2H (park write: GPU -> CPU DRAM)
     - pinned vs pageable host memory
     NOTE: vmm_probe.py measured 23.7-26.1 GB/s over the CUDA VMM *host location*
     path. The real implementation will use pinned host memory, so the paper must
     quote THIS number, not that one.
  C. amortized staging: pin one buffer once and reuse it N times (the (b) option)
  D. decision summary: alloc cost as a fraction of transfer time, per size

Sizes default to what a session actually needs: Llama-3.1-8B is 128 KiB of KV per
token, so a 6000-token session is 750 MiB.

사용:
  conda activate sglang
  python benchmark/pinned_host_probe.py                       # 128Mi 768Mi 1Gi 2Gi
  python benchmark/pinned_host_probe.py --sizes-mb 256 768 2048 --iters 5
  python benchmark/pinned_host_probe.py --no-raw               # skip cuda-python part
"""
import argparse
import gc
import time

import torch


# ----------------------------------------------------------------- host counters

def status_mb():
    """VmRSS / VmLck of this process, in MB. VmLck is the mlock'd (pinned) subset --
    pinned host memory shows up there, which is how we tell a real free from a
    cached one."""
    out = {}
    try:
        with open("/proc/self/status") as f:
            for line in f:
                k, _, rest = line.partition(":")
                if k in ("VmRSS", "VmLck", "VmHWM"):
                    out[k] = float(rest.strip().split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return out


def meminfo_mb():
    want = {"MemAvailable", "MemFree", "AnonPages", "Mlocked", "Unevictable"}
    out = {}
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                k, _, rest = line.partition(":")
                if k in want:
                    out[k] = float(rest.strip().split()[0]) / 1024.0
    except Exception:  # noqa: BLE001
        pass
    return out


def snap():
    s, m = status_mb(), meminfo_mb()
    return {"rss": s.get("VmRSS", 0.0), "lck": s.get("VmLck", 0.0),
            "mlocked": m.get("Mlocked", 0.0), "avail": m.get("MemAvailable", 0.0)}


def try_host_empty_cache():
    """Ask PyTorch to release cached pinned host memory. The API has moved around
    across versions, so try what exists and report which worked."""
    for name, fn in (
        ("torch._C._host_emptyCache", getattr(torch._C, "_host_emptyCache", None)),
        ("torch.cuda.host_empty_cache",
         getattr(getattr(torch, "cuda", None), "host_empty_cache", None)),
        ("torch._C._cuda_hostEmptyCache",
         getattr(torch._C, "_cuda_hostEmptyCache", None)),
    ):
        if fn is None:
            continue
        try:
            fn()
            return name
        except Exception:  # noqa: BLE001
            continue
    return None


# ------------------------------------------------------------------- allocators

def alloc_torch_pinned(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8, pin_memory=True)


def alloc_torch_pageable(nbytes):
    return torch.empty(nbytes, dtype=torch.uint8)


class RawPinned:
    """cudaHostAlloc / cudaFreeHost through cuda-python, wrapped as a tensor.
    Unlike torch's caching host allocator this returns pages to the OS on free,
    so it is the reference for 'did the memory actually come back'."""

    def __init__(self):
        from cuda.bindings import runtime as rt
        self.rt = rt

    def _chk(self, res):
        err, *rest = res if isinstance(res, tuple) else (res,)
        if int(err) != 0:
            raise RuntimeError(f"cuda runtime error {err}")
        return rest[0] if len(rest) == 1 else tuple(rest)

    def alloc(self, nbytes):
        return self._chk(self.rt.cudaHostAlloc(nbytes, 0))   # cudaHostAllocDefault

    def free(self, ptr):
        self._chk(self.rt.cudaFreeHost(ptr))


# -------------------------------------------------------------------- section A

def section_a(sizes, iters, do_raw):
    print("\n" + "=" * 78)
    print("[A] allocation cost, and whether freeing returns the memory")
    print("=" * 78)
    print("  'warm' repeats the same size -- if warm is far cheaper than cold, an")
    print("  allocator is caching. Then check whether rss/lck come back after free.")

    for nbytes in sizes:
        gb = nbytes / 2**30
        print(f"\n  --- {nbytes/2**20:.0f} MiB ({gb:.2f} GiB) ---")

        # torch pinned: cold
        base = snap()
        t0 = time.perf_counter()
        buf = alloc_torch_pinned(nbytes)
        cold = (time.perf_counter() - t0) * 1e3
        after = snap()
        del buf
        gc.collect()
        freed = snap()
        # RSS is the reliable signal: pinned pages are always resident. VmLck/Mlocked
        # may stay 0 because the CUDA driver pins through its own path rather than
        # mlock(), so do NOT judge by lck alone.
        held = freed["rss"] - base["rss"]
        print(f"  torch pinned  cold alloc {cold:8.1f} ms  ({cold/gb:7.1f} ms/GiB)")
        print(f"                rss +{after['rss']-base['rss']:7.0f} MB   "
              f"lck +{after['lck']-base['lck']:7.0f} MB   (during)")
        print(f"                rss +{held:7.0f} MB   "
              f"lck +{freed['lck']-base['lck']:7.0f} MB   (AFTER free)"
              f"{'   <-- NOT returned: allocator is caching' if held > gb * 512 else '   <-- returned'}")

        # torch pinned: warm (repeat alloc/free of the same size)
        warm = []
        for _ in range(iters):
            t0 = time.perf_counter()
            buf = alloc_torch_pinned(nbytes)
            warm.append((time.perf_counter() - t0) * 1e3)
            del buf
            gc.collect()
        w = sorted(warm)[len(warm) // 2]
        print(f"  torch pinned  warm alloc {w:8.1f} ms  ({w/gb:7.1f} ms/GiB)  "
              f"[median of {iters}]")

        # does an explicit host cache flush give it back?
        api = try_host_empty_cache()
        gc.collect()
        flushed = snap()
        print(f"  host_empty_cache: {api or 'no API found in this torch build'}"
              f"   -> rss +{flushed['rss']-base['rss']:.0f} MB "
              f"(was +{held:.0f} MB before the flush)")

        # raw cudaHostAlloc for comparison
        if do_raw:
            try:
                raw = RawPinned()
                b0 = snap()
                t0 = time.perf_counter()
                ptr = raw.alloc(nbytes)
                rcold = (time.perf_counter() - t0) * 1e3
                b1 = snap()
                t0 = time.perf_counter()
                raw.free(ptr)
                rfree = (time.perf_counter() - t0) * 1e3
                b2 = snap()
                r_held = b2["rss"] - b0["rss"]
                print(f"  raw cudaHostAlloc  alloc {rcold:8.1f} ms "
                      f"({rcold/gb:7.1f} ms/GiB)  free {rfree:7.1f} ms")
                print(f"                     rss +{b1['rss']-b0['rss']:7.0f} MB during, "
                      f"+{r_held:7.0f} MB after free"
                      f"{'   <-- returned to OS' if r_held < gb * 128 else '   <-- held'}")
            except ImportError:
                print("  raw cudaHostAlloc: cuda-python not importable (skipped)")
            except Exception as e:  # noqa: BLE001
                print(f"  raw cudaHostAlloc: failed ({e})")


# -------------------------------------------------------------------- section B

def bw(dst, src, iters):
    """GB/s of dst.copy_(src) with a sync outside the timing loop."""
    dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        dst.copy_(src, non_blocking=True)
    torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    return src.numel() * iters / dt / 1e9


def section_b(sizes, iters, dev):
    print("\n" + "=" * 78)
    print("[B] transfer bandwidth on the REAL implementation path (pinned host memory)")
    print("=" * 78)
    print("  H2D = fetch (CPU DRAM -> GPU).  D2H = park write (GPU -> CPU DRAM).")
    print("  Quote these in the paper, not vmm_probe's VMM-host-location number.")
    print(f"\n  {'size':>9}  {'H2D pinned':>11}  {'D2H pinned':>11}  "
          f"{'H2D pageable':>13}  {'D2H pageable':>13}")
    res = {}
    for nbytes in sizes:
        gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)
        row = {}
        try:
            hp = alloc_torch_pinned(nbytes)
            row["h2d_pin"] = bw(gpu, hp, iters)
            row["d2h_pin"] = bw(hp, gpu, iters)
            del hp
            gc.collect()
        except Exception as e:  # noqa: BLE001
            print(f"  pinned failed at {nbytes/2**20:.0f} MiB: {e}")
        try:
            ho = alloc_torch_pageable(nbytes)
            row["h2d_page"] = bw(gpu, ho, max(1, iters // 2))
            row["d2h_page"] = bw(ho, gpu, max(1, iters // 2))
            del ho
            gc.collect()
        except Exception as e:  # noqa: BLE001
            print(f"  pageable failed at {nbytes/2**20:.0f} MiB: {e}")
        del gpu
        torch.cuda.empty_cache()
        g = lambda k: f"{row[k]:11.1f}" if k in row else f"{'n/a':>11}"  # noqa: E731
        print(f"  {nbytes/2**20:>7.0f}Mi  {g('h2d_pin')}  {g('d2h_pin')}  "
              f"{row.get('h2d_page', float('nan')):13.1f}  "
              f"{row.get('d2h_page', float('nan')):13.1f}")
        res[nbytes] = row
    return res


# -------------------------------------------------------------------- section C

def section_c(nbytes, reuse, dev):
    """Option (b): pin ONE staging buffer and reuse it. Amortized cost per transfer
    is what matters if per-park allocation turns out to be expensive."""
    print("\n" + "=" * 78
          + f"\n[C] amortized staging: pin once ({nbytes/2**20:.0f} MiB), reuse "
          f"{reuse}x\n" + "=" * 78)
    t0 = time.perf_counter()
    staging = alloc_torch_pinned(nbytes)
    pin_ms = (time.perf_counter() - t0) * 1e3
    gpu = torch.empty(nbytes, dtype=torch.uint8, device=dev)
    t0 = time.perf_counter()
    for _ in range(reuse):
        gpu.copy_(staging, non_blocking=True)
    torch.cuda.synchronize()
    total_ms = (time.perf_counter() - t0) * 1e3
    print(f"  one-time pin      : {pin_ms:8.1f} ms")
    print(f"  {reuse} transfers      : {total_ms:8.1f} ms "
          f"({total_ms/reuse:.1f} ms each)")
    print(f"  pin amortized     : {pin_ms/reuse:8.3f} ms per transfer "
          f"({100*pin_ms/total_ms:.2f}% overhead)")
    del staging, gpu
    gc.collect()
    torch.cuda.empty_cache()


# -------------------------------------------------------------------- section D

def section_d(sizes, bwres):
    print("\n" + "=" * 78)
    print("[D] decision: on-demand (a) vs pre-reserved (c) host overflow")
    print("=" * 78)
    print("  on-demand is viable when alloc time is a small fraction of the transfer")
    print("  it precedes AND freeing returns the memory (see [A]).")
    print(f"\n  {'size':>9}  {'H2D (ms)':>9}  {'alloc cold (ms)':>16}  "
          f"{'alloc as % of xfer':>19}")
    for nbytes in sizes:
        row = bwres.get(nbytes, {})
        if "h2d_pin" not in row:
            continue
        xfer_ms = nbytes / (row["h2d_pin"] * 1e9) * 1e3
        t0 = time.perf_counter()
        try:
            b = alloc_torch_pinned(nbytes)
        except Exception:  # noqa: BLE001
            continue
        a_ms = (time.perf_counter() - t0) * 1e3
        del b
        gc.collect()
        print(f"  {nbytes/2**20:>7.0f}Mi  {xfer_ms:>9.1f}  {a_ms:>16.1f}  "
              f"{100*a_ms/xfer_ms:>18.0f}%")
    print("\n  Read it like this:")
    print("   - alloc << transfer AND memory returns on free  -> use (a) on-demand;")
    print("     committed host DRAM == cached KV, which is the paper's claim.")
    print("   - alloc >> transfer, or memory does NOT return   -> use (b): one small")
    print("     pinned staging buffer reused, with the KV body in pageable memory.")
    print("     Still avoids a big pre-reservation; costs a second copy.")
    print("   - only fall back to (c) pre-reserved if both fail, and then report the")
    print("     reservation honestly as committed host DRAM.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes-mb", nargs="+", type=int, default=[128, 768, 1024, 2048],
                    help="transfer sizes; 768 MiB = a 6000-token session for "
                         "Llama-3.1-8B (128 KiB of KV per token)")
    ap.add_argument("--iters", type=int, default=5)
    ap.add_argument("--reuse", type=int, default=20, help="section C reuse count")
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--no-raw", action="store_true",
                    help="skip the raw cudaHostAlloc comparison")
    ap.add_argument("--min-avail-gb", type=float, default=12.0,
                    help="refuse to allocate if MemAvailable would drop below this")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        raise SystemExit("no CUDA device visible")
    sizes = sorted(int(m) * 2**20 for m in args.sizes_mb)
    avail = meminfo_mb().get("MemAvailable", 0.0) / 1024.0
    biggest = sizes[-1] / 2**30
    print(f"[pinned_host_probe] torch {torch.__version__}, dev {args.device}, "
          f"MemAvailable {avail:.1f} GiB")
    print(f"  sizes: {', '.join(f'{s/2**20:.0f}Mi' for s in sizes)}  iters={args.iters}")
    if avail - biggest * 3 < args.min_avail_gb:
        raise SystemExit(
            f"MemAvailable {avail:.1f} GiB too low for a {biggest:.1f} GiB test "
            f"(floor {args.min_avail_gb} GiB). Stop the servers or lower --sizes-mb.")

    torch.cuda.set_device(args.device)
    section_a(sizes, args.iters, not args.no_raw)
    bwres = section_b(sizes, args.iters, args.device)
    section_c(sizes[len(sizes) // 2], args.reuse, args.device)
    section_d(sizes, bwres)


if __name__ == "__main__":
    main()
