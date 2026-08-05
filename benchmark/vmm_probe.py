#!/usr/bin/env python3
"""
CUDA VMM feasibility probe for Unified KV Memory Management.

Answers the four questions that decide whether the "one logical KV space, explicit
physical residency" design is implementable on THIS machine, before any SGLang
integration work is done:

  Q1 granularity  -- what is the minimum/recommended slab size? (cuMemGetAllocationGranularity)
                     The KV slab size must be a multiple of it; 2MiB is typical.
  Q2 remap cost   -- how long does it take to change a slab's physical residency
                     while keeping the SAME virtual address? (cuMemUnmap + cuMemMap +
                     cuMemSetAccess) This is the per-migration control-plane cost; it
                     must be small next to a prefill of the same KV.
  Q3 peer-backed  -- can a slab physically resident in a PEER GPU's HBM be mapped into
                     this GPU's address space and read directly (no copy)? At what
                     bandwidth vs local HBM? This is what makes "park to the idle
                     peer GPU" a residency change rather than a copy.
  Q4 host-backed  -- same for host DRAM (CU_MEM_LOCATION_TYPE_HOST / HOST_NUMA), the
                     overflow tier. Bandwidth here is the cost of overflowing.

Nothing here touches SGLang; it allocates its own slabs and frees them. Read-only
with respect to the serving stack, but it does consume a little VRAM on the GPUs it
probes -- run it when the servers are down, or point --dev/--peer at free GPUs.

NOT RUNNABLE HERE: needs a real GPU + cuda-python. Run on server17 inside the
`sglang` env. API names/enums were verified against cuda-bindings 12.9.

사용:
  conda activate sglang
  python benchmark/vmm_probe.py                        # dev 0, peer 1, 32MiB slabs
  python benchmark/vmm_probe.py --dev 0 --peer 1 --slab-mb 64 --iters 200
  python benchmark/vmm_probe.py --pairs                # P2P matrix for all GPUs
"""
import argparse
import ctypes
import time

try:
    from cuda.bindings import driver as cuda
except ImportError:  # older cuda-python layout
    try:
        from cuda import cuda  # type: ignore
    except ImportError:
        raise SystemExit(
            "cuda-python not importable. It is a SGLang dependency "
            "(cuda-python==12.9) -- run this inside the 'sglang' conda env."
        )

SUCCESS = cuda.CUresult.CUDA_SUCCESS


def _c(res):
    """Unwrap cuda-python's (CUresult, *values) return convention, raising on error."""
    if not isinstance(res, tuple):
        res = (res,)
    err, rest = res[0], res[1:]
    if err != SUCCESS:
        try:
            msg = cuda.cuGetErrorString(err)[1].decode()
        except Exception:  # noqa: BLE001
            msg = str(err)
        raise RuntimeError(f"{err}: {msg}")
    if not rest:
        return None
    return rest[0] if len(rest) == 1 else rest


def _prop(loc_type, loc_id, exportable=True):
    """CUmemAllocationProp for a physical allocation at a given location.

    exportable=True requests POSIX-fd handles, which a multi-process (2P2D) deployment
    needs to pass a slab to another process. HOST/HOST_NUMA allocations do NOT support
    fd export on every driver, and asking for it there returns CUDA_ERROR_INVALID_VALUE
    -- which is why the first version of this probe reported host allocation as
    unsupported when it was really the handle-type request that failed."""
    p = cuda.CUmemAllocationProp()
    p.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = loc_type
    p.location.id = loc_id
    if exportable:
        p.requestedHandleTypes = (
            cuda.CUmemAllocationHandleType.CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR
        )
    return p


def _access(dev_ids):
    descs = []
    for d in dev_ids:
        a = cuda.CUmemAccessDesc()
        a.location.type = cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE
        a.location.id = d
        a.flags = cuda.CUmemAccess_flags.CU_MEM_ACCESS_FLAGS_PROT_READWRITE
        descs.append(a)
    return descs


def round_up(x, g):
    return ((x + g - 1) // g) * g


def bandwidth_dtod(src_ptr, dst_ptr, nbytes, iters):
    """GB/s of cuMemcpyDtoD src->dst, as a proxy for how fast a kernel reads `src`.
    Both pointers are device-addressable; src may be backed by local HBM, peer HBM, or
    host DRAM -- that is exactly the comparison we want."""
    _c(cuda.cuMemcpyDtoD(dst_ptr, src_ptr, nbytes))          # warm
    _c(cuda.cuCtxSynchronize())
    t0 = time.perf_counter()
    for _ in range(iters):
        _c(cuda.cuMemcpyDtoD(dst_ptr, src_ptr, nbytes))
    _c(cuda.cuCtxSynchronize())
    dt = time.perf_counter() - t0
    return nbytes * iters / dt / 1e9


def print_topo():
    """nvidia-smi's own view -- the authoritative answer to 'is there an NVLink bridge'.
    cuDeviceCanAccessPeer says only that a P2P path exists, NOT that it is NVLink: PCIe
    P2P also reports 1, at a fraction of the bandwidth."""
    import subprocess
    for cmd, title in ((["nvidia-smi", "topo", "-m"], "topology matrix"),
                       (["nvidia-smi", "nvlink", "-s"], "NVLink status")):
        try:
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=20)
            out = (r.stdout or r.stderr).strip()
        except Exception as e:  # noqa: BLE001
            out = f"(failed: {e})"
        print(f"\n[topo] {title} ({' '.join(cmd)}):")
        for line in out.splitlines()[:24]:
            print(f"  {line}")


def pair_bandwidth(n_dev, slab_mb, iters):
    """Measure peer-read bandwidth for EVERY ordered pair. This is what separates an
    NVLink-bridged pair (tens of GB/s more) from a PCIe P2P pair -- the placement policy
    must prefer the former, and a single 0->1 measurement cannot tell you which is which."""
    print(f"\n[Q3b] peer-read bandwidth per ordered pair ({slab_mb} MiB slab, "
          f"cuMemcpyDtoD peer->local)")
    print("      rows = reading dev (owns the address space), cols = dev holding the data")
    hdr = "      " + "".join(f"{j:>10}" for j in range(n_dev))
    print(hdr)
    for i in range(n_dev):
        dev_i = _c(cuda.cuDeviceGet(i))
        ctx_i = _c(cuda.cuDevicePrimaryCtxRetain(dev_i))
        _c(cuda.cuCtxSetCurrent(ctx_i))
        prop_i = _prop(cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, i)
        gran = _c(cuda.cuMemGetAllocationGranularity(
            prop_i,
            cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
        size = round_up(slab_mb * 2**20, gran)
        row, va, hs, ms = [], None, [], []
        try:
            va = _c(cuda.cuMemAddressReserve(size * 2, gran, 0, 0))
            h_dst = _c(cuda.cuMemCreate(size, prop_i, 0))
            hs.append(h_dst)
            _c(cuda.cuMemMap(int(va), size, 0, h_dst, 0))
            _c(cuda.cuMemSetAccess(int(va), size, _access([i]), 1))
            ms.append((int(va), size))
            for j in range(n_dev):
                if i == j:
                    row.append(f"{'  --':>10}")
                    continue
                src = int(va) + size
                try:
                    prop_j = _prop(cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, j)
                    h = _c(cuda.cuMemCreate(size, prop_j, 0))
                    hs.append(h)
                    _c(cuda.cuMemMap(src, size, 0, h, 0))
                    _c(cuda.cuMemSetAccess(src, size, _access([i, j]), 2))
                    bw = bandwidth_dtod(src, int(va), size, iters)
                    row.append(f"{bw:>9.1f}")
                    _c(cuda.cuMemUnmap(src, size))
                except Exception:  # noqa: BLE001
                    row.append(f"{'  n/a':>10}")
        finally:
            for ptr, sz in ms:
                try:
                    cuda.cuMemUnmap(ptr, sz)
                except Exception:  # noqa: BLE001
                    pass
            for h in hs:
                try:
                    cuda.cuMemRelease(h)
                except Exception:  # noqa: BLE001
                    pass
            if va is not None:
                try:
                    cuda.cuMemAddressFree(va, size * 2)
                except Exception:  # noqa: BLE001
                    pass
            cuda.cuDevicePrimaryCtxRelease(dev_i)
        print(f"  {i:>3} " + "".join(row) + "   GB/s")


def batched_remap(prop, g_min, page, n_pages, dev, iters):
    """Cost of committing n_pages of `page` bytes as ONE logical region:
    n_pages x cuMemMap, then a SINGLE cuMemSetAccess over the whole range, then a
    SINGLE cuMemUnmap of the whole range.

    Why this matters: the per-slab sweep shows map+setAccess+unmap costs ~87 us
    REGARDLESS of slab size, i.e. the cost is per CALL, not per byte. So the way to
    keep 2 MiB granularity (little internal waste) without paying 44 ms/GiB is to
    batch: cuMemSetAccess and cuMemUnmap both take a RANGE that may span many mapped
    handles, so they are paid once per region instead of once per page.
    Returns (us_total_p50, us_map_p50, us_access_p50, us_unmap_p50)."""
    size = page * n_pages
    descs = _access([dev])
    tot, tm, ta, tu = [], [], [], []
    for _ in range(iters):
        va = _c(cuda.cuMemAddressReserve(size, g_min, 0, 0))
        hs = [_c(cuda.cuMemCreate(page, prop, 0)) for _ in range(n_pages)]
        try:
            t0 = time.perf_counter()
            for i, h in enumerate(hs):
                _c(cuda.cuMemMap(int(va) + i * page, page, 0, h, 0))
            t1 = time.perf_counter()
            _c(cuda.cuMemSetAccess(int(va), size, descs, len(descs)))
            t2 = time.perf_counter()
            _c(cuda.cuMemUnmap(int(va), size))
            t3 = time.perf_counter()
            tm.append((t1 - t0) * 1e6); ta.append((t2 - t1) * 1e6)
            tu.append((t3 - t2) * 1e6); tot.append((t3 - t0) * 1e6)
        finally:
            for h in hs:
                try:
                    cuda.cuMemRelease(h)
                except Exception:  # noqa: BLE001
                    pass
            try:
                cuda.cuMemAddressFree(va, size)
            except Exception:  # noqa: BLE001
                pass
    mid = lambda xs: sorted(xs)[len(xs) // 2]     # noqa: E731
    return mid(tot), mid(tm), mid(ta), mid(tu)


def kv_math(results, lengths, layers, kv_heads, head_dim, dtype_bytes, recompute_ms):
    """Turn the measured bandwidths into the number the paper actually argues about:
    how long it takes to RESTORE an L-token prefix from each residency, versus
    recomputing it. Recompute costs are the measured ones from
    results/intro/fig_ttft_ctx_sweep.json."""
    per_tok = layers * kv_heads * head_dim * 2 * dtype_bytes   # 2 = K and V
    print(f"\n[KV math] {per_tok/1024:.0f} KiB of KV per token "
          f"({layers} layers x {kv_heads} kv-heads x {head_dim} dim x K/V x "
          f"{dtype_bytes}B)")
    cols = [k for k in ("local", "peer", "host") if k in results]
    head = f"  {'ctx':>7}  {'KV size':>9}" + "".join(f"{k+' (ms)':>13}" for k in cols)
    print(head + f"{'recompute':>12}  {'speedup vs recompute':>22}")
    for L in lengths:
        nbytes = L * per_tok
        cells, best = [], None
        for k in cols:
            ms = nbytes / (results[k] * 1e9) * 1e3
            cells.append(f"{ms:>13.1f}")
            if k == "peer":
                best = ms
        rc = recompute_ms.get(L)
        rc_s = f"{rc:>12.0f}" if rc else f"{'-':>12}"
        sp = f"{rc/best:>21.0f}x" if (rc and best) else f"{'-':>22}"
        print(f"  {L:>7}  {nbytes/2**30:>8.2f}G" + "".join(cells) + rc_s + sp)
    print("  -> 'peer' is the transfer a park/fetch pays instead of a full prefill.")
    print("     A peer-resident restore only has to beat RECOMPUTE, not local HBM.")


def probe_pairs(n_dev):
    print("\n[Q3a] P2P reachability matrix (cuDeviceCanAccessPeer)")
    print("      rows = accessing dev, cols = peer dev.  1 = can map/read peer memory")
    devs = [_c(cuda.cuDeviceGet(i)) for i in range(n_dev)]
    hdr = "      " + " ".join(f"{i:>3}" for i in range(n_dev))
    print(hdr)
    for i in range(n_dev):
        row = []
        for j in range(n_dev):
            if i == j:
                row.append("  -")
                continue
            ok = _c(cuda.cuDeviceCanAccessPeer(devs[i], devs[j]))
            row.append(f"{int(ok):>3}")
        print(f"  {i:>3} " + " ".join(row))
    print("      NOTE: reachability is NOT bandwidth. On server17 every pair reports 1,")
    print("      but --all-pairs-bw measures 27-53 GB/s inside an NV4-bridged pair and")
    print("      only 3.3 GB/s across pairs -- slower than host DRAM (~24-26 GB/s).")
    print("      Placement must rank locations by MEASURED bandwidth, not by 'GPU first'.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=int, default=0, help="GPU whose address space we build")
    ap.add_argument("--peer", type=int, default=1, help="GPU to place a peer-backed slab on")
    ap.add_argument("--slab-mb", type=int, default=32,
                    help="slab size for the bandwidth tests (rounded up to granularity)")
    ap.add_argument("--iters", type=int, default=100, help="copies per bandwidth point")
    ap.add_argument("--remap-iters", type=int, default=200, help="remap latency samples")
    ap.add_argument("--numa", type=int, default=-1,
                    help="host slab NUMA node (>=0 uses CU_MEM_LOCATION_TYPE_HOST_NUMA)")
    ap.add_argument("--pairs", action="store_true", help="also print the P2P matrix")
    ap.add_argument("--all-pairs-bw", action="store_true",
                    help="measure peer-read bandwidth for EVERY ordered pair -- the only "
                         "way to tell an NVLink-bridged pair from a PCIe P2P one")
    ap.add_argument("--topo", action="store_true",
                    help="print nvidia-smi topo -m / nvlink -s (authoritative on NVLink)")
    ap.add_argument("--remap-slabs", nargs="+", type=int, default=[2, 8, 32, 128],
                    help="slab sizes (MiB) for the remap-latency sweep")
    ap.add_argument("--batch-pages", nargs="+", type=int, default=[1, 8, 64, 384, 512],
                    help="page counts for the batched-commit test (N x map, 1 x "
                         "setAccess, 1 x unmap). 384 pages of 2 MiB = a 6000-token "
                         "session slab for Llama-3.1-8B")
    # KV geometry for the restore-vs-recompute table (default: Llama-3.1-8B-Instruct)
    ap.add_argument("--layers", type=int, default=32)
    ap.add_argument("--kv-heads", type=int, default=8, help="num_key_value_heads (GQA)")
    ap.add_argument("--head-dim", type=int, default=128)
    ap.add_argument("--dtype-bytes", type=int, default=2, help="2 = fp16/bf16")
    ap.add_argument("--kv-lengths", nargs="+", type=int,
                    default=[1000, 2000, 4000, 8000, 16000, 32000])
    args = ap.parse_args()

    _c(cuda.cuInit(0))
    n_dev = _c(cuda.cuDeviceGetCount())
    ver = _c(cuda.cuDriverGetVersion())
    print(f"[vmm_probe] {n_dev} GPU(s) visible, driver CUDA {ver//1000}.{(ver%1000)//10}; "
          f"space on dev {args.dev}, peer slab on dev {args.peer}")
    if args.topo:
        print_topo()
    if args.pairs:
        probe_pairs(n_dev)
    if args.all_pairs_bw:
        pair_bandwidth(n_dev, args.slab_mb, args.iters)

    dev = _c(cuda.cuDeviceGet(args.dev))
    ctx = _c(cuda.cuDevicePrimaryCtxRetain(dev))
    _c(cuda.cuCtxSetCurrent(ctx))

    # ---- Q1 granularity
    prop_local = _prop(cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, args.dev)
    g_min = _c(cuda.cuMemGetAllocationGranularity(
        prop_local, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_MINIMUM))
    g_rec = _c(cuda.cuMemGetAllocationGranularity(
        prop_local, cuda.CUmemAllocationGranularity_flags.CU_MEM_ALLOC_GRANULARITY_RECOMMENDED))
    print(f"\n[Q1] granularity: minimum={g_min/2**20:.2f} MiB  recommended={g_rec/2**20:.2f} MiB")
    print("     -> the KV slab must be a multiple of the minimum; pick the slab so that")
    print("        slab_tokens * bytes_per_token_per_layer is a whole number of these.")

    slab = round_up(args.slab_mb * 2**20, g_min)
    n_slots = 4
    print(f"     using slab={slab/2**20:.0f} MiB, VA reservation={n_slots} slots")

    # ---- one VA reservation = the "unified KV space" of this rank
    va = _c(cuda.cuMemAddressReserve(slab * n_slots, g_min, 0, 0))
    print(f"\n[VA] reserved {slab*n_slots/2**20:.0f} MiB of contiguous VA at 0x{int(va):x}")
    print("     every slot below keeps ITS OWN virtual address for the whole run; only")
    print("     the physical backing changes. This is the property the design needs:")
    print("     attention kernels keep the same pointers across a migration.")

    handles, mapped = [], []          # (handle, ptr, size) bookkeeping for cleanup
    results = {}

    def make_and_map(slot, prop, access_devs, tag):
        ptr = int(va) + slot * slab
        h = _c(cuda.cuMemCreate(slab, prop, 0))
        _c(cuda.cuMemMap(ptr, slab, 0, h, 0))
        descs = _access(access_devs)
        _c(cuda.cuMemSetAccess(ptr, slab, descs, len(descs)))
        handles.append(h); mapped.append((ptr, slab))
        print(f"     [{tag}] mapped at 0x{ptr:x}")
        return h, ptr

    try:
        # slot 0: local HBM, used as the copy destination and the local baseline
        _, p_local = make_and_map(0, prop_local, [args.dev], "local HBM")
        # slot 1: second local slab -> destination for the bandwidth copies
        _, p_dst = make_and_map(1, prop_local, [args.dev], "local HBM (dst)")

        # correctness: write a pattern through the VA and read it back
        host = (ctypes.c_ubyte * 4096)(*([0xA5] * 4096))
        _c(cuda.cuMemcpyHtoD(p_local, host, 4096))
        back = (ctypes.c_ubyte * 4096)()
        _c(cuda.cuMemcpyDtoH(back, p_local, 4096))
        ok = all(b == 0xA5 for b in back)
        print(f"\n[sanity] write/read through the mapped VA: {'OK' if ok else 'MISMATCH'}")

        results["local"] = bandwidth_dtod(p_local, p_dst, slab, args.iters)
        print(f"\n[Q3] read bandwidth by physical residency (same VA space, dev {args.dev}):")
        print(f"     local HBM        : {results['local']:8.1f} GB/s   (baseline)")

        # ---- Q3 peer-backed slab
        if args.peer != args.dev and args.peer < n_dev:
            try:
                prop_peer = _prop(
                    cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_DEVICE, args.peer)
                # access for BOTH devices: the owner and the reader
                _, p_peer = make_and_map(2, prop_peer, [args.dev, args.peer],
                                         f"peer HBM (dev {args.peer})")
                results["peer"] = bandwidth_dtod(p_peer, p_dst, slab, args.iters)
                ratio = results["peer"] / results["local"]
                print(f"     peer HBM (dev {args.peer}) : {results['peer']:8.1f} GB/s   "
                      f"({ratio*100:.0f}% of local)")
            except Exception as e:  # noqa: BLE001
                print(f"     peer HBM (dev {args.peer}) : FAILED -- {e}")
                print("       -> no P2P path between these GPUs (no NVLink bridge, or")
                print("          PCIe P2P disabled). Zero-copy peer residency is not")
                print("          available for this pair; migration must be a copy.")

        # ---- Q4 host-backed slab. Several (location, exportable) combinations are
        # legal on paper but not all are supported by a given driver, and the failure
        # is an indistinguishable INVALID_VALUE -- so try them in order and report
        # which one works instead of concluding "host is unsupported" from one attempt.
        LT = cuda.CUmemLocationType
        host_attempts = [
            (LT.CU_MEM_LOCATION_TYPE_HOST, 0, False, "host DRAM"),
            (LT.CU_MEM_LOCATION_TYPE_HOST, 0, True, "host DRAM (fd-exportable)"),
            (LT.CU_MEM_LOCATION_TYPE_HOST_NUMA, max(0, args.numa), False,
             f"host DRAM NUMA{max(0, args.numa)}"),
            (LT.CU_MEM_LOCATION_TYPE_HOST_NUMA, max(0, args.numa), True,
             f"host DRAM NUMA{max(0, args.numa)} (fd-exportable)"),
        ]
        if hasattr(LT, "CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT"):
            host_attempts.append(
                (LT.CU_MEM_LOCATION_TYPE_HOST_NUMA_CURRENT, 0, False,
                 "host DRAM NUMA_CURRENT"))
        for lt, lid, exp, tag in host_attempts:
            try:
                _, p_host = make_and_map(3, _prop(lt, lid, exportable=exp),
                                         [args.dev], tag)
            except Exception as e:  # noqa: BLE001
                print(f"     {tag:<32}: unsupported ({e})")
                continue
            results["host"] = bandwidth_dtod(p_host, p_dst, slab, args.iters)
            results["host_kind"] = tag
            print(f"     {tag:<32}: {results['host']:8.1f} GB/s   "
                  f"({results['host']/results['local']*100:.0f}% of local)")
            break
        else:
            print("     host DRAM: no VMM host location worked on this driver.")
            print("       -> the overflow tier must use cuMemHostAlloc + a separate host")
            print("          pool (what HiCache does). The unified-space claim then covers")
            print("          local+peer HBM only; say so explicitly in the paper.")

        # ---- Q2 remap latency vs SLAB SIZE, at a fixed VA.
        # Swept, not measured at one size: the cost that matters is
        #   (bytes to migrate / slab size) x per-slab latency,
        # so if latency is flat in slab size, small slabs are catastrophic (a 1 GiB
        # migration in 2 MiB slabs is 512 remaps) and large slabs are nearly free.
        print(f"\n[Q2] residency change at a FIXED virtual address "
              f"(cuMemMap + cuMemSetAccess + cuMemUnmap):")
        print(f"  {'slab':>8}  {'p50 (us)':>9}  {'p99 (us)':>9}  "
              f"{'remaps/GiB':>11}  {'us/GiB':>9}")
        for mb in args.remap_slabs:
            sz = round_up(mb * 2**20, g_min)
            sva = _c(cuda.cuMemAddressReserve(sz, g_min, 0, 0))
            ha = _c(cuda.cuMemCreate(sz, prop_local, 0))
            hb = _c(cuda.cuMemCreate(sz, prop_local, 0))
            descs = _access([args.dev])
            lat = []
            try:
                for i in range(args.remap_iters):
                    h = ha if i % 2 == 0 else hb
                    t0 = time.perf_counter()
                    _c(cuda.cuMemMap(int(sva), sz, 0, h, 0))
                    _c(cuda.cuMemSetAccess(int(sva), sz, descs, len(descs)))
                    _c(cuda.cuMemUnmap(int(sva), sz))
                    lat.append((time.perf_counter() - t0) * 1e6)
            finally:
                cuda.cuMemRelease(ha); cuda.cuMemRelease(hb)
                cuda.cuMemAddressFree(sva, sz)
            lat.sort()
            p50, p99 = lat[len(lat) // 2], lat[int(0.99 * len(lat))]
            n_per_gib = 2**30 / sz
            print(f"  {sz/2**20:>6.0f}Mi  {p50:>9.1f}  {p99:>9.1f}  "
                  f"{n_per_gib:>11.0f}  {p50*n_per_gib:>9.0f}")
        print("     -> latency is flat in slab size, so the cost is per CALL, not per")
        print("        byte. Either use large slabs, or batch (next table).")

        # ---- Q2b batched commit: many 2 MiB pages, ONE setAccess, ONE unmap.
        # This is the number the implementation should be designed around: it keeps
        # fine granularity (little internal waste) while paying the expensive calls
        # once per region.
        print(f"\n[Q2b] batched commit of a region built from {g_min/2**20:.0f} MiB pages "
              f"(N x cuMemMap, 1 x cuMemSetAccess, 1 x cuMemUnmap):")
        print(f"  {'region':>9}  {'pages':>6}  {'total us':>9}  {'map':>8}  "
              f"{'access':>8}  {'unmap':>8}  {'us/GiB':>9}")
        for n_pages in args.batch_pages:
            try:
                tot, tm_, ta_, tu_ = batched_remap(
                    prop_local, g_min, g_min, n_pages, args.dev,
                    max(3, args.remap_iters // 20))
            except Exception as e:  # noqa: BLE001
                print(f"  {n_pages*g_min/2**20:>8.0f}Mi  {n_pages:>6}  failed: {e}")
                continue
            per_gib = tot * (2**30 / (n_pages * g_min))
            print(f"  {n_pages*g_min/2**20:>8.0f}Mi  {n_pages:>6}  {tot:>9.1f}  "
                  f"{tm_:>8.1f}  {ta_:>8.1f}  {tu_:>8.1f}  {per_gib:>9.0f}")
        print("     -> compare us/GiB against the ~38600 us/GiB transfer time at "
              "27 GB/s.")
        print("     MEASURED ON server17: us/GiB is FLAT (~42000) no matter how many")
        print("     pages are batched under one cuMemSetAccess -- so the driver walks")
        print("     each MAPPING, and batching does not help. The same 128 MiB range")
        print("     costs 90 us when backed by ONE handle (Q2) but 5295 us when backed")
        print("     by 64 x 2 MiB handles. => allocate ONE LARGE HANDLE per park slab.")

        # ---- turn bandwidth into restore-vs-recompute, the paper's actual claim
        if results:
            kv_math(results, args.kv_lengths, args.layers, args.kv_heads,
                    args.head_dim, args.dtype_bytes,
                    {1000: 161, 2000: 293, 4000: 592, 8000: 1203,
                     16000: 2706, 32000: 6681})

        print("\n[summary] read bandwidth by residency")
        for k in ("local", "peer", "host"):
            if k in results:
                extra = f"  [{results['host_kind']}]" if k == "host" and \
                    "host_kind" in results else ""
                print(f"  {k:>5}: {results[k]:8.1f} GB/s{extra}")

    finally:
        for ptr, size in mapped:
            try:
                cuda.cuMemUnmap(ptr, size)
            except Exception:  # noqa: BLE001
                pass
        for h in handles:
            try:
                cuda.cuMemRelease(h)
            except Exception:  # noqa: BLE001
                pass
        try:
            cuda.cuMemAddressFree(va, slab * n_slots)
        except Exception:  # noqa: BLE001
            pass
        cuda.cuDevicePrimaryCtxRelease(dev)


if __name__ == "__main__":
    main()
