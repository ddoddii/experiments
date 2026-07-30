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


def _prop(loc_type, loc_id):
    """CUmemAllocationProp for a physical allocation at a given location."""
    p = cuda.CUmemAllocationProp()
    p.type = cuda.CUmemAllocationType.CU_MEM_ALLOCATION_TYPE_PINNED
    p.location.type = loc_type
    p.location.id = loc_id
    # POSIX fd handles are what a multi-process (2P2D) deployment needs in order to
    # pass a slab to another process; requesting it here also proves it is available.
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
    print("      NOTE on RTX A6000: NVLink is bridged in PAIRS (0-1, 2-3). A pair with")
    print("      no bridge falls back to PCIe P2P (much slower) or is unreachable --")
    print("      zero-copy peer-resident KV is only attractive inside a bridged pair.")


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
    args = ap.parse_args()

    _c(cuda.cuInit(0))
    n_dev = _c(cuda.cuDeviceGetCount())
    print(f"[vmm_probe] {n_dev} GPU(s) visible; building the space on dev {args.dev}, "
          f"peer slab on dev {args.peer}")
    if args.pairs:
        probe_pairs(n_dev)

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

        # ---- Q4 host-backed slab
        try:
            if args.numa >= 0:
                prop_host = _prop(
                    cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST_NUMA, args.numa)
                tag = f"host DRAM (NUMA {args.numa})"
            else:
                prop_host = _prop(cuda.CUmemLocationType.CU_MEM_LOCATION_TYPE_HOST, 0)
                tag = "host DRAM"
            _, p_host = make_and_map(3, prop_host, [args.dev], tag)
            results["host"] = bandwidth_dtod(p_host, p_dst, slab, args.iters)
            print(f"     {tag:<17}: {results['host']:8.1f} GB/s   "
                  f"({results['host']/results['local']*100:.0f}% of local)")
        except Exception as e:  # noqa: BLE001
            print(f"     host DRAM        : FAILED -- {e}")
            print("       -> host-located VMM allocations need CUDA >= 12.2; fall back")
            print("          to cuMemHostAlloc + a separate host pool for the overflow tier.")

        # ---- Q2 remap latency: same VA, different physical backing
        h_a = _c(cuda.cuMemCreate(slab, prop_local, 0))
        h_b = _c(cuda.cuMemCreate(slab, prop_local, 0))
        handles += [h_a, h_b]
        ptr = int(va)                       # reuse slot 0's address
        _c(cuda.cuMemUnmap(ptr, slab))
        descs = _access([args.dev])
        lat = []
        for i in range(args.remap_iters):
            h = h_a if i % 2 == 0 else h_b
            t0 = time.perf_counter()
            _c(cuda.cuMemMap(ptr, slab, 0, h, 0))
            _c(cuda.cuMemSetAccess(ptr, slab, descs, len(descs)))
            _c(cuda.cuMemUnmap(ptr, slab))
            lat.append((time.perf_counter() - t0) * 1e6)
        _c(cuda.cuMemMap(ptr, slab, 0, h_a, 0))
        _c(cuda.cuMemSetAccess(ptr, slab, descs, len(descs)))
        lat.sort()
        p50, p99 = lat[len(lat) // 2], lat[int(0.99 * len(lat))]
        print(f"\n[Q2] residency change at a FIXED virtual address "
              f"(map+setAccess+unmap, {slab/2**20:.0f} MiB slab):")
        print(f"     p50 = {p50:.1f} us   p99 = {p99:.1f} us   "
              f"({p50/1000:.3f} ms per slab)")
        print("     -> compare against the prefill this migration avoids: a 8k-token")
        print("        recompute cost ~1200 ms on this model (results/intro sweep), so a")
        print("        remap is worth it as long as slabs-per-migration x p50 << that.")

        print("\n[summary]")
        for k in ("local", "peer", "host"):
            if k in results:
                print(f"  {k:>5} residency: {results[k]:8.1f} GB/s")
        print("  Use these three numbers to justify the placement policy: peer HBM is")
        print("  only worth preferring over host DRAM if its bandwidth ratio says so.")

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
