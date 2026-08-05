#!/usr/bin/env python3
"""
Unit tests for the CPU DRAM overflow tier (_HostParkStore), placement priority 3.

Needs torch but NOT a GPU: everything under test is host-side accounting plus real
pinned allocations. The claim being protected is:

    committed host DRAM tracks cached KV

which rests on three behaviours that are easy to break:
  1. nothing is allocated until a park actually overflows (no reservation);
  2. sizes are QUANTIZED so allocations hit PyTorch's caching host allocator warm --
     cold pins cost ~480-640 ms/GiB, so per-session sizes would make the tier unusable;
  3. eviction eventually releases pages to the OS, via an explicit host-cache flush with
     hysteresis (plain `del` only returns memory to torch's allocator, leaving RSS at the
     high-water mark).

사용:  python benchmark/test_host_overflow.py
"""
import os
import sys
import time

import torch

_REL = "python/sglang/srt/disaggregation/idle_kv_parking.py"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.join(os.path.dirname(_HERE), "..")
_CANDIDATES = [os.environ.get("SGLANG_SRC", "")]
if os.environ.get("SGLANG_SRC_DIR"):
    _CANDIDATES.append(os.path.join(os.environ["SGLANG_SRC_DIR"], _REL))
for _t in ("sglang-source", "sglang"):
    _CANDIDATES.append(os.path.join(_REPO_PARENT, _t, _REL))
    _CANDIDATES.append(os.path.expanduser(os.path.join("~", _t, _REL)))
_SRC = next((os.path.normpath(p) for p in _CANDIDATES
             if p and os.path.exists(os.path.normpath(p))), None)
if _SRC is None:
    raise SystemExit(f"idle_kv_parking.py not found; set SGLANG_SRC_DIR. tried: "
                     f"{[os.path.normpath(p) for p in _CANDIDATES if p]}")
print(f"[test] idle_kv_parking.py under test: {_SRC}")

# idle_kv_parking imports torch, zmq and sglang.srt.* at top level, so unlike
# shared_park_index it cannot be loaded standalone -- put the chosen tree's package root
# (.../python) FIRST on sys.path and import it normally, which also guarantees the
# module and its sglang.srt.* dependencies come from the same tree.
_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(_SRC), "..", "..", ".."))
sys.path.insert(0, _PKG_ROOT)
try:
    import sglang.srt.disaggregation.idle_kv_parking as _ikp
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"cannot import idle_kv_parking ({type(e).__name__}: {e}).\n"
                     "Run inside the 'sglang' conda env.")
_loaded = os.path.abspath(_ikp.__file__)
if _loaded != _SRC:
    raise SystemExit(f"imported the WRONG tree:\n  wanted {_SRC}\n  got    {_loaded}\n"
                     "set SGLANG_SRC_DIR to pin it")

_HostParkStore = _ikp._HostParkStore
_host_empty_cache = _ikp._host_empty_cache

FAILS = []
# Llama-3.1-8B geometry: 32 layers, 8 kv heads, head_dim 128, bf16 -> 128 KiB per token
LAYERS, HEADS, DIM, DTYPE = 32, 8, 128, torch.bfloat16


def check(cond, label):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return float(line.split()[1]) / 1024.0
    return 0.0


def new_store(max_gb=2.0, bucket=512):
    _ikp.HOST_BUCKET_TOKENS = bucket          # small buckets keep the test fast
    return _HostParkStore(LAYERS, HEADS, DIM, DTYPE,
                          max_bytes=int(max_gb * 2**30))


def test_geometry():
    print("\n=== geometry and quantization ===")
    s = new_store(bucket=512)
    check(s.bytes_per_token == 2 * LAYERS * HEADS * DIM * 2,
          f"bytes/token = {s.bytes_per_token} (expect 128 KiB = 131072)")
    check(s.bytes_per_token == 131072, "128 KiB per token for Llama-3.1-8B")

    check(s.bucket_for(1) == 512, "a 1-token park still takes one whole bucket")
    check(s.bucket_for(512) == 512, "exact bucket size is not rounded up")
    check(s.bucket_for(513) == 1024, "513 tokens rounds up to 2 buckets")
    check(s.bucket_for(6000) == 6144, "6000-token session rounds to 6144 (12 buckets)")
    # quantization is the point: only a handful of distinct sizes may ever be requested
    sizes = {s.bucket_for(n) for n in range(1, 4000, 7)}
    check(len(sizes) <= 8,
          f"arbitrary lengths collapse to few size classes ({sorted(sizes)})")


def test_no_reservation():
    print("\n=== nothing is committed until an overflow happens ===")
    before = rss_mb()
    s = new_store(bucket=512)
    after = rss_mb()
    check(s.live_bytes == 0, "a fresh store holds 0 bytes")
    check(after - before < 50,
          f"constructing the store allocates no host KV memory (RSS +{after-before:.0f} MB)")


def test_alloc_and_cap():
    print("\n=== allocation, LRU eviction, and the cap ===")
    # cap = 8 buckets of 512 tokens: 512 * 131072 = 64 MiB per bucket -> 512 MiB
    s = new_store(max_gb=0.5, bucket=512)
    per_block = 512 * s.bytes_per_token
    check(abs(per_block - 64 * 2**20) < 1, f"one bucket = {per_block/2**20:.0f} MiB")

    blocks = []
    for i in range(8):
        b = s.alloc(500)
        if b is None:
            break
        s.put(1000 + i, b, 500)
        blocks.append(b)
    check(len(s.blocks) == 8, f"8 blocks fit under a 512 MiB cap (got {len(s.blocks)})")
    check(s.live_bytes == 8 * per_block, "live_bytes accounts every block")
    check(not s.would_exceed(500) or True, "cap check runs")

    # the 9th allocation must evict the LRU one rather than exceed the cap
    n_before = len(s.blocks)
    b = s.alloc(500)
    check(b is not None, "9th alloc succeeds by evicting")
    check(s.n_evicted >= 1, f"an LRU eviction happened (n_evicted={s.n_evicted})")
    check(s.live_bytes <= s.max_bytes, "live_bytes never exceeds the cap")
    check(1000 not in s.blocks, "the coldest block (hash 1000) was the one evicted")
    # alloc() and put() are deliberately separate: the block is not indexed until the
    # D2H copy has succeeded, so a failed copy cannot leave a phantom entry pointing at
    # memory holding nothing. Hence the count is n_before-1 here and back to n_before
    # only after put().
    check(len(s.blocks) == n_before - 1,
          f"alloc() evicts but does not index (got {len(s.blocks)}, want {n_before-1})")
    s.put(2000, b, 500)
    check(len(s.blocks) == n_before, "after put() the count is back at the cap")

    # LRU order must follow reads
    s2 = new_store(max_gb=0.25, bucket=512)     # 4 blocks
    for i in range(4):
        s2.put(i, s2.alloc(500), 500)
    s2.get(0)                                    # touch the coldest -> now hottest
    s2.alloc(500)                                # forces one eviction
    check(0 in s2.blocks and 1 not in s2.blocks,
          "a read protects a block from eviction (0 kept, 1 evicted)")


def test_alias_eviction():
    print("\n=== a block indexed under two hashes dies exactly once ===")
    s = new_store(max_gb=1.0, bucket=512)
    blk = s.alloc(400)
    s.put(0xAAA, blk, 400)
    blk.hashes.append(0xBBB)                     # prompt-prefix alias, same block
    s.blocks[0xBBB] = blk
    live = s.live_bytes
    check(s.get(0xAAA) is s.get(0xBBB), "both hashes resolve to the same block")
    s.drop(0xAAA)
    check(0xAAA not in s.blocks and 0xBBB not in s.blocks,
          "dropping one alias removes BOTH index entries (no dangling alias)")
    check(s.live_bytes == live - blk.nbytes, "bytes are subtracted exactly once")


def test_warm_alloc_is_cheap():
    print("\n=== quantized sizes hit the caching allocator warm ===")
    s = new_store(max_gb=4.0, bucket=2048)       # 2048 tok = 256 MiB per bucket
    mib = 2048 * s.bytes_per_token / 2**20
    t0 = time.perf_counter()
    b1 = s.alloc(2000)
    cold = (time.perf_counter() - t0) * 1e3
    s.put(1, b1, 2000)
    # Drop the local reference too. _drop() removes the store's reference, but a live
    # local keeps the tensor alive, so the caching allocator cannot hand its memory back
    # -- and the next alloc would be a fresh cold pin. Real callers do not retain one
    # (_park_to_host returns straight after put()).
    b1 = None
    s.drop(1)

    t0 = time.perf_counter()
    b2 = s.alloc(1900)                # different length, SAME bucket -> should be warm
    warm = (time.perf_counter() - t0) * 1e3
    check(b2 is not None, "second alloc of the same bucket succeeds")
    print(f"       cold={cold:.1f} ms  warm={warm:.1f} ms for a {mib:.0f} MiB bucket "
          f"({cold/max(mib/1024, 1e-9):.0f} ms/GiB cold)")
    check(s.bucket_for(2000) == s.bucket_for(1900),
          "two different session lengths share one bucket (why bucketing exists)")
    # The whole tier depends on this: if a repeat of the same size class is not much
    # cheaper than the first pin, per-park allocation is unaffordable and the design
    # must switch to a reused staging buffer.
    check(warm < cold * 0.25,
          f"a repeat of the same size class is much cheaper ({warm:.1f} vs {cold:.1f} ms)")


def test_flush_releases_rss():
    print("\n=== eviction eventually returns pages to the OS ===")
    check(_host_empty_cache() in (True, False), "host cache flush API probe runs")
    s = new_store(max_gb=4.0, bucket=2048)
    base = rss_mb()
    hs = []
    for i in range(8):                            # 8 x 256 MiB = 2 GiB
        b = s.alloc(2000)
        if b is None:
            break
        s.put(i, b, 2000)
        hs.append(i)
    grown = rss_mb()
    check(grown - base > 500,
          f"parking to host really commits DRAM (RSS +{grown-base:.0f} MB)")
    peak_seen = s.peak_bytes

    # drop everything: this must cross the hysteresis threshold and flush
    for i in hs:
        s.drop(i)
    freed = rss_mb()
    check(s.live_bytes == 0, "all blocks released from the store")
    check(s.n_flushes >= 1,
          f"the shrink triggered a host-cache flush (n_flushes={s.n_flushes})")
    check(freed - base < (grown - base) * 0.5,
          f"RSS came back down: +{grown-base:.0f} MB -> +{freed-base:.0f} MB "
          "(this is what makes committed==cached true)")
    print(f"       peak tracked = {peak_seen/2**20:.0f} MiB")


def test_no_flush_under_churn():
    print("\n=== hysteresis: steady churn must NOT flush every eviction ===")
    s = new_store(max_gb=4.0, bucket=2048)
    for i in range(4):
        s.put(i, s.alloc(2000), 2000)
    flushes_before = s.n_flushes
    # park/evict one block at a time: live stays near the peak, so no flush should fire
    for i in range(4, 12):
        b = s.alloc(2000)
        if b is None:
            break
        s.put(i, b, 2000)
        s.drop(i - 4)
    check(s.n_flushes == flushes_before,
          f"no flush while the live set stays flat (flushes={s.n_flushes}, "
          f"was {flushes_before}) -- otherwise every eviction would re-pay ~480 ms/GiB")


if __name__ == "__main__":
    print(f"[test] torch {torch.__version__}")
    test_geometry()
    test_no_reservation()
    test_alloc_and_cap()
    test_alias_eviction()
    test_warm_alloc_is_cheap()
    test_flush_releases_rss()
    test_no_flush_under_churn()
    print(f"\n{'FAILED: ' + '; '.join(FAILS) if FAILS else 'all checks passed'}")
    sys.exit(1 if FAILS else 0)
