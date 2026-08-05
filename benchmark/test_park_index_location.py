#!/usr/bin/env python3
"""
Unit tests for the two Phase-3 pieces that need no GPU:

  1. SharedParkIndex location field -- LOC_GPU vs LOC_HOST round-trip, cross-process
     visibility, host entries only readable in the owning process, backshift deletion
     still correct with the wider entry, and the magic bump rejecting an old-format file.
  2. Bandwidth-ranked selection ordering -- the (slow_link, serving, -headroom) key must
     prefer a BUSY fast-link GPU over an IDLE slow-link one, which is the whole point:
     a PCIe-only peer at 3.3 GB/s is worse than CPU DRAM at 26.3.

사용:  python benchmark/test_park_index_location.py
"""
import importlib.util
import os
import struct
import sys
import tempfile

# Load shared_park_index.py directly by path. Importing it as
# sglang.srt.disaggregation.shared_park_index would execute sglang/__init__.py, which
# pulls in torch/pybase64/etc -- unnecessary here since this module is stdlib-only.
_REL = "python/sglang/srt/disaggregation/shared_park_index.py"
_HERE = os.path.dirname(os.path.abspath(__file__))
_REPO_PARENT = os.path.join(os.path.dirname(_HERE), "..")
# Tree names differ per machine: the working copy is ~/sglang-source on server17 and
# ~/sglang in the dev mirror. SGLANG_SRC_DIR is the same variable the start scripts use.
_TREES = ["sglang-source", "sglang"]
_CANDIDATES = [os.environ.get("SGLANG_SRC", "")]
_SRC_DIR = os.environ.get("SGLANG_SRC_DIR", "")
if _SRC_DIR:
    _CANDIDATES.append(os.path.join(_SRC_DIR, _REL))
for _t in _TREES:
    _CANDIDATES.append(os.path.join(_REPO_PARENT, _t, _REL))          # <repo>/../<tree>
    _CANDIDATES.append(os.path.expanduser(os.path.join("~", _t, _REL)))
_SPI = next((os.path.normpath(p) for p in _CANDIDATES
             if p and os.path.exists(os.path.normpath(p))), None)
if _SPI is None:
    raise SystemExit("shared_park_index.py not found; set SGLANG_SRC=<path to it>. "
                     f"tried: {[os.path.normpath(p) for p in _CANDIDATES if p]}")
_spec = importlib.util.spec_from_file_location("shared_park_index", _SPI)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
LOC_GPU, LOC_HOST, SharedParkIndex = _mod.LOC_GPU, _mod.LOC_HOST, _mod.SharedParkIndex
# Print it: with several candidate trees on a machine, a pass against the WRONG tree is
# worse than a failure. Set SGLANG_SRC_DIR (or SGLANG_SRC) to pin it.
print(f"[test] shared_park_index.py under test: {_SPI}")

FAILS = []


def check(cond, label):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def test_index():
    print("\n=== SharedParkIndex: location field ===")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "idx.bin")
        idx = SharedParkIndex(path, capacity=256)

        # GPU entry round-trip
        idx.insert(0xABCD, dev=2, start=6000, n=1500)              # loc defaults to GPU
        e = idx.lookup(0xABCD)
        check(e is not None, "gpu entry found")
        check(e.loc == LOC_GPU and e.dev == 2 and e.start == 6000 and e.n == 1500,
              f"gpu entry fields round-trip (got {e})")
        check(e.is_gpu, "is_gpu true for a GPU entry")
        check(e.readable_here(), "gpu entry readable in any process")

        # HOST entry round-trip; dev carries the owning pid
        idx.insert(0x1234, dev=os.getpid(), start=0, n=800, loc=LOC_HOST)
        h = idx.lookup(0x1234)
        check(h.loc == LOC_HOST and h.dev == os.getpid() and h.n == 800,
              f"host entry fields round-trip (got {h})")
        check(not h.is_gpu, "is_gpu false for a host entry")
        check(h.readable_here(), "host entry readable in the OWNING process")

        # a host entry owned by someone else must NOT be treated as readable
        idx.insert(0x5678, dev=os.getpid() + 12345, start=0, n=900, loc=LOC_HOST)
        o = idx.lookup(0x5678)
        check(not o.readable_here(),
              "host entry of ANOTHER process is not readable here (no IPC for pinned host)")

        # update in place must be able to change the location (GPU -> host demotion)
        idx.insert(0xABCD, dev=os.getpid(), start=42, n=1500, loc=LOC_HOST)
        e2 = idx.lookup(0xABCD)
        check(e2.loc == LOC_HOST and e2.dev == os.getpid() and e2.start == 42,
              f"demotion GPU->HOST updates the entry in place (got {e2})")

        # removal + the backshift path with the wider entry
        check(idx.remove(0xABCD) is True, "remove returns True for a present key")
        check(idx.lookup(0xABCD) is None, "removed key is gone")
        check(idx.remove(0xABCD) is False, "remove returns False for an absent key")

        # collision chain: same home slot, then remove the first -> the second must
        # still be reachable (probe stops at the first empty slot)
        cap = idx.capacity
        a, b = 7, 7 + cap                      # identical home slot
        idx.insert(a, dev=1, start=1, n=10)
        idx.insert(b, dev=3, start=2, n=20)
        idx.remove(a)
        rb = idx.lookup(b)
        check(rb is not None and rb.dev == 3 and rb.start == 2,
              "backshift keeps a colliding key reachable after the earlier one is removed")

        # many entries, mixed locations, all retrievable
        for i in range(100):
            idx.insert(1000 + i, dev=i % 4, start=i * 16,
                       n=100 + i, loc=LOC_GPU if i % 2 else LOC_HOST)
        bad = []
        for i in range(100):
            e = idx.lookup(1000 + i)
            want_loc = LOC_GPU if i % 2 else LOC_HOST
            if e is None or e.loc != want_loc or e.start != i * 16 or e.n != 100 + i:
                bad.append(i)
        check(not bad, f"100 mixed-location entries round-trip (bad: {bad[:5]})")
        idx.close()

        # a second attach sees the same data (this is the cross-process contract; the
        # file is the only channel, so a fresh object is a faithful stand-in)
        idx2 = SharedParkIndex(path, capacity=256)
        e3 = idx2.lookup(1001)
        check(e3 is not None and e3.loc == LOC_GPU,
              "re-attaching to the same file sees entries written earlier")
        check(idx2.capacity == 256, "capacity read back from the header")
        idx2.close()


def test_magic_bump():
    print("\n=== old-format file is rejected, not misparsed ===")
    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "old.bin")
        # Write a header with the OLD magic ('PKIX') and one old-format 28-byte entry.
        old_magic = 0x504B4958
        old_entry = struct.Struct("<qiiid")           # hash, gpu, start, n, ts (28 B)
        header = struct.Struct("<qq")
        cap, stale_hash = 256, 99
        # Place the stale record at the slot the NEW layout would probe for this hash.
        # Putting it at slot 0 would pass trivially: lookup() probes home = h % cap and
        # would never read slot 0 at all. The leading 8 bytes are the hash in both
        # layouts, so if reinit does not ZERO the region, lookup() matches here and
        # returns (dev, start, n) decoded from a 28-byte record read as 32 -- garbage
        # slot coordinates pointing into someone else's KV.
        buf = bytearray(header.size + cap * 32)
        header.pack_into(buf, 0, old_magic, cap)
        off = header.size + (stale_hash % cap) * 32
        buf[off:off + old_entry.size] = old_entry.pack(stale_hash, 3, 111, 222, 1.0)
        with open(path, "wb") as f:
            f.write(buf)

        idx = SharedParkIndex(path, capacity=cap)
        check(idx.lookup(stale_hash) is None,
              "stale old-format entry at its own home slot is gone after reinit "
              "(reinit must zero, not just truncate)")
        # and the file is usable afterwards
        idx.insert(stale_hash, dev=1, start=5, n=7)
        e = idx.lookup(stale_hash)
        check(e is not None and e.dev == 1 and e.start == 5 and e.n == 7,
              "reinitialized file accepts new entries")
        idx.close()


def test_selection_order():
    """The ordering key from _select_pool, tested in isolation (no GPU needed).
    key = (slow_link, serving_usage, -headroom); min() wins."""
    print("\n=== bandwidth-ranked selection ordering ===")

    # measured on server17: dev1 is NVLink-bridged to dev0, dev2/3 are PCIe-only
    peer_bw = {(1, 0): 27.2, (2, 0): 3.3, (3, 0): 3.3}
    host = 26.3

    def slow(src, dst=0):
        return 0 if peer_bw.get((src, dst), 0.0) >= host else 1

    def key(gpu, serving, headroom):
        return (slow(gpu), round(serving, 2), -headroom)

    # dev2 is completely idle with a huge pool; dev1 is busy with less room.
    # Bandwidth-aware selection must still pick dev1, because dev2's link (3.3 GB/s)
    # is slower than CPU DRAM -- parking there would be worse than the host tier.
    cands = [
        ("dev1 NVLink, busy, small pool", key(1, serving=0.88, headroom=10_000)),
        ("dev2 PCIe, idle, huge pool", key(2, serving=0.05, headroom=200_000)),
        ("dev3 PCIe, idle, huge pool", key(3, serving=0.02, headroom=200_000)),
    ]
    winner = min(cands, key=lambda c: c[1])[0]
    check(winner.startswith("dev1"),
          f"fast link beats an idler slow-link GPU (picked: {winner})")

    # Among equally fast links, the idler GPU wins.
    peer_bw[(3, 0)] = 30.0                       # pretend dev3 is bridged too
    cands = [
        ("dev1 busy", key(1, serving=0.88, headroom=50_000)),
        ("dev3 idle", key(3, serving=0.10, headroom=10_000)),
    ]
    winner = min(cands, key=lambda c: c[1])[0]
    check(winner == "dev3 idle",
          f"among fast links the idler GPU wins (picked: {winner})")

    # Among equally fast AND equally idle, more headroom wins.
    cands = [
        ("dev1 small", key(1, serving=0.10, headroom=10_000)),
        ("dev3 large", key(3, serving=0.10, headroom=90_000)),
    ]
    winner = min(cands, key=lambda c: c[1])[0]
    check(winner == "dev3 large",
          f"ties broken by headroom (picked: {winner})")

    # Sanity on the thresholds actually measured.
    check(slow(1) == 0, "NVLink peer (27.2 GB/s) counts as fast vs host 26.3")
    check(slow(2) == 1, "PCIe-only peer (3.3 GB/s) counts as SLOW vs host 26.3")


if __name__ == "__main__":
    test_index()
    test_magic_bump()
    test_selection_order()
    print(f"\n{'FAILED: ' + '; '.join(FAILS) if FAILS else 'all checks passed'}")
    sys.exit(1 if FAILS else 0)
