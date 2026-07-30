#!/usr/bin/env python3
"""
Unit tests for reuse-value-based eviction (Phase 3, replaces plain LRU).

The claim: what we give up first should be what is CHEAPEST TO REBUILD, discounted by
staleness -- not simply whatever was touched longest ago. Re-prefilling 32k tokens costs
6681 ms against 161 ms for 1k (measured), so under LRU a slightly-staler long session
loses to a fresh short one and the system pays ~40x more to rebuild it.

This also matters for the paper, not just for performance: the CUDA-Unified-Memory
argument lists "UM cannot know how long a re-prefill a page costs" as a semantic gap. If
our own policy were pure LRU, that criticism would apply to us too.

Needs torch (the module imports it) but no GPU.

사용:  python benchmark/test_reuse_value.py
"""
import os
import sys
import time

import torch  # noqa: F401  (required by the module under test)

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

_PKG_ROOT = os.path.abspath(os.path.join(os.path.dirname(_SRC), "..", "..", ".."))
sys.path.insert(0, _PKG_ROOT)
try:
    import sglang.srt.disaggregation.idle_kv_parking as _ikp
except Exception as e:  # noqa: BLE001
    raise SystemExit(f"cannot import idle_kv_parking ({type(e).__name__}: {e}).")
if os.path.abspath(_ikp.__file__) != _SRC:
    raise SystemExit(f"imported the WRONG tree: {_ikp.__file__}")

FAILS = []
# measured re-prefill costs, results/intro/fig_ttft_ctx_sweep.json
MEASURED = {1000: 161, 2000: 293, 4000: 592, 8000: 1203, 16000: 2706, 32000: 6681}


def check(cond, label):
    print(f"  [{'ok ' if cond else 'FAIL'}] {label}")
    if not cond:
        FAILS.append(label)


def test_cost_model():
    print("\n=== re-prefill cost model tracks the measured sweep ===")
    print(f"  {'n':>7} {'measured':>9} {'model':>8} {'err':>7}")
    worst = 0.0
    for n, meas in MEASURED.items():
        got = _ikp._reprefill_ms(n)
        err = (got - meas) / meas
        worst = max(worst, abs(err))
        print(f"  {n:>7} {meas:>9} {got:>8.0f} {err*100:>6.1f}%")
    check(worst < 0.20, f"worst-case error under 20% (got {worst*100:.0f}%)")
    # accuracy where it matters most -- long prefixes dominate the decision
    for n in (8000, 16000, 32000):
        err = abs(_ikp._reprefill_ms(n) - MEASURED[n]) / MEASURED[n]
        check(err < 0.03, f"within 3% at n={n} (got {err*100:.1f}%)")
    check(_ikp._reprefill_ms(0) == 0.0, "zero tokens cost nothing")
    check(_ikp._reprefill_ms(-5) == 0.0, "negative length is clamped, not negative")

    # super-linear: doubling the length must more than double the cost at large n,
    # otherwise the quadratic attention term is missing and long prefixes are undervalued
    r = _ikp._reprefill_ms(32000) / _ikp._reprefill_ms(16000)
    check(r > 2.0, f"cost is super-linear at 16k->32k (ratio {r:.2f} > 2)")
    prev = -1.0
    mono = True
    for n in range(0, 40001, 250):
        v = _ikp._reprefill_ms(n)
        if v < prev:
            mono = False
        prev = v
    check(mono, "cost model is monotone increasing in length")


def test_value_ordering():
    print("\n=== a long stale prefix outranks a short fresh one ===")
    now = time.time()
    # the case LRU gets wrong: 32k parked one halflife ago vs 1k parked just now
    hl = _ikp.REUSE_HALFLIFE_S
    v_long_stale = _ikp._reuse_value(32000, now - hl, now)
    v_short_fresh = _ikp._reuse_value(1000, now, now)
    print(f"       32k aged {hl:.0f}s -> {v_long_stale:.0f}   "
          f"1k fresh -> {v_short_fresh:.0f}")
    check(v_long_stale > v_short_fresh,
          "32k one halflife old is still worth more than a brand-new 1k "
          "(LRU would evict the 32k)")

    # but staleness must eventually win, or dead sessions would pin the pool forever
    v_long_dead = _ikp._reuse_value(32000, now - 20 * hl, now)
    check(v_long_dead < v_short_fresh,
          f"after 20 halflives the 32k block loses to a fresh 1k "
          f"({v_long_dead:.3g} < {v_short_fresh:.0f})")

    # same age -> bigger wins
    check(_ikp._reuse_value(8000, now - 5, now) > _ikp._reuse_value(2000, now - 5, now),
          "at equal age the more expensive prefix is worth more")
    # same size -> fresher wins
    check(_ikp._reuse_value(8000, now - 1, now) > _ikp._reuse_value(8000, now - 60, now),
          "at equal size the fresher block is worth more")
    # decay is exactly halving per halflife
    a = _ikp._reuse_value(8000, now, now)
    b = _ikp._reuse_value(8000, now - hl, now)
    check(abs(b / a - 0.5) < 1e-6, f"one halflife halves the value (ratio {b/a:.4f})")


def test_lru_fallback():
    print("\n=== REUSE_AWARE=0 reproduces LRU ===")
    now = time.time()
    saved = _ikp.REUSE_AWARE
    try:
        _ikp.REUSE_AWARE = False
        # with the flag off the value is just the timestamp, so min() = oldest = LRU,
        # regardless of size
        v_old_big = _ikp._reuse_value(32000, now - 100, now)
        v_new_small = _ikp._reuse_value(1000, now, now)
        check(v_old_big < v_new_small,
              "with REUSE_AWARE=0 the older block is the victim even though it is bigger")
        check(v_old_big == now - 100, "value degenerates to the raw timestamp")
    finally:
        _ikp.REUSE_AWARE = saved
    check(_ikp.REUSE_AWARE is saved, "flag restored")


def test_host_store_picks_least_valuable():
    print("\n=== host store evicts the least valuable block, not the oldest ===")
    _ikp.HOST_BUCKET_TOKENS = 256          # 32 MiB per bucket -> cheap test
    store = _ikp._HostParkStore(32, 8, 128, torch.bfloat16, max_bytes=8 * 2**30)
    now = time.time()
    hl = _ikp.REUSE_HALFLIFE_S

    # a big, somewhat stale block and a small, brand-new one
    big = store.alloc(2000)
    big.n = 2000
    store.put(1, big, 2000)
    big.ts = now - hl                       # aged one halflife
    small = store.alloc(200)
    small.n = 200
    store.put(2, small, 200)
    small.ts = now                          # fresh

    check(big.value(now) > small.value(now),
          f"big/stale ({big.value(now):.0f}) outranks small/fresh ({small.value(now):.0f})")
    store.evict_victim()
    check(1 in store.blocks and 2 not in store.blocks,
          "the SMALL fresh block was evicted (LRU would have taken the big one)")
    check(store.n_evicted == 1, "one eviction recorded")

    # and a truly dead block does get taken
    dead = store.alloc(2000)
    dead.n = 2000
    store.put(3, dead, 2000)
    dead.ts = now - 30 * hl
    store.evict_victim()
    check(3 not in store.blocks and 1 in store.blocks,
          "a long-dead block is evicted ahead of a live one of the same size")

    # reading a block refreshes it, protecting it from the next eviction
    store2 = _ikp._HostParkStore(32, 8, 128, torch.bfloat16, max_bytes=8 * 2**30)
    a = store2.alloc(1000); a.n = 1000; store2.put(10, a, 1000); a.ts = now - 5 * hl
    b = store2.alloc(1000); b.n = 1000; store2.put(11, b, 1000); b.ts = now - 1 * hl
    store2.get(10)                          # touch the stale one -> now the hottest
    store2.evict_victim()
    check(10 in store2.blocks and 11 not in store2.blocks,
          "get() refreshes the timestamp, so the touched block survives")


def test_slab_victim_selection():
    """The GPU-pool path keeps (n, hashes, ts) tuples; _evict_slab must pick by value.
    Tested as the same key function over a plain dict, no GPU pool needed."""
    print("\n=== GPU slab victim selection uses the same key ===")
    now = time.time()
    hl = _ikp.REUSE_HALFLIFE_S
    blocks = {
        0: (6000, ["h0"], now - hl),        # big, one halflife old
        1: (500, ["h1"], now),              # small, fresh
        2: (6000, ["h2"], now - 25 * hl),   # big but long dead
    }
    victim = min(blocks, key=lambda b: _ikp._reuse_value(blocks[b][0], blocks[b][2], now))
    check(victim == 2, f"the long-dead big slab is the victim (picked {victim})")
    del blocks[2]
    victim = min(blocks, key=lambda b: _ikp._reuse_value(blocks[b][0], blocks[b][2], now))
    check(victim == 1, f"then the small fresh slab, not the big stale one (picked {victim})")


if __name__ == "__main__":
    print(f"[test] REUSE_AWARE={_ikp.REUSE_AWARE} halflife={_ikp.REUSE_HALFLIFE_S}s "
          f"a={_ikp.PREFILL_MS_PER_TOK} b={_ikp.PREFILL_MS_PER_TOK2}")
    test_cost_model()
    test_value_ordering()
    test_lru_fallback()
    test_host_store_picks_least_valuable()
    test_slab_victim_selection()
    print(f"\n{'FAILED: ' + '; '.join(FAILS) if FAILS else 'all checks passed'}")
    sys.exit(1 if FAILS else 0)
