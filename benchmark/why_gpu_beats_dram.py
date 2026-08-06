#!/usr/bin/env python3
"""
Why storing a victim KV block on a peer GPU beats storing it in CPU DRAM -- decomposed,
because the obvious reason is not the real one.

THE OBVIOUS ARGUMENT DOES NOT SURVIVE ARITHMETIC. Peer GPU measures 52.7 GB/s against
26.3 GB/s to host (results/nvlink_microbench.json), so the medium is 2x, or 4x once a park
plus a fetch pays the host link twice. But at 128 KiB per token that is 2.49 us/token
against 4.98, while the re-prefill a hit avoids costs 131.7 us/token. The medium is worth
1.9% of what a hit saves. A paper that rests "GPU is faster than DRAM" on bandwidth is
resting it on a 2% term.

The measured penalty is far larger than that: park_host against park_gpu, identical code
and policy with only SGLANG_KV_PARK_FORCE_HOST changed, came out 438 ms/request apart --
121x more than the bandwidth budget allows. So something other than bandwidth dominates,
and THAT is the claim worth proving.

THE MECHANISM THIS SCRIPT TESTS. A GPU park pool is allocated once at startup and reused
for the life of the process, so a park into it is pure DMA. Host memory cannot be a DMA
target until it is PINNED, and pinning is a kernel operation costing ~480-640 ms/GiB on
this machine. That leaves no good option, which is the actual argument:

  pin a large pool up front   committed host DRAM regardless of occupancy -- exactly
                              HiCache's 61 GB problem, and the thing the design rejects.
  pin on demand               committed tracks cached, but every new size class pays the
                              pin, on the scheduler thread, in the park path.

The GPU tier escapes the dilemma because HBM that is already allocated to a serving
process needs no equivalent step. That asymmetry is structural, not a bandwidth number,
and it does not shrink as interconnects get faster.

WHAT COUNTS AS PROOF HERE: alloc time must be a large share of the host park path AND
absent from the GPU one, and the cold-pin count must line up with the size classes and
evictions rather than with the number of parks. If alloc turns out to be small, the
hypothesis is wrong and the real cause is still unfound -- the script says so.

사용:
  python benchmark/why_gpu_beats_dram.py --dir results/why/media_r1
"""
import argparse
import glob
import json
import os

MICRO = {"peer_gpu_gbps": 52.7, "host_gbps": 26.3}
KIB_PER_TOKEN = 128.0
PREFILL_MS_PER_TOK = 0.131656


def load(d, arm):
    out = {"park": {}, "host": {}, "park_bytes": 0, "host_bytes": 0,
           "park_n": 0, "host_n": 0, "cold_pins": 0, "cold_ms": 0.0, "files": 0}
    for f in glob.glob(os.path.join(d, f"parked_gpu*.{arm}.json")):
        try:
            j = json.load(open(f))
        except Exception:  # noqa: BLE001
            continue
        out["files"] += 1
        for k, v in (j.get("park_phase_ms") or {}).items():
            out["park"][k] = out["park"].get(k, 0.0) + v
        for k, v in (j.get("host_phase_ms") or {}).items():
            out["host"][k] = out["host"].get(k, 0.0) + v
        out["park_bytes"] += j.get("park_bytes_moved", 0) or 0
        out["host_bytes"] += j.get("host_bytes_moved", 0) or 0
        out["park_n"] += j.get("park_n", 0) or 0
        out["host_n"] += j.get("host_park_n", 0) or 0
        out["cold_pins"] += j.get("host_cold_pins", 0) or 0
        out["cold_ms"] += j.get("host_cold_pin_ms", 0.0) or 0.0
    return out if out["files"] else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--gpu-arm", default="park_gpu")
    ap.add_argument("--host-arm", default="park_host")
    args = ap.parse_args()

    g = load(args.dir, args.gpu_arm)
    h = load(args.dir, args.host_arm)
    if not g or not h:
        raise SystemExit(f"need both {args.gpu_arm} and {args.host_arm} telemetry in "
                         f"{args.dir}")

    # Failed requests are excluded from the latency averages, so an arm that TIMES OUT
    # gets its slowest requests deleted from its own mean. Comparing that against an arm
    # with no failures flatters the failing one, and a latency gap quoted across such a
    # pair is a lower bound at best. Say so before printing any of it.
    errs = {}
    for lab, arm in (("gpu", args.gpu_arm), ("host", args.host_arm)):
        p = os.path.join(args.dir, f"bench_{arm}.json")
        if os.path.exists(p):
            try:
                errs[lab] = json.load(open(p))["summary"]
            except Exception:  # noqa: BLE001
                pass
    if len(errs) == 2:
        eg, eh = errs["gpu"].get("error_items", 0), errs["host"].get("error_items", 0)
        og = errs["gpu"].get("total_output_tokens", 0)
        oh = errs["host"].get("total_output_tokens", 0)
        if eg != eh or (og and oh and abs(og - oh) / max(og, oh) > 0.2):
            print(f"\n!!! THE ARMS DID NOT DO THE SAME WORK !!!")
            print(f"    {args.gpu_arm:>16}: {eg:>4} failed, {og:>7} output tokens")
            print(f"    {args.host_arm:>16}: {eh:>4} failed, {oh:>7} output tokens")
            print("    Failures are dropped from the latency averages, so the arm that")
            print("    failed had its slowest requests removed from its own mean. Any")
            print("    latency gap below is a LOWER BOUND, and if the failure counts are")
            print("    large the right claim is that the arm could not sustain the load,")
            print("    not that it was slower.")

    bpt = KIB_PER_TOKEN * 1024
    print("\n=== 1. the bandwidth argument, and why it is not enough ===")
    us_g = bpt / (MICRO["peer_gpu_gbps"] * 1e9) * 1e6
    us_h = bpt / (MICRO["host_gbps"] * 1e9) * 1e6
    print(f"  peer GPU {MICRO['peer_gpu_gbps']} GB/s = {us_g:.2f} us/token")
    print(f"  host     {MICRO['host_gbps']} GB/s = {us_h:.2f} us/token")
    print(f"  re-prefill avoided by a hit  = {PREFILL_MS_PER_TOK * 1000:.1f} us/token")
    print(f"  => the medium is worth {100 * (us_h - us_g) / (PREFILL_MS_PER_TOK * 1000):.1f}% "
          f"of what a hit saves. Bandwidth alone cannot carry the claim.")

    print("\n=== 2. where each tier's park time actually goes ===")
    for lab, d, phases, nbytes, n in (
            (f"{args.gpu_arm} (peer GPU)", g, g["park"], g["park_bytes"], g["park_n"]),
            (f"{args.host_arm} (CPU DRAM)", h, h["host"], h["host_bytes"], h["host_n"])):
        tot = sum(phases.values())
        gb = nbytes / 2**30
        if not tot or not n:
            print(f"  {lab}: no parks recorded")
            continue
        print(f"  {lab}: {n} parks, {gb:.0f} GB, {tot / 1000:.1f} s "
              f"({tot / n:.1f} ms/park, {gb / (tot / 1000):.1f} GB/s effective)")
        for k, v in sorted(phases.items(), key=lambda kv: -kv[1]):
            if v <= 0:
                continue
            print(f"       {k:>13} {v / 1000:>7.1f} s  {100 * v / tot:>5.1f}%  "
                  f"{v / n:>7.2f} ms/park")

    print("\n=== 3. the asymmetry the design rests on ===")
    ha = h["host"].get("alloc", 0.0)
    ht = sum(h["host"].values())
    ga = g["park"].get("alloc", 0.0)     # absent by construction: the GPU pool is preallocated
    print(f"  host tier spends {ha / 1000:.1f} s pinning memory "
          f"({100 * ha / ht if ht else 0:.1f}% of its park path)")
    print(f"  GPU tier spends  {ga / 1000:.1f} s -- its pool was allocated once at startup")
    if h["host_n"]:
        print(f"  cold pins: {h['cold_pins']} of {h['host_n']} parks "
              f"({100 * h['cold_pins'] / h['host_n']:.1f}%) cost "
              f"{h['cold_ms'] / 1000:.1f} s total, "
              f"{h['cold_ms'] / max(1, h['cold_pins']):.0f} ms each")
    # The verdict is computed, not written, because the hypothesis can fail here.
    share = 100 * ha / ht if ht else 0
    if share >= 25:
        print(f"\n  => CONFIRMED. Pinning is {share:.0f}% of the host park path and has no")
        print("     counterpart on the GPU side. This is structural: HBM already owned by a")
        print("     serving process is DMA-ready, host pages are not, and pinning them up")
        print("     front is the committed-DRAM problem the design exists to avoid.")
    elif share >= 5:
        print(f"\n  => PARTIAL. Pinning is {share:.0f}% of the host park path -- real but not")
        print("     dominant. Report it as one term and find what else differs before")
        print("     claiming allocation is the reason.")
    else:
        print(f"\n  => NOT CONFIRMED. Pinning is only {share:.0f}% of the host park path, so")
        print("     the allocation hypothesis does not explain the gap. Do not claim it.")
        print("     The measured host penalty is real but its cause is still unidentified.")

    if ht and sum(g["park"].values()):
        gt = sum(g["park"].values())
        gpn, hpn = max(1, g["park_n"]), max(1, h["host_n"])
        print(f"\n  per park: GPU {gt / gpn:.1f} ms vs host {ht / hpn:.1f} ms "
              f"= {(ht / hpn) / (gt / gpn):.1f}x")
        print(f"  of that, pinning alone is {ha / hpn:.1f} ms/park, "
              f"{ha / hpn / (gt / gpn):.1f}x the ENTIRE GPU park")
    print()


if __name__ == "__main__":
    main()
