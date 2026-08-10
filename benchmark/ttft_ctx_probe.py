#!/usr/bin/env python3
"""
TTFT vs context length, 3 arms (recompute / hicache / park) -- same context-length
x-axis convention as host_gpu_traffic_probe.py (controlled, exact token lengths), but
measuring Time To First Token instead of KV traffic.

For each context length L, per rep:
  recompute arm: ONE fresh, never-before-seen doc -> its streaming TTFT IS the
                 datapoint. No cache exists at all (DISABLE_RADIX_CACHE=1 -- see
                 run_host_gpu_traffic_sweep.sh's start_arm()), so every request is a
                 full cold prefill; there's nothing to warm or evict.
  hicache/park:  reuses host_gpu_traffic_probe.py's build_doc() and its 4-phase
                 write+evict+reload sequence (warm hit -> write hit -> flood/evict ->
                 verify hit). The PLOTTED datapoint is phase 4's TTFT: a GENUINE
                 post-eviction reload from host (hicache) or park pool (park), not a
                 same-GPU no-op hit that happened to still be resident -- measuring
                 phase 1 or 2 instead would just be re-measuring "cache still warm",
                 which every arm can do and answers a different, less interesting
                 question than "what does recovering an evicted prefix cost".

TTFT is measured via streaming (time to first content token), the same convention as
every other benchmark script in this repo, including the guard
sglang_BFCL_multi_turn_concurrent.py needed for sglang's occasional empty-choices SSE
chunk under load (see that script's run_turn() for the incident this was fixed after).

사용:
  MODEL_PATH=/home/uhmturks/hf_models/Llama-3.1-8B-Instruct \
  PREFILL_MAX_TOTAL_TOKENS=170000 PREFILL_PORTS=30000,30001 \
  PARK_DIR=/dev/shm/sglang_kv_parking \
  python benchmark/ttft_ctx_probe.py --arm park \
      --context-lens 4096,8192,16384,32768,65536,131072 --reps 3 \
      --out results/ttft_ctx/park.json
"""
import argparse
import json
import os
import time

import requests

from host_gpu_traffic_probe import (
    MODEL, MODEL_PATH, ROUTER_URL, TIMEOUT, build_doc, clear_hicache_disk_and_check_free,
    snapshot_park,
)

FLOOD_MULTIPLE = 2.0   # phase-3 flood size, x pool-tokens -- eviction only, no TTFT recorded
MIN_FLOOD = 4          # matches host_gpu_traffic_probe.py's cache_aware routing-coverage floor


def send_ttft(prompt):
    """Streaming request; returns (ttft_s, error) -- error is None on success, else the
    exception repr. Same guard host_gpu_traffic_probe.py's SSE-adjacent code and
    sglang_BFCL_multi_turn_concurrent.py's run_turn() both needed: sglang can emit an
    SSE chunk with an EMPTY choices list under load, and chunk["choices"][0] on that
    raises IndexError, misreported as a request failure if unguarded."""
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\nReply with one word."}],
        "max_tokens": 4,
        "stream": True,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=TIMEOUT)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            line = line.decode("utf-8")
            if not line.startswith("data: "):
                continue
            data = line[len("data: "):]
            if data == "[DONE]":
                break
            try:
                chunk = json.loads(data)
            except json.JSONDecodeError:
                continue
            if not chunk.get("choices"):
                continue
            delta = chunk["choices"][0].get("delta", {})
            if delta.get("content"):
                return time.perf_counter() - t0, None
        return None, "stream ended with no content chunk"
    except Exception as e:  # noqa: BLE001
        return None, repr(e)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["recompute", "radix", "hicache", "park"])
    ap.add_argument("--context-lens", default="4096,8192,16384,32768,65536,131072")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument(
        "--pool-tokens", type=int,
        default=int(os.environ.get("PREFILL_MAX_TOTAL_TOKENS", "60000")),
    )
    ap.add_argument("--settle-s", type=float, default=3.0)
    ap.add_argument(
        "--min-free-gb", type=float, default=15.0,
        help="hard-fail before starting a context length if less than this much disk "
             "is free (checked fresh each L, after clearing /tmp/hicache) -- hicache's "
             "file-backend write failures are silent to this script otherwise, see "
             "host_gpu_traffic_probe.py's clear_hicache_disk_and_check_free()",
    )
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not MODEL_PATH:
        raise SystemExit("MODEL_PATH env var required (see host_gpu_traffic_probe.py)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    tok.model_max_length = int(1e9)
    print(f"[ttft-probe] arm={args.arm} pool={args.pool_tokens} reps={args.reps}")

    rows = []
    for L in [int(x) for x in args.context_lens.split(",") if x]:
        clear_hicache_disk_and_check_free(args.min_free_gb)
        n_flood = max(MIN_FLOOD, int((args.pool_tokens * FLOOD_MULTIPLE) // L))
        print(f"\n=== context_len={L}  n_flood={n_flood}  reps={args.reps} ===")
        ttfts, errors = [], []
        # A fast phase-4 TTFT is ambiguous on its own: a genuine park restore and a
        # phase-3 flood that simply FAILED to evict (leaving the prefix GPU-resident, so
        # phase 4 is a no-op local radix hit) both produce one. They differ in the park
        # counters -- a restore moves tokens through a tier, a no-op hit moves none --
        # so record them per context length. Motivated by a live 128k point that came
        # back at 0.30s while every other length tracked recompute exactly: 0.30s is
        # simultaneously consistent with a 17.2GB NVLink peer restore (~0.32s at 53GB/s)
        # and with no transfer at all, and nothing in the TTFT alone separates the two.
        h0, g0, hits0, miss0 = snapshot_park()

        for rep in range(args.reps):
            tag = f"ttftL{L}_r{rep}"
            print(f"  rep {rep}: building target doc...", flush=True)
            doc = build_doc(tok, tag, L)

            if args.arm in ("recompute", "radix"):
                dt, err = send_ttft(doc)
                print(f"  rep {rep}: ttft={dt}  err={err}")
                if err:
                    errors.append(err)
                else:
                    ttfts.append(dt)
                continue

            # hicache/park: warm -> write -> flood/evict -> verify (the plotted point).
            # Printed BEFORE the build, not after: this is the longest silent stretch in
            # the run (n_flood can be 80+ documents at small L), and with no output here
            # a slow build is indistinguishable from a hang -- which is exactly how the
            # O(n^2) build_doc regression presented, as "hicache stuck" with a healthy
            # server logging nothing but /health polls.
            print(f"  rep {rep}: building {n_flood} flood docs...", flush=True)
            flood = [build_doc(tok, f"{tag}_flood{i}", L) for i in range(n_flood)]

            print(f"  rep {rep}: phase 1/4 (warm hit)...")
            _, err1 = send_ttft(doc)              # phase 1: warm hit (hit_count=1)
            print(f"  rep {rep}: phase 2/4 (write hit)...")
            _, err2 = send_ttft(doc)               # phase 2: write hit (crosses threshold)
            print(f"  rep {rep}: settling {args.settle_s}s...")
            time.sleep(args.settle_s)
            print(f"  rep {rep}: phase 3/4 (flood x{n_flood})...")
            for i, f in enumerate(flood):           # phase 3: flood/evict
                send_ttft(f)
            print(f"  rep {rep}: phase 4/4 (verify hit -- plotted point)...")
            dt4, err4 = send_ttft(doc)              # phase 4: verify hit (genuine reload)

            print(f"  rep {rep}: phase1_err={err1} phase2_err={err2} "
                  f"phase4_ttft={dt4} phase4_err={err4}")
            if err4:
                errors.append(err4)
            else:
                ttfts.append(dt4)

        h1, g1, hits1, miss1 = snapshot_park()
        ttfts.sort()
        n = len(ttfts)
        row = {
            "context_len": L,
            "reps": args.reps,
            "n_ok": n,
            "n_err": len(errors),
            "ttft_mean_s": round(sum(ttfts) / n, 4) if n else None,
            "ttft_median_s": round(ttfts[n // 2], 4) if n else None,
            "ttft_min_s": round(ttfts[0], 4) if n else None,
            "ttft_max_s": round(ttfts[-1], 4) if n else None,
            # park arm only (hicache/recompute never publish this telemetry, so these
            # stay 0 there). Nonzero host/gpu tokens = a real restore happened and the
            # fast TTFT is the mechanism working; all-zero alongside a fast TTFT = the
            # prefix was never evicted and phase 4 measured nothing.
            "park_fetch_hits": max(0, hits1 - hits0),
            "park_fetch_miss": max(0, miss1 - miss0),
            "park_host_tokens": max(0, h1 - h0),
            "park_gpu_tokens": max(0, g1 - g0),
        }
        print(f"  -> {row}")
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"arm": args.arm, "model": MODEL, "rows": rows}, fh, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
