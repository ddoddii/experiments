#!/usr/bin/env python3
"""
Check that each node's ACTUAL KV pool matches what the run asked for, before the run.

WHY THIS IS NOT REDUNDANT WITH THE SWEEP'S BUDGET ARITHMETIC
  --max-total-tokens is a CEILING, not a request. SGLang sizes the pool as

      rest_memory = available_gpu_memory - total_gpu_memory * (1 - mem_fraction_static)

  (model_runner_kv_cache_mixin.py), so anything else already holding memory on that card
  comes straight out of the pool. A park pool allocated by a PREFILL process onto a
  DECODE GPU is exactly that: device-wide free memory drops by 5.86 GB before the decode
  server sizes itself, and the pool silently lands far under the ceiling.

  That is not hypothetical. The C=4 park_decode run asked for 176000 decode tokens and
  got 133263 -- below the 170078-token peak live usage measured on the previous run, so
  decode ran into its own ceiling (occupancy peaked at 98.5%) and TTFT was being charged
  to queueing rather than to placement. Nothing failed; the numbers were just no longer
  measuring the thing. The budget check passed it because it modelled decode's
  mem-fraction requirement without the park pool the arm puts on that same card.

  An arithmetic model cannot close this: the reserve depends on the driver's idea of free
  memory, on CUDA context size, and on allocation ORDER between four processes. Asking
  the servers what they actually built is the only reliable check, and it costs a second.

사용:
  python benchmark/verify_pool_sizes.py --expect prefill:112000 --expect decode:176000
  python benchmark/verify_pool_sizes.py --expect decode:176000 --tolerance 0.10
"""
import argparse
import sys
import urllib.request

DEFAULT_NODES = "prefill:30000,prefill:30001,decode:30002,decode:30003"
KEY = "sglang:max_total_num_tokens"


def scrape(port):
    try:
        with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
            text = r.read().decode("utf-8", "replace")
    except Exception as e:  # noqa: BLE001
        return None, str(e)
    for line in text.splitlines():
        if line.startswith("#"):
            continue
        parts = line.rsplit(None, 1)
        if len(parts) == 2 and parts[0].split("{", 1)[0] == KEY:
            try:
                return float(parts[1]), None
            except ValueError:
                pass
    return None, f"{KEY} not exported"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--nodes", default=DEFAULT_NODES, help="role:port pairs")
    ap.add_argument("--expect", action="append", default=[], metavar="ROLE:TOKENS",
                    help="required pool size for every node of ROLE; repeatable")
    ap.add_argument(
        "--tolerance", type=float, default=0.05,
        help="fraction the actual pool may fall SHORT by. A pool larger than asked for is "
             "never an error -- --max-total-tokens is a ceiling, and coming in under it is "
             "the failure mode.")
    args = ap.parse_args()

    want = {}
    for spec in args.expect:
        role, _, tok = spec.partition(":")
        if not tok:
            print(f"[fatal] --expect wants ROLE:TOKENS, got {spec!r}")
            return 2
        want[role.strip()] = int(tok)
    if not want:
        print("[fatal] nothing to check: pass at least one --expect ROLE:TOKENS")
        return 2

    bad, unreachable = False, False
    for spec in args.nodes.split(","):
        role, _, port = spec.partition(":")
        role, port = role.strip(), port.strip()
        if not port or role not in want:
            continue
        actual, err = scrape(int(port))
        target = want[role]
        if actual is None:
            print(f"  {role}:{port}  UNREACHABLE ({err})")
            unreachable = True
            continue
        ratio = actual / target
        flag = "ok" if ratio >= 1 - args.tolerance else "SHORT"
        print(f"  {role}:{port}  asked {target:>7d}  got {actual:>9.0f}  "
              f"({ratio * 100:5.1f}%)  {flag}")
        if flag == "SHORT":
            bad = True

    if unreachable:
        # Kept distinct from a short pool: the fix is completely different, and reporting
        # a down server as "pool too small" would send someone resizing memory knobs.
        print("\nERROR: a node did not answer /metrics. It is still starting, dead, or on"
              "\n       a different port -- check the server logs before resizing anything.")
        return 1
    if bad:
        print("\nERROR: at least one pool is materially smaller than the run asked for.")
        print("       The usual cause is another allocation on the same card -- a park")
        print("       pool placed on a DECODE GPU comes out of that decode's pool, since")
        print("       SGLang sizes from free memory. Raise that role's")
        print("       --mem-fraction-static, shrink the park pool, or lower the ceiling to")
        print("       what actually fits. Do not run the point: it would measure a pool")
        print("       size nobody chose.")
        return 1
    print("\nall pools within tolerance")
    return 0


if __name__ == "__main__":
    sys.exit(main())
