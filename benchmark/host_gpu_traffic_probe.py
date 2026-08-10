#!/usr/bin/env python3
"""
Measure Host(DRAM)->GPU KV-cache traffic as context length grows -- the RULER-style
"Figure 5(a) Host-to-GPU Traffic" panel, for our own arms instead of quantization
baselines.

For each context length L: build N distinct documents of exactly L tokens (a unique
per-doc header token prevents cross-document prefix sharing on the radix tree), send
each once (round 1 -- fills the serving pool and, once full, pushes the rest into
whichever overflow tier the running arm uses), then re-send the same N documents
(round 2 -- forces a read-back of anything that got evicted). Snapshot each arm's
read-back counters before/after and report the delta as GB actually moved from host
DRAM to GPU HBM:

  - park arm:    idle_kv_parking's own telemetry (parked_gpu*.json -> fetch_tok_tier),
                 which splits cumulative FETCHED (read-back) tokens by the tier that
                 served them: "host" (CPU DRAM overflow, actual Host->GPU PCIe traffic),
                 "local"/"peer" (GPU->GPU park-pool restore, no host DRAM involved).
                 host_bytes_moved / park_bytes_moved in that same file are a DIFFERENT,
                 easily-confused pair of counters -- they track GPU->host WRITES (park
                 admission), not reads, and would silently report the wrong direction of
                 traffic if used here.
  - hicache arm: sglang's built-in Prometheus counter sglang:load_back_tokens_total,
                 scraped from each prefill's /metrics and converted to bytes with the
                 same formula idle_kv_parking.py's _bytes_per_token() uses, so the two
                 arms are directly comparable in GB.
  - recompute/radix: no host tier exists, so this is expected to read ~0 GB -- it's run
                 as the negative control, not skipped.

N auto-scales with --pool-tokens (read from PREFILL_MAX_TOTAL_TOKENS by default) so the
working set is a fixed multiple of the serving pool at every L, and is inflated further
because the router round-robins across 2 prefill replicas -- each one only ever sees
about half the documents.

사용:
  MODEL_PATH=/home/uhmturks/hf_models/Llama-3.1-8B-Instruct \
  PREFILL_MAX_TOTAL_TOKENS=60000 PREFILL_PORTS=30000,30001 \
  PARK_DIR=/dev/shm/sglang_kv_parking \
  python benchmark/host_gpu_traffic_probe.py --arm park \
      --context-lens 4096,8192,16384,32768 \
      --out results/host_gpu_traffic/park.json
"""
import argparse
import glob
import json
import os
import time
import urllib.request

import requests

ROUTER_URL = os.environ.get("ROUTER_URL", "http://127.0.0.1:8000/v1/chat/completions")
MODEL = os.environ.get("MODEL", "meta-llama/Llama-3.1-8B-Instruct")
MODEL_PATH = os.environ.get("MODEL_PATH", "")
PARK_DIR = os.environ.get("PARK_DIR", "/dev/shm/sglang_kv_parking")
PREFILL_PORTS = [int(p) for p in os.environ.get("PREFILL_PORTS", "30000,30001").split(",") if p]
TIMEOUT = int(os.environ.get("TIMEOUT", "600"))

_LOAD_BACK_KEY = "sglang:load_back_tokens_total"


def bytes_per_token():
    """Same formula idle_kv_parking.py's _bytes_per_token() uses, computed from the
    model's own config so every arm agrees on the GB conversion regardless of whether
    idle_kv_parking is even loaded (hicache/recompute arms never import that module)."""
    from transformers import AutoConfig

    cfg = AutoConfig.from_pretrained(MODEL_PATH)
    n_layers = cfg.num_hidden_layers
    n_kv_heads = getattr(cfg, "num_key_value_heads", None) or cfg.num_attention_heads
    head_dim = getattr(cfg, "head_dim", None) or (cfg.hidden_size // cfg.num_attention_heads)
    dtype_bytes = 2  # bf16/fp16 -- matches how start_2P_2D.sh launches these servers
    return 2 * n_layers * n_kv_heads * head_dim * dtype_bytes


def snapshot_park():
    """Cumulative FETCHED tokens by serving tier, summed across every prefill's
    telemetry file. host = read back from CPU DRAM (Host->GPU traffic); local/peer =
    read back from a GPU park pool (GPU->GPU, no host DRAM involved)."""
    host_tok = gpu_tok = 0
    for path in glob.glob(os.path.join(PARK_DIR, "parked_gpu*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        tier = d.get("fetch_tok_tier") or {}
        host_tok += int(tier.get("host") or 0)
        gpu_tok += int(tier.get("local") or 0) + int(tier.get("peer") or 0)
    return host_tok, gpu_tok


def snapshot_hicache_tokens():
    """Sum sglang:load_back_tokens_total across prefill replicas. Exact key match, not
    startswith -- prometheus_client also emits a `..._created` sibling line (the
    counter's creation timestamp) for the same metric name, and startswith("...total")
    would silently add that timestamp into the token count."""
    total = 0.0
    for port in PREFILL_PORTS:
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
                text = r.read().decode("utf-8", "replace")
        except Exception as e:
            print(f"  [warn] /metrics unreachable on {port}: {e}")
            continue
        for line in text.splitlines():
            if line.startswith("#"):
                continue
            parts = line.rsplit(None, 1)
            if len(parts) != 2:
                continue
            key = parts[0].split("{", 1)[0]
            if key == _LOAD_BACK_KEY:
                try:
                    total += float(parts[1])
                except ValueError:
                    pass
    return total


def build_doc(tokenizer, doc_id, n_tokens):
    header = f"[[DOC {doc_id} START]] "
    filler = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 400
    text = header + filler
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    while len(ids) < n_tokens:
        text += filler
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    ids = ids[:n_tokens]
    return tokenizer.decode(ids)


def send(prompt):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": prompt + "\n\nReply with one word."}],
        "max_tokens": 4,
        "stream": False,
    }
    t0 = time.perf_counter()
    resp = requests.post(ROUTER_URL, json=payload, timeout=TIMEOUT)
    resp.raise_for_status()
    return time.perf_counter() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--arm", required=True,
        choices=["recompute", "radix", "hicache", "hicache_memfrac", "park"],
    )
    ap.add_argument("--context-lens", default="4096,8192,16384,32768")
    ap.add_argument(
        "--pool-tokens", type=int,
        default=int(os.environ.get("PREFILL_MAX_TOTAL_TOKENS", "60000")),
    )
    ap.add_argument(
        "--work-multiple", type=float, default=4.0,
        help="working-set size as a multiple of pool-tokens, at every L -- kept high "
             "because the router splits documents across 2 prefill replicas",
    )
    ap.add_argument("--min-docs", type=int, default=6)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    if not MODEL_PATH:
        raise SystemExit("MODEL_PATH env var required (local tokenizer needed for exact "
                          "token-length prompts)")

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    bpt = bytes_per_token()
    print(f"[probe] arm={args.arm} bytes/token={bpt} ({bpt/1024:.1f} KiB)  "
          f"pool={args.pool_tokens}  ports={PREFILL_PORTS}")

    rows = []
    for L in [int(x) for x in args.context_lens.split(",") if x]:
        n_docs = max(args.min_docs, int((args.pool_tokens * args.work_multiple) // L))
        print(f"\n=== context_len={L}  n_docs={n_docs}  "
              f"working_set={n_docs * L} tokens ===")
        docs = [build_doc(tok, i, L) for i in range(n_docs)]

        h0, p0 = snapshot_park()
        lb0 = snapshot_hicache_tokens()

        print("  round 1 (fill + spill)...")
        for i, d in enumerate(docs):
            dt = send(d)
            print(f"    doc {i}: {dt:.2f}s")

        print("  round 2 (re-hit, forces load-back)...")
        for i, d in enumerate(docs):
            dt = send(d)
            print(f"    doc {i}: {dt:.2f}s")

        h1, p1 = snapshot_park()
        lb1 = snapshot_hicache_tokens()

        host_bytes_park = max(0, h1 - h0) * bpt
        gpu_gpu_bytes_park = max(0, p1 - p0) * bpt
        host_bytes_hicache = max(0.0, lb1 - lb0) * bpt

        row = {
            "context_len": L,
            "n_docs": n_docs,
            "working_set_tokens": n_docs * L,
            "host_gb": round(
                (host_bytes_park if args.arm == "park" else host_bytes_hicache) / 1e9, 4
            ),
            "park_gpu_gpu_gb": round(gpu_gpu_bytes_park / 1e9, 4),
            "bytes_per_token": bpt,
        }
        print(f"  -> {row}")
        rows.append(row)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"arm": args.arm, "model": MODEL, "rows": rows}, fh, indent=2)
    print(f"\n[saved] {args.out}")


if __name__ == "__main__":
    main()
