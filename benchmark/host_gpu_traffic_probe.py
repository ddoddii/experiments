#!/usr/bin/env python3
"""
Measure Host(DRAM)->GPU KV-cache traffic as context length grows -- the RULER-style
"Figure 5(a) Host-to-GPU Traffic" panel, for our own arms instead of quantization
baselines.

For each context length L: build N distinct documents of exactly L tokens (a unique
per-doc header token prevents cross-document prefix sharing on the radix tree), then
run 4 phases to force a GENUINE host round-trip rather than hoping one falls out of
random eviction timing:

  1. warm hit    -- send each doc once (hit_count=1 on hicache's radix node; no write
                     yet -- see below).
  2. write hit   -- send each doc a 2nd time. hicache's write_through_selective policy
                     only starts writing a node to host once hit_count reaches
                     write_through_threshold=2 (hiradix_cache.py) -- this is the hit
                     that crosses it. Still served from GPU, so this phase alone never
                     produces a load-back read; skipping it means nothing is ever
                     written to host to read back in phase 4. park's on-evict admission
                     has no such hit-count gate, but phase 2 still helps make sure
                     something has been evicted-and-parked by the time phase 3 runs.
  3. flood/evict -- send a SEPARATE disposable batch, sized to exceed the pool, so the
                     phase-1/2 docs get evicted from the GPU pool. Without this, a doc
                     that never happened to get evicted by the ambient traffic simply
                     stays GPU-resident forever and phase 4 reads it locally -- zero
                     host traffic, indistinguishable from "the mechanism doesn't work".
  4. verify hit  -- re-send the original docs a 3rd time. This is the ONLY phase that
                     can produce a genuine load-back: written-to-host (phase 2) AND
                     evicted-from-GPU (phase 3) AND referenced again (phase 4).

An earlier 2-phase version of this script (send once, send again) produced a single
isolated nonzero point per arm with zero everywhere else -- not a believable trend, and
exactly what phase 2 alone predicts: it can enqueue a write but can't produce a read,
so any nonzero reading was eviction-timing noise, not signal.

Snapshot each arm's read-back counters before phase 1 and after phase 4, and report the
delta as GB actually moved from host DRAM to GPU HBM:

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
working set is a modest multiple of the serving pool at every L -- enough to guarantee
some ambient eviction pressure in phases 1-2, but deliberately NOT so large that the
overflow tiers (host DRAM cap, park pool capacity) themselves get overrun and start
silently dropping data instead of parking it (an earlier --work-multiple=4.0 default
put ~230-240k tokens, ~28-30GB of KV, through arms whose overflow capacity was never
verified to hold that much). Phase 3's flood is what actually guarantees eviction now,
so N no longer has to carry that job alone.

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


_SUFFIX = "\n\nReply with one word."
_GEN_TOKENS = 4          # matches send()'s max_tokens
_SAFETY_BUFFER = 16      # slack for tokenizer nondeterminism right at the boundary


def _wrapped_len(tokenizer, text):
    """Token count of the EXACT request send() will issue: chat-template-wrapped, with
    the generation prompt appended -- not just tokenizer(text), which is what the
    server actually enforces its context-length limit against."""
    msgs = [{"role": "user", "content": text + _SUFFIX}]
    return len(tokenizer.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True))


def build_doc(tokenizer, doc_id, n_tokens):
    """Builds a document whose full wrapped request fits in n_tokens total.

    Naively tokenizing the raw filler text to n_tokens ids and decoding back to a
    string does NOT guarantee the server sees n_tokens: decode -> chat-template
    re-encode is not a stable round trip. Confirmed live -- a document built to exactly
    131072 ids produced a 134407-token wrapped request server-side (2.5% drift), a 400
    Bad Request past the model's own 131072-token limit. Measures the ACTUAL wrapped
    length and trims proportionally to the measured excess until it fits under budget,
    rather than assuming a fixed overhead constant."""
    header = f"[[DOC {doc_id} START]] "
    filler = "The quick brown fox jumps over the lazy dog near the riverbank at dawn. " * 400
    budget = n_tokens - _GEN_TOKENS - _SAFETY_BUFFER

    text = header + filler
    while len(tokenizer(text, add_special_tokens=False)["input_ids"]) < budget:
        text += filler
    ids = tokenizer(text, add_special_tokens=False)["input_ids"][:budget]
    text = tokenizer.decode(ids)

    for _ in range(8):
        excess = _wrapped_len(tokenizer, text) - budget
        if excess <= 0:
            break
        ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        cut = max(1, int(len(ids) * (excess / len(ids)) * 1.3))  # overshoot the cut a bit
        text = tokenizer.decode(ids[:-cut])
    return text


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
        help="the SERVER's actual --max-total-tokens -- must exceed the largest "
             "--context-lens value (with margin) or that single request can't even be "
             "admitted. Informational here; doc-count scaling uses --work-pool-tokens.",
    )
    ap.add_argument(
        "--work-pool-tokens", type=int, default=None,
        help="reference pool size FOR DOC-COUNT SCALING ONLY, decoupled from "
             "--pool-tokens. Needed once --pool-tokens has to be raised past ~131072 "
             "tokens to admit a single 128k document: scaling doc count off that same "
             "huge number would blow up n_docs at small L too (e.g. ~100+ docs at 4k) "
             "for no reason. Defaults to min(--pool-tokens, 60000).",
    )
    ap.add_argument(
        "--work-multiple", type=float, default=1.5,
        help="working-set size as a multiple of work-pool-tokens, at every L -- ambient "
             "eviction pressure only; phase 3 forces the real eviction",
    )
    ap.add_argument(
        "--flood-multiple", type=float, default=2.0,
        help="phase-3 flood size as a multiple of work-pool-tokens -- must exceed the "
             "pool on its own (with margin) so it evicts the phase-1/2 docs regardless "
             "of how full the pool already was",
    )
    ap.add_argument(
        "--min-docs", type=int, default=3,
        help="floor on doc count -- keep this LOW. The host/park overflow budget is "
             "roughly fixed in GB (e.g. hicache-ratio x device pool, or "
             "SGLANG_KV_PARK_HOST_MAX_GB), not proportional to L, so a large floor "
             "silently balloons the working set past that budget at big L (was 6: at "
             "32768 tokens x 128 KiB/token, 6 docs alone is ~25GB against an ~8-9GB "
             "host budget) and most of it gets evicted-before-write or dropped instead "
             "of actually round-tripping through host -- the opposite of what a higher "
             "floor was meant to guarantee.",
    )
    ap.add_argument(
        "--min-flood", type=int, default=4,
        help="SEPARATE floor for phase-3 flood docs, independent of --min-docs. The "
             "router defaults to cache_aware routing, which sends each doc's repeated "
             "hits to the SAME prefill replica it first landed on -- but a flood doc is "
             "a novel, never-seen prefix, and cache_aware has no cache signal for it, "
             "so which of the (here) 2 prefill replicas it lands on is effectively a "
             "coin flip per flood doc. --min-docs 1 with --flood-multiple's formula "
             "giving n_flood=1 confirmed this live: the single flood doc had a real "
             "chance of landing on the OTHER replica from the target doc, applying zero "
             "pressure to the pool that actually needed it, and measuring a clean 0 GB "
             "that looked like 'the mechanism doesn't work' rather than 'got unlucky'. "
             "4 flood docs makes both replicas getting hit overwhelmingly likely without "
             "needing to know or hardcode the replica count.",
    )
    ap.add_argument(
        "--settle-s", type=float, default=3.0,
        help="pause after phase 2 so hicache's async write-to-host can land before "
             "phase 3 evicts the node -- too short and phase 4 reads nothing back "
             "because there was nothing on host yet to read",
    )
    ap.add_argument("--out", required=True)
    ap.add_argument(
        "--append", action="store_true",
        help="merge new context-lens into an existing --out file instead of "
             "overwriting it -- for adding e.g. 65536,131072 to an earlier 4k-32k run "
             "without re-measuring the small L points. Rows with a matching "
             "context_len are replaced, not duplicated.",
    )
    args = ap.parse_args()

    if not MODEL_PATH:
        raise SystemExit("MODEL_PATH env var required (local tokenizer needed for exact "
                          "token-length prompts)")

    work_pool_tokens = args.work_pool_tokens or min(args.pool_tokens, 60000)

    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL_PATH)
    # build_doc()'s trim loop deliberately measures an over-length candidate before
    # cutting it down (that's how it knows how much to cut) -- harmless by design, but
    # HF's tokenizer warns on every call past model_max_length, which would otherwise
    # print noise on every trim iteration at large L.
    tok.model_max_length = int(1e9)
    bpt = bytes_per_token()
    print(f"[probe] arm={args.arm} bytes/token={bpt} ({bpt/1024:.1f} KiB)  "
          f"server_pool={args.pool_tokens}  work_pool={work_pool_tokens}  "
          f"ports={PREFILL_PORTS}")

    rows = []
    for L in [int(x) for x in args.context_lens.split(",") if x]:
        n_docs = max(args.min_docs, int((work_pool_tokens * args.work_multiple) // L))
        n_flood = max(args.min_flood, int((work_pool_tokens * args.flood_multiple) // L))
        print(f"\n=== context_len={L}  n_docs={n_docs}  n_flood={n_flood}  "
              f"working_set={n_docs * L} tokens ===")
        docs = [build_doc(tok, i, L) for i in range(n_docs)]
        flood = [build_doc(tok, f"flood{i}", L) for i in range(n_flood)]

        h0, p0 = snapshot_park()
        lb0 = snapshot_hicache_tokens()

        print("  phase 1/4 (warm hit, hit_count=1)...")
        for i, d in enumerate(docs):
            dt = send(d)
            print(f"    doc {i}: {dt:.2f}s")

        print("  phase 2/4 (write hit, crosses write_through_threshold)...")
        for i, d in enumerate(docs):
            dt = send(d)
            print(f"    doc {i}: {dt:.2f}s")

        print(f"  settling {args.settle_s}s for the async host write to land...")
        time.sleep(args.settle_s)

        print("  phase 3/4 (flood -- evicts the phase-1/2 docs from GPU)...")
        for i, d in enumerate(flood):
            dt = send(d)
            print(f"    flood {i}: {dt:.2f}s")

        print("  phase 4/4 (verify hit -- forces the genuine load-back)...")
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
            "n_flood": n_flood,
            "working_set_tokens": n_docs * L,
            "host_gb": round(
                (host_bytes_park if args.arm == "park" else host_bytes_hicache) / 1e9, 4
            ),
            "park_gpu_gpu_gb": round(gpu_gpu_bytes_park / 1e9, 4),
            "bytes_per_token": bpt,
        }
        print(f"  -> {row}")
        rows.append(row)

    if args.append and os.path.exists(args.out):
        with open(args.out) as fh:
            prior = json.load(fh)
        by_len = {r["context_len"]: r for r in prior.get("rows", [])}
        for r in rows:
            by_len[r["context_len"]] = r  # new measurement replaces an old one at the same L
        rows = sorted(by_len.values(), key=lambda r: r["context_len"])

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"arm": args.arm, "model": MODEL, "rows": rows}, fh, indent=2)
    print(f"\n[saved] {args.out}  ({len(rows)} context lengths)")


if __name__ == "__main__":
    main()
