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
HICACHE_DISK_DIR = os.environ.get("HICACHE_DISK_DIR", "/tmp/hicache")

_LOAD_BACK_KEY = "sglang:load_back_tokens_total"
_CACHE_GAUGE_KEYS = (
    "sglang:cache_occupancy", "sglang:token_usage", "sglang:evictable_tokens",
    "sglang:max_total_num_tokens",
)


def snapshot_hicache_disk_bytes():
    """Direct, unambiguous evidence of host-side writes for the hicache arm: the L3
    file-storage backend's on-disk footprint (HICACHE_STORAGE_BACKEND=file, see
    start_2P_2D.sh) rather than inferring "it must have written something" backwards
    from a nonzero read counter. 0 if the directory doesn't exist (recompute/park
    never create it)."""
    import subprocess
    if not os.path.isdir(HICACHE_DISK_DIR):
        return 0
    try:
        out = subprocess.run(["du", "-sb", HICACHE_DISK_DIR], capture_output=True,
                              text=True, timeout=30)
        return int(out.stdout.split()[0])
    except Exception:
        return 0


def clear_hicache_disk_and_check_free(min_free_gb=15.0):
    """Call at the START of every context length, not just once per arm.

    Two live incidents drove this: (1) hicache's file-storage write failures are
    SWALLOWED by sglang -- it logs "Failed to save tensor ... 0 written" / "Write page
    to storage: N pages failed" and keeps serving degraded to a cache miss rather than
    raising, so this script's own requests.post() never sees an error and the run
    "succeeds" with numbers that quietly stopped meaning what they should; the failure
    was only visible by hand-reading the prefill server's log. (2) a one-time disk
    check before the whole arm starts (run_*_sweep.sh's preflight) isn't enough once
    documents are this large (up to 17GB each at 128k) -- /tmp/hicache was never
    cleared BETWEEN context lengths within one arm's run, so usage climbs
    monotonically across the whole CONTEXT_LENS list and can exhaust the disk mid-run
    even though there was plenty of room when the arm started.

    Clearing it before each L keeps peak usage bounded to roughly one L's worth of
    documents, and re-checking free space here (not just once at the top of the shell
    script) means a still-too-small disk fails LOUDLY in Python with a clear message
    instead of silently degrading two layers down in a log file the caller isn't
    watching.

    Empties the directory's CONTENTS, not the directory itself. rmtree()-ing
    HICACHE_DISK_DIR outright was the first version of this fix and immediately broke
    the very thing it was protecting: sglang doesn't create/reopen that directory per
    write, so with it gone every subsequent write failed with ENOENT ("No such file or
    directory: '/tmp/hicache/<hash>.bin'") -- confirmed live, right after this function
    started running. Removing rmtree()'s target one level down (the directory's entries)
    instead of the directory node itself keeps the path sglang expects to already exist
    intact throughout."""
    import shutil
    if os.path.isdir(HICACHE_DISK_DIR):
        for name in os.listdir(HICACHE_DISK_DIR):
            path = os.path.join(HICACHE_DISK_DIR, name)
            if os.path.isdir(path) and not os.path.islink(path):
                shutil.rmtree(path, ignore_errors=True)
            else:
                try:
                    os.remove(path)
                except OSError:
                    pass
    free_gb = shutil.disk_usage(".").free / 1e9
    print(f"  disk free: {free_gb:.1f}G  (/tmp/hicache cleared)")
    if free_gb < min_free_gb:
        raise SystemExit(
            f"only {free_gb:.1f}G free (need {min_free_gb}G). hicache's file backend "
            f"writes ~1 doc's worth of KV bytes per document at this L, and sglang "
            f"won't error loudly when it runs out -- it'll just silently degrade to a "
            f"cache miss. Free space (check `df -h` / `du -sh /tmp/hicache`) and re-run."
        )


def snapshot_cache_gauges():
    """Current (not delta) sglang:cache_occupancy / evictable_tokens / token_usage per
    prefill port -- the same gauges probe_cache_metric.sh reads. evictable staying near
    0 after a load means the GPU-side radix tree isn't holding what phase 4 needs, i.e.
    it had to come from somewhere else (host or a genuine miss) rather than a local hit
    quietly padding the "it worked" read."""
    out = {}
    for port in PREFILL_PORTS:
        vals = {}
        try:
            with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=5) as r:
                text = r.read().decode("utf-8", "replace")
            for line in text.splitlines():
                if line.startswith("#"):
                    continue
                parts = line.rsplit(None, 1)
                if len(parts) != 2:
                    continue
                key = parts[0].split("{", 1)[0]
                if key in _CACHE_GAUGE_KEYS:
                    try:
                        vals[key.split(":", 1)[1]] = float(parts[1])
                    except ValueError:
                        pass
        except Exception:
            pass
        out[port] = vals
    return out


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
    """Cumulative FETCHED tokens by serving tier, plus raw hit/miss counts, summed
    across every prefill's telemetry file. host = read back from CPU DRAM (Host->GPU
    traffic); local/peer = read back from a GPU park pool (GPU->GPU, no host DRAM
    involved). hits/miss distinguish "0 GB because the mechanism avoided host traffic"
    from "0 GB because nothing was ever found parked and every phase-4 hit just
    recomputed from scratch" -- the two look identical on host_gb/park_gpu_gpu_gb
    alone but mean opposite things."""
    host_tok = gpu_tok = hits = miss = 0
    for path in glob.glob(os.path.join(PARK_DIR, "parked_gpu*.json")):
        try:
            with open(path) as fh:
                d = json.load(fh)
        except Exception:
            continue
        tier = d.get("fetch_tok_tier") or {}
        host_tok += int(tier.get("host") or 0)
        gpu_tok += int(tier.get("local") or 0) + int(tier.get("peer") or 0)
        hits += int(d.get("fetch_hits") or 0)
        miss += int(d.get("fetch_miss") or 0)
    return host_tok, gpu_tok, hits, miss


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
    # doc_id embedded in EVERY repetition, not just a header -- a header alone only
    # diverges the first ~10 tokens, after which every document (and every flood doc,
    # and every OTHER context length's documents) shares the exact same repeated
    # sentence verbatim. The radix tree matches on shared prefix, so "N distinct
    # documents" was actually one giant shared node with N tiny distinct fronts:
    # eviction pressure, hit_count (accumulates on the SHARED node, not per document),
    # and every downstream count in this script assumed independence that didn't exist.
    # Confirmed live -- hicache read back exactly 4020 tokens at 8k, 16k, AND 32k, three
    # different requested lengths converging on the same number, consistent with
    # different-L runs all resolving to the same shared filler node rather than to
    # independent per-length content.
    filler = f"[[DOC {doc_id}]] The quick brown fox jumps over the lazy dog near the riverbank at dawn. "
    budget = n_tokens - _GEN_TOKENS - _SAFETY_BUFFER

    text = filler
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
    ap.add_argument(
        "--min-free-gb", type=float, default=15.0,
        help="hard-fail before starting a context length if less than this much disk "
             "is free (checked fresh each L, after clearing /tmp/hicache -- see "
             "clear_hicache_disk_and_check_free)",
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
        clear_hicache_disk_and_check_free(args.min_free_gb)
        n_docs = max(args.min_docs, int((work_pool_tokens * args.work_multiple) // L))
        n_flood = max(args.min_flood, int((work_pool_tokens * args.flood_multiple) // L))
        print(f"\n=== context_len={L}  n_docs={n_docs}  n_flood={n_flood}  "
              f"working_set={n_docs * L} tokens ===")
        # doc_id includes L: without it, doc_id=0 at L=8192 and doc_id=0 at a LATER
        # L=16384 iteration would still share their first 8192 tokens verbatim (both
        # built from the same "[[DOC 0]] ..." repeated unit), letting a later context
        # length's phase-1 "warm hit" partially hit whatever the earlier length's run
        # left resident/parked -- the same cross-request sharing bug this function
        # exists to avoid, just moved from cross-doc to cross-L.
        docs = [build_doc(tok, f"L{L}_{i}", L) for i in range(n_docs)]
        flood = [build_doc(tok, f"L{L}_flood{i}", L) for i in range(n_flood)]

        h0, p0, hits0, miss0 = snapshot_park()
        lb0 = snapshot_hicache_tokens()
        disk0 = snapshot_hicache_disk_bytes()

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

        h1, p1, hits1, miss1 = snapshot_park()
        lb1 = snapshot_hicache_tokens()
        disk1 = snapshot_hicache_disk_bytes()
        gauges = snapshot_cache_gauges()

        host_bytes_park = max(0, h1 - h0) * bpt
        gpu_gpu_bytes_park = max(0, p1 - p0) * bpt
        host_bytes_hicache = max(0.0, lb1 - lb0) * bpt
        park_hits = max(0, hits1 - hits0)
        park_miss = max(0, miss1 - miss0)
        disk_delta_gb = round(max(0, disk1 - disk0) / 1e9, 4)
        print(f"  hicache disk (/tmp/hicache) delta: {disk_delta_gb} GB   "
              f"cache gauges (post-phase-4): {gauges}")

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
            # park-only diagnostic: distinguishes "0 GB, avoided host traffic" from
            # "0 GB, phase 4 never even found anything parked and just recomputed" --
            # only meaningful for arm=park (hicache/recompute don't publish this file).
            "park_fetch_hits": park_hits,
            "park_fetch_miss": park_miss,
            # direct evidence of host-side writes for hicache (L3 file-backend disk
            # footprint), rather than only inferring "it must have written something"
            # backwards from a nonzero read counter.
            "hicache_disk_delta_gb": disk_delta_gb,
            "cache_gauges_post": gauges,
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
