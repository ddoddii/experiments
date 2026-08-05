"""
SGLang KV Cache Tier Latency Benchmark
=======================================
GPU L1 vs CPU DRAM L2 vs SSD L3 KV fetch 시간 측정

실험 원리:
  같은 "anchor" prefix를 갖는 요청을 반복하면서 TTFT를 측정한다.
  Anchor KV를 특정 tier에 두기 위해 GPU/DRAM KV pool을 다른 요청들로 채워
  LRU eviction을 강제한다.

  Phase 0 COLD : anchor 첫 요청 → GPU에서 계산 (cache miss)
  Phase 1 L1   : 즉시 재요청    → GPU L1 hit (baseline)
  Phase 2 L2   : GPU 채움 → anchor GPU 축출, DRAM에서 로드
  Phase 3 L3   : DRAM 채움 → anchor DRAM 축출, SSD에서 로드

  TTFT(L2) - TTFT(L1) = DRAM→GPU KV 전송 오버헤드
  TTFT(L3) - TTFT(L1) = SSD→DRAM→GPU KV 전송 오버헤드

L3 anchor 설계 (핵심):
  L2 측정에서 anchor를 DRAM에서 읽으면 anchor의 DRAM LRU가 "지금"으로 갱신된다.
  이후 Phase 3에서 flood를 보내도 anchor가 "최신"이라 DRAM에서 밀려나지 않는다.

  해결: L3 전용 별도 anchor를 실험 시작 전(Pre-Phase)에 미리 DRAM에 써둔다.
  이 L3 anchor의 DRAM LRU는 실험 전체에서 가장 오래됐다.
  Phase 2 flood, L2 측정, Phase 3 소량 flood 후에도 L3 anchor가 제일 오래됐으므로
  DRAM이 조금만 넘쳐도 L3 anchor가 SSD로 먼저 축출된다.
  → SSD 기록량: anchor KV (~780MB) + 소량. 디스크 가득 찰 위험 없음.

실험 조건 (중요):
  - 모든 요청이 동일한 P 노드에서 처리되어야 한다.
  - 2P2D 라우터는 P1/P2를 랜덤 분산하므로, 단일 P+D 라우터 사용 권장:

    python -m sglang_router.launch_router \\
      --pd-disaggregation \\
      --prefill http://127.0.0.1:30000 8998 \\
      --decode  http://127.0.0.1:30002 \\
      --host 0.0.0.0 --port 8001

  - hicache SSD 경로가 실제 디스크인지 확인:
    df -T /tmp  # tmpfs면 SSD 측정 의미 없음
    SGLang 서버에 --hicache-storage-backend-extra-config '{"path":"/data/hicache"}' 추가 권장

메모리 예산 (Qwen3-14B, A6000):
  모델 가중치 : 28 GB (14B × 2B BF16)
  GPU KV 여유 : ~18 GB  → ~115,200 tokens
  DRAM KV     : ~22 GB  → ~140,800 tokens (hicache_ratio=1.2 × 18GB)

Usage:
  # 단일 P+D 라우터 기준
  SERVER_URL=http://127.0.0.1:8001 \\
    python benchmark/sglang_kv_tier_latency.py

  # 파라미터 커스텀
  SERVER_URL=http://127.0.0.1:8001 \\
  ANCHOR_CHARS=15000 GPU_EVICT_N=35 DRAM_EVICT_N=8 \\
    python benchmark/sglang_kv_tier_latency.py

Environment variables:
  SERVER_URL       SGLang 라우터 주소        (default: http://127.0.0.1:8001)
  P_NODE_URL       P 노드 직접 주소 (stats용)   (default: http://127.0.0.1:30000)
  MODEL            모델 경로                  (default: /home/uhmturks/hf_models/Qwen3-14B)
  ANCHOR_CHARS     anchor 텍스트 길이        (default: 15000 ≈ 5000 tokens)
  EVICT_CHARS      eviction 요청 텍스트 길이  (default: 12000 ≈ 4000 tokens)
  GPU_EVICT_N      GPU 축출용 eviction 요청 수   (default: 32)
  L3_FLOOD_N       L3 eviction 요청 수              (default: 20)
                   L3 anchor (Pre-Phase에 DRAM에 써둔 것)이 DRAM 최고령이라
                   DRAM 조금만 넘쳐도 SSD로 먼저 밀려남. 20개면 충분히 여유있음.
                   GPU KV가 예상보다 크면 30으로 늘려볼 것.
  REPEAT_N         phase당 측정 반복 수      (default: 5)
  GPU_KV_GB        GPU KV 예산 GB            (default: 18.0)
  DRAM_KV_GB       DRAM KV 예산 GB           (default: 21.6)
  HICACHE_PATH     SGLang hicache SSD 경로   (default: /tmp/hicache)
  MAX_TOKENS_OUT   응답 최대 토큰 수         (default: 16)
"""

import json
import os
import shutil
import statistics
import sys
import time

import requests
from tqdm import tqdm

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

# ─── Config ─────────────────────────────────────────────────────────────────
SERVER_URL     = os.environ.get("SERVER_URL",    "http://127.0.0.1:8001")
P_NODE_URL     = os.environ.get("P_NODE_URL",    "http://127.0.0.1:30000")
MODEL          = os.environ.get("MODEL",         "/home/uhmturks/hf_models/Qwen3-14B")
ANCHOR_CHARS   = int(os.environ.get("ANCHOR_CHARS",   "15000"))  # ≈ 5000 tokens
EVICT_CHARS    = int(os.environ.get("EVICT_CHARS",    "12000"))  # ≈ 4000 tokens per request
GPU_EVICT_N    = int(os.environ.get("GPU_EVICT_N",    "32"))     # GPU 채우기 요청 수
DRAM_EVICT_N   = int(os.environ.get("DRAM_EVICT_N",   "30"))     # Phase 2 GPU evict 이후 L2 측정용 (사용 안함)
L3_FLOOD_N     = int(os.environ.get("L3_FLOOD_N",     "20"))     # L3 eviction 요청 수 (소량으로 충분)
REPEAT_N       = int(os.environ.get("REPEAT_N",       "5"))
GPU_KV_GB      = float(os.environ.get("GPU_KV_GB",    "18.0"))
DRAM_KV_GB     = float(os.environ.get("DRAM_KV_GB",   "21.6"))   # 1.2 × GPU_KV_GB
HICACHE_PATH   = os.environ.get("HICACHE_PATH",       "/tmp/hicache")
MAX_TOKENS_OUT = int(os.environ.get("MAX_TOKENS_OUT", "16"))

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
CHAT_URL   = f"{SERVER_URL.rstrip('/')}/v1/chat/completions"

# ─── Memory estimation ───────────────────────────────────────────────────────
# Qwen3-14B: 40 layers, 8 KV heads, 128 head_dim, BF16 (2 bytes)
KV_LAYERS    = int(os.environ.get("KV_LAYERS",    "40"))
KV_HEADS     = int(os.environ.get("KV_HEADS",     "8"))
KV_HEAD_DIM  = int(os.environ.get("KV_HEAD_DIM",  "128"))
KV_BYTES_PER_TOKEN = 2 * KV_LAYERS * KV_HEADS * KV_HEAD_DIM * 2  # BF16

CHARS_PER_TOKEN    = 3.0  # rough: 1 token ≈ 3 chars

gpu_max_tokens  = int(GPU_KV_GB  * (1024 ** 3) / KV_BYTES_PER_TOKEN)
dram_max_tokens = int(DRAM_KV_GB * (1024 ** 3) / KV_BYTES_PER_TOKEN)
anchor_tok_est  = int(ANCHOR_CHARS / CHARS_PER_TOKEN)
evict_tok_est   = int(EVICT_CHARS  / CHARS_PER_TOKEN)

# ─── Startup banner ─────────────────────────────────────────────────────────
print("=" * 68)
print("SGLang KV Cache Tier Latency Benchmark")
print("=" * 68)
print(f"  Model        : {MODEL_SLUG}")
print(f"  Server       : {CHAT_URL}")
print(f"  P-node       : {P_NODE_URL} (for cache stats)")
print()
print(f"  KV bytes/tok : {KV_BYTES_PER_TOKEN:,} B  ({KV_BYTES_PER_TOKEN/1024:.1f} KB)")
print(f"  GPU KV       : {GPU_KV_GB:.1f} GB → ~{gpu_max_tokens:,} tokens")
print(f"  DRAM KV      : {DRAM_KV_GB:.1f} GB → ~{dram_max_tokens:,} tokens")
print()
print(f"  Anchor       : {ANCHOR_CHARS:,} chars ≈ {anchor_tok_est:,} tokens"
      f"  ({anchor_tok_est * KV_BYTES_PER_TOKEN / 1024**3:.2f} GB KV)")
print(f"  Evict/req    : {EVICT_CHARS:,} chars ≈ {evict_tok_est:,} tokens")
print(f"  GPU_EVICT_N  : {GPU_EVICT_N} requests"
      f"  ({GPU_EVICT_N*evict_tok_est:,} tokens = {GPU_EVICT_N*evict_tok_est/gpu_max_tokens*100:.0f}% of GPU KV)")
print(f"  DRAM_EVICT_N : {DRAM_EVICT_N} requests"
      f"  ({DRAM_EVICT_N*evict_tok_est:,} tokens = {DRAM_EVICT_N*evict_tok_est/dram_max_tokens*100:.0f}% of DRAM KV)")
print(f"  Repeat N     : {REPEAT_N} per phase")
print(f"  L3_FLOOD_N   : {L3_FLOOD_N} requests  (small flood to push pre-aged L3 anchor to SSD)")
print(f"  HiCache path : {HICACHE_PATH}")
print("=" * 68)
print()
print("  L3 anchor design: computed at experiment START (before Phase 0) so its DRAM")
print("  LRU timestamp is the oldest. Phase 3 flood overflows DRAM by just enough to")
print("  evict it to SSD. No hicache cleanup needed — SSD writes are tiny (~few GB).")
print("=" * 68)

# ─── Storage tier check ──────────────────────────────────────────────────────
def check_tmp_is_tmpfs() -> bool:
    try:
        import subprocess
        out = subprocess.check_output(["df", "-T", "/tmp"], text=True)
        return "tmpfs" in out
    except Exception:
        return False

if check_tmp_is_tmpfs():
    print("⚠  WARNING: /tmp is tmpfs (RAM). L3 'SSD' measurements will show")
    print("   near-zero extra latency. Configure hicache to a real disk path")
    print("   (--hicache-storage-backend-extra-config '{\"path\":\"/data/hicache\"}')")
    print()

# ─── Text generation ────────────────────────────────────────────────────────
def _anchor_text(n: int) -> str:
    """Fixed deterministic English text for L1/L2 anchor."""
    unit = "the quick brown fox jumps over the lazy dog "
    return (unit * (n // len(unit) + 1))[:n]

def _l3_anchor_text(n: int) -> str:
    """Different phrase for L3 anchor — no shared prefix with L1/L2 anchor.
    Computed BEFORE Phase 0 so its DRAM LRU timestamp is the oldest."""
    unit = "pack my box with five dozen liquor jugs now "
    return (unit * (n // len(unit) + 1))[:n]

def _evict_text(seed: int, n: int) -> str:
    """Unique per seed; starts with EVICT{seed} → no shared prefix with anchors."""
    unit = f"evict{seed:06d}x "
    return (unit * (n // len(unit) + 1))[:n]

ANCHOR_TEXT   = _anchor_text(ANCHOR_CHARS)
ANCHOR_MESSAGES = [
    {"role": "system", "content": f"ANCHOR context: {ANCHOR_TEXT}"},
    {"role": "user",   "content": "Summarize in one word."},
]

L3_ANCHOR_TEXT = _l3_anchor_text(ANCHOR_CHARS)
L3_ANCHOR_MESSAGES = [
    {"role": "system", "content": f"L3ANCHOR context: {L3_ANCHOR_TEXT}"},
    {"role": "user",   "content": "Summarize in one word."},
]

def evict_messages(seed: int) -> list:
    return [
        {"role": "system", "content": f"EVICT context: {_evict_text(seed, EVICT_CHARS)}"},
        {"role": "user",   "content": "Summarize in one word."},
    ]

# ─── Request helper ──────────────────────────────────────────────────────────
def do_request(messages: list, label: str = "") -> tuple[float | None, float | None]:
    """Stream request; returns (ttft_s, e2e_s). None on failure."""
    payload = {
        "model":       MODEL,
        "messages":    messages,
        "max_tokens":  MAX_TOKENS_OUT,
        "temperature": 0,
        "stream":      True,
    }
    t0 = time.perf_counter()
    try:
        resp = requests.post(CHAT_URL, json=payload, stream=True, timeout=300)
        resp.raise_for_status()
    except Exception as e:
        tqdm.write(f"  [ERR {label}] {e}")
        return None, None

    t_first = t_last = None
    for line in resp.iter_lines():
        if not line:
            continue
        raw = line.decode("utf-8")
        if not raw.startswith("data: "):
            continue
        data = raw[6:]
        if data == "[DONE]":
            break
        try:
            chunk = json.loads(data)
        except json.JSONDecodeError:
            continue
        if not chunk.get("choices"):
            continue
        delta = chunk["choices"][0]["delta"]
        if delta.get("content"):
            if t_first is None:
                t_first = time.perf_counter()
            t_last = time.perf_counter()

    ttft = (t_first - t0) if t_first else None
    e2e  = (t_last  - t0) if t_last  else None
    return ttft, e2e

# ─── Cache stats (P node) ────────────────────────────────────────────────────
def get_cache_stats() -> dict:
    try:
        r = requests.get(f"{P_NODE_URL}/get_server_info", timeout=5)
        info = r.json()
        return {
            "token_usage":       info.get("token_usage"),
            "num_total_tokens":  info.get("num_total_tokens"),
            "num_cached_tokens": info.get("num_cached_tokens"),
        }
    except Exception:
        return {}

def print_cache_stats(label: str):
    stats = get_cache_stats()
    if stats.get("token_usage") is not None:
        print(f"    [cache {label}] token_usage={stats['token_usage']:.3f}  "
              f"cached={stats.get('num_cached_tokens')}/"
              f"{stats.get('num_total_tokens')}")

# ─── Disk / hicache helpers ──────────────────────────────────────────────────
def free_disk_gb(path: str = "/tmp") -> float:
    try:
        st = os.statvfs(path)
        return st.f_bavail * st.f_frsize / 1024 ** 3
    except Exception:
        return float("inf")

def hicache_used_gb() -> float:
    try:
        total = 0
        for root, _dirs, files in os.walk(HICACHE_PATH):
            for f in files:
                try:
                    total += os.path.getsize(os.path.join(root, f))
                except OSError:
                    pass
        return total / 1024 ** 3
    except Exception:
        return 0.0

def clean_hicache(reason: str = ""):
    """Remove all SSD hicache data to free disk space.
    Safe to call when anchor is in DRAM (not yet evicted to SSD)."""
    if not os.path.isdir(HICACHE_PATH):
        return
    used_gb = hicache_used_gb()
    shutil.rmtree(HICACHE_PATH, ignore_errors=True)
    os.makedirs(HICACHE_PATH, exist_ok=True)
    msg = f"  Cleaned {HICACHE_PATH} ({used_gb:.1f} GB freed)"
    if reason:
        msg += f" — {reason}"
    print(msg)

# ─── Phase runner ────────────────────────────────────────────────────────────
def measure_phase(label: str, messages: list | None = None) -> tuple[list[float], float | None]:
    """Run REPEAT_N requests; return (ttft_list_ms, first_miss_ms).

    first_miss_ms is the first request's TTFT — this is the actual tier load
    latency (cache miss). Subsequent requests hit GPU cache and measure ~L1.
    """
    msgs = messages if messages is not None else ANCHOR_MESSAGES
    ttfts = []
    first_miss: float | None = None
    for i in range(REPEAT_N):
        ttft, e2e = do_request(msgs, label=f"{label}[{i}]")
        ms = ttft * 1000 if ttft else None
        e2e_ms = e2e * 1000 if e2e else None
        if ms is not None:
            ttfts.append(ms)
            if first_miss is None:
                first_miss = ms
            marker = " ← tier miss" if i == 0 else ""
            print(f"    [{label} {i+1}/{REPEAT_N}]  TTFT={ms:7.1f} ms  e2e={e2e_ms:.1f} ms{marker}")
        else:
            print(f"    [{label} {i+1}/{REPEAT_N}]  FAILED")
    return ttfts, first_miss

def flood(desc: str, seeds: range):
    """Send eviction requests to fill KV pool."""
    for seed in tqdm(seeds, desc=desc, leave=False):
        do_request(evict_messages(seed), label=f"flood_{seed}")

def stats_dict(vals: list[float]) -> dict:
    if not vals:
        return {}
    return {
        "min_ms":    round(min(vals),               1),
        "mean_ms":   round(statistics.mean(vals),   1),
        "median_ms": round(statistics.median(vals), 1),
        "max_ms":    round(max(vals),               1),
        "samples":   len(vals),
    }

def fmt_stats(vals: list[float]) -> str:
    if not vals:
        return "N/A"
    s = stats_dict(vals)
    return (f"min={s['min_ms']} "
            f"mean={s['mean_ms']} "
            f"median={s['median_ms']} "
            f"max={s['max_ms']} ms")

# ─── Experiment ──────────────────────────────────────────────────────────────
record = {}

# ── Pre-Phase: L3 anchor into DRAM (MUST run before Phase 0) ─────────────────
# L3 anchor is computed NOW so its DRAM LRU timestamp (t_pre) is the oldest.
# Phase 0/1/2 run after this, so Phase 2 evictions and L2 re-access all have
# timestamps NEWER than t_pre. When Phase 3 overflows DRAM, L3 anchor (oldest)
# is evicted to SSD first — no need for large flood or hicache cleanup.
print("\n[Pre-Phase] Computing L3 anchor into GPU+DRAM ...")
print(f"  Text: 'L3ANCHOR context: {L3_ANCHOR_TEXT[:60]}...'")
do_request(L3_ANCHOR_MESSAGES, "L3-PRE")
print("  L3 anchor written to DRAM (LRU timestamp = oldest in experiment).")

# Phase 0: COLD
print("\n[Phase 0] COLD — no cache (initial prefill)")
ttft_cold, _ = do_request(ANCHOR_MESSAGES, "COLD")
cold_ms = ttft_cold * 1000 if ttft_cold else None
print(f"  COLD TTFT = {cold_ms:.1f} ms" if cold_ms else "  COLD FAILED")
record["cold"] = {"ttft_ms": round(cold_ms, 1) if cold_ms else None}

# Phase 1: L1 GPU
print("\n[Phase 1] L1 GPU — KV in GPU (immediate re-request)")
ttfts_l1, _ = measure_phase("L1")
print(f"  → {fmt_stats(ttfts_l1)}")
print_cache_stats("after-L1")
record["l1_gpu"] = stats_dict(ttfts_l1)

# Phase 2: L2 DRAM
# Evict anchor from GPU by filling GPU KV pool with GPU_EVICT_N unique requests.
# (anchor stays in DRAM via write_through)
print(f"\n[Phase 2] L2 DRAM — evicting anchor from GPU with {GPU_EVICT_N} requests")
print(f"  Loading ~{GPU_EVICT_N * evict_tok_est:,} tokens into GPU KV "
      f"({GPU_EVICT_N * evict_tok_est / gpu_max_tokens * 100:.0f}% of capacity)")
flood("GPU evict", range(1000, 1000 + GPU_EVICT_N))
print_cache_stats("after-GPU-flood")

print("  Measuring anchor TTFT (expected: L2 DRAM hit) ...")
ttfts_l2, l2_miss_ms = measure_phase("L2")
print(f"  → {fmt_stats(ttfts_l2)}")
print(f"    (first-miss = {l2_miss_ms:.1f} ms = DRAM load latency)")
record["l2_dram"] = {
    **stats_dict(ttfts_l2),
    "first_miss_ms": round(l2_miss_ms, 1) if l2_miss_ms else None,
    "gpu_evict_n": GPU_EVICT_N,
    "evict_tokens": GPU_EVICT_N * evict_tok_est,
}

# Phase 3: L3 SSD
# L3 anchor (computed in Pre-Phase) is the OLDEST item in DRAM (LRU = t_pre).
# Phase 2 evictions and L2 re-access of anchor all happened AFTER t_pre.
# A small flood overflows DRAM → L3 anchor evicted to SSD first (LRU policy).
# SSD writes are tiny: only ~anchor_kv_mb + a bit more (no multi-GB flood needed).
print(f"\n[Phase 3] L3 SSD — overflowing DRAM to evict pre-aged L3 anchor to SSD")
print(f"  L3 anchor DRAM LRU = oldest in experiment (t_pre, before Phase 0).")
print(f"  Flood: {L3_FLOOD_N} requests overflow DRAM → L3 anchor → SSD.")
print(f"  (Disk writes: ~{max(L3_FLOOD_N * evict_tok_est - dram_max_tokens, evict_tok_est) * KV_BYTES_PER_TOKEN / 1024**3:.1f} GB est.)")
flood("L3 flood", range(5000, 5000 + L3_FLOOD_N))

print("  Waiting 5s for async SSD writes to complete ...")
time.sleep(5)
print_cache_stats("after-L3-flood")

print("  Measuring L3 anchor TTFT (expected: SSD hit) ...")
ttfts_l3, l3_miss_ms = measure_phase("L3", messages=L3_ANCHOR_MESSAGES)
print(f"  → {fmt_stats(ttfts_l3)}")
print(f"    (first-miss = {l3_miss_ms:.1f} ms = SSD load latency)")
record["l3_ssd"] = {
    **stats_dict(ttfts_l3),
    "first_miss_ms": round(l3_miss_ms, 1) if l3_miss_ms else None,
    "l3_flood_n":   L3_FLOOD_N,
    "evict_tokens": L3_FLOOD_N * evict_tok_est,
}

# ─── Summary ─────────────────────────────────────────────────────────────────
# Use first-miss (= [0]) for overhead/bandwidth — it's the actual tier load.
# Subsequent hits go to GPU cache and reflect L1-like latency, not tier latency.
l1_baseline = min(ttfts_l1) if ttfts_l1 else None  # pure GPU hit, no load overhead

l2_overhead = round(l2_miss_ms - l1_baseline, 1) if (l2_miss_ms and l1_baseline) else None
l3_overhead = round(l3_miss_ms - l1_baseline, 1) if (l3_miss_ms and l1_baseline) else None
l3_vs_l2    = round(l3_miss_ms - l2_miss_ms,  1) if (l3_miss_ms and l2_miss_ms)  else None

anchor_kv_mb = anchor_tok_est * KV_BYTES_PER_TOKEN / 1024 / 1024

print("\n" + "=" * 68)
print("RESULTS SUMMARY  (overheads use first-miss = actual tier load)")
print("=" * 68)
print(f"  Anchor KV    : {anchor_tok_est:,} tokens  {anchor_kv_mb:.0f} MB")
print()
print(f"  Phase 0 COLD  : {cold_ms:.1f} ms" if cold_ms else "  Phase 0 COLD  : FAILED")
print(f"  Phase 1 L1 GPU: {fmt_stats(ttfts_l1)}")
print(f"  Phase 2 L2 RAM: first-miss={l2_miss_ms:.1f} ms  │  GPU re-hits: {fmt_stats(ttfts_l2[1:])}"
      if (l2_miss_ms and len(ttfts_l2) > 1) else f"  Phase 2 L2 RAM: {fmt_stats(ttfts_l2)}")
print(f"  Phase 3 L3 SSD: first-miss={l3_miss_ms:.1f} ms  │  GPU re-hits: {fmt_stats(ttfts_l3[1:])}"
      if (l3_miss_ms and len(ttfts_l3) > 1) else f"  Phase 3 L3 SSD: {fmt_stats(ttfts_l3)}")
print()
if l2_overhead is not None:
    print(f"  L2 vs L1 overhead : +{l2_overhead:.1f} ms  (DRAM→GPU fetch)")
if l3_vs_l2 is not None:
    print(f"  L3 vs L2 overhead : +{l3_vs_l2:.1f} ms  (SSD→DRAM additional)")
if l3_overhead is not None:
    print(f"  L3 vs L1 overhead : +{l3_overhead:.1f} ms  (total SSD→DRAM→GPU fetch)")
print()

if l2_overhead and l2_overhead > 0:
    dram_bw = anchor_kv_mb / (l2_overhead / 1000)
    print(f"  Est. DRAM→GPU bandwidth : {dram_bw:.0f} MB/s  (PCIe limit ~32,000 MB/s)")
if l3_vs_l2 and l3_vs_l2 > 0:
    ssd_bw = anchor_kv_mb / (l3_vs_l2 / 1000)
    print(f"  Est. SSD→DRAM bandwidth : {ssd_bw:.0f} MB/s  (NVMe limit ~7,000 MB/s)")

# ── Recompute vs Fetch comparison ────────────────────────────────────────────
print()
print("  ── Recompute vs Fetch (for this anchor length) ──")
if cold_ms:
    print(f"  Cold recompute : {cold_ms:.1f} ms  (P-node GPU prefill from scratch)")
if l2_miss_ms and cold_ms:
    delta_l2 = l2_miss_ms - cold_ms
    sign = "+" if delta_l2 >= 0 else ""
    verdict = "SLOWER than recompute" if delta_l2 > 0 else "faster than recompute"
    print(f"  L2 DRAM fetch  : {l2_miss_ms:.1f} ms  ({sign}{delta_l2:.1f} ms vs recompute → {verdict})")
if l3_miss_ms and cold_ms:
    delta_l3 = l3_miss_ms - cold_ms
    sign = "+" if delta_l3 >= 0 else ""
    verdict = "SLOWER than recompute" if delta_l3 > 0 else "faster than recompute"
    print(f"  L3 SSD  fetch  : {l3_miss_ms:.1f} ms  ({sign}{delta_l3:.1f} ms vs recompute → {verdict})")
print()
# Crossover insight
if cold_ms and l2_miss_ms and l1_baseline:
    if l2_miss_ms >= cold_ms:
        # For this anchor length, recompute wins. Estimate crossover.
        # compute_time ∝ N (approximate: prefill = O(N) for causal mask amortized)
        # fetch_time ≈ const (DRAM bandwidth overhead is mostly transfer-size independent latency + N*bytes/bw)
        # At crossover: cold_ms/anchor_tok_est * N_cross = l2_overhead * (N_cross/anchor_tok_est)
        # Both sides scale with N → no simple crossover unless there's a fixed component
        # Simpler: note that DRAM bw overhead = l2_overhead ms for anchor_tok_est tokens
        # GPU compute speed = anchor_tok_est / (cold_ms - l1_baseline) tokens/ms (minus decode overhead)
        compute_only_ms = cold_ms - l1_baseline  # subtract decode overhead
        if compute_only_ms > 0 and l2_overhead and l2_overhead > 0:
            compute_tok_per_ms = anchor_tok_est / compute_only_ms
            dram_tok_per_ms    = anchor_kv_mb * 1024 / (anchor_kv_mb / (l2_overhead / 1000) / 1024)
            crossover_tok = int(compute_tok_per_ms * (l2_overhead / 1000) * 1000)
            print(f"  Note: At {anchor_tok_est:,} tokens, DRAM fetch ({l2_miss_ms:.0f}ms) >= recompute ({cold_ms:.0f}ms).")
            print(f"  Hicache L2/L3 helps TTFT only for shorter contexts or faster PCIe bandwidth.")
        else:
            print(f"  Note: DRAM fetch >= recompute at {anchor_tok_est:,} tokens.")
            print(f"  For shorter contexts hicache would be beneficial.")
    else:
        print(f"  Note: DRAM fetch < recompute at {anchor_tok_est:,} tokens → hicache beneficial here.")

# Sanity checks on first-miss values
if l1_baseline and l2_miss_ms and l2_miss_ms < l1_baseline + 50:
    print()
    print("  ⚠  L2 first-miss ≈ L1: anchor may not have been evicted from GPU.")
    print(f"     Try increasing GPU_EVICT_N (current: {GPU_EVICT_N})")

if l2_miss_ms and l3_miss_ms and l3_miss_ms < l2_miss_ms + 50:
    print()
    print("  ⚠  L3 first-miss ≈ L2 first-miss: L3 anchor may not be on SSD.")
    print(f"     Try increasing L3_FLOOD_N (current: {L3_FLOOD_N}).")
    if check_tmp_is_tmpfs():
        print("     Also: /tmp is tmpfs — SSD backend is actually in RAM.")

print("=" * 68)

# ─── Save results ─────────────────────────────────────────────────────────────
summary = {
    "model":            MODEL_SLUG,
    "server_url":       SERVER_URL,
    "anchor_chars":     ANCHOR_CHARS,
    "anchor_tokens_est": anchor_tok_est,
    "anchor_kv_mb":     round(anchor_kv_mb, 1),
    "gpu_kv_gb":        GPU_KV_GB,
    "dram_kv_gb":       DRAM_KV_GB,
    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
    "gpu_evict_n":      GPU_EVICT_N,
    "l3_flood_n":       L3_FLOOD_N,
    "repeat_n":         REPEAT_N,
    "cold_ttft_ms":       round(cold_ms,    1) if cold_ms    else None,
    "l1_baseline_ms":     round(l1_baseline, 1) if l1_baseline else None,
    "l2_first_miss_ms":   round(l2_miss_ms,  1) if l2_miss_ms  else None,
    "l3_first_miss_ms":   round(l3_miss_ms,  1) if l3_miss_ms  else None,
    "l2_overhead_ms":     l2_overhead,
    "l3_overhead_ms":     l3_overhead,
    "l3_vs_l2_ms":        l3_vs_l2,
    "l2_vs_recompute_ms": round(l2_miss_ms - cold_ms, 1) if (l2_miss_ms and cold_ms) else None,
    "l3_vs_recompute_ms": round(l3_miss_ms - cold_ms, 1) if (l3_miss_ms and cold_ms) else None,
}

out_dir  = _p(f"results/sglang_hicache/{MODEL_SLUG}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "kv_tier_latency.json")
with open(out_path, "w") as f:
    json.dump({"summary": summary, "phases": record}, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장: {out_path}")
