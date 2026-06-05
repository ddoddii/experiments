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

Phase 3 디자인 (disk space):
  write-through 정책으로 모든 KV가 GPU와 동시에 DRAM에도 기록된다.
  Phase 3a (GPU 재축출) 중 DRAM이 가득 차면 Phase 2 eviction 데이터가
  /tmp/hicache (SSD)에 써져 수십 GB를 차지한다.
  Phase 3b (DRAM 축출) 전에 hicache를 정리하여 anchor→SSD 기록에 필요한
  공간을 확보한다 (anchor KV는 DRAM에 있으므로 정리해도 안전).

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
  DRAM_EVICT_N     DRAM 축출용 추가 eviction 수  (default: 30)
                   Phase 3a 후 DRAM은 가득 참; anchor를 밀어내려면
                   Phase2 잔여 + anchor 토큰 수 이상이 필요.
                   실제 GPU KV 예산에 따라 필요량이 달라지므로 30이 안전한 기본값.
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
DRAM_EVICT_N   = int(os.environ.get("DRAM_EVICT_N",   "30"))     # DRAM 채우기 추가 요청 수
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
print(f"  HiCache path : {HICACHE_PATH}  (cleaned before Phase 3b)")
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
    """Fixed deterministic English text. Always identical."""
    unit = "the quick brown fox jumps over the lazy dog "
    return (unit * (n // len(unit) + 1))[:n]

def _evict_text(seed: int, n: int) -> str:
    """Unique per seed; starts with EVICT{seed} → no shared prefix with anchor."""
    unit = f"evict{seed:06d}x "
    return (unit * (n // len(unit) + 1))[:n]

ANCHOR_TEXT   = _anchor_text(ANCHOR_CHARS)
ANCHOR_MESSAGES = [
    {"role": "system", "content": f"ANCHOR context: {ANCHOR_TEXT}"},
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
def measure_phase(label: str) -> list[float]:
    """Run REPEAT_N anchor requests; return TTFT list in ms."""
    ttfts = []
    for i in range(REPEAT_N):
        ttft, e2e = do_request(ANCHOR_MESSAGES, label=f"{label}[{i}]")
        ms = ttft * 1000 if ttft else None
        e2e_ms = e2e * 1000 if e2e else None
        if ms is not None:
            ttfts.append(ms)
            print(f"    [{label} {i+1}/{REPEAT_N}]  TTFT={ms:7.1f} ms  e2e={e2e_ms:.1f} ms")
        else:
            print(f"    [{label} {i+1}/{REPEAT_N}]  FAILED")
    return ttfts

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

# Phase 0: COLD
print("\n[Phase 0] COLD — no cache (initial prefill)")
ttft_cold, _ = do_request(ANCHOR_MESSAGES, "COLD")
cold_ms = ttft_cold * 1000 if ttft_cold else None
print(f"  COLD TTFT = {cold_ms:.1f} ms" if cold_ms else "  COLD FAILED")
record["cold"] = {"ttft_ms": round(cold_ms, 1) if cold_ms else None}

# Phase 1: L1 GPU
print("\n[Phase 1] L1 GPU — KV in GPU (immediate re-request)")
ttfts_l1 = measure_phase("L1")
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
ttfts_l2 = measure_phase("L2")
print(f"  → {fmt_stats(ttfts_l2)}")
record["l2_dram"] = {
    **stats_dict(ttfts_l2),
    "gpu_evict_n": GPU_EVICT_N,
    "evict_tokens": GPU_EVICT_N * evict_tok_est,
}

# Phase 3: L3 SSD
# After L2 measurement, anchor is back in GPU+DRAM (loaded from DRAM, LRU refreshed).
# Step 3a: flood GPU with fresh requests → anchor evicted GPU→DRAM (already there).
#          This also overflows DRAM: Phase 2 evictions (oldest) spill to SSD.
# Clean:   /tmp/hicache holds the Phase 2 overflow (can be 10-20 GB).
#          Anchor is still safely in DRAM. Cleaning frees disk for Step 3b.
# Step 3b: a few more requests overflow DRAM → oldest (Phase2 leftovers + anchor) → SSD.
#          Only ~3 requests needed; default 8 gives a comfortable margin.
total_l3_flood = GPU_EVICT_N + DRAM_EVICT_N
print(f"\n[Phase 3] L3 SSD — evicting anchor from GPU+DRAM")
print(f"  Step 3a: re-fill GPU ({GPU_EVICT_N} requests)  →  anchor GPU→DRAM, Phase2 data→SSD")
flood("L3 GPU re-evict", range(3000, 3000 + GPU_EVICT_N))

# Clean hicache: Phase 3a flushed Phase 2 eviction data to SSD (can be ~10-20 GB).
# Anchor is in DRAM (not SSD), so this is safe.
free_before = free_disk_gb(HICACHE_PATH)
print(f"  Disk free before hicache clean: {free_before:.1f} GB")
clean_hicache("freeing disk space before anchor DRAM→SSD eviction")
free_after = free_disk_gb(HICACHE_PATH)
print(f"  Disk free after  hicache clean: {free_after:.1f} GB")

print(f"  Step 3b: overflow DRAM ({DRAM_EVICT_N} requests)  →  anchor DRAM→SSD")
flood("L3 DRAM evict", range(4000, 4000 + DRAM_EVICT_N))

print("  Waiting 5s for async SSD writes to complete ...")
time.sleep(5)
print_cache_stats("after-DRAM-flood")

print("  Measuring anchor TTFT (expected: L3 SSD hit) ...")
ttfts_l3 = measure_phase("L3")
print(f"  → {fmt_stats(ttfts_l3)}")
record["l3_ssd"] = {
    **stats_dict(ttfts_l3),
    "total_evict_n": total_l3_flood,
    "evict_tokens":  total_l3_flood * evict_tok_est,
}

# ─── Summary ─────────────────────────────────────────────────────────────────
l1_mean  = statistics.mean(ttfts_l1) if ttfts_l1 else None
l2_mean  = statistics.mean(ttfts_l2) if ttfts_l2 else None
l3_mean  = statistics.mean(ttfts_l3) if ttfts_l3 else None

l2_overhead = round(l2_mean - l1_mean, 1) if (l1_mean and l2_mean) else None
l3_overhead = round(l3_mean - l1_mean, 1) if (l1_mean and l3_mean) else None
l3_vs_l2    = round(l3_mean - l2_mean, 1) if (l2_mean and l3_mean) else None

print("\n" + "=" * 68)
print("RESULTS SUMMARY")
print("=" * 68)
print(f"  Anchor prefix : {ANCHOR_CHARS:,} chars ≈ {anchor_tok_est:,} tokens"
      f"  ({anchor_tok_est * KV_BYTES_PER_TOKEN / 1024**2:.0f} MB KV)")
print()
print(f"  Phase 0 COLD  : {cold_ms:.1f} ms" if cold_ms else "  Phase 0 COLD  : FAILED")
print(f"  Phase 1 L1 GPU: {fmt_stats(ttfts_l1)}")
print(f"  Phase 2 L2 RAM: {fmt_stats(ttfts_l2)}")
print(f"  Phase 3 L3 SSD: {fmt_stats(ttfts_l3)}")
print()
if l2_overhead is not None:
    print(f"  L2 vs L1 overhead : +{l2_overhead:.1f} ms  (DRAM→GPU fetch)")
if l3_vs_l2 is not None:
    print(f"  L3 vs L2 overhead : +{l3_vs_l2:.1f} ms  (SSD→DRAM additional)")
if l3_overhead is not None:
    print(f"  L3 vs L1 overhead : +{l3_overhead:.1f} ms  (total SSD→DRAM→GPU fetch)")
print()

anchor_kv_mb = anchor_tok_est * KV_BYTES_PER_TOKEN / 1024 / 1024
if l2_overhead and l2_overhead > 0:
    dram_bw = anchor_kv_mb / (l2_overhead / 1000)
    print(f"  Est. DRAM→GPU bandwidth : {dram_bw:.0f} MB/s  (PCIe limit ~32,000 MB/s)")
if l3_vs_l2 and l3_vs_l2 > 0:
    ssd_bw = anchor_kv_mb / (l3_vs_l2 / 1000)
    print(f"  Est. SSD→DRAM bandwidth : {ssd_bw:.0f} MB/s  (NVMe limit ~5,000 MB/s)")

# Sanity check: warn if L2 ≈ L1 (anchor may not have been evicted)
if l1_mean and l2_mean and abs(l2_mean - l1_mean) < 20:
    print()
    print("  ⚠  L2 TTFT ≈ L1 TTFT: anchor may not have been evicted from GPU.")
    print(f"     Try increasing GPU_EVICT_N (current: {GPU_EVICT_N})")

if l2_mean and l3_mean and abs(l3_mean - l2_mean) < 100:
    print()
    print("  ⚠  L3 TTFT ≈ L2 TTFT: anchor may not have been evicted from DRAM.")
    print(f"     Try increasing DRAM_EVICT_N (current: {DRAM_EVICT_N}).")
    print(f"     Verify actual DRAM token budget:")
    print(f"       curl -s http://127.0.0.1:30000/get_server_info | python3 -m json.tool | grep num_total_tokens")
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
    "dram_evict_n":     DRAM_EVICT_N,
    "repeat_n":         REPEAT_N,
    "cold_ttft_ms":     round(cold_ms, 1) if cold_ms else None,
    "l1_mean_ms":       round(l1_mean,  1) if l1_mean  else None,
    "l2_mean_ms":       round(l2_mean,  1) if l2_mean  else None,
    "l3_mean_ms":       round(l3_mean,  1) if l3_mean  else None,
    "l2_overhead_ms":   l2_overhead,
    "l3_overhead_ms":   l3_overhead,
    "l3_vs_l2_ms":      l3_vs_l2,
}

out_dir  = _p(f"results/sglang_hicache/{MODEL_SLUG}")
os.makedirs(out_dir, exist_ok=True)
out_path = os.path.join(out_dir, "kv_tier_latency.json")
with open(out_path, "w") as f:
    json.dump({"summary": summary, "phases": record}, f, indent=2, ensure_ascii=False)

print(f"\n결과 저장: {out_path}")
