"""
SGLang KV Break-even Map Benchmark  (research.md §5.A — 논문 Figure 1)
======================================================================
(tool call duration d) × (context length L) 격자에서 최적 KV tier를 결정하는
break-even 맵을 실측 데이터로 생성한다.

방법론:
  sglang_kv_tier_latency.py의 4-phase 측정(COLD / L1 GPU / L2 DRAM / L3 SSD)을
  context 길이 축으로 sweep한다. 각 길이 L에서:

    Pre   : L3 anchor를 GPU+DRAM에 기록 (DRAM LRU 최고령 확보 — 원 스크립트와 동일 트릭)
    Phase0: COLD     — anchor 첫 요청 (cache miss = re-prefill 비용)
    Phase1: L1 GPU   — 즉시 재요청 (GPU radix hit = baseline)
    Phase2: L2 DRAM  — GPU flood로 anchor 축출 후 재요청 (first-miss = DRAM 복구)
    Phase3: L3 SSD   — DRAM 소량 overflow로 L3 anchor를 SSD로 밀어낸 후 재요청

  복구 패널티:
    penalty(tier, L) = first_miss(tier, L) − TTFT_L1(L)

  Break-even 맵 (prefetch 모델):
    tool duration d 동안 prefetch로 복구를 숨길 수 있는 가장 깊은 tier 선택:
      optimal(d, L) = deepest tier s.t. penalty(tier, L) ≤ d × PREFETCH_ALPHA
    EVICT는 cold(L) ≤ l3_first_miss(L) (재계산이 SSD fetch보다 빠름)일 때만 후보.

  No-prefetch 비교 맵:
    penalty가 resume 시 그대로 노출 → penalty ≤ SLO_PENALTY_MS인 가장 깊은 tier.
    (d와 무관 — prefetch의 가치를 보여주는 대조군)

Tier 매핑 주의 (research.md §5.A):
  실측 사다리 L1/L2/L3/cold = 제안 4-tier의 Tier2(P-HBM)/Tier3(DRAM)/Tier4(SSD)/EVICT.
  Tier1(D-HBM 유지)은 D 노드 append-prefill 메커니즘 구현 전에는 측정 불가 —
  L1(GPU radix hit)이 "HBM hit"의 하한 근사.

실험 조건:
  - 모든 요청이 동일한 P 노드에서 처리되어야 한다 (단일 P+D 라우터, port 8001)
    → 서버 시작 스크립트: scripts/sglang/start_1P_1D_breakeven.sh
  - hicache SSD 경로가 실제 디스크인지 확인: df -T /tmp  (tmpfs면 L3 측정 무의미)

예상 소요: 길이당 ~3–5분 (flood 지배적) × 5 lengths ≈ 20–30분

Hit/miss 가설 판별 (research.md §2.7):
  각 phase의 첫 요청(tier miss) 전후로 P 노드 /server_info의 캐시 관련 숫자 필드
  (hicache/storage/prefetch/hit/cache/evict/host 매칭)를 snapshot하고 차분을 출력·저장한다.
  → L2/L3 first-miss에서 카운터가 안 움직이면 "hit 미발생(재계산 fallback)",
    hit 카운터가 증가하는데도 TTFT ≈ cold면 "load 경로가 느림"으로 판별.
  (/get_server_info는 deprecated — /server_info 우선, 실패 시 fallback)

Usage:
  # 1) 서버 시작 (1P+1D + hicache + router 8001)
  bash scripts/sglang/start_1P_1D_breakeven.sh

  # 2) 벤치마크 실행
  SERVER_URL=http://127.0.0.1:8001 python benchmark/sglang_kv_breakeven_map.py

  # 단일 서버(비-PD) 대조 실험 — PD가 hit 경로를 깨는지 분리 확인:
  bash scripts/sglang/start_single_hicache.sh
  SERVER_URL=http://127.0.0.1:30000 P_NODE_URL=http://127.0.0.1:30000 \\
    python benchmark/sglang_kv_breakeven_map.py

  # 3-tier 모드 (research.md §2.9 — SSD 제외, D-HBM proxy 결합):
  #   1) 단일 서버 결과를 T1 proxy로 저장해 두고 (위 대조 실험 결과 JSON)
  #   2) PD(1p1d)에서 SSD phase 생략 + T1 ref 결합:
  SKIP_L3=1 T1_REF_JSON=results/sglang_hicache/Qwen3-14B/1server_kv_breakeven_map.json \\
    SERVER_URL=http://127.0.0.1:8001 python benchmark/sglang_kv_breakeven_map.py
  #   → kv_breakeven_map_3tier.{json,png} 추가 생성
  #   (T1=D-HBM 유지 proxy, T2=이 실행의 L1(P radix+전송), T3=이 실행의 L2(DRAM))

  # 길이/duration 격자 커스텀
  SWEEP_CHARS=2000,6000,15000 DURATIONS=0.5,1,5,10 \\
    SERVER_URL=http://127.0.0.1:8001 python benchmark/sglang_kv_breakeven_map.py

Environment variables:
  SERVER_URL       SGLang 라우터 주소           (default: http://127.0.0.1:8001)
  P_NODE_URL       P 노드 직접 주소 (stats용)    (default: http://127.0.0.1:30000)
  MODEL            모델 경로                     (default: /home/uhmturks/hf_models/Qwen3-14B)
  SWEEP_CHARS      context 길이 sweep (chars)    (default: 2000,6000,15000,30000,60000)
                   ≈ 0.7k / 2k / 5k / 10k / 20k tokens
  DURATIONS        tool duration 격자 (seconds)  (default: 0.1,0.2,0.5,1,2,5,10,30,60)
  PREFETCH_ALPHA   duration 중 prefetch에 쓸 수 있는 비율 (default: 1.0)
  SLO_PENALTY_MS   no-prefetch 맵의 가시적 TTFT 패널티 허용치 (default: 200)
  EVICT_CHARS      eviction 요청 텍스트 길이     (default: 12000 ≈ 4000 tokens)
  GPU_EVICT_N      GPU 축출용 요청 수            (default: 32)
  L3_FLOOD_N       DRAM overflow 요청 수         (default: 20)
  REPEAT_N         phase당 측정 반복 수          (default: 5)
  GPU_KV_GB        GPU KV 예산 GB                (default: 18.0)
  DRAM_KV_GB       DRAM KV 예산 GB               (default: 21.6)
  HICACHE_PATH     SGLang hicache SSD 경로       (default: /tmp/hicache)
  MAX_TOKENS_OUT   응답 최대 토큰 수             (default: 16)
  KV_LAYERS/KV_HEADS/KV_HEAD_DIM  모델 KV 형상 (default: Qwen3-14B = 40/8/128)
"""

import json
import os
import re
import statistics
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
SWEEP_CHARS    = [int(x) for x in os.environ.get("SWEEP_CHARS", "2000,6000,15000,30000,60000").split(",")]
DURATIONS_S    = [float(x) for x in os.environ.get("DURATIONS", "0.1,0.2,0.5,1,2,5,10,30,60").split(",")]
PREFETCH_ALPHA = float(os.environ.get("PREFETCH_ALPHA", "1.0"))
SLO_PENALTY_MS = float(os.environ.get("SLO_PENALTY_MS", "200"))
EVICT_CHARS    = int(os.environ.get("EVICT_CHARS",    "12000"))
GPU_EVICT_N    = int(os.environ.get("GPU_EVICT_N",    "32"))
L3_FLOOD_N     = int(os.environ.get("L3_FLOOD_N",     "20"))
REPEAT_N       = int(os.environ.get("REPEAT_N",       "5"))
GPU_KV_GB      = float(os.environ.get("GPU_KV_GB",    "18.0"))
DRAM_KV_GB     = float(os.environ.get("DRAM_KV_GB",   "21.6"))
HICACHE_PATH   = os.environ.get("HICACHE_PATH",       "/tmp/hicache")
MAX_TOKENS_OUT = int(os.environ.get("MAX_TOKENS_OUT", "16"))
# 3-tier 모드 (research.md §2.9): SSD tier 측정 생략 + D-HBM(T1) proxy 결합
SKIP_L3        = os.environ.get("SKIP_L3", "0") == "1"
T1_REF_JSON    = os.environ.get("T1_REF_JSON", "")

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
CHAT_URL   = f"{SERVER_URL.rstrip('/')}/v1/chat/completions"

# ─── Memory estimation ───────────────────────────────────────────────────────
# Qwen3-14B: 40 layers, 8 KV heads, 128 head_dim, BF16 (2 bytes)
KV_LAYERS    = int(os.environ.get("KV_LAYERS",    "40"))
KV_HEADS     = int(os.environ.get("KV_HEADS",     "8"))
KV_HEAD_DIM  = int(os.environ.get("KV_HEAD_DIM",  "128"))
KV_BYTES_PER_TOKEN = 2 * KV_LAYERS * KV_HEADS * KV_HEAD_DIM * 2  # BF16

CHARS_PER_TOKEN = 3.0  # rough: 1 token ≈ 3 chars

gpu_max_tokens  = int(GPU_KV_GB  * (1024 ** 3) / KV_BYTES_PER_TOKEN)
dram_max_tokens = int(DRAM_KV_GB * (1024 ** 3) / KV_BYTES_PER_TOKEN)
evict_tok_est   = int(EVICT_CHARS / CHARS_PER_TOKEN)

# ─── Startup banner ─────────────────────────────────────────────────────────
print("=" * 68)
print("SGLang KV Break-even Map Benchmark")
print("=" * 68)
print(f"  Model        : {MODEL_SLUG}")
print(f"  Server       : {CHAT_URL}")
print(f"  P-node       : {P_NODE_URL} (for cache stats)")
print()
print(f"  KV bytes/tok : {KV_BYTES_PER_TOKEN:,} B  ({KV_BYTES_PER_TOKEN/1024:.1f} KB)")
print(f"  GPU KV       : {GPU_KV_GB:.1f} GB → ~{gpu_max_tokens:,} tokens")
print(f"  DRAM KV      : {DRAM_KV_GB:.1f} GB → ~{dram_max_tokens:,} tokens")
print()
print(f"  Length sweep : {SWEEP_CHARS} chars"
      f"  (≈ {[int(c / CHARS_PER_TOKEN) for c in SWEEP_CHARS]} tokens)")
print(f"  Durations    : {DURATIONS_S} s")
print(f"  Prefetch α   : {PREFETCH_ALPHA}   SLO penalty: {SLO_PENALTY_MS:.0f} ms")
print(f"  GPU_EVICT_N  : {GPU_EVICT_N}   L3_FLOOD_N: {L3_FLOOD_N}   REPEAT_N: {REPEAT_N}")
print(f"  HiCache path : {HICACHE_PATH}")
if SKIP_L3:
    print("  SKIP_L3=1    : SSD(L3) phase 생략 — 3-tier 모드")
if T1_REF_JSON:
    print(f"  T1_REF_JSON  : {T1_REF_JSON} (D-HBM Tier-1 proxy = 단일 서버 L1 baseline)")
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
def _repeat_text(unit: str, n: int) -> str:
    return (unit * (n // len(unit) + 1))[:n]

def anchor_messages(tag: str, chars: int) -> list:
    """Unique prefix per (tag, length) — no shared prefix across iterations."""
    unit = f"{tag}{chars:06d}word "
    return [
        {"role": "system", "content": f"{tag.upper()}{chars} context: {_repeat_text(unit, chars)}"},
        {"role": "user",   "content": "Summarize in one word."},
    ]

_evict_seed = 1000  # global counter — flood prefixes unique across the whole run
def evict_messages() -> list:
    global _evict_seed
    _evict_seed += 1
    unit = f"evict{_evict_seed:06d}x "
    return [
        {"role": "system", "content": f"EVICT context: {_repeat_text(unit, EVICT_CHARS)}"},
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
        resp = requests.post(CHAT_URL, json=payload, stream=True, timeout=600)
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

def fetch_server_info() -> dict:
    """P 노드 server info. /server_info 우선 (/get_server_info는 deprecated)."""
    for ep in ("/server_info", "/get_server_info"):
        try:
            r = requests.get(f"{P_NODE_URL}{ep}", timeout=5)
            if r.ok:
                return r.json()
        except Exception:
            continue
    return {}

def print_cache_stats(label: str):
    info = fetch_server_info()
    if info.get("token_usage") is not None:
        print(f"    [cache {label}] token_usage={info['token_usage']:.3f}  "
              f"cached={info.get('num_cached_tokens')}/"
              f"{info.get('num_total_tokens')}")

# ─── HiCache counter snapshot/diff (가설 판별용) ──────────────────────────────
# server_info의 숫자 필드 중 캐시 관련 키만 추출해 요청 전후 차분을 본다.
# 필드 이름이 SGLang 버전마다 달라서 정확한 키를 가정하지 않고 regex로 거른다.
CACHE_KEY_RE = re.compile(r"hicache|storage|prefetch|hit|cache|evict|host", re.I)

def _flatten_numeric(obj, prefix: str = "", out: dict | None = None) -> dict:
    if out is None:
        out = {}
    if isinstance(obj, dict):
        for k, v in obj.items():
            _flatten_numeric(v, f"{prefix}{k}.", out)
    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            _flatten_numeric(v, f"{prefix}{i}.", out)
    elif isinstance(obj, bool):
        pass
    elif isinstance(obj, (int, float)):
        out[prefix[:-1]] = obj
    return out

def cache_snapshot() -> dict:
    flat = _flatten_numeric(fetch_server_info())
    return {k: v for k, v in flat.items() if CACHE_KEY_RE.search(k)}

def counter_diff(before: dict, after: dict) -> dict:
    """변화한 캐시 카운터만 {key: [before, after]}로 반환."""
    return {k: [before.get(k), after[k]]
            for k in after if after[k] != before.get(k)}

def print_counter_diff(label: str, diff: dict):
    if not diff:
        print(f"    [counters {label}] 변화한 캐시 카운터 없음"
              " (hicache hit/load가 안 일어났거나 server_info에 미노출)")
        return
    for k in sorted(diff)[:15]:
        b, a = diff[k]
        print(f"    [counters {label}] {k}: {b} → {a}")
    if len(diff) > 15:
        print(f"    [counters {label}] ... ({len(diff) - 15} more, JSON에 전체 저장)")

# ─── Phase runner ────────────────────────────────────────────────────────────
def measure_phase(label: str, messages: list) -> tuple[list[float], float | None, dict]:
    """Run REPEAT_N requests; return (ttft_list_ms, first_miss_ms, counter_diff).

    counter_diff = 첫 요청(tier miss) 전후 캐시 카운터 차분 — hit/miss 가설 판별용.
    """
    ttfts = []
    first_miss: float | None = None
    miss_counters: dict = {}
    for i in range(REPEAT_N):
        snap_before = cache_snapshot() if i == 0 else None
        ttft, e2e = do_request(messages, label=f"{label}[{i}]")
        if i == 0:
            miss_counters = counter_diff(snap_before, cache_snapshot())
        if ttft is not None:
            ms = ttft * 1000
            ttfts.append(ms)
            if first_miss is None:
                first_miss = ms
            marker = " ← tier miss" if i == 0 else ""
            print(f"    [{label} {i+1}/{REPEAT_N}]  TTFT={ms:7.1f} ms"
                  f"  e2e={e2e*1000:.1f} ms{marker}")
            if i == 0:
                print_counter_diff(f"{label}-miss", miss_counters)
        else:
            print(f"    [{label} {i+1}/{REPEAT_N}]  FAILED")
    return ttfts, first_miss, miss_counters

def flood(desc: str, n: int):
    for _ in tqdm(range(n), desc=desc, leave=False):
        do_request(evict_messages(), label="flood")

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

# ─── Per-length 4-phase measurement ──────────────────────────────────────────
def measure_length(chars: int) -> dict:
    """Run the full COLD/L1/L2/L3 ladder for one context length. Returns metrics."""
    tok_est = int(chars / CHARS_PER_TOKEN)
    kv_mb   = tok_est * KV_BYTES_PER_TOKEN / 1024 / 1024
    print("\n" + "─" * 68)
    print(f"LENGTH {chars:,} chars ≈ {tok_est:,} tokens  (KV ≈ {kv_mb:.0f} MB)")
    print("─" * 68)

    anchor    = anchor_messages("anchor", chars)
    l3_anchor = anchor_messages("late",   chars)   # distinct prefix, pre-aged for L3

    if not SKIP_L3:
        # Pre-phase: L3 anchor into GPU+DRAM — its DRAM LRU becomes the oldest of
        # this iteration, so the small Phase-3 flood evicts it to SSD first.
        print("  [Pre] L3 anchor → GPU+DRAM (oldest LRU of this iteration)")
        do_request(l3_anchor, "L3-PRE")

    # Phase 0: COLD (unique prefix → guaranteed miss)
    print("  [Phase 0] COLD")
    snap_before = cache_snapshot()
    ttft_cold, _ = do_request(anchor, "COLD")
    cold_counters = counter_diff(snap_before, cache_snapshot())
    cold_ms = ttft_cold * 1000 if ttft_cold else None
    print(f"    COLD TTFT = {cold_ms:.1f} ms" if cold_ms else "    COLD FAILED")
    print_counter_diff("COLD", cold_counters)

    # Phase 1: L1 GPU hit
    print("  [Phase 1] L1 GPU")
    ttfts_l1, _, _ = measure_phase("L1", anchor)
    l1_baseline = min(ttfts_l1) if ttfts_l1 else None

    # Phase 2: evict anchor from GPU → DRAM hit
    print(f"  [Phase 2] L2 DRAM — GPU flood ({GPU_EVICT_N} reqs, "
          f"{GPU_EVICT_N*evict_tok_est/gpu_max_tokens*100:.0f}% of GPU KV)")
    flood("GPU evict", GPU_EVICT_N)
    print_cache_stats("after-GPU-flood")
    ttfts_l2, l2_miss_ms, l2_counters = measure_phase("L2", anchor)

    # Phase 3: overflow DRAM → pre-aged L3 anchor lands on SSD
    if SKIP_L3:
        ttfts_l3, l3_miss_ms, l3_counters = [], None, {}
    else:
        print(f"  [Phase 3] L3 SSD — DRAM overflow ({L3_FLOOD_N} reqs)")
        flood("L3 flood", L3_FLOOD_N)
        print("    Waiting 5s for async SSD writes ...")
        time.sleep(5)
        print_cache_stats("after-L3-flood")
        ttfts_l3, l3_miss_ms, l3_counters = measure_phase("L3", l3_anchor)

    # Sanity warnings (same thresholds as sglang_kv_tier_latency.py)
    if l1_baseline and l2_miss_ms and l2_miss_ms < l1_baseline + 50:
        print("    ⚠  L2 first-miss ≈ L1: anchor may not have left GPU. "
              f"Increase GPU_EVICT_N (current {GPU_EVICT_N}).")
    if l2_miss_ms and l3_miss_ms and l3_miss_ms < l2_miss_ms + 50:
        print("    ⚠  L3 first-miss ≈ L2: L3 anchor may not be on SSD. "
              f"Increase L3_FLOOD_N (current {L3_FLOOD_N}).")

    pen = lambda miss: round(miss - l1_baseline, 1) if (miss and l1_baseline) else None
    return {
        "chars":            chars,
        "tokens_est":       tok_est,
        "kv_mb":            round(kv_mb, 1),
        "cold_ttft_ms":     round(cold_ms, 1)      if cold_ms      else None,
        "l1_baseline_ms":   round(l1_baseline, 1)  if l1_baseline  else None,
        "l2_first_miss_ms": round(l2_miss_ms, 1)   if l2_miss_ms   else None,
        "l3_first_miss_ms": round(l3_miss_ms, 1)   if l3_miss_ms   else None,
        "penalty_l2_ms":    pen(l2_miss_ms),
        "penalty_l3_ms":    pen(l3_miss_ms),
        "penalty_evict_ms": pen(cold_ms),
        "phases": {
            "l1": stats_dict(ttfts_l1),
            "l2": stats_dict(ttfts_l2),
            "l3": stats_dict(ttfts_l3),
        },
        # 첫 요청(tier miss) 전후 캐시 카운터 차분 — hit 미발생 vs 느린 load 판별 근거
        "counters_first_miss": {
            "cold": cold_counters,
            "l2":   l2_counters,
            "l3":   l3_counters,
        },
    }

# ─── Break-even map ──────────────────────────────────────────────────────────
# Tier order: shallow → deep. Deeper = cheaper memory, larger recovery penalty.
TIER_GPU, TIER_RAM, TIER_SSD, TIER_EVT, TIER_UNK = "GPU", "RAM", "SSD", "EVT", "?"

def _evict_candidate(m: dict) -> bool:
    """EVICT는 가장 깊은 tier보다 재계산이 빠를 때만 후보.
    L3 미측정(SKIP_L3)이면 L2와 비교, 둘 다 없으면 항상 후보."""
    if m["cold_ttft_ms"] is None:
        return False
    deepest = m["l3_first_miss_ms"] if m["l3_first_miss_ms"] is not None \
        else m["l2_first_miss_ms"]
    return deepest is None or m["cold_ttft_ms"] <= deepest

def _deepest_within(m: dict, budget_ms: float) -> str:
    choice = TIER_GPU  # always feasible (penalty 0)
    if m["penalty_l2_ms"] is not None and m["penalty_l2_ms"] <= budget_ms:
        choice = TIER_RAM
    if m["penalty_l3_ms"] is not None and m["penalty_l3_ms"] <= budget_ms:
        choice = TIER_SSD
    if (m["penalty_evict_ms"] is not None
            and m["penalty_evict_ms"] <= budget_ms
            and _evict_candidate(m)):
        choice = TIER_EVT
    return choice

def optimal_tier_prefetch(m: dict, d_s: float) -> str:
    """Deepest tier whose recovery penalty hides within d (prefetch model)."""
    if m["l1_baseline_ms"] is None:
        return TIER_UNK
    return _deepest_within(m, d_s * 1000 * PREFETCH_ALPHA)

def optimal_tier_no_prefetch(m: dict) -> str:
    """No prefetch: penalty is fully visible at resume → bound by SLO budget."""
    if m["l1_baseline_ms"] is None:
        return TIER_UNK
    return _deepest_within(m, SLO_PENALTY_MS)

def print_map(rows: list[dict], map_grid: list[list[str]]):
    header = "  tokens \\ d(s) │" + "".join(f"{d:>7g}" for d in DURATIONS_S)
    print(header)
    print("  " + "─" * (len(header) - 2))
    for m, row in zip(rows, map_grid):
        print(f"  {m['tokens_est']:>13,} │" + "".join(f"{t:>7}" for t in row))

def save_heatmap(rows: list[dict], map_grid: list[list[str]], out_png: str,
                 tier_order: list[str], legend: str):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.colors import ListedColormap
    except ImportError:
        print("  (matplotlib 미설치 — PNG 생략)")
        return
    tier_idx = {t: i for i, t in enumerate(tier_order)}
    tier_idx[TIER_UNK] = len(tier_order)
    data = [[tier_idx[t] for t in row] for row in map_grid]
    palette = ["#d73027", "#fdae61", "#74add1", "#4575b4", "#cccccc"]
    cmap = ListedColormap(palette[:len(tier_order)] + [palette[-1]])
    fig, ax = plt.subplots(figsize=(1.2 * len(DURATIONS_S) + 2, 0.8 * len(rows) + 2))
    ax.imshow(data, cmap=cmap, vmin=0, vmax=len(tier_order), aspect="auto")
    ax.set_xticks(range(len(DURATIONS_S)), [f"{d:g}" for d in DURATIONS_S])
    ax.set_yticks(range(len(rows)), [f"{m['tokens_est']:,}" for m in rows])
    ax.set_xlabel("Tool call duration (s)")
    ax.set_ylabel("Context length (tokens)")
    ax.set_title(f"Optimal KV tier — {MODEL_SLUG} (prefetch α={PREFETCH_ALPHA})\n{legend}")
    for i, row in enumerate(map_grid):
        for j, t in enumerate(row):
            ax.text(j, i, t, ha="center", va="center", fontsize=9, color="white")
    fig.tight_layout()
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Heatmap 저장: {out_png}")

# ─── Run sweep ───────────────────────────────────────────────────────────────
results = [measure_length(c) for c in SWEEP_CHARS]

# ─── Summary: latency ladder per length ──────────────────────────────────────
print("\n" + "=" * 68)
print("LATENCY LADDER  (first-miss = actual tier load)")
print("=" * 68)
print(f"  {'tokens':>8} {'KV MB':>7} {'cold':>9} {'L1 GPU':>9} {'L2 RAM':>9} {'L3 SSD':>9}"
      f" {'pen.L2':>8} {'pen.L3':>8}")
for m in results:
    f = lambda v: f"{v:>9.1f}" if v is not None else f"{'—':>9}"
    g = lambda v: f"{v:>8.1f}" if v is not None else f"{'—':>8}"
    print(f"  {m['tokens_est']:>8,} {m['kv_mb']:>7.0f} {f(m['cold_ttft_ms'])}"
          f" {f(m['l1_baseline_ms'])} {f(m['l2_first_miss_ms'])} {f(m['l3_first_miss_ms'])}"
          f" {g(m['penalty_l2_ms'])} {g(m['penalty_l3_ms'])}")

# Bandwidth estimates from penalties
print("\n  Estimated transfer bandwidth (KV size / penalty):")
for m in results:
    parts = []
    if m["penalty_l2_ms"] and m["penalty_l2_ms"] > 0:
        parts.append(f"DRAM→GPU {m['kv_mb'] / (m['penalty_l2_ms'] / 1000):,.0f} MB/s")
    if (m["penalty_l3_ms"] and m["penalty_l2_ms"]
            and m["penalty_l3_ms"] > m["penalty_l2_ms"]):
        ssd_ms = m["penalty_l3_ms"] - m["penalty_l2_ms"]
        parts.append(f"SSD→DRAM {m['kv_mb'] / (ssd_ms / 1000):,.0f} MB/s")
    if parts:
        print(f"    {m['tokens_est']:>8,} tok: " + "   ".join(parts))

# ─── Break-even maps ─────────────────────────────────────────────────────────
map_prefetch = [[optimal_tier_prefetch(m, d) for d in DURATIONS_S] for m in results]
map_no_pf    = [optimal_tier_no_prefetch(m) for m in results]

print("\n" + "=" * 68)
print(f"BREAK-EVEN MAP — WITH prefetch (α={PREFETCH_ALPHA})")
print("  GPU=keep in P-HBM(≈Tier1/2)  RAM=DRAM(Tier3)  SSD(Tier4)  EVT=evict+re-prefill")
print("=" * 68)
print_map(results, map_prefetch)

print("\n" + "-" * 68)
print(f"NO-prefetch 대조군 (visible penalty ≤ SLO {SLO_PENALTY_MS:.0f} ms, duration 무관):")
for m, t in zip(results, map_no_pf):
    print(f"  {m['tokens_est']:>8,} tok → {t}")
print("-" * 68)

# ─── Save results ─────────────────────────────────────────────────────────────
out_dir = _p(f"results/sglang_hicache/{MODEL_SLUG}")
os.makedirs(out_dir, exist_ok=True)

summary = {
    "model":              MODEL_SLUG,
    "server_url":         SERVER_URL,
    "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
    "sweep_chars":        SWEEP_CHARS,
    "durations_s":        DURATIONS_S,
    "prefetch_alpha":     PREFETCH_ALPHA,
    "slo_penalty_ms":     SLO_PENALTY_MS,
    "gpu_evict_n":        GPU_EVICT_N,
    "l3_flood_n":         L3_FLOOD_N,
    "repeat_n":           REPEAT_N,
    "tier_legend": {
        TIER_GPU: "keep in GPU/P-HBM (Tier 1/2)",
        TIER_RAM: "CPU DRAM (Tier 3)",
        TIER_SSD: "SSD (Tier 4)",
        TIER_EVT: "evict + re-prefill",
    },
}
out_json = os.path.join(out_dir, "kv_breakeven_map.json")
with open(out_json, "w") as f:
    json.dump({
        "summary":              summary,
        "lengths":              results,
        "breakeven_map":        map_prefetch,
        "breakeven_no_prefetch": map_no_pf,
    }, f, indent=2, ensure_ascii=False)
print(f"\n결과 저장: {out_json}")

save_heatmap(results, map_prefetch, os.path.join(out_dir, "kv_breakeven_map.png"),
             tier_order=[TIER_GPU, TIER_RAM, TIER_SSD, TIER_EVT],
             legend="GPU=P-HBM radix  RAM=DRAM  SSD  EVT=re-prefill")

# ─── 3-tier map: T1(D-HBM proxy) / T2(P-HBM) / RAM(DRAM) / EVT ───────────────
# research.md §2.9: 이 실행(PD)의 L1 = P radix hit + Mooncake 전송 = Tier 2,
# 단일 서버(비-PD) 실행의 L1 = 전송·재계산 없는 hit = Tier 1(D-HBM 유지)의 proxy.
# T1_REF_JSON에 단일 서버 결과를 주면 두 측정을 결합해 3-tier 맵을 만든다.
TIER_T1, TIER_T2 = "T1", "T2"

def build_3tier_rows(results: list[dict], ref_path: str) -> list[dict]:
    ref = json.load(open(ref_path))
    ref_by_chars = {m["chars"]: m for m in ref.get("lengths", [])}
    rows = []
    for m in results:
        r = ref_by_chars.get(m["chars"])
        if not (r and r.get("l1_baseline_ms") is not None
                and m["l1_baseline_ms"] is not None):
            print(f"  (3-tier: {m['chars']} chars — T1 ref 없음/측정 실패, 생략)")
            continue
        t1 = r["l1_baseline_ms"]
        pen = lambda v: round(v - t1, 1) if v is not None else None
        rows.append({
            "chars":         m["chars"],
            "tokens_est":    m["tokens_est"],
            "kv_mb":         m["kv_mb"],
            "t1_ms":         t1,                       # D-HBM 유지 proxy (1server L1)
            "t2_ms":         m["l1_baseline_ms"],      # P-HBM radix hit (+transfer)
            "t3_ms":         m["l2_first_miss_ms"],    # DRAM 복구
            "cold_ttft_ms":  m["cold_ttft_ms"],
            "penalty_t2_ms":    pen(m["l1_baseline_ms"]),
            "penalty_t3_ms":    pen(m["l2_first_miss_ms"]),
            "penalty_evict_ms": pen(m["cold_ttft_ms"]),
        })
    return rows

def optimal_3tier(row: dict, budget_ms: float) -> str:
    choice = TIER_T1
    if row["penalty_t2_ms"] is not None and row["penalty_t2_ms"] <= budget_ms:
        choice = TIER_T2
    if row["penalty_t3_ms"] is not None and row["penalty_t3_ms"] <= budget_ms:
        choice = TIER_RAM
    evt_ok = (row["cold_ttft_ms"] is not None
              and (row["t3_ms"] is None or row["cold_ttft_ms"] <= row["t3_ms"]))
    if (row["penalty_evict_ms"] is not None
            and row["penalty_evict_ms"] <= budget_ms and evt_ok):
        choice = TIER_EVT
    return choice

if T1_REF_JSON:
    rows3 = build_3tier_rows(results, T1_REF_JSON)
    if rows3:
        map3 = [[optimal_3tier(r, d * 1000 * PREFETCH_ALPHA) for d in DURATIONS_S]
                for r in rows3]
        print("\n" + "=" * 68)
        print(f"3-TIER BREAK-EVEN MAP (α={PREFETCH_ALPHA})")
        print("  T1=D-HBM 유지(proxy)  T2=P-HBM radix(+transfer)  RAM=DRAM  EVT=re-prefill")
        print("=" * 68)
        print(f"  {'tokens':>8} {'T1':>9} {'T2':>9} {'T3 RAM':>9} {'cold':>9}"
              f" {'pen.T2':>8} {'pen.T3':>8}")
        for r in rows3:
            f = lambda v: f"{v:>9.1f}" if v is not None else f"{'—':>9}"
            g = lambda v: f"{v:>8.1f}" if v is not None else f"{'—':>8}"
            print(f"  {r['tokens_est']:>8,} {f(r['t1_ms'])} {f(r['t2_ms'])}"
                  f" {f(r['t3_ms'])} {f(r['cold_ttft_ms'])}"
                  f" {g(r['penalty_t2_ms'])} {g(r['penalty_t3_ms'])}")
        print()
        print_map(rows3, map3)

        out3 = os.path.join(out_dir, "kv_breakeven_map_3tier.json")
        with open(out3, "w") as f:
            json.dump({
                "summary": {**summary, "t1_ref_json": T1_REF_JSON,
                            "tier_legend": {
                                TIER_T1:  "keep in D-HBM (Tier 1, proxy = 1server L1)",
                                TIER_T2:  "P-HBM radix (Tier 2, incl. P→D transfer)",
                                TIER_RAM: "CPU DRAM (Tier 3)",
                                TIER_EVT: "evict + re-prefill",
                            }},
                "lengths":       rows3,
                "breakeven_map": map3,
            }, f, indent=2, ensure_ascii=False)
        print(f"\n3-tier 결과 저장: {out3}")
        save_heatmap(rows3, map3, os.path.join(out_dir, "kv_breakeven_map_3tier.png"),
                     tier_order=[TIER_T1, TIER_T2, TIER_RAM, TIER_EVT],
                     legend="T1=D-HBM keep  T2=P-HBM radix  RAM=DRAM  EVT=re-prefill")
