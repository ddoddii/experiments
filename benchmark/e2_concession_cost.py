"""
E2: P-radix vs D-HBM concession cost asymmetry   — research.md §2.14 E2
============================================================================
"P-HBM을 Tier2로 쓰려고 P가 자기 radix를 evict하는 비용 < D가 세션 KV를 evict하는 비용"
을 직접 측정해 §2.12 correlated-demand 단서(C=16에서 P 23%로 상승)를 닫는다.

────────────────────────────────────────────────────────────────────────────
설계: "양보 비용" = 그 tier를 비웠을 때 다음에 그 KV가 필요해지는 비용 (= recover TTFT)
────────────────────────────────────────────────────────────────────────────
하나의 anchor prefix에 대해 3 상태를 측정 (context 길이별 sweep):

  warm       : anchor를 즉시 재요청 → GPU L1 hit (복원 없음, 기준선)
  P_restore  : GPU pool을 flood해 anchor를 GPU radix에서 evict (write_through라
               host 복제본은 유지) → 재요청 → host→GPU 복원 비용
  cold       : fresh anchor 첫 요청 → 어디에도 없음 → full re-prefill

  D 양보 비용 = cold − warm     (D가 세션 KV evict → 다음 turn full re-prefill)
  P 양보 비용 = P_restore − warm (P가 radix evict → host에서 복원, 복원 작동 시 저렴)
  비대칭     = (D 양보) / (P 양보)   ≫1 이면 "P 양보가 훨씬 싸다"

⚠ §2.8에서 host(L2) first-miss ≈ cold로 측정됨 → 현재 hicache 복원 경로가 깨져
  P_restore ≈ cold일 수 있음(= 가설 거짓). 이 벤치마크는 그걸 *가정 않고 측정*한다.
  판정:
    P_restore ≪ cold  → host 복원 작동 → "P 양보 거의 공짜" 성립
    P_restore ≈ cold  → host 복원 깨짐 → recovery-cost 비대칭 없음.
                        → 비대칭 논거를 opportunity-cost로 전환(§2.12 P idle + §2.14 hang):
                          P 양보는 active decode 방해 안 함(future prefix-miss일 뿐),
                          D 양보는 active 세션 재계산 + hang 유발 → robust한 비대칭.

  L2-revival 확인: 서버를 page_first + ratio↑로 재시작해 다시 측정(§2.9 step2):
    HICACHE_MEM_LAYOUT=page_first HICACHE_RATIO=3.0 bash scripts/sglang/start_1P_1D_breakeven.sh
    RUN_TAG=revival python benchmark/e2_concession_cost.py

실험 조건: 모든 요청이 같은 P 노드 경유 (단일 P+D 라우터 8001). hicache write_through.

Usage:
  # 서버: scripts/sglang/start_1P_1D_breakeven.sh
  SERVER_URL=http://127.0.0.1:8001 python benchmark/e2_concession_cost.py

Environment variables:
  SERVER_URL     라우터 주소            (default: http://127.0.0.1:8001)
  P_NODE_URL     P 노드 직접(stats용)    (default: http://127.0.0.1:30000)
  MODEL          모델 이름              (default: /home/uhmturks/hf_models/Qwen3-14B)
  SWEEP_CHARS    anchor 길이 sweep      (default: 2000,6000,15000,30000)
  EVICT_CHARS    flood 요청당 chars      (default: 12000)
  GPU_EVICT_N    GPU evict flood 요청 수 (default: 32)
  REPEAT_N       상태별 반복(중앙값)     (default: 3)
  RUN_TAG        결과 파일 prefix        (default: "")
  MAX_TOKENS_OUT 응답 토큰 수            (default: 16)
"""

import json
import os
import statistics
import time

import requests

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

SERVER_URL   = os.environ.get("SERVER_URL", "http://127.0.0.1:8001").rstrip("/")
P_NODE_URL   = os.environ.get("P_NODE_URL", "http://127.0.0.1:30000").rstrip("/")
MODEL        = os.environ.get("MODEL", "/home/uhmturks/hf_models/Qwen3-14B")
SWEEP_CHARS  = [int(x) for x in os.environ.get("SWEEP_CHARS", "2000,6000,15000,30000").split(",")]
EVICT_CHARS  = int(os.environ.get("EVICT_CHARS", "12000"))
GPU_EVICT_N  = int(os.environ.get("GPU_EVICT_N", "32"))
REPEAT_N     = int(os.environ.get("REPEAT_N", "3"))
RUN_TAG      = os.environ.get("RUN_TAG", "")
MAX_TOKENS_OUT = int(os.environ.get("MAX_TOKENS_OUT", "16"))
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
CHARS_PER_TOKEN = 3.0

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
CHAT_URL   = f"{SERVER_URL}/v1/chat/completions"
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{RUN_TAG}_" if RUN_TAG else ""

# ─── helpers ──────────────────────────────────────────────────────────────────
def _rep(unit: str, n: int) -> str:
    return (unit * (n // len(unit) + 1))[:n]

def anchor_msgs(tag: str, chars: int) -> list:
    unit = f"{tag}{chars:06d}word "
    return [{"role": "system", "content": f"{tag.upper()}{chars}: {_rep(unit, chars)}"},
            {"role": "user", "content": "Summarize in one word."}]

_evict_seed = 1000
def evict_msgs() -> list:
    global _evict_seed
    _evict_seed += 1
    unit = f"evict{_evict_seed:06d}x "
    return [{"role": "system", "content": f"EVICT: {_rep(unit, EVICT_CHARS)}"},
            {"role": "user", "content": "Summarize in one word."}]

def do_request(messages: list) -> float | None:
    """stream → TTFT(ms). None on fail."""
    payload = {"model": MODEL, "messages": messages, "max_tokens": MAX_TOKENS_OUT,
               "temperature": 0, "stream": True}
    t0 = time.perf_counter()
    try:
        resp = requests.post(CHAT_URL, json=payload, stream=True, timeout=300)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line:
                continue
            raw = line.decode("utf-8")
            if not raw.startswith("data: "):
                continue
            if raw[6:] == "[DONE]":
                break
            try:
                ch = json.loads(raw[6:]).get("choices")
            except json.JSONDecodeError:
                continue
            if ch and ch[0].get("delta", {}).get("content"):
                return (time.perf_counter() - t0) * 1000
    except Exception as e:
        print(f"    [ERR] {e}")
    return None

def flood():
    for _ in range(GPU_EVICT_N):
        do_request(evict_msgs())

def med(label: str, msgs: list) -> float | None:
    vals = [v for v in (do_request(msgs) for _ in range(REPEAT_N)) if v is not None]
    return statistics.median(vals) if vals else None

# ─── per-length measurement ───────────────────────────────────────────────────
def measure(chars: int, idx: int) -> dict:
    tok = int(chars / CHARS_PER_TOKEN)
    kv_mb = tok * KV_BYTES_PER_TOKEN / 1024 / 1024
    print(f"\n── {chars} chars ≈ {tok} tok ({kv_mb:.0f} MB KV) ──")
    anchor = anchor_msgs(f"anc{idx}", chars)   # fresh anchor (never seen → cold)

    # cold: 첫 요청
    cold = do_request(anchor)
    print(f"  cold (fresh)        : {cold:.1f} ms" if cold else "  cold FAILED")
    # warm: 즉시 재요청 (GPU L1)
    warm = med("warm", anchor)
    print(f"  warm (GPU L1)       : {warm:.1f} ms" if warm else "  warm FAILED")
    # P_restore: GPU flood로 evict (host 유지) → 재요청
    print(f"  flood GPU ({GPU_EVICT_N} reqs) → evict anchor from GPU radix ...")
    flood()
    p_restore = med("P_restore", anchor)
    print(f"  P_restore (host→GPU): {p_restore:.1f} ms" if p_restore else "  P_restore FAILED")

    d_cost = round(cold - warm, 1) if (cold and warm) else None
    p_cost = round(p_restore - warm, 1) if (p_restore and warm) else None
    asym = round(d_cost / p_cost, 2) if (d_cost and p_cost and p_cost > 0) else None
    return {"chars": chars, "tokens": tok, "kv_mb": round(kv_mb, 1),
            "cold_ms": round(cold, 1) if cold else None,
            "warm_ms": round(warm, 1) if warm else None,
            "p_restore_ms": round(p_restore, 1) if p_restore else None,
            "d_concession_ms": d_cost, "p_concession_ms": p_cost, "asymmetry": asym}

# ─── run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"E2: P-radix vs D-HBM concession cost  [{RUN_TAG or 'default'}]")
print("=" * 70)
print(f"  server={CHAT_URL}  lengths={SWEEP_CHARS}  repeat={REPEAT_N}")
print("=" * 70)
rows = [measure(c, i) for i, c in enumerate(SWEEP_CHARS)]

# ─── report ───────────────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULT — concession cost asymmetry")
print("=" * 70)
print(f"  {'tok':>6} {'cold':>8} {'warm':>7} {'P_rest':>8} {'D양보':>8} {'P양보':>8} {'D/P':>6}")
for r in rows:
    f = lambda v, w=8: f"{v:>{w}.0f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
    print(f"  {r['tokens']:>6} {f(r['cold_ms'])} {f(r['warm_ms'],7)} {f(r['p_restore_ms'])} "
          f"{f(r['d_concession_ms'])} {f(r['p_concession_ms'])} {f(r['asymmetry'],6)}")

# 판정
asyms = [r["asymmetry"] for r in rows if r["asymmetry"]]
pr_vs_cold = [(r["p_restore_ms"], r["cold_ms"]) for r in rows if r["p_restore_ms"] and r["cold_ms"]]
# P양보가 0 근처면 D/P가 폭발(아티팩트) → median + 절대값으로 보고
d_costs = [r["d_concession_ms"] for r in rows if r["d_concession_ms"] is not None]
p_costs = [r["p_concession_ms"] for r in rows if r["p_concession_ms"] is not None]
print()
if d_costs and p_costs:
    print(f"  D 양보(cold−warm)   : {min(d_costs):.0f}–{max(d_costs):.0f} ms (context 비례)")
    print(f"  P 양보(P_rest−warm) : {min(p_costs):.0f}–{max(p_costs):.0f} ms (거의 평평)")
    if asyms:
        print(f"  비대칭 D/P (median) : {statistics.median(asyms):.0f}×  "
              f"(P양보≈0 케이스는 div 폭발이라 median 사용)")
if pr_vs_cold:
    ratios = [pr / cd for pr, cd in pr_vs_cold]
    avg_pr_cold = statistics.mean(ratios)
    print(f"  P_restore / cold 평균 = {avg_pr_cold:.2f}")
    if avg_pr_cold < 0.6:
        print("  → P_restore ≪ cold: host 복원 작동 → 'P 양보 거의 공짜' 가설 성립.")
        if asyms:
            print(f"    비대칭(D양보/P양보) 평균 = {statistics.mean(asyms):.1f}× → P 양보가 훨씬 쌈.")
    else:
        print("  → P_restore ≈ cold: host 복원 깨짐(§2.8 재확인) → recovery-cost 비대칭 없음.")
        print("    가설(host 복제본 = 공짜)은 현재 hicache 설정에선 거짓.")
        print("    L2-revival 시도: HICACHE_MEM_LAYOUT=page_first HICACHE_RATIO=3.0 재시작 후 RUN_TAG=revival.")
        print("    또는 opportunity-cost 비대칭으로 전환(§2.12 P idle 83-90% + §2.14 D evict→hang).")

# ─── save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"{_prefix}e2_concession_cost.json")
with open(out, "w") as f:
    json.dump({"config": {"server": SERVER_URL, "run_tag": RUN_TAG, "sweep_chars": SWEEP_CHARS,
                          "gpu_evict_n": GPU_EVICT_N, "repeat_n": REPEAT_N},
               "lengths": rows}, f, indent=2, ensure_ascii=False)
print(f"\n결과 저장: {out}")

# ─── plot ─────────────────────────────────────────────────────────────────────
def make_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
        return
    toks = [r["tokens"] for r in rows]
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(toks, [r["cold_ms"] for r in rows], "-o", color="#d73027",
            label="cold = D concession (full re-prefill)")
    ax.plot(toks, [r["p_restore_ms"] for r in rows], "-s", color="#1a9850",
            label="P_restore = P concession (host->GPU)")
    ax.plot(toks, [r["warm_ms"] for r in rows], "--^", color="#4575b4",
            label="warm (GPU L1, baseline)")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("TTFT (ms)")
    tag = f" [{RUN_TAG}]" if RUN_TAG else ""
    ax.set_title(f"E2: concession cost — {MODEL_SLUG}{tag}\n"
                 "P concession (host restore) vs D concession (re-prefill)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{_prefix}e2_concession_cost.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"plot saved: {png}")

make_plot()
