"""
E1b: Natural LRU Eviction under Concurrency   — research.md §2.14 E1 후속
============================================================================
E1(`e1_emulation.py`)은 `/flush_cache`로 eviction을 *강제*했다(C=1, 결정적).
E1b는 **C=8~16 동시성으로 실제 메모리 압박을 만들어 자연 LRU eviction**이 일어나게 하고,
그 압박 하에서 "evict된 turn(cold) vs 보존된 turn(warm)"의 TTFT 격차가 E1과 같이
재현되는지를 본다. flush를 쓰지 않는다 — 압박이 알아서 LRU evict를 유발.

────────────────────────────────────────────────────────────────────────────
메커니즘: 자연 상태에선 eviction이 비결정적 → cached_tokens로 turn을 사후 분류
────────────────────────────────────────────────────────────────────────────
C개 multi-turn 세션을 병렬 구동. tool-call window 동안 세션은 idle → 그 KV는 LRU-
evictable. 다른 세션들이 pool을 쓰면 idle 세션 prefix가 자연 evict될 수 있음.

각 turn을 **SGLang이 보고하는 cached_tokens로 HIT/MISS 분류** (서버 `--enable-cache-report`):
  hit_ratio = cached_tokens / prompt_tokens
  HIT  (prefix 보존, warm)        : hit_ratio ≥ HIT_THRESHOLD   → 이전 context 재사용
  MISS (prefix evict, cold)       : hit_ratio < HIT_THRESHOLD   → full re-prefill
  (turn 0은 이전 context 없음 → 항상 MISS, 분석에서 제외)

보여줄 것:
  (1) C↑ → 자연 MISS율 상승 = 압박 motivation (flush 없이 evict가 실제로 일어남)
  (2) MISS turn = cold TTFT, HIT turn = warm TTFT — E1의 격차가 자연 발생으로 재현
  (3) migration 효과 합성: MISS turn의 TTFT를 (warm_ref + pen.T2)로 치환하면
      mean/P95 TTFT가 얼마나 개선되는가 = "migration이 자연 cold re-prefill을 제거".

warm_ref: 같은 turn(=같은 ctx·부하단계)의 HIT median (robust; 큐잉 효과 포함, self-contained).
  migration_ttft(MISS) = warm_ref(turn) + pen.T2(ctx)   [§2.11 fit]
  ⚠ 고동시성에선 TTFT가 큐잉 지배라 cold/warm 분리가 흐려질 수 있음 → turn별 분리도 함께 출력.
  improvement = measured − migration_ttft  (MISS turn에만 적용; HIT은 그대로)

정직성: cached_tokens는 prefix-cache 적중을 직접 보고하므로 TTFT 임계 추정보다 rigorous.
HIT/MISS는 P node radix(=prefill 캐시) 상태를 반영. PD에서 prefill이 P에서 일어나므로
이 cache 상태가 곧 re-prefill 여부를 결정.

Usage:
  # 서버 (--enable-cache-report 포함): scripts/sglang/start_1P_1D_breakeven.sh
  # 압박이 약하면(자연 MISS=0) D pool을 줄여 강제: 서버를 MEM_FRACTION 낮춰 재시작
  for C in 8 16; do
    CONCURRENCY=$C NUM_TURNS=8 TOOL_DELAY_SEC=6 OUT_TAG=nvlink_c$C \
      python benchmark/e1b_natural_eviction.py
  done

Environment variables:
  ROUTER_URL        chat 요청 경로                 (default: .../8001/v1/chat/completions)
  MODEL             모델 이름                       (default: /home/uhmturks/hf_models/Qwen3-14B)
  CONCURRENCY       병렬 세션 수                    (default: 8)
  NUM_TURNS         세션당 turn 수                  (default: 8)
  TURN_OUTPUT_TOKENS turn당 decode 토큰             (default: 256)
  INIT_PROMPT_CHARS  turn0 프롬프트 chars           (default: 3000)
  TURN_GROWTH_CHARS  turn마다 추가 입력 chars       (default: 3000)
  TOOL_DELAY_SEC     tool call idle 초              (default: 6)
  HIT_THRESHOLD      cached/prompt HIT 임계         (default: 0.4)
  PENT2_FIXED_MS / PENT2_MS_PER_KTOK   §2.11 pen.T2 (default: 38, 3.35)
  KV_BYTES_PER_TOKEN  토큰당 KV 바이트              (default: 163840)
  OUT_DIR/OUT_TAG    결과 경로/태그
"""

import json
import os
import statistics
import threading
import time

import requests

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

# ─── Config ─────────────────────────────────────────────────────────────────
ROUTER_URL    = os.environ.get("ROUTER_URL", "http://127.0.0.1:8001/v1/chat/completions")
MODEL         = os.environ.get("MODEL", "/home/uhmturks/hf_models/Qwen3-14B")
CONCURRENCY   = int(os.environ.get("CONCURRENCY", "8"))
NUM_TURNS     = int(os.environ.get("NUM_TURNS", "8"))
TURN_OUTPUT_TOKENS = int(os.environ.get("TURN_OUTPUT_TOKENS", "256"))
INIT_PROMPT_CHARS  = int(os.environ.get("INIT_PROMPT_CHARS", "3000"))
TURN_GROWTH_CHARS  = int(os.environ.get("TURN_GROWTH_CHARS", "3000"))
TOOL_DELAY_SEC     = float(os.environ.get("TOOL_DELAY_SEC", "6"))
HIT_THRESHOLD      = float(os.environ.get("HIT_THRESHOLD", "0.4"))
PENT2_FIXED_MS     = float(os.environ.get("PENT2_FIXED_MS", "38"))
PENT2_MS_PER_KTOK  = float(os.environ.get("PENT2_MS_PER_KTOK", "3.35"))
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
OUT_TAG       = os.environ.get("OUT_TAG", "")
CHARS_PER_TOKEN = 3.0

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{OUT_TAG}_" if OUT_TAG else ""

def pen_t2_ms(tokens: float) -> float:
    return PENT2_FIXED_MS + (tokens / 1000.0) * PENT2_MS_PER_KTOK

# ─── Shared sample collection ─────────────────────────────────────────────────
samples: list[dict] = []          # {sid, turn, ctx_tokens, cached_tokens, hit_ratio, ttft_ms}
_lock = threading.Lock()

def _filler(nchars: int, seed: int) -> str:
    unit = f"ctx-seg-{seed:05d} the agent observed the environment and planned next actions. "
    return (unit * (nchars // len(unit) + 1))[:nchars]

def do_chat(messages: list) -> tuple[float | None, str, int | None, int | None]:
    """stream 요청 → (ttft_s, text, prompt_tokens, cached_tokens)."""
    payload = {"model": MODEL, "messages": messages, "max_tokens": TURN_OUTPUT_TOKENS,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    t_first = None
    text = ""
    ptok = ctok = None
    try:
        resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=600)
        resp.raise_for_status()
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
            usage = chunk.get("usage")
            if usage:
                if usage.get("prompt_tokens"):
                    ptok = usage["prompt_tokens"]
                # OpenAI 형식: prompt_tokens_details.cached_tokens (SGLang --enable-cache-report)
                det = usage.get("prompt_tokens_details") or {}
                if det.get("cached_tokens") is not None:
                    ctok = det["cached_tokens"]
                elif usage.get("cached_tokens") is not None:
                    ctok = usage["cached_tokens"]
            ch = chunk.get("choices")
            if ch and ch[0].get("delta", {}).get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                text += ch[0]["delta"]["content"]
    except Exception as e:
        with _lock:
            print(f"    [chat ERR] {e}")
        return None, "", None, None
    ttft = (t_first - t0) if t_first else None
    return ttft, text, ptok, ctok

def run_session(sid: int):
    history = [{"role": "system", "content": f"You are agent {sid}. " + _filler(INIT_PROMPT_CHARS, sid)}]
    for turn in range(NUM_TURNS):
        history.append({"role": "user",
                        "content": f"[s{sid} turn {turn}] " + _filler(TURN_GROWTH_CHARS, sid * 100 + turn + 1)})
        ttft, text, ptok, ctok = do_chat(history)
        ctx = ptok if ptok else int(sum(len(m["content"]) for m in history) / CHARS_PER_TOKEN)
        hit_ratio = (ctok / ctx) if (ctok is not None and ctx) else None
        with _lock:
            samples.append({
                "sid": sid, "turn": turn, "ctx_tokens": ctx,
                "cached_tokens": ctok, "hit_ratio": round(hit_ratio, 3) if hit_ratio is not None else None,
                "ttft_ms": round(ttft * 1000, 1) if ttft else None,
            })
        history.append({"role": "assistant", "content": text or "(no output)"})
        time.sleep(TOOL_DELAY_SEC)   # tool call idle window (flush 없음 — 자연 LRU만)
        history.append({"role": "user",
                        "content": f"[s{sid} tool {turn}] " + _filler(TURN_GROWTH_CHARS // 2, sid * 100 + 500 + turn)})

def pct(vals: list[float], q: float) -> float:
    if not vals:
        return float("nan")
    s = sorted(vals)
    i = min(len(s) - 1, int(q * (len(s) - 1) + 0.5))
    return s[i]

# ─── Run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"E1b: Natural LRU Eviction under Concurrency  C={CONCURRENCY}")
print("=" * 70)
print(f"  router={ROUTER_URL}  turns={NUM_TURNS} tool_delay={TOOL_DELAY_SEC}s")
print(f"  HIT_THRESHOLD={HIT_THRESHOLD}  pen.T2={PENT2_FIXED_MS}+{PENT2_MS_PER_KTOK}/1k tok")
print("=" * 70)

threads = [threading.Thread(target=run_session, args=(sid,), daemon=True) for sid in range(CONCURRENCY)]
t0 = time.time()
for i, th in enumerate(threads):
    th.start()
    time.sleep(0.25)  # stagger
for th in threads:
    th.join()
print(f"\n완료: {len(samples)} samples in {time.time()-t0:.0f}s")

# cached_tokens 보고 확인
n_with_ctok = sum(1 for s in samples if s["cached_tokens"] is not None)
if n_with_ctok == 0:
    print("⚠ cached_tokens가 None — 서버에 --enable-cache-report 없음. HIT/MISS 분류 불가.")
    print("  서버를 --enable-cache-report로 재시작하세요 (start 스크립트에 반영됨).")

# ─── Classify + compose ───────────────────────────────────────────────────────
analyz = [s for s in samples if s["turn"] >= 1 and s["ttft_ms"] is not None and s["hit_ratio"] is not None]
hits = [s for s in analyz if s["hit_ratio"] >= HIT_THRESHOLD]
misses = [s for s in analyz if s["hit_ratio"] < HIT_THRESHOLD]
miss_rate = len(misses) / len(analyz) if analyz else float("nan")

# ─── Robust warm_ref: MISS turn을 같은 turn(=같은 ctx·부하단계)의 HIT median으로 매칭 ──
# (이전 global linear fit은 고-ctx HIT의 큐잉 노이즈에 기울기가 부풀려져 절편 음수 등
#  비현실적 warm_ref를 만들었음 — research.md §2.14 E1b 메모. turn-matched median이 robust.)
from collections import defaultdict as _dd
_hit_ttft_by_turn = _dd(list)
for _s in hits:
    _hit_ttft_by_turn[_s["turn"]].append(_s["ttft_ms"])
_hit_med_by_turn = {t: statistics.median(v) for t, v in _hit_ttft_by_turn.items()}
_global_hit_med = statistics.median([s["ttft_ms"] for s in hits]) if hits else (
    statistics.median([s["ttft_ms"] for s in analyz]) if analyz else 0.0)

def warm_ref_for(s: dict) -> float:
    """그 MISS turn이 HIT였다면 가졌을 TTFT 추정 (같은 turn HIT median, 없으면 인접/전역)."""
    t = s["turn"]
    if t in _hit_med_by_turn:
        return _hit_med_by_turn[t]
    # 인접 turn HIT median (ctx 비슷)
    near = [tt for tt in _hit_med_by_turn if abs(tt - t) <= 1]
    if near:
        return statistics.median([_hit_med_by_turn[tt] for tt in near])
    return _global_hit_med

# baseline vs migration TTFT (turn>=1)
baseline_ttft = [s["ttft_ms"] for s in analyz]
migration_ttft = []
for s in analyz:
    if s["hit_ratio"] >= HIT_THRESHOLD:
        migration_ttft.append(s["ttft_ms"])           # 이미 warm → 변화 없음
    else:
        migration_ttft.append(warm_ref_for(s) + pen_t2_ms(s["ctx_tokens"]))

def stat(v): return {"mean": round(statistics.mean(v), 1), "p50": round(pct(v, 0.5), 1),
                     "p95": round(pct(v, 0.95), 1), "max": round(max(v), 1)} if v else {}

base_s, mig_s = stat(baseline_ttft), stat(migration_ttft)

print("\n" + "=" * 70)
print(f"RESULT  C={CONCURRENCY}  (turn≥1, n={len(analyz)})")
print("=" * 70)
print(f"  자연 MISS율 (LRU evict 발생): {miss_rate*100:.0f}%  ({len(misses)} cold / {len(analyz)})")
if hits:
    print(f"  HIT  turn TTFT: mean={statistics.mean([s['ttft_ms'] for s in hits]):.0f}ms (warm, n={len(hits)})")
if misses:
    print(f"  MISS turn TTFT: mean={statistics.mean([s['ttft_ms'] for s in misses]):.0f}ms (cold, n={len(misses)})")
if hits and misses:
    gap = statistics.mean([s['ttft_ms'] for s in misses]) - statistics.mean([s['ttft_ms'] for s in hits])
    print(f"  → cold−warm 격차(aggregate): {gap:.0f}ms")

# per-turn HIT vs MISS median (큐잉 지배 여부 진단 — turn별 cold/warm 분리도 확인)
_miss_by_turn = _dd(list)
for _s in misses:
    _miss_by_turn[_s["turn"]].append(_s["ttft_ms"])
print(f"\n  turn별 HIT/MISS median (분리도 점검):")
print(f"    {'turn':>4} {'ctx':>6} {'HITmed':>8} {'MISSmed':>8} {'gap':>7} {'nH/nM':>7}")
turn_table = []
for t in sorted(set([s["turn"] for s in analyz])):
    h = _hit_med_by_turn.get(t); m = statistics.median(_miss_by_turn[t]) if _miss_by_turn.get(t) else None
    ctx = next((s["ctx_tokens"] for s in analyz if s["turn"] == t), 0)
    gap = f"{m-h:>7.0f}" if (h and m) else f"{'—':>7}"
    turn_table.append({"turn": t, "ctx": ctx, "hit_med": h, "miss_med": m,
                       "n_hit": len(_hit_ttft_by_turn.get(t, [])), "n_miss": len(_miss_by_turn.get(t, []))})
    print(f"    {t:>4} {ctx:>6} {h if h else '—':>8} {m if m else '—':>8} {gap} "
          f"{len(_hit_ttft_by_turn.get(t,[])):>3}/{len(_miss_by_turn.get(t,[])):<3}")

print(f"\n  TTFT 분포          {'baseline':>10} {'migration':>10}  개선")
for k in ("mean", "p50", "p95", "max"):
    if base_s.get(k) and mig_s.get(k):
        imp = (1 - mig_s[k] / base_s[k]) * 100
        print(f"    {k:>4}            {base_s[k]:>9.0f}ms {mig_s[k]:>9.0f}ms  {imp:>5.0f}%")

# ─── Save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_json = os.path.join(OUT_DIR, f"{_prefix}e1b_natural_eviction.json")
with open(out_json, "w") as f:
    json.dump({"config": {"concurrency": CONCURRENCY, "num_turns": NUM_TURNS,
                          "tool_delay_sec": TOOL_DELAY_SEC, "hit_threshold": HIT_THRESHOLD,
                          "pent2_fixed_ms": PENT2_FIXED_MS, "pent2_ms_per_ktok": PENT2_MS_PER_KTOK},
               "summary": {"miss_rate": miss_rate, "n_analyzed": len(analyz),
                           "n_hit": len(hits), "n_miss": len(misses),
                           "warm_med_by_turn": _hit_med_by_turn, "per_turn": turn_table,
                           "baseline_ttft": base_s, "migration_ttft": mig_s},
               "samples": samples}, f, indent=2, ensure_ascii=False)
print(f"\n결과 저장: {out_json}")

# ─── Plot ─────────────────────────────────────────────────────────────────────
def make_plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
        return
    # 영문 라벨만 (한글 글리프 tofu 방지)
    fig, ax = plt.subplots(figsize=(11, 6))
    hx = [s["ctx_tokens"] for s in hits];   hy = [s["ttft_ms"] for s in hits]
    mx = [s["ctx_tokens"] for s in misses]; my = [s["ttft_ms"] for s in misses]
    ax.scatter(hx, hy, c="#4575b4", s=28, alpha=0.7, label=f"HIT / warm (retained, n={len(hits)})")
    ax.scatter(mx, my, c="#d73027", s=28, alpha=0.7, label=f"MISS / cold re-prefill (evicted, n={len(misses)})")
    # turn-matched warm_ref median + migration overlay (robust; no linear fit)
    tt = sorted(t for t in _hit_med_by_turn)
    if tt:
        wx = [next((s["ctx_tokens"] for s in analyz if s["turn"] == t), 0) for t in tt]
        wy = [_hit_med_by_turn[t] for t in tt]
        ax.plot(wx, wy, "--o", color="#4575b4", lw=1.5, ms=5, label="warm_ref (HIT median / turn)")
        ax.plot(wx, [y + pen_t2_ms(x) for x, y in zip(wx, wy)], "-s", color="#1a9850", lw=1.8, ms=5,
                label="migration* = warm_ref + pen.T2")
    ax.set_xlabel("context tokens")
    ax.set_ylabel("next-turn TTFT (ms)")
    ax.set_title(f"E1b: natural LRU eviction under load — {MODEL_SLUG}  C={CONCURRENCY}\n"
                 f"natural MISS rate={miss_rate*100:.0f}%,  P95 TTFT "
                 f"{base_s.get('p95',0):.0f}->{mig_s.get('p95',0):.0f}ms with migration")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{_prefix}e1b_natural_eviction.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"plot saved: {png}")

make_plot()
