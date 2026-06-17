"""
E1: Tier2 Migration End-to-End Benefit (Emulation)   — research.md §2.14 E1
============================================================================
"tool call 동안 idle 세션 KV를 P-HBM(Tier2)에 두면, 압박 시 LRU evict(=다음 turn
cold re-prefill) 대비 TTFT가 얼마나 줄어드는가"를 multi-turn agent 워크로드에서 직접 측정.

────────────────────────────────────────────────────────────────────────────
왜 "emulation"인가 — KV를 P node로 옮기는 메커니즘은 아직 SGLang에 없다
────────────────────────────────────────────────────────────────────────────
우리는 migration을 *구현*하지 않는다. migration의 본질은 "tool call 동안 idle KV를
접근 가능한 상태로 보존해, 다음 turn이 cold re-prefill이 되지 않게 하는 것"이고,
핵심 관측량은 **다음 turn의 TTFT** 하나다. 그 TTFT는 KV가 보존되었는지에만 좌우되므로,
KV가 물리적으로 P-HBM에 있든 D radix에 있든 결과는 같다. 따라서 두 끝점으로 bracket:

  [EVICT]  baseline (migration 없음):
     tool window 동안 /flush_cache 로 세션 prefix를 evict
     → 다음 turn = cold re-prefill   (압박 시 SGLang LRU의 실제 동작, §2.10 cold)

  [RETAIN] 이상적 migration (이득의 상한):
     tool window 동안 그냥 idle → prefix 캐시 유지
     → 다음 turn = warm prefix hit   (KV 보존 = migration 성공, §2.11 T1 수준)

  현실적 migration = RETAIN + NVLink fetch(pen.T2, §2.11 직접 측정값)을 해석적 합성:
     - prefetch로 tool call과 겹치면      → 가시 비용 ≈ RETAIN (best case)
     - 안 겹치면                          → RETAIN + pen.T2 (worst case)
     pen.T2(L) = PENT2_FIXED_MS + L_tok × PENT2_MS_PER_KTOK/1000   (§2.11 fit)

  migration 이득 = EVICT_TTFT − migrate_TTFT   (turn마다, context 길이 따라 증가)

정직성: RETAIN은 KV를 P GPU radix에 두는 것이지 literally "P-HBM as Tier2"는 아니다.
그러나 다음 turn TTFT는 migration이 KV를 되살렸을 때와 동일하고, NVLink 전송 비용만
§2.11에서 따로 더한다. 즉 **측정된 조각(EVICT/RETAIN/pen.T2)의 합성**이지 미구현
메커니즘에 의존하지 않는다.

────────────────────────────────────────────────────────────────────────────
eviction 수단
────────────────────────────────────────────────────────────────────────────
EVICT_METHOD=flush (기본): tool window에 P/D node로 POST /flush_cache (결정적, 저렴).
  ⚠ running/waiting 요청 있으면 flush 거부 → idle한 tool window에 호출해야 함(설계상 OK).
EVICT_METHOD=flood: tool window에 unique 대형 요청 flood → LRU로 세션 evict (현실적
  부분 압박 재현, 비용 큼). pool이 크면 FLOOD_TOKENS↑ 또는 서버 mem_fraction↓ 필요.

Usage:
  # 서버: scripts/sglang/start_1P_1D_breakeven.sh (NVLink, router 8001)
  ROUTER_URL=http://127.0.0.1:8001/v1/chat/completions OUT_TAG=nvlink \
    python benchmark/e1_emulation.py

  # 실험 재실행 없이 기존 JSON으로 PNG만 다시 그리기 (figure 수정용):
  REPLOT_JSON=results/sglang_hicache/Qwen3-14B/nvlink_e1_migration_benefit.json \
    python benchmark/e1_emulation.py
  # 플롯 텍스트는 영문만 사용 — matplotlib 기본 폰트에 한글 글리프가 없어 tofu(네모) 발생.

Environment variables:
  ROUTER_URL        chat 요청 경로 (PD via router)  (default: .../8001/v1/chat/completions)
  P_NODE_URL        P node (flush 대상)            (default: http://127.0.0.1:30000)
  D_NODE_URL        D node (flush 대상)            (default: http://127.0.0.1:30002)
  MODEL             모델 이름                       (default: /home/uhmturks/hf_models/Qwen3-14B)
  NUM_TURNS         turn 수                         (default: 8)
  TURN_OUTPUT_TOKENS turn당 decode 토큰             (default: 256)
  INIT_PROMPT_CHARS  turn0 프롬프트 chars           (default: 3000)
  TURN_GROWTH_CHARS  turn마다 추가 입력 chars       (default: 3000)
  TOOL_DELAY_SEC     tool call idle 초              (default: 6)
  EVICT_METHOD       flush | flood                  (default: flush)
  FLOOD_TOKENS       flood 요청당 토큰 (flood용)    (default: 4000)
  FLOOD_N            flood 요청 수 (flood용)        (default: 30)
  PENT2_FIXED_MS     migration 고정 오버헤드 ms     (default: 38, §2.11)
  PENT2_MS_PER_KTOK  토큰 1k당 NVLink fetch ms      (default: 3.35, §2.11 fit)
  KV_BYTES_PER_TOKEN 토큰당 KV 바이트               (default: 163840)
  REPEAT            arm당 반복(중앙값 사용)         (default: 1)
  OUT_DIR/OUT_TAG   결과 경로/태그
"""

import json
import os
import statistics
import time

import requests

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

# ─── Config ─────────────────────────────────────────────────────────────────
ROUTER_URL    = os.environ.get("ROUTER_URL", "http://127.0.0.1:8001/v1/chat/completions")
P_NODE_URL    = os.environ.get("P_NODE_URL", "http://127.0.0.1:30000").rstrip("/")
D_NODE_URL    = os.environ.get("D_NODE_URL", "http://127.0.0.1:30002").rstrip("/")
MODEL         = os.environ.get("MODEL", "/home/uhmturks/hf_models/Qwen3-14B")
NUM_TURNS     = int(os.environ.get("NUM_TURNS", "8"))
TURN_OUTPUT_TOKENS = int(os.environ.get("TURN_OUTPUT_TOKENS", "256"))
INIT_PROMPT_CHARS  = int(os.environ.get("INIT_PROMPT_CHARS", "3000"))
TURN_GROWTH_CHARS  = int(os.environ.get("TURN_GROWTH_CHARS", "3000"))
TOOL_DELAY_SEC     = float(os.environ.get("TOOL_DELAY_SEC", "6"))
EVICT_METHOD       = os.environ.get("EVICT_METHOD", "flush")
FLOOD_TOKENS       = int(os.environ.get("FLOOD_TOKENS", "4000"))
FLOOD_N            = int(os.environ.get("FLOOD_N", "30"))
PENT2_FIXED_MS     = float(os.environ.get("PENT2_FIXED_MS", "38"))
PENT2_MS_PER_KTOK  = float(os.environ.get("PENT2_MS_PER_KTOK", "3.35"))
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
REPEAT        = int(os.environ.get("REPEAT", "1"))
OUT_TAG       = os.environ.get("OUT_TAG", "")
CHARS_PER_TOKEN = 3.0

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{OUT_TAG}_" if OUT_TAG else ""

def pen_t2_ms(tokens: float) -> float:
    """§2.11 fit: NVLink fetch 비용 = 고정 오버헤드 + 토큰 비례."""
    return PENT2_FIXED_MS + (tokens / 1000.0) * PENT2_MS_PER_KTOK

def make_plot(plot_rows: list[dict]):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib not installed — skipping plot)")
        return
    # 플롯 텍스트는 영문만 사용 (matplotlib 기본 폰트에 한글 글리프 없음 → tofu 방지)
    turns = [r["turn"] for r in plot_rows]
    ev = [r["evict_ttft_ms"] for r in plot_rows]
    rt = [r["retain_ttft_ms"] for r in plot_rows]
    mig = [r["migrate_est_ms"] for r in plot_rows]
    fig, ax = plt.subplots(figsize=(11, 6))
    ax.plot(turns, ev, "-o", color="#d73027", label="EVICT (no migration -> re-prefill)")
    ax.plot(turns, mig, "-s", color="#1a9850", label="migrate* (RETAIN + NVLink pen.T2)")
    ax.plot(turns, rt, "--^", color="#4575b4", label="RETAIN (KV preserved, benefit upper bound)")
    ax.fill_between(turns, mig, ev, where=[(e or 0) >= (m or 0) for e, m in zip(ev, mig)],
                    color="#1a9850", alpha=0.12, label="migration benefit")
    ax.set_xlabel("turn (context accumulates ->)")
    ax.set_ylabel("next-turn TTFT (ms)")
    ax.set_title(f"E1: Tier2 migration benefit (emul) — {MODEL_SLUG}\n"
                 f"evict={EVICT_METHOD}, pen.T2={PENT2_FIXED_MS}+{PENT2_MS_PER_KTOK}/1k tok (sec 2.11)")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    # context tokens as secondary x labels
    for r in plot_rows:
        ax.annotate(f"{r['ctx_tokens']//1000}k", (r["turn"], 0), textcoords="offset points",
                    xytext=(0, -16), ha="center", fontsize=7, color="gray")
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{_prefix}e1_migration_benefit.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"plot saved: {png}")

# REPLOT 모드: 기존 JSON으로 PNG만 재생성 (실험 재실행 없이 figure 수정용)
#   REPLOT_JSON=results/.../nvlink_e1_migration_benefit.json python benchmark/e1_emulation.py
_REPLOT = os.environ.get("REPLOT_JSON", "")
if _REPLOT:
    _d = json.load(open(_REPLOT))
    print(f"[REPLOT] {_REPLOT} → PNG 재생성")
    make_plot(_d["composed"])
    raise SystemExit(0)

# ─── Helpers ──────────────────────────────────────────────────────────────────
def _filler(nchars: int, seed: int) -> str:
    unit = f"ctx-seg-{seed:05d} the agent observed the environment and planned next actions. "
    return (unit * (nchars // len(unit) + 1))[:nchars]

def do_chat(messages: list, max_tokens: int) -> tuple[float | None, str, int | None]:
    """stream 요청 → (ttft_s, text, prompt_tokens). prompt_tokens는 usage에서(가능 시)."""
    payload = {"model": MODEL, "messages": messages, "max_tokens": max_tokens,
               "temperature": 0, "stream": True,
               "stream_options": {"include_usage": True}}
    t0 = time.perf_counter()
    t_first = None
    text = ""
    prompt_tokens = None
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
            if usage and usage.get("prompt_tokens"):
                prompt_tokens = usage["prompt_tokens"]
            ch = chunk.get("choices")
            if ch and ch[0].get("delta", {}).get("content"):
                if t_first is None:
                    t_first = time.perf_counter()
                text += ch[0]["delta"]["content"]
    except Exception as e:
        print(f"    [chat ERR] {e}")
        return None, "", None
    ttft = (t_first - t0) if t_first else None
    return ttft, text, prompt_tokens

def flush_node(base: str) -> bool:
    """idle 상태에서 radix cache flush. timeout으로 idle 대기."""
    for path in (f"{base}/flush_cache?timeout=5", f"{base}/flush_cache"):
        try:
            r = requests.post(path, timeout=10)
            if r.status_code == 200:
                return True
        except Exception:
            pass
    return False

def flood_evict():
    """unique 대형 요청으로 pool을 채워 세션 prefix를 LRU evict."""
    for i in range(FLOOD_N):
        msgs = [{"role": "system", "content": f"FLOOD{i:05d} " + _filler(FLOOD_TOKENS * 3, 90000 + i)},
                {"role": "user", "content": "ok"}]
        do_chat(msgs, max_tokens=1)

# ─── One arm: multi-turn session under a tool-window policy ────────────────────
def run_arm(mode: str, seed: int) -> list[dict]:
    """mode ∈ {retain, evict}. 각 turn의 TTFT + context 토큰 측정.
    retain: tool window에 idle (KV 보존) → warm
    evict : tool window에 flush/flood (KV evict) → cold"""
    # 시작 전 깨끗한 상태로: 양쪽 노드 flush
    flush_node(P_NODE_URL); flush_node(D_NODE_URL)
    time.sleep(1)
    history = [{"role": "system", "content": f"You are agent {seed}. " + _filler(INIT_PROMPT_CHARS, seed)}]
    out = []
    for turn in range(NUM_TURNS):
        history.append({"role": "user",
                        "content": f"[turn {turn}] " + _filler(TURN_GROWTH_CHARS, seed * 100 + turn + 1)})
        ttft, text, ptok = do_chat(history, TURN_OUTPUT_TOKENS)
        ctx_tok = ptok if ptok else int(sum(len(m["content"]) for m in history) / CHARS_PER_TOKEN)
        out.append({"turn": turn, "ctx_tokens": ctx_tok,
                    "ttft_ms": round(ttft * 1000, 1) if ttft else None})
        tag = "warm" if mode == "retain" else "COLD"
        print(f"    [{mode} turn {turn}]  ctx={ctx_tok:>6} tok  TTFT={ttft*1000:7.1f} ms  ({tag} 기대)"
              if ttft else f"    [{mode} turn {turn}]  FAILED")
        history.append({"role": "assistant", "content": text or "(no output)"})
        # ── tool call idle window: policy ──
        if mode == "evict":
            # KV를 버린다 (migration 없음)
            t0 = time.perf_counter()
            if EVICT_METHOD == "flush":
                okp, okd = flush_node(P_NODE_URL), flush_node(D_NODE_URL)
                if not (okp or okd):
                    print("      ⚠ flush_cache 실패 (요청 active?) — EVICT_METHOD=flood 고려")
            else:
                flood_evict()
            # 남은 tool window 시간 idle
            rest = TOOL_DELAY_SEC - (time.perf_counter() - t0)
            if rest > 0:
                time.sleep(rest)
        else:
            # KV 보존 (migration 성공 emulation)
            time.sleep(TOOL_DELAY_SEC)
        # tool 결과 누적 (context 성장)
        history.append({"role": "user",
                        "content": f"[tool {turn}] " + _filler(TURN_GROWTH_CHARS // 2, seed * 100 + 500 + turn)})
    return out

# ─── Run both arms ────────────────────────────────────────────────────────────
def median_arm(mode: str, seed0: int) -> list[dict]:
    runs = [run_arm(mode, seed0 + r) for r in range(REPEAT)]
    merged = []
    for turn in range(NUM_TURNS):
        ttfts = [r[turn]["ttft_ms"] for r in runs if r[turn]["ttft_ms"] is not None]
        ctx = statistics.median([r[turn]["ctx_tokens"] for r in runs])
        merged.append({"turn": turn, "ctx_tokens": int(ctx),
                       "ttft_ms": round(statistics.median(ttfts), 1) if ttfts else None})
    return merged

print("=" * 70)
print("E1: Tier2 Migration Benefit (Emulation)")
print("=" * 70)
print(f"  router={ROUTER_URL}")
print(f"  turns={NUM_TURNS} tool_delay={TOOL_DELAY_SEC}s evict={EVICT_METHOD} repeat={REPEAT}")
print(f"  pen.T2 model: {PENT2_FIXED_MS}ms + {PENT2_MS_PER_KTOK}ms/1k tok (§2.11)")
print("=" * 70)

print("\n[RETAIN arm] tool window에 idle → KV 보존 (migration 성공 emul)")
retain = median_arm("retain", seed0=1)
print("\n[EVICT arm] tool window에 flush → KV evict (migration 없음)")
evict = median_arm("evict", seed0=101)

# ─── Compose + report ─────────────────────────────────────────────────────────
print("\n" + "=" * 70)
print("RESULT  (turn0=cold 양쪽 동일; turn1+ 에서 격차 발생)")
print("=" * 70)
print(f"  {'turn':>4} {'ctx_tok':>8} {'EVICT':>9} {'RETAIN':>9} {'migrate*':>9} "
      f"{'benefit':>9}  (migrate=RETAIN+pen.T2)")
rows = []
for rt, ev in zip(retain, evict):
    ctx = ev["ctx_tokens"]
    rt_ms, ev_ms = rt["ttft_ms"], ev["ttft_ms"]
    mig = round(rt_ms + pen_t2_ms(ctx), 1) if rt_ms is not None else None
    benefit = round(ev_ms - mig, 1) if (ev_ms is not None and mig is not None) else None
    rows.append({"turn": ev["turn"], "ctx_tokens": ctx,
                 "evict_ttft_ms": ev_ms, "retain_ttft_ms": rt_ms,
                 "migrate_est_ms": mig, "benefit_ms": benefit,
                 "pen_t2_ms": round(pen_t2_ms(ctx), 1)})
    f = lambda v: f"{v:>9.1f}" if isinstance(v, (int, float)) else f"{'—':>9}"
    print(f"  {ev['turn']:>4} {ctx:>8} {f(ev_ms)} {f(rt_ms)} {f(mig)} {f(benefit)}")

# turn1+ 평균 이득 (turn0은 양쪽 cold라 제외)
ben = [r["benefit_ms"] for r in rows[1:] if r["benefit_ms"] is not None]
if ben:
    print(f"\n  turn1+ 평균 migration 이득: {statistics.mean(ben):.0f} ms  "
          f"(최대 {max(ben):.0f} ms @ turn{rows[1:][ben.index(max(ben))]['turn']})")
    evs = [r["evict_ttft_ms"] for r in rows[1:] if r["evict_ttft_ms"]]
    migs = [r["migrate_est_ms"] for r in rows[1:] if r["migrate_est_ms"]]
    if evs and migs:
        print(f"  turn1+ 평균 TTFT: EVICT {statistics.mean(evs):.0f}ms → migrate {statistics.mean(migs):.0f}ms "
              f"({statistics.mean(evs)/statistics.mean(migs):.1f}× 단축)")

# ─── Save + plot ──────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out_json = os.path.join(OUT_DIR, f"{_prefix}e1_migration_benefit.json")
with open(out_json, "w") as f:
    json.dump({"config": {
        "router": ROUTER_URL, "num_turns": NUM_TURNS, "tool_delay_sec": TOOL_DELAY_SEC,
        "evict_method": EVICT_METHOD, "pent2_fixed_ms": PENT2_FIXED_MS,
        "pent2_ms_per_ktok": PENT2_MS_PER_KTOK, "repeat": REPEAT},
        "retain": retain, "evict": evict, "composed": rows}, f, indent=2, ensure_ascii=False)
print(f"\n결과 저장: {out_json}")

make_plot(rows)
