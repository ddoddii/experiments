"""
E4: Per-turn re-prefill & re-transfer cost  — research.md §2.14 E4 (motivation Figure)
======================================================================================
SGLang PD multi-turn의 핵심 비효율을 *코드 변경 없이* 정량화:
"매 turn P가 conversation history를 얼마나 재계산(re-prefill)하고, 얼마나 재전송하는가."

문헌(SGLang PD docs / PPD arXiv:2603.13358): multi-turn에서 P node가 전체 history(이전
출력 포함) KV를 재계산 → prefill 비용의 최대 99%가 historical token 재계산. E4는 이걸
우리 셋업에서 직접 측정해 논문 motivation Figure로 만든다.

────────────────────────────────────────────────────────────────────────────
측정 (서버 `--enable-cache-report` 필요 — start 스크립트에 반영됨)
────────────────────────────────────────────────────────────────────────────
단일 세션 multi-turn(누적 context). 각 turn 응답의 usage에서:
  prompt_tokens          = 이번 turn 입력 context 총 토큰 (history + 새 입력)
  cached_tokens          = 그 중 prefix-cache로 재사용된 토큰 (P radix hit)
  recompute_tokens       = prompt_tokens − cached_tokens = **재계산된 historical/new 토큰**
  recompute_frac         = recompute / prompt  = "이번 turn prefill 중 재계산 비율"

  재전송량(추정): D가 turn 종료 시 KV를 free(§2.14 E3)하므로 다음 turn 전체 prompt_tokens의
  KV가 P→D 재전송됨. transfer_bytes ≈ prompt_tokens × KV_BYTES_PER_TOKEN.
  transfer_ms(추정) = §2.11 모델: 고정 + prompt_tokens × KV/대역폭.

  TTFT(실측): 각 turn 실제 TTFT — 재계산+전송+큐잉의 합.

핵심 산출 (motivation):
  (1) turn이 깊어질수록 recompute_tokens가 누적 history만큼 커지는가? (history 지배)
  (2) recompute_frac (재계산이 prefill의 X%)
  (3) "history 재계산"이 없다면(이상적 = 전부 cached) 절감될 TTFT
      = TTFT − warm_ref(새 입력만 prefill) — break-even/E1 데이터와 연결

비교 축:
  CONCURRENCY=1 (순수 재계산 비용, 큐잉 없음) 권장.
  여러 C로 돌리면 부하 하 재계산+큐잉 합산 효과도 관찰.

Usage:
  # 서버 (--enable-cache-report 포함): scripts/sglang/start_1P_1D_breakeven.sh
  CONCURRENCY=1 NUM_TURNS=10 OUT_TAG=c1 python benchmark/e4_reprefill_cost.py

Environment variables:
  ROUTER_URL/MODEL/CONCURRENCY/NUM_TURNS/TURN_OUTPUT_TOKENS/INIT_PROMPT_CHARS/
  TURN_GROWTH_CHARS/TOOL_DELAY_SEC  — e1b와 동일 기본값 (단 TOOL_DELAY 기본 2s)
  PENT2_FIXED_MS / PENT2_MS_PER_KTOK / KV_BYTES_PER_TOKEN — §2.11 전송 모델
  OUT_DIR/OUT_TAG
"""

import json
import os
import statistics
import threading
import time

import requests

SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

ROUTER_URL    = os.environ.get("ROUTER_URL", "http://127.0.0.1:8001/v1/chat/completions")
MODEL         = os.environ.get("MODEL", "/home/uhmturks/hf_models/Qwen3-14B")
CONCURRENCY   = int(os.environ.get("CONCURRENCY", "1"))
NUM_TURNS     = int(os.environ.get("NUM_TURNS", "10"))
TURN_OUTPUT_TOKENS = int(os.environ.get("TURN_OUTPUT_TOKENS", "256"))
INIT_PROMPT_CHARS  = int(os.environ.get("INIT_PROMPT_CHARS", "3000"))
TURN_GROWTH_CHARS  = int(os.environ.get("TURN_GROWTH_CHARS", "3000"))
TOOL_DELAY_SEC     = float(os.environ.get("TOOL_DELAY_SEC", "2"))
PENT2_FIXED_MS     = float(os.environ.get("PENT2_FIXED_MS", "38"))
PENT2_MS_PER_KTOK  = float(os.environ.get("PENT2_MS_PER_KTOK", "3.35"))
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
OUT_TAG       = os.environ.get("OUT_TAG", "")
CHARS_PER_TOKEN = 3.0

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{OUT_TAG}_" if OUT_TAG else ""

def transfer_ms(tok): return PENT2_FIXED_MS + (tok / 1000.0) * PENT2_MS_PER_KTOK

samples: list[dict] = []
_lock = threading.Lock()

def _filler(n, seed):
    unit = f"ctx-{seed:05d} the agent observed the environment and planned next actions. "
    return (unit * (n // len(unit) + 1))[:n]

def do_chat(messages):
    """stream → (ttft_ms, text, prompt_tokens, cached_tokens)."""
    payload = {"model": MODEL, "messages": messages, "max_tokens": TURN_OUTPUT_TOKENS,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; text = ""; ptok = ctok = None
    try:
        resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=600)
        resp.raise_for_status()
        for line in resp.iter_lines():
            if not line: continue
            raw = line.decode("utf-8")
            if not raw.startswith("data: "): continue
            if raw[6:] == "[DONE]": break
            try: chunk = json.loads(raw[6:])
            except json.JSONDecodeError: continue
            u = chunk.get("usage")
            if u:
                if u.get("prompt_tokens"): ptok = u["prompt_tokens"]
                det = u.get("prompt_tokens_details") or {}
                if det.get("cached_tokens") is not None: ctok = det["cached_tokens"]
                elif u.get("cached_tokens") is not None: ctok = u["cached_tokens"]
            ch = chunk.get("choices")
            if ch and ch[0].get("delta", {}).get("content"):
                if first is None: first = time.perf_counter()
                text += ch[0]["delta"]["content"]
    except Exception as e:
        with _lock: print(f"    [ERR] {e}")
        return None, "", None, None
    return ((first - t0) * 1000 if first else None), text, ptok, ctok

def run_session(sid):
    hist = [{"role": "system", "content": f"agent {sid}. " + _filler(INIT_PROMPT_CHARS, sid)}]
    for turn in range(NUM_TURNS):
        hist.append({"role": "user", "content": f"[s{sid} t{turn}] " + _filler(TURN_GROWTH_CHARS, sid*100+turn+1)})
        ttft, text, ptok, ctok = do_chat(hist)
        ctx = ptok if ptok else int(sum(len(m["content"]) for m in hist) / CHARS_PER_TOKEN)
        recompute = (ctx - ctok) if (ctok is not None) else None
        with _lock:
            samples.append({
                "sid": sid, "turn": turn,
                "prompt_tokens": ctx, "cached_tokens": ctok,
                "recompute_tokens": recompute,
                "recompute_frac": round(recompute / ctx, 3) if (recompute is not None and ctx) else None,
                "ttft_ms": round(ttft, 1) if ttft else None,
                "transfer_ms_est": round(transfer_ms(ctx), 1),
            })
        hist.append({"role": "assistant", "content": text or "(none)"})
        time.sleep(TOOL_DELAY_SEC)
        hist.append({"role": "user", "content": f"[s{sid} tool{turn}] " + _filler(TURN_GROWTH_CHARS//2, sid*100+500+turn)})

# ─── run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"E4: per-turn re-prefill & re-transfer cost  C={CONCURRENCY}")
print("=" * 70)
print(f"  router={ROUTER_URL}  turns={NUM_TURNS} tool_delay={TOOL_DELAY_SEC}s")
print("=" * 70)
ths = [threading.Thread(target=run_session, args=(s,), daemon=True) for s in range(CONCURRENCY)]
for t in ths: t.start(); time.sleep(0.25)
for t in ths: t.join()

n_ctok = sum(1 for s in samples if s["cached_tokens"] is not None)
if n_ctok == 0:
    print("\n⚠ cached_tokens=None — 서버 --enable-cache-report 없음. recompute 분해 불가.")

# ─── per-turn aggregate (median across sessions) ──────────────────────────────
from collections import defaultdict
byturn = defaultdict(list)
for s in samples: byturn[s["turn"]].append(s)
def medf(rows, k):
    vs = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return statistics.median(vs) if vs else None

print("\n" + "=" * 70)
print("RESULT — turn별 재계산/재전송 (median)")
print("=" * 70)
print(f"  {'turn':>4} {'prompt':>8} {'cached':>8} {'recomp':>8} {'recomp%':>8} {'TTFT':>8} {'xfer_est':>9}")
turn_rows = []
for t in sorted(byturn):
    rows = byturn[t]
    pt = medf(rows, "prompt_tokens"); ct = medf(rows, "cached_tokens")
    rc = medf(rows, "recompute_tokens"); rf = medf(rows, "recompute_frac")
    tf = medf(rows, "ttft_ms"); xf = medf(rows, "transfer_ms_est")
    turn_rows.append({"turn": t, "prompt_tokens": pt, "cached_tokens": ct,
                      "recompute_tokens": rc, "recompute_frac": rf, "ttft_ms": tf, "transfer_ms_est": xf})
    g = lambda v, w=8: f"{v:>{w}.0f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
    print(f"  {t:>4} {g(pt)} {g(ct)} {g(rc)} "
          f"{(f'{rf*100:>7.0f}%' if rf is not None else '      —')} {g(tf)} {g(xf,9)}")

# turn>=1 요약 (turn0은 cold)
later = [r for r in turn_rows if r["turn"] >= 1 and r["recompute_frac"] is not None]
if later:
    rfs = [r["recompute_frac"] for r in later]
    print(f"\n  turn≥1 재계산 비율: median={statistics.median(rfs)*100:.0f}%  "
          f"max={max(rfs)*100:.0f}%  (= prefill 중 history 재계산 몫)")
    rcs = [r["recompute_tokens"] for r in later if r["recompute_tokens"] is not None]
    if rcs:
        print(f"  turn≥1 재계산 토큰: median={statistics.median(rcs):.0f}  "
              f"(이게 0이면 prefix 완전 재사용; 클수록 history 재계산)")
    # 재전송: D free로 매 turn 전체 prompt KV 재전송
    last = turn_rows[-1]
    if last["prompt_tokens"]:
        print(f"  마지막 turn 재전송량 ≈ {last['prompt_tokens']:.0f} tok "
              f"({last['prompt_tokens']*KV_BYTES_PER_TOKEN/1024**2:.0f} MB, est {last['transfer_ms_est']:.0f}ms)")

# ─── save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"{_prefix}e4_reprefill_cost.json")
with open(out, "w") as f:
    json.dump({"config": {"concurrency": CONCURRENCY, "num_turns": NUM_TURNS,
                          "tool_delay_sec": TOOL_DELAY_SEC, "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
                          "pent2_fixed_ms": PENT2_FIXED_MS, "pent2_ms_per_ktok": PENT2_MS_PER_KTOK},
               "per_turn": turn_rows, "samples": samples}, f, indent=2, ensure_ascii=False)
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
    ts = [r["turn"] for r in turn_rows]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [2, 1]})
    # panel1: cached vs recompute stacked (tokens)
    cached = [r["cached_tokens"] or 0 for r in turn_rows]
    recomp = [r["recompute_tokens"] or 0 for r in turn_rows]
    ax1.bar(ts, cached, color="#4575b4", label="cached (prefix reuse)")
    ax1.bar(ts, recomp, bottom=cached, color="#d73027", label="recomputed (re-prefill)")
    ax1.set_ylabel("prompt tokens")
    ax1.set_title(f"E4: per-turn re-prefill — {MODEL_SLUG} C={CONCURRENCY}\n"
                  "red = history KV recomputed each turn (the inefficiency)")
    ax1.legend(fontsize=9); ax1.grid(alpha=0.3, axis="y")
    # panel2: TTFT + transfer est
    ax2.plot(ts, [r["ttft_ms"] for r in turn_rows], "-o", color="#333", label="measured TTFT")
    ax2.plot(ts, [r["transfer_ms_est"] for r in turn_rows], "--s", color="#1a9850", label="re-transfer est (§2.11)")
    ax2.set_xlabel("turn"); ax2.set_ylabel("ms")
    ax2.legend(fontsize=9); ax2.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{_prefix}e4_reprefill_cost.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"plot saved: {png}")

make_plot()
