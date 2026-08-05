"""
E5: decode-offload-kvcache on/off baseline  — research.md §2.14 E5
======================================================================
"기존 부분 해법"의 대비선: `--disaggregation-decode-enable-offload-kvcache`를 켜면
D가 생성한 KV가 storage(host/SSD)로 offload되어 P가 다음 turn에 재사용 → turn 2+의
history 재계산(E4)이 줄어드는가?

이건 우리 연구(생성 KV를 P-HBM에 tool-call-aware하게 보존)의 *기존 baseline*이다:
  - decode-offload: 생성 KV를 host/SSD로 (느린 tier, write_through 비용, #11016 불안정)
  - 우리 제안: 생성 KV를 P-HBM으로 (NVLink, tool-duration-aware)
E5는 decode-offload가 얼마나 재계산을 줄이고 turn 2+ TTFT를 낮추는지(또는 #11016로
불안정한지)를 측정해, 우리 제안이 넘어서야 할 선을 긋는다.

────────────────────────────────────────────────────────────────────────────
실행 (off / on 두 번, 서버 재시작으로 토글)
────────────────────────────────────────────────────────────────────────────
  # OFF (baseline): 생성 KV offload 없음
  D_OFFLOAD_KVCACHE=0 bash scripts/sglang/start_1P_1D_breakeven.sh
  RUN_TAG=offload_off python benchmark/e5_decode_offload.py

  # ON: 생성 KV → storage → P 재사용  (#11016 CUDA error / hang 주의)
  D_OFFLOAD_KVCACHE=1 bash scripts/sglang/start_1P_1D_breakeven.sh
  RUN_TAG=offload_on D_LOG=logs/sglang_1p1d/d1.log python benchmark/e5_decode_offload.py

  # 비교 (두 JSON diff)
  COMPARE=results/.../offload_off_e5_decode_offload.json,results/.../offload_on_e5_decode_offload.json \
    python benchmark/e5_decode_offload.py

측정 (서버 --enable-cache-report 필요):
  turn별 prompt_tokens / cached_tokens / recompute / TTFT  (E4와 동일)
  + 안정성: request 실패, hang(timeout), D_LOG의 #retracted-req / CUDA error (#11016 신호)
핵심 신호:
  offload ON이 작동하면 → cached_tokens↑ (P가 D 생성 KV 재사용) → recompute↓ → turn2+ TTFT↓
  offload가 불안정하면 → request 실패 / hang / CUDA error 로그

Environment variables:
  ROUTER_URL/MODEL/NUM_TURNS/TURN_OUTPUT_TOKENS/INIT_PROMPT_CHARS/TURN_GROWTH_CHARS/
  TOOL_DELAY_SEC/CONCURRENCY  — E4와 동일 (CONCURRENCY 기본 1)
  RUN_TAG           결과 prefix (offload_off / offload_on)
  D_LOG             decode 로그 (offload ON에서 #retracted/CUDA error 포착용)
  COMPARE           "off.json,on.json" 주면 비교만 하고 종료
  KV_BYTES_PER_TOKEN / PENT2_*  전송 모델 (§2.11)
"""

import json
import os
import re
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
RUN_TAG       = os.environ.get("RUN_TAG", "")
D_LOG         = os.environ.get("D_LOG", "")
COMPARE       = os.environ.get("COMPARE", "")
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
PENT2_FIXED_MS     = float(os.environ.get("PENT2_FIXED_MS", "38"))
PENT2_MS_PER_KTOK  = float(os.environ.get("PENT2_MS_PER_KTOK", "3.35"))
OUT_TAG       = RUN_TAG
CHARS_PER_TOKEN = 3.0

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{OUT_TAG}_" if OUT_TAG else ""

# ─── COMPARE 모드: 두 JSON diff하고 종료 ──────────────────────────────────────
def compare_mode(paths):
    a, b = [json.load(open(p.strip())) for p in paths.split(",")]
    ta = a["config"].get("run_tag") or "A"; tb = b["config"].get("run_tag") or "B"
    print("=" * 70); print(f"E5 비교: {ta}  vs  {tb}"); print("=" * 70)
    da = {r["turn"]: r for r in a["per_turn"]}; db = {r["turn"]: r for r in b["per_turn"]}
    print(f"  {'turn':>4} | {ta[:14]:>14} {'cached':>7} {'recomp%':>8} | {tb[:14]:>14} {'cached':>7} {'recomp%':>8} | {'ΔTTFT':>8}")
    impr = []
    for t in sorted(set(da) & set(db)):
        ra, rb = da[t], db[t]
        f = lambda v, w=7: f"{v:>{w}.0f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
        pf = lambda v: f"{v*100:>7.0f}%" if isinstance(v, (int, float)) else f"{'—':>8}"
        dttft = (rb["ttft_ms"] - ra["ttft_ms"]) if (ra.get("ttft_ms") and rb.get("ttft_ms")) else None
        if t >= 1 and dttft is not None: impr.append(dttft)
        print(f"  {t:>4} | {f(ra.get('ttft_ms'),14)} {f(ra.get('cached_tokens'))} {pf(ra.get('recompute_frac'))}"
              f" | {f(rb.get('ttft_ms'),14)} {f(rb.get('cached_tokens'))} {pf(rb.get('recompute_frac'))}"
              f" | {f(dttft,8)}")
    print(f"\n  안정성: {ta} fails={a['summary'].get('request_fails')} hang={a['summary'].get('hang')}"
          f"  |  {tb} fails={b['summary'].get('request_fails')} hang={b['summary'].get('hang')}"
          f" cuda_err={b['summary'].get('cuda_errors')}")
    if impr:
        print(f"  turn≥1 평균 ΔTTFT ({tb}−{ta}) = {statistics.mean(impr):+.0f} ms "
              f"({'offload 이득' if statistics.mean(impr)<0 else 'offload 손해/무효'})")
    raise SystemExit(0)

if COMPARE:
    compare_mode(COMPARE)

# ─── 측정 ─────────────────────────────────────────────────────────────────────
samples: list[dict] = []
_lock = threading.Lock()
_stop = threading.Event()
log_state = {"retracted_cum": 0, "cuda_errors": 0}

def _filler(n, seed):
    unit = f"ctx-{seed:05d} the agent observed the environment and planned next actions. "
    return (unit * (n // len(unit) + 1))[:n]

def do_chat(messages):
    payload = {"model": MODEL, "messages": messages, "max_tokens": TURN_OUTPUT_TOKENS,
               "temperature": 0, "stream": True, "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; text = ""; ptok = ctok = None
    try:
        resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=300)
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
        return None, "", None, None, str(e)[:100]
    return ((first - t0) * 1000 if first else None), text, ptok, ctok, None

def run_session(sid):
    hist = [{"role": "system", "content": f"agent {sid}. " + _filler(INIT_PROMPT_CHARS, sid)}]
    for turn in range(NUM_TURNS):
        if _stop.is_set(): return
        hist.append({"role": "user", "content": f"[s{sid} t{turn}] " + _filler(TURN_GROWTH_CHARS, sid*100+turn+1)})
        ttft, text, ptok, ctok, err = do_chat(hist)
        ctx = ptok if ptok else int(sum(len(m["content"]) for m in hist) / CHARS_PER_TOKEN)
        recompute = (ctx - ctok) if (ctok is not None) else None
        with _lock:
            samples.append({
                "sid": sid, "turn": turn, "prompt_tokens": ctx, "cached_tokens": ctok,
                "recompute_tokens": recompute,
                "recompute_frac": round(recompute / ctx, 3) if (recompute is not None and ctx) else None,
                "ttft_ms": round(ttft, 1) if ttft else None,
                "fail": err,
            })
        hist.append({"role": "assistant", "content": text or "(none)"})
        _stop.wait(TOOL_DELAY_SEC)
        hist.append({"role": "user", "content": f"[s{sid} tool{turn}] " + _filler(TURN_GROWTH_CHARS//2, sid*100+500+turn)})

# ─── D_LOG tailer: #retracted-req + CUDA error (#11016 신호) ───────────────────
_RE_RETRACT = re.compile(r"#retracted-req:\s*(\d+)")
_RE_CUDA = re.compile(r"CUDA error|illegal memory access|an illegal", re.I)
def log_tailer():
    try: f = open(D_LOG)
    except OSError: return
    f.seek(0, os.SEEK_END)
    while not _stop.is_set():
        line = f.readline()
        if not line: _stop.wait(0.2); continue
        m = _RE_RETRACT.search(line)
        if m and int(m.group(1)) > 0: log_state["retracted_cum"] += int(m.group(1))
        if _RE_CUDA.search(line): log_state["cuda_errors"] += 1

# ─── run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"E5: decode-offload baseline  [{RUN_TAG or 'default'}]  C={CONCURRENCY}")
print("=" * 70)
print(f"  router={ROUTER_URL}  turns={NUM_TURNS} tool_delay={TOOL_DELAY_SEC}s")
print("=" * 70)
if D_LOG:
    threading.Thread(target=log_tailer, daemon=True).start()
ths = [threading.Thread(target=run_session, args=(s,), daemon=True) for s in range(CONCURRENCY)]
for t in ths: t.start(); time.sleep(0.25)
for t in ths: t.join()
_stop.set(); time.sleep(0.3)

n_ctok = sum(1 for s in samples if s["cached_tokens"] is not None)
if n_ctok == 0:
    print("\n⚠ cached_tokens=None — 서버 --enable-cache-report 없음.")

from collections import defaultdict
byturn = defaultdict(list)
for s in samples: byturn[s["turn"]].append(s)
def medf(rows, k):
    vs = [r[k] for r in rows if isinstance(r.get(k), (int, float))]
    return statistics.median(vs) if vs else None

fails = sum(1 for s in samples if s.get("fail"))
hang = any("timeout" in (s.get("fail") or "").lower() for s in samples)

print("\n" + "=" * 70)
print(f"RESULT [{RUN_TAG or 'default'}]")
print("=" * 70)
print(f"  {'turn':>4} {'prompt':>8} {'cached':>8} {'recomp%':>8} {'TTFT':>9}")
turn_rows = []
for t in sorted(byturn):
    rows = byturn[t]
    pt = medf(rows, "prompt_tokens"); ct = medf(rows, "cached_tokens")
    rf = medf(rows, "recompute_frac"); tf = medf(rows, "ttft_ms")
    turn_rows.append({"turn": t, "prompt_tokens": pt, "cached_tokens": ct,
                      "recompute_frac": rf, "recompute_tokens": medf(rows, "recompute_tokens"),
                      "ttft_ms": tf})
    g = lambda v, w=8: f"{v:>{w}.0f}" if isinstance(v, (int, float)) else f"{'—':>{w}}"
    print(f"  {t:>4} {g(pt)} {g(ct)} {(f'{rf*100:>7.0f}%' if rf is not None else '       —')} {g(tf,9)}")

later = [r for r in turn_rows if r["turn"] >= 1 and r["recompute_frac"] is not None]
if later:
    print(f"\n  turn≥1 재계산 비율 median = {statistics.median([r['recompute_frac'] for r in later])*100:.0f}%")
print(f"  안정성: request fails={fails}  hang={hang}  "
      f"retractions={log_state['retracted_cum']}  cuda_errors={log_state['cuda_errors']}")
if log_state["cuda_errors"] > 0:
    print("  ⚠ CUDA error 감지 — #11016 (decode-offload 불안정) 발현.")

# ─── save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"{_prefix}e5_decode_offload.json")
with open(out, "w") as f:
    json.dump({"config": {"run_tag": RUN_TAG, "concurrency": CONCURRENCY, "num_turns": NUM_TURNS,
                          "tool_delay_sec": TOOL_DELAY_SEC},
               "summary": {"request_fails": fails, "hang": hang,
                           "retractions": log_state["retracted_cum"], "cuda_errors": log_state["cuda_errors"],
                           "recompute_frac_median": statistics.median([r["recompute_frac"] for r in later])
                                                    if later else None},
               "per_turn": turn_rows, "samples": samples}, f, indent=2, ensure_ascii=False)
print(f"\n결과 저장: {out}")
print(f"\n비교: COMPARE={OUT_DIR}/offload_off_e5_decode_offload.json,{out}  python benchmark/e5_decode_offload.py")
