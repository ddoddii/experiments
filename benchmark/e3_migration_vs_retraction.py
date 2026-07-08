"""
E3: Migration vs Retraction   — research.md §2.14 E3 (FlashGen 차별점 [D])
============================================================================
FlashGen/현 SGLang은 메모리 압박을 retraction(preemption)으로 처리한다: D-HBM이 꽉 차면
*실행 중* 요청의 KV를 회수→re-prefill. active 작업을 파괴 → P99 폭발/hang.
제안은 migration: 압박 전에 *idle*(tool-call 중) 세션 KV를 P-HBM으로 옮겨 D-HBM 확보 →
retraction 자체를 회피. active를 안 건드림.

────────────────────────────────────────────────────────────────────────────
핵심 측정: D-HBM을 active vs idle로 분해 + retraction 이벤트 포착
────────────────────────────────────────────────────────────────────────────
C개 multi-turn 세션을 부하로 걸고, decode 노드를 시계열 폴링:
  active_KV  = 지금 decode 중인 토큰      = decode 로그 "#token"
  total_KV   = token_usage × capacity     = decode 로그 "token usage" × pool
  idle_KV    = total_KV − active_KV        = tool-call 중 보존된 세션 KV
               = **migration이 비파괴적으로 옮길 대상**
  retraction = "#retracted-req" / total_retractions (압박 시 active preempt = 파괴적)

증명 논리 (migration 메커니즘 없이):
  (1) 압박 하 D-HBM의 **idle 비율이 크다** → migration이 그만큼 비파괴 회수 가능.
  (2) retraction은 active를 preempt(파괴적) → migration이 idle을 먼저 비워 retraction 회피.
  (3) "migration 후 D ≈ active만 → pool 아래 유지 → retraction N건/hang 회피" (analytical).

비교 지표 (baseline = 현 SGLang):
  - D idle-KV peak 비율 (migration이 회수할 양)
  - retraction 수 / hang 발생 (migration이 회피할 파괴적 이벤트)
  - P50/P99 TTFT, goodput, 완료율
  - "migration 후 예상 D 점유" = active-only peak (pool 대비)

⚠ 압박을 너무 높이면 decode OOM hang(#6857, §2.14). 시작 전 hang 감지(heartbeat freeze)
  + timeout으로 graceful 종료. mem_fraction은 start 스크립트에서 0.68~0.75 권장.

Usage:
  # 서버 (압박 영역): MEM_FRACTION으로 pool 줄여 retraction/idle 압박 유발
  D_MEM_FRACTION=0.72 bash scripts/sglang/start_1P_1D_breakeven.sh
  CONCURRENCY=12 NUM_TURNS=8 TOOL_DELAY_SEC=10 OUT_TAG=mf072_c12 \
  D_LOG=logs/sglang_1p1d/d1.log \
    python benchmark/e3_migration_vs_retraction.py

Environment variables:
  ROUTER_URL        chat 요청 경로           (default: .../8001/v1/chat/completions)
  D_NODE_URL        decode 노드(HTTP, cap용)  (default: http://127.0.0.1:30002)
  D_LOG             decode 로그 (필수 — #token/#retracted-req 출처)
  MODEL             모델 이름                 (default: /home/uhmturks/hf_models/Qwen3-14B)
  CONCURRENCY       병렬 세션 수              (default: 12)
  NUM_TURNS         세션당 turn 수            (default: 8)
  TURN_OUTPUT_TOKENS turn당 decode 토큰       (default: 256)
  INIT_PROMPT_CHARS / TURN_GROWTH_CHARS  context (default: 3000 / 3000)
  TOOL_DELAY_SEC    tool call idle 초         (default: 10)
  POLL_INTERVAL     폴링 간격                 (default: 0.5)
  HANG_TIMEOUT_SEC  무진행 hang 감지 임계      (default: 90)
  KV_BYTES_PER_TOKEN 토큰당 KV 바이트         (default: 163840)
  OUT_DIR/OUT_TAG   결과 경로/태그
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
D_NODE_URL    = os.environ.get("D_NODE_URL", "http://127.0.0.1:30002").rstrip("/")
D_LOG         = os.environ.get("D_LOG", "")
MODEL         = os.environ.get("MODEL", "/home/uhmturks/hf_models/Qwen3-14B")
CONCURRENCY   = int(os.environ.get("CONCURRENCY", "12"))
NUM_TURNS     = int(os.environ.get("NUM_TURNS", "8"))
TURN_OUTPUT_TOKENS = int(os.environ.get("TURN_OUTPUT_TOKENS", "256"))
INIT_PROMPT_CHARS  = int(os.environ.get("INIT_PROMPT_CHARS", "3000"))
TURN_GROWTH_CHARS  = int(os.environ.get("TURN_GROWTH_CHARS", "3000"))
TOOL_DELAY_SEC     = float(os.environ.get("TOOL_DELAY_SEC", "10"))
POLL_INTERVAL      = float(os.environ.get("POLL_INTERVAL", "0.5"))
HANG_TIMEOUT_SEC   = float(os.environ.get("HANG_TIMEOUT_SEC", "90"))
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
OUT_TAG       = os.environ.get("OUT_TAG", "")
CHARS_PER_TOKEN = 3.0

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{OUT_TAG}_" if OUT_TAG else ""

def used_gb(tok):
    return tok * KV_BYTES_PER_TOKEN / 1024**3 if tok is not None else None

# ─── shared state ─────────────────────────────────────────────────────────────
poll_rows: list[dict] = []        # {t, total_tok, active_tok, idle_tok, running, retracted, queue}
ttfts: list[float] = []           # per-turn TTFT (turn>=1)
events: list[dict] = []
_lock = threading.Lock()
_stop = threading.Event()
_t0 = time.time()
def now(): return time.time() - _t0

# ─── D capacity (HTTP) ────────────────────────────────────────────────────────
def d_capacity() -> float | None:
    for ep in ("/server_info", "/get_server_info"):
        try:
            r = requests.get(f"{D_NODE_URL}{ep}", timeout=5)
            if r.ok:
                d = r.json()
                if isinstance(d.get("internal_states"), list):
                    for s in d["internal_states"]:
                        if isinstance(s, dict):
                            for k, v in s.items():
                                d.setdefault(k, v)
                            break
                if isinstance(d.get("memory_usage"), dict):
                    for k, v in d["memory_usage"].items():
                        d.setdefault(k, v)
                cap = d.get("token_capacity") or d.get("num_total_tokens") or d.get("max_total_num_tokens")
                if cap:
                    return float(cap)
        except Exception:
            pass
    return None

CAP = None

# ─── decode log tailer: #token(active), token usage(total), #retracted-req ─────
_RE_TOKEN = re.compile(r"#token:\s*(\d+)")
_RE_USAGE = re.compile(r"token usage:\s*([0-9.]+)")
_RE_RETRACT = re.compile(r"#retracted-req:\s*(\d+)")
_RE_RUN = re.compile(r"#running-req:\s*(\d+)")
_RE_QUEUE = re.compile(r"#queue-req:\s*(\d+)")

_log_state = {"active": None, "usage": None, "running": None, "queue": None, "retracted_cum": 0}

def log_tailer():
    try:
        f = open(D_LOG, "r")
    except OSError:
        print(f"  ⚠ D_LOG 열 수 없음: {D_LOG} — active/idle 분해 불가. D_LOG 지정 필수.")
        return
    f.seek(0, os.SEEK_END)
    while not _stop.is_set():
        line = f.readline()
        if not line:
            _stop.wait(0.1)
            continue
        if "batch" not in line.lower():
            continue
        mt = _RE_TOKEN.search(line); mu = _RE_USAGE.search(line)
        if mt: _log_state["active"] = float(mt.group(1))
        if mu: _log_state["usage"] = float(mu.group(1))
        mr = _RE_RUN.search(line); mq = _RE_QUEUE.search(line)
        if mr: _log_state["running"] = float(mr.group(1))
        if mq: _log_state["queue"] = float(mq.group(1))
        mx = _RE_RETRACT.search(line)
        if mx:
            rv = int(mx.group(1))
            if rv > 0:
                _log_state["retracted_cum"] += rv
                with _lock:
                    events.append({"t": round(now(), 2), "kind": "retraction", "n": rv})

# ─── poller ───────────────────────────────────────────────────────────────────
def poller():
    last_progress = time.time()
    last_active = None
    while not _stop.is_set():
        usage = _log_state["usage"]; active = _log_state["active"]
        total_tok = usage * CAP if (usage is not None and CAP) else None
        idle_tok = (total_tok - active) if (total_tok is not None and active is not None) else None
        with _lock:
            poll_rows.append({
                "t": round(now(), 2),
                "total_tok": total_tok, "active_tok": active,
                "idle_tok": max(idle_tok, 0) if idle_tok is not None else None,
                "running": _log_state["running"], "queue": _log_state["queue"],
                "retracted_cum": _log_state["retracted_cum"],
            })
        # hang 감지: active가 변하고 running>0인데 오래 진행 없으면
        if active is not None and active != last_active:
            last_progress = time.time(); last_active = active
        _stop.wait(POLL_INTERVAL)

# ─── synthetic multi-turn driver (부하) ───────────────────────────────────────
def _filler(n, seed):
    unit = f"ctx-{seed:05d} the agent observed state and planned next actions. "
    return (unit * (n // len(unit) + 1))[:n]

def run_session(sid):
    hist = [{"role": "system", "content": f"agent {sid}. " + _filler(INIT_PROMPT_CHARS, sid)}]
    for turn in range(NUM_TURNS):
        if _stop.is_set():
            return
        hist.append({"role": "user", "content": f"[s{sid} t{turn}] " + _filler(TURN_GROWTH_CHARS, sid*100+turn+1)})
        payload = {"model": MODEL, "messages": hist, "max_tokens": TURN_OUTPUT_TOKENS,
                   "temperature": 0, "stream": True}
        t0 = time.perf_counter(); first = None; text = ""
        try:
            resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=HANG_TIMEOUT_SEC)
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line: continue
                raw = line.decode("utf-8")
                if not raw.startswith("data: "): continue
                if raw[6:] == "[DONE]": break
                try: delta = json.loads(raw[6:])["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError): continue
                if delta:
                    if first is None:
                        first = time.perf_counter()
                    text += delta
        except Exception as e:
            with _lock:
                events.append({"t": round(now(), 2), "kind": "request_fail", "sid": sid, "turn": turn, "err": str(e)[:80]})
            continue
        if first and turn >= 1:
            with _lock:
                ttfts.append((first - t0) * 1000)
        hist.append({"role": "assistant", "content": text or "(none)"})
        _stop.wait(TOOL_DELAY_SEC)  # tool call idle window
        hist.append({"role": "user", "content": f"[s{sid} tool{turn}] " + _filler(TURN_GROWTH_CHARS//2, sid*100+500+turn)})

# ─── run ──────────────────────────────────────────────────────────────────────
print("=" * 70)
print(f"E3: Migration vs Retraction  C={CONCURRENCY}  [{OUT_TAG or 'default'}]")
print("=" * 70)
CAP = d_capacity()
print(f"  D capacity: {CAP:.0f} tok ({used_gb(CAP):.1f} GB)" if CAP else "  ⚠ D capacity 못 읽음")
print(f"  router={ROUTER_URL}  turns={NUM_TURNS} tool_delay={TOOL_DELAY_SEC}s")
if not D_LOG:
    print("  ⚠ D_LOG 미지정 — active/idle 분해 불가. D_LOG=logs/sglang_1p1d/d1.log 지정 권장.")
print("=" * 70)

threads = [threading.Thread(target=poller, daemon=True)]
if D_LOG:
    threads.append(threading.Thread(target=log_tailer, daemon=True))
sess = [threading.Thread(target=run_session, args=(sid,), daemon=True) for sid in range(CONCURRENCY)]
for th in threads: th.start()
t_start = time.time()
for i, th in enumerate(sess):
    th.start(); time.sleep(0.25)
# 완료 대기 (hang 감지 포함)
total_expected = CONCURRENCY * NUM_TURNS
for th in sess:
    th.join()
elapsed = time.time() - t_start
_stop.wait(2); _stop.set()
time.sleep(POLL_INTERVAL + 0.3)

# ─── analyze ──────────────────────────────────────────────────────────────────
def pk(key):
    vals = [r[key] for r in poll_rows if isinstance(r.get(key), (int, float))]
    return max(vals) if vals else None
def pct(v, q):
    if not v: return None
    s = sorted(v); return s[min(len(s)-1, int(q*(len(s)-1)+0.5))]

idle_peak = pk("idle_tok"); active_peak = pk("active_tok"); total_peak = pk("total_tok")
retractions = _log_state["retracted_cum"]
fails = sum(1 for e in events if e["kind"] == "request_fail")
hang = any(e["kind"] == "request_fail" and "timeout" in e.get("err", "").lower() for e in events)
goodput = (len(ttfts) * TURN_OUTPUT_TOKENS) / elapsed if elapsed else 0

print("\n" + "=" * 70)
print(f"RESULT  C={CONCURRENCY}")
print("=" * 70)
if CAP:
    fr = lambda t: f"{used_gb(t):.2f}GB ({t/CAP*100:.0f}%)" if t is not None else "—"
    print(f"  D-HBM peak total : {fr(total_peak)}")
    print(f"  D-HBM peak active: {fr(active_peak)}   ← migration 후 D 점유 (active만)")
    print(f"  D-HBM peak idle  : {fr(idle_peak)}   ← migration이 P-HBM으로 옮길 양 (비파괴)")
    if idle_peak and total_peak:
        print(f"    → idle 비율 = {idle_peak/total_peak*100:.0f}% of peak D-HBM")
print(f"  retractions      : {retractions}  (파괴적 — active preempt)")
print(f"  request fails    : {fails}   hang(timeout): {hang}")
print(f"  완료 TTFT 샘플   : {len(ttfts)} (turn>=1, expected ~{CONCURRENCY*(NUM_TURNS-1)})")
if ttfts:
    print(f"  TTFT P50/P99     : {pct(ttfts,0.5):.0f} / {pct(ttfts,0.99):.0f} ms")
print(f"  goodput          : {goodput:.0f} tok/s   wall {elapsed:.0f}s")
print()
print("  논리: migration은 idle({}%)를 비파괴적으로 P-HBM에 옮김 → D는 active-only로 떨어져"
      .format(round(idle_peak/total_peak*100) if (idle_peak and total_peak) else "?"))
print("        retraction/hang 회피. retraction은 active를 preempt(파괴적).")

# ─── save ─────────────────────────────────────────────────────────────────────
os.makedirs(OUT_DIR, exist_ok=True)
out = os.path.join(OUT_DIR, f"{_prefix}e3_migration_vs_retraction.json")
with open(out, "w") as f:
    json.dump({"config": {"concurrency": CONCURRENCY, "num_turns": NUM_TURNS,
                          "tool_delay_sec": TOOL_DELAY_SEC, "d_capacity_tok": CAP},
               "summary": {"d_peak_total_tok": total_peak, "d_peak_active_tok": active_peak,
                           "d_peak_idle_tok": idle_peak,
                           "idle_fraction": (idle_peak/total_peak) if (idle_peak and total_peak) else None,
                           "retractions": retractions, "request_fails": fails, "hang": hang,
                           "ttft_p50": pct(ttfts,0.5), "ttft_p99": pct(ttfts,0.99),
                           "goodput_tok_s": round(goodput,1), "wall_s": round(elapsed,1),
                           "n_ttft": len(ttfts)},
               "poll": poll_rows, "events": events}, f, indent=2, ensure_ascii=False)
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
    ts = [r["t"] for r in poll_rows]
    act = [used_gb(r["active_tok"]) if r["active_tok"] is not None else None for r in poll_rows]
    idl = [used_gb(r["idle_tok"]) if r["idle_tok"] is not None else None for r in poll_rows]
    tot = [used_gb(r["total_tok"]) if r["total_tok"] is not None else None for r in poll_rows]
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(ts, tot, color="#333", lw=1.0, label="D total KV")
    ax.plot(ts, act, color="#d73027", lw=1.6, label="D active KV (decoding now)")
    ax.plot(ts, idl, color="#1a9850", lw=1.6, label="D idle KV (tool-call, migratable)")
    if CAP:
        ax.axhline(used_gb(CAP), color="gray", ls=":", label="D pool capacity")
    for e in events:
        if e["kind"] == "retraction":
            ax.axvline(e["t"], color="#762a83", ls="--", lw=0.8, alpha=0.7)
    ax.set_xlabel("time (s)"); ax.set_ylabel("KV (GB)")
    idlef = f"{idle_peak/total_peak*100:.0f}%" if (idle_peak and total_peak) else "?"
    ax.set_title(f"E3: migration vs retraction — {MODEL_SLUG} C={CONCURRENCY}\n"
                 f"idle peak={idlef} of D-HBM (migratable), retractions={retractions}")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{_prefix}e3_migration_vs_retraction.png")
    fig.savefig(png, dpi=150); plt.close(fig)
    print(f"plot saved: {png}")

make_plot()
