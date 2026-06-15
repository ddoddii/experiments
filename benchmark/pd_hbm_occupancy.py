"""
PD Node KV/HBM Occupancy Monitor  (research.md §2.12 — "P node idle HBM" 전제 검증)
================================================================================
1P1D disaggregation에서 P node와 D node의 KV cache 점유를 **동시에** 시계열로
폴링하고, P→D KV 전송 시점을 추정해 함께 기록·플롯한다.

연구 동기 (FlashGen 대비 차별점):
  "decode phase 동안 P node HBM은 idle하니 Tier2로 쓰자"가 연구의 핵심 전제.
  이 도구로 (1) P node HBM 여유분 = Tier2 가용 용량, (2) D node HBM 압박과의
  시간 상관 (correlated demand 리스크), (3) multi-turn에서 context가 커지며
  KV가 어떻게 쌓이는지를 측정한다.

multi-turn agent 가정:
  --drive(DRIVE=1) 모드는 합성 multi-turn 대화를 생성한다. 매 turn마다
  (이전 prompt + decode output + tool 결과)가 누적되어 context가 단조 증가하고,
  turn 사이에 TOOL_DELAY_SEC만큼 tool call idle window를 둔다.
  → 각 turn의 prefill이 점점 커지고, P→D 전송량(=D 점프 크기)이 turn마다 증가.

P→D 전송 시점 추정 (3가지 신호, 강건한 순):
  1. D.num_used_tokens 양(+)의 점프 ≥ JUMP_THRESHOLD tokens
     = P가 prefill한 KV가 D에 도착(전송 완료). decode는 토큰당 +1이라 임계값으로 분리.
     multi-turn에서는 점프 크기 = 그 turn의 전체 context 길이 → turn마다 증가.
  2. D node 로그(--decode-log)의 "#transfer-req: N" / "#prealloc-req: N"
     = 전송 in-flight 카운터 (있으면 전송 window를 직접 표시).
  3. /metrics에 disagg 관련 metric이 노출되면 자동 발견해 함께 기록.

기본 SGLang PD 동작 메모 (로그 관측): turn 종료 후 D는 token usage→0으로 복귀
(turn 사이 KV 미보존). 세션 KV 지속은 P node radix/hicache 쪽 → "P-HBM as Tier2"
설계의 근거.

⚠ live used_tokens 읽기: SGLang은 enable_metrics=False면 /metrics가 비어 있어
  HTTP로 used_tokens/usage를 못 읽는다(capacity만 나옴). 두 가지 중 하나 필요:
    (a) 서버를 --enable-metrics로 재시작 (start 스크립트에 반영됨), 또는
    (b) D_LOG/P_LOG로 decode/prefill 로그를 넘기면 batch 라인에서 live 점유를 추출
        (재시작 불필요; 단 tool-call idle 중엔 batch 출력이 없어 값이 잠시 고정).
  → 가능하면 (a)+(b) 둘 다. (a)는 idle 중에도 폴링되고, (b)는 #transfer-req 신호 제공.

Usage:
  # 1) 서버 (1P1D, NVLink 권장): scripts/sglang/start_1P_1D_breakeven.sh  (--enable-metrics 포함)
  # 2-A) 합성 multi-turn 부하를 직접 구동하며 모니터 (자체 완결, 권장)
  DRIVE=1 NUM_TURNS=8 TOOL_DELAY_SEC=10 OUT_TAG=nvlink_c1 \
  D_LOG=logs/sglang_1p1d/d1.log P_LOG=logs/sglang_1p1d/p1.log \
    python benchmark/pd_hbm_occupancy.py

  # 2-B) 외부 벤치마크와 병행 (별 터미널에서 BFCL multi-turn 실행, 여기선 폴링만)
  DRIVE=0 DURATION_SEC=300 OUT_TAG=bfcl_c8 \
  D_LOG=logs/sglang_1p1d/d1.log P_LOG=logs/sglang_1p1d/p1.log \
    python benchmark/pd_hbm_occupancy.py
  # Ctrl-C로 중단 → CSV/플롯 저장

Environment variables:
  P_NODE_URL    P node 주소        (default: http://127.0.0.1:30000)
  D_NODE_URL    D node 주소        (default: http://127.0.0.1:30002)
  ROUTER_URL    drive 모드 요청 경로 (default: http://127.0.0.1:8001/v1/chat/completions)
  POLL_INTERVAL 폴링 간격 초       (default: 0.5)
  KV_BYTES_PER_TOKEN  토큰당 KV 바이트 (default: 163840 = Qwen3-14B)
  JUMP_THRESHOLD      D 전송 감지 임계 tokens (default: 150)
  OUT_DIR / OUT_TAG   결과 경로/태그 (default: results/sglang_hicache/<model>, "")
  D_LOG / P_LOG       decode/prefill 로그 경로 (transfer-req tailing, optional)
  ── drive 모드 ──
  DRIVE              1이면 합성 multi-turn 구동 (default: 0)
  MODEL              모델 이름        (default: /home/uhmturks/hf_models/Qwen3-14B)
  NUM_TURNS          turn 수          (default: 8)
  TOOL_DELAY_SEC     turn 사이 tool idle 초 (default: 10)
  TURN_OUTPUT_TOKENS turn당 decode 토큰 (default: 256)
  INIT_PROMPT_CHARS  turn 0 프롬프트 chars (default: 3000)
  TURN_GROWTH_CHARS  turn마다 추가 입력 chars (default: 2500)
  ── drive 비활성 시 ──
  DURATION_SEC       폴링 지속 초 (default: 0 = Ctrl-C까지)
"""

import csv
import json
import os
import re
import signal
import sys
import threading
import time

import requests

# ─── Paths ──────────────────────────────────────────────────────────────────
SCRIPT_DIR   = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
def _p(rel): return os.path.join(PROJECT_ROOT, rel)

# ─── Config ─────────────────────────────────────────────────────────────────
P_NODE_URL    = os.environ.get("P_NODE_URL", "http://127.0.0.1:30000").rstrip("/")
D_NODE_URL    = os.environ.get("D_NODE_URL", "http://127.0.0.1:30002").rstrip("/")
ROUTER_URL    = os.environ.get("ROUTER_URL", "http://127.0.0.1:8001/v1/chat/completions")
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "0.5"))
KV_BYTES_PER_TOKEN = int(os.environ.get("KV_BYTES_PER_TOKEN", "163840"))
JUMP_THRESHOLD = int(os.environ.get("JUMP_THRESHOLD", "150"))
OUT_TAG       = os.environ.get("OUT_TAG", "")
D_LOG         = os.environ.get("D_LOG", "")
P_LOG         = os.environ.get("P_LOG", "")

DRIVE             = os.environ.get("DRIVE", "0") == "1"
MODEL             = os.environ.get("MODEL", "/home/uhmturks/hf_models/Qwen3-14B")
NUM_TURNS         = int(os.environ.get("NUM_TURNS", "8"))
TOOL_DELAY_SEC    = float(os.environ.get("TOOL_DELAY_SEC", "10"))
TURN_OUTPUT_TOKENS = int(os.environ.get("TURN_OUTPUT_TOKENS", "256"))
INIT_PROMPT_CHARS = int(os.environ.get("INIT_PROMPT_CHARS", "3000"))
TURN_GROWTH_CHARS = int(os.environ.get("TURN_GROWTH_CHARS", "2500"))
DURATION_SEC      = float(os.environ.get("DURATION_SEC", "0"))

MODEL_SLUG = os.path.basename(MODEL.rstrip("/"))
OUT_DIR    = os.environ.get("OUT_DIR", _p(f"results/sglang_hicache/{MODEL_SLUG}"))
_prefix    = f"{OUT_TAG}_" if OUT_TAG else ""

# ─── Metrics scraping (compact, reused from sglang_hicache_exporter logic) ────
# disagg/transfer 관련 metric 이름은 버전마다 다르므로 정확히 가정하지 않고
# 아래 정규식으로 후보를 자동 발견한다.
_DISAGG_RE = re.compile(r"transfer|prealloc|bootstrap|disagg", re.I)

def _scrape_metrics(base: str) -> dict:
    """SGLang /metrics(Prometheus) → {metric_name: value} (sglang* 만)."""
    out = {}
    try:
        r = requests.get(f"{base}/metrics", timeout=2.0)
        if r.status_code != 200 or not r.text.strip():
            return out
        for line in r.text.splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split(None, 2)
            if len(parts) < 2:
                continue
            name_lbl, val = parts[0], parts[1]
            brace = name_lbl.find("{")
            name = name_lbl[:brace] if brace >= 0 else name_lbl
            if not name.startswith("sglang"):
                continue
            try:
                out[name] = float(val)
            except ValueError:
                pass
    except Exception:
        pass
    return out

def _server_info(base: str) -> dict:
    for path in ("/server_info", "/get_server_info"):
        try:
            r = requests.get(f"{base}{path}", timeout=2.0)
            if r.status_code == 200:
                d = r.json()
                break
        except Exception:
            d = None
    else:
        d = None
    if not isinstance(d, dict):
        return {}
    # flatten internal_states[0] + memory_usage
    if isinstance(d.get("internal_states"), list):
        for st in d["internal_states"]:
            if isinstance(st, dict):
                for k, v in st.items():
                    d.setdefault(k, v)
                break
    if isinstance(d.get("memory_usage"), dict):
        for k, v in d["memory_usage"].items():
            d.setdefault(k, v)
    return d

def _norm(name_re, metrics: dict):
    """첫 매칭 metric 값 반환."""
    for k, v in metrics.items():
        if name_re.search(k):
            return v
    return None

_USED_RE  = re.compile(r"num_used_tokens|used_tokens")
# token_usage는 prealloc/swa 변종(pending_prealloc_token_usage, swa_token_usage)과
# 헷갈리지 않게 prefix 직후로 앵커.
_USAGE_RE = re.compile(r"^sglang[:_]token_usage$")
_RUN_RE   = re.compile(r"running_req")
_QUEUE_RE = re.compile(r"(queue|waiting)_req")
_XFER_RE  = re.compile(r"transfer.*req|num_transfer")
_PRE_RE   = re.compile(r"prealloc.*req|num_prealloc")
# decode쪽 pending_prealloc_token_usage > 0 = P→D 전송 위해 D가 KV 슬롯 예약 중
# (전송 in-flight 신호; #transfer-req 로그보다 폴링으로 더 잘 잡힘).
_PREALLOC_USAGE_RE = re.compile(r"pending_prealloc_token_usage")

def scrape_node(base: str, node: str) -> dict:
    """한 노드의 KV 점유 스냅샷. HTTP(/server_info + /metrics)에 로그 fallback 병합.

    enable_metrics=False면 /metrics가 비어 used/usage가 None → 로그 batch 라인에서
    추출한 latest_log[node]로 채운다. capacity는 /server_info에서 항상 얻을 수 있다.
    """
    info = _server_info(base)
    metrics = _scrape_metrics(base)
    log = latest_log.get(node, {})

    cap = (info.get("token_capacity") or info.get("num_total_tokens")
           or _norm(re.compile(r"token_capacity|max_total_num_tokens"), metrics))
    cap = float(cap) if cap is not None else None
    # used/usage: HTTP metrics 우선, 없으면 로그, 그래도 없으면 usage×capacity 역산
    used = (info.get("num_used_tokens") or info.get("used_token_num")
            or _norm(_USED_RE, metrics))
    used = float(used) if used is not None else log.get("used_tokens")
    usage = info.get("token_usage") or _norm(_USAGE_RE, metrics) or log.get("usage")
    if (used is None) and (usage is not None) and cap:
        used = float(usage) * cap
    if (usage in (None, 0.0)) and used is not None and cap and cap > 0:
        usage = used / cap

    running  = _norm(_RUN_RE, metrics) or info.get("num_running_reqs") or log.get("running")
    queue    = _norm(_QUEUE_RE, metrics) or info.get("num_queue_reqs") or log.get("queue")
    transfer = _norm(_XFER_RE, metrics)
    transfer = transfer if transfer is not None else log.get("transfer_req")
    prealloc = _norm(_PRE_RE, metrics)
    prealloc = prealloc if prealloc is not None else log.get("prealloc_req")
    prealloc_usage = _norm(_PREALLOC_USAGE_RE, metrics)
    return {
        "capacity":     cap,
        "used_tokens":  float(used) if used is not None else None,
        "usage":        float(usage) if usage is not None else None,
        "running":      running,
        "queue":        queue,
        "transfer_req": transfer,
        "prealloc_req": prealloc,
        "prealloc_usage": prealloc_usage,  # >0 = P→D 전송 in-flight (D측)
        "src_log":      bool(log) and (_norm(_USED_RE, metrics) is None),
    }

def used_gb(used_tokens):
    if used_tokens is None:
        return None
    return used_tokens * KV_BYTES_PER_TOKEN / 1024**3

# ─── Shared state ─────────────────────────────────────────────────────────────
rows: list[dict] = []          # 폴링 시계열
events: list[dict] = []        # 전송/turn/tool 마커
# 로그 batch 라인에서 추출한 최신 live 상태 (enable_metrics=False여도 동작).
# SGLang은 enable_metrics 없이도 decode/prefill batch 라인을 항상 출력하므로,
# HTTP /metrics가 비어도 여기서 used_tokens/usage/transfer_req를 얻는다.
latest_log: dict = {"P": {}, "D": {}}
_stop = threading.Event()
_t0 = time.time()
def now() -> float: return time.time() - _t0

def add_event(kind: str, t: float, **kw):
    events.append({"t": round(t, 3), "kind": kind, **kw})

# ─── Poller thread ────────────────────────────────────────────────────────────
def poller():
    prev_d_used = None
    print(f"{'t(s)':>7} {'P_used':>8} {'P_use%':>7} {'D_used':>8} {'D_use%':>7} "
          f"{'D_run':>6} {'D_xfer':>6}  event")
    while not _stop.is_set():
        t = now()
        P = scrape_node(P_NODE_URL, "P")
        D = scrape_node(D_NODE_URL, "D")
        row = {
            "t": round(t, 3),
            "p_used_tokens": P["used_tokens"], "p_usage": P["usage"],
            "p_used_gb": used_gb(P["used_tokens"]), "p_running": P["running"],
            "p_queue": P["queue"], "p_capacity": P["capacity"],
            "d_used_tokens": D["used_tokens"], "d_usage": D["usage"],
            "d_used_gb": used_gb(D["used_tokens"]), "d_running": D["running"],
            "d_queue": D["queue"], "d_capacity": D["capacity"],
            "d_transfer_req": D["transfer_req"], "d_prealloc_req": D["prealloc_req"],
            "d_prealloc_usage": D["prealloc_usage"],
        }
        rows.append(row)

        # ── P→D 전송 감지 ──
        # (1) D.used_tokens 양의 점프 ≥ JUMP_THRESHOLD = 전송 완료(KV 도착)
        # (2) D.pending_prealloc_token_usage > 0 = 전송 in-flight (예약 중)
        ev_mark = ""
        du = D["used_tokens"]
        if du is not None and prev_d_used is not None:
            delta = du - prev_d_used
            if delta >= JUMP_THRESHOLD:
                add_event("p2d_transfer", t, delta_tokens=round(delta),
                          delta_gb=round(used_gb(delta), 3) if used_gb(delta) else None)
                ev_mark = f"← P→D xfer +{int(delta)} tok ({used_gb(delta):.2f} GB)"
        if du is not None:
            prev_d_used = du
        if D["prealloc_usage"] and D["prealloc_usage"] > 0:
            add_event("prealloc_inflight", t, usage=round(D["prealloc_usage"], 4))
            ev_mark = ev_mark or f"(D prealloc {D['prealloc_usage']*100:.1f}% — transfer in-flight)"
        elif D["transfer_req"]:
            ev_mark = ev_mark or f"(D transfer-req={D['transfer_req']:.0f})"

        def f(x, w=8, p=0): return f"{x:>{w}.{p}f}" if isinstance(x, (int, float)) else f"{'—':>{w}}"
        print(f"{t:>7.1f} {f(P['used_tokens'])} {f((P['usage'] or 0)*100,7,1)} "
              f"{f(D['used_tokens'])} {f((D['usage'] or 0)*100,7,1)} "
              f"{f(D['running'],6)} {f(D['transfer_req'],6)}  {ev_mark}")
        _stop.wait(POLL_INTERVAL)

# ─── Log tailer — batch 라인에서 live 점유 + 전송 신호 추출 ────────────────────
# decode batch 예: "Decode batch, #running-req: 1, #token: 3792, token usage: 0.02,
#   pre-allocated usage: 0.00, #prealloc-req: 0, #transfer-req: 0, ..., #queue-req: 0"
# prefill batch 예: "Prefill batch, #new-seq: 1, #new-token: 4096, ..., token usage: 0.05, ..."
def _grab(pat: str, line: str):
    m = re.search(pat, line)
    return float(m.group(1)) if m else None

def _parse_batch_line(line: str) -> dict | None:
    if "batch" not in line.lower():
        return None
    usage = _grab(r"token usage:\s*([0-9.]+)", line)
    tok   = _grab(r"#token:\s*(\d+)", line)
    if usage is None and tok is None:
        return None
    return {
        "used_tokens":  tok,
        "usage":        usage,
        "running":      _grab(r"#running-req:\s*(\d+)", line),
        "queue":        _grab(r"#queue-req:\s*(\d+)", line),
        "transfer_req": _grab(r"#transfer-req:\s*(\d+)", line),
        "prealloc_req": _grab(r"#prealloc-req:\s*(\d+)", line),
    }

def log_tailer(path: str, node: str):
    """decode/prefill 로그를 tail하며 batch 라인을 latest_log[node]에 반영하고,
    transfer-req>0 / prealloc-req>0 구간을 이벤트로 기록."""
    try:
        f = open(path, "r")
    except OSError:
        print(f"  (log tailer: {path} 열 수 없음 — 건너뜀)")
        return
    f.seek(0, os.SEEK_END)
    last_xfer = 0.0
    while not _stop.is_set():
        line = f.readline()
        if not line:
            _stop.wait(0.2)
            continue
        st = _parse_batch_line(line)
        if st is None:
            continue
        # capacity는 HTTP에서 오므로 여기선 used_tokens가 None이고 usage만 있으면 유지
        latest_log[node] = {k: v for k, v in st.items() if v is not None}
        xfer = st.get("transfer_req") or 0.0
        if xfer > 0 and last_xfer == 0:
            add_event("transfer_req_start", now(), node=node, n=int(xfer))
        elif xfer == 0 and last_xfer > 0:
            add_event("transfer_req_end", now(), node=node)
        last_xfer = xfer

# ─── Synthetic multi-turn agent driver (optional) ─────────────────────────────
def _filler(nchars: int, seed: int) -> str:
    unit = f"context-segment-{seed:04d} the agent observed state and planned actions. "
    return (unit * (nchars // len(unit) + 1))[:nchars]

def driver():
    """context가 turn마다 누적 증가하는 multi-turn 대화를 router로 구동."""
    history = [{"role": "system",
                "content": "You are an agent. " + _filler(INIT_PROMPT_CHARS, 0)}]
    print(f"\n[driver] {NUM_TURNS} turns, tool_delay={TOOL_DELAY_SEC}s, "
          f"growth={TURN_GROWTH_CHARS} chars/turn\n")
    for turn in range(NUM_TURNS):
        if _stop.is_set():
            break
        # 사용자/tool 결과 입력 (turn마다 누적 → context 성장)
        history.append({"role": "user",
                        "content": f"[turn {turn}] " + _filler(TURN_GROWTH_CHARS, turn + 1)})
        add_event("turn_start", now(), turn=turn)
        # streaming 요청 → decode output 수집
        payload = {"model": MODEL, "messages": history,
                   "max_tokens": TURN_OUTPUT_TOKENS, "temperature": 0, "stream": True}
        text, t_first = "", None
        try:
            resp = requests.post(ROUTER_URL, json=payload, stream=True, timeout=300)
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
                    delta = json.loads(data)["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError):
                    continue
                if delta:
                    if t_first is None:
                        t_first = now()
                        add_event("first_token", t_first, turn=turn)
                    text += delta
        except Exception as e:
            print(f"  [driver turn {turn}] ERR {e}")
        add_event("decode_done", now(), turn=turn, out_chars=len(text))
        # 응답 누적 (context 성장) + tool 결과
        history.append({"role": "assistant", "content": text or "(no output)"})
        # tool call idle window
        add_event("tool_start", now(), turn=turn)
        _stop.wait(TOOL_DELAY_SEC)
        add_event("tool_end", now(), turn=turn)
        history.append({"role": "user",
                        "content": f"[tool result {turn}] " + _filler(TURN_GROWTH_CHARS // 2, 900 + turn)})
    # 마지막 tool window 후 잠깐 더 폴링하고 종료
    _stop.wait(3)
    _stop.set()

# ─── Output: CSV + events + plot ──────────────────────────────────────────────
def save_outputs():
    os.makedirs(OUT_DIR, exist_ok=True)
    csv_path = os.path.join(OUT_DIR, f"{_prefix}pd_hbm_occupancy.csv")
    if rows:
        with open(csv_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f"\n시계열 CSV 저장: {csv_path}  ({len(rows)} rows)")
    ev_path = os.path.join(OUT_DIR, f"{_prefix}pd_hbm_events.json")
    with open(ev_path, "w") as f:
        json.dump({"events": events, "config": {
            "p_node": P_NODE_URL, "d_node": D_NODE_URL, "drive": DRIVE,
            "num_turns": NUM_TURNS, "tool_delay_sec": TOOL_DELAY_SEC,
            "jump_threshold": JUMP_THRESHOLD, "kv_bytes_per_token": KV_BYTES_PER_TOKEN,
        }}, f, indent=2)
    n_xfer = sum(1 for e in events if e["kind"] == "p2d_transfer")
    print(f"이벤트 JSON 저장: {ev_path}  (P→D 전송 감지 {n_xfer}건)")

    # ── 전송 감지 요약 (multi-turn: turn마다 점프 크기 증가 기대) ──
    xfers = [e for e in events if e["kind"] == "p2d_transfer"]
    if xfers:
        print("\nP→D 전송 이벤트 (multi-turn이면 turn마다 KV 증가):")
        for i, e in enumerate(xfers):
            print(f"  #{i+1}  t={e['t']:>6.1f}s  +{e['delta_tokens']:>6} tok"
                  f"  ({e.get('delta_gb')} GB)")
    _plot()

def _plot():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("  (matplotlib 미설치 — 플롯 생략)")
        return
    if not rows:
        return
    ts = [r["t"] for r in rows]
    pg = [r["p_used_gb"] for r in rows]
    dg = [r["d_used_gb"] for r in rows]
    pu = [(r["p_usage"] or 0) * 100 for r in rows]
    du = [(r["d_usage"] or 0) * 100 for r in rows]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13, 8), sharex=True,
                                   gridspec_kw={"height_ratios": [3, 1]})
    # ── Panel 1: KV 점유 (GB) ──
    ax1.plot(ts, pg, label="P node KV (GB)", color="#d73027", lw=1.8)
    ax1.plot(ts, dg, label="D node KV (GB)", color="#4575b4", lw=1.8)
    ax1.set_ylabel("KV cache (GB)")
    ax1.set_title(f"P/D node KV occupancy — {MODEL_SLUG}"
                  f"{'  [drive: '+str(NUM_TURNS)+' turns, tool '+str(TOOL_DELAY_SEC)+'s]' if DRIVE else ''}")
    # tool call idle window 음영
    tool_starts = {e["turn"]: e["t"] for e in events if e["kind"] == "tool_start"}
    for e in events:
        if e["kind"] == "tool_end" and e["turn"] in tool_starts:
            ax1.axvspan(tool_starts[e["turn"]], e["t"], color="#ffe08a", alpha=0.35,
                        label="_tool call idle")
    # P→D 전송 마커
    for e in events:
        if e["kind"] == "p2d_transfer":
            ax1.axvline(e["t"], color="#1a9850", ls="--", lw=1.0, alpha=0.8)
            ax1.annotate(f"+{e['delta_tokens']//1000}k" if e["delta_tokens"] >= 1000
                         else f"+{e['delta_tokens']}",
                         (e["t"], max([g for g in dg if g] or [0])),
                         fontsize=7, color="#1a9850", rotation=90, va="bottom")
    # turn 경계
    for e in events:
        if e["kind"] == "turn_start":
            ax1.axvline(e["t"], color="gray", ls=":", lw=0.6, alpha=0.5)
    # 범례 (중복 제거)
    h, l = ax1.get_legend_handles_labels()
    seen = dict(zip(l, h))
    # tool 음영 범례 1개만
    from matplotlib.patches import Patch
    seen["tool call idle"] = Patch(facecolor="#ffe08a", alpha=0.35)
    seen["P→D transfer"] = plt.Line2D([], [], color="#1a9850", ls="--")
    ax1.legend(seen.values(), seen.keys(), loc="upper left", fontsize=8)
    ax1.grid(alpha=0.3)

    # ── Panel 2: running reqs / transfer-req ──
    ax2.plot(ts, [r["d_running"] or 0 for r in rows], label="D running reqs",
             color="#4575b4", lw=1.2)
    ax2.plot(ts, [r["p_running"] or 0 for r in rows], label="P running reqs",
             color="#d73027", lw=1.2)
    xfer = [r["d_transfer_req"] for r in rows]
    if any(x for x in xfer):
        ax2.plot(ts, [x or 0 for x in xfer], label="D transfer-req",
                 color="#1a9850", lw=1.2)
    ax2.set_ylabel("# reqs")
    ax2.set_xlabel("time (s)")
    ax2.legend(loc="upper left", fontsize=8)
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    png = os.path.join(OUT_DIR, f"{_prefix}pd_hbm_occupancy.png")
    fig.savefig(png, dpi=150)
    plt.close(fig)
    print(f"플롯 저장: {png}")

# ─── Main ─────────────────────────────────────────────────────────────────────
def main():
    print("=" * 68)
    print("PD Node KV/HBM Occupancy Monitor")
    print("=" * 68)
    print(f"  P node : {P_NODE_URL}")
    print(f"  D node : {D_NODE_URL}")
    print(f"  poll   : {POLL_INTERVAL}s   KV/token: {KV_BYTES_PER_TOKEN/1024:.0f} KB"
          f"   jump≥{JUMP_THRESHOLD} tok")
    print(f"  mode   : {'DRIVE (synthetic multi-turn)' if DRIVE else 'observe-only'}")
    print("=" * 68)

    # 노드 도달성 점검
    http_live = True
    for name, base in (("P", P_NODE_URL), ("D", D_NODE_URL)):
        s = scrape_node(base, name)
        cap = f"{s['capacity']:,.0f} tok" if s["capacity"] else "capacity 미상"
        print(f"  {name} node reachable: used={s['used_tokens']} cap={cap}")
        if s["capacity"] is None and s["used_tokens"] is None:
            print(f"  ⚠ {name} node({base})에서 KV 지표를 못 읽음 — URL/포트 확인")
        if s["used_tokens"] is None:
            http_live = False
    if not http_live and not (D_LOG or P_LOG):
        print("\n  ⚠ HTTP /metrics에서 live used_tokens를 못 읽음 (enable_metrics=False?).")
        print("    → 서버를 --enable-metrics로 재시작하거나, D_LOG/P_LOG로 로그를 넘기세요:")
        print("      D_LOG=logs/sglang_1p1d/d1.log P_LOG=logs/sglang_1p1d/p1.log ...")

    threads = [threading.Thread(target=poller, daemon=True)]
    if D_LOG:
        threads.append(threading.Thread(target=log_tailer, args=(D_LOG, "D"), daemon=True))
    if P_LOG:
        threads.append(threading.Thread(target=log_tailer, args=(P_LOG, "P"), daemon=True))
    if DRIVE:
        threads.append(threading.Thread(target=driver, daemon=True))
    for t in threads:
        t.start()

    def _sigint(*_):
        _stop.set()
    signal.signal(signal.SIGINT, _sigint)

    try:
        if DRIVE:
            while not _stop.is_set():
                time.sleep(0.2)
        elif DURATION_SEC > 0:
            _stop.wait(DURATION_SEC)
            _stop.set()
        else:
            print("\n관측 모드 — Ctrl-C로 종료\n")
            while not _stop.is_set():
                time.sleep(0.2)
    except KeyboardInterrupt:
        _stop.set()

    time.sleep(POLL_INTERVAL + 0.3)  # 마지막 폴링 flush
    save_outputs()

if __name__ == "__main__":
    main()
