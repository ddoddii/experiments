#!/usr/bin/env python3
"""
SGLang hicache Prometheus Exporter
===================================
SGLang의 /metrics 엔드포인트가 비어있는 경우 (/get_server_info 기반으로 우회),
3-tier hicache 사용량을 Prometheus 형식으로 노출한다.

Exposed metrics (all with node="{label}" label):
  sglang_token_usage          - L1 GPU KV cache 사용률 (0.0~1.0); write_through 시 항상 ≈0
  sglang_num_used_tokens      - L1 GPU KV cache 점유 토큰 수
  sglang_kvcache_allocated_gb - GPU에 할당된 KV cache 총 크기 (GB, static)
  sglang_num_running_reqs     - 현재 처리 중인 요청 수
  sglang_num_queue_reqs       - 대기 중인 요청 수
  sglang_cache_hit_rate       - prefix cache hit rate (scheduler log 파싱; 0-1)
  sglang_hicache_l2_dram_mb   - L2 CPU DRAM 추정 사용량 (process VmRSS 기반)
  sglang_hicache_l3_ssd_mb    - L3 SSD 파일 백엔드 사용량 (du 기반)
  sglang_scrape_ok            - 1=스크레이핑 성공, 0=실패

Note on L1 token_usage=0: SGLang hicache_write_policy='write_through' immediately
evicts KV blocks from GPU to SSD after prefill, so GPU usage stays near 0 during
steady-state. This is correct behavior. sglang_kvcache_allocated_gb shows the
static GPU memory reserved for KV cache.

Usage (standalone):
    python sglang_hicache_exporter.py
    # => http://localhost:9199/metrics

    # Prometheus에서 scrape:
    # - job_name: 'sglang-hicache-exporter'
    #   static_configs:
    #     - targets: ['host.docker.internal:9199']

Environment variables:
    SGLANG_INSTANCES  comma-separated "label:port", default: p1:30000,p2:30001,d1:30002,d2:30003
    EXPORTER_PORT     default: 9199
    HICACHE_DIRS      comma-separated "label:dir_path" overrides for SSD tier monitoring
                      e.g. "p1:/tmp/hicache,p2:/tmp/hicache"
    LOG_DIR           directory containing p1.log, p2.log etc. for cache hit rate parsing
                      default: auto-detect from exporter script location
    POLL_INTERVAL     seconds between internal cache refresh (default: 3.0)
"""

import os
import re
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import requests


# ─── Configuration ────────────────────────────────────────────────────────────

def _parse_instances(raw: str) -> dict[str, int]:
    result = {}
    for part in raw.split(","):
        part = part.strip()
        if ":" in part:
            label, port = part.rsplit(":", 1)
            result[label.strip()] = int(port.strip())
    return result


def _parse_hicache_dirs(raw: str, instances: dict[str, int]) -> dict[str, str | None]:
    """Build label → directory mapping.

    If HICACHE_DIRS is not set, probe common default locations.
    """
    explicit = {}
    if raw:
        for part in raw.split(","):
            part = part.strip()
            if ":" in part:
                label, path = part.split(":", 1)
                explicit[label.strip()] = path.strip()

    result = {}
    for label, port in instances.items():
        if label in explicit:
            result[label] = explicit[label]
        else:
            # Common default paths SGLang uses for file-backend hicache
            candidates = [
                f"/tmp/sglang_hicache_{port}",
                f"/tmp/sglang_hicache/port_{port}",
                f"/tmp/hicache_{port}",
                os.path.expanduser(f"~/.sglang/hicache_{port}"),
                os.path.expanduser(f"~/.sglang/hicache/port_{port}"),
            ]
            found = next((c for c in candidates if os.path.isdir(c)), None)
            result[label] = found
    return result


INSTANCES = _parse_instances(
    os.environ.get("SGLANG_INSTANCES", "p1:30000,p2:30001,d1:30002,d2:30003")
)
EXPORTER_PORT = int(os.environ.get("EXPORTER_PORT", "9199"))
HICACHE_DIRS: dict[str, str | None] = _parse_hicache_dirs(
    os.environ.get("HICACHE_DIRS", ""), INSTANCES
)
POLL_INTERVAL = float(os.environ.get("POLL_INTERVAL", "3.0"))

# LOG_DIR: directory containing {label}.log files for cache hit rate parsing
# Auto-detect: look for logs/ sibling to the benchmark/ directory
_script_dir = os.path.dirname(os.path.abspath(__file__))
_default_log_dir = os.path.join(os.path.dirname(_script_dir), "logs")
LOG_DIR: str = os.environ.get("LOG_DIR", _default_log_dir)


# ─── Metric collection ────────────────────────────────────────────────────────

def _get_server_info(port: int) -> dict | None:
    try:
        r = requests.get(f"http://localhost:{port}/get_server_info", timeout=2.0)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def _find_sglang_pid(port: int) -> int | None:
    """Return PID of the process listening on the given port."""
    try:
        # ss -tlnp shows listening TCP sockets with PIDs
        out = subprocess.run(
            ["ss", "-tlnp", f"sport = :{port}"],
            capture_output=True, text=True, timeout=3
        ).stdout
        # Pattern: pid=12345
        m = re.search(r"pid=(\d+)", out)
        if m:
            return int(m.group(1))
    except Exception:
        pass
    try:
        # Fallback: lsof
        out = subprocess.run(
            ["lsof", "-ti", f":{port}"],
            capture_output=True, text=True, timeout=3
        ).stdout.strip()
        if out:
            return int(out.splitlines()[0])
    except Exception:
        pass
    return None


def _proc_rss_mb(pid: int) -> float:
    """Return process VmRSS in MB from /proc/{pid}/status."""
    try:
        with open(f"/proc/{pid}/status") as fh:
            for line in fh:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024.0
    except Exception:
        pass
    return 0.0


def _ssd_dir_mb(path: str | None) -> float:
    """Return directory size in MB using `du -sb`."""
    if not path or not os.path.isdir(path):
        return 0.0
    try:
        out = subprocess.run(
            ["du", "-sb", path],
            capture_output=True, text=True, timeout=5
        ).stdout
        if out:
            return int(out.split()[0]) / (1024 * 1024)
    except Exception:
        pass
    return 0.0


# Matches SGLang scheduler stat lines, e.g.:
#   "#running-req: 2 ... cache hit rate: 45.67%"
#   "Avg ... cache hit rate: 45.67%"
_CACHE_HIT_RE = re.compile(
    r"cache hit rate:\s*([\d.]+)\s*%", re.IGNORECASE
)


def _parse_log_hit_rate(label: str) -> float | None:
    """
    Return the most recent prefix cache hit rate (0.0–1.0) from the
    SGLang scheduler log for the given instance label.

    SGLang's scheduler periodically prints a stats line containing
    "cache hit rate: XX.XX%" which we tail-parse from the log file.
    Returns None if no match found or file not accessible.
    """
    log_path = os.path.join(LOG_DIR, f"{label}.log")
    if not os.path.isfile(log_path):
        return None
    try:
        # Read last 8 KB of the log file — scheduler prints stats every few seconds
        with open(log_path, "rb") as fh:
            fh.seek(0, 2)
            size = fh.tell()
            fh.seek(max(0, size - 8192))
            tail = fh.read().decode("utf-8", errors="replace")
        # Find all matches, take the last one (most recent)
        matches = _CACHE_HIT_RE.findall(tail)
        if matches:
            return float(matches[-1]) / 100.0
    except Exception:
        pass
    return None


def collect_metrics() -> str:
    """Collect all metrics and return Prometheus text format."""
    lines: list[str] = []

    # Headers
    lines += [
        "# HELP sglang_token_usage Fraction of GPU KV cache in use (L1 VRAM tier; ~0 with write_through)",
        "# TYPE sglang_token_usage gauge",
        "# HELP sglang_num_used_tokens Tokens currently occupying GPU KV cache",
        "# TYPE sglang_num_used_tokens gauge",
        "# HELP sglang_kvcache_allocated_gb GPU memory allocated for KV cache in GB (static from config)",
        "# TYPE sglang_kvcache_allocated_gb gauge",
        "# HELP sglang_num_running_reqs Requests being actively processed",
        "# TYPE sglang_num_running_reqs gauge",
        "# HELP sglang_num_queue_reqs Requests waiting in queue",
        "# TYPE sglang_num_queue_reqs gauge",
        "# HELP sglang_cache_hit_rate Prefix cache hit rate parsed from scheduler log (0-1)",
        "# TYPE sglang_cache_hit_rate gauge",
        "# HELP sglang_hicache_l2_dram_mb Estimated L2 CPU DRAM usage in MB (VmRSS)",
        "# TYPE sglang_hicache_l2_dram_mb gauge",
        "# HELP sglang_hicache_l3_ssd_mb L3 SSD file-backend hicache size in MB",
        "# TYPE sglang_hicache_l3_ssd_mb gauge",
        "# HELP sglang_scrape_ok 1 if /get_server_info succeeded, 0 otherwise",
        "# TYPE sglang_scrape_ok gauge",
    ]

    for label, port in INSTANCES.items():
        # 'instance' is reserved by Prometheus (scrape target addr) → use 'node' instead
        lbl = f'node="{label}",port="{port}"'

        info = _get_server_info(port)
        if info is None:
            lines.append(f"sglang_scrape_ok{{{lbl}}} 0")
            continue

        lines.append(f"sglang_scrape_ok{{{lbl}}} 1")

        token_usage = info.get("token_usage", 0.0)
        lines.append(f"sglang_token_usage{{{lbl}}} {token_usage:.6f}")

        num_used = info.get("num_used_tokens", 0)
        lines.append(f"sglang_num_used_tokens{{{lbl}}} {num_used}")

        # Static GPU memory allocated for KV cache (from internal_states config)
        for state in info.get("internal_states", []):
            if isinstance(state, dict) and "memory_usage" in state:
                mem = state["memory_usage"]
                if isinstance(mem, dict) and "kvcache" in mem:
                    lines.append(
                        f"sglang_kvcache_allocated_gb{{{lbl}}} {float(mem['kvcache']):.3f}"
                    )
                break  # only need the first worker state

        num_running = info.get("num_running_reqs", 0)
        lines.append(f"sglang_num_running_reqs{{{lbl}}} {num_running}")

        num_queue = info.get("num_queue_reqs", 0)
        lines.append(f"sglang_num_queue_reqs{{{lbl}}} {num_queue}")

        # Prefix cache hit rate: try /get_server_info fields first, then log parsing
        hit_rate = (
            info.get("cache_hit_rate")
            or info.get("radix_cache_hit_rate")
            or info.get("avg_cache_hit_rate")
        )
        # Also check internal_states for hicache-specific stats
        if hit_rate is None:
            for state in info.get("internal_states", []):
                if isinstance(state, dict):
                    for k in ("cache_hit_rate", "hicache_hit_rate",
                              "l2_cache_hit_rate", "l3_cache_hit_rate",
                              "radix_cache_hit_rate"):
                        if k in state:
                            hit_rate = state[k]
                            break
                if hit_rate is not None:
                    break
        # Last resort: parse most recent stat line from scheduler log
        if hit_rate is None:
            hit_rate = _parse_log_hit_rate(label)
        if hit_rate is not None:
            lines.append(f"sglang_cache_hit_rate{{{lbl}}} {float(hit_rate):.6f}")

        # L2 DRAM: process RSS
        pid = _find_sglang_pid(port)
        if pid:
            dram_mb = _proc_rss_mb(pid)
            lines.append(f"sglang_hicache_l2_dram_mb{{{lbl}}} {dram_mb:.1f}")

        # L3 SSD: directory size
        ssd_dir = HICACHE_DIRS.get(label)
        ssd_mb = _ssd_dir_mb(ssd_dir)
        lines.append(f"sglang_hicache_l3_ssd_mb{{{lbl}}} {ssd_mb:.1f}")

    lines.append("")  # trailing newline
    return "\n".join(lines)


# ─── HTTP Server ──────────────────────────────────────────────────────────────

class MetricsHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path in ("/metrics", "/"):
            body = collect_metrics().encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; version=0.0.4; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, fmt, *args):
        pass  # suppress request logs


# ─── Benchmark companion class (same interface as KVCachePoller) ───────────────

class SGLangHiCachePoller(threading.Thread):
    """
    Background thread that polls SGLang /get_server_info and collects
    per-instance stats — mirrors KVCachePoller interface.

    Usage:
        poller = SGLangHiCachePoller()
        poller.start()
        # ... benchmark ...
        poller.stop()
        stats = poller.stats()
        # {"p1": {"token_usage": {"min":.., "max":.., "mean":.., "samples":..}, ...}}
    """

    def __init__(self, instances: dict | None = None, interval: float = 2.0):
        super().__init__(daemon=True, name="SGLangHiCachePoller")
        self.instances = instances if instances is not None else INSTANCES
        self.interval = interval
        self._stop_event = threading.Event()
        self.samples: dict[str, dict[str, list[float]]] = {
            label: {"token_usage": [], "num_running_reqs": []}
            for label in self.instances
        }

    def run(self):
        while not self._stop_event.wait(self.interval):
            for label, port in self.instances.items():
                info = _get_server_info(port)
                if info is not None:
                    self.samples[label]["token_usage"].append(
                        float(info.get("token_usage", 0.0))
                    )
                    self.samples[label]["num_running_reqs"].append(
                        int(info.get("num_running_reqs", 0))
                    )

    def stop(self):
        self._stop_event.set()
        self.join(timeout=5)

    def stats(self) -> dict:
        result = {}
        for label, metrics in self.samples.items():
            result[label] = {}
            for metric, vals in metrics.items():
                if vals:
                    result[label][metric] = {
                        "min":     round(min(vals), 4),
                        "max":     round(max(vals), 4),
                        "mean":    round(sum(vals) / len(vals), 4),
                        "samples": len(vals),
                    }
                else:
                    result[label][metric] = None
        return result


# ─── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"SGLang hicache exporter starting on :{EXPORTER_PORT}")
    print(f"  Instances   : {INSTANCES}")
    print(f"  Hicache dirs: {HICACHE_DIRS}")
    print(f"  Log dir     : {LOG_DIR}")
    print(f"  Scrape URL  : http://localhost:{EXPORTER_PORT}/metrics")

    server = HTTPServer(("0.0.0.0", EXPORTER_PORT), MetricsHandler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nExporter stopped.")
