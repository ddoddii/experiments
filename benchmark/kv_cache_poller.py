"""
KV Cache Background Poller
==========================
vLLM /metrics 엔드포인트를 백그라운드 스레드에서 주기적으로 폴링해
GPU별 KV cache 사용률을 수집한다.

벤치마크 스크립트에서 아래처럼 사용:
    from kv_cache_poller import KVCachePoller

    poller = KVCachePoller()  # 기본: p1=8100, p2=8101, d1=8200, d2=8201
    poller.start()
    # ... 벤치마크 실행 ...
    poller.stop()
    stats = poller.stats()
    # stats = {
    #   "p1": {"min": 0.0, "max": 0.031, "mean": 0.008, "samples": 120},
    #   "d1": {"min": 0.0, "max": 0.087, "mean": 0.045, "samples": 120},
    #   ...
    # }

환경변수로 포트 오버라이드:
    P1_PORT=8100 P2_PORT=8101 D1_PORT=8200 D2_PORT=8201
"""

import os
import threading
import requests


class KVCachePoller(threading.Thread):
    """Daemon thread that polls vLLM /metrics for kv_cache_usage_perc."""

    DEFAULT_PORTS = {
        "p1": int(os.environ.get("P1_PORT", "8100")),
        "p2": int(os.environ.get("P2_PORT", "8101")),
        "d1": int(os.environ.get("D1_PORT", "8200")),
        "d2": int(os.environ.get("D2_PORT", "8201")),
    }

    def __init__(self, ports: dict = None, interval: float = 2.0):
        """
        Args:
            ports:    {"label": port} dict.  None → DEFAULT_PORTS.
            interval: polling interval in seconds.
        """
        super().__init__(daemon=True, name="KVCachePoller")
        self.ports    = ports if ports is not None else self.DEFAULT_PORTS
        self.interval = interval
        self._stop    = threading.Event()
        self.samples: dict[str, list[float]] = {k: [] for k in self.ports}

    # ── internal ─────────────────────────────────────────────────────────────

    def _scrape_one(self, port: int) -> float | None:
        """Return kv_cache_usage_perc value from one vLLM instance, or None."""
        try:
            r = requests.get(f"http://localhost:{port}/metrics", timeout=1.5)
            for line in r.text.splitlines():
                if line.startswith("vllm:kv_cache_usage_perc") and not line.startswith("#"):
                    return float(line.split()[-1])
        except Exception:
            pass
        return None

    def run(self):
        while not self._stop.wait(self.interval):
            for label, port in self.ports.items():
                val = self._scrape_one(port)
                if val is not None:
                    self.samples[label].append(val)

    # ── public API ────────────────────────────────────────────────────────────

    def stop(self):
        """Signal stop and wait for thread to exit."""
        self._stop.set()
        self.join(timeout=5)

    def stats(self) -> dict:
        """
        Return min / max / mean / samples per label.
        Labels with no data return None.

        Example:
            {
              "p1": {"min": 0.0, "max": 0.031, "mean": 0.008, "samples": 120},
              "p2": {"min": 0.0, "max": 0.029, "mean": 0.007, "samples": 120},
              "d1": {"min": 0.0, "max": 0.087, "mean": 0.045, "samples": 120},
              "d2": {"min": 0.0, "max": 0.095, "mean": 0.048, "samples": 120},
            }
        """
        result = {}
        for label, vals in self.samples.items():
            if vals:
                result[label] = {
                    "min":     round(min(vals), 4),
                    "max":     round(max(vals), 4),
                    "mean":    round(sum(vals) / len(vals), 4),
                    "samples": len(vals),
                }
            else:
                result[label] = None
        return result
