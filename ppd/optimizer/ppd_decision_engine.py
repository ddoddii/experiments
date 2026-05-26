#!/usr/bin/env python3
"""
PPD Decision Engine — data-driven PD vs PPD routing for Turn 2+ requests.

Scoring formula (absolute ms, output-length aware):
    score = w_ttft * TTFT_savings_ms - w_tpot * (delta_tpot_ms_per_token * output_tokens)
    use_ppd = score > 0

Lookup table is built from comprehensive benchmark results.
An entry is included only when:
  - Both pd_tpot and pd_ttft are non-zero (measurement not failed)
  - PD success rate >= MIN_PD_SR (default 80%); below this PD is unreliable → force PPD

At runtime, _get_perf_data() does linear QPS interpolation between the two
nearest benchmark points (clamped at edges).  If no data exists for the
requested (ctx, workload) pair, it falls back to large-context data before
returning None and letting should_use_ppd() use the configured default.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, Optional, Tuple
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

CONTEXT_THRESHOLDS = {
    "small": (0, 512),
    "large": (512, 4096),
    "huge":  (4096, float("inf")),
}

T2_WORKLOAD_CONFIGS = {
    "tiny":         {"input": 16,   "output": 32},
    "short_gen":    {"input": 32,   "output": 256},
    "long_gen":     {"input": 32,   "output": 512},
    "very_long_gen":{"input": 64,   "output": 1024},
    "small_bal":    {"input": 64,   "output": 64},
    "mid_bal":      {"input": 128,  "output": 128},
    "mid_paste":    {"input": 256,  "output": 64},
    "big_paste":    {"input": 512,  "output": 64},
    "huge_paste":   {"input": 1024, "output": 32},
}

# PD ↔ PPD config pairs
CONFIG_PAIRS = {
    "1P_3D": "1P_3pD",
    "2P_2D": "2P_2pD",
    "3P_1D": "3P_1pD",
    "1R_1P_2D": "1R_1P_2pD",
    "1R_2P_1D": "1R_2P_1pD",
    "2R_1P_1D": "2R_1P_1pD",
}

# Benchmark QPS points (must match filenames: _1.0.json, _2.0.json, …)
QPS_POINTS = [0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 20.0]

# Minimum PD success rate to consider an entry reliable.
# Below this threshold PD is too unreliable to prefer even if score > 0.
_DEFAULT_MIN_PD_SR = 0.80


# ---------------------------------------------------------------------------
# Request classifier helpers
# ---------------------------------------------------------------------------

def classify_context_length(context_tokens: int) -> str:
    for cls, (lo, hi) in CONTEXT_THRESHOLDS.items():
        if lo < context_tokens <= hi:
            return cls
    return "small"


def classify_workload(input_tokens: int, output_tokens: int) -> str:
    """Map (input, output) token counts to one of the 9 T2 workload buckets."""
    if output_tokens == 0:
        output_tokens = 1
    ratio = input_tokens / output_tokens

    if ratio < 0.25:                        # decode-heavy (long generation)
        if output_tokens <= 64:   return "tiny"
        if output_tokens <= 256:  return "short_gen"
        if output_tokens <= 512:  return "long_gen"
        return "very_long_gen"
    elif ratio <= 2.0:                      # balanced
        if input_tokens <= 64:    return "small_bal"
        return "mid_bal"
    else:                                   # prefill-heavy (paste)
        if input_tokens <= 256:   return "mid_paste"
        if input_tokens <= 512:   return "big_paste"
        return "huge_paste"


# ---------------------------------------------------------------------------
# Decision stats
# ---------------------------------------------------------------------------

@dataclass
class DecisionStats:
    total_decisions: int = 0
    ppd_decisions: int = 0
    pd_decisions: int = 0
    turn1_skipped: int = 0
    decisions_by_workload: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def record(self, workload: str, use_ppd: bool, is_turn1: bool = False):
        self.total_decisions += 1
        if is_turn1:
            self.turn1_skipped += 1
            return
        if use_ppd:
            self.ppd_decisions += 1
        else:
            self.pd_decisions += 1
        bucket = self.decisions_by_workload.setdefault(workload, {"ppd": 0, "pd": 0})
        bucket["ppd" if use_ppd else "pd"] += 1

    def to_dict(self) -> dict:
        denom = max(1, self.ppd_decisions + self.pd_decisions)
        return {
            "total_decisions": self.total_decisions,
            "ppd_decisions": self.ppd_decisions,
            "pd_decisions": self.pd_decisions,
            "turn1_skipped": self.turn1_skipped,
            "ppd_ratio": self.ppd_decisions / denom,
            "decisions_by_workload": self.decisions_by_workload,
        }


# ---------------------------------------------------------------------------
# Main engine
# ---------------------------------------------------------------------------

class PPDDecisionEngine:
    """Benchmark-driven PD vs PPD routing decision engine.

    Lookup table is keyed by (context_class, workload_type, qps).
    At runtime, _get_perf_data() retrieves linearly-interpolated performance
    metrics for the actual QPS and computes a fresh score using the request's
    real output_tokens — so the TPOT penalty is correctly weighted.
    """

    def __init__(
        self,
        benchmark_data_path: str,
        base_config: str = "2P_2D",
        default_use_ppd: bool = True,
        w_ttft: float = 1.0,
        w_tpot: float = 1.0,
        min_pd_sr: float = _DEFAULT_MIN_PD_SR,
    ):
        self.benchmark_path = Path(benchmark_data_path)
        self.base_config    = base_config
        self.ppd_config     = CONFIG_PAIRS.get(base_config, base_config.replace("D", "pD"))
        self.default_use_ppd = default_use_ppd
        self.w_ttft    = w_ttft
        self.w_tpot    = w_tpot
        self.min_pd_sr = min_pd_sr

        # (ctx_class, workload_type, qps) → bool
        self.lookup_table: Dict[Tuple[str, str, float], bool] = {}
        # (ctx_class, workload_type, qps) → {pd_ttft, ppd_ttft, pd_tpot, ppd_tpot, score, …}
        self.performance_data: Dict[Tuple[str, str, float], dict] = {}

        self.stats = DecisionStats()
        self._build_lookup_table()

        logger.info(
            "PPDDecisionEngine initialized | config=%s→%s | "
            "w_ttft=%.1f w_tpot=%.1f | min_pd_sr=%.0f%% | "
            "perf_entries=%d lookup_entries=%d",
            self.base_config, self.ppd_config,
            self.w_ttft, self.w_tpot,
            self.min_pd_sr * 100,
            len(self.performance_data), len(self.lookup_table),
        )

    # ------------------------------------------------------------------
    # Table construction
    # ------------------------------------------------------------------

    def _load_result(self, config: str, workload: str, qps: float) -> Optional[dict]:
        path = self.benchmark_path / config / f"{config}_{workload}_{qps}.json"
        if not path.exists():
            return None
        try:
            with open(path) as f:
                return json.load(f)
        except Exception as e:
            logger.warning("Failed to load %s: %s", path, e)
            return None

    def _build_lookup_table(self):
        logger.info("Building lookup table from %s", self.benchmark_path)

        for ctx in ("small", "large"):
            for wl, t2_cfg in T2_WORKLOAD_CONFIGS.items():
                typical_out = t2_cfg["output"]
                full_wl = f"{ctx}_{wl}"

                for qps in QPS_POINTS:
                    pd_res  = self._load_result(self.base_config, full_wl, qps)
                    ppd_res = self._load_result(self.ppd_config,  full_wl, qps)

                    if pd_res is None or ppd_res is None:
                        # No benchmark data — defer to default at runtime
                        self.lookup_table[(ctx, wl, qps)] = self.default_use_ppd
                        continue

                    pd_t2   = pd_res.get("turn2", {})
                    ppd_t2  = ppd_res.get("turn2", {})
                    pd_ttft  = pd_t2.get("avg_ttft_ms", 0)
                    pd_tpot  = pd_t2.get("avg_tpot_ms", 0)
                    ppd_ttft = ppd_t2.get("avg_ttft_ms", 0)
                    ppd_tpot = ppd_t2.get("avg_tpot_ms", 0)
                    pd_sr    = pd_res.get("success_rate", 0) / 100

                    # ── validity checks ────────────────────────────────
                    skip_reason = None
                    if pd_ttft == 0 or ppd_ttft == 0:
                        skip_reason = "ttft=0 (measurement failure)"
                    elif pd_tpot == 0 or ppd_tpot == 0:
                        skip_reason = "tpot=0 (measurement failure)"
                    elif pd_sr < self.min_pd_sr:
                        skip_reason = f"pd_sr={pd_sr:.0%} < {self.min_pd_sr:.0%} (PD unreliable → force PPD)"

                    if skip_reason:
                        # PD unreliable → PPD is the safe choice; exclude from
                        # performance_data so QPS interpolation skips this point.
                        self.lookup_table[(ctx, wl, qps)] = True
                        logger.debug("  %s @ QPS=%.1f: SKIP — %s", full_wl, qps, skip_reason)
                        continue

                    score = self._score(pd_ttft, ppd_ttft, pd_tpot, ppd_tpot, typical_out)
                    use_ppd = score > 0

                    self.performance_data[(ctx, wl, qps)] = {
                        "pd_ttft":  pd_ttft,
                        "ppd_ttft": ppd_ttft,
                        "pd_tpot":  pd_tpot,
                        "ppd_tpot": ppd_tpot,
                        "score":    score,
                        "pd_sr":    pd_sr,
                        "typical_out": typical_out,
                    }
                    self.lookup_table[(ctx, wl, qps)] = use_ppd

                    logger.debug(
                        "  %s @ QPS=%.1f | TTFT_save=%+.0fms TPOT_cost=%+.0fms "
                        "score=%+.0fms → %s",
                        full_wl, qps,
                        pd_ttft - ppd_ttft,
                        (ppd_tpot - pd_tpot) * typical_out,
                        score,
                        "PPD" if use_ppd else "PD",
                    )

        ppd_n = sum(1 for v in self.lookup_table.values() if v)
        tot   = len(self.lookup_table)
        logger.info(
            "Lookup table built: %d PPD / %d PD out of %d entries "
            "(%d backed by benchmark data, %d fallback/forced)",
            ppd_n, tot - ppd_n, tot,
            len(self.performance_data), tot - len(self.performance_data),
        )

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _score(
        self,
        pd_ttft: float,
        ppd_ttft: float,
        pd_tpot: float,
        ppd_tpot: float,
        output_tokens: int,
    ) -> float:
        """Absolute-ms score.

        score = w_ttft * TTFT_savings_ms
              - w_tpot * (delta_tpot_ms_per_token * output_tokens)

        Positive → PPD saves more latency than it costs.
        Uses actual output_tokens so TPOT penalty scales correctly with
        generation length (unlike a relative-% formula).
        """
        ttft_savings = pd_ttft  - ppd_ttft
        tpot_extra   = (ppd_tpot - pd_tpot) * output_tokens
        return self.w_ttft * ttft_savings - self.w_tpot * tpot_extra

    # ------------------------------------------------------------------
    # QPS-interpolated performance data lookup
    # ------------------------------------------------------------------

    def _get_perf_data(
        self,
        ctx_class: str,
        workload_type: str,
        qps: float,
    ) -> Optional[dict]:
        """Return benchmark perf metrics interpolated to the given QPS.

        Interpolation order:
          1. Linear interpolation between the two nearest valid QPS points
             for (ctx_class, workload_type).
          2. Clamp to available range (no extrapolation beyond data edge).
          3. If no data for ctx_class, fall back to "large" context
             (conservative proxy — large-context KV transfer cost ≥ small).
          4. Return None if no data exists at all.
        """
        def _interp(pts: list, target: float) -> dict:
            """pts: sorted list of (qps, data_dict)."""
            if target <= pts[0][0]:
                return pts[0][1]
            if target >= pts[-1][0]:
                return pts[-1][1]
            for i in range(len(pts) - 1):
                q_lo, d_lo = pts[i]
                q_hi, d_hi = pts[i + 1]
                if q_lo <= target <= q_hi:
                    t = (target - q_lo) / (q_hi - q_lo)
                    return {
                        k: d_lo[k] * (1 - t) + d_hi[k] * t
                        for k in ("pd_ttft", "ppd_ttft", "pd_tpot", "ppd_tpot")
                    }
            return pts[-1][1]

        def _collect(ctx: str) -> list:
            return sorted(
                (q, self.performance_data[(ctx, workload_type, q)])
                for q in QPS_POINTS
                if (ctx, workload_type, q) in self.performance_data
            )

        pts = _collect(ctx_class)
        if not pts and ctx_class == "small":
            pts = _collect("large")   # conservative fallback
        if not pts:
            return None
        return _interp(pts, qps)

    # ------------------------------------------------------------------
    # Huge-context extrapolation
    # ------------------------------------------------------------------

    def _extrapolate_huge(
        self,
        workload_type: str,
        qps: float,
        output_tokens: int,
    ) -> bool:
        """Huge context (>4096 tokens) extrapolated from large-context data.

        Huge context has larger KV cache → larger transfer overhead → PPD
        benefit is at least as large as for large context.
        """
        perf = self._get_perf_data("large", workload_type, qps)
        if perf is not None:
            return self._score(
                perf["pd_ttft"], perf["ppd_ttft"],
                perf["pd_tpot"],  perf["ppd_tpot"],
                output_tokens,
            ) > 0
        return self.default_use_ppd

    # ------------------------------------------------------------------
    # Main decision API
    # ------------------------------------------------------------------

    def should_use_ppd(
        self,
        turn: int,
        input_tokens: int,
        output_tokens: int,
        current_qps: float,
        context_tokens: int = 0,
    ) -> bool:
        """Decide whether to route this Turn 2+ request to PPD (local) or PD.

        Args:
            turn:           Turn number (1-indexed).
            input_tokens:   New input tokens in the current turn.
            output_tokens:  Expected output tokens (from dataset / predictor).
            current_qps:    Observed system QPS at decision time.
            context_tokens: Accumulated tokens from previous turns.

        Returns:
            True → PPD (handle on decode node directly).
            False → PD  (route through prefill node).
        """
        # Turn 1: no KV cache on decode node yet → always PD
        if turn == 1:
            self.stats.record("turn1", False, is_turn1=True)
            return False

        ctx_class     = classify_context_length(context_tokens)
        workload_type = classify_workload(input_tokens, output_tokens)

        # Retrieve linearly-interpolated benchmark metrics
        perf = self._get_perf_data(ctx_class, workload_type, current_qps)

        if perf is not None:
            score   = self._score(
                perf["pd_ttft"], perf["ppd_ttft"],
                perf["pd_tpot"],  perf["ppd_tpot"],
                output_tokens,
            )
            use_ppd = score > 0
            logger.debug(
                "decision: turn=%d ctx=%s wl=%s qps=%.1f out=%d "
                "TTFT_save=%+.0f TPOT_cost=%+.0f score=%+.0fms → %s",
                turn, ctx_class, workload_type, current_qps, output_tokens,
                perf["pd_ttft"] - perf["ppd_ttft"],
                (perf["ppd_tpot"] - perf["pd_tpot"]) * output_tokens,
                score,
                "PPD" if use_ppd else "PD",
            )
        elif ctx_class == "huge":
            use_ppd = self._extrapolate_huge(workload_type, current_qps, output_tokens)
            logger.debug(
                "decision: huge-context extrapolation wl=%s → %s",
                workload_type, "PPD" if use_ppd else "PD",
            )
        else:
            use_ppd = self.default_use_ppd
            logger.debug(
                "decision: no benchmark data for %s_%s, default=%s",
                ctx_class, workload_type, use_ppd,
            )

        self.stats.record(workload_type, use_ppd)
        return use_ppd

    # ------------------------------------------------------------------
    # Inspection helpers
    # ------------------------------------------------------------------

    def get_decision_stats(self) -> dict:
        return self.stats.to_dict()

    def get_performance_comparison(self) -> Dict[str, dict]:
        """Aggregate performance_data by workload for reporting."""
        agg: Dict[str, dict] = {}
        for (ctx, wl, _qps), d in self.performance_data.items():
            key = f"{ctx}_{wl}"
            if key not in agg:
                agg[key] = {
                    "pd_ttft": [], "ppd_ttft": [],
                    "pd_tpot": [], "ppd_tpot": [],
                    "scores": [], "ppd_wins": 0, "total": 0,
                }
            a = agg[key]
            a["pd_ttft"].append(d["pd_ttft"])
            a["ppd_ttft"].append(d["ppd_ttft"])
            a["pd_tpot"].append(d["pd_tpot"])
            a["ppd_tpot"].append(d["ppd_tpot"])
            a["scores"].append(d["score"])
            a["total"] += 1
            if d["score"] > 0:
                a["ppd_wins"] += 1

        summary = {}
        for key, a in agg.items():
            n = len(a["pd_ttft"])
            summary[key] = {
                "pd_avg_ttft_ms":  sum(a["pd_ttft"])  / n,
                "ppd_avg_ttft_ms": sum(a["ppd_ttft"]) / n,
                "pd_avg_tpot_ms":  sum(a["pd_tpot"])  / n,
                "ppd_avg_tpot_ms": sum(a["ppd_tpot"]) / n,
                "avg_score_ms":    sum(a["scores"])    / n,
                "ppd_win_rate":    a["ppd_wins"] / a["total"],
            }
        return summary

    def export_lookup_table(self, output_path: str):
        """Export the full lookup table to JSON for offline inspection."""
        rows = []
        for (ctx, wl, qps) in sorted(self.lookup_table):
            use_ppd = self.lookup_table[(ctx, wl, qps)]
            d = self.performance_data.get((ctx, wl, qps), {})
            rows.append({
                "context":  ctx,
                "workload": wl,
                "qps":      qps,
                "use_ppd":  use_ppd,
                "has_data": bool(d),
                "pd_ttft_ms":  d.get("pd_ttft"),
                "ppd_ttft_ms": d.get("ppd_ttft"),
                "pd_tpot_ms":  d.get("pd_tpot"),
                "ppd_tpot_ms": d.get("ppd_tpot"),
                "score_ms":    d.get("score"),
                "pd_sr":       d.get("pd_sr"),
                "typical_out": d.get("typical_out"),
            })
        export = {
            "base_config": self.base_config,
            "ppd_config":  self.ppd_config,
            "weights":     {"w_ttft": self.w_ttft, "w_tpot": self.w_tpot},
            "min_pd_sr":   self.min_pd_sr,
            "entries":     rows,
        }
        with open(output_path, "w") as f:
            json.dump(export, f, indent=2)
        logger.info("Lookup table exported → %s (%d entries)", output_path, len(rows))


# ---------------------------------------------------------------------------
# CLI smoke test
# ---------------------------------------------------------------------------

def main():
    import sys
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    benchmark_path = Path(__file__).parent.parent.parent / "results" / "comprehensive"
    if not benchmark_path.exists():
        print(f"Benchmark path not found: {benchmark_path}")
        sys.exit(1)

    engine = PPDDecisionEngine(benchmark_data_path=str(benchmark_path))

    print(f"\nperf_data entries : {len(engine.performance_data)}")
    print(f"lookup entries    : {len(engine.lookup_table)}")

    # ── per-entry decision table ──────────────────────────────────────────
    print("\n" + "=" * 105)
    print(f"{'Context':8s} {'Workload':16s} {'QPS':>5s}  "
          f"{'TTFT_save':>10s} {'TPOT_cost':>10s}  {'Score':>9s}  "
          f"{'PD_SR':>6s}  {'Decision':>8s}")
    print("-" * 105)

    for (ctx, wl, qps) in sorted(engine.performance_data):
        d = engine.performance_data[(ctx, wl, qps)]
        out = d["typical_out"]
        ttft_save  = d["pd_ttft"]  - d["ppd_ttft"]
        tpot_cost  = (d["ppd_tpot"] - d["pd_tpot"]) * out
        decision   = "PPD" if engine.lookup_table[(ctx, wl, qps)] else "PD "
        print(f"{ctx:8s} {wl:16s} {qps:>5.1f}  "
              f"{ttft_save:>+10.0f}ms {tpot_cost:>+10.0f}ms  "
              f"{d['score']:>+9.0f}ms  "
              f"{d['pd_sr']:>6.0%}  [{decision}]")

    # ── runtime test cases ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("Runtime decisions (actual output_tokens, QPS interpolation)")
    print("=" * 70)

    cases = [
        # (turn, inp, out, qps, ctx_tokens, label)
        (1,  128, 128, 4.0, 0,    "Turn 1 — always PD"),
        (2,  16,   32, 4.0, 256,  "small ctx  tiny    QPS 4"),
        (2,  32,  256, 4.0, 256,  "small ctx  short_gen QPS 4"),
        (2,  32,  512, 3.0, 256,  "small ctx  long_gen  QPS 3 (interpolated)"),
        (2,  64, 1024, 2.0, 256,  "small ctx  very_long_gen QPS 2"),
        (2, 128,  128, 4.0, 256,  "small ctx  mid_bal  QPS 4"),
        (2, 256,   64, 4.0, 256,  "small ctx  mid_paste QPS 4"),
        (2, 512,   64, 4.0, 256,  "small ctx  big_paste QPS 4"),
        (2, 512,   64, 8.0, 256,  "small ctx  big_paste QPS 8"),
        (2,1024,   32, 4.0, 256,  "small ctx  huge_paste QPS 4"),
        (2,  16,   32, 4.0, 2048, "large ctx  tiny    QPS 4"),
        (2,  32,  512, 4.0, 2048, "large ctx  long_gen QPS 4"),
        (2,  64, 1024, 1.0, 2048, "large ctx  very_long_gen QPS 1"),
        (2, 512,   64, 2.0, 2048, "large ctx  big_paste QPS 2"),
        (2, 512,   64, 8.0, 2048, "large ctx  big_paste QPS 8 (low SR → ?)"),
        (2,1024,   32, 8.0, 2048, "large ctx  huge_paste QPS 8"),
        (2, 512,   64, 6.0, 2048, "large ctx  big_paste QPS 6 (interpolated)"),
        (2,1024,   32, 4.0, 5000, "huge ctx   huge_paste QPS 4 (extrapolated)"),
    ]

    for turn, inp, out, qps, ctx_tok, label in cases:
        decision = engine.should_use_ppd(turn, inp, out, qps, ctx_tok)
        marker = "PPD ✓" if decision else "PD  ·"
        print(f"  [{marker}]  {label}")

    # ── export ────────────────────────────────────────────────────────────
    out_path = str(benchmark_path.parent / "ppd_lookup_table.json")
    engine.export_lookup_table(out_path)
    print(f"\nExported → {out_path}")
    print(f"\nStats: {engine.get_decision_stats()}")


if __name__ == "__main__":
    main()
