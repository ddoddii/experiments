# INVALID RUN — the metric was wrong, not the premise

`M1 stranded headroom = 0.0 GB*s` in this directory is **an artifact of the wrong gauge**,
not a negative result. Do not cite it, and do not conclude from it that P/D imbalance is
absent.

## What happened

The sampler read `sglang:token_usage`. That gauge is defined in
`scheduler_runtime_checker_mixin.py::_get_token_info` as:

```python
available_size = self.token_to_kv_pool_allocator.available_size()
evictable_size = self.tree_cache.evictable_size()
num_used      = self.max_total_num_tokens - (available_size + evictable_size)
token_usage   = num_used / self.max_total_num_tokens
```

`evictable_size` — the prefix-cached KV — is **subtracted**. For admission control that is
correct: a cached block is reclaimable, so it is "available". But it means the gauge
measures only the *actively referenced* working set, and **a prefill pool that is 100%
full of retained prefixes reports a `token_usage` near zero.**

That is exactly what this run shows: P0/P1 at `use_mean = 0.019`. The contradiction was
visible immediately — Exp 1 measured a 55.7% cache hit rate under the same 60k pool, which
is impossible if the pool were really 2% occupied.

The failure direction is what makes it dangerous: a metric that ignores the cache reports
*no imbalance*, which is indistinguishable from a legitimate "premise is false, abandon the
experiment" verdict.

## Fix

`sglang:cache_occupancy` and `sglang:evictable_tokens` were added to the fork
(`python/sglang/srt/metrics/collector.py`, `scheduler_metrics_mixin.py`):

```
cache_occupancy = (num_used + evictable) / max_total_num_tokens
```

`kv_occupancy_timeseries.py` now prefers it and records `token_usage` alongside as
`*_pressure`, so the gap between the two — which *is* the prefix cache — stays visible.
`collect_imbalance.py` refuses to present a CSV that has only the old column without
printing a warning first.

## What this run is still good for

- Infrastructure works end to end under `PD_LAYOUT=b`: 144 sessions, 743 turns, **0 errors**,
  15m20s, avg TTFT 0.322 s. The single balanced router did not produce the 503 storm that
  invalidated the earlier two-router Exp 2.
- Decode-side headroom is real and is measured correctly here, because decode runs a chunk
  cache with essentially no evictable pool, so `token_usage ≈ cache_occupancy` on D:
  **D0/D1 held ~21.3 GB free each, ~42.6 GB combined, throughout.**
- The P/D roll-up on the *active* metric is negative (P 0.019 vs D 0.062), which is itself
  informative: on in-flight working set decode is the busier side. The imbalance this
  experiment is about lives in the **cache**, not in the working set.
