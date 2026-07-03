# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| none | - | - | - | - | - | 0.9854 | 0.0239 | 20.1700 | 200/200 |
| radix | - | - | - | - | - | 0.6099 | 0.0239 | 25.5000 | 200/200 |
| hicache_host | - | - | - | - | - | 0.6400 | 0.0239 | 25.1800 | 200/200 |
| hicache_file | - | - | - | - | - | 0.6395 | 0.0239 | 25.2300 | 200/200 |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

