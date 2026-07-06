# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| radix | 0.6906 | 0.3094 | 0.0000 | 0.0000 | - | 1.2091 | 0.0220 | 66.5100 | 60/60 |
| hicache_host | 0.6907 | 0.3093 | 0.0000 | 0.0000 | - | 1.4620 | 0.0226 | 60.0900 | 60/60 |
| hicache_file | 0.6969 | 0.3031 | 0.0000 | 6330 | - | 1.3059 | 0.0233 | 65.4600 | 60/60 |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

- **radix**: t0:1.4866, t1:1.0905, t2:1.0148, t3:1.0183, t4:0.789
- **hicache_host**: t0:1.8134, t1:1.251, t2:1.2642, t3:1.1737, t4:1.3696
- **hicache_file**: t0:1.7668, t1:1.0877, t2:0.9407, t3:1.1522, t4:0.9442
