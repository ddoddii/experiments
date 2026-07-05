# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| radix | 0.0601 | 0.9399 | 0.0000 | 0.0000 | - | 10.7595 | 0.0134 | 54.5700 | 200/200 |
| hicache_host | 0.1537 | 0.8463 | 0.0000 | 0.0000 | - | 12.1160 | 0.0139 | 48.9600 | 200/200 |
| hicache_file | 0.3527 | 0.6473 | 0.0000 | 924061 | - | 9.2753 | 0.0176 | 61.4200 | 200/200 |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

- **radix**: t0:10.6453, t1:10.8933, t2:10.8531, t3:10.7415, t4:10.3378, t5:9.7045, t6:8.855
- **hicache_host**: t0:11.9378, t1:12.177, t2:12.3723, t3:11.8459, t4:11.4173, t5:10.3275, t6:8.3573
- **hicache_file**: t0:9.0912, t1:9.405, t2:9.4826, t3:9.0284, t4:8.5097, t5:7.8403, t6:7.9708
