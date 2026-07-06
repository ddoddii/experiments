# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| radix | 0.2574 | 0.7426 | 0.0000 | 0.0000 | - | 2.4010 | 0.0174 | 56.6800 | 200/200 |
| hicache_host | 0.7434 | 0.2566 | 0.0000 | 0.0000 | - | 1.4544 | 0.0234 | 65.4800 | 200/200 |
| hicache_file | 0.7434 | 0.2566 | 0.0000 | 0.0000 | - | 1.5042 | 0.0234 | 65.1300 | 200/200 |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

- **radix**: t0:2.7089, t1:2.2203, t2:2.2465, t3:2.3212, t4:2.1781, t5:2.1484, t6:2.1111
- **hicache_host**: t0:1.8201, t1:1.2894, t2:1.2268, t3:1.3286, t4:1.206, t5:1.0718, t6:1.3491
- **hicache_file**: t0:1.8321, t1:1.3094, t2:1.3236, t3:1.396, t4:1.3897, t5:1.1822, t6:1.4763
