# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| radix | 0.5620 | 0.4380 | 0.0000 | 0.0000 | - | 4.8307 | 0.0213 | 77.8700 | 200/200 |
| hicache_host | 0.7437 | 0.2563 | 0.0000 | 0.0000 | - | 4.9614 | 0.0230 | 78.3500 | 200/200 |
| hicache_file | 0.7522 | 0.2478 | 0.0000 | 32398 | - | 4.9222 | 0.0236 | 79.2400 | 200/200 |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

- **radix**: t0:4.9569, t1:4.8344, t2:4.8333, t3:4.831, t4:4.8082, t5:5.1626, t6:4.962
- **hicache_host**: t0:5.1696, t1:4.9946, t2:4.8394, t3:4.8918, t4:4.7524, t5:4.8496, t6:5.4048
- **hicache_file**: t0:5.1106, t1:4.872, t2:4.8506, t3:4.8387, t4:4.7201, t5:4.94, t6:5.403
