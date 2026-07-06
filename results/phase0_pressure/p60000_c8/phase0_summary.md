# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| radix | 0.7433 | 0.2567 | 0.0000 | 0.0000 | - | 2.7012 | 0.0235 | 83.3800 | 200/200 |
| hicache_host | 0.7438 | 0.2562 | 0.0000 | 0.0000 | - | 2.9982 | 0.0232 | 77.8900 | 200/200 |
| hicache_file | 0.7514 | 0.2486 | 0.0000 | 29664 | - | 3.0351 | 0.0231 | 78.2000 | 200/200 |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

- **radix**: t0:2.7921, t1:2.6986, t2:2.6226, t3:2.7259, t4:2.7968, t5:2.6721, t6:3.3933
- **hicache_host**: t0:3.1675, t1:2.9415, t2:2.8552, t3:2.9455, t4:3.0543, t5:3.1551, t6:3.4966
- **hicache_file**: t0:3.1585, t1:2.9801, t2:2.9518, t3:2.9928, t4:3.085, t5:3.2251, t6:3.4493
