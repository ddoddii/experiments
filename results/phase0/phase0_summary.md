# Phase 0 baseline — recompute vs fetch

| mode | reuse_ratio | recompute_ratio | cache_hit_rate | L3_prefetched | L2_host_used | avg_ttft_s | avg_tpot_s | overall_tput | success |
|---|---|---|---|---|---|---|---|---|---|
| none | 0.0000 | 1.0000 | 0.0000 | 0.0000 | - | - | - | - | ?/? |
| radix | 0.7451 | 0.2549 | 0.0000 | 0.0000 | - | - | - | - | ?/? |
| hicache_host | 0.7451 | 0.2549 | 0.0000 | 0.0000 | - | - | - | - | ?/? |
| hicache_file | 0.7451 | 0.2549 | 0.0000 | 0.0000 | - | - | - | - | ?/? |

## Per-turn 평균 TTFT (context 성장 → TTFT 성장; prefix hit면 완만)

