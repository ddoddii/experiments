
### recompute

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.343 | 180.2 | 0.4261 | 0.9377 | 9 | 0/17 |  |
| 0.1 | 0.796 | 379.76 | 0.3837 | 1.0942 | 18 | 0/39 |  |
| 0.2 | 1.435 | 594.56 | 0.3376 | 0.9459 | 31 | 0/70 |  |
| 0.35 | 2.041 | 899.09 | 0.5092 | 5.5006 | 58 | 0/112 |  |
| 0.5 | 2.725 | 1047.5 | 3.0183 | 34.3193 | 98 | 3/162 |  |
| 0.75 | 2.632 | 1105.36 | 11.1789 | 30.3736 | 140 | 60/200 | SATURATED |

### hicache

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.351 | 191.35 | 0.3413 | 0.6704 | 9 | 0/17 |  |
| 0.1 | 0.82 | 401.32 | 0.3045 | 0.7926 | 19 | 0/39 |  |
| 0.2 | 1.427 | 615.18 | 0.3158 | 1.1431 | 31 | 0/70 |  |
| 0.35 | 2.164 | 912.6 | 0.324 | 1.1383 | 54 | 0/112 |  |
| 0.5 | 2.878 | 1211.03 | 0.6293 | 20.0188 | 93 | 0/162 |  |
| 0.75 | 3.244 | 1248.72 | 0.9216 | 21.7966 | 116 | 8/200 |  |

### hicache_memfrac

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|
| 0.35 | 2.16 | 949.87 | 0.3614 | 1.162 | 54 | 0/112 |  |
| 0.5 | 2.87 | 1234.89 | 0.7648 | 20.3339 | 95 | 4/162 |  |
| 0.75 | 3.135 | 1193.56 | 1.2786 | 23.2023 | 117 | 43/200 | SATURATED |

### park

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|
| 0.05 | 0.339 | 176.14 | 0.2135 | 0.4175 | 9 | 0/17 |  |
| 0.1 | 0.8 | 389.43 | 0.2294 | 0.6274 | 18 | 0/39 |  |
| 0.2 | 1.415 | 611.42 | 0.2529 | 1.0761 | 31 | 0/70 |  |
| 0.35 | 2.102 | 929.98 | 0.4501 | 4.4262 | 57 | 0/112 |  |
| 0.5 | 2.895 | 1135.08 | 1.0653 | 20.6407 | 90 | 0/162 |  |
| 0.75 | 2.728 | 1135.47 | 2.7348 | 40.6641 | 136 | 80/200 | SATURATED |

### SLO capacity (delivered tok/s while median TTFT stays under budget)

| arm | <= 0.5s | <= 1.0s | <= 2.0s |
|---|---|---|---|
| recompute | 883 | 928 | 987 |
| hicache | 1085 | 1249 | 1249 |
| hicache_memfrac | 987 | 1235 | 1235 |
| park | 947 | 1113 | 1135 |

- **at 0.5s median TTFT, Ours carries 1.07x recompute, 0.87x hicache**

- **at 1.0s median TTFT, Ours carries 1.20x recompute, 0.89x hicache**

- **at 2.0s median TTFT, Ours carries 1.15x recompute, 0.91x hicache**

### capacity (peak delivered throughput)

| arm | peak tok/s | at rate | TTFT p50 there |
|---|---|---|---|
| recompute | 1105.36 | 0.75 | 11.1789 |
| hicache | 1248.72 | 0.75 | 0.9216 |
| hicache_memfrac | 1234.89 | 0.5 | 0.7648 |
| park | 1135.47 | 0.75 | 2.7348 |

- **Ours vs recompute: 1.03x peak throughput**

- **Ours vs hicache: 0.91x peak throughput**

[saved] results/qps/qps_llama8b/qps.json
