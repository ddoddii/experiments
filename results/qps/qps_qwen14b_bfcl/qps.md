
### recompute

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.395 | 138.37 | 1.5487 | 2.4305 | 63.3416 | 95.501 | 11 | 0/31 |  |
| 0.2 | 0.623 | 178.11 | 1.9771 | 4.9191 | 49.1842 | 81.493 | 17 | 0/59 |  |
| 0.35 | 0.826 | 251.23 | 3.1807 | 9.0987 | 50.97 | 95.556 | 29 | 0/85 |  |
| 0.5 | 0.005 | 0.42 | 28.8322 | 28.8322 | None | None | 55 | 3/125 |  |
| 0.7 | 0.41 | 139.21 | 47.6163 | 96.2643 | 58.8436 | 176.799 | 137 | 132/176 | SATURATED |

### hicache

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.371 | 131.99 | 1.4636 | 2.5485 | 60.5112 | 91.645 | 11 | 0/31 |  |
| 0.2 | 0.641 | 173.73 | 1.7464 | 3.8251 | 45.6619 | 79.884 | 17 | 0/59 |  |
| 0.35 | 0.789 | 236.81 | 6.0454 | 20.7636 | 61.2665 | 116.714 | 34 | 7/85 |  |
| 0.5 | 0.014 | 3.88 | 23.7865 | 35.3596 | 36.766 | 43.568 | 35 | 0/125 |  |
| 0.7 | 0.467 | 124.53 | 42.7013 | 115.0587 | 51.7643 | 171.856 | 135 | 127/176 | SATURATED |

### park

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.1 | 0.395 | 145.8 | 0.88 | 1.9587 | 65.4119 | 103.981 | 11 | 0/31 |  |
| 0.2 | 0.609 | 169.0 | 1.7 | 3.2952 | 45.1215 | 68.806 | 17 | 0/59 |  |
| 0.35 | 0.742 | 213.09 | 8.2381 | 20.7243 | 56.7465 | 105.291 | 36 | 9/85 | SATURATED |
| 0.5 | 0.019 | 3.01 | 26.0737 | 47.8728 | 46.1457 | 55.908 | 48 | 1/125 |  |
| 0.7 | 0.363 | 127.08 | 49.4196 | 102.2968 | 91.771 | 125.786 | 137 | 130/176 | SATURATED |

### SLO capacity (delivered tok/s while median TTFT stays under budget)

| arm | <= 0.5s | <= 1.0s | <= 2.0s |
|---|---|---|---|
| recompute | — | — | 180 |
| hicache | — | — | 177 |
| park | — | 149 | 171 |

- **at 2.0s median TTFT, Ours carries 0.95x recompute, 0.96x hicache**

### capacity (peak delivered throughput)

| arm | peak tok/s | at rate | TTFT p50 there |
|---|---|---|---|
| recompute | 251.23 | 0.35 | 3.1807 |
| hicache | 236.81 | 0.35 | 6.0454 |
| park | 213.09 | 0.35 | 8.2381 |

- **Ours vs recompute: 0.85x peak throughput**

- **Ours vs hicache: 0.90x peak throughput**

[saved] results/qps/qps_qwen14b_bfcl/qps.json
