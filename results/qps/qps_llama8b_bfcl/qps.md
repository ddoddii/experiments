
### recompute

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 1.26 | 16.41 | 1.2993 | 2.4848 | 5.9696 | 9.174 | 9 | 0/48 |  |
| 0.6 | 1.621 | 11.6 | 1.5277 | 3.4463 | 6.4571 | 12.366 | 11 | 0/70 |  |
| 1.0 | 1.4 | 27.75 | 7.4038 | 12.6754 | 34.6209 | 55.01 | 32 | 0/124 |  |
| 1.8 | 1.247 | 13.08 | 10.6526 | 18.2361 | 19.6167 | 40.834 | 83 | 0/201 |  |
| 3.0 | 2.405 | 40.87 | 36.4664 | 56.3696 | 58.1074 | 73.142 | 327 | 65/365 | SATURATED |

### hicache

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 1.26 | 17.56 | 1.172 | 2.7645 | 5.9471 | 9.074 | 9 | 0/48 |  |
| 0.6 | 1.68 | 18.17 | 1.2825 | 2.715 | 6.0382 | 11.991 | 9 | 0/70 |  |
| 1.0 | 1.39 | 27.42 | 4.1534 | 11.5399 | 25.4352 | 44.417 | 23 | 0/124 |  |
| 1.8 | 1.485 | 14.84 | 8.1675 | 17.2911 | 20.4824 | 37.154 | 76 | 0/201 |  |
| 3.0 | 2.336 | 42.56 | 33.9788 | 58.0715 | 58.1558 | 76.097 | 324 | 222/365 | SATURATED |

### park

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.3 | 1.28 | 15.53 | 0.8585 | 1.7166 | 4.8324 | 6.784 | 7 | 0/48 |  |
| 0.6 | 1.591 | 36.06 | 1.1112 | 1.9919 | 5.0162 | 10.255 | 8 | 0/70 |  |
| 1.0 | 1.4 | 35.54 | 5.0314 | 15.3177 | 33.5605 | 55.196 | 30 | 0/124 |  |
| 1.8 | 1.317 | 17.37 | 9.5475 | 16.5901 | 18.4635 | 32.965 | 80 | 0/201 |  |
| 3.0 | 2.416 | 43.83 | 35.6096 | 55.5824 | 58.5078 | 87.22 | 320 | 220/365 | SATURATED |

### SLO capacity (delivered tok/s while median TTFT stays under budget)

| arm | <= 0.5s | <= 1.0s | <= 2.0s |
|---|---|---|---|
| recompute | — | — | 18 |
| hicache | — | — | 20 |
| park | — | 16 | 36 |

- **at 2.0s median TTFT, Ours carries 2.05x recompute, 1.77x hicache**

### capacity (peak delivered throughput)

| arm | peak tok/s | at rate | TTFT p50 there |
|---|---|---|---|
| recompute | 40.87 | 3.0 | 36.4664 |
| hicache | 42.56 | 3.0 | 33.9788 |
| park | 43.83 | 3.0 | 35.6096 |

- **Ours vs recompute: 1.07x peak throughput**

- **Ours vs hicache: 1.03x peak throughput**

[saved] results/qps/qps_llama8b_bfcl/qps.json
