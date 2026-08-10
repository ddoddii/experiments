
### park

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.2 | 0.692 | 10.15 | 0.7986 | 1.3178 | 4.5088 | 6.851 | 4 | 0/59 |  |
| 0.4 | 1.072 | 10.68 | 0.8993 | 1.5421 | 4.2587 | 8.042 | 6 | 0/96 |  |
| 0.6 | 0.038 | 0.1 | 1.0907 | 1.64 | 1.8974 | 2.673 | 12 | 0/138 |  |
| 0.9 | 2.535 | 91.71 | 7.928 | 12.6609 | 28.725 | 59.097 | 38 | 0/206 |  |
| 1.2 | 2.141 | 63.27 | 29.7397 | 62.6471 | 70.1028 | 119.786 | 179 | 108/309 | SATURATED |
| 1.6 | 1.022 | 30.42 | 32.3973 | 78.9878 | 83.8752 | 167.55 | 191 | 68/380 | SATURATED |

### SLO capacity (delivered tok/s while median TTFT stays under budget)

| arm | <= 0.5s | <= 1.0s | <= 2.0s |
|---|---|---|---|
| park | — | 11 | 11 |

### capacity (peak delivered throughput)

| arm | peak tok/s | at rate | TTFT p50 there |
|---|---|---|---|
| park | 91.71 | 0.9 | 7.928 |

[saved] results/qps/qps_llama8b_bfcl/qps.json
