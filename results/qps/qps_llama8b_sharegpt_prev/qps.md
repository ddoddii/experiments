
### recompute

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.343 | 178.59 | 0.4432 | 0.9269 | 96.89 | 164.577 | 9 | 0/17 |  |
| 0.1 | 0.808 | 380.52 | 0.374 | 0.961 | 107.0998 | 158.447 | 18 | 0/39 |  |
| 0.2 | 1.447 | 599.26 | 0.361 | 1.0827 | 97.3587 | 160.947 | 30 | 0/70 |  |
| 0.35 | 2.107 | 918.86 | 0.4887 | 2.1116 | 106.8562 | 178.083 | 56 | 0/112 |  |
| 0.5 | 3.172 | 1218.14 | 0.8263 | 15.0449 | 126.2945 | 193.7 | 86 | 0/162 |  |
| 0.75 | 2.973 | 1225.85 | 1.4036 | 23.3203 | 127.9257 | 188.051 | 131 | 39/200 | SATURATED |

### hicache

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.343 | 182.88 | 0.2978 | 0.6869 | 99.6617 | 167.402 | 9 | 0/17 |  |
| 0.1 | 0.82 | 385.03 | 0.3145 | 0.7713 | 97.6565 | 136.942 | 19 | 0/39 |  |
| 0.2 | 1.435 | 618.72 | 0.3119 | 0.9201 | 101.2109 | 161.719 | 31 | 0/70 |  |
| 0.35 | 1.946 | 885.98 | 0.4909 | 11.4663 | 114.8638 | 194.269 | 63 | 0/112 |  |
| 0.5 | 2.132 | 745.42 | 4.6517 | 38.8203 | 109.7686 | 185.881 | 123 | 76/162 | SATURATED |

### park

| rate (sess/s) | turn rate/s | throughput (tok/s) | TTFT p50 | TTFT p95 | job delay mean | job delay p95 | peak inflight | unfinished | flags |
|---|---|---|---|---|---|---|---|---|---|
| 0.05 | 0.347 | 185.6 | 0.2955 | 0.6228 | 109.8716 | 165.787 | 10 | 0/17 |  |
| 0.1 | 0.824 | 401.04 | 0.2797 | 0.6602 | 99.0259 | 139.731 | 18 | 0/39 |  |
| 0.2 | 1.459 | 604.04 | 0.2964 | 0.7713 | 99.2709 | 162.174 | 31 | 0/70 |  |
| 0.35 | 2.094 | 903.28 | 0.4236 | 11.0356 | 108.6067 | 186.757 | 56 | 0/112 |  |
| 0.5 | 2.754 | 1075.15 | 1.2238 | 28.7792 | 126.5004 | 188.392 | 96 | 0/162 |  |
| 0.75 | 2.599 | 1058.93 | 5.4658 | 48.9982 | 146.6943 | 213.141 | 144 | 62/200 | SATURATED |

### SLO capacity (delivered tok/s while median TTFT stays under budget)

| arm | <= 0.5s | <= 1.0s | <= 2.0s |
|---|---|---|---|
| recompute | 929 | 1220 | 1226 |
| hicache | 886 | 886 | 886 |
| park | 906 | 921 | 1075 |

- **at 0.5s median TTFT, Ours carries 0.97x recompute, 1.02x hicache**

- **at 1.0s median TTFT, Ours carries 0.75x recompute, 1.04x hicache**

- **at 2.0s median TTFT, Ours carries 0.88x recompute, 1.21x hicache**

### capacity (peak delivered throughput)

| arm | peak tok/s | at rate | TTFT p50 there |
|---|---|---|---|
| recompute | 1225.85 | 0.75 | 1.4036 |
| hicache | 885.98 | 0.35 | 0.4909 |
| park | 1075.15 | 0.5 | 1.2238 |

- **Ours vs recompute: 0.88x peak throughput**

- **Ours vs hicache: 1.21x peak throughput**

[saved] results/qps/qps_llama8b_sharegpt/qps.json
