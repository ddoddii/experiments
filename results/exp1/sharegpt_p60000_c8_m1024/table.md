| metric | radix | hicache | park |
|---|---|---|---|
| **Validity** | | | |
| turns | 1440 | 1440 | 1440 |
| infra failures (outage) | 0 | 0 | 0 |
| empty (200, no tokens) | 0 | 0 | 0 |
| infra fail rate | 0 | 0 | 0 |
| workload rejections (400) | 0 | 0 | 0 |
| workload fail rate | 0 | 0 | 0 |
| **Memory (peak)** | | | |
| host RSS (GB) | 65.83 | 83.51 | 67.22 |
| page cache (GB) | 37.94 | 82.81 | 31.71 |
| host footprint RSS+cache (GB) | 103.8 | 166.3 | 98.92 |
| AnonPages (GB) | 17.48 | 35.75 | 17.82 |
| MemAvailable min (GB) | 103.9 | 84.7 | 102.6 |
| GPU HBM total (GB) | 135.8 | 135.3 | 144.6 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 15.66 |
| local GPU park (GB) | 0 | 0 | 3.932 |
| peer GPU (GB) | 0 | 0 | 3.932 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.4441 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 1,211 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 1,211 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 0.2937 | 0.2958 | 0.2368 |
| TTFT p95 (s) | 0.7087 | 0.7641 | 0.5934 |
| TTFT p99 (s) | 1.022 | 1.353 | 1.582 |
| goodput (tok/s) | 248.5 | 248.3 | 247.7 |
| prefix reuse ratio | 0.5415 | 0.5231 | 0.9547 |
| recomputed tokens | 522,422 | 525,300 | 51,246 |
| peak gpu0 HBM (GB) | 25.32 | 25.18 | 29.38 |
| peak gpu1 HBM (GB) | 25.22 | 24.9 | 28.66 |
| peak gpu2 HBM (GB) | 42.63 | 42.63 | 43.29 |
| peak gpu3 HBM (GB) | 42.63 | 42.63 | 43.29 |

[collect] -> results/exp1/sharegpt_p60000_c8_m1024/table.json
