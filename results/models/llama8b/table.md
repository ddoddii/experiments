| metric | recompute | hicache | park |
|---|---|---|---|
| **Validity** | | | |
| turns | 1440 | 1440 | 1440 |
| infra failures (outage) | 0 | 0 | 0 |
| empty (200, no tokens) | 0 | 0 | 0 |
| infra fail rate | 0 | 0 | 0 |
| workload rejections (400) | 0 | 0 | 0 |
| workload fail rate | 0 | 0 | 0 |
| **Memory (peak)** | | | |
| host RSS (GB) | 65.39 | 83.45 | 67.1 |
| page cache (GB) | 50.49 | 84.13 | 44.29 |
| host footprint RSS+cache (GB) | 115.9 | 167.5 | 111.4 |
| AnonPages (GB) | 16.56 | 34.18 | 16.64 |
| MemAvailable min (GB) | 104.9 | 86.36 | 103.6 |
| GPU HBM total (GB) | 138 | 136.1 | 145 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 15.6 |
| local GPU park (GB) | 0 | 0 | 3.932 |
| peer GPU (GB) | 0 | 0 | 3.932 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.4582 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 1,204 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 1,204 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 0.3549 | 0.3131 | 0.2381 |
| TTFT p95 (s) | 0.9547 | 0.7646 | 0.6202 |
| TTFT p99 (s) | 1.232 | 1.057 | 0.9917 |
| goodput (tok/s) | 245.9 | 249.1 | 247.8 |
| prefix reuse ratio | 0 | 0.5572 | 0.9571 |
| recomputed tokens | 1,176,324 | 544,324 | 52,393 |
| peak gpu0 HBM (GB) | 26.34 | 25 | 29.46 |
| peak gpu1 HBM (GB) | 26.44 | 25.81 | 28.93 |
| peak gpu2 HBM (GB) | 42.63 | 42.63 | 43.29 |
| peak gpu3 HBM (GB) | 42.63 | 42.63 | 43.29 |

[collect] -> results/models/llama8b/table.json
