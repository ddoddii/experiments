| metric | recompute | hicache | park |
|---|---|---|---|
| **Validity** | | | |
| turns | 800 | 800 | 800 |
| infra failures (outage) | 0 | 0 | 0 |
| empty (200, no tokens) | 0 | 0 | 0 |
| infra fail rate | 0 | 0 | 0 |
| workload rejections (400) | 0 | 0 | 0 |
| workload fail rate | 0 | 0 | 0 |
| **Memory (peak)** | | | |
| host RSS (GB) | 64.32 | 79.51 | 65.8 |
| page cache (GB) | 73.11 | 87.36 | 29.98 |
| host footprint RSS+cache (GB) | 137.4 | 166.6 | 95.78 |
| AnonPages (GB) | 14.81 | 29.79 | 15.17 |
| MemAvailable min (GB) | 106.5 | 90.14 | 104.9 |
| GPU HBM total (GB) | 147.2 | 147 | 165 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 13 |
| local GPU park (GB) | 0 | 0 | 8.192 |
| peer GPU (GB) | 0 | 0 | 8.192 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.4274 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 346 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 346 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 0.5409 | 0.5599 | 0.571 |
| TPOT p50 (s/token) | 0.043 | 0.0431 | 0.0433 |
| TTFT p95 (s) | 1.452 | 1.533 | 1.382 |
| TTFT p99 (s) | 2.693 | 2.885 | 2.118 |
| goodput (tok/s) | 151.9 | 151 | 151.8 |
| prefix reuse ratio | 0 | 0.3515 | 0.583 |
| recomputed tokens | 261,394 | 177,165 | 107,506 |
| peak gpu0 HBM (GB) | 32.08 | 32.11 | 40.38 |
| peak gpu1 HBM (GB) | 32.09 | 31.88 | 40 |
| peak gpu2 HBM (GB) | 41.51 | 41.51 | 42.31 |
| peak gpu3 HBM (GB) | 41.51 | 41.52 | 42.32 |

[collect] -> results/models/llama13b/table.json
