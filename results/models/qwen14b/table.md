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
| host RSS (GB) | 65.65 | 73.5 | 67.21 |
| page cache (GB) | 45.08 | 93.52 | 40.02 |
| host footprint RSS+cache (GB) | 110.7 | 166.9 | 107.2 |
| AnonPages (GB) | 16.3 | 24.73 | 16.91 |
| MemAvailable min (GB) | 105 | 96.47 | 103.6 |
| GPU HBM total (GB) | 152.1 | 150.4 | 166.1 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 6.517 |
| local GPU park (GB) | 0 | 0 | 7.373 |
| peer GPU (GB) | 0 | 0 | 7.373 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.439 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 0 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 0 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 0.6995 | 0.5368 | 0.5516 |
| TTFT p95 (s) | 2.026 | 1.708 | 1.992 |
| TTFT p99 (s) | 3.686 | 3.923 | 4.526 |
| goodput (tok/s) | 155.6 | 156.6 | 156.4 |
| prefix reuse ratio | 0 | 0.5056 | 0.4511 |
| recomputed tokens | 1,399,595 | 681,236 | 759,795 |
| peak gpu0 HBM (GB) | 33.9 | 32.97 | 40.64 |
| peak gpu1 HBM (GB) | 34.06 | 33.29 | 39.98 |
| peak gpu2 HBM (GB) | 42.06 | 42.06 | 42.72 |
| peak gpu3 HBM (GB) | 42.06 | 42.06 | 42.72 |

[collect] -> results/models/qwen14b/table.json
