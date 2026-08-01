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
| host RSS (GB) | 65.66 | 73.58 | 67.35 |
| page cache (GB) | 48.8 | 94.84 | 48.38 |
| host footprint RSS+cache (GB) | 114.5 | 168.2 | 115.7 |
| AnonPages (GB) | 16.53 | 23.36 | 16.16 |
| MemAvailable min (GB) | 104.9 | 97.21 | 104.3 |
| GPU HBM total (GB) | 150.8 | 151.2 | 165.2 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 6.535 |
| local GPU park (GB) | 0 | 0 | 7.373 |
| peer GPU (GB) | 0 | 0 | 7.373 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.3755 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 659 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 659 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 0.5914 | 0.4915 | 0.4176 |
| TTFT p95 (s) | 1.589 | 1.195 | 0.9127 |
| TTFT p99 (s) | 3.542 | 3.515 | 2.341 |
| goodput (tok/s) | 165.5 | 165.8 | 165.7 |
| prefix reuse ratio | 0 | 0.5654 | 0.7806 |
| recomputed tokens | 1,119,306 | 501,058 | 252,158 |
| peak gpu0 HBM (GB) | 33.39 | 33.42 | 40.24 |
| peak gpu1 HBM (GB) | 33.33 | 33.69 | 39.52 |
| peak gpu2 HBM (GB) | 42.06 | 42.06 | 42.72 |
| peak gpu3 HBM (GB) | 42.06 | 42.06 | 42.72 |

[collect] -> results/models/qwen14b/table.json
