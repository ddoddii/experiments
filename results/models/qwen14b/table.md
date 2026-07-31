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
| host RSS (GB) | 66.18 | 73.46 | 67.3 |
| page cache (GB) | 93.46 | 94.43 | 43.64 |
| host footprint RSS+cache (GB) | 159.6 | 167.5 | 110.9 |
| AnonPages (GB) | 21 | 33 | 18.33 |
| MemAvailable min (GB) | 100.3 | 88.17 | 102 |
| GPU HBM total (GB) | 150.2 | 151.1 | 156.7 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 6.537 |
| local GPU park (GB) | 0 | 0 | 1.966 |
| peer GPU (GB) | 0 | 0 | 1.966 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.4872 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 0 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 0 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 0.5523 | 0.5306 | 0.5476 |
| TTFT p95 (s) | 1.926 | 1.735 | 1.957 |
| TTFT p99 (s) | 5.009 | 3.25 | 4.167 |
| goodput (tok/s) | 156.7 | 157.5 | 156 |
| prefix reuse ratio | 0.4746 | 0.4662 | 0.468 |
| recomputed tokens | 750,508 | 718,090 | 764,844 |
| peak gpu0 HBM (GB) | 33.19 | 34.09 | 35.69 |
| peak gpu1 HBM (GB) | 32.93 | 32.88 | 35.51 |
| peak gpu2 HBM (GB) | 42.06 | 42.06 | 42.72 |
| peak gpu3 HBM (GB) | 42.06 | 42.06 | 42.72 |

[collect] -> results/models/qwen14b/table.json
