| metric | radix | hicache | park |
|---|---|---|---|
| **Memory (peak)** | | | |
| host RSS (GB) | 65.64 | 83.19 | 66.71 |
| page cache (GB) | 98.68 | 86.1 | 71.99 |
| host footprint RSS+cache (GB) | 164.3 | 169.1 | 138.7 |
| AnonPages (GB) | 16.07 | 33.62 | 16.2 |
| MemAvailable min (GB) | 105.4 | 87.6 | 104.3 |
| GPU HBM total (GB) | 135 | 134.9 | 144.9 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 15.63 |
| local GPU park (GB) | 0 | 0 | 3.932 |
| peer GPU (GB) | 0 | 0 | 3.932 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.4584 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 14 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 14 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 11 |
| **Performance** | | | |
| TTFT p50 (s) | 3.535 | 4.75 | 4.604 |
| TTFT p95 (s) | 7.543 | 7.852 | 7.291 |
| TTFT p99 (s) | 7.543 | 7.852 | 7.291 |
| goodput (tok/s) | 6.98 | 6.99 | 6.79 |
| prefix reuse ratio | 0.6835 | 0.7849 | 0.7253 |
| recomputed tokens | 166,928 | 108,454 | 160,513 |
| peak gpu0 HBM (GB) | 24.82 | 24.84 | 29.42 |
| peak gpu1 HBM (GB) | 24.96 | 24.84 | 28.9 |
| peak gpu2 HBM (GB) | 42.63 | 42.63 | 43.29 |
| peak gpu3 HBM (GB) | 42.63 | 42.63 | 43.29 |

[collect] -> results/exp1/p60000_c16_d3/table.json
