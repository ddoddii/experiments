| metric | radix | hicache | park_local | park |
|---|---|---|---|---|
| **Memory (peak)** | | | | |
| host RSS (GB) | 65.3 | 83.04 | 66.57 | 66.6 |
| page cache (GB) | 53.08 | 64.51 | 53.1 | 53.12 |
| host footprint RSS+cache (GB) | 118.4 | 147.6 | 119.7 | 119.7 |
| AnonPages (GB) | 15.63 | 33.38 | 16.11 | 16.16 |
| MemAvailable min (GB) | 105.9 | 87.85 | 104.3 | 104.3 |
| GPU HBM total (GB) | 134.4 | 134.3 | 144.2 | 144.2 |
| **KV residency** | | | | |
| local GPU serving (GB) | 0 | 0 | 7.826 | 7.851 |
| local GPU park (GB) | 0 | 0 | 3.932 | 1.966 |
| peer GPU (GB) | 0 | 0 | 0 | 0 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0 | 0 |
| host share of parked | 0 | 0 | 0 | 0 |
| **Fetch source** | | | | |
| fetch hits | 0 | 0 | 5 | 2 |
|   from local park | 0 | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 5 | 2 |
|   from CPU DRAM | 0 | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 1 | 5 |
| **Performance** | | | | |
| TTFT p50 (s) | 5.193 | 3.414 | 6.316 | 6.763 |
| TTFT p95 (s) | 9.822 | 11.08 | 10.82 | 10.75 |
| TTFT p99 (s) | 10.38 | 13.26 | 10.82 | 10.75 |
| goodput (tok/s) | 45.35 | 48.43 | 6.66 | 6.55 |
| prefix reuse ratio | — | — | 0.4994 | 0.5496 |
| recomputed tokens | — | — | 386,110 | 413,071 |
| peak gpu0 HBM (GB) | 24.21 | 24.22 | 29.5 | 29.54 |
| peak gpu1 HBM (GB) | 24.95 | 24.84 | 28.22 | 28.26 |
| peak gpu2 HBM (GB) | 42.63 | 42.63 | 43.23 | 43.23 |
| peak gpu3 HBM (GB) | 42.63 | 42.63 | 43.23 | 43.23 |

[collect] -> results/exp2/skew0.9_p60000_c16/table.json
