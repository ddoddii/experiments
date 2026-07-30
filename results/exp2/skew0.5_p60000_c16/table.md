| metric | radix | hicache | park_local | park |
|---|---|---|---|---|
| **Memory (peak)** | | | | |
| host RSS (GB) | 65.47 | 83.08 | 66.54 | 66.57 |
| page cache (GB) | 72.01 | 85.89 | 53.06 | 53.07 |
| host footprint RSS+cache (GB) | 137.5 | 169 | 119.6 | 119.6 |
| AnonPages (GB) | 15.92 | 33.45 | 16.08 | 16.1 |
| MemAvailable min (GB) | 105.6 | 87.77 | 104.4 | 104.4 |
| GPU HBM total (GB) | 134.3 | 134.3 | 144.2 | 144.2 |
| **KV residency** | | | | |
| local GPU serving (GB) | 0 | 0 | 7.85 | 7.863 |
| local GPU park (GB) | 0 | 0 | 3.932 | 1.966 |
| peer GPU (GB) | 0 | 0 | 0 | 0 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0 | 0 |
| host share of parked | 0 | 0 | 0 | 0 |
| **Fetch source** | | | | |
| fetch hits | 0 | 0 | 4 | 0 |
|   from local park | 0 | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 4 | 0 |
|   from CPU DRAM | 0 | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 12 | 1 |
| **Performance** | | | | |
| TTFT p50 (s) | 6.421 | 4.559 | 6.761 | 6.99 |
| TTFT p95 (s) | 10.95 | 11.75 | 11.28 | 10.98 |
| TTFT p99 (s) | 10.95 | 12.95 | 11.28 | 10.98 |
| goodput (tok/s) | 7.25 | 29.18 | 7.14 | 1.92 |
| prefix reuse ratio | 0.4775 | — | 0.4407 | 0.2932 |
| recomputed tokens | 460,090 | — | 455,423 | 1,887,580 |
| peak gpu0 HBM (GB) | 24.82 | 24.22 | 29.5 | 29.54 |
| peak gpu1 HBM (GB) | 24.21 | 24.84 | 28.22 | 28.26 |
| peak gpu2 HBM (GB) | 42.63 | 42.63 | 43.22 | 43.22 |
| peak gpu3 HBM (GB) | 42.63 | 42.63 | 43.23 | 43.23 |

[collect] -> results/exp2/skew0.5_p60000_c16/table.json
