| metric | radix | hicache | park |
|---|---|---|---|
| **Validity** | | | |
| turns | 688 | 692 | 698 |
| infra failures (outage) | 0 | 0 | 0 |
| empty (200, no tokens) | 0 | 0 | 0 |
| infra fail rate | 0 | 0 | 0 |
| workload rejections (400) | 39 | 34 | 37 |
| workload fail rate | 0.057 | 0.049 | 0.053 |
| **Memory (peak)** | | | |
| host RSS (GB) | 65.87 | 83.33 | 67.09 |
| page cache (GB) | 53.16 | 85.33 | 31.18 |
| host footprint RSS+cache (GB) | 119 | 168.6 | 98.27 |
| AnonPages (GB) | 16.28 | 33.89 | 16.51 |
| MemAvailable min (GB) | 104.6 | 86.46 | 102.5 |
| GPU HBM total (GB) | 135 | 134.9 | 145.1 |
| **KV residency** | | | |
| local GPU serving (GB) | 0 | 0 | 15.66 |
| local GPU park (GB) | 0 | 0 | 3.932 |
| peer GPU (GB) | 0 | 0 | 3.932 |
| CPU DRAM overflow (GB) | 0 | 0 | 0 |
| dropped, cumulative (GB) | 0 | 0 | 0 |
| peer share of parked | 0 | 0 | 0.448 |
| host share of parked | 0 | 0 | 0 |
| **Fetch source** | | | |
| fetch hits | 0 | 0 | 126 |
|   from local park | 0 | 0 | 0 |
|   from peer GPU | 0 | 0 | 126 |
|   from CPU DRAM | 0 | 0 | 0 |
| gave up (no space) | 0 | 0 | 0 |
| **Performance** | | | |
| TTFT p50 (s) | 1.688 | 2.001 | 2.106 |
| TTFT p95 (s) | 3.61 | 4.336 | 4.838 |
| TTFT p99 (s) | 4.576 | 5.443 | 6.506 |
| goodput (tok/s) | 105.7 | 102 | 122.4 |
| prefix reuse ratio | 0.4463 | 0.4766 | 0.4972 |
| recomputed tokens | 864,500 | 798,913 | 787,288 |
| peak gpu0 HBM (GB) | 24.96 | 24.8 | 29.62 |
| peak gpu1 HBM (GB) | 24.78 | 24.84 | 28.92 |
| peak gpu2 HBM (GB) | 42.63 | 42.63 | 43.29 |
| peak gpu3 HBM (GB) | 42.63 | 42.63 | 43.3 |
