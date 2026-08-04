
3 repeats: repeat_nocap_c32_r1, repeat_nocap_c32_r2, repeat_nocap_c32_r3

  park hit rate (%)
      46.86→48.69       45.74→49.56       44.67→50.19  
    Ours better in ALL, median |Δ| = 8.4%

  park-fetch misses
     762.00→723.00     751.00→692.00     763.00→668.00 
    Ours better in ALL, median |Δ| = 7.9%

  parked KV (GB)
       2.62→7.86         2.62→7.86         2.62→7.86   
    Ours better in ALL, median |Δ| = 199.9%

  TTFT p50 (s)
       0.32→0.32         0.32→0.27         0.30→0.28   
    Ours better in ALL, median |Δ| = 7.5%

  TTFT p95 (s)
       1.82→4.33         3.08→1.97         1.45→2.39   
    SPLIT 1 better / 2 worse / 0 tied, median |Δ| = 64.7%  <-- SPLIT

  TTFT p99 (s)
       6.32→8.22         7.05→8.33         4.10→7.30   
    Ours worse in ALL, median |Δ| = 30.0%

  throughput (t/s)
     834.21→843.09     762.83→759.19     803.88→804.52 
    SPLIT 1 better / 0 worse / 2 tied, median |Δ| = 0.5%  <-- SPLIT

================================================================
  NOT reproducible on: ['ttft_p95_s', 'overall_throughput_tok_per_s']
  Do not quote a single-run figure for these; report the range, or run more repeats.

[collect] -> results/exp2/repeat_nocap_c32_summary.json
