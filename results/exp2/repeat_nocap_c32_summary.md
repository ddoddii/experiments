
3 repeats: repeat_nocap_c32_r1, repeat_nocap_c32_r2, repeat_nocap_c32_r3

  park-fetch hit rate (%)
      46.86→48.69       45.74→49.56       44.67→50.19  
    Ours better in ALL, median |Δ| = 8.4%
    baseline alone varies 4.8% across repeats

  park-fetch misses
     762.00→723.00     751.00→692.00     763.00→668.00 
    Ours better in ALL, median |Δ| = 7.9%
    baseline alone varies 1.6% across repeats

  parked KV (GB)   [configured, not measured]
       2.62→7.86         2.62→7.86         2.62→7.86   
    Ours better in ALL, median |Δ| = 199.9%

  TTFT p50 (s)
       0.32→0.32         0.32→0.27         0.30→0.28   
    Ours better in ALL, median |Δ| = 7.5%
    baseline alone varies 5.7% across repeats

  TTFT p95 (s)
       1.82→4.33         3.08→1.97         1.45→2.39   
    SPLIT 1 better / 2 worse / 0 tied, median |Δ| = 64.7%  <-- SPLIT
    baseline alone varies 89.4% across repeats   <-- EFFECT IS SMALLER THAN THIS

  TTFT p99 (s)
       6.32→8.22         7.05→8.33         4.10→7.30   
    Ours worse in ALL, median |Δ| = 30.0%
    baseline alone varies 46.8% across repeats   <-- EFFECT IS SMALLER THAN THIS

  throughput (t/s)
     834.21→843.09     762.83→759.19     803.88→804.52 
    SPLIT 1 better / 0 worse / 2 tied, median |Δ| = 0.5%  <-- SPLIT
    baseline alone varies 8.9% across repeats   <-- EFFECT IS SMALLER THAN THIS

================================================================
  NOT reproducible on: ['ttft_p95_s', 'overall_throughput_tok_per_s']
  Do not quote a single-run figure for these; report the range, or run more repeats.
  Effect smaller than the baseline's own run-to-run spread: ['ttft_p95_s', 'ttft_p99_s', 'overall_throughput_tok_per_s']
  A consistent sign here is not yet evidence -- n is too small to separate the effect from the noise the baseline shows on its own.

[collect] -> results/exp2/repeat_nocap_c32_summary.json
