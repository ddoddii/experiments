[WARN] occupancy == pressure on every sample: the server did not export sglang:cache_occupancy and the sampler fell back to token_usage. Rebuild the SGLang source tree before trusting these numbers.

### hicache   (1889 samples, 944.1s)
  metric: token_usage (FALLBACK)
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%
   P0     P     7.32   0.022   0.022   0.000   0.075      7.16    0.0%
   P1     P     7.32   0.021   0.021   0.000   0.068      7.17    0.0%
   D0     D    22.64   0.062   0.062   0.000   0.114     21.24    0.0%
   D1     D    22.64   0.065   0.065   0.000   0.114     21.17    0.0%
  P mean 0.021 vs D mean 0.064   -> P/D gap -0.042
  M1 stranded headroom : 0.0 GB*s   (0.0 GB average while saturated)
     saturated for      : 0.0s of 944.1s (0.0%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 0.0, 'D1': 0.0}
     vs threshold       : {'hi=0.7': 0.0, 'hi=0.8': 0.0, 'hi=0.85': 0.0, 'hi=0.9': 0.0, 'hi=0.95': 0.0}
  M2 imbalance         : CoV 0.7585  spread mean 0.0761  p95 0.1249  Jain 0.6438

[collect] -> results/exp2/pd_layoutb_p60000_c16/imbalance.json
