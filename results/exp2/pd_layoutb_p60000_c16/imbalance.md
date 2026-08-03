[WARN] ['D0', 'D1'] reported via {'P0': 'cache_occupancy', 'P1': 'cache_occupancy', 'D0': 'token_usage', 'D1': 'token_usage'}. Those servers do not export sglang:cache_occupancy, so their occupancy EXCLUDES prefix-cached KV. Pull the SGLang source tree and restart before trusting them.

### hicache   (1849 samples, 924.1s)
  metric: token_usage (FALLBACK)
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%
   P0     P     7.32   0.880   0.017   0.863   0.998      0.88   83.6%
   P1     P     7.32   0.894   0.021   0.873   0.998      0.78   84.3%
   D0     D    22.64   0.065   0.065   0.000   0.114     21.18    0.0%
   D1     D    22.64   0.061   0.060   0.001   0.109     21.26    0.0%
  P mean 0.887 vs D mean 0.063   -> P/D gap +0.824
  M1 stranded headroom : 33028.7 GB*s   (42.42 GB average while saturated)
     saturated for      : 778.6s of 924.1s (84.2%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 16454.1, 'D1': 16574.6}
     vs threshold       : {'hi=0.7': 34191.0, 'hi=0.8': 34001.7, 'hi=0.85': 33874.4, 'hi=0.9': 33028.7, 'hi=0.95': 32515.6}
  M2 imbalance         : CoV 0.8456  spread mean 0.8518  p95 0.9684  Jain 0.5845
  GPU HBM peak/free    : gpu0:25.67/23.33free gpu1:39.75/9.25free gpu2:25.33/23.67free gpu3:39.75/9.25free
     idle HBM cluster-wide (at each GPU's peak): 65.5 GB

[collect] -> results/exp2/pd_layoutb_p60000_c16/imbalance.json
