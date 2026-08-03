
### park_pd   (1808 samples, 903.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.856   0.026   0.830   0.993      1.05   74.2%  0.000
   P1     P     7.32   0.850   0.030   0.820   0.988       1.1   72.4%  0.000
   D0     D    20.15   0.069   0.069   0.000   0.124     18.77    0.0%  0.000
   D1     D    20.15   0.076   0.076   0.000   0.144     18.61    0.0%  0.000
  P mean 0.853 vs D mean 0.073   -> P/D gap +0.780
  M1 stranded headroom : 26503.0 GB*s   (37.3 GB average while saturated)
     saturated for      : 710.6s of 903.6s (78.6%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 13285.7, 'D1': 13217.4}
     vs threshold       : {'hi=0.7': 29075.1, 'hi=0.8': 28249.3, 'hi=0.85': 28101.6, 'hi=0.9': 26503.0, 'hi=0.95': 20714.1}
  M2 imbalance         : CoV 0.8187  spread mean 0.8176  p95 0.9622  Jain 0.6
  GPU HBM peak/free    : gpu0:26.89/22.11free gpu1:40.41/8.59free gpu2:26.39/22.61free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 61.9 GB

[collect] -> results/exp2/pd_layoutb_p60000_c16_fix/imbalance.json
