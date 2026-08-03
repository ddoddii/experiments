
### park_local   (1899 samples, 949.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.910   0.031   0.879   0.995      0.66   90.0%  0.989
   P1     P     7.32   0.902   0.040   0.862   0.994      0.72   85.2%  0.993
   D0     D    22.64   0.125   0.125   0.000   0.194      19.8    0.0%  0.000
   D1     D    22.64   0.146   0.146   0.000   0.241     19.33    0.0%  0.000
  P mean 0.906 vs D mean 0.136   -> P/D gap +0.770
  M1 stranded headroom : 33438.7 GB*s   (38.94 GB average while saturated)
     saturated for      : 858.6s of 949.1s (90.5%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 16961.4, 'D1': 16477.3}
     vs threshold       : {'hi=0.7': 33937.8, 'hi=0.8': 33708.0, 'hi=0.85': 33573.9, 'hi=0.9': 33438.7, 'hi=0.95': 32304.0}
  M2 imbalance         : CoV 0.7283  spread mean 0.8133  p95 0.9343  Jain 0.6537
  GPU HBM peak/free    : gpu0:27.17/21.83free gpu1:40.41/8.59free gpu2:26.64/22.36free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 61.37 GB

[collect] -> results/exp2/probe_nocap/imbalance.json
