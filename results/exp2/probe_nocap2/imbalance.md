
### park_local   (1906 samples, 952.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.817   0.011   0.806   0.998      4.57   69.5%  0.993
   P1     P     25.0   0.816   0.008   0.808   0.998       4.6   69.4%  0.996
   D0     D    22.64   0.142   0.142   0.000   0.232     19.43    0.0%  0.000
   D1     D    22.64   0.133   0.133   0.000   0.216     19.63    0.0%  0.000
  P mean 0.817 vs D mean 0.137   -> P/D gap +0.679
  M1 stranded headroom : 25769.1 GB*s   (38.95 GB average while saturated)
     saturated for      : 661.6s of 952.6s (69.5%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 12846.9, 'D1': 12922.2}
     vs threshold       : {'hi=0.7': 28272.5, 'hi=0.8': 26862.5, 'hi=0.85': 26432.1, 'hi=0.9': 25769.1, 'hi=0.95': 25174.3}
  M2 imbalance         : CoV 0.6801  spread mean 0.7169  p95 0.9506  Jain 0.6837
  GPU HBM peak/free    : gpu0:44.72/4.28free gpu1:40.41/8.59free gpu2:44.18/4.82free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 26.28 GB

[collect] -> results/exp2/probe_nocap2/imbalance.json
