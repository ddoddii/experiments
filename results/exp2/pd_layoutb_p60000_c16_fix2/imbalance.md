
### park_pd   (1827 samples, 913.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.864   0.025   0.839   0.994      0.99   78.0%  0.000
   P1     P     7.32   0.857   0.027   0.830   0.992      1.05   76.0%  0.000
   D0     D    20.15   0.062   0.062   0.000   0.115      18.9    0.0%  0.000
   D1     D    20.15   0.081   0.081   0.000   0.148     18.53    0.0%  0.000
  P mean 0.861 vs D mean 0.071   -> P/D gap +0.789
  M1 stranded headroom : 28422.5 GB*s   (37.39 GB average while saturated)
     saturated for      : 760.1s of 913.1s (83.2%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 14389.1, 'D1': 14033.4}
     vs threshold       : {'hi=0.7': 29858.3, 'hi=0.8': 29282.9, 'hi=0.85': 28818.1, 'hi=0.9': 28422.5, 'hi=0.95': 22672.5}
  M2 imbalance         : CoV 0.8209  spread mean 0.8272  p95 0.9683  Jain 0.5988
  GPU HBM peak/free    : gpu0:26.88/22.12free gpu1:40.41/8.59free gpu2:26.34/22.66free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 61.96 GB

[collect] -> results/exp2/pd_layoutb_p60000_c16_fix2/imbalance.json
