
### park_local   (1931 samples, 965.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.813   0.011   0.803   0.998      4.67   68.8%  0.000
   P1     P     25.0   0.815   0.009   0.805   0.998      4.64   69.0%  0.000
   D0     D    22.64   0.128   0.128   0.000   0.210     19.73    0.0%  0.000
   D1     D    22.64   0.144   0.144   0.000   0.231     19.37    0.0%  0.000
  P mean 0.814 vs D mean 0.137   -> P/D gap +0.677
  M1 stranded headroom : 25966.8 GB*s   (38.98 GB average while saturated)
     saturated for      : 666.1s of 965.1s (69.0%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 13115.2, 'D1': 12851.6}
     vs threshold       : {'hi=0.7': 28503.6, 'hi=0.8': 27236.2, 'hi=0.85': 26716.8, 'hi=0.9': 25966.8, 'hi=0.95': 25210.6}
  M2 imbalance         : CoV 0.6743  spread mean 0.7113  p95 0.9512  Jain 0.6865
  GPU HBM peak/free    : gpu0:45.01/3.99free gpu1:40.41/8.59free gpu2:44.54/4.46free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 25.63 GB

### park_pd   (1896 samples, 947.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.800   0.010   0.790   0.997      5.01   66.0%  0.991
   P1     P     25.0   0.825   0.011   0.814   0.999      4.37   70.0%  0.995
   D0     D    20.15   0.161   0.161   0.000   0.245     16.91    0.0%  0.000
   D1     D    20.15   0.148   0.148   0.000   0.237     17.16    0.0%  0.000
  P mean 0.812 vs D mean 0.155   -> P/D gap +0.658
  M1 stranded headroom : 22525.2 GB*s   (33.97 GB average while saturated)
     saturated for      : 663.1s of 947.6s (70.0%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 11260.3, 'D1': 11264.8}
     vs threshold       : {'hi=0.7': 25057.3, 'hi=0.8': 23509.4, 'hi=0.85': 23139.0, 'hi=0.9': 22525.2, 'hi=0.95': 22050.9}
  M2 imbalance         : CoV 0.6608  spread mean 0.7096  p95 0.9332  Jain 0.6957
  GPU HBM peak/free    : gpu0:44.94/4.06free gpu1:40.42/8.58free gpu2:44.09/4.91free gpu3:40.42/8.58free
     idle HBM cluster-wide (at each GPU's peak): 26.13 GB

[collect] -> results/exp2/repeat_nocap_c32_r1/imbalance.json
