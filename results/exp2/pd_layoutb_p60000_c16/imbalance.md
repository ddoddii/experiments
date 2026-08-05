
### park_local   (1852 samples, 925.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.854   0.031   0.823   0.987      1.07   70.2%  0.997
   P1     P     7.32   0.855   0.023   0.832   0.987      1.06   75.8%  0.000
   D0     D    22.64   0.061   0.061   0.000   0.128     21.27    0.0%  0.000
   D1     D    22.64   0.065   0.064   0.000   0.118     21.18    0.0%  0.000
  P mean 0.855 vs D mean 0.062   -> P/D gap +0.792
  M1 stranded headroom : 31093.0 GB*s   (42.38 GB average while saturated)
     saturated for      : 733.6s of 925.6s (79.2%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 15566.3, 'D1': 15526.6}
     vs threshold       : {'hi=0.7': 34157.5, 'hi=0.8': 33107.2, 'hi=0.85': 32216.4, 'hi=0.9': 31093.0, 'hi=0.95': 24608.2}
  M2 imbalance         : CoV 0.8424  spread mean 0.8265  p95 0.9573  Jain 0.5862
  GPU HBM peak/free    : gpu0:28.98/20.02free gpu1:40.41/8.59free gpu2:28.43/20.57free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 57.77 GB

### park_pd   (1846 samples, 922.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.857   0.029   0.828   0.990      1.05   77.7%  0.000
   P1     P     7.32   0.853   0.028   0.825   0.988      1.08   72.3%  0.998
   D0     D    20.15   0.074   0.073   0.001   0.139     18.67    0.0%  0.000
   D1     D    20.15   0.070   0.070   0.000   0.132     18.75    0.0%  0.000
  P mean 0.855 vs D mean 0.072   -> P/D gap +0.783
  M1 stranded headroom : 28312.1 GB*s   (37.39 GB average while saturated)
     saturated for      : 757.1s of 922.6s (82.1%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 14134.7, 'D1': 14177.4}
     vs threshold       : {'hi=0.7': 29877.9, 'hi=0.8': 29527.8, 'hi=0.85': 29235.1, 'hi=0.9': 28312.1, 'hi=0.95': 24004.2}
  M2 imbalance         : CoV 0.8289  spread mean 0.8249  p95 0.9568  Jain 0.5941
  GPU HBM peak/free    : gpu0:26.62/22.38free gpu1:40.41/8.59free gpu2:26.28/22.72free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 62.28 GB

### park_pd_blind   (1883 samples, 941.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.857   0.026   0.831   0.993      1.05   76.0%  0.997
   P1     P     7.32   0.869   0.029   0.840   0.996      0.96   74.6%  0.997
   D0     D    20.15   0.067   0.067   0.000   0.126      18.8    0.0%  0.000
   D1     D    20.15   0.075   0.074   0.000   0.136     18.65    0.0%  0.000
  P mean 0.863 vs D mean 0.071   -> P/D gap +0.792
  M1 stranded headroom : 28378.7 GB*s   (37.41 GB average while saturated)
     saturated for      : 758.6s of 941.1s (80.6%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 14268.9, 'D1': 14109.8}
     vs threshold       : {'hi=0.7': 30988.0, 'hi=0.8': 30518.7, 'hi=0.85': 29529.9, 'hi=0.9': 28378.7, 'hi=0.95': 23402.6}
  M2 imbalance         : CoV 0.8275  spread mean 0.8304  p95 0.9641  Jain 0.595
  GPU HBM peak/free    : gpu0:27.02/21.98free gpu1:40.39/8.61free gpu2:26.56/22.44free gpu3:40.39/8.61free
     idle HBM cluster-wide (at each GPU's peak): 61.64 GB

### park_slowlink   (1848 samples, 923.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.845   0.030   0.815   0.991      1.13   66.5%  0.997
   P1     P     7.32   0.849   0.032   0.817   0.988      1.11   73.0%  0.997
   D0     D    22.64   0.061   0.061   0.000   0.130     21.26    0.0%  0.000
   D1     D    22.64   0.062   0.062   0.000   0.115     21.24    0.0%  0.000
  P mean 0.847 vs D mean 0.061   -> P/D gap +0.785
  M1 stranded headroom : 30023.5 GB*s   (42.37 GB average while saturated)
     saturated for      : 708.6s of 923.6s (76.7%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 15028.9, 'D1': 14994.6}
     vs threshold       : {'hi=0.7': 33646.6, 'hi=0.8': 32883.4, 'hi=0.85': 31572.1, 'hi=0.9': 30023.5, 'hi=0.95': 22796.8}
  M2 imbalance         : CoV 0.8431  spread mean 0.8198  p95 0.9572  Jain 0.5858
  GPU HBM peak/free    : gpu0:29.16/19.84free gpu1:40.41/8.59free gpu2:28.75/20.25free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 57.27 GB

[collect] -> results/exp2/pd_layoutb_p60000_c16/imbalance.json
