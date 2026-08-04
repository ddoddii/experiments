
### park_local   (2049 samples, 1024.2s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.015   0.003   0.012   0.034     24.62    0.0%  0.989
   P1     P     25.0   0.858   0.032   0.825   0.994      3.56   77.3%  0.991
   D0     D    13.21   0.246   0.246   0.000   0.436      9.96    0.0%  0.000
   D1     D    13.21   0.227   0.227   0.000   0.380     10.21    0.0%  0.000
  P mean 0.436 vs D mean 0.237   -> P/D gap +0.200
  M1 stranded headroom : 15795.2 GB*s   (19.97 GB average while saturated)
     saturated for      : 791.1s of 1024.2s (77.2%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 7759.4, 'D1': 8035.8}
     vs threshold       : {'hi=0.7': 16842.0, 'hi=0.8': 16392.1, 'hi=0.85': 16055.3, 'hi=0.9': 15795.2, 'hi=0.95': 15235.0}
  M2 imbalance         : CoV 0.6677  spread mean 0.6817  p95 0.9049  Jain 0.6918
  GPU HBM peak/free    : gpu0:47.4/1.6free gpu1:30.91/18.09free gpu2:47.02/1.98free gpu3:30.91/18.09free
     idle HBM cluster-wide (at each GPU's peak): 39.76 GB

### park_pd   (2421 samples, 1210.2s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.041   0.004   0.037   0.129     23.97    0.0%  0.985
   P1     P     25.0   0.858   0.068   0.790   0.994      3.55   76.5%  0.992
   D0     D     6.85   0.439   0.439   0.000   0.664      3.84    0.0%  0.000
   D1     D     6.85   0.452   0.452   0.000   0.708      3.75    0.0%  0.000
  P mean 0.450 vs D mean 0.445   -> P/D gap +0.004
  M1 stranded headroom : 4920.5 GB*s   (5.32 GB average while saturated)
     saturated for      : 925.1s of 1210.2s (76.4%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 2429.5, 'D1': 2491.0}
     vs threshold       : {'hi=0.7': 5393.0, 'hi=0.8': 5099.3, 'hi=0.85': 5008.5, 'hi=0.9': 4920.5, 'hi=0.95': 4743.3}
  M2 imbalance         : CoV 0.4171  spread mean 0.5337  p95 0.8469  Jain 0.8467
  GPU HBM peak/free    : gpu0:47.4/1.6free gpu1:34.54/14.46free gpu2:47.03/1.97free gpu3:34.54/14.46free
     idle HBM cluster-wide (at each GPU's peak): 32.49 GB

[collect] -> results/exp2/big_c32_r1/imbalance.json
