
### park_local   (1944 samples, 971.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.812   0.012   0.801   0.997      4.69   69.7%  0.992
   P1     P     25.0   0.820   0.010   0.810   0.997      4.49   69.5%  0.000
   D0     D    26.41   0.109   0.109   0.000   0.175     23.54    0.0%  0.000
   D1     D    26.41   0.128   0.128   0.000   0.197     23.03    0.0%  0.000
  P mean 0.816 vs D mean 0.118   -> P/D gap +0.698
  M1 stranded headroom : 31456.2 GB*s   (46.49 GB average while saturated)
     saturated for      : 676.6s of 971.6s (69.6%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 15995.6, 'D1': 15460.6}
     vs threshold       : {'hi=0.7': 34560.7, 'hi=0.8': 32892.9, 'hi=0.85': 32036.4, 'hi=0.9': 31456.2, 'hi=0.95': 30903.9}
  M2 imbalance         : CoV 0.7055  spread mean 0.7296  p95 0.9578  Jain 0.6676
  GPU HBM peak/free    : gpu0:45.37/3.63free gpu1:44.16/4.84free gpu2:44.53/4.47free gpu3:44.16/4.84free
     idle HBM cluster-wide (at each GPU's peak): 17.78 GB

### park_pd   (1934 samples, 966.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.811   0.012   0.800   0.998      4.71   68.6%  0.000
   P1     P     25.0   0.808   0.012   0.796   0.996      4.79   67.7%  0.995
   D0     D    18.84   0.154   0.154   0.000   0.249     15.94    0.0%  0.000
   D1     D    18.84   0.184   0.184   0.000   0.269     15.36    0.0%  0.000
  P mean 0.810 vs D mean 0.169   -> P/D gap +0.641
  M1 stranded headroom : 20672.4 GB*s   (31.2 GB average while saturated)
     saturated for      : 662.6s of 966.6s (68.5%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 10412.8, 'D1': 10259.6}
     vs threshold       : {'hi=0.7': 22868.2, 'hi=0.8': 21760.0, 'hi=0.85': 21124.9, 'hi=0.9': 20672.4, 'hi=0.95': 20311.3}
  M2 imbalance         : CoV 0.6248  spread mean 0.6863  p95 0.9133  Jain 0.7163
  GPU HBM peak/free    : gpu0:44.8/4.2free gpu1:44.54/4.46free gpu2:44.2/4.8free gpu3:44.54/4.46free
     idle HBM cluster-wide (at each GPU's peak): 17.92 GB

[collect] -> results/exp2/big_c32_r1/imbalance.json
