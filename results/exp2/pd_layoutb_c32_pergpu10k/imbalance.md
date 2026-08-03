
### park_local   (1902 samples, 950.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.905   0.035   0.870   0.994      0.69   88.6%  0.980
   P1     P     7.32   0.904   0.037   0.868   0.996       0.7   87.2%  0.996
   D0     D    22.64   0.132   0.132   0.000   0.224     19.66    0.0%  0.000
   D1     D    22.64   0.142   0.142   0.000   0.222     19.44    0.0%  0.000
  P mean 0.905 vs D mean 0.137   -> P/D gap +0.768
  M1 stranded headroom : 33627.8 GB*s   (38.89 GB average while saturated)
     saturated for      : 864.6s of 950.6s (91.0%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 16910.1, 'D1': 16717.7}
     vs threshold       : {'hi=0.7': 34015.5, 'hi=0.8': 33764.5, 'hi=0.85': 33647.4, 'hi=0.9': 33627.8, 'hi=0.95': 29765.8}
  M2 imbalance         : CoV 0.7246  spread mean 0.8131  p95 0.9346  Jain 0.6559
  GPU HBM peak/free    : gpu0:27.42/21.58free gpu1:40.41/8.59free gpu2:26.57/22.43free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 61.19 GB

### park_pd   (1923 samples, 961.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.903   0.040   0.864   0.995      0.71   86.5%  0.000
   P1     P     7.32   0.906   0.040   0.866   0.994      0.69   84.1%  0.000
   D0     D    20.15   0.141   0.141   0.000   0.231      17.3    0.0%  0.000
   D1     D    20.15   0.164   0.164   0.000   0.257     16.84    0.0%  0.000
  P mean 0.904 vs D mean 0.153   -> P/D gap +0.751
  M1 stranded headroom : 29816.8 GB*s   (33.96 GB average while saturated)
     saturated for      : 878.1s of 961.1s (91.4%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 15109.0, 'D1': 14707.8}
     vs threshold       : {'hi=0.7': 30203.7, 'hi=0.8': 30118.4, 'hi=0.85': 30033.4, 'hi=0.9': 29816.8, 'hi=0.95': 26391.5}
  M2 imbalance         : CoV 0.6996  spread mean 0.798  p95 0.9138  Jain 0.6715
  GPU HBM peak/free    : gpu0:27.13/21.87free gpu1:40.42/8.58free gpu2:26.37/22.63free gpu3:40.42/8.58free
     idle HBM cluster-wide (at each GPU's peak): 61.66 GB

[collect] -> results/exp2/pd_layoutb_c32_pergpu10k/imbalance.json
