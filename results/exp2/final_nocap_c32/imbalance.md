
### park_local   (1938 samples, 968.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.828   0.013   0.815   0.999      4.31   71.0%  0.000
   P1     P     25.0   0.816   0.012   0.804   0.998      4.59   68.7%  0.992
   D0     D    22.64   0.139   0.139   0.000   0.215      19.5    0.0%  0.000
   D1     D    22.64   0.139   0.139   0.000   0.222      19.5    0.0%  0.000
  P mean 0.822 vs D mean 0.139   -> P/D gap +0.683
  M1 stranded headroom : 26776.1 GB*s   (38.94 GB average while saturated)
     saturated for      : 687.6s of 968.6s (71.0%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 13431.6, 'D1': 13344.5}
     vs threshold       : {'hi=0.7': 28755.3, 'hi=0.8': 27813.3, 'hi=0.85': 27217.2, 'hi=0.9': 26776.1, 'hi=0.95': 26278.9}
  M2 imbalance         : CoV 0.6832  spread mean 0.7222  p95 0.9509  Jain 0.6804
  GPU HBM peak/free    : gpu0:44.57/4.43free gpu1:40.41/8.59free gpu2:44.41/4.59free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 26.2 GB

### park_pd   (1915 samples, 957.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.818   0.009   0.809   0.998      4.56   69.5%  0.991
   P1     P     25.0   0.814   0.011   0.803   0.998      4.66   67.8%  0.995
   D0     D    20.15   0.150   0.150   0.000   0.255     17.13    0.0%  0.000
   D1     D    20.15   0.163   0.163   0.000   0.263     16.87    0.0%  0.000
  P mean 0.816 vs D mean 0.157   -> P/D gap +0.659
  M1 stranded headroom : 22497.5 GB*s   (33.83 GB average while saturated)
     saturated for      : 665.1s of 957.1s (69.5%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 11294.9, 'D1': 11202.6}
     vs threshold       : {'hi=0.7': 24516.7, 'hi=0.8': 23293.7, 'hi=0.85': 23064.1, 'hi=0.9': 22497.5, 'hi=0.95': 22053.4}
  M2 imbalance         : CoV 0.6464  spread mean 0.6996  p95 0.952  Jain 0.7038
  GPU HBM peak/free    : gpu0:45.06/3.94free gpu1:40.42/8.58free gpu2:44.51/4.49free gpu3:40.42/8.58free
     idle HBM cluster-wide (at each GPU's peak): 25.59 GB

[collect] -> results/exp2/final_nocap_c32/imbalance.json
