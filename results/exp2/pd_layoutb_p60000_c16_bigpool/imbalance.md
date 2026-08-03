
### park_pd   (1864 samples, 931.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     7.32   0.866   0.035   0.832   0.989      0.98   72.7%  0.997
   P1     P     7.32   0.834   0.031   0.803   0.989      1.21   64.6%  0.000
   D0     D    16.15   0.086   0.086   0.000   0.179     14.76    0.0%  0.000
   D1     D    16.15   0.090   0.090   0.000   0.160      14.7    0.0%  0.000
  P mean 0.850 vs D mean 0.088   -> P/D gap +0.762
  M1 stranded headroom : 21273.7 GB*s   (29.34 GB average while saturated)
     saturated for      : 725.1s of 931.6s (77.8%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 10683.2, 'D1': 10590.5}
     vs threshold       : {'hi=0.7': 24511.9, 'hi=0.8': 23932.8, 'hi=0.85': 23799.2, 'hi=0.9': 21273.7, 'hi=0.95': 14661.1}
  M2 imbalance         : CoV 0.792  spread mean 0.8161  p95 0.9565  Jain 0.6153
  GPU HBM peak/free    : gpu0:28.95/20.05free gpu1:41.42/7.58free gpu2:28.39/20.61free gpu3:41.42/7.58free
     idle HBM cluster-wide (at each GPU's peak): 55.82 GB

[collect] -> results/exp2/pd_layoutb_p60000_c16_bigpool/imbalance.json
