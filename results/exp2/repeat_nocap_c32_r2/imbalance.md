
### park_local   (1883 samples, 941.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.810   0.009   0.801   0.998      4.75   66.6%  0.000
   P1     P     25.0   0.821   0.009   0.812   0.998      4.47   69.7%  0.998
   D0     D    22.64   0.102   0.101   0.001   0.175     20.33    0.0%  0.000
   D1     D    22.64   0.120   0.120   0.000   0.198     19.91    0.0%  0.000
  P mean 0.816 vs D mean 0.111   -> P/D gap +0.704
  M1 stranded headroom : 26497.5 GB*s   (40.39 GB average while saturated)
     saturated for      : 656.1s of 941.1s (69.7%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 13397.2, 'D1': 13100.3}
     vs threshold       : {'hi=0.7': 29098.8, 'hi=0.8': 27641.6, 'hi=0.85': 27085.9, 'hi=0.9': 26497.5, 'hi=0.95': 25733.6}
  M2 imbalance         : CoV 0.7177  spread mean 0.736  p95 0.959  Jain 0.6609
  GPU HBM peak/free    : gpu0:44.87/4.13free gpu1:40.41/8.59free gpu2:44.08/4.92free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 26.23 GB

### park_pd   (1905 samples, 952.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.813   0.009   0.804   0.998      4.67   68.0%  0.998
   P1     P     25.0   0.822   0.011   0.811   0.998      4.45   68.9%  0.000
   D0     D    20.15   0.120   0.119   0.001   0.204     17.74    0.0%  0.000
   D1     D    20.15   0.128   0.128   0.000   0.225     17.57    0.0%  0.000
  P mean 0.818 vs D mean 0.124   -> P/D gap +0.694
  M1 stranded headroom : 23249.7 GB*s   (35.44 GB average while saturated)
     saturated for      : 656.1s of 952.1s (68.9%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 11660.1, 'D1': 11589.5}
     vs threshold       : {'hi=0.7': 25971.3, 'hi=0.8': 24519.4, 'hi=0.85': 24014.9, 'hi=0.9': 23249.7, 'hi=0.95': 22773.9}
  M2 imbalance         : CoV 0.6997  spread mean 0.7293  p95 0.963  Jain 0.6724
  GPU HBM peak/free    : gpu0:44.81/4.19free gpu1:40.41/8.59free gpu2:44.35/4.65free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 26.02 GB

[collect] -> results/exp2/repeat_nocap_c32_r2/imbalance.json
