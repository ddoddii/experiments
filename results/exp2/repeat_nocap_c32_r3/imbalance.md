
### park_local   (1850 samples, 924.6s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.798   0.009   0.788   0.999      5.06   64.9%  0.998
   P1     P     25.0   0.798   0.008   0.790   0.997      5.05   66.6%  0.000
   D0     D    22.64   0.119   0.119   0.000   0.196     19.95    0.0%  0.000
   D1     D    22.64   0.134   0.133   0.001   0.229     19.62    0.0%  0.000
  P mean 0.798 vs D mean 0.126   -> P/D gap +0.672
  M1 stranded headroom : 24449.8 GB*s   (39.69 GB average while saturated)
     saturated for      : 616.1s of 924.6s (66.6%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 12378.5, 'D1': 12071.3}
     vs threshold       : {'hi=0.7': 27035.6, 'hi=0.8': 25802.5, 'hi=0.85': 25318.6, 'hi=0.9': 24449.8, 'hi=0.95': 23660.5}
  M2 imbalance         : CoV 0.6793  spread mean 0.7079  p95 0.9599  Jain 0.6836
  GPU HBM peak/free    : gpu0:44.95/4.05free gpu1:40.41/8.59free gpu2:44.42/4.58free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 25.81 GB

### park_pd   (1867 samples, 933.1s)
  metric: cache_occupancy
  gpu  role  cap(GB)   occup  active  cached     p95  free(GB)    >90%    hit
   P0     P     25.0   0.790   0.010   0.779   0.998      5.26   64.7%  0.000
   P1     P     25.0   0.797   0.009   0.788   0.998      5.08   65.3%  0.998
   D0     D    20.15   0.149   0.148   0.000   0.248     17.16    0.0%  0.000
   D1     D    20.15   0.135   0.135   0.000   0.223     17.44    0.0%  0.000
  P mean 0.793 vs D mean 0.142   -> P/D gap +0.651
  M1 stranded headroom : 21184.0 GB*s   (34.75 GB average while saturated)
     saturated for      : 609.6s of 933.1s (65.3%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 10509.1, 'D1': 10674.9}
     vs threshold       : {'hi=0.7': 23557.1, 'hi=0.8': 22594.1, 'hi=0.85': 22085.0, 'hi=0.9': 21184.0, 'hi=0.95': 20546.0}
  M2 imbalance         : CoV 0.6531  spread mean 0.689  p95 0.9545  Jain 0.6996
  GPU HBM peak/free    : gpu0:44.78/4.22free gpu1:40.41/8.59free gpu2:44.24/4.76free gpu3:40.41/8.59free
     idle HBM cluster-wide (at each GPU's peak): 26.16 GB

[collect] -> results/exp2/repeat_nocap_c32_r3/imbalance.json
