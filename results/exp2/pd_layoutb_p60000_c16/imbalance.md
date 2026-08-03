
### hicache   (1847 samples, 923.1s)
  gpu  role  cap(GB)  use_mean  use_p95  free(GB)    >90%
   P0     P     7.32     0.019    0.071      7.18    0.0%
   P1     P     7.32     0.019    0.067      7.19    0.0%
   D0     D    22.64     0.060    0.127     21.28    0.0%
   D1     D    22.64     0.064    0.108      21.2    0.0%
  P mean 0.019 vs D mean 0.062   -> P/D gap -0.043
  M1 stranded headroom : 0.0 GB*s   (0.0 GB average while saturated)
     saturated for      : 0.0s of 923.1s (0.0%)
     by gpu (GB*s)      : {'P0': 0.0, 'P1': 0.0, 'D0': 0.0, 'D1': 0.0}
     vs threshold       : {'hi=0.7': 0.0, 'hi=0.8': 0.0, 'hi=0.85': 0.0, 'hi=0.9': 0.0, 'hi=0.95': 0.0}
  M2 imbalance         : CoV 0.8011  spread mean 0.076  p95 0.1229  Jain 0.6237

[collect] -> results/exp2/pd_layoutb_p60000_c16/imbalance.json
