
### park_pd   (744 parks)
  selection accuracy (idlest fast-link candidate) : 90.9%
  usage of chosen GPU  : 0.116
  usage of rejected    : 0.4458
  USAGE GAP            : -0.33   (negative = placed toward idle)
  cross-GPU parks      : 94.0%
  target share         : {'0': 0.0309, '1': 0.3817, '2': 0.0296, '3': 0.5578}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.351
     round_robin    0.3364
     always_local   0.8196
     policy         0.1159  <- actual

[collect] -> results/exp2/pd_layoutb_p60000_c16_bigpool/decisions.json
