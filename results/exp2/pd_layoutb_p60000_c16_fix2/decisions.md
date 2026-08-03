
### park_pd   (744 parks)
  selection accuracy (idlest fast-link candidate) : 95.4%
  usage of chosen GPU  : 0.083
  usage of rejected    : 0.4532
  USAGE GAP            : -0.3704   (negative = placed toward idle)
  cross-GPU parks      : 96.6%
  target share         : {'0': 0.0175, '1': 0.4583, '2': 0.0161, '3': 0.5081}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3445
     round_robin    0.3293
     always_local   0.8365
     policy         0.0828  <- actual

[collect] -> results/exp2/pd_layoutb_p60000_c16_fix2/decisions.json
