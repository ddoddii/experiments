
### park_pd   (744 parks)
  selection accuracy (idlest fast-link candidate) : 97.0%
  usage of chosen GPU  : 0.081
  usage of rejected    : 0.453
  USAGE GAP            : -0.3722   (negative = placed toward idle)
  cross-GPU parks      : 96.5%
  target share         : {'0': 0.0121, '1': 0.3522, '2': 0.0228, '3': 0.6129}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3441
     round_robin    0.3297
     always_local   0.8335
     policy         0.0809  <- actual

[collect] -> results/exp2/pd_layoutb_p60000_c16_fix/decisions.json
