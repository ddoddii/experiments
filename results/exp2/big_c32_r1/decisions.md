
### park_local   (1450 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.788
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.6462, '2': 0.3538}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.7882
     round_robin    0.7882
     always_local   0.7882
     policy         0.7882  <- actual

### park_pd   (1448 parks)
  selection accuracy (idlest fast-link candidate) : 94.3%
  usage of chosen GPU  : 0.146
  usage of rejected    : 0.4904
  USAGE GAP            : -0.344   (negative = placed toward idle)
  cross-GPU parks      : 94.8%
  target share         : {'0': 0.0262, '1': 0.7659, '2': 0.0262, '3': 0.1816}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.381
     round_robin    0.3759
     always_local   0.7833
     policy         0.1465  <- actual

[collect] -> results/exp2/big_c32_r1/decisions.json
