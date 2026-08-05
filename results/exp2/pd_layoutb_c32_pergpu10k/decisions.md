
### park_local   (1450 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.880
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.5379, '2': 0.4621}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.88
     round_robin    0.88
     always_local   0.88
     policy         0.88  <- actual

### park_pd   (1447 parks)
  selection accuracy (idlest fast-link candidate) : 96.1%
  usage of chosen GPU  : 0.143
  usage of rejected    : 0.5247
  USAGE GAP            : -0.3822   (negative = placed toward idle)
  cross-GPU parks      : 97.7%
  target share         : {'0': 0.0076, '1': 0.7851, '2': 0.0159, '3': 0.1914}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.4027
     round_robin    0.3978
     always_local   0.8791
     policy         0.1426  <- actual

[collect] -> results/exp2/pd_layoutb_c32_pergpu10k/decisions.json
