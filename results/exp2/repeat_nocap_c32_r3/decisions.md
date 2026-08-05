
### park_local   (1376 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.768
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.5283, '2': 0.4717}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.7677
     round_robin    0.7677
     always_local   0.7677
     policy         0.7677  <- actual

### park_pd   (1375 parks)
  selection accuracy (idlest fast-link candidate) : 96.7%
  usage of chosen GPU  : 0.140
  usage of rejected    : 0.4614
  USAGE GAP            : -0.3213   (negative = placed toward idle)
  cross-GPU parks      : 94.8%
  target share         : {'0': 0.0138, '1': 0.7018, '2': 0.0385, '3': 0.2458}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3593
     round_robin    0.3544
     always_local   0.7634
     policy         0.1402  <- actual

[collect] -> results/exp2/repeat_nocap_c32_r3/decisions.json
