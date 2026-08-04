
### park_local   (1450 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.785
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.4676, '2': 0.5324}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.7847
     round_robin    0.7847
     always_local   0.7847
     policy         0.7847  <- actual

### park_pd   (1448 parks)
  selection accuracy (idlest fast-link candidate) : 97.0%
  usage of chosen GPU  : 0.167
  usage of rejected    : 0.4655
  USAGE GAP            : -0.2985   (negative = placed toward idle)
  cross-GPU parks      : 91.2%
  target share         : {'0': 0.0753, '1': 0.6354, '2': 0.0131, '3': 0.2762}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3702
     round_robin    0.3662
     always_local   0.7822
     policy         0.1669  <- actual

[collect] -> results/exp2/repeat_nocap_c32_r1/decisions.json
