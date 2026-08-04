
### park_local   (1382 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.766
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.6462, '2': 0.3538}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.7657
     round_robin    0.7657
     always_local   0.7657
     policy         0.7657  <- actual

### park_pd   (1382 parks)
  selection accuracy (idlest fast-link candidate) : 96.4%
  usage of chosen GPU  : 0.136
  usage of rejected    : 0.4513
  USAGE GAP            : -0.3149   (negative = placed toward idle)
  cross-GPU parks      : 96.0%
  target share         : {'0': 0.0174, '1': 0.6093, '2': 0.0232, '3': 0.3502}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3536
     round_robin    0.3465
     always_local   0.7652
     policy         0.1364  <- actual

[collect] -> results/exp2/repeat_nocap_c32_r2/decisions.json
