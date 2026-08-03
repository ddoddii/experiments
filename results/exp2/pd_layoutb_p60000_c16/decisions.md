
### park_local   (744 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.836
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.5054, '2': 0.4946}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.836
     round_robin    0.836
     always_local   0.836
     policy         0.836  <- actual

### park_pd   (744 parks)
  selection accuracy (idlest fast-link candidate) : 97.7%
  usage of chosen GPU  : 0.063
  usage of rejected    : 0.466
  USAGE GAP            : -0.4031   (negative = placed toward idle)
  cross-GPU parks      : 98.5%
  target share         : {'0': 0.0134, '1': 0.7083, '2': 0.0013, '3': 0.2769}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3481
     round_robin    0.3313
     always_local   0.8415
     policy         0.0629  <- actual

### park_pd_blind   (744 parks)
  selection accuracy (idlest fast-link candidate) : 1.5%
  usage of chosen GPU  : 0.876
  usage of rejected    : 0.0774
  USAGE GAP            : 0.799   (negative = placed toward idle)
  cross-GPU parks      : 1.6%
  target share         : {'2': 0.9839, '3': 0.0161}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3616
     round_robin    0.3438
     always_local   0.8766
     policy         0.8764  <- actual

### park_slowlink   (744 parks)
  selection accuracy (idlest fast-link candidate) : 97.0%
  usage of chosen GPU  : 0.826
  usage of rejected    : 0.855
  USAGE GAP            : -0.0293   (negative = placed toward idle)
  cross-GPU parks      : 3.2%
  target share         : {'0': 0.504, '2': 0.496}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.8402
     round_robin    0.8403
     always_local   0.8257
     policy         0.8257  <- actual

[collect] -> results/exp2/pd_layoutb_p60000_c16/decisions.json
