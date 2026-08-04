
### park_local   (1449 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.797
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.3368, '2': 0.6632}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.7968
     round_robin    0.7968
     always_local   0.7968
     policy         0.7968  <- actual

### park_pd   (1450 parks)
  selection accuracy (idlest fast-link candidate) : 97.2%
  usage of chosen GPU  : 0.164
  usage of rejected    : 0.472
  USAGE GAP            : -0.3084   (negative = placed toward idle)
  cross-GPU parks      : 93.5%
  target share         : {'0': 0.0255, '1': 0.4545, '2': 0.04, '3': 0.48}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.3746
     round_robin    0.3686
     always_local   0.7858
     policy         0.1636  <- actual

[collect] -> results/exp2/final_nocap_c32/decisions.json
