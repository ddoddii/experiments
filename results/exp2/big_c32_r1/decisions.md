
### park_local   (1436 parks)
  selection accuracy (idlest fast-link candidate) : 100.0%
  usage of chosen GPU  : 0.843
  usage of rejected    : None
  USAGE GAP            : None   (negative = placed toward idle)
  cross-GPU parks      : 0.0%
  target share         : {'0': 0.0167, '2': 0.9833}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.8426
     round_robin    0.8426
     always_local   0.8426
     policy         0.8426  <- actual

### park_pd   (1443 parks)
  selection accuracy (idlest fast-link candidate) : 95.9%
  usage of chosen GPU  : 0.362
  usage of rejected    : 0.6918
  USAGE GAP            : -0.3298   (negative = placed toward idle)
  cross-GPU parks      : 90.5%
  target share         : {'0': 0.0298, '1': 0.4207, '2': 0.0651, '3': 0.4844}
  null models (mean usage of the GPU each would pick, lower is better):
     random         0.5861
     round_robin    0.5825
     always_local   0.8416
     policy         0.362  <- actual

[collect] -> results/exp2/big_c32_r1/decisions.json
