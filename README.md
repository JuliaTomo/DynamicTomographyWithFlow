# Welcome to XfromProjections (under development)

XfromProjections aims to provide solutions X from tomographic projection data. X can be images, edges (or ). Instead of providing a solver as a blackbox, we sometimes divide the function into multiple steps. For example, for filtered back-projection, the user should use filtering and back projection seperately. 

XfromProjectiions depends on [TomoForward](https://github.com/JuliaTomo/TomoForward.jl) package for forward operators of images.

## Install

Install [Julia](https://julialang.org/downloads/) and in [Julia REPL](https://docs.julialang.org/en/v1/stdlib/REPL/),

```
julia> ]
pkg> add https://github.com/JuliaTomo/TomoForward.jl
pkg> add https://github.com/JuliaTomo/XfromProjections.jl
```

## Examples

Please see codes in test folder.


# Features

## Image reconstruction from Projections

### Analytic methods

- FBP with different filters of Ram-Lak, Henning, Hann, Kaiser

### Iterative methods

- SIRT [Andersen, Kak 1984]
- TV using primal dual [Chambolle, Pock 2011]
- TNV [Duran et al, 2016] using primal dual

## Edges from projections

- Laplacian of Gaussian from projections [Srinivasa et al. 1992]

## Shape form Projections

- (Todo) Parametric level set (Todo) []

# Todos

- 3D geometry
- Supporting GPU
- Forward projection of one closed mesh

# Reference

- Andersen, A.H., Kak, A.C., 1984. Simultaneous Algebraic Reconstruction Technique (SART): A superior implementation of the ART algorithm. Ultrasonic Imaging 6. https://doi.org/10.1016/0161-7346(84)90008-7
- Chambolle, A., Pock, T., 2011. A First-Order Primal-Dual Algorithm for Convex Problems with Applications to Imaging. Journal of Mathematical Imaging and Vision 40, 120–145. https://doi.org/10.1007/s10851-010-0251-1
- Srinivasa, N., Ramakrishnan, K.R., Rajgopal, K., 1992. Detection of edges from projections. IEEE Transactions on Medical Imaging 11, 76–80. https://doi.org/10.1109/42.126913
- Duran, J., Moeller, M., Sbert, C., Cremers, D., 2016. Collaborative Total Variation: A General Framework for Vectorial TV Models. SIAM Journal on Imaging Sciences 9, 116–151. https://doi.org/10.1137/15M102873X
