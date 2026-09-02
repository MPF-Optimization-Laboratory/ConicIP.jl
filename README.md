# ConicIP.jl

[![CI](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MPF-Optimization-Laboratory/ConicIP.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MPF-Optimization-Laboratory/ConicIP.jl)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/dev/)
[![Version](https://juliahub.com/docs/General/ConicIP/stable/version.svg)](https://juliahub.com/ui/Packages/General/ConicIP)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/blob/master/LICENSE.md)

[ConicIP.jl](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl) (Conic **I**nterior **P**oint) is a pure-Julia interior-point solver for quadratic programs with linear equality constraints and polyhedral, second-order cone, and (experimental) semidefinite cone constraints. It solves

```
minimize    ½yᵀQy - cᵀy
subject to  Ay ≥_K b,   K = K₁ × ⋯ × Kⱼ
            Gy  = d
```

where `Q ⪰ 0` and each `Kᵢ` is a nonnegative orthant, a second-order cone, or a cone of positive semidefinite matrices. Because ConicIP is written in Julia, it accepts abstract matrix types and custom KKT solver callbacks for exploiting problem structure.

## Features

- **Pure Julia.** No binary dependencies or external solver installations.
- **Cones.** Cartesian products of the cones below, in any order.

  | Cone | Spec | Description |
  |------|------|-------------|
  | Nonnegative orthant | `("R", n)` | Linear inequalities |
  | Second-order cone | `("Q", n)` | Norm constraints |
  | Semidefinite (experimental) | `("S", k)` | Matrix positivity |

- **Quadratic objectives.** Handled natively by the direct API, without reformulation to a second-order cone. Through JuMP, quadratic objectives are supported via MathOptInterface bridges.
- **Custom KKT solvers.** Plug in your own factorization or iterative method at each interior-point iteration; built-in dense, sparse, and reduced 2×2 solvers with automatic selection.
- **Nesterov-Todd scaling.** Symmetric primal-dual scaling for good numerical behaviour.
- **Infeasibility detection.** Returns validated certificates for infeasible and unbounded problems.
- **JuMP support.** A [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) wrapper makes ConicIP a drop-in solver for [JuMP](https://github.com/jump-dev/JuMP.jl).

## Installation

ConicIP requires Julia 1.10 or later. Install it from the General registry:

```julia
import Pkg
Pkg.add("ConicIP")
```

## Quick start

### With JuMP

Minimize the Euclidean norm of a vector whose entries sum to one:

```julia
using JuMP, ConicIP

model = Model(ConicIP.Optimizer)
set_silent(model)
@variable(model, x[1:3])
@variable(model, t)
@constraint(model, sum(x) == 1)
@constraint(model, [t; x] in SecondOrderCone())
@objective(model, Min, t)
optimize!(model)

termination_status(model)   # OPTIMAL
value.(x)                   # ≈ [0.3333, 0.3333, 0.3333]
```

See the [JuMP integration guide](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/guides/jump/) for solver options and supported constraint types.

### Direct API

The direct interface is

```julia
sol = conicIP(Q, c, A, b, K, G, d)
```

where `K` is a vector of `(cone, dimension)` tuples. For example, the cone `K = R² × Q³ × R²` is written

```julia
K = [("R", 2), ("Q", 3), ("R", 2)]
```

To solve the bound-constrained QP `minimize ½yᵀQy - cᵀy subject to y ≥ 0`:

```julia
using ConicIP
using SparseArrays, LinearAlgebra

n = 1000
Q = sprandn(n, n, 0.1)
Q = Q'*Q
c = ones(n)
A = sparse(1.0I, n, n)
b = zeros(n)
K = [("R", n)]

sol = conicIP(Q, c, A, b, K, verbose=true);
```

`sol` holds the status (`sol.status`), primal variables (`sol.y`), dual variables (`sol.v`, `sol.w`), objective values (`sol.pobj`, `sol.dobj`), and convergence residuals (`sol.prFeas`, `sol.duFeas`, `sol.muFeas`).

## Documentation

The [stable documentation](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/) (or the [development version](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/dev/)) covers

- [Tutorials](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/tutorials/generated/getting_started/): linear, quadratic, second-order cone, and semidefinite programs; reading the iteration log; detecting infeasibility.
- How-to guides: [JuMP integration](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/guides/jump/), [KKT solvers](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/guides/kkt_solvers/), and [preprocessing](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/guides/preprocessing/).
- [Mathematical background](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/background/) on the infeasible-start Mehrotra predictor-corrector method with Nesterov-Todd scaling.
- [API reference](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/api/).

## Citing

If you use ConicIP in your research, please cite it as

```bibtex
@software{ConicIP.jl,
  author  = {Friedlander, Michael P. and Goh, Gabriel},
  title   = {{ConicIP.jl}: A conic interior-point solver in {Julia}},
  year    = {2026},
  version = {0.3.2},
  url     = {https://github.com/MPF-Optimization-Laboratory/ConicIP.jl}
}
```

The repository also ships a [CITATION.cff](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/blob/master/CITATION.cff) file that GitHub renders under "Cite this repository".

## Contributing

Bug reports, fixes, and documentation improvements are welcome. See [CONTRIBUTING.md](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/blob/master/CONTRIBUTING.md) for the workflow and the policy on disclosing AI-assisted contributions, and [CHANGELOG.md](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/blob/master/CHANGELOG.md) for release notes.

## Getting help

If you need help, please ask a question by [opening a GitHub issue](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/issues/new/choose). Questions about modelling with JuMP are also welcome on the [JuMP community forum](https://jump.dev/forum).

## Affiliation

ConicIP.jl is maintained by the
[MPF Optimization Laboratory](https://github.com/MPF-Optimization-Laboratory)
at the University of British Columbia.

## History

ConicIP.jl was originally written by [Gabriel Goh](https://github.com/gabgoh)
in 2016 as a PhD project under the supervision of Michael P. Friedlander at the
University of California, Davis. Development paused after Gabriel graduated.
Michael Friedlander revived the package in 2026, modernized it for current
Julia and JuMP, and registered it in the General registry. Thanks also to
Tony Kelman and Miles Lubin for early contributions.

## License

ConicIP.jl is licensed under the [MIT License](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/blob/master/LICENSE.md).
