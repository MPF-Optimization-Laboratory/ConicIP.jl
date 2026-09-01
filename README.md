# ConicIP.jl

[![CI](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/actions/workflows/CI.yml)
[![codecov](https://codecov.io/gh/MPF-Optimization-Laboratory/ConicIP.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/MPF-Optimization-Laboratory/ConicIP.jl)
[![Documentation (stable)](https://img.shields.io/badge/docs-stable-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/)
[![Documentation (dev)](https://img.shields.io/badge/docs-dev-blue.svg)](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/dev/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE.md)

`ConicIP` (Conic **I**nterior **P**oint) is a pure-Julia interior-point solver for optimizing quadratic objectives with linear equality constraints, and polyhedral, second-order cone, and (experimental) semidefinite cone constraints. Because ConicIP is written in Julia, it allows abstract input and custom KKT solver callbacks for exploiting problem structure.

## Affiliation

ConicIP.jl is maintained by the
[MPF Optimization Laboratory](https://github.com/MPF-Optimization-Laboratory)
at the University of British Columbia. It is not affiliated with JuMP-dev.

## History

ConicIP.jl was originally written by [Gabriel Goh](https://github.com/gabgoh)
in 2016 as a PhD project under the supervision of Michael P. Friedlander at the
University of California, Davis. Development paused after Gabriel graduated.
Michael Friedlander revived the package in 2026, modernized it for current
Julia and JuMP, and registered it in the General registry. Thanks also to
Tony Kelman and Miles Lubin for early contributions.

## License

ConicIP.jl is licensed under the [MIT License](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/blob/master/LICENSE.md).

## Getting help

If you need help, please ask a question by
[opening a GitHub issue](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/issues/new).

## Installation

Install ConicIP.jl from the Julia General registry:

```julia
import Pkg
Pkg.add("ConicIP")
```

Requires Julia 1.10 or later.

## Basic usage

ConicIP has the interface
```julia
sol = conicIP( Q , c , A , b , 𝐾 , G , d )
```
For the problem
```
minimize    ½yᵀQy - cᵀy
s.t         Ay ≧𝐾 b,  𝐾 = 𝐾₁  × ⋯ × 𝐾ⱼ
            Gy  = d
```

`𝐾` is a list of tuples of the form `(Cone Type ∈ {"R", "Q", "S"}, Cone Dimension)` specifying the cone `𝐾ᵢ`. For example, the cone `𝐾 = 𝑅² × 𝑄³ × 𝑅²` has the following specification:

```julia
𝐾 = [ ("R",2) , ("Q",3),  ("R",2) ]
```

ConicIP returns `sol`, a structure containing the status (`sol.status`), primal variables (`sol.y`), dual variables (`sol.v`, `sol.w`), objective values (`sol.pobj`, `sol.dobj`), and convergence residuals (`sol.prFeas`, `sol.duFeas`, `sol.muFeas`).

To solve the problem

```
minimize    ½yᵀQy - cᵀy
such that   y ≧ 0
```

for example, use `ConicIP` as follows

```julia
using ConicIP
using SparseArrays, LinearAlgebra

n = 1000

Q = sprandn(n, n, 0.1)
Q = Q'*Q
c = ones(n)
A = sparse(1.0I, n, n)
b = zeros(n)
𝐾 = [("R", n)]

sol = conicIP(Q, c, A, b, 𝐾, verbose=true);
```

## Use with JuMP

ConicIP implements a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) wrapper, so it can be used as a solver in [JuMP](https://github.com/jump-dev/JuMP.jl). Note: only linear objectives are supported through JuMP; use the direct `conicIP` API for quadratic programs.

```julia
using JuMP
using ConicIP

model = Model(ConicIP.Optimizer)
@variable(model, x[1:10] >= 0)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, sum(x))
optimize!(model)
value.(x) # should be ≈ [0.1, 0.1, …, 0.1]
```

## Documentation

For full documentation including tutorials, how-to guides, and API reference, see the [stable documentation](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/stable/) (or the [development version](https://MPF-Optimization-Laboratory.github.io/ConicIP.jl/dev/)).
