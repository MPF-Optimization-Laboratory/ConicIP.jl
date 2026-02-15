# ConicIP.jl

**ConicIP.jl** is a pure-Julia conic interior-point solver for optimization problems with linear, second-order cone, and (experimental) semidefinite constraints.

## Features

- **Pure Julia** -- no external solver dependencies
- **JuMP / MathOptInterface integration** -- use as `Model(ConicIP.Optimizer)`
- **Custom KKT solver callbacks** -- exploit problem structure for speed
- **Nesterov-Todd scaling** -- symmetric primal-dual scaling
- **Infeasibility and unboundedness detection** -- returns certificates

## Supported Problem Types

- Linear programs (LP)
- Quadratic programs (QP)
- Second-order cone programs (SOCP)
- Semidefinite programs (SDP) -- *experimental*

## Quick Example

```@example quickstart
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

# Box-constrained QP: minimize ½ y'Qy - c'y subject to 0 ≤ y ≤ 1
n = 5
Q = sparse(Diagonal(rand(n) .+ 0.1))
c = randn(n, 1)

# Constraints: [I; -I] y >= [0; -1] (i.e., y >= 0 and y <= 1)
A = sparse([I(n); -I(n)])
b = [zeros(n, 1); -ones(n, 1)]
cone_dims = [("R", 2n)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol.status
```

```@example quickstart
round.(sol.y, digits=4)
```

## Contents

```@contents
Pages = [
    "index.md",
]
Depth = 2
```
