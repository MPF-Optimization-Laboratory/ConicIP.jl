# ConicIP.jl

A pure-Julia conic interior-point solver for optimization problems with linear, second-order cone, and semidefinite constraints.

## Features

- **Pure Julia** — no external solver dependencies
- **LP / QP / SOCP / SDP** — supports linear, quadratic, second-order cone, and semidefinite programming
- **JuMP integration** — use via the MathOptInterface wrapper
- **Custom KKT solvers** — plug in your own factorization callbacks

## Quick Example

```@example quickstart
using ConicIP
using Random
using SparseArrays
using LinearAlgebra: I
Random.seed!(42)

n = 5
Q = sparse(Matrix(1.0I, n, n))
c = randn(n, 1)
A = sparse(Matrix(1.0I, n, n))
b = zeros(n, 1)
cone_dims = [("R", n)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol.status
```

## Next Steps

- API reference and tutorials coming soon.
