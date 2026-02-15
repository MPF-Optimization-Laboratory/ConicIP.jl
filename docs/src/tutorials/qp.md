# Quadratic Programs

A quadratic program (QP) has a positive semidefinite Hessian `Q`:

```
minimize    ½ yᵀQy - cᵀy
subject to   Ay ≥ b
             Gy  = d
```

## Example: Projection onto the Simplex

Project the point `c = [1, 2, 3, 4, 5]` onto the probability simplex
`{y : y ≥ 0, ∑yᵢ = 1}`:

```@example qp
using ConicIP, SparseArrays, LinearAlgebra

n = 5
Q = sparse(1.0I, n, n)
c = reshape(collect(1.0:n), :, 1)

# Nonnegativity constraints: y ≥ 0
A = sparse(1.0I, n, n)
b = zeros(n, 1)
cone_dims = [("R", n)]

# Simplex constraint: sum(y) = 1
G = ones(1, n)
d = ones(1, 1)

sol = conicIP(Q, Q * c, A, b, cone_dims, G, d;
              verbose=false, optTol=1e-7)
sol.status
```

```@example qp
round.(sol.y, digits=4)
```

The solution concentrates weight on the largest components of `c`, as
expected for the nearest point on the simplex.

## Convergence Information

The [`Solution`](@ref ConicIP.Solution) struct reports convergence diagnostics:

- `prFeas` -- primal feasibility: residual of `Ay ≥ b` and `Gy = d`
- `duFeas` -- dual feasibility: residual of the KKT stationarity condition
- `muFeas` -- complementarity: residual of the complementary slackness condition
- `pobj`, `dobj` -- primal and dual objective values

```@example qp
(prFeas=sol.prFeas, duFeas=sol.duFeas, muFeas=sol.muFeas, pobj=round(sol.pobj, digits=6), dobj=round(sol.dobj, digits=6))
```

At optimality, `pobj ≈ dobj` and all residuals are below the specified
tolerance (`optTol`).
