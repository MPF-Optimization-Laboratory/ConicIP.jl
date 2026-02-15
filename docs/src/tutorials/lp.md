# Linear Programs

A linear program (LP) is a special case of the ConicIP problem formulation
with `Q = 0`:

```
minimize    -cᵀy
subject to   Ay ≥ b
             Gy  = d
```

## Example: LP with Equality and Inequality Constraints

Solve a small LP with nonnegativity and an equality constraint:

```
minimize    -2y₁ - 3y₂ - y₃ - y₄ - y₅
subject to   y₁ + y₂ + y₃ + y₄ + y₅ = 4
             y ≥ 0
```

```@example lp
using ConicIP, SparseArrays, LinearAlgebra

n = 5
Q = spzeros(n, n)
c = reshape([2.0, 3.0, 1.0, 1.0, 1.0], :, 1)

# Nonnegativity: y ≥ 0
A = sparse(1.0I, n, n)
b = zeros(n, 1)
cone_dims = [("R", n)]

# Equality constraint: sum(y) = 4
G = ones(1, n)
d = reshape([4.0], 1, 1)

sol = conicIP(Q, c, A, b, cone_dims, G, d; verbose=false)
sol.status
```

```@example lp
round.(sol.y, digits=4)
```

The optimal solution puts all weight on the variable with the largest
objective coefficient (y₂ = 4).

## Reading Primal and Dual Solutions

The dual variables `sol.v` give the shadow prices for the inequality
constraints, and `sol.w` gives the dual for the equality constraint:

```@example lp
round.(sol.v, digits=4)
```

```@example lp
round.(sol.w, digits=4)
```

The primal and dual objective values should be equal at optimality:

```@example lp
(pobj=round(sol.pobj, digits=6), dobj=round(sol.dobj, digits=6))
```
