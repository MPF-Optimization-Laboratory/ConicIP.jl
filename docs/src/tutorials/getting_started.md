# Getting Started

## Problem Formulation

ConicIP solves optimization problems of the form

```
minimize    ½ yᵀQy - cᵀy
subject to  Ay ≥_K b
            Gy  = d
```

where `≥_K` denotes a generalized inequality with respect to a cone `K`.

## Arguments

- `Q` -- positive semidefinite Hessian matrix (n × n). Set to zero for linear programs.
- `c` -- linear objective term (n × 1 **matrix**, not a vector)
- `A` -- inequality constraint matrix (m × n)
- `b` -- inequality right-hand side (m × 1 matrix)
- `cone_dims` -- cone specification for the inequality constraints
- `G` -- equality constraint matrix (p × n), optional
- `d` -- equality right-hand side (p × 1 matrix), optional

!!! note "Column matrices"
    ConicIP expects `c`, `b`, and `d` to be **n × 1 matrices** (created with
    `reshape` or literal `[1.0; 2.0;; ]`), not Julia `Vector`s. Use
    `reshape(v, :, 1)` to convert a vector `v`.

## Cone Specification

The `cone_dims` argument is a vector of `(type, dimension)` tuples describing
how the rows of `A` and `b` are partitioned into cone constraints:

- `("R", n)` -- nonnegative orthant: the first `n` rows satisfy `Ay - b ≥ 0`
- `("Q", m)` -- second-order cone: the next `m` rows satisfy `‖(Ay-b)[2:end]‖ ≤ (Ay-b)[1]`
- `("S", k)` -- semidefinite cone (experimental): the next `k` rows represent a vectorized
  symmetric matrix that must be positive semidefinite, where `k = n(n+1)/2`

For example, `[("R", 3), ("Q", 5)]` means the first 3 rows of `Ay ≥ b`
are nonnegative constraints, and the next 5 rows form a second-order cone
constraint.

## Solution

The solver returns a [`Solution`](@ref ConicIP.Solution) struct with the following key fields:

- `sol.y` -- primal variables
- `sol.w` -- dual variables for equality constraints
- `sol.v` -- dual variables for inequality constraints
- `sol.status` -- `:Optimal`, `:Infeasible`, `:Unbounded`, `:Abandoned`, or `:Error`
- `sol.pobj`, `sol.dobj` -- primal and dual objective values
- `sol.prFeas`, `sol.duFeas`, `sol.muFeas` -- feasibility residuals

## Example: Box-Constrained QP

Solve a box-constrained quadratic program: minimize ½ yᵀQy - cᵀy subject
to 0 ≤ y ≤ 1.

```@example getting_started
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

n = 10
Q = sparse(Diagonal(rand(n) .+ 0.1))
c = randn(n, 1)

# Encode 0 ≤ y ≤ 1 as [I; -I] y ≥ [0; -1]
A = sparse([I(n); -I(n)])
b = [zeros(n, 1); -ones(n, 1)]
cone_dims = [("R", 2n)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol.status
```

The primal solution:

```@example getting_started
round.(sol.y, digits=4)
```

The convergence residuals confirm the solve quality:

```@example getting_started
(prFeas=sol.prFeas, duFeas=sol.duFeas, muFeas=sol.muFeas)
```
