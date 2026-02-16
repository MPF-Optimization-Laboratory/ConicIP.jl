# ConicIP.jl

**ConicIP.jl** is a pure-Julia conic interior-point solver for optimization
problems with linear, second-order cone, and (experimental) semidefinite
constraints.

## Why ConicIP?

- **Pure Julia** — no binary dependencies or external solver installations
- **Custom KKT solver callbacks** — exploit problem structure for speed
- **Extensible** — plug in your own factorization at each interior-point iteration
- **Nesterov-Todd scaling** — symmetric primal-dual scaling for good numerical behavior
- **Infeasibility detection** — returns certificates for infeasible/unbounded problems

## Problem Formulation

ConicIP solves problems of the form:

```
minimize    ½ yᵀQy - cᵀy
subject to  Ay ≥_K b
            Gy  = d
```

where `Q ≽ 0` and `K` is a Cartesian product of cones:

| Cone | Spec | Description |
|------|------|-------------|
| Nonnegative orthant | `("R", n)` | Linear inequalities |
| Second-order cone | `("Q", n)` | Norm constraints |
| Semidefinite (experimental) | `("S", k)` | Matrix positivity |

## Quick Example

```@example quickstart
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

# Box-constrained QP: minimize ½ y'Qy - c'y subject to 0 ≤ y ≤ 1
n = 5
Q = sparse(Diagonal(rand(n) .+ 0.1))
c = randn(n)

# Constraints: [I; -I] y ≥ [0; -1]
A = sparse([I(n); -I(n)])
b = [zeros(n); -ones(n)]
cone_dims = [("R", 2n)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol.status
```

```@example quickstart
round.(sol.y, digits=4)
```

## Two Ways to Use ConicIP

**Direct API** — full control, supports quadratic objectives:

```julia
sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
```

**JuMP/MOI** — algebraic modeling, linear objectives only:

```julia
using JuMP, ConicIP
model = Model(ConicIP.Optimizer)
@variable(model, x[1:n] >= 0)
@objective(model, Min, sum(x))
optimize!(model)
```

See the [JuMP Integration](@ref) guide for details.

## Choosing a Solver

ConicIP is a good fit when you need:

- A **pure-Julia** solver with no binary dependencies
- **Custom KKT solver callbacks** to exploit problem structure
- A solver for **moderate-size** LP/QP/SOCP problems

For large-scale production use, consider:

| Solver | Pure Julia | QP | SOCP | SDP | Custom KKT |
|--------|-----------|-----|------|-----|------------|
| **ConicIP** | ✓ | ✓ | ✓ | experimental | ✓ |
| [COSMO.jl](https://github.com/oxfordcontrol/COSMO.jl) | ✓ | ✓ | ✓ | ✓ | ✗ |
| [Hypatia.jl](https://github.com/chriscoey/Hypatia.jl) | ✓ | ✓ | ✓ | ✓ | ✗ |
| [SCS](https://github.com/jump-dev/SCS.jl) | ✗ (C) | ✗ | ✓ | ✓ | ✗ |
| [ECOS](https://github.com/jump-dev/ECOS.jl) | ✗ (C) | ✗ | ✓ | ✗ | ✗ |

## Contents

```@contents
Pages = [
    "installation.md",
    "tutorials/generated/getting_started.md",
    "tutorials/generated/lp.md",
    "tutorials/generated/qp.md",
    "tutorials/generated/socp.md",
    "tutorials/generated/sdp.md",
    "guides/jump.md",
    "guides/kkt_solvers.md",
    "guides/preprocessing.md",
    "background.md",
    "api.md",
]
Depth = 2
```
