# # Getting Started
#
# This tutorial introduces ConicIP.jl's problem formulation, core API, and
# solution interpretation. By the end you will have solved your first
# optimization problem in under a minute.
#
# ## Problem Formulation
#
# ConicIP solves optimization problems of the form
#
# ```
# minimize    ½ yᵀQy - cᵀy
# subject to  Ay ≥_K b
#             Gy  = d
# ```
#
# where `≥_K` denotes a generalized inequality with respect to a cone `K`.
#
# ## Arguments at a Glance
#
# | Argument | Size | Description |
# |----------|------|-------------|
# | `Q` | n × n | Positive semidefinite Hessian (use `spzeros(n,n)` for LPs) |
# | `c` | n × 1 **Matrix** | Linear objective term |
# | `A` | m × n | Inequality constraint matrix |
# | `b` | m × 1 **Matrix** | Inequality right-hand side |
# | `cone_dims` | `Vector` of `(String,Int)` | Cone specification for rows of `A` |
# | `G` | p × n | Equality constraint matrix (optional) |
# | `d` | p × 1 **Matrix** | Equality right-hand side (optional) |
#
# !!! note "Column matrices"
#     ConicIP expects `c`, `b`, and `d` to be **n × 1 matrices** (two-dimensional),
#     not Julia `Vector`s. Use `reshape(v, :, 1)` to convert a vector `v`.
#
# ## Cone Specification
#
# The `cone_dims` argument is a vector of `(type, dimension)` tuples describing
# how the rows of `A` and `b` are partitioned into cone constraints:
#
# - `("R", n)` -- nonnegative orthant: the first `n` rows satisfy `Ay - b ≥ 0`
# - `("Q", m)` -- second-order cone: the next `m` rows satisfy `‖(Ay-b)[2:end]‖ ≤ (Ay-b)[1]`
# - `("S", k)` -- semidefinite cone (experimental): the next `k` rows represent a vectorized
#   symmetric matrix that must be positive semidefinite, where `k = n(n+1)/2`
#
# For example, `[("R", 3), ("Q", 5)]` means the first 3 rows of `Ay ≥ b`
# are nonnegative constraints, and the next 5 rows form a second-order cone
# constraint.
#
# ## Example: Box-Constrained QP
#
# Let's solve a box-constrained quadratic program:
# minimize ½ yᵀQy - cᵀy subject to 0 ≤ y ≤ 1.
#
# We encode the box constraints `0 ≤ y ≤ 1` as `[I; -I] y ≥ [0; -1]`.

using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

n = 10
Q = sparse(Diagonal(rand(n) .+ 0.1))
c = randn(n, 1)   # note: n × 1 Matrix, not a Vector

## Encode 0 ≤ y ≤ 1 as [I; -I] y ≥ [0; -1]
A = sparse([I(n); -I(n)])
b = [zeros(n, 1); -ones(n, 1)]
cone_dims = [("R", 2n)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol.status

# The primal solution:

round.(sol.y, digits=4)

# ## Interpreting the Solution
#
# The solver returns a [`Solution`](@ref ConicIP.Solution) struct with these key fields:
#
# | Field | Description |
# |-------|-------------|
# | `sol.y` | Primal variables (n × 1 Matrix) |
# | `sol.v` | Dual variables for inequality constraints |
# | `sol.w` | Dual variables for equality constraints |
# | `sol.status` | `:Optimal`, `:Infeasible`, `:Unbounded`, `:Abandoned`, or `:Error` |
# | `sol.pobj`, `sol.dobj` | Primal and dual objective values |
# | `sol.prFeas` | Primal feasibility residual |
# | `sol.duFeas` | Dual feasibility residual |
# | `sol.muFeas` | Complementarity residual |
# | `sol.Iter` | Number of iterations |
#
# Let's inspect the convergence residuals:

(prFeas=sol.prFeas, duFeas=sol.duFeas, muFeas=sol.muFeas)

# All residuals are well below the default tolerance (`optTol = 1e-6`).
#
# ## Using JuMP Instead
#
# You can also model problems via [JuMP](https://jump.dev/JuMP.jl/) instead
# of the direct `conicIP` API. Here's the same box-constrained QP:

using JuMP

model = Model(() -> ConicIP.Optimizer(verbose=false))

@variable(model, 0 <= x[i=1:n] <= 1)
@objective(model, Min, 0.5 * sum(Q[i,i] * x[i]^2 for i in 1:n) - sum(c[i] * x[i] for i in 1:n))

optimize!(model)
termination_status(model)

# ```@raw html
# <br>
# ```
#
# The JuMP solution matches the direct API:

round.(value.(x), digits=4)

# ## Next Steps
#
# - [Linear Programs](@ref) -- LPs with equality constraints and duals
# - [Quadratic Programs](@ref) -- portfolio optimization on the simplex
# - [Second-Order Cone Programs](@ref) -- norm minimization and mixed cones
# - [JuMP Integration](@ref) -- full guide on using ConicIP through JuMP
