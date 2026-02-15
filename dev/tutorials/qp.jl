# # Quadratic Programs
#
# A quadratic program (QP) has a positive semidefinite Hessian `Q`:
#
# ```
# minimize    ½ yᵀQy - cᵀy
# subject to   Ay ≥ b
#              Gy  = d
# ```
#
# ## Example: Projection onto the Simplex
#
# Project the point `p = [1, 2, 3, 4, 5]` onto the probability simplex
# `{y : y ≥ 0, ∑yᵢ = 1}`. This is equivalent to solving
# `minimize ½ ‖y - p‖² subject to y ≥ 0, 1ᵀy = 1`,
# which in ConicIP form becomes `minimize ½ yᵀIy - pᵀy` with
# appropriate constraints.

using ConicIP, SparseArrays, LinearAlgebra

n = 5
Q = sparse(1.0I, n, n)
p = reshape(collect(1.0:n), :, 1)

## Nonnegativity constraints: y ≥ 0
A = sparse(1.0I, n, n)
b = zeros(n, 1)
cone_dims = [("R", n)]

## Simplex constraint: sum(y) = 1
G = ones(1, n)
d = ones(1, 1)

## Note: we pass Q*p as the linear term c = Qp = Ip = p
sol = conicIP(Q, Q * p, A, b, cone_dims, G, d;
              verbose=false, optTol=1e-7)
sol.status

#

round.(sol.y, digits=4)

# The solution concentrates weight on the largest components of `p`, as
# expected for the nearest point on the simplex.
#
# ## Convergence Diagnostics
#
# The [`Solution`](@ref ConicIP.Solution) struct reports convergence diagnostics:
#
# - `prFeas` -- primal feasibility residual
# - `duFeas` -- dual feasibility (KKT stationarity) residual
# - `muFeas` -- complementarity residual
# - `pobj`, `dobj` -- primal and dual objective values

(prFeas=sol.prFeas, duFeas=sol.duFeas, muFeas=sol.muFeas,
 pobj=round(sol.pobj, digits=6), dobj=round(sol.dobj, digits=6))

# At optimality, `pobj ≈ dobj` and all residuals are below `optTol`.
#
# ## Example: Portfolio Optimization
#
# A classic application of QP is mean-variance portfolio optimization.
# Given expected returns `μ` and a covariance matrix `Σ`, find the
# minimum-variance portfolio with expected return at least `r_min`:
#
# ```
# minimize    ½ yᵀΣy
# subject to  μᵀy ≥ r_min
#             1ᵀy = 1
#             y ≥ 0
# ```

using Random
Random.seed!(42)

nassets = 8
returns = rand(nassets) * 0.1          # expected returns: 0% to 10%
## Build a realistic covariance matrix from factor model
F = randn(nassets, 3) * 0.05
Sigma = F * F' + Diagonal(rand(nassets) * 0.01 .+ 0.001)
Sigma = sparse(Symmetric(Sigma))

r_min = 0.05  # target minimum return

## Q = Sigma, c = 0 (pure quadratic objective)
Q_port = Sigma
c_port = zeros(nassets, 1)

## Inequality constraints: [μᵀ; I] y ≥ [r_min; 0]
A_port = sparse([returns'; I(nassets)])
b_port = [reshape([r_min], 1, 1); zeros(nassets, 1)]
cone_dims_port = [("R", 1 + nassets)]

## Equality constraint: sum(y) = 1
G_port = ones(1, nassets)
d_port = ones(1, 1)

sol_port = conicIP(Q_port, c_port, A_port, b_port, cone_dims_port,
                   G_port, d_port; verbose=false, optTol=1e-7)
sol_port.status

#

weights = round.(sol_port.y, digits=4)

# Portfolio expected return and variance:

port_return = round(dot(returns, sol_port.y), digits=6)
port_variance = round((sol_port.y' * Sigma * sol_port.y)[1], digits=6)
(expected_return=port_return, variance=port_variance)
