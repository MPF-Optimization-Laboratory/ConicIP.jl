# # Second-Order Cone Programs
#
# A second-order cone (SOC) constraint has the form
#
# ```
# ‖x‖ ≤ t
# ```
#
# or equivalently `(t, x) ∈ Q`, where `Q` is the second-order (Lorentz) cone.
# In ConicIP, this is specified as `("Q", dim)` where `dim = 1 + length(x)`.
# The **first** component of the cone block is the scalar bound `t`.
#
# SOC constraints appear naturally in robust optimization, norm minimization,
# and chance-constrained programming.
#
# ## Encoding SOC Constraints
#
# To enforce `‖y‖ ≤ t`, arrange the constraint rows so that:
# - Row 1 of the SOC block gives `t` (the bound)
# - Rows 2:end give the components of `y`
#
# Concretely, if `Ay - b` restricted to the SOC block yields a vector
# `[t; x₁; x₂; ...]`, then the constraint is `‖[x₁, x₂, ...]‖ ≤ t`.
#
# ## Example: Projection onto the Unit Ball
#
# Project a point onto the unit Euclidean ball `{y : ‖y‖ ≤ 1}`.
# The expected result is the normalized vector `a / ‖a‖`.

using ConicIP, SparseArrays, LinearAlgebra

n = 3
Q = sparse(1.0I, n, n)
a = ones(n)  # point to project

## SOC constraint: (t, y) ∈ Q with t = 1
## Encode as: [0ᵀ; I] y - [-1; 0] ∈ Q
## i.e., the SOC vector is (1, y₁, y₂, y₃) and we need ‖y‖ ≤ 1
A = [spzeros(1, n); sparse(1.0I, n, n)]
b = [-1.0; zeros(n)]
cone_dims = [("Q", n + 1)]

sol = conicIP(Q, Q * a, A, b, cone_dims;
              verbose=false, optTol=1e-7)
sol.status

#

round.(sol.y, digits=4)

# Compare to the expected answer `a / ‖a‖`:

round.(a ./ norm(a), digits=4)

# ## Example: Mixed Cones (Nonnegative + SOC)
#
# Combine nonnegative (`"R"`) and second-order cone (`"Q"`) constraints.
# Here we minimize a linear objective subject to both `y ≥ 0` and `‖y‖ ≤ 1`:

using Random
Random.seed!(42)

n = 5
Q = sparse(1.0I, n, n)
c = randn(n)

## Stack constraints: first n rows are R+ (y ≥ 0),
## next n+1 rows are SOC (‖y‖ ≤ 1)
A = [sparse(1.0I, n, n);          # y ≥ 0
     spzeros(1, n);                # t placeholder for SOC bound
     sparse(1.0I, n, n)]           # y components in SOC
b = [zeros(n);                     # R+ bound
     -1.0;                         # t ≥ 1
     zeros(n)]                     # SOC body
cone_dims = [("R", n), ("Q", n + 1)]

sol2 = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol2.status

#

round.(sol2.y, digits=4)

# The solution lies in the intersection of the nonnegative orthant and the
# unit ball — the nonnegative part of the unit sphere.
#
# ## Example: Robust Least-Squares
#
# Given an uncertain measurement matrix `A₀ + δA` and observations `b₀`,
# robust least-squares minimizes the worst-case residual:
#
# ```
# minimize  ‖A₀ y - b₀‖ + ρ ‖y‖
# ```
#
# This can be reformulated as an SOCP. We introduce auxiliary variables
# `t₁ ≥ ‖A₀ y - b₀‖` and `t₂ ≥ ‖y‖` and minimize `t₁ + ρ t₂`.

Random.seed!(99)

m, n_var = 10, 5
A0 = randn(m, n_var)
y_true = randn(n_var)
b0 = A0 * y_true + 0.1 * randn(m)
ρ = 0.5  # regularization weight

## Decision variables: [y; t₁; t₂] of length n_var + 2
nz = n_var + 2
Qz = spzeros(nz, nz)

## Objective: minimize t₁ + ρ t₂  →  c = -[0; 1; ρ] (ConicIP minimizes -c'z)
cz = zeros(nz)
cz[n_var+1] = 1.0
cz[n_var+2] = ρ

## SOC 1: (t₁, A₀ y - b₀) ∈ Q^{m+1}
##   Row for t₁: [0...0  1  0] z ≥ 0  (extracts t₁)
##   Rows for residual: [A₀  0  0] z ≥ b₀
A_soc1_t = spzeros(1, nz); A_soc1_t[1, n_var+1] = 1.0
A_soc1_r = [sparse(A0) spzeros(m, 2)]
b_soc1 = [0.0; b0]

## SOC 2: (t₂, y) ∈ Q^{n_var+1}
##   Row for t₂: [0...0  0  1] z ≥ 0
##   Rows for y: [I  0  0] z ≥ 0
A_soc2_t = spzeros(1, nz); A_soc2_t[1, n_var+2] = 1.0
A_soc2_y = [sparse(1.0I, n_var, n_var) spzeros(n_var, 2)]
b_soc2 = zeros(1 + n_var)

A_all = [A_soc1_t; A_soc1_r; A_soc2_t; A_soc2_y]
b_all = [b_soc1; b_soc2]
cone_dims_robust = [("Q", m + 1), ("Q", n_var + 1)]

sol3 = conicIP(Qz, cz, sparse(A_all), b_all, cone_dims_robust; verbose=false)
sol3.status

# The robust solution:

round.(sol3.y[1:n_var], digits=4)
