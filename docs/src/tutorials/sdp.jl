# # Semidefinite Programs
#
# A semidefinite constraint requires a symmetric matrix to be positive
# semidefinite. ConicIP handles semidefinite blocks natively, alongside
# linear and second-order cone blocks, both through the direct
# [`conicIP`](@ref) interface and through JuMP's `PSDCone()`.
#
# ## Semidefinite support
#
# **What is supported.** Dense semidefinite blocks of any order, any number
# of them, freely mixed with `"R"` and `"Q"` blocks in one cone
# specification; linear equality constraints `Gy = d`; and quadratic
# objectives through the direct [`conicIP`](@ref) API. Through JuMP, see the
# [JuMP Integration](@ref) guide for what the wrapper accepts.
#
# **How it is solved.** A problem with any `"S"` block is routed to
# [`kktsolver_qr`](@ref ConicIP.kktsolver_qr), the dense double-QR method —
# see the [KKT Solvers](@ref) guide. The Nesterov-Todd scaling block of a
# semidefinite cone is a dense congruence transform, so the sparse solver has
# nothing to exploit. The cost model follows from that routing: the dense
# variable-space and reduced matrices grow with the *total* cone-vector
# dimension times the number of variables, not with the order of a single
# block. An order-`n` block contributes `n(n+1)/2` rows, so a single
# order-100 block is 5050 rows, and a hundred order-10 blocks are 5500 rows —
# comparable cost, even though the largest matrix differs by an order of
# magnitude. Memory, not iteration count, is the binding constraint.
#
# **Accuracy in practice.** The SDPLIB 1.2 benchmark set (Borchers 1999) is
# the reference. `benchmark/sdplib.jl` runs a spread of instances through the
# MOI wrapper and reports status, iterations, time and relative error against
# the published optimal values; `test/sdplib/README.md` records provenance,
# checksums and the SDPA sign conventions for the six-instance subset that
# serves as a CI gate (fetched from a pinned upstream commit and cached; no
# SDPLIB data is stored in the repository). On an Apple-Silicon laptop, with
# relative error
# measured as `|obj - ref| / (1 + |ref|)`:
#
# | Instances | `optTol` | Outcome |
# |:---|:---|:---|
# | `control1`, `truss1`–`truss4`, `theta1`, `theta2`, `mcp100`, `mcp124-1`, `qap5`, `arch0` | `1e-8` | `OPTIMAL`, relative error `1e-7` … `1e-10` |
# | `control2`, `control3`, `gpp100`, `hinf2` | `1e-6` (default) | `OPTIMAL`, relative error `2e-7` … `2e-5` |
# | `hinf1`, `hinf3`, `qap6` | `1e-4` | `OPTIMAL`, relative error `3e-4` … `1e-3` |
#
# Each row gives a tolerance that solves those instances; `optTol = 1e-4`
# solves every instance in the table. Solve times are modest:
# `theta1` (order 50) takes about 0.1 s, `theta2` (order 100) about 1.7 s,
# and `arch0` (order 335) about 8 s.
#
# **What is not there.**
#
# - No chordal decomposition and no sparse-SDP specialization. A large
#   sparse coefficient matrix is factored densely once an `"S"` block is
#   present.
# - The degenerate instances need a loose tolerance. The `hinf` family lacks
#   strict complementarity, and `qap6` behaves the same way: once feasibility
#   reaches roughly `1e-6` the KKT system is ill-conditioned, the Newton step
#   loses accuracy, and the iteration stagnates. At the default `optTol =
#   1e-6`, `hinf1` and `qap6` return `ITERATION_LIMIT` and `hinf3` returns
#   `NUMERICAL_ERROR`. Set `optTol = 1e-4` for these.
# - Tighter is not always better. `control2`, `control3`, `gpp100` and
#   `hinf2` give six correct digits at the default tolerance, but at
#   `optTol = 1e-8` they stagnate in the same way and come back as
#   `ITERATION_LIMIT` or `NUMERICAL_ERROR`. Loosen before tightening.
#
# **Failures are statuses, not exceptions.** Every factorization and cone
# line search that can fail on a boundary iterate is guarded. A failure ends
# the solve with `status = :Error` (`NUMERICAL_ERROR` through MOI) and a
# reason in `sol.message`, returning the best iterate seen so far. It is
# never an escaped `PosDefException`.
#
# ## Vectorization Convention
#
# The cone specification `("S", k)` describes a semidefinite cone where
# `k = n(n+1)/2` is the dimension of the vectorized representation of an
# `n × n` symmetric matrix.
#
# - [`vecm(Z)`](@ref ConicIP.vecm) vectorizes a symmetric matrix, reading the
#   upper triangle row by row and scaling off-diagonal entries by `√2` so
#   that `dot(vecm(X), vecm(Y)) == tr(X*Y)`.
# - [`mat(x)`](@ref ConicIP.mat) reconstructs the symmetric matrix from its
#   vectorized form.
#
# JuMP's `PSDCone()` uses MathOptInterface's
# `PositiveSemidefiniteConeTriangle`, which reads the upper triangle *column*
# by column and applies no scaling. The MOI wrapper translates between the
# two conventions, so a JuMP model never sees the `√2`.
#
# ## Example: Projection onto the PSD Cone
#
# Project the diagonal matrix `diag(1, 1, 1, -1, -1, -1)` onto the cone
# of positive semidefinite matrices. The expected result clips the negative
# eigenvalues to zero: `diag(1, 1, 1, 0, 0, 0)`.

using ConicIP, SparseArrays, LinearAlgebra

## 6×6 matrix → vectorized dimension k = 6*7/2 = 21
k = 21
Q = sparse(1.0I, k, k)
target = diagm(0 => [1.0, 1, 1, -1, -1, -1])
c = ConicIP.vecm(target)

A = sparse(1.0I, k, k)
b = zeros(k)
cone_dims = [("S", k)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false, optTol=1e-7)
sol.status

# Reconstruct the matrix from the solution and check its eigenvalues:

result = ConicIP.mat(sol.y)
round.(eigvals(Symmetric(result)), digits=4)

# The negative eigenvalues have been projected to (approximately) zero.
#
# ## Understanding `vecm` and `mat`
#
# Let's see how the vectorization works on a small example:

M = [1.0 2.0 3.0;
     2.0 5.0 6.0;
     3.0 6.0 9.0]

v = ConicIP.vecm(M)

# The vector `v` has length `n(n+1)/2 = 6`. Off-diagonal entries are
# scaled by `√2`:

round.(v, digits=4)

# Reconstruct the original matrix:

M_recovered = ConicIP.mat(v)
round.(M_recovered, digits=4)

# ## Modelling with JuMP
#
# The same cone is available through JuMP as `PSDCone()`, with no
# vectorization to think about. As a worked example, compute the Lovász
# theta number of the five-cycle ``C_5``. For a graph ``G`` with edge set
# ``E``,
#
# ```math
# \vartheta(G) \;=\; \max_X \; \operatorname{tr}(JX)
# \quad\text{s.t.}\quad
# \operatorname{tr}(X) = 1,\;\;
# X_{ij} = 0 \;\;\forall\, (i,j) \in E,\;\;
# X \succeq 0,
# ```
#
# where ``J`` is the all-ones matrix, so that ``\operatorname{tr}(JX)`` is
# just the sum of the entries of ``X``. Lovász's original paper gives the
# closed form for a cycle of odd length ``n``,
# ``\vartheta(C_n) = n\cos(\pi/n) / (1 + \cos(\pi/n))``, which for ``n = 5``
# collapses to ``\vartheta(C_5) = \sqrt{5} \approx 2.2360680``.

using JuMP, LinearAlgebra

model = Model(ConicIP.Optimizer)
set_silent(model)

@variable(model, X[1:5, 1:5], Symmetric)
@constraint(model, psd, X in PSDCone())
@constraint(model, tr(X) == 1)
edges = [(i, mod1(i + 1, 5)) for i in 1:5]
@constraint(model, [e in edges], X[e[1], e[2]] == 0)
@objective(model, Max, sum(X))

optimize!(model)
termination_status(model)

# Compare the computed value with the closed form:

objective_value(model), sqrt(5)

#-

abs(objective_value(model) - sqrt(5))

# ## Reading the dual
#
# `dual` on a `PSDCone()` constraint returns the dual matrix, not a vector:

Y = dual(psd)

# JuMP reshapes the MOI dual back into an `n × n` symmetric matrix, so each
# off-diagonal entry appears once and carries no scaling — `Y[1,2]` is the
# dual matrix entry ``Y_{12}`` itself. The `√2` of the internal `vecm` form
# never surfaces. Note the pairing that goes with this convention: the
# complementarity product is `tr(X*Y)`, which in the entries above means the
# off-diagonals count twice,
#
# ```math
# \operatorname{tr}(XY) \;=\; \sum_i X_{ii} Y_{ii} \;+\; 2\sum_{i<j} X_{ij} Y_{ij}.
# ```
#
# The dual is positive semidefinite, and complementary to `X` at optimality —
# both statements hold to the requested `optTol`, here the default `1e-6`:

round.(eigvals(Y), digits=6)

#-

X_opt = value.(X)
round(tr(X_opt * Y), digits=8)
