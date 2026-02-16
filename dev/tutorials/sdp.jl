# # Semidefinite Programs (Experimental)
#
# !!! warning "Experimental"
#     Semidefinite cone support in ConicIP is experimental. It works for
#     small problems but has not been extensively tested.
#
# A semidefinite cone constraint requires a symmetric matrix to be positive
# semidefinite. In ConicIP, matrices are stored in a vectorized form using
# [`vecm`](@ref ConicIP.vecm) and [`mat`](@ref ConicIP.mat).
#
# ## Vectorization Convention
#
# The cone specification `("S", k)` describes a semidefinite cone where
# `k = n(n+1)/2` is the dimension of the vectorized representation of an
# `n × n` symmetric matrix.
#
# - [`vecm(Z)`](@ref ConicIP.vecm) vectorizes a symmetric matrix, scaling
#   off-diagonal entries by `√2` so that `dot(vecm(X), vecm(Y)) == tr(X*Y)`.
# - [`mat(x)`](@ref ConicIP.mat) reconstructs the symmetric matrix from its
#   vectorized form.
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

X = [1.0 2.0 3.0;
     2.0 5.0 6.0;
     3.0 6.0 9.0]

v = ConicIP.vecm(X)

# The vector `v` has length `n(n+1)/2 = 6`. Off-diagonal entries are
# scaled by `√2`:

round.(v, digits=4)

# Reconstruct the original matrix:

X_recovered = ConicIP.mat(v)
round.(X_recovered, digits=4)
