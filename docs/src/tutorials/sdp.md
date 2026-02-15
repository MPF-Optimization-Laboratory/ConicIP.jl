# Semidefinite Programs (Experimental)

!!! warning "Experimental"
    Semidefinite cone support in ConicIP is experimental. It works for
    small problems but has not been extensively tested.

A semidefinite cone constraint requires a symmetric matrix to be positive
semidefinite. In ConicIP, matrices are stored in a vectorized form using
[`vecm`](@ref ConicIP.vecm) and [`mat`](@ref ConicIP.mat).

## Vectorization Convention

The cone specification `("S", k)` describes a semidefinite cone where
`k = n(n+1)/2` is the dimension of the vectorized matrix for an `n × n`
symmetric matrix.

- [`vecm(Z)`](@ref ConicIP.vecm) vectorizes a symmetric matrix, scaling
  off-diagonal entries by `√2` so that inner products are preserved:
  `dot(vecm(X), vecm(Y)) == tr(X*Y)`.
- [`mat(x)`](@ref ConicIP.mat) reconstructs the symmetric matrix from its
  vectorized form.

## Example: Projection onto the PSD Cone

Project the diagonal matrix `diag(1, 1, 1, -1, -1, -1)` onto the cone
of positive semidefinite matrices. The expected result clips the negative
eigenvalues to zero: `diag(1, 1, 1, 0, 0, 0)`.

```@example sdp
using ConicIP, SparseArrays, LinearAlgebra

# 6×6 matrix → vectorized dimension k = 6*7/2 = 21
k = 21
Q = sparse(1.0I, k, k)
target = diagm(0 => [1.0, 1, 1, -1, -1, -1])
c = reshape(ConicIP.vecm(target), :, 1)

A = sparse(1.0I, k, k)
b = zeros(k, 1)
cone_dims = [("S", k)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false, optTol=1e-7)
sol.status
```

Reconstruct the matrix from the solution and check its eigenvalues:

```@example sdp
result = ConicIP.mat(sol.y)
round.(eigvals(Symmetric(result)), digits=4)
```

The negative eigenvalues have been projected to (approximately) zero.
