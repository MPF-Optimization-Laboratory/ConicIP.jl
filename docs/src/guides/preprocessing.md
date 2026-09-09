# Preprocessing

ConicIP includes a preprocessing wrapper that removes redundant constraints
before solving. This can prevent numerical issues caused by rank-deficient
constraint matrices.

## When to Use Preprocessing

Use [`preprocess_conicIP`](@ref ConicIP.preprocess_conicIP) instead of `conicIP` when:

- The equality constraint matrix `G` may have redundant rows
  (i.e., `rank(G) < size(G, 1)`)
- The combined matrix `[Q  A'  G']` may be rank-deficient
- You receive `:Error` or `:Abandoned` status from the solver due to
  singular KKT systems

## Usage

`preprocess_conicIP` has the same signature as `conicIP` and passes all
keyword arguments through:

```julia
sol = preprocess_conicIP(Q, c, A, b, cone_dims, G, d; verbose=false)
```

## What It Does

The preprocessor performs three steps:

1. **Singleton fixing:** an equality row with a single nonzero,
   `gᵢⱼ yⱼ = dᵢ`, fixes `yⱼ = dᵢ/gᵢⱼ`. The row and the column are removed
   and the right-hand sides and objective adjusted; the fixed value and
   the row's dual multiplier are restored on the way out. Two singletons
   on the same column are dropped when their fixed values agree to
   tolerance; conflicting values are screened for a Farkas certificate.

2. **Equality constraint reduction** (`rank_check`): uses
   [`imcols`](@ref ConicIP.imcols) to identify and remove linearly
   dependent rows from `G` and `d`, ensuring `rank(G) = size(G, 1)`.

3. **Dual constraint check** (`rank_check`): checks that the combined
   system `[Q  A'  G']` has full row rank and, if not, opts into the
   solver's static KKT regularization rather than altering the problem.

Steps 2 and 3 are the expensive ones (two sparse QR factorizations) and
exist for KKT solvers that need a full-rank `G`. `rank_check = :auto`
(default) runs them only when such a solver will be used — dense QR for
small and semidefinite problems, or an explicit solver other than
[`kktsolver_ldl`](@ref ConicIP.kktsolver_ldl) or its cached callable — since the LDLᵀ solver
regularizes the equality block itself. `:always` and `:never` override.

After preprocessing, the reduced problem is passed to `conicIP`. Postsolve
recomputes feasibility against the original data and revalidates certificates.
A reduced optimum that fails the original `optTol` is returned as `:Error`,
with the restored point available for inspection.

## The `imcols` Function

[`imcols`](@ref ConicIP.imcols) identifies independent columns (rows) in a
linear system `Ax = b`. It returns the indices of a maximal linearly
independent subset and checks consistency of the system.

```julia
using ConicIP, SparseArrays

A = sparse([1.0 2.0 3.0;
            2.0 4.0 6.0;   # redundant (2× row 1)
            0.0 1.0 1.0])
b = [1.0; 2.0; 1.0]

# imcols returns indices of independent rows
idx = ConicIP.imcols(A', b)
```

## Example

```@example preprocess
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

n = 5
Q = sparse(1.0I, n, n)
c = randn(n)

A = sparse(1.0I, n, n)
b = zeros(n)
cone_dims = [("R", n)]

# Redundant equality constraints: row 3 = row 1 + row 2
G = [1.0 1.0 0.0 0.0 0.0;
     0.0 0.0 1.0 1.0 0.0;
     1.0 1.0 1.0 1.0 0.0]   # redundant!
d = [1.0; 1.0; 2.0]

# preprocess_conicIP handles the redundancy automatically
sol = preprocess_conicIP(Q, c, A, b, cone_dims, G, d; verbose=false)
sol.status
```

```@example preprocess
round.(sol.y, digits=4)
```
