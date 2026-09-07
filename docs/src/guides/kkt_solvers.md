# KKT Solvers

At each interior-point iteration, ConicIP solves a 3×3 block KKT system.
The choice of solver for this system has a significant impact on
performance. This guide covers the built-in solvers, when to use each,
and how to write custom solvers.

## The KKT System

The system solved at each iteration is:

```
┌             ┐ ┌   ┐   ┌   ┐
│ Q   G'  -A' │ │ a │   │ x │
│ G           │ │ b │ = │ y │
│ A       FᵀF │ │ c │   │ z │
└             ┘ └   ┘   └   ┘
```

where `F` is a [`Block`](@ref ConicIP.Block) diagonal matrix representing the
Nesterov-Todd scaling. The block type depends on the cone:

| Cone | Block type | Description |
|------|-----------|-------------|
| `"R"` | `Diagonal` | Diagonal scaling for nonnegative orthant |
| `"Q"` | `SymWoodbury` | Low-rank-plus-diagonal for second-order cone |
| `"S"` | `VecCongurance` | Congruence transform for semidefinite cone |

## Automatic Selection (default)

By default (`kktsolver = default_kktsolver`), ConicIP picks the solver
per problem via [`choose_kktsolver`](@ref ConicIP.choose_kktsolver),
using cone mix, size, and *structural* sparsity — never the storage
type of the inputs:

0. **Dense storage over budget → `kktsolver_ldl`.** If the dense
   solver's storage estimate ([`dense_kkt_bytes`](@ref ConicIP.dense_kkt_bytes),
   roughly `2n² + 2m(n−p) + 2(n−p)²` doubles) exceeds
   `dense_bytes_max` (default 4 GiB), the LDLᵀ solver is used whatever
   the rules below would say. At that size dense QR is an out-of-memory
   error, not a slow solve.
1. **Any SDP cone → `kktsolver_qr`.** The dense double-QR method is the
   numerically robust choice for the dense SDP scaling blocks. This routing
   sets the cost of a semidefinite solve; see
   [Semidefinite support](@ref) for the cost model and for what the
   solver does and does not handle.
2. **Small problems (`n + m + p < 200`) → `kktsolver_qr`.** Dense
   factorization wins at small sizes, and the symbolic analysis below
   would cost more than it saves.
3. **Otherwise, predicted flops decide.** A symbolic analysis of the
   quasi-definite KKT pattern gives the LDLᵀ cost `Σⱼ nnz(L₍:,ⱼ₎)²`; the
   dense cost is `m(n−p)² + (n−p)³/3`
   ([`dense_kkt_flops`](@ref ConicIP.dense_kkt_flops)). Dense QR is chosen
   when its estimate is below ten times the LDLᵀ estimate (the factor
   reflects BLAS versus scalar code), otherwise `kktsolver_ldl`. Many
   small SOCs over a 10%-dense `A` go dense; banded, block-structured, and
   [issue #10](https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/issues/10)-shaped
   problems go sparse. Nonzeros per column are no longer used: box
   constraints and identity blocks inflate that count without creating
   fill, which sent a banded LP with 13 nnz/col to a solver 100× slower.

`kktsolver_sparse` is never selected automatically any more; it remains
available by name. With `verbose = true` the solver prints the choice it
made. Passing `kktsolver` explicitly always overrides the heuristic.

## Built-in Solvers

### `kktsolver_ldl`

Sparse LDLᵀ factorization of the symmetric quasi-definite form of the KKT
system. Negating the third block row of the 3×3 system gives

```
┌ Q + δ_p I     Gᵀ        −Aᵀ           ┐
│ G            −δ_e I      0            │
└ −A            0        −(FᵀF + δ_c I) ┘
```

whose leading block is positive definite and whose trailing block is
negative definite, so it has an LDLᵀ factorization with diagonal `D`
for every symmetric permutation. The pattern is assembled once, ordered
by AMD, and analysed symbolically once; each iteration rewrites the
scaling entries and refactorizes numerically in place (QDLDL.jl, pure
Julia).

Second-order cones of dimension six or more are lifted: `FᵀF = D² + uuᵀ − vvᵀ`
([`soc_uv`](@ref ConicIP.soc_uv)) becomes a diagonal, two columns, and two
extra pivots of known sign, `3k + 2` entries instead of `k(k+1)/2`.
Smaller SOC blocks and SDP blocks are stored densely (the SDP block is
`O(k⁴)`, which is why SDP problems stay on `kktsolver_qr`).

Regularization is static (`static_reg`, default `1e-8`, on the primal
and equality blocks) plus QDLDL's dynamic pivot repair; each solve is
refined against the *unregularized* matrix (`refine_steps`, `refine_tol`),
so the perturbation acts as a preconditioner rather than a change of
problem. The refinement is bounded and keeps the best iterate: the
residual is evaluated after every correction, a correction that does
not reduce it is discarded and ends the loop, and `refine_tol` is a
target rather than a guarantee. `G` may have dependent rows: they are
absorbed by `δ_e` and the refinement rather than requiring the
preprocessor (this holds for the LDLᵀ route only; `kktsolver_qr` still
needs independent rows).

```julia
sol = conicIP(Q, c, A, b, cone_dims; kktsolver = ConicIP.kktsolver_ldl)
# or, with options:
ks = (Q, A, G, cd) -> ConicIP.kktsolver_ldl(Q, A, G, cd; static_reg = 1e-7)
sol = conicIP(Q, c, A, b, cone_dims; kktsolver = ks)
```

For repeated solves of problems with the same structure (new data, same
sparsity), [`cached_kktsolver_ldl`](@ref ConicIP.cached_kktsolver_ldl)
keeps the fill-reducing ordering between calls:

```julia
ks = ConicIP.cached_kktsolver_ldl()
for t in 1:T
    sol = conicIP(Q, c[t], A, b[t], cone_dims; kktsolver = ks)
end
```

### `kktsolver_qr`

QR-based solver using the double QR method from CVXOPT: a thin QR of
`G'` at setup, then per iteration a Cholesky factorization of the
reduced Hessian `Q₂'(Q + AᵀF⁻¹F⁻ᵀA)Q₂` on the null space of `G`.

**When to use:** SDP problems, and small or dense problems.

**Trade-offs:** Numerically robust, but the per-iteration factorization
is dense — `O((n-p)³)` — so it is the wrong choice for large sparse
problems.

### `kktsolver_sparse`

Sparse LU solver that intelligently chooses between two internal
strategies:

- **Lifted formulation:** Replaces large diagonal-plus-low-rank blocks
  with lifted variables, keeping the system sparse. Better for large
  second-order cones.
- **Dense formulation:** Converts all scaling blocks to dense matrices.
  Better when constraints are the product of many small cones.

The solver estimates the nonzero count for each strategy and picks the
sparser one automatically. (SDP blocks cannot be lifted, so any `"S"`
cone forces the dense formulation.) The sparse LU factorization reuses
its symbolic analysis across iterations once the sparsity pattern
stabilizes.

**When to use:** Large problems with sparse `Q` and `A`.

```julia
sol = conicIP(Q, c, A, b, cone_dims; kktsolver=kktsolver_sparse)
```

### `kktsolver_2x2` (with `pivot`)

A 2×2 sparse LU solver that works on the Schur complement system obtained
by pivoting on the third block. Must be wrapped with [`pivot`](@ref ConicIP.pivot):

```julia
sol = conicIP(Q, c, A, b, cone_dims; kktsolver=pivot(kktsolver_2x2))
```

**When to use:** Problems where the Schur complement `Q + Aᵀ(FᵀF)⁻¹A`
is sparser or better conditioned than the full 3×3 system.

## Choosing a Solver Manually

The automatic default covers the common cases. Override it when you
know better:

| Problem characteristics | Recommended solver |
|------------------------|-------------------|
| Small/medium, any structure | `kktsolver_qr` (auto picks this) |
| Any SDP cone | `kktsolver_qr` (auto picks this) |
| Large, sparse Q and A | `kktsolver_sparse` (auto picks this) |
| Large, few large SOC cones | `kktsolver_sparse` (uses lifted form) |
| Structured/sparse Schur complement | `pivot(kktsolver_2x2)` |
| Custom problem structure | Write a custom solver (see below) |

If a factorization breaks down (e.g. rank-deficient `G` passed directly
to [`conicIP`](@ref)), the solver returns a `Solution` with
`status == :Error` and the failure reason in `sol.message` rather than
throwing; [`preprocess_conicIP`](@ref) removes redundant rows up front,
and structurally degenerate inputs (an all-zero equality row, a variable
appearing in no constraint) are detected exactly in `O(nnz)` and either
deflated or answered with a certified `:Infeasible`/`:DualInfeasible`.

## Writing a Custom Solver

A custom KKT solver is a three-level nested function. Note that with the
default `equilibrate = true` the solver receives the *Ruiz-scaled* data
(see [`equilibrate_conicIP`](@ref ConicIP.equilibrate_conicIP)); a solver
that exploits known structure must derive it from the `Q`, `A`, `G` it is
handed, not from the caller's original matrices.

```julia
function my_kktsolver(Q, A, G, cone_dims)
    # Level 1: One-time setup (symbolic factorization, preallocation)
    # Called once before the solve loop.

    function solve3x3gen(F, F⁻ᵀ)
        # Level 2: Per-iteration setup (F changes each iteration)
        # Compute numeric factorization using current scaling F.

        function solve3x3(x, y, z)
            # Level 3: Solve the 3×3 system given RHS (x, y, z).
            # Return (a, b, c) where:
            #   Qa + G'b - A'c = x
            #   Ga = y
            #   Aa + FᵀFc = z
            return (a, b, c)
        end
        return solve3x3
    end
    return solve3x3gen
end
```

Pass it to the solver via the `kktsolver` keyword:

```julia
sol = conicIP(Q, c, A, b, cone_dims; kktsolver=my_kktsolver)
```

### The contract

What `conicIP` guarantees to a custom solver, and what it expects back:

- **Data.** `Q`, `A`, `G` are whatever `conicIP` was given, after
  equilibration (dense stays dense, sparse stays sparse; `Q` is
  symmetric). Level 1 runs once per solve, before the initial point.
- **Signs.** Level 3 solves exactly the system in the skeleton above:
  `−Aᵀ` in the first block row and `+A`, `+FᵀF` in the third. A solver
  that factors the symmetric quasi-definite form (third block row
  negated: `−A`, `−FᵀF`) must negate the third *right-hand-side* block
  before its back-solve and return the third *solution* block `c`
  unnegated — the unknown is the same `c` in both forms. This is what
  `kktsolver_ldl` does: `rhs[oz+1:oz+m] .= .-bz`, and `sol[oz+1:oz+m]`
  is returned as is.
- **Scaling blocks.** `F` is a `Block` whose elements are `Diagonal`
  (`"R"` cones, and *every* cone at the identity-scaled initial point),
  `SymWoodbury` (`"Q"`), or `VecCongurance` (`"S"`); `F⁻ᵀ` is its
  inverse adjoint. `FᵀF` is symmetric positive definite; `F` itself is
  not self-adjoint for `"S"` blocks. Level 2 runs once per iteration
  (plus once for the initial point) and may keep any state across
  iterations; the sparsity pattern of `FᵀF` is fixed after the initial
  point.
- **Calls per iteration.** Level 3 is called for the predictor, the
  corrector, and up to `maxRefinementSteps` refinements of each: between
  two and `2 + 2·maxRefinementSteps` times per factorization.
- **Ownership of the result.** Return fresh arrays. `conicIP` builds
  its step from them and updates it in place during refinement; a view
  into a buffer that the next call overwrites corrupts the step.
- **Accuracy.** No tolerance is passed down. `conicIP` measures the 4×4
  residual of every step and refines it towards `refineAbsTol +
  refineRelTol·‖r‖`: at most `maxRefinementSteps` corrections, each one
  more level-3 call, with the residual re-evaluated after every
  correction; a correction that does not reduce the residual is undone
  and ends the refinement, so the step used is the best one seen and
  never worse than the unrefined solve. The tolerance is a target, not
  a guarantee — a step that still misses it is used as is, and the
  solve is not aborted for it. A solver may therefore regularize its
  factorization (`kktsolver_ldl` does) as long as each solve is
  reasonably accurate for the *unregularized* system; the refinement
  recovers the rest only while it contracts. A solver that returns a
  poor solve every time shows as `maxRefinementSteps` refinements per
  step in the verbose `refine` column and a red row.
- **Failure.** Throw one of `ConicIP.KKT_FAILURES`
  (`SingularException`, `PosDefException`, `LAPACKException`,
  `ZeroPivotException`) from level 2 or level 3 to report a
  factorization failure; `conicIP` returns `status = :Error` with the
  stage in `sol.message`. Any other exception propagates as a bug.
  Return non-finite values and `conicIP` reports
  `"non-finite ... direction"` the same way.

### Example: Diagonal QP

For `minimize ½ xᵀQx - cᵀx subject to x ≥ 0` with diagonal `Q`,
the KKT system simplifies (no equality constraints, `G` is empty) to:

```
┌            ┐ ┌   ┐   ┌   ┐
│ Q       -I │ │ a │   │ x │
│ I    FᵀF   │ │ c │ = │ z │
└            ┘ └   ┘   └   ┘
```

Since `F` is `Diagonal` for `"R"` cones, pivoting on the second block
gives `(Q + (FᵀF)⁻¹) a = x + (FᵀF)⁻¹ z`, solvable by Cholesky:

```@example custom_kkt
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

n = 50
Q = sprandn(n, n, 0.3); Q = Q'Q + 0.1I  # make positive definite
c = ones(n)
A = sparse(1.0I, n, n)
b = zeros(n)
cone_dims = [("R", n)]

function my_kktsolver(Q, A, G, cone_dims)
    function solve3x3gen(F, F⁻ᵀ)
        invFᵀF = inv(F'F)
        QpD = cholesky(Q + spdiagm(0 => (F[1].diag).^(-2)))

        function solve3x3(x, y, z)
            a = QpD \ (x + A' * (invFᵀF * z))
            c = invFᵀF * (z - A * a)
            b = zeros(0)
            return (a, b, c)
        end
    end
end

sol = conicIP(Q, c, A, b, cone_dims; kktsolver=my_kktsolver, verbose=false)
sol.status
```

## The `pivot` Wrapper

The pattern of reducing a 3×3 system to 2×2 by pivoting on the third block
is common enough that ConicIP provides [`pivot`](@ref ConicIP.pivot) to automate it.

A 2×2 solver has the signature:

```julia
function my_2x2_solver(Q, A, G, cone_dims)
    function solve2x2gen(F, F⁻ᵀ)
        # Build and factor the Schur complement: Q + Aᵀ(FᵀF)⁻¹A
        function solve2x2(y, w)
            # Solve for (Δy, Δw) and return them
            return (Δy, Δw)
        end
        return solve2x2
    end
    return solve2x2gen
end
```

Then `pivot(my_2x2_solver)` produces a valid 3×3 solver. Here's the same
diagonal QP using `pivot`:

```@example custom_kkt
function my_2x2_solver(Q, A, G, cone_dims)
    function solve2x2gen(F, F⁻ᵀ)
        QpD = cholesky(Q + spdiagm(0 => (F[1].diag).^(-2)))
        return (y, w) -> (QpD \ y, zeros(0))
    end
end

sol2 = conicIP(Q, c, A, b, cone_dims;
               kktsolver=pivot(my_2x2_solver), verbose=false)
sol2.status
```

## Performance Tips

1. **Preallocate buffers** in level 1 (the outer function) and reuse them
   in levels 2 and 3.
2. **Reuse symbolic factorizations** when the sparsity pattern doesn't
   change between iterations (only the numeric values of `F` change).
3. **Avoid `inv(F'F)` for large blocks** — compute the action of
   `(FᵀF)⁻¹` on a vector instead.
4. For problems with a `callback.ipynb` example, see the `examples/`
   directory in the repository.
