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

## Built-in Solvers

### `kktsolver_qr` (default)

QR-based solver using the double QR method from CVXOPT. This is the
default and works reliably for all problem types.

**When to use:** General-purpose; a safe default for any problem.

**Trade-offs:** More numerically robust than sparse LU, but slower for
large sparse problems.

### `kktsolver_sparse`

Sparse LU solver that intelligently chooses between two internal
strategies:

- **Lifted formulation:** Replaces large diagonal-plus-low-rank blocks
  with lifted variables, keeping the system sparse. Better for large
  second-order cones.
- **Dense formulation:** Converts all scaling blocks to dense matrices.
  Better when constraints are the product of many small cones.

The solver estimates the nonzero count for each strategy and picks the
sparser one automatically.

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

## Choosing a Solver

| Problem characteristics | Recommended solver |
|------------------------|-------------------|
| Small/medium, any structure | `kktsolver_qr` (default) |
| Large, sparse Q and A | `kktsolver_sparse` |
| Large, few large SOC cones | `kktsolver_sparse` (uses lifted form) |
| Large, many small cones | `kktsolver_sparse` (uses dense form) |
| Structured Schur complement | `pivot(kktsolver_2x2)` |
| Custom problem structure | Write a custom solver (see below) |

## Writing a Custom Solver

A custom KKT solver is a three-level nested function:

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
