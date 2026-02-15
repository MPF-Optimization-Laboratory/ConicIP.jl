# Custom KKT Solvers

The interior-point solver can be sped up significantly by exploiting the
structure of `Q`, `A`, and `G`. ConicIP supports custom KKT solver callbacks
that replace the default factorization at each iteration.

## The KKT System

At each interior-point iteration, ConicIP solves a 3×3 block system:

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

## The `kktsolver` Interface

A custom KKT solver is a function with the signature:

```julia
function my_kktsolver(Q, A, G, cone_dims)
    # One-time setup (problem structure analysis, symbolic factorization, etc.)

    function solve3x3gen(F, F⁻ᵀ)
        # Per-iteration setup (F changes each iteration)

        function solve3x3(x, y, z)
            # Solve the 3×3 system and return (a, b, c)
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

## Example: Diagonal QP

For the problem `minimize ½ xᵀQx - cᵀx subject to x ≥ 0` with a sparse `Q`,
the KKT system simplifies (no equality constraints, so `G` is empty) to:

```
┌            ┐ ┌   ┐   ┌   ┐
│ Q       I  │ │ a │   │ x │
│ I    -FᵀF  │ │ c │ = │ z │
└            ┘ └   ┘   └   ┘
```

Since `F` is `Diagonal` for `"R"` cones, `(FᵀF)⁻¹` is diagonal. Pivoting on
the second block gives `(Q + (FᵀF)⁻¹) a = x + (FᵀF)⁻¹ z`, which is a
diagonal perturbation of `Q` solvable by Cholesky factorization:

```@example custom_kkt
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

n = 50
Q = sprandn(n, n, 0.3); Q = Q'Q + 0.1I  # make positive definite
c = ones(n, 1)
A = sparse(1.0I, n, n)
b = zeros(n, 1)
cone_dims = [("R", n)]

function my_kktsolver(Q, A, G, cone_dims)
    function solve3x3gen(F, F⁻ᵀ)
        invFᵀF = inv(F'F)
        QpD = cholesky(Q + spdiagm(0 => (F[1].diag).^(-2)))

        function solve3x3(x, y, z)
            a = QpD \ (x + A' * (invFᵀF * z))
            c = invFᵀF * (z - A * a)
            b = zeros(0, 1)
            return (a, b, c)
        end
    end
end

sol = conicIP(Q, c, A, b, cone_dims; kktsolver=my_kktsolver, verbose=false)
sol.status
```

## The `pivot` Wrapper

The pattern of reducing a 3×3 system to a 2×2 system by pivoting on
the third block is common. The [`pivot`](@ref ConicIP.pivot) function
automates this: it wraps a 2×2 solver into a 3×3 solver.

A 2×2 solver has the signature:

```julia
function my_2x2_solver(Q, A, G, cone_dims)
    function solve2x2gen(F, F⁻ᵀ)
        # Build the Schur complement Q + Aᵀ(FᵀF)⁻¹A
        function solve2x2(y, w)
            return (Δy, Δw)
        end
        return solve2x2
    end
    return solve2x2gen
end
```

Then `pivot(my_2x2_solver)` produces a valid 3×3 solver.

Here is the same diagonal QP example using `pivot`:

```@example custom_kkt
function my_2x2_solver(Q, A, G, cone_dims)
    function solve2x2gen(F, F⁻ᵀ)
        QpD = cholesky(Q + spdiagm(0 => (F[1].diag).^(-2)))
        return (y, w) -> (QpD \ y, zeros(0, 1))
    end
end

sol2 = conicIP(Q, c, A, b, cone_dims;
               kktsolver=pivot(my_2x2_solver), verbose=false)
sol2.status
```

## Built-in Solvers

ConicIP provides three built-in KKT solvers:

- [`kktsolver_qr`](@ref ConicIP.kktsolver_qr) -- QR-based solver (default),
  works for all problem types
- [`kktsolver_sparse`](@ref ConicIP.kktsolver_sparse) -- sparse LU solver,
  automatically chooses between lifted and dense formulations
- [`kktsolver_2x2`](@ref ConicIP.kktsolver_2x2) -- 2×2 sparse LU solver
  (use with [`pivot`](@ref ConicIP.pivot))
