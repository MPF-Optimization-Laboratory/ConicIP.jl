# Mathematical Background

This page provides the mathematical context needed to understand ConicIP's
behavior, interpret its output, and tune its parameters. For full derivations,
see the references at the end.

## Primal Problem

ConicIP solves the conic optimization problem

```
minimize    ½ yᵀQy - cᵀy
subject to  Ay - b ∈ K
            Gy = d
```

where `K` is a Cartesian product of cones and `Q` is positive semidefinite.

## Supported Cones

**Nonnegative orthant** (`"R"`): the set of vectors with all nonneg entries,
`R₊ⁿ = { x ∈ Rⁿ : xᵢ ≥ 0 }`.

**Second-order cone** (`"Q"`): also called the Lorentz cone,
`Qⁿ = { (t, x) ∈ R × Rⁿ⁻¹ : ‖x‖₂ ≤ t }`.

**Positive semidefinite cone** (`"S"`, experimental):
`Sⁿ₊ = { X ∈ Sⁿ : X ≽ 0 }`.
Matrices are stored in vectorized form using [`vecm`](@ref ConicIP.vecm),
which scales off-diagonal entries by `√2` to preserve inner products.

## Interior-Point Method

ConicIP implements a homogeneous self-dual interior-point method based on
the approach described by Andersen, Dahl, and Vandenberghe (2003). The
method solves the primal and dual problems simultaneously and can detect
infeasibility and unboundedness without a separate Phase I.

### Nesterov-Todd Scaling

At each iteration, the algorithm computes the Nesterov-Todd scaling point
such that the scaling operator `F` satisfies `F z = F⁻¹ s = λ`, where
`z` and `s` are the current primal and dual slack variables.

The scaling matrix type depends on the cone:

| Cone | Scaling type |
|------|-------------|
| Nonneg orthant | `Diagonal` — `F = Diagonal(√(s./z))` |
| Second-order cone | `SymWoodbury` — rank-2 update of a diagonal |
| Semidefinite cone | `VecCongurance` — congruence transform |

### Predictor-Corrector Steps

Each iteration consists of two phases:

1. **Predictor (affine) step:** Solve the KKT system with current residuals
   to estimate how much the complementarity gap can be reduced.

2. **Corrector (combined) step:** Solve a modified system that includes
   a centering term `σμe` and a second-order correction. The centering
   parameter `σ` is chosen adaptively based on the predictor step length.

## Convergence Criteria

The solver monitors three residuals:

- **Primal feasibility** (`prFeas`): `‖Ay - s - b‖ / (1 + ‖b‖)`
- **Dual feasibility** (`duFeas`): `‖Qy + Gᵀw - Aᵀv - c‖ / (1 + ‖c‖)`
- **Complementarity** (`muFeas`): `sᵀv / (1 + |cᵀy|)`

The solver terminates with status `:Optimal` when all three residuals
fall below the tolerance `optTol` (default: `1e-6`).

## Troubleshooting Solver Output

### Status: `:Optimal`

All convergence criteria met. Check `sol.prFeas`, `sol.duFeas`, and
`sol.muFeas` to confirm the solution quality. Values below `1e-8` indicate
a high-accuracy solution.

### Status: `:Infeasible`

The solver found a certificate `(w, v)` proving no feasible point exists.
Common causes:
- Contradictory constraints (e.g., `x ≥ 1` and `x ≤ 0`)
- Overly tight bounds combined with equality constraints

**What to try:** Relax constraints or check problem data for errors.

### Status: `:Unbounded`

The solver found a ray along which the objective decreases without bound.
Common causes:
- Missing constraints that should bound the feasible region
- `Q = 0` (LP) with an unbounded feasible direction

**What to try:** Add bounding constraints or verify the objective.

### Status: `:Abandoned`

The solver stalled — step sizes became too small to make progress.
Common causes:
- Near-degenerate problem (constraints nearly dependent)
- Poor numerical conditioning of `Q` or `A`
- Tolerance `optTol` set too tight for the problem's condition number

**What to try:**
- Use [`preprocess_conicIP`](@ref ConicIP.preprocess_conicIP) to remove redundant constraints
- Loosen `optTol` (e.g., `1e-5` instead of `1e-8`)
- Scale the problem data so that entries are of moderate magnitude

### Status: `:Error`

An unexpected error occurred (e.g., singular factorization). This usually
indicates a problem with the input data.

**What to try:** Check that `Q` is positive semidefinite and `A` has full row rank.

## Reading Residuals

The three residuals in the `Solution` struct measure different aspects
of solution quality:

| Residual | Measures | Good value |
|----------|----------|------------|
| `prFeas` | Constraint satisfaction | `< optTol` |
| `duFeas` | KKT stationarity | `< optTol` |
| `muFeas` | Complementary slackness | `< optTol` |

If `prFeas` is large but `duFeas` is small, the solver found a nearly
optimal point that doesn't quite satisfy the constraints — try scaling.

If `muFeas` is large but the others are small, the solver found a feasible
point but the duality gap hasn't closed — try more iterations (`maxIters`).

## Parameter Tuning

| Parameter | Default | Effect |
|-----------|---------|--------|
| `optTol` | `1e-6` | Convergence tolerance for all three residuals |
| `maxIters` | `100` | Maximum interior-point iterations |
| `DTB` | `0.01` | Distance-to-boundary parameter; controls step conservatism |
| `maxRefinementSteps` | `3` | Iterative refinement steps for KKT solve |
| `infeasTol` | `optTol` | Threshold for infeasibility detection |

**`optTol`:** Decrease for higher accuracy (e.g., `1e-8`); increase if the
solver stalls (e.g., `1e-5`). Tighter tolerances require more iterations.

**`DTB`:** Controls how close the step can get to the cone boundary. Smaller
values (e.g., `0.001`) are more conservative but more stable; larger values
(e.g., `0.1`) are more aggressive and may converge faster or oscillate.

**`maxIters`:** Increase if the solver reports `:Abandoned` after reaching
the iteration limit. Most well-conditioned problems converge in 20–50
iterations.

## References

- E.D. Andersen, C. Roos, and T. Terlaky. "On implementing a primal-dual
  interior-point method for conic quadratic optimization."
  *Mathematical Programming*, 95(2):249-277, 2003.
- Y.E. Nesterov and M.J. Todd. "Self-scaled barriers and interior-point
  methods for convex programming." *Mathematics of Operations Research*,
  22(1):1-42, 1997.
- L. Vandenberghe and S. Boyd. "Semidefinite programming."
  *SIAM Review*, 38(1):49-95, 1996.
- M.S. Andersen, J. Dahl, and L. Vandenberghe. "CVXOPT: A Python package
  for convex optimization." Available at https://cvxopt.org.
