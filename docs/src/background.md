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

ConicIP implements an **infeasible-start primal-dual interior-point method**
with Mehrotra predictor-corrector steps and Nesterov-Todd scaling. The
iterates are not required to be feasible: the method starts from a point
strictly inside the cone but generally violating `Ay - s = b` and `Gy = d`,
and drives the primal residual, dual residual, and complementarity gap to
zero together. There is no Phase I, and no homogeneous self-dual embedding —
the iterates carry no `τ`/`κ` variables.

The KKT and scaling machinery follows the cone program solver in CVXOPT
(Andersen, Dahl, and Vandenberghe), which is the closest algorithmic
reference for the linear algebra at each iteration.

Because the embedding is absent, infeasibility and unboundedness are not
read off a single variable. They are detected *post hoc*: when the problem
has no solution, the iterates diverge along a ray, and that ray — once it
is validated against the original problem data — is the certificate. See
[Convergence Criteria](@ref) below.

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

### The Certificate Pipeline

Detecting infeasibility or unboundedness is a second, independent test,
run in two stages so that the expensive part is paid only when it is likely
to succeed.

**Stage 1 — cheap per-iteration screens.** At every iteration the solver
forms the scaled Farkas residuals used by CVXOPT and ECOS. For primal
infeasibility these measure `‖Gᵀw - Aᵀv‖` relative to `‖w‖ + ‖v‖` (CVXOPT
style) and relative to `|dᵀw - bᵀv|` (ECOS style); the analogous screens for
unboundedness measure `‖Ay - s‖`, `‖Gy‖`, and `‖Qy‖` against `cᵀy`. These
are functions of the *current, scaled* iterate, so they are cheap but not
conclusive. A screen falling below `infeasTol` (default `1e-7`, decoupled
from `optTol`) only nominates the current iterate as a *candidate ray*.

**Stage 2 — validation against the original data.** A candidate is then
checked by `ConicIP.validate_infeasibility_certificate` or
`ConicIP.validate_unboundedness_certificate`. These are pure functions of the
problem data as the user supplied it — they never consult solver state,
scalings, or preprocessed copies. Each performs three checks:

1. **Normalization.** The separation value must be strictly positive, and
   the ray is rescaled by it: `dᵀw̄ - bᵀv̄ = -1` for infeasibility, `cᵀȳ = +1`
   for unboundedness.
2. **Farkas residual.** `Gᵀw̄ - Aᵀv̄ ≈ 0` for infeasibility; `Qȳ ≈ 0` and
   `Gȳ ≈ 0` for unboundedness, measured in the `∞`-norm.
3. **Cone membership.** The blockwise margin of the ray (`v̄` for
   infeasibility, `Aȳ` for unboundedness) must be nonnegative: the minimum
   component for an `"R"` block, `x₁ - ‖x₂:ₙ‖` for a `"Q"` block, and the
   minimum eigenvalue for an `"S"` block.

Checks 2 and 3 are accepted within `infeasAbsTol + infeasTol·(1 + ‖ray‖)`,
so `infeasAbsTol` (default `1e-9`) sets the floor and `infeasTol` the
relative slack. Only a candidate that passes all three is reported as a
certificate.

## Troubleshooting Solver Output

### Status: `:Optimal`

All convergence criteria met. Check `sol.prFeas`, `sol.duFeas`, and
`sol.muFeas` to confirm the solution quality. Values below `1e-8` indicate
a high-accuracy solution.

### Status: `:Infeasible`

No feasible point exists. The status is claimed only after a candidate ray
has passed validation against the original problem data, so it is not an
inference from a diverging iterate alone.

When `sol.has_certificate` is `true`, `sol.w` and `sol.v` hold the verified
Farkas ray, normalized so that

```
dᵀw̄ - bᵀv̄ = -1,    Gᵀw̄ - Aᵀv̄ ≈ 0,    v̄ ∈ K
```

and the remaining fields are `NaN`. Any nonnegative combination of the
constraints therefore yields the contradiction `0 ≤ -1`, which you can
check yourself from the returned vectors.

When `sol.has_certificate` is `false`, the problem was still established as
infeasible, but no usable ray is returned (all of `y`, `w`, `v`, `s` are
`NaN`). This happens when infeasibility is settled before the ray exists —
for example by the consistency checks in
[`preprocess_conicIP`](@ref ConicIP.preprocess_conicIP), which detect an
inconsistent equality system directly — or when the residual construction
that would produce the ray breaks down numerically.

Common causes:
- Contradictory constraints (e.g., `x ≥ 1` and `x ≤ 0`)
- Overly tight bounds combined with equality constraints

**What to try:** Relax constraints or check problem data for errors.

### Status: `:Unbounded`

The objective decreases without bound over the feasible set. As with
`:Infeasible`, the status is claimed only after validation against the
original data.

When `sol.has_certificate` is `true`, `sol.y` holds the verified recession
ray, normalized so that

```
cᵀȳ = +1,    Qȳ ≈ 0,    Gȳ ≈ 0,    Aȳ ∈ K
```

with `sol.s = A*ȳ`; `w` and `v` are `NaN`. Moving from any feasible point
along `ȳ` stays feasible and decreases the objective at unit rate.

When `sol.has_certificate` is `false`, unboundedness was detected but no
usable ray is returned.

Common causes:
- Missing constraints that should bound the feasible region
- `Q = 0` (LP) with an unbounded feasible direction

**What to try:** Add bounding constraints or verify the objective.

### Status: `:AlmostInfeasible` / `:AlmostUnbounded`

The iteration limit was reached with a candidate ray that validates only
when the tolerances are relaxed by a factor of 100. This is the gray zone
between a stall and a proof: the evidence points at infeasibility (or
unboundedness), but not strongly enough to assert it.

No certificate is returned — `has_certificate` is `false` and the solution
fields hold the best iterate, not a ray. Treat the result as advisory.

**What to try:**
- Increase `maxIters`; the candidate may sharpen into a real certificate
- Loosen `infeasTol` if you are confident the problem is infeasible
- Tighten `infeasTol` if you believe the problem is feasible and the
  detection is spurious
- Rescale the problem data, which is the usual cause of a ray that will
  not quite validate

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
| `infeasTol` | `1e-7` | Relative tolerance for certificate screening and validation |
| `infeasAbsTol` | `1e-9` | Absolute floor for certificate validation |
| `staticReg` | `0` | Static regularization of the KKT factorization |
| `certFallback` | `true` | Attempt an auxiliary certificate solve on stall |

**`optTol`:** Decrease for higher accuracy (e.g., `1e-8`); increase if the
solver stalls (e.g., `1e-5`). Tighter tolerances require more iterations.

**`DTB`:** Controls how close the step can get to the cone boundary. Smaller
values (e.g., `0.001`) are more conservative but more stable; larger values
(e.g., `0.1`) are more aggressive and may converge faster or oscillate.

**`maxIters`:** Increase if the solver reports `:Abandoned` after reaching
the iteration limit. Most well-conditioned problems converge in 20–50
iterations.

**`infeasTol`:** Governs certificate detection only, and is deliberately
decoupled from `optTol` — changing the accuracy you demand of an optimal
solution should not change how readily the solver declares a problem
infeasible. Decrease it if the solver reports `:Infeasible` or `:Unbounded`
for a problem you know has a solution; increase it if a genuinely infeasible
problem is reported as `:Abandoned` or `:AlmostInfeasible`.

**`infeasAbsTol`:** The absolute term in the validation tolerance
`infeasAbsTol + infeasTol·(1 + ‖ray‖)`. It matters only for rays whose norm
is small, where the relative term alone would be too strict. Rarely needs
adjustment.

**`staticReg`:** Adds `δI` with `δ = staticReg·(1 + ‖Q‖∞)` to the matrix
that is factorized, so a rank-deficient `[Q Aᵀ Gᵀ]` still yields a usable
factorization. The perturbation is confined to the factorization: residuals,
objective values, and iterative refinement all use the true `Q`, so the
regularized factorization acts as a preconditioner whose error the
refinement loop removes. The default `0` disables it;
[`preprocess_conicIP`](@ref ConicIP.preprocess_conicIP) enables it
automatically when it detects rank deficiency, so setting it by hand is
usually unnecessary.

**`certFallback`:** When the solver stalls or hits the iteration limit
without deciding the problem, it solves an auxiliary strongly convex problem
whose solution is a certificate candidate, then puts that candidate through
the same validation as any other. The cost is one extra solve on problems
that would otherwise return `:Abandoned`; set it to `false` if you would
rather have the stall reported immediately.

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
