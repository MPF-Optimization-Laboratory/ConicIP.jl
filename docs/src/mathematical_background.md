# Mathematical Background

## Primal Problem

ConicIP solves the conic optimization problem

```
minimize    ½ yᵀQy - cᵀy
subject to  Ay - b ∈ K
            Gy = d
```

where `K` is a Cartesian product of cones and `Q` is positive semidefinite.

## Supported Cones

**Nonnegative orthant** (`"R"`):
```
R₊ⁿ = { x ∈ Rⁿ : xᵢ ≥ 0 }
```

**Second-order cone** (`"Q"`):
```
Qⁿ = { (t, x) ∈ R × Rⁿ⁻¹ : ‖x‖₂ ≤ t }
```

**Positive semidefinite cone** (`"S"`, experimental):
```
Sⁿ₊ = { X ∈ Sⁿ : X ≽ 0 }
```

Matrices are stored in vectorized form using [`vecm`](@ref ConicIP.vecm),
which scales off-diagonal entries by `√2` to preserve inner products.

## Interior-Point Method

ConicIP implements a homogeneous self-dual interior-point method based on
the approach described by Andersen, Dahl, and Vandenberghe (2003). The
method solves the primal and dual problems simultaneously and can detect
infeasibility and unboundedness without a separate Phase I.

### Nesterov-Todd Scaling

At each iteration, the algorithm computes the Nesterov-Todd scaling point
`w` such that the scaling operator `F` satisfies

```
F z = F⁻¹ s = λ
```

where `z` and `s` are the current primal and dual slack variables, and
`λ` is the scaled point. This symmetric scaling ensures equal treatment
of primal and dual and leads to better numerical behavior than
primal-only or dual-only scaling.

The scaling matrix type depends on the cone:

| Cone | Scaling type |
|------|-------------|
| Nonnegative orthant | Diagonal matrix: `F = Diagonal(√(s./z))` |
| Second-order cone | Rank-2 update of a diagonal (`SymWoodbury`) |
| Semidefinite cone | Congruence transform (`VecCongurance`) |

### Predictor-Corrector Steps

Each iteration consists of two phases:

1. **Predictor (affine) step:** Solve the linearized KKT system with the
   current residuals to compute an affine scaling direction. This step
   estimates how much the complementarity gap can be reduced.

2. **Corrector (combined) step:** Solve a modified system that includes
   a centering term `σμe` and a second-order correction from the
   predictor step. The centering parameter `σ` is chosen adaptively
   based on the predictor step length.

### Convergence Criteria

The solver monitors three residuals:

- **Primal feasibility** (`prFeas`): `‖Ay - s - b‖ / (1 + ‖b‖)`
- **Dual feasibility** (`duFeas`): `‖Qy + Gᵀw - Aᵀv - c‖ / (1 + ‖c‖)`
- **Complementarity** (`muFeas`): `sᵀv / (1 + |cᵀy|)`

The solver terminates with status `:Optimal` when all three residuals
fall below the tolerance `optTol`.

### Infeasibility Detection

**Primal infeasibility** (`:Infeasible`): The solver detects a certificate
`(w, v)` satisfying `Gᵀw + Aᵀv ≈ 0` and `bᵀv + dᵀw < 0`, proving that
the primal problem has no feasible point.

**Dual infeasibility** (`:Unbounded`): The solver detects a certificate `y`
satisfying `Ay ≥ 0`, `Gy ≈ 0`, `Qy ≈ 0`, and `cᵀy > 0`, proving that
the dual problem has no feasible point (equivalently, the primal is
unbounded).

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
