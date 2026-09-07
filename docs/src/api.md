# API Reference

## Solver

The main entry point for solving conic optimization problems.

```@docs
ConicIP.conicIP
ConicIP.preprocess_conicIP
```

## Solution

The solver returns a `Solution` struct containing primal/dual variables,
status, and convergence information.

```@docs
ConicIP.Solution
```

**Key fields:**

| Field | Type | Description |
|-------|------|-------------|
| `y` | `Matrix` | Primal variables |
| `w` | `Matrix` | Dual variables for equality constraints (`Gy = d`) |
| `v` | `Matrix` | Dual variables for inequality constraints (`Ay ≥ b`) |
| `status` | `Symbol` | Termination status (see below) |
| `pobj` | `Real` | Primal objective value |
| `dobj` | `Real` | Dual objective value |
| `prFeas` | `Real` | Primal feasibility residual |
| `duFeas` | `Real` | Dual feasibility residual |
| `muFeas` | `Real` | Complementarity residual |
| `Iter` | `Integer` | Number of iterations |
| `Mu` | `Real` | Final barrier parameter |

**Status values:**

| Status | Meaning |
|--------|---------|
| `:Optimal` | Converged to an optimal solution |
| `:Infeasible` | Problem is primal infeasible (validated Farkas ray when `has_certificate`) |
| `:DualInfeasible` | Problem is dual infeasible: a recession ray decreases the objective without bound (validated when `has_certificate`). The primal is unbounded if it is also feasible, which this status does not establish. |
| `:AlmostInfeasible` | Iteration limit with a near-validating infeasibility candidate (no certificate) |
| `:AlmostDualInfeasible` | Iteration limit with a near-validating recession-ray candidate (no certificate) |
| `:Abandoned` | Solver stalled (step size too small or numerical issues) |
| `:TimeLimit` | `timeLimit` seconds elapsed; the solution holds the best iterate so far |
| `:Error` | Solver encountered an error |

See [Troubleshooting Solver Output](@ref) in the Mathematical Background
for guidance on non-optimal statuses.

## Certificate Validation

Infeasibility and unboundedness claims are backed by rays validated against
the original problem data. See
[The Certificate Pipeline](@ref) in the Mathematical Background.

```@docs
ConicIP.CertificateCheck
ConicIP.cone_margin
ConicIP.validate_infeasibility_certificate
ConicIP.validate_unboundedness_certificate
```

When the iterate loop exhausts with evidence of a ray, the solver can
recover a certificate by solving an auxiliary min-norm QP:

```@docs
ConicIP.fallback_infeasibility_ray
ConicIP.fallback_unbounded_ray
```

## JuMP / MathOptInterface

```@docs
ConicIP.Optimizer
```

## KKT Solver Functions

Three built-in KKT solvers are provided, and the default picks among
them automatically per problem. See the [KKT Solvers](@ref) guide
for detailed usage and custom solver development.

```@docs
ConicIP.equilibrate_conicIP
ConicIP.default_kktsolver
ConicIP.choose_kktsolver
ConicIP.dense_kkt_bytes
ConicIP.dense_kkt_flops
ConicIP.kktsolver_ldl
ConicIP.soc_uv
ConicIP.kktsolver_qr
ConicIP.kktsolver_sparse
ConicIP.kktsolver_2x2
ConicIP.pivot
```

## Block Diagonal Matrices

The Nesterov-Todd scaling matrix is represented as a block diagonal matrix
where each block corresponds to a cone in the cone specification.

```@docs
ConicIP.Block
ConicIP.block_idx
ConicIP.broadcastf
```

## Utilities

```@docs
ConicIP.Id
ConicIP.VecCongurance
ConicIP.mat
ConicIP.mat!
ConicIP.vecm
ConicIP.vecm!
ConicIP.imcols
```

## Internal

These functions are implementation details and not part of the public API.

```@docs
ConicIP.inv_adjoint!
ConicIP.pivotgen
ConicIP.placeholder
ConicIP.identical_sparse_structure
ConicIP.count_lift
ConicIP.count_dense
ConicIP._psd_moi_vecm_info
ConicIP._psd_vecm_to_moi
ConicIP._psd_scale_input!
```
