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
| `:Infeasible` | Problem is primal infeasible (dual certificate found) |
| `:Unbounded` | Problem is dual infeasible / primal unbounded |
| `:Abandoned` | Solver stalled (step size too small or numerical issues) |
| `:Error` | Solver encountered an error |

See [Troubleshooting Solver Output](@ref) in the Mathematical Background
for guidance on non-optimal statuses.

## JuMP / MathOptInterface

```@docs
ConicIP.Optimizer
```

## KKT Solver Functions

Three built-in KKT solvers are provided. See the [KKT Solvers](@ref) guide
for detailed usage and custom solver development.

```@docs
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
ConicIP.vecm
ConicIP.imcols
```

## Internal

These functions are implementation details and not part of the public API.

```@docs
ConicIP.pivotgen
ConicIP.placeholder
ConicIP.identical_sparse_structure
ConicIP.count_lift
ConicIP.count_dense
```
