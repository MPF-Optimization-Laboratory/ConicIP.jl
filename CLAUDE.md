# ConicIP.jl

Pure-Julia conic interior-point solver for quadratic programs with linear, second-order cone, and semidefinite constraints.

## Architecture

- `src/ConicIP.jl` — Main module, core `conicIP()` solver, type definitions (`v4x1`, `VecCongurance`, `Solution`)
- `src/blockmatrices.jl` — Block diagonal matrix type (`Block`)
- `src/kktsolvers.jl` — KKT system solvers (`kktsolver_qr`, `kktsolver_sparse`, `kktsolver_2x2`, `pivot`)
- `src/preprocessor.jl` — Redundancy removal (`imcols`, `preprocess_conicIP`)
- `src/MOI_wrapper.jl` — MathOptInterface wrapper (`Optimizer`)

## Key Design Patterns

- The solver works with `Matrix` (n×1) columns rather than `Vector` for `c`, `b`, `d`
- Cone dimensions specified as `[("R", n), ("Q", m), ("S", k)]` tuples
- Block diagonal matrices (`Block`) hold per-cone scaling matrices (Diagonal, SymWoodbury, VecCongurance)
- Nesterov-Todd scaling computed per cone block
- `Id(n)` creates n×n identity as `Diagonal(ones(n))`

## Running Tests

```
julia --project -e 'using Pkg; Pkg.test()'
```
