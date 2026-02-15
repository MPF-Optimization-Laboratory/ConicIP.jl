# API Reference

## Solver

```@docs
ConicIP.conicIP
ConicIP.preprocess_conicIP
```

## Solution

```@docs
ConicIP.Solution
```

## JuMP / MathOptInterface

```@docs
ConicIP.Optimizer
```

## Block Diagonal Matrices

```@docs
ConicIP.Block
ConicIP.block_idx
ConicIP.broadcastf
```

## KKT Solvers

```@docs
ConicIP.kktsolver_qr
ConicIP.kktsolver_sparse
ConicIP.kktsolver_2x2
ConicIP.pivot
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
