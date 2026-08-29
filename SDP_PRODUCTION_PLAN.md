# SDP Solver: Experimental → Production Plan

## Current State Assessment

The SDP implementation in ConicIP.jl is more complete than the documentation
suggests. The core interior-point loop already handles semidefinite cones end
to end: Nesterov-Todd scaling (`nestod_sdc`), line search (`maxstep_sdc`),
cone arithmetic (`dsdc!`/`xsdc!`), and the MOI wrapper accepts
`PositiveSemidefiniteConeTriangle`. The docstring at `src/ConicIP.jl:431`
declares SDP "NOT supported and purely experimental", but the solver does
produce correct results on small problems.

What is missing falls into six categories: **correctness coverage**, **numerical
robustness**, **performance**, **KKT solver integration**, **MOI wrapper
completeness**, and **documentation**.

---

## Phase 1: Correctness and Test Coverage

The single existing SDP test (PSD projection, k=6) is not sufficient to ship.

### 1a. Expand the SDP test suite

Add tests covering these scenarios (all currently untested):

| Test case | Why it matters |
|-----------|----------------|
| **Multiple SDP blocks** (`[("S",k1), ("S",k2)]`) | Exercises `Block` with >1 `VecCongurance` entry |
| **Mixed cones** (`[("R",n), ("Q",m), ("S",k)]`) | Verifies block indexing and heterogeneous scaling |
| **SDP + equality constraints** (G, d non-empty) | Equality rows interact with KKT system |
| **Larger matrix sizes** (k=15, k=45, k=100) | Numerical issues surface at scale |
| **Known-optimal SDP instances** (Lovász theta, max-cut relaxation) | Validates objective value to known solutions |
| **Infeasible SDP** | Confirms infeasibility certificate for SDP cones |
| **Unbounded SDP** | Confirms unboundedness certificate |
| **SDP via MOI/JuMP interface** | End-to-end MOI wrapper test (`PositiveSemidefiniteConeTriangle`) |
| **Dual variable recovery** | Checks that MOI dual extraction returns correct matrix |

**Files:** `test/runtests.jl`, possibly a new `test/sdp_problems.jl` for the
problem generators.

### 1b. Validate the `vecm`/`mat` round-trip

Add property-based tests that for random symmetric matrices X:
- `mat(vecm(X)) == X` (exact round-trip)
- `dot(vecm(X), vecm(Y)) ≈ tr(X*Y)` (inner product preservation)
- These should run at sizes n = 2, 5, 10, 20.

### 1c. Validate Nesterov-Todd scaling identity

For random strictly feasible (z, s) pairs in the SDP cone, verify:
- `F*z ≈ inv(F)*s` where `F = nestod_sdc(z, s)`
- `F*z ≈ F'*z` (self-adjointness of the scaled point λ)

---

## Phase 2: Numerical Robustness

### 2a. Guard `nestod_sdc` against non-PD inputs

`nestod_sdc` (`src/ConicIP.jl:198-212`) calls `cholesky(mat(s))` and
`cholesky(mat(z))` without checking positive definiteness first. If z or s
drifts to the cone boundary (common in late iterations), Cholesky will throw
a `PosDefException`.

**Fix:** Add a `try/catch` or eigenvalue check. If the Cholesky fails, fall
back to an eigenvalue-based factorization (`eigen` with clamped eigenvalues),
or signal the solver to reduce the step size. This is the most likely source
of runtime failures on real SDP problems.

### 2b. Guard `maxstep_sdc` against near-singular matrices

`maxstep_sdc` (`src/ConicIP.jl:274-295`) computes `X^(-1/2)` via
`X^(-1/2)`. When X has eigenvalues near zero, this is numerically
catastrophic.

**Fix:** Use the eigendecomposition `X = U*Λ*U'`, compute `X^{-1/2} =
U*Λ^{-1/2}*U'` with explicit clamping of small eigenvalues, and symmetrize the
result. This is more expensive but avoids silent NaN propagation.

### 2c. Symmetrize `xsdc!` output

`xsdc!` computes `X*Y + Y*X` (`src/ConicIP.jl:357-362`). Due to floating-point
arithmetic, the result may not be perfectly symmetric after `vecm`. Add an
explicit symmetrization step: `M = 0.5*(M + M')` before `vecm`.

### 2d. Handle the `lyap` call in `dsdc!`

`dsdc!` (`src/ConicIP.jl:349-355`) uses `lyap(mat(Y), -mat(x))` from
LinearAlgebra. The Lyapunov solver can fail or return inaccurate results when Y
has eigenvalues close to zero or when Y has eigenvalues that nearly sum to zero.
Add a residual check and fall back to a direct solve if needed.

---

## Phase 3: Performance

These items correspond to issues already identified in `benchmark/report.md`
but are specifically blocking for production SDP at scale.

### 3a. Fix `VecCongurance` materialization in KKT solvers

The `lift()` function in `src/kktsolvers.jl:60-105` has specialized paths for
`SymWoodbury` and `Diagonal` blocks, but **no path for `VecCongurance`**. When
an SDP block is present, `lift()` silently produces an incomplete
factorization. The `kktsolver_sparse` and `kktsolver_2x2` solvers fall back to
`sparse(Block)`, which materializes `VecCongurance` column-by-column into a
dense matrix (`src/ConicIP.jl:71-79`) — O(n³) work creating a full n×n dense
matrix for each SDP block.

**Fix options (in order of increasing complexity):**

1. **Direct dense embedding:** In `lift()`, add a `VecCongurance` branch that
   extracts the dense matrix directly via `Matrix(Blk)` and inserts its
   entries into the sparse IJV arrays. This avoids the column-by-column
   application but still materializes the block.

2. **Cholesky-based sparse lift:** Since `VecCongurance(R)` represents
   `x → vecm(R'*mat(x)*R)`, we can express this as a sparse Kronecker
   product `(R⊗R) * P` where P is the vec↔vecm permutation/scaling matrix.
   This gives `lift()` a structured sparse representation without
   materializing.

3. **SDP-specialized KKT solver:** For problems with large SDP blocks, bypass
   `lift()` entirely and use a Schur complement approach that works directly
   with the `R` matrix from `VecCongurance`. This is the standard approach in
   production SDP solvers (SeDuMi, SDPT3, MOSEK).

Recommendation: Implement option 1 first to fix correctness, then option 3
for production-scale performance.

### 3b. Vectorize `mat()` and `vecm()`

`mat()` and `vecm()` (`src/ConicIP.jl:93-151`) use scalar loops with
conditional branches for the √2 scaling. These are called multiple times per
iteration per SDP block.

**Fix:** Use index arithmetic to separate diagonal and off-diagonal entries,
then apply the scaling as a single vectorized multiply. This eliminates the
branch and enables SIMD.

### 3c. Cache eigendecompositions in `maxstep_sdc`

`maxstep_sdc` is called twice per iteration (predictor and corrector steps)
and computes `eigvals(Symmetric(X))` each time. The matrix X (the current
primal SDP iterate) is the same in both calls within a single iteration.

**Fix:** Cache the eigendecomposition of the current iterate and reuse it
across the two line search calls.

### 3d. Reduce allocations in SDP cone operations

`nestod_sdc` allocates intermediate matrices for `Ls`, `Lz`, `F.U`, `Λ`, and
`R` on every iteration. For a k×k SDP block these are each k×k matrices.

**Fix:** Pre-allocate workspace buffers for the SDP cone operations and pass
them through the iteration loop. This requires adding a workspace struct to
`conicIP()` and threading it through `nt_scaling`, `maxstep`, etc.

---

## Phase 4: KKT Solver Correctness for SDP

### 4a. Add `VecCongurance` branch to `lift()`

As noted in 3a, `lift()` (`src/kktsolvers.jl:60-105`) has no handling for
`VecCongurance` blocks. This means `kktsolver_sparse` produces wrong results
silently when SDP cones are present.

**This is a correctness bug, not just a performance issue.** It must be fixed
before SDP can be considered production-ready with any KKT solver other than
`kktsolver_qr` (which bypasses `lift()`).

### 4b. Verify all three KKT solvers produce correct results for SDP

After fixing `lift()`, add a test that solves the same SDP problem with
`kktsolver_qr`, `kktsolver_sparse`, and `pivot(kktsolver_2x2)`, and verifies
they produce the same optimal solution.

### 4c. Benchmark KKT solver selection for SDP

The current `kktsolver` default selection logic may not be optimal for SDP
problems. Profile each solver on SDP problems of varying size and determine
the appropriate crossover points.

---

## Phase 5: MOI Wrapper Completeness

### 5a. Dual variable extraction for SDP

The MOI wrapper returns dual variables in vectorized form. For SDP cones,
users typically expect the dual matrix. The wrapper should:
- Correctly map vectorized duals back to `MOI.ConstraintDual` results
- Verify the scaling convention matches what MOI expects (ConicIP uses √2
  off-diagonal scaling; MOI's `PositiveSemidefiniteConeTriangle` expects the
  same, but this should be verified)

### 5b. Support `PositiveSemidefiniteConeSquare`

MOI also has `PositiveSemidefiniteConeSquare` for full (non-triangular) matrix
representations. Adding this is straightforward: symmetrize the input and
delegate to the triangle path.

### 5c. Add MOI `Test` suite for SDP

Run the standard `MOI.Test` regressions with SDP constraints to catch edge
cases in constraint handling, variable bridging, and result queries.

### 5d. Support quadratic objectives with SDP constraints

Currently the MOI wrapper only supports `ScalarAffineFunction` and
`VariableIndex` objectives (`src/MOI_wrapper.jl:60-64`). A
`ScalarQuadraticFunction` objective combined with SDP constraints is a valid
problem class (QCQP with SDP relaxation). Verify this works end-to-end or
document the limitation.

---

## Phase 6: Documentation and API

### 6a. Remove "experimental" labels

- Update docstring at `src/ConicIP.jl:431-432`
- Update `docs/src/tutorials/sdp.jl` warning banner
- Update any README references

### 6b. Document the SDP vectorization convention

The `vecm`/`mat` convention (scaled lower-triangular, √2 on off-diagonals) is
standard (sometimes called "svec") but must be clearly documented since users
constructing problems via the direct API need to know it.

### 6c. Add an SDP tutorial with a realistic problem

The existing tutorial (`docs/src/tutorials/sdp.jl`) only shows PSD projection.
Add a more realistic example: Lovász theta number or a max-cut SDP relaxation,
showing both the direct API and the JuMP interface.

### 6d. Document solver limitations

Be explicit about:
- Maximum practical SDP block size (determined by Phase 3 benchmarking)
- Which KKT solver to use for SDP problems
- Known numerical sensitivities

---

## Recommended Execution Order

```
Priority  Phase   Item   Description
────────  ──────  ─────  ─────────────────────────────────────────
  P0      4a      ★      Fix lift() for VecCongurance (correctness bug)
  P0      2a      ★      Guard nestod_sdc against non-PD inputs
  P1      1a             Expand SDP test suite
  P1      4b             Verify all KKT solvers with SDP
  P1      2b             Guard maxstep_sdc against near-singular X
  P1      2c             Symmetrize xsdc! output
  P2      1b             vecm/mat round-trip tests
  P2      1c             NT scaling identity tests
  P2      2d             Guard lyap in dsdc!
  P2      5a             MOI dual extraction for SDP
  P3      3a             VecCongurance materialization (performance)
  P3      3b             Vectorize mat()/vecm()
  P3      3c             Cache eigendecompositions
  P3      3d             Reduce allocations
  P3      5b             PositiveSemidefiniteConeSquare
  P4      5c             MOI Test suite for SDP
  P4      5d             Quadratic + SDP verification
  P4      4c             KKT solver benchmarking for SDP
  P4      6a-6d          Documentation updates
```

P0 items are **blocking correctness bugs** that will cause wrong answers or
crashes on non-trivial SDP problems. P1 items are needed to have confidence
the solver works. P2 items improve robustness. P3 items address performance
for larger problems. P4 items are polish for a public release.

---

## Estimated Scope

- **P0 (blocking bugs):** 2 items, touching `kktsolvers.jl` and `ConicIP.jl`
- **P1 (must-have):** 4 items, primarily new tests + defensive guards
- **P2 (robustness):** 4 items, scattered across solver core and MOI wrapper
- **P3 (performance):** 4 items, primarily `ConicIP.jl` and `kktsolvers.jl`
- **P4 (polish):** 5 items, docs + MOI test suite

The P0+P1 items represent the minimum viable path to removing the
"experimental" label.
