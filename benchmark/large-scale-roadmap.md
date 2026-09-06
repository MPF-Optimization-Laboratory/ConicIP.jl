# ConicIP.jl large-scale roadmap

**Date:** 2026-09-06
**Baseline commit:** `25431eb` (master)
**Status:** planning document. No tranche has started. Update the Status line of a
tranche when work on it lands, and keep the `file:line` references current.

`docs/src/index.md:83` positions ConicIP as a solver for **moderate-size** LP/QP/SOCP
problems and sends large-scale users to COSMO, Hypatia, SCS, and ECOS. This document
records what would have to change to drop that qualifier honestly, and in what order.

## What "large-scale" means here

For a primal-dual interior-point method, large-scale means the Clarabel/ECOS class:
sparse LP/QP/SOCP with 10⁵–10⁶ variables and 10⁶–10⁷ nonzeros, memory proportional to
nnz plus factorization fill, and each iteration costing one sparse numeric
factorization. It does not mean the SCS class (first-order methods at 10⁷+ nonzeros),
which is a different algorithm with different accuracy trade-offs.

### Baseline

From the issue #10 close-out (PR #28), on that issue's 6010-variable SOCP
(`benchmark/issue10.jl` downloads and reproduces it):

| configuration | time | iterations | status |
|---|---|---|---|
| old default (dense QR) | ~11 s/iteration, ≳5 min/solve | — | did not converge usefully |
| current default (auto → sparse), with preprocessing | 0.78 s | 29 | Optimal |
| ECOS (yardstick) | 0.17 s | 23 | Optimal |

The remaining ~5× gap is per-iteration constants (numeric refactorization with an
unsymmetric LU each iteration) plus the preprocessor's own sparse QR solves.

## Diagnosis: what caps scale today, ranked by impact

### 1. KKT linear algebra (the ceiling)

- **No LDLᵀ / quasi-definite solver.** `kktsolver_sparse` (`src/kktsolvers.jl:255-342`)
  factors the symmetric 3×3 KKT matrix with UMFPACK's *unsymmetric* LU: roughly twice the
  memory and flops of a symmetric LDLᵀ, no fill-reducing ordering computed for the
  symmetric pattern, and no way to express the quasi-definite regularization
  `[Q+δI Gᵀ; G −δI]`. `staticReg` perturbs only the Q block (`src/ConicIP.jl:890-891`)
  and cannot repair a rank-deficient G. Issue #10 already names LDLᵀ as the route to
  ECOS-class per-iteration cost; no LDL code exists in the tree.
- **Dense QR is still the default** for `n+m+p < 1000`, for more than 10 structural
  nonzeros per column, and for every problem with an SDP cone
  (`src/kktsolvers.jl:28-42`). It materializes an n×n orthogonal factor at setup
  (`src/kktsolvers.jl:85`) and a dense (n−p)² Cholesky per iteration
  (`src/kktsolvers.jl:99-106`). O(n²) memory and O(n³) time, unconditionally.
- **`kktsolver_2x2`** re-runs symbolic and numeric `lu` and re-forms an explicit Schur
  complement every iteration (`src/kktsolvers.jl:361-367`). No factorization caching.
- **Up to five back-solves per iteration.** `refinementThreshold = optTol/1e7`, i.e.
  1e-13 (`src/ConicIP.jl:680`), is effectively unreachable, so all three refinement steps
  (`src/ConicIP.jl:1219-1235`) run on top of the predictor and corrector solves.

### 2. SDP cones are dense at O(k⁴)

- `VecCongurance` stores a dense k×k `R` (`src/ConicIP.jl:40`, field typed as the
  abstract `Matrix`). `Matrix(W)` and `sparse(W)` build a dense k(k+1)/2-square matrix
  (`src/ConicIP.jl:94-106`), i.e. O(k⁴) memory. At k=100 that is ~200 MB per block per
  iteration on the sparse path, which is why SDP problems are routed to dense QR.
- Per iteration per block: two Cholesky factorizations, one SVD, and one explicit inverse
  in `nestod_sdc` (`src/ConicIP.jl:233-247`); `lyap` in `dsdc!`; two `eigvals` calls plus
  an inverse square root in `maxstep_sdc`, which is called four times per iteration.
- The standard large-SDP approach (SDPA, SeDuMi, Mosek) never forms these operators. It
  builds the m×m Schur complement `M_ij = tr(A_i W A_j W)` per SDP block and factors it
  densely. That is a new cone-specific KKT path, not a tweak to the current one.

### 3. No time limit, and a 2–3× tail on hard instances

- There is no wall-clock check anywhere. `MOI.TimeLimitSec` is unimplemented (only
  `SolveTimeSec`, `src/MOI_wrapper.jl:724-725`). The only escape is `maxIters = 100`.
- On iteration exhaustion, `certFallback` launches two more full solves of 50 iterations
  each (`src/ConicIP.jl:1295-1328`), and `preprocess_conicIP` retries the whole solve with
  `staticReg = 1e-8` on `:Error` (`src/preprocessor.jl:171-175`). Exactly the instances
  that are already slow get multiplied.

### 4. Presolve does sparse rank detection on the whole problem

- `imcols` runs SPQR `qr(sparse(Aᵀ))` (`src/preprocessor.jl:29`) plus a second
  factorization for the consistency test (`src/preprocessor.jl:42`). `preprocess_conicIP`
  applies it to `G` and to the n×(n+m+p) concatenation `[Q Aᵀ Gᵀ]`
  (`src/preprocessor.jl:89`, `:115`). Sparse QR fill can dwarf the KKT factorization
  itself. ECOS and Clarabel do not rank-detect; they regularize. With a quasi-definite
  LDLᵀ, full-matrix `imcols` becomes optional.
- There is no data equilibration (Ruiz or similar). `docs/src/background.md:217` asks the
  user to scale the data. At scale, badly scaled real-world data is the norm.

### 5. MOI front end is quadratic in constraint count

- One `spzeros(dim, n)` per constraint, filled by scalar CSC insertion
  (`src/MOI_wrapper.jl:206-240`); PSD scaling does `dim` row assignments into CSC
  (`src/MOI_wrapper.jl:277-288`); a full `UniversalFallback` model copy precedes
  extraction (`src/MOI_wrapper.jl:310-314`).
- Result queries linear-scan `eq_ci_map` and `ineq_ci_map` per constraint and re-slice
  `eq_G[rows,:]` per call (`src/MOI_wrapper.jl:643-717`). Retrieving all duals is
  O(#constraints²).
- No quadratic objective through MOI (`Q = spzeros(n,n)` at `src/MOI_wrapper.jl:343`);
  JuMP QP users pay a bridge that adds an SOC and a variable.

### 6. Secondary constants

- A new `Block` per iteration from `nt_scaling` (`src/ConicIP.jl:838-854`); `block_idx`
  and `size(::Block)` allocate on every call (`src/blockmatrices.jl:45-49, 66-80`); a
  `Block` cannot multiply a `SparseMatrixCSC`, which forces `sparse(F⁻ᵀ)` in the 2×2
  solver.
- `Solution` has abstract-typed numeric fields (`src/ConicIP.jl:451-457`). Minor.
- No multithreading. Not a priority: an interior-point iteration is factorization-bound,
  and BLAS and UMFPACK thread internally.

### 7. No evidence base

- The largest problem in CI has n=650 (`test/runtests.jl:1363`). The largest tutorial
  has 12 variables. The 6010-variable issue-#10 instance is download-only and outside CI.
- There are no runs against Maros–Mészáros (QP), CBLIB (SOCP), or SDPLIB, and no
  comparison harness against Clarabel.jl, ECOS, or COSMO. A large-scale claim without
  this is not credible.

## Roadmap, in tranches

Ordering reflects dependency and payoff. Tranche 1 is the enabling piece. Tranches 2 and
3 are independent of each other. Tranches 4 and 5 are optional directions. Tranche 6
gates the documentation change.

### Tranche 1: sparse LDLᵀ core

**Status:** not started. **Estimate:** 1–2 weeks. Removes the ceiling for LP/QP/SOCP.

1. New `kktsolver_ldl` in `src/kktsolvers.jl` with the existing `solve3x3gen(F, F⁻ᵀ)`
   interface. Assemble the quasi-definite system
   `[Q+δI  Gᵀ  −Aᵀ; G  −δI  0; A  0  FᵀF (+lift)]` as one symmetric CSC matrix with a fixed
   sparsity pattern. Reuse `lift` (`src/kktsolvers.jl:135-180`) so SOC blocks enter as
   diagonal plus low-rank and the pattern never changes between iterations. Symbolic
   analysis and AMD ordering once; numeric refactorization per iteration; dynamic
   regularization on tiny pivots. Backend: QDLDL.jl (pure Julia, used by Clarabel and
   COSMO, keeps the "pure Julia" claim), with CHOLMOD `ldlt` as a fallback option.
2. Make `kktsolver_ldl` the default in `choose_kktsolver` for all non-SDP problems. Lower
   the dense gate from 1000 to about 200 total dimension. Give `kktsolver_2x2` the
   `factor!` cache or retire it.
3. Make the refinement stop relative: `refinementThreshold` about `0.1·optTol` relative
   to the right-hand-side norm, and refine only when the step residual exceeds it.
4. Wall-clock limit: a `timeLimit` keyword checked once per iteration in `conicIP`,
   returning `:TimeLimit` with the best iterate; wire `MOI.TimeLimitSec` and
   `MOI.TIME_LIMIT`.
5. Cap the exhaustion tail: run `certFallback` only when the screens came within 100× of
   `infeasTol`, and never when the time budget is spent.

**Verification targets**

- `Pkg.test()` green, with SDP tests still routed through `kktsolver_qr`.
- `benchmark/issue10.jl`: at most 0.4 s (within about 2× of ECOS) at the same iteration
  count.
- Synthetic sparse LP and SOCP at n = 10⁵, nnz ≈ 10⁶: peak memory proportional to
  factorization fill, no O(n²) allocation (`@allocated`, `Sys.maxrss`).
- `benchmark/regressions.jl` bounds hold; `jetls check` shows no new inference failures.
- `timeLimit = 1` on the n = 10⁵ instance returns `:TimeLimit` with a finite best iterate.

### Tranche 2: front end and presolve

**Status:** not started. **Estimate:** 1 week.

1. MOI assembly by triplet accumulation (`I`, `J`, `V` vectors and one
   `sparse(I, J, V, m, n)` call per block); PSD input scaling as a diagonal multiply;
   constraint-index maps as dense `Vector{UnitRange}` for O(1) result lookup; cache
   `G*y` and `A*y` once per solve.
2. Native quadratic objectives through MOI. The direct API already supports `Q`.
3. Ruiz equilibration of `[Q Aᵀ Gᵀ; A; G]` with cone-aware uniform scaling inside each
   SOC and SDP block, undone on `Solution`. Belongs in `preprocess_conicIP`.
4. Make full-matrix `imcols` rank detection opt-in (`rank_check = :auto | :always |
   :never`), defaulting to the O(nnz) structural checks plus LDLᵀ regularization. Drop the
   `staticReg` retry once LDLᵀ carries its own regularization.

**Verification targets**

- MOI `copy_to` plus solve on a 10⁵-constraint LP: assembly time linear in nnz, dual
  retrieval for all constraints under one second.
- Equilibration: the issue-#10 instance and a deliberately badly scaled variant (rows
  scaled by 10⁶) converge in the same iteration count.
- Preprocessing without rank detection: `Pkg.test()` green with `rank_check = :never`
  except for the tests that construct rank-deficient `G` on purpose.

### Tranche 3: iteration count

**Status:** not started. **Estimate:** 1–2 weeks, higher risk.

The 29-versus-23 iteration gap to ECOS is algorithmic, not linear-algebraic.

1. Gondzio multiple centrality correctors. Cheap: reuses the factorization, typically
   cuts 10–20 % of iterations.
2. Homogeneous self-dual embedding, replacing the certificate screens and the
   `certFallback` double solve. Cleaner infeasibility detection at scale, but it is a
   rewrite of the main loop and of `src/certificates.jl`. Defer until Tranches 1–2 are
   measured.

**Verification targets**

- Median iteration count on the Tranche 6 suite within 10 % of ECOS and Clarabel.
- The infeasible and unbounded tests in `test/runtests.jl` keep their statuses and
  certificates.

### Tranche 4: SDP at scale

**Status:** not started. **Estimate:** 3–4 weeks. Only if SDP is a goal.

Schur-complement KKT path for problems with SDP blocks: per block store the constraint
rows as symmetric matrices, form `M_ij = tr(A_i W A_j W)`, and take a dense Cholesky of
the m×m system. Replace `Matrix(::VecCongurance)`; vectorize `mat` and `vecm`; reduce
`maxstep_sdc` to one eigendecomposition per call; compute the NT scaling with one
eigendecomposition instead of Cholesky, SVD, and inverse. Without this, the honest claim
is "large-scale LP/QP/SOCP; SDP experimental", which is what the README already says.

Build on the in-flight branches `sdp-solver-fixes`, `sdp-test-suite`, and
`sdplib-validation` (worktrees `.worktrees/sdp-fixes`, `sdp-tests`, `sdplib`), which
already carry an SDP test suite, a vendored SDPLIB subset, and an SDPLIB benchmark script.

**Verification targets**

- SDPLIB instances with block size k up to 300 and m up to 10⁴ solve in memory
  proportional to m² + k², not k⁴.
- The vendored SDPLIB subset matches SDPA reference objectives to 1e-6 relative.

### Tranche 5: matrix-free operators

**Status:** not started. **Estimate:** 2–3 weeks. The differentiating direction.

ConicIP's stated niche is the pluggable KKT solver. The main loop already touches `Q`,
`A`, and `G` only through matrix-vector products, so accepting `LinearMap`-style
operators is close. The blockers are `structurally_zero_cols` (needs CSC),
`norm(Q, Inf)`, the deflation sub-indexing in `conicIP`, and `src/certificates.jl`. Ship
an iterative KKT solver alongside: MINRES on the regularized quasi-definite system with a
block-diagonal preconditioner, or PCG on the reduced normal equations for R and Q cones,
with an inexact-Newton tolerance schedule. This is the one axis where ConicIP could do
something Clarabel does not.

**Verification targets**

- A problem given with `Q`, `A`, `G` as operators (no explicit matrices) solves to
  `optTol` with the iterative KKT solver and never calls `sparse` or `Matrix` on them.
- Iteration count with inexact solves within 20 % of the direct-solver count on the
  Tranche 6 suite.

### Tranche 6: evidence, then the docs change

**Status:** not started. **Estimate:** 1 week.

1. `benchmark/suite.jl`: the Maros–Mészáros QP set, a CBLIB SOCP subset, and the
   issue-#10 instance; baselines Clarabel.jl, ECOS, COSMO; Dolan–Moré performance
   profiles.
2. One medium instance (n ≈ 10⁴) in CI with iteration and allocation bounds, extending
   `benchmark/regressions.jl`.
3. Only then rewrite `docs/src/index.md:76-93` and the README positioning.

**Verification targets**

- Performance profiles show ConicIP within 2× of Clarabel on at least 90 % of the QP and
  SOCP instances it solves, with the same solved set.
- The CI medium instance runs in under 30 s on the GitHub Actions runner.

## What not to do

- Threading, Float32, GPU: not where the time goes for an interior-point method.
- Chasing per-iteration constants before the LDLᵀ solver exists.
- Dropping the "moderate-size" phrase before Tranche 6 numbers exist.
