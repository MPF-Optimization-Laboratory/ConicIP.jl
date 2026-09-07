# Changelog

All notable changes to ConicIP.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Changed
- **Breaking (direct API):** the status `:Unbounded` is now `:DualInfeasible`
  and `:AlmostUnbounded` is `:AlmostDualInfeasible`. The validated ray
  certifies dual infeasibility; primal unboundedness additionally needs
  primal feasibility, which the solver does not establish. The MOI mapping
  (`DUAL_INFEASIBLE`) is unchanged.
- `choose_kktsolver` refuses the dense solver when its storage estimate
  (`dense_kkt_bytes`) exceeds `dense_bytes_max` (default 4 GiB), whatever
  the cone mix or nonzero density says. Previously a very sparse problem
  with more than 10 nonzeros per column was routed to dense QR at any size.
- The KKT factorization is performed after the termination and certificate
  checks, so a converged iterate no longer pays for one, and a factorization
  failure cannot mask convergence.
- The dense-QR gate in `choose_kktsolver` drops from 1000 to 200 total
  dimension; `kktsolver_2x2` reuses its symbolic factorization across
  iterations like `kktsolver_sparse`.
- Line search: the trial iterate is checked for strict interiority with
  the same quantities the NT scaling computes, backing the step off
  geometrically if needed, so an accepted step can no longer fail the next
  scaling. Three consecutive steps below 1e-8 end the loop as a stall
  (`sol.message` says so) and hand over to the post-loop certificate
  screens instead of spinning to `maxIters`.
- Iterative refinement re-evaluates the step residual after the last
  correction, so the reported residual describes the step taken, and the
  predictor step is refined as well as the corrector.
- **Breaking (direct API):** the `refinementThreshold` keyword (an absolute
  bound on a size-scaled residual) is replaced by `refineRelTol = 1e-13`
  and `refineAbsTol = 1e-12`: refinement stops when
  `‖r − KΔz‖ ≤ refineAbsTol + refineRelTol·‖r‖`.
- Termination also requires the relative duality gap
  `⟨v,s⟩/(1 + |pobj|) < optTol`; the complementarity residual alone is a
  2-norm whose enforced accuracy drifted with the number of cones. The gap
  is the complementarity itself rather than `pobj − dobj`, whose residual
  products floor the computed value once the duals are large.
  `Solution.prFeas` now includes the equality residual.
- Feasibility residuals are normalized by the size of the equation they
  measure (right-hand side or data norm times iterate norm, whichever is
  larger) instead of the right-hand side alone. A homogeneous row
  (`d = 0`) with a matrix of size 10⁸ was an absolute test that rounding
  alone could never meet.

### Added
- `kktsolver_ldl`: sparse LDLᵀ factorization of the symmetric
  quasi-definite KKT system (QDLDL.jl backend, pure Julia). Fixed pattern
  with AMD ordering analysed once, numeric refactorization in place each
  iteration, second-order cones of dimension ≥ 6 lifted to diagonal plus
  two columns, static and dynamic regularization with refinement against
  the unregularized matrix. Dependent equality rows no longer need the
  preprocessor. It is the new automatic choice for large non-SDP problems:
  `choose_kktsolver` now decides between dense QR and LDLᵀ by predicted
  flops from a symbolic analysis instead of nonzeros per column, and
  `kktsolver_sparse` (UMFPACK LU) is no longer selected automatically.
  New dependency: QDLDL.jl.
- Ruiz equilibration of the problem data (`equilibrate = true` by default,
  also an MOI option), with uniform scaling inside each second-order and
  semidefinite block, and an objective rescale applied only when `‖c‖∞`
  is outside `[1e-3, 1e3]` (the iteration is not invariant to it, and on
  well-scaled problems it costs iterations). Termination is tested on
  residuals mapped back to the original coordinates, so `optTol` keeps
  its meaning; the solution and rays are mapped back as well. A custom
  `kktsolver` now receives the scaled data.
- `timeLimit` keyword (seconds) and `MOI.TimeLimitSec`: checked once per
  iteration, covering preprocessing and retries; returns `:TimeLimit`
  (`MOI.TIME_LIMIT`) with the best iterate, and skips the certificate
  fallback solves.
- `cached_kktsolver_ldl()`: an LDLᵀ solver object that reuses the AMD
  ordering across solves with the same structure (repeated solves, MPC);
  a modest saving on well-structured problems, larger where fill is heavy.
  The ordering is also computed once per solve now instead of twice
  (solver selection and factorization each ran AMD). New direct
  dependency: AMD.jl (already required by QDLDL.jl).
- `preprocess_conicIP` fixes variables from singleton equality rows
  (`gᵢⱼ yⱼ = dᵢ`), solves the reduced problem, and restores the fixed
  values and their equality duals; conflicting singletons surface as a
  certified infeasibility. Its sparse-QR rank detection is now governed by
  `rank_check = :auto | :always | :never` (also an MOI option): `:auto`
  runs it only when the KKT solver in use cannot regularize a
  rank-deficient `G` (dense QR, i.e. small and semidefinite problems, or
  an explicit non-LDLᵀ solver), so large problems no longer pay for two
  sparse QR factorizations before the solve.
- Native quadratic objectives through MOI/JuMP: `ScalarQuadraticFunction`
  objectives become the solver's `Q` instead of a second-order-cone
  reformulation, so positive semidefinite but singular Hessians solve
  (Maros–Mészáros `cvxqp1_s` failed in the bridge before).
- MOI assembly builds the constraint matrices from triplets in one
  `sparse` call each (no per-constraint CSC object, no PSD row
  reassignment), merges adjacent orthant blocks, looks constraint results
  up in O(1), and computes `G*y` once; `SolveTimeSec` now covers assembly
  as well as the solve, with the assembly part in `Optimizer.assembly_time`.
- `Solution.kkt_solves`: the number of KKT back-solves the main loop
  performed (initial point, predictor, corrector, refinements).
- `benchmark/suite.jl`: reproducible harness with phase timings, solve
  counts, fill proxies, peak RSS in a fresh process, and residuals
  recomputed from the original data.
- `benchmark/large-scale-roadmap.md`: diagnosis of the scale limits and the
  tranche plan to lift them.
- The KKT-solver guide documents the callback contract: data and signs,
  the scaling-block types, calls per iteration, ownership of returned
  arrays, accuracy and regularization, and failure reporting.

### Fixed
- The verbose "refine" column always printed 1; it now reports the number of
  refinement corrections applied in the previous iteration.

## [0.4.0] - 2026-09-06

### Added
- KKT-validated SDP test suite with known-answer instances
  (`test/sdp_tests.jl`, `test/sdp_problems.jl`).
- SDPLIB validation gate: an SDPLIB 1.2 subset fetched on demand from a
  pinned upstream commit, sha256-verified and cached, read through
  `MOI.FileFormats.SDPA` (`test/sdplib_tests.jl`; skipped offline), plus
  `benchmark/sdplib.jl` for the larger instances.
- JuMP `PSDCone()` tutorial (Lovász theta of `C₅`), a semidefinite scope and
  cost-model section, and a note on reading the semidefinite dual.
- README: Features, Quick start, Citing, and Contributing sections; version badge.
- `CHANGELOG.md`, GitHub issue forms, and `codecov.yml`.
- Enriched `CITATION.cff` (version, release date, abstract, keywords).

### Changed
- Semidefinite support is no longer labelled experimental; its scope, cost
  model and measured SDPLIB accuracy are stated in the semidefinite tutorial.
- Iteration counts and the complementarity residual on `"S"` blocks differ
  from 0.3.x, because the corrector now centres semidefinite blocks correctly.
- `maxstep_sdc` performs one generalized symmetric-definite eigen-solve
  instead of three decompositions.
- CI: coverage upload now fails loudly if the Codecov token is missing;
  Julia nightly failures no longer fail the workflow.
- `fallback_infeasibility_ray` / `fallback_unbounded_ray` now catch only KKT
  factorization failures; other exceptions from a custom `kktsolver` propagate
  instead of being reported as "no certificate".
- Tests: staged KKT-failure fault injection, post-loop certificate exits,
  MOI status mapping and metadata, `Block` `inv`/`Adjoint` products.

### Fixed
- The semidefinite cone product was `XY + YX`, twice the Jordan product,
  while the cone identity `e = vecm(I)` assumed the true Jordan product. The
  Mehrotra corrector therefore centred `"S"` blocks at `σ/2`, so a mixed-cone
  problem was centred inconsistently across its blocks.
- `maxstep_sdc` returned `-Inf` on a signed-zero search direction — which
  `kktsolver_sparse` produces — breaking multi-block semidefinite problems on
  the sparse solver.
- Nesterov-Todd scaling, the cone line searches, and the iterative refinement
  solve now honour the `:Error` contract: a boundary iterate no longer
  escapes as a raw `PosDefException`.
- Non-finite search directions and iterates are screened before they reach
  LAPACK.
- `preprocess_conicIP` reported `:Infeasible`/`:Unbounded` (without a
  certificate) when the reduced problem's ray failed revalidation against the
  original data, e.g. on a bounded problem whose equality rows are dependent
  only to tolerance. The status is now `:Error` with an explanatory message.
- `MOI.SolverVersion` returned a stale hard-coded `"0.2"`; it now reports the
  package version.

### Removed
- Unused `+`/`-` methods on the internal `v4x1` block vector.

## [0.3.2] - 2026-09-01

### Fixed
- Absolute badge and license links so the README renders correctly in the
  JuMP documentation.

## [0.3.1] - 2026-09-01

### Changed
- README credits the original author and records the project history.
- Log banners drop stale dates and use consistent colour output.

## [0.3.0] - 2026-09-01

### Added
- Validated termination with `Almost*` statuses and post-hoc certificate checks.
- Fallback minimum-norm certificate QPs when the iteration stalls.
- Automatic KKT solver selection with sparse-first heuristics; solver
  crashes are reported as a status instead of an exception (#10).
- MOI options plumbing: `RawOptimizerAttribute` and `Silent` (#10).
- Tutorials on reading the iteration log and detecting infeasibility.
- Regression suite and performance harness for #10.

### Fixed
- Verbose log printed `rDu`/`rPr` under transposed `prFeas`/`duFeas` headers.

## [0.2.0] - 2026-08-29

First registered release. Modernized the 2016 code base for Julia ≥ 1.10,
MathOptInterface 1.x, and JuMP; added Documenter.jl documentation and CI.

[Unreleased]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.4.0...HEAD
[0.4.0]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.3.2...v0.4.0
[0.3.2]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/releases/tag/v0.2.0
