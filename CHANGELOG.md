# Changelog

All notable changes to ConicIP.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

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
