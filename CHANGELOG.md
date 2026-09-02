# Changelog

All notable changes to ConicIP.jl are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project
uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- README: Features, Quick start, Citing, and Contributing sections; version badge.
- `CHANGELOG.md`, GitHub issue forms, and `codecov.yml`.
- Enriched `CITATION.cff` (version, release date, abstract, keywords).

### Changed
- CI: coverage upload now fails loudly if the Codecov token is missing;
  Julia nightly failures no longer fail the workflow.

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

[Unreleased]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.3.2...HEAD
[0.3.2]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.3.1...v0.3.2
[0.3.1]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.3.0...v0.3.1
[0.3.0]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/compare/v0.2.0...v0.3.0
[0.2.0]: https://github.com/MPF-Optimization-Laboratory/ConicIP.jl/releases/tag/v0.2.0
