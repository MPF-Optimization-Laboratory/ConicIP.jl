# Resuming the large-scale work (private)

Read this first when picking the project up again, then the status paragraphs of
`large-scale-roadmap.md` (Tranches 0–3) and §7 of `hsd-design.md`. Everything under
`benchmark/` except `profile.jl`, `report.md`, `regressions.jl`, `sdplib.jl` and
`issue10.jl` is private development material and never goes upstream.

## Branch layout

| branch | role | pushed? |
|---|---|---|
| `master` | public, what is released | yes |
| `dev` | private mainline: everything in `master` plus the private artifacts and the tests that depend on them | never |
| `release-<ver>` | `dev` minus the private artifacts, PR'd into `master` | yes, per release |
| `large-scale-roadmap`, `tranche-3-prep` | historical; both end at `7095299`, which is `dev`'s starting point | `large-scale-roadmap` was pushed once by mistake (delete it on the remote) |

Rule: develop on `dev` (or short branches off it, merged back with `--ff-only`). Cutting a
release: `benchmark/strip-release.sh <ver>` creates `release-<ver>` from `dev` with the
private paths removed and prints the remaining references to clean by hand (CLAUDE.md
benchmarking lines, CHANGELOG mentions, test includes); then bump `Project.toml`, add the
CHANGELOG header, run the suite under the current Julia and 1.10, build the docs, push,
`gh pr create`, merge, `@JuliaRegistrator register`. After the merge, `git merge master`
into `dev` (conflicts only in the files the strip touched; keep `dev`'s side).

## Private paths (the strip list)

```
benchmark/suite.jl            harness: instances, residuals, CSV, --opt, --timeout
benchmark/baselines.jl        Clarabel / ECOS / ConicIP-via-MOI on the same tuples
benchmark/moi_model.jl        tuple -> MOI model and back (used by baselines and tests)
benchmark/compare.jl          join CSVs, solved/verified counts, shifted geometric means
benchmark/sumnorms_diag.jl    attribution script for the sum-of-norms regression
benchmark/Project.toml        env for baselines (Clarabel, ECOS); Manifest gitignored
benchmark/large-scale-roadmap.md
benchmark/hsd-design.md
benchmark/RESUME.md           this file
benchmark/strip-release.sh
test/review_benchmark.jl      includes suite.jl
test/harness_tests.jl         includes suite.jl and moi_model.jl
```
`test/Project.toml` carries `Dates` only for `suite.jl`; `.gitignore` carries
`benchmark/.cache/`, `benchmark/results/`, `benchmark/Manifest.toml` only on `dev`.

## Running things

```
julia --project benchmark/suite.jl --quick                     # 6 instances, ~1 min
julia --project benchmark/suite.jl --out results/x.csv         # 23 instances, ~10 min (subprocess per instance)
julia --project benchmark/suite.jl --opt centralityCorrectors=2
julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'   # once
julia --project=benchmark benchmark/baselines.jl --solver clarabel --timeout 120 --out results/clarabel.csv
julia --project=benchmark benchmark/compare.jl results/x.csv results/clarabel.csv
julia --project benchmark/sumnorms_diag.jl [--quick]
```
Downloads cache in `benchmark/.cache/` (gitignored); a fresh worktree needs
`Manifest.toml` copied or `Pkg.instantiate()`, and a symlink to an existing cache saves
the downloads. For timing runs use a detached worktree at a fixed commit and keep the
machine otherwise idle; the subprocess mode restarts Julia per instance (RSS is real,
wall time includes nothing of that).

## Where things stand (2026-09-08)

- Solver: LDLᵀ default for large non-SDP problems, Ruiz equilibration, termination in
  original coordinates with componentwise normalization plus a row-wise feasibility test,
  singleton presolve, native QP through MOI, diagnostics hook, optional shift retry and
  Gondzio correctors (both off by default after harness sweeps).
- Numbers: 23/23 harness instances verified; Clarabel 1.6× faster in shifted geometric
  mean at about one fewer iteration; v0.4.0 comparison and the sweep tables are in the
  roadmap's Tranche 2/3 status paragraphs.
- Open decisions: `hsd-design.md` §7 (ten items; go/no-go on the homogeneous embedding
  first). Candidate next steps, in the order I would take them: per-phase timing to
  locate the 1.6× per-iteration gap (assembly, scaling, back-solves, cone operations)
  since the embedding will not close it; then the embedding behind `method = :hsd`; then
  Tranche 4 (SDP Schur complement) or 5 (operator inputs) by demand.
- Process that worked: plan → adversarial review of the plan → parallel subagents with
  exclusive file ownership → coordinator commits at phase boundaries → adversarial review
  of the result → harness sweeps for every default change. Skipping the last two is
  where earlier mistakes came from.

Session notes for the agent side live in `~/.claude/plans/conicip-is-currently-described-buzzing-flurry.md`
and the memory files under `~/.claude/projects/-Users-mpf-projects-Software-ConicIP-jl/memory/`.
