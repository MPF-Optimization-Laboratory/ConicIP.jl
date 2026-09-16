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
into `dev`. **The merge does not conflict: it silently deletes every private path and
applies the strip's edits to CLAUDE.md, CHANGELOG, .gitignore, test/Project.toml and
test/runtests.jl** (master's strip commit removed them and `dev` had not touched them
since). Immediately `git checkout <dev-before-merge> -- <strip list and the five files>`,
re-apply master's public CHANGELOG hunks (version header, links), check
`git diff --stat master dev` shows only the private paths, and `--amend` the merge.

## Private paths (the strip list)

```
benchmark/suite.jl            harness: instances, residuals, CSV, --opt, --timeout
benchmark/baselines.jl        Clarabel / ECOS / ConicIP-via-MOI on the same tuples
benchmark/moi_model.jl        tuple -> MOI model and back (used by baselines and tests)
benchmark/compare.jl          join CSVs, solved/verified counts, shifted geometric means
benchmark/sumnorms_diag.jl    attribution script for the sum-of-norms regression
benchmark/phases.jl           per-phase timing driver (ConicIP and Clarabel through the same MOI route)
benchmark/compare_phases.jl   joins two phase CSVs: per-instance Δ per phase, shares, counts
benchmark/alloc_phases.jl     allocation sites per phase (Profile.Allocs)
benchmark/setup_diag.jl       attribution of t_setup on the direct instances
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
julia --project benchmark/phases.jl --solver conicip [--quick] [--only a,b] [--reps 2] [--out results/phase-conicip.csv]
julia --project=benchmark benchmark/phases.jl --solver clarabel [same flags]
julia --project=benchmark benchmark/compare_phases.jl results/phase-conicip.csv results/phase-clarabel.csv [--md out.md]
julia --project benchmark/setup_diag.jl [instance ...]
julia --project benchmark/alloc_phases.jl [--quick] [--top N] [--out out.md]
```
Downloads cache in `benchmark/.cache/` (gitignored); a fresh worktree needs
`Manifest.toml` copied or `Pkg.instantiate()`, and a symlink to an existing cache saves
the downloads. For timing runs use a detached worktree at a fixed commit and keep the
machine otherwise idle; the subprocess mode restarts Julia per instance (RSS is real,
wall time includes nothing of that).

## Where things stand (2026-09-16)

- Solver: LDLᵀ default for large non-SDP problems, Ruiz equilibration (in place),
  termination in original coordinates with componentwise normalization plus a row-wise
  feasibility test, singleton presolve, native QP through MOI (assembled from `src`,
  structural convexity guard before CHOLMOD), diagnostics hook, optional shift retry and
  Gondzio correctors (both off by default after harness sweeps). The main loop is
  allocation-free on the LDLᵀ path with R/Q cones (in-place `Block` products,
  buffer-owned directions, NT scaling and residual span in place). The outer refinement
  evaluates the 4×4 residual of every base solve; the backend's `last_bound` is a
  report, not a screen (the screen was removed after the second adversarial pass, see
  the roadmap's "Adversarial review" section for the counterexample and the numbers).
- Numbers (dev 3881ac2 study, still the reference: `results/*-3881ac2.*`): 23/23
  verified; totals 8.94 s vs Clarabel 8.78 s, sgm wall ratio 1.16 (was 1.46 at 482d797).
  The seven review fixes and the screen removal leave statuses, iteration and solve
  counts and residual columns identical to 3881ac2 on every instance; the removal
  gives back the screen's saving (band instances +2–9 % wall vs 3881ac2, same
  session, best of two). Day-to-day drift of this machine is of the same size
  (lp-band-200000 4.28 s on 09-15, 4.78 s on 09-16 for the same commit), so compare
  only within a session.
- Open decisions: `hsd-design.md` §7 (go/no-go on the homogeneous embedding; it stays
  behind the SOCP allocation work); whether `Solution` may store raw residual norms and
  normalize at exit (would free 30–46 % of `t_residuals`, see roadmap item 4 outcome);
  a cheaper 4×4 residual evaluation now that the screen is gone (roadmap (b')).
- Not yet done on this stretch: a release cut (`release-0.6.0`) with the CHANGELOG's
  Unreleased section (solve3x3 view contract, MOI front-end assembly,
  `LDLDiagnostics.last_bound`/`lift_gain`/`last_rtol`, `mul!` for `Block`, the fixes).
- Test suite: 5945 tests, 15 min wall on an idle M3 Pro (9.5 min in tests, of which
  `MOI.Test` 4.3 min; the rest is instantiate/precompile). The runner is
  `verbose = true`, so the summary lists per-testset times; `results/test-times-*.txt`
  keeps one. Runs that took an hour were sharing the machine with timing jobs. Wrap
  every run in `timeout`.
- `benchmark/setup_diag.jl` (private) attributes `t_setup` on the direct instances.
- Process that worked: plan → adversarial review of the plan → parallel subagents with
  exclusive file ownership → coordinator commits at phase boundaries → adversarial review
  of the result → a SECOND adversarial pass on the fixes (that is what caught the
  screen) → harness sweeps for every default change.

Session notes for the agent side live in `~/.claude/plans/conicip-is-currently-described-buzzing-flurry.md`
and the memory files under `~/.claude/projects/-Users-mpf-projects-Software-ConicIP-jl/memory/`.
