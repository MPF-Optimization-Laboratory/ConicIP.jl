# # Reading the Iteration Log
#
# Every ConicIP solve can narrate itself. Passing `verbose = true` to
# [`conicIP`](@ref ConicIP.conicIP) prints one row per interior-point
# iteration, so you can watch the residuals fall, the duality gap close, and
# the infeasibility screens stay quiet. This page reads one such log column
# by column.
#
# The solver always works on the same problem,
#
# ```
# minimize    ½yᵀQy - cᵀy
# subject to  Ay ≥_K b
#             Gy  = d
# ```
#
# where `≥_K` means `Ay - b` lies in the cone described by `cone_dims`. The
# log is a solver-level feature, so we call `conicIP` directly rather than
# going through JuMP. All data below is deterministic, so the output is
# reproducible.
#
# The log comes in two styles. In an interactive terminal the default is the
# *rich* style, a Unicode table with colour described at the end of this
# page; anywhere else (a file, a pipe, a notebook, these docs) it is the
# *classic* ASCII table. Most of this page reads the classic table, so the
# calls below ask for it with `log_style = :classic`.
#
# ## A small quadratic program
#
# Minimize a strictly convex quadratic over the nonnegative orthant:

using ConicIP, SparseArrays, LinearAlgebra

Q = [2.0 0.5 0.0
     0.5 2.0 0.5
     0.0 0.5 2.0]
c = [1.0, 2.0, 3.0]

## Nonnegativity: y ≥ 0
A = sparse(1.0I, 3, 3)
b = zeros(3)
cone_dims = [("R", 3)]

sol = conicIP(Q, c, A, b, cone_dims; verbose = true, log_style = :classic)
println("status: ", sol.status)

# ## The three optimality columns
#
# The first block of the log tracks the three residuals that define
# convergence — constraint satisfaction, KKT stationarity, and complementary
# slackness — each scaled by the size of the data:
#
# ```
# ‖Ay - s - b‖ / (1 + ‖b‖)      primal feasibility
# ‖Qy + Gᵀw - Aᵀv - c‖ / (1 + ‖c‖)   dual feasibility
# sᵀv / (1 + |cᵀy|)             complementarity
# ```
#
# The run above needs five iterations to drive all three below `optTol`
# (default `1e-6`), gaining about two digits per iteration once the
# predictor–corrector steps take hold. The same three numbers are returned in
# the [`Solution`](@ref ConicIP.Solution) struct:

(prFeas = sol.prFeas, duFeas = sol.duFeas, muFeas = sol.muFeas)

# ## The objective columns
#
# `pobj` is `½yᵀQy - cᵀy` at the current iterate; `dobj` is the value of the
# dual objective there. Weak duality puts `dobj ≤ pobj` for any feasible
# pair, and the two squeeze together as complementarity closes. Watching them
# converge is the quickest read on how far along a solve is:

(pobj = sol.pobj, dobj = sol.dobj, gap = sol.pobj - sol.dobj)

# ## The infeasibility columns
#
# `icertp` and `icertd` are the *screens* for primal and dual infeasibility —
# cheap per-iteration tests that ask whether the current iterate looks like a
# Farkas ray (`icertp`) or a recession ray (`icertd`). Each is a scaled
# residual that has to fall below `infeasTol` (default `1e-7`) before the
# solver will even consider the iterate as a certificate candidate.
#
# A screen prints `NaN` when the sign condition that makes a ray meaningful
# fails — `dᵀw - bᵀv < 0` for `icertp`, `cᵀy > 0` for `icertd`. On a solvable
# problem like this one both columns should stay `NaN` or stay large, which
# is exactly what happens above. Falling below the tolerance only *nominates*
# a ray; the claim is made by a separate validator run against the original
# data. See [The Certificate Pipeline](@ref) for the full story, and
# [Detecting Infeasibility](@ref) for logs where these columns do fire.
#
# ## The refinement column
#
# `refine` counts the iterative-refinement steps spent on that iteration's
# KKT solve, capped by `maxRefinementSteps` (default `3`). Zero on the first
# iteration and one thereafter is the healthy pattern. If the scaled KKT
# residual of the step is still above `1e-3` after refinement, the whole row
# is printed in bold red — a signal that the KKT system is badly conditioned
# and that `staticReg` or a different `kktsolver` may be needed. Colour
# follows Julia's `--color` setting, so it never appears in captured output.
#
# ## The kkt column
#
# `kkt` is filled in only by KKT solvers that report diagnostics (the
# sparse LDLᵀ solver does; the dense QR solver leaves it blank). It reads
# `repaired/refactors` for the factorization that produced the row's
# iterate: how many pivots dynamic regularization had to replace, and how
# many times the static shifts were bumped and the matrix refactorized
# (always `0` unless `kktsolver_ldl` is called with `retry_max > 0`). `0/0`
# on every row is the healthy pattern; the totals over the solve are in
# `sol.kkt_repaired` and `sol.kkt_refactors`.
#
# ## The cc column
#
# `cc` is blank unless `centralityCorrectors > 0`. It then reads
# `accepted/tried` for the Gondzio centrality correctors of the step that
# produced the row's iterate: each corrector is one extra back-solve of the
# current KKT factorization, kept only if it lengthens the step, and the
# loop stops at the first rejection. `0/0` means the step was already full
# (or there was no centering target), so nothing was tried. The extra solves
# are included in `sol.kkt_solves`.
#
# ## The exit line
#
# The log ends with a one-line verdict. `EXIT -- Below Tolerance!` means all
# three optimality residuals cleared `optTol`, and the status returned is
# `:Optimal`. The other exits — `Certificate of Infeasibility Found!`,
# `Certificate of Dual Infeasibility Found!`, and `Error!` — are the subject
# of the next page. A solve that runs out of iterations ends with
# `No verdict (...)` and the iteration of the best iterate it returns, and
# `Almost infeasible` / `Almost dual infeasible` mark a ray that validates
# only at a relaxed tolerance. Finally, the checks that run after the solver
# (on the original, unscaled data) can overturn its verdict; the log then
# ends with a `POSTSOLVE -- status changed` line, so the last line printed
# is always the status you get back.
#
# ## The same log across cones
#
# Nothing about the log is specific to the nonnegative orthant. Here is a
# second-order cone program — maximize `1ᵀy` over the unit ball, whose answer
# is `1/√3` in each coordinate — with a cone block `("Q", 4)` in place of
# `("R", 3)`:

n = 3
Qsoc = spzeros(n, n)                # no quadratic term
csoc = ones(n)                      # maximize 1ᵀy

## SOC block: (1, y₁, y₂, y₃) ∈ Q⁴, i.e. ‖y‖ ≤ 1
Asoc = [spzeros(1, n); sparse(1.0I, n, n)]
bsoc = [-1.0; zeros(n)]

solsoc = conicIP(Qsoc, csoc, Asoc, bsoc, [("Q", n + 1)]; verbose = true,
                 log_style = :classic)
println("status: ", solsoc.status)

# The optimizer, as expected:

round.(solsoc.y, digits = 4)

# Same columns, same shape, same exit line: the Nesterov–Todd scaling makes
# the second-order cone look like the orthant to the outer loop. Mixing cone
# blocks — `[("R", 3), ("Q", 4)]` — changes nothing about how the log reads.
#
# ## The rich style
#
# `log_style = :rich` (the default in an interactive terminal) prints the
# same iteration as a Unicode table. Here it is for the first problem, with
# colour off:

sol = conicIP(Q, c, A, b, cone_dims; verbose = true, log_style = :rich);

# What is different from the classic table:
#
# - a four-line summary of the problem: sizes, a cone summary, the KKT
#   solver (and whether it was chosen automatically), and the tolerances;
# - `primal` is the full primal measure the solver reports as `prFeas`
#   (cone rows *and* equality rows), and `gap` is the relative duality gap
#   `|vᵀs| / (1 + |pobj|)`, the third stopping test, which the classic table
#   does not show;
# - `step` is the length of the step that produced the row's iterate
#   (`·` on the initial point), shown in yellow when it is below 0.1;
# - in colour, residuals turn green once they pass `optTol`, and a step
#   whose KKT residual is still large after refinement marks its `ref`
#   cell with a red `!`;
# - `progress` is how far the stopping measure has fallen from its first
#   value toward `optTol`, on a log scale, so the column reads as a chart
#   of convergence. When an infeasibility screen gets close to nominating
#   a ray the cell shows that screen (`p 3.4e-04`, with `⇢` once it is
#   handed to the validator); when the aggregate residuals pass but a
#   row-wise feasibility test does not, it shows `rows …`.
#
# The `kkt` and `cc` columns appear only when they carry information, and
# `time` and `progress` are dropped on a narrow terminal. The card at the
# end gives the verdict, the objective, sparklines of each residual and of
# μ and the step length over the run, and the totals; with
# `timing = ConicIP.PhaseTimes()` it adds where the time went.
#
# `ENV["CONICIP_LOG_STYLE"] = "classic"` (or `"rich"`) changes the default
# without touching code, which is handy under JuMP; `NO_COLOR` turns
# colour off.
#
# ## The iteration history as data
#
# Everything the log prints is also available as data. Pass a
# [`SolverTrace`](@ref ConicIP.SolverTrace) as `trace` and every event lands
# in it as a `NamedTuple`, whether or not the log is printed:

tr = SolverTrace()
sol = conicIP(Q, c, A, b, cone_dims; verbose = false, trace = tr)
[(r.iter, r.rDu, r.alpha) for r in ConicIP.iterations(tr)]

# Each iteration record carries every residual of the stopping test
# (including the gap and row-wise tests), the objectives, μ, both
# infeasibility screens, the step length and centering parameter, the
# refinement count and residual, the KKT diagnostics and the correctors;
# the `final` record holds the status the caller received. `trace` also
# accepts any function of one record, or a file path, which appends JSON
# Lines (`write_jsonl` saves a collected trace the same way). Setting
# `ENV["CONICIP_TRACE"] = "run.jsonl"` traces every solve, including those
# made through JuMP, and two runs can then be compared with standard tools:
#
# ```
# jq -c 'select(.event == "iter") | [.iter, .rPr, .rDu, .alpha]' run.jsonl
# ```
#
# ## Debug diagnostics
#
# Rare internal events — how the KKT solver was chosen, pivots repaired by
# regularization, refinement corrections rolled back, line-search
# back-offs, a ray nominated and rejected, a verdict overturned by
# postsolve — are reported through Julia's logging at the `Debug` level.
# They cost nothing until you ask for them:
#
# ```julia
# ENV["JULIA_DEBUG"] = "ConicIP"
# ```
#
# ## Where to go next
#
# - [Detecting Infeasibility](@ref) — the logs that end in a certificate
# - [Reading Residuals](@ref) — what to do when one residual lags the others
# - [Troubleshooting Solver Output](@ref) — every exit status and its remedy
