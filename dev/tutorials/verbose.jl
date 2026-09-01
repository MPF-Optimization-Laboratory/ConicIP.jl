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

sol = conicIP(Q, c, A, b, cone_dims; verbose = true)
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
# ## The exit line
#
# The log ends with a one-line verdict. `EXIT -- Below Tolerance!` means all
# three optimality residuals cleared `optTol`, and the status returned is
# `:Optimal`. The other exits — `Certificate of Infeasiblity Found!`,
# `Certificate of Dual Infeasibility Found!`, and `Error!` — are the subject
# of the next page.
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

solsoc = conicIP(Qsoc, csoc, Asoc, bsoc, [("Q", n + 1)]; verbose = true)
println("status: ", solsoc.status)

# The optimizer, as expected:

round.(solsoc.y, digits = 4)

# Same columns, same shape, same exit line: the Nesterov–Todd scaling makes
# the second-order cone look like the orthant to the outer loop. Mixing cone
# blocks — `[("R", 3), ("Q", 4)]` — changes nothing about how the log reads.
#
# ## Where to go next
#
# - [Detecting Infeasibility](@ref) — the logs that end in a certificate
# - [Reading Residuals](@ref) — what to do when one residual lags the others
# - [Troubleshooting Solver Output](@ref) — every exit status and its remedy
