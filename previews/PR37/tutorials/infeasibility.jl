# # Detecting Infeasibility
#
# A conic problem can fail to have a solution in two ways: the feasible set
# is empty, or the objective slides off to `-∞` along a direction that stays
# feasible forever. ConicIP does not merely *report* either verdict — it
# returns a short vector that proves it, and refuses to claim the status
# unless that vector passes a check against the problem data you supplied.
#
# This page runs both cases with `verbose = true`, reads the exit lines, and
# then verifies the returned rays by hand. It assumes the column layout
# introduced in [Reading the Iteration Log](@ref).

using ConicIP, SparseArrays, LinearAlgebra

# ## An infeasible problem
#
# Ask for a box that contradicts itself: every coordinate at least `1` and at
# most `0`. Written as `Ay ≥ b`, that is `y ≥ 1` stacked on `-y ≥ 0`. The
# objective is a strictly convex quadratic; its exact form does not matter,
# because no point satisfies the constraints.

n = 2
Q = Matrix(0.5I, n, n)
c = ones(n)

A = [sparse(1.0I, n, n)      # y ≥ 1
     -sparse(1.0I, n, n)]    # -y ≥ 0, i.e. y ≤ 0
b = [ones(n); zeros(n)]
cone_dims = [("R", 2n)]

G = spzeros(0, n)            # no equality constraints
d = Float64[]

sol = conicIP(Q, c, A, b, cone_dims, G, d; verbose = true)
println("status: ", sol.status)

# The `icertp` column tells the story: it starts at `1.5e+00`, falls by one
# or two orders of magnitude per iteration, and once it drops under `infeasTol`
# the nominated ray is handed to the validator, which accepts it — hence
# `EXIT -- Certificate of Infeasiblity Found!`. Meanwhile `muFeas` blows up
# and `pobj` stalls: the iterates are running off to infinity, which is what
# an infeasible-start interior-point method does on an empty feasible set.
#
# ## Checking the certificate by hand
#
# When the status is claimed with a certificate, `sol.w` and `sol.v` hold the
# Farkas ray and the primal fields are `NaN`:

sol.has_certificate

#

(w = sol.w, v = sol.v, y = sol.y)

# The ray is normalized so that three facts hold simultaneously. First, the
# ray separates:

dot(d, sol.w) - dot(b, sol.v)     # dᵀw̄ - bᵀv̄ = -1

# Second, it annihilates the constraint operator:

norm(G' * sol.w - A' * sol.v)     # ‖Gᵀw̄ - Aᵀv̄‖ ≈ 0

# Third, the inequality multipliers lie in the cone — here the nonnegative
# orthant, so every component must be nonnegative:

all(sol.v .>= 0)

# Together these three facts are a proof. Scaling the constraints `Ay ≥ b` by
# the nonnegative weights `v̄` and the equalities `Gy = d` by `w̄` and adding
# them up gives `(Gᵀw̄ - Aᵀv̄)ᵀy ≤ dᵀw̄ - bᵀv̄` for any feasible `y`; the second
# fact makes the left side `0` and the first makes the right side `-1`. So a
# feasible `y` would give `0 ≤ -1`, and no such `y` exists.
#
# ## An unbounded problem
#
# Now drop the upper bound and push the objective the other way: minimize
# `-1ᵀy` over `y ≥ 0`. Every ray of the orthant is a direction of descent.

Qu = spzeros(n, n)
cu = ones(n)
Au = sparse(1.0I, n, n)
bu = zeros(n)

solu = conicIP(Qu, cu, Au, bu, [("R", n)], G, d; verbose = true)
println("status: ", solu.status)

# This one is settled on the first iteration: `icertd` reads `0.0e+00`, the
# recession screen fires immediately, and the exit line is
# `EXIT -- Certificate of Dual Infeasibility Found!`. This time it is `y`
# that carries the ray, and the duals that are `NaN`:

(y = solu.y, w = solu.w, has_certificate = solu.has_certificate)

# The recession ray is normalized to unit rate of descent,

dot(cu, solu.y)                   # cᵀȳ = +1

# it is flat for the quadratic term (and for the equalities, of which there
# are none here),

(norm(Qu * solu.y), norm(G * solu.y))    # Qȳ ≈ 0, Gȳ ≈ 0

# and it points into the cone:

minimum(Au * solu.y)              # cone margin of Aȳ, must be ≥ 0

# So from any feasible point, moving along `ȳ` stays feasible forever while
# the objective drops at unit rate — the definition of unbounded.
#
# ## When the loop runs out of iterations
#
# The screens above ride on the current iterate, so a solve that is cut short
# can end with the ray still out of focus. Give the infeasible box seven
# iterations instead of the eight it wanted:

solfb = conicIP(Q, c, A, b, cone_dims, G, d; verbose = true, maxIters = 7)
println("status: ", solfb.status, "   has_certificate: ", solfb.has_certificate)

# The verdict survives. At iteration 7 the screen is at `1.8e-07`, just above
# `infeasTol`, so nothing was nominated inside the loop; the answer comes
# from `certFallback`, which solves a small auxiliary minimum-norm problem
# whose feasible set is exactly the set of Farkas rays, then puts the result
# through the same validator as any other candidate. Turning it off shows
# what the loop alone had:

conicIP(Q, c, A, b, cone_dims, G, d;
        verbose = false, maxIters = 7, certFallback = false).status

# The recovered ray is a certificate on the same terms as before — nothing is
# accepted on the fallback's word:

(dot(d, solfb.w) - dot(b, solfb.v),
 norm(G' * solfb.w - A' * solfb.v),
 all(solfb.v .>= 0))

# ## Where to go next
#
# - [The Certificate Pipeline](@ref) — the screens, the validators, and the
#   tolerances that separate a candidate from a claim
# - [Troubleshooting Solver Output](@ref) — `:AlmostInfeasible`,
#   `:Abandoned`, and what to try next
# - [`validate_infeasibility_certificate`](@ref ConicIP.validate_infeasibility_certificate)
#   and [`validate_unboundedness_certificate`](@ref ConicIP.validate_unboundedness_certificate)
#   — the same checks, callable on any ray you like
