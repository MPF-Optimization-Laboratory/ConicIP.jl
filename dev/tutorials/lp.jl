# # Linear Programs
#
# A linear program (LP) is a special case of the ConicIP problem formulation
# with `Q = 0`:
#
# ```
# minimize    -cᵀy
# subject to   Ay ≥ b
#              Gy  = d
# ```
#
# ## Example: LP with Equality and Inequality Constraints
#
# Solve a small LP with nonnegativity and an equality constraint:
#
# ```
# minimize    -2y₁ - 3y₂ - y₃ - y₄ - y₅
# subject to   y₁ + y₂ + y₃ + y₄ + y₅ = 4
#              y ≥ 0
# ```

using ConicIP, SparseArrays, LinearAlgebra

n = 5
Q = spzeros(n, n)
c = reshape([2.0, 3.0, 1.0, 1.0, 1.0], :, 1)

## Nonnegativity: y ≥ 0
A = sparse(1.0I, n, n)
b = zeros(n, 1)
cone_dims = [("R", n)]

## Equality constraint: sum(y) = 4
G = ones(1, n)
d = reshape([4.0], 1, 1)

sol = conicIP(Q, c, A, b, cone_dims, G, d; verbose=false)
sol.status

#

round.(sol.y, digits=4)

# The optimal solution puts all weight on the variable with the largest
# objective coefficient (y₂ = 4).
#
# ## Reading Dual Variables
#
# The dual variables `sol.v` give the shadow prices for the inequality
# constraints. Each component of `v` corresponds to one row of `A y ≥ b`.

round.(sol.v, digits=4)

# The dual variable for the equality constraint is in `sol.w`:

round.(sol.w, digits=4)

# ## Verifying Optimality
#
# At optimality, the primal and dual objectives should match:

(pobj=round(sol.pobj, digits=6), dobj=round(sol.dobj, digits=6))

# And all feasibility residuals should be small:

(prFeas=sol.prFeas, duFeas=sol.duFeas, muFeas=sol.muFeas)

# ## A Larger Example: Transportation Problem
#
# Consider a transportation problem with 3 supply nodes and 4 demand nodes.
# The cost of shipping one unit from supply node `i` to demand node `j` is
# `cost[i,j]`. We want to minimize total shipping cost subject to supply
# and demand constraints.

using Random
Random.seed!(123)

nsupply, ndemand = 3, 4
cost = rand(nsupply, ndemand) .+ 0.1

## Decision variables: x[i,j] = amount shipped from i to j
nvar = nsupply * ndemand
Q = spzeros(nvar, nvar)
c = reshape(cost, :, 1)         # minimize (note: conicIP minimizes -c'y, so negate)
c = -c                           # we want to minimize cost'y, so set c = -cost

## Nonnegativity
A = sparse(1.0I, nvar, nvar)
b_ineq = zeros(nvar, 1)
cone_dims = [("R", nvar)]

## Supply constraints: sum_j x[i,j] = supply[i]
## Demand constraints: sum_i x[i,j] = demand[j]
supply = [10.0, 15.0, 20.0]
demand = [8.0, 12.0, 10.0, 15.0]  # sum(demand) = sum(supply) = 45

G_supply = zeros(nsupply, nvar)
for i in 1:nsupply
    for j in 1:ndemand
        G_supply[i, (i-1)*ndemand + j] = 1.0
    end
end

G_demand = zeros(ndemand, nvar)
for j in 1:ndemand
    for i in 1:nsupply
        G_demand[j, (i-1)*ndemand + j] = 1.0
    end
end

G = [G_supply; G_demand]
d = reshape([supply; demand], :, 1)

sol2 = conicIP(Q, c, A, b_ineq, cone_dims, G, d; verbose=false)
sol2.status

# The optimal shipping plan:

shipments = reshape(round.(sol2.y, digits=2), nsupply, ndemand)
