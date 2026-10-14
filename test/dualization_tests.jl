# The Dualization.jl route for matrix-variable SDPs (docs/src/guides/sdp_formulation.md).
#
# A PSD *variable* (`VectorOfVariables`-in-PSD, JuMP's `@variable(X, PSD)`)
# gives the solver one variable per matrix entry, and the dense KKT path
# then costs O(k⁶) in the block order k. `Dualization.dual_optimizer`
# hands ConicIP the dual, whose variables are the equality constraints.
# These tests pin that the route is faithful: the same optimum, the same
# constraint duals, and correctly mapped infeasibility statuses and
# certificates. Comparisons of iterates use the solve tolerance, and the
# dual values are compared only on instances whose duals are unique.

using Dualization
import MathOptInterface as MOI

@testset "Dualization route for matrix-variable SDPs" begin

    tri(i, j) = i <= j ? div(j * (j - 1), 2) + i : tri(j, i)
    saf(terms, c = 0.0) = MOI.ScalarAffineFunction(terms, c)
    tm(c, x) = MOI.ScalarAffineTerm(c, x)

    function solve(model, optimizer)
        opt = MOI.instantiate(optimizer; with_bridge_type = Float64)
        MOI.set(opt, MOI.Silent(), true)
        index_map = MOI.copy_to(opt, model)
        MOI.optimize!(opt)
        return (opt, index_map)
    end
    routes = (ConicIP.Optimizer, Dualization.dual_optimizer(ConicIP.Optimizer))

    # max ⟨L, X⟩/4 s.t. X_ii = 1, X ⪰ 0, on a 5-cycle (unique optimum).
    function maxcut_c5()
        k = 5
        W = zeros(k, k)
        for i in 1:k
            j = mod1(i + 1, k); W[i, j] = W[j, i] = 1.0
        end
        L = Diagonal(vec(sum(W; dims = 2))) - W
        model = MOI.Utilities.Model{Float64}()
        X = MOI.add_variables(model, div(k * (k + 1), 2))
        cX = MOI.add_constraint(model, MOI.VectorOfVariables(X),
                                MOI.PositiveSemidefiniteConeTriangle(k))
        ceq = [MOI.add_constraint(model, saf([tm(1.0, X[tri(i, i)])]), MOI.EqualTo(1.0))
               for i in 1:k]
        obj = [tm((i == j ? 1.0 : 2.0) * L[i, j] / 4, X[tri(i, j)])
               for j in 1:k for i in 1:j if L[i, j] != 0]
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf(obj))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        return (model = model, X = X, cX = cX, ceq = ceq, k = k)
    end

    @testset "same optimum, primal values and duals" begin
        inst = maxcut_c5()
        res = map(routes) do r
            (opt, map_) = solve(inst.model, r)
            (status = MOI.get(opt, MOI.TerminationStatus()),
             pstat = MOI.get(opt, MOI.PrimalStatus()),
             dstat = MOI.get(opt, MOI.DualStatus()),
             obj = MOI.get(opt, MOI.ObjectiveValue()),
             dobj = MOI.get(opt, MOI.DualObjectiveValue()),
             x = [MOI.get(opt, MOI.VariablePrimal(), map_[v]) for v in inst.X],
             yeq = [MOI.get(opt, MOI.ConstraintDual(), map_[c]) for c in inst.ceq],
             S = MOI.get(opt, MOI.ConstraintDual(), map_[inst.cX]))
        end
        direct, dualized = res
        for r in res
            @test r.status == MOI.OPTIMAL
            @test r.pstat == MOI.FEASIBLE_POINT
            @test r.dstat == MOI.FEASIBLE_POINT
        end
        # The max-cut value of C₅ under the SDP relaxation: (25 + 5√5)/8.
        θ = (25 + 5 * sqrt(5)) / 8
        @test dualized.obj ≈ θ rtol = 1e-6
        @test dualized.dobj ≈ θ rtol = 1e-6
        @test dualized.obj ≈ direct.obj rtol = 1e-6
        @test dualized.x ≈ direct.x atol = 1e-5
        @test dualized.yeq ≈ direct.yeq atol = 1e-5
        @test dualized.S ≈ direct.S atol = 1e-5
        # The dualized route's primal point is feasible in the original model.
        Xm = zeros(inst.k, inst.k)
        for j in 1:inst.k, i in 1:j
            Xm[i, j] = Xm[j, i] = dualized.x[tri(i, j)]
        end
        @test maximum(abs.(diag(Xm) .- 1)) <= 1e-6
        @test eigmin(Symmetric(Xm)) >= -1e-6
    end

    @testset "affine PSD constraint: same point and PSD dual" begin
        # max y₁ + y₂ s.t. C − y₁E₁₁ − y₂E₂₂ ⪰ 0, optimum y = (1, 2).
        model = MOI.Utilities.Model{Float64}()
        y = MOI.add_variables(model, 2)
        f = MOI.VectorAffineFunction(
            [MOI.VectorAffineTerm(1, tm(-1.0, y[1])), MOI.VectorAffineTerm(3, tm(-1.0, y[2]))],
            [2.0, 1.0, 3.0])
        c = MOI.add_constraint(model, f, MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([tm(1.0, y[1]), tm(1.0, y[2])]))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        out = map(routes) do r
            (opt, map_) = solve(model, r)
            (MOI.get(opt, MOI.TerminationStatus()),
             [MOI.get(opt, MOI.VariablePrimal(), map_[v]) for v in y],
             MOI.get(opt, MOI.ConstraintDual(), map_[c]))
        end
        for (st, yv, S) in out
            @test st == MOI.OPTIMAL
            @test yv ≈ [1.0, 2.0] atol = 1e-5
            @test S ≈ [1.0, -1.0, 1.0] atol = 1e-5
        end
    end

    @testset "infeasibility statuses and certificates map back" begin
        # Primal infeasible: X ⪰ 0 with X₁₁ = −1.
        model = MOI.Utilities.Model{Float64}()
        X = MOI.add_variables(model, 3)
        MOI.add_constraint(model, MOI.VectorOfVariables(X), MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.add_constraint(model, saf([tm(1.0, X[1])]), MOI.EqualTo(-1.0))
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([tm(1.0, X[1]), tm(1.0, X[3])]))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        for r in routes
            (opt, _) = solve(model, r)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
            @test MOI.get(opt, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
        end

        # Unbounded: min −X₁₁ s.t. X₂₂ = 1, X ⪰ 0.
        model = MOI.Utilities.Model{Float64}()
        X = MOI.add_variables(model, 3)
        MOI.add_constraint(model, MOI.VectorOfVariables(X), MOI.PositiveSemidefiniteConeTriangle(2))
        MOI.add_constraint(model, saf([tm(1.0, X[3])]), MOI.EqualTo(1.0))
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([tm(-1.0, X[1])]))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        for r in routes
            (opt, map_) = solve(model, r)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
            # The ray is a recession direction: PSD, X₂₂ = 0, and it
            # decreases the objective.
            ray = [MOI.get(opt, MOI.VariablePrimal(), map_[v]) for v in X]
            R = [ray[1] ray[2]; ray[2] ray[3]]
            @test eigmin(Symmetric(R)) >= -1e-6 * norm(ray)
            @test abs(ray[3]) <= 1e-6 * norm(ray)
            @test -ray[1] < 0
        end
    end
end
