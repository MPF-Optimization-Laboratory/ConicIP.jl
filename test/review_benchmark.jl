module ReviewBenchmark
using Test
include("../benchmark/suite.jl")

@testset "Review: independent benchmark residuals" begin
    src = MOI.Utilities.Model{Float64}()
    x = MOI.add_variable(src)
    MOI.add_constraint(src, x, MOI.GreaterThan(1.0))
    MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction([MOI.ScalarQuadraticTerm(2.0,x,x)],
            MOI.ScalarAffineTerm{Float64}[], 0.0))
    raw = ConicIP.Optimizer()
    MOI.optimize!(raw, src)
    @test raw.cone_dims == [("R",1)]
    prob = (Q=raw.Q_int, c=raw.c_int, A=raw.ineq_A, b=raw.ineq_b,
            G=raw.eq_G, d=raw.eq_d, cone_dims=raw.cone_dims)
    @test verified(raw.sol, residuals(prob, raw.sol))
    # Solver diagnostics and objective fields cannot determine verification.
    raw.sol.duFeas = 0.0; raw.sol.prFeas = 0.0; raw.sol.dobj = raw.sol.pobj
    raw.sol.w = zeros(0); raw.sol.v .= -1.0
    res = residuals(prob, raw.sol)
    @test res.rDu > 0.1
    @test res.dual_margin < 0
    @test !verified(raw.sol, res)
    MOI.empty!(raw)
    @test isempty(raw.cone_dims)
end
end
