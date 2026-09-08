using Test, ConicIP, LinearAlgebra, SparseArrays

@testset "Review: original-data postsolve" begin
    Q = sparse(1.0I, 2, 2); A = copy(Q)
    c = zeros(2); b = zeros(2); cd = [("R", 2)]
    # The fixed-value tolerance is looser than the requested solve tolerance.
    G = sparse([1.0 0; 1 0]); d = [1.0, 1 + 1e-9]
    sol = preprocess_conicIP(Q, c, A, b, cd, G, d;
                            optTol = 1e-12, verbose = false)
    @test sol.status == :Error
    @test occursin("original-data", sol.message)
    rEq = norm(G*sol.y-d)/(1+max(norm(d),norm(abs.(G)*abs.(sol.y))))
    @test sol.prFeas >= rEq > 1e-12

    # Rank detection can also drop approximately dependent rows.
    G = sparse([1.0 1; 1 1]); d = [1.0, 1 + 1e-9]
    sol = preprocess_conicIP(Q, c, A, b, cd, G, d;
        fix_singletons = false, rank_check = :always, optTol = 1e-12, verbose = false)
    @test sol.status == :Error
    @test sol.prFeas > 1e-12

    # Full Q times a lifted recession ray includes the fixed columns.
    # The reduced Q can accept a numerical null vector that the full Q rejects.
    Q = sparse([1e-10 1.0; 1.0 1e10])
    G = sparse([0.0 1.0]); d = [0.0]; c = [1.0, 0.0]
    sol = preprocess_conicIP(Q, c, A, b, cd, G, d; verbose = false)
    @test !sol.has_certificate
    @test sol.status == :Abandoned
    @test occursin("presolved data", sol.message)
end

@testset "Review: cached LDL rank policy" begin
    Q = sparse(1.0I, 2, 2); A = copy(Q)
    G = sparse([1.0 1; 2 2]); d = [1.0, 2.0]
    ks = ConicIP.cached_kktsolver_ldl()
    sol = preprocess_conicIP(Q, zeros(2), A, zeros(2), [("R", 2)], G, d;
        kktsolver = ks, verbose = false)
    @test sol.status == :Optimal
    @test ks.key[3] == size(G) # Both rows reached LDL; no rank reduction.
end

@testset "Review: finite normalized certificates" begin
    Q = spzeros(1, 1); A = spzeros(0, 1); b = zeros(0)
    cd = Tuple{String,Int}[]; G = spzeros(0, 1); d = zeros(0)
    chk, ray = ConicIP.validate_unboundedness_certificate(
        Q, [1e-320], A, b, cd, G, d, [1.0]; abstol = 1e-9, reltol = 1e-7)
    @test !chk.valid && !chk.finite
    @test all(isfinite, ray)
    chk, w, v = ConicIP.validate_infeasibility_certificate(
        Q, [0.0], A, b, cd, spzeros(1,1), [1e-320], [-1.0], zeros(0);
        abstol = 1e-9, reltol = 1e-7)
    @test !chk.valid && !chk.finite
    @test all(isfinite, w)
    # Failure to represent the normalized certificate must not make a
    # contradictory zero row (or improving zero column) disappear.
    for eq in (false, true)
        sol = conicIP(Q, [1e-320], A, b, cd; equilibrate = eq, verbose = false)
        @test sol.status == :Error && !sol.has_certificate
        sol = conicIP(Q, [0.0], A, b, cd, spzeros(1,1), [1e-320];
                      equilibrate = eq, verbose = false)
        @test sol.status == :Error && !sol.has_certificate
    end
end

@testset "Review: bounded Ruiz sweeps" begin
    # Row 1 and column 1 saturate at 0.1 on the first sweep. Column 2
    # already has norm 1, and must stay there while row 2 is balanced.
    A = sparse([1e12 100.0; 1.0 1.0])
    eq = ConicIP.equilibrate_conicIP(spzeros(2,2), zeros(2), A, zeros(2),
        [("R",2)], spzeros(0,2), zeros(0); bound = 10.0, iters = 10)
    @test eq.Dc ≈ [0.1, 0.1]
    @test eq.Dr[1] ≈ 0.1
    @test maximum(abs, eq.A[:,2]) ≈ 1.0
    @test all(x -> 0.1 <= x <= 10.0, [eq.Dc; eq.Dr])
    @test eq.A ≈ Diagonal(eq.Dr)*A*Diagonal(eq.Dc)
end

@testset "Review: exhausted time budgets" begin
    Q = sparse(1.0I, 2, 2); A = copy(Q); c = ones(2); b = zeros(2)
    cd = [("R",2)]; G = spzeros(0,2); d = zeros(0)
    # The post-loop path must honor time even when there was no loop check.
    sol = conicIP(Q, c, A, b, cd; maxIters = 0, timeLimit = 0.0, verbose = false)
    @test sol.status == :TimeLimit
    @test all(isfinite, sol.y)
    never_called(args...) = error("expired fallback started a KKT solver")
    @test ConicIP.fallback_infeasibility_ray(Q,c,A,b,cd,G,d;
        timeLimit = 0.0, kktsolver = never_called) === nothing
    @test ConicIP.fallback_unbounded_ray(Q,c,A,b,cd,G,d;
        timeLimit = 0.0, kktsolver = never_called) === nothing
end

@testset "Review: objective values and convexity" begin
    # An unrelated huge positive block must not regularize away a
    # negative direction in the MOI convexity check.
    @test !ConicIP._is_psd(sparse(Diagonal([-1.0, 1e16])))
    @test !ConicIP._is_psd(sparse([1.0 2 0; 2 1 0; 0 0 1e16]))
    @test ConicIP._is_psd(sparse([1e-16 1.0; 1.0 1e16]))
    Q = sparse([2.0 0; 0 3]); c = [1.0, 2.0]
    A = sparse([1.0 0; 0 1; -1 0]); b = [1.0, 0.0, -2.0]
    G = sparse([1.0 1]); d = [3.0]
    for eq in (false, true)
        sol = conicIP(Q,c,A,b,[("R",3)],G,d;
            equilibrate = eq, maxIters = 1, certFallback = false, verbose = false)
        expected = -dot(sol.y,Q*sol.y)/2 - dot(d,sol.w) + dot(b,sol.v)
        @test sol.dobj ≈ expected
    end
    import MathOptInterface as MOI
    model = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.EqualTo(1.0))
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
        MOI.ScalarQuadraticFunction([MOI.ScalarQuadraticTerm(-2.0,x,x)],
            MOI.ScalarAffineTerm{Float64}[], 5.0))
    MOI.set(model, MOI.ObjectiveSense(), MOI.FEASIBILITY_SENSE)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    @test MOI.get(model, MOI.ObjectiveValue()) == 0.0
end
