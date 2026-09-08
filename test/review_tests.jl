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
