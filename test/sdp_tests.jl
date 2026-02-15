# ──────────────────────────────────────────────────────────────
#  SDP Test Suite
#
#  Unit tests for SDP internals, standard SDP problem instances,
#  edge cases, and evaluation harness integration.
# ──────────────────────────────────────────────────────────────

include("sdp_problems.jl")
include("sdp_harness.jl")

@testset "SDP Suite" begin

# ──────────────────────────────────────────────────────────────
#  Unit Tests: vecm / mat
# ──────────────────────────────────────────────────────────────

@testset "vecm/mat round-trip" begin
    for n in [1, 2, 3, 5, 8, 10]
        Random.seed!(n)
        X = randn(n, n)
        X = (X + X') / 2  # symmetric
        v = ConicIP.vecm(X)
        X_rec = ConicIP.mat(v)
        @test X_rec ≈ X atol=1e-12
        @test length(v) == _vdim(n)
    end
end

@testset "vecm inner product preserves trace" begin
    for n in [2, 3, 5, 8]
        Random.seed!(100 + n)
        X = randn(n, n); X = (X + X') / 2
        Y = randn(n, n); Y = (Y + Y') / 2
        @test dot(ConicIP.vecm(X), ConicIP.vecm(Y)) ≈ tr(X * Y) atol=1e-10
    end
end

@testset "vecm(I) is cone identity" begin
    for n in [2, 4, 6]
        I_n = Matrix{Float64}(I, n, n)
        e = ConicIP.vecm(I_n)
        # cone product with identity: X ∘ e = X
        Random.seed!(200 + n)
        M = randn(n, n)
        X = M' * M + I  # PSD
        x = reshape(ConicIP.vecm(X), :, 1)
        e_col = reshape(e, :, 1)
        o = zeros(length(x), 1)
        ConicIP.xsdc!(x, e_col, o)
        # X ∘ I = X*I + I*X = 2X
        @test ConicIP.mat(o) ≈ 2 * X atol=1e-10
    end
end

@testset "ord() inverse of vdim" begin
    for n in [1, 2, 3, 5, 10, 15, 20]
        k = _vdim(n)
        x = zeros(k)
        @test ConicIP.ord(x) == n
    end
end

# ──────────────────────────────────────────────────────────────
#  Unit Tests: VecCongurance
# ──────────────────────────────────────────────────────────────

@testset "VecCongurance operations" begin
    Random.seed!(42)
    for n in [2, 3, 5]
        k = _vdim(n)
        R = randn(n, n) + n * I  # well-conditioned
        W = ConicIP.VecCongurance(R)

        x = randn(k, 1)
        X = ConicIP.mat(x)

        # W*x should equal vecm(R'*X*R)
        expected = ConicIP.vecm(R' * X * R)
        @test W * x ≈ expected atol=1e-10

        # inv(W) * W * x ≈ x
        Winv = inv(W)
        @test Winv * (W * x) ≈ x atol=1e-9

        # adjoint: (W'*x) should equal vecm(R*X*R')
        Wadj = W'
        expected_adj = ConicIP.vecm(R * X * R')
        @test Wadj * x ≈ expected_adj atol=1e-10

        # Matrix conversion
        Wmat = Matrix(W)
        @test Wmat * vec(x) ≈ vec(W * x) atol=1e-10

        # Sparse conversion
        Wsp = sparse(W)
        @test Wsp ≈ Wmat atol=1e-10

        # Size
        @test size(W, 1) == k

        # Composition: W1 * W2
        R2 = randn(n, n) + n * I
        W2 = ConicIP.VecCongurance(R2)
        x_comp = randn(k, 1)
        @test (W * W2) * x_comp ≈ W * (W2 * x_comp) atol=1e-9
    end
end

# ──────────────────────────────────────────────────────────────
#  Unit Tests: Nesterov-Todd Scaling
# ──────────────────────────────────────────────────────────────

@testset "Nesterov-Todd scaling identity: F*z ≈ inv(F)*s" begin
    for n in [2, 3, 4, 5]
        Random.seed!(300 + n)
        k = _vdim(n)

        # Generate strictly feasible points (interior of PSD cone)
        M1 = randn(n, n)
        Z = M1' * M1 + I   # strictly PSD
        M2 = randn(n, n)
        S = M2' * M2 + I   # strictly PSD

        z = reshape(ConicIP.vecm(Z), :, 1)
        s = reshape(ConicIP.vecm(S), :, 1)

        F = ConicIP.nestod_sdc(z, s)

        Fz  = F * z
        Fis = inv(F) * s

        @test Fz ≈ Fis atol=1e-8
    end
end

@testset "NT scaling: λ = F*z is in interior of PSD cone" begin
    for n in [2, 3, 5]
        Random.seed!(400 + n)
        M1 = randn(n, n); Z = M1' * M1 + I
        M2 = randn(n, n); S = M2' * M2 + I
        z = reshape(ConicIP.vecm(Z), :, 1)
        s = reshape(ConicIP.vecm(S), :, 1)

        F = ConicIP.nestod_sdc(z, s)
        λ = F * z
        Λ_mat = ConicIP.mat(λ)

        # λ should be PSD (all eigenvalues > 0)
        eigs = eigvals(Symmetric(Λ_mat))
        @test all(eigs .> -1e-10)
    end
end

# ──────────────────────────────────────────────────────────────
#  Unit Tests: Cone Arithmetic
# ──────────────────────────────────────────────────────────────

@testset "SDP cone product xsdc!" begin
    for n in [2, 3, 4]
        Random.seed!(500 + n)
        k = _vdim(n)

        X = randn(n, n); X = (X + X') / 2
        Y = randn(n, n); Y = (Y + Y') / 2

        x = reshape(ConicIP.vecm(X), :, 1)
        y = reshape(ConicIP.vecm(Y), :, 1)
        o = zeros(k, 1)

        ConicIP.xsdc!(x, y, o)

        # x ∘ y = X*Y + Y*X (symmetric product)
        expected = ConicIP.vecm(X * Y + Y * X)
        @test o ≈ expected atol=1e-10
    end
end

@testset "SDP cone division dsdc!" begin
    for n in [2, 3, 4]
        Random.seed!(600 + n)
        k = _vdim(n)

        # Y must be in the interior of PSD cone for Lyapunov to be well-defined
        M = randn(n, n)
        Y = M' * M + I  # strictly PSD
        X = randn(n, n); X = (X + X') / 2

        x = reshape(ConicIP.vecm(X), :, 1)
        y = reshape(ConicIP.vecm(Y), :, 1)
        o = zeros(k, 1)

        ConicIP.dsdc!(x, y, o)
        O_mat = ConicIP.mat(o)

        # Division Z = X ÷ Y means Y*Z + Z*Y = X
        residual = Y * O_mat + O_mat * Y - X
        @test norm(residual, Inf) < 1e-9
    end
end

@testset "SDP cone product/division inverse relationship" begin
    for n in [2, 3, 4]
        Random.seed!(700 + n)
        k = _vdim(n)

        # Y strictly PSD, X arbitrary symmetric
        M = randn(n, n); Y = M' * M + I
        X = randn(n, n); X = (X + X') / 2

        x = reshape(ConicIP.vecm(X), :, 1)
        y = reshape(ConicIP.vecm(Y), :, 1)
        o_div = zeros(k, 1)
        o_prod = zeros(k, 1)

        # Z = X ÷ Y, then Y ∘ Z should ≈ X
        ConicIP.dsdc!(x, y, o_div)
        ConicIP.xsdc!(y, o_div, o_prod)
        @test o_prod ≈ x atol=1e-8
    end
end

# ──────────────────────────────────────────────────────────────
#  Unit Tests: Line Search
# ──────────────────────────────────────────────────────────────

@testset "maxstep_sdc: PSD point returns finite step" begin
    Random.seed!(800)
    n = 3
    X = Matrix{Float64}(I, n, n)  # strictly PSD
    D = -0.5 * I + 0.1 * randn(n, n)
    D = (D + D') / 2

    x = reshape(ConicIP.vecm(X), :, 1)
    d = reshape(ConicIP.vecm(D), :, 1)

    α = ConicIP.maxstep_sdc(x, d)
    @test isfinite(α)
    @test α > 0

    # At step α, the matrix should be on the boundary (smallest eigenvalue ≈ 0)
    X_boundary = ConicIP.mat(x - α * d)
    eigs = eigvals(Symmetric(X_boundary))
    @test minimum(eigs) ≈ 0 atol=1e-6
end

@testset "maxstep_sdc: non-PSD point returns Inf" begin
    n = 3
    X = -Matrix{Float64}(I, n, n)  # negative definite
    D = Matrix{Float64}(I, n, n)
    @test ConicIP.maxstep_sdc(ConicIP.vecm(X), ConicIP.vecm(D)) == Inf
end

@testset "maxstep_sdc: initial point feasibility check" begin
    # Strictly feasible point → returns 0
    n = 3
    X = 2.0 * Matrix{Float64}(I, n, n)
    @test ConicIP.maxstep_sdc(ConicIP.vecm(X), nothing) == 0

    # Infeasible point → returns negative value
    X_bad = -Matrix{Float64}(I, n, n)
    α = ConicIP.maxstep_sdc(ConicIP.vecm(X_bad), nothing)
    @test α < 0
end

# ──────────────────────────────────────────────────────────────
#  Unit Tests: Block with VecCongurance
# ──────────────────────────────────────────────────────────────

@testset "Block with VecCongurance blocks" begin
    Random.seed!(900)

    # Single VecCongurance block
    n = 3; k = _vdim(n)
    R = randn(n, n) + n * I
    W = ConicIP.VecCongurance(R)
    B = Block(1); B[1] = W

    x = randn(k)
    @test B * x ≈ vec(W * reshape(x, :, 1)) atol=1e-10
    @test Matrix(B) ≈ Matrix(W) atol=1e-10
end

@testset "Block with mixed Diagonal + VecCongurance" begin
    Random.seed!(901)

    # R+ block (Diagonal) + SDP block (VecCongurance)
    n_r = 3; n_s = 2; k_s = _vdim(n_s)
    D = Diagonal(rand(n_r) .+ 0.5)
    R = randn(n_s, n_s) + n_s * I
    W = ConicIP.VecCongurance(R)

    B = Block(2); B[1] = D; B[2] = W

    x = randn(n_r + k_s)
    y = B * x

    @test y[1:n_r] ≈ D * x[1:n_r] atol=1e-10
    @test y[n_r+1:end] ≈ vec(W * reshape(x[n_r+1:end], :, 1)) atol=1e-10

    # inv and adjoint
    Binv = inv(B)
    @test Binv * (B * x) ≈ x atol=1e-8
    @test B' * x ≈ Matrix(B)' * x atol=1e-10
end

# ──────────────────────────────────────────────────────────────
#  Standard SDP Problems (direct API)
# ──────────────────────────────────────────────────────────────

@testset "SDP: PSD projection (n=6)" begin
    prob = sdp_psd_projection(n=6)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X - prob.known_X, Inf) < tol
    @test all(eigvals(Symmetric(X)) .> -tol)
end

@testset "SDP: PSD projection (n=8)" begin
    prob = sdp_psd_projection(n=8)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X - prob.known_X, Inf) < tol
end

@testset "SDP: Trace minimization (n=4)" begin
    prob = sdp_trace_minimization(n=4)
    r = sdp_solve(prob; optTol=optTol, verbose=false)
    @test r.status_match
    @test r.status == :Optimal
end

@testset "SDP: Nearest correlation matrix (n=5)" begin
    prob = sdp_nearest_correlation(n=5)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                  prob.G, prob.d;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    # Result should have unit diagonal
    X = ConicIP.mat(sol.y)
    @test all(abs.(diag(X) .- 1.0) .< tol)
    # Result should be PSD
    @test all(eigvals(Symmetric(X)) .> -tol)
end

@testset "SDP: Max-cut relaxation (n=5)" begin
    prob = sdp_max_cut_relaxation(n=5)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                  prob.G, prob.d;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    # Check diag(X) = 1
    X = ConicIP.mat(sol.v)
    @test all(abs.(diag(X) .- 1.0) .< tol)
    # Check PSD
    @test all(eigvals(Symmetric(X)) .> -tol)
end

@testset "SDP: Lovász theta (n=5)" begin
    prob = sdp_lovasz_theta(n=5)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                  prob.G, prob.d;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
end

@testset "SDP: Rank-1 solution (n=4)" begin
    prob = sdp_rank_one(n=4)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                  prob.G, prob.d;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    # Objective should match minimum eigenvalue
    @test abs(sol.pobj - prob.known_obj) < tol
end

@testset "SDP: Identity feasible (n=3)" begin
    prob = sdp_identity_feasible(n=3)
    r = sdp_solve(prob; optTol=optTol, verbose=false)
    @test r.status_match
end

# ──────────────────────────────────────────────────────────────
#  Multiple blocks and mixed cones
# ──────────────────────────────────────────────────────────────

@testset "SDP: Multiple SDP blocks" begin
    prob = sdp_multiple_blocks(n1=3, n2=4)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
end

@testset "SDP: Mixed R + Q + S cones" begin
    prob = sdp_mixed_cones(n_r=4, n_q=3, n_s=3)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal

    # Verify each cone constraint
    n_r = 4; n_q = 4; k_s = _vdim(3)
    y = sol.y

    # R+ block: y[1:4] ≥ 0
    # SOC block: y[5:8] ∈ SOC
    # SDP block: mat(y[9:end]) ≽ 0
end

@testset "SDP: With equality constraints (n=4)" begin
    prob = sdp_with_equality(n=4)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                  prob.G, prob.d;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    # Check equality constraints are satisfied
    X = ConicIP.mat(sol.v)
    @test all(eigvals(Symmetric(X)) .> -tol)
end

# ──────────────────────────────────────────────────────────────
#  Scaling / Size tests
# ──────────────────────────────────────────────────────────────

@testset "SDP: Larger PSD projection (n=10)" begin
    prob = sdp_larger(n=10)
    sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                  optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X - prob.known_X, Inf) < tol
end

@testset "SDP: PSD projection across sizes" begin
    for n in [2, 4, 6, 8, 10, 12]
        prob = sdp_psd_projection(n=n)
        sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                      optTol=optTol, verbose=false)
        @test sol.status == :Optimal
        X = ConicIP.mat(sol.y)
        @test norm(X - prob.known_X, Inf) < tol
    end
end

# ──────────────────────────────────────────────────────────────
#  Edge Cases
# ──────────────────────────────────────────────────────────────

@testset "SDP: 1×1 matrix (scalar case)" begin
    # 1×1 PSD = nonnegative scalar
    k = 1
    Q = ones(1, 1)
    c = reshape([-2.0], :, 1)   # min ½x² + 2x s.t. x ≥ 0 → x* = 0...
    # Actually: min ½x² - (-2)x = ½x² + 2x → x* = 0 since x ≥ 0 (PSD for 1x1)
    # Wait: ConicIP minimizes ½y'Qy - c'y. With c = [-2], minimizes ½x² + 2x.
    # Subject to x ∈ S(1) means x ≥ 0. Minimum at x=0.
    A = sparse([1.0], [1], [1.0], 1, 1)
    b = zeros(1, 1)
    sol = conicIP(Q, c, A, b, [("S", 1)]; optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    @test abs(sol.y[1]) < tol
end

@testset "SDP: 2×2 matrix" begin
    # Project [[1, 2], [2, 1]] onto PSD cone
    # Eigenvalues: 3, -1. PSD projection clips -1→0.
    n = 2; k = _vdim(n)
    target = [1.0 2.0; 2.0 1.0]
    λ, V = eigen(Symmetric(target))
    expected = V * diagm(0 => max.(λ, 0.0)) * V'

    Q = Matrix{Float64}(I, k, k)
    c = reshape(ConicIP.vecm(target), :, 1)
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    sol = conicIP(Q, c, A, b, [("S", k)]; optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X - expected, Inf) < tol
end

@testset "SDP: Already-PSD matrix stays unchanged" begin
    n = 4; k = _vdim(n)
    target = Matrix{Float64}(I, n, n)  # already PSD

    Q = Matrix{Float64}(I, k, k)
    c = reshape(ConicIP.vecm(target), :, 1)
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    sol = conicIP(Q, c, A, b, [("S", k)]; optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X - target, Inf) < tol
end

@testset "SDP: Zero matrix projection" begin
    # Project zero matrix onto PSD cone → stays at zero
    n = 3; k = _vdim(n)
    Q = Matrix{Float64}(I, k, k)
    c = zeros(k, 1)  # target = zero matrix
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    sol = conicIP(Q, c, A, b, [("S", k)]; optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X, Inf) < tol
end

@testset "SDP: Highly non-PSD target" begin
    # Project -10*I onto PSD cone → should get zero matrix
    n = 3; k = _vdim(n)
    target = -10.0 * Matrix{Float64}(I, n, n)
    Q = Matrix{Float64}(I, k, k)
    c = reshape(ConicIP.vecm(target), :, 1)
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    sol = conicIP(Q, c, A, b, [("S", k)]; optTol=optTol, verbose=false)
    @test sol.status == :Optimal
    X = ConicIP.mat(sol.y)
    @test norm(X, Inf) < tol
end

# ──────────────────────────────────────────────────────────────
#  SDP across KKT solvers (via harness)
# ──────────────────────────────────────────────────────────────

@testset "SDP: Consistent results across KKT solvers" begin
    prob = sdp_psd_projection(n=6)

    sols = []
    for ks in (ConicIP.kktsolver_qr,
               ConicIP.kktsolver_sparse,
               pivot(ConicIP.kktsolver_2x2))
        sol = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                      kktsolver=ks, optTol=optTol, verbose=false)
        push!(sols, sol)
    end

    # All should be optimal
    @test all(s.status == :Optimal for s in sols)

    # Solutions should agree
    for i in 2:length(sols)
        @test norm(sols[1].y - sols[i].y, Inf) < tol
    end
end

# ──────────────────────────────────────────────────────────────
#  Evaluation Harness Integration
# ──────────────────────────────────────────────────────────────

@testset "SDP Harness: full problem suite" begin
    problems = [
        sdp_psd_projection(n=6),
        sdp_psd_projection(n=8),
        sdp_trace_minimization(n=4),
        sdp_nearest_correlation(n=5),
        sdp_max_cut_relaxation(n=5),
        sdp_lovasz_theta(n=5),
        sdp_rank_one(n=4),
        sdp_identity_feasible(n=3),
        sdp_multiple_blocks(n1=3, n2=4),
        sdp_mixed_cones(n_r=4, n_q=3, n_s=3),
        sdp_with_equality(n=4),
        sdp_larger(n=10),
    ]

    results = run_sdp_harness(problems; optTol=optTol, verbose=false)
    summary = print_sdp_report(results)

    # All problems should have matching status
    for r in results
        @test r.status_match
    end
end

end # @testset "SDP Suite"
