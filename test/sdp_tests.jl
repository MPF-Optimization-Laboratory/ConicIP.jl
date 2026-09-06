# Semidefinite test suite: unit tests for the SDP internals, the standard
# problem instances from sdp_problems.jl, cross-solver consistency, and
# a handful of edge cases.

include("sdp_problems.jl")

# Solve one instance from sdp_all_problems(), with or without equalities.
function sdp_solve(prob; kwargs...)
    if size(prob.G, 1) == 0
        conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                verbose = false, kwargs...)
    else
        conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims, prob.G, prob.d;
                verbose = false, kwargs...)
    end
end

# Extract the first SDP block of a solution as a matrix.  The modeled
# matrix variable is `sol.y`, read here by cone offset, which is valid
# because every instance in this file uses A = I so the cone blocks of
# `y` line up with `cone_dims`.  The cone slack is `sol.s = A*y - b`,
# which differs from `y` whenever b ≠ 0 (trace minimization, multiple
# blocks, and the with-equality instance).  `sol.v` is the inequality
# dual, never the primal.
function sdp_block(sol, cone_dims)
    offset = 0
    for (ctype, cdim) in cone_dims
        ctype == "S" && return ConicIP.mat(sol.y[offset+1:offset+cdim])
        offset += cdim
    end
    return nothing
end

# Residuals of the KKT system the solver itself certifies.  The signs and
# the normalizations follow src/ConicIP.jl, where rDu/rPr/rEq are formed:
#
#   stationarity   Q*y + G'*w - A'*v - c    scaled by 1 + ‖c‖
#   slack          A*y - s - b              scaled by 1 + ‖b‖
#   equality       G*y - d                  scaled by 1 + ‖d‖
#
# plus the complementarity gap s'v, the objective identity
# pobj = ½y'Qy - c'y, and cone membership of the slack and the dual.
function kkt_residuals(prob, sol)
    y, w, v, s = sol.y, sol.w, sol.v, sol.s
    return (stat  = norm(prob.Q * y + prob.G' * w - prob.A' * v - prob.c) /
                    (1 + norm(prob.c)),
            slack = norm(prob.A * y - s - prob.b) / (1 + norm(prob.b)),
            eq    = isempty(prob.d) ? 0.0 :
                    norm(prob.G * y - prob.d) / (1 + norm(prob.d)),
            gap   = abs(dot(s, v)) / (1 + abs(dot(prob.c, y))),
            pobj  = abs(sol.pobj - (0.5 * dot(y, prob.Q * y) - dot(prob.c, y))),
            s_margin = ConicIP.cone_margin(s, prob.cone_dims),
            v_margin = ConicIP.cone_margin(v, prob.cone_dims))
end

# Assert every KKT residual of an optimal solution. `δ` defaults to a
# hundred times the requested optimality tolerance; observed residuals on
# this suite are all below 1e-7.
function test_kkt(prob, sol; δ = 100 * optTol)
    r = kkt_residuals(prob, sol)
    @test r.stat     < δ
    @test r.slack    < δ
    @test r.eq       < δ
    @test r.gap      < δ
    @test r.pobj     < δ
    @test r.s_margin > -δ
    @test r.v_margin > -δ
end

@testset "SDP Suite" begin

    # ──────────────────────────────────────────────────────────────
    #  Cone arithmetic
    # ──────────────────────────────────────────────────────────────

    @testset "Cone product and division" begin
        for n in [2, 3, 4]
            Random.seed!(500 + n)
            k = _vdim(n)

            M = randn(n, n); Y = M' * M + I      # strictly PSD
            R = randn(n, n); X = (R + R') / 2

            x = ConicIP.vecm(X); y = ConicIP.vecm(Y)
            o = zeros(k)

            # x ∘ y = XY + YX
            ConicIP.xsdc!(x, y, o)
            @test o ≈ ConicIP.vecm(X * Y + Y * X) atol=1e-10

            # x ÷ y solves the Lyapunov equation Y*O + O*Y = X
            od = zeros(k)
            ConicIP.dsdc!(x, y, od)
            O = ConicIP.mat(od)
            @test norm(Y * O + O * Y - X, Inf) < 1e-9

            # and the two are inverse: y ∘ (x ÷ y) ≈ x
            ConicIP.xsdc!(y, od, o)
            @test o ≈ x atol=1e-8
        end
    end

    @testset "Cone identity vecm(I)" begin
        for n in [2, 4, 6]
            Random.seed!(200 + n)
            M = randn(n, n); X = M' * M + I
            x = ConicIP.vecm(X)
            o = zeros(length(x))
            ConicIP.xsdc!(x, ConicIP.vecm(Matrix{Float64}(I, n, n)), o)
            @test ConicIP.mat(o) ≈ 2 * X atol=1e-10
        end
    end

    @testset "ord() inverts vdim" begin
        for n in [1, 2, 3, 5, 10, 15, 20]
            @test ConicIP.ord(zeros(_vdim(n))) == n
        end
    end

    @testset "Diagonal position in vecm ordering" begin
        # _diagpos must name the unique nonzero of vecm(Eᵢᵢ), and the value
        # there must be exactly 1 — vecm scales off-diagonals by √2 but
        # leaves the diagonal alone, and the equality rows of the max-cut
        # and nearest-correlation instances depend on that.
        for n in (3, 5)
            k = _vdim(n)
            for i = 1:n
                E = zeros(n, n); E[i, i] = 1.0
                e = ConicIP.vecm(E)
                @test length(e) == k
                @test findall(!iszero, e) == [_diagpos(n, i)]
                @test e[_diagpos(n, i)] == 1.0
            end
        end
    end

    # ──────────────────────────────────────────────────────────────
    #  VecCongurance and Block
    #
    #  runtests.jl "Misc Tests" already checks Matrix/sparse agreement,
    #  size, inv-against-backslash and Z'*Z for one 3×3 R.  What is
    #  additive here is the *meaning* of the operator — W*x = vecm(R'XR)
    #  and W'*x = vecm(RXR') — over several orders.
    # ──────────────────────────────────────────────────────────────

    @testset "VecCongurance acts as X ↦ R'XR" begin
        Random.seed!(42)
        for n in [2, 3, 5]
            k = _vdim(n)
            R = randn(n, n) + n * I
            W = ConicIP.VecCongurance(R)
            x = randn(k)
            X = ConicIP.mat(x)

            @test W * x ≈ ConicIP.vecm(R' * X * R) atol=1e-10
            @test W' * x ≈ ConicIP.vecm(R * X * R') atol=1e-10
            @test inv(W) * (W * x) ≈ x atol=1e-9

            W2 = ConicIP.VecCongurance(randn(n, n) + n * I)
            z  = randn(k)
            @test (W * W2) * z ≈ W * (W2 * z) atol=1e-9
        end
    end

    @testset "Block with VecCongurance" begin
        Random.seed!(900)

        n = 3; k = _vdim(n)
        W = ConicIP.VecCongurance(randn(n, n) + n * I)
        B = Block(1); B[1] = W
        x = randn(k)
        @test B * x ≈ W * x atol=1e-10
        @test Matrix(B) ≈ Matrix(W) atol=1e-10

        # mixed Diagonal (R₊) + VecCongurance (S)
        n_r = 3; n_s = 2; k_s = _vdim(n_s)
        D = Diagonal(rand(n_r) .+ 0.5)
        Ws = ConicIP.VecCongurance(randn(n_s, n_s) + n_s * I)
        Bm = Block(2); Bm[1] = D; Bm[2] = Ws

        z = randn(n_r + k_s)
        y = Bm * z
        @test y[1:n_r] ≈ D * z[1:n_r] atol=1e-10
        @test y[n_r+1:end] ≈ Ws * z[n_r+1:end] atol=1e-10
        @test inv(Bm) * (Bm * z) ≈ z atol=1e-8
        @test Bm' * z ≈ Matrix(Bm)' * z atol=1e-10
    end

    # ──────────────────────────────────────────────────────────────
    #  Nesterov-Todd scaling and line search
    # ──────────────────────────────────────────────────────────────

    @testset "Nesterov-Todd scaling F*z = inv(F')*s" begin
        for n in [2, 3, 4, 5]
            Random.seed!(300 + n)
            M1 = randn(n, n); Z = M1' * M1 + I
            M2 = randn(n, n); S = M2' * M2 + I
            z = ConicIP.vecm(Z); s = ConicIP.vecm(S)

            F  = ConicIP.nestod_sdc(z, s)
            Fz = F * z
            @test Fz ≈ inv(F') * s atol=1e-8
            # the same identity without relying on inv: F'*(F*z) = s
            @test F' * Fz ≈ s atol=1e-8

            # the common value is vecm(Λ) for Λ the diagonal matrix of
            # singular values, so mat(F*z) is diagonal and positive
            Λ = ConicIP.mat(Fz)
            @test norm(Λ - Diagonal(Λ), Inf) < 1e-8
            @test minimum(diag(Λ)) > 0
        end
    end

    @testset "maxstep_sdc boundary step" begin
        # Commuting case: X = I and a direction with one positive
        # eigenvalue, so the maximum α with mat(x - α*d) ⪰ 0 is 1/λmax(D).
        n = 3
        x = ConicIP.vecm(Matrix{Float64}(I, n, n))
        d = ConicIP.vecm(diagm(0 => [0.5, -0.3, -0.1]))

        α = ConicIP.maxstep_sdc(x, d)
        @test α ≈ 2.0
        @test eigmin(Symmetric(ConicIP.mat(x - α * d))) ≈ 0 atol=1e-9

        # Non-commuting case: X = L*L' and D = L*Λ*L' for a fixed
        # non-orthogonal L.  X^{-1/2}*L is orthogonal, so the eigenvalues
        # of X^{-1/2}*D*X^{-1/2} are exactly diag(Λ) and α = 1/λmax = 2
        # again — even though X and D do not commute.
        L = [1.0 0.0 0.0
             0.7 1.2 0.0
            -0.4 0.5 0.9]
        Λ  = diagm(0 => [0.5, -0.3, -0.1])
        XL = L * L'
        DL = L * Λ * L'
        @test norm(XL * DL - DL * XL, Inf) > 0.1        # genuinely non-commuting

        xL = ConicIP.vecm(XL)
        dL = ConicIP.vecm(DL)
        αL = ConicIP.maxstep_sdc(xL, dL)
        @test αL ≈ 2.0
        @test eigmin(Symmetric(ConicIP.mat(xL - αL * dL))) ≈ 0 atol=1e-9

        # All-negative spectrum: x - α*d stays PSD for every α, so the
        # step to the boundary is infinite (a branch distinct from the
        # non-PD-x branch that runtests.jl "Misc Tests" already covers).
        dNeg = ConicIP.vecm(L * diagm(0 => [-0.5, -0.3, -0.1]) * L')
        @test ConicIP.maxstep_sdc(xL, dNeg) == Inf
    end

    @testset "maxstep_sdc generalized eigen form" begin
        # maxstep_sdc solves the pencil (D, X) instead of forming
        # X^{-1/2}*D*X^{-1/2}. The two must agree wherever the explicit
        # form is defined, and the pencil form must additionally be exact
        # on the degenerate inputs that broke the explicit one.
        for n in 2:6, seed in (11, 12, 13)
            Random.seed!(1000 * n + seed)
            A = randn(n, n); X = A' * A + I           # strictly PD
            R = randn(n, n); D = (R + R') / 2

            x = ConicIP.vecm(X); d = ConicIP.vecm(D)
            α = ConicIP.maxstep_sdc(x, d)

            Xih = X^(-1 / 2)
            M   = Symmetric((Xih * D * Xih + (Xih * D * Xih)') / 2)
            λmax = maximum(eigvals(M))

            if λmax > 0
                @test α ≈ 1 / λmax rtol=1e-8
                # X - α*D sits exactly on the cone boundary
                @test eigmin(Symmetric(X - α * D)) ≈ 0 atol=1e-10 * norm(X)
            else
                @test α == Inf
            end
        end

        # A zero direction never limits the step. kktsolver_sparse returns
        # -0.0 for a mathematically zero affine dual step; the old sign
        # mask let those through and produced 1/(-0.0) = -Inf.
        Xi = ConicIP.vecm(Matrix{Float64}(I, 3, 3))
        @test ConicIP.maxstep_sdc(Xi,  ConicIP.vecm(zeros(3, 3))) == Inf
        @test ConicIP.maxstep_sdc(Xi, -ConicIP.vecm(zeros(3, 3))) == Inf

        # X positive definite but scaled into the subnormal range: LAPACK
        # sygvd returns NaN rather than throwing, which must still be
        # reported as a KKT failure (never as NaN, never as ArgumentError).
        @test_throws ConicIP.KKT_FAILURES ConicIP.maxstep_sdc(
            ConicIP.vecm(diagm(0 => [1.0, 1e-310])),
            ConicIP.vecm(Matrix{Float64}(I, 2, 2)))

        # And X ⋡ 0 is a factorization failure, not an infinite step.
        @test_throws ConicIP.KKT_FAILURES ConicIP.maxstep_sdc(
            ConicIP.vecm(-Matrix{Float64}(I, 3, 3)),
            ConicIP.vecm(Matrix{Float64}(I, 3, 3)))
    end

    @testset "Order-0 S block" begin
        # MOI's PositiveSemidefiniteConeTriangle(0) reaches the solver as
        # ("S", 0) — the wrapper does not filter it — so every per-cone
        # primitive has to survive an empty block.  `maximum` and `eigmin`
        # both throw on an empty spectrum, so both line searches need the
        # explicit answer: an order-0 block constrains nothing.
        @test ConicIP.maxstep_sdc(Float64[], Float64[]) == Inf
        @test ConicIP.maxstep_sdc(Float64[], nothing) == 0
        @test ConicIP.nestod_sdc(Float64[], Float64[]) isa ConicIP.VecCongurance

        # and end to end, next to a live block
        n = 2
        sol = conicIP(Matrix{Float64}(I, n, n), ones(n),
                      sparse(1.0I, n, n), zeros(n), [("R", n), ("S", 0)];
                      verbose = false, optTol = optTol)
        @test sol.status == :Optimal
        @test norm(sol.y - ones(n), Inf) < tol
    end

    # ──────────────────────────────────────────────────────────────
    #  Standard problem instances
    # ──────────────────────────────────────────────────────────────

    for prob in sdp_all_problems()
        @testset "$(prob.description)" begin
            sol = sdp_solve(prob; optTol = optTol)
            @test sol.status == prob.known_status

            if prob.known_obj !== nothing
                @test abs(sol.pobj - prob.known_obj) < tol
            end
            if prob.known_X !== nothing
                @test norm(sdp_block(sol, prob.cone_dims) - prob.known_X, Inf) < tol
            end
            known_y = get(prob, :known_y, nothing)
            if known_y !== nothing
                @test norm(sol.y - known_y, Inf) < tol
            end
            for (name, chk) in get(prob, :checks, ())
                @testset "$name" begin
                    @test chk(sol, tol)
                end
            end
            if sol.status == :Optimal
                @testset "KKT residuals" begin
                    test_kkt(prob, sol)
                end
            end
        end
    end

    # ──────────────────────────────────────────────────────────────
    #  Cross-solver consistency
    # ──────────────────────────────────────────────────────────────

    @testset "Consistency across KKT solvers" begin
        for prob in sdp_all_problems()
            solvers = Any[("qr", ConicIP.kktsolver_qr),
                          ("sparse", ConicIP.kktsolver_sparse),
                          ("pivot(2x2)", pivot(ConicIP.kktsolver_2x2))]

            # Where the optimal set is not a singleton, solvers may return
            # different points; compare objectives instead, and let the KKT
            # residuals carry the burden of proving each point optimal.
            unique_y = get(prob, :unique_y, true)

            @testset "$(prob.description)" begin
                ref = nothing
                for (name, ks) in solvers
                    sol = sdp_solve(prob; kktsolver = ks, optTol = optTol)
                    @testset "$name" begin
                        @test sol.status == prob.known_status
                        if sol.status == :Optimal
                            test_kkt(prob, sol)
                        end
                        if ref === nothing
                            ref = sol
                        elseif unique_y
                            @test norm(ref.y - sol.y, Inf) < tol
                        else
                            @test abs(ref.pobj - sol.pobj) < tol
                        end
                    end
                end
            end
        end
    end

    # ──────────────────────────────────────────────────────────────
    #  Edge cases
    #
    #  The projections below have s = v = 0 at the optimum — the worst
    #  case for strict complementarity, where the attainable accuracy is
    #  about √optTol — so they are solved to a tighter optTol.
    # ──────────────────────────────────────────────────────────────

    @testset "1×1 SDP is a nonnegative scalar" begin
        # min ½x² + 2x s.t. x ≥ 0  →  x⋆ = 0
        sol = conicIP(ones(1, 1), [-2.0], sparse(1.0I, 1, 1), zeros(1),
                      [("S", 1)]; optTol = 1e-9, verbose = false)
        @test sol.status == :Optimal
        @test abs(sol.y[1]) < tol
    end

    @testset "2×2 PSD projection clips one eigenvalue" begin
        target = [1.0 2.0; 2.0 1.0]            # eigenvalues 3, -1
        λ, V   = eigen(Symmetric(target))
        expected = V * diagm(0 => max.(λ, 0.0)) * V'

        k = _vdim(2)
        sol = conicIP(Matrix{Float64}(I, k, k), ConicIP.vecm(target),
                      sparse(1.0I, k, k), zeros(k), [("S", k)];
                      optTol = optTol, verbose = false)
        @test sol.status == :Optimal
        @test norm(ConicIP.mat(sol.y) - expected, Inf) < tol
    end

    @testset "Already-PSD target is unchanged" begin
        n = 4; k = _vdim(n)
        target = Matrix{Float64}(I, n, n)
        sol = conicIP(Matrix{Float64}(I, k, k), ConicIP.vecm(target),
                      sparse(1.0I, k, k), zeros(k), [("S", k)];
                      optTol = optTol, verbose = false)
        @test sol.status == :Optimal
        @test norm(ConicIP.mat(sol.y) - target, Inf) < tol
    end

    @testset "Zero and strongly negative targets project to zero" begin
        n = 3; k = _vdim(n)
        for target in (zeros(n, n), -10.0 * Matrix{Float64}(I, n, n))
            sol = conicIP(Matrix{Float64}(I, k, k), ConicIP.vecm(target),
                          sparse(1.0I, k, k), zeros(k), [("S", k)];
                          optTol = 1e-9, verbose = false)
            @test sol.status == :Optimal
            @test norm(ConicIP.mat(sol.y), Inf) < tol
        end
    end

end
