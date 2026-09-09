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

# The same for the inequality dual `sol.v`.  An instance whose `A` is not
# the identity models its matrix variable in the dual rather than in `y`
# (see `sdp_affine_eigmax`), and reports it as `known_V`.
function sdp_dual_block(sol, cone_dims)
    offset = 0
    for (ctype, cdim) in cone_dims
        ctype == "S" && return ConicIP.mat(sol.v[offset+1:offset+cdim])
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

# Assert every KKT residual of an optimal solution.  The tolerance is
# absolute, not a multiple of `optTol`: tying it to the requested
# tolerance would make the assertion vacuous the moment the suite asks
# for less accuracy.  At `optTol = 1e-7` the observed relative residuals
# on this suite are below 1e-8 and the observed gaps below 1e-7, so 1e-6
# is a real bound with an order of magnitude of headroom.
function test_kkt(prob, sol; δ = 1e-6)
    r = kkt_residuals(prob, sol)
    @test r.stat     < δ
    @test r.slack    < δ
    @test r.eq       < δ
    @test r.gap      < δ
    @test r.pobj     < δ
    @test r.s_margin > -δ
    @test r.v_margin > -δ

    # Dual objective reconstructed from the problem data alone, rather
    # than read back from the solver's own running value:
    #   g(w,v) = b'v - d'w - ½y'Qy
    # (the sign on `d` follows the conicIP Lagrangian, whose stationarity
    # row is Qy + G'w - A'v - c = 0).  Strong duality closes it onto pobj.
    dobj = dot(prob.b, sol.v) - dot(prob.d, sol.w) -
           0.5 * dot(sol.y, prob.Q * sol.y)
    @test abs(dobj - sol.pobj) / (1 + abs(sol.pobj)) < δ
end

@testset "SDP Suite" begin

    # ──────────────────────────────────────────────────────────────
    #  Cone arithmetic
    # ──────────────────────────────────────────────────────────────

    # ── Regression evidence for the S-cone centering fix ──
    #
    #  Two assertions below fail against the pre-fix `xsdc!`/`dsdc!`, which
    #  computed the unnormalized product XY + YX (twice the Jordan product,
    #  with identity I/2 rather than the vecm(I) that conicIP assembles as
    #  `e`):
    #
    #    * "Cone product and division": `o ≈ vecm((XY + YX)/2)` and the
    #      division residual `‖YO + OY - 2X‖`;
    #    * "vecm(I) is the identity of the cone product" and the S case of
    #      "x ∘ e = x", i.e. `xsdc!(x, e) == x`.
    #
    #  Those are the tests that detect the bug. The testsets after them
    #  ("Jordan product identity", "order-1 S block", the affine-eigmax
    #  fixture, the certificate tests) pass on the old code too — they are
    #  invariants worth pinning, not regression evidence. The end-to-end
    #  effect of the fix is one extra interior-point iteration on the
    #  with-equality instance and nothing else; that is pinned where the
    #  instance is solved.

    @testset "Cone product and division" begin
        # The cone product of the semidefinite block is the *Jordan*
        # product (XY + YX)/2 — the one whose identity is I, matching the
        # `e = vecm(I)` that conicIP assembles and that the corrector
        # subtracts σμ times.  Both assertions here fail on the pre-fix
        # code, which returned XY + YX.
        for n in [2, 3, 4]
            Random.seed!(500 + n)
            k = _vdim(n)

            M = randn(n, n); Y = M' * M + I      # strictly PSD
            R = randn(n, n); X = (R + R') / 2

            x = ConicIP.vecm(X); y = ConicIP.vecm(Y)
            o = zeros(k)

            # x ∘ y = (XY + YX)/2
            ConicIP.xsdc!(x, y, o)
            @test o ≈ ConicIP.vecm((X * Y + Y * X) / 2) atol=1e-10

            # x ÷ y inverts it: (Y*O + O*Y)/2 = X
            od = zeros(k)
            ConicIP.dsdc!(x, y, od)
            O = ConicIP.mat(od)
            @test norm(Y * O + O * Y - 2 * X, Inf) < 1e-9

            # and the two are inverse: y ∘ (x ÷ y) ≈ x
            ConicIP.xsdc!(y, od, o)
            @test o ≈ x atol=1e-8
        end
    end

    @testset "vecm(I) is the identity of the cone product" begin
        # x ∘ e = x for the S block.  Fails on the pre-fix code, which
        # returned 2X here.
        for n in [2, 4, 6]
            Random.seed!(200 + n)
            M = randn(n, n); X = M' * M + I
            x = ConicIP.vecm(X)
            o = zeros(length(x))
            ConicIP.xsdc!(x, ConicIP.vecm(Matrix{Float64}(I, n, n)), o)
            @test ConicIP.mat(o) ≈ X atol=1e-10
        end
    end

    @testset "x ∘ e = x for every cone" begin
        # conicIP builds one `e` for the whole cone group and the corrector
        # subtracts σμ*e from the product λ ∘ λ.  That is only the intended
        # centering target if `e` is the identity of each block's product.
        # The R and Q cases held before the fix; the S case did not.
        Random.seed!(4242)

        n_r = 5
        x_r = rand(n_r) .+ 0.5
        o_r = zeros(n_r)
        ConicIP.xrp!(x_r, ones(n_r), o_r)
        @test o_r ≈ x_r atol=1e-12

        n_q = 4
        x_q = vcat(3.0, randn(n_q - 1))                 # interior of Q
        o_q = zeros(n_q)
        ConicIP.xsoc!(x_q, vcat(1.0, zeros(n_q - 1)), o_q)
        @test o_q ≈ x_q atol=1e-12

        n_s = 4; k_s = _vdim(n_s)
        Ms = randn(n_s, n_s); Xs = Ms' * Ms + I
        x_s = ConicIP.vecm(Xs)
        o_s = zeros(k_s)
        ConicIP.xsdc!(x_s, ConicIP.vecm(Matrix{Float64}(I, n_s, n_s)), o_s)
        @test o_s ≈ x_s atol=1e-12
    end

    @testset "Jordan product identity" begin
        # The algebra the corrector row composes: λ ∘ λ = Λ², and the
        # product of two symmetric direction blocks.  This does *not*
        # exercise the corrector, and it is written in terms of the
        # post-fix semantics, so it is an invariant rather than a
        # regression test — the σ/2 bug is caught by `xsdc!(x, e) == x`
        # above.
        Random.seed!(31415)
        n = 4; k = _vdim(n)
        Ml = randn(n, n); Λ = Ml' * Ml + I               # strictly PD λ
        λ = ConicIP.vecm(Λ)
        e = ConicIP.vecm(Matrix{Float64}(I, n, n))
        σ = 0.3; μ = 0.7

        o = zeros(k)
        ConicIP.xsdc!(λ, λ, o)
        @test o ≈ ConicIP.vecm(Λ * Λ) atol=1e-10
        @test o - σ * μ * e ≈ ConicIP.vecm(Λ * Λ - σ * μ * Matrix{Float64}(I, n, n)) atol=1e-10

        # The other half of the corrector row is the product of two
        # symmetric search-direction blocks, F⁻ᵀΔs ∘ FΔv.
        R1 = randn(n, n); Δ1 = (R1 + R1') / 2
        R2 = randn(n, n); Δ2 = (R2 + R2') / 2
        ConicIP.xsdc!(ConicIP.vecm(Δ1), ConicIP.vecm(Δ2), o)
        @test o ≈ ConicIP.vecm((Δ1 * Δ2 + Δ2 * Δ1) / 2) atol=1e-10
    end

    @testset "Cone algebra of an order-1 S block matches R₊" begin
        # For n = 1 the Jordan algebra of symmetric matrices *is* R₊: the
        # product, the division, the identity, the scaling and the line
        # search all collapse to the scalar case.  Solving the same QP
        # under both declarations must therefore trace the same iterates.
        # (Both runs are pinned to kktsolver_qr so the comparison is not
        # confounded by the automatic solver choice.)
        #
        # This is an invariant, not regression evidence: the pre-fix code
        # was internally consistent, so `2xy` on both sides of the S/R
        # comparison cancelled, and σ ≈ 0 on this instance anyway.
        Qs = ones(1, 1); cs = [2.0]                 # min ½y² - 2y, y ≥ 0
        As = sparse(1.0I, 1, 1); bs = zeros(1)

        sol_s = conicIP(Qs, cs, As, bs, [("S", 1)];
                        verbose = false, optTol = optTol,
                        kktsolver = ConicIP.kktsolver_qr)
        sol_r = conicIP(Qs, cs, As, bs, [("R", 1)];
                        verbose = false, optTol = optTol,
                        kktsolver = ConicIP.kktsolver_qr)

        @test sol_s.status == :Optimal
        @test sol_r.status == :Optimal
        @test sol_s.Iter == sol_r.Iter
        @test norm(sol_s.y - sol_r.y, Inf) < 1e-10
        @test norm(sol_s.v - sol_r.v, Inf) < 1e-10
        @test norm(sol_s.s - sol_r.s, Inf) < 1e-10
        @test abs(sol_s.y[1] - 2.0) < 1e-6           # the actual optimum
    end

    @testset "Centering fix moves only the with-equality trajectory" begin
        # End-to-end consequence of centering the S blocks at σμ*I instead
        # of (σ/2)μ*I: the S contribution to rCp and to the refinement
        # residual is halved at an identical iterate, so both stopping
        # rules shift slightly.  Across the whole suite exactly one
        # instance changes its iteration count, 8 -> 9, and it does so
        # under all three KKT solvers -- which is what makes the count
        # safe to pin here.
        # Pinned on the unequilibrated trajectory: the argument above is
        # about the centering of the S blocks at an identical iterate, and
        # equilibration (which rescales the objective) moves the count.
        prob = sdp_with_equality(n = 4)
        for ks in (ConicIP.kktsolver_qr, ConicIP.kktsolver_sparse,
                   pivot(ConicIP.kktsolver_2x2))
            sol = sdp_solve(prob; optTol = optTol, kktsolver = ks,
                            equilibrate = false)
            @test sol.status == :Optimal
            @test sol.Iter == 9
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
            known_V = get(prob, :known_V, nothing)
            if known_V !== nothing
                # An instance whose A is not the identity carries its
                # matrix variable in the dual; hold it to a tight bound,
                # since the off-diagonals are what pin the vecm scaling.
                @test norm(sdp_dual_block(sol, prob.cone_dims) - known_V) < 1e-6
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
    #  These projections sit at the extremes of strict complementarity.
    #  Where the optimal X is on the boundary of the cone the attainable
    #  accuracy is about √optTol, so those instances are solved to a
    #  tighter optTol.  Note that s and v do not both vanish in general:
    #  for an already-PSD target the primal slack is s = X ≠ 0 and the
    #  dual v = 0, while for a strictly negative-definite target X⋆ = 0
    #  and it is the dual v = -target ≠ 0 that carries the solution.
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
