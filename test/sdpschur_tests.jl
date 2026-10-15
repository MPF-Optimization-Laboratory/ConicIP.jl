# kktsolver_sdpschur: the equality Schur complement of standard-form
# semidefinite programs (src/schur.jl).
#
# A PSD *variable* makes every cone coordinate a variable (A a scaled
# permutation) and, with Q = 0, the Hessian AᵀF⁻¹F⁻ᵀA has an explicit
# inverse, so only the p×p matrix G H⁻¹ Gᵀ is factored. These tests check
# the reduced solve against kktsolver_qr on the 3×3 system itself, the
# routing (including what must decline), the presolve exception for
# singleton rows, and full solves against the dual formulation.

import MathOptInterface as MOI

# Records which backend the router picks for the main solve.
mutable struct _RouteProbe; name::String; end
function (r::_RouteProbe)(Q, A, G, cd)
    isempty(r.name) && (r.name = string(nameof(ConicIP.choose_kktsolver(Q, A, G, cd))))
    return ConicIP.default_kktsolver(Q, A, G, cd)
end

@testset "kktsolver_sdpschur" begin

    C = ConicIP

    # Interior point of a cone product, and its NT scaling Block.
    function interior(cd)
        v = Float64[]
        for (t, d) in cd
            if t == "R"
                append!(v, rand(d) .+ 0.1)
            elseif t == "Q"
                x = randn(d); x[1] = norm(x[2:end]) + 0.5 + rand(); append!(v, x)
            else
                k = C.ord(1:d); M = randn(k, k); append!(v, C.vecm(M * M' + 0.1I))
            end
        end
        return v
    end
    function nt_block(cd, z, s)
        F = C.Block(length(cd)); off = 0
        for (i, (t, d)) in enumerate(cd)
            I = (off+1):(off+d); off += d
            F[i] = t == "R" ? Diagonal(sqrt.(s[I] ./ z[I])) :
                   t == "Q" ? C.nestod_soc(z[I], s[I]) : C.nestod_sdc(z[I], s[I])
        end
        return F
    end
    # A random standard-form instance: A a scaled permutation over all cones.
    function standard_form(cd, p; seed)
        Random.seed!(seed)
        m = sum(last, cd)
        A = sparse(randperm(m), 1:m, (rand(m) .+ 0.5) .* rand([-1, 1], m), m, m)
        G = sprandn(p, m, 0.3) + sparse(1:p, 1:p, ones(p), p, m)
        return (Q = spzeros(m, m), A = A, G = G, m = m, p = p)
    end
    # Relative residual of the 3×3 system [Q Gᵀ −Aᵀ; G 0 0; A 0 FᵀF].
    function resid3(Q, A, G, F, b, sol)
        (y, w, v) = sol; (bx, by, bz) = b
        Fd = Matrix(F)
        r = [Q * y + G' * w - A' * v - bx; G * y - by; A * y + Fd' * (Fd * v) - bz]
        return norm(r) / norm([bx; by; bz])
    end

    @testset "3×3 solve agrees with kktsolver_qr" begin
        cases = ([("S", 10)], [("R", 3), ("S", 10), ("S", 6)],
                 [("S", 6), ("Q", 4), ("R", 2), ("S", 15)])
        for (ci, cd) in enumerate(cases), p in (0, 4)
            P = standard_form(cd, p; seed = 10ci + p)
            @test C.sdpschur_eligible(P.Q, P.A, cd)
            z = interior(cd); s = interior(cd)
            F = nt_block(cd, z, s); F⁻ᵀ = inv(F)'
            sq = C.kktsolver_qr(P.Q, P.A, P.G, cd)(F, F⁻ᵀ)
            ss = C.kktsolver_sdpschur(P.Q, P.A, P.G, cd)(F, F⁻ᵀ)
            for _ in 1:2
                b = (randn(P.m), randn(P.p), randn(P.m))
                xs = ss(b...); xq = sq(b...)
                @test resid3(P.Q, P.A, P.G, F, b, xs) <= 1e-12
                for i in 1:3
                    @test norm(xs[i] - xq[i]) <= 1e-10 * max(1, norm(xq[i]))
                end
            end
            # The initial point factors with identity scaling.
            Iblk = C.Block([Diagonal(ones(d)) for (_, d) in cd])
            s0 = C.kktsolver_sdpschur(P.Q, P.A, P.G, cd)(Iblk, Iblk)
            b = (randn(P.m), randn(P.p), randn(P.m))
            @test resid3(P.Q, P.A, P.G, Iblk, b, s0(b...)) <= 1e-12
        end
    end

    @testset "sparse, low-rank and dense trace-form paths agree" begin
        # One order-40 block with columns of 1, 3 and all coordinates, so
        # the Gram kernel's cost model takes the four-index, rank-two and
        # dense branches respectively.
        cd = [("S", 820)]; m = 820; p = 9
        Random.seed!(5)
        Gt = spzeros(m, p)
        Gt[rand(1:m), 1] = 1.0; Gt[rand(1:m), 2] = -2.0
        for j in 3:4, c in rand(1:m, 3); Gt[c, j] = randn(); end
        for j in 5:9; Gt[:, j] = randn(m); end
        G = sparse(Gt'); A = sparse(1.0I, m, m); Q = spzeros(m, m)
        z = interior(cd); s = interior(cd); F = nt_block(cd, z, s)
        plan = C._gram_plan(sparse(Gt), cd)
        M = zeros(p, p); C._gram!(M, plan, F)
        Fd = Matrix(F)
        @test M ≈ Gt' * (Fd' * Fd) * Gt rtol = 1e-12
        @test M == M'
        @test all(>(0), plan.sdp[1].stats[1:3])     # each branch was taken
    end

    @testset "trace-form cancellation is caught" begin
        # Review reproducer: R with a leading block [1 0; 1 1e-4] and the
        # column vecm(J), J = [1 −1; −1 1] there. The four-index sum came out
        # 0 against a true 1e-16.
        R = Matrix(1.0I, 4, 4); R[1:2, 1:2] = [1 0; 1 1e-4]
        J = zeros(4, 4); J[1:2, 1:2] = [1 -1; -1 1]
        cd = [("S", 10)]
        B = sparse(reshape(C.vecm(J), :, 1))
        plan = C._gram_plan(B, cd)
        M = zeros(1, 1); C._gram!(M, plan, C.Block([C.VecCongurance(R)]))
        wb = C.VecCongurance(R) * C.vecm(J)
        @test M[1] ≈ dot(wb, wb) rtol = 1e-10
        @test plan.sdp[1].stats[1] == 1 && plan.sdp[1].stats[5] == 1
        @test plan.fallback[1] == 1
        # The same through a real Nesterov-Todd scaling's inverse (1e-20).
        Rz = [1 0 0 0; 1 1e-5 0 0; 0 0 1 0; 0 0 0 1.0]
        F = C.nestod_sdc(C.vecm(Rz * Rz'), C.vecm(inv(Rz)' * inv(Rz)))
        W = inv(F)'
        M = zeros(1, 1); C._gram!(M, C._gram_plan(B, cd), C.Block([W]))
        wb = W * C.vecm(J)
        @test M[1] ≈ dot(wb, wb) rtol = 1e-10

        # Stress: ill-conditioned scalings (κ(T) up to 1e12) against
        # columns of every branch, including ones aligned with the small
        # eigen-directions. Compared on the diagonally scaled matrix, which
        # is what the factorization sees.
        Random.seed!(17)
        for k in (6, 12), κ in (1e2, 1e6, 1e12)
            m = div(k * (k + 1), 2); cd = [("S", m)]
            U = Matrix(qr(randn(k, k)).Q)
            R = U * Diagonal(exp10.(range(0, -log10(sqrt(κ)), length = k)))
            W = C.VecCongurance(R)
            cols = Vector{Float64}[]
            for _ in 1:3                       # sparse: two or three entries
                v = zeros(m); v[rand(1:m, 3)] .= randn(3); push!(cols, v)
            end
            u1, u2 = U[:, end], U[:, end-1]    # small directions of T
            push!(cols, C.vecm(u1 * u2' + u2 * u1'))
            push!(cols, C.vecm(u1 * u1'))
            push!(cols, C.vecm(Symmetric(randn(k, k))))
            e = zeros(k, k); e[1, 1] = 1; e[2, 2] = -1; push!(cols, C.vecm(U * e * U'))
            B = sparse(reduce(hcat, cols)); B = dropzeros!(B)
            plan = C._gram_plan(B, cd)
            M = zeros(size(B, 2), size(B, 2)); C._gram!(M, plan, C.Block([W]))
            Y = reduce(hcat, [W * Vector(B[:, j]) for j in 1:size(B, 2)])
            Md = Y' * Y
            D = Diagonal(1 ./ sqrt.(diag(Md)))
            @test opnorm(D * (M - Md) * D) <= 1e-6
            @test all(>(0), diag(M))
        end
    end

    @testset "kktsolver_qr assembles the SDP Hessian by the trace form" begin
        # Affine (LMI) form: n free variables, A with sparse and dense
        # columns over R, Q and two S blocks, with and without equalities.
        # kktsolver_qr now forms Q + AᵀF⁻¹F⁻ᵀA with the Gram kernel rather
        # than (F⁻ᵀAQ₂)ᵀ(F⁻ᵀAQ₂); the 3×3 residual must stay at rounding.
        cd = [("R", 3), ("Q", 4), ("S", 10), ("S", 21)]
        m = sum(last, cd); n = 9
        for p in (0, 3)
            Random.seed!(60 + p)
            A = sprandn(m, n, 0.15); A[:, 1] = randn(m)
            Q = p == 0 ? spzeros(n, n) : (B = sprandn(n, n, 0.2); B' * B)
            G = sprandn(p, n, 0.5) + sparse(1:p, 1:p, ones(p), p, n)
            z = interior(cd); s = interior(cd)
            F = nt_block(cd, z, s); F⁻ᵀ = inv(F)'
            s3 = C.kktsolver_qr(Q, A, G, cd)(F, F⁻ᵀ)
            for _ in 1:2
                b = (randn(n), randn(p), randn(m))
                @test resid3(Q, A, G, F, b, s3(b...)) <= 1e-11
            end
        end
    end

    @testset "kktsolver_qr with equalities keeps the projected product" begin
        # Projecting an assembled Hessian onto ker G cancels when it is badly
        # scaled along range(Gᵀ): here the reduced Hessian is ≈ diag(0.5, 1),
        # but Q2ᵀHQ2 came out as diag(2e-17, 1) and y ≈ ±4e16.
        cd = [("S", 3)]
        A = sparse([1 1 0; 1e-4 0 0; 0 0 1.0]); G = sparse([1 1 0.0]); Q = spzeros(3, 3)
        F⁻ᵀ = C.Block([C.VecCongurance(Matrix(Diagonal([1e4, 1.0])))])
        F = C.Block([inv(F⁻ᵀ[1])'])
        b = ([1.0, -1, 0], [0.0], zeros(3))
        sol = C.kktsolver_qr(Q, A, G, cd)(F, F⁻ᵀ)(b...)
        @test resid3(Q, A, G, F, b, sol) <= 1e-12
        @test norm(sol[1]) <= 10
    end

    @testset "kktsolver_qr does not densify a large SOC when n − p is small" begin
        # One order-1 SDP block made every factorization assemble the full
        # Hessian, densifying the SOC slice of A (dimension × touched
        # columns, 2 × 160 MB here) although the null space of G is one
        # direction. The projected product is m × (n − p).
        d = 200_000; n = 50; p = 49
        cd = [("Q", d), ("S", 1)]
        Random.seed!(23)
        A = [sprandn(d, n, 5 / d); sparse([1], [1], [1.0], 1, n)]
        G = sparse([Matrix(1.0I, p, p) zeros(p, 1)] .+ [zeros(p, n - 1) randn(p)])
        z = interior(cd); s = interior(cd); F = nt_block(cd, z, s); F⁻ᵀ = inv(F)'
        gen = C.kktsolver_qr(spzeros(n, n), A, G, cd)
        gen(F, F⁻ᵀ)
        @test (@allocated gen(F, F⁻ᵀ)) < 40 * 2^20
        b = (randn(n), randn(p), randn(d + 1))
        (y, w, v) = gen(F, F⁻ᵀ)(b...)
        @test norm(G * y - b[2]) <= 1e-10 * norm(b[2])
    end

    @testset "dual-side regularization and dependent rows" begin
        cd = [("S", 10)]
        P = standard_form(cd, 3; seed = 41)
        z = interior(cd); s = interior(cd); F = nt_block(cd, z, s); F⁻ᵀ = inv(F)'
        b = (randn(P.m), randn(P.p), randn(P.m))
        exact = C.kktsolver_sdpschur(P.Q, P.A, P.G, cd)(F, F⁻ᵀ)(b...)
        reg = C.kktsolver_sdpschur(P.Q, P.A, P.G, cd; static_reg = 1e-10)(F, F⁻ᵀ)(b...)
        @test norm(reg[1] - exact[1]) <= 1e-6 * norm(exact[1])
        # A repeated equality row makes M singular; a consistent right-hand
        # side is still solved (the small dual shift picks one multiplier).
        G2 = [P.G; P.G[1:1, :]]
        s2 = C.kktsolver_sdpschur(P.Q, P.A, G2, cd)(F, F⁻ᵀ)
        b2 = (b[1], [b[2]; b[2][1]], b[3])
        (y, w, v) = s2(b2...)
        @test norm(G2 * y - b2[2]) <= 1e-6 * norm(b2[2])
    end

    @testset "routing and eligibility" begin
        cd = [("S", 6)]; m = 6
        A = sparse(1.0I, m, m); G = sparse(ones(1, m)); Q = spzeros(m, m)
        @test C.choose_kktsolver(Q, A, G, cd) === C.kktsolver_sdpschur
        # Declines: a quadratic objective, a variable in two cone rows (a
        # bound), a cone row in two variables, no semidefinite cone.
        @test !C.sdpschur_eligible(sparse(1.0I, m, m), A, cd)
        @test !C.sdpschur_eligible(Q, [A; sparse([1], [1], [1.0], 1, m)], [("S", 6), ("R", 1)])
        A2 = copy(A); A2[1, 2] = 1.0
        @test !C.sdpschur_eligible(Q, A2, cd)
        @test !C.sdpschur_eligible(Q, A, [("R", 6)])
        @test C.choose_kktsolver(sparse(1.0I, m, m), A, G, cd) === C.kktsolver_qr
        @test_throws ArgumentError C.kktsolver_sdpschur(sparse(1.0I, m, m), A, G, cd)
        # The storage budget does not apply: the k(k+1)/2 variables are never
        # factored, so an order-150 block routes here rather than being refused.
        k = 150; mk = div(k * (k + 1), 2)
        @test C.choose_kktsolver(spzeros(mk, mk), sparse(1.0I, mk, mk),
                                 sparse(1:k, 1:k, ones(k), k, mk),
                                 [("S", mk)]; dense_bytes_max = 2^20) === C.kktsolver_sdpschur
        # …but the Schur matrix is: with a budget below its 3·8p² bytes the
        # router falls through to the dense rules, which refuse an SDP.
        @test_throws ArgumentError C.choose_kktsolver(spzeros(mk, mk), sparse(1.0I, mk, mk),
            sparse(1:k, 1:k, ones(k), k, mk), [("S", mk)]; dense_bytes_max = 2^18)
        # Review reproducer: n = p = 30 000 with A = G = I needs 21.6 GB for
        # the Schur matrix and its copies; it used to route to the Schur
        # solver even with dense_bytes_max = 1. Refused by the router and by
        # conicIP's direct shortcut alike.
        N = 30_000; cdN = [("R", N - 1), ("S", 1)]
        IN = sparse(1.0I, N, N)
        @test C.sdpschur_bytes(N) > 4 * 2^30
        @test_throws ArgumentError C.choose_kktsolver(spzeros(N, N), IN, IN, cdN)
        @test_throws ArgumentError C.choose_kktsolver(spzeros(N, N), IN, IN, cdN; dense_bytes_max = 1)
        @test_throws ArgumentError conicIP(spzeros(N, N), ones(N), IN, zeros(N), cdN, IN, ones(N);
                                           verbose = false)
    end

    # max ⟨L, X⟩/4  s.t.  X_ii = 1, X ⪰ 0 as a PSD variable through MOI, and
    # min Σyᵢ  s.t.  Diag(y) − L/4 ⪰ 0 as the dual formulation.
    tri(i, j) = i <= j ? div(j * (j - 1), 2) + i : tri(j, i)
    function maxcut_models(k; seed = 1)
        rng = MersenneTwister(seed)
        W = zeros(k, k)
        for j in 1:k, i in 1:j-1
            rand(rng) < 0.4 && (W[i, j] = W[j, i] = 1.0)
        end
        L = Diagonal(vec(sum(W; dims = 2))) - W
        pm = MOI.Utilities.Model{Float64}()
        X = MOI.add_variables(pm, div(k * (k + 1), 2))
        MOI.add_constraint(pm, MOI.VectorOfVariables(X), MOI.PositiveSemidefiniteConeTriangle(k))
        for i in 1:k
            MOI.add_constraint(pm, MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, X[tri(i, i)])], 0.0),
                               MOI.EqualTo(1.0))
        end
        obj = [MOI.ScalarAffineTerm((i == j ? 1.0 : 2.0) * L[i, j] / 4, X[tri(i, j)])
               for j in 1:k for i in 1:j if L[i, j] != 0]
        MOI.set(pm, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(obj, 0.0))
        MOI.set(pm, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        dm = MOI.Utilities.Model{Float64}()
        y = MOI.add_variables(dm, k)
        f = MOI.VectorAffineFunction([MOI.VectorAffineTerm(tri(i, i), MOI.ScalarAffineTerm(1.0, y[i])) for i in 1:k],
                                     [-L[i, j] / 4 for j in 1:k for i in 1:j])
        MOI.add_constraint(dm, f, MOI.PositiveSemidefiniteConeTriangle(k))
        MOI.set(dm, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, yi) for yi in y], 0.0))
        MOI.set(dm, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return (pm, dm)
    end

    @testset "PSD-variable max-cut through MOI" begin
        for k in (6, 25)
            (pm, dm) = maxcut_models(k)
            probe = _RouteProbe("")
            opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
            MOI.set(opt, MOI.Silent(), true)
            MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), probe)
            MOI.copy_to(opt, pm); MOI.optimize!(opt)
            # Presolve keeps the X_ii = 1 rows, so the default route is the
            # Schur solver rather than the dense path over k(k+1)/2 variables.
            @test probe.name == "kktsolver_sdpschur"
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            dopt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
            MOI.set(dopt, MOI.Silent(), true)
            MOI.copy_to(dopt, dm); MOI.optimize!(dopt)
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ MOI.get(dopt, MOI.ObjectiveValue()) rtol = 1e-6
            raw = MOI.get(opt, MOI.RawSolver())::ConicIP.Optimizer
            sol = raw.sol
            # Original-data checks: stationarity, feasibility, cone membership.
            Q = raw.Q_int === nothing ? spzeros(raw.n, raw.n) : raw.Q_int
            rDu = norm(Q * sol.y + raw.eq_G' * sol.w - raw.ineq_A' * sol.v - raw.c_int) /
                  (1 + norm(raw.c_int))
            @test rDu <= 1e-6
            @test norm(raw.eq_G * sol.y - raw.eq_d) <= 1e-6 * (1 + norm(raw.eq_d))
            @test C.cone_margin(sol.s, raw.cone_dims) >= -1e-8
            @test C.cone_margin(sol.v, raw.cone_dims) >= -1e-8
        end
    end

    @testset "certificates on the Schur route" begin
        # Primal infeasible: X ⪰ 0 with X₁₁ = −1 (direct API, A = I).
        cd = [("S", 3)]
        A = sparse(1.0I, 3, 3); b = zeros(3); Q = spzeros(3, 3)
        G = sparse([1.0 0 0]); d = [-1.0]; c = -[1.0, 0, 1.0]
        @test C.choose_kktsolver(Q, A, G, cd) === C.kktsolver_sdpschur
        sol = conicIP(Q, c, A, b, cd, G, d; verbose = false)
        @test sol.status == :Infeasible
        # Unbounded: min −X₁₁ s.t. X₂₂ = 1.
        G = sparse([0 0 1.0]); d = [1.0]; c = [1.0, 0, 0]
        sol = conicIP(Q, c, A, b, cd, G, d; verbose = false)
        @test sol.status == :DualInfeasible
    end
end
