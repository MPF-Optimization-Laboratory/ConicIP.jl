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
            @test C.sdpschur_eligible(P.Q, P.A, P.G, cd)
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

    # Iterative refinement on the 3×3 system with a (possibly perturbed)
    # solve, as conicIP applies it; returns the final relative residual.
    function refined_resid(Q, A, G, F, solve, b; steps = 3)
        Fd = Matrix(F)
        K(x) = (Q * x[1] + G' * x[2] - A' * x[3], G * x[1], A * x[1] + Fd' * (Fd * x[3]))
        x = solve(b...)
        for _ in 1:steps
            r = map(-, b, K(x))
            dx = solve(r...)
            x = map(+, x, dx)
        end
        return (resid3(Q, A, G, F, b, x), x)
    end

    @testset "dual-side regularization and dependent rows" begin
        cd = [("S", 10)]
        P = standard_form(cd, 3; seed = 41)
        z = interior(cd); s = interior(cd); F = nt_block(cd, z, s); F⁻ᵀ = inv(F)'
        b = (randn(P.m), randn(P.p), randn(P.m))
        exact = C.kktsolver_sdpschur(P.Q, P.A, P.G, cd)(F, F⁻ᵀ)(b...)
        regs = C.kktsolver_sdpschur(P.Q, P.A, P.G, cd; static_reg = 1e-10)(F, F⁻ᵀ)
        reg = regs(b...)
        @test norm(reg[1] - exact[1]) <= 1e-6 * norm(exact[1])
        # Refinement against the unperturbed system removes the shift.
        @test refined_resid(P.Q, P.A, P.G, F, regs, b)[1] <= 1e-12
        # A repeated equality row makes M singular; a consistent right-hand
        # side is still solved (the small dual shift picks one multiplier).
        G2 = [P.G; P.G[1:1, :]]
        s2 = C.kktsolver_sdpschur(P.Q, P.A, G2, cd)(F, F⁻ᵀ)
        b2 = (b[1], [b[2]; b[2][1]], b[3])
        (y, w, v) = s2(b2...)
        @test norm(G2 * y - b2[2]) <= 1e-6 * norm(b2[2])
        @test refined_resid(P.Q, P.A, G2, F, s2, b2)[1] <= 1e-10
    end

    @testset "a small Schur row is not a dependent one" begin
        # Review reproducer: equality rows on packed coordinates 1 and 3 of
        # an order-2 block with F = VecCongurance(Diag(1, 1e-4)), so
        # M = diag(1, 1e-16) — nonsingular and exact. The unequilibrated
        # pivot test shifted it by 1e-12, returning y₃ = 1e-4 for 1, and
        # three refinement steps reached only 4e-4.
        cd = [("S", 3)]
        A = sparse(1.0I, 3, 3); G = sparse([1.0 0 0; 0 0 1]); Q = spzeros(3, 3)
        F = C.Block([C.VecCongurance(Matrix(Diagonal([1, 1e-4])))])
        F⁻ᵀ = C.Block([inv(F[1])'])
        s5 = C.kktsolver_sdpschur(Q, A, G, cd)(F, F⁻ᵀ)
        b = (zeros(3), [0.0, 1.0], zeros(3))
        (y, w, v) = s5(b...)
        @test y[3] ≈ 1 rtol = 1e-10
        @test resid3(Q, A, G, F, b, (y, w, v)) <= 1e-12
    end

    @testset "routing and eligibility" begin
        cd = [("S", 6)]; m = 6
        A = sparse(1.0I, m, m); G = sparse(ones(1, m)); Q = spzeros(m, m)
        @test C.choose_kktsolver(Q, A, G, cd) === C.kktsolver_sdpschur
        # Declines: a quadratic objective on the cone variables, a cone row
        # in two variables, no semidefinite cone.
        @test !C.sdpschur_eligible(sparse(1.0I, m, m), A, G, cd)
        A2 = copy(A); A2[1, 2] = 1.0
        @test !C.sdpschur_eligible(Q, A2, G, cd)
        @test !C.sdpschur_eligible(Q, A, G, [("R", 6)])
        # A bound on one entry is mixed form with a small reduced system.
        @test C.sdpschur_eligible(Q, [A; sparse([1], [1], [1.0], 1, m)], G, [("S", 6), ("R", 1)])
        @test C.choose_kktsolver(sparse(1.0I, m, m), A, G, cd) === C.kktsolver_qr
        @test_throws ArgumentError C.kktsolver_sdpschur(sparse(1.0I, m, m), A, G, cd)
        # The dense path's storage estimate does not apply: the k(k+1)/2
        # variables are never factored, so an order-150 block routes here
        # under a budget far below the dense estimate — its own (the Schur
        # matrix and the Gram workspaces) is what must fit.
        k = 150; mk = div(k * (k + 1), 2)
        Gk = sparse(1:k, 1:k, ones(k), k, mk); cdk = [("S", mk)]
        own = C._sdpschur_bytes(C._schur_split(spzeros(mk, mk), sparse(1.0I, mk, mk), cdk),
                                k, mk, cdk)
        @test own < C.dense_kkt_bytes(mk, mk, k) / 100
        @test C.choose_kktsolver(spzeros(mk, mk), sparse(1.0I, mk, mk), Gk, cdk;
                                 dense_bytes_max = own) === C.kktsolver_sdpschur
        @test_throws ArgumentError C.choose_kktsolver(spzeros(mk, mk), sparse(1.0I, mk, mk),
                                                      Gk, cdk; dense_bytes_max = own - 1)
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
            @test MOI.get(dopt, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ MOI.get(dopt, MOI.ObjectiveValue()) rtol = 1e-6
            raw = MOI.get(opt, MOI.RawSolver())::ConicIP.Optimizer
            sol = raw.sol
            # Original-data checks: stationarity, feasibility, cone membership.
            Q = raw.Q_int === nothing ? spzeros(raw.n, raw.n) : raw.Q_int
            rDu = norm(Q * sol.y + raw.eq_G' * sol.w - raw.ineq_A' * sol.v - raw.c_int) /
                  (1 + norm(raw.c_int))
            @test rDu <= 1e-6
            @test norm(raw.eq_G * sol.y - raw.eq_d) <= 1e-6 * (1 + norm(raw.eq_d))
            @test norm(raw.ineq_A * sol.y - raw.ineq_b - sol.s) <= 1e-6 * (1 + norm(raw.ineq_b))
            @test abs(dot(sol.s, sol.v)) <= 1e-6 * (1 + abs(MOI.get(opt, MOI.ObjectiveValue())))
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
        # The Farkas ray, checked here rather than by the solver's own
        # validator: Gᵀw = Aᵀv, v ⪰ 0, bᵀv − dᵀw > 0.
        sep = dot(b, sol.v) - dot(d, sol.w)
        @test sep > 0
        @test norm(G' * sol.w - A' * sol.v) <= 1e-8 * sep
        @test eigmin(Symmetric(C.mat(sol.v))) >= -1e-8 * sep
        # Unbounded: min −X₁₁ s.t. X₂₂ = 1.
        G = sparse([0 0 1.0]); d = [1.0]; c = [1.0, 0, 0]
        sol = conicIP(Q, c, A, b, cd, G, d; verbose = false)
        @test sol.status == :DualInfeasible
        # The recession ray: Gy = 0, Ay ⪰ 0, and cᵀy > 0 (the objective
        # ½yᵀQy − cᵀy decreases along it; Q = 0).
        slope = dot(c, sol.y)
        @test slope > 0
        @test norm(G * sol.y) <= 1e-8 * slope
        @test eigmin(Symmetric(C.mat(A * sol.y))) >= -1e-8 * slope
    end

    # ── Mixed form: a PSD variable among other variables and rows ──────
    #
    # Only the cone variables of the eliminated blocks go; free variables,
    # bounds or a second cone on matrix entries, LMIs in the free variables
    # and a quadratic objective on them stay in a reduced system factored
    # by Bunch-Kaufman.

    # Columns: the PSD variables' coordinates (shuffled), then `nfree`
    # free variables, then `nonneg` nonnegative variables used nowhere else.
    # Rows: the PSD blocks; a nonnegative block with `nbound` bounds on
    # matrix entries, the nonnegative variables and (if `freebounds`) bounds
    # on the free variables; optionally an SOC on (free₁, two entries) and an
    # order-3 LMI in the free variables. `rrow` makes the last equality row
    # touch a free variable only, so C_S loses rank.
    function mixed_form(; ks, nfree, nbound, nonneg = 0, soc = false, lmi = false,
                        freebounds = true, p, rrow = false, qfree = true, seed)
        Random.seed!(seed)
        mS = [div(k * (k + 1), 2) for k in ks]; nS = sum(mS)
        n = nS + nfree + nonneg
        perm = randperm(nS)
        fr = nS .+ (1:nfree); nn = nS + nfree .+ (1:nonneg)
        blocks = SparseMatrixCSC{Float64,Int}[]; cd = Tuple{String,Int}[]
        off = 0
        for mk in mS
            push!(blocks, sparse(1:mk, perm[off+1:off+mk],
                                 (rand(mk) .+ 0.5) .* rand([-1, 1], mk), mk, n))
            push!(cd, ("S", mk)); off += mk
        end
        rc = [rand(1:nS, nbound); nn; freebounds ? fr : Int[]]
        if !isempty(rc)
            push!(blocks, sparse(1:length(rc), rc, rand(length(rc)) .+ 0.5, length(rc), n))
            push!(cd, ("R", length(rc)))
        end
        if soc
            push!(blocks, sparse(1:3, [fr[1]; rand(1:nS, 2)], [1.0, 0.7, -0.4], 3, n))
            push!(cd, ("Q", 3))
        end
        if lmi
            B = spzeros(6, n)
            for j in fr; B[:, j] = randn(6); end
            push!(blocks, B); push!(cd, ("S", 6))
        end
        A = reduce(vcat, blocks)
        G = sprandn(p, n, 0.3)
        for i in 1:min(p, nfree); G[i, fr[i]] = 1.0 + rand(); end
        for i in nfree+1:p; G[i, perm[i]] += 2.0; end
        if rrow
            G[p, :] .= 0; G[p, fr[end]] = 1.0
        end
        Q = spzeros(n, n)
        if qfree && nfree > 0
            B = randn(nfree, nfree); Q[fr, fr] = B' * B
        end
        dropzeros!(G)
        return (Q = Q, A = A, G = G, cd = cd, n = n, m = size(A, 1), p = p)
    end

    @testset "mixed form: split" begin
        # X (order 3, cols 1:6), t (col 7), z (col 8).
        n = 8
        A = [sparse(1:6, 1:6, ones(6), 6, n);                 # X ⪰ 0         → P
             sparse(1:3, [1, 8, 7], ones(3), 3, n);           # X₁, z, t ≥ 0  → K, P, O
             sparse(1:3, [7, 2, 3], ones(3), 3, n);           # (t, X₂, X₃) ∈ SOC → K
             sparse([1, 3], [7, 7], [1.0, 1.0], 3, n)]        # LMI in t      → O
        cd = [("S", 6), ("R", 3), ("Q", 3), ("S", 3)]
        sp = C._schur_split(spzeros(n, n), A, cd)
        P, K, O = C._ROW_P, C._ROW_K, C._ROW_O
        @test sp.kind == [P, P, P, P, P, P, K, P, O, K, K, K, O, O, O]
        @test sp.S == [1:6; 8] && sp.R == [7] && sp.K == [7, 10, 11, 12]
        @test !sp.pure
        # A quadratic term on z keeps it (its row becomes O); on X, no split.
        Qz = sparse([8], [8], [1.0], n, n)
        @test C._schur_split(Qz, A, cd).kind[8] == O
        @test C._schur_split(sparse([1], [1], [1.0], n, n), A, cd) === nothing
        # An SDP block on claimed variables is K, not a second P.
        A2 = [A; sparse(1:6, 1:6, ones(6), 6, n)]
        @test all(==(K), C._schur_split(spzeros(n, n), A2, [cd; ("S", 6)]).kind[16:21])
    end

    @testset "mixed form: 3×3 solve agrees with kktsolver_qr" begin
        cases = [
            # everything at once, quadratic objective on the free variables
            (ks = [3], nfree = 2, nbound = 2, nonneg = 1, soc = true, lmi = true, p = 3),
            # free variables in equality rows only: H_R = 0, and C_S rank-deficient
            (ks = [4], nfree = 2, nbound = 0, freebounds = false, p = 3, rrow = true, qfree = false),
            # two PSD variables, bounds and an SOC on entries, no equalities
            (ks = [3, 4], nfree = 1, nbound = 3, nonneg = 2, soc = true, p = 0),
            # no free variables, only bounds on entries
            (ks = [5], nfree = 0, nbound = 4, p = 2),
        ]
        for (ci, kw) in enumerate(cases)
            P = mixed_form(; seed = 70 + ci, kw...)
            sp = C._schur_split(P.Q, P.A, P.cd)
            @test sp !== nothing && !sp.pure
            z = interior(P.cd); s = interior(P.cd)
            F = nt_block(P.cd, z, s); F⁻ᵀ = inv(F)'
            sq = C.kktsolver_qr(P.Q, P.A, P.G, P.cd)(F, F⁻ᵀ)
            ss = C.kktsolver_sdpschur(P.Q, P.A, P.G, P.cd)(F, F⁻ᵀ)
            for _ in 1:2
                b = (randn(P.n), randn(P.p), randn(P.m))
                xs = ss(b...); xq = sq(b...)
                @test resid3(P.Q, P.A, P.G, F, b, xs) <= 1e-11
                for i in 1:3
                    @test norm(xs[i] - xq[i]) <= 1e-9 * max(1, norm(xq[i]))
                end
            end
            Iblk = C.Block([Diagonal(ones(d)) for (_, d) in P.cd])
            s0 = C.kktsolver_sdpschur(P.Q, P.A, P.G, P.cd)(Iblk, Iblk)
            b = (randn(P.n), randn(P.p), randn(P.m))
            @test resid3(P.Q, P.A, P.G, Iblk, b, s0(b...)) <= 1e-11
        end
    end

    @testset "mixed form: regularization and dependent rows" begin
        P = mixed_form(; ks = [3], nfree = 2, nbound = 2, soc = true, p = 3, seed = 81)
        z = interior(P.cd); s = interior(P.cd); F = nt_block(P.cd, z, s); F⁻ᵀ = inv(F)'
        b = (randn(P.n), randn(P.p), randn(P.m))
        exact = C.kktsolver_sdpschur(P.Q, P.A, P.G, P.cd)(F, F⁻ᵀ)(b...)
        regs = C.kktsolver_sdpschur(P.Q, P.A, P.G, P.cd; static_reg = 1e-10)(F, F⁻ᵀ)
        reg = regs(b...)
        @test norm(reg[1] - exact[1]) <= 1e-6 * norm(exact[1])
        @test refined_resid(P.Q, P.A, P.G, F, regs, b)[1] <= 1e-12
        # A repeated equality row makes the reduced matrix singular; the
        # retry's shift still solves a consistent right-hand side.
        G2 = [P.G; P.G[1:1, :]]
        s2 = C.kktsolver_sdpschur(P.Q, P.A, G2, P.cd)(F, F⁻ᵀ)
        b2 = (b[1], [b[2]; b[2][1]], b[3])
        (y, w, v) = s2(b2...)
        @test norm(G2 * y - b2[2]) <= 1e-6 * norm(b2[2])
        @test refined_resid(P.Q, P.A, G2, F, s2, b2)[1] <= 1e-10
    end

    @testset "mixed form: a small row is not a dependent one" begin
        # The standard-form reproducer with a free variable t: equality rows
        # on packed coordinates 1 and 3 (the second also on t, the only
        # place t appears, so H_R = 0) and F = VecCongurance(Diag(1, 1e-4)):
        # the reduced matrix has entries of order 1 and 1e-16.
        # Equilibration keeps the small row from being taken for a
        # dependent one.
        cd = [("S", 3)]
        A = sparse(1.0I, 3, 4)
        G = sparse([1.0 0 0 0; 0 0 1 1]); Q = spzeros(4, 4)
        F = C.Block([C.VecCongurance(Matrix(Diagonal([1, 1e-4])))])
        F⁻ᵀ = C.Block([inv(F[1])'])
        @test C._schur_split(Q, A, cd).R == [4]
        b = (zeros(4), [0.0, 1.0], zeros(3))
        xs = C.kktsolver_sdpschur(Q, A, G, cd)(F, F⁻ᵀ)(b...)
        xq = C.kktsolver_qr(Q, A, G, cd)(F, F⁻ᵀ)(b...)
        @test resid3(Q, A, G, F, b, xs) <= 1e-12
        @test norm(xs[1] - xq[1]) <= 1e-8 * norm(xq[1])
    end

    @testset "mixed form: routing" begin
        # Order 8 (36 coordinates), one free variable, two bounds: the
        # reduced system (1 + 3 + 2 + 1) is far below n/2.
        P = mixed_form(; ks = [8], nfree = 1, nbound = 2, p = 3, seed = 91)
        @test C.sdpschur_eligible(P.Q, P.A, P.G, P.cd)
        @test C.choose_kktsolver(P.Q, P.A, P.G, P.cd) === C.kktsolver_sdpschur
        # Doubly nonnegative: X ⪰ 0 and X .≥ 0 keeps every entry's row, so
        # the reduced system is as large as the dense one: declined.
        k = 6; mk = div(k * (k + 1), 2)
        A = [sparse(1.0I, mk, mk); sparse(1.0I, mk, mk)]
        G = sparse(1:k, [div(i * (i + 1), 2) for i in 1:k], ones(k), k, mk)
        cd = [("S", mk), ("R", mk)]
        @test C._schur_split(spzeros(mk, mk), A, cd) !== nothing
        @test !C.sdpschur_eligible(spzeros(mk, mk), A, G, cd)
        @test C.choose_kktsolver(spzeros(mk, mk), A, G, cd) === C.kktsolver_qr
        # The reduced matrix is budgeted like the standard-form one.
        sp = C._schur_split(P.Q, P.A, P.cd)
        bytes = C._sdpschur_bytes(sp, P.p, size(P.A, 1), P.cd)
        @test C._sdpschur_fits(P.Q, P.A, P.G, P.cd, bytes)
        @test !C._sdpschur_fits(P.Q, P.A, P.G, P.cd, bytes - 1)
        # (Over budget, the dense rules refuse an SDP outright.)
        @test_throws ArgumentError C.choose_kktsolver(P.Q, P.A, P.G, P.cd;
                                                      dense_bytes_max = bytes - 1)
        # Asked for by name, the solver still takes it.
        z = interior(cd); s = interior(cd); F = nt_block(cd, z, s)
        b = (randn(mk), randn(k), randn(2mk))
        sol = C.kktsolver_sdpschur(spzeros(mk, mk), A, G, cd)(F, inv(F)')(b...)
        @test resid3(spzeros(mk, mk), A, G, F, b, sol) <= 1e-11
    end

    # Mixed-form models through MOI, each solved by the default route and by
    # kktsolver_qr on the same model.
    saf(terms, c = 0.0) = MOI.ScalarAffineFunction(terms, c)
    term(c, x) = MOI.ScalarAffineTerm(c, x)
    psd_variable(model, k) = (X = MOI.add_variables(model, div(k * (k + 1), 2));
        MOI.add_constraint(model, MOI.VectorOfVariables(X), MOI.PositiveSemidefiniteConeTriangle(k)); X)

    # Lovász theta with a free scalar: ϑ(H) = min t  s.t.  Z ⪰ 0,
    # Z_ii = t − 1, Z_ij = −1 for the non-adjacent pairs ij ∈ E of H.
    function theta_free_model(k, E)
        model = MOI.Utilities.Model{Float64}()
        Z = psd_variable(model, k)
        t = MOI.add_variable(model)
        for i in 1:k
            MOI.add_constraint(model, saf([term(1.0, Z[tri(i, i)]), term(-1.0, t)]), MOI.EqualTo(-1.0))
        end
        for (i, j) in E
            MOI.add_constraint(model, saf([term(1.0, Z[tri(i, j)])]), MOI.EqualTo(-1.0))
        end
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf([term(1.0, t)]))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return model
    end
    # The same ϑ(H) in standard form: max ⟨J, X⟩, tr X = 1, X_ij = 0 on
    # the edges of H (the pairs not in E).
    function theta_std_model(k, E)
        model = MOI.Utilities.Model{Float64}()
        X = psd_variable(model, k)
        MOI.add_constraint(model, saf([term(1.0, X[tri(i, i)]) for i in 1:k]), MOI.EqualTo(1.0))
        for j in 1:k, i in 1:j-1
            (i, j) in E && continue
            MOI.add_constraint(model, saf([term(1.0, X[tri(i, j)])]), MOI.EqualTo(0.0))
        end
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([term(i == j ? 1.0 : 2.0, X[tri(i, j)]) for j in 1:k for i in 1:j]))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
        return model
    end
    # Shor relaxation of a box-constrained QP: Y = [1 xᵀ; x X] ⪰ 0, Y₁₁ = 1,
    # X_ii ≤ x_i, 0 ≤ x ≤ 1 (bounds on entries of the PSD variable).
    function boxqp_model(k; seed = 3)
        rng = MersenneTwister(seed)
        Qb = randn(rng, k, k); Qb = (Qb + Qb') / 2; c = randn(rng, k)
        model = MOI.Utilities.Model{Float64}()
        Y = psd_variable(model, k + 1)
        MOI.add_constraint(model, saf([term(1.0, Y[tri(1, 1)])]), MOI.EqualTo(1.0))
        for i in 1:k
            x = Y[tri(1, i + 1)]
            MOI.add_constraint(model, saf([term(1.0, Y[tri(i + 1, i + 1)]), term(-1.0, x)]), MOI.LessThan(0.0))
            MOI.add_constraint(model, x, MOI.GreaterThan(0.0))
            MOI.add_constraint(model, x, MOI.LessThan(1.0))
        end
        obj = [term((i == j ? 0.5 : 1.0) * Qb[i, j], Y[tri(i + 1, j + 1)]) for j in 1:k for i in 1:j]
        append!(obj, [term(c[i], Y[tri(1, i + 1)]) for i in 1:k])
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf(obj))
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return model
    end
    # A quadratic objective on free variables: min ⟨L, X⟩/4 + Σ tᵢ²  s.t.
    # X_ii − tᵢ = 1, X ⪰ 0.
    function quadfree_model(k)
        rng = MersenneTwister(4)
        W = zeros(k, k)
        for j in 1:k, i in 1:j-1
            rand(rng) < 0.5 && (W[i, j] = W[j, i] = 1.0)
        end
        L = Diagonal(vec(sum(W; dims = 2))) - W
        model = MOI.Utilities.Model{Float64}()
        X = psd_variable(model, k)
        t = MOI.add_variables(model, k)
        for i in 1:k
            MOI.add_constraint(model, saf([term(1.0, X[tri(i, i)]), term(-1.0, t[i])]), MOI.EqualTo(1.0))
        end
        aff = [term((i == j ? 1.0 : 2.0) * L[i, j] / 4, X[tri(i, j)]) for j in 1:k for i in 1:j if L[i, j] != 0]
        f = MOI.ScalarQuadraticFunction([MOI.ScalarQuadraticTerm(2.0, ti, ti) for ti in t], aff, 0.0)
        MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(), f)
        MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        return model
    end

    function moi_solve(model, kkt)
        opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
        MOI.set(opt, MOI.Silent(), true)
        MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), kkt)
        MOI.copy_to(opt, model); MOI.optimize!(opt)
        return opt
    end
    function original_residuals_ok(opt)
        raw = MOI.get(opt, MOI.RawSolver())::ConicIP.Optimizer
        sol = raw.sol
        Q = raw.Q_int === nothing ? spzeros(raw.n, raw.n) : raw.Q_int
        rDu = norm(Q * sol.y + raw.eq_G' * sol.w - raw.ineq_A' * sol.v - raw.c_int) /
              (1 + norm(raw.c_int))
        rPr = norm(raw.ineq_A * sol.y - sol.s - raw.ineq_b) / (1 + norm(raw.ineq_b))
        pobj = 0.5 * dot(sol.y, Q * sol.y) - dot(raw.c_int, sol.y)
        gap = abs(dot(sol.v, sol.s)) / (1 + abs(pobj))
        return rDu <= 1e-6 && rPr <= 1e-6 && gap <= 1e-6 &&
               norm(raw.eq_G * sol.y - raw.eq_d) <= 1e-6 * (1 + norm(raw.eq_d)) &&
               C.cone_margin(sol.s, raw.cone_dims) >= -1e-8 &&
               C.cone_margin(sol.v, raw.cone_dims) >= -1e-8
    end

    @testset "mixed form through MOI: $name" for (name, build) in (
            ("theta with a free scalar", () -> theta_free_model(12,
                [(i, j) for j in 1:12 for i in 1:j-1 if (i + 2j) % 5 == 0])),
            ("box QP relaxation", () -> boxqp_model(12)),
            ("quadratic objective on free variables", () -> quadfree_model(8)))
        model = build()
        probe = _RouteProbe("")
        opt = moi_solve(model, probe)
        @test probe.name == "kktsolver_sdpschur"
        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test original_residuals_ok(opt)
        ref = moi_solve(model, C.kktsolver_qr)
        @test MOI.get(ref, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ MOI.get(ref, MOI.ObjectiveValue()) rtol = 1e-6
        if name == "theta with a free scalar"
            E = [(i, j) for j in 1:12 for i in 1:j-1 if (i + 2j) % 5 == 0]
            std = moi_solve(theta_std_model(12, E), C.default_kktsolver)
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ MOI.get(std, MOI.ObjectiveValue()) rtol = 1e-6
        end
    end

    # ── Second review pass (2026-10-16) ─────────────────────────────────
    # Reproducers from the adversarial review of the first round of fixes;
    # each failed before the change it pins.

    @testset "second review: adjacent subnormal divisors" begin
        # 2·(x/(λᵢ+λⱼ)): the sum 3u is exact, halving it first rounded the
        # denominator to 2u and gave 1/2 for 2/3.
        u = nextfloat(0.0); o = zeros(3)
        C.dsdc_diag!(fill(u, 3), [u, 0.0, 2u], o)
        ref = Float64(2 * big(u) / (big(u) + 2 * big(u)))
        @test o[2] == ref
        @test o[1] == 1.0 && o[3] == 0.5
    end

    @testset "second review: staticReg is dimensionless on the Schur route" begin
        # A quadratic term of scale 1e12 on a free variable beside a PSD
        # variable. conicIP's absolute δ = staticReg·(1 + ‖Q‖) applied as a
        # relative shift on the equilibrated matrix was a shift of 1e4:
        # Abandoned with y₄ ≈ 0.015, where kktsolver_qr solves it.
        A = sparse([1.0 0 0 0; 0 1 0 0; 0 0 1 0]); Q = spdiagm([0, 0, 0, 1e12])
        c = [-1.0, 0, -1, 1e12]; cd = [("S", 3)]
        @test C.choose_kktsolver(Q, A, spzeros(0, 4), cd) === C.kktsolver_sdpschur
        for ks in (C.default_kktsolver, C.kktsolver_qr)
            s = conicIP(Q, c, A, zeros(3), cd; staticReg = 1e-8, equilibrate = false,
                        maxIters = 40, verbose = false, kktsolver = ks)
            @test s.status == :Optimal
            @test s.y[4] ≈ 1.0 atol = 1e-6
        end
    end

    @testset "second review: a non-finite trace form is recomputed" begin
        # T = RRᵀ = 1e200 overflows in the four-index products although
        # RᵀGR = 1: the diagonal came out NaN and passed the check.
        plan = C._gram_plan(sparse(reshape([1e-200], 1, 1)), [("S", 1)])
        M = zeros(1, 1)
        C._gram!(M, plan, C.Block([C.VecCongurance(fill(1e100, 1, 1))]))
        @test M[1] ≈ 1.0 rtol = 1e-12
        @test plan.suspect == [1]
    end

    @testset "second review: suspect rows recomputed in bounded chunks" begin
        # Two columns that cancel in the trace form (R's leading block
        # [1 0; 1 1e-4], J₁ = [1 −1; −1 1] there, J₂ = J₁ + 1e-8 e₃e₃ᵀ) next
        # to unrelated ones. The diagonals were repaired but the
        # off-diagonal M₁₂ = 1e-16 stayed 0 whenever the dense W·B exceeded
        # its 1 GiB cap. Now the suspect rows are recomputed from W·B
        # streamed in chunks; a one-column workspace forces many.
        k = 6; m = div(k * (k + 1), 2); cd = [("S", m)]
        R = Matrix(1.0I, k, k); R[1:2, 1:2] = [1 0; 1 1e-4]
        J1 = zeros(k, k); J1[1:2, 1:2] = [1 -1; -1 1]
        J2 = copy(J1); J2[3, 3] = 1e-8
        cols = [C.vecm(J1), C.vecm(J2)]
        for (i, j) in ((4, 4), (4, 5), (5, 6), (6, 6))
            E = zeros(k, k); E[i, j] = E[j, i] = 1.0; push!(cols, C.vecm(E))
        end
        B = sparse(reduce(hcat, cols))
        W = C.VecCongurance(R)
        Y = reduce(hcat, [W * Vector(B[:, j]) for j in 1:size(B, 2)])
        Md = Y' * Y
        saved = C._GRAM_WORKSPACE_BYTES[]
        try
            C._GRAM_WORKSPACE_BYTES[] = 8 * 4 * m           # one column per chunk
            plan = C._gram_plan(B, cd)
            M = zeros(size(B, 2), size(B, 2)); C._gram!(M, plan, C.Block([W]))
            @test plan.suspect == [1, 2]
            @test plan.fallback[1] == 1
            @test M[1:2, 1:2] ≈ Md[1:2, 1:2] rtol = 1e-8
            @test M[1, 2] > 0.5e-16
            D = Diagonal(1 ./ sqrt.(diag(Md)))
            @test opnorm(D * (M - Md) * D) <= 1e-8
        finally
            C._GRAM_WORKSPACE_BYTES[] = saved
        end
    end

    @testset "second review: zero-diagonal equality in the mixed form" begin
        # An equality on a remaining variable whose Hessian entry the
        # scaling makes 1e16: diagonal equilibration left [1 1e-8; 1e-8 0],
        # the small pivot triggered the shift, and y₄ came out 1e-4 for 1.
        # Ruiz balancing scales the zero-diagonal row by its coupling; no
        # dense fallback is needed.
        A = sparse(1.0I, 4, 4); Q = spdiagm([0, 0, 0, 1.0]); G = sparse([0 0 0 1.0])
        cd = [("S", 3), ("R", 1)]
        @test C.choose_kktsolver(Q, A, G, cd) === C.kktsolver_sdpschur
        F = C.Block([C.VecCongurance(Matrix(1.0I, 2, 2)), Diagonal([1e-8])])
        Fi = C.Block([inv(F[1])', Diagonal([1e8])])
        g = C.kktsolver_sdpschur(Q, A, G, cd)
        b = (zeros(4), [1.0], zeros(4))
        sol = g(F, Fi)(b...)
        @test sol[1][4] ≈ 1.0 rtol = 1e-12
        @test resid3(Q, A, G, F, b, sol) <= 1e-12
        @test g.stats[2] == 0
    end

    @testset "second review: correlated equality rows fall back to kktsolver_qr" begin
        # Independent, well-conditioned equality rows whose Schur rows agree
        # to 1e-16 under the scaling: forming M loses the difference and the
        # retry shift overwhelms it (residual 0.7). The residual check hands
        # the factorization to kktsolver_qr, which solves it exactly.
        A = sparse(1.0I, 3, 3); Q = spzeros(3, 3); G = sparse([1.0 0 0; 1 0 1]); cd = [("S", 3)]
        F = C.Block([C.VecCongurance(Matrix(Diagonal([1, 1e-4])))]); Fi = C.Block([inv(F[1])'])
        b = (zeros(3), [0.0, 1.0], zeros(3))
        g = C.kktsolver_sdpschur(Q, A, G, cd)
        sol = g(F, Fi)(b...)
        @test sol[1] ≈ [0.0, 0.0, 1.0] atol = 1e-12
        @test resid3(Q, A, G, F, b, sol) <= 1e-12
        @test g.stats[2] == 1
        # Without room for the dense path the best Schur solve is returned
        # and counted, not raised.
        g0 = C.kktsolver_sdpschur(Q, A, G, cd; dense_bytes_max = 0)
        sol0 = g0(F, Fi)(b...)
        @test all(isfinite, sol0[1])
        @test g0.stats[2] == 0 && g0.stats[3] == 1
        # A well-conditioned solve is accepted as it is.
        P = standard_form([("S", 10)], 3; seed = 7)
        z = interior([("S", 10)]); s = interior([("S", 10)])
        Fg = nt_block([("S", 10)], z, s)
        gg = C.kktsolver_sdpschur(P.Q, P.A, P.G, [("S", 10)])
        gg(Fg, inv(Fg)')(randn(P.m), randn(P.p), randn(P.m))
        @test gg.stats == [0, 0, 0, 0]
    end

    @testset "second review: a large SOC is streamed and budgeted" begin
        # A PSD variable of order 14 beside 50 remaining variables (Q_RR = I)
        # that touch a second-order cone of 200 000 rows. The reduced system
        # is 50×50, but the SOC's dense products W_B B_B (two 200 000 × 50
        # buffers, 160 MB) were neither bounded nor budgeted: the router
        # estimated 80 kB. They now stream within the Gram workspace, and
        # the estimate covers what a factorization allocates.
        Random.seed!(3)
        k = 14; mS = div(k * (k + 1), 2); nR = 50; d = 200_000
        n = mS + nR; cd = [("S", mS), ("Q", d)]
        As = sparse(1:mS, 1:mS, ones(mS), mS, n)
        rows = [1; rand(2:d, 4nR)]; colsR = [mS + 1; mS .+ rand(1:nR, 4nR)]
        Aq = sparse(rows, colsR, randn(length(rows)), d, n)
        A = [As; Aq]; Q = blockdiag(spzeros(mS, mS), sparse(1.0I, nR, nR))
        G = spzeros(0, n)
        sp = C._schur_split(Q, A, cd)
        @test sp !== nothing && !sp.pure && length(sp.R) == nR
        z = interior(cd); s = interior(cd); F = nt_block(cd, z, s); Fi = inv(F)'
        saved = C._GRAM_WORKSPACE_BYTES[]
        try
            C._GRAM_WORKSPACE_BYTES[] = 2^25                 # 32 MiB: 6-column chunks
            est = C._sdpschur_bytes(sp, 0, size(A, 1), cd)
            g = C.kktsolver_sdpschur(Q, A, G, cd)
            g(F, Fi); g(F, Fi)                               # warm up
            bytes = @allocated g(F, Fi)
            @test bytes <= est
            @test bytes < 40 * 2^20
            @test C._sdpschur_fits(Q, A, G, cd, est)
            @test !C._sdpschur_fits(Q, A, G, cd, est - 1)
        finally
            C._GRAM_WORKSPACE_BYTES[] = saved
        end
    end

    # ── Third review pass (2026-10-16) ──────────────────────────────────

    @testset "third review: a tiny numerator over a normal mean" begin
        # 2·(x/σ) underflowed u/2 to zero before doubling; x/(σ/2) is exact.
        u = nextfloat(0.0); o = zeros(3)
        C.dsdc_diag!([0.0, u, 0.0], [1.0, 0.0, 1.0], o)
        @test o[2] == u
        C.dsdc_diag!(fill(u, 3), [u, 0.0, 2u], o)          # still 2/3
        @test o[2] == Float64(2 * big(u) / (3 * big(u)))
    end

    @testset "third review: the dense fallback never replaces a better solve" begin
        # A 1e12 quadratic term on the free variable and staticReg = 1e-8:
        # the dense path's absolute δ ≈ 1e4 lands on every primal
        # coordinate and its solve is wrong (residual ≈ 1), while the Schur
        # solve is accurate. The better one must come back.
        A = sparse(1.0I, 4, 4); Q = spdiagm([0, 0, 0, 1e12]); G = spzeros(0, 4)
        cd = [("S", 3), ("R", 1)]
        F = C.Block([C.VecCongurance(100 * Matrix(1.0I, 2, 2)), Diagonal([1.0])])
        Fi = C.Block([inv(F[1])', Diagonal([1.0])])
        b = ([1.0, 2, 3, 0], zeros(0), [0.01, 0.1, 0.3, 0])
        g = C.kktsolver_sdpschur(Q, A, G, cd; static_reg = 1e-8)
        sol = g(F, Fi)(b...)
        ref = C.kktsolver_qr(Q, A, G, cd)(F, Fi)(b...)        # unregularized
        @test sol[1] ≈ ref[1] rtol = 1e-6
        @test resid3(Q, A, G, F, b, sol) <= 1e-6
    end

    @testset "third review: a large legitimate solution is accepted" begin
        # Q = 0, A = I, no equalities, F = 100·I: the solution is ~1e8 for an
        # order-one right-hand side and cancellation leaves ~1e-8 of it in
        # the residual. That is backward-stable; no dense attempt is made.
        A = sparse(1.0I, 3, 3); Q = spzeros(3, 3); G = spzeros(0, 3); cd = [("S", 3)]
        F = C.Block([C.VecCongurance(100 * Matrix(1.0I, 2, 2))]); Fi = C.Block([inv(F[1])'])
        g = C.kktsolver_sdpschur(Q, A, G, cd)
        g(F, Fi)([1.0, 2, 3], zeros(0), [0.01, 0.1, 0.3])
        @test g.stats == [0, 0, 0, 0]
    end

    @testset "third review: the residual check is scale-invariant" begin
        # The correlated-rows reproducer with its right-hand side scaled by
        # 1e±170 (squared norms underflowed to a residual of 0 and accepted
        # a wrong solve), and a scalar system with by = 1e155 under
        # staticReg (the squared norm overflowed).
        A = sparse(1.0I, 3, 3); Q = spzeros(3, 3); G = sparse([1.0 0 0; 1 0 1]); cd = [("S", 3)]
        F = C.Block([C.VecCongurance(Matrix(Diagonal([1, 1e-4])))]); Fi = C.Block([inv(F[1])'])
        for sc in (1e-170, 1.0, 1e170)
            g = C.kktsolver_sdpschur(Q, A, G, cd)
            sol = g(F, Fi)(zeros(3), [0.0, sc], zeros(3))
            @test sol[1] ≈ [0.0, 0.0, sc] atol = 1e-12 * sc
            @test g.stats[2] == 1
        end
        A1 = sparse(reshape([1.0], 1, 1)); cd1 = [("S", 1)]
        F1 = C.Block([C.VecCongurance(fill(1.0, 1, 1))])
        for by in (1e-155, 1.0, 1e155)
            g = C.kktsolver_sdpschur(spzeros(1, 1), A1, A1, cd1; static_reg = 1e-8)
            sol = g(F1, F1)(zeros(1), [by], zeros(1))
            @test sol[1][1] ≈ by rtol = 1e-12
        end
    end

    @testset "third review: a failing dense fallback keeps the Schur solve" begin
        # The correlated rows plus a repeated (consistent) row: the dense
        # path's QR of Gᵀ is singular. The Schur solve is returned, the
        # fallback is switched off and counted, nothing is raised.
        A = sparse(1.0I, 3, 3); Q = spzeros(3, 3)
        G = sparse([1.0 0 0; 1 0 1; 1 0 1]); cd = [("S", 3)]
        F = C.Block([C.VecCongurance(Matrix(Diagonal([1, 1e-4])))]); Fi = C.Block([inv(F[1])'])
        g = C.kktsolver_sdpschur(Q, A, G, cd)
        sol = g(F, Fi)(zeros(3), [0.0, 1.0, 1.0], zeros(3))
        @test all(isfinite, sol[1])
        @test g.stats[4] == 1 && g.stats[3] == 1
        g(F, Fi)(zeros(3), [0.0, 1.0, 1.0], zeros(3))      # not retried
        @test g.stats[2] <= 1 && g.stats[4] == 1
    end

    @testset "third review: stable row recomputation stays within its estimate" begin
        # Suspect rows of a plan with a 100 000-row SOC: the products went
        # through allocating block multiplications (six live SOC-sized
        # arrays). Now two preallocated chunk buffers.
        Random.seed!(11)
        k = 8; mS = div(k * (k + 1), 2); d = 100_000; nc = 32
        cd = [("S", mS), ("Q", d)]; m = mS + d
        B = [sprandn(mS, nc, 0.3); sprandn(d, nc, 1e-4)]
        z = interior(cd); s = interior(cd); W = nt_block(cd, z, s)
        plan = C._gram_plan(B, cd)
        M = zeros(nc, nc)
        C._gram_rows!(M, plan, W, collect(1:nc))
        bytes = @allocated C._gram_rows!(M, plan, W, collect(1:nc))
        @test bytes <= C._gram_bytes(m, nc, cd)
        Y = W * Matrix(B)
        @test M ≈ Y' * Y rtol = 1e-12
    end
end
