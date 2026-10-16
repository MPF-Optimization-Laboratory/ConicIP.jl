# Krylov feasibility study on ConicIP's captured KKT systems.
#
#   julia --project=. experiments.jl [instance ...]

using Serialization, SparseArrays, LinearAlgebra, Printf, Statistics
using Krylov, QDLDL, KrylovKit, LimitedLDLFactorizations, LinearOperators

const DIR = @__DIR__
const DATA = joinpath(DIR, "data")
BLAS.set_num_threads(1)

# ── helpers ───────────────────────────────────────────────────────────
upperK(meta, nz) = SparseMatrixCSC(meta.N, meta.N, copy(meta.colptr),
                                   copy(meta.rowval), copy(nz))
function fullK(Ku)
    d = Diagonal(diag(Ku))
    return Ku + sparse(Ku') - d
end

function dsigns(meta)
    s = Vector{Int}(undef, meta.N)
    s[1:meta.n] .= 1
    s[meta.n+1:meta.oa] .= -1
    for j in 1:meta.nlift
        s[meta.oa + 2j - 1] = 1
        s[meta.oa + 2j] = -1
    end
    return s
end

fac(Ku, meta) = QDLDL.qdldl(Ku; perm = copy(meta.perm), Dsigns = dsigns(meta),
                            regularize_eps = 1e-13, regularize_delta = 2e-7)

# LinearOperator wrapping an action v ↦ apply!(z, v); prod! is (res, v, α, β).
function actionop(N, apply!)
    z = zeros(N)
    return LinearOperator(Float64, N, N, true, true,
        (res, v, α, β) -> begin
            apply!(z, v)
            if β == 0
                res .= α .* z
            else
                res .= α .* z .+ β .* res
            end
            res
        end)
end
qdop(F, N) = actionop(N, (z, v) -> (z .= v; QDLDL.solve!(F, z); z))
lldlop(P, N) = actionop(N, (z, v) -> (z .= P \ v; z))

# iterations of a residual history to reach a relative tolerance
function iters_to(hist, nb, tol)
    for (i, r) in enumerate(hist)
        r <= tol * nb && return i - 1        # hist[1] is the initial residual
    end
    return -1
end

# extreme |eigenvalue| estimates; κ ≈ |λ|max(K) · |λ|max(K⁻¹)
function cond_est(Kf, solveop, N; tol = 1e-4)
    vals, = KrylovKit.eigsolve(x -> Kf * x, randn(N), 1, :LM;
                               issymmetric = true, tol = tol, maxiter = 80,
                               krylovdim = 24)
    lmax = abs(vals[1])
    ivals, = KrylovKit.eigsolve(solveop, randn(N), 1, :LM;
                                issymmetric = true, tol = tol, maxiter = 80,
                                krylovdim = 24)
    imax = abs(ivals[1])
    return (lmax, 1 / imax, lmax * imax)
end

# ── one instance ──────────────────────────────────────────────────────
function run_instance(name; itcap = 2000, ircap = 60)
    d = deserialize(joinpath(DATA, name * ".jls"))
    meta = d.meta; recs = d.recs; T = length(recs)
    @printf("\n================ %s  (n=%d m=%d p=%d N=%d T=%d) ================\n",
            name, meta.n, meta.m, meta.p, meta.N, T)
    nzK = length(meta.rowval)
    isLP = meta.p == 0 && all(k -> k == :diag, meta.kinds)
    @printf("nnz(triu K)=%d  δp=%g δe=%g δc=%g  nlift=%d  normal-eq path=%s\n",
            nzK, meta.dp, meta.de, meta.dc, meta.nlift, isLP)

    tset = sort(unique(clamp.([1, max(1, T ÷ 2), max(1, T - 2), T], 1, T)))

    Ku = [upperK(meta, r.nzval) for r in recs]
    Facs = Dict{Int, Any}()
    getfac(t) = get!(Facs, t) do; fac(Ku[t], meta); end

    # ---- direct-solve reference timings -------------------------------
    t_fac_total = sum(r.tfac for r in recs)
    t_sol_total = sum(sum(r.tsolve) for r in recs)
    nsolves = sum(length(r.rhss) for r in recs)
    F1 = getfac(T)
    xb = randn(meta.N)
    tbs = minimum(( (y = copy(xb); @elapsed QDLDL.solve!(F1, y)) for _ in 1:20 ))
    tfacbare = minimum(( @elapsed(fac(Ku[T], meta)) for _ in 1:3 ))
    Kf_ref = fullK(Ku[T]); v = randn(meta.N); w = similar(v)
    tmv = minimum(( @elapsed(mul!(w, Kf_ref, v)) for _ in 1:50 ))
    @printf("direct: T_fac=%.4fs over %d factorizations, T_solve=%.4fs over %d solves\n",
            t_fac_total, T, t_sol_total, nsolves)
    @printf("bare:   factor %.3f ms | back-solve %.3f ms | matvec %.3f ms  (nnz(L)=%d)\n",
            1e3tfacbare, 1e3tbs, 1e3tmv, nnz(F1.L))

    @printf("break-even: one factorization = %.1f back-solves; Krylov iteration ≈ 1 back-solve + %.2f matvec\n",
            tfacbare / tbs, tmv / tbs)

    rows = String[]
    push!(rows, "# $name")

    # ---- 1. conditioning ----------------------------------------------
    println("\n[1] conditioning")
    @printf("  %-4s %12s %12s %12s | %12s\n", "t", "|λ|max(K)", "|λ|min(K)", "cond(K)", "cond(N)")
    condK = Dict{Int,Float64}()
    for t in tset
        Kf = fullK(Ku[t]); Ft = getfac(t)
        sop = x -> (y = copy(x); QDLDL.solve!(Ft, y); y)
        (lmax, lmin, κ) = cond_est(Kf, sop, meta.N)
        condK[t] = κ
        κN = NaN
        if isLP
            NN = normal_matrix(Ku[t], meta)
            κN = cond_spd(NN)
        end
        @printf("  %-4d %12.3e %12.3e %12.3e | %12.3e\n", t, lmax, lmin, κ, κN)
        push!(rows, @sprintf("cond,%d,%.6e,%.6e,%.6e,%.6e", t, lmax, lmin, κ, κN))
    end

    # ---- 2. unpreconditioned / Jacobi MINRES ---------------------------
    println("\n[2] MINRES on K_t x = rhs_t (predictor rhs)")
    @printf("  %-4s %9s %9s %9s %8s | %9s %9s %9s %8s\n",
            "t", "plain@8", "plain@12", "best_rel", "t_plain",
            "jac@8", "jac@12", "best_rel", "t_jac")
    for t in tset
        Kf = fullK(Ku[t]); b = recs[t].rhss[1]; nb = norm(b)
        cap = min(itcap, meta.N)
        st1 = @timed Krylov.minres(Kf, b; rtol = 1e-12, atol = 0.0,
                                   itmax = cap, history = true)
        h1 = st1.value[2].residuals
        Mj = Diagonal(1 ./ max.(abs.(diag(Kf)), 1e-300))
        st2 = @timed Krylov.minres(Kf, b; M = Mj, rtol = 1e-12, atol = 0.0,
                                   itmax = cap, history = true)
        h2 = st2.value[2].residuals
        # Jacobi-preconditioned MINRES history is in the M-norm; recompute true
        i2_8 = iters_to(h2, h2[1], 1e-8); i2_12 = iters_to(h2, h2[1], 1e-12)
        @printf("  %-4d %9s %9s %9.1e %7.2fs | %9s %9s %9.1e %7.2fs\n", t,
                fmt(iters_to(h1, nb, 1e-8)), fmt(iters_to(h1, nb, 1e-12)),
                minimum(h1)/nb, st1.time,
                fmt(i2_8), fmt(i2_12), minimum(h2)/h2[1], st2.time)
        push!(rows, @sprintf("minres,%d,%d,%d,%.4f,%d,%d,%.4f", t,
                             iters_to(h1, nb, 1e-8), iters_to(h1, nb, 1e-12), st1.time,
                             i2_8, i2_12, st2.time))
    end

    # ---- 3. stale-factorization preconditioner -------------------------
    println("\n[3] stale LDLᵀ of K_{t−k} as preconditioner for K_t")
    @printf("  %-4s %-3s %9s %8s %8s %9s %9s %9s %9s\n",
            "t", "k", "relΔK", "IR@8", "IR@12", "IR_best", "GMRES@8", "GMRES@12", "t_gmres12")
    for t in tset, k in (1, 2, 3, 5)
        t - k >= 1 || continue
        Kf = fullK(Ku[t]); b = recs[t].rhss[1]; nb = norm(b)
        Fs = getfac(t - k)
        # iterative refinement x ← x + K_{t-k}⁻¹ (b − K_t x)
        x = zeros(meta.N); r = copy(b); hist = [nb]
        i8 = -1; i12 = -1
        for it in 1:ircap
            y = copy(r); QDLDL.solve!(Fs, y); x .+= y
            mul!(r, Kf, x); r .= b .- r
            rn = norm(r); push!(hist, rn)
            i8 < 0 && rn <= 1e-8nb && (i8 = it)
            i12 < 0 && rn <= 1e-12nb && (i12 = it)
            i12 > 0 && break
            rn > 1e6nb && break                 # diverging
        end
        pop = qdop(Fs, meta.N)
        stg = @timed Krylov.gmres(Kf, b; N = pop, rtol = 1e-12, atol = 0.0,
                                  itmax = min(200, meta.N), memory = 50,
                                  restart = true, history = true)
        hg = stg.value[2].residuals
        dK = norm(recs[t].nzval .- recs[t-k].nzval) / norm(recs[t].nzval)
        @printf("  %-4d %-3d %9.2e %8s %8s %9.1e %9s %9s %8.2fs\n", t, k, dK,
                fmt(i8), fmt(i12), minimum(hist[2:end])/nb, fmt(iters_to(hg, nb, 1e-8)),
                fmt(iters_to(hg, nb, 1e-12)), stg.time)
        push!(rows, @sprintf("stale,%d,%d,%d,%d,%d,%d,%.4f", t, k, i8, i12,
                             iters_to(hg, nb, 1e-8), iters_to(hg, nb, 1e-12), stg.time))
    end

    # ---- 4. incomplete factorization ----------------------------------
    println("\n[4] limited-memory LDLᵀ preconditioner")
    for t in tset
        b = recs[t].rhss[1]
        if isLP
            NN = normal_matrix(Ku[t], meta)
            bn = normal_rhs(Ku[t], meta, b)
            nbn = norm(bn)
            for mem in (0, 5, 20)
                ok = true; P = nothing; tp = 0.0
                try
                    tp = @elapsed P = lldl(sparse(tril(NN)); memory = mem)
                catch e; ok = false; end
                if !ok; @printf("  t=%-3d N_t lldl(mem=%d): failed\n", t, mem); continue; end
                op = lldlop(P, size(NN,1))
                st = @timed Krylov.cg(NN, bn; M = op, rtol = 1e-12, atol = 0.0,
                                      itmax = min(500, size(NN,1)), history = true)
                h = st.value[2].residuals
                @printf("  t=%-3d N_t  lldl(mem=%2d) nnz(L)=%8d setup %.3fs | PCG@8=%s @12=%s  %.3fs\n",
                        t, mem, nnz(P.L), tp, fmt(iters_to(h, nbn, 1e-8)),
                        fmt(iters_to(h, nbn, 1e-12)), st.time)
                push!(rows, @sprintf("illdl_N,%d,%d,%d,%.4f,%d,%d,%.4f", t, mem,
                                     nnz(P.L), tp, iters_to(h, nbn, 1e-8),
                                     iters_to(h, nbn, 1e-12), st.time))
            end
            # Jacobi PCG on N, for reference
            let Mj = Diagonal(1 ./ diag(NN))
                st = @timed Krylov.cg(NN, bn; M = Mj, rtol = 1e-12, atol = 0.0,
                                      itmax = min(2000, size(NN,1)), history = true)
                h = st.value[2].residuals
                @printf("  t=%-3d N_t  jacobi           | PCG@8=%s @12=%s  best %.1e  %.3fs\n",
                        t, fmt(iters_to(h, nbn, 1e-8)), fmt(iters_to(h, nbn, 1e-12)),
                        minimum(h)/nbn, st.time)
                push!(rows, @sprintf("jac_N,%d,%d,%d,%.4f", t,
                                     iters_to(h, nbn, 1e-8), iters_to(h, nbn, 1e-12), st.time))
            end
            # plain CG on N
            let st = @timed Krylov.cg(NN, bn; rtol = 1e-12, atol = 0.0,
                                      itmax = min(2000, size(NN,1)), history = true)
                h = st.value[2].residuals
                @printf("  t=%-3d N_t  none             | CG@8=%s  @12=%s  best %.1e  %.3fs\n",
                        t, fmt(iters_to(h, nbn, 1e-8)), fmt(iters_to(h, nbn, 1e-12)),
                        minimum(h)/nbn, st.time)
                push!(rows, @sprintf("cg_N,%d,%d,%d,%.4f", t,
                                     iters_to(h, nbn, 1e-8), iters_to(h, nbn, 1e-12), st.time))
            end
            # exact Cholesky reference on N
            tchol = @elapsed Ch = cholesky(Symmetric(NN))
            tcs = minimum(((@elapsed (Ch \ bn)) for _ in 1:10))
            @printf("  t=%-3d N_t  chol nnz(L)=%d  %.3fs factor, %.3f ms solve\n",
                    t, nnz(sparse(Ch.L)), tchol, 1e3tcs)
            push!(rows, @sprintf("chol_N,%d,%d,%.4f,%.6f", t, nnz(sparse(Ch.L)), tchol, tcs))
        else
            Kf = fullK(Ku[t]); nb = norm(b)
            for mem in (0, 10, 30)
                P = nothing; tp = 0.0
                try
                    tp = @elapsed P = lldl(sparse(tril(Kf)); memory = mem)
                catch e
                    @printf("  t=%-3d K_t lldl(mem=%d): failed (%s)\n", t, mem, sprint(showerror, e)[1:min(60,end)])
                    continue
                end
                op = lldlop(P, meta.N)
                st = @timed Krylov.gmres(Kf, b; N = op, rtol = 1e-12, atol = 0.0,
                                         itmax = min(300, meta.N), memory = 50,
                                         restart = true, history = true)
                h = st.value[2].residuals
                @printf("  t=%-3d K_t  lldl(mem=%2d) nnz(L)=%8d α=%.1e setup %.3fs | GMRES@8=%s @12=%s  %.3fs\n",
                        t, mem, nnz(P.L), (hasproperty(P, :α) ? P.α : NaN), tp, fmt(iters_to(h, nb, 1e-8)),
                        fmt(iters_to(h, nb, 1e-12)), st.time)
                push!(rows, @sprintf("illdl_K,%d,%d,%d,%.4f,%d,%d,%.4f", t, mem,
                                     nnz(P.L), tp, iters_to(h, nb, 1e-8),
                                     iters_to(h, nb, 1e-12), st.time))
            end
        end
    end

    push!(rows, @sprintf("direct,%d,%d,%.6f,%.6f,%.6f,%.6f,%.6f,%d", T, nsolves,
                         t_fac_total, t_sol_total, tfacbare, tbs, tmv, nnz(F1.L)))
    write(joinpath(DIR, "res-$name.csv"), join(rows, "\n") * "\n")
    return nothing
end

fmt(i) = i < 0 ? "—" : string(i)

# Normal equations for the LP/QP case (p = 0, diagonal FᵀF):
#   K = [Q+δp I  −Aᵀ ; −A  −(FᵀF+δc I)] ⇒ N = Q + δp I + Aᵀ D⁻¹ A, D = FᵀF+δc I
function normal_matrix(Ku, meta)
    n = meta.n; oz = meta.oz; m = meta.m
    Q11 = Ku[1:n, 1:n]                      # upper triangle of Q + δp I
    Q11 = Q11 + sparse(Q11') - Diagonal(diag(Q11))
    At = -Ku[1:n, oz+1:oz+m]                # block (1,3) of K is −Aᵀ
    D = -diag(Ku[oz+1:oz+m, oz+1:oz+m])     # = FᵀF + δc I  (positive)
    return Q11 + At * Diagonal(1 ./ D) * sparse(At')
end
function normal_rhs(Ku, meta, b)
    n = meta.n; oz = meta.oz; m = meta.m
    At = -Ku[1:n, oz+1:oz+m]
    D = -diag(Ku[oz+1:oz+m, oz+1:oz+m])
    return Vector(b[1:n] - At * ((b[oz+1:oz+m]) ./ D))
end
function cond_spd(NN)
    Ch = cholesky(Symmetric(NN))
    n = size(NN, 1)
    vals, = KrylovKit.eigsolve(x -> NN * x, randn(n), 1, :LM;
                               issymmetric = true, tol = 1e-4, maxiter = 80, krylovdim = 24)
    ivals, = KrylovKit.eigsolve(x -> Ch \ x, randn(n), 1, :LM;
                                issymmetric = true, tol = 1e-4, maxiter = 80, krylovdim = 24)
    return abs(vals[1]) * abs(ivals[1])
end

names = isempty(ARGS) ? ["qp-band-20000", "lp-band-20000", "issue10",
                         "nql30", "chainsing-1000-1", "socp-sumnorms-1500"] : ARGS
for nm in names
    try
        run_instance(nm)
    catch e
        println("FAILED $nm: ", sprint(showerror, e))
        for (i, f) in enumerate(stacktrace(catch_backtrace())[1:min(8,end)]); println("   ", f); end
    end
end
println("\nALL DONE")
