# exp-activeset, experiments 1-5 for LP instances (v2).
include(joinpath(@__DIR__, "lib.jl"))

# ── Ye projection for a *given* predicted N set ────────────────────────────
function ye_project(prob, Nhat::BitVector, y, w, v)
    m = size(prob.A, 1); p = size(prob.G, 1)
    Nidx = findall(Nhat); Bidx = findall(.!Nhat)
    AN = prob.A[Nidx, :]
    t1 = @elapsed yp = minnorm_proj([AN; prob.G], [prob.b[Nidx]; prob.d], y)[1]
    Md = hcat(sparse(prob.G'), -sparse(AN'))
    t2 = @elapsed xp = minnorm_proj(Md, Vector(prob.c), [w; v[Nidx]])[1]
    wp = xp[1:p]; up = xp[p+1:end]
    sp = zeros(m); sp[Bidx] = prob.A[Bidx, :] * yp - prob.b[Bidx]
    vp = zeros(m); vp[Nidx] = up
    normb = norm(prob.b); normd = norm(prob.d); normc = norm(prob.c)
    rPr = norm(prob.A * yp - sp - prob.b) / (1 + normb)
    rEq = p == 0 ? 0.0 : norm(prob.G * yp - prob.d) / (1 + normd)
    rDu = norm(prob.G' * wp - prob.A' * vp - prob.c) / (1 + normc)
    ss = max(maximum(abs, sp), 1.0); vs = max(maximum(abs, vp), 1.0)
    mins = isempty(Bidx) ? 0.0 : minimum(sp[Bidx]) / ss
    minv = isempty(Nidx) ? 0.0 : minimum(up) / vs
    ok = rPr < 1e-12 && rEq < 1e-12 && rDu < 1e-12 && mins > -1e-12 && minv > -1e-12
    return (ok = ok, rPr = rPr, rEq = rEq, rDu = rDu, mins = mins, minv = minv,
            pobj = -dot(prob.c, yp), t = t1 + t2, nN = length(Nidx), y = yp)
end

function analyze_lp(name, prob; T = 60, verbose_rows = true, do_rank = true)
    println("\n" * "="^78); println("### ", name)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    @printf("n=%d  cone rows m=%d  equalities p=%d  nnz(A)=%d nnz(G)=%d\n",
            n, m, p, nnz(prob.A), nnz(prob.G))

    # ── 1. reference solution ──
    ref = nothing; usedRef = 0.0
    for tol in (1e-10, 1e-9, 1e-8)
        r = solve1(prob; optTol = tol, maxIters = 200)
        if r.status == :Optimal; ref = r; usedRef = tol; break; end
    end
    sr, vr = ref.s, ref.v
    scale = max(maximum(sr), maximum(vr))
    Bref = sr .> vr
    ndeg = count(max.(sr, vr) .< 1e-8 * scale)
    @printf("reference: tol=%.0e iters=%d pobj=%.16e  |B*|=%d |N*|=%d  #non-strict=%d\n",
            usedRef, ref.Iter, ref.pobj, count(Bref), count(.!Bref), ndeg)

    # iteration counts at the two stopping tolerances (same iterate sequence)
    t6 = solve1(prob; optTol = 1e-6, maxIters = 200)
    t8 = solve1(prob; optTol = 1e-8, maxIters = 200)
    tm6 = @elapsed solve1(prob; optTol = 1e-6, maxIters = 200)
    @printf("stop at 1e-6: %d iters (%.3f s, %.3f ms/iter)   stop at 1e-8: %d iters (%s)\n",
            t6.Iter, tm6, 1e3 * tm6 / t6.Iter, t8.Iter, t8.status)

    # ── capture the iterate sequence (run to the tight tolerance) ──
    sols, _, _ = capture(prob; optTol = usedRef, T = T)
    Tcap = length(sols)

    # ── 2/3. indicators + Ye projection at every iteration ──
    println("  t  |  mu      | ratio ok/fp/fn | tapia ok/fp | proj(ratio) rPr/rDu min  | proj(tapia)")
    tr_exact = 0; tt_exact = 0; pr_first = 0; pt_first = 0
    pr_info = nothing; pt_info = nothing; pr_t = NaN; pt_t = NaN
    prevB = BitVector[]; t_stab = 0
    for t in 1:Tcap
        s = sols[t].s; v = sols[t].v; y = sols[t].y; w = sols[t].w
        Bhat = ratio_B(s, v)
        okr = count(Bhat .== Bref); fpr = count(Bhat .& .!Bref); fnr = count(.!Bhat .& Bref)
        okr == m && tr_exact == 0 && (tr_exact = t)
        if t >= 2
            Bt, _, _ = tapia_B(s, sols[t-1].s, v, sols[t-1].v)
            okt = count(Bt .== Bref); fpt = count(Bt .& .!Bref)
        else
            Bt = Bhat; okt = 0; fpt = m
        end
        okt == m && tt_exact == 0 && (tt_exact = t)
        push!(prevB, Bhat)
        if t_stab == 0 && t >= 3 && prevB[t] == prevB[t-1] && prevB[t] == prevB[t-2]
            t_stab = t
        end
        P1 = ye_project(prob, .!Bhat, y, w, v)
        P2 = ye_project(prob, .!Bt,  y, w, v)
        if P1.ok && pr_first == 0; pr_first = t; pr_info = P1; pr_t = P1.t; end
        if P2.ok && pt_first == 0; pt_first = t; pt_info = P2; pt_t = P2.t; end
        verbose_rows && @printf("  %2d | %8.2e | %5d/%4d/%4d | %5d/%4d | %.1e %.1e %+.0e %s | %.1e %.1e %+.0e %s\n",
                t, sols[t].Mu, okr, fpr, fnr, okt, fpt,
                P1.rPr, P1.rDu, min(P1.mins, P1.minv), P1.ok ? "EXACT" : "     ",
                P2.rPr, P2.rDu, min(P2.mins, P2.minv), P2.ok ? "EXACT" : "")
    end
    @printf("  exact identification: ratio=%s tapia=%s ; exact projection: ratio=%s tapia=%s\n",
            tr_exact == 0 ? "never" : string(tr_exact), tt_exact == 0 ? "never" : string(tt_exact),
            pr_first == 0 ? "never" : string(pr_first), pt_first == 0 ? "never" : string(pt_first))

    # ── 4. accuracy: default 1e-6 termination vs the projected point ──
    normb = norm(prob.b); normd = norm(prob.d); normc = norm(prob.c)
    yd = t6.y
    rPrd = norm(prob.A * yd - t6.s - prob.b) / (1 + normb)
    rEqd = p == 0 ? 0.0 : norm(prob.G * yd - prob.d) / (1 + normd)
    rDud = norm(prob.G' * t6.w - prob.A' * t6.v - prob.c) / (1 + normc)
    gapd = abs(dot(t6.v, t6.s)) / (1 + abs(t6.pobj))
    best = pt_info === nothing ? pr_info : (pr_info === nothing ? pt_info :
           (pr_first <= pt_first ? pr_info : pt_info))
    bestt = min(pr_first == 0 ? 10^6 : pr_first, pt_first == 0 ? 10^6 : pt_first)
    # the projected point is exactly complementary, so its objective is the
    # exact optimum (to the backward error of the residuals): use it as truth
    truth = best === nothing ? ref.pobj : best.pobj
    oerr(x) = abs(x - truth) / (1 + abs(truth))
    @printf("  ACCURACY default(1e-6, %d it): rPr=%.2e rEq=%.2e rDu=%.2e gap=%.2e objerr=%.2e\n",
            t6.Iter, rPrd, rEqd, rDud, gapd, oerr(t6.pobj))
    @printf("  partition (ratio) stable for 3 consecutive iterations from t=%s\n",
            t_stab == 0 ? "never" : string(t_stab))
    if best !== nothing
        @printf("  ACCURACY projected  (t=%d):      rPr=%.2e rEq=%.2e rDu=%.2e gap=0        objerr=%.2e\n",
                bestt, best.rPr, best.rEq, best.rDu, oerr(best.pobj))
        @printf("  SAVING vs 1e-6 stop: %d of %d iterations (%+.0f%%);  t_proj=%.1f ms = %.1f IPM iterations\n",
                t6.Iter - bestt, t6.Iter, 100 * (t6.Iter - bestt) / t6.Iter,
                1e3 * best.t, best.t / (tm6 / t6.Iter))
    else
        println("  projection never certified an exact optimum")
    end

    # ── 5. crossover / basis ──
    Nidx = findall(.!Bref)
    M = [prob.A[Nidx, :]; prob.G]
    q = size(M, 1)
    rk = -1
    if do_rank
        try; rk = rank(qr(sparse(M'))); catch; rk = -1; end
    end
    @printf("  CROSSOVER: |N*|+p = %d rows vs n = %d (excess %d); rank=%s ⇒ %s\n",
            q, n, q - n, rk < 0 ? "n/a" : string(rk),
            rk == n && q == n ? "unique nondegenerate vertex" :
            rk == n ? "vertex, primal-degenerate ($(q-n) redundant active rows)" :
            "not a vertex: optimal face of dim $(n - max(rk,0)); crossover must pivot $(n - max(rk,0)) columns")
    return (name = name, n = n, m = m, it6 = t6.Iter, it8 = t8.Iter,
            tr = tr_exact, tt = tt_exact, pr = pr_first, pt = pt_first,
            stab = t_stab, ndeg = ndeg, q = q, rank = rk, msiter = 1e3 * tm6 / t6.Iter,
            tproj = best === nothing ? NaN : 1e3 * best.t,
            objd = oerr(t6.pobj),
            maxrp = best === nothing ? NaN : max(best.rPr, best.rEq, best.rDu),
            maxrd = max(rPrd, rEqd, rDud),
            rdud = rDud, rdup = best === nothing ? NaN : best.rDu, gapd = gapd)
end

results = []
nb = joinpath(@__DIR__, "netlib")
push!(results, analyze_lp("lp-band-2000",  lp_band(2_000)))
push!(results, analyze_lp("lp-band-20000", lp_band(20_000)))
push!(results, analyze_lp("afiro",    load_netlib(nb, "lp_afiro")))
push!(results, analyze_lp("adlittle", load_netlib(nb, "lp_adlittle")))
push!(results, analyze_lp("share2b",  load_netlib(nb, "lp_share2b")))
push!(results, analyze_lp("scagr7",   load_netlib(nb, "lp_scagr7")))
push!(results, analyze_lp("degen2",   load_netlib(nb, "lp_degen2")))
push!(results, analyze_lp("degen-synth", degen_lp(300, 60)))

println("\n\n=== SUMMARY ===")
@printf("%-14s %6s %6s %5s %5s %6s %6s %6s %6s %5s %5s %9s %9s\n",
        "instance","n","m","it1e6","it1e8","tRatio","tTapia","pRatio","pTapia","stab","ndeg","ms/iter","ms/proj")
for r in results
    f(x) = x == 0 ? "never" : string(x)
    @printf("%-14s %6d %6d %5d %5d %6s %6s %6s %6s %5s %5d %9.3f %9.3f\n",
            r.name, r.n, r.m, r.it6, r.it8, f(r.tr), f(r.tt), f(r.pr), f(r.pt),
            f(r.stab), r.ndeg, r.msiter, r.tproj)
end
println()
@printf("%-14s %10s %10s %10s %10s %10s\n", "instance","maxres1e6","maxresProj","gap@1e-6","objerr6","objerrP")
for r in results
    @printf("%-14s %10.2e %10.2e %10.2e %10.2e %10s\n", r.name, r.maxrd, r.maxrp, r.gapd, r.objd, "0 (exact)")
end
println()
@printf("%-14s %8s %8s %8s\n", "instance","|N*|+p","n","rank")
for r in results; @printf("%-14s %8d %8d %8d\n", r.name, r.q, r.n, r.rank); end
