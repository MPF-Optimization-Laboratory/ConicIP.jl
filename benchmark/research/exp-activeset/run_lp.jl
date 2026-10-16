# exp-activeset, experiments 1-5 for LP instances.
include(joinpath(@__DIR__, "lib.jl"))

const OUT = @__DIR__

function analyze_lp(name, prob; optTol = 1e-6, refTol = 1e-10, T = 60,
                    do_rank = true, verbose_rows = true)
    println("\n" * "="^78); println("### ", name)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    @printf("n=%d  cone rows m=%d  equalities p=%d  nnz(A)=%d nnz(G)=%d\n",
            n, m, p, nnz(prob.A), nnz(prob.G))

    # ── 1. reference solution ──
    ref = nothing; usedRef = refTol
    for tol in (refTol, 1e-9, 1e-8)
        r = solve1(prob; optTol = tol, maxIters = 200)
        if r.status == :Optimal; ref = r; usedRef = tol; break; end
    end
    ref === nothing && (ref = solve1(prob; optTol = 1e-6, maxIters = 200); usedRef = 1e-6)
    sr, vr = ref.s, ref.v
    scale = max(maximum(sr), maximum(vr))
    Bref = sr .> vr
    ndeg = count(max.(sr, vr) .< 1e-8 * scale)
    @printf("reference: tol=%.0e status=%s iters=%d pobj=%.16e  |B*|=%d |N*|=%d  #non-strict(<1e-8 rel)=%d\n",
            usedRef, ref.status, ref.Iter, ref.pobj, count(Bref), count(.!Bref), ndeg)

    # ── default-tolerance run, iterate by iterate ──
    sols, twall, _ = capture(prob; optTol = optTol, T = T)
    Tdef = length(sols)
    tdef = @elapsed soldef = solve1(prob; optTol = optTol, maxIters = T)
    @printf("default run (optTol=%.0e): %s in %d iters, %.4f s (%.2f ms/iter)\n",
            optTol, soldef.status, soldef.Iter, tdef, 1e3 * tdef / max(soldef.Iter, 1))

    # ── 2. indicators ──
    rows = []
    t_ratio_exact = 0; t_tapia_exact = 0
    for t in 1:Tdef
        s = sols[t].s; v = sols[t].v
        Bhat = ratio_B(s, v)
        okr = count(Bhat .== Bref); fpr = count(Bhat .& .!Bref); fnr = count(.!Bhat .& Bref)
        if t >= 2
            s0 = sols[t-1].s; v0 = sols[t-1].v
            Bt, Ts, Tv = tapia_B(s, s0, v, v0)
            okt = count(Bt .== Bref); fpt = count(Bt .& .!Bref)
        else
            okt = 0; fpt = m
        end
        (okr == m) && t_ratio_exact == 0 && (t_ratio_exact = t)
        (okt == m) && t_tapia_exact == 0 && (t_tapia_exact = t)
        push!(rows, (t = t, mu = sols[t].Mu, ratio_ok = okr, ratio_fp = fpr,
                     ratio_fn = fnr, tapia_ok = okt, tapia_fp = fpt))
    end
    if verbose_rows
        println("  t   mu        ratio: ok/m   fp  fn   |  tapia: ok/m   fp")
        for r in rows
            @printf("  %2d  %8.2e  %6d/%d %5d %4d   |  %6d/%d %5d\n", r.t, r.mu,
                    r.ratio_ok, m, r.ratio_fp, r.ratio_fn, r.tapia_ok, m, r.tapia_fp)
        end
    end
    @printf("  identification exact from iteration: ratio=%s  tapia=%s  (default run ends at %d)\n",
            t_ratio_exact == 0 ? "never" : string(t_ratio_exact),
            t_tapia_exact == 0 ? "never" : string(t_tapia_exact), Tdef)

    # ── 3. Ye-style projection at each iteration ──
    normc = norm(prob.c); normb = norm(prob.b); normd = norm(prob.d)
    tproj_first = 0; tproj_time = NaN; projinfo = nothing
    println("  Ye projection per iteration (predicted N from s_i > v_i):")
    for t in 1:Tdef
        s = sols[t].s; v = sols[t].v; y = sols[t].y; w = sols[t].w
        Bhat = ratio_B(s, v); Nhat = .!Bhat
        Nidx = findall(Nhat); Bidx = findall(Bhat)
        AN = prob.A[Nidx, :]
        M = [AN; prob.G]; r = [prob.b[Nidx]; prob.d]
        yp, t1 = minnorm_proj_timed(M, r, y)
        # dual: G'w - AN'u = c, u ≥ 0 (v_B = 0)
        Md = hcat(sparse(prob.G'), -sparse(AN'))
        x0 = [w; v[Nidx]]
        xp, t2 = minnorm_proj_timed(Md, Vector(prob.c), x0)
        wp = xp[1:p]; up = xp[p+1:end]
        sp = zeros(m); sp[Bidx] = prob.A[Bidx, :] * yp - prob.b[Bidx]
        vp = zeros(m); vp[Nidx] = up
        rPr = norm(prob.A * yp - sp - prob.b) / (1 + normb)
        rEq = p == 0 ? 0.0 : norm(prob.G * yp - prob.d) / (1 + normd)
        rDu = norm(prob.G' * wp - prob.A' * vp - prob.c) / (1 + normc)
        mins = isempty(Bidx) ? 0.0 : minimum(sp[Bidx])
        minv = isempty(Nidx) ? 0.0 : minimum(up)
        sscale = max(maximum(abs, sp), 1.0); vscale = max(maximum(abs, vp), 1.0)
        ok = rPr < 1e-12 && rEq < 1e-12 && rDu < 1e-12 &&
             mins > -1e-12 * sscale && minv > -1e-12 * vscale
        pobjp = -dot(prob.c, yp)
        @printf("    t=%2d |N|=%5d rPr=%.2e rEq=%.2e rDu=%.2e min sB=%+.2e min vN=%+.2e  t_proj=%.1f ms  %s\n",
                t, length(Nidx), rPr, rEq, rDu, mins / sscale, minv / vscale,
                1e3 * (t1 + t2), ok ? "EXACT" : "")
        if ok && tproj_first == 0
            tproj_first = t; tproj_time = t1 + t2
            projinfo = (y = yp, s = sp, v = vp, w = wp, Nidx = Nidx, Bidx = Bidx,
                        rPr = rPr, rEq = rEq, rDu = rDu, pobj = pobjp)
        end
    end

    # control: projection with the TRUE partition, from the last default iterate
    let s = sols[Tdef].s, v = sols[Tdef].v, y = sols[Tdef].y, w = sols[Tdef].w
        Nidx = findall(.!Bref); Bidx = findall(Bref)
        AN = prob.A[Nidx, :]
        yp, t1 = minnorm_proj_timed([AN; prob.G], [prob.b[Nidx]; prob.d], y)
        xp, t2 = minnorm_proj_timed(hcat(sparse(prob.G'), -sparse(AN')),
                                    Vector(prob.c), [w; v[Nidx]])
        wp = xp[1:p]; up = xp[p+1:end]
        sp = zeros(m); sp[Bidx] = prob.A[Bidx, :] * yp - prob.b[Bidx]
        vp = zeros(m); vp[Nidx] = up
        @printf("  CONTROL (true partition, last iterate): rPr=%.2e rEq=%.2e rDu=%.2e min sB=%+.2e min vN=%+.2e objerr=%.2e\n",
                norm(prob.A * yp - sp - prob.b) / (1 + normb),
                p == 0 ? 0.0 : norm(prob.G * yp - prob.d) / (1 + normd),
                norm(prob.G' * wp - prob.A' * vp - prob.c) / (1 + normc),
                isempty(Bidx) ? 0.0 : minimum(sp[Bidx]),
                isempty(Nidx) ? 0.0 : minimum(up),
                abs(-dot(prob.c, yp) + ref.pobj) / (1 + abs(ref.pobj)))
    end

    # ── 4. accuracy gain ──
    if projinfo !== nothing
        yd = soldef.y
        rPrd = norm(prob.A * yd - soldef.s - prob.b) / (1 + normb)
        rEqd = p == 0 ? 0.0 : norm(prob.G * yd - prob.d) / (1 + normd)
        rDud = norm(prob.G' * soldef.w - prob.A' * soldef.v - prob.c) / (1 + normc)
        gapd = abs(dot(soldef.v, soldef.s)) / (1 + abs(soldef.pobj))
        objref = -ref.pobj
        @printf("  ACCURACY  default(t=%d): rPr=%.2e rEq=%.2e rDu=%.2e gap=%.2e objerr=%.2e\n",
                Tdef, rPrd, rEqd, rDud, gapd, abs(-dot(prob.c, yd) - objref) / (1 + abs(objref)))
        @printf("            projected(t=%d): rPr=%.2e rEq=%.2e rDu=%.2e gap=0 (exact)  objerr=%.2e\n",
                tproj_first, projinfo.rPr, projinfo.rEq, projinfo.rDu,
                abs(projinfo.pobj - objref) / (1 + abs(objref)))
        @printf("  SAVING: projection succeeds at t=%d of %d iterations (%.0f%% of iterations saved); t_proj=%.1f ms vs %.1f ms/IPM-iteration\n",
                tproj_first, Tdef, 100 * (Tdef - tproj_first) / Tdef,
                1e3 * tproj_time, 1e3 * tdef / max(soldef.Iter, 1))
    else
        println("  Ye projection never certified an exact optimum within the default run.")
    end

    # ── 5. crossover / basis check ──
    Nidx = findall(.!Bref)
    M = [prob.A[Nidx, :]; prob.G]
    q = size(M, 1)
    rk = do_rank ? try rank(qr(sparse(M'))) catch; -1 end : -1
    @printf("  CROSSOVER: |N*|+p = %d rows vs n = %d  (excess %d);  rank = %s  ⇒ %s\n",
            q, n, q - n, rk < 0 ? "n/a" : string(rk),
            rk == n && q == n ? "unique nondegenerate vertex basis" :
            rk == n ? "vertex, primal-degenerate (crossover must choose $(q-n) redundant rows)" :
            "NOT a vertex: face of dimension $(n - max(rk,0)); crossover must pivot $(n - max(rk,0)) more")
    return (name = name, n = n, m = m, p = p, Tdef = Tdef, tdef = tdef,
            iters = soldef.Iter, t_ratio = t_ratio_exact, t_tapia = t_tapia_exact,
            t_proj = tproj_first, tproj_ms = 1e3 * tproj_time, ndeg = ndeg,
            q = q, rank = rk)
end

results = []
nb = joinpath(@__DIR__, "netlib")
push!(results, analyze_lp("lp-band-2000", lp_band(2_000)))
push!(results, analyze_lp("afiro", load_netlib(nb, "lp_afiro")))
push!(results, analyze_lp("adlittle", load_netlib(nb, "lp_adlittle")))
push!(results, analyze_lp("share2b", load_netlib(nb, "lp_share2b")))
push!(results, analyze_lp("scagr7", load_netlib(nb, "lp_scagr7")))
push!(results, analyze_lp("degen-synth", degen_lp(300, 60); verbose_rows = true))

println("\n\n=== SUMMARY ===")
@printf("%-14s %6s %6s %6s %8s %8s %8s %9s %9s %6s\n",
        "instance", "n", "m", "iters", "t_ratio", "t_tapia", "t_proj", "t_def(ms)", "t_proj(ms)", "ndeg")
for r in results
    @printf("%-14s %6d %6d %6d %8s %8s %8s %9.1f %9.1f %6d\n", r.name, r.n, r.m, r.iters,
            r.t_ratio == 0 ? "never" : string(r.t_ratio),
            r.t_tapia == 0 ? "never" : string(r.t_tapia),
            r.t_proj == 0 ? "never" : string(r.t_proj),
            1e3 * r.tdef, r.tproj_ms, r.ndeg)
end
