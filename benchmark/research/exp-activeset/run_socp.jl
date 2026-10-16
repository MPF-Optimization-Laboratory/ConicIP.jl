# exp-activeset, experiment 2 (cone-state identification) for SOCPs.
include(joinpath(@__DIR__, "lib.jl"))

# Jordan-frame eigenvalues of a SOC vector x = (x0, x̄): λ± = x0 ± ‖x̄‖
eigs(x) = (x[1] - norm(@view x[2:end]), x[1] + norm(@view x[2:end]))

# reference state of a cone block, from (s, v) at high accuracy
function ref_state(s, v; tol = 1e-6)
    ns = norm(s); nv = norm(v); sc = max(ns, nv, 1e-300)
    ns / sc < tol && return :zero
    nv / sc < tol && return :interior
    return :boundary
end

# predicted state at an iterate: pure ratio rule with a neutral band
function pred_ratio(s, v; τ = 1e-2)
    ns = norm(s); nv = norm(v)
    nv / max(ns, 1e-300) < τ && return :interior
    ns / max(nv, 1e-300) < τ && return :zero
    return :boundary
end

# Tapia rule: block-norm ratios between consecutive iterates
function pred_tapia(s1, s0, v1, v0; θ = 0.5)
    Ts = norm(s1) / max(norm(s0), 1e-300)
    Tv = norm(v1) / max(norm(v0), 1e-300)
    Ts < θ && Tv >= θ && return :zero
    Tv < θ && Ts >= θ && return :interior
    Ts < θ && Tv < θ && return (Ts < Tv ? :zero : :interior)
    return :boundary
end

blocks(cone_dims) = begin
    rs = UnitRange{Int}[]; off = 0
    for (k, dim) in cone_dims
        if k == "R"
            for i in 1:dim; push!(rs, (off+i):(off+i)); end
        else
            push!(rs, (off+1):(off+dim))
        end
        off += dim
    end
    rs
end

function analyze_socp(name, prob; T = 60)
    println("\n" * "="^78); println("### ", name)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    rs = blocks(prob.cone_dims)
    socidx = findall(r -> length(r) > 1, rs)
    @printf("n=%d m=%d p=%d  blocks=%d (SOC blocks=%d, R rows=%d)\n",
            n, m, p, length(rs), length(socidx), length(rs) - length(socidx))

    ref = nothing; usedRef = 0.0
    for tol in (1e-10, 1e-9, 1e-8, 1e-7)
        r = solve1(prob; optTol = tol, maxIters = 200)
        if r.status == :Optimal; ref = r; usedRef = tol; break; end
    end
    println("reference tol=$usedRef status=$(ref.status) iters=$(ref.Iter) pobj=$(ref.pobj)")
    states = [ref_state(ref.s[r], ref.v[r]) for r in rs]
    ssoc = states[socidx]
    @printf("reference cone states (all blocks): interior=%d boundary=%d zero=%d\n",
            count(==(:interior), states), count(==(:boundary), states), count(==(:zero), states))
    @printf("                     (SOC blocks): interior=%d boundary=%d zero=%d\n",
            count(==(:interior), ssoc), count(==(:boundary), ssoc), count(==(:zero), ssoc))
    # how sharp is the reference separation?
    rr = [norm(ref.s[r]) / max(norm(ref.v[r]), 1e-300) for r in rs[socidx]]
    bnd = findall(==(:boundary), ssoc)
    if !isempty(bnd)
        lam = [eigs(ref.s[rs[socidx[i]]]) for i in bnd]
        @printf("  boundary blocks: median λ₋(s)/λ₊(s) = %.2e   (0 ⇒ exactly on the boundary)\n",
                median_(sort([abs(l[1]) / max(l[2], 1e-300) for l in lam])))
    end

    t6 = solve1(prob; optTol = 1e-6, maxIters = 200)
    tm6 = @elapsed solve1(prob; optTol = 1e-6, maxIters = 200)
    @printf("stop at 1e-6: %d iters, %.3f s (%.2f ms/iter)\n", t6.Iter, tm6, 1e3*tm6/t6.Iter)

    sols, _, _ = capture(prob; optTol = usedRef, T = T)
    Tcap = length(sols)
    K = length(socidx)
    tr_exact = 0; tt_exact = 0
    println("  t |  mu      | ratio ok/K   (int/bnd/zero wrong) | tapia ok/K")
    for t in 1:Tcap
        s = sols[t].s; v = sols[t].v
        pr = [pred_ratio(s[rs[i]], v[rs[i]]) for i in socidx]
        okr = count(pr .== ssoc)
        if t >= 2
            s0 = sols[t-1].s; v0 = sols[t-1].v
            pt = [pred_tapia(s[rs[i]], s0[rs[i]], v[rs[i]], v0[rs[i]]) for i in socidx]
            okt = count(pt .== ssoc)
        else
            okt = 0
        end
        okr == K && tr_exact == 0 && (tr_exact = t)
        okt == K && tt_exact == 0 && (tt_exact = t)
        wi = count(i -> ssoc[i] == :interior && pr[i] != :interior, 1:K)
        wb = count(i -> ssoc[i] == :boundary && pr[i] != :boundary, 1:K)
        wz = count(i -> ssoc[i] == :zero && pr[i] != :zero, 1:K)
        @printf("  %2d | %8.2e | %5d/%-5d (%d/%d/%d) | %5d/%-5d\n",
                t, sols[t].Mu, okr, K, wi, wb, wz, okt, K)
    end
    @printf("  exact cone-state identification: ratio=%s tapia=%s  (1e-6 stop at %d iters)\n",
            tr_exact == 0 ? "never" : string(tr_exact),
            tt_exact == 0 ? "never" : string(tt_exact), t6.Iter)
    return (name = name, K = K, it6 = t6.Iter, tr = tr_exact, tt = tt_exact,
            nint = count(==(:interior), ssoc), nbnd = count(==(:boundary), ssoc),
            nzero = count(==(:zero), ssoc))
end

median_(x) = isempty(x) ? NaN : x[div(length(x)+1, 2)]

res = []
push!(res, analyze_socp("socp-sumnorms-150", socp_sum_of_norms(150; d = 200)))
if get(ENV, "DO_ISSUE10", "0") == "1"
    include(joinpath("/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap",
                     "benchmark", "issue10.jl"))
    push!(res, analyze_socp("issue10", load_issue10()))
end
println("\n=== SOCP SUMMARY ===")
@printf("%-20s %6s %6s %7s %7s %6s %6s %6s\n", "instance","K","it1e6","tRatio","tTapia","int","bnd","zero")
for r in res
    @printf("%-20s %6d %6d %7s %7s %6d %6d %6d\n", r.name, r.K, r.it6,
            r.tr == 0 ? "never" : string(r.tr), r.tt == 0 ? "never" : string(r.tt),
            r.nint, r.nbnd, r.nzero)
end
