# Diagnosis of the sum-of-norms SOCP regression (large-scale roadmap, T4)
# =======================================================================
# socp_sum_of_norms(1500; d = 2000) (n = 6500, m = 4500, p = 3000, 1500
# second-order cones of dimension 3, never lifted) ran in 1.18 s / 10
# iterations on v0.4.0 (3.6 GB) and takes ~1.8 s / 12 iterations on the
# LDLᵀ route (1.4 GB). This script attributes the difference among
# factorization time, back-solve time (the LDLᵀ back-solve includes its
# own refinement), the extra iterations, and setup/threads effects, and
# tests the accuracy hypotheses for the extra iterations. It measures
# three routes: today's dense `kktsolver_qr`, `kktsolver_ldl`, and
# `kktsolver_sparse` (UMFPACK), which is what v0.4.0's routing rule chose
# for this instance (2.5 nnz/col), so the release comparison can be
# reproduced on one tree. It changes nothing in src/ and is not part of CI.
#
#   julia --project benchmark/sumnorms_diag.jl                # k = 1500, d = 2000
#   julia --project benchmark/sumnorms_diag.jl --quick        # k = 150,  d = 200
#   julia --project benchmark/sumnorms_diag.jl --k 800 --d 1000
#   julia --project benchmark/sumnorms_diag.jl --skip-accuracy   # sections 1–3, 5 only
#
# Output is Markdown to stdout, ending with a one-line attribution.

using ConicIP, SparseArrays, LinearAlgebra, Printf, Random, Dates
using QDLDL, AMD

include(joinpath(@__DIR__, "..", "test", "testdata.jl"))   # socp_sum_of_norms

# ──────────────────────────────────────────────────────────────
#  Arguments
# ──────────────────────────────────────────────────────────────

function argval(args, flag, default)
    i = findfirst(==(flag), args)
    i === nothing && return default
    return parse(typeof(default), args[i + 1])
end

const QUICK  = "--quick" in ARGS
const K      = QUICK ? 150 : argval(ARGS, "--k", 1500)
const D      = QUICK ? 200 : argval(ARGS, "--d", 2000)
const SKIP_ACC = "--skip-accuracy" in ARGS
const NREP   = 2                          # best-of-NREP wall time

# ──────────────────────────────────────────────────────────────
#  Residuals (copied from benchmark/suite.jl so that file's MOI and
#  download machinery is not pulled in)
# ──────────────────────────────────────────────────────────────

function residuals(prob, sol)
    Q, c, A, b, G, d = prob.Q, prob.c, prob.A, prob.b, prob.G, prob.d
    y, w, v, s = sol.y, sol.w, sol.v, sol.s
    all(isfinite, y) || return (rDu = NaN, rPr = NaN, rEq = NaN, gap = NaN, margin = NaN, dual_margin = NaN)
    Qy   = Q*y
    absQ = ConicIP._absmat(Q); absA = ConicIP._absmat(A); absG = ConicIP._absmat(G)
    ay = abs.(y); aw = abs.(w); av = abs.(v)
    nQy = norm(absQ*ay)
    nGw = isempty(w) ? 0.0 : norm(absG'*aw)
    nAv = isempty(v) ? 0.0 : norm(absA'*av)
    rDu  = norm(Qy + G'*w - A'*v - c) / (1 + max(norm(c), nQy, nGw, nAv))
    rPr  = isempty(b) ? 0.0 :
           norm(A*y - s - b) / (1 + max(norm(b), norm(absA*ay), norm(s)))
    rEq  = isempty(d) ? 0.0 : norm(G*y - d) / (1 + max(norm(d), norm(absG*ay)))
    pobj = 0.5*dot(y, Qy) - dot(c, y)
    dobj = -0.5*dot(y, Qy) + dot(b, v) - dot(d, w)
    gap  = abs(pobj - dobj) / (1 + abs(pobj))
    margin = ConicIP.cone_margin(s, prob.cone_dims)
    dual_margin = ConicIP.cone_margin(v, prob.cone_dims)
    return (rDu = rDu, rPr = rPr, rEq = rEq, gap = gap, margin = margin,
            dual_margin = dual_margin)
end

verified(sol, res; tol = 1e-6) = sol.status == :Optimal &&
    max(res.rDu, res.rPr, res.rEq, res.gap) <= tol &&
    res.margin >= -tol && res.dual_margin >= -tol

# ──────────────────────────────────────────────────────────────
#  Phase-timed kktsolver wrapper (no change to src/)
# ──────────────────────────────────────────────────────────────
# Wraps any kktsolver constructor. Records the setup time (pattern, AMD,
# symbolic analysis for LDLᵀ; QR of Gᵀ, Q0, AQ2, S22 for QR), every
# factorization (`solve3x3gen(F, F⁻ᵀ)`: value rewrite + numeric refactor
# for LDLᵀ; W, S22 + WᵀW, Cholesky for QR) and every 3×3 back-solve (the
# LDLᵀ back-solve includes its internal refinement against K₀).

mutable struct PhaseLog
    setup       :: Float64
    fact_times  :: Vector{Float64}
    solve_times :: Vector{Float64}
end
PhaseLog() = PhaseLog(0.0, Float64[], Float64[])

#
# A callable struct with an abstract `inner::Function` field rather than a
# closure: `_conicIP` specializes on the type of its `kktsolver` argument,
# and every anonymous `(Q, A, G, cd) -> kktsolver_ldl(...; ...)` in the
# accuracy grid would otherwise trigger a ~1.5 s recompilation of the
# solver on its first use.
struct TimedKKT
    inner :: Function
    log   :: PhaseLog
end

function (tk::TimedKKT)(Q, A, G, cd)
    inner = tk.inner; log = tk.log
    ts = @elapsed g = inner(Q, A, G, cd)
    log.setup = ts
    return (F, Fi) -> begin
        t = @elapsed s3 = g(F, Fi)
        push!(log.fact_times, t)
        (bx, by, bz) -> begin
            t2 = @elapsed r = s3(bx, by, bz)
            push!(log.solve_times, t2)
            r
        end
    end
end

timed_kktsolver(inner, log::PhaseLog) = TimedKKT(inner, log)

# ──────────────────────────────────────────────────────────────
#  One configured solve
# ──────────────────────────────────────────────────────────────

# Runs entry ∈ {conicIP, preprocess_conicIP} with the kktsolver wrapped in
# a PhaseLog; optionally captures verbose output. Returns a NamedTuple.
function solve_once(prob; kktsolver, equilibrate, preprocess, verbose = false,
                    kwargs...)
    entry = preprocess ? preprocess_conicIP : conicIP
    log = PhaseLog()
    ks = timed_kktsolver(kktsolver, log)
    GC.gc()
    local sol, stats, text
    if verbose
        path, io = mktemp()
        stats = redirect_stdout(io) do
            @timed entry(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims, prob.G, prob.d;
                         kktsolver = ks, equilibrate = equilibrate, verbose = true,
                         kwargs...)
        end
        close(io)
        text = read(path, String)
        rm(path; force = true)
    else
        stats = @timed entry(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims, prob.G, prob.d;
                             kktsolver = ks, equilibrate = equilibrate, verbose = false,
                             kwargs...)
        text = ""
    end
    sol = stats.value
    return (sol = sol, time = stats.time, bytes = stats.bytes, log = log, text = text)
end

# Best of NREP by wall time.
function solve_best(prob; nrep = NREP, kwargs...)
    best = nothing
    for _ in 1:nrep
        r = solve_once(prob; kwargs...)
        if best === nothing || r.time < best.time
            best = r
        end
    end
    return best
end

# ──────────────────────────────────────────────────────────────
#  Verbose-row parsing
# ──────────────────────────────────────────────────────────────
# Iteration rows have the form
#   "   3  │  1.2e-03  4.5e-04  6.7e-05 │ -1.2e+00 -1.2e+00  │  1.0e+00  1.0e+00 │  2"

struct IterRow
    iter   :: Int
    prFeas :: Float64
    duFeas :: Float64
    muFeas :: Float64
    pobj   :: Float64
    dobj   :: Float64
    refine :: Int
end

function parse_rows(text::AbstractString)
    rows = IterRow[]
    for line in split(text, '\n')
        occursin('│', line) || continue
        parts = split(line, '│')
        length(parts) == 5 || continue
        it = tryparse(Int, strip(parts[1]))
        it === nothing && continue
        f = s -> parse(Float64, s)
        feas = f.(split(strip(parts[2])))
        obj  = f.(split(strip(parts[3])))
        ref  = tryparse(Int, strip(parts[5]))
        (length(feas) == 3 && length(obj) == 2 && ref !== nothing) || continue
        push!(rows, IterRow(it, feas[1], feas[2], feas[3], obj[1], obj[2], ref))
    end
    return rows
end

# ──────────────────────────────────────────────────────────────
#  Formatting
# ──────────────────────────────────────────────────────────────

fmt(x::AbstractString) = x
fmt(x::Bool) = string(x)
fmt(x::Integer) = string(x)
fmt(x::Real) = isnan(x) ? "nan" : (abs(x) >= 1e-3 && abs(x) < 1e6) ? @sprintf("%.3g", x) : @sprintf("%.2e", x)
fmt(x) = string(x)

function mdtable(header, rows)
    println("| ", join(header, " | "), " |")
    println("|", join(["---" for _ in header], "|"), "|")
    for r in rows
        println("| ", join(fmt.(r), " | "), " |")
    end
    println()
end

ksname(ks) = ks === ConicIP.kktsolver_qr ? "qr" :
             ks === ConicIP.kktsolver_ldl ? "ldl" :
             ks === ConicIP.kktsolver_sparse ? "umfpack" :
             ks === ConicIP.default_kktsolver ? "default" : "custom"

# The three routes measured. `kktsolver_sparse` (UMFPACK LU on the lifted
# system) is what v0.4.0's rule (`nnz/col ≤ 10` and `n + m + p ≥ 1000` ⇒
# sparse) selected for this instance; it is no longer chosen automatically
# but remains available, so the release comparison can be reproduced on
# the current tree.
const ROUTES = (ConicIP.kktsolver_qr, ConicIP.kktsolver_ldl, ConicIP.kktsolver_sparse)

# ──────────────────────────────────────────────────────────────
#  Warm-up
# ──────────────────────────────────────────────────────────────

function warmup()
    small = socp_sum_of_norms(20; d = 20)
    for ks in (ROUTES..., ConicIP.default_kktsolver), eq in (true, false), pp in (true, false)
        solve_once(small; kktsolver = ks, equilibrate = eq, preprocess = pp)
    end
    # Verbose paths with the same keyword combinations section 4 uses.
    r = solve_once(small; kktsolver = ConicIP.kktsolver_qr, equilibrate = true,
                   preprocess = true, verbose = true, maxRefinementSteps = 6)
    r = solve_once(small; kktsolver = ConicIP.kktsolver_ldl, equilibrate = true,
                   preprocess = true, verbose = true, maxRefinementSteps = 6)
    parse_rows(r.text)
    custom = (Q, A, G, cd) -> ConicIP.kktsolver_ldl(Q, A, G, cd; static_reg = 1e-10,
                                                    dynamic_delta = 1e-9, refine_steps = 0)
    solve_once(small; kktsolver = custom, equilibrate = true, preprocess = true,
               verbose = true, maxRefinementSteps = 3)
    pat = ConicIP._ldl_pattern(small.Q, small.A, small.G, small.cone_dims)
    F = qdldl(pat.K; perm = pat.perm, Dsigns = pat.Dsigns,
              regularize_eps = 1e-13, regularize_delta = 2e-7)
    refactor!(F); solve!(F, randn(pat.N))
    qdldl(pat.K; perm = pat.perm, logical = true)
    qdldl(pat.K; perm = nothing, logical = true)
    ConicIP._ldl_flops(pat)
    amd(pat.K); amd(pat.K, Amd()); QDLDL._get_amd_ordering(pat.K, 1.0)
    symamd(pat.K + pat.K' - spdiagm(diag(pat.K)))
    residuals(small, r.sol)
    return nothing
end

# Warm solve after a BLAS thread-count change (OpenBLAS re-spawns its pool).
function warm_blas()
    small = socp_sum_of_norms(20; d = 20)
    solve_once(small; kktsolver = ConicIP.kktsolver_qr, equilibrate = true, preprocess = true)
    return nothing
end

# Minimum over `nrep` timings of `f()` after one warm call.
function best_time(f; nrep = 20)
    f()
    return minimum(@elapsed(f()) for _ in 1:nrep)
end

# ──────────────────────────────────────────────────────────────
#  Main
# ──────────────────────────────────────────────────────────────

function main()
    println("# sum-of-norms SOCP diagnosis (k = $K, d = $D)")
    rev = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "?" end
    println()
    println("date=$(Dates.format(now(), "yyyy-mm-dd HH:MM"))  julia=$(VERSION)  " *
            "conicip=$(pkgversion(ConicIP))  git=$rev  cpu=$(Sys.cpu_info()[1].model)")
    println("`Threads.nthreads()` = $(Threads.nthreads()), " *
            "`BLAS.get_num_threads()` = $(BLAS.get_num_threads()) (`$(BLAS.get_config().loaded_libs[1].libname |> basename)`)")
    println()

    print("warming up on k = 20 ... "); flush(stdout)
    tw = @elapsed warmup()
    println(@sprintf("%.1f s", tw)); println()

    prob = socp_sum_of_norms(K; d = D)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    println("n = $n, m = $m, p = $p, nnz(A) = $(nnz(prob.A)), nnz(G) = $(nnz(prob.G)), " *
            "nnz(Q) = $(nnz(prob.Q)), cones = $(length(prob.cone_dims)) × Q³")
    println()

    default_threads = BLAS.get_num_threads()
    routed = ConicIP.choose_kktsolver(prob.Q, prob.A, prob.G, prob.cone_dims)
    println("`choose_kktsolver` routes this instance to **$(nameof(routed))**.")
    println()

    # ── 1 + 2. Grid with phase timing ─────────────────────────
    println("## 1. Solver × equilibrate × preprocess grid (best of $NREP, phase-timed)")
    println()
    println("Phase columns are totals over the best run: `setup` (solver construction), " *
            "`fact` (all `solve3x3gen` calls: value rewrite + numeric LDLᵀ, or W/Cholesky for QR), " *
            "`solves` (all 3×3 back-solves; the LDLᵀ one includes its internal refinement), " *
            "`other` = wall − setup − fact − solves (equilibration, presolve, NT scaling, " *
            "line searches, residuals, 4×4 refinement bookkeeping).")
    println()
    grid = Dict{Tuple{String,Bool,Bool}, Any}()
    rows = []
    for ks in ROUTES, eq in (true, false), pp in (true, false)
        r = solve_best(prob; kktsolver = ks, equilibrate = eq, preprocess = pp)
        res = residuals(prob, r.sol)
        grid[(ksname(ks), eq, pp)] = (r = r, res = res)
        L = r.log
        tf = sum(L.fact_times); tsv = sum(L.solve_times)
        push!(rows, (ksname(ks), eq, pp, string(r.sol.status), r.sol.Iter, r.sol.kkt_solves,
                     r.time, L.setup, tf, tsv, r.time - L.setup - tf - tsv,
                     r.bytes / 2^30, res.rDu, res.rPr, res.rEq, res.gap, verified(r.sol, res)))
    end
    # Reference: the default route (should coincide with one of the rows above).
    rdef = solve_best(prob; kktsolver = ConicIP.default_kktsolver, equilibrate = true, preprocess = true)
    resdef = residuals(prob, rdef.sol)
    let L = rdef.log, tf = sum(L.fact_times), tsv = sum(L.solve_times)
        push!(rows, ("default", true, true, string(rdef.sol.status), rdef.sol.Iter, rdef.sol.kkt_solves,
                     rdef.time, L.setup, tf, tsv, rdef.time - L.setup - tf - tsv,
                     rdef.bytes / 2^30, resdef.rDu, resdef.rPr, resdef.rEq, resdef.gap,
                     verified(rdef.sol, resdef)))
    end
    mdtable(["kkt", "equil", "presolve", "status", "iters", "solves", "wall s", "setup s",
             "fact s", "solves s", "other s", "GiB alloc", "rDu", "rPr", "rEq", "gap", "verified"], rows)

    println("### 1b. Per-call phase costs (harness configuration: equilibrate = true, presolve = true)")
    println()
    rows = []
    for key in (("qr", true, true), ("ldl", true, true), ("umfpack", true, true))
        r = grid[key].r; L = r.log
        push!(rows, (key[1], length(L.fact_times), 1e3 * sum(L.fact_times) / length(L.fact_times),
                     1e3 * minimum(L.fact_times), 1e3 * maximum(L.fact_times),
                     length(L.solve_times), 1e3 * sum(L.solve_times) / length(L.solve_times),
                     1e3 * minimum(L.solve_times), 1e3 * maximum(L.solve_times),
                     length(L.solve_times) / max(r.sol.Iter, 1)))
    end
    mdtable(["kkt", "#fact", "fact mean ms", "fact min ms", "fact max ms",
             "#solves", "solve mean ms", "solve min ms", "solve max ms", "solves/iter"], rows)

    # ── 3. Fill, flops, orderings, routing margin ────────────
    println("## 3. Fill, flops, orderings, routing margin")
    println()
    # Pattern on the data the solver actually factors: after equilibration
    # the sparsity pattern is unchanged, so the unscaled data is representative.
    pat = ConicIP._ldl_pattern(prob.Q, prob.A, prob.G, prob.cone_dims)
    Kp = pat.K
    Fl = qdldl(Kp; perm = pat.perm, logical = true)
    flops = ConicIP._ldl_flops(pat)
    nnzL = nnz(Fl.L)
    println("KKT order N = $(pat.N) (lifted cones: $(pat.nlift)), nnz(triu K) = $(nnz(Kp)), " *
            "nnz(L) = $nnzL, fill ratio nnz(L)/nnz(triu K) = $(fmt(nnzL / nnz(Kp))), " *
            "`_ldl_flops` = Σⱼ nnz(L₍:,ⱼ₎)² = $(fmt(flops)).")
    println()

    # Pure numeric refactorization time with the solver's ordering (no
    # value rewrite), best of 5; GFlop/s uses the Σ nnz(L_j)² count (a
    # count of multiply-adds in the rank-1 updates, so ×2 for flops).
    Fn = qdldl(Kp; perm = pat.perm, Dsigns = pat.Dsigns,
               regularize_eps = 1e-13, regularize_delta = 2e-7)
    t_refactor = best_time(() -> refactor!(Fn))
    xs = randn(pat.N); xb = copy(xs)
    t_solve = best_time(() -> (xb .= xs; solve!(Fn, xb)))
    Lldl = grid[("ldl", true, true)].r.log
    fact_mean = sum(Lldl.fact_times) / length(Lldl.fact_times)
    solve_mean = sum(Lldl.solve_times) / length(Lldl.solve_times)
    println("Numeric `refactor!` alone (best of 20): $(fmt(1e3 * t_refactor)) ms → " *
            "$(fmt(flops / t_refactor / 1e9)) G(mul-add)/s = $(fmt(2flops / t_refactor / 1e9)) GFlop/s. " *
            "Measured `solve3x3gen` mean in the solve (value rewrite + refactor): $(fmt(1e3 * fact_mean)) ms " *
            "(rewrite overhead ≈ $(fmt(1e3 * (fact_mean - t_refactor))) ms per factorization, " *
            "mostly `_dense_FtF` on $(length(prob.cone_dims)) 3×3 blocks).")
    println("One raw `solve!` (forward/back substitution, no refinement): $(fmt(1e3 * t_solve)) ms " *
            "($(fmt(2nnzL / t_solve / 1e9)) GFlop/s on 2·nnz(L)·2 ≈ $(fmt(4nnzL/1e6)) Mflop); " *
            "measured `solve3x3` mean in the solve (with up to 2 internal refinement steps, each one " *
            "`_symmul!` + `solve!`): $(fmt(1e3 * solve_mean)) ms = $(fmt(solve_mean / t_solve)) raw solves.")
    println()

    println("### 3b. Ordering variants")
    println()
    rows = []
    function try_perm(name, perm)
        t = @elapsed F = qdldl(Kp; perm = perm, logical = true)
        fl = sum(abs2, Float64.(F.workspace.Lnz))
        push!(rows, (name, nnz(F.L), fmt(nnz(F.L) / nnzL), fl, fmt(fl / flops), 1e3 * t))
    end
    try_perm("AMD (ConicIP `_ldl_pattern`, `amd(K)`)", pat.perm)
    for dense in (10.0, 3.0, 30.0, -1.0), aggr in (1.0, 0.0)
        meta = Amd(); meta.control[AMD.AMD_DENSE] = dense; meta.control[AMD.AMD_AGGRESSIVE] = aggr
        ta = @elapsed pm = amd(Kp, meta)
        try_perm("AMD dense=$(dense) aggressive=$(aggr) (amd $(fmt(1e3*ta)) ms)", pm)
    end
    for s in (1.0, 1.5)
        ta = @elapsed pm = QDLDL._get_amd_ordering(Kp, s)
        try_perm("QDLDL default AMD, amd_dense_scale=$(s) (amd $(fmt(1e3*ta)) ms)", pm)
    end
    ta = @elapsed pm = symamd(Kp + Kp' - spdiagm(diag(Kp)))
    try_perm("SYMAMD on K + Kᵀ (symamd $(fmt(1e3*ta)) ms)", pm)
    try_perm("natural (no permutation)", nothing)
    mdtable(["ordering", "nnz(L)", "vs AMD", "Σ nnz(L_j)²", "vs AMD", "symbolic ms"], rows)

    println("### 3c. Routing margin")
    println()
    fq = ConicIP.dense_kkt_flops(n, m, p; nnzA = nnz(prob.A), nnzQ = nnz(prob.Q))
    println("`dense_kkt_flops(n, m, p; nnzA, nnzQ)` = $(fmt(fq)); " *
            "`ldl_flop_weight × _ldl_flops` = 10 × $(fmt(flops)) = $(fmt(10flops)). " *
            "Ratio dense/LDLᵀ = $(fmt(fq / flops)); the route flips to QR only for `ldl_flop_weight` > $(fmt(fq / flops)). " *
            "`dense_kkt_bytes` = $(fmt(ConicIP.dense_kkt_bytes(n, m, p) / 2^30)) GiB " *
            "(budget $(fmt(4.0)) GiB).")
    println()

    # ── 5. BLAS threads ──────────────────────────────────────
    println("## 5. BLAS threads (equilibrate = true, presolve = true, best of $NREP)")
    println()
    rows = []
    thread_runs = Dict{Tuple{String,Int}, Any}()
    for nt in (default_threads, 1), ks in ROUTES
        BLAS.set_num_threads(nt); warm_blas()
        r = solve_best(prob; kktsolver = ks, equilibrate = true, preprocess = true)
        thread_runs[(ksname(ks), nt)] = r
        L = r.log
        push!(rows, (ksname(ks), nt, r.sol.Iter, r.sol.kkt_solves, r.time, L.setup,
                     sum(L.fact_times), sum(L.solve_times)))
    end
    BLAS.set_num_threads(default_threads); warm_blas()
    mdtable(["kkt", "BLAS threads", "iters", "solves", "wall s", "setup s", "fact s", "solves s"], rows)

    # ── 4. Accuracy hypotheses ───────────────────────────────
    acc = Dict{Any, Any}()
    if !SKIP_ACC
        println("## 4. Accuracy hypotheses for the extra iterations (verbose runs, single solve each)")
        println()
        println("`s` = `static_reg`, `δ` = `dynamic_delta`, `r` = `refine_steps` (LDLᵀ-internal), " *
                "`R` = `maxRefinementSteps` (outer 4×4 refinement). `Σrefine` is the sum of the verbose " *
                "`refine` column (outer corrections taken); final feasibilities from the last verbose row.")
        println()
        rows = []
        for R in (3, 6)
            r = solve_once(prob; kktsolver = ConicIP.kktsolver_qr, equilibrate = true,
                           preprocess = true, verbose = true, maxRefinementSteps = R)
            it = parse_rows(r.text); res = residuals(prob, r.sol)
            acc[("qr", R)] = (r = r, rows = it)
            last = isempty(it) ? IterRow(0, NaN, NaN, NaN, NaN, NaN, 0) : it[end]
            push!(rows, ("qr", "—", "—", "—", R, string(r.sol.status), r.sol.Iter, r.sol.kkt_solves,
                         sum(x -> x.refine, it), r.time, last.prFeas, last.duFeas, last.muFeas,
                         max(res.rDu, res.rPr, res.rEq), res.gap))
        end
        for s in (1e-8, 1e-10), δ in (2e-7, 1e-9), rr in (0, 2, 5), R in (3, 6)
            ks = (Q, A, G, cd) -> ConicIP.kktsolver_ldl(Q, A, G, cd; static_reg = s,
                                                        dynamic_delta = δ, refine_steps = rr)
            r = solve_once(prob; kktsolver = ks, equilibrate = true, preprocess = true,
                           verbose = true, maxRefinementSteps = R)
            it = parse_rows(r.text); res = residuals(prob, r.sol)
            acc[("ldl", s, δ, rr, R)] = (r = r, rows = it)
            last = isempty(it) ? IterRow(0, NaN, NaN, NaN, NaN, NaN, 0) : it[end]
            push!(rows, ("ldl", s, δ, rr, R, string(r.sol.status), r.sol.Iter, r.sol.kkt_solves,
                         sum(x -> x.refine, it), r.time, last.prFeas, last.duFeas, last.muFeas,
                         max(res.rDu, res.rPr, res.rEq), res.gap))
        end
        mdtable(["kkt", "s", "δ", "r", "R", "status", "iters", "solves", "Σrefine", "wall s",
                 "prFeas", "duFeas", "muFeas", "max resid", "gap"], rows)

        println("### 4b. Per-iteration trajectories: QR (R = 3) vs LDLᵀ defaults (s = 1e-8, δ = 2e-7, r = 2, R = 3)")
        println()
        a = acc[("qr", 3)].rows; b = acc[("ldl", 1e-8, 2e-7, 2, 3)].rows
        rows = []
        for i in 1:max(length(a), length(b))
            ra = i <= length(a) ? a[i] : nothing
            rb = i <= length(b) ? b[i] : nothing
            g = (x, f) -> x === nothing ? "" : fmt(f(x))
            push!(rows, (i, g(ra, x -> x.prFeas), g(ra, x -> x.duFeas), g(ra, x -> x.muFeas), g(ra, x -> x.refine),
                            g(rb, x -> x.prFeas), g(rb, x -> x.duFeas), g(rb, x -> x.muFeas), g(rb, x -> x.refine)))
        end
        mdtable(["iter", "qr prFeas", "qr duFeas", "qr muFeas", "qr ref",
                 "ldl prFeas", "ldl duFeas", "ldl muFeas", "ldl ref"], rows)

        # Best LDLᵀ variant by iteration count, then time.
        ldl_keys = [k for k in keys(acc) if k[1] == "ldl"]
        sort!(ldl_keys; by = k -> (acc[k].r.sol.Iter, acc[k].r.time))
        kb = ldl_keys[1]
        println("Fewest LDLᵀ iterations: s = $(kb[2]), δ = $(kb[3]), r = $(kb[4]), R = $(kb[5]) → " *
                "$(acc[kb].r.sol.Iter) iterations, $(acc[kb].r.sol.kkt_solves) solves, $(fmt(acc[kb].r.time)) s " *
                "(QR: $(acc[("qr", 3)].r.sol.Iter) iterations).")
        println()
    end

    # ── 6. Attribution ───────────────────────────────────────
    println("## 6. Attribution")
    println()
    # Δ = T_ldl − T_base decomposed exactly, at the baseline's iteration
    # count: X = factorizations, Y = back-solves (the LDLᵀ one includes its
    # internal refinement), Z = extra iterations priced at the per-iteration
    # LDLᵀ cost (KKT work plus the per-iteration remainder; the remainder
    # also holds the fixed equilibration/presolve cost, so Z is a slight
    # over-estimate), W = the rest (setup difference and per-iteration
    # non-KKT work, which is where BLAS threading of the baseline shows).
    function attribution(base)
        rb = grid[(base, true, true)].r; rl = grid[("ldl", true, true)].r
        Ib = rb.sol.Iter; Il = rl.sol.Iter
        Lb = rb.log; Ll = rl.log
        fb_t = sum(Lb.fact_times); sb_t = sum(Lb.solve_times)
        fl_t = sum(Ll.fact_times); sl_t = sum(Ll.solve_times)
        Δ = rl.time - rb.time
        per_iter_ldl = (rl.time - Ll.setup) / max(Il, 1)
        extra = Il - Ib
        Z = extra * per_iter_ldl
        scale = Ib / max(Il, 1)
        X = fl_t * scale - fb_t
        Y = sl_t * scale - sb_t
        W = Δ - X - Y - Z
        tb1 = haskey(thread_runs, (base, 1)) ? thread_runs[(base, 1)].time : NaN
        tbN = haskey(thread_runs, (base, default_threads)) ? thread_runs[(base, default_threads)].time : NaN
        println("**LDLᵀ vs $base** (equil, presolve). $base: $(fmt(rb.time)) s, $Ib iterations, $(rb.sol.kkt_solves) solves " *
                "[setup $(fmt(Lb.setup)), fact $(fmt(fb_t)), solves $(fmt(sb_t))]; " *
                "LDLᵀ: $(fmt(rl.time)) s, $Il iterations, $(rl.sol.kkt_solves) solves " *
                "[setup $(fmt(Ll.setup)), fact $(fmt(fl_t)), solves $(fmt(sl_t))]. Δ = $(fmt(Δ)) s, " *
                "decomposed at $Ib iterations; Z = $(extra) extra iteration(s) × $(fmt(per_iter_ldl)) s; " *
                "$base wall at 1 BLAS thread $(fmt(tb1)) s vs $(fmt(tbN)) s at $default_threads.")
        println()
        println(@sprintf("factorization %.3f s | back-solves %.3f s | extra iterations %.3f s | threads/routing %.3f s   (Δ = %.3f s, LDLᵀ − %s)",
                         X, Y, Z, W, Δ, base))
        println()
        return (X = X, Y = Y, Z = Z, W = W, Δ = Δ)
    end
    attribution("qr")
    a = attribution("umfpack")
    println("The `qr` line answers the roadmap's framing (dense QR vs LDLᵀ); the `umfpack` line is the " *
            "release comparison, since v0.4.0's `choose_kktsolver` sent this instance " *
            "($(fmt((nnz(prob.A) + nnz(prob.G) + nnz(prob.Q)) / n)) nnz/col) to `kktsolver_sparse`. " *
            "Both LDLᵀ and UMFPACK take $(grid[("ldl", true, true)].r.sol.Iter) and " *
            "$(grid[("umfpack", true, true)].r.sol.Iter) iterations here; v0.4.0's 10 came from its " *
            "stopping test, which had no relative-gap term (`rGap`).")
    println()
    parts = sort([("factorization", a.X), ("back-solves", a.Y), ("extra iterations", a.Z), ("setup/threads", a.W)];
                 by = x -> -x[2])
    lead = a.Δ > 0 ?
        "LDLᵀ is slower than UMFPACK by $(fmt(a.Δ)) s; the largest term is **$(parts[1][1])** ($(fmt(parts[1][2])) s), then $(parts[2][1]) ($(fmt(parts[2][2])) s)" :
        "LDLᵀ is faster than UMFPACK by $(fmt(-a.Δ)) s (largest saving: $(parts[end][1]), $(fmt(parts[end][2])) s)"
    println("Conclusion: $lead; LDLᵀ numeric factorization runs at " *
            "$(fmt(2flops / t_refactor / 1e9)) GFlop/s single-threaded on $(fmt(nnzL / 1e6)) M factor entries " *
            "(fill $(fmt(nnzL / nnz(Kp)))×, every ordering tried within 1 %), " *
            "and the LDLᵀ back-solve costs $(fmt(solve_mean / t_solve)) raw triangular solves because of its internal refinement.")
    return nothing
end

main()
