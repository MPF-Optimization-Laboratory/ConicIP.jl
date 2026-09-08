# Benchmark harness (large-scale roadmap, tranche 0)
# ==================================================
# A small reproducible instance set with the measurements the roadmap says
# every later tranche must be judged by: partial timing estimates, iteration and
# KKT-solve counts, a fill proxy for the KKT factorization, peak RSS in a
# fresh process, and residuals recomputed from the original data.
#
#   julia --project benchmark/suite.jl                 # full set
#   julia --project benchmark/suite.jl --quick         # small subset, fast
#   julia --project benchmark/suite.jl --only nql30,lp-band-10000
#   julia --project benchmark/suite.jl --in-process    # no subprocess (no RSS)
#   julia --project benchmark/suite.jl --out results.csv
#   julia --project benchmark/suite.jl --opt maxIters=50,kktsolver=ldl
#                                      # solver options (Int/Float/Bool/String)
#   julia --project benchmark/suite.jl --timeout 300   # kill a child after 300 s
#
# Instances come from three sources:
#   * synthetic families with bounded factorization fill (banded LP, banded
#     box QP, the sum-of-norms SOCP from test/testdata.jl), at several sizes;
#   * the issue-#10 SOCP (downloaded, as benchmark/issue10.jl does);
#   * Maros–Mészáros QPs (QPS via the MOI MPS reader) and CBLIB SOCPs (CBF),
#     downloaded on demand into benchmark/.cache/ (gitignored).
#
# Direct-API instances are solved through preprocess_conicIP with the
# default KKT solver. File instances go through MOI with bridges, so their
# time includes model assembly; the inner Solution is read back through
# MOI.RawSolver.
#
# Each instance runs in a fresh Julia process by default so that peak RSS
# is attributable and compilation of one instance cannot warm another. The
# child warms up on a tiny problem first; the reported time is the better
# of two solves. Peak RSS still includes compilation memory, so compare RSS
# across sizes within a family, not against an absolute budget.
#
# Not part of CI. Results depend on hardware; the header records the
# environment so that two runs can be compared honestly.

using ConicIP, SparseArrays, LinearAlgebra, Printf, Random, Downloads, Dates
using MathOptInterface
const MOI = MathOptInterface

# testdata.jl is guarded so that a module that already loaded it (the test
# suite, or issue10.jl below) does not redefine its methods.
isdefined(@__MODULE__, :socp_sum_of_norms) ||
    include(joinpath(@__DIR__, "..", "test", "testdata.jl"))
include(joinpath(@__DIR__, "issue10.jl"))          # load_issue10()

const CACHE = joinpath(@__DIR__, ".cache")
const MM_URL = "https://bitbucket.org/optrove/maros-meszaros/raw/master/"
const CBLIB_URL = "https://cblib.zib.de/download/all/"

# ──────────────────────────────────────────────────────────────
#  Synthetic families (bounded fill by construction)
# ──────────────────────────────────────────────────────────────

# Banded LP: min cᵀy  s.t.  B y ≥ b₁ (band of half-width w),  −10 ≤ y ≤ 10.
# y = 0 is feasible for the band rows, the box bounds the problem.
function lp_band(n; w = 5, seed = 1)
    Random.seed!(seed)
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for i in 1:n, j in max(1, i - w):min(n, i + w)
        push!(I_, i); push!(J_, j); push!(V_, randn())
    end
    B = sparse(I_, J_, V_, n, n)
    A = [B; sparse(1.0I, n, n); -sparse(1.0I, n, n)]
    b = [-rand(n); fill(-10.0, n); fill(-10.0, n)]
    return (Q = spzeros(n, n), c = randn(n), A = A, b = b,
            cone_dims = [("R", 3n)], G = spzeros(0, n), d = zeros(0))
end

# Banded box QP: min ½yᵀQy − cᵀy  s.t.  −1 ≤ y ≤ 1,  Q = LLᵀ + I banded.
function qp_band(n; w = 3, seed = 1)
    Random.seed!(seed)
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for i in 1:n, j in max(1, i - w):i
        push!(I_, i); push!(J_, j); push!(V_, randn())
    end
    L = sparse(I_, J_, V_, n, n)
    Q = L*L' + sparse(1.0I, n, n)
    A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
    b = fill(-1.0, 2n)
    return (Q = Q, c = randn(n), A = A, b = b,
            cone_dims = [("R", 2n)], G = spzeros(0, n), d = zeros(0))
end

# ──────────────────────────────────────────────────────────────
#  Instance registry
# ──────────────────────────────────────────────────────────────

struct Instance
    name   :: String
    kind   :: Symbol        # :direct (ConicIP tuple) or :moi (file)
    family :: String
    quick  :: Bool
    load   :: Function      # () -> problem tuple, or () -> path
    format :: Union{Nothing, MOI.FileFormats.FileFormat}
end

direct(name, family, thunk; quick = false) =
    Instance(name, :direct, family, quick, thunk, nothing)

function fetched(name, family, url, fname, format; quick = false)
    path = joinpath(CACHE, fname)
    thunk = () -> begin
        mkpath(CACHE)
        isfile(path) || Downloads.download(url, path; timeout = 300)
        path
    end
    return Instance(name, :moi, family, quick, thunk, format)
end

mm(name; quick = false) =
    fetched(lowercase(name), "maros-meszaros", MM_URL * name * ".SIF",
            name * ".SIF", MOI.FileFormats.FORMAT_MPS; quick = quick)
cblib(name; quick = false) =
    fetched(name, "cblib", CBLIB_URL * name * ".cbf.gz",
            name * ".cbf.gz", MOI.FileFormats.FORMAT_CBF; quick = quick)

const INSTANCES = Instance[
    direct("lp-band-2000",   "lp-band",  () -> lp_band(2_000);   quick = true),
    direct("lp-band-20000",  "lp-band",  () -> lp_band(20_000)),
    direct("lp-band-200000", "lp-band",  () -> lp_band(200_000)),
    direct("qp-band-2000",   "qp-band",  () -> qp_band(2_000);   quick = true),
    direct("qp-band-20000",  "qp-band",  () -> qp_band(20_000)),
    direct("qp-band-200000", "qp-band",  () -> qp_band(200_000)),
    direct("socp-sumnorms-150",  "socp-sumnorms",
           () -> socp_sum_of_norms(150; d = 200); quick = true),
    direct("socp-sumnorms-1500", "socp-sumnorms",
           () -> socp_sum_of_norms(1500; d = 2000)),
    direct("issue10", "issue10", load_issue10),
    mm("HS21"; quick = true), mm("HS35"), mm("HS76"), mm("QAFIRO"; quick = true),
    mm("CVXQP1_S"), mm("DUAL1"), mm("PRIMAL1"), mm("QPCBLEND"),
    cblib("sambal"; quick = true), cblib("chainsing-1000-1"),
    cblib("nb"), cblib("nql30"), cblib("qssp30"), cblib("sched_50_50_scaled"),
]

# ──────────────────────────────────────────────────────────────
#  Measurements
# ──────────────────────────────────────────────────────────────

# Residuals of a direct-API solution against the original data, with the
# solver's own normalization (relative to the right-hand side or to the
# componentwise products |Q||y|, |Gᵀ||w|, |Aᵀ||v|, |A||y|, |G||y|, whichever
# is larger), plus the cone-membership margin of the slack (≥ 0 is inside).
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
    # Lagrangian ½yᵀQy − cᵀy + wᵀ(Gy − d) − vᵀ(Ay − b): the dual objective at
    # stationarity is −½yᵀQy + bᵀv − dᵀw.
    pobj = 0.5*dot(y, Qy) - dot(c, y)
    dobj = -0.5*dot(y, Qy) + dot(b, v) - dot(d, w)
    gap  = abs(pobj - dobj) / (1 + abs(pobj))
    margin = ConicIP.cone_margin(s, prob.cone_dims)
    dual_margin = ConicIP.cone_margin(v, prob.cone_dims)
    return (rDu = rDu, rPr = rPr, rEq = rEq, gap = gap, margin = margin,
            dual_margin = dual_margin)
end

# Structural nonzeros of the 3×3 KKT matrix at identity scaling, and the
# fill of its UMFPACK LU (nnz(L)+nnz(U)). This is an identity-scaling
# diagnostic, NOT the selected solver's factor fill: SOC off-diagonals
# and lifted auxiliaries are absent. Skipped above `fill_max` rows.
function kkt_fill(prob; fill_max = 300_000)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    Q = sparse(prob.Q); A = sparse(prob.A); G = sparse(prob.G)
    Z = [Q  G'  -A'
         G  spzeros(p, p)  spzeros(p, m)
         A  spzeros(m, p)  sparse(1.0I, m, m)]
    kkt_nnz = nnz(Z)
    n + m + p > fill_max && return (kkt_nnz = kkt_nnz, lu_nnz = -1)
    lu_nnz = try nnz(lu(Z)) catch; -1 end
    return (kkt_nnz = kkt_nnz, lu_nnz = lu_nnz)
end

# Accuracy is checked independently of the solver's termination fields.
verified(sol, res; tol = 1e-6) = sol.status == :Optimal &&
    max(res.rDu, res.rPr, res.rEq, res.gap) <= tol &&
    res.margin >= -tol && res.dual_margin >= -tol

# ──────────────────────────────────────────────────────────────
#  Solver options (`--opt key=value[,key=value]`)
# ──────────────────────────────────────────────────────────────

# Parsed once per process from the command line into SOLVER_OPTS; the
# parent forwards the raw string to each child (see run_subprocess).
const SOLVER_OPTS = Dict{Symbol, Any}()
const OPT_STRING = Ref("")

_parse_opt_value(s::AbstractString) = something(
    tryparse(Int, s), tryparse(Float64, s),
    s == "true" ? true : s == "false" ? false : nothing, String(s))

# "maxIters=3,optTol=1e-8,equilibrate=false,kktsolver=ldl" →
# Dict(:maxIters => 3, :optTol => 1e-8, :equilibrate => false, :kktsolver => "ldl")
function parse_opts(str::AbstractString)
    opts = Dict{Symbol, Any}()
    for kv in split(str, ','; keepempty = false)
        occursin('=', kv) || throw(ArgumentError("--opt entry \"$kv\" is not key=value"))
        k, v = strip.(split(kv, '='; limit = 2))
        (isempty(k) || isempty(v)) &&
            throw(ArgumentError("--opt entry \"$kv\" has an empty key or value"))
        opts[Symbol(k)] = _parse_opt_value(v)
    end
    return opts
end

function set_opts!(str::AbstractString)
    OPT_STRING[] = String(str)
    empty!(SOLVER_OPTS)
    merge!(SOLVER_OPTS, parse_opts(str))
    return SOLVER_OPTS
end

# Keyword arguments for the direct API: strings name a KKT solver, and
# `preprocess` chooses the entry point rather than being forwarded.
function direct_kwargs(opts = SOLVER_OPTS)
    kw = Dict{Symbol, Any}(k => v for (k, v) in opts if k != :preprocess)
    if haskey(kw, :kktsolver)
        kw[:kktsolver] = ConicIP._resolve_kktsolver(kw[:kktsolver])
    end
    return kw
end

# Name of the KKT solver `choose_kktsolver` picks for this data, or the
# `--opt kktsolver=` override; "" if the choice itself throws.
function kkt_solver_name(prob; opts = SOLVER_OPTS)
    haskey(opts, :kktsolver) && return string(opts[:kktsolver])
    ks = try
        ConicIP.choose_kktsolver(prob.Q, prob.A, prob.G, prob.cone_dims)
    catch
        return ""
    end
    return ks isa Function ? string(nameof(ks)) : string(nameof(typeof(ks)))
end

# Solution fields another tranche adds (LDLᵀ repair/refactor counters); blank
# when the running ConicIP does not have them.
_sol_field(sol, name) = (sol !== nothing && hasproperty(sol, name)) ? getproperty(sol, name) : ""

function solve_direct(prob; preprocess = true)
    entry = (preprocess && get(SOLVER_OPTS, :preprocess, true)) ? preprocess_conicIP : conicIP
    return entry(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims, prob.G, prob.d;
                 verbose = false, direct_kwargs()...)
end

function run_direct(inst::Instance)
    prob = inst.load()
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    data_nnz = nnz(sparse(prob.Q)) + nnz(sparse(prob.A)) + nnz(sparse(prob.G))
    fill = kkt_fill(prob)
    kktname = kkt_solver_name(prob)
    # Two solves with preprocessing (report the better), one without to
    # estimate the presolve cost by subtraction. This is NOT an instrumented
    # phase time: the two solves can take different routes/iterations.
    t1 = @timed solve_direct(prob)
    t2 = @timed solve_direct(prob)
    (stats, sol) = t1.time <= t2.time ? (t1, t1.value) : (t2, t2.value)
    t_raw = @elapsed solve_direct(prob; preprocess = false)
    res = residuals(prob, sol)
    return Dict(
        "n" => n, "m" => m, "p" => p, "data_nnz" => data_nnz,
        "kkt_nnz" => fill.kkt_nnz, "lu_nnz" => fill.lu_nnz,
        "status" => string(sol.status), "iters" => sol.Iter,
        "kkt_solves" => sol.kkt_solves,
        "t_total" => stats.time, "t_presolve_est" => max(stats.time - t_raw, 0.0),
        "t_assembly" => 0.0,
        "alloc_gib" => stats.bytes / 2^30,
        "rDu" => res.rDu, "rPr" => res.rPr, "rEq" => res.rEq,
        "gap" => res.gap, "margin" => res.margin, "dual_margin" => res.dual_margin,
        "verified" => verified(sol, res),
        "kktsolver" => kktname,
        "kkt_repaired" => _sol_field(sol, :kkt_repaired),
        "kkt_refactors" => _sol_field(sol, :kkt_refactors),
    )
end

function run_moi(inst::Instance)
    path = inst.load()
    src = MOI.FileFormats.Model(format = inst.format, filename = path)
    MOI.read_from_file(src, path)
    function once()
        opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
        MOI.set(opt, MOI.Silent(), true)
        for (k, v) in SOLVER_OPTS
            MOI.set(opt, MOI.RawOptimizerAttribute(string(k)), v)
        end
        t_asm = @elapsed MOI.copy_to(opt, src)
        t_opt = @elapsed MOI.optimize!(opt)
        return (opt = opt, t_asm = t_asm, t_opt = t_opt)
    end
    r1 = once(); r2 = once()
    r = (r1.t_asm + r1.t_opt) <= (r2.t_asm + r2.t_opt) ? r1 : r2
    raw = MOI.get(r.opt, MOI.RawSolver())::ConicIP.Optimizer
    sol = raw.sol
    n = raw.n
    m = raw.ineq_A === nothing ? 0 : size(raw.ineq_A, 1)
    p = raw.eq_G === nothing ? 0 : size(raw.eq_G, 1)
    data_nnz = (raw.Q_int === nothing ? 0 : nnz(raw.Q_int)) +
               (raw.ineq_A === nothing ? 0 : nnz(raw.ineq_A)) +
               (raw.eq_G === nothing ? 0 : nnz(raw.eq_G))
    # Reconstruct from the original assembled data, not sol.prFeas/duFeas
    # or the solver's dual objective. Presolve and equilibration have both
    # been undone on sol; use the internal minimization sense consistently.
    prob = (Q = raw.Q_int === nothing ? spzeros(n,n) : raw.Q_int,
            c = raw.c_int, A = raw.ineq_A, b = raw.ineq_b,
            G = raw.eq_G, d = raw.eq_d, cone_dims = raw.cone_dims)
    res = residuals(prob, sol)
    # The assembled data is at hand, so the automatic choice is cheap here.
    kktname = (raw.ineq_A === nothing || raw.eq_G === nothing) ? "" : kkt_solver_name(prob)
    return Dict(
        "n" => n, "m" => m, "p" => p, "data_nnz" => data_nnz,
        "kkt_nnz" => -1, "lu_nnz" => -1,
        "status" => string(MOI.get(r.opt, MOI.TerminationStatus())),
        "iters" => sol === nothing ? 0 : sol.Iter,
        "kkt_solves" => sol === nothing ? 0 : sol.kkt_solves,
        "t_total" => r.t_asm + r.t_opt, "t_presolve_est" => NaN,
        "t_assembly" => r.t_asm + raw.assembly_time,
        "alloc_gib" => NaN,
        "rDu" => res.rDu, "rPr" => res.rPr, "rEq" => res.rEq,
        "gap" => res.gap, "margin" => res.margin, "dual_margin" => res.dual_margin,
        "verified" => verified(sol, res),
        "kktsolver" => kktname,
        "kkt_repaired" => _sol_field(sol, :kkt_repaired),
        "kkt_refactors" => _sol_field(sol, :kkt_refactors),
    )
end

# Compile the solver paths on tiny problems so that timings and the RSS
# high-water mark of the instance itself are not dominated by the first call.
function warmup()
    small = socp_sum_of_norms(10; d = 10)
    preprocess_conicIP(small.Q, small.c, small.A, small.b, small.cone_dims,
                       small.G, small.d; verbose = false)
    lp = lp_band(50)
    preprocess_conicIP(lp.Q, lp.c, lp.A, lp.b, lp.cone_dims, lp.G, lp.d; verbose = false)
    return nothing
end

function run_one(inst::Instance)
    warmup()
    result = inst.kind == :direct ? run_direct(inst) : run_moi(inst)
    result["maxrss_mb"] = Sys.maxrss() / 2^20
    return result
end

# ──────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────

const COLUMNS = ["name", "family", "n", "m", "p", "data_nnz", "kkt_nnz", "lu_nnz",
                 "status", "iters", "kkt_solves", "t_total", "t_presolve_est",
                 "t_assembly", "alloc_gib", "maxrss_mb",
                 "rDu", "rPr", "rEq", "gap", "margin", "dual_margin", "verified",
                 "kktsolver", "kkt_repaired", "kkt_refactors"]

# Run one instance in a fresh Julia process: `script --one name extra_args...`
# (the script defaults to this file; baselines.jl passes itself). The child
# is killed after `timeout` seconds (wall clock, warm-up and compilation
# included) and the row then reads status=TIMEOUT.
function run_subprocess(inst::Instance; timeout = Inf, script = @__FILE__,
                        extra_args = String[])
    args = String["--one", inst.name]
    isempty(OPT_STRING[]) || append!(args, ["--opt", OPT_STRING[]])
    append!(args, extra_args)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $script $args`
    result = Dict{String, Any}()
    mktempdir() do dir
        outpath = joinpath(dir, "out"); errpath = joinpath(dir, "err")
        proc = run(pipeline(cmd; stdout = outpath, stderr = errpath); wait = false)
        killed = Ref(false)
        timer = isfinite(timeout) ? Timer(timeout) do _
            process_running(proc) || return
            killed[] = true
            kill(proc)
        end : nothing
        wait(proc)
        timer === nothing || close(timer)
        for line in eachline(outpath)
            occursin('=', line) || continue
            k, v = split(line, '='; limit = 2)
            result[String(k)] = String(v)
        end
        if killed[]
            result["status"] = "TIMEOUT"
        elseif !success(proc)
            # Child stderr (precompilation chatter included) is shown only on failure.
            println(stderr, read(errpath, String))
            result["status"] = get(result, "status", "CRASHED")
        end
    end
    return result
end

fmt(x::AbstractString) = x
fmt(x::Integer) = string(x)
fmt(x::Real) = isnan(x) ? "" : (abs(x) >= 1e-3 && abs(x) < 1e6) ? @sprintf("%.3g", x) : @sprintf("%.1e", x)

function environment_header()
    rev = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "?" end
    return join([
        "date=$(Dates.format(now(), "yyyy-mm-dd HH:MM"))",
        "julia=$(VERSION)", "conicip=$(pkgversion(ConicIP))", "git=$rev",
        "cpu=$(Sys.cpu_info()[1].model)", "threads=$(Threads.nthreads())",
        "blas_threads=$(BLAS.get_num_threads())", "os=$(Sys.KERNEL) $(Sys.MACHINE)",
    ], "  ")
end

# Value of `--flag value` in args, or `default` when the flag is absent.
function arg_value(args, flag, default = nothing)
    i = findfirst(==(flag), args)
    return i === nothing ? default : args[i + 1]
end

_num(x) = x isa AbstractString ? something(tryparse(Float64, x), NaN) : x

# Run `selected` instances one after another, printing a progress line per
# instance and returning the result rows. `runner(inst)` is the in-process
# path; otherwise each instance goes to a child process running `script`.
function run_selected(selected; inproc = false, timeout = Inf, runner = run_one,
                      script = @__FILE__, extra_args = String[])
    rows = Vector{Dict{String, Any}}()
    for inst in selected
        print(rpad(inst.name, 24)); flush(stdout)
        result = try
            inproc ? Dict{String, Any}(k => v for (k, v) in runner(inst)) :
                     run_subprocess(inst; timeout = timeout, script = script,
                                    extra_args = extra_args)
        catch err
            Dict{String, Any}("status" => "FAILED: " * sprint(showerror, err)[1:min(end, 60)])
        end
        result["name"] = inst.name; result["family"] = inst.family
        push!(rows, result)
        @printf("%-14s iters=%-4s solves=%-4s t=%-9s rss=%s MB\n",
                get(result, "status", "?"), get(result, "iters", "?"),
                get(result, "kkt_solves", "?"),
                fmt(_num(get(result, "t_total", NaN))),
                fmt(_num(get(result, "maxrss_mb", NaN))))
    end
    return rows
end

# CSV with the environment header as a comment line; `""` for missing cells.
function write_csv(path, rows; columns = COLUMNS, header = environment_header())
    open(path, "w") do io
        println(io, "# ", header)
        println(io, join(columns, ","))
        for r in rows
            println(io, join([string(get(r, c, "")) for c in columns], ","))
        end
    end
    return path
end

function select_instances(args)
    quick = "--quick" in args
    only = "--only" in args ? Set(split(arg_value(args, "--only"), ',')) : nothing
    return filter(INSTANCES) do inst
        only !== nothing ? inst.name in only : (!quick || inst.quick)
    end
end

function main(args)
    "--opt" in args && set_opts!(arg_value(args, "--opt"))
    if "--one" in args
        name = arg_value(args, "--one")
        inst = INSTANCES[findfirst(i -> i.name == name, INSTANCES)]
        for (k, v) in run_one(inst)
            println(k, "=", v)
        end
        return
    end
    inproc = "--in-process" in args
    timeout = parse(Float64, arg_value(args, "--timeout", "Inf"))
    outpath = if "--out" in args
        arg_value(args, "--out")
    else
        mkpath(joinpath(@__DIR__, "results"))
        joinpath(@__DIR__, "results", "suite-$(Dates.format(now(), "yyyymmdd-HHMM")).csv")
    end
    selected = select_instances(args)

    println("# ConicIP benchmark suite")
    println(environment_header())
    isempty(OPT_STRING[]) || println("options: ", OPT_STRING[])
    println()
    rows = run_selected(selected; inproc = inproc, timeout = timeout)
    write_csv(outpath, rows)
    println("\nwrote ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
