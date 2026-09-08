# Baseline solvers on the benchmark instance set
# ==============================================
# Runs the suite.jl instances through Clarabel, ECOS, or ConicIP-via-MOI,
# all fed the *same* ConicIP problem tuple through benchmark/moi_model.jl,
# and verifies every answer against the original data with the residuals
# from suite.jl. Rows carry the suite.jl columns plus `solver` and `note`.
#
# One-time setup (ConicIP is dev'd from the repository root; the Manifest
# is gitignored):
#
#   julia --project=benchmark -e 'using Pkg; Pkg.develop(path="."); Pkg.instantiate()'
#
# Usage:
#
#   julia --project=benchmark benchmark/baselines.jl --solver clarabel|ecos|conicip-moi
#         [--quick] [--only a,b] [--timeout 120] [--out f.csv] [--in-process]
#         [--opt key=value,...]          # raw options of the selected solver, e.g.
#                                        # conicip-moi: maxIters=50,kktsolver=ldl
#                                        # ecos: feastol=1e-9,abstol=1e-9,reltol=1e-9
#
# Direct instances hand their tuple over as is. File instances (MPS/CBF) are
# converted once through `ConicIP.Optimizer` with `assemble_only = true`
# (falling back to a zero-iteration solve when this ConicIP lacks that
# option), so every solver sees the assembled `(Q, c, A, b, cone_dims, G, d)`.
#
# Each instance runs in a fresh process (as in suite.jl): two solves, the
# faster one is reported; `t_total` is the `optimize!` wall time (model copy
# into the solver included), `t_assembly` the MOI model build. `--timeout T`
# sets `MOI.TimeLimitSec = T` where the solver supports it, skips the second
# solve when the first one exceeded T, and kills the child after 2T + 120 s
# (compilation and warm-up included), recording status=TIMEOUT.
#
# Failures never abort the run: an unsupported constraint or attribute
# (a bridge gap) is recorded as status=FAILED, note=BRIDGE_FAIL; other
# exceptions as status=FAILED with the message in `note`.

include(joinpath(@__DIR__, "suite.jl"))       # ConicIP, MOI, INSTANCES, residuals, …
include(joinpath(@__DIR__, "moi_model.jl"))   # build_moi_model, recover_solution

const BASE_COLUMNS = vcat(COLUMNS, ["solver", "note"])
const SOLVERS = ("clarabel", "ecos", "conicip-moi")

# The baseline package is loaded lazily (only the requested one, and not at
# all for conicip-moi). It must be loaded from top level *before* `main`
# runs: methods defined by a package loaded inside a running function live
# in a newer world, and `MOI.instantiate` would then not see `Optimizer()`.
function load_solver_package(name)
    name == "clarabel" && Base.require(Main, :Clarabel)
    name == "ecos" && Base.require(Main, :ECOS)
    return nothing
end

function solver_constructor(name)
    name == "conicip-moi" && return ConicIP.Optimizer
    name == "clarabel" && return Base.require(Main, :Clarabel).Optimizer
    name == "ecos" && return Base.require(Main, :ECOS).Optimizer
    throw(ArgumentError("unknown solver \"$name\" (expected one of $(join(SOLVERS, ", ")))"))
end

solver_name(args) = lowercase(arg_value(args, "--solver", "conicip-moi"))

# ──────────────────────────────────────────────────────────────
#  Problem tuples for every instance
# ──────────────────────────────────────────────────────────────

_or_empty(M, n) = M === nothing ? spzeros(0, n) : M

# `(prob, note)`: file instances go through ConicIP's MOI front end so the
# tuple is exactly what ConicIP itself would solve.
function load_problem(inst::Instance)
    inst.kind == :direct && return (inst.load(), "")
    path = inst.load()
    src = MOI.FileFormats.Model(format = inst.format, filename = path)
    MOI.read_from_file(src, path)
    opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
    MOI.set(opt, MOI.Silent(), true)
    note = ""
    if "assemble_only" in ConicIP._SUPPORTED_OPTIONS
        MOI.set(opt, MOI.RawOptimizerAttribute("assemble_only"), true)
    else
        note = "ASSEMBLE_FALLBACK"
        println(stderr, "note: this ConicIP has no `assemble_only` option; ",
                "converting $(inst.name) with maxIters = 0 instead")
        for (k, v) in ("maxIters" => 0, "certFallback" => false,
                       "preprocess" => false, "equilibrate" => false)
            MOI.set(opt, MOI.RawOptimizerAttribute(k), v)
        end
    end
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)
    raw = MOI.get(opt, MOI.RawSolver())::ConicIP.Optimizer
    n = raw.n
    prob = (Q = raw.Q_int === nothing ? spzeros(n, n) : raw.Q_int, c = raw.c_int,
            A = _or_empty(raw.ineq_A, n), b = raw.ineq_b, cone_dims = raw.cone_dims,
            G = _or_empty(raw.eq_G, n), d = raw.eq_d)
    return (prob, note)
end

# ──────────────────────────────────────────────────────────────
#  Solving
# ──────────────────────────────────────────────────────────────

function solve_once(ctor, prob; timeout = Inf, opts = Dict{Symbol, Any}())
    opt = bridged_cached(ctor)
    MOI.set(opt, MOI.Silent(), true)
    if isfinite(timeout)
        try MOI.set(opt, MOI.TimeLimitSec(), timeout) catch end
    end
    for (k, v) in opts
        MOI.set(opt, MOI.RawOptimizerAttribute(string(k)), v)
    end
    t_asm = @elapsed maps = build_moi_model(opt, prob)
    st = @timed MOI.optimize!(opt)
    return (opt = opt, maps = maps, t_asm = t_asm, t_opt = st.time, bytes = st.bytes)
end

_short(err) = replace(first(sprint(showerror, err), 80), ',' => ';', '\n' => ' ')

function run_baseline(inst::Instance, solver; timeout = Inf)
    ctor = solver_constructor(solver)
    opts = SOLVER_OPTS            # raw attributes of the selected solver
    prob, note = load_problem(inst)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    data_nnz = nnz(sparse(prob.Q)) + nnz(sparse(prob.A)) + nnz(sparse(prob.G))
    fill = kkt_fill(prob)
    row = Dict{String, Any}(
        "n" => n, "m" => m, "p" => p, "data_nnz" => data_nnz,
        "kkt_nnz" => fill.kkt_nnz, "lu_nnz" => fill.lu_nnz,
        "t_presolve_est" => NaN, "solver" => solver, "note" => note,
        "kktsolver" => solver == "conicip-moi" ? kkt_solver_name(prob) : "")
    r = try
        r1 = solve_once(ctor, prob; timeout = timeout, opts = opts)
        if r1.t_opt < timeout
            r2 = solve_once(ctor, prob; timeout = timeout, opts = opts)
            r1.t_opt <= r2.t_opt ? r1 : r2
        else
            r1
        end
    catch err
        row["status"] = "FAILED"
        row["note"] = err isa MOI.UnsupportedError ? "BRIDGE_FAIL" : "ERROR: " * _short(err)
        return row
    end
    rec = recover_solution(r.opt, r.maps, prob)
    res = residuals(prob, rec)
    sol = solver == "conicip-moi" ? MOI.get(r.opt, MOI.RawSolver()).sol : nothing
    merge!(row, Dict{String, Any}(
        "status" => string(rec.status), "iters" => rec.iters,
        "kkt_solves" => sol === nothing ? "" : sol.kkt_solves,
        "t_total" => r.t_opt, "t_assembly" => r.t_asm,
        "alloc_gib" => r.bytes / 2^30,
        "rDu" => res.rDu, "rPr" => res.rPr, "rEq" => res.rEq,
        "gap" => res.gap, "margin" => res.margin, "dual_margin" => res.dual_margin,
        "verified" => verified(rec, res),
        "kkt_repaired" => _sol_field(sol, :kkt_repaired),
        "kkt_refactors" => _sol_field(sol, :kkt_refactors)))
    return row
end

# Compile the MOI path of the chosen solver on tiny problems (suite's
# warmup covers ConicIP's direct path, which load_problem also exercises).
function warmup_baseline(solver)
    warmup()
    ctor = solver_constructor(solver)
    for small in (socp_sum_of_norms(10; d = 10), lp_band(50), qp_band(50))
        r = solve_once(ctor, small)
        recover_solution(r.opt, r.maps, small)
    end
    return nothing
end

function run_one_baseline(inst::Instance, solver; timeout = Inf)
    warmup_baseline(solver)
    row = run_baseline(inst, solver; timeout = timeout)
    row["maxrss_mb"] = Sys.maxrss() / 2^20
    return row
end

# ──────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────

function main(args)
    solver = solver_name(args)
    solver in SOLVERS || throw(ArgumentError("--solver must be one of $(join(SOLVERS, ", "))"))
    "--opt" in args && set_opts!(arg_value(args, "--opt"))
    timeout = parse(Float64, arg_value(args, "--timeout", "Inf"))
    if "--one" in args
        name = arg_value(args, "--one")
        inst = INSTANCES[findfirst(i -> i.name == name, INSTANCES)]
        for (k, v) in run_one_baseline(inst, solver; timeout = timeout)
            println(k, "=", v)
        end
        return
    end
    inproc = "--in-process" in args
    outpath = if "--out" in args
        arg_value(args, "--out")
    else
        mkpath(joinpath(@__DIR__, "results"))
        joinpath(@__DIR__, "results",
                 "baseline-$solver-$(Dates.format(now(), "yyyymmdd-HHMM")).csv")
    end
    selected = select_instances(args)

    println("# ConicIP benchmark baselines: ", solver)
    println(environment_header())
    isempty(OPT_STRING[]) || println("options: ", OPT_STRING[])
    isfinite(timeout) && println("timeout: ", timeout, " s per solve")
    println()
    child_timeout = isfinite(timeout) ? 2timeout + 120 : Inf
    rows = run_selected(selected; inproc = inproc, timeout = child_timeout,
                        runner = inst -> run_one_baseline(inst, solver; timeout = timeout),
                        script = @__FILE__, extra_args = ["--solver", solver])
    write_csv(outpath, rows; columns = BASE_COLUMNS)
    println("\nwrote ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    load_solver_package(solver_name(ARGS))
    # Same top-level expression as the load above, so the world age has not
    # advanced yet: invokelatest lets main see the package's methods.
    Base.invokelatest(main, ARGS)
end
