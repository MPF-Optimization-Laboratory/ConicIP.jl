# Per-phase timing of ConicIP and Clarabel on the benchmark instances (private)
# ============================================================================
# Both solvers take the SAME route as the headline comparison in baselines.jl:
# the problem tuple goes through `build_moi_model` (moi_model.jl) into a
# bridged, cached MOI model, and `wall` is `@elapsed MOI.optimize!(opt)`.
# For ConicIP, `optimize!` covers the wrapper's assembly (t_frontend, charged
# by src/MOI_wrapper.jl) and the solver phases of `ConicIP.PhaseTimes`
# (src/timing.jl). For Clarabel the CachingOptimizer also performs the
# `copy_to` inside `optimize!` (Clarabel's MOI wrapper builds the
# `Clarabel.Solver`, and hence runs `setup!`, in `copy_to`), so the two walls
# are comparable; Clarabel's phases are read from `solver.timers`.
#
#   julia --project           benchmark/phases.jl --solver conicip  [--quick] [--only a,b]
#   julia --project=benchmark benchmark/phases.jl --solver clarabel [--quick] [--only a,b]
#         [--reps 2] [--timeout N] [--out results/phase-<solver>.csv] [--in-process]
#
# Protocol per instance (fresh Julia process each, as in suite.jl): warm-up
# on three tiny problems through the same route, with a `PhaseTimes` attached
# for ConicIP so that the timed code path is compiled; then `--reps` timed
# repetitions, each with a fresh `PhaseTimes`; the whole record (phases,
# counts, bytes, wall) of the FASTEST repetition is kept. `--timeout N` kills
# the child after N seconds of wall clock (warm-up and compilation included).
#
# Clarabel timer mapping (Clarabel 0.11.1, src/solver.jl; values in ns):
#
#   t_frontend    = wall − (setup! + solve! + post-process)   [by subtraction:
#                   MOI copy/translation through bridges and the cache, plus
#                   Clarabel's own MOI wrapper work; ConicIP's t_frontend is
#                   the analogous wrapper work, measured inside its wrapper]
#   t_presolve    = setup!/presolve
#   t_equilibrate = setup!/equilibration
#   t_setup       = setup!/kkt init
#   t_init        = solve!/default start
#   t_loop        = solve!/IP iteration
#   t_postsolve   = post-process                              (a ROOT section)
#   t_scaling     = solve!/IP iteration/scale cones
#   t_kktupdate   = solve!/IP iteration/kkt update  (includes a constant-RHS
#                   solve with refinement, kktsystem.jl:62; compare
#                   t_kkt_total = t_kktupdate + t_direction across solvers)
#   t_direction   = solve!/IP iteration/kkt solve
#   t_loop_rest   = IP iteration − (scale cones + kkt update + kkt solve)
#                   (ConicIP: t_residuals + t_rhs + t_linesearch)
#   t_other_call  = wall − Σ whole-call phases: for Clarabel the untimed
#                   parts of setup!/solve! (cone and variable construction,
#                   info reset); for ConicIP whatever the instrumentation
#                   does not cover, plus the bridge/cache layer
#   iters         = solver.info.iterations (ncalls of "IP iteration" is 1);
#                   ConicIP's `iters` column is n_passes. `iters_reported`
#                   is MOI.BarrierIterations for both.
#
# ConicIP-only columns are left empty in Clarabel rows. All times are seconds.

include(joinpath(@__DIR__, "baselines.jl"))   # suite.jl + moi_model.jl, guarded mains

const PHASE_SOLVERS = ("conicip", "clarabel")

const WHOLE_CALL = [:t_frontend, :t_presolve, :t_equilibrate, :t_setup, :t_init,
                    :t_loop, :t_final, :t_fallback, :t_postsolve]
const LOOP_CHILDREN = [:t_scaling, :t_residuals, :t_kktupdate, :t_direction, :t_rhs, :t_linesearch]
const DIAGNOSTICS = [:t_dir_base, :t_dir_refine, :t_dir_refine_resid,
                     :t_ldl_factor, :t_ldl_solve, :t_ldl_resid]
const BYTES = [:b_presolve, :b_equilibrate, :b_setup, :b_scaling, :b_residuals,
               :b_kktupdate, :b_direction, :b_rhs, :b_linesearch]
const COUNTS = [:n_steps, :n_kktupdate, :n_solve, :n_refine_attempt, :n_refine_resid,
                :n_ldl_factor, :n_ldl_solve, :n_ldl_resid]

const PHASE_COLUMNS = vcat(
    ["instance", "solver", "status", "verified", "wall", "iters", "iters_reported"],
    string.(COUNTS),
    string.(WHOLE_CALL), ["t_other_call"],
    string.(LOOP_CHILDREN), ["t_loop_rest", "t_other_loop", "t_kkt_total"],
    string.(DIAGNOSTICS), string.(BYTES), ["t_gc"],
    ["t_build_moi", "n", "m", "p", "data_nnz", "maxrss_mb",
     "julia", "threads", "blas_threads", "blas", "sha", "note"])

phase_ctor(solver) = solver == "conicip" ? ConicIP.Optimizer : solver_constructor(solver)

# ──────────────────────────────────────────────────────────────
#  One timed optimize! through the baselines route
# ──────────────────────────────────────────────────────────────

# `(opt, maps, pt, wall, t_build)`: a fresh bridged/cached optimizer, the
# model built from `prob`, and the wall time of `optimize!`. `pt` is the
# `PhaseTimes` attached to a ConicIP run, `nothing` for Clarabel.
function solve_phased(ctor, prob, solver; opts = SOLVER_OPTS)
    opt = bridged_cached(ctor)
    MOI.set(opt, MOI.Silent(), true)
    for (k, v) in opts
        MOI.set(opt, MOI.RawOptimizerAttribute(string(k)), v)
    end
    pt = nothing
    if solver == "conicip"
        pt = ConicIP.PhaseTimes()
        MOI.set(opt, MOI.RawOptimizerAttribute("timing"), pt)
    end
    t_build = @elapsed maps = build_moi_model(opt, prob)
    wall = @elapsed MOI.optimize!(opt)
    return (opt = opt, maps = maps, pt = pt, wall = wall, t_build = t_build)
end

ns2s(x) = Float64(x) / 1e9

# Phase columns of a ConicIP repetition, from its PhaseTimes.
function conicip_record(pt::ConicIP.PhaseTimes, wall)
    row = Dict{String, Any}()
    for f in vcat(WHOLE_CALL, LOOP_CHILDREN, DIAGNOSTICS)
        row[string(f)] = ns2s(getfield(pt, f))
    end
    for f in vcat(BYTES, COUNTS)
        row[string(f)] = getfield(pt, f)
    end
    row["iters"] = pt.n_passes
    row["t_gc"] = ns2s(pt.t_gc)
    row["t_other_call"] = wall - sum(row[string(f)] for f in WHOLE_CALL)
    row["t_other_loop"] = row["t_loop"] - sum(row[string(f)] for f in LOOP_CHILDREN)
    row["t_loop_rest"] = row["t_residuals"] + row["t_rhs"] + row["t_linesearch"]
    row["t_kkt_total"] = row["t_kktupdate"] + row["t_direction"]
    return row
end

# Time in seconds of the timer section at `path` under `to`, 0 when a
# section is absent (early exits skip children).
function timer_section(TO, to, path...)
    node = to
    for name in path
        haskey(node.inner_timers, name) || return 0.0
        node = node[name]
    end
    return ns2s(TO.time(node))
end

# Phase columns of a Clarabel repetition, from the solver's TimerOutput.
function clarabel_record(solver_obj, wall)
    TO = clarabel_module().TimerOutputs
    to = solver_obj.timers
    sec(path...) = timer_section(TO, to, path...)
    row = Dict{String, Any}()
    t_setup_root = sec("setup!"); t_solve_root = sec("solve!"); t_post = sec("post-process")
    row["t_frontend"] = wall - (t_setup_root + t_solve_root + t_post)
    row["t_presolve"] = sec("setup!", "presolve")
    row["t_equilibrate"] = sec("setup!", "equilibration")
    row["t_setup"] = sec("setup!", "kkt init")
    row["t_init"] = sec("solve!", "default start")
    row["t_loop"] = sec("solve!", "IP iteration")
    row["t_final"] = 0.0; row["t_fallback"] = 0.0
    row["t_postsolve"] = t_post
    row["t_scaling"] = sec("solve!", "IP iteration", "scale cones")
    row["t_kktupdate"] = sec("solve!", "IP iteration", "kkt update")
    row["t_direction"] = sec("solve!", "IP iteration", "kkt solve")
    row["t_loop_rest"] = row["t_loop"] - (row["t_scaling"] + row["t_kktupdate"] + row["t_direction"])
    row["t_other_call"] = wall - sum(row[string(f)] for f in WHOLE_CALL)
    row["t_other_loop"] = 0.0          # t_loop_rest is the remainder by construction
    row["t_kkt_total"] = row["t_kktupdate"] + row["t_direction"]
    row["iters"] = Int(solver_obj.info.iterations)
    row["n_kktupdate"] = Int(solver_obj.info.iterations)
    row["note"] = string(solver_obj.info.status)
    return row
end

clarabel_module() = Base.require(Main, :Clarabel)

# The `Clarabel.Solver` behind a bridged/cached optimizer. `MOI.get(opt,
# MOI.RawSolver())` cannot be used here: the CachingOptimizer tries to
# `map_indices` over the returned object and has no method for it.
function clarabel_solver(opt)
    inner = opt
    while true
        if inner isa MOI.Utilities.CachingOptimizer
            inner = inner.optimizer
        elseif inner isa MOI.Bridges.AbstractBridgeOptimizer
            inner = inner.model
        else
            break
        end
    end
    return inner.solver
end

# ──────────────────────────────────────────────────────────────
#  Per-instance driver (runs in the child)
# ──────────────────────────────────────────────────────────────

function environment_fields()
    sha = try strip(read(`git -C $(@__DIR__) rev-parse --short HEAD`, String)) catch; "?" end
    blas = try
        join(unique(basename(lib.libname) for lib in BLAS.get_config().loaded_libs), ";")
    catch
        "?"
    end
    return Dict{String, Any}("julia" => string(VERSION), "threads" => Threads.nthreads(),
                             "blas_threads" => BLAS.get_num_threads(), "blas" => blas,
                             "sha" => sha)
end

function warmup_phases(solver)
    warmup()
    ctor = phase_ctor(solver)
    for small in (socp_sum_of_norms(10; d = 10), lp_band(50), qp_band(50))
        r = solve_phased(ctor, small, solver)
        recover_solution(r.opt, r.maps, small)
        solver == "clarabel" && clarabel_record(clarabel_solver(r.opt), r.wall)
    end
    return nothing
end

function run_phases(inst::Instance, solver; reps = 2)
    ctor = phase_ctor(solver)
    prob, note = load_problem(inst)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    row = Dict{String, Any}(
        "instance" => inst.name, "solver" => solver, "note" => note,
        "n" => n, "m" => m, "p" => p,
        "data_nnz" => nnz(sparse(prob.Q)) + nnz(sparse(prob.A)) + nnz(sparse(prob.G)))
    merge!(row, environment_fields())
    best = nothing
    try
        for _ in 1:max(reps, 1)
            r = solve_phased(ctor, prob, solver)
            (best === nothing || r.wall < best.wall) && (best = r)
        end
    catch err
        row["status"] = "FAILED"
        row["note"] = err isa MOI.UnsupportedError ? "BRIDGE_FAIL" : "ERROR: " * _short(err)
        return row
    end
    rec = recover_solution(best.opt, best.maps, prob)
    res = residuals(prob, rec)
    phases = solver == "conicip" ? conicip_record(best.pt, best.wall) :
             clarabel_record(clarabel_solver(best.opt), best.wall)
    isempty(get(phases, "note", "")) || (row["note"] = strip(row["note"] * " " * phases["note"]))
    delete!(phases, "note")
    merge!(row, phases)
    merge!(row, Dict{String, Any}(
        "status" => string(rec.status), "verified" => verified(rec, res),
        "wall" => best.wall, "t_build_moi" => best.t_build,
        "iters_reported" => rec.iters,
        # keys read by run_selected's progress line
        "t_total" => best.wall, "kkt_solves" => get(row, "n_solve", "")))
    return row
end

function run_one_phases(inst::Instance, solver; reps = 2)
    warmup_phases(solver)
    row = run_phases(inst, solver; reps = reps)
    row["maxrss_mb"] = Sys.maxrss() / 2^20
    return row
end

# ──────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────

phase_solver(args) = lowercase(arg_value(args, "--solver", "conicip"))

function main(args)
    solver = phase_solver(args)
    solver in PHASE_SOLVERS || throw(ArgumentError("--solver must be one of $(join(PHASE_SOLVERS, ", "))"))
    "--opt" in args && set_opts!(arg_value(args, "--opt"))
    reps = parse(Int, arg_value(args, "--reps", "2"))
    if "--one" in args
        name = arg_value(args, "--one")
        inst = INSTANCES[findfirst(i -> i.name == name, INSTANCES)]
        for (k, v) in run_one_phases(inst, solver; reps = reps)
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
        joinpath(@__DIR__, "results", "phase-$solver.csv")
    end
    selected = select_instances(args)

    println("# ConicIP per-phase timing: ", solver, "  (reps=", reps, ")")
    println(environment_header())
    isempty(OPT_STRING[]) || println("options: ", OPT_STRING[])
    println()
    rows = run_selected(selected; inproc = inproc, timeout = timeout,
                        runner = inst -> run_one_phases(inst, solver; reps = reps),
                        script = @__FILE__,
                        extra_args = ["--solver", solver, "--reps", string(reps)])
    for r in rows
        r["instance"] = r["name"]; r["solver"] = solver
    end
    mkpath(dirname(abspath(outpath)))
    write_csv(outpath, rows; columns = PHASE_COLUMNS)
    println("\nwrote ", outpath)
end

if abspath(PROGRAM_FILE) == @__FILE__
    load_solver_package(phase_solver(ARGS))
    # Same top-level expression as the load above (see baselines.jl).
    Base.invokelatest(main, ARGS)
end
