# SDPLIB benchmark
# ================
# Solves a spread of SDPLIB 1.2 instances through the MOI wrapper and
# reports size, status, iterations, wall time and relative error against the
# published optimal objective values. NOT part of the CI gate — the six
# instances cheap enough to run every build are checked by
# test/sdplib_tests.jl. This script is for the larger instances and for
# tracking solver progress over time:
#
#   julia --project benchmark/sdplib.jl
#   julia --project benchmark/sdplib.jl --max-seconds 120
#   julia --project benchmark/sdplib.jl --only 'truss|theta'
#
# Instances are fetched on demand from a pinned upstream commit, verified
# against the sha256 table in test/sdplib/instances.jl, and cached in the
# same ConicIP scratch space the test suite uses — so an instance is
# downloaded at most once per machine no matter which entry point asks for
# it. Nothing is redistributed with the package. See test/sdplib/README.md
# for the citation, the licence position, and the SDPA sign conventions —
# in short, the SDPA file format's *primal* is the geometric form MOI
# imports, so ObjectiveValue() matches the reference table with no sign flip.
#
# Each instance is solved in a worker process so that --max-seconds is a real
# budget: an instance that overruns is killed and reported as skipped, and a
# fresh worker is started for the next one.

using Downloads, Printf, Distributed, SHA

include(joinpath(@__DIR__, "..", "test", "sdplib", "instances.jl"))

# name => published optimal objective value, quoted from the SDPLIB 1.2
# README (vsdp/SDPLIB @ fa11b45c1d8c896a6abad2648d5dad46d8ecefaa).
const SDPLIB_REF = Dict(
    "control1" =>  1.778463e+01,
    "control2" =>  8.300000e+00,
    "control3" =>  1.363327e+01,
    "hinf1"    =>  2.0326e+00,
    "hinf2"    =>  1.0967e+01,
    "hinf3"    =>  5.69e+01,
    "truss1"   => -8.999996e+00,
    "truss2"   => -1.233804e+02,
    "truss3"   => -9.109996e+00,
    "truss4"   => -9.009996e+00,
    "theta1"   =>  2.300000e+01,
    "theta2"   =>  3.287917e+01,
    "mcp100"   =>  2.261574e+02,
    "mcp124-1" =>  1.419905e+02,
    "gpp100"   => -4.49435e+01,
    "qap5"     => -4.360e+02,
    "qap6"     => -3.8144e+02,
    "arch0"    =>  5.66517e-01,
)

const INSTANCES = [
    "control1", "control2", "control3",
    "hinf1", "hinf2", "hinf3",
    "truss1", "truss2", "truss3", "truss4",
    "theta1", "theta2",
    "mcp100", "mcp124-1", "gpp100",
    "qap5", "qap6",
    "arch0",
]

# ── options ───────────────────────────────────────────────────────────────

function parse_args(args)
    opts = (max_seconds = 60.0, only = r"", optTol = 1e-8)
    i = 1
    while i <= length(args)
        a = args[i]
        if a == "--max-seconds"
            opts = merge(opts, (max_seconds = parse(Float64, args[i+1]),)); i += 2
        elseif a == "--only"
            opts = merge(opts, (only = Regex(args[i+1]),)); i += 2
        elseif a == "--opt-tol"
            opts = merge(opts, (optTol = parse(Float64, args[i+1]),)); i += 2
        else
            error("unrecognised argument $a")
        end
    end
    return opts
end

# ── data ──────────────────────────────────────────────────────────────────
#
# `sdplib_fetch!` (test/sdplib/instances.jl) downloads on a cache miss,
# verifies sha256, and returns the cached path. The cache is resolved once so
# every instance shares it.

const CACHE_DIR = sdplib_cache_dir()

ensure_instance(name) = sdplib_fetch!(name; dir = CACHE_DIR)

# ── the solve, run inside a worker ────────────────────────────────────────

const WORKER_SETUP = quote
    using ConicIP
    import MathOptInterface as MOI

    # ConicIP does not implement MOI.BarrierIterations, so reach the raw
    # Solution through the wrapper stack. Returns -1 if the path changes.
    function _sdplib_iters(opt)
        inner = opt
        for _ in 1:8
            if inner isa ConicIP.Optimizer
                return inner.sol === nothing ? -1 : inner.sol.Iter
            elseif inner isa MOI.Bridges.AbstractBridgeOptimizer
                inner = inner.model
            elseif inner isa MOI.Utilities.CachingOptimizer
                inner = inner.optimizer
            else
                return -1
            end
        end
        return -1
    end

    # Total order of the block-diagonal matrix: PSD blocks contribute their
    # side dimension, linear blocks their length. This is SDPLIB's `n`.
    function _sdplib_order(model)
        n = 0
        for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
            for ci in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
                s = MOI.get(model, MOI.ConstraintSet(), ci)
                n += s isa MOI.PositiveSemidefiniteConeTriangle ?
                     MOI.side_dimension(s) : MOI.dimension(s)
            end
        end
        return n
    end

    function sdplib_run(path, optTol)
        model = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_SDPA)
        MOI.read_from_file(model, path)
        m = MOI.get(model, MOI.NumberOfVariables())
        n = _sdplib_order(model)
        opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
        MOI.set(opt, MOI.Silent(), true)
        MOI.set(opt, MOI.RawOptimizerAttribute("optTol"), optTol)
        secs = @elapsed begin
            MOI.copy_to(opt, model)
            MOI.optimize!(opt)
        end
        status = MOI.get(opt, MOI.TerminationStatus())
        obj = MOI.get(opt, MOI.ResultCount()) > 0 ?
              MOI.get(opt, MOI.ObjectiveValue()) : NaN
        return (m = m, n = n, status = string(status),
                iters = _sdplib_iters(opt), secs = secs, obj = obj)
    end

    # `remotecall_eval` ships the block's value back to the driver, and a
    # function object defined only on the worker cannot be deserialized
    # there. Return nothing instead.
    nothing
end

# ── worker lifecycle ──────────────────────────────────────────────────────

function start_worker()
    w = only(addprocs(1; exeflags = "--project=$(Base.active_project())"))
    Distributed.remotecall_eval(Main, w, WORKER_SETUP)
    return w
end

kill_worker(w) = (try rmprocs(w; waitfor = 0) catch end; nothing)

# Run one instance under a wall-clock budget. Returns
#   (:ok, result)       solved (whatever the termination status)
#   (:timeout, reason)  overran the budget; the worker is now unusable
#   (:error, reason)    the solve threw; the worker is still usable
function run_budgeted(w, path, optTol, budget)
    # `sdplib_run` exists only on the worker, so send an expression rather
    # than a function object (which the driver cannot serialize).
    fut = remotecall(Core.eval, w, Main, :(sdplib_run($path, $optTol)))
    # Do NOT poll `isready(fut)`: on a remote `Future` that call queries the
    # owning worker, which is busy running the very solve being timed, so it
    # blocks and the budget never fires. Wait on a local task instead. The
    # task swallows its own exception so that abandoning it on a timeout
    # does not surface later as an unhandled task failure.
    task = @async try fetch(fut) catch e; e end
    if timedwait(() -> istaskdone(task), budget; pollint = 0.05) !== :ok
        return (:timeout, "exceeded --max-seconds ($budget s)")
    end
    res = fetch(task)
    if res isa Exception
        res isa RemoteException && (res = res.captured.ex)
        return (:error, "errored: " * first(sprint(showerror, res), 72))
    end
    return (:ok, res)
end

# ── driver ────────────────────────────────────────────────────────────────

function main(args = ARGS)
    opts = parse_args(args)
    names = filter(n -> occursin(opts.only, n), INSTANCES)

    @printf("%-10s %6s %6s  %-17s %5s %9s  %14s %10s\n",
            "instance", "m", "n", "status", "iter", "seconds", "objective", "rel.err")
    println("-"^88)

    w = start_worker()
    # Warm up the worker so the first timed solve is not measuring compilation.
    warm = try ensure_instance("truss1") catch; nothing end
    warm === nothing || run_budgeted(w, warm, opts.optTol, 300.0)

    for name in names
        path = try
            ensure_instance(name)
        catch e
            @printf("%-10s %s\n", name, "skipped: download failed: " *
                    first(sprint(showerror, e), 60))
            continue
        end

        outcome, res = run_budgeted(w, path, opts.optTol, opts.max_seconds)
        if outcome !== :ok
            @printf("%-10s %s\n", name, "skipped: " * res)
            flush(stdout)
            if outcome === :timeout
                # The runaway solve cannot be interrupted, only killed.
                kill_worker(w)
                w = start_worker()
                warm === nothing || run_budgeted(w, warm, opts.optTol, 300.0)
            end
            continue
        end

        ref = SDPLIB_REF[name]
        relerr = isnan(res.obj) ? NaN : abs(res.obj - ref) / (1 + abs(ref))
        @printf("%-10s %6d %6d  %-17s %5d %9.3f  %14.7e %10.2e\n",
                name, res.m, res.n, res.status, res.iters, res.secs,
                res.obj, relerr)
        flush(stdout)
    end

    kill_worker(w)
    return 0
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(main())
end
