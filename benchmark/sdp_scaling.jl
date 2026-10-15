# SDP scaling harness (roadmap Tranche 4)
# =======================================
# How the cost of a semidefinite solve grows with the PSD block order k, and
# how it depends on the formulation the model is written in:
#
#   primal  the matrix is the variable (JuMP's `@variable(X, PSD)`, MOI
#           `VectorOfVariables`-in-`PositiveSemidefiniteConeTriangle`), so the
#           solver sees n ≈ k(k+1)/2 variables;
#   dual    the matrix is affine in a few variables
#           (`VectorAffineFunction`-in-PSD), so n is the constraint count.
#
#   mixed   a PSD variable among other variables and rows (the mixed form of
#           kktsolver_sdpschur): free scalars, bounds on matrix entries.
#
# Families: the max-cut relaxation (p = k diagonal equalities) and the Lovász
# theta function of a sparse random graph (p = 1 + |E|, |E| ≈ 2k). Both have
# equal primal and dual optimal values, which the CSV records for comparison.
# Mixed form: `theta-mixed` is theta with a free scalar,
# ϑ(H) = min t s.t. Z ⪰ 0, Z_ii = t − 1, Z_ij = −1 on the |E| ≈ 2k
# non-adjacent pairs of H (H is the complement of the theta graph, so its
# value differs from theta-primal's); `boxqp-mixed` is the Shor relaxation
# of a box-constrained QP, Y = [1 xᵀ; x X] ⪰ 0 of order k + 1 with Y₁₁ = 1,
# X_ii ≤ x_i and 0 ≤ x ≤ 1 (3k inequality rows on entries of Y).
#
#   julia --project benchmark/sdp_scaling.jl                 # full set
#   julia --project benchmark/sdp_scaling.jl --quick         # small subset
#   julia --project benchmark/sdp_scaling.jl --only maxcut-primal-100
#   julia --project benchmark/sdp_scaling.jl --blas-threads 4 --timeout 600
#   julia --project benchmark/sdp_scaling.jl --opt kktsolver=qr --out x.csv
#
# Each instance runs in a fresh process (the suite.jl machinery): the child
# warms up on an order-10 instance of the same family and form, then solves
# twice and keeps the faster run. Recorded per row: the assembled dimensions
# (n, m, p) and the dimensions the KKT backend actually receives after
# presolve (n_kkt, m_kkt, p_kkt), the backend chosen, iterations, wall time
# and seconds per iteration, the phase timers (KKT update, and its dense
# assembly / factorization split), peak RSS, and residuals recomputed from the
# assembled data. BLAS threads are pinned (default 4) and recorded.
#
# Not part of CI. Results depend on hardware; the header records the
# environment.

isdefined(@__MODULE__, :run_selected) || include(joinpath(@__DIR__, "suite.jl"))

using Random

# ──────────────────────────────────────────────────────────────
#  Model builders (MOI only, no JuMP)
# ──────────────────────────────────────────────────────────────

# Column-major upper triangle, the PositiveSemidefiniteConeTriangle order.
tri(i, j) = i <= j ? div(j * (j - 1), 2) + i : tri(j, i)

function maxcut_laplacian(k; density = 0.3, seed = 1)
    rng = MersenneTwister(seed)
    W = zeros(k, k)
    for j in 1:k, i in 1:j-1
        rand(rng) < density && (W[i, j] = W[j, i] = 1.0)
    end
    return Diagonal(vec(sum(W; dims = 2))) - W
end

function theta_edges(k; degree = 4, seed = 1)
    rng = MersenneTwister(seed)
    E = Set{Tuple{Int,Int}}()
    target = div(degree * k, 2)
    while length(E) < target
        i, j = rand(rng, 1:k), rand(rng, 1:k)
        i == j && continue
        push!(E, (min(i, j), max(i, j)))
    end
    return sort!(collect(E))
end

saf(terms, c = 0.0) = MOI.ScalarAffineFunction(terms, c)
term(c, x) = MOI.ScalarAffineTerm(c, x)

# max ⟨L, X⟩/4  s.t.  X_ii = 1,  X ⪰ 0
function maxcut_primal(k)
    L = maxcut_laplacian(k)
    model = MOI.Utilities.Model{Float64}()
    X = MOI.add_variables(model, div(k * (k + 1), 2))
    MOI.add_constraint(model, MOI.VectorOfVariables(X), MOI.PositiveSemidefiniteConeTriangle(k))
    for i in 1:k
        MOI.add_constraint(model, saf([term(1.0, X[tri(i, i)])]), MOI.EqualTo(1.0))
    end
    obj = [term((i == j ? 1.0 : 2.0) * L[i, j] / 4, X[tri(i, j)]) for j in 1:k for i in 1:j if L[i, j] != 0]
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf(obj))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    return model
end

# min Σ yᵢ  s.t.  Diag(y) − L/4 ⪰ 0
function maxcut_dual(k)
    L = maxcut_laplacian(k)
    model = MOI.Utilities.Model{Float64}()
    y = MOI.add_variables(model, k)
    terms = [MOI.VectorAffineTerm(tri(i, i), term(1.0, y[i])) for i in 1:k]
    consts = [-L[i, j] / 4 for j in 1:k for i in 1:j]
    MOI.add_constraint(model, MOI.VectorAffineFunction(terms, consts),
                       MOI.PositiveSemidefiniteConeTriangle(k))
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
            saf([term(1.0, yi) for yi in y]))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return model
end

# max ⟨J, X⟩  s.t.  tr X = 1,  X_ij = 0 (ij ∈ E),  X ⪰ 0
function theta_primal(k)
    E = theta_edges(k)
    model = MOI.Utilities.Model{Float64}()
    X = MOI.add_variables(model, div(k * (k + 1), 2))
    MOI.add_constraint(model, MOI.VectorOfVariables(X), MOI.PositiveSemidefiniteConeTriangle(k))
    MOI.add_constraint(model, saf([term(1.0, X[tri(i, i)]) for i in 1:k]), MOI.EqualTo(1.0))
    for (i, j) in E
        MOI.add_constraint(model, saf([term(1.0, X[tri(i, j)])]), MOI.EqualTo(0.0))
    end
    obj = [term(i == j ? 1.0 : 2.0, X[tri(i, j)]) for j in 1:k for i in 1:j]
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf(obj))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
    return model
end

# min t  s.t.  t I − J − Σ_{ij∈E} y_ij (e_i e_jᵀ + e_j e_iᵀ) ⪰ 0
function theta_dual(k)
    E = theta_edges(k)
    model = MOI.Utilities.Model{Float64}()
    t = MOI.add_variable(model)
    y = MOI.add_variables(model, length(E))
    terms = [MOI.VectorAffineTerm(tri(i, i), term(1.0, t)) for i in 1:k]
    for (e, (i, j)) in enumerate(E)
        push!(terms, MOI.VectorAffineTerm(tri(i, j), term(-1.0, y[e])))
    end
    consts = fill(-1.0, div(k * (k + 1), 2))
    MOI.add_constraint(model, MOI.VectorAffineFunction(terms, consts),
                       MOI.PositiveSemidefiniteConeTriangle(k))
    MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(), saf([term(1.0, t)]))
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    return model
end

# min t  s.t.  Z ⪰ 0,  Z_ii − t = −1,  Z_ij = −1 (ij ∈ E)
function theta_mixed(k)
    E = theta_edges(k)
    model = MOI.Utilities.Model{Float64}()
    Z = MOI.add_variables(model, div(k * (k + 1), 2))
    MOI.add_constraint(model, MOI.VectorOfVariables(Z), MOI.PositiveSemidefiniteConeTriangle(k))
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

# min ½⟨Q, X⟩ + cᵀx  s.t.  [1 xᵀ; x X] ⪰ 0,  X_ii ≤ x_i,  0 ≤ x ≤ 1
function boxqp_mixed(k; seed = 1)
    rng = MersenneTwister(seed)
    Qb = randn(rng, k, k); Qb = (Qb + Qb') / 2; c = randn(rng, k)
    model = MOI.Utilities.Model{Float64}()
    Y = MOI.add_variables(model, div((k + 1) * (k + 2), 2))
    MOI.add_constraint(model, MOI.VectorOfVariables(Y), MOI.PositiveSemidefiniteConeTriangle(k + 1))
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

const BUILDERS = Dict(("maxcut", "primal") => maxcut_primal, ("maxcut", "dual") => maxcut_dual,
                      ("theta", "primal") => theta_primal, ("theta", "dual") => theta_dual,
                      ("theta", "mixed") => theta_mixed, ("boxqp", "mixed") => boxqp_mixed)

# ──────────────────────────────────────────────────────────────
#  Instance registry
# ──────────────────────────────────────────────────────────────

const SDP_SIZES = Dict(
    ("maxcut", "primal") => (50, 70, 100, 150, 200),
    ("maxcut", "dual")   => (50, 100, 150, 200, 300),
    ("theta", "primal")  => (50, 70, 100, 150),
    ("theta", "dual")    => (50, 100, 150, 200),
    ("theta", "mixed")   => (50, 70, 100, 150, 200),
    ("boxqp", "mixed")   => (50, 70, 100, 150, 200),
)
const SDP_QUICK = (50, 70)

function sdp_instances()
    insts = Instance[]
    for (fam, form) in (("maxcut", "primal"), ("maxcut", "dual"), ("theta", "primal"),
                        ("theta", "dual"), ("theta", "mixed"), ("boxqp", "mixed")),
        k in SDP_SIZES[(fam, form)]
        name = "$fam-$form-$k"
        push!(insts, Instance(name, :sdp, "$fam-$form", k in SDP_QUICK,
                              () -> BUILDERS[(fam, form)](k), nothing))
    end
    return insts
end
const SDP_INSTANCES = sdp_instances()

parse_inst(name) = (m = match(r"^(\w+)-(primal|dual|mixed)-(\d+)$", name);
                    (fam = m[1], form = m[2], k = parse(Int, m[3])))

# ──────────────────────────────────────────────────────────────
#  Solve and measure (runs in the child)
# ──────────────────────────────────────────────────────────────

# Records what the KKT backend receives (after presolve and equilibration)
# and which backend the router picks, then delegates. The first call is the
# main solve; later calls (certificate fallback) are not recorded.
mutable struct KKTProbe
    inner::Any
    dims::Union{Nothing, NTuple{3, Int}}
    name::String
end
KKTProbe(inner) = KKTProbe(inner, nothing, "")
function (pr::KKTProbe)(Q, A, G, cd)
    if pr.dims === nothing
        pr.dims = (size(Q, 1), size(A, 1), size(G, 1))
        pr.name = if pr.inner === ConicIP.default_kktsolver
            try string(nameof(ConicIP.choose_kktsolver(Q, A, G, cd))) catch; "rejected" end
        else
            string(pr.inner)
        end
    end
    return pr.inner(Q, A, G, cd)
end

function sdp_solve(model)
    opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
    MOI.set(opt, MOI.Silent(), true)
    inner = ConicIP.default_kktsolver
    for (k, v) in SOLVER_OPTS
        k == :kktsolver ? (inner = ConicIP._resolve_kktsolver(v)) :
                          MOI.set(opt, MOI.RawOptimizerAttribute(string(k)), v)
    end
    probe = KKTProbe(inner)
    MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), probe)
    pt = ConicIP.PhaseTimes()
    MOI.set(opt, MOI.RawOptimizerAttribute("timing"), pt)
    t = @elapsed begin
        MOI.copy_to(opt, model)
        MOI.optimize!(opt)
    end
    return (opt = opt, t = t, pt = pt, probe = probe)
end

function run_sdp(inst::Instance)
    (; fam, form, k) = parse_inst(inst.name)
    sdp_solve(BUILDERS[(fam, form)](10))             # warm-up, same code paths
    model = inst.load()
    r1 = sdp_solve(model); r2 = sdp_solve(model)
    r = r1.t <= r2.t ? r1 : r2
    raw = MOI.get(r.opt, MOI.RawSolver())::ConicIP.Optimizer
    sol = raw.sol
    n = raw.n
    m = raw.ineq_A === nothing ? 0 : size(raw.ineq_A, 1)
    p = raw.eq_G === nothing ? 0 : size(raw.eq_G, 1)
    prob = (Q = raw.Q_int === nothing ? spzeros(n, n) : raw.Q_int,
            c = raw.c_int,
            A = raw.ineq_A === nothing ? spzeros(0, n) : raw.ineq_A,
            b = raw.ineq_b,
            G = raw.eq_G === nothing ? spzeros(0, n) : raw.eq_G,
            d = raw.eq_d, cone_dims = raw.cone_dims)
    res = sol === nothing ? nothing : residuals(prob, sol)
    iters = sol === nothing ? 0 : sol.Iter
    dims = something(r.probe.dims, (-1, -1, -1))
    pt = r.pt
    status = string(MOI.get(r.opt, MOI.TerminationStatus()))
    obj = MOI.get(r.opt, MOI.ResultCount()) > 0 ? MOI.get(r.opt, MOI.ObjectiveValue()) : NaN
    return Dict{String, Any}(
        "form" => form, "k" => k, "n" => n, "m" => m, "p" => p,
        "n_kkt" => dims[1], "m_kkt" => dims[2], "p_kkt" => dims[3],
        "kktsolver" => r.probe.name, "status" => status, "iters" => iters,
        "objective" => obj, "t_total" => r.t, "s_per_iter" => r.t / max(iters, 1),
        "t_init" => pt.t_init / 1e9, "t_loop" => pt.t_loop / 1e9,
        "t_kktupdate" => pt.t_kktupdate / 1e9, "t_direction" => pt.t_direction / 1e9,
        "t_kkt_assemble" => pt.t_kkt_assemble / 1e9, "t_kkt_factor" => pt.t_kkt_factor / 1e9,
        "n_kkt_factor" => pt.n_kkt_factor, "t_fallback" => pt.t_fallback / 1e9,
        "rDu" => res === nothing ? NaN : res.rDu, "rPr" => res === nothing ? NaN : res.rPr,
        "rEq" => res === nothing ? NaN : res.rEq, "gap" => res === nothing ? NaN : res.gap,
        "margin" => res === nothing ? NaN : res.margin,
        "dual_margin" => res === nothing ? NaN : res.dual_margin,
        "verified" => sol !== nothing && verified(sol, res),
    )
end

function run_one_sdp(inst::Instance)
    # The router refuses an SDP whose dense KKT storage exceeds its budget
    # with an ArgumentError out of optimize!; record that as the outcome.
    result = try
        run_sdp(inst)
    catch err
        err isa ArgumentError || rethrow()
        msg = sprint(showerror, err)
        gib = match(r"estimated ([0-9.]+) GiB", msg)
        Dict{String, Any}("status" => gib === nothing ? "REJECTED" : "REJECTED($(gib[1])GiB)",
                          "k" => parse_inst(inst.name).k, "form" => parse_inst(inst.name).form)
    end
    result["maxrss_mb"] = Sys.maxrss() / 2^20
    return result
end

# ──────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────

const SDP_COLUMNS = ["name", "family", "form", "k", "n", "m", "p", "n_kkt", "m_kkt", "p_kkt",
                     "kktsolver", "status", "iters", "objective", "t_total", "s_per_iter",
                     "t_init", "t_loop", "t_kktupdate", "t_direction", "t_kkt_assemble",
                     "t_kkt_factor", "n_kkt_factor", "t_fallback", "maxrss_mb",
                     "rDu", "rPr", "rEq", "gap", "margin", "dual_margin", "verified"]

function sdp_main(args)
    "--opt" in args && set_opts!(arg_value(args, "--opt"))
    nblas = parse(Int, arg_value(args, "--blas-threads", "4"))
    BLAS.set_num_threads(nblas)
    if "--one" in args
        name = arg_value(args, "--one")
        inst = SDP_INSTANCES[findfirst(i -> i.name == name, SDP_INSTANCES)]
        for (k, v) in run_one_sdp(inst)
            println(k, "=", v)
        end
        return
    end
    inproc = "--in-process" in args
    timeout = parse(Float64, arg_value(args, "--timeout", "600"))
    outpath = if "--out" in args
        arg_value(args, "--out")
    else
        mkpath(joinpath(@__DIR__, "results"))
        joinpath(@__DIR__, "results", "sdp-$(Dates.format(now(), "yyyymmdd-HHMM")).csv")
    end
    quick = "--quick" in args
    only = "--only" in args ? Set(split(arg_value(args, "--only"), ',')) : nothing
    selected = filter(SDP_INSTANCES) do inst
        only !== nothing ? inst.name in only : (!quick || inst.quick)
    end

    println("# ConicIP SDP scaling")
    println(environment_header())
    isempty(OPT_STRING[]) || println("options: ", OPT_STRING[])
    println()
    rows = run_selected(selected; inproc = inproc, timeout = timeout, runner = run_one_sdp,
                        script = @__FILE__, extra_args = ["--blas-threads", string(nblas)])
    write_csv(outpath, rows; columns = SDP_COLUMNS)
    println("\nwrote ", outpath)
    println()
    @printf("%-22s %-18s %7s %7s %5s %10s %10s %9s %9s %s\n", "instance", "kktsolver",
            "n_kkt", "p_kkt", "iter", "s/iter", "assemble", "factor", "rss MB", "verified")
    for r in rows
        @printf("%-22s %-18s %7s %7s %5s %10s %10s %9s %9s %s\n", r["name"],
                get(r, "kktsolver", ""), get(r, "n_kkt", ""), get(r, "p_kkt", ""),
                get(r, "iters", ""), fmt(_num(get(r, "s_per_iter", NaN))),
                fmt(_num(get(r, "t_kkt_assemble", NaN))), fmt(_num(get(r, "t_kkt_factor", NaN))),
                fmt(_num(get(r, "maxrss_mb", NaN))), get(r, "verified", ""))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    sdp_main(ARGS)
end
