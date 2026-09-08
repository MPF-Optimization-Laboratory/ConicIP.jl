# Tests for the benchmark harness pieces that are MOI-only (model builder,
# solution recovery, option parsing). Wrapped in a module because
# review_benchmark.jl already includes benchmark/suite.jl in its own module.
module HarnessTests
using Test, LinearAlgebra, SparseArrays, Random
include("../benchmark/suite.jl")       # ConicIP, MOI, residuals, verified, parse_opts, …
include("../benchmark/moi_model.jl")   # build_moi_model, recover_solution, bridged_cached

# R × Q × S problem with equalities, feasible by construction: a strictly
# interior slack s₀ and a point y₀ define b = A y₀ − s₀ and d = G y₀; the
# strictly convex Q makes the minimizer unique.
function mixed_cone_problem(; seed = 11)
    rng = MersenneTwister(seed)
    cone_dims = [("R", 3), ("Q", 4), ("S", 6)]
    n = 12; m = 13; p = 2
    A = sparse(randn(rng, m, n))
    G = sparse(randn(rng, p, n))
    y0 = randn(rng, n)
    u = randn(rng, 3)
    M = randn(rng, 3, 3)
    s0 = vcat(rand(rng, 3) .+ 0.5, [norm(u) + 1.0; u], ConicIP.vecm(M * M' + I))
    b = A * y0 - s0
    d = G * y0
    L = randn(rng, n, n)
    Q = sparse(L' * L / n + 0.1I)
    c = randn(rng, n)
    return (Q = Q, c = c, A = A, b = b, cone_dims = cone_dims, G = G, d = d)
end

@testset "Benchmark harness" begin
    @testset "PSD triangle permutation matches ConicIP" begin
        for nmat in 1:5
            k = nmat * (nmat + 1) ÷ 2
            @test psd_moi_vecm_info(k) == ConicIP._psd_moi_vecm_info(k)
            # Round trip through the row map: vecm → MOI → vecm.
            x = randn(k)
            src, scale = _block_rowmap("S", k)
            moi = [x[src[t]] * scale[t] for t in 1:k]
            back = _to_vecm!(zeros(k), 1:k, moi, "S")
            @test back ≈ x
        end
        @test_throws ArgumentError psd_moi_vecm_info(4)
    end

    @testset "MOI round trip on an R+Q+S problem with equalities" begin
        prob = mixed_cone_problem()
        direct = ConicIP.conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                                 prob.G, prob.d; verbose = false, optTol = 1e-9)
        @test direct.status == :Optimal

        opt = bridged_cached(ConicIP.Optimizer)
        MOI.set(opt, MOI.Silent(), true)
        MOI.set(opt, MOI.RawOptimizerAttribute("optTol"), 1e-9)
        maps = build_moi_model(opt, prob)
        @test length(maps.x) == 12
        @test maps.ci_eq !== nothing
        @test [blk.kind for blk in maps.blocks] == ["R", "Q", "S"]
        @test MOI.get(opt, MOI.ObjectiveFunctionType()) == MOI.ScalarQuadraticFunction{Float64}
        MOI.optimize!(opt)
        rec = recover_solution(opt, maps, prob)
        @test rec.status == :Optimal
        @test rec.iters >= 1
        @test rec.time >= 0
        res = residuals(prob, rec)
        @test max(res.rDu, res.rPr, res.rEq, res.gap) <= 1e-6
        @test res.margin >= -1e-6 && res.dual_margin >= -1e-6
        @test verified(rec, res)
        @test norm(rec.y - direct.y) <= 1e-5
        # The slack read back from MOI is the solver's A y − b in vecm order.
        @test norm(rec.s - (prob.A * rec.y - prob.b)) <= 1e-6
        # ConicIP's own wrapper agrees row by row (same data, same options),
        # which pins the sign and √2 conventions of the dual recovery.
        raw = MOI.get(opt, MOI.RawSolver())
        @test norm(rec.v - raw.sol.v) <= 1e-8 * (1 + norm(raw.sol.v))
        @test norm(rec.w - raw.sol.w) <= 1e-8 * (1 + norm(raw.sol.w))
    end

    @testset "Affine objective and no equalities" begin
        prob = lp_band(30)
        opt = bridged_cached(ConicIP.Optimizer)
        MOI.set(opt, MOI.Silent(), true)
        maps = build_moi_model(opt, prob)
        @test maps.ci_eq === nothing
        @test MOI.get(opt, MOI.ObjectiveFunctionType()) == MOI.ScalarAffineFunction{Float64}
        MOI.optimize!(opt)
        rec = recover_solution(opt, maps, prob)
        @test verified(rec, residuals(prob, rec))
        @test isempty(rec.w)
    end

    @testset "Dual-infeasible model recovers without throwing" begin
        # min −y  s.t.  y ≥ 0  is unbounded below.
        prob = (Q = spzeros(1, 1), c = [1.0], A = sparse(ones(1, 1)), b = [0.0],
                cone_dims = [("R", 1)], G = spzeros(0, 1), d = zeros(0))
        opt = bridged_cached(ConicIP.Optimizer)
        MOI.set(opt, MOI.Silent(), true)
        maps = build_moi_model(opt, prob)
        MOI.optimize!(opt)
        rec = recover_solution(opt, maps, prob)
        @test rec.status == :DUAL_INFEASIBLE
        @test length(rec.y) == 1 && length(rec.v) == 1 && isempty(rec.w)
        @test !verified(rec, residuals(prob, rec))
    end

    @testset "--opt parsing and passthrough" begin
        opts = parse_opts("maxIters=3,optTol=1e-8,equilibrate=false,kktsolver=ldl")
        @test opts == Dict(:maxIters => 3, :optTol => 1e-8, :equilibrate => false,
                           :kktsolver => "ldl")
        @test opts[:maxIters] isa Int && opts[:optTol] isa Float64
        @test opts[:equilibrate] isa Bool && opts[:kktsolver] isa String
        @test parse_opts("") == Dict{Symbol, Any}()
        @test parse_opts(" verbose = true ") == Dict(:verbose => true)
        @test_throws ArgumentError parse_opts("maxIters")
        # negative numbers, floats without an integer form, and values that
        # themselves contain "=" (split once); empty keys or values are errors
        @test parse_opts("staticReg=-1e-8,maxIters=-3,x=a=b,timeLimit=Inf") ==
              Dict(:staticReg => -1e-8, :maxIters => -3, :x => "a=b", :timeLimit => Inf)
        @test_throws ArgumentError parse_opts("=3")
        @test_throws ArgumentError parse_opts("a=")

        prob = lp_band(40)
        @test kkt_solver_name(prob) in ("kktsolver_qr", "kktsolver_ldl")
        try
            set_opts!("maxIters=3,kktsolver=ldl,preprocess=false")
            kw = direct_kwargs()
            @test kw[:kktsolver] === ConicIP.kktsolver_ldl
            @test kw[:maxIters] == 3 && !haskey(kw, :preprocess)
            @test kkt_solver_name(prob) == "ldl"
            sol = solve_direct(prob)
            @test sol.Iter <= 3
            @test sol.status != :Optimal
        finally
            set_opts!("")
        end
        @test isempty(SOLVER_OPTS) && isempty(OPT_STRING[])
        @test solve_direct(prob).status == :Optimal
    end

    @testset "Non-optimal statuses recover without throwing" begin
        function rec(prob; kw...)
            opt = bridged_cached(ConicIP.Optimizer)
            MOI.set(opt, MOI.Silent(), true)
            for (k, v) in kw
                MOI.set(opt, MOI.RawOptimizerAttribute(string(k)), v)
            end
            maps = build_moi_model(opt, prob)
            MOI.optimize!(opt)
            r = recover_solution(opt, maps, prob)
            @test !verified(r, residuals(prob, r))
            return r
        end
        # primal infeasible (certificate result: v is the ray, y is NaN)
        r = rec((Q = spzeros(1, 1), c = [0.0], A = sparse([1.0; -1.0][:, :]), b = [1.0, 1.0],
                 cone_dims = [("R", 2)], G = spzeros(0, 1), d = zeros(0)))
        @test r.status == :INFEASIBLE
        @test all(isnan, r.y) && all(isfinite, r.v) && isempty(r.w)
        # iteration limit and time limit: statuses are the MOI names, no throw
        r = rec(lp_band(200); maxIters = 2, certFallback = false)
        @test r.status == :ITERATION_LIMIT && length(r.y) == 200
        r = rec(lp_band(200); timeLimit = 0.0)
        @test r.status == :TIME_LIMIT
    end

    @testset "Subprocess timeout" begin
        # A child that never finishes is killed after `timeout` seconds and
        # reported as TIMEOUT, with whatever it printed before.
        script = tempname() * ".jl"
        write(script, "println(\"status=RUNNING\"); flush(stdout); sleep(120)\n")
        inst = direct("sleepy", "test", () -> nothing)
        t = @elapsed row = run_subprocess(inst; timeout = 5.0, script = script)
        @test row["status"] == "TIMEOUT"
        @test t < 60
        # A child that exits normally is read back as it printed
        write(script, "println(\"status=DONE\"); println(\"iters=3\")\n")
        row = run_subprocess(inst; timeout = 60.0, script = script)
        @test row["status"] == "DONE" && row["iters"] == "3"
        rm(script)
    end

    @testset "CSV writer and result row shape" begin
        prob = lp_band(30)
        inst = direct("lp-band-30", "lp-band", () -> prob)
        row = run_direct(inst)
        for col in COLUMNS
            col in ("name", "family", "maxrss_mb") && continue
            @test haskey(row, col)
        end
        @test row["status"] == "Optimal"
        @test row["kktsolver"] in ("kktsolver_qr", "kktsolver_ldl")
        row["name"] = inst.name; row["family"] = inst.family
        path = write_csv(tempname(), [row]; header = "test")
        lines = readlines(path)
        @test lines[1] == "# test"
        @test split(lines[2], ',') == COLUMNS
        @test length(split(lines[3], ','; keepempty = true)) == length(COLUMNS)
        rm(path)
    end

    @testset "assemble_only reproduces the direct tuple" begin
        # baselines.jl converts file instances this way: the MOI model built
        # from a tuple, assembled by ConicIP.Optimizer without solving, must
        # give the tuple back (vecm row order and √2 scaling included).
        if "assemble_only" in ConicIP._SUPPORTED_OPTIONS
            prob = mixed_cone_problem()
            opt = bridged_cached(ConicIP.Optimizer)
            MOI.set(opt, MOI.Silent(), true)
            MOI.set(opt, MOI.RawOptimizerAttribute("assemble_only"), true)
            build_moi_model(opt, prob)
            MOI.optimize!(opt)
            raw = MOI.get(opt, MOI.RawSolver())
            @test raw isa ConicIP.Optimizer
            @test raw.n == length(prob.c)
            @test raw.cone_dims == prob.cone_dims
            @test raw.c_int ≈ prob.c
            @test raw.Q_int !== nothing && norm(raw.Q_int - prob.Q) <= 1e-12 * norm(prob.Q)
            @test norm(raw.ineq_A - prob.A) <= 1e-12 * norm(prob.A)
            @test norm(raw.ineq_b - prob.b) <= 1e-12 * norm(prob.b)
            @test norm(raw.eq_G - prob.G) <= 1e-12 * norm(prob.G)
            @test norm(raw.eq_d - prob.d) <= 1e-12 * norm(prob.d)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
        else
            @test_broken false   # assemble_only not available in this ConicIP
        end
    end
end
end
