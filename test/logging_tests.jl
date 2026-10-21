# Solver log: the classic table is pinned by golden skeletons (numbers
# masked, see log_skeleton) in test/golden/. To regenerate them after an
# intended change, run the tests with CONICIP_UPDATE_GOLDEN=1 and review
# the diff.

include("logging_cases.jl")

import MathOptInterface as MOI
import JSON
using Logging

@testset "Solver log" begin

    @testset "classic golden skeletons" begin
        update = get(ENV, "CONICIP_UPDATE_GOLDEN", "") == "1"
        for (name, f) in LOG_CASES
            path = joinpath(@__DIR__, "golden", name * ".txt")
            got = log_skeleton(capture_log(f))
            if update
                write(path, got)
            end
            @test isfile(path)
            isfile(path) || continue
            want = read(path, String)
            got == want || @info "Log skeleton mismatch for $name" want got
            @test got == want
        end
    end

    @testset "style resolution" begin
        withenv("CONICIP_LOG_STYLE" => nothing) do
            @test ConicIP._resolve_style(:auto, IOBuffer()) === :classic
            @test ConicIP._resolve_style(:auto, IOContext(IOBuffer(), :color => true)) === :classic
            @test ConicIP._resolve_style("rich", IOBuffer()) === :rich
            @test ConicIP._resolve_style(:classic, IOBuffer()) === :classic
        end
        withenv("CONICIP_LOG_STYLE" => "rich") do
            @test ConicIP._resolve_style(:auto, IOBuffer()) === :rich
            @test ConicIP._resolve_style(:classic, IOBuffer()) === :classic   # keyword wins
        end
        withenv("CONICIP_LOG_STYLE" => "sparkly") do                          # ignored
            @test ConicIP._resolve_style(:auto, IOBuffer()) === :classic
        end
        @test_throws ArgumentError ConicIP._check_log_style(:fancy)
        @test_throws ArgumentError ConicIP._check_log_style(1)
        # checked even when nothing is printed
        @test_throws ArgumentError conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K;
                                           verbose = false, log_style = :fancy)
        @test_throws ArgumentError ConicIP.Optimizer(log_style = "fancy")
    end

    rich(f; color = false, width = 140, kw...) = begin
        buf = IOBuffer()
        f(; log_style = :rich, log_io = IOContext(buf, :color => color,
                                                  :displaysize => (40, width)), kw...)
        String(take!(buf))
    end
    case(name) = LOG_CASES[findfirst(c -> c.first == name, LOG_CASES)].second

    @testset "rich style" begin
        out = rich(case("qp_optimal"))
        @test !occursin("\e[", out)
        @test occursin("✓ OPTIMAL", out)
        for label in ("problem", "kkt", "stop", "objective", "residuals", "path")
            @test occursin(Regex("^ +$label ", "m"), out)
        end
        @test occursin("ℝ₊ 3", out)
        @test occursin(r"^ +iter +primal +dual +compl +gap +pobj +dobj +step +ref +time +progress"m, out)
        @test !occursin(r"\bkkt +time"m, out) && !occursin(r"\bcc\b", out)
        @test occursin("██████████ ✓", out)                 # the last row reaches optTol
        # optional columns
        @test occursin(r"ref +kkt +time"m, rich(case("qp_ldl")))
        @test occursin(r"ref +cc +time"m, rich(case("qp_correctors")))
        # a narrow terminal drops time and progress
        narrow = rich(case("qp_optimal"); width = 80)
        @test !occursin("progress", narrow)
        @test all(l -> textwidth(l) <= 80, filter(l -> occursin(r"^ +\d+  ", l), split(narrow, '\n')))
        # colour only when the io asks for it, and never under NO_COLOR
        @test occursin("\e[", rich(case("qp_optimal"); color = true))
        withenv("NO_COLOR" => "1") do
            @test !occursin("\e[", rich(case("qp_optimal"); color = true))
        end
        # every exit has a card
        titles = Dict("qp_abandoned" => "⚠ ABANDONED", "infeasible" => "✗ INFEASIBLE",
                      "dual_infeasible" => "✗ DUAL INFEASIBLE", "kkt_error" => "✗ ERROR",
                      "struct_infeasible" => "✗ INFEASIBLE", "preprocess" => "✓ OPTIMAL",
                      "preprocess_infeasible" => "✗ INFEASIBLE", "soc" => "✓ OPTIMAL",
                      "sdp" => "✓ OPTIMAL")
        for (name, title) in titles
            out = rich(case(name))
            @test occursin(title, out)
            @test !occursin(" > ", out)                     # nothing from the classic table
        end
        @test occursin("› presolve", rich(case("preprocess")))
        @test occursin(r"p \d\.\de-\d+ ⇢", rich(case("infeasible")))   # the screen nominated a ray
        # the final card reports a postsolve downgrade
        buf = IOBuffer()
        log = ConicIP.SolverLog(true, true, :rich, buf, 0)
        ConicIP._log_exit!(log, :optimal)
        sol = conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K; verbose = false)
        sol.status = :Error; sol.message = "postsolve check failed"
        ConicIP._log_final!(log, sol)
        out = String(take!(buf))
        @test occursin("OPTIMAL → ERROR", out) && occursin("postsolve check failed", out)
    end

    @testset "sparklines" begin
        sp = ConicIP._sparkline([1.0, 1e-2, 1e-4, 1e-6])
        @test sp == "█▆▃▁"
        @test length(ConicIP._sparkline(rand(200) .+ 0.1; width = 40)) == 40
        @test ConicIP._sparkline([1e-17, 3e-17, 1e-8]; floor = 1e-8) == "▁▁▁"
        @test ConicIP._sparkline([1.0, NaN, 0.5]; scale = :lin) == "█·▅"
        @test ConicIP._sparkline(Float64[]) == ""
        @test ConicIP._bar(1.0, 10) == "█"^10
        @test ConicIP._bar(0.0, 10) == ""
        @test ConicIP._bar(0.5 + 1/80, 10) == "█████▏"
    end

    @testset "MOI options" begin
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(src, 2)
        MOI.add_constraint(src, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
                           MOI.EqualTo(1.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], x), 0.0))
        buf = IOBuffer()
        opt = ConicIP.Optimizer(verbose = true, log_style = "rich", log_io = buf)
        @test MOI.get(opt, MOI.RawOptimizerAttribute("log_style")) == "rich"
        MOI.optimize!(opt, src)
        @test occursin("✓ OPTIMAL", String(take!(buf)))
        MOI.set(opt, MOI.Silent(), true)
        MOI.optimize!(opt, src)
        @test isempty(take!(buf))
        @test_throws ArgumentError MOI.set(opt, MOI.RawOptimizerAttribute("log_io"), "stdout")
    end

    @testset "trace" begin
        P = LOG_QP
        tr = SolverTrace()
        # unequilibrated, so that the returned residuals are the logged ones
        # (postsolve recomputes them on the original data otherwise)
        sol = conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false, trace = tr, equilibrate = false,
                      kktsolver = ConicIP.kktsolver_ldl, centralityCorrectors = 1)
        kinds = [e.event for e in tr.events]
        @test kinds[1] === :header
        @test count(==(:header), kinds) == 1 && count(==(:final), kinds) == 1
        @test kinds[end] === :final && kinds[end-1] === :exit
        h = tr.events[findfirst(==(:header), kinds)]
        @test h.schema == ConicIP.TRACE_SCHEMA
        @test (h.n, h.m, h.p) == (3, 3, 0) && h.kktsolver == "kktsolver_ldl" && !h.routed
        its = ConicIP.iterations(tr)
        @test [r.iter for r in its] == 1:length(its)
        @test length(its) == sol.Iter                   # optimal: the last iterate is returned
        last = its[end]
        @test last.best && last.optMeasure < 1e-6
        @test sol.prFeas == max(last.rPr, last.rEq)
        @test sol.duFeas == last.rDu && sol.rGap == last.rGap
        @test isnan(its[1].alpha) && all(r -> 0 < r.alpha <= 1, its[2:end])
        @test all(r -> r.kkt_repaired >= 0, its)        # kktsolver_ldl reports diagnostics
        @test all(r -> 0 <= r.cc_acc <= r.cc_try <= 1, its[2:end])
        @test issorted([r.kkt_solves for r in its])
        @test its[end].kkt_solves <= sol.kkt_solves
        fin = tr.events[end]
        @test fin.status === sol.status && fin.changed_from === nothing && fin.phases === nothing
        @test all(e -> e.depth == 0 && e.t >= 0, tr.events)
        # off: correctors and diagnostics read -1
        tr2 = SolverTrace()
        conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false, trace = tr2,
                kktsolver = ConicIP.kktsolver_qr)
        @test all(r -> r.cc_try == -1 && r.kkt_repaired == -1, ConicIP.iterations(tr2))
        # a callable sink, and timing adds phases to the final record
        recs = Any[]
        conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false, trace = r -> push!(recs, r),
                timing = ConicIP.PhaseTimes())
        @test recs[end].phases.t_loop > 0
        # the trace does not print
        @test capture_log(() -> conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false,
                                        trace = SolverTrace())) == ""
        # presolve notes and its final verdict, through the owner
        tr3 = SolverTrace()
        LOG_CASES[findfirst(c -> c.first == "preprocess", LOG_CASES)].second(;
            verbose = false, trace = tr3)
        @test any(e -> e.event === :note && e.source === :presolve, tr3.events)
        @test count(e -> e.event === :final, tr3.events) == 1
    end

    @testset "JSON Lines" begin
        P = LOG_QP
        tr = SolverTrace()
        conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false, trace = tr)
        mktempdir() do dir
            path = joinpath(dir, "run.jsonl")
            write_jsonl(path, tr)
            lines = readlines(path)
            @test length(lines) == length(tr.events)
            objs = JSON.parse.(lines)
            @test objs[findfirst(o -> o["event"] == "header", objs)]["schema"] == "conicip-trace/1"
            it1 = objs[findfirst(o -> o["event"] == "iter", objs)]
            @test it1["alpha"] === nothing                  # NaN is written as null
            @test it1["iter"] == 1
            # trace = path appends, one header and one final per solve
            path2 = joinpath(dir, "append.jsonl")
            for _ in 1:2
                conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false, trace = path2)
            end
            objs2 = JSON.parse.(readlines(path2))
            @test count(o -> o["event"] == "header", objs2) == 2
            @test count(o -> o["event"] == "final", objs2) == 2
            # CONICIP_TRACE, and nothing from the fallback's auxiliary solves
            # (this infeasible box runs the fallback)
            path3 = joinpath(dir, "env.jsonl")
            withenv("CONICIP_TRACE" => path3) do
                conicIP(zeros(1, 1), [0.0], sparse(reshape([1.0, -1.0], 2, 1)), [1.0, 0.0],
                        [("R", 2)]; verbose = false, maxIters = 3)
            end
            objs3 = JSON.parse.(readlines(path3))
            @test count(o -> o["event"] == "header", objs3) == 1
            @test all(o -> o["depth"] == 0, objs3)
        end
        # escaping
        io = IOBuffer()
        ConicIP._write_json(io, (; msg = "a \"quoted\"\n\tline \\ \x01", x = Inf, v = [1, NaN]))
        s = String(take!(io))
        @test JSON.parse(s)["msg"] == "a \"quoted\"\n\tline \\ \x01"
        @test JSON.parse(s)["x"] === nothing
        @test occursin("\"v\":[1.0,null]", s)
    end

    @testset "MOI trace" begin
        src = MOI.Utilities.Model{Float64}()
        x = MOI.add_variables(src, 2)
        MOI.add_constraint(src, MOI.VectorOfVariables(x), MOI.Nonnegatives(2))
        MOI.add_constraint(src, MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 1.0], x), 0.0),
                           MOI.EqualTo(1.0))
        MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
        MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.([1.0, 2.0], x), 0.0))
        tr = SolverTrace()
        opt = ConicIP.Optimizer(trace = tr)
        MOI.set(opt, MOI.Silent(), true)               # silences printing, not the trace
        MOI.optimize!(opt, src)
        @test !isempty(ConicIP.iterations(tr))
        @test tr.events[end].event === :final && tr.events[end].status === :Optimal
    end

    @testset "debug diagnostics" begin
        P = LOG_QP
        dbg(f) = @test_logs (:debug, r"KKT routing") match_mode = :any min_level = Logging.Debug f()
        dbg(() -> conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false))
        @test_logs (:debug, r"Farkas ray nominated") match_mode = :any min_level = Logging.Debug conicIP(
            zeros(1, 1), [0.0], sparse(reshape([1.0, -1.0], 2, 1)), [1.0, 0.0], [("R", 2)];
            verbose = false)
        # An LP with Q = 0, no static shift and the natural ordering: LDLᵀ
        # repairs zero pivots, and with retry on bumps the shifts (see the
        # deflation test in tranche3_tests.jl)
        n = 6
        ksr = (Q, A, G, cd) -> ConicIP.kktsolver_ldl(Q, A, G, cd; static_reg = 0.0,
                  dynamic_delta = 1e-2, retry_max = 2,
                  pattern = ConicIP._ldl_pattern(Q, A, G, cd;
                              perm_hint = collect(1:(size(Q, 1) + size(A, 1) + size(G, 1)))))
        lp() = conicIP(spzeros(n, n), ones(n), sparse(1.0I, n, n), zeros(n), [("R", n)];
                       verbose = false, kktsolver = ksr, equilibrate = false)
        @test_logs (:debug, r"dynamic regularization") match_mode = :any min_level = Logging.Debug lp()
        @test_logs (:debug, r"static shifts bumped") match_mode = :any min_level = Logging.Debug lp()
        # nothing at Info and above: the solve path never warns on its own
        @test_logs conicIP(P.Q, P.c, P.A, P.b, P.K; verbose = false)
    end

    @testset "final verdict line" begin
        # The owner adds a line only when postsolve changed the verdict the
        # solver announced, or when no solve announced one.
        sol = conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K; verbose = false)
        final(exit_kind) = begin
            io = IOBuffer()
            log = ConicIP.SolverLog(true, true, :classic, io, 0)
            exit_kind === nothing || ConicIP._log_exit!(log, exit_kind)
            truncate(io, 0)
            ConicIP._log_final!(log, sol)
            String(take!(io))
        end
        @test final(:optimal) == ""
        @test occursin("POSTSOLVE -- status changed: Infeasible → Optimal", final(:infeasible))
        @test occursin("EXIT -- Optimal", final(nothing))
    end

end
