# Solver log: the classic table is pinned by golden skeletons (numbers
# masked, see log_skeleton) in test/golden/. To regenerate them after an
# intended change, run the tests with CONICIP_UPDATE_GOLDEN=1 and review
# the diff.

include("logging_cases.jl")

import MathOptInterface as MOI

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
