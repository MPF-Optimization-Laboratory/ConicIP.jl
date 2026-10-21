# Solver log: the classic table is pinned by golden skeletons (numbers
# masked, see log_skeleton) in test/golden/. To regenerate them after an
# intended change, run the tests with CONICIP_UPDATE_GOLDEN=1 and review
# the diff.

include("logging_cases.jl")

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
