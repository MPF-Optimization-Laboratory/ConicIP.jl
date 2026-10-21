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

end
