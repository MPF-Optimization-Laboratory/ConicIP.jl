# Tests for the benchmark harness pieces that are MOI-only (model builder,
# solution recovery, option parsing). Wrapped in a module because
# review_benchmark.jl already includes benchmark/suite.jl in its own module.
module HarnessTests
using Test
@testset "Benchmark harness" begin
end
end
