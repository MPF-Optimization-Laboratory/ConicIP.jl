# Performance regression checks
# ==============================
# Deterministic iteration-count and allocation bounds for representative
# problems. NOT part of the CI gate (wall time is too noisy there); run
# manually after performance-sensitive changes:
#
#   julia --project benchmark/regressions.jl
#
# Bounds are deliberately loose (~2× the measured value at the time the
# entry was added) so only genuine regressions trip them. Iterations are
# deterministic (seeded generators); allocations are stable per
# Julia/dependency version.

using ConicIP, SparseArrays, LinearAlgebra, Printf

include(joinpath(@__DIR__, "..", "test", "testdata.jl"))

# (name, problem thunk, solver, max_iters, max_alloc_MiB)
CHECKS = [
    ("issue10 k=150 auto",
     () -> socp_sum_of_norms(150; d = 200), nothing, 20, 100.0),
    ("issue10 k=400 sparse",
     () -> socp_sum_of_norms(400; d = 800), ConicIP.kktsolver_sparse, 25, 200.0),
    ("issue10 k=400 2x2",
     () -> socp_sum_of_norms(400; d = 800), pivot(ConicIP.kktsolver_2x2), 25, 200.0),
]

function run_checks()
    fails = 0
    for (name, thunk, solver, max_iters, max_mib) in CHECKS
        p = thunk()
        kw = solver === nothing ? (;) : (; kktsolver = solver)
        # warmup (compilation)
        conicIP(p.Q, p.c, p.A, p.b, p.cone_dims, p.G, p.d;
                verbose = false, maxIters = 3, kw...)
        stats = @timed conicIP(p.Q, p.c, p.A, p.b, p.cone_dims, p.G, p.d;
                               verbose = false, kw...)
        sol = stats.value
        mib = stats.bytes / 2^20
        ok = sol.status == :Optimal && sol.Iter <= max_iters && mib <= max_mib
        fails += !ok
        @printf("%-24s  %-9s  iters %2d (≤%2d)  alloc %7.1f MiB (≤%7.1f)  %s\n",
                name, sol.status, sol.Iter, max_iters, mib, max_mib,
                ok ? "OK" : "REGRESSION")
    end
    return fails
end

if abspath(PROGRAM_FILE) == @__FILE__
    exit(run_checks() == 0 ? 0 : 1)
end
