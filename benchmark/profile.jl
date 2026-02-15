#!/usr/bin/env julia
#
# ConicIP.jl Performance Profiling Script
# ========================================
# Produces timing, allocation, profiling, and type-stability data
# for a representative set of conic optimization problems.

using ConicIP
using LinearAlgebra
using SparseArrays
using Random
using Profile
using Printf
using InteractiveUtils: @code_warntype

# ══════════════════════════════════════════════════════════════════
#  1A. Problem Generators
# ══════════════════════════════════════════════════════════════════

function prob_box_qp_dense(; n=500)
    Random.seed!(42)
    M = randn(n, n); Q = M'M / n
    c = randn(n, 1)
    A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
    b = [-ones(n, 1); -ones(n, 1)]
    cone_dims = [("R", 2n)]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Box QP dense Q (n=$n)")
end

function prob_box_qp_sparse(; n=1000)
    Random.seed!(42)
    Q = spdiagm(0 => 1.0 .+ rand(n))
    c = randn(n, 1)
    A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
    b = [-ones(n, 1); -ones(n, 1)]
    cone_dims = [("R", 2n)]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Box QP sparse Q (n=$n)")
end

function prob_single_soc(; n=500)
    Random.seed!(42)
    Q = sparse(1.0I, n, n)
    c = randn(n, 1)
    A = [spzeros(1, n); sparse(1.0I, n, n)]
    b = [-1.0; zeros(n, 1)]
    cone_dims = [("Q", n + 1)]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Single large SOC (n=$n)")
end

function prob_many_small_socs(; n=500, k=250)
    Random.seed!(42)
    # Each SOC has dimension 3: (t, x1, x2), so k cones need 3k rows
    # We have n variables, m = n + k constraint rows
    m_soc = 3k  # 3 rows per SOC cone
    Q = sparse(1.0I, n, n)
    c = randn(n, 1)
    A = sprandn(m_soc, n, 0.1)
    # Make first row of each SOC block have a negative b for feasibility
    b = zeros(m_soc, 1)
    for i in 1:k
        b[3(i-1)+1] = -1.0
    end
    cone_dims = [("Q", 3) for _ in 1:k]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Many small SOCs (k=$k, n=$n)")
end

function prob_small_sdp(; k=10)
    Random.seed!(42)
    n = div(k * (k + 1), 2)  # vectorized size
    Q = sparse(1.0I, n, n)
    c = reshape(ConicIP.vecm(Diagonal(ones(k))), :, 1)
    A = sparse(1.0I, n, n)
    b = zeros(n, 1)
    cone_dims = [("S", n)]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Small SDP (k=$k)")
end

function prob_larger_sdp(; k=30)
    Random.seed!(42)
    n = div(k * (k + 1), 2)
    Q = sparse(1.0I, n, n)
    c = reshape(ConicIP.vecm(Diagonal(ones(k))), :, 1)
    A = sparse(1.0I, n, n)
    b = zeros(n, 1)
    cone_dims = [("S", n)]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Larger SDP (k=$k)")
end

function prob_mixed_rq_eq(; n=200)
    Random.seed!(42)
    Q = sparse(1.0I, n, n)
    c = randn(n, 1)
    # R+ cone for first n vars, SOC of dim 51
    n_r = n
    n_q = 51
    A_r = sparse(1.0I, n, n)
    A_q = sprandn(n_q, n, 0.2)
    A_q[1, :] .= 0  # first row of SOC is the bound
    A = [A_r; A_q]
    b_r = zeros(n_r, 1)
    b_q = [-1.0; zeros(n_q - 1, 1)]
    b = [b_r; b_q]
    cone_dims = [("R", n_r), ("Q", n_q)]
    p = 10
    G = randn(p, n)
    d = G * ones(n, 1)  # feasible point y = ones(n)
    (; Q, c, A, b, cone_dims, G, d, name="Mixed R+Q + equalities (n=$n, p=$p)")
end

function prob_mixed_rqs(; n=86)
    Random.seed!(42)
    # R+ for first 50, Q for next 21, S(k=5) for last 15
    n_r = 50; n_q = 21; k_s = 5; n_s = div(k_s*(k_s+1), 2)  # 15
    m = n_r + n_q + n_s  # 86
    Q = sparse(1.0I, n, n)
    c = randn(n, 1)
    A = sparse(1.0I, m, n)
    b_r = zeros(n_r, 1)
    b_q = [-1.0; zeros(n_q - 1, 1)]
    b_s = reshape(ConicIP.vecm(Matrix{Float64}(I, k_s, k_s)), :, 1) .* 0  # zeros
    b = [b_r; b_q; b_s]
    cone_dims = [("R", n_r), ("Q", n_q), ("S", n_s)]
    G = spzeros(0, n); d = zeros(0, 1)
    (; Q, c, A, b, cone_dims, G, d, name="Mixed R+Q+S (n=$n)")
end

# ══════════════════════════════════════════════════════════════════
#  1B. Macro Benchmarks
# ══════════════════════════════════════════════════════════════════

function run_macro_benchmarks()
    problems = [
        prob_box_qp_dense,
        prob_box_qp_sparse,
        prob_single_soc,
        prob_many_small_socs,
        prob_small_sdp,
        prob_larger_sdp,
        prob_mixed_rq_eq,
        prob_mixed_rqs,
    ]

    solvers = [
        ("kktsolver_qr",     ConicIP.kktsolver_qr),
        ("kktsolver_sparse", ConicIP.kktsolver_sparse),
        ("pivot(2x2)",       pivot(ConicIP.kktsolver_2x2)),
    ]

    println("="^100)
    println("  MACRO BENCHMARKS")
    println("="^100)

    results = []

    for prob_fn in problems
        prob = prob_fn()
        println("\n--- $(prob.name) ---")
        println("  n=$(size(prob.Q,1)), m=$(size(prob.A,1)), p=$(size(prob.G,1)), cones=$(prob.cone_dims)")

        for (solver_name, solver) in solvers
            # Warmup
            try
                conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                        prob.G, prob.d,
                        kktsolver=solver, verbose=false, maxIters=3)
            catch e
                println("  [$solver_name] warmup failed: $e")
                continue
            end

            # Timed runs
            times = Float64[]
            allocs = Int[]
            gc_times = Float64[]
            status = :None
            iters = 0

            run_ok = true
            for trial in 1:3
                GC.gc()
                try
                    stats = @timed conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                                           prob.G, prob.d,
                                           kktsolver=solver, verbose=false, maxIters=100)
                    push!(times, stats.time)
                    push!(allocs, stats.bytes)
                    push!(gc_times, stats.gctime)
                    status = stats.value.status
                    iters = stats.value.Iter
                catch e
                    println("  [$solver_name] trial $trial failed: $e")
                    run_ok = false
                    break
                end
            end

            if !run_ok || isempty(times); continue; end

            med_idx = sortperm(times)[2]  # median of 3
            t = times[med_idx]
            a = allocs[med_idx]
            gc = gc_times[med_idx]
            gc_pct = t > 0 ? 100 * gc / t : 0.0

            push!(results, (; problem=prob.name, solver=solver_name,
                             time_s=t, alloc_mb=a/1e6, gc_pct, iters, status))

            @printf("  [%-18s] %8.4fs  %8.1f MB  GC=%4.1f%%  iters=%3d  status=%s\n",
                    solver_name, t, a/1e6, gc_pct, iters, status)
        end
    end

    return results
end

# ══════════════════════════════════════════════════════════════════
#  1C. Statistical Profiling
# ══════════════════════════════════════════════════════════════════

function run_profiling()
    println("\n" * "="^100)
    println("  STATISTICAL PROFILING (flat, top 30 by count)")
    println("="^100)

    # Representative problems: one R, one Q, one S, one mixed
    profile_problems = [
        ("Box QP sparse",  prob_box_qp_sparse),
        ("Single SOC",     prob_single_soc),
        ("Small SDP",      prob_small_sdp),
        ("Mixed R+Q+S",    prob_mixed_rqs),
    ]

    for (label, prob_fn) in profile_problems
        prob = prob_fn()
        println("\n--- Profile: $label ---")

        # Warmup
        conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                prob.G, prob.d, verbose=false, maxIters=3)

        Profile.clear()
        @profile conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                         prob.G, prob.d, verbose=false, maxIters=100)
        Profile.print(format=:flat, sortedby=:count, mincount=5, maxdepth=80)
    end
end

# ══════════════════════════════════════════════════════════════════
#  1D. Allocation Tracking
# ══════════════════════════════════════════════════════════════════

function run_allocation_tracking()
    println("\n" * "="^100)
    println("  ALLOCATION TRACKING")
    println("="^100)

    Random.seed!(42)

    # --- Block * Matrix for different block types ---
    println("\n--- Block * Matrix allocations ---")

    # Diagonal block (R+ cone)
    n_r = 500
    F_diag = Block([Diagonal(1.0 .+ rand(n_r))])
    x_r = randn(n_r, 1)
    F_diag * x_r  # warmup
    a = @allocated F_diag * x_r
    println("  Diagonal Block($n_r) * Matrix:  $a bytes")

    # SymWoodbury block (SOC cone)
    n_q = 500
    F_sw = Block([ConicIP.nestod_soc(
        [1.0; zeros(n_q-1, 1)] + 0.01*randn(n_q, 1),
        [1.0; zeros(n_q-1, 1)] + 0.01*randn(n_q, 1)
    )])
    x_q = randn(n_q, 1)
    F_sw * x_q  # warmup
    a = @allocated F_sw * x_q
    println("  SymWoodbury Block($n_q) * Matrix:  $a bytes")

    # VecCongurance block (SDP cone)
    k_s = 10
    n_s = div(k_s*(k_s+1), 2)
    R = randn(k_s, k_s) + k_s * I
    F_vc = Block([ConicIP.VecCongurance(R)])
    x_s = randn(n_s, 1)
    F_vc * x_s  # warmup
    a = @allocated F_vc * x_s
    println("  VecCongurance Block($n_s) * Matrix:  $a bytes")

    # --- inv(Block) and inv(Block)' ---
    println("\n--- inv(Block) allocations ---")

    inv(F_diag)  # warmup
    a = @allocated inv(F_diag)
    println("  inv(Diagonal Block):  $a bytes")

    inv(F_sw)
    a = @allocated inv(F_sw)
    println("  inv(SymWoodbury Block):  $a bytes")

    inv(F_vc)
    a = @allocated inv(F_vc)
    println("  inv(VecCongurance Block):  $a bytes")

    println("\n--- inv(Block)' allocations ---")
    inv(F_diag)'
    a = @allocated inv(F_diag)'
    println("  inv(Diagonal Block)':  $a bytes")

    inv(F_sw)'
    a = @allocated inv(F_sw)'
    println("  inv(SymWoodbury Block)':  $a bytes")

    inv(F_vc)'
    a = @allocated inv(F_vc)'
    println("  inv(VecCongurance Block)':  $a bytes")

    # --- sparse(Block) ---
    println("\n--- sparse(Block) allocations ---")

    sparse(F_diag)
    a = @allocated sparse(F_diag)
    println("  sparse(Diagonal Block):  $a bytes")

    sparse(F_sw)
    a = @allocated sparse(F_sw)
    println("  sparse(SymWoodbury Block):  $a bytes")

    sparse(F_vc)
    a = @allocated sparse(F_vc)
    println("  sparse(VecCongurance Block):  $a bytes")

    # --- Per-cone operations ---
    println("\n--- Per-cone arithmetic (÷ and ∘ primitives) ---")

    # drp! / xrp!
    n = 500
    x = rand(n) .+ 0.1; y = rand(n) .+ 0.1; o = zeros(n)
    ConicIP.drp!(x, y, o)
    a = @allocated ConicIP.drp!(x, y, o)
    println("  drp!($n):  $a bytes")

    ConicIP.xrp!(x, y, o)
    a = @allocated ConicIP.xrp!(x, y, o)
    println("  xrp!($n):  $a bytes")

    # dsoc! / xsoc!
    n = 500
    x = [1.0; 0.01*randn(n-1)]; y = [1.0; 0.01*randn(n-1)]; o = zeros(n)
    ConicIP.dsoc!(x, y, o)
    a = @allocated ConicIP.dsoc!(x, y, o)
    println("  dsoc!($n):  $a bytes")

    ConicIP.xsoc!(x, y, o)
    a = @allocated ConicIP.xsoc!(x, y, o)
    println("  xsoc!($n):  $a bytes")

    # dsdc! / xsdc!
    k = 10; n = div(k*(k+1), 2)
    X = ConicIP.vecm(Diagonal(ones(k)) + 0.1*randn(k,k))
    Y = ConicIP.vecm(Diagonal(ones(k)) + 0.1*randn(k,k))
    o = zeros(n)
    ConicIP.dsdc!(X, Y, o)
    a = @allocated ConicIP.dsdc!(X, Y, o)
    println("  dsdc!($n):  $a bytes")

    ConicIP.xsdc!(X, Y, o)
    a = @allocated ConicIP.xsdc!(X, Y, o)
    println("  xsdc!($n):  $a bytes")

    # --- mat / vecm ---
    println("\n--- mat/vecm allocations ---")

    k = 30; n = div(k*(k+1), 2)
    x = randn(n, 1)
    ConicIP.mat(x)
    a = @allocated ConicIP.mat(x)
    println("  mat(k=$k, n=$n):  $a bytes")

    Z = randn(k, k); Z = Z + Z'
    ConicIP.vecm(Z)
    a = @allocated ConicIP.vecm(Z)
    println("  vecm(k=$k):  $a bytes")
end

# ══════════════════════════════════════════════════════════════════
#  1E. Type Stability Audit
# ══════════════════════════════════════════════════════════════════

function run_type_stability_audit()
    println("\n" * "="^100)
    println("  TYPE STABILITY AUDIT")
    println("="^100)

    Random.seed!(42)

    # Block * Matrix
    println("\n--- @code_warntype: Block * Matrix{Float64} ---")
    n = 10
    F = Block([Diagonal(ones(n))])
    x = randn(n, 1)
    @code_warntype broadcastf(*, F, x)

    println("\n--- @code_warntype: inv(Block) ---")
    @code_warntype broadcastf(inv, F)

    println("\n--- @code_warntype: Block' (adjoint) ---")
    @code_warntype broadcastf(adjoint, F)

    # broadcastf with mixed blocks
    println("\n--- @code_warntype: broadcastf(*, mixed Block, Matrix) ---")
    F_mixed = Block(Any[Diagonal(ones(5)),
                        ConicIP.nestod_soc([1.0; zeros(4)], [1.0; zeros(4)])])
    x_mixed = randn(10, 1)
    @code_warntype broadcastf(*, F_mixed, x_mixed)

    println("\n--- Block.Blocks field type ---")
    println("  typeof(F.Blocks) = $(typeof(F.Blocks))")
    println("  typeof(F_mixed.Blocks) = $(typeof(F_mixed.Blocks))")
    println("  eltype(F_mixed.Blocks) = $(eltype(F_mixed.Blocks))")
end

# ══════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════

function main()
    println("ConicIP.jl Performance Profile")
    println("Julia $(VERSION)")
    println("Threads: $(Threads.nthreads())")
    println("Time: $(Dates.now())")
    println()

    macro_results = run_macro_benchmarks()
    run_profiling()
    run_allocation_tracking()
    run_type_stability_audit()

    println("\n" * "="^100)
    println("  PROFILING COMPLETE")
    println("="^100)

    return macro_results
end

using Dates
main()
