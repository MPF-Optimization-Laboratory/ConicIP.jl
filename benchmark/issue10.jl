# Benchmark for GitHub issue #10 (mlubin's sparse SOCP).
#
# Downloads the original gist instance at run time (never vendored, not part
# of CI), converts it to ConicIP internal form via test/testdata.jl's
# mpb_to_conicip, and times a solve per KKT solver.
#
# Run:  julia --project benchmark/issue10.jl [qr|sparse|2x2|all]
#
# The instance: n=6010 vars, 1002 equality rows (the first all-zero — the
# preprocessor must handle it), 2 NonPos rows, 1000 3-dim SOCs, NonNeg +
# Free variable blocks, 16010 nnz. ECOS: 0.17 s / 23 iters.

using ConicIP, SparseArrays, LinearAlgebra, Downloads

include(joinpath(@__DIR__, "..", "test", "testdata.jl"))

const GIST_URL = "https://gist.githubusercontent.com/mlubin/" *
    "79304a15043498a2f7da35d548f3610c/raw/" *
    "1abdf49e7110656a5153c16050b1ef4b9088a031/slow.jl"

function load_issue10()
    path = joinpath(mktempdir(), "slow.jl")
    Downloads.download(GIST_URL, path)
    src = read(path, String)
    # Keep only the data assignments; drop the MathProgBase driver loop.
    keep = filter(l -> occursin(r"^(c|b|con_cones|var_cones|I|J|V|A) = ", l),
                  split(src, '\n'))
    m = Module(:Issue10Data)
    Base.eval(m, :(using SparseArrays))
    Base.eval(m, Meta.parseall(join(keep, '\n')))
    g(s) = Base.invokelatest(getglobal, m, s)
    return mpb_to_conicip(g(:c), g(:A), g(:b), g(:con_cones), g(:var_cones))
end

function run_instance(prob; kktsolver, maxIters = 100, verbose = false)
    stats = @timed preprocess_conicIP(
        prob.Q, prob.c, prob.A, prob.b, prob.cone_dims, prob.G, prob.d;
        kktsolver = kktsolver, maxIters = maxIters, verbose = verbose)
    sol = stats.value
    return (status = sol.status, iters = sol.Iter, time = stats.time,
            gib = stats.bytes / 2^30)
end

function main(which = "all")
    println("Downloading gist instance …")
    prob = load_issue10()
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    nnz_total = nnz(prob.Q) + nnz(prob.A) + nnz(prob.G)
    println("n=$n  m=$m  p=$p  nnz=$nnz_total  (≈$(round(nnz_total/n, digits=1)) nnz/col)")

    solvers = Dict(
        "qr"     => ConicIP.kktsolver_qr,
        "sparse" => ConicIP.kktsolver_sparse,
        "2x2"    => pivot(ConicIP.kktsolver_2x2),
    )
    names = which == "all" ? ["sparse", "2x2", "qr"] : [which]

    for name in names
        solver = solvers[name]
        # warmup on a small instance for compilation
        small = socp_sum_of_norms(20; d = 20)
        preprocess_conicIP(small.Q, small.c, small.A, small.b,
                           small.cone_dims, small.G, small.d;
                           kktsolver = solver, verbose = false)
        if name == "qr"
            println("kktsolver_qr: 3-iteration probe only (full solve is impractical)")
            r = run_instance(prob; kktsolver = solver, maxIters = 3)
            println("  status=$(r.status)  iters=$(r.iters)  " *
                    "time=$(round(r.time, digits=1))s  alloc=$(round(r.gib, digits=2)) GiB")
        else
            r = run_instance(prob; kktsolver = solver)
            println("kktsolver_$name: status=$(r.status)  iters=$(r.iters)  " *
                    "time=$(round(r.time, digits=2))s  alloc=$(round(r.gib, digits=2)) GiB")
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(isempty(ARGS) ? "all" : ARGS[1])
end
