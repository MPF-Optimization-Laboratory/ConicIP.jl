# Capture ConicIP's actual KKT systems along its actual iterates.
#
# The wrapper builds the LDL pattern itself, hands it to the real
# ConicIP.kktsolver_ldl, and snapshots pat.K.nzval (which kktsolver_ldl
# rewrites in place) after each numeric factorization, plus every rhs and
# the wall time of the factorization and of each back-solve.
#
# Run from the worktree:  julia --project benchmark-free; see run.sh

using ConicIP, SparseArrays, LinearAlgebra, Serialization, Printf, Random
using MathOptInterface
const MOI = MathOptInterface

const WT = "/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap"
const OUT = "/private/tmp/claude-502/-Users-mpf-projects-Software-ConicIP-jl/8d6f2e95-bb86-4902-8603-0c2255592e24/scratchpad/exp-krylov/data"
mkpath(OUT)

include(joinpath(WT, "benchmark", "suite.jl"))   # instance constructors

mutable struct Store
    meta :: Any
    recs :: Vector{Any}
end
Store() = Store(nothing, Any[])

function capture_kktsolver(store::Store; static_reg = 1e-8)
    function ks(Q, A, G, cone_dims)
        pat = ConicIP._ldl_pattern(Q, A, G, cone_dims; lift_min = 6)
        gen = ConicIP.kktsolver_ldl(Q, A, G, cone_dims; pattern = pat,
                                    static_reg = static_reg)
        store.meta = (; n = pat.n, m = pat.m, p = pat.p, N = pat.N,
                        oz = pat.oz, oa = pat.oa, nlift = pat.nlift,
                        perm = copy(pat.perm),
                        colptr = copy(pat.K.colptr), rowval = copy(pat.K.rowval),
                        dp = static_reg, de = static_reg, dc = 0.0,
                        kinds = copy(pat.kinds))
        t = Ref(0)
        function solve3x3gen(F, F⁻ᵀ)
            tfac = @elapsed s3 = gen(F, F⁻ᵀ)
            t[] += 1
            rec = (iter = t[], nzval = copy(pat.K.nzval), tfac = tfac,
                   rhss = Vector{Vector{Float64}}(), tsolve = Float64[])
            push!(store.recs, rec)
            function solve3x3(bx, by, bz)
                r = zeros(pat.N)
                @views r[1:pat.n] .= bx
                @views r[pat.n+1:pat.n+pat.p] .= by
                @views r[pat.oz+1:pat.oz+pat.m] .= .-bz
                ts = @elapsed out = s3(bx, by, bz)
                push!(rec.rhss, r); push!(rec.tsolve, ts)
                return out
            end
            return solve3x3
        end
        return solve3x3gen
    end
    return ks
end

function capture_direct(name, prob)
    store = Store()
    t = @elapsed sol = preprocess_conicIP(prob.Q, prob.c, prob.A, prob.b,
                                          prob.cone_dims, prob.G, prob.d;
                                          verbose = false,
                                          kktsolver = capture_kktsolver(store))
    return (store, sol, t)
end

function capture_moi(name, path, format)
    src = MOI.FileFormats.Model(format = format, filename = path)
    MOI.read_from_file(src, path)
    store = Store()
    opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
    MOI.set(opt, MOI.Silent(), true)
    MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), capture_kktsolver(store))
    MOI.copy_to(opt, src)
    t = @elapsed MOI.optimize!(opt)
    inner = MOI.get(opt, MOI.RawSolver())
    sol = hasproperty(inner, :sol) ? inner.sol : inner
    return (store, sol, t)
end

function dump(name, store, sol, t)
    nsolve = sum(length(r.rhss) for r in store.recs; init = 0)
    @printf("%-20s T=%2d solves=%3d t=%.3fs status=%s\n",
            name, length(store.recs), nsolve, t,
            sol === nothing ? "?" : string(sol.status))
    serialize(joinpath(OUT, name * ".jls"),
              (; name, meta = store.meta, recs = store.recs, t_total = t,
                 status = sol === nothing ? "?" : string(sol.status),
                 iters = sol === nothing ? -1 : sol.Iter))
    return nothing
end

# Warm up compilation on a tiny problem
let p = lp_band(200)
    capture_direct("warmup", p)
end

todo = something(get(ENV, "ONLY", nothing), "")
want(x) = isempty(todo) || occursin(x, todo)

want("lp-band-20000")   && dump("lp-band-20000",   capture_direct("lp-band-20000", lp_band(20_000))...)
want("qp-band-20000")   && dump("qp-band-20000",   capture_direct("qp-band-20000", qp_band(20_000))...)
want("issue10")         && dump("issue10",         capture_direct("issue10", load_issue10())...)
want("socp-sumnorms-1500") && dump("socp-sumnorms-1500", capture_direct("socp-sumnorms-1500", socp_sum_of_norms(1500; d = 2000))...)

for nm in ("chainsing-1000-1", "nql30")
    want(nm) || continue
    inst = INSTANCES[findfirst(i -> i.name == nm, INSTANCES)]
    path = inst.load()
    dump(nm, capture_moi(nm, path, inst.format)...)
end

println("done")
