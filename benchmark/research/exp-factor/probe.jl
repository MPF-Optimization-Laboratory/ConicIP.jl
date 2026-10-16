# Probe: what does a kktsolver_ldl generator expose, and what does QDLDL's
# factorisation object look like?  No src edits: the closure's captured
# variables are fields of the closure object.
using ConicIP, SparseArrays, LinearAlgebra, Random
import QDLDL
using Pkg

println("QDLDL version: ", [v.version for (k,v) in Pkg.dependencies() if v.name == "QDLDL"])

include("/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap/benchmark/suite.jl")

prob = lp_band(2000)
captured = Ref{Any}(nothing)

mutable struct Wrap
    count::Int
    target::Int
    K::Any
    inner::Any
end
function (w::Wrap)(Q, A, G, cone_dims)
    g = ConicIP.kktsolver_ldl(Q, A, G, cone_dims)
    inner = g.gen
    w.inner = inner
    println("LDLGen fields: ", fieldnames(typeof(g)))
    println("closure fields: ", fieldnames(typeof(inner)))
    return function (F, Finv)
        s = g(F, Finv)
        w.count += 1
        if w.count == w.target
            w.K = copy(getfield(inner, :K))
        end
        return s
    end
end

w = Wrap(0, 5, nothing, nothing)
sol = ConicIP.preprocess_conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                                 prob.G, prob.d; verbose = false, kktsolver = w)
println("status=", sol.status, " iters=", sol.Iter, " gens=", w.count)
K = w.K
println("K: ", size(K), " nnz=", nnz(K), " issym=", issymmetric(K))
F = getfield(w.inner, :Fact)
println("QDLDL factorisation type: ", typeof(F))
println("  fields: ", fieldnames(typeof(F)))
for f in fieldnames(typeof(F))
    v = getfield(F, f)
    println("   ", f, " :: ", typeof(v))
    if v !== nothing && !(v isa Number) && !(v isa Bool)
        try
            println("      fields: ", fieldnames(typeof(v)))
        catch
        end
    end
end
