# Part C: inside `kktupdate`, how much is the KKT value update / assembly and
# how much is QDLDL's numeric phase?
#
# Two independent measurements:
#  (1) the package's own phase timers (`timing = PhaseTimes()`): t_kktupdate
#      (loop child) against t_ldl_factor (inclusive diagnostic), per call;
#  (2) a direct micro-benchmark: with the generator captured through a
#      kktsolver wrapper, time `solve3x3gen(F, F⁻ᵀ)` (value update + numeric
#      refactorization) against `refactor!(Fact)` alone (numeric only).
#
#   julia --project=<env> partC.jl <instance> [outdir]

using ConicIP, SparseArrays, LinearAlgebra, Printf, Random
import QDLDL
using QDLDL: refactor!

const WT = "/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap"
include(joinpath(WT, "benchmark", "suite.jl"))

const NAME = ARGS[1]
const OUTDIR = length(ARGS) >= 2 ? ARGS[2] : @__DIR__
inst = INSTANCES[findfirst(i -> i.name == NAME, INSTANCES)]
prob = inst.load()

mutable struct Cap
    count::Int
    gen::Any        # the returned solve3x3gen closure (LDLGen)
    inner::Any      # the closure object, whose fields are the captured vars
    F::Any
    Finv::Any
end
function (c::Cap)(Q, A, G, cone_dims)
    g = ConicIP.kktsolver_ldl(Q, A, G, cone_dims)
    c.gen = g; c.inner = g.gen
    return function (F, Finv)
        c.count += 1
        c.F = F; c.Finv = Finv
        return g(F, Finv)
    end
end

# ── (1) phase timers ─────────────────────────────────────────────────────
pt = ConicIP.PhaseTimes()
ConicIP.preprocess_conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                           prob.G, prob.d; verbose = false)          # warm
ConicIP.reset!(pt)
sol = ConicIP.preprocess_conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                                 prob.G, prob.d; verbose = false, timing = pt)
ns = 1e-9
@printf("%s: status=%s iters=%d\n", NAME, string(sol.status), sol.Iter)
@printf("  t_loop        = %8.2f ms\n", ns * pt.t_loop * 1e3)
@printf("  t_kktupdate   = %8.2f ms   (n_kktupdate = %d)\n", ns * pt.t_kktupdate * 1e3, pt.n_kktupdate)
@printf("  t_ldl_factor  = %8.2f ms   (n_ldl_factor = %d)  [inclusive, whole call]\n",
        ns * pt.t_ldl_factor * 1e3, pt.n_ldl_factor)
@printf("  t_setup       = %8.2f ms   t_init = %8.2f ms\n", ns * pt.t_setup * 1e3, ns * pt.t_init * 1e3)
@printf("  kktupdate − ldl_factor = %8.2f ms  (assembly, loop, biased low by the\n",
        (ns * pt.t_kktupdate - ns * pt.t_ldl_factor) * 1e3)
@printf("                                      initial factorization charged to setup)\n")

# ── (2) direct micro-benchmark ───────────────────────────────────────────
c = Cap(0, nothing, nothing, nothing, nothing)
ConicIP.preprocess_conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                           prob.G, prob.d; verbose = false, kktsolver = c)
Fact = getfield(c.inner, :Fact)
K = getfield(c.inner, :K)
gen = c.gen

best(f, reps = 7) = minimum(begin; GC.gc(false); @elapsed f(); end for _ in 1:reps)
gen(c.F, c.Finv); refactor!(Fact)                     # warm
t_gen  = best(() -> gen(c.F, c.Finv))                 # value update + refactor
t_num  = best(() -> refactor!(Fact))                  # numeric only
t_asm  = t_gen - t_num
@printf("  micro: solve3x3gen(F,F⁻ᵀ) = %8.3f ms  (assembly + numeric)\n", 1e3t_gen)
@printf("  micro: refactor!(Fact)    = %8.3f ms  (numeric only)\n", 1e3t_num)
@printf("  micro: assembly           = %8.3f ms  (%.1f%% of kktupdate)\n",
        1e3t_asm, 100 * t_asm / t_gen)
@printf("  N=%d nnz(triu K)=%d nnz(L)=%d factorizations=%d\n",
        size(K, 1), nnz(K), nnz(Fact.L), c.count)

open(joinpath(OUTDIR, "partC-$(NAME).txt"), "w") do io
    for (k, v) in ("instance" => NAME, "iters" => sol.Iter,
                   "t_loop_s" => ns * pt.t_loop, "t_kktupdate_s" => ns * pt.t_kktupdate,
                   "t_ldl_factor_s" => ns * pt.t_ldl_factor,
                   "n_kktupdate" => pt.n_kktupdate, "n_ldl_factor" => pt.n_ldl_factor,
                   "t_setup_s" => ns * pt.t_setup, "t_init_s" => ns * pt.t_init,
                   "micro_gen_s" => t_gen, "micro_refactor_s" => t_num,
                   "micro_assembly_s" => t_asm,
                   "assembly_frac_of_kktupdate" => t_asm / t_gen,
                   "N" => size(K, 1), "nnz_triuK" => nnz(K), "nnz_L" => nnz(Fact.L))
        println(io, k, "=", v)
    end
end
println("wrote partC-$(NAME).txt")
