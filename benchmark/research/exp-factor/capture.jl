# Part B: capture the KKT matrix ConicIP factors at a mid-run iteration and
# time five factorization back ends on it.  No src edits: `kktsolver_ldl`
# returns an `LDLGen` whose `.gen` field is the closure `solve3x3gen`, and a
# closure's captured variables (`K`, `Fact`, …) are its fields.
#
#   julia --project=. capture.jl <instance-name> [outdir]
#
# Writes <outdir>/partB-<instance>.csv (one row per timed back end) and
# <outdir>/partB-<instance>.stats (matrix / supernode statistics).

using ConicIP, SparseArrays, LinearAlgebra, Printf, Random
import QDLDL
using QDLDL: qdldl, refactor!, update_values!

const WT = "/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap"
include(joinpath(WT, "benchmark", "suite.jl"))

const NAME = ARGS[1]
const OUTDIR = length(ARGS) >= 2 ? ARGS[2] : @__DIR__

# iteration counts from benchmark/results/phase-conicip-3881ac2.csv; the
# capture target is the halfway factorization.
const ITERS = Dict("lp-band-20000" => 21, "lp-band-200000" => 23,
                   "qp-band-200000" => 9, "socp-sumnorms-1500" => 12,
                   "nb" => 19, "chainsing-1000-1" => 11, "qssp30" => 18,
                   "issue10" => 25)

mutable struct Wrap
    count::Int
    target::Int
    K::Any
    perm::Any
    Dsigns::Any
    inner::Any
end
Wrap(target) = Wrap(0, target, nothing, nothing, nothing, nothing)

function (w::Wrap)(Q, A, G, cone_dims)
    g = ConicIP.kktsolver_ldl(Q, A, G, cone_dims)
    inner = g.gen
    w.inner = inner
    return function (F, Finv)
        s = g(F, Finv)
        w.count += 1
        if w.count == w.target
            w.K = copy(getfield(inner, :K))
            fact = getfield(inner, :Fact)
            w.perm = copy(fact.perm)
            w.Dsigns = copy(fact.workspace.Dsigns)
        end
        return s
    end
end

# ── run the instance, capturing K ────────────────────────────────────────
inst = INSTANCES[findfirst(i -> i.name == NAME, INSTANCES)]
target = max(1, ITERS[NAME] ÷ 2)
w = Wrap(target)

t_solve = @elapsed begin
    if inst.kind == :direct
        prob = inst.load()
        global sol = ConicIP.preprocess_conicIP(prob.Q, prob.c, prob.A, prob.b,
                                                prob.cone_dims, prob.G, prob.d;
                                                verbose = false, kktsolver = w)
        global status = string(sol.status); global iters = sol.Iter
    else
        path = inst.load()
        src = MOI.FileFormats.Model(format = inst.format, filename = path)
        MOI.read_from_file(src, path)
        opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
        MOI.set(opt, MOI.Silent(), true)
        MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), w)
        MOI.copy_to(opt, src)
        MOI.optimize!(opt)
        raw = MOI.get(opt, MOI.RawSolver())::ConicIP.Optimizer
        global status = string(MOI.get(opt, MOI.TerminationStatus()))
        global iters = raw.sol === nothing ? 0 : raw.sol.Iter
    end
end

if w.K === nothing          # fewer factorizations than expected: take the last
    w.K = copy(getfield(w.inner, :K))
    fact = getfield(w.inner, :Fact)
    w.perm = copy(fact.perm); w.Dsigns = copy(fact.workspace.Dsigns)
end

K     = w.K                              # UPPER triangle, CSC, N×N
N     = size(K, 1)
nnzK  = nnz(K)                           # stored upper triangle incl. diagonal
Kfull = sparse(Symmetric(K, :U))         # full symmetric copy for UMFPACK
@printf("instance=%s status=%s iters=%d captured_at=%d/%d N=%d nnz(triu K)=%d t_solve=%.3f\n",
        NAME, status, iters, min(w.count, target), w.count, N, nnzK, t_solve)

# ── QDLDL: symbolic once, numeric refactorization timed ──────────────────
Fq = qdldl(K; perm = w.perm, Dsigns = w.Dsigns,
           regularize_eps = 1e-13, regularize_delta = 2e-7)
nnzL  = nnz(Fq.L)
Lnz   = Fq.workspace.Lnz                 # subdiagonal count per column of L
etree = Fq.workspace.etree
flops = sum(x -> Float64(x)^2, Lnz)      # Σ colcount²  (≈ 2× the true flop count)

# fundamental-supernode chains: column j joins its etree parent p when
# p == j+1, |L_j| == |L_p| + 1 (identical pattern below the diagonal) and j
# is p's only child.  Chains of ≥ 16 columns are what a supernodal code
# turns into BLAS-3 panels.
function supernode_stats(N, Lnz, etree)
    nchild = zeros(Int, N)
    for j in 1:N
        p = etree[j]
        (p > 0 && p <= N) && (nchild[p] += 1)
    end
    joins = falses(N)
    for j in 1:(N-1)
        p = etree[j]
        (p == j + 1 && Lnz[j] == Lnz[p] + 1 && nchild[p] == 1) && (joins[j] = true)
    end
    fl_ge16 = 0.0; nchains = 0; maxchain = 0; len = 1
    chainf = Float64(Lnz[1])^2
    for j in 1:N
        if j < N && joins[j]
            len += 1; chainf += Float64(Lnz[j+1])^2
        else
            nchains += 1; maxchain = max(maxchain, len)
            len >= 16 && (fl_ge16 += chainf)
            len = 1; chainf = j < N ? Float64(Lnz[j+1])^2 : 0.0
        end
    end
    return (nchains = nchains, maxchain = maxchain, fl_ge16 = fl_ge16)
end
sn = supernode_stats(N, Lnz, etree)
chain_flops_ge16 = sn.fl_ge16; nchains = sn.nchains; maxchain = sn.maxchain

# ── timing helpers ───────────────────────────────────────────────────────
best(f, reps = 5) = minimum(begin; GC.gc(false); @elapsed f(); end for _ in 1:reps)
results = Tuple{String,Int,Float64,String}[]   # (what, blas_threads, seconds, note)
push_r!(w_, t_, s_, n_ = "") = push!(results, (w_, t_, s_, n_))

function try_time(label, blas, f; reps = 5)
    BLAS.set_num_threads(blas)
    try
        f()                                  # warm / compile
        t = best(f, reps)
        push_r!(label, blas, t, "")
        @printf("  %-34s blas=%d  %8.2f ms\n", label, blas, 1e3t)
        return t
    catch e
        push_r!(label, blas, NaN, first(sprint(showerror, e), 90))
        @printf("  %-34s blas=%d  FAILED: %s\n", label, blas,
                first(sprint(showerror, e), 90))
        return NaN
    end
end

BLAS.set_num_threads(1)
println("timings (best of 5, warmed):")
try_time("qdldl refactor!", 1, () -> refactor!(Fq))

# ── CHOLMOD on the quasi-definite matrix itself (simplicial LDLᵀ, no
#    pivoting: it may or may not survive the indefinite diagonal) ─────────
const CHM = SparseArrays.CHOLMOD
Ksym = Symmetric(Kfull)
Fl = nothing
try
    global Fl = ldlt(Ksym)
    @printf("  cholmod ldlt(K) quasidefinite: nnz(L)=%d\n", nnz(sparse(Fl.L)))
    global nnzL_chmldlt = nnz(sparse(Fl.L))
catch e
    @printf("  cholmod ldlt(K) FAILED: %s\n", first(sprint(showerror, e), 120))
    global nnzL_chmldlt = -1
end
if Fl !== nothing
    for nt in (1, 6)
        try_time("cholmod ldlt! (K, numeric only)", nt, () -> ldlt!(Fl, Ksym))
    end
end

# ── CHOLMOD on an SPD proxy with the IDENTICAL sparsity pattern ──────────
# |K| off-diagonal, diagonal replaced by 1 + Σ|off-diagonal in that row|:
# same pattern as K, symmetric, strictly diagonally dominant, hence SPD.
Kp = copy(Kfull)
Kp.nzval .= abs.(Kp.nzval)
rs = zeros(N)
for j in 1:N, q in Kp.colptr[j]:(Kp.colptr[j+1]-1)
    Kp.rowval[q] == j || (rs[j] += Kp.nzval[q])
end
for j in 1:N, q in Kp.colptr[j]:(Kp.colptr[j+1]-1)
    Kp.rowval[q] == j && (Kp.nzval[q] = rs[j] + 1.0)
end
Kpsym = Symmetric(Kp)
Fc = nothing
nsuper = -1; is_super = -1
try
    global Fc = cholesky(Kpsym)
    nnzLc = nnz(sparse(Fc.L))
    @printf("  cholmod cholesky(SPD proxy): nnz(L)=%d\n", nnzLc)
    global nnzLc_g = nnzLc
    ptr = unsafe_load(pointer(Fc))
    global is_super = Int(ptr.is_super)
    global nsuper = Int(ptr.nsuper)
    @printf("  cholmod is_super=%d nsuper=%d\n", is_super, nsuper)
catch e
    @printf("  cholesky(SPD proxy) FAILED: %s\n", first(sprint(showerror, e), 120))
    global nnzLc_g = -1
end
if Fc !== nothing
    for nt in (1, 6)
        try_time("cholmod cholesky! (SPD proxy, auto)", nt, () -> cholesky!(Fc, Kpsym))
    end
end

# forced supernodal (BLAS-3) Cholesky of the same SPD proxy
Fs = nothing; nsuper_forced = -1
try
    CHM.getcommon()[].supernodal = 2
    global Fs = cholesky(Kpsym)
    p = unsafe_load(pointer(Fs))
    global nsuper_forced = Int(p.nsuper)
    @printf("  cholmod forced supernodal: is_super=%d nsuper=%d\n", Int(p.is_super), Int(p.nsuper))
catch e
    @printf("  forced-supernodal cholesky FAILED: %s\n", first(sprint(showerror, e), 120))
end
if Fs !== nothing
    for nt in (1, 6)
        try_time("cholmod cholesky! (SPD proxy, SUPERNODAL)", nt, () -> cholesky!(Fs, Kpsym))
    end
end
CHM.getcommon()[].supernodal = 1

# ── UMFPACK ──────────────────────────────────────────────────────────────
Fu = nothing
try
    global Fu = lu(Kfull)
    @printf("  umfpack lu: nnz(L)+nnz(U)=%d\n", nnz(Fu.L) + nnz(Fu.U))
    global nnzLU = nnz(Fu.L) + nnz(Fu.U)
catch e
    @printf("  lu FAILED: %s\n", first(sprint(showerror, e), 120))
    global nnzLU = -1
end
if Fu !== nothing
    for nt in (1, 6)
        try_time("umfpack lu! refactorization", nt, () -> lu!(Fu, Kfull); reps = 3)
    end
end

# ── LDLFactorizations (optional, scratch env) ────────────────────────────
have_ldlf = false
try
    @eval using LDLFactorizations
    global have_ldlf = true
catch
    println("  LDLFactorizations not available in this environment")
end
if have_ldlf
    LDLF = Base.invokelatest(getfield(Main, :ldl_analyze), Symmetric(Kfull, :U))
    try_time("LDLFactorizations ldl_factorize!", 1,
             () -> Base.invokelatest(getfield(Main, :ldl_factorize!),
                                     Symmetric(Kfull, :U), LDLF); reps = 3)
    global nnzL_ldlf = nnz(sparse(LDLF.L))
else
    global nnzL_ldlf = -1
end

# ── back-solve, one RHS ──────────────────────────────────────────────────
b = randn(N)
xq = similar(b)
BLAS.set_num_threads(1)
try_time("qdldl solve! (1 rhs)", 1, () -> (copyto!(xq, b); QDLDL.solve!(Fq, xq)); reps = 9)
if Fc !== nothing
    for nt in (1, 6)
        try_time("cholmod \\ (1 rhs, SPD proxy)", nt, () -> (Fc \ b); reps = 9)
    end
end
if Fs !== nothing
    try_time("cholmod \\ (1 rhs, SUPERNODAL proxy)", 1, () -> (Fs \ b); reps = 9)
end
if Fl !== nothing
    try_time("cholmod \\ (1 rhs, ldlt of K)", 1, () -> (Fl \ b); reps = 9)
end
if Fu !== nothing
    try_time("umfpack \\ (1 rhs)", 1, () -> (Fu \ b); reps = 5)
end

# ── write ────────────────────────────────────────────────────────────────
open(joinpath(OUTDIR, "partB-$(NAME).csv"), "w") do io
    println(io, "instance,what,blas_threads,seconds,note")
    for (a, bt, t, note) in results
        println(io, NAME, ",", a, ",", bt, ",", t, ",\"", note, "\"")
    end
end
open(joinpath(OUTDIR, "partB-$(NAME).stats"), "w") do io
    for (k, v) in ("instance" => NAME, "status" => status, "iters" => iters,
                   "captured_at" => min(w.count, target), "n_factorizations" => w.count,
                   "N" => N, "nnz_triuK" => nnzK, "nnz_Kfull" => nnz(Kfull),
                   "nnz_L_qdldl" => nnzL,
                   "fill_ratio_L_over_triuK" => nnzL / nnzK,
                   "flops_sum_colcount2" => flops,
                   "nnz_L_cholmod_spdproxy" => nnzLc_g,
                   "nnz_L_cholmod_ldlt_K" => nnzL_chmldlt,
                   "cholmod_nsuper_forced" => nsuper_forced,
                   "cholmod_is_super" => is_super, "cholmod_nsuper" => nsuper,
                   "nnz_LU_umfpack" => nnzLU, "nnz_L_ldlfactorizations" => nnzL_ldlf,
                   "supernode_chains" => nchains, "max_chain_len" => maxchain,
                   "flop_frac_in_chains_ge16" => chain_flops_ge16 / flops,
                   "t_solve_capture_run" => t_solve)
        println(io, k, "=", v)
    end
end
println("wrote partB-$(NAME).csv / .stats")
