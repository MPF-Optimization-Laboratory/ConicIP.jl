# MOI model builder for ConicIP problem tuples
# ============================================
# Turns a ConicIP problem tuple `(Q, c, A, b, cone_dims, G, d)`, meaning
#
#   min ½ xᵀQx − cᵀx   s.t.   A x − b ∈ K,   G x = d,
#
# into an MOI model on any optimizer, and reads a solution back in ConicIP's
# own `Solution` field layout (`y, w, v, s, status, ...`) so that
# `residuals(prob, sol)` and `verified(sol, res)` from suite.jl apply to a
# foreign solver's answer unchanged.
#
# Conventions (matching src/MOI_wrapper.jl, which is the reference):
#   * objective: MOI's ScalarQuadraticFunction is ½xᵀQx + aᵀx, so a diagonal
#     quadratic coefficient q means ½q·xᵢ² and an off-diagonal one means
#     q·xᵢxⱼ — pass Q_ii and, once, Q_ij (i < j); affine terms are −c;
#   * equalities: VectorAffineFunction(G x − d) ∈ Zeros(p);
#   * cone blocks: VectorAffineFunction(A[I,:] x − b[I]) ∈ Nonnegatives,
#     SecondOrderCone, or PositiveSemidefiniteConeTriangle;
#   * "S" blocks: ConicIP rows are vecm (row-major upper triangle, √2 on the
#     off-diagonals); MOI wants the column-major triangle, unscaled. Both the
#     primal and the dual map back the same way (vecm = √2 × MOI off the
#     diagonal) because MOI's triangle inner product counts off-diagonal
#     entries twice;
#   * duals: MOI's equality dual is −w and the cone dual is v, from the
#     stationarity condition Qy − c + Gᵀw − Aᵀv = 0.
#
# Loads with MathOptInterface, SparseArrays, LinearAlgebra only (no ConicIP),
# so it can be used from the benchmark and the test environment alike.

using MathOptInterface, SparseArrays, LinearAlgebra
const MOI = MathOptInterface

# ──────────────────────────────────────────────────────────────
#  PSD triangle ordering
# ──────────────────────────────────────────────────────────────

# `perm[moi_k]` is the vecm position of MOI triangle position `moi_k`, and
# `is_offdiag[moi_k]` marks off-diagonal entries. Same formula as
# `ConicIP._psd_moi_vecm_info`; harness_tests.jl checks the two agree.
#
#   MOI  (column-major): (1,1),(1,2),(2,2),(1,3),(2,3),(3,3),…
#   vecm (row-major):    (1,1),(1,2),(1,3),(2,2),(2,3),(3,3),…
function psd_moi_vecm_info(k::Int)
    n = round(Int, (sqrt(1 + 8k) - 1) / 2)
    n * (n + 1) ÷ 2 == k || throw(ArgumentError("$k is not a triangle number"))
    perm = zeros(Int, k)
    is_offdiag = falses(k)
    moi_k = 0
    for j in 1:n, i in 1:j
        moi_k += 1
        before_i = (i - 1) * n - (i - 1) * (i - 2) ÷ 2   # entries in rows < i
        perm[moi_k] = before_i + (j - i + 1)
        is_offdiag[moi_k] = i != j
    end
    return perm, is_offdiag
end

# Row map for one cone block of ConicIP rows `I`: MOI output row `t` reads
# ConicIP row `I[src[t]]` times `scale[t]`.
function _block_rowmap(kind, k)
    if kind == "S"
        perm, od = psd_moi_vecm_info(k)
        return perm, [o ? 1 / √2 : 1.0 for o in od]
    end
    return collect(1:k), ones(k)
end

# ──────────────────────────────────────────────────────────────
#  Model construction
# ──────────────────────────────────────────────────────────────

_moi_set(kind, k) = kind == "R" ? MOI.Nonnegatives(k) :
                    kind == "Q" ? MOI.SecondOrderCone(k) :
                    kind == "S" ? MOI.PositiveSemidefiniteConeTriangle(
                                      round(Int, (sqrt(1 + 8k) - 1) / 2)) :
                    throw(ArgumentError("unknown cone kind $kind"))

# VectorAffineFunction for rows `I` of the sparse (M, r): M[I,:] x − r[I],
# output row t taking source row I[src[t]] scaled by scale[t].
function _vector_affine(M::SparseMatrixCSC, r, I, x, src, scale)
    k = length(I)
    dst = zeros(Int, k)               # source-local row → output row
    for t in 1:k
        dst[src[t]] = t
    end
    lo, hi = first(I), last(I)
    terms = MOI.VectorAffineTerm{Float64}[]
    rows = rowvals(M); vals = nonzeros(M)
    for j in axes(M, 2), ptr in nzrange(M, j)
        i = rows[ptr]
        lo <= i <= hi || continue
        t = dst[i - lo + 1]
        push!(terms, MOI.VectorAffineTerm(t,
                  MOI.ScalarAffineTerm(scale[t] * vals[ptr], x[j])))
    end
    consts = [-scale[t] * r[I[src[t]]] for t in 1:k]
    return MOI.VectorAffineFunction(terms, consts)
end

"""
    build_moi_model(opt, prob) -> maps

Load the ConicIP problem tuple `prob = (Q, c, A, b, cone_dims, G, d)` into the
MOI model `opt` (an instantiated optimizer that supports incremental
modification, e.g. one wrapped in a `CachingOptimizer`; see
[`bridged_cached`](@ref)). Returns

    (x = variables, ci_eq = equality constraint index or nothing,
     blocks = [(ci = ..., I = rows, kind = "R"|"Q"|"S"), ...])

for [`recover_solution`](@ref).
"""
function build_moi_model(opt, prob)
    Q = sparse(prob.Q); A = sparse(prob.A); G = sparse(prob.G)
    c = prob.c; b = prob.b; d = prob.d
    n = length(c); m = size(A, 1); p = size(G, 1)
    size(Q) == (n, n) || throw(DimensionMismatch("Q is $(size(Q)), expected ($n, $n)"))
    size(A, 2) == n && length(b) == m || throw(DimensionMismatch("A/b sizes"))
    size(G, 2) == n && length(d) == p || throw(DimensionMismatch("G/d sizes"))
    sum(last, prob.cone_dims; init = 0) == m ||
        throw(DimensionMismatch("cone_dims cover $(sum(last, prob.cone_dims; init = 0)) rows, A has $m"))

    x = MOI.add_variables(opt, n)
    MOI.set(opt, MOI.ObjectiveSense(), MOI.MIN_SENSE)

    affine = [MOI.ScalarAffineTerm(-c[j], x[j]) for j in 1:n if c[j] != 0]
    if nnz(Q) == 0
        MOI.set(opt, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(affine, 0.0))
    else
        quad = MOI.ScalarQuadraticTerm{Float64}[]
        rows = rowvals(Q); vals = nonzeros(Q)
        for j in axes(Q, 2), ptr in nzrange(Q, j)
            i = rows[ptr]
            i <= j || continue                     # upper triangle, once
            vals[ptr] == 0 && continue
            push!(quad, MOI.ScalarQuadraticTerm(vals[ptr], x[i], x[j]))
        end
        MOI.set(opt, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}(),
                MOI.ScalarQuadraticFunction(quad, affine, 0.0))
    end

    ci_eq = nothing
    if p > 0
        f = _vector_affine(G, d, 1:p, x, collect(1:p), ones(p))
        ci_eq = MOI.add_constraint(opt, f, MOI.Zeros(p))
    end

    blocks = NamedTuple{(:ci, :I, :kind), Tuple{Any, UnitRange{Int}, String}}[]
    offset = 0
    for (kind, k) in prob.cone_dims
        I = offset + 1 : offset + k
        offset += k
        src, scale = _block_rowmap(kind, k)
        f = _vector_affine(A, b, I, x, src, scale)
        ci = MOI.add_constraint(opt, f, _moi_set(kind, k))
        push!(blocks, (ci = ci, I = I, kind = kind))
    end
    return (x = x, ci_eq = ci_eq, blocks = blocks)
end

# ──────────────────────────────────────────────────────────────
#  Solution recovery
# ──────────────────────────────────────────────────────────────

# MOI block vector → ConicIP rows (inverse of the map used in the builder).
function _to_vecm!(dest, I, vals, kind)
    src, scale = _block_rowmap(kind, length(I))
    for t in eachindex(I)
        dest[I[src[t]]] = vals[t] / scale[t]
    end
    return dest
end

_try_get(f, default) = try f() catch; default end

"""
    recover_solution(opt, maps, prob) -> (y, w, v, s, status, iters, time)

Read the solution of `opt` back into ConicIP's conventions: `y` is the
primal, `s = A y − b` the block slacks, `v ∈ K*` the cone duals, `w` the
equality duals, `status = :Optimal` when `TerminationStatus == OPTIMAL` and
the MOI status name as a Symbol otherwise. Missing results become `NaN`;
nothing here throws on an infeasible/unbounded model.
"""
function recover_solution(opt, maps, prob)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    term = MOI.get(opt, MOI.TerminationStatus())
    status = term == MOI.OPTIMAL ? :Optimal : Symbol(string(term))
    y = fill(NaN, n); s = fill(NaN, m); v = fill(NaN, m); w = fill(NaN, p)
    if MOI.get(opt, MOI.ResultCount()) >= 1
        yy = _try_get(() -> MOI.get(opt, MOI.VariablePrimal(), maps.x), nothing)
        yy === nothing || (y .= yy)
        for blk in maps.blocks
            sv = _try_get(() -> MOI.get(opt, MOI.ConstraintPrimal(), blk.ci), nothing)
            sv === nothing || _to_vecm!(s, blk.I, sv, blk.kind)
            dv = _try_get(() -> MOI.get(opt, MOI.ConstraintDual(), blk.ci), nothing)
            dv === nothing || _to_vecm!(v, blk.I, dv, blk.kind)
        end
        if maps.ci_eq !== nothing
            wv = _try_get(() -> MOI.get(opt, MOI.ConstraintDual(), maps.ci_eq), nothing)
            wv === nothing || (w .= -wv)
        end
    end
    iters = _try_get(() -> Int(MOI.get(opt, MOI.BarrierIterations())), -1)
    time = _try_get(() -> Float64(MOI.get(opt, MOI.SolveTimeSec())), NaN)
    return (y = y, w = w, v = v, s = s, status = status, iters = iters, time = time)
end

"""
    bridged_cached(constructor) -> optimizer

`constructor()` wrapped in full bridging and a `CachingOptimizer`, so that
models can be built incrementally on optimizers that only implement
`copy_to`-style `optimize!` (ConicIP, Clarabel, ECOS).
"""
function bridged_cached(constructor)
    inner = MOI.instantiate(constructor; with_bridge_type = Float64)
    cache = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    return MOI.Utilities.CachingOptimizer(cache, inner)
end
