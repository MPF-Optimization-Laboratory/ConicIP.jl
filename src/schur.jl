# ──────────────────────────────────────────────────────────────
#  Reduced Schur complements for semidefinite problems
# ──────────────────────────────────────────────────────────────
#
# Two dense KKT paths reduce to the same product: a Gram matrix
#
#     M = Bᵀ Wᵀ W B,
#
# where B is a sparse matrix whose columns are coefficient vectors in cone
# coordinates and W is a block-diagonal Nesterov-Todd scaling (a `Block`).
#
#   * kktsolver_qr's reduced Hessian Q + Aᵀ F⁻¹F⁻ᵀ A is Q + M with B = A and
#     W = F⁻ᵀ;
#   * kktsolver_sdpschur's equality Schur complement G H⁻¹ Gᵀ is M with
#     B = (G A⁻¹)ᵀ and W = F.
#
# For a semidefinite block of order k, with W acting as X ↦ RᵀXR and
# T = RRᵀ, the entry is the trace form tr(Gᵢ T Gⱼ T) with Gᵢ = mat(bᵢ)
# (Fujisawa-Kojima-Nakata). It is assembled one column at a time: form
# Wᵢ = T Gᵢ T and take its inner product with the sparse bⱼ, never storing
# the (k(k+1)/2)×ncol product W B. Wᵢ is formed densely (two k³ products),
# as a sum of rank-two updates when bᵢ has few nonzeros, or not at all
# when both columns are sparse enough for the four-index formula below.
# Diagonal blocks contribute Bᵀ diag(w²) B sparsely and any other block
# (second-order cone, dense) through a dense W_B B_B.

# Per-block slice of B, fixed at setup: the columns of B with a nonzero in
# the block's rows, and B[I, cols] with local row numbering.
struct _GramBlock
  I::UnitRange{Int}
  cols::Vector{Int}
  Bs::SparseMatrixCSC{Float64,Int}
end

# Workspace of one semidefinite block of order k.
struct _GramSDP
  k::Int
  ia::Vector{Int}       # packed coordinate → row index a (a ≤ b)
  ib::Vector{Int}       # packed coordinate → column index b
  T::Matrix{Float64}
  Wb::Matrix{Float64}   # T Gᵢ T
  Gb::Matrix{Float64}   # mat(bᵢ)
  tmp::Matrix{Float64}
end

struct GramPlan
  ncol::Int
  blocks::Vector{_GramBlock}
  sdp::Vector{Union{Nothing,_GramSDP}}
end

# Packed (vecm) coordinate c of an order-k block ↦ (a, b), a ≤ b: row-major
# upper triangle, the order `vecm` writes.
function _packed_pairs(k)
  ia = Int[]; ib = Int[]
  for a = 1:k, b = a:k
    push!(ia, a); push!(ib, b)
  end
  return (ia, ib)
end

"""
    _gram_plan(B, cone_dims) -> GramPlan

Setup for [`_gram!`](@ref): slice the sparse `B` (rows = cone coordinates
in `cone_dims` order) per cone block and allocate the semidefinite
workspaces. `B`'s values are read here once and never again.
"""
function _gram_plan(B::SparseMatrixCSC, cone_dims)
  blocks = _GramBlock[]
  sdp = Union{Nothing,_GramSDP}[]
  Bt = sparse(B')                      # rows of B are columns of Bᵗ
  off = 0
  for (btype, dim) in cone_dims
    I = (off+1):(off+dim)
    off += dim
    # Columns of B touching rows I: the union of the patterns of Bᵗ[:, I].
    touched = falses(size(B, 2))
    for r in I, t in nzrange(Bt, r)
      touched[rowvals(Bt)[t]] = true
    end
    cols = findall(touched)
    push!(blocks, _GramBlock(I, cols, B[I, cols]))
    if btype == "S"
      k = ord(1:dim)
      (ia, ib) = _packed_pairs(k)
      push!(sdp, _GramSDP(k, ia, ib, zeros(k, k), zeros(k, k), zeros(k, k), zeros(k, k)))
    else
      push!(sdp, nothing)
    end
  end
  return GramPlan(size(B, 2), blocks, sdp)
end

"""
    _gram!(M, plan, W::Block) -> M

Overwrite the dense `M` (ncol × ncol) with Bᵀ Wᵀ W B for the `B` of
`plan`. Only the upper triangle is accumulated; it is mirrored at the end,
so `M` is exactly symmetric.
"""
function _gram!(M::Matrix{Float64}, plan::GramPlan, W::Block)
  fill!(M, 0.0)
  for (bi, gb) in enumerate(plan.blocks)
    isempty(gb.cols) && continue
    Blk = W.Blocks[bi]
    if Blk isa DiagBlock
      _gram_diag!(M, gb, Blk.diag)
    elseif Blk isa VecCongurance
      _gram_sdp!(M, gb, plan.sdp[bi], Blk.R)
    else
      _gram_dense!(M, gb, Blk)
    end
  end
  n = size(M, 1)
  @inbounds for j = 1:n, i = j+1:n
    M[i, j] = M[j, i]
  end
  return M
end

# Bsᵀ diag(w²) Bs, sparse, scattered into the upper triangle.
function _gram_diag!(M, gb::_GramBlock, w)
  Bs = gb.Bs
  S = Bs' * (Diagonal(w .^ 2) * Bs)
  rv = rowvals(S); nz = nonzeros(S); cols = gb.cols
  @inbounds for jl = 1:size(S, 2), t in nzrange(S, jl)
    il = rv[t]
    il <= jl && (M[cols[il], cols[jl]] += nz[t])
  end
  return M
end

# Any other block (second-order cone SymWoodbury, dense Matrix): Y = W_B B_B
# densely, then YᵀY.
function _gram_dense!(M, gb::_GramBlock, Blk)
  X = Matrix(gb.Bs)
  Y = similar(X)
  for j in axes(X, 2)
    _blk_mul!(view(Y, :, j), Blk, view(X, :, j), false)
  end
  S = Y' * Y
  cols = gb.cols
  @inbounds for jl in axes(S, 2), il = 1:jl
    M[cols[il], cols[jl]] += S[il, jl]
  end
  return M
end

# Semidefinite block, W: X ↦ RᵀXR, T = RRᵀ.
function _gram_sdp!(M, gb::_GramBlock, ws::_GramSDP, R)
  k = ws.k; T = ws.T; Wb = ws.Wb; Gb = ws.Gb
  ia = ws.ia; ib = ws.ib
  mul!(T, R, R')
  @inbounds for j = 1:k, i = j+1:k          # exact symmetry
    T[i, j] = T[j, i]
  end
  Bs = gb.Bs; rv = rowvals(Bs); nz = nonzeros(Bs); cols = gb.cols
  nc = length(cols)
  nB = nnz(Bs)
  s2 = sqrt(2.0); is2 = 1/s2
  for il = 1:nc
    rng = nzrange(Bs, il)
    q = length(rng)
    q == 0 && continue
    # Rough flop counts of the three ways to get row il of the block's
    # contribution (the dense path is discounted for running in BLAS-3).
    c_f3    = 6.0 * q * nB / 2
    c_rank  = 4.0 * q * k^2 + 2.0 * nB
    c_dense = 0.5 * k^3 + k^2 + 2.0 * nB
    if c_f3 <= min(c_rank, c_dense)
      # Four-index formula: with mat(e_c) = β_c (e_a e_bᵀ + e_b e_aᵀ),
      # β = 1/2 on the diagonal and 1/√2 off it,
      #   tr(mat(e_s) T mat(e_t) T) = 2β_sβ_t (T_{b_s c_t} T_{d_t a_s} + T_{b_s d_t} T_{c_t a_s}).
      for jl = il:nc
        acc = 0.0
        for s in rng
          cs = rv[s]; as = ia[cs]; bs = ib[cs]
          gs = nz[s] * (as == bs ? 0.5 : is2)
          for t in nzrange(Bs, jl)
            ct = rv[t]; at = ia[ct]; bt = ib[ct]
            gt = nz[t] * (at == bt ? 0.5 : is2)
            acc += 2 * gs * gt * (T[bs, at] * T[bt, as] + T[bs, bt] * T[at, as])
          end
        end
        M[cols[il], cols[jl]] += acc
      end
      continue
    end
    if c_rank <= c_dense
      # Wb = Σ_s g_s β_s (T_a T_bᵀ + T_b T_aᵀ)
      fill!(Wb, 0.0)
      for s in rng
        cs = rv[s]; a = ia[cs]; b = ib[cs]
        g = nz[s] * (a == b ? 0.5 : is2)
        BLAS.ger!(g, view(T, :, a), view(T, :, b), Wb)
        BLAS.ger!(g, view(T, :, b), view(T, :, a), Wb)
      end
    else
      fill!(Gb, 0.0)
      for s in rng
        cs = rv[s]; a = ia[cs]; b = ib[cs]
        if a == b
          Gb[a, a] = nz[s]
        else
          Gb[a, b] = Gb[b, a] = nz[s] * is2
        end
      end
      mul!(ws.tmp, T, Gb)
      mul!(Wb, ws.tmp, T)
    end
    # ⟨vecm(Wb), b_j⟩ over the nonzeros of each later column.
    for jl = il:nc
      acc = 0.0
      for t in nzrange(Bs, jl)
        ct = rv[t]; a = ia[ct]; b = ib[ct]
        acc += nz[t] * (a == b ? Wb[a, a] : s2 * Wb[a, b])
      end
      M[cols[il], cols[jl]] += acc
    end
  end
  return M
end

# ──────────────────────────────────────────────────────────────
#  Standard-form conic programs: the equality Schur complement
# ──────────────────────────────────────────────────────────────

# A as a scaled permutation: column j has its single nonzero a[j] in row
# r[j]. `nothing` unless every row and column of A has exactly one nonzero.
function _scaled_permutation(A)
  m, n = size(A)
  m == n || return nothing
  As = sparse(A)
  r = zeros(Int, n); a = zeros(n)
  seen = falses(m)
  rv = rowvals(As); nz = nonzeros(As)
  for j = 1:n
    found = 0
    for t in nzrange(As, j)
      nz[t] == 0 && continue
      found += 1
      found > 1 && return nothing
      r[j] = rv[t]; a[j] = nz[t]
    end
    found == 1 || return nothing
    seen[r[j]] && return nothing
    seen[r[j]] = true
  end
  return (r = r, a = a)
end

"""
    sdpschur_eligible(Q, A, cone_dims) -> Bool

Whether [`kktsolver_sdpschur`](@ref) applies: a linear objective (`Q`
zero), at least one semidefinite cone, and every variable the coordinate
of exactly one cone entry — `A` a scaled permutation, so `Ay ≥_K b` says
`y` (up to order and scale) lies in the cone. This is the standard form a
modelling layer produces for a PSD *variable* (`@variable(X, PSD)`), whose
`k(k+1)/2` entries would otherwise make the dense KKT path cost O(k⁶).
"""
function sdpschur_eligible(Q, A, cone_dims)
  any(cd -> cd[1] == "S", cone_dims) || return false
  iszero(Q) || return false
  return _scaled_permutation(A) !== nothing
end

"""
    kktsolver_sdpschur(Q, A, G, cone_dims; static_reg = 0.0)

KKT solver for standard-form conic programs (see
[`sdpschur_eligible`](@ref)): with `Q = 0` and `A` a scaled permutation,
the Hessian `H = AᵀF⁻¹F⁻ᵀA` has the explicit inverse `A⁻¹FᵀF A⁻ᵀ`, so the
3×3 system reduces to the p×p equality Schur complement

    M = G H⁻¹ Gᵀ,   M_ij = tr(Ĝᵢ T Ĝⱼ T) on a semidefinite block,

with Ĝ = G A⁻¹, assembled by the sparse trace-form kernel and factored by
Cholesky. Per iteration this costs O(p k³ + p² k² + p³) for p equality
rows instead of the O(k⁶) of a dense factorization in the k(k+1)/2
variables, and stores O(p² + k²) besides the data.

`static_reg = δ` regularizes on the dual side, `M + δ·max(1, maxᵢ Mᵢᵢ)·I`,
instead of the `Q + δI` the other solvers receive: a primal shift would
destroy the explicit inverse. Either way the factorization is a
preconditioner whose error `conicIP`'s refinement removes. A Cholesky
that fails on dependent equality rows is retried once with a small dual
shift, and otherwise raises (an `:Error` status from `conicIP`).
"""
function kktsolver_sdpschur(Q, A, G, cone_dims; static_reg = 0.0)

  sdpschur_eligible(Q, A, cone_dims) || throw(ArgumentError(
    "kktsolver_sdpschur needs Q = 0, a semidefinite cone, and A a scaled " *
    "permutation (every variable one cone coordinate); see sdpschur_eligible"))
  n = size(A, 2); m = size(A, 1); p = size(G, 1)
  (; r, a) = _scaled_permutation(A)
  As = sparse(A); Gs = sparse(G)

  # Ĝᵀ = A⁻ᵀ Gᵀ (m × p): row r[j] of Ĝᵀ is row j of Gᵀ divided by a[j].
  (Ig, Jg, Vg) = findnz(Gs)
  Ght = sparse([r[j] for j in Jg], Ig, Vg ./ a[Jg], m, p)
  plan = _gram_plan(Ght, cone_dims)
  M = zeros(p, p)
  tm = zeros(m); tm2 = zeros(m)

  # u ↦ A⁻¹u (cone coordinates → variables) and v ↦ A⁻ᵀv.
  ainv!(y, u) = (@inbounds for j = 1:n; y[j] = u[r[j]] / a[j]; end; y)
  ainvt!(u, v) = (@inbounds for j = 1:n; u[r[j]] = v[j] / a[j]; end; u)

  function solve3x3gen(F, F⁻ᵀ, pt = nothing)

    @phase pt t_kkt_assemble _gram!(M, plan, F)
    dM = maximum(diag(M); init = 0.0)
    dM > 0 || (dM = 1.0)
    δ = static_reg * max(1.0, dM)
    L = @phase pt t_kkt_factor begin
      # A pivot below 1e-14·max Mᵢᵢ is a dependent row that rounding let
      # through, not a factorization to trust: shift and try once more.
      fact = nothing
      for shift in (δ, max(δ, 1e-12 * dM))
        Ms = shift == 0 ? M : M + shift * I
        f = cholesky(Symmetric(Ms); check = false)
        if issuccess(f) && minimum(diag(f.U); init = Inf)^2 > 1e-14 * dM
          fact = f; break
        end
      end
      fact === nothing && throw(LinearAlgebra.PosDefException(0))
      fact
    end
    pt === nothing || (pt.n_kkt_factor += 1)

    # K = FᵀF and K⁻¹ = F⁻¹F⁻ᵀ, on cone vectors.
    Kmul!(out, x)  = (mul!(tm2, F, x); mul_adjoint!(out, F, tm2))
    Kinv!(out, x)  = (mul!(tm2, F⁻ᵀ, x); mul_adjoint!(out, F⁻ᵀ, tm2))
    # H⁻¹u = A⁻¹ K A⁻ᵀ u
    Hinv(u) = (ainvt!(tm, u); Kmul!(tm, copy(tm)); ainv!(zeros(n), tm))

    function solve3x3(bx, by, bz)
      Kinv!(tm, bz)
      h  = bx + As' * tm                  # h = bx + AᵀK⁻¹bz
      t  = Hinv(h)
      w  = p == 0 ? zeros(0) : L \ (Gs * t - by)
      y  = p == 0 ? t : t - Hinv(Gs' * w)
      Kinv!(tm, bz - As * y)              # v = K⁻¹(bz − Ay)
      return (y, w, copy(tm))
    end

    return solve3x3

  end

  return DenseKKTGen(solve3x3gen, nothing)

end
