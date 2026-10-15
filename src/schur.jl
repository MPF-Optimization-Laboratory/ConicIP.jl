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
  Rt::Matrix{Float64}   # Rᵀ, for the stable diagonal RᵀGᵢR
  Z::Matrix{Float64}    # RᵀGᵢR
  # Columns taken by each branch (four-index, rank-two, dense), diagonals
  # recomputed stably, and those whose trace-form value failed the check;
  # cumulative over calls (diagnostics and tests).
  stats::Vector{Int}
end

struct GramPlan
  ncol::Int
  blocks::Vector{_GramBlock}
  sdp::Vector{Union{Nothing,_GramSDP}}
  B::SparseMatrixCSC{Float64,Int}   # for the dense fallback
  suspect::Vector{Int}              # [1]: failed diagonals in the last _gram!
  fallback::Vector{Int}             # [1]: dense fallbacks taken, cumulative
end

# Largest dense W·B (bytes) the fallback of _gram! forms.
const _GRAM_FALLBACK_BYTES = 2^30

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
      push!(sdp, _GramSDP(k, ia, ib, zeros(k, k), zeros(k, k), zeros(k, k), zeros(k, k),
                          zeros(k, k), zeros(k, k), zeros(Int, 5)))
    else
      push!(sdp, nothing)
    end
  end
  return GramPlan(size(B, 2), blocks, sdp, B, [0], [0])
end

"""
    _gram!(M, plan, W::Block) -> M

Overwrite the dense `M` (ncol × ncol) with Bᵀ Wᵀ W B for the `B` of
`plan`. Only the upper triangle is accumulated; it is mirrored at the end,
so `M` is exactly symmetric.

The trace form expands each semidefinite entry into products of entries of
T = RRᵀ, which can cancel catastrophically when the scaling is badly
conditioned (a true 1e-16 came out as 0, or as −7e-16 with Nesterov-Todd
scalings). Every diagonal entry whose terms cancel beyond what rounding
allows is recomputed stably as ‖RᵀGᵢR‖²; if that disagrees with the trace
form, the off-diagonals of its row cannot be trusted either, and `M` is
recomputed as the dense (WB)ᵀ(WB) when W·B fits in
`_GRAM_FALLBACK_BYTES` (otherwise the stable diagonals are kept).
"""
function _gram!(M::Matrix{Float64}, plan::GramPlan, W::Block)
  plan.suspect[1] = 0
  _gram_trace!(M, plan, W)
  if plan.suspect[1] > 0 && 8 * size(plan.B, 1) * plan.ncol <= _GRAM_FALLBACK_BYTES
    Y = W * Matrix(plan.B)
    mul!(M, Y', Y)
    n = size(M, 1)
    @inbounds for j = 1:n, i = j+1:n          # exact symmetry
      M[i, j] = M[j, i]
    end
    plan.fallback[1] += 1
  end
  return M
end

function _gram_trace!(M::Matrix{Float64}, plan::GramPlan, W::Block)
  fill!(M, 0.0)
  for (bi, gb) in enumerate(plan.blocks)
    isempty(gb.cols) && continue
    Blk = W.Blocks[bi]
    if Blk isa DiagBlock
      _gram_diag!(M, gb, Blk.diag)
    elseif Blk isa VecCongurance
      plan.suspect[1] += _gram_sdp!(M, gb, plan.sdp[bi], Blk.R)
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
  copyto!(ws.Rt, R')
  nT2 = sum(abs2, T)                        # ‖T‖_F², for the dense-branch check
  Bs = gb.Bs; rv = rowvals(Bs); nz = nonzeros(Bs); cols = gb.cols
  nc = length(cols)
  nB = nnz(Bs)
  s2 = sqrt(2.0); is2 = 1/s2
  # A diagonal whose terms add up to less than τ times their magnitude may
  # have lost more than ~1e-6 of its value to rounding.
  τ = 1e6 * eps()
  nsuspect = 0
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
      ws.stats[1] += 1
      # Four-index formula: with mat(e_c) = β_c (e_a e_bᵀ + e_b e_aᵀ),
      # β = 1/2 on the diagonal and 1/√2 off it,
      #   tr(mat(e_s) T mat(e_t) T) = 2β_sβ_t (T_{b_s c_t} T_{d_t a_s} + T_{b_s d_t} T_{c_t a_s}).
      for jl = il:nc
        acc = 0.0; mag = 0.0
        for s in rng
          cs = rv[s]; as = ia[cs]; bs = ib[cs]
          gs = nz[s] * (as == bs ? 0.5 : is2)
          for t in nzrange(Bs, jl)
            ct = rv[t]; at = ia[ct]; bt = ib[ct]
            gt = nz[t] * (at == bt ? 0.5 : is2)
            u = T[bs, at] * T[bt, as]; v = T[bs, bt] * T[at, as]
            acc += 2 * gs * gt * (u + v)
            jl == il && (mag += 2 * abs(gs * gt) * (abs(u) + abs(v)))
          end
        end
        if jl == il && acc <= τ * mag
          (acc, bad) = _gram_sdp_diag(ws, Bs, rng, acc)
          nsuspect += bad
        end
        M[cols[il], cols[jl]] += acc
      end
      continue
    end
    if c_rank <= c_dense
      ws.stats[2] += 1
      # Wb = Σ_s g_s β_s (T_a T_bᵀ + T_b T_aᵀ)
      fill!(Wb, 0.0)
      for s in rng
        cs = rv[s]; a = ia[cs]; b = ib[cs]
        g = nz[s] * (a == b ? 0.5 : is2)
        BLAS.ger!(g, view(T, :, a), view(T, :, b), Wb)
        BLAS.ger!(g, view(T, :, b), view(T, :, a), Wb)
      end
      # Magnitude of the diagonal's terms: the four-index sum in absolute
      # values, O(q²) against the O(q k²) above.
      mag = 0.0
      for s in rng, t in rng
        cs = rv[s]; as = ia[cs]; bs = ib[cs]
        ct = rv[t]; at = ia[ct]; bt = ib[ct]
        gs = nz[s] * (as == bs ? 0.5 : is2); gt = nz[t] * (at == bt ? 0.5 : is2)
        mag += 2 * abs(gs * gt) * (abs(T[bs, at] * T[bt, as]) + abs(T[bs, bt] * T[at, as]))
      end
    else
      ws.stats[3] += 1
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
      # |tr(GTGT)| ≤ ‖T‖²‖G‖², and ‖G‖_F = ‖bᵢ‖.
      mag = nT2 * sum(abs2, view(nz, rng))
    end
    # ⟨vecm(Wb), b_j⟩ over the nonzeros of each later column.
    for jl = il:nc
      acc = 0.0
      for t in nzrange(Bs, jl)
        ct = rv[t]; a = ia[ct]; b = ib[ct]
        acc += nz[t] * (a == b ? Wb[a, a] : s2 * Wb[a, b])
      end
      if jl == il && acc <= τ * mag
        (acc, bad) = _gram_sdp_diag(ws, Bs, rng, acc)
        nsuspect += bad
      end
      M[cols[il], cols[jl]] += acc
    end
  end
  return nsuspect
end

# The diagonal contribution ‖RᵀGᵢR‖_F² of a column (vecm is an isometry),
# computed without T: as a sum of rank-two terms in the rows of R when the
# column is sparse, by two products otherwise. Returns it and whether the
# trace-form value `d` disagrees with it beyond 1e-6 relative.
function _gram_sdp_diag(ws::_GramSDP, Bs, rng, d)
  k = ws.k; Z = ws.Z; Rt = ws.Rt; ia = ws.ia; ib = ws.ib
  rv = rowvals(Bs); nz = nonzeros(Bs)
  is2 = 1/sqrt(2.0)
  if 4 * length(rng) <= k
    fill!(Z, 0.0)
    for s in rng
      c = rv[s]; a = ia[c]; b = ib[c]
      g = nz[s] * (a == b ? 0.5 : is2)
      BLAS.ger!(g, view(Rt, :, a), view(Rt, :, b), Z)
      BLAS.ger!(g, view(Rt, :, b), view(Rt, :, a), Z)
    end
  else
    Gb = ws.Gb
    fill!(Gb, 0.0)
    for s in rng
      c = rv[s]; a = ia[c]; b = ib[c]
      if a == b
        Gb[a, a] = nz[s]
      else
        Gb[a, b] = Gb[b, a] = nz[s] * is2
      end
    end
    mul!(ws.tmp, Rt, Gb)                  # RᵀG
    mul!(Z, ws.tmp, Rt')                  # RᵀGR
  end
  ds = sum(abs2, Z)
  ws.stats[4] += 1
  bad = abs(d - ds) > 1e-6 * ds
  ws.stats[5] += bad
  return (ds, Int(bad))
end

# ──────────────────────────────────────────────────────────────
#  Standard-form conic programs: the equality Schur complement
# ──────────────────────────────────────────────────────────────

# Row classes of the split below.
const _ROW_P = 0x01   # eliminated together with its variable
const _ROW_K = 0x02   # kept: its multiplier stays in the reduced system
const _ROW_O = 0x03   # absorbed into the Hessian of the remaining variables

"""
    _schur_split(Q, A, cone_dims) -> NamedTuple or nothing

Partition of the variables into *cone variables* `S`, each the coordinate
of exactly one row of `A` (its *P row*, `A[r, j] = a`, no other variable in
that row), and the rest `R`; and of the rows of `A` into

- `P` — the rows that make `S` a cone coordinate (`A[P, S]` a scaled
  permutation, `A[P, R] = 0`);
- `K` — any other row touching `S` (a bound or a second cone on entries of
  a PSD variable). A second-order or semidefinite block is classified
  whole, since its scaling couples its rows; a nonnegative row on its own;
- `O` — the rows touching only `R`.

A semidefinite block becomes `P` when each of its rows has one nonzero, on
distinct unclaimed variables with zero `Q` rows and columns; its variables
may appear in other rows of `A`, which then become `K`. A nonnegative row
or second-order block becomes `P` only when its variables appear nowhere
else in `A` (a nonnegative or SOC *variable*), so claiming it never moves
a row into `K`. `nothing` when no semidefinite block qualifies.

`pure` marks standard form: every row `P` and every variable in `S`.
"""
function _schur_split(Q, A, cone_dims)
  any(cd -> cd[1] == "S", cone_dims) || return nothing
  m, n = size(A)
  At = sparse(A')                        # column i of Aᵗ is row i of A
  Qs = sparse(Q)
  qfree = trues(n)                       # Q[j, :] = Q[:, j] = 0
  qv = rowvals(Qs); qz = nonzeros(Qs)
  for j = 1:size(Qs, 2), t in nzrange(Qs, j)
    qz[t] == 0 && continue
    qfree[qv[t]] = false; qfree[j] = false
  end
  rv = rowvals(At); nz = nonzeros(At)
  rowcnt = zeros(Int, m); rowcol = zeros(Int, m); rowval = zeros(m)
  colcnt = zeros(Int, n)
  for i = 1:m, t in nzrange(At, i)
    nz[t] == 0 && continue
    rowcnt[i] += 1; rowcol[i] = rv[t]; rowval[i] = nz[t]
    colcnt[rv[t]] += 1
  end
  claimed = falses(n)
  kind = zeros(UInt8, m)
  takes(i, exclusive) = rowcnt[i] == 1 && qfree[rowcol[i]] && !claimed[rowcol[i]] &&
                        (!exclusive || colcnt[rowcol[i]] == 1)
  ranges = Tuple{String,UnitRange{Int}}[]
  off = 0
  for (bt, dim) in cone_dims
    push!(ranges, (bt, (off+1):(off+dim))); off += dim
  end

  # Semidefinite blocks first: they are what the elimination is for.
  for (bt, I) in ranges
    bt == "S" || continue
    all(i -> takes(i, false), I) && allunique(view(rowcol, I)) || continue
    claimed[view(rowcol, I)] .= true
    kind[I] .= _ROW_P
  end
  any(((bt, I),) -> bt == "S" && !isempty(I) && kind[first(I)] == _ROW_P, ranges) ||
    return nothing
  # Nonnegative and second-order variables that appear nowhere else.
  for (bt, I) in ranges
    if bt == "R"
      for i in I
        takes(i, true) || continue
        claimed[rowcol[i]] = true; kind[i] = _ROW_P
      end
    elseif bt == "Q" && all(i -> takes(i, true), I)
      claimed[view(rowcol, I)] .= true
      kind[I] .= _ROW_P
    end
  end
  # Every other row: kept when it touches a cone variable.
  touches(i) = any(t -> nz[t] != 0 && claimed[rv[t]], nzrange(At, i))
  for (bt, I) in ranges
    if bt == "R"
      for i in I
        kind[i] == 0 && (kind[i] = touches(i) ? _ROW_K : _ROW_O)
      end
    elseif !isempty(I) && kind[first(I)] != _ROW_P
      kind[I] .= any(touches, I) ? _ROW_K : _ROW_O
    end
  end
  S = findall(claimed); R = findall(!, claimed)
  return (S = S, R = R, K = findall(==(_ROW_K), kind), kind = kind,
          rowcol = rowcol, rowval = rowval,
          pure = isempty(R) && all(==(_ROW_P), kind))
end

"""
    sdpschur_eligible(Q, A, G, cone_dims) -> Bool

Whether the router sends the problem to [`kktsolver_sdpschur`](@ref).

*Standard form* always qualifies: a linear objective (`Q` zero), at least
one semidefinite cone, and every variable the coordinate of exactly one
cone entry — `A` a scaled permutation, so `Ay ≥_K b` says `y` (up to order
and scale) lies in the cone. This is what a modelling layer produces for a
PSD *variable* (`@variable(X, PSD)`), whose `k(k+1)/2` entries would
otherwise make the dense KKT path cost O(k⁶).

*Mixed form* — a PSD variable alongside free variables, bounds or a second
cone on some of its entries, or a quadratic objective on the other
variables — qualifies when at least one semidefinite block can be
eliminated with its variables (see the split in `src/schur.jl`) and the
reduced system is small: `n_R + p + m_K ≤ n/2`, for `n_R` variables left,
`p` equality rows and `m_K` inequality rows kept. Beyond that bound the
reduced system is no cheaper than [`kktsolver_qr`](@ref)'s; a doubly
nonnegative model (`X ⪰ 0` and `X .≥ 0`, `m_K ≈ n`) is the typical
declined case.
"""
sdpschur_eligible(Q, A, G, cone_dims) =
  _split_eligible(_schur_split(Q, A, cone_dims), size(A, 2), size(G, 1))

_split_eligible(sp, n, p) =
  sp !== nothing && (sp.pure || 2 * (length(sp.R) + p + length(sp.K)) <= n)

# Dense storage of kktsolver_sdpschur for a split: in standard form the
# p×p Schur matrix (sdpschur_bytes); in mixed form the N×N reduced matrix,
# its equilibrated copy and Bunch-Kaufman factor, plus the Gram outputs.
function _sdpschur_bytes(sp, p)
  sp.pure && return sdpschur_bytes(p)
  nR = length(sp.R); nu = p + length(sp.K); N = nR + nu
  return 8 * (3N^2 + nu^2 + nR^2 + nu * nR)
end

# Eligible, and its dense storage fits `budget`: the default router's test.
function _sdpschur_fits(Q, A, G, cone_dims, budget)
  sp = _schur_split(Q, A, cone_dims)
  _split_eligible(sp, size(A, 2), size(G, 1)) || return false
  return _sdpschur_bytes(sp, size(G, 1)) <= budget
end

"""
    kktsolver_sdpschur(Q, A, G, cone_dims; static_reg = 0.0)

KKT solver for conic programs with PSD *variables* (see
[`sdpschur_eligible`](@ref)), which eliminates the cone variables exactly.

**Standard form** (`Q = 0`, `A` a scaled permutation). The Hessian
`H = AᵀF⁻¹F⁻ᵀA` has the explicit inverse `A⁻¹FᵀF A⁻ᵀ`, so the 3×3 system
reduces to the p×p equality Schur complement

    M = G H⁻¹ Gᵀ,   M_ij = tr(Ĝᵢ T Ĝⱼ T) on a semidefinite block,

with Ĝ = G A⁻¹, assembled by the sparse trace-form kernel and factored by
Cholesky. Per iteration this costs O(p k³ + p² k² + p³) for p equality
rows instead of the O(k⁶) of a dense factorization in the k(k+1)/2
variables, and stores O(p² + k²) besides the data.

**Mixed form.** Only the cone variables `y_S` of the eliminated blocks
(rows `P`, `A[P, S]` a scaled permutation) go; the other variables `y_R`,
the equality multipliers `w` and the multipliers `v_K` of the inequality
rows `K` that also touch `y_S` stay. With `C = [G; A_K]`, `u = (w, −v_K)`
and `H_S = A_PSᵀ D_P⁻¹ A_PS` (explicitly invertible as above), the reduced
system is

    ┌ H_R          C_Rᵀ               ┐ ┌ y_R ┐
    └ C_R   −(E + C_S H_S⁻¹ C_Sᵀ)     ┘ └  u  ┘,    E = diag(0, D_K),

where `H_R = Q_RR + A_ORᵀ D_O⁻¹ A_OR` collects the rows `O` that touch only
`y_R` and `D = FᵀF`. Both `E + C_S H_S⁻¹ C_Sᵀ` (one Gram product) and `H_R`
(another) come from the trace-form kernel. The matrix is symmetric but not
quasi-definite in general: `H_R` is singular for a free variable that
enters only equality rows, and `C_S` can lose rank (an equality row on
`y_R` alone) though `[G; A]` has full rank. It is factored by dense
Bunch-Kaufman, which needs only nonsingularity, at (n_R + p + m_K)³/3
flops.

Both matrices are equilibrated to unit diagonal (`D M D`,
`D = |diag|^(-1/2)`, entries with a zero diagonal left unscaled) before
they are factored, so rows that are merely small — a constraint along a
direction the scaling shrinks — are not mistaken for dependent ones.
`static_reg = δ` regularizes on that scale: on the dual side,
`M + δ·diag(M)` (and the `u` block alike), and on the remaining variables
`H_R + δ·diag(H_R)`; a primal shift on `y_S` would destroy the explicit
inverse. Either way the factorization is a preconditioner whose error
`conicIP`'s refinement removes. A factorization that fails, or has a
pivot below `1e-14` on the unit diagonal (dependent rows that rounding let
through), is retried once with a shift of `1e-12` on that scale (primal
side positive, dual side negative), and otherwise raises (an `:Error`
status from `conicIP`).

Called directly, the solver accepts any problem with an eliminable
semidefinite block, whatever the size of the reduced system; the default
router applies the size test of [`sdpschur_eligible`](@ref).
"""
function kktsolver_sdpschur(Q, A, G, cone_dims; static_reg = 0.0)

  sp = _schur_split(Q, A, cone_dims)
  sp === nothing && throw(ArgumentError(
    "kktsolver_sdpschur needs a semidefinite block whose rows each hold one " *
    "variable, on variables with no quadratic objective term (a PSD variable); " *
    "see sdpschur_eligible"))
  sp.pure || return _sdpschur_mixed(Q, A, G, cone_dims, sp; static_reg = static_reg)
  n = size(A, 2); m = size(A, 1); p = size(G, 1)
  # Column j is the coordinate of row r[j], with coefficient a[j].
  r = zeros(Int, n); a = zeros(n)
  for i = 1:m
    r[sp.rowcol[i]] = i; a[sp.rowcol[i]] = sp.rowval[i]
  end
  As = sparse(A); Gs = sparse(G)

  # Ĝᵀ = A⁻ᵀ Gᵀ (m × p): row r[j] of Ĝᵀ is row j of Gᵀ divided by a[j].
  (Ig, Jg, Vg) = findnz(Gs)
  Ght = sparse([r[j] for j in Jg], Ig, Vg ./ a[Jg], m, p)
  plan = _gram_plan(Ght, cone_dims)
  M = zeros(p, p); Ms = zeros(p, p); dsc = zeros(p)
  tm = zeros(m); tm2 = zeros(m)

  # u ↦ A⁻¹u (cone coordinates → variables) and v ↦ A⁻ᵀv.
  ainv!(y, u) = (@inbounds for j = 1:n; y[j] = u[r[j]] / a[j]; end; y)
  ainvt!(u, v) = (@inbounds for j = 1:n; u[r[j]] = v[j] / a[j]; end; u)

  function solve3x3gen(F, F⁻ᵀ, pt = nothing)

    @phase pt t_kkt_assemble _gram!(M, plan, F)
    L = @phase pt t_kkt_factor begin
      # Equilibrate first, Mₛ = D M D with D = diag(M)^(-1/2): the rows of
      # M scale independently with the cone iterate, and a pivot that is
      # small only because its row is (M = diag(1, 1e-16)) is no
      # dependence. On the unit diagonal, a pivot below 1e-14 is a
      # dependent row that rounding let through: shift and try once more.
      @inbounds for i = 1:p
        dsc[i] = M[i, i] > 0 ? 1 / sqrt(M[i, i]) : 1.0
      end
      @inbounds for j = 1:p, i = 1:p
        Ms[i, j] = dsc[i] * M[i, j] * dsc[j]
      end
      fact = nothing
      for shift in (static_reg, max(static_reg, 1e-12))
        shift == 0 || (@inbounds for i = 1:p; Ms[i, i] += shift; end)
        f = cholesky(Symmetric(Ms); check = false)
        if issuccess(f) && minimum(diag(f.U); init = Inf)^2 > 1e-14
          fact = f; break
        end
        shift == 0 || (@inbounds for i = 1:p; Ms[i, i] -= shift; end)
      end
      fact === nothing && throw(LinearAlgebra.PosDefException(0))
      fact
    end
    pt === nothing || (pt.n_kkt_factor += 1)
    d = copy(dsc)          # this factorization's scaling, kept with it

    # K = FᵀF and K⁻¹ = F⁻¹F⁻ᵀ, on cone vectors.
    Kmul!(out, x)  = (mul!(tm2, F, x); mul_adjoint!(out, F, tm2))
    Kinv!(out, x)  = (mul!(tm2, F⁻ᵀ, x); mul_adjoint!(out, F⁻ᵀ, tm2))
    # H⁻¹u = A⁻¹ K A⁻ᵀ u
    Hinv(u) = (ainvt!(tm, u); Kmul!(tm, copy(tm)); ainv!(zeros(n), tm))

    function solve3x3(bx, by, bz)
      Kinv!(tm, bz)
      h  = bx + As' * tm                  # h = bx + AᵀK⁻¹bz
      t  = Hinv(h)
      w  = p == 0 ? zeros(0) : d .* (L \ (d .* (Gs * t - by)))
      y  = p == 0 ? t : t - Hinv(Gs' * w)
      Kinv!(tm, bz - As * y)              # v = K⁻¹(bz − Ay)
      return (y, w, copy(tm))
    end

    return solve3x3

  end

  return DenseKKTGen(solve3x3gen, nothing)

end

# Smallest eigenvalue magnitude over the 1×1 and 2×2 pivots of a
# Bunch-Kaufman factorization: how close the factored matrix came to
# singular.
function _bk_minpivot(f)
  D = f.D
  N = size(D, 1); μ = Inf; i = 1
  while i <= N
    if i < N && D.du[i] != 0
      a = D.d[i]; b = D.du[i]; c = D.d[i+1]
      ρ = abs((a + c) / 2) + hypot((a - c) / 2, b)    # larger |λ|
      μ = min(μ, ρ == 0 ? 0.0 : abs(a * c - b * b) / ρ)
      i += 2
    else
      μ = min(μ, abs(D.d[i])); i += 1
    end
  end
  return μ
end

# Mixed form: eliminate the cone variables y_S only (see kktsolver_sdpschur).
function _sdpschur_mixed(Q, A, G, cone_dims, sp; static_reg = 0.0)

  n = size(A, 2); m = size(A, 1); p = size(G, 1)
  S = sp.S; Rc = sp.R; Kr = sp.K
  nS = length(S); nR = length(Rc); mK = length(Kr)
  nu = p + mK; N = nR + nu
  As = sparse(A); Gs = sparse(G)

  # Eliminated variable S[l] is the coordinate of P row rS[l], coefficient aS[l].
  rS = zeros(Int, nS); aS = zeros(nS)
  pos = zeros(Int, n); pos[S] = 1:nS
  for i = 1:m
    sp.kind[i] == _ROW_P || continue
    l = pos[sp.rowcol[i]]; rS[l] = i; aS[l] = sp.rowval[i]
  end

  # C = [G; A_K], split by columns. Its S part is only applied; its R part
  # is a constant block of the reduced matrix.
  Cm  = [Gs; As[Kr, :]]
  CS  = Cm[:, S]; CSt = sparse(CS')
  CRd = Matrix(Cm[:, Rc])

  # E + C_S H_S⁻¹ C_Sᵀ = B_uᵀ FᵀF B_u with B_u (cone coordinates × u): the
  # columns of Ĉᵀ = A_PS⁻ᵀ C_Sᵀ on the P rows and the identity on the K
  # rows. The two row sets share no cone block (a P block is whole, and a
  # K row of a nonnegative block is decoupled), so no cross term appears.
  (Ic, Jc, Vc) = findnz(CS)
  Bu = sparse([rS[Jc]; Kr], [Ic; p .+ (1:mK)], [Vc ./ aS[Jc]; ones(mK)], m, nu)
  plan_u = _gram_plan(Bu, cone_dims)
  # H_R − Q_RR = A_ORᵀ F⁻¹F⁻ᵀ A_OR: the R columns on the O rows only.
  BR = sparse(Diagonal(Float64.(sp.kind .== _ROW_O))) * As[:, Rc]
  dropzeros!(BR)
  plan_R = _gram_plan(BR, cone_dims)
  QR = Matrix{Float64}(sparse(Q)[Rc, Rc])

  Mu = zeros(nu, nu); HR = zeros(nR, nR); Kred = zeros(N, N)
  Ks = zeros(N, N); dsc = zeros(N)
  tm = zeros(m); tm2 = zeros(m); tm3 = zeros(m)
  iR = 1:nR; iu = nR+1:N

  function solve3x3gen(F, F⁻ᵀ, pt = nothing)

    @phase pt t_kkt_assemble begin
      _gram!(Mu, plan_u, F)
      _gram!(HR, plan_R, F⁻ᵀ)
      HR .+= QR
      Kred[iR, iR] .= HR
      Kred[iu, iR] .= CRd
      Kred[iR, iu] .= CRd'
      Kred[iu, iu] .= .-Mu
    end
    fact = @phase pt t_kkt_factor begin
      # Equilibrate to unit |diagonal| (see kktsolver_sdpschur), then
      # factor; the pivot test and the shifts act on that scale.
      @inbounds for i = 1:N
        dsc[i] = Kred[i, i] != 0 ? 1 / sqrt(abs(Kred[i, i])) : 1.0
      end
      @inbounds for j = 1:N, i = 1:N
        Ks[i, j] = dsc[i] * Kred[i, j] * dsc[j]
      end
      fact = nothing
      for σ in (static_reg, max(static_reg, 1e-12))
        # A fresh copy: an earlier solve3x3 may still hold its factors.
        Kf = copy(Ks)
        @inbounds for i in iR; Kf[i, i] += σ; end
        @inbounds for i in iu; Kf[i, i] -= σ; end
        f = bunchkaufman!(Symmetric(Kf, :U), false; check = false)
        if issuccess(f) && _bk_minpivot(f) > 1e-14
          fact = f; break
        end
      end
      fact === nothing && throw(LinearAlgebra.SingularException(0))
      fact
    end
    pt === nothing || (pt.n_kkt_factor += 1)
    d = copy(dsc)          # this factorization's scaling, kept with it

    # D = FᵀF and D⁻¹ = F⁻¹F⁻ᵀ, on cone vectors.
    Dmul!(out, x) = (mul!(tm2, F, x); mul_adjoint!(out, F, tm2))
    Dinv!(out, x) = (mul!(tm2, F⁻ᵀ, x); mul_adjoint!(out, F⁻ᵀ, tm2))
    # H_S⁻¹ g = A_PS⁻¹ D_P A_PS⁻ᵀ g; the P rows are whole blocks or
    # decoupled nonnegative rows, so D applied to a vector supported on P
    # is D_P there.
    function HSinv(g)
      fill!(tm3, 0.0)
      @inbounds for l = 1:nS; tm3[rS[l]] = g[l] / aS[l]; end
      Dmul!(tm, tm3)
      return [tm[rS[l]] / aS[l] for l = 1:nS]
    end

    function solve3x3(bx, by, bz)
      Dinv!(tm, bz)
      tm[Kr] .= 0.0                       # D⁻¹bz on the P and O rows
      g  = bx + As' * tm
      gS = g[S]
      hS = HSinv(gS)
      rhs = [g[Rc]; [by; bz[Kr]] - CS * hS]
      z  = d .* (fact \ (d .* rhs))
      u  = z[iu]
      y  = zeros(n)
      y[Rc] = z[iR]
      y[S]  = HSinv(gS - CSt * u)
      Dinv!(tm, bz - As * y)              # v = D⁻¹(bz − Ay) off K
      v = copy(tm)
      v[Kr] = -u[p+1:end]
      return (y, u[1:p], v)
    end

    return solve3x3

  end

  return DenseKKTGen(solve3x3gen, nothing)

end
