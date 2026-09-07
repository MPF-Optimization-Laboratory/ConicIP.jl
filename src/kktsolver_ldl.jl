# ──────────────────────────────────────────────────────────────
#  Sparse quasi-definite LDLᵀ KKT solver (roadmap tranche 1)
# ──────────────────────────────────────────────────────────────
#
# The 3×3 system every kktsolver must solve,
#
#     ┌             ┐ ┌   ┐   ┌    ┐
#     │ Q   Gᵀ  −Aᵀ │ │ x │ = │ bx │
#     │ G           │ │ y │   │ by │
#     │ A       FᵀF │ │ z │   │ bz │
#     └             ┘ └   ┘   └    ┘
#
# is not symmetric as written. Negating the third block row (and bz)
# gives the symmetric quasi-definite matrix
#
#     K_δ = ┌ Q + δ_p I     Gᵀ        −Aᵀ           ┐
#           │ G            −δ_e I      0            │
#           └ −A            0        −(FᵀF + δ_c I) ┘
#
# whose (1,1) block is positive definite and whose remaining diagonal
# block is negative definite for any δ_p, δ_e > 0 and δ_c ≥ 0. Such a
# matrix has an LDLᵀ factorization with diagonal D for *every* symmetric
# permutation (Vanderbei 1995), so a fill-reducing ordering can be fixed
# once and the numeric factorization repeated each iteration in place.
#
# The scaling block FᵀF is block diagonal over the cones:
#
#   R   : FᵀF = Diagonal(d²)                     — k diagonal entries
#   Q   : FᵀF = D² + u uᵀ − v vᵀ                 — see `soc_uv` below
#   S   : FᵀF dense k×k                          — small blocks only
#
# Large second-order cones are *lifted*: with a = uᵀz and b = vᵀz the
# block −(D² + uuᵀ − vvᵀ) becomes
#
#     ┌ −D²    −u    v ┐
#     │ −uᵀ     1    0 │        (a-row: −uᵀz + a = 0,  b-row: vᵀz − b = 0)
#     └  vᵀ     0   −1 ┘
#
# which is quasi-definite in its own right (the +1 pivot joins the
# positive group, and the Schur complement D² − vvᵀ of the −1 pivot is
# positive definite for the NT scaling, where D² = β²I and
# vᵀv = β² − σ₂² < β² with σ₂ the smaller eigenvalue of the scaling on
# the cone's 2-dimensional subspace). Storage per cone is 3k + 2 entries
# instead of k(k+1)/2.
#
# The pattern never changes between iterations, including at the
# identity-scaled initial point: every entry is stored structurally, and
# only values are rewritten before each numeric refactorization.

using QDLDL: qdldl, update_values!, refactor!, solve!

"""
    soc_uv(Blk::SymWoodbury) -> (d², u, v)

Decompose the square of a diagonal-plus-rank-one scaling `F = D + c·w·wᵀ`
(the second-order-cone Nesterov–Todd scaling from `nestod_soc`, with
`D = Diagonal([−β; β; …; β])` and `c = 1`) as

    FᵀF = D² + u uᵀ − v vᵀ,     uᵀv = 0,

returning the vector `d² = diag(D)²` and the two vectors. Writing
`w̃ = √c·w`, `FᵀF − D² = B M Bᵀ` with `B = [D w̃  w̃]` and
`M = [0 1; 1 w̃ᵀw̃]`; `det M = −1`, so the rank-two term has exactly one
positive and one negative eigenvalue, obtained from the 2×2 symmetric
matrix `T^{1/2} M T^{1/2}`, `T = BᵀB`. For the NT scaling the negative
part satisfies `vᵀv < β²` (its eigenvalues on the cone's two-dimensional
subspace multiply to `β²`), which is what keeps the lifted system
quasi-definite.
"""
function soc_uv(Blk::SymWoodbury)
  dvec = Blk.A.diag
  c    = Blk.D isa Number ? Blk.D : Blk.D[1, 1]
  w    = vec(Blk.B) .* sqrt(c)
  k    = length(w)
  d²   = dvec .^ 2
  u    = zeros(k); v = zeros(k)
  ww   = dot(w, w)
  ww == 0 && return (d², u, v)
  Dw   = dvec .* w
  # T = BᵀB for B = [Dw w]
  t11 = dot(Dw, Dw); t12 = dot(Dw, w); t22 = ww
  T   = Symmetric([t11 t12; t12 t22])
  ET  = eigen(T)
  if ET.values[1] <= 1e-14 * ET.values[2]
    # Dw ∥ w: the rank-two term collapses to (2α + ww)·wwᵀ with α = Dwᵀw/wᵀw
    γ = 2 * t12 / ww + ww
    if γ >= 0
      u .= sqrt(γ) .* w
    else
      v .= sqrt(-γ) .* w
    end
    return (d², u, v)
  end
  Vt  = ET.vectors
  Th  = Vt * Diagonal(sqrt.(ET.values)) * Vt'        # T^{1/2}
  Tih = Vt * Diagonal(1 ./ sqrt.(ET.values)) * Vt'   # T^{-1/2}
  M   = [0.0 1.0; 1.0 ww]
  EC  = eigen(Symmetric(Th * M * Th))                 # λ₋ < 0 < λ₊
  P   = Tih * EC.vectors                              # B*P has orthonormal columns
  λ₋  = EC.values[1]; λ₊ = EC.values[2]
  su  = sqrt(max(λ₊, 0.0)); sv = sqrt(max(-λ₋, 0.0))
  @inbounds for i in 1:k
    u[i] = su * (Dw[i] * P[1, 2] + w[i] * P[2, 2])
    v[i] = sv * (Dw[i] * P[1, 1] + w[i] * P[2, 1])
  end
  return (d², u, v)
end

# The identity-scaled initial point hands every cone a Diagonal block.
soc_uv(Blk::Diagonal) = (Blk.diag .^ 2, zeros(length(Blk.diag)), zeros(length(Blk.diag)))

# Dense k×k value of FᵀF for a scaling block (small SOC and SDP blocks).
_dense_FtF(Blk::Diagonal)      = Diagonal(Blk.diag .^ 2)
function _dense_FtF(Blk::SymWoodbury)
  W = Matrix(Blk.A) + vec(Blk.B) * (Blk.D isa Number ? Blk.D : Blk.D[1, 1]) * vec(Blk.B)'
  return W'W
end
function _dense_FtF(Blk::VecCongurance)
  W = Matrix(Blk)
  return W'W
end
_dense_FtF(Blk::AbstractMatrix) = Blk'Blk

# Position of the structural entry (i, j) in K.nzval (K upper triangular).
function _nzindex(K::SparseMatrixCSC, i::Int, j::Int)
  r = @view K.rowval[K.colptr[j]:K.colptr[j+1]-1]
  t = searchsortedfirst(r, i)
  (t <= length(r) && r[t] == i) || error("structural entry ($i,$j) missing")
  return K.colptr[j] - 1 + t
end

# y = K x for K stored as its upper triangle.
function _symmul!(y, K::SparseMatrixCSC, x)
  fill!(y, 0.0)
  rows = K.rowval; vals = K.nzval
  @inbounds for j in 1:size(K, 2)
    xj = x[j]
    for t in K.colptr[j]:K.colptr[j+1]-1
      i = rows[t]; v = vals[t]
      y[i] += v * xj
      if i != j
        y[j] += v * x[i]
      end
    end
  end
  return y
end

"""
    kktsolver_ldl(Q, A, G, cone_dims;
                  static_reg = 1e-8, cone_reg = 0.0,
                  dynamic_eps = 1e-13, dynamic_delta = 2e-7,
                  lift_min = 6, refine_steps = 2, refine_tol = 1e-13)

Sparse LDLᵀ KKT solver for the quasi-definite form of the 3×3 system.
The upper triangle of

    K_δ = [ Q + δ_p I   Gᵀ   −Aᵀ ;  G   −δ_e I   0 ;  −A   0   −(FᵀF + δ_c I) ]

(second-order cones of dimension at least `lift_min` lifted to
`diagonal + two columns + two pivots`, smaller SOC and SDP blocks dense)
is assembled once with every entry stored structurally, ordered by AMD,
and analysed symbolically once; each iteration rewrites the scaling
entries and refactorizes numerically in place (QDLDL.jl).

Regularization: `static_reg` is `δ_p = δ_e`, `cone_reg` is `δ_c`; the
factorization also applies QDLDL's dynamic regularization, replacing any
pivot whose sign disagrees with the quasi-definite pattern or whose
magnitude is below `dynamic_eps` by `±dynamic_delta`. Each solve is then
refined against the *unregularized* matrix (`δ = 0`) for up to
`refine_steps` corrections or until the residual is below
`refine_tol · (1 + ‖rhs‖)`, so the perturbation acts as a
preconditioner rather than a change of problem. `conicIP`'s own
refinement against the 4×4 system runs on top of this.

No rank assumption on `G`: dependent equality rows are handled by
`δ_e` and the refinement. Semidefinite blocks are supported through a
dense k×k block, which is O(k⁴) in memory and not meant for large `k`;
[`choose_kktsolver`](@ref) keeps SDP problems on [`kktsolver_qr`](@ref).

Returns `solve3x3gen(F, F⁻ᵀ)` per the `conicIP` KKT solver interface.
"""
function kktsolver_ldl(Q, A, G, cone_dims;
                       static_reg::Float64 = 1e-8,
                       cone_reg::Float64 = 0.0,
                       dynamic_eps::Float64 = 1e-13,
                       dynamic_delta::Float64 = 2e-7,
                       lift_min::Int = 6,
                       refine_steps::Int = 2,
                       refine_tol::Float64 = 1e-13,
                       pattern = nothing)

  # `pattern` lets default_kktsolver hand over the pattern it already
  # analysed; it must have been built with the same lift_min and static_reg.
  pat = pattern !== nothing ? pattern :
        _ldl_pattern(Q, A, G, cone_dims; lift_min = lift_min,
                     δp = static_reg, δe = static_reg)
  (; K, kinds, ranges, blk_idx, Dsigns, n, m, p, N, oz, oa) = pat
  δp = static_reg; δe = static_reg; δc = cone_reg

  # Symbolic analysis once (AMD ordering inside qdldl); numeric
  # factorizations happen in refactor! after each value update.
  Fact = qdldl(K; logical = true, Dsigns = Dsigns,
               regularize_eps = dynamic_eps, regularize_delta = dynamic_delta)

  # Diagonal shifts to remove when evaluating the unregularized residual
  shift = zeros(N)
  shift[1:n] .= δp
  shift[n+1:n+p] .= -δe
  shift[oz+1:oz+m] .= -δc

  rhs  = zeros(N); sol = zeros(N); res = zeros(N); tmp = zeros(N)
  vbuf = Float64[]

  function solve3x3gen(F::Block, F⁻ᵀ)

    # Rewrite the scaling entries (in K, for the residual, and in the
    # factorization's internal copy), then refactorize numerically.
    for (bi, (kind, I)) in enumerate(zip(kinds, ranges))
      Blk = F.Blocks[bi]
      idx = blk_idx[bi]
      if kind == :diag
        d = Blk isa Diagonal ? Blk.diag : diag(Matrix(Blk))
        resize!(vbuf, length(idx))
        @inbounds for t in eachindex(idx)
          vbuf[t] = -(d[t]^2) - δc
        end
      elseif kind == :dense
        M = _dense_FtF(Blk)
        k = length(I)
        resize!(vbuf, length(idx))
        t = 0
        for b in 1:k, a in 1:b
          t += 1
          vbuf[t] = -M[a, b] - (a == b ? δc : 0.0)
        end
      else
        (d², u, v) = soc_uv(Blk)
        k = length(I)
        resize!(vbuf, length(idx))
        @inbounds for i in 1:k
          vbuf[i]      = -d²[i] - δc
          vbuf[k + i]  = -u[i]
          vbuf[2k + i] =  v[i]
        end
        vbuf[3k + 1] =  1.0
        vbuf[3k + 2] = -1.0
      end
      K.nzval[idx] .= vbuf
      update_values!(Fact, idx, vbuf)
    end
    refactor!(Fact)

    function solve3x3(bx, by, bz)
      rhs[1:n] .= bx
      rhs[n+1:n+p] .= by
      rhs[oz+1:oz+m] .= .-bz
      rhs[oa+1:N] .= 0.0
      sol .= rhs
      solve!(Fact, sol)
      # Refinement against the unregularized matrix K₀ = K_δ − Diagonal(shift)
      nrhs = norm(rhs)
      for _ in 1:refine_steps
        _symmul!(res, K, sol)
        @inbounds for i in 1:N
          res[i] = rhs[i] - (res[i] - shift[i] * sol[i])
        end
        norm(res) <= refine_tol * (1 + nrhs) && break
        tmp .= res
        solve!(Fact, tmp)
        sol .+= tmp
      end
      return (sol[1:n], sol[n+1:n+p], sol[oz+1:oz+m])
    end

    return solve3x3

  end

  return solve3x3gen

end

# _ldl_pattern(Q, A, G, cone_dims; lift_min = 6, δp = 1e-8, δe = 1e-8)
#
# Assemble the upper triangle of the quasi-definite KKT matrix with
# placeholder scaling entries, and the index maps kktsolver_ldl uses to
# rewrite them. Also used by choose_kktsolver to estimate the
# factorization's fill before committing to a solver.
function _ldl_pattern(Q, A, G, cone_dims; lift_min::Int = 6,
                      δp::Float64 = 1e-8, δe::Float64 = 1e-8)

  n = size(Q, 1); m = size(A, 1); p = size(G, 1)
  Qs = sparse(Q); As = sparse(A); Gs = sparse(G)

  ranges = cum_range([cd[2] for cd in cone_dims])
  kinds  = Symbol[]              # :diag, :dense, :lift per block
  for (cd, I) in zip(cone_dims, ranges)
    k = length(I)
    kind = cd[1] == "R" ? :diag :
           (cd[1] == "Q" && k >= lift_min) ? :lift : :dense
    push!(kinds, kind)
  end
  nlift = count(==(:lift), kinds)
  N = n + p + m + 2nlift
  oz = n + p                      # offset of the z block
  oa = n + p + m                  # offset of the auxiliary block

  # ── Upper-triangular pattern with placeholder values ──
  Ii = Int[]; Jj = Int[]; Vv = Float64[]
  function put!(i, j, v)
    push!(Ii, i); push!(Jj, j); push!(Vv, v)
  end
  rows, cols, vals = findnz(Qs)
  for t in eachindex(rows)
    rows[t] <= cols[t] && put!(rows[t], cols[t], vals[t])
  end
  for i in 1:n
    put!(i, i, δp)                # summed onto Qᵢᵢ by sparse()
  end
  rows, cols, vals = findnz(Gs)
  for t in eachindex(rows)
    put!(cols[t], n + rows[t], vals[t])
  end
  for r in 1:p
    put!(n + r, n + r, -δe)
  end
  rows, cols, vals = findnz(As)
  for t in eachindex(rows)
    put!(cols[t], oz + rows[t], -vals[t])
  end
  ilift = 0
  for (kind, I) in zip(kinds, ranges)
    if kind == :diag
      for i in I; put!(oz + i, oz + i, -1.0); end
    elseif kind == :dense
      for b in I, a in first(I):b
        put!(oz + a, oz + b, a == b ? -1.0 : 1.0)
      end
    else
      ilift += 1
      ca = oa + 2ilift - 1; cb = oa + 2ilift
      for i in I
        put!(oz + i, oz + i, -1.0)
        put!(oz + i, ca, -1.0)
        put!(oz + i, cb,  1.0)
      end
      put!(ca, ca,  1.0)
      put!(cb, cb, -1.0)
    end
  end
  K = sparse(Ii, Jj, Vv, N, N)

  # ── Index maps from each block's entries into K.nzval ──
  blk_idx = Vector{Vector{Int}}(undef, length(kinds))
  ilift = 0
  for (bi, (kind, I)) in enumerate(zip(kinds, ranges))
    if kind == :diag
      blk_idx[bi] = [_nzindex(K, oz + i, oz + i) for i in I]
    elseif kind == :dense
      idx = Int[]
      for b in I, a in first(I):b
        push!(idx, _nzindex(K, oz + a, oz + b))
      end
      blk_idx[bi] = idx
    else
      ilift += 1
      ca = oa + 2ilift - 1; cb = oa + 2ilift
      idx = Int[]
      for i in I; push!(idx, _nzindex(K, oz + i, oz + i)); end
      for i in I; push!(idx, _nzindex(K, oz + i, ca)); end
      for i in I; push!(idx, _nzindex(K, oz + i, cb)); end
      push!(idx, _nzindex(K, ca, ca)); push!(idx, _nzindex(K, cb, cb))
      blk_idx[bi] = idx
    end
  end

  # Expected pivot signs of the quasi-definite pattern
  Dsigns = Vector{Int}(undef, N)
  Dsigns[1:n] .= 1
  Dsigns[n+1:oa] .= -1
  for j in 1:nlift
    Dsigns[oa + 2j - 1] = 1
    Dsigns[oa + 2j] = -1
  end

  return (; K, kinds, ranges, blk_idx, Dsigns, n, m, p, N, oz, oa, nlift)

end
