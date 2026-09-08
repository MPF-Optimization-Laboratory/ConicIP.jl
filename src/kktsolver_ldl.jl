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

using QDLDL: qdldl, update_values!, refactor!, solve!,
             regularized_entries, positive_inertia
using AMD: amd

"""
    soc_uv(Blk::SymWoodbury) -> (d², u, v)

Decompose the square of a diagonal-plus-rank-one scaling `F = D + c·w·wᵀ`
(the second-order-cone Nesterov–Todd scaling from `nestod_soc`, with
`D = Diagonal([−β; β; …; β])` and `c = 1`) as

    FᵀF = D² + u uᵀ − v vᵀ,     uᵀv = 0,

returning the vector `d² = diag(D)²` and the two vectors. Writing
`w̃ = √c·w`, `FᵀF − D² = B M Bᵀ` with `B = [D w̃  w̃]` and
`M = [0 1; 1 w̃ᵀw̃]`; `det M = −1`, so the rank-two term has exactly one
positive and one negative eigenvalue when `B` has full column rank.

The construction goes through a thin Householder QR `B = Qb R` (`Qb`
k×2 with orthonormal columns, `R` 2×2 upper triangular), so that
`B M Bᵀ = Qb (R M Rᵀ) Qbᵀ`. The 2×2 symmetric matrix `R M Rᵀ = V Λ Vᵀ`
is eigendecomposed and `u = √λ₊·Qb V[:, +]`, `v = √(−λ₋)·Qb V[:, −]`.
Because `Qb V` has orthonormal columns, `uᵀv = 0` to rounding. Nothing
is inverted: when `Dw` and `w̃` are nearly (or exactly) parallel the
second row of `R` is ~0 and the formula degrades gracefully to the
rank-one term, with reconstruction error O(ε‖B‖²‖M‖) regardless of the
conditioning of `B`. (The earlier route through `T = BᵀB` and `T^{±1/2}`
squared the condition number and needed a scale-sensitive degeneracy
branch that could discard part of the rank-two term.)

For the NT scaling the negative part satisfies `vᵀv < β²` (its
eigenvalues on the cone's two-dimensional subspace multiply to `β²`),
which is what keeps the lifted system quasi-definite.
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
  B    = [dvec .* w  w]                              # k×2, columns Dw and w
  QRB  = qr(B)
  Qb   = Matrix(QRB.Q)                                # thin k×min(k,2)
  R    = QRB.R
  M    = [0.0 1.0; 1.0 ww]
  E    = eigen(Symmetric(R * M * R'))                 # ascending: λ₋ ≤ λ₊
  λ₋   = E.values[1]; λ₊ = E.values[end]
  # det(R M Rᵀ) = −det(R)² ≤ 0, so the exact eigenvalues have opposite
  # signs (or one is zero); clamp what rounding may push across zero.
  su   = sqrt(max(λ₊, 0.0)); sv = sqrt(max(-λ₋, 0.0))
  su > 0 && mul!(u, Qb, E.vectors[:, end], su, 0.0)
  sv > 0 && mul!(v, Qb, E.vectors[:, 1],   sv, 0.0)
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
    LDLDiagnostics

Per-factorization and per-solve counters of [`kktsolver_ldl`](@ref),
reachable through `kkt_diagnostics(solve3x3)` on the object its
`solve3x3gen` returns.

- `repaired` -- pivots QDLDL's dynamic regularization replaced in the
  current factorization; `repaired_total` sums them over the solve
- `pos_inertia` -- positive pivots of the current factorization. Recorded
  only: with `Dsigns` QDLDL forces every pivot's sign, so the count
  always matches the quasi-definite pattern and detects nothing
- `δp`, `δe`, `δc` -- static shifts in effect (`δp`, `δe` grow under the
  retry policy; `δc` is fixed)
- `refactors` -- shift bumps applied to the current factorization;
  `refactors_total` sums them over the solve
- `last_residual` -- unregularized residual norm `‖rhs − K₀x‖` of the last
  `solve3x3` return
"""
mutable struct LDLDiagnostics
  repaired        :: Int
  pos_inertia     :: Int
  δp              :: Float64
  δe              :: Float64
  δc              :: Float64
  refactors       :: Int
  refactors_total :: Int
  repaired_total  :: Int
  last_residual   :: Float64
end
LDLDiagnostics(δp, δe, δc) = LDLDiagnostics(0, 0, δp, δe, δc, 0, 0, 0, NaN)

# The callable `kktsolver_ldl` hands back from `solve3x3gen`: the solve
# closure, the regularization bump for tests and diagnosis, and the shared
# diagnostics record (one per `kktsolver_ldl` instance).
struct LDLSolve3x3{S, B}
  solve :: S
  bump! :: B
  diag  :: LDLDiagnostics
end
(s::LDLSolve3x3)(bx, by, bz) = s.solve(bx, by, bz)
kkt_diagnostics(s::LDLSolve3x3) = s.diag

"""
    kktsolver_ldl(Q, A, G, cone_dims;
                  static_reg = 1e-8, cone_reg = 0.0,
                  dynamic_eps = 1e-13, dynamic_delta = 2e-7,
                  lift_min = 6, refine_steps = 2, refine_tol = 1e-13,
                  retry_max = 0, retry_factor = 10.0,
                  shift_floor = 1e-8, shift_max = 1e-4)

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
preconditioner rather than a change of problem. The residual is
evaluated after every correction; a correction that does not reduce it
is discarded and ends the refinement, so the returned solution is the
best one seen and never worse than the unrefined solve. The tolerance
is a target, not a guarantee. `conicIP`'s own refinement against the
4×4 system runs on top of this.

Retry (off by default, `retry_max = 0`): when a solve is still above the
refinement tolerance and either the factorization repaired a pivot or
the first correction failed to reduce the residual, the static shifts
are bumped, `δ ← min(max(δ·retry_factor, shift_floor), shift_max)` for
`δ_p` and `δ_e` (a zero base shift starts at `shift_floor`), the matrix
is refactorized, and the same right-hand side is solved again; the
better of the two results by unregularized residual is returned. At
most `retry_max` bumps per factorization. Bumped shifts last for the
remaining solves with that factorization only: the next `solve3x3gen`
call restores the base shifts. `δ_c` and `dynamic_delta` are never
bumped.

Diagnostics: the object `solve3x3gen` returns is callable as before and
carries an `LDLDiagnostics` record, reachable through
`ConicIP.kkt_diagnostics(solve3x3)`; `conicIP` prints its counts in the
verbose `kkt` column and sums them into `Solution.kkt_repaired` and
`Solution.kkt_refactors`.

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
                       retry_max::Int = 0,
                       retry_factor::Float64 = 10.0,
                       shift_floor::Float64 = 1e-8,
                       shift_max::Float64 = 1e-4,
                       pattern = nothing)

  # `pattern` lets default_kktsolver and cached_kktsolver_ldl hand over the
  # pattern they already analysed; it must have been built with the same
  # lift_min. The static shifts are written onto the (1,1) and (2,2)
  # diagonals below, so the pattern's own placeholder shifts do not matter.
  pat = pattern !== nothing ? pattern :
        _ldl_pattern(Q, A, G, cone_dims; lift_min = lift_min)
  (; K, kinds, ranges, blk_idx, Dsigns, n, m, p, N, oz, oa,
     qdiag_idx, qdiag, ediag_idx) = pat
  δp0 = static_reg; δe0 = static_reg; δc = cone_reg
  δp = δp0; δe = δe0

  # Base static shifts in K before the symbolic analysis copies it.
  K.nzval[qdiag_idx] .= qdiag .+ δp
  K.nzval[ediag_idx] .= -δe

  # Symbolic analysis once, with the ordering the pattern carries; numeric
  # factorizations happen in refactor! after each value update.
  Fact = qdldl(K; perm = pat.perm, logical = true, Dsigns = Dsigns,
               regularize_eps = dynamic_eps, regularize_delta = dynamic_delta)

  # Diagonal shifts to remove when evaluating the unregularized residual
  shift = zeros(N)
  shift[oz+1:oz+m] .= -δc

  diag = LDLDiagnostics(δp, δe, δc)

  # Write the static shifts in effect into K, into the factorization's
  # internal copy, and into `shift`, so that K − Diagonal(shift) is always
  # the exact unregularized matrix.
  function write_static!()
    @inbounds for t in 1:n
      K.nzval[qdiag_idx[t]] = qdiag[t] + δp
    end
    K.nzval[ediag_idx] .= -δe
    update_values!(Fact, qdiag_idx, view(K.nzval, qdiag_idx))
    update_values!(Fact, ediag_idx, view(K.nzval, ediag_idx))
    shift[1:n] .= δp
    shift[n+1:n+p] .= -δe
    diag.δp = δp; diag.δe = δe
    return nothing
  end
  write_static!()

  rhs  = zeros(N); sol = zeros(N); res = zeros(N); tmp = zeros(N)
  cand = zeros(N); keep = zeros(N)
  vbuf = Float64[]

  # res = rhs − K₀ x for the unregularized K₀ = K_δ − Diagonal(shift);
  # returns ‖res‖.
  function residual!(x)
    _symmul!(res, K, x)
    @inbounds for i in 1:N
      res[i] = rhs[i] - (res[i] - shift[i] * x[i])
    end
    return norm(res)
  end

  # Numeric refactorization, recording what QDLDL did to the pivots.
  function numeric_factor!()
    refactor!(Fact)
    diag.repaired        = regularized_entries(Fact)
    diag.repaired_total += diag.repaired
    diag.pos_inertia     = positive_inertia(Fact)
    return nothing
  end

  # One regularization bump: raise δp and δe geometrically (a zero base
  # shift starts at shift_floor), cap at shift_max, and refactorize. The
  # bumped shifts stay in effect until the next solve3x3gen call.
  function bump!()
    δp = min(max(δp * retry_factor, shift_floor), shift_max)
    δe = min(max(δe * retry_factor, shift_floor), shift_max)
    write_static!()
    numeric_factor!()
    diag.refactors       += 1
    diag.refactors_total += 1
    return nothing
  end

  function solve3x3gen(F::Block, F⁻ᵀ)

    # Every factorization starts from the base shifts: a retry in the
    # previous iteration does not carry its bumps forward.
    if δp != δp0 || δe != δe0
      δp = δp0; δe = δe0
      write_static!()
    end
    diag.refactors = 0

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
    numeric_factor!()

    # Solve for the right-hand side loaded in `rhs`, leaving the result in
    # `sol`. Refinement against the unregularized matrix K₀: each candidate
    # sol + K_δ⁻¹ res is evaluated before it is accepted: refinement
    # contracts only when ‖I − K_δ⁻¹K₀‖ < 1, and a correction that
    # increases the residual is the evidence that it does not, so it is
    # discarded and the loop ends with the best iterate seen. At most
    # refine_steps + 1 residual evaluations. Returns the best residual, the
    # tolerance, and whether the *first* correction already failed to
    # contract while the residual was still above the tolerance.
    function solve_loaded!()
      sol .= rhs
      solve!(Fact, sol)
      rtol  = refine_tol * (1 + norm(rhs))
      rbest = residual!(sol)
      noncontract = false
      for k in 1:refine_steps
        rbest <= rtol && break
        tmp .= res
        solve!(Fact, tmp)
        cand .= sol .+ tmp
        rcand = residual!(cand)
        if !(rcand < rbest)
          noncontract = (k == 1)
          break
        end
        sol .= cand
        rbest = rcand
      end
      return (rbest, rtol, noncontract)
    end

    function solve3x3(bx, by, bz)
      rhs[1:n] .= bx
      rhs[n+1:n+p] .= by
      rhs[oz+1:oz+m] .= .-bz
      rhs[oa+1:N] .= 0.0
      (rbest, rtol, noncontract) = solve_loaded!()
      # Bounded retry: a solve that misses the tolerance on a factorization
      # that repaired pivots (or whose refinement did not contract at all)
      # is repeated with larger static shifts; the better result by
      # unregularized residual is kept.
      while rbest > rtol && (diag.repaired > 0 || noncontract) &&
            diag.refactors < retry_max
        keep .= sol; rkeep = rbest
        bump!()
        (rnew, _, noncontract) = solve_loaded!()
        if rnew < rkeep
          rbest = rnew
        else
          sol .= keep; rbest = rkeep
        end
      end
      diag.last_residual = rbest
      return (sol[1:n], sol[n+1:n+p], sol[oz+1:oz+m])
    end

    return LDLSolve3x3(solve3x3, bump!, diag)

  end

  return solve3x3gen

end

# _ldl_pattern(Q, A, G, cone_dims; lift_min = 6, δp = 1e-8, δe = 1e-8)
#
# Assemble the upper triangle of the quasi-definite KKT matrix with
# placeholder scaling entries, and the index maps kktsolver_ldl uses to
# rewrite them: `blk_idx` for the scaling blocks, `qdiag_idx`/`ediag_idx`
# for the (1,1) and (2,2) diagonals, with `qdiag` the diagonal of Q
# before any shift so that kktsolver_ldl can write its own δp/δe (the
# placeholders δp, δe here only make the matrix quasi-definite for the
# fill estimate). Also used by choose_kktsolver to estimate the
# factorization's fill before committing to a solver.
function _ldl_pattern(Q, A, G, cone_dims; lift_min::Int = 6,
                      δp::Float64 = 1e-8, δe::Float64 = 1e-8,
                      perm_hint = nothing)

  n = size(Q, 1); m = size(A, 1); p = size(G, 1)
  Qs = sparse(Q); As = sparse(A); Gs = sparse(G)
  qdiag = Vector{Float64}(diag(Qs))     # Q's diagonal before the δp shift

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
  # Static-shift diagonals; every (i,i) and (n+r,n+r) exists structurally
  # because of the put! calls above.
  qdiag_idx = [_nzindex(K, i, i) for i in 1:n]
  ediag_idx = [_nzindex(K, n + r, n + r) for r in 1:p]

  # Expected pivot signs of the quasi-definite pattern
  Dsigns = Vector{Int}(undef, N)
  Dsigns[1:n] .= 1
  Dsigns[n+1:oa] .= -1
  for j in 1:nlift
    Dsigns[oa + 2j - 1] = 1
    Dsigns[oa + 2j] = -1
  end

  # Fill-reducing ordering, computed here once so that the solver choice,
  # the symbolic analysis, and any cached reuse all share it.
  perm = perm_hint === nothing ? amd(K) : perm_hint

  return (; K, kinds, ranges, blk_idx, Dsigns, n, m, p, N, oz, oa, nlift, perm,
            qdiag_idx, qdiag, ediag_idx)

end

# Structure key of a problem, for reusing an ordering across solves.
_ldl_structure_key(Q, A, G, cone_dims) = begin
  Qs = sparse(Q); As = sparse(A); Gs = sparse(G)
  (size(Qs), size(As), size(Gs), copy(cone_dims),
   copy(Qs.colptr), copy(Qs.rowval), copy(As.colptr), copy(As.rowval),
   copy(Gs.colptr), copy(Gs.rowval))
end

"""
    cached_kktsolver_ldl(; kwargs...)

A [`kktsolver_ldl`](@ref) that remembers the fill-reducing ordering of
the last problem it saw and reuses it when the next problem has the same
structure (dimensions, cone list, and sparsity patterns of `Q`, `A`, `G`),
which is the situation in model-predictive control, sequential convex
programming, or any loop that re-solves with new data. The AMD ordering
is the only super-linear part of the symbolic setup; the rest is rebuilt
in O(nnz) each solve. On banded and block-structured problems AMD is a
few percent of a solve, so the saving is modest; it grows with the fill
of the pattern. `kwargs` are forwarded to `kktsolver_ldl`.

```julia
ks = ConicIP.cached_kktsolver_ldl()
for t in 1:T
    sol = conicIP(Q, c[t], A, b[t], cone_dims; kktsolver = ks)
end
ks.hits   # number of solves that reused the ordering
```
"""
mutable struct cached_kktsolver_ldl
  key    :: Any
  perm   :: Union{Nothing, Vector{Int}}
  hits   :: Int
  kwargs :: Any
end
cached_kktsolver_ldl(; kwargs...) = cached_kktsolver_ldl(nothing, nothing, 0, kwargs)

function (ks::cached_kktsolver_ldl)(Q, A, G, cone_dims)
  key = _ldl_structure_key(Q, A, G, cone_dims)
  hint = nothing
  if ks.perm !== nothing && key == ks.key
    hint = ks.perm
    ks.hits += 1
  end
  # kktsolver_ldl writes its own static shifts onto the pattern, so only
  # lift_min has to agree.
  pat = _ldl_pattern(Q, A, G, cone_dims;
                     lift_min = get(ks.kwargs, :lift_min, 6),
                     perm_hint = hint)
  ks.key = key
  ks.perm = pat.perm
  return kktsolver_ldl(Q, A, G, cone_dims; pattern = pat, ks.kwargs...)
end
