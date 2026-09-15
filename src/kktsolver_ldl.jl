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

import QDLDL
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

# `sparse(M)` without the copy when M already is one. The pattern assembly
# and the structure key below only read these matrices, and a
# SparseMatrixCSC always stores each column's row indices in ascending
# order, which is exactly what the CSC assembly needs.
_csc(M::SparseMatrixCSC) = M
_csc(M::AbstractMatrix)  = sparse(M)

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
  `solve3x3` return, over the LIFTED system (auxiliary rows included)
- `last_rtol` -- the tolerance `refine_tol·(1 + ‖rhs‖)` that the internal
  refinement of that solve aimed at, so a caller can tell a solve that met
  its own target from one that gave up
- `last_bound` -- upper bound on the residual norm of the UNLIFTED 3×3
  system `‖(bx, by, −bz) − K₃ₓ₃ x₃ₓ₃‖` for the same solution. Equal to
  `last_residual` when no second-order cone was lifted; otherwise
  `‖r_u‖ + lift_gain·‖r_a‖`, where `r_u`/`r_a` split the lifted residual
  into its 3×3 rows and its auxiliary rows (eliminating the auxiliaries
  `a = uᵀz`, `b = vᵀz` from a residual `(r_u, r_a)` leaves
  `r_u + [u v]·r_a` on the 3×3 rows, and `[u v]` per lifted block has
  spectral norm at most `√(‖u‖² + ‖v‖²)`)
- `lift_gain` -- `maxᵦ √(‖uᵦ‖² + ‖vᵦ‖²)` over the lifted second-order-cone
  blocks of the current factorization; `0` when nothing is lifted
- `timing` -- the `PhaseTimes` the backend reports into, or `nothing`
  (the default). Set by `kkt_attach_timing!`; while attached, every
  numeric refactorization, triangular solve, and residual evaluation adds
  its wall time and a count to the `t_ldl_*` / `n_ldl_*` fields. With
  `nothing` each of those sites costs one pointer comparison.
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
  last_rtol       :: Float64
  last_bound      :: Float64
  lift_gain       :: Float64
  timing          :: Union{Nothing, PhaseTimes}
end
LDLDiagnostics(δp, δe, δc) =
  LDLDiagnostics(0, 0, δp, δe, δc, 0, 0, 0, NaN, NaN, NaN, 0.0, nothing)

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

# The generator `kktsolver_ldl` returns (what the main loop calls
# `solve3x3gen`): the factorization closure plus the shared diagnostics
# record, so timing can be attached BEFORE the first factorization.
struct LDLGen{G}
  gen  :: G
  diag :: LDLDiagnostics
end
(g::LDLGen)(F, F⁻ᵀ) = g.gen(F, F⁻ᵀ)
kkt_diagnostics(g::LDLGen) = g.diag

# The diagnostics record is shared by every factorization of one
# `kktsolver_ldl` instance, so attaching once (to the generator or to any
# factorization) keeps reporting until a different (or no) record is
# attached.
function kkt_attach_timing!(s::LDLSolve3x3, pt::PhaseTimes)
  s.diag.timing = pt
  return nothing
end
function kkt_attach_timing!(g::LDLGen, pt::PhaseTimes)
  g.diag.timing = pt
  return nothing
end

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
`Solution.kkt_refactors`. `ConicIP.kkt_attach_timing!(solve3x3, pt)`
makes the backend add the time and count of every numeric
refactorization, triangular solve, and refinement residual to the
`t_ldl_*` / `n_ldl_*` fields of `pt::PhaseTimes` (`conicIP` does this
when called with `timing = pt`); the record is shared across the
factorizations of one `kktsolver_ldl` instance, so the attachment
persists.

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

  # The timed sites below branch on `pt === nothing`, where `pt` is
  # `diag.timing` read once per solve3x3 call and PASSED DOWN as an argument:
  # capturing `diag` in these closures instead would grow the per-
  # factorization object (closures are embedded by value), which is the
  # allocation the timing-off path must not pay. Timestamps are local. See
  # timing.jl.

  # res = rhs − K₀ x for the unregularized K₀ = K_δ − Diagonal(shift);
  # returns (‖res‖, ‖res over the lifted blocks' auxiliary rows‖). The
  # second entry is what `last_bound` charges the lift for; it is zero
  # when no second-order cone was lifted (N == oa).
  function residual!(x, pt)
    t0 = pt === nothing ? UInt64(0) : time_ns()
    _symmul!(res, K, x)
    @inbounds for i in 1:N
      res[i] = rhs[i] - (res[i] - shift[i] * x[i])
    end
    r  = norm(res)
    ra = 0.0
    @inbounds for i in (oa+1):N
      ra += res[i]^2
    end
    ra = sqrt(ra)
    if pt !== nothing
      pt.t_ldl_resid += time_ns() - t0
      pt.n_ldl_resid += 1
    end
    return (r, ra)
  end

  # Numeric refactorization, recording what QDLDL did to the pivots
  # (its dynamic regularization is part of refactor! and so of the time).
  function numeric_factor!()
    pt = diag.timing
    t0 = pt === nothing ? UInt64(0) : time_ns()
    refactor!(Fact)
    diag.repaired        = regularized_entries(Fact)
    diag.repaired_total += diag.repaired
    diag.pos_inertia     = positive_inertia(Fact)
    if pt !== nothing
      pt.t_ldl_factor += time_ns() - t0
      pt.n_ldl_factor += 1
    end
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
    diag.lift_gain = 0.0

    # Rewrite the scaling entries (in K, for the residual, and in the
    # factorization's internal copy), then refactorize numerically.
    for (bi, (kind, I)) in enumerate(zip(kinds, ranges))
      Blk = F.Blocks[bi]
      idx = blk_idx[bi]
      if kind == :diag
        # `diag` is the diagnostics record here; qualify the function.
        d = Blk isa Diagonal ? Blk.diag : LinearAlgebra.diag(Matrix(Blk))
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
        # Spectral-norm bound of this block's [u v], for `last_bound`.
        diag.lift_gain = max(diag.lift_gain, sqrt(dot(u, u) + dot(v, v)))
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
    function solve_loaded!(pt)
      sol .= rhs
      t0 = pt === nothing ? UInt64(0) : time_ns()
      solve!(Fact, sol)
      if pt !== nothing
        pt.t_ldl_solve += time_ns() - t0
        pt.n_ldl_solve += 1
      end
      rtol  = refine_tol * (1 + norm(rhs))
      (rbest, abest) = residual!(sol, pt)
      noncontract = false
      for k in 1:refine_steps
        rbest <= rtol && break
        tmp .= res
        t0 = pt === nothing ? UInt64(0) : time_ns()
        solve!(Fact, tmp)
        if pt !== nothing
          pt.t_ldl_solve += time_ns() - t0
          pt.n_ldl_solve += 1
        end
        cand .= sol .+ tmp
        (rcand, acand) = residual!(cand, pt)
        if !(rcand < rbest)
          noncontract = (k == 1)
          break
        end
        sol .= cand
        rbest = rcand
        abest = acand
      end
      return (rbest, rtol, noncontract, abest)
    end

    function solve3x3(bx, by, bz)
      rhs[1:n] .= bx
      rhs[n+1:n+p] .= by
      rhs[oz+1:oz+m] .= .-bz
      rhs[oa+1:N] .= 0.0
      pt = diag.timing
      (rbest, rtol, noncontract, abest) = solve_loaded!(pt)
      # Bounded retry: a solve that misses the tolerance on a factorization
      # that repaired pivots (or whose refinement did not contract at all)
      # is repeated with larger static shifts; the better result by
      # unregularized residual is kept.
      while rbest > rtol && (diag.repaired > 0 || noncontract) &&
            diag.refactors < retry_max
        keep .= sol; rkeep = rbest; akeep = abest
        bump!()
        (rnew, _, noncontract, anew) = solve_loaded!(pt)
        if rnew < rkeep
          rbest = rnew; abest = anew
        else
          sol .= keep; rbest = rkeep; abest = akeep
        end
      end
      diag.last_residual = rbest
      diag.last_rtol     = rtol
      # ‖r_u‖ + gain·‖r_a‖ with ‖r_u‖ = √(‖res‖² − ‖r_a‖²); an exact
      # unlifted solve (abest == 0) reports the residual itself.
      diag.last_bound = abest == 0.0 ? rbest :
        sqrt(max(rbest*rbest - abest*abest, 0.0)) + diag.lift_gain * abest
      # Views, not slices: three copies of the solution per back-solve were
      # the third-largest allocation site in the loop. The documented
      # contract lets a backend hand back views of its own workspace — the
      # main loop copies what it needs out before calling again — but a
      # caller that keeps a result across two solves must copy it itself.
      return (view(sol, 1:n), view(sol, (n+1):(n+p)), view(sol, (oz+1):(oz+m)))
    end

    return LDLSolve3x3(solve3x3, bump!, diag)

  end

  return LDLGen(solve3x3gen, diag)

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
  Qs = _csc(Q); As = _csc(A); Gs = _csc(G)
  qdiag = zeros(n)                # Q's diagonal before the δp shift; read
                                  # off in the counting pass below

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

  # The assembly below indexes K's columns straight from the source
  # patterns, so a shape that does not fit the KKT block structure has to
  # be rejected before the first @inbounds loop rather than inside it.
  mcone = isempty(ranges) ? 0 : last(last(ranges))
  (size(Qs, 2) == n && size(As, 2) == n && size(Gs, 2) == n && mcone == m) ||
    throw(DimensionMismatch(
      "_ldl_pattern: Q, A and G must have $n columns and cone_dims must " *
      "cover all $m rows of A (got $(size(Qs, 2)), $(size(As, 2)), " *
      "$(size(Gs, 2)) columns and $mcone cone rows)"))

  # ── Upper triangle of K, assembled straight into CSC in O(nnz) ──
  #
  # Column layout (every column's rows come out ascending, because each
  # source loop below runs over the source matrix column by column and the
  # block entries of a column all sit below its data entries):
  #
  #   j ≤ n      strict upper triangle of Q's column j, then (j,j) = Qⱼⱼ + δp
  #              (always present, whatever Q's pattern);
  #   n + r      row r of G (as column r of Gᵀ), then (n+r, n+r) = −δe;
  #   oz + i     column i of −Aᵀ, then the scaling entries of i's cone block;
  #   oa + 2k∓1  a lifted block's two spike columns, then their pivots.
  #
  # Building the column counts first and filling through a per-column write
  # cursor replaces the COO triplet list, its `sparse()` assembly, and the
  # binary search that used to locate every scaling entry afterwards: the
  # cursor *is* the index, so `blk_idx`, `qdiag_idx` and `ediag_idx` fall
  # out of the same pass.
  colptr = zeros(Int, N + 1)            # counts in colptr[c+1], cumsum below
  @inbounds for j in 1:n
    cnt = 1                             # the (j,j) entry, always present
    for t in nzrange(Qs, j)
      i = Qs.rowval[t]
      if i < j
        cnt += 1
      else
        # This column's own diagonal, on the way past: `diag(Qs)` would
        # walk the same columns a second time.
        i == j && (qdiag[j] = Qs.nzval[t])
        break
      end
    end
    colptr[j+1] = cnt
  end
  @inbounds for j in 1:size(Gs, 2), t in nzrange(Gs, j)
    colptr[n + Gs.rowval[t] + 1] += 1
  end
  @inbounds for r in 1:p
    colptr[n + r + 1] += 1              # the −δe pivot
  end
  @inbounds for j in 1:size(As, 2), t in nzrange(As, j)
    colptr[oz + As.rowval[t] + 1] += 1
  end
  ilift = 0
  @inbounds for (kind, I) in zip(kinds, ranges)
    if kind == :dense
      f = first(I)
      for b in I; colptr[oz + b + 1] += b - f + 1; end
    else
      for i in I; colptr[oz + i + 1] += 1; end
      if kind == :lift
        ilift += 1
        colptr[oa + 2ilift] += length(I) + 1        # column oa + 2ilift − 1
        colptr[oa + 2ilift + 1] += length(I) + 1    # column oa + 2ilift
      end
    end
  end
  colptr[1] = 1
  @inbounds for c in 1:N; colptr[c+1] += colptr[c]; end
  rowval = Vector{Int}(undef, colptr[N+1] - 1)
  nzval  = Vector{Float64}(undef, colptr[N+1] - 1)
  w = colptr[1:N]                       # per-column write cursor

  qdiag_idx = Vector{Int}(undef, n)
  @inbounds for j in 1:n
    k = w[j]
    for t in nzrange(Qs, j)
      i = Qs.rowval[t]
      i >= j && break
      rowval[k] = i; nzval[k] = Qs.nzval[t]; k += 1
    end
    rowval[k] = j; nzval[k] = qdiag[j] + δp
    qdiag_idx[j] = k
    w[j] = k + 1
  end
  @inbounds for j in 1:size(Gs, 2), t in nzrange(Gs, j)
    c = n + Gs.rowval[t]; k = w[c]
    rowval[k] = j; nzval[k] = Gs.nzval[t]; w[c] = k + 1
  end
  ediag_idx = Vector{Int}(undef, p)
  @inbounds for r in 1:p
    c = n + r; k = w[c]
    rowval[k] = c; nzval[k] = -δe; w[c] = k + 1
    ediag_idx[r] = k
  end
  @inbounds for j in 1:size(As, 2), t in nzrange(As, j)
    c = oz + As.rowval[t]; k = w[c]
    rowval[k] = j; nzval[k] = -As.nzval[t]; w[c] = k + 1
  end

  # ── Scaling blocks, recording each entry's index into K.nzval as it is
  #    written (same order as the block updates in kktsolver_ldl) ──
  blk_idx = Vector{Vector{Int}}(undef, length(kinds))
  ilift = 0
  @inbounds for (bi, (kind, I)) in enumerate(zip(kinds, ranges))
    nI = length(I)
    if kind == :diag
      idx = Vector{Int}(undef, nI)
      for (q, i) in enumerate(I)
        c = oz + i; k = w[c]
        rowval[k] = c; nzval[k] = -1.0; w[c] = k + 1
        idx[q] = k
      end
      blk_idx[bi] = idx
    elseif kind == :dense
      f = first(I)
      idx = Vector{Int}(undef, (nI * (nI + 1)) >> 1)
      q = 0
      for b in I
        c = oz + b
        for a in f:b
          k = w[c]
          rowval[k] = oz + a; nzval[k] = (a == b ? -1.0 : 1.0); w[c] = k + 1
          idx[q += 1] = k
        end
      end
      blk_idx[bi] = idx
    else
      ilift += 1
      ca = oa + 2ilift - 1; cb = oa + 2ilift
      idx = Vector{Int}(undef, 3nI + 2)
      for (q, i) in enumerate(I)
        c = oz + i; k = w[c]
        rowval[k] = c; nzval[k] = -1.0; w[c] = k + 1
        idx[q] = k
      end
      for (q, i) in enumerate(I)
        k = w[ca]
        rowval[k] = oz + i; nzval[k] = -1.0; w[ca] = k + 1
        idx[nI + q] = k
      end
      for (q, i) in enumerate(I)
        k = w[cb]
        rowval[k] = oz + i; nzval[k] = 1.0; w[cb] = k + 1
        idx[2nI + q] = k
      end
      k = w[ca]; rowval[k] = ca; nzval[k] =  1.0; w[ca] = k + 1
      idx[3nI + 1] = k
      k = w[cb]; rowval[k] = cb; nzval[k] = -1.0; w[cb] = k + 1
      idx[3nI + 2] = k
      blk_idx[bi] = idx
    end
  end
  K = SparseMatrixCSC(N, N, colptr, rowval, nzval)

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
  Qs = _csc(Q); As = _csc(A); Gs = _csc(G)
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
