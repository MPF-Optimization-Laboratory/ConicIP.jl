# ──────────────────────────────────────────────────────────────
#  Ruiz equilibration (roadmap tranche 1, item 6)
# ──────────────────────────────────────────────────────────────
#
# The scaled problem is
#
#     min ½ỹᵀQ̃ỹ − c̃ᵀỹ   s.t.  Ãỹ ≥_K b̃,  G̃ỹ = d̃
#
# with  ỹ = Dc⁻¹ y,  Q̃ = σ Dc Q Dc,  c̃ = σ Dc c,
#       Ã = Dr A Dc,  b̃ = Dr b,  G̃ = De G Dc,  d̃ = De d,
#
# where Dc, Dr, De are positive diagonal and σ > 0 scales the objective.
# Dr is constant on every second-order and semidefinite block, so that
# s̃ = Dr s ∈ K ⟺ s ∈ K; only the nonnegative orthant is scaled entrywise.
# Stationarity Q̃ỹ + G̃ᵀw̃ − Ãᵀṽ = c̃ identifies the duals as
# w̃ = σ De⁻¹ w and ṽ = σ Dr⁻¹ v.
#
# Ruiz iterations equalize the ∞-norms of the rows and columns of the
# symmetric matrix [Q Aᵀ Gᵀ; A 0 0; G 0 0]; each sweep multiplies the
# current scaling by 1/√(row norm).

# Column-wise and row-wise ∞-norms of a sparse matrix, accumulated into
# `out` with `max`. A sweep needs the largest column norm over the three
# blocks of one KKT column, so the accumulating form lets one buffer serve
# them all and keeps the sweep free of allocation.
function _col_infnorms!(out, S::SparseMatrixCSC)
  colptr = S.colptr; vals = S.nzval
  @inbounds for j in 1:size(S, 2)
    mx = out[j]
    for t in colptr[j]:colptr[j+1]-1
      mx = max(mx, abs(vals[t]))
    end
    out[j] = mx
  end
  return out
end
function _row_infnorms!(out, S::SparseMatrixCSC)
  rows = S.rowval; vals = S.nzval
  @inbounds for t in eachindex(vals)
    i = rows[t]
    out[i] = max(out[i], abs(vals[t]))
  end
  return out
end

# S .= Diagonal(dl) * S * Diagonal(dr) on the stored values, in the same
# multiplication order as the two-sided product it replaces
# (`(dl[i] * S[i,j]) * dr[j]`), so the scaled entries are bitwise identical
# to the matrix the allocating form would have built.
function _scale_rows_cols!(S::SparseMatrixCSC, dl, dr)
  colptr = S.colptr; rows = S.rowval; vals = S.nzval
  @inbounds for j in 1:size(S, 2)
    dj = dr[j]
    for t in colptr[j]:colptr[j+1]-1
      vals[t] = (dl[rows[t]] * vals[t]) * dj
    end
  end
  return S
end

# Diagonal(dl) * M * Diagonal(dr), one allocation for a sparse M; the
# generic fallback keeps any other storage type (dense stays dense).
_scaled(M, dl, dr) = Diagonal(dl) * M * Diagonal(dr)
_scaled(M::SparseMatrixCSC, dl, dr) =
  _scale_rows_cols!(SparseMatrixCSC(size(M, 1), size(M, 2), copy(M.colptr),
                                    copy(M.rowval), copy(M.nzval)), dl, dr)

# Slot of the mirror entry (j,i) for every stored (i,j), or `nothing` when
# the structure is not symmetric. Walking the columns in order appends the
# entries of column i of Sᵀ in increasing row order, which is exactly the
# CSC order of column i of S when S is structurally symmetric — so the
# running cursor `ptr[i]` is the mirror slot, and any mismatch proves the
# structure asymmetric.
function _mirror_slots(S::SparseMatrixCSC)
  n = size(S, 2)
  size(S, 1) == n || return nothing
  colptr = S.colptr; rows = S.rowval
  ptr = copy(colptr)
  mirror = Vector{Int}(undef, length(rows))
  @inbounds for j in 1:n, t in colptr[j]:colptr[j+1]-1
    i = rows[t]
    q = ptr[i]
    (q < colptr[i+1] && rows[q] == j) || return nothing
    ptr[i] = q + 1
    mirror[t] = q
  end
  return mirror
end

# σ * (Diagonal(d) * M * Diagonal(d)), symmetrized. The two-sided diagonal
# product is not bitwise symmetric (the products are formed in different
# orders above and below the diagonal), so the result is averaged with its
# transpose: solvers and user callbacks may check `issymmetric(Q)`.
_congruence_generic(M, d, σ) =
  (X = σ * (Diagonal(d) * M * Diagonal(d)); (X + X') / 2)
_congruence(M, d, σ) = _congruence_generic(M, d, σ)
function _congruence(M::SparseMatrixCSC, d, σ)
  mirror = _mirror_slots(M)
  mirror === nothing && return _congruence_generic(M, d, σ)
  colptr = M.colptr; rows = M.rowval; vals = M.nzval
  out = similar(vals)
  @inbounds for j in 1:size(M, 2)
    dj = d[j]
    for t in colptr[j]:colptr[j+1]-1
      out[t] = σ * ((d[rows[t]] * vals[t]) * dj)
    end
  end
  # Average each entry with its mirror, visiting every pair once (the
  # lower triangle, where the mirror slot still holds its unaveraged value).
  @inbounds for j in 1:size(M, 2), t in colptr[j]:colptr[j+1]-1
    if rows[t] >= j
      q = mirror[t]
      a = (out[t] + out[q]) / 2
      out[t] = a; out[q] = a
    end
  end
  return SparseMatrixCSC(size(M, 1), size(M, 2), copy(colptr), copy(rows), out)
end

"""
    equilibrate_conicIP(Q, c, A, b, cone_dims, G, d;
                        iters = 10, tol = 1e-2, bound = 1e6,
                        σ_range = (1e-3, 1e3))

Ruiz-equilibrate the problem data. Returns a named tuple with the scaled
data `Q, c, A, b, G, d` and the scalings `Dc` (variables), `Dr` (cone
rows, constant on each second-order and semidefinite block), `De`
(equality rows) and `σ` (objective), such that

    Q = Dc⁻¹ Q̃ Dc⁻¹ / σ,  c = Dc⁻¹ c̃ / σ,  A = Dr⁻¹ Ã Dc⁻¹,  b = Dr⁻¹ b̃,
    G = De⁻¹ G̃ Dc⁻¹,  d = De⁻¹ d̃.

At most `iters` sweeps are made, stopping early when every row and column
∞-norm of `[Q Aᵀ Gᵀ; A 0 0; G 0 0]` is within `tol` of one. Individual
scale factors are kept within `[1/bound, bound]`; a zero row or column
is left unscaled. The objective is rescaled to unit ∞-norm
(`σ = 1/‖c̃‖∞`) only when `‖c̃‖∞` falls outside `σ_range`: the iteration
is not invariant to the objective scale, and on well-scaled problems a
σ ≠ 1 costs iterations. The matrices keep their storage type (dense
stays dense, sparse stays sparse).
"""
function equilibrate_conicIP(Q, c, A, b, cone_dims, G, d;
                             iters::Int = 10, tol::Float64 = 1e-2,
                             bound::Float64 = 1e6,
                             σ_range::Tuple{Float64,Float64} = (1e-3, 1e3))
  n = length(c); m = size(A, 1); p = size(G, 1)
  ranges = cum_range([cd[2] for cd in cone_dims])
  uniform = [cd[1] != "R" for cd in cone_dims]

  # Private working copies: the sweeps rescale their stored values in
  # place, so they must not alias the caller's matrices.
  Qs = sparse(Q); Qs === Q && (Qs = copy(Qs))
  As = sparse(A); As === A && (As = copy(As))
  Gs = sparse(G); Gs === G && (Gs = copy(Gs))
  Dc = ones(n); Dr = ones(m); De = ones(p)
  lo = 1 / bound; hi = bound
  cn = zeros(n); rn = zeros(m); en = zeros(p)
  sc = ones(n); sr = ones(m); se = ones(p)

  for _ in 1:iters
    # KKT row norms: variable rows see Q, Aᵀ, Gᵀ; cone rows see A; equality rows see G
    fill!(cn, 0.0); fill!(rn, 0.0); fill!(en, 0.0)
    _col_infnorms!(cn, Qs); _col_infnorms!(cn, As); _col_infnorms!(cn, Gs)
    _row_infnorms!(rn, As)
    _row_infnorms!(en, Gs)
    for (u, I) in zip(uniform, ranges)
      if u && !isempty(I)
        mx = maximum(view(rn, I))
        rn[I] .= mx
      end
    end
    converged = all(x -> x == 0 || abs(1 - x) <= tol, cn) &&
                all(x -> x == 0 || abs(1 - x) <= tol, rn) &&
                all(x -> x == 0 || abs(1 - x) <= tol, en)
    converged && break
    # Limit the applied increments, not just the recorded cumulative
    # factors. Otherwise Qs/As/Gs drift from the scalings returned below
    # once a bound is hit, and later sweeps balance a fictitious matrix.
    @inbounds for i in eachindex(cn)
      x = cn[i]
      sc[i] = x == 0 ? 1.0 : clamp(1 / sqrt(x), lo / Dc[i], hi / Dc[i])
    end
    @inbounds for i in eachindex(rn)
      x = rn[i]
      sr[i] = x == 0 ? 1.0 : clamp(1 / sqrt(x), lo / Dr[i], hi / Dr[i])
    end
    @inbounds for i in eachindex(en)
      x = en[i]
      se[i] = x == 0 ? 1.0 : clamp(1 / sqrt(x), lo / De[i], hi / De[i])
    end
    # Rescale the stored values in place: a sweep is a pure reweighting of
    # the existing nonzeros, so there is no reason to build three new
    # sparse matrices per sweep.
    _scale_rows_cols!(Qs, sc, sc)
    _scale_rows_cols!(As, sr, sc)
    _scale_rows_cols!(Gs, se, sc)
    Dc .*= sc; Dr .*= sr; De .*= se
  end

  c̃ = Dc .* c
  # Objective scale. The interior-point iteration is not invariant to it
  # (the initial point and centering treat primal and dual asymmetrically),
  # and on well-scaled problems a σ ≠ 1 measurably costs iterations, so it
  # is applied only when the objective is genuinely out of range.
  nc = norm(c̃, Inf)
  σ = (nc == 0 || σ_range[1] <= nc <= σ_range[2]) ? 1.0 :
      1 / clamp(nc, 1e-8, 1e8)
  # Scale the caller's matrices in their own storage type, each in a
  # single pass over its stored values (see `_congruence` / `_scaled`).
  Q̃ = _congruence(Q, Dc, σ)
  Ã = _scaled(A, Dr, Dc)
  G̃ = _scaled(G, De, Dc)
  return (Q = Q̃, c = σ .* c̃, A = Ã, b = Dr .* b, G = G̃, d = De .* d,
          Dc = Dc, Dr = Dr, De = De, σ = σ)
end

# Re-validate a certificate found on the equilibrated data against the
# ORIGINAL data. Tolerance-based validity is not invariant under diagonal
# scaling: a ray with residual 1e-8 on the scaled data can have residual
# 1e-4 against the original (a column scaled by 1e-6 amplifies its share
# of Gᵀw − Aᵀv by 1e6 relative to the normalization). The status
# vocabulary mirrors the post-loop re-screen in `_conicIP`: a ray valid
# at the nominal tolerance keeps its claim (and takes the validator's
# normalization); one valid only at 100× reltol downgrades to
# :AlmostInfeasible / :AlmostDualInfeasible; otherwise :Abandoned. In
# both downgraded cases `has_certificate` is cleared and the ray is left
# in place for inspection.
function _revalidate_certificate!(sol::Solution, Q, c, A, b, cone_dims, G, d;
                                  infeasTol::Float64 = 1e-7,
                                  infeasAbsTol::Float64 = 1e-9,
                                  source = "equilibrated data")
  sol.has_certificate || return sol
  sol.status in (:Infeasible, :DualInfeasible) || return sol

  relaxed = 100 * infeasTol
  if sol.status == :Infeasible
    (chk, w̄, v̄) = validate_infeasibility_certificate(
                     Q, c, A, b, cone_dims, G, d, sol.w, sol.v;
                     abstol = infeasAbsTol, reltol = infeasTol)
    if chk.valid
      sol.w[:] = w̄; sol.v[:] = v̄
      return sol
    end
    (chk100, _, _) = validate_infeasibility_certificate(
                       Q, c, A, b, cone_dims, G, d, sol.w, sol.v;
                       abstol = infeasAbsTol, reltol = relaxed)
    almost = :AlmostInfeasible
  else
    (chk, ȳ) = validate_unboundedness_certificate(
                 Q, c, A, b, cone_dims, G, d, sol.y;
                 abstol = infeasAbsTol, reltol = infeasTol)
    if chk.valid
      sol.y[:] = ȳ
      sol.s .= A * ȳ
      return sol
    end
    (chk100, _) = validate_unboundedness_certificate(
                    Q, c, A, b, cone_dims, G, d, sol.y;
                    abstol = infeasAbsTol, reltol = relaxed)
    almost = :AlmostDualInfeasible
  end

  kind = sol.status == :Infeasible ? "infeasibility" : "unboundedness"
  sol.has_certificate = false
  sol.status = chk100.valid ? almost : :Abandoned
  g3(x) = @sprintf("%.3g", x)
  sol.message = "$kind certificate valid on the $source but not " *
                "on the original data (residual $(g3(chk.farkas_residual)), " *
                "cone margin $(g3(chk.cone_margin)) at infeasTol = $(g3(infeasTol))" *
                (chk100.valid ? "; valid at $(g3(relaxed)))" :
                                " and at $(g3(relaxed)))")
  return sol
end

# Map a Solution of the scaled problem back to the original coordinates,
# renormalize certificates, and recompute the reported residuals from
# the original data.
function unequilibrate!(sol::Solution, eq, Q, c, A, b, cone_dims, G, d;
                        objective_offset = 0.0)
  Dc, Dr, De, σ = eq.Dc, eq.Dr, eq.De, eq.σ
  sol.y .= Dc .* sol.y
  sol.s .= sol.s ./ Dr
  sol.v .= (Dr .* sol.v) ./ σ
  sol.w .= (De .* sol.w) ./ σ

  if sol.has_certificate && sol.status == :Infeasible
    # dᵀw̄ − bᵀv̄ = −1 in the original coordinates
    t = dot(d, sol.w) - dot(b, sol.v)
    if isfinite(t) && t < 0
      sol.w ./= -t; sol.v ./= -t
    end
    return sol
  elseif sol.has_certificate && sol.status == :DualInfeasible
    # cᵀȳ = +1 in the original coordinates
    t = dot(c, sol.y)
    if isfinite(t) && t > 0
      sol.y ./= t
    end
    sol.s .= A * sol.y
    return sol
  end

  sol.Mu /= σ
  return _refresh_point!(sol, Q, c, A, b, G, d; objective_offset = objective_offset)
end

# Recompute point diagnostics after any change of coordinates or presolve.
# In particular, residuals on dropped equality rows must not disappear.
# The relative gap uses the one formula of the main loop's termination
# test, |vᵀs| / (1 + |pobj + objective_offset|).
function _refresh_point!(sol::Solution, Q, c, A, b, G, d; objective_offset = 0.0)
  y, w, v, s = sol.y, sol.w, sol.v, sol.s
  if all(isfinite, y) && all(isfinite, v) && all(isfinite, s) && all(isfinite, w)
    Qy   = Q * y
    cᵀy  = dot(c, y)
    # Same backward-error normalization as the termination test in
    # _conicIP: the residual is relative to the right-hand side or to the
    # componentwise products |Q||y|, |Gᵀ||w|, |Aᵀ||v|, |A||y|, |G||y|.
    absQ = _absmat(Q); absA = _absmat(A); absG = _absmat(G)
    ay = abs.(y); aw = abs.(w); av = abs.(v)
    nQy = norm(absQ * ay)
    nGw = isempty(w) ? 0.0 : norm(absG' * aw)
    nAv = isempty(v) ? 0.0 : norm(absA' * av)
    rDu  = norm(Qy + (G' * w - A' * v) - c) / (1 + max(norm(c), nQy, nGw, nAv))
    rPr  = isempty(b) ? 0.0 :
           norm(A * y - s - b) / (1 + max(norm(b), norm(absA * ay), norm(s)))
    rEq  = isempty(d) ? 0.0 :
           norm(G * y - d) / (1 + max(norm(d), norm(absG * ay)))
    pobj = 0.5 * dot(y, Qy) - cᵀy
    dobj = -0.5 * dot(y, Qy) - dot(d, w) + dot(b, v)
    sol.duFeas = rDu
    sol.prFeas = max(rPr, rEq)
    sol.rEq    = rEq
    sol.rGap   = abs(dot(v, s)) / (1 + abs(pobj + objective_offset))
    sol.pobj   = pobj
    sol.dobj   = dobj
  end
  return sol
end

# A numerically reduced problem can pass its stopping test while the
# restored point fails the caller's tolerance. Keep the point for diagnosis,
# but retract optimality rather than exposing it as a feasible MOI result.
function _check_postsolve!(sol::Solution, Q, c, A, b, cone_dims, G, d;
                           optTol = 1e-6, objective_offset = 0.0,
                           infeasTol = 1e-7, infeasAbsTol = 1e-9)
  if sol.has_certificate
    return _revalidate_certificate!(sol, Q, c, A, b, cone_dims, G, d;
        infeasTol = Float64(infeasTol), infeasAbsTol = Float64(infeasAbsTol),
        source = "presolved data")
  end
  _refresh_point!(sol, Q, c, A, b, G, d; objective_offset = objective_offset)
  if sol.status == :Optimal
    if !(max(sol.prFeas, sol.duFeas, sol.rGap) < optTol)
      sol.status = :Error
      sol.message = "presolved point fails the original-data optimality tolerance"
    end
  end
  return sol
end
