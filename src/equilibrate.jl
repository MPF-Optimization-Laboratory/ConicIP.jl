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

# Column-wise and row-wise ∞-norms of a sparse matrix.
function _col_infnorms(S::SparseMatrixCSC)
  out = zeros(size(S, 2))
  @inbounds for j in 1:size(S, 2)
    mx = 0.0
    for t in S.colptr[j]:S.colptr[j+1]-1
      mx = max(mx, abs(S.nzval[t]))
    end
    out[j] = mx
  end
  return out
end
function _row_infnorms(S::SparseMatrixCSC)
  out = zeros(size(S, 1))
  rows = S.rowval; vals = S.nzval
  @inbounds for t in eachindex(vals)
    i = rows[t]
    out[i] = max(out[i], abs(vals[t]))
  end
  return out
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

  Qs = sparse(Q); As = sparse(A); Gs = sparse(G)
  Dc = ones(n); Dr = ones(m); De = ones(p)
  lo = 1 / bound; hi = bound

  for _ in 1:iters
    # KKT row norms: variable rows see Q, Aᵀ, Gᵀ; cone rows see A; equality rows see G
    cn = max.(_col_infnorms(Qs), _col_infnorms(As), _col_infnorms(Gs))
    rn = _row_infnorms(As)
    en = _row_infnorms(Gs)
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
    sc = [x == 0 ? 1.0 : clamp(1 / sqrt(x), lo, hi) for x in cn]
    sr = [x == 0 ? 1.0 : clamp(1 / sqrt(x), lo, hi) for x in rn]
    se = [x == 0 ? 1.0 : clamp(1 / sqrt(x), lo, hi) for x in en]
    Qs = Diagonal(sc) * Qs * Diagonal(sc)
    As = Diagonal(sr) * As * Diagonal(sc)
    Gs = Diagonal(se) * Gs * Diagonal(sc)
    Dc .*= sc; Dr .*= sr; De .*= se
    # keep the cumulative factors bounded as well
    clamp!(Dc, lo, hi); clamp!(Dr, lo, hi); clamp!(De, lo, hi)
  end

  c̃ = Dc .* c
  # Objective scale. The interior-point iteration is not invariant to it
  # (the initial point and centering treat primal and dual asymmetrically),
  # and on well-scaled problems a σ ≠ 1 measurably costs iterations, so it
  # is applied only when the objective is genuinely out of range.
  nc = norm(c̃, Inf)
  σ = (nc == 0 || σ_range[1] <= nc <= σ_range[2]) ? 1.0 :
      1 / clamp(nc, 1e-8, 1e8)
  # Scale the caller's matrices in their own storage type. The two-sided
  # diagonal product is not bitwise symmetric (the products are formed in
  # different orders above and below the diagonal), so symmetrize
  # explicitly: solvers and user callbacks may check `issymmetric(Q)`.
  Q̃ = σ * (Diagonal(Dc) * Q * Diagonal(Dc))
  Q̃ = (Q̃ + Q̃') / 2
  Ã = Diagonal(Dr) * A * Diagonal(Dc)
  G̃ = Diagonal(De) * G * Diagonal(Dc)
  return (Q = Q̃, c = σ .* c̃, A = Ã, b = Dr .* b, G = G̃, d = De .* d,
          Dc = Dc, Dr = Dr, De = De, σ = σ)
end

# Map a Solution of the scaled problem back to the original coordinates,
# renormalize certificates, and recompute the reported residuals from
# the original data.
function unequilibrate!(sol::Solution, eq, Q, c, A, b, cone_dims, G, d)
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

  # A point (optimal, best iterate). The loop already evaluated prFeas,
  # duFeas and muFeas in the original coordinates (see `scaling` in
  # _conicIP); recompute the feasibility residuals and objectives from
  # the original data as the postsolve truth, and rescale μ.
  y, w, v, s = sol.y, sol.w, sol.v, sol.s
  if all(isfinite, y) && all(isfinite, v) && all(isfinite, s) && all(isfinite, w)
    Qy   = Q * y
    cᵀy  = dot(c, y)
    rDu  = norm(Qy + G' * w - A' * v - c) / (1 + norm(c))
    rPr  = isempty(b) ? 0.0 : norm(A * y - s - b) / (1 + norm(b))
    rEq  = isempty(d) ? 0.0 : norm(G * y - d) / (1 + norm(d))
    pobj = 0.5 * dot(y, Qy) - cᵀy
    dobj = pobj + dot(w, G * y - d) + dot(v, A * y - s - b) - dot(v, s)
    sol.duFeas = rDu
    sol.prFeas = max(rPr, rEq)
    sol.pobj   = pobj
    sol.dobj   = dobj
    sol.Mu     = sol.Mu / σ
  end
  return sol
end
