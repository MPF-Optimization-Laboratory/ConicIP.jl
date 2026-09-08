"""
  imcols(A, b, ϵ = 1e-8)

Removes redundant inequalities in a system of equations

Ax = b

and checks if the equations are consistent. Returns `(R, consistent)`
where `R` are the indices of a maximal independent row set.
"""
function imcols(A, b, ϵ = 1e-8)

  A = sparse(A)

  if isempty(A); return ([], true); end

  # 0·x = b is consistent iff b ≈ 0; guard the normalization against 0/0
  nA = norm(A)
  if nA == 0; return ([], norm(b, Inf) <= ϵ); end

  A = A/nA; b = b/nA

  # SPQR of A'. Row selection uses Heath-style dead-column detection:
  # a small |R[i,i]| marks column pcol[i] of A' (row pcol[i] of A) as
  # dependent on the kept ones — the same heuristic SPQR itself uses for
  # rank detection. It is a heuristic, not a strong rank-revealing
  # factorization; the certificate validators downstream re-validate any
  # verdict against the original data, which is what makes this sound.
  F = qr(sparse(A'))
  diag_R = abs.(diag(F.R))
  n_r = min(size(F.R)...)
  piv = F.pcol  # fill-reducing column permutation
  R = sort(piv[findall(view(diag_R, 1:n_r) .> ϵ)])

  if isempty(R); return ([], true); end

  # Consistency: solve with the kept rows and verify against all rows.
  # NB this is a second sparse factorization. Reusing F alone is not
  # sound: SPQR's dead columns leave nonzero off-diagonal rows in R, so
  # an R-only test with the dead coordinates zeroed checks membership in
  # a subspace of the true row space and can misreport inconsistency.
  return (norm(A*(A[R,:]\b[R]) - b, Inf) < ϵ) ? (R, true) : ([], false)

end

"""
ConicIP with preprocessing to ensure the following
rank constraints

Primal equailty constraints : Gx = d
Rank condition              : rank(G) = size(G,1)

Dual equality constraints   : [ Q A' G'] = c
Rank condition              : rank([Q A' G']) = size(Q,1)

Inconsistent data is reported with a certificate whenever one can be
constructed and verified against the original problem data:

- `Gy = d` inconsistent  → `:Infeasible` with a Farkas ray `(w,v)`
- `c ∉ range([Q Aᵀ Gᵀ])` → `:DualInfeasible` with a recession ray `y`

Rank deficiency that is *not* an inconsistency is handled by opting into
`conicIP`'s static KKT regularization rather than by perturbing `Q`.

`rank_check` controls the sparse-QR rank detection, which is the
expensive part: `:always` runs it; `:never` skips it and relies on the
KKT solver's regularization (what [`kktsolver_ldl`](@ref) provides);
`:auto` (default) runs it only when the KKT solver that will be used
needs a full-rank `G` — the dense QR solver, chosen for small and
semidefinite problems, and any explicit solver other than `kktsolver_ldl`.

Before either, singleton equality rows (`gᵢⱼ yⱼ = dᵢ`) fix their variable
and are removed together with the column; the fixed values, their
equality duals, and the objective values are restored on the way out
(`fix_singletons = false` disables this step).
"""
function preprocess_conicIP(Q, c::AbstractVector,
  A, b::AbstractVector, cone_dims,
  G = spzeros(0,length(c)), d = zeros(0);
  verbose = false,
  rank_check = :auto,
  fix_singletons = true,
  options...)

  if verbose == true
    println()
    println(" > INTERIOR POINT PREPROCESSOR")
    println()
  end

  fx = fix_singletons ? _singleton_fixings(G, d) : nothing
  if fx === nothing
    return _preprocess_core(Q, c, A, b, cone_dims, G, d;
                            verbose = verbose, rank_check = rank_check, options...)
  end

  n = length(c); m = size(A, 1); p = size(G, 1)
  # rows: every singleton row dropped (rowcols: its column; primary: the
  # first row on that column, or a consistent duplicate); cols/vals: the
  # distinct fixed columns and their values.
  (; rows, rowcols, primary, cols, vals, conflict) = fx

  if conflict !== nothing
    # Two singleton rows i, k on column j with dᵢ/gᵢⱼ ≠ dₖ/gₖⱼ. The Farkas
    # ray w = ±(eᵢ/gᵢⱼ − eₖ/gₖⱼ), v = 0 has Gᵀw = 0 and dᵀw ≠ 0; pick the
    # sign with dᵀw < 0 and let the validator normalize and confirm it.
    (i, k, j) = conflict
    Gs = sparse(G)
    w0 = zeros(p); w0[i] = 1 / Gs[i, j]; w0[k] = -1 / Gs[k, j]
    dot(d, w0) > 0 && (w0 .= .-w0)
    opts = (; options...)
    (chk, w̄, v̄) = validate_infeasibility_certificate(Q, c, A, b, cone_dims, G, d,
        w0, zeros(m); abstol = get(opts, :infeasAbsTol, 1e-9),
                      reltol = get(opts, :infeasTol, 1e-7))
    if verbose
      println("   - Conflicting singleton equality rows $i and $k on variable $j",
              chk.valid ? " (certified infeasible)" : " (no valid certificate)")
    end
    nanv(k) = fill(NaN, k)
    return chk.valid ?
      Solution(nanv(n), w̄, v̄, nanv(m), :Infeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, true) :
      Solution(nanv(n), nanv(p), nanv(m), nanv(m), :Infeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, false)
  end

  keepc = setdiff(1:n, cols)
  keepr = setdiff(1:p, rows)
  yfix  = zeros(n); yfix[cols] = vals
  if verbose
    println("   - Fixing $(length(cols)) variable(s) from singleton equality rows")
  end

  # Reduced data: ½ȳᵀQ̄ȳ − c̄ᵀȳ with c̄ = c − Q[:,F] y_F on the kept columns,
  # Ā ȳ ≥ b − A[:,F] y_F, Ḡ ȳ = d − G[:,F] y_F on the kept rows.
  Qs = sparse(Q); As = sparse(A); Gs = sparse(G)
  cr = (c - Qs * yfix)[keepc]
  br = b - As * yfix
  dr = (d - Gs * yfix)[keepr]
  # The reduced objective omits the constant ½y_FᵀQ_FF y_F − c_Fᵀy_F of the
  # fixed variables. It is passed through as `objective_offset` so that the
  # reduced solve's relative gap test ⟨v,s⟩/(1 + |pobj|) is the full
  # problem's, not one scaled by a possibly much smaller reduced objective.
  # A caller-supplied offset (nested presolves) is added to it.
  opts0  = (; options...)
  offset = 0.5 * dot(yfix, Qs * yfix) - dot(c, yfix) + get(opts0, :objective_offset, 0.0)
  sol = _preprocess_core(Qs[keepc, keepc], cr, As[:, keepc], br, cone_dims,
                         Gs[keepr, keepc], dr;
                         verbose = verbose, rank_check = rank_check,
                         objective_offset = offset,
                         Base.structdiff(opts0, NamedTuple{(:objective_offset,)})...)

  # Postsolve. The primal is the fixed value on F. The dual of the primary
  # singleton row i on variable j comes from stationarity of column j:
  #   (Qy)ⱼ + (Gᵀw)ⱼ − (Aᵀv)ⱼ = cⱼ  ⇒  gᵢⱼ wᵢ = cⱼ − (Qy)ⱼ + (Aᵀv)ⱼ − Σ_{k≠i} gₖⱼ wₖ.
  # A consistent duplicate singleton row on the same column gets wᵢ = 0 (the
  # dual of a redundant row is free to be anything; zero keeps the primary
  # row's stationarity formula exact). Rays: a primal ray has zero fixed
  # components; a Farkas ray needs wᵢ chosen so that column j of Gᵀw − Aᵀv
  # vanishes, which is the same formula with c and Qy dropped.
  y = fill(NaN, n); w = fill(NaN, p)
  if all(isfinite, sol.y)
    y[keepc] = sol.y
    if sol.status == :DualInfeasible && sol.has_certificate
      y[cols] .= 0.0
    else
      y[cols] = vals
    end
  end
  if all(isfinite, sol.w) && all(isfinite, sol.v)
    w[keepr] = sol.w
    ray = sol.status == :Infeasible && sol.has_certificate
    wk = zeros(p); wk[keepr] = sol.w
    rhs = ray ? (As' * sol.v) : (c - Qs * y + As' * sol.v)
    w[rows[.!primary]] .= 0.0                     # duplicates first: they enter `other` as 0
    for (i, j, isprimary) in zip(rows, rowcols, primary)
      isprimary || continue
      gij = Gs[i, j]
      other = dot(Gs[:, j], wk) - gij * wk[i]     # Σ_{k≠i} gₖⱼ wₖ (wk[i] = 0)
      w[i] = (rhs[j] - other) / gij
      wk[i] = w[i]
    end
  end
  sol.y = y; sol.w = w
  if sol.status == :DualInfeasible && sol.has_certificate && all(isfinite, y)
    sol.s = As * y
  elseif all(isfinite, y) && all(isfinite, w) && all(isfinite, sol.v) && all(isfinite, sol.s)
    # Objective values of the full problem: the reduced solve dropped the
    # constant ½y_FᵀQ_FF y_F − c_Fᵀy_F carried by the fixed variables.
    Qy = Qs * y
    sol.pobj = 0.5 * dot(y, Qy) - dot(c, y)
    sol.dobj = sol.pobj + dot(w, Gs * y - d) + dot(sol.v, As * y - sol.s - b) -
               dot(sol.v, sol.s)
  end
  return _check_postsolve!(sol, Q, c, A, b, cone_dims, G, d;
      optTol = get(opts0, :optTol, 1e-6),
      objective_offset = get(opts0, :objective_offset, 0.0),
      infeasTol = get(opts0, :infeasTol, 1e-7),
      infeasAbsTol = get(opts0, :infeasAbsTol, 1e-9))
end

# Singleton equality rows: rows of G with exactly one structural nonzero.
# Returns `nothing` when there are none, else a named tuple with
#   rows, rowcols, primary — every dropped row, the column it fixes, and
#                            whether it is the first (primary) row on that
#                            column; a later consistent duplicate is
#                            dropped as redundant (`primary = false`);
#   cols, vals             — the distinct fixed columns and values dᵢ/gᵢⱼ,
#                            in primary-row order (`cols == rowcols[primary]`);
#   conflict               — `nothing`, or `(i, k, j)`: two singleton rows
#                            on column j whose fixed values disagree beyond
#                            `tol` (relative), which makes the problem
#                            infeasible.
function _singleton_fixings(G, d; tol = 1e-9)
  Gs = sparse(G)
  p = size(Gs, 1)
  p == 0 && return nothing
  cnt = zeros(Int, p); lastcol = zeros(Int, p)
  rowsv = rowvals(Gs); nzv = nonzeros(Gs)
  for j in 1:size(Gs, 2), t in nzrange(Gs, j)
    if nzv[t] != 0
      cnt[rowsv[t]] += 1; lastcol[rowsv[t]] = j
    end
  end
  rows = Int[]; rowcols = Int[]; primary = Bool[]
  cols = Int[]; vals = Float64[]
  fixrow = zeros(Int, size(Gs, 2))       # row that fixed each column
  fixval = zeros(size(Gs, 2))
  conflict = nothing
  for i in 1:p
    cnt[i] == 1 || continue
    j = lastcol[i]
    v = d[i] / Gs[i, j]
    if fixrow[j] == 0
      fixrow[j] = i; fixval[j] = v
      push!(rows, i); push!(rowcols, j); push!(primary, true)
      push!(cols, j); push!(vals, v)
    elseif abs(v - fixval[j]) <= tol * (1 + abs(fixval[j]))
      push!(rows, i); push!(rowcols, j); push!(primary, false)   # consistent duplicate
    elseif conflict === nothing
      conflict = (fixrow[j], i, j)
    end
  end
  isempty(rows) && return nothing
  return (rows = rows, rowcols = rowcols, primary = primary,
          cols = cols, vals = vals, conflict = conflict)
end

function _preprocess_core(Q, c::AbstractVector,
  A, b::AbstractVector, cone_dims,
  G = spzeros(0,length(c)), d = zeros(0);
  verbose = false,
  rank_check = :auto,
  options...)

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  # Certificate tolerances: honour whatever is forwarded to conicIP, and
  # otherwise fall back on conicIP's own defaults.
  t_start = time()
  opts   = (; options...)
  reltol = get(opts, :infeasTol,    1e-7)
  abstol = get(opts, :infeasAbsTol, 1e-9)

  nanvec(k) = fill(NaN, k)

  # Rank detection is needed only by KKT solvers without regularization
  # of the equality block. Resolve the solver the same way conicIP will.
  rc = Symbol(rank_check)
  rc in (:auto, :always, :never) ||
    throw(ArgumentError("rank_check must be :auto, :always or :never (got $rank_check)"))
  ks = get(opts, :kktsolver, default_kktsolver)
  do_rank = rc == :always ? true :
            rc == :never  ? false :
            (ks === default_kktsolver ? choose_kktsolver(Q, A, G, cone_dims) !== kktsolver_ldl :
                                        !(ks === kktsolver_ldl || ks isa cached_kktsolver_ldl))

  if !do_rank
    if verbose; println("   - Rank detection skipped (KKT solver regularizes)"); end
    IP = collect(1:p); pconsistent = true
    ID = collect(1:n); dconsistent = true
  else
    (IP, pconsistent) = imcols(G, d)
  end

  if !pconsistent

    # Gy = d has no solution. The least-squares residual r = d - G(G\d) is
    # orthogonal to range(G), so Gᵀ(-r) = 0 and dᵀ(-r) = -‖r‖² < 0: the pair
    # (w,v) = (-r, 0) is a Farkas ray (v = 0 ∈ K trivially). Solve through
    # sparse QR, which is rank-revealing — plain \ throws SingularException
    # for a square rank-deficient G. A failed solve just means no certificate.
    r = try d - G*(qr(sparse(G)) \ d) catch; fill(NaN, p) end
    (check, w̄, v̄) = validate_infeasibility_certificate(Q, c, A, b, cone_dims,
      G, d, -r, zeros(m); abstol = abstol, reltol = reltol)

    if verbose == true
      println("   - Primal equality constraints inconsistent",
              check.valid ? " (certified)" : " (no valid certificate)")
    end

    return check.valid ?
      ConicIP.Solution(nanvec(n), w̄, v̄, nanvec(m),
        :Infeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, true) :
      ConicIP.Solution(nanvec(n), nanvec(p), nanvec(m), nanvec(m),
        :Infeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, false)

  end

  if do_rank
    (ID, dconsistent) = imcols([Q A' G[IP,:]'], c)
  end

  if !dconsistent

    # c ∉ range(M), M = [Q Aᵀ G_IPᵀ]: the dual is inconsistent, which is
    # primal unboundedness. The residual y = c - M(M\c) lies in null(Mᵀ), so
    # Qy = 0, Ay = 0 ∈ K (on the boundary), Gy = 0, and cᵀy = ‖y‖² > 0.
    M = [Q A' G[IP,:]']
    y_res = try c - M*(qr(sparse(M)) \ c) catch; fill(NaN, n) end
    (check, ȳ) = validate_unboundedness_certificate(Q, c, A, b, cone_dims,
      G, d, y_res; abstol = abstol, reltol = reltol)

    if verbose == true
      println("   - Dual equality constraints inconsistent (primal unbounded)",
              check.valid ? " (certified)" : " (no valid certificate)")
    end

    return check.valid ?
      ConicIP.Solution(ȳ, nanvec(p), nanvec(m), A*ȳ,
        :DualInfeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, true) :
      ConicIP.Solution(nanvec(n), nanvec(p), nanvec(m), nanvec(m),
        :DualInfeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, false)

  end

  if (verbose == true) && (length(IP) != p)
    println("   - Removing $(p - length(IP)) redundant primal constraints ");
  end

  if (verbose == true) && (length(ID) != n)
    println("   - Rank deficient dual constraints: enabling static regularization");
  end

  if (verbose == true) && do_rank && (length(ID) == n) && (length(IP) == p)
    println("   - No changes made")
  end

  # Rank deficiency in [Q Aᵀ G_IPᵀ] used to be patched by adding a 0/1 diagonal
  # to Q, which silently changes the objective. Instead opt into conicIP's static
  # KKT regularization, which leaves the problem alone. A caller-supplied
  # staticReg always wins (and is stripped from options to avoid a duplicate
  # keyword argument).
  reg  = haskey(opts, :staticReg) ? opts[:staticReg] :
         (length(ID) < n ? 1e-8 : 0.0)
  rest = Base.structdiff(opts, NamedTuple{(:staticReg, :timeLimit)})

  # The wall-clock budget covers the rank detection above and any retry
  # below, not just the solve.
  timeLimit = get(opts, :timeLimit, Inf)
  time_left() = timeLimit - (time() - t_start)

  sol = conicIP(Q, c, A, b, cone_dims, G[IP,:], d[IP];
    verbose = verbose,       #                   |
    staticReg = reg,         # Removed redundant linear constraints
    timeLimit = time_left(),
    rest...)                 # TODO : (use view?)

  # One retry with static regularization on a numerical (:Error) failure,
  # only when the first attempt ran unregularized and the caller did not
  # pin staticReg themselves. A rank-deficient G is deliberately NOT
  # retried this way: staticReg touches only the Q block and cannot cure
  # it (imcols already trimmed dependent rows above).
  if sol.status == :Error && do_rank && reg == 0.0 && !haskey(opts, :staticReg) &&
     time_left() > 0
    if verbose; println("   - KKT failure; retrying once with staticReg = 1e-8"); end
    sol = conicIP(Q, c, A, b, cone_dims, G[IP,:], d[IP];
      verbose = verbose, staticReg = 1e-8, timeLimit = time_left(), rest...)
  end

  # Re-expand the equality duals over the original rows, zero on the dropped
  # ones. This is exact for the two quantities the certificate identity uses:
  # Gᵀw_full = Σ_{i∈IP} w_i gᵢ = G[IP,:]ᵀ sol.w and dᵀw_full = d[IP]ᵀ sol.w,
  # since the dropped entries are zero. What the dropped rows being (numerical)
  # combinations of the kept ones buys us is that the reduced feasible set is
  # the original one -- imcols only checked that to tolerance ϵ, so a ray for
  # the reduced data need not certify the original data to the certificate
  # tolerance. Hence the re-validation below rather than a bare re-expansion.
  if all(isfinite, sol.w)
    w = zeros(p); w[IP] = sol.w; sol.w = w
  else
    sol.w = nanvec(p)   # keep a non-certificate ray NaN rather than part-zero
  end

  # A ray that certifies the reduced data but not the original one means the
  # reduction changed the problem (imcols dropped a row that was only
  # dependent to tolerance ϵ). The reduced problem's verdict then says
  # nothing about the caller's problem, so it must not survive as a terminal
  # status: downgrade to :Error rather than report an uncertified claim.
  # The iterate belongs to a different problem, so it is NaN'd rather than
  # returned as a "best iterate".
  function retract!(sol, claim)
    dropped = setdiff(1:p, IP)
    sol.y = nanvec(n); sol.w = nanvec(p); sol.v = nanvec(m); sol.s = nanvec(m)
    sol.has_certificate = false
    sol.status  = :Error
    sol.message = string(claim, " claimed on the reduced equality system ",
                         "(dropped rows ", dropped, ") but the ray does not ",
                         "certify the original data")
    return sol
  end

  if sol.status == :Infeasible && sol.has_certificate
    (check, w̄, v̄) = validate_infeasibility_certificate(Q, c, A, b, cone_dims,
      G, d, sol.w, sol.v; abstol = abstol, reltol = reltol)
    if check.valid
      sol.w = w̄; sol.v = v̄
    else
      return retract!(sol, :Infeasible)
    end
  end

  # Same for an unbounded ray: it satisfies Gȳ ≈ 0 on the reduced rows only,
  # so re-validate against the full G before letting the certificate stand.
  if sol.status == :DualInfeasible && sol.has_certificate
    (check, ȳ) = validate_unboundedness_certificate(Q, c, A, b, cone_dims,
      G, d, sol.y; abstol = abstol, reltol = reltol)
    if check.valid
      sol.y = ȳ; sol.s = A*ȳ
    else
      return retract!(sol, :DualInfeasible)
    end
  end

  length(IP) == p && return sol
  return _check_postsolve!(sol, Q, c, A, b, cone_dims, G, d;
      optTol = get(opts, :optTol, 1e-6),
      objective_offset = get(opts, :objective_offset, 0.0),
      infeasTol = reltol, infeasAbsTol = abstol)

end
