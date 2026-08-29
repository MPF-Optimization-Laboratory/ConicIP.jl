"""
  imcols(A, b; ϵ = 1e-10)

Removes redundant inequalities in a system of equations

Ax = b

and checks if the equations are consistent.
"""
function imcols(A, b, ϵ = 1e-8)

  A = sparse(A)

  if isempty(A); return ([], true); end

  # 0·x = b is consistent iff b ≈ 0; guard the normalization against 0/0
  nA = norm(A)
  if nA == 0; return ([], norm(b, Inf) <= ϵ); end

  A = A/nA; b = b/nA

  F = qr(sparse(A'))
  R_mat = F.R
  n_r = min(size(R_mat)...)
  diag_R = [abs(R_mat[i,i]) for i in 1:n_r]
  piv = F.pcol  # column permutation
  R = sort(piv[findall(diag_R .> ϵ)])

  if isempty(R); return ([], true); end

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
- `c ∉ range([Q Aᵀ Gᵀ])` → `:Unbounded` with a recession ray `y`

Rank deficiency that is *not* an inconsistency is handled by opting into
`conicIP`'s static KKT regularization rather than by perturbing `Q`.
"""
function preprocess_conicIP(Q, c::AbstractVector,
  A, b::AbstractVector, cone_dims,
  G = spzeros(0,length(c)), d = zeros(0);
  verbose = false,
  options...)

  if verbose == true
    println()
    println(" > INTERIOR POINT PREPROCESSOR v0.7.1 (Aug 2016)")
    println()
  end

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  # Certificate tolerances: honour whatever is forwarded to conicIP, and
  # otherwise fall back on conicIP's own defaults.
  opts   = (; options...)
  reltol = get(opts, :infeasTol,    1e-7)
  abstol = get(opts, :infeasAbsTol, 1e-9)

  nanvec(k) = fill(NaN, k)

  (IP, pconsistent) = imcols(G, d)

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

  (ID, dconsistent) = imcols([Q A' G[IP,:]'], c)

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
        :Unbounded, 0, NaN, NaN, NaN, NaN, NaN, NaN, true) :
      ConicIP.Solution(nanvec(n), nanvec(p), nanvec(m), nanvec(m),
        :Unbounded, 0, NaN, NaN, NaN, NaN, NaN, NaN, false)

  end

  if (verbose == true) && (length(IP) != p)
    println("   - Removing $(p - length(IP)) redundant primal constraints ");
  end

  if (verbose == true) && (length(ID) != n)
    println("   - Rank deficient dual constraints: enabling static regularization");
  end

  if (verbose == true) &&  (length(ID) == n) && (length(IP) == p)
    println("   - No changes made")
  end

  # Rank deficiency in [Q Aᵀ G_IPᵀ] used to be patched by adding a 0/1 diagonal
  # to Q, which silently changes the objective. Instead opt into conicIP's static
  # KKT regularization, which leaves the problem alone. A caller-supplied
  # staticReg always wins (and is stripped from options to avoid a duplicate
  # keyword argument).
  reg  = haskey(opts, :staticReg) ? opts[:staticReg] :
         (length(ID) < n ? 1e-8 : 0.0)
  rest = Base.structdiff(opts, NamedTuple{(:staticReg,)})

  sol = conicIP(Q, c, A, b, cone_dims, G[IP,:], d[IP];
    verbose = verbose,       #                   |
    staticReg = reg,         # Removed redundant linear constraints
    rest...)                 # TODO : (use view?)

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

  if sol.status == :Infeasible && sol.has_certificate
    (check, w̄, v̄) = validate_infeasibility_certificate(Q, c, A, b, cone_dims,
      G, d, sol.w, sol.v; abstol = abstol, reltol = reltol)
    if check.valid
      sol.w = w̄; sol.v = v̄
    else
      sol.w = nanvec(p); sol.v = nanvec(m); sol.has_certificate = false
    end
  end

  # Same for an unbounded ray: it satisfies Gȳ ≈ 0 on the reduced rows only,
  # so re-validate against the full G before letting the certificate stand.
  if sol.status == :Unbounded && sol.has_certificate
    (check, ȳ) = validate_unboundedness_certificate(Q, c, A, b, cone_dims,
      G, d, sol.y; abstol = abstol, reltol = reltol)
    if check.valid
      sol.y = ȳ; sol.s = A*ȳ
    else
      sol.y = nanvec(n); sol.s = nanvec(m); sol.has_certificate = false
    end
  end

  return sol

end
