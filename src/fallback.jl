# ──────────────────────────────────────────────────────────────
#  Fallback certificate QPs
#
#  When the interior-point loop exhausts its iteration budget
#  without a verdict, the best iterate may still be *pointing* at
#  a ray without carrying one accurately enough to validate. The
#  two routines below recover a candidate ray directly, by solving
#  a small strongly convex auxiliary QP whose feasible set is
#  exactly the set of certificates of the original problem:
#
#    infeasibility (Farkas)   min ½‖(w,v)‖²
#                             s.t.  Gᵀw - Aᵀv = 0
#                                   dᵀw - bᵀv = -1
#                                   v ∈ K
#
#    unboundedness (recession) min ½‖y‖²
#                              s.t.  Gy = 0,  Qy = 0,  cᵀy = 1
#                                    Ay ∈ K
#
#  (Recall conicIP MINIMIZES ½yᵀQy - cᵀy, so the auxiliary min-norm
#   objective is Q_aux = I, c_aux = 0.)
#
#  Each auxiliary problem is strongly convex, so it has a unique
#  minimum-norm solution when a certificate exists, and is
#  *infeasible* exactly when none does — returning `nothing` for a
#  well-posed original problem is the expected outcome, not a
#  failure.
#
#  Recursion guard: both calls pass `certFallback = false`, so an
#  auxiliary solve can never spawn a fallback of its own. This is
#  structural, not a depth counter.
#
#  These routines never claim anything. They return a *candidate*
#  ray; the caller must run it through the validators in
#  certificates.jl against the ORIGINAL problem data.
# ──────────────────────────────────────────────────────────────

"""
    fallback_infeasibility_ray(Q, c, A, b, cone_dims, G, d;
                               kktsolver = kktsolver_qr, maxIters = 50)

Solve the minimum-norm Farkas auxiliary QP and return a candidate
infeasibility ray `(w, v)`, or `nothing` if the auxiliary solve does not
reach `:Optimal` (which includes the common case that the original problem
is feasible, making the auxiliary problem infeasible).

The returned pair is *not* validated and *not* normalized — pass it to
[`validate_infeasibility_certificate`](@ref).
"""
function fallback_infeasibility_ray(Q, c, A, b, cone_dims, G, d;
                                    kktsolver = kktsolver_qr,
                                    maxIters = 50)

  n = length(c)
  m = size(A, 1)
  p = size(G, 1)

  (p + m) > 0 || return nothing

  # min ½‖(w,v)‖²  (conicIP form: min ½xᵀQx - cᵀx)
  Q_aux = sparse(1.0I, p + m, p + m)
  c_aux = zeros(p + m)

  # [Gᵀ  -Aᵀ] [w;v] = 0     (n rows)
  # [dᵀ  -bᵀ] [w;v] = -1    (1 row)
  G_aux = [sparse(G')                            -sparse(A')                           ;
           sparse(reshape(collect(Float64, d), 1, p))  -sparse(reshape(collect(Float64, b), 1, m))]
  d_aux = [zeros(n); -1.0]

  # v ∈ K
  A_aux = [spzeros(m, p) sparse(1.0I, m, m)]
  b_aux = zeros(m)

  # G_aux is routinely rank deficient (dependent rows of G, or simply
  # n + 1 > p + m). staticReg regularizes the Q block only, and every
  # kktsolver factors a singular KKT matrix here, so trim to an independent
  # consistent row set first. Dropping dependent rows leaves the feasible set
  # alone up to imcols' tolerance, and the caller re-validates the ray against
  # the original data anyway.
  (IP, consistent) = try imcols(G_aux, d_aux) catch; (Int[], false) end
  IP = collect(Int, IP)
  # An empty row set would drop the normalizing row and leave a meaningless
  # auxiliary problem, so treat it as a failure rather than solving it.
  (consistent && !isempty(IP)) || return nothing
  G_aux = G_aux[IP, :]; d_aux = d_aux[IP]

  sol = try
    conicIP(Q_aux, c_aux, A_aux, b_aux, cone_dims, G_aux, d_aux;
            kktsolver = kktsolver,
            maxIters = maxIters,
            verbose = false,
            staticReg = 1e-8,
            certFallback = false)   # recursion guard
  catch
    return nothing
  end

  sol.status == :Optimal || return nothing

  w = sol.y[1:p]
  v = sol.y[(p + 1):(p + m)]

  return (w, v)

end

"""
    fallback_unbounded_ray(Q, c, A, b, cone_dims, G, d;
                           kktsolver = kktsolver_qr, maxIters = 50)

Solve the minimum-norm recession auxiliary QP and return a candidate
unboundedness ray `y`, or `nothing` if the auxiliary solve does not reach
`:Optimal` (in particular when no recession ray exists).

The `Qy = 0` rows are omitted when `Q` is identically zero, which keeps the
auxiliary system at its smallest for the LP case. The returned ray is *not*
validated and *not* normalized — pass it to
[`validate_unboundedness_certificate`](@ref).
"""
function fallback_unbounded_ray(Q, c, A, b, cone_dims, G, d;
                                kktsolver = kktsolver_qr,
                                maxIters = 50)

  n = length(c)
  m = size(A, 1)
  p = size(G, 1)

  n > 0 || return nothing

  # min ½‖y‖²
  Q_aux = sparse(1.0I, n, n)
  c_aux = zeros(n)

  # Gy = 0 ; (Qy = 0 when Q ≠ 0) ; cᵀy = 1
  Qzero = normsafe(Q) == 0
  crow  = sparse(reshape(collect(Float64, c), 1, n))

  G_aux = Qzero ? [sparse(G); crow] : [sparse(G); sparse(Q); crow]
  d_aux = [zeros(size(G_aux, 1) - 1); 1.0]

  # Ay ∈ K
  A_aux = A
  b_aux = zeros(m)

  # G_aux is rank deficient by construction whenever the Qy = 0 rows are
  # present, and it then has more rows than columns (p + n + 1 > n), which
  # kktsolver_qr cannot factor at all. Trim to an independent consistent row
  # set; the caller re-validates the ray against the original data.
  (IP, consistent) = try imcols(G_aux, d_aux) catch; (Int[], false) end
  IP = collect(Int, IP)
  # An empty row set would drop the normalizing row and leave a meaningless
  # auxiliary problem, so treat it as a failure rather than solving it.
  (consistent && !isempty(IP)) || return nothing
  G_aux = G_aux[IP, :]; d_aux = d_aux[IP]

  sol = try
    conicIP(Q_aux, c_aux, A_aux, b_aux, cone_dims, G_aux, d_aux;
            kktsolver = kktsolver,
            maxIters = maxIters,
            verbose = false,
            staticReg = 1e-8,
            certFallback = false)   # recursion guard
  catch
    return nothing
  end

  sol.status == :Optimal || return nothing

  return sol.y

end
