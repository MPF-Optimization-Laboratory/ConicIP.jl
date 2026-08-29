# ──────────────────────────────────────────────────────────────
#  Infeasibility / Unboundedness Certificates
#
#  A certificate is a ray which proves that the problem
#
#    minimize    ½yᵀQy - cᵀy
#    s.t         Ay ≥_K b
#                Gy  = d
#
#  is primal infeasible (Farkas ray) or unbounded below
#  (recession ray). The validators below are pure functions of
#  the ORIGINAL problem data — they never look at solver state.
# ──────────────────────────────────────────────────────────────

"""
    CertificateCheck

Verdict returned by the certificate validators.

# Fields
- `valid::Bool` -- all checks passed
- `farkas_residual::Float64` -- ‖·‖_∞ of the linear residual of the ray
- `separation::Float64` -- the (pre-normalization) separation value; must be > 0
- `cone_margin::Float64` -- blockwise minimum cone margin of the ray (≥ 0 if in K)
- `finite::Bool` -- the candidate ray had only finite entries
"""
struct CertificateCheck
    valid            :: Bool
    farkas_residual  :: Float64
    separation       :: Float64
    cone_margin      :: Float64
    finite           :: Bool
end

# ‖x‖_∞, returning 0 for empty x
norminfsafe(x) = isempty(x) ? 0.0 : Float64(norm(x, Inf))

# M'x, returning zeros(size(M,2)) when x is empty
adjmulsafe(M, x) = isempty(x) ? zeros(size(M,2)) : M'*x

"""
    cone_margin(x, cone_dims)

Blockwise minimum margin of `x` with respect to the cone `K` described by
`cone_dims`. Nonnegative iff `x ∈ K`; the magnitude measures the distance
to the boundary (violation depth when negative).

Per block:

```
"R"  minimum(x[I])
"Q"  x[I][1] - ‖x[I][2:end]‖
"S"  eigmin(Symmetric(mat(x[I])))
```

Returns `+Inf` when there are no blocks.
"""
function cone_margin(x::AbstractVector, cone_dims) :: Float64

  block_types = [i[1] for i in cone_dims]
  block_sizes = [i[2] for i in cone_dims]

  isempty(block_sizes) && return Inf

  min_α = Inf
  for (btype, I) = zip(block_types, cum_range(block_sizes))
    xI = view(x, I)
    if     btype == "R"; α = isempty(xI) ? Inf : Float64(minimum(xI))
    elseif btype == "Q"; α = Float64(xI[1] - norm(view(xI, 2:length(xI))))
    elseif btype == "S"; α = Float64(eigmin(Symmetric(mat(xI))))
    else   error("Unrecognized cone type $btype")
    end
    min_α = min(min_α, α)
  end

  return min_α

end

"""
    validate_infeasibility_certificate(Q, c, A, b, cone_dims, G, d, w, v;
                                       abstol, reltol)

Validate `(w,v)` as a Farkas ray proving primal infeasibility of
`{y : Ay ≥_K b, Gy = d}`. A valid ray satisfies

```
Gᵀw̄ - Aᵀv̄ ≈ 0,    v̄ ∈ K,    dᵀw̄ - bᵀv̄ = -1
```

Returns `(check::CertificateCheck, w̄, v̄)` with the ray normalized so that
`dᵀw̄ - bᵀv̄ = -1`. If the separation `-(dᵀw - bᵀv)` is nonpositive, or the
candidate has nonfinite entries, the verdict is invalid and the candidates
are returned unchanged (no normalization).
"""
function validate_infeasibility_certificate(Q, c, A, b, cone_dims, G, d, w, v;
                                            abstol :: Float64,
                                            reltol :: Float64)

  finite = all(isfinite, w) && all(isfinite, v)

  # Separation: must be > 0 for the ray to separate b,d from range(A,G)
  separation = finite ? Float64(-(dot(d, w) - dot(b, v))) : NaN

  if !finite || !isfinite(separation) || separation <= 0
    return (CertificateCheck(false, NaN, separation, NaN, finite), w, v)
  end

  w̄ = w/separation
  v̄ = v/separation

  # Gᵀw̄ - Aᵀv̄ ≈ 0
  r = adjmulsafe(G, w̄) - adjmulsafe(A, v̄)
  farkas_residual = norminfsafe(r)

  margin = cone_margin(v̄, cone_dims)

  valid = finite &&
          farkas_residual <= abstol + reltol*(1 + normsafe(w̄) + normsafe(v̄)) &&
          margin >= -(abstol + reltol)

  return (CertificateCheck(valid, farkas_residual, separation, margin, finite), w̄, v̄)

end

"""
    validate_unboundedness_certificate(Q, c, A, b, cone_dims, G, d, y;
                                       abstol, reltol)

Validate `y` as a recession ray proving `½yᵀQy - cᵀy` is unbounded below
over the feasible set. A valid ray satisfies

```
Qȳ ≈ 0,    Gȳ ≈ 0,    Aȳ ∈ K,    cᵀȳ = +1
```

so that the objective decreases without bound along `ȳ`. Returns
`(check::CertificateCheck, ȳ)` with the ray normalized so that `cᵀȳ = +1`.
If the separation `cᵀy` is nonpositive, or the candidate has nonfinite
entries, the verdict is invalid and the candidate is returned unchanged.
"""
function validate_unboundedness_certificate(Q, c, A, b, cone_dims, G, d, y;
                                            abstol :: Float64,
                                            reltol :: Float64)

  finite = all(isfinite, y)

  # Objective slope along the ray is -cᵀy; require it strictly negative
  separation = finite ? Float64(dot(c, y)) : NaN

  if !finite || !isfinite(separation) || separation <= 0
    return (CertificateCheck(false, NaN, separation, NaN, finite), y)
  end

  ȳ = y/separation

  farkas_residual = max(norminfsafe(Q*ȳ), norminfsafe(G*ȳ))

  margin = cone_margin(A*ȳ, cone_dims)

  valid = finite &&
          farkas_residual <= abstol + reltol*(1 + normsafe(ȳ)) &&
          margin >= -(abstol + reltol)

  return (CertificateCheck(valid, farkas_residual, separation, margin, finite), ȳ)

end
