module ConicIP

export Id, conicIP, pivot, preprocess_conicIP,
  Optimizer, Block, choose_kktsolver, default_kktsolver

import Base: +, *, -, \, ^
using LinearAlgebra
using LinearAlgebra.BLAS: axpy!, scal!
using SparseArrays
using WoodburyMatrices
using Printf

# Verbose log: an iteration row is printed in bold red when the scaled KKT
# residual of the step, ‖r - K Δz‖/(n + p + 2m), is still above this after
# the iterative-refinement loop. It flags an ill-conditioned KKT system.
const REFINE_WARN_NORM = 1e-3

"""
    Id(n)

Create an `n`-by-`n` identity matrix as `Diagonal(ones(n))`.
"""
Id(n::Integer) = Diagonal(ones(n))

# ──────────────────────────────────────────────────────────────
#  Linear operator representing a congruence transform of a
#  matrix in vectorized form
#  (struct defined here so blockmatrices.jl can reference it
#   in the BlockElem union; methods added after mat/vecm below)
# ──────────────────────────────────────────────────────────────

"""
    VecCongurance(R)

Linear operator representing a congruence transform in vectorized form.
The action `W * x` computes `vecm(R' * mat(x) * R)`.

Used internally as the Nesterov-Todd scaling matrix for semidefinite cones.
"""
mutable struct VecCongurance; R :: Matrix; end

Base.adjoint(W::VecCongurance)         = VecCongurance(W.R');
Base.inv(W::VecCongurance)             = VecCongurance(inv(W.R))
Base.size(W::VecCongurance, i)         = round(Int, size(W.R,1)*(size(W.R,1)+1)/2)
*(W1::VecCongurance, W2::VecCongurance) = VecCongurance(W2.R * W1.R)

include("blockmatrices.jl")
include("kktsolvers.jl")

ViewTypes   = Union{SubArray}
VectorTypes = Union{Vector, ViewTypes}
MatrixTypes = Union{Matrix, Array{Real,2},
                    SparseMatrixCSC{Real,Integer}}

# returns 0 for matrices with dimension 0.
normsafe(x) = isempty(x) ? 0 : norm(x)

# ──────────────────────────────────────────────────────────────
#  3x1 block vector
# ──────────────────────────────────────────────────────────────

mutable struct v4x1; y::Vector{Float64}; w::Vector{Float64}; v::Vector{Float64}; s::Vector{Float64}; end

LinearAlgebra.norm(a::v4x1) = norm(a.y) + normsafe(a.w) + normsafe(a.v) + normsafe(a.s)

# All four blocks finite. Screening for this *before* handing an iterate or
# a direction to LAPACK matters: a NaN reaching a factorization raises
# ArgumentError("matrix contains Infs or NaNs"), which is deliberately not a
# KKT_FAILURE and would escape the solver. (`all(isfinite, Δ)` would be a
# MethodError -- v4x1 is not iterable.)
isfinite4(Δ::v4x1) = all(isfinite, Δ.y) && all(isfinite, Δ.w) &&
                     all(isfinite, Δ.v) && all(isfinite, Δ.s)

function axpy4!(α::Number, x::v4x1, y::v4x1)
    axpy!(α, x.y, y.y); axpy!(α, x.w, y.w)
    axpy!(α, x.v, y.v); axpy!(α, x.s, y.s)
end

# dest = a - b, in place (no allocation)
function sub4!(dest::v4x1, a::v4x1, b::v4x1)
    dest.y .= a.y .- b.y; dest.w .= a.w .- b.w
    dest.v .= a.v .- b.v; dest.s .= a.s .- b.s
    return dest
end

# VecCongurance methods that depend on mat/vecm (defined below)
*(W::VecCongurance, x::AbstractVector) = vecm(W.R'*mat(x)*W.R)

# Column-wise action on matrices. Needed for Block*Matrix products with
# SDP blocks; the old `x::VectorTypes` signature also captured 2-D
# SubArrays and fed whole matrices to mat().
function *(W::VecCongurance, X::AbstractMatrix)
  Y = zeros(size(X,1), size(X,2))
  for j in axes(X,2)
    Y[:,j] = W * view(X,:,j)
  end
  return Y
end

function Base.Matrix(W::VecCongurance)
  n = size(W,1)
  Imat = Matrix{Float64}(LinearAlgebra.I, n, n)
  Z = zeros(n,n)
  for i = 1:n
    Z[:,i] = W*Imat[:,i][:]
  end
  return Z
end

function SparseArrays.sparse(W::VecCongurance)
  return sparse(Matrix(W))
end

ord(x) = begin; n = length(x); round(Int, (sqrt(1+8*n) - 1)/2); end

"""
    mat(x)

Convert a vectorized symmetric matrix (scaled lower-triangular form) back
to a full symmetric matrix. Inverse of [`vecm`](@ref).
"""
function mat(x)

  # inverse of vecm
  # > mat([1,2,3,4,5,6)]
  #  1    2/√2  3√2
  #  2    4     5√2
  #  3√2  5√2   6

  n = ord(x)
  return mat!(zeros(n,n), x)

end

"""
    mat!(Z, x)

In-place [`mat`](@ref): fill the symmetric matrix `Z` from the
vectorized form `x`. `Z` must be `ord(x)` square.
"""
function mat!(Z, x)

  n = size(Z,1)
  s = 1/√2
  c = 1
  @inbounds for i = 1:n
    Z[i,i] = x[c]; c += 1
    for j = i+1:n
      v = x[c]*s
      Z[i,j] = v; Z[j,i] = v
      c += 1
    end
  end
  return Z

end

"""
    vecm(Z)

Vectorize a symmetric matrix `Z` into scaled lower-triangular form.
Off-diagonal entries are scaled by `√2` so that
`dot(vecm(X), vecm(Y)) == tr(X*Y)`. Inverse of [`mat`](@ref).
"""
function vecm(Z)

  # inverse of mat
  # > vecm([1 2 3; 2 4 5; 3 5 6])
  # [1 2√2 3√2 4 5√2 6]

  n = size(Z,1)
  return vecm!(zeros((n*(n+1)) >> 1), Z)

end

"""
    vecm!(x, Z)

In-place [`vecm`](@ref): write the vectorized form of the symmetric
matrix `Z` into `x`, which must have length `n(n+1)/2`.
"""
function vecm!(x, Z)

  n = size(Z,1)
  c = 1
  @inbounds for i = 1:n
    x[c] = Z[i,i]; c += 1
    for j = i+1:n
      x[c] = Z[i,j]*√2; c += 1
    end
  end
  return x

end

# ──────────────────────────────────────────────────────────────
#  Misc Helper Types/Functions
# ──────────────────────────────────────────────────────────────

# Example:  block_ranges([1,1,3]) = [1:1,2:2,3:5]
cum_range(x) = [i:(j-1) for (i,j) in
        zip(cumsum([1;x])[1:end-1], cumsum([1;x])[2:end])]
QF(r) = 2*r[1]*r[1] - dot(r,r)
Q(x::VectorTypes,y::VectorTypes) = 2*x[1]*y[1] - dot(x,y) # xᵀJy
fts(x₁, α₁, y₁, x₂, α₂, y₂)      = dot(x₁,x₂) - α₂*dot(x₁,y₂) -
          α₁*dot(y₁,x₂) + α₁*α₂*dot(y₁,y₂) # (x₁ - α₁*y₁)'(x₂ - α₂y₂)

function nestod_soc(z,s)

  # Nesterov-Todd Scaling Matrix for the second order cone
  # Matrix which satisfies the properties
  # W*z = inv(W)*s

  n = size(z,1)

  # QF(x) = x₁² - ‖x̄‖² is the Jordan determinant of the SOC. It is
  # positive in the interior, and roundoff on a near-boundary iterate can
  # push it to zero or below — at which point β and γ silently become Inf
  # or NaN and a non-finite scaling matrix escapes into the KKT solve.
  # Refuse it here so that a boundary SOC iterate surfaces as a guarded
  # factorization failure, exactly as a boundary SDP iterate does through
  # the cholesky in nestod_sdc.
  (QF(z) > 0 && QF(s) > 0) || throw(LinearAlgebra.PosDefException(1))

  β = (QF(s)/QF(z))^(1/4)

  # Normalize z,s vectors
  z = z/sqrt(QF(z))
  s = s/sqrt(QF(s))

  γ = sqrt((1 + dot(z,s))/2)

  # Jz = J*z;
  scal!(length(z), -1., z, 1)
  z[1] = -z[1]

  w = (1.0 ./ (2.0 .* γ)) .* (s + z)
  w[1] = w[1] + 1
  scal!(length(w), (sqrt(2*β)/sqrt(2*w[1])), w, 1)

  J = Diagonal(Float64[β for i = 1:n])
  J = Diagonal([-β; fill(β, n-1)])

  return SymWoodbury(J, vec(w), 1.)

end

function nestod_sdc(z,s)

  # Nesterov-Todd Scaling Matrix for the Semidefinite Cone
  # Matrix which satisfies the property
  # F*z = inv(F')*s
  # (equivalently F'*(F*z) = s).  Note inv(F') and not inv(F): the two
  # agree only when mat(z) and mat(s) commute.

  Ls  = cholesky(Symmetric(mat(s))).L
  Lz  = cholesky(Symmetric(mat(z))).L
  F   = svd(Lz'*Ls)
  U   = F.U
  Λ   = F.S
  # R = inv(Lz)'*U*diagm(sqrt.(Λ)), formed as a triangular solve plus a
  # column scaling: inv(Lz)' = inv(Lz') so inv(Lz)'*U is Lz' \ U, and
  # right-multiplying by a diagonal scales the columns.
  R = (Lz' \ U) .* sqrt.(Λ)'
  return VecCongurance(R)

end

function maxstep_rp(x,d)

  # Assume x in R+.
  # Returns maximum α such that x - α*d in R+.
  # (every cone's maxstep uses the minus convention: the solver step is
  #  z ← z - α*Δz.)

  minVal = Inf
  for i = 1:length(x)
    if d[i] > 0
      minVal = min(minVal, x[i]/d[i])
    end
  end
  return minVal

end

function maxstep_rp(x, e::Nothing)

  # Let α = inf { α | -x + αe >= 0 }
  # Then this returns
  # 0       if α < 0   (point is STRICTLY feasible)
  # 1 + α   otherwise

  if all(x .> 0)
    return 0;
  else
    return -1 + minimum(x);
  end

end

function maxstep_soc(x,d)

  # Assume x in Q.
  # Returns maximum α such that x - α*d in Q.

  d = -d;
  γ = Q(x,x)
  xbar = x/sqrt(γ)
  β = Q(xbar,d)

  ρ1 = β /sqrt(γ)
  μ  = (β + d[1])/(xbar[1] + 1)
  ρ2 = (d[2:end] - μ*xbar[2:end])
  alpha = norm(ρ2)/sqrt(γ) - ρ1
  if alpha < 0
    return Inf
  else
    return 1/alpha
  end

end

function maxstep_soc(x, e::Nothing)

  # Maximum step to cone
  α = norm(x[2:end]) - x[1];
  return α < 0 ? 0 : -1 - α;

end

function maxstep_sdc(x,d)

  # Maximum α such that mat(x) - α*mat(d) ⪰ 0, for X = mat(x) ≻ 0.
  #
  # For X ≻ 0,
  #
  #     X - αD ⪰ 0  ⟺  I - α X^{-1/2} D X^{-1/2} ⪰ 0  ⟺  α·λmax ≤ 1,
  #
  # where λmax is the largest eigenvalue of X^{-1/2} D X^{-1/2}. Those
  # eigenvalues are exactly the generalized eigenvalues of the symmetric-
  # definite pencil (D, X), so one LAPACK sygvd call — a Cholesky of X
  # plus one symmetric eigen-decomposition — replaces the three
  # decompositions the explicit form needs.
  #
  # sygvd factors X, so X ⋡ 0 raises PosDefException instead of being
  # answered with Inf; conicIP catches it and reports :Error.
  Λ = eigvals(Symmetric(mat(d)), Symmetric(mat(x)))
  # An order-0 block constrains nothing, so it never limits the step.
  isempty(Λ) && return Inf
  # sygvd can return NaN rather than throw when X is positive definite but
  # scaled into the subnormal range; treat that as a factorization failure.
  all(isfinite, Λ) || throw(LinearAlgebra.LAPACKException(0))
  λmax = maximum(Λ)
  # λmax ≤ 0 means every direction of D moves *into* the cone. The
  # comparison (rather than a sign mask) is what makes a direction of
  # signed zeros — which kktsolver_sparse produces for a mathematically
  # zero step — return +Inf rather than 1/(-0.0) = -Inf.
  return λmax <= 0 ? Inf : 1/λmax

end

function maxstep_sdc(x,d::Nothing)

  # Maximum step to Semidefinite cone (see maxstep_rp(x, ::Nothing))
  # An order-0 block is vacuously strictly feasible and needs no shift.
  isempty(x) && return 0
  minΛ = eigmin(Symmetric(mat(x)))
  return minΛ > 0 ? 0 : -1 + minΛ

end

function drp!(x, y, o)

    @inbounds @simd for i = 1:length(x); o[i] = x[i]/y[i]; end

end

function xrp!(x, y, o)

    @inbounds @simd for i = 1:length(x); o[i] = x[i]*y[i]; end

end

function dsoc!(y,x, o)

  # Inverse of arrow matrix
  #     ┌                         ┐ ┌    ┐
  # α⁻¹ │  y1  -yb                │ │ x1 │
  #     │ -yb   (αI + yb*yb')/y1  │ │ xb │
  #     └                         ┘ └    ┘

  @inbounds y1 = x[1];
  @inbounds yb = view(x,2:length(x))
  α = y1^2 - dot(yb,yb)

  @inbounds x1 = y[1];
  @inbounds xb = view(y,2:length(x))
  o[1] = (y1*x1 - dot(yb,xb) )/α
  β1 = ((-x1/α) + dot(yb,xb)/(y1*α))
  β2 = 1/y1
  @inbounds @simd for i = 2:length(o)
    o[i] = yb[i-1]*β1 + xb[i-1]*β2
  end

end

function xsoc!(x, y, o)

  o[1] = dot(x,y)
  @inbounds @simd for i = 2:length(x); o[i] = x[1]*y[i] + y[1]*x[i]; end

end

function dsdc!(x, y, o)

  # Inverse of the Jordan product: solve (Y*Z + Z*Y)/2 = X for Z.
  # lyap(A,C) solves A*Z + Z*A' + C = 0, and Y is symmetric, so the
  # equation is Z = lyap(Y, -2X). lyap goes through LAPACK's
  # overflow-scaled Sylvester solver, which an eigen-basis Hadamard
  # formula would give up.
  X = mat(x); Y = mat(y)
  Z = lyap(Y, -2 .* X)
  vecm!(o, (Z .+ Z') ./ 2)

end

function xsdc!(x, y, o)

  # Jordan product (X*Y + Y*X)/2 of the semidefinite cone, whose identity
  # is I = mat(e) — matching the e assembled in conicIP. For symmetric X
  # and Y, (X*Y)' = Y'*X' = Y*X, so the symmetrization below is exactly
  # the Jordan product.
  X = mat(x); Y = mat(y)
  XY = X*Y
  vecm!(o, (XY .+ XY') ./ 2)

end

# ──────────────────────────────────────────────────────────────
#  Interior Point
# ──────────────────────────────────────────────────────────────

"""
    Solution

Return type of [`conicIP`](@ref) and [`preprocess_conicIP`](@ref).

# Fields
- `y::Vector{Float64}` -- primal variables
- `w::Vector{Float64}` -- dual variables for equality constraints (Gy = d)
- `v::Vector{Float64}` -- dual variables for inequality constraints (Ay ≥_K b)
- `s::Vector{Float64}` -- cone slack variables (Ay - s = b, s ∈ K)
- `status::Symbol` -- `:Optimal`, `:Infeasible`, `:Unbounded`,
  `:AlmostInfeasible`, `:AlmostUnbounded`, `:Abandoned`, or `:Error`
- `Iter::Integer` -- number of interior-point iterations
- `Mu::Real` -- final complementarity gap parameter
- `prFeas::Real` -- primal feasibility residual
- `duFeas::Real` -- dual feasibility residual
- `muFeas::Real` -- complementarity residual
- `pobj::Real` -- primal objective value
- `dobj::Real` -- dual objective value
- `has_certificate::Bool` -- the returned vectors carry a *verified* ray
  certifying infeasibility or unboundedness (see the table below)

# Field conventions by status

| status | y | w | v | s | pobj/dobj | has_certificate |
|:--|:--|:--|:--|:--|:--|:--|
| `:Optimal` | solution | dual (eq) | dual (ineq), ∈ K | slack, ∈ K | real | `false` |
| `:Infeasible` *with ray* | all `NaN` | ray `w̄` | ray `v̄` ∈ K | all `NaN` | `NaN` | `true` |
| `:Unbounded` *with ray* | ray `ȳ` | all `NaN` | all `NaN` | `A*ȳ` | `NaN` | `true` |
| `:Infeasible`/`:Unbounded` *without ray* | all `NaN` | all `NaN` | all `NaN` | all `NaN` | `NaN` | `false` |
| `:Abandoned`, `:AlmostInfeasible`, `:AlmostUnbounded`, `:Error` | best iterate | best iterate | best iterate | best iterate | best iterate | `false` |

The infeasibility ray is normalized so that `dᵀw̄ - bᵀv̄ = -1` with
`Gᵀw̄ - Aᵀv̄ ≈ 0`; the unboundedness ray is normalized so that `cᵀȳ = +1`
with `Qȳ ≈ 0`, `Gȳ ≈ 0` and `Aȳ ∈ K`. See
[`validate_infeasibility_certificate`](@ref) and
[`validate_unboundedness_certificate`](@ref).

A 12-argument constructor is provided which defaults `has_certificate` to
`false`.
"""
mutable struct Solution

  y      :: Vector{Float64}  # primal
  w      :: Vector{Float64}  # dual (linear equality)
  v      :: Vector{Float64}  # dual (linear inequality)
  s      :: Vector{Float64}  # cone slack (Ay - s = b, s ∈ K)
  status :: Symbol  # :Optimal, :Infeasible
  Iter   :: Integer # number of iterations
  Mu     :: Real    # optimality conditions
  prFeas :: Real
  duFeas :: Real
  muFeas :: Real
  pobj   :: Real
  dobj   :: Real
  has_certificate :: Bool  # y/w/v carry a verified ray
  message :: String        # diagnostic detail (e.g. the factorization
                           # failure behind an :Error status); "" otherwise

end

# 12/13-argument constructors: no certificate / no message by default
Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj) =
  Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, false, "")
Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate) =
  Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate, "")

# Overwrite sol with a *verified* infeasibility ray (dᵀw̄ - bᵀv̄ = -1).
# The primal iterate is discarded: it means nothing on an empty feasible set.
function claim_infeasible!(sol::Solution, w̄, v̄)
  fill!(sol.y, NaN); fill!(sol.s, NaN)
  sol.w[:] = w̄; sol.v[:] = v̄
  sol.pobj = NaN; sol.dobj = NaN
  sol.status = :Infeasible
  sol.has_certificate = true
  return sol
end

# Overwrite sol with a *verified* recession ray (cᵀȳ = +1). The duals are
# discarded: they mean nothing when the dual is infeasible.
function claim_unbounded!(sol::Solution, ȳ, A)
  sol.y[:] = ȳ; sol.s[:] = A*ȳ
  fill!(sol.w, NaN); fill!(sol.v, NaN)
  sol.pobj = NaN; sol.dobj = NaN
  sol.status = :Unbounded
  sol.has_certificate = true
  return sol
end

# ──────────────────────────────────────────────────────────────
#  Structural degeneracy and KKT failure handling (issue #10)
# ──────────────────────────────────────────────────────────────

# Factorization failures that degrade to a clean :Error status instead of
# escaping the solver. Deliberately narrow: anything else (BoundsError,
# ArgumentError, …) signals a broken invariant and must propagate.
const KKT_FAILURES = Union{LinearAlgebra.SingularException,
                           LinearAlgebra.PosDefException,
                           LinearAlgebra.ZeroPivotException,
                           LinearAlgebra.LAPACKException}

# Rows/columns of M with no structural nonzero entry, in O(nnz).
function structurally_zero_rows(M::SparseMatrixCSC)
  z = trues(size(M,1))
  rv = rowvals(M); nz = nonzeros(M)
  @inbounds for k in 1:length(rv)
    if nz[k] != 0; z[rv[k]] = false; end
  end
  return z
end
structurally_zero_rows(M::AbstractMatrix) =
  BitVector(Bool[all(iszero, view(M,i,:)) for i in 1:size(M,1)])

function structurally_zero_cols(M::SparseMatrixCSC)
  z = trues(size(M,2))
  nz = nonzeros(M)
  @inbounds for j in 1:size(M,2)
    # NB: a comma-form `for j, k` would exit both loops on break
    for k in nzrange(M,j)
      if nz[k] != 0; z[j] = false; break; end
    end
  end
  return z
end
structurally_zero_cols(M::AbstractMatrix) =
  BitVector(Bool[all(iszero, view(M,:,j)) for j in 1:size(M,2)])

"""
  conicIP(Q, c, A, b, cone_dims, G, d;
  kktsolver = default_kktsolver,
  optTol = 1e-6,
  DTB = 0.01,
  verbose = true,
  maxRefinementSteps = 3,
  maxIters = 100,
  cache_nestodd = false,
  infeasTol = 1e-7,
  infeasAbsTol = 1e-9,
  staticReg = 0.0,
  certFallback = true,
  certFallbackIters = 50,
  refinementThreshold = optTol/1e7)

Interior point solver for the system

```
minimize    ½yᵀQy - cᵀy
s.t         Ay >= b
            Gy  = d
```

c, b, d are vectors (or any AbstractVector)

cone_dims is an array of tuples (Cone Type, Dimension)

```
e.g. [("R",2),("Q",4)] means
(y₁, y₂)          in  R+
(y₃, y₄, y₅, y₆)  in  Q
```

A semidefinite block is `("S", k)` with `k = n(n+1)/2` for an `n × n`
symmetric matrix. Those `k` rows carry the matrix in the `vecm` form:
the upper triangle read row by row, with the off-diagonal entries
scaled by `√2`, so that `dot(vecm(X), vecm(Y)) == tr(X*Y)`. See
[`vecm`](@ref) and [`mat`](@ref).

Returns a [`Solution`](@ref) whose `status` is one of

- `:Optimal` — `max(rDu, rPr, rCp, rEq) < optTol`.
- `:Infeasible` / `:Unbounded` — a ray passed a screen *and* was accepted
  by the corresponding validator; `has_certificate` is then `true`.
- `:AlmostInfeasible` / `:AlmostUnbounded` — set only at loop exhaustion,
  when the best iterate carries a ray that validates at `100*infeasTol`
  but not at `infeasTol`. The best iterate is retained.
- `:Abandoned` — iteration limit reached with no verdict.
- `:Error` — nonfinite residuals, a nonfinite search direction or
  iterate, or a KKT factorization failure (the reason is recorded in
  `sol.message`). Rank-deficient `G` handed directly to `conicIP`
  typically lands here; use [`preprocess_conicIP`](@ref) to trim
  redundant rows first. `staticReg` regularizes only the `Q` block of
  the KKT system and cannot repair a rank-deficient `G`.

Every factorization or cone line search that can fail on a boundary
iterate is guarded, so such a failure is reported as an `:Error` status
with a reason in `sol.message` and never as an escaped exception (issue
#10). A guarded failure returns immediately with the current iterate (the
initial point, with `Iter = 0` and `pobj = Inf`, if no iteration
completed); it does not go through the certificate-fallback path, which
runs only when the iteration limit is exhausted.

Structurally degenerate inputs are handled exactly before any
factorization: an all-zero row of `G` is deflated (`dᵢ = 0`) or answered
with a certified `:Infeasible` (`dᵢ ≠ 0`), and a variable absent from
`Q`, `A`, and `G` is deflated (`cⱼ = 0`) or answered with a certified
`:Unbounded` (`cⱼ ≠ 0`).

Selected keyword arguments:

- `infeasTol` — infeasibility-certificate tolerance, decoupled from `optTol`.
- `infeasAbsTol` — absolute tolerance for certificate validation.
- `staticReg` — static KKT regularization scale; `0` (default) disables it.
  `preprocess_conicIP` enables it when it detects rank deficiency.
- `certFallback` — enable fallback certificate solve on stall.

The parameter solve3x3gen allows the passing of a custom solver
for the KKT System, as follows

```
julia> L = solve3x3gen(F,F⁻ᵀ,Q,A,G)

Then this

julia> (a,b,c) = L(y,w,v)

solves the system
┌             ┐ ┌   ┐   ┌   ┐
│ Q   G'  -A' │ │ a │ = │ y │
│ G           │ │ b │   │ w │
│ A       FᵀF │ │ c │   │ v │
└             ┘ └   ┘   └   ┘
```

We can also wrap a 2x2 solver using pivot3gen(solve2x2gen)
The 2x2 solves the system

```
julia> L = solve2x2gen(F,F⁻ᵀ,Q,A,G)

Then this

julia> (a,b) = L(y,w)

solves the system

┌                     ┐ ┌   ┐   ┌   ┐
│ Q + Aᵀinv(FᵀF)A  G' │ │ a │ = │ y │
│ G                   │ │ b │   │ w │
└                     ┘ └   ┘   └   ┘
```
"""
function conicIP(

  # ½xᵀQx - cᵀx
  Q, c::AbstractVector,

  # Ax ≧ b
  A, b::AbstractVector, cone_dims,

  # Gx = d
  G = spzeros(0,length(c)), d = zeros(0);

  # Solver Parameters

  # L = solve3x3gen(F,F⁻ᵀ,Q,A,G)
  # L(a,b,c) solves the system
  # ┌             ┐ ┌   ┐   ┌   ┐
  # │ Q   G'  -A' │ │ a │ = │ y │
  # │ G           │ │ b │   │ w │
  # │ A       FᵀF │ │ c │   │ v │
  # └             ┘ └   ┘   └   ┘
  #
  # We can also wrap a 2x2 solver using pivot3gen(solve2x2gen)
  # The 2x2 solves the system
  #
  # L = solve2x2gen(F,F⁻ᵀ,Q,A,G)
  # L(a,b) solves
  # ┌                ┐ ┌   ┐   ┌   ┐
  # │ Q + AᵀFᵀFA  G' │ │ a │ = │ y │
  # │ G              │ │ b │   │ w │
  # └                ┘ └   ┘   └   ┘
  kktsolver = default_kktsolver,

  optTol = 1e-6,           # Optimal Tolerance
  DTB = 0.01,              # Distance to Boundary
  verbose = true,          # Verbose Output
  maxRefinementSteps = 3,  # Maximum number of IR Steps
  maxIters = 100,          # Maximum number of interior iterations
  cache_nestodd = false,   # Set to true if there are many small blocks
  infeasTol = 1e-7,        # Infeasibility threshold (this shouldn't need to be tweaked,
                           # but set it small if the program returns infeasible/unbounded when
                           # you are sure it isn't)
  infeasAbsTol = 1e-9,     # used by certificate validation (WP3b)
  staticReg = 0.0,         # Static regularization scale for the KKT factorization
                           # (0 disables it; preprocess_conicIP opts in when it
                           # detects rank deficiency in [Q A' G'])
  certFallback = true,     # enables fallback certificate solve (WP5)
  certFallbackIters = 50,  # iteration budget for each fallback solve
  refinementThreshold = optTol/1e7 # Accuracy of refinement steps
  )

  # Precomputed transposition matrices
  Aᵀ = A'; Gᵀ = G'

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  block_types  = [i[1] for i in cone_dims]
  block_sizes  = [i[2] for i in cone_dims]
  block_data   = zip(block_types, cum_range(block_sizes),
                     [i for i in 1:length(block_types)])

  # Pre-allocated buffers for in-place cone_div!/cone_prod! (avoids
  # zeros(m) per call), and v4x1 scratch for the refinement loop
  _div_buf   = zeros(m)
  _prod_buf1 = zeros(m)
  _prod_buf2 = zeros(m)
  _rkkt = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  _rIr  = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))

  # Pre-allocated Block for inv(F)' — reused each iteration
  F⁻ᵀ_cache = Block(size(block_sizes, 1))

  normc = norm(c)
  normd = isempty(d) ? -Inf : norm(d)
  normb = normsafe(b)
  normdsafe = normsafe(d)   # 0 for empty d (normd is -Inf there)

  # Sanity Checks
  ◂ = nothing
  size(Q,1) != size(Q,2) ? error("Q is not square") : ◂
  size(b,1) != m        ? error("Inconsistency in inequalities") : ◂
  size(c,1) != n        ? error("Inconsistency in inequalities/objective") : ◂
  size(d,1) != p        ? error("Inconsistency in equalities") : ◂
  size(G,2) != n        ? error("Inconsistency in equalities/objective") : ◂

  # ────────────────────────────────────────────────────────────
  #  Structural degeneracy: exact O(nnz) handling (issue #10)
  #
  #  A structurally zero row of G is either trivially infeasible
  #  (dᵢ ≠ 0 ⇒ the Farkas ray -sign(dᵢ)eᵢ) or vacuous (dᵢ = 0 ⇒
  #  deflate the row and re-expand the dual). A zero column of
  #  [Q; A; G] with cⱼ ≠ 0 admits the recession ray sign(cⱼ)eⱼ;
  #  with cⱼ = 0 the variable is deflated. Without these checks
  #  every KKT factorization below is exactly singular.
  # ────────────────────────────────────────────────────────────

  Gzr = structurally_zero_rows(G)
  Zc  = structurally_zero_cols(Q) .& structurally_zero_cols(A) .&
        structurally_zero_cols(G)

  i0 = findfirst(i -> Gzr[i] && d[i] != 0, 1:p)
  if i0 !== nothing
    w̄0 = zeros(p); w̄0[i0] = -sign(d[i0])
    (chk, w̄, v̄) = validate_infeasibility_certificate(
        Q, c, A, b, cone_dims, G, d, w̄0, zeros(m);
        abstol = infeasAbsTol, reltol = infeasTol)
    if chk.valid
      if verbose
        print("\n > EXIT -- Structurally infeasible (zero row $(i0) of G, d[$(i0)] ≠ 0)\n\n")
      end
      sol0 = Solution(fill(NaN,n), zeros(p), zeros(m), fill(NaN,m),
                      :None, 0, 0, Inf, Inf, Inf, NaN, NaN)
      return claim_infeasible!(sol0, w̄, v̄)
    end
  end

  j0 = findfirst(j -> Zc[j] && c[j] != 0, 1:n)
  if j0 !== nothing
    ȳ0 = zeros(n); ȳ0[j0] = sign(c[j0])
    (chk, ȳ) = validate_unboundedness_certificate(
        Q, c, A, b, cone_dims, G, d, ȳ0;
        abstol = infeasAbsTol, reltol = infeasTol)
    if chk.valid
      if verbose
        print("\n > EXIT -- Structurally unbounded (zero column $(j0) of [Q; A; G], c[$(j0)] ≠ 0)\n\n")
      end
      sol0 = Solution(zeros(n), fill(NaN,p), fill(NaN,m), zeros(m),
                      :None, 0, 0, Inf, Inf, Inf, NaN, NaN)
      return claim_unbounded!(sol0, ȳ, A)
    end
  end

  if any(Gzr) || any(Zc)
    keep_r = findall(.!Gzr)
    keep_c = findall(.!Zc)
    solr = conicIP(Q[keep_c, keep_c], c[keep_c], A[:, keep_c], b, cone_dims,
                   G[keep_r, keep_c], d[keep_r];
                   kktsolver = kktsolver, optTol = optTol, DTB = DTB,
                   verbose = verbose, maxRefinementSteps = maxRefinementSteps,
                   maxIters = maxIters, cache_nestodd = cache_nestodd,
                   infeasTol = infeasTol, infeasAbsTol = infeasAbsTol,
                   staticReg = staticReg, certFallback = certFallback,
                   certFallbackIters = certFallbackIters,
                   refinementThreshold = refinementThreshold)
    # Re-expand, inserting zeros at deflated positions when the block is
    # meaningful for the returned status (see the Solution field table);
    # blocks the convention leaves NaN stay all-NaN.
    best_iter = solr.status in (:Optimal, :Abandoned, :AlmostInfeasible,
                                :AlmostUnbounded, :Error, :None)
    w_ok = best_iter || (solr.status == :Infeasible && solr.has_certificate)
    y_ok = best_iter || (solr.status == :Unbounded && solr.has_certificate)
    w = fill(NaN, p); y = fill(NaN, n)
    if w_ok; fill!(w, 0.0); w[keep_r] = solr.w; end
    if y_ok; fill!(y, 0.0); y[keep_c] = solr.y; end
    return Solution(y, w, solr.v, solr.s, solr.status, solr.Iter, solr.Mu,
                    solr.prFeas, solr.duFeas, solr.muFeas, solr.pobj,
                    solr.dobj, solr.has_certificate, solr.message)
  end

  # Number to scale (z's) by
  # 1 for each R_+ dimension
  # 1 for each Q cone (regardless of dimension)
  conedim = 0
  for (btype, I, i) = block_data
    if btype == "R"; conedim += length(I);  end
    if btype == "Q"; conedim += 1;          end
    if btype == "S"; conedim += ord(I);     end
  end

  # e = conic group identity
  # Concatenate the vectors
  # [1, 1, … , 1] for R_+
  # [1, 0, … , 0] for Q
  # vecm(I)       for S
  e = zeros(m)
  for (btype, I, i) = block_data
    m_i = length(I)
    if btype == "R"; e[I] = ones(m_i);             end
    if btype == "Q"; e[I] = [1; zeros(m_i-1)];     end
    if btype == "S"; e[I] = vecm(Matrix{Float64}(LinearAlgebra.I, ord(I), ord(I)));  end
  end

  # ──────────────────────────────────────────────────────────────
  #  Functions capturing cone_dims
  # ──────────────────────────────────────────────────────────────

  function maxstep(x, d)

    # Linesearch

    min_α = Inf;
    @inbounds for (btype, I, i) = block_data
      xI = view(x,I)
      dI = ( d == nothing ? nothing : view(d,I) )
      if btype == "R"; α = maxstep_rp(xI,dI);  end
      if btype == "Q"; α = maxstep_soc(xI,dI); end
      if btype == "S"; α = maxstep_sdc(xI,dI); end
      min_α = min(α, min_α)
    end

    return min_α;

  end

  function nt_scaling(x, y)

    # Compute Nesterov-Todd scaling matrix, F s.t.
    # λ = F*x = inv(F')*y
    # For the self-adjoint R and Q blocks this is λ = F*x = F\y; for
    # an S block F is a congruence and only the adjoint form holds
    # (see nestod_sdc).

    B = Block(size(block_sizes,1));

    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I);
      if btype == "R"; B[i] = Diagonal(sqrt.(yI./xI)); end
      if btype == "Q"; B[i] = nestod_soc(xI, yI); end
      if btype == "S"; B[i] = nestod_sdc(xI, yI); end
    end

    return B;

  end

  function cone_div!(o,x,y)

    # In-place group division x ○\ y → o
    # (each per-cone primitive fully overwrites its output view)

    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I); oI = view(o,I)
      if btype == "R"; drp!(xI, yI, oI);  end
      if btype == "Q"; dsoc!(xI, yI, oI); end
      if btype == "S"; dsdc!(xI, yI, oI); end
    end
    return o;

  end

  function cone_prod!(o,x,y)

    # In-place group product x ○ y → o
    # (each per-cone primitive fully overwrites its output view)

    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I); oI = view(o,I)
      if btype == "R"; xrp!(xI, yI, oI);  end
      if btype == "Q"; xsoc!(xI, yI, oI); end
      if btype == "S"; xsdc!(xI, yI, oI); end
    end
    return o;

  end

  # Static regularization of the KKT factorization only. The factorization
  # sees Q + δI, but every other use of Q (residuals, objective, iterative
  # refinement) keeps the original Q — so the perturbed factorization acts as
  # a preconditioner whose error the refinement loop corrects.
  δ = staticReg*(1 + norm(Q, Inf))
  Qᵣ = δ == 0 ? Q : Q + δ*Id(n)

  # A KKT factorization failure (rank-deficient G, cone iterate at the
  # boundary, …) is reported as an :Error status with the reason in
  # sol.message, never as an escaped exception (issue #10).
  function kkt_error(where, err)
    msg = "KKT solve failed ($where): $(sprint(showerror, err))"
    if verbose; print("\n > EXIT -- Error! ($msg)\n\n"); end
    return msg
  end

  # :Error status carrying no iterate (nothing has been computed yet).
  errsol(msg) = Solution(fill(NaN,n), fill(NaN,p), fill(NaN,m), fill(NaN,m),
                         :Error, 0, 0, Inf, Inf, Inf, NaN, NaN, false, msg)

  if verbose && kktsolver === default_kktsolver
    chosen = choose_kktsolver(Qᵣ, A, G, cone_dims)
    nnz_pc = (_structural_nnz(Q) + _structural_nnz(A) + _structural_nnz(G)) / max(n, 1)
    @printf(" > KKT solver: %s (auto, %.1f nnz/col)\n", nameof(chosen), nnz_pc)
  end

  solve3x3gen = try
    kktsolver(Qᵣ,A,G,cone_dims)
  catch err
    err isa KKT_FAILURES || rethrow()
    return errsol(kkt_error("solver setup", err))
  end

  function solve4x4gen(λ, F, F⁻ᵀ, solve3x3gen = solve3x3gen)

    #
    # solve4x4gen(λ, F)(r) solves the 4x4 KKT System
    # ┌                  ┐ ┌    ┐   ┌     ┐
    # │ Q   G'  -A'      │ │ Δy │ = │ r.y │
    # │ G                │ │ Δw │   │ r.w │ S = block(λ)*F
    # │ A             -I │ │ Δv │   │ r.v │ V = block(λ)*F⁻ᵀ
    # │          S     V │ │ Δs │   │ r.s │
    # └                  ┘ └    ┘   └     ┘
    # F = Nesterov-Todd scaling matrix
    #

    solve3x3 = solve3x3gen(F, F⁻ᵀ)

    function solve4x4(r)

      cone_div!(_div_buf, r.s, λ)
      t1 = F'*_div_buf
      (Δy, Δw, Δv)  = solve3x3(r.y, r.w, r.v + t1)
      axpy!(-1, F'*(F*Δv), t1) # > Δs = t1 - F*(F*Δv)
      return v4x1(Δy,Δw,Δv,t1)

    end

  end

  if verbose
      print("\n > INTERIOR POINT SOLVER (ConicIP v$(pkgversion(@__MODULE__)))\n\n")
  end

  # ────────────────────────────────────────────────────────────
  #  Initial Point
  # ────────────────────────────────────────────────────────────

  I  = Block([Diagonal(ones(i)) for i = block_sizes])
  # Named apart from the loop's r0 so that the loop-local residual has a
  # single assignment site and is not boxed when the predictor closure
  # captures it.
  r_init = v4x1(c, d, b, zeros(m))
  z  = try
    solve4x4gen(e,I,I)(r_init)
  catch err
    err isa KKT_FAILURES || rethrow()
    return errsol(kkt_error("initial point", err))
  end

  # A nonfinite initial point would reach LAPACK through maxstep below.
  isfinite4(z) || return errsol("non-finite initial point (initial point)")

  (α_v, α_s) = try
    (maxstep(z.v, nothing), maxstep(z.s, nothing))
  catch err
    err isa KKT_FAILURES || rethrow()
    return errsol(kkt_error("initial point", err))
  end

  # Change to +
  z.v = z.v - α_v*e
  z.s = z.s - α_s*e

  if verbose
      println("            Optimality                      Objective              Infeasibility       ")
      println()
      printstyled(@sprintf(" %-6s  │  %-8s  %-8s  %-8s │  %-8s  %-8s  │  %-8s  %-8s │  %-8s \n",
                  "  Iter","prFeas","duFeas","muFeas","pobj","dobj","icertp","icertd","refine");
                  bold = true)
  end

  # ────────────────────────────────────────────────────────────
  #  Iterate Loop
  # ────────────────────────────────────────────────────────────

  sol     = Solution(copy(z.y), copy(z.w), copy(z.v), copy(z.s), :None, 0, 0, Inf, Inf, Inf, Inf, -Inf)
  optBest = Inf
  rStep   = 0
  rnorm   = 0
  μ_history = Float64[]   # complementarity gap per iteration (exhaustion path only)

  # ── Guards ──
  #
  # Every factorization and every cone line search below can raise a
  # KKT_FAILURE on a boundary iterate, and the documented contract is an
  # :Error status rather than an escaped exception. `guarded` stamps the
  # solution and returns `nothing`; a `return` inside the closure would
  # not leave conicIP, so each call site must `return sol` itself.
  # `nothing` is usable as the failure sentinel because no guarded closure
  # below can legitimately return it (they return a Block, a v4x1, or a
  # tuple of step lengths).
  function guarded(f, stage)
    try
      return f()
    catch err
      err isa KKT_FAILURES || rethrow()
      sol.status = :Error
      sol.message = kkt_error(stage, err)
      return nothing
    end
  end

  # Same verdict for a nonfinite direction or iterate, which has no
  # exception to report but must not be handed to LAPACK.
  function nonfinite!(what, stage)
    sol.status = :Error
    sol.message = "non-finite $what ($stage)"
    if verbose; print("\n > EXIT -- Error! ($(sol.message))\n\n"); end
    return sol
  end

  for Iter = 1:maxIters

    # Nesterov-Todd scaling matrix. nestod_sdc factors both cone iterates,
    # so a boundary iterate surfaces here as a PosDefException.
    Fλ = guarded("NT scaling, iteration $Iter") do
      Fi = nt_scaling(z.v, z.s)
      inv_adjoint!(F⁻ᵀ_cache, Fi)
      (Fi, Fi*z.v)                 # λ = F*z.v is also F⁻ᵀ*z.s
    end
    Fλ === nothing && return sol
    (F, λ) = Fλ
    F⁻ᵀ    = F⁻ᵀ_cache
    # A scaling that is non-finite without having thrown would reach
    # inv_adjoint! and the KKT solve as Inf/NaN.
    all(isfinite, λ) ||
      return nonfinite!("NT scaling", "NT scaling, iteration $Iter")

    solve = try
      solve4x4gen(λ,F,F⁻ᵀ)         # Caches 4x4 solver
                                   # (used a few times, at least 2)
    catch err
      err isa KKT_FAILURES || rethrow()
      sol.status = :Error
      sol.message = kkt_error("factorization, iteration $Iter", err)
      return sol
    end

    #         ┌                   ┐ ┌     ┐
    # rleft = │ Q   G'   -A'      │ │ z.y │
    #         │ G                 │ │ z.w │  V = block(λ)*F⁻ᵀ
    #         │ A              -I │ │ z.v │    = block(λ)*λ
    #         │           S     V │ │ z.s │
    #         └                   ┘ └     ┘
    cone_prod!(_prod_buf1, λ, λ)
    rleft = v4x1( Q*z.y + Gᵀ*z.w - Aᵀ*z.v ,
                  G*z.y                   ,
                  A*z.y - z.s             ,
                  _prod_buf1              )

    # True Residual of nonlinear KKT System
    r0 = v4x1(rleft.y - c, rleft.w - d, rleft.v - b, rleft.s);

    # Gap
    μbar = dot(z.v,z.s)
    μ    = μbar/conedim
    push!(μ_history, μ)

    # ────────────────────────────────────────────────────────────
    #  Print iterate status, save best iterate
    # ────────────────────────────────────────────────────────────

    cᵀy = dot(c,z.y)
    rDu = norm(r0.y)/(1+normc)
    rPr = normsafe(r0.v)/(1+normb)
    rCp = normsafe(r0.s)/(1+abs(cᵀy));
    rEq = normsafe(r0.w)/(1+normdsafe)   # Gy - d

    pobj = 0.5*dot(z.y, Q*z.y) - dot(c, z.y)
    dobj = pobj + dot(z.w, r0.w) + dot(z.v, r0.v) - dot(z.v, z.s)

    if max(rDu, rPr, rCp, rEq) < optBest
      sol.y[:] = z.y; sol.w[:] = z.w; sol.v[:] = z.v; sol.s[:] = z.s
      sol.Iter = Iter; sol.Mu = μ;
      sol.duFeas = rDu; sol.prFeas = rPr; sol.muFeas = rCp
      sol.pobj = pobj; sol.dobj = dobj
      optBest = max(rDu, rPr, rCp, rEq)
    end

    # ────────────────────────────────────────────────────────────
    #  Termination : candidate screen → validate → claim
    #
    #  Strict precedence. Optimality wins outright and returns; only
    #  a non-optimal iterate is screened for a ray. A screen hit is a
    #  *nomination* only — the status is claimed if and only if the
    #  validator in certificates.jl accepts the ray against the
    #  original problem data, and the claim returns immediately.
    # ────────────────────────────────────────────────────────────

    optimal = max(rDu, rPr, rCp, rEq) < optTol

    # Defined even when no screen runs (verbose row below reads them)
    p_infeas = NaN
    d_infeas = NaN

    claim = :None                      # set only by a validated ray
    w̄ = z.w; v̄ = z.v; ȳ = z.y          # normalized ray, once validated

    if !optimal && !(p == 0 && m == 0)

      # Primal Infeasibility (Farkas ray)
      #
      #  (w,v) with w free and v ∈ K certifies {y : Ay ≥_K b, Gy = d}
      #  is empty when
      #
      #    Gᵀw - Aᵀv = 0,   v ∈ K,   dᵀw - bᵀv < 0
      #
      #  The screen scales the residual ‖Gᵀw - Aᵀv‖ two ways, both
      #  gated on dᵀw - bᵀv < 0:
      #
      #   CVXOPT style             ECOS style
      #   ────────────────────     ────────────────────────
      #    ‖Gᵀw - Aᵀv‖              ‖Gᵀw - Aᵀv‖
      #    ───────────              ────────────────────────
      #      ‖w‖ + ‖v‖              max(1,‖c‖)·|dᵀw - bᵀv|
      #
      #  Passing the screen only nominates (w,v); the claim is made
      #  by validate_infeasibility_certificate, which also normalizes
      #  the ray to dᵀw̄ - bᵀv̄ = -1.
      dᵀw_bᵀv = dot(d,z.w) - dot(b,z.v)

      p_infeas_unscaled = norm(Gᵀ*z.w - Aᵀ*z.v)
      p_infeas_cvx  = dᵀw_bᵀv < 0 ? p_infeas_unscaled/(normsafe(z.w) + normsafe(z.v)) : NaN
      p_infeas_ecos = dᵀw_bᵀv < 0 ? p_infeas_unscaled/(max(1,normc)*abs(dᵀw_bᵀv)) : NaN
      p_infeas = max(p_infeas_cvx, p_infeas_ecos)

      if p_infeas < infeasTol
        (pchk, w̄, v̄) = validate_infeasibility_certificate(
                          Q, c, A, b, cone_dims, G, d, z.w, z.v;
                          abstol = infeasAbsTol, reltol = infeasTol)
        if pchk.valid; claim = :Infeasible; end
      end

      # Dual Infeasibility (recession ray)
      #
      #  y certifies ½yᵀQy - cᵀy is unbounded below over the feasible
      #  set when
      #
      #    Ay - s = 0, s ∈ K   (d_infeas1)
      #    Gy = 0              (d_infeas2)
      #    Qy = 0              (d_infeas3)
      #    cᵀy > 0
      #
      #  The screen scales max(d_infeas1, d_infeas2, d_infeas3) two
      #  ways, both gated on cᵀy > 0:
      #
      #   CVXOPT style                    ECOS style
      #   ─────────────────────────       ──────────────────────
      #    max(d₁/max(1,‖b‖),              max(d₁, d₂, d₃)
      #        d₂/max(1,‖d‖),              ───────────────
      #        d₃/max(1,‖c‖)) / |cᵀy|            ‖y‖
      #
      #  Again a nomination only: validate_unboundedness_certificate
      #  makes the claim and normalizes the ray to cᵀȳ = +1.
      d_infeas1 = isempty(A) ? -Inf : norm(A*z.y - z.s)
      d_infeas2 = isempty(G) ? -Inf : norm(G*z.y)
      d_infeas3 = all(isfinite.(z.y)) ? norm(Q*z.y) : NaN

      d_infeas_cvx  = cᵀy > 0 ? max(d_infeas1/max(1,normb), d_infeas2/max(1,normd), d_infeas3/max(1,normc))/abs(cᵀy) : NaN
      d_infeas_ecos = cᵀy > 0 ? max(d_infeas1, d_infeas2, d_infeas3)/norm(z.y) : NaN
      d_infeas = abs(max(d_infeas_cvx, d_infeas_ecos))

      if claim == :None && d_infeas < infeasTol
        (dchk, ȳ) = validate_unboundedness_certificate(
                       Q, c, A, b, cone_dims, G, d, z.y;
                       abstol = infeasAbsTol, reltol = infeasTol)
        if dchk.valid; claim = :Unbounded; end
      end

    end

    if verbose
      # A row is highlighted in red when the KKT step is still inaccurate
      # after iterative refinement (see REFINE_WARN_NORM).
      row = @sprintf(" %6i  │  %-8.1e  %-8.1e  %-8.1e │  % -8.1e  % -8.1e  │  %-8.1e  %-8.1e │  %i\n",
                     Iter, rPr, rDu, rCp, pobj, dobj, p_infeas, d_infeas, rStep)
      if rnorm > REFINE_WARN_NORM
        printstyled(row; bold = true, color = :red)
      else
        print(row)
      end
    end

    if optimal
      if verbose; print("\n > EXIT -- Below Tolerance!\n\n"); end
      sol.status = :Optimal
      return sol
    end

    if claim == :Infeasible
      if verbose; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
      return claim_infeasible!(sol, w̄, v̄)
    end

    if claim == :Unbounded
      if verbose; print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
      return claim_unbounded!(sol, ȳ, A)
    end

    # Cause of Divergence Unknown
    if !(isfinite(μ) && isfinite(rDu) && isfinite(rPr) && isfinite(rCp))
      if verbose; print("\n > EXIT -- Error!\n\n"); end
      sol.status = :Error; return sol
    end

    # ────────────────────────────────────────────────────────────
    #  Predictor
    # ────────────────────────────────────────────────────────────

    d_aff = guarded("predictor, iteration $Iter") do
      solve(r0)
    end
    d_aff === nothing && return sol
    isfinite4(d_aff) ||
      return nonfinite!("predictor direction", "predictor, iteration $Iter")

    α_aff_vs = guarded("predictor line search, iteration $Iter") do
      ( min( maxstep( z.v, d_aff.v ) , 1 ),
        min( maxstep( z.s, d_aff.s ) , 1 ) )
    end
    α_aff_vs === nothing && return sol
    α_aff = min( α_aff_vs[1] , α_aff_vs[2] )

    # >> ρ  = (z.v - α_aff*d_aff.v)'*(z.s - α_aff*d_aff.s)/μbar
    ρ  = fts(z.v, α_aff, d_aff.v, z.s, α_aff,d_aff.s)/μbar
    σ  = max(0,min(1,ρ))^3
    (isfinite(ρ) && isfinite(σ)) ||
      return nonfinite!("centering parameter", "predictor, iteration $Iter")

    # ────────────────────────────────────────────────────────────
    #  Corrector
    # ────────────────────────────────────────────────────────────

    F⁻ᵀdfs = F⁻ᵀ*d_aff.s
    Fdfs   = F*d_aff.v

    # >> lc = -(F⁻ᵀdfs ∘ Fdfs) + (σ*μ)[1]*e;
    cone_prod!(_prod_buf2, F⁻ᵀdfs, Fdfs); lc = _prod_buf2
    axpy!(-σ*μ, e, lc);
    scal!(length(e), -1., lc, 1)

    r  =  v4x1(r0.y, r0.w, r0.v, rleft.s - lc)

    # ────────────────────────────────────────────────────────────
    #  Take newton step, with iterative refinement
    # ────────────────────────────────────────────────────────────

    Δz = guarded("corrector, iteration $Iter") do
      solve(r)
    end
    Δz === nothing && return sol
    isfinite4(Δz) ||
      return nonfinite!("corrector direction", "corrector, iteration $Iter")

    rStep = 1;
    for rStep = 1:maxRefinementSteps
      cone_prod!(_prod_buf1, λ, F*Δz.v)
      cone_prod!(_prod_buf2, λ, F⁻ᵀ*Δz.s)
      # KKT residual of the step, into preallocated scratch:
      #   rkkt = (QΔy + GᵀΔw − AᵀΔv, GΔy, AΔy − Δs, λ∘FΔv + λ∘F⁻ᵀΔs)
      mul!(_rkkt.y, Q, Δz.y)
      mul!(_rkkt.y, Gᵀ, Δz.w, 1.0, 1.0)
      mul!(_rkkt.y, Aᵀ, Δz.v, -1.0, 1.0)
      mul!(_rkkt.w, G, Δz.y)
      mul!(_rkkt.v, A, Δz.y); _rkkt.v .-= Δz.s
      _rkkt.s .= _prod_buf1 .+ _prod_buf2
      sub4!(_rIr, r, _rkkt)
      rnorm = norm(_rIr)/(n + p + 2*m)
      if rnorm < refinementThreshold; break; end
      Δzr = guarded("refinement, iteration $Iter") do
        solve(_rIr)
      end
      Δzr === nothing && return sol
      isfinite4(Δzr) ||
        return nonfinite!("refinement direction", "refinement, iteration $Iter")
      axpy4!(1.0, Δzr, Δz)
    end
    isfinite4(Δz) ||
      return nonfinite!("search direction", "search direction, iteration $Iter")

    # ────────────────────────────────────────────────────────────
    # Make Step
    # ────────────────────────────────────────────────────────────

    # maxstep is homogeneous of degree -1 in the direction, so scaling the
    # step back from the boundary by (1-DTB) is the same as searching along
    # Δz/(1-DTB) — without forming the scaled direction.
    α_vs = guarded("line search, iteration $Iter") do
      ( min( 1, (1-DTB)*maxstep(z.v, Δz.v) ),
        min( 1, (1-DTB)*maxstep(z.s, Δz.s) ) )
    end
    α_vs === nothing && return sol
    α = min( α_vs[1], α_vs[2] )

    # >> z = z - α*Δz;
    axpy4!(-α, Δz, z)

    # The next iteration's nt_scaling factors this iterate.
    isfinite4(z) ||
      return nonfinite!("iterate", "line search, iteration $Iter")

  end

  # ────────────────────────────────────────────────────────────
  #  Loop exhausted : re-screen the best iterate
  #
  #  The screens above are evaluated on the *current* iterate and can
  #  miss a ray that the best iterate carries. Re-validate that
  #  iterate here, first at the nominal tolerance (a full claim, rare)
  #  and then relaxed 100×, which downgrades to :AlmostInfeasible /
  #  :AlmostUnbounded rather than claiming.
  # ────────────────────────────────────────────────────────────

  (pchk, w̄, v̄) = validate_infeasibility_certificate(
                    Q, c, A, b, cone_dims, G, d, sol.w, sol.v;
                    abstol = infeasAbsTol, reltol = infeasTol)
  (dchk, ȳ)    = validate_unboundedness_certificate(
                    Q, c, A, b, cone_dims, G, d, sol.y;
                    abstol = infeasAbsTol, reltol = infeasTol)

  (pchk100, _, _) = validate_infeasibility_certificate(
                      Q, c, A, b, cone_dims, G, d, sol.w, sol.v;
                      abstol = infeasAbsTol, reltol = 100*infeasTol)
  (dchk100, _)    = validate_unboundedness_certificate(
                      Q, c, A, b, cone_dims, G, d, sol.y;
                      abstol = infeasAbsTol, reltol = 100*infeasTol)

  # Secondary signal: complementarity collapsed while the residuals did
  # not — the signature of a problem with no interior optimum. It only
  # corroborates a *relaxed* verdict; a ray valid at 1× is never vetoed.
  μ_collapsed = length(μ_history) > 1 && isfinite(μ_history[end]) &&
                μ_history[end] <= 1e-3*maximum(μ_history)

  # Diverging complementarity is the classic infeasible-start signature of
  # an infeasible or unbounded problem (measured: μ can blow up by 1e38 on
  # an infeasible box). Either extreme — collapse or divergence — is
  # evidence that no interior optimum exists.
  μ_diverged = length(μ_history) > 1 && (!isfinite(μ_history[end]) ||
                μ_history[end] >= 1e3*minimum(μ_history))

  # ── WP5 fallback: recover a ray by an auxiliary min-norm QP ──
  #  Only when both 1× validations failed AND there is evidence a ray
  #  exists (a relaxed validation passed, or complementarity collapsed).
  #  A clean :Abandoned with no such signal does not earn a solve.
  #  At most one attempt of each kind. The auxiliary problems run under
  #  default_kktsolver rather than the caller's solver: they have a
  #  different structure (min-norm, wide equalities, regularized), so
  #  the selection heuristic is re-run on the auxiliary data.
  if certFallback && !pchk.valid && !dchk.valid &&
     (pchk100.valid || dchk100.valid || μ_collapsed || μ_diverged)

    # The auxiliary solves use their own iteration budget: the outer
    # maxIters is small in exactly the regime the fallback exists for.
    if (pchk100.valid || μ_collapsed || μ_diverged) && p + m > 0
      ray = fallback_infeasibility_ray(Q, c, A, b, cone_dims, G, d;
                                       maxIters = certFallbackIters)
      if ray !== nothing
        (fchk, fw̄, fv̄) = validate_infeasibility_certificate(
                            Q, c, A, b, cone_dims, G, d, ray[1], ray[2];
                            abstol = infeasAbsTol, reltol = infeasTol)
        if fchk.valid
          if verbose; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
          return claim_infeasible!(sol, fw̄, fv̄)
        end
      end
    end

    if (dchk100.valid || μ_collapsed || μ_diverged) && n > 0
      ray = fallback_unbounded_ray(Q, c, A, b, cone_dims, G, d;
                                   maxIters = certFallbackIters)
      if ray !== nothing
        (fchk, fȳ) = validate_unboundedness_certificate(
                       Q, c, A, b, cone_dims, G, d, ray;
                       abstol = infeasAbsTol, reltol = infeasTol)
        if fchk.valid
          if verbose; print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
          return claim_unbounded!(sol, fȳ, A)
        end
      end
    end

  end

  if pchk.valid
    if verbose; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
    return claim_infeasible!(sol, w̄, v̄)
  elseif dchk.valid
    if verbose; print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
    return claim_unbounded!(sol, ȳ, A)
  elseif pchk100.valid && μ_collapsed
    sol.status = :AlmostInfeasible
  elseif dchk100.valid && μ_collapsed
    sol.status = :AlmostUnbounded
  else
    sol.status = :Abandoned
  end

  return sol

end

include("certificates.jl")
include("fallback.jl")
include("preprocessor.jl")
include("MOI_wrapper.jl")

end
