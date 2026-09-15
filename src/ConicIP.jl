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
include("timing.jl")
include("kktsolvers.jl")
include("kktsolver_ldl.jl")
include("correctors.jl")

ViewTypes   = Union{SubArray}
VectorTypes = Union{Vector, ViewTypes}
MatrixTypes = Union{Matrix, Array{Real,2},
                    SparseMatrixCSC{Real,Integer}}

# returns 0 for matrices with dimension 0.
normsafe(x) = isempty(x) ? 0 : norm(x)

# Entrywise absolute value |M| of a data matrix, for the backward-error
# style residual normalizations ‖|M||x|‖. Storage is preserved: sparse
# stays sparse, Diagonal stays Diagonal, and a Symmetric wrapper is kept
# (broadcasting through it would densify the parent).
_absmat(M::AbstractMatrix) = abs.(M)
_absmat(M::Symmetric) = Symmetric(_absmat(parent(M)), Symbol(M.uplo))

# ‖ |M| x ‖ for a nonnegative vector x, formed in `out`; with D a positive
# diagonal (vector) the product is divided entrywise by D and by σ first,
# which maps an equilibrated product back to the original coordinates
# (see the termination test in _conicIP).
function _absprod_norm!(out, M, x, D = nothing, σ = 1.0)
  (isempty(out) || isempty(x)) && return 0.0
  mul!(out, M, x)
  D === nothing || (out ./= D)
  return norm(out) / σ
end

# Largest row-wise relative residual of a block of rows,
#   max_i |r_i| / (1 + |b_i| + prod_i + |s_i|),
# with `prod` the componentwise product |M||y| already in the original
# coordinates (as _absprod_norm! leaves it) and `s` the slack (`nothing`
# for equality rows). `D` maps the equilibrated `r`, `b`, `s` back to the
# original coordinates (each divided by D_i). Allocation-free.
function _rowwise_max(r, b, prod, s, D = nothing)
  isempty(r) && return 0.0
  mx = 0.0
  @inbounds for i in eachindex(r)
    Di  = D === nothing ? 1.0 : D[i]
    si  = s === nothing ? 0.0 : abs(s[i]) / Di
    den = 1 + abs(b[i]) / Di + prod[i] + si
    mx  = max(mx, abs(r[i]) / Di / den)
  end
  return mx
end

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

# dest .= src, in place (no allocation)
function copy4!(dest::v4x1, src::v4x1)
    copyto!(dest.y, src.y); copyto!(dest.w, src.w)
    copyto!(dest.v, src.v); copyto!(dest.s, src.s)
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

# Jordan determinant of the unit vector x/‖x‖, QF(x)/‖x‖², computed on the
# normalized entries so that it neither overflows nor underflows for any
# finite nonzero x (QF(x) itself is 0 for ‖x‖ < 1e-154 and Inf beyond
# 1e154). It is positive exactly on the interior of the second-order cone
# (together with x₁ > 0) and is the quantity both the line search's
# interiority check and `nestod_soc` test.
function QFunit(x)
  nx = norm(x)
  nx > 0 || return 0.0
  ss = 0.0
  @inbounds for xi in x
    t = xi / nx
    ss += t * t
  end
  t1 = x[1] / nx
  return 2 * t1 * t1 - ss
end
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
  #
  # The scaling is homogeneous, W(αz, βs) = √(β/α)·W(z, s), so it is
  # computed on the unit vectors z/‖z‖, s/‖s‖ and rescaled: a strictly
  # interior pair with jointly extreme magnitudes (z ~ 1e-150, s ~ 1e150)
  # would otherwise overflow in QF(s)/QF(z) and hand back an Inf scaling.
  qz = QFunit(z); qs = QFunit(s)
  (qz > 0 && qs > 0) || throw(LinearAlgebra.PosDefException(1))
  nz = norm(z); ns = norm(s)

  # β = (QF(s)/QF(z))^(1/4) = (qs/qz)^(1/4) · √(‖s‖/‖z‖)
  β = (qs/qz)^(1/4) * (sqrt(ns)/sqrt(nz))

  # Normalize z,s vectors to QF = 1 (two divisions so that neither
  # ‖z‖·√qz nor its reciprocal is formed)
  z = (z ./ nz) ./ sqrt(qz)
  s = (s ./ ns) ./ sqrt(qs)

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

# ──────────────────────────────────────────────────────────────
#  In-place Nesterov-Todd scaling for the second-order cone
#
#  `nestod_soc` above allocates six vectors per cone per iteration
#  (the two normalized iterates, w, the J diagonal, and the two n-length
#  temporaries every SymWoodbury carries), and `adjoint(inv(·))` on the
#  result allocates as many again. The solver instead keeps one
#  `SOCScratch` per cone for the whole solve and calls `nestod_soc!` /
#  `soc_inv_adjoint!`, which perform exactly the operations of
#  `nestod_soc` and `adjoint(inv(·))`, in the same order, on those
#  buffers. The only per-iteration allocation left is the immutable
#  `SymWoodbury` wrapper itself.
# ──────────────────────────────────────────────────────────────

"""
    SOCScratch(n)

Per-cone buffers for the in-place second-order-cone NT scaling: the
`SymWoodbury` factors of `F` (`j`, `w`) and of `F⁻ᵀ` (`ij`, `iw`), the
normalized iterates, and the internal temporaries both wrappers need.
"""
struct SOCScratch
  n    :: Int
  j    :: Vector{Float64}   # Diagonal part of F
  w    :: Vector{Float64}   # rank-one factor of F
  zn   :: Vector{Float64}   # normalized z (scratch)
  sn   :: Vector{Float64}   # normalized s (scratch)
  scr  :: Vector{Float64}   # A\B scratch for the Woodbury Dp
  tN1  :: Vector{Float64}; tN2 :: Vector{Float64}
  tk1  :: Vector{Float64}; tk2 :: Vector{Float64}
  ij   :: Vector{Float64}   # Diagonal part of F⁻ᵀ
  iw   :: Vector{Float64}   # rank-one factor of F⁻ᵀ
  iscr :: Vector{Float64}
  itN1 :: Vector{Float64}; itN2 :: Vector{Float64}
  itk1 :: Vector{Float64}; itk2 :: Vector{Float64}
end

SOCScratch(n::Integer) = SOCScratch(Int(n),
  zeros(n), zeros(n), zeros(n), zeros(n), zeros(n),
  zeros(n), zeros(n), zeros(1), zeros(1),
  zeros(n), zeros(n), zeros(n),
  zeros(n), zeros(n), zeros(1), zeros(1))

# The 8-argument default constructor of SymWoodbury, which takes the
# precomputed Dp and the four temporaries rather than allocating them.
# Spelled out here so that the one place that bypasses the checked
# 3-argument constructor is easy to find.
@inline _symwoodbury(A, B, D, Dp, tN1, tN2, tk1, tk2) =
  SymWoodbury(A, B, D, Dp, tN1, tN2, tk1, tk2)

# Dp = safeinv(safeinv(D) .+ B'*(A\B)) for A = Diagonal(d), B a vector and
# D a scalar — what the 3-argument SymWoodbury constructor computes.
@inline function _woodbury_Dp(d::Vector{Float64}, B::Vector{Float64},
                              D::Float64, scr::Vector{Float64})
  @inbounds for i in eachindex(d)
    iszero(d[i]) && throw(SingularException(i))
    scr[i] = B[i] / d[i]
  end
  return inv(inv(D) + dot(B, scr))
end

"""
    nestod_soc!(sc::SOCScratch, z, s)

In-place [`nestod_soc`](@ref): the same scaling matrix, with its factors
written into `sc` instead of freshly allocated vectors.
"""
function nestod_soc!(sc::SOCScratch, z, s)

  n = length(z)
  qz = QFunit(z); qs = QFunit(s)
  (qz > 0 && qs > 0) || throw(LinearAlgebra.PosDefException(1))
  nz = norm(z); ns = norm(s)

  β = (qs/qz)^(1/4) * (sqrt(ns)/sqrt(nz))

  zb = sc.zn; sb = sc.sn
  rqz = sqrt(qz); rqs = sqrt(qs)
  @inbounds for i = 1:n
    zb[i] = (z[i] / nz) / rqz
    sb[i] = (s[i] / ns) / rqs
  end

  γ = sqrt((1 + dot(zb, sb))/2)

  # Jz = J*z
  scal!(n, -1., zb, 1)
  zb[1] = -zb[1]

  w = sc.w
  c = 1.0 / (2.0 * γ)
  @inbounds for i = 1:n
    w[i] = c * (sb[i] + zb[i])
  end
  w[1] = w[1] + 1
  scal!(n, (sqrt(2*β)/sqrt(2*w[1])), w, 1)

  j = sc.j
  j[1] = -β
  @inbounds for i = 2:n; j[i] = β; end

  J  = Diagonal(j)
  Dp = _woodbury_Dp(j, w, 1.0, sc.scr)
  return _symwoodbury(J, w, 1.0, Dp, sc.tN1, sc.tN2, sc.tk1, sc.tk2)

end

"""
    soc_inv_adjoint!(sc::SOCScratch, W)

In-place `adjoint(inv(W))` for the second-order-cone scaling block `W`
built by [`nestod_soc!`](@ref) (a `SymWoodbury` of real type is its own
adjoint, so this is `inv(W)` written into `sc`).
"""
function soc_inv_adjoint!(sc::SOCScratch, W::SOCBlock)

  # WoodburyMatrices.calc_inv: W′ = inv(A), X = W′B,
  # Z = safeinv(-safeinv(D) - dot(B, X)), result SymWoodbury(W′, X, Z).
  d = W.A.diag; B = W.B; D = W.D
  n = length(d)
  ij = sc.ij; iw = sc.iw
  @inbounds for i = 1:n
    iszero(d[i]) && throw(SingularException(i))
    ij[i] = inv(d[i])
  end
  @inbounds for i = 1:n
    iw[i] = ij[i] * B[i]      # Diagonal * Vector
  end
  Z  = inv(-inv(D) - dot(B, iw))
  Dp = _woodbury_Dp(ij, iw, Z, sc.iscr)
  return _symwoodbury(Diagonal(ij), iw, Z, Dp, sc.itN1, sc.itN2, sc.itk1, sc.itk2)

end

function nestod_sdc(z,s)

  # Nesterov-Todd Scaling Matrix for the Semidefinite Cone
  # Matrix which satisfies the property
  # F*z = inv(F')*s
  # (equivalently F'*(F*z) = s).  Note inv(F') and not inv(F): the two
  # agree only when mat(z) and mat(s) commute.

  # Homogeneity: F(αz, βs) = √(β/α)·F(z, s), and F acts as X ↦ RᵀXR, so
  # R scales as (β/α)^(1/4). Factor the unit-norm matrices and rescale,
  # so that jointly extreme magnitudes cannot overflow or underflow in
  # the Cholesky factors and their product. (‖vecm(X)‖ = ‖X‖_F.)
  isempty(z) && return VecCongurance(zeros(0, 0))   # order-0 block
  nz = norm(z); ns = norm(s)
  (nz > 0 && ns > 0) || throw(LinearAlgebra.PosDefException(1))
  Sm  = mat(s); Sm ./= ns
  Zm  = mat(z); Zm ./= nz
  Ls  = cholesky(Symmetric(Sm)).L
  Lz  = cholesky(Symmetric(Zm)).L
  F   = svd(Lz'*Ls)
  U   = F.U
  Λ   = F.S
  # R = inv(Lz)'*U*diagm(sqrt.(Λ)), formed as a triangular solve plus a
  # column scaling: inv(Lz)' = inv(Lz') so inv(Lz)'*U is Lz' \ U, and
  # right-multiplying by a diagonal scales the columns.
  R = (Lz' \ U) .* sqrt.(Λ)'
  R .*= sqrt(sqrt(ns) / sqrt(nz))
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
- `status::Symbol` -- `:Optimal`, `:Infeasible`, `:DualInfeasible`,
  `:AlmostInfeasible`, `:AlmostDualInfeasible`, `:Abandoned`, `:TimeLimit`,
  or `:Error`
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
| `:DualInfeasible` *with ray* | ray `ȳ` | all `NaN` | all `NaN` | `A*ȳ` | `NaN` | `true` |
| `:Infeasible`/`:DualInfeasible` *without ray* | all `NaN` | all `NaN` | all `NaN` | all `NaN` | `NaN` | `false` |
| `:Abandoned`, `:AlmostInfeasible`, `:AlmostDualInfeasible`, `:TimeLimit`, `:Error` | best iterate | best iterate | best iterate | best iterate | best iterate | `false` |

One exception: when a ray found on equilibrated or presolved data fails revalidation
against the original data (`sol.message` says so), the `:Almost*` or
`:Abandoned` solution holds that ray in the ray fields, not the best
iterate.

The infeasibility ray is normalized so that `dᵀw̄ - bᵀv̄ = -1` with
`Gᵀw̄ - Aᵀv̄ ≈ 0`; the unboundedness ray is normalized so that `cᵀȳ = +1`
with `Qȳ ≈ 0`, `Gȳ ≈ 0` and `Aȳ ∈ K`. See
[`validate_infeasibility_certificate`](@ref) and
[`validate_unboundedness_certificate`](@ref).

`kkt_solves` counts the KKT back-solves the main loop performed (initial
point, predictor, corrector, and refinement corrections); solves made by
the certificate-fallback auxiliary problems are not included.

Residual and diagnostic tail:

- `rEq::Real` -- relative equality residual `‖Gy − d‖ / (1 + max(‖d‖, ‖|G||y|‖))`
  of the returned point (`prFeas = max(rPr, rEq)`); `NaN` when no iterate
  was evaluated or when a certificate is returned (the iterate is discarded)
- `rGap::Real` -- relative duality gap `|vᵀs| / (1 + |pobj + objective_offset|)`,
  the quantity the termination test compares with `optTol`; `NaN` when
  unset or when a certificate is returned
- `kkt_repaired::Int` -- pivots the KKT solver dynamically regularized,
  summed over every factorization of the solve (0 unless the solver
  reports diagnostics; see `kkt_diagnostics`)
- `kkt_refactors::Int` -- refactorizations after a regularization bump
  (`kktsolver_ldl` with `retry_max > 0`), summed over the solve

Constructors with 12, 13, 14, or 15 positional arguments default the
trailing fields to `has_certificate = false`, `message = ""`,
`kkt_solves = 0`, `rEq = rGap = NaN`, `kkt_repaired = kkt_refactors = 0`.
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
  kkt_solves :: Int        # KKT back-solves performed by the main loop
  rEq    :: Real           # equality residual ‖Gy − d‖ (relative); NaN if unset
  rGap   :: Real           # relative gap |vᵀs| / (1 + |pobj + offset|); NaN if unset
  kkt_repaired  :: Int     # dynamically regularized pivots, summed over the solve
  kkt_refactors :: Int     # retry refactorizations (shift bumps), summed over the solve

end

# 12/13/14/15-argument constructors: no certificate / no message / no count /
# no residual-and-diagnostics tail. The tail defaults rEq = rGap = NaN and
# kkt_repaired = kkt_refactors = 0.
Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj) =
  Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, false, "", 0)
Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate) =
  Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate, "", 0)
Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate, message) =
  Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate, message, 0)
Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate, message, kkt_solves) =
  Solution(y, w, v, s, status, Iter, Mu, prFeas, duFeas, muFeas, pobj, dobj, has_certificate, message, kkt_solves,
           NaN, NaN, 0, 0)

# Overwrite sol with a *verified* infeasibility ray (dᵀw̄ - bᵀv̄ = -1).
# The primal iterate is discarded: it means nothing on an empty feasible set.
function claim_infeasible!(sol::Solution, w̄, v̄)
  fill!(sol.y, NaN); fill!(sol.s, NaN)
  sol.w[:] = w̄; sol.v[:] = v̄
  sol.pobj = NaN; sol.dobj = NaN
  sol.rEq  = NaN; sol.rGap = NaN     # they described the discarded iterate
  sol.status = :Infeasible
  sol.has_certificate = true
  return sol
end

# Overwrite sol with a *verified* recession ray (cᵀȳ = +1). The duals are
# discarded: they mean nothing when the dual is infeasible.
function claim_dual_infeasible!(sol::Solution, ȳ, A)
  sol.y[:] = ȳ; sol.s[:] = A*ȳ
  fill!(sol.w, NaN); fill!(sol.v, NaN)
  sol.pobj = NaN; sol.dobj = NaN
  sol.rEq  = NaN; sol.rGap = NaN     # they described the discarded iterate
  sol.status = :DualInfeasible
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
  refineRelTol = 1e-13,
  refineAbsTol = 1e-12,
  timeLimit = Inf,
  centralityCorrectors = 0,
  timing = nothing)

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

- `:Optimal` — `max(rDu, rPr, rCp, rEq, rGap) < optTol`, and every cone
  and equality row also passes the row-wise test
  `|rᵢ| / (1 + |bᵢ| + (|A||y|)ᵢ + |sᵢ|) < optTol`
  (`|rᵢ| / (1 + |dᵢ| + (|G||y|)ᵢ)` for equalities) in the original
  coordinates; the aggregate 2-norm test alone can accept a point that
  violates a small-scale row when the data span many orders of magnitude.
- `:Infeasible` / `:DualInfeasible` — a ray passed a screen *and* was accepted
  by the corresponding validator; `has_certificate` is then `true`.
- `:AlmostInfeasible` / `:AlmostDualInfeasible` — set only at loop exhaustion,
  when the best iterate carries a ray that validates at `100*infeasTol`
  but not at `infeasTol`. The best iterate is retained.
- `:Abandoned` — iteration limit reached with no verdict.
- `:TimeLimit` — `timeLimit` seconds elapsed; the best iterate is retained.
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
`:DualInfeasible` (`cⱼ ≠ 0`).

Selected keyword arguments:

- `infeasTol` — infeasibility-certificate tolerance, decoupled from `optTol`.
- `infeasAbsTol` — absolute tolerance for certificate validation.
- `staticReg` — static KKT regularization scale; `0` (default) disables it.
  `preprocess_conicIP` enables it when it detects rank deficiency.
- `certFallback` — enable fallback certificate solve on stall.
- `maxRefinementSteps`, `refineRelTol`, `refineAbsTol` — the predictor and
  corrector steps are refined against the 4×4 KKT system until
  `‖r − KΔz‖ ≤ refineAbsTol + refineRelTol·‖r‖` or the step budget is spent.
- `timeLimit` — wall-clock budget in seconds, checked once per iteration
  (a single factorization can overrun it). On expiry the status is
  `:TimeLimit` and the solution holds the best iterate so far; the
  certificate-fallback solves are skipped.
- `objective_offset` — a constant added to the objective for the purpose
  of the relative gap test only (`⟨v,s⟩/(1 + |pobj + objective_offset|)`).
  `preprocess_conicIP` passes the constant carried by fixed variables, so a
  reduced problem terminates by the same criterion as the full one.
- `centralityCorrectors` — number of Gondzio multiple centrality correctors
  tried per iteration (default `0`, off). Each corrector re-solves the
  current KKT factorization once for a direction that pushes the trial
  complementarity toward the box `[0.1·σμ, 10·σμ]` (in the Jordan frame of
  each cone) and is kept only if it lengthens the step; the loop stops at
  the first rejected corrector, and nothing is tried when the step is
  already full. Extra solves are counted in `kkt_solves`; the verbose
  `cc` column shows `accepted/tried`.
- `timing` — a [`PhaseTimes`](@ref) object to accumulate per-phase wall
  times, allocation bytes, and counts into (see `src/timing.jl` for the
  contract); `nothing` (default) leaves the solver uninstrumented.

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

`a`, `b` and `c` may be fresh vectors or views into the solver's own
workspace, valid only until the next call to `L` — `conicIP` copies them
into its direction buffers before calling again. [`kktsolver_ldl`](@ref)
returns views; the other built-in solvers return fresh vectors.

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

`equilibrate = true` (default) applies Ruiz equilibration to the data
before solving and maps the solution, rays, and residuals back to the
original coordinates; see [`equilibrate_conicIP`](@ref). A custom
`kktsolver` then receives the scaled data.
"""
function conicIP(Q, c::AbstractVector, A, b::AbstractVector, cone_dims,
                 G = spzeros(0, length(c)), d = zeros(0);
                 equilibrate = true, timeLimit = Inf, timing = nothing, kwargs...)
  t_start = time()
  # t_gc: every entry point (this one, preprocess_conicIP, _preprocess_core,
  # the MOI optimize!) assigns it on exit; the outermost writes last and
  # wins (see gc_start/gc_stop! in timing.jl). `_conicIP` never touches it.
  gc0 = gc_start(timing)
  if !equilibrate
    sol = _conicIP(Q, c, A, b, cone_dims, G, d; timeLimit = timeLimit, timing = timing, kwargs...)
    return gc_stop!(timing, gc0, sol)
  end
  eq = @phase timing t_equilibrate b_equilibrate equilibrate_conicIP(Q, c, A, b, cone_dims, G, d)
  # The core tests termination on residuals mapped back to the original
  # coordinates, so optTol keeps its meaning under any scaling.
  scaling = @phase timing t_equilibrate b_equilibrate (
    Dc = eq.Dc, Dr = eq.Dr, De = eq.De, σ = eq.σ,
    normc = norm(c), normb = normsafe(b), normd = normsafe(d))
  sol = _conicIP(eq.Q, eq.c, eq.A, eq.b, cone_dims, eq.G, eq.d;
                 scaling = scaling, timeLimit = timeLimit - (time() - t_start),
                 timing = timing, kwargs...)
  sol = @phase timing t_postsolve begin
    unequilibrate!(sol, eq, Q, c, A, b, cone_dims, G, d;
                   objective_offset = get(kwargs, :objective_offset, 0.0))
    # A ray validated on the scaled data need not validate on the original
    # data (tolerances are not scaling-invariant): re-run the validator in
    # the caller's coordinates with the caller's tolerances, and downgrade
    # the claim if it fails there.
    _revalidate_certificate!(sol, Q, c, A, b, cone_dims, G, d;
                             infeasTol = Float64(get(kwargs, :infeasTol, 1e-7)),
                             infeasAbsTol = Float64(get(kwargs, :infeasAbsTol, 1e-9)))
  end
  return gc_stop!(timing, gc0, sol)
end

function _conicIP(

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
  refineRelTol = 1e-13,    # refinement stops when ‖r − KΔz‖ ≤ refineAbsTol + refineRelTol‖r‖
  refineAbsTol = 1e-12,
  timeLimit = Inf,         # wall-clock budget in seconds, checked once per iteration
  objective_offset = 0.0,  # constant part of the objective (from presolve), used
                           # only in the relative gap test's denominator
  scaling = nothing,       # set by conicIP when the data are equilibrated: the
                           # termination residuals are evaluated in the
                           # original coordinates (see equilibrate.jl)
  centralityCorrectors::Integer = 0,  # Gondzio correctors per iteration (0 = off;
                           # see the corrector block after the line search)
  timing = nothing         # PhaseTimes to accumulate into (timing.jl); bound
                           # once here and only ever read, so the closures
                           # below capture it by value
  )

  centralityCorrectors >= 0 ||
    throw(ArgumentError("centralityCorrectors must be a nonnegative integer (got $centralityCorrectors)"))
  centralityCorrectors = Int(centralityCorrectors)

  t_start = time()
  # t_setup runs from here to the KKT solver construction. It is opened
  # by hand rather than with @phase because the scratch below is captured
  # by closures, and a @phase block would give each of those variables
  # two assignment sites (one per branch) and so box them.
  (t_setup0, b_setup0) = @phase_start timing
  # Closes t_setup; every exit between here and the end of solver
  # construction (structural certificates, deflation, setup failure) goes
  # through it, as does the normal path.
  function exit_setup(x)
    @phase_stop timing t_setup b_setup t_setup0 b_setup0
    return x
  end
  over_time() = time() - t_start > timeLimit
  time_left() = timeLimit - (time() - t_start)

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
  # Scratch for the step-residual evaluation only: _prod_buf1 is aliased by
  # rleft.s / r0.s for the whole iteration and must not be overwritten.
  _res_buf1  = zeros(m)
  _res_buf2  = zeros(m)
  # Block products F*Δv and F⁻ᵀ*Δs inside step_residual!; kept apart from
  # _res_buf1/_res_buf2 because the cone product that consumes them must
  # not alias its own output.
  _res_buf3  = zeros(m)
  _res_buf4  = zeros(m)
  # solve4x4! scratch: the v-block right-hand side handed to solve3x3, and
  # the two Block products of the Δs recovery. All three are live only for
  # the duration of one solve.
  _s3_rhs    = zeros(m)
  _dir_buf1  = zeros(m)
  _dir_buf2  = zeros(m)
  # Corrector right-hand side: the two Block products of d_aff, and the
  # s-block itself (its y/w/v blocks alias r0's).
  _rhs_buf1  = zeros(m)
  _rhs_buf2  = zeros(m)
  _rhs_s     = zeros(m)
  # λ = F*z.v, recomputed at the top of every iteration
  _λ         = zeros(m)
  # Trial iterate for the interiority check of the line search
  _trial_v   = zeros(m)
  _trial_s   = zeros(m)
  _rkkt = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  _rIr  = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  # Best step seen so far during refinement (restored when a correction
  # increases the residual)
  _Δz_keep = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  # The directions the loop owns. `solve4x4!` writes into the buffer it is
  # given, so each direction that has to outlive another needs its own:
  # the predictor d_aff is still read while the corrector rhs is formed,
  # the corrector Δz is read by the line search and by every refinement
  # residual, and the refinement correction Δzr is consumed immediately.
  _d_aff = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  _Δz    = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  _Δzr   = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  # Centrality-corrector scratch (allocated only when the option is on):
  # trial scaled iterates ṽ, s̃, their product w, the correction Δw, the
  # corrector right-hand side (zero except the s block), and the candidate
  # direction Δz + Δz_c.
  if centralityCorrectors > 0
    _cc_v  = zeros(m); _cc_s = zeros(m); _cc_w = zeros(m); _cc_dw = zeros(m)
    _cc_b1 = zeros(m); _cc_b2 = zeros(m)   # FΔv and F⁻ᵀΔs
    _cc_r  = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
    _cc_Δz = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
    _Δz_c  = v4x1(zeros(n), zeros(p), zeros(m), zeros(m))
  end

  # KKT back-solve counter, reported as Solution.kkt_solves
  _nsolve = Ref(0)
  # Level-3 solve object of the current factorization, read through the
  # kkt_diagnostics hook (kktsolvers.jl) for the verbose kkt column and
  # the Solution counters.
  _s3_cur = Ref{Any}(nothing)
  # Stamp the solve count and, when the KKT solver reports them, the
  # regularization diagnostics onto a Solution. Bumps happen inside
  # solve3x3 calls, after any per-iteration snapshot, so every site that
  # records the count records the diagnostics too.
  function _stamp_kkt!(sol)
    sol.kkt_solves = _nsolve[]
    dg = kkt_diagnostics(_s3_cur[])
    if dg !== nothing
      sol.kkt_repaired  = dg.repaired_total
      sol.kkt_refactors = dg.refactors_total
    end
    return sol
  end

  # Pre-allocated Blocks for the NT scaling F and for inv(F)' — the shells
  # and the per-cone buffers behind them are reused every iteration.
  # "R" blocks are installed once and only their `.diag` is overwritten;
  # "Q" blocks get a fresh (immutable) SymWoodbury wrapper per iteration
  # around the buffers in `_soc_scr`; "S" blocks still allocate.
  F_cache   = Block(size(block_sizes, 1))
  F⁻ᵀ_cache = Block(size(block_sizes, 1))
  _soc_scr  = Vector{Union{Nothing,SOCScratch}}(nothing, size(block_sizes, 1))
  for (btype, I, i) = block_data
    if btype == "R"
      F_cache.Blocks[i]   = Diagonal(zeros(length(I)))
      F⁻ᵀ_cache.Blocks[i] = Diagonal(zeros(length(I)))
    elseif btype == "Q"
      _soc_scr[i] = SOCScratch(length(I))
    end
  end

  normc = norm(c)
  normd = isempty(d) ? -Inf : norm(d)
  normb = normsafe(b)
  normdsafe = normsafe(d)   # 0 for empty d (normd is -Inf there)
  # Entrywise absolute values of the data, for the residual normalizations
  # ‖|Q||y|‖, ‖|Gᵀ||w|‖, ‖|Aᵀ||v|‖, ‖|A||y|‖, ‖|G||y|‖ (transposes are
  # lazy wrappers; the products below never materialize them).
  absQ = _absmat(Q); absA = _absmat(A); absG = _absmat(G)
  absAᵀ = absA'; absGᵀ = absG'
  # Scratch for those products: |y|, |w|, |v| and the outputs
  _absy = zeros(n); _absw = zeros(p); _absv = zeros(m)
  _nrm_n = zeros(n); _nrm_m = zeros(m); _nrm_p = zeros(p)

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
      return exit_setup(claim_infeasible!(sol0, w̄, v̄))
    end
    return exit_setup(Solution(fill(NaN,n), fill(NaN,p), fill(NaN,m), fill(NaN,m),
                    :Error, 0, NaN, Inf, Inf, Inf, NaN, NaN, false,
                    "zero equality row certificate could not be normalized and validated", 0))
  end

  j0 = findfirst(j -> Zc[j] && c[j] != 0, 1:n)
  if j0 !== nothing
    ȳ0 = zeros(n); ȳ0[j0] = sign(c[j0])
    (chk, ȳ) = validate_unboundedness_certificate(
        Q, c, A, b, cone_dims, G, d, ȳ0;
        abstol = infeasAbsTol, reltol = infeasTol)
    if chk.valid
      if verbose
        print("\n > EXIT -- Structurally dual infeasible (zero column $(j0) of [Q; A; G], c[$(j0)] ≠ 0)\n\n")
      end
      sol0 = Solution(zeros(n), fill(NaN,p), fill(NaN,m), zeros(m),
                      :None, 0, 0, Inf, Inf, Inf, NaN, NaN)
      return exit_setup(claim_dual_infeasible!(sol0, ȳ, A))
    end
    return exit_setup(Solution(fill(NaN,n), fill(NaN,p), fill(NaN,m), fill(NaN,m),
                    :Error, 0, NaN, Inf, Inf, Inf, NaN, NaN, false,
                    "zero column certificate could not be normalized and validated", 0))
  end

  if any(Gzr) || any(Zc)
    keep_r = findall(.!Gzr)
    keep_c = findall(.!Zc)
    # The recursive call accumulates into the same object; this call's own
    # work so far is setup.
    exit_setup(nothing)
    solr = _conicIP(Q[keep_c, keep_c], c[keep_c], A[:, keep_c], b, cone_dims,
                   G[keep_r, keep_c], d[keep_r];
                   kktsolver = kktsolver, optTol = optTol, DTB = DTB,
                   verbose = verbose, maxRefinementSteps = maxRefinementSteps,
                   maxIters = maxIters, cache_nestodd = cache_nestodd,
                   infeasTol = infeasTol, infeasAbsTol = infeasAbsTol,
                   staticReg = staticReg, certFallback = certFallback,
                   certFallbackIters = certFallbackIters,
                   refineRelTol = refineRelTol, refineAbsTol = refineAbsTol,
                   timeLimit = time_left(),
                   objective_offset = objective_offset,
                   centralityCorrectors = centralityCorrectors,
                   timing = timing,
                   scaling = scaling === nothing ? nothing :
                             (; scaling..., Dc = scaling.Dc[keep_c],
                                            De = scaling.De[keep_r]))
    # Re-expand, inserting zeros at deflated positions when the block is
    # meaningful for the returned status (see the Solution field table);
    # blocks the convention leaves NaN stay all-NaN.
    best_iter = solr.status in (:Optimal, :Abandoned, :AlmostInfeasible,
                                :AlmostDualInfeasible, :TimeLimit, :Error, :None)
    w_ok = best_iter || (solr.status == :Infeasible && solr.has_certificate)
    y_ok = best_iter || (solr.status == :DualInfeasible && solr.has_certificate)
    w = fill(NaN, p); y = fill(NaN, n)
    if w_ok; fill!(w, 0.0); w[keep_r] = solr.w; end
    if y_ok; fill!(y, 0.0); y[keep_c] = solr.y; end
    return Solution(y, w, solr.v, solr.s, solr.status, solr.Iter, solr.Mu,
                    solr.prFeas, solr.duFeas, solr.muFeas, solr.pobj,
                    solr.dobj, solr.has_certificate, solr.message,
                    solr.kkt_solves, solr.rEq, solr.rGap,
                    solr.kkt_repaired, solr.kkt_refactors)
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

  # Strict interiority of x with respect to the cone product, tested with
  # the same quantities the NT scaling will compute (QF for SOC blocks,
  # a Cholesky for SDP blocks), so that an accepted step cannot fail the
  # next iteration's scaling.
  function interior(x)
    @inbounds for (btype, I, i) = block_data
      xI = view(x, I)
      if btype == "R"
        all(>(0.0), xI) || return false
      elseif btype == "Q"
        (xI[1] > 0 && QFunit(xI) > 0) || return false
      elseif btype == "S"
        isposdef(Symmetric(mat(xI))) || return false
      end
    end
    return true
  end

  function nt_scaling!(B::Block, x, y)

    # Compute Nesterov-Todd scaling matrix, F s.t.
    # λ = F*x = inv(F')*y
    # For the self-adjoint R and Q blocks this is λ = F*x = F\y; for
    # an S block F is a congruence and only the adjoint form holds
    # (see nestod_sdc).
    #
    # In place: the "R" block installed at setup keeps its Diagonal and only
    # its `.diag` is rewritten, and the "Q" block is rebuilt around the
    # buffers of its SOCScratch. Only "S" blocks allocate.

    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I);
      # √y/√x rather than √(y/x): the quotient overflows for jointly
      # extreme magnitudes (y ~ 1e160, x ~ 1e-160), the square roots do not.
      if btype == "R"
        # NB: not named `d` — a plain assignment in this nested function
        # would rebind _conicIP's equality right-hand side.
        Blk = B.Blocks[i]
        dR = (Blk isa DiagBlock) ? Blk.diag : zeros(length(I))
        for t in eachindex(dR)
          dR[t] = sqrt(yI[t]) / sqrt(xI[t])
        end
        Blk isa DiagBlock || (B.Blocks[i] = Diagonal(dR))
      end
      if btype == "Q"
        sc = _soc_scr[i]
        B.Blocks[i] = sc === nothing ? nestod_soc(xI, yI) : nestod_soc!(sc, xI, yI)
      end
      if btype == "S"; B[i] = nestod_sdc(xI, yI); end
    end

    return B;

  end

  # adjoint(inv(F)) into F⁻ᵀ_cache, reusing the per-cone buffers for the
  # "R" and "Q" blocks (blockmatrices.jl's inv_adjoint! handles "R"; the
  # SOC blocks go through soc_inv_adjoint!, which keeps `inv`'s operations
  # but writes into the SOCScratch).
  function nt_inv_adjoint!(dest::Block, src::Block)
    @inbounds for (btype, I, i) = block_data
      Blk = src.Blocks[i]
      sc  = btype == "Q" ? _soc_scr[i] : nothing
      if sc !== nothing && Blk isa SOCBlock
        dest.Blocks[i] = soc_inv_adjoint!(sc, Blk)
      else
        inv_adjoint_block!(dest, src, i)
      end
    end
    return dest
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
  errsol(msg) = _stamp_kkt!(Solution(fill(NaN,n), fill(NaN,p), fill(NaN,m), fill(NaN,m),
                                     :Error, 0, 0, Inf, Inf, Inf, NaN, NaN, false, msg,
                                     _nsolve[]))

  if verbose && kktsolver === default_kktsolver
    chosen = choose_kktsolver(Qᵣ, A, G, cone_dims)
    nnz_pc = (_structural_nnz(Q) + _structural_nnz(A) + _structural_nnz(G)) / max(n, 1)
    @printf(" > KKT solver: %s (auto, %.1f nnz/col)\n", nameof(chosen), nnz_pc)
  end

  solve3x3gen = try
    kktsolver(Qᵣ,A,G,cone_dims)
  catch err
    err isa KKT_FAILURES || rethrow()
    return exit_setup(errsol(kkt_error("solver setup", err)))
  end
  exit_setup(nothing)
  # Attach before the first factorization so the backend counts it too.
  timing === nothing || kkt_attach_timing!(solve3x3gen, timing)

  function solve4x4gen(λ, F, F⁻ᵀ, solve3x3gen = solve3x3gen)

    #
    # solve4x4gen(λ, F)(out, r) solves the 4x4 KKT System into `out`
    # ┌                  ┐ ┌    ┐   ┌     ┐
    # │ Q   G'  -A'      │ │ Δy │ = │ r.y │
    # │ G                │ │ Δw │   │ r.w │ S = block(λ)*F
    # │ A             -I │ │ Δv │   │ r.v │ V = block(λ)*F⁻ᵀ
    # │          S     V │ │ Δs │   │ r.s │
    # └                  ┘ └    ┘   └     ┘
    # F = Nesterov-Todd scaling matrix
    #

    solve3x3 = solve3x3gen(F, F⁻ᵀ)
    _s3_cur[] = solve3x3
    # The wall time of this factorization is charged by the caller
    # (t_init for the initial point, t_kktupdate in the loop).
    if timing !== nothing
      timing.n_kktupdate += 1
      kkt_attach_timing!(solve3x3, timing)
    end

    # The direction is written into the caller's `out`; nothing on this
    # path allocates (bar an SDP scaling block's cone products). `out.s`
    # doubles as the t1 scratch, so `out` must not alias `r` — every call
    # site pairs a distinct preallocated direction with its right-hand
    # side. The three vectors a backend's solve3x3 returns may be views
    # into its own workspace, so they are copied out before that workspace
    # is touched again.
    function solve4x4!(out::v4x1, r)

      _nsolve[] += 1
      timing === nothing || (timing.n_solve += 1)
      cone_div!(_div_buf, r.s, λ)
      mul_adjoint!(out.s, F, _div_buf)    # t1 = F'*(r.s ○\ λ)
      _s3_rhs .= r.v .+ out.s
      (Δy, Δw, Δv)  = solve3x3(r.y, r.w, _s3_rhs)
      copyto!(out.y, Δy); copyto!(out.w, Δw); copyto!(out.v, Δv)
      mul!(_dir_buf1, F, out.v)
      mul_adjoint!(_dir_buf2, F, _dir_buf1)
      axpy!(-1, _dir_buf2, out.s)         # > Δs = t1 - F'*(F*Δv)
      return out

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
  # t_init: the identity-scaling factorization and solve, and the shift
  # into the interior (neither counts toward t_kktupdate / t_direction).
  t_init0 = timing === nothing ? UInt64(0) : time_ns()
  function exit_init(x)
    @phase_stop timing t_init t_init0
    return x
  end
  z  = try
    solve4x4gen(e,I,I)(v4x1(zeros(n), zeros(p), zeros(m), zeros(m)), r_init)
  catch err
    err isa KKT_FAILURES || rethrow()
    return exit_init(errsol(kkt_error("initial point", err)))
  end

  # A nonfinite initial point would reach LAPACK through maxstep below.
  isfinite4(z) || return exit_init(errsol("non-finite initial point (initial point)"))

  (α_v, α_s) = try
    (maxstep(z.v, nothing), maxstep(z.s, nothing))
  catch err
    err isa KKT_FAILURES || rethrow()
    return exit_init(errsol(kkt_error("initial point", err)))
  end

  # Change to +
  z.v = z.v - α_v*e
  z.s = z.s - α_s*e
  exit_init(nothing)

  if verbose
      println("            Optimality                      Objective              Infeasibility       ")
      println()
      printstyled(@sprintf(" %-6s  │  %-8s  %-8s  %-8s │  %-8s  %-8s  │  %-8s  %-8s │  %-6s  %-8s  %-5s\n",
                  "  Iter","prFeas","duFeas","muFeas","pobj","dobj","icertp","icertd","refine","kkt","cc");
                  bold = true)
  end

  # ────────────────────────────────────────────────────────────
  #  Iterate Loop
  # ────────────────────────────────────────────────────────────

  sol     = Solution(copy(z.y), copy(z.w), copy(z.v), copy(z.s), :None, 0, 0, Inf, Inf, Inf, Inf, -Inf)
  _stamp_kkt!(sol)
  optBest = Inf
  nref    = 0      # refinement corrections applied in the previous iteration
  cc_acc  = 0      # centrality correctors accepted / tried in the previous
  cc_try  = 0      #   iteration (verbose cc column)
  rnorm   = 0
  nstall  = 0      # consecutive iterations with a negligible step
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
      _stamp_kkt!(sol)
      return nothing
    end
  end

  # Same verdict for a nonfinite direction or iterate, which has no
  # exception to report but must not be handed to LAPACK.
  function nonfinite!(what, stage)
    sol.status = :Error
    sol.message = "non-finite $what ($stage)"
    _stamp_kkt!(sol)
    if verbose; print("\n > EXIT -- Error! ($(sol.message))\n\n"); end
    return sol
  end

  # t_loop is the wall time of the whole iterate loop. The loop is left
  # through `return` at many sites, so the phase is opened here and every
  # `return` inside the loop goes through exit_loop, which closes it and
  # passes the value on; the break/exhaustion path closes it after the
  # loop. The loop children (t_scaling … t_linesearch) are @phase spans
  # inside the body, laid out so that no closure-captured loop variable
  # (F, λ, r0, r, solve, d_aff, Δz) is assigned inside a span other than
  # as `x = @phase … expr`, which keeps its single assignment site.
  t_loop0 = timing === nothing ? UInt64(0) : time_ns()
  function exit_loop(x)
    timing === nothing || (timing.t_loop += time_ns() - t_loop0)
    return x
  end
  # Exit from inside the termination span: close t_residuals (stamps are
  # passed in, so nothing per-pass is captured), then the loop.
  function exit_res(x, t0, b0)
    @phase_stop timing t_residuals b_residuals t0 b0
    return exit_loop(x)
  end

  for Iter = 1:maxIters

    timing === nothing || (timing.n_passes += 1)

    # Every solve of the previous iteration is accounted for here; the
    # termination returns below happen before this iteration's first solve.
    _stamp_kkt!(sol)

    if over_time()
      if verbose; print("\n > EXIT -- Time limit reached ($(timeLimit) s)\n\n"); end
      sol.status = :TimeLimit
      return exit_loop(sol)
    end

    # Nesterov-Todd scaling matrix. nestod_sdc factors both cone iterates,
    # so a boundary iterate surfaces here as a PosDefException.
    Fok = @phase timing t_scaling b_scaling guarded("NT scaling, iteration $Iter") do
      nt_scaling!(F_cache, z.v, z.s)
      nt_inv_adjoint!(F⁻ᵀ_cache, F_cache)
      mul!(_λ, F_cache, z.v)       # λ = F*z.v is also F⁻ᵀ*z.s
      true
    end
    Fok === nothing && return exit_loop(sol)
    F      = F_cache
    λ      = _λ
    F⁻ᵀ    = F⁻ᵀ_cache
    # A scaling that is non-finite without having thrown would reach
    # inv_adjoint! and the KKT solve as Inf/NaN.
    all(isfinite, λ) ||
      return exit_loop(nonfinite!("NT scaling", "NT scaling, iteration $Iter"))

    # The KKT factorization is deferred until after the termination and
    # certificate checks below: a converged iterate never pays for it, and
    # a factorization that would fail on it cannot mask the convergence.

    # Products of the iterate with the data, each formed once per
    # iteration and shared by the residuals, the objective, and the
    # infeasibility screens.
    rleft = @phase timing t_residuals b_residuals begin
      Qy   = Q*z.y
      Gᵀw_Aᵀv = Gᵀ*z.w - Aᵀ*z.v

      #         ┌                   ┐ ┌     ┐
      # rleft = │ Q   G'   -A'      │ │ z.y │
      #         │ G                 │ │ z.w │  V = block(λ)*F⁻ᵀ
      #         │ A              -I │ │ z.v │    = block(λ)*λ
      #         │           S     V │ │ z.s │
      #         └                   ┘ └     ┘
      cone_prod!(_prod_buf1, λ, λ)
      v4x1( Qy + Gᵀw_Aᵀv ,
            G*z.y        ,
            A*z.y - z.s  ,
            _prod_buf1   )
    end

    # True Residual of nonlinear KKT System
    r0 = @phase timing t_residuals b_residuals v4x1(rleft.y - c, rleft.w - d, rleft.v - b, rleft.s)

    # Residual norms, objectives, best-iterate bookkeeping, certificate
    # screens and the termination verdicts, as one span so that the
    # termination returns stay inside it.
    (t_res0, b_res0) = @phase_start timing

    # Gap
    μbar = dot(z.v,z.s)
    μ    = μbar/conedim
    push!(μ_history, μ)

    # ────────────────────────────────────────────────────────────
    #  Print iterate status, save best iterate
    # ────────────────────────────────────────────────────────────

    cᵀy  = dot(c,z.y)
    yᵀQy = dot(z.y, Qy)
    pobj = 0.5*yᵀQy - cᵀy
    # Stationary quadratic dual objective. Away from stationarity this
    # is an estimate, not a certified bound; complementarity is tested
    # separately and must not be substituted for the objective difference.
    dobj = -0.5*yᵀQy - dot(d, z.w) + dot(b, z.v)

    # Convexity guard. The method assumes Q ⪰ 0 (equilibration preserves
    # this); an iterate with yᵀQy < 0 beyond rounding is a witness that it
    # is not, and the iteration would otherwise converge to a stationary
    # point that is not a minimizer and report it as :Optimal.
    if yᵀQy < -1e-10 * (1 + norm(Qy) * norm(z.y))
      sol.status  = :Error
      sol.message = "objective Hessian is not positive semidefinite " *
                    "(yᵀQy < 0 at iteration $Iter); ConicIP requires a convex objective"
      _stamp_kkt!(sol)
      if verbose; print("\n > EXIT -- Error! ($(sol.message))\n\n"); end
      return exit_res(sol, t_res0, b_res0)
    end

    # rGap is the relative duality gap measured as the complementarity
    # ⟨v,s⟩ (equal to pobj − dobj at a feasible point). It is not taken
    # from the dobj formula above, whose residual products wᵀr_w + vᵀr_v
    # put a floor of ‖w‖‖r_w‖ under the computed gap once the duals are
    # large; feasibility has its own tests. rCp measures the same
    # complementarity as a 2-norm of the Jordan product, √(cones) smaller
    # for equal components, so the gap test is what keeps the enforced
    # accuracy independent of problem size.
    # Feasibility residuals are relative to the size of the equation they
    # measure — the right-hand side or the terms actually summed to form
    # the residual, whichever is larger (a backward-error normalization):
    # ‖c‖, ‖|Q||y|‖, ‖|Gᵀ||w|‖, ‖|Aᵀ||v|‖ for stationarity; ‖b‖, ‖|A||y|‖,
    # ‖s‖ for the cone rows; ‖d‖, ‖|G||y|‖ for the equalities. The
    # componentwise products matter: the cruder bound ‖A‖‖y‖ (largest
    # entry times the iterate) is far larger than any row of |A||y| when
    # y has a huge component in a column that row does not touch, and it
    # then normalizes a genuine residual away — an infeasible pair
    # y₁ ≥ 1, −y₁ ≥ 0 was reported optimal beside a legitimate y₂ ≈ 10¹⁶.
    # Normalizing by the right-hand side alone is wrong the other way: a
    # homogeneous row (d = 0) becomes an absolute test, which a G of size
    # 10⁸ can never meet, since rounding alone leaves ‖Gy‖ ≈ ‖|G||y|‖ε.
    _absy .= abs.(z.y); _absw .= abs.(z.w); _absv .= abs.(z.v)
    if scaling === nothing
      nQy = _absprod_norm!(_nrm_n, absQ,  _absy)
      nGw = _absprod_norm!(_nrm_n, absGᵀ, _absw)
      nAv = _absprod_norm!(_nrm_n, absAᵀ, _absv)
      nAy = _absprod_norm!(_nrm_m, absA,  _absy)
      nGy = _absprod_norm!(_nrm_p, absG,  _absy)
      rDu = norm(r0.y)/(1 + max(normc, nQy, nGw, nAv))
      rPr = normsafe(r0.v)/(1 + max(normb, nAy, normsafe(z.s)))
      rCp = normsafe(r0.s)/(1+abs(cᵀy));
      rEq = normsafe(r0.w)/(1 + max(normdsafe, nGy))   # Gy - d
      rGap = abs(μbar)/(1 + abs(pobj + objective_offset))
    else
      # Residuals of the equilibrated iterate in the original coordinates:
      # r_y = r̃_y/(σDc), r_v = r̃_v/Dr, r_w = r̃_w/De, y = Dc ỹ, w = De w̃/σ,
      # v = Dr ṽ/σ, s = s̃/Dr, and every complementarity quantity (λ∘λ,
      # ⟨v,s⟩, the objectives) is 1/σ times its scaled value. The
      # normalizing products map the same way as the residual they scale,
      # since |Q̃| = σDc|Q|Dc, |Ã| = Dr|A|Dc, |G̃| = De|G|Dc entrywise:
      #   |Q||y| = |Q̃||ỹ|/(σDc),  |Gᵀ||w| = |G̃ᵀ||w̃|/(σDc),
      #   |Aᵀ||v| = |Ãᵀ||ṽ|/(σDc),  |A||y| = |Ã||ỹ|/Dr,  |G||y| = |G̃||ỹ|/De.
      σs = scaling.σ; Dc = scaling.Dc; Dr = scaling.Dr; De = scaling.De
      nQy = _absprod_norm!(_nrm_n, absQ,  _absy, Dc, σs)
      nGw = _absprod_norm!(_nrm_n, absGᵀ, _absw, Dc, σs)
      nAv = _absprod_norm!(_nrm_n, absAᵀ, _absv, Dc, σs)
      nAy = _absprod_norm!(_nrm_m, absA,  _absy, Dr)
      nGy = _absprod_norm!(_nrm_p, absG,  _absy, De)
      rDu = norm(r0.y ./ Dc)/σs / (1 + max(scaling.normc, nQy, nGw, nAv))
      rPr = normsafe(r0.v ./ Dr) /
            (1 + max(scaling.normb, nAy, normsafe(z.s ./ Dr)))
      rCp = normsafe(r0.s)/σs/(1+abs(cᵀy)/σs)
      rEq = normsafe(r0.w ./ De) / (1 + max(scaling.normd, nGy))
      rGap = abs(μbar)/(σs + abs(pobj + σs*objective_offset))
    end

    # The retained "best" iterate is judged on feasibility and
    # complementarity only: on an infeasible or unbounded problem the gap
    # diverges while the iterate sharpens into a certificate, and the
    # post-loop screens and fallback need that late iterate, not an early
    # one with a small gap.
    bestMeasure = max(rDu, rPr, rCp, rEq)
    optMeasure  = max(bestMeasure, rGap)
    # Row-wise feasibility, in the original coordinates, as an additional
    # requirement for :Optimal. The aggregate tests above normalize by the
    # 2-norm of the whole block, so when the data span many orders of
    # magnitude a small-scale row can be violated by its own scale and
    # still vanish in the aggregate: x ≥ 1 (row weight 1e-4) together with
    # x ≤ 0.5 (weight 1e4) was reported optimal at x ≈ 0.5, unequilibrated,
    # with prFeas = 4e-8 and a row-wise residual of 1.5e-4. Every row is
    # therefore also held to |r_i| / (1 + |b_i| + (|A||y|)_i + |s_i|) < optTol
    # (cone rows) and |r_i| / (1 + |d_i| + (|G||y|)_i) < optTol
    # (equalities). Measured on the harness and the scaling tests, the
    # iterate that passes the aggregate test passes these too, so the
    # iteration counts do not change; the stationarity rows are not tested
    # this way (a free variable's row with a tiny cost coefficient becomes
    # an absolute test, and the κ-scaling instances then never terminate).
    # _nrm_m and _nrm_p still hold |A||y| and |G||y| in original coordinates.
    if scaling === nothing
      rPrRow = _rowwise_max(r0.v, b, _nrm_m, z.s)
      rEqRow = _rowwise_max(r0.w, d, _nrm_p, nothing)
    else
      rPrRow = _rowwise_max(r0.v, b, _nrm_m, z.s, scaling.Dr)
      rEqRow = _rowwise_max(r0.w, d, _nrm_p, nothing, scaling.De)
    end
    optimal     = optMeasure < optTol && max(rPrRow, rEqRow) < optTol
    # An iterate that passes the full test is stored unconditionally: the
    # gap is not part of bestMeasure, so an earlier iterate with a smaller
    # bestMeasure but a larger gap could otherwise be the one returned as
    # :Optimal (it is sol, not z, that the exit below hands back).
    if optimal || bestMeasure < optBest
      sol.y[:] = z.y; sol.w[:] = z.w; sol.v[:] = z.v; sol.s[:] = z.s
      sol.Iter = Iter; sol.Mu = μ;
      sol.duFeas = rDu; sol.prFeas = max(rPr, rEq); sol.muFeas = rCp
      sol.rEq = rEq; sol.rGap = rGap
      sol.pobj = pobj; sol.dobj = dobj
      optBest = bestMeasure
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

      p_infeas_unscaled = norm(Gᵀw_Aᵀv)
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
      d_infeas1 = isempty(A) ? -Inf : norm(rleft.v)     # ‖Ay − s‖
      d_infeas2 = isempty(G) ? -Inf : norm(rleft.w)     # ‖Gy‖
      d_infeas3 = all(isfinite, z.y) ? norm(Qy) : NaN

      d_infeas_cvx  = cᵀy > 0 ? max(d_infeas1/max(1,normb), d_infeas2/max(1,normd), d_infeas3/max(1,normc))/abs(cᵀy) : NaN
      d_infeas_ecos = cᵀy > 0 ? max(d_infeas1, d_infeas2, d_infeas3)/norm(z.y) : NaN
      d_infeas = abs(max(d_infeas_cvx, d_infeas_ecos))

      if claim == :None && d_infeas < infeasTol
        (dchk, ȳ) = validate_unboundedness_certificate(
                       Q, c, A, b, cone_dims, G, d, z.y;
                       abstol = infeasAbsTol, reltol = infeasTol)
        if dchk.valid; claim = :DualInfeasible; end
      end

    end

    if verbose
      # A row is highlighted in red when the KKT step is still inaccurate
      # after iterative refinement (see REFINE_WARN_NORM). The kkt cell is
      # repaired/refactors of the factorization that produced this
      # iterate (the previous iteration's, or the initial point's); blank
      # when the solver reports no diagnostics.
      # The cc cell is accepted/tried centrality correctors of the step
      # that produced this iterate; blank when the option is off.
      dg  = kkt_diagnostics(_s3_cur[])
      kkt = dg === nothing ? "" : @sprintf("%d/%d", dg.repaired, dg.refactors)
      cc  = centralityCorrectors > 0 ? @sprintf("%d/%d", cc_acc, cc_try) : ""
      row = @sprintf(" %6i  │  %-8.1e  %-8.1e  %-8.1e │  % -8.1e  % -8.1e  │  %-8.1e  %-8.1e │  %-6i  %-8s  %s\n",
                     Iter, rPr, rDu, rCp, pobj, dobj, p_infeas, d_infeas, nref, kkt, cc)
      if rnorm > REFINE_WARN_NORM
        printstyled(row; bold = true, color = :red)
      else
        print(row)
      end
    end

    if optimal
      if verbose; print("\n > EXIT -- Below Tolerance!\n\n"); end
      sol.status = :Optimal
      return exit_res(sol, t_res0, b_res0)
    end

    if claim == :Infeasible
      if verbose; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
      return exit_res(claim_infeasible!(sol, w̄, v̄), t_res0, b_res0)
    end

    if claim == :DualInfeasible
      if verbose; print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
      return exit_res(claim_dual_infeasible!(sol, ȳ, A), t_res0, b_res0)
    end

    # Cause of Divergence Unknown
    if !(isfinite(μ) && isfinite(rDu) && isfinite(rPr) && isfinite(rCp))
      if verbose; print("\n > EXIT -- Error!\n\n"); end
      sol.status = :Error; return exit_res(sol, t_res0, b_res0)
    end

    @phase_stop timing t_residuals b_residuals t_res0 b_res0

    # ────────────────────────────────────────────────────────────
    #  Factorization (only for an iterate that is going to be stepped)
    # ────────────────────────────────────────────────────────────

    # t_kktupdate: scaling-block assembly and numeric factorization.
    (t_kk0, b_kk0) = @phase_start timing
    solve = try
      solve4x4gen(λ,F,F⁻ᵀ)         # Caches 4x4 solver
                                   # (used a few times, at least 2)
    catch err
      err isa KKT_FAILURES || rethrow()
      sol.status = :Error
      sol.message = kkt_error("factorization, iteration $Iter", err)
      @phase_stop timing t_kktupdate b_kktupdate t_kk0 b_kk0
      return exit_loop(sol)
    end
    @phase_stop timing t_kktupdate b_kktupdate t_kk0 b_kk0

    # ────────────────────────────────────────────────────────────
    #  Predictor
    # ────────────────────────────────────────────────────────────

    # Scaled KKT residual of the step Δz against the right-hand side r,
    # left in the preallocated _rIr:
    #   rkkt = (QΔy + GᵀΔw − AᵀΔv, GΔy, AΔy − Δs, λ∘FΔv + λ∘F⁻ᵀΔs)
    function step_residual!(Δz, r)
      mul!(_res_buf3, F, Δz.v)
      cone_prod!(_res_buf1, λ, _res_buf3)
      mul!(_res_buf4, F⁻ᵀ, Δz.s)
      cone_prod!(_res_buf2, λ, _res_buf4)
      mul!(_rkkt.y, Q, Δz.y)
      mul!(_rkkt.y, Gᵀ, Δz.w, 1.0, 1.0)
      mul!(_rkkt.y, Aᵀ, Δz.v, -1.0, 1.0)
      mul!(_rkkt.w, G, Δz.y)
      mul!(_rkkt.v, A, Δz.y); _rkkt.v .-= Δz.s
      _rkkt.s .= _res_buf1 .+ _res_buf2
      sub4!(_rIr, r, _rkkt)
      return norm(_rIr)
    end

    # Iterative refinement of a step Δz for the right-hand side r: at most
    # maxRefinementSteps corrections, each one more back-solve with the
    # same factorization, until ‖r − KΔz‖ ≤ refineAbsTol + refineRelTol‖r‖.
    # The residual is evaluated after every correction, and a correction
    # that does not reduce it is undone and ends the loop: refinement with
    # an approximate factorization K̃ contracts only when ‖I − K̃⁻¹K‖ < 1,
    # and a worsening step is the evidence that it does not. The step that
    # leaves this function is therefore the best one seen, and `rnorm`
    # (relative, shown in red in the verbose row when large) describes it.
    # The tolerance is a target, not a guarantee: a step that still misses
    # it is used as is. Returns false after stamping sol when a correction
    # solve fails.
    # Timing: the whole call is t_dir_refine (inclusive diagnostic inside
    # t_direction, which the call sites wrap); the residual evaluations
    # are t_dir_refine_resid. A failed correction leaves the span early
    # and unaccounted.
    function refine!(Δz, r, stage)
      @phase timing t_dir_refine begin
      nr   = norm(r)
      rtol = refineAbsTol + refineRelTol * nr
      timing === nothing || (timing.n_refine_resid += 1)
      rres = @phase timing t_dir_refine_resid step_residual!(Δz, r)
      k    = 0                          # this call's own budget
      while k < maxRefinementSteps && rres > rtol
        timing === nothing || (timing.n_refine_attempt += 1)
        Δzr = guarded("refinement, $stage") do
          solve(_Δzr, _rIr)
        end
        Δzr === nothing && return false
        if !isfinite4(Δzr)
          nonfinite!("refinement direction", "refinement, $stage")
          return false
        end
        copy4!(_Δz_keep, Δz)
        axpy4!(1.0, Δzr, Δz)
        k += 1
        timing === nothing || (timing.n_refine_resid += 1)
        rnew = @phase timing t_dir_refine_resid step_residual!(Δz, r)
        if !(rnew < rres)               # also catches a NaN residual
          copy4!(Δz, _Δz_keep)
          break
        end
        rres = rnew
      end
      nref += k                         # iteration total, for the verbose row
      rnorm = rres / (1 + nr)
      true
      end # @phase t_dir_refine
    end
    nref = 0

    # t_direction covers the base solve (also t_dir_base) and the
    # refinement; the finiteness checks between them are left out.
    d_aff = @phase timing t_direction b_direction @phase timing t_dir_base guarded("predictor, iteration $Iter") do
      solve(_d_aff, r0)
    end
    d_aff === nothing && return exit_loop(sol)
    isfinite4(d_aff) ||
      return exit_loop(nonfinite!("predictor direction", "predictor, iteration $Iter"))
    (@phase timing t_direction b_direction refine!(d_aff, r0, "predictor, iteration $Iter")) ||
      return exit_loop(sol)

    α_aff_vs = @phase timing t_linesearch b_linesearch guarded("predictor line search, iteration $Iter") do
      ( min( maxstep( z.v, d_aff.v ) , 1 ),
        min( maxstep( z.s, d_aff.s ) , 1 ) )
    end
    α_aff_vs === nothing && return exit_loop(sol)
    α_aff = min( α_aff_vs[1] , α_aff_vs[2] )

    # t_rhs: centering parameter and the corrector right-hand side.
    r = @phase timing t_rhs b_rhs begin

    # >> ρ  = (z.v - α_aff*d_aff.v)'*(z.s - α_aff*d_aff.s)/μbar
    ρ  = fts(z.v, α_aff, d_aff.v, z.s, α_aff,d_aff.s)/μbar
    σ  = max(0,min(1,ρ))^3
    (isfinite(ρ) && isfinite(σ)) ||
      return exit_loop(nonfinite!("centering parameter", "predictor, iteration $Iter"))

    # ────────────────────────────────────────────────────────────
    #  Corrector
    # ────────────────────────────────────────────────────────────

    F⁻ᵀdfs = mul!(_rhs_buf1, F⁻ᵀ, d_aff.s)
    Fdfs   = mul!(_rhs_buf2, F, d_aff.v)

    # >> lc = -(F⁻ᵀdfs ∘ Fdfs) + (σ*μ)[1]*e;
    cone_prod!(_prod_buf2, F⁻ᵀdfs, Fdfs); lc = _prod_buf2
    axpy!(-σ*μ, e, lc);
    scal!(length(e), -1., lc, 1)

    # The y, w and v blocks are r0's (read only from here on); the s block
    # is the loop's own buffer, distinct from rleft.s (_prod_buf1) and
    # from lc (_prod_buf2).
    _rhs_s .= rleft.s .- lc
    v4x1(r0.y, r0.w, r0.v, _rhs_s)

    end # @phase t_rhs

    # ────────────────────────────────────────────────────────────
    #  Take newton step, with iterative refinement
    # ────────────────────────────────────────────────────────────

    Δz = @phase timing t_direction b_direction @phase timing t_dir_base guarded("corrector, iteration $Iter") do
      solve(_Δz, r)
    end
    Δz === nothing && return exit_loop(sol)
    isfinite4(Δz) ||
      return exit_loop(nonfinite!("corrector direction", "corrector, iteration $Iter"))

    (@phase timing t_direction b_direction refine!(Δz, r, "corrector, iteration $Iter")) ||
      return exit_loop(sol)
    isfinite4(Δz) ||
      return exit_loop(nonfinite!("search direction", "search direction, iteration $Iter"))

    # ────────────────────────────────────────────────────────────
    # Make Step
    # ────────────────────────────────────────────────────────────

    # maxstep is homogeneous of degree -1 in the direction, so scaling the
    # step back from the boundary by (1-DTB) is the same as searching along
    # Δz/(1-DTB) — without forming the scaled direction.
    α_vs = @phase timing t_linesearch b_linesearch guarded("line search, iteration $Iter") do
      ( min( 1, (1-DTB)*maxstep(z.v, Δz.v) ),
        min( 1, (1-DTB)*maxstep(z.s, Δz.s) ) )
    end
    α_vs === nothing && return exit_loop(sol)
    α = min( α_vs[1], α_vs[2] )

    # ────────────────────────────────────────────────────────────
    #  Gondzio multiple centrality correctors (off by default)
    #
    #  Each corrector asks for a longer step α̃ = min(1, α + δα), forms
    #  the trial scaled complementarity that step would produce,
    #      ṽ = λ − α̃·FΔv,   s̃ = λ − α̃·F⁻ᵀΔs,   w = ṽ ∘ s̃,
    #  and a correction Δw = Π_[βmin σμ, βmax σμ](w) − w, capped below at
    #  −βmax σμ, taken in the Jordan frame of w (correctors.jl). One more
    #  back-solve with the current factorization gives Δz_c; the candidate
    #  Δz + Δz_c is kept iff it lengthens the step by at least γ(α̃ − α).
    #
    #  Sign of the right-hand side. The fourth block row of the 4×4 system
    #  is λ∘(FΔv) + λ∘(F⁻ᵀΔs) = r_s, and the step is z ← z − αΔz, so an
    #  extra direction Δz_c changes the trial complementarity by
    #      −α·(λ∘FΔz_c.v + λ∘F⁻ᵀΔz_c.s) = −α·r_s   (to first order).
    #  Solving with r_s = −Δw therefore moves w by +α̃·Δw, toward the box
    #  (verified numerically in test/tranche3_tests.jl, "corrector sign").
    #  The corrector rhs is not divided by α̃ (Gondzio's convention): the
    #  realized move is a fraction of Δw, and the acceptance test decides.
    # ────────────────────────────────────────────────────────────
    #  Timing: the corrector formation and its line search are
    #  t_linesearch, the extra back-solve is t_direction; the spans are
    #  closed before every `break` and `return`.
    # ────────────────────────────────────────────────────────────
    if centralityCorrectors > 0
      cc_acc = 0; cc_try = 0
      σμ = σ*μ
      for _ in 1:centralityCorrectors
        (t_cc0, b_cc0) = @phase_start timing
        α̃ = min(1.0, α + GONDZIO_δα)
        # A full step needs no lengthening, and a zero centering target
        # (σ = 0, or μ ≤ 0) leaves no box to aim for.
        cc_go = α̃ > α && σμ > 0
        if cc_go
          mul!(_cc_b1, F,   Δz.v); _cc_v .= λ .- α̃ .* _cc_b1
          mul!(_cc_b2, F⁻ᵀ, Δz.s); _cc_s .= λ .- α̃ .* _cc_b2
          cone_prod!(_cc_w, _cc_v, _cc_s)
          centrality_correction!(_cc_dw, _cc_w, GONDZIO_βmin*σμ, GONDZIO_βmax*σμ,
                                 GONDZIO_βmax*σμ, cone_dims)
          # A non-finite correction, or one already inside the box (exactly
          # zero): nothing to solve.
          cc_go = all(isfinite, _cc_dw) && !all(iszero, _cc_dw)
          cc_go && (_cc_r.s .= .-_cc_dw)
        end
        @phase_stop timing t_linesearch b_linesearch t_cc0 b_cc0
        cc_go || break
        Δz_c = @phase timing t_direction b_direction guarded("centrality corrector, iteration $Iter") do
          solve(_Δz_c, _cc_r)
        end
        Δz_c === nothing && return exit_loop(sol)
        isfinite4(Δz_c) || break
        (t_cc1, b_cc1) = @phase_start timing
        cc_try += 1
        copy4!(_cc_Δz, Δz)
        axpy4!(1.0, Δz_c, _cc_Δz)
        α_new_vs = guarded("centrality corrector line search, iteration $Iter") do
          ( min( 1, (1-DTB)*maxstep(z.v, _cc_Δz.v) ),
            min( 1, (1-DTB)*maxstep(z.s, _cc_Δz.s) ) )
        end
        if α_new_vs === nothing
          @phase_stop timing t_linesearch b_linesearch t_cc1 b_cc1
          return exit_loop(sol)
        end
        α_new = min(α_new_vs[1], α_new_vs[2])
        cc_ok = α_new >= α + GONDZIO_γ*(α̃ - α)
        if cc_ok
          copy4!(Δz, _cc_Δz)
          α = α_new
          cc_acc += 1
        end
        @phase_stop timing t_linesearch b_linesearch t_cc1 b_cc1
        cc_ok || break
      end
    end

    # Verified interiority. maxstep is exact in exact arithmetic, but a
    # step that lands within rounding of the boundary makes the next
    # NT scaling fail (a terminal :Error today). Check the trial iterate
    # exactly and back off geometrically before accepting it.
    @phase timing t_linesearch b_linesearch begin
    ok = false
    for _ in 1:30
      _trial_v .= z.v .- α .* Δz.v
      _trial_s .= z.s .- α .* Δz.s
      if interior(_trial_v) && interior(_trial_s)
        ok = true
        break
      end
      α /= 2
    end
    if !ok
      sol.status = :Error
      sol.message = "no interior point along the search direction (line search, iteration $Iter)"
      _stamp_kkt!(sol)
      if verbose; print("\n > EXIT -- Error! ($(sol.message))\n\n"); end
      return exit_loop(sol)
    end

    # >> z = z - α*Δz;
    axpy4!(-α, Δz, z)

    # The next iteration's nt_scaling factors this iterate.
    isfinite4(z) ||
      return exit_loop(nonfinite!("iterate", "line search, iteration $Iter"))
    timing === nothing || (timing.n_steps += 1)
    end # @phase t_linesearch

    # Stall: three consecutive negligible steps mean the iteration is no
    # longer moving. Leave the loop and let the post-loop screens decide
    # between a certificate, an "almost" verdict, and :Abandoned.
    nstall = α < 1e-8 ? nstall + 1 : 0
    if nstall >= 3
      sol.message = "stalled: step length below 1e-8 for 3 consecutive iterations (iteration $Iter)"
      if verbose; print("\n > Stalled: step length below 1e-8 for 3 iterations\n"); end
      break
    end

  end
  exit_loop(nothing)                   # break / exhaustion path

  _stamp_kkt!(sol)
  if over_time()
    sol.status = :TimeLimit
    return sol
  end

  # ────────────────────────────────────────────────────────────
  #  Loop exhausted : re-screen the best iterate
  #
  #  The screens above are evaluated on the *current* iterate and can
  #  miss a ray that the best iterate carries. Re-validate that
  #  iterate here, first at the nominal tolerance (a full claim, rare)
  #  and then relaxed 100×, which downgrades to :AlmostInfeasible /
  #  :AlmostDualInfeasible rather than claiming.
  # ────────────────────────────────────────────────────────────

  # t_final: the post-loop screens and validations (the fallback solves
  # below are t_fallback; the final status assignment is not timed).
  @phase timing t_final begin
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
  end # @phase t_final

  # ── WP5 fallback: recover a ray by an auxiliary min-norm QP ──
  #  Only when both 1× validations failed AND there is evidence a ray
  #  exists (a relaxed validation passed, or complementarity collapsed).
  #  A clean :Abandoned with no such signal does not earn a solve.
  #  At most one attempt of each kind. The auxiliary problems run under
  #  default_kktsolver rather than the caller's solver: they have a
  #  different structure (min-norm, wide equalities, regularized), so
  #  the selection heuristic is re-run on the auxiliary data.
  if certFallback && !over_time() && !pchk.valid && !dchk.valid &&
     (pchk100.valid || dchk100.valid || μ_collapsed || μ_diverged)

    # The auxiliary solves use their own iteration budget: the outer
    # maxIters is small in exactly the regime the fallback exists for.
    # Each solve is gated on the deadline separately: the first can use
    # up what is left of the budget, and the second must not then start.
    if (pchk100.valid || μ_collapsed || μ_diverged) && p + m > 0 && !over_time()
      # The auxiliary solve runs untimed inside (no `timing` reaches it).
      ray = @phase timing t_fallback fallback_infeasibility_ray(
                                       Q, c, A, b, cone_dims, G, d;
                                       maxIters = certFallbackIters,
                                       timeLimit = max(time_left(), 0.0))
      if ray !== nothing
        (fchk, fw̄, fv̄) = @phase timing t_final validate_infeasibility_certificate(
                            Q, c, A, b, cone_dims, G, d, ray[1], ray[2];
                            abstol = infeasAbsTol, reltol = infeasTol)
        if fchk.valid
          if verbose; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
          return claim_infeasible!(sol, fw̄, fv̄)
        end
      end
    end

    if (dchk100.valid || μ_collapsed || μ_diverged) && n > 0 && !over_time()
      ray = @phase timing t_fallback fallback_unbounded_ray(
                                   Q, c, A, b, cone_dims, G, d;
                                   maxIters = certFallbackIters,
                                   timeLimit = max(time_left(), 0.0))
      if ray !== nothing
        (fchk, fȳ) = @phase timing t_final validate_unboundedness_certificate(
                       Q, c, A, b, cone_dims, G, d, ray;
                       abstol = infeasAbsTol, reltol = infeasTol)
        if fchk.valid
          if verbose; print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
          return claim_dual_infeasible!(sol, fȳ, A)
        end
      end
    end

  end

  if over_time()
    sol.status = :TimeLimit
    return sol
  end

  if pchk.valid
    if verbose; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
    return claim_infeasible!(sol, w̄, v̄)
  elseif dchk.valid
    if verbose; print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
    return claim_dual_infeasible!(sol, ȳ, A)
  elseif pchk100.valid && μ_collapsed
    sol.status = :AlmostInfeasible
  elseif dchk100.valid && μ_collapsed
    sol.status = :AlmostDualInfeasible
  else
    sol.status = :Abandoned
  end

  return sol

end

include("equilibrate.jl")
include("certificates.jl")
include("fallback.jl")
include("preprocessor.jl")
include("MOI_wrapper.jl")

end
