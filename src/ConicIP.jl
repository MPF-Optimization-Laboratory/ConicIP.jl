module ConicIP

export Id, conicIP, pivot, preprocess_conicIP,
  Optimizer, Block

import Base: +, *, -, \, ^
using LinearAlgebra
using LinearAlgebra.BLAS: axpy!, scal!
using SparseArrays
using WoodburyMatrices
using Printf

"""
    Id(n)

Create an `n`-by-`n` identity matrix as `Diagonal(ones(n))`.
"""
Id(n::Integer) = Diagonal(ones(n))

include("blockmatrices.jl")
include("kktsolvers.jl")

ViewTypes   = Union{SubArray}
VectorTypes = Union{Matrix, Vector, ViewTypes}
MatrixTypes = Union{Matrix, Array{Real,2},
                    SparseMatrixCSC{Real,Integer}}

# returns 0 for matrices with dimension 0.
normsafe(x) = isempty(x) ? 0 : norm(x)

# ──────────────────────────────────────────────────────────────
#  3x1 block vector
# ──────────────────────────────────────────────────────────────

mutable struct v4x1; y::Matrix; w::Matrix; v::Matrix; s::Matrix; end

+(a::v4x1, b::v4x1) = v4x1(a.y + b.y, a.w + b.w, a.v + b.v, a.s + b.s)
-(a::v4x1, b::v4x1) = v4x1(a.y - b.y, a.w - b.w, a.v - b.v, a.s - b.s)
LinearAlgebra.norm(a::v4x1) = norm(a.y) + normsafe(a.w) + normsafe(a.v) + normsafe(a.s)

function axpy4!(α::Number, x::v4x1, y::v4x1)
    axpy!(α, x.y, y.y); axpy!(α, x.w, y.w)
    axpy!(α, x.v, y.v); axpy!(α, x.s, y.s)
end

# ──────────────────────────────────────────────────────────────
#  Linear operator representing a congruence transform of a
#  matrix in vectorized form
# ──────────────────────────────────────────────────────────────

"""
    VecCongurance(R)

Linear operator representing a congruence transform in vectorized form.
The action `W * x` computes `vecm(R' * mat(x) * R)`.

Used internally as the Nesterov-Todd scaling matrix for semidefinite cones.
"""
mutable struct VecCongurance; R :: Matrix; end

*(W::VecCongurance, x::VectorTypes)    = vecm(W.R'*mat(x)*W.R)
Base.adjoint(W::VecCongurance)         = VecCongurance(W.R');
Base.inv(W::VecCongurance)             = VecCongurance(inv(W.R))
Base.size(W::VecCongurance, i)         = round(Int, size(W.R,1)*(size(W.R,1)+1)/2)
*(W1::VecCongurance, W2::VecCongurance) = VecCongurance(W2.R * W1.R)

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
  Z = zeros(n,n)
  for i = 1:n
    k = round(Int, length(x) - (n-i+2)*(n-i+1)/2)
    for j = 1:n
      if i <= j
        if i == j
          Z[i,j] = x[k+j-i+1]
        else
          Z[i,j] = x[k+j-i+1]/√2
        end
      else
        Z[i,j] = Z[j,i];
      end
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
  x = zeros(round(Int, n*(n+1)/2),1)
  c = 1
  for i = 1:n
    for j = 1:n
      if i <= j
        if i == j
          x[c] = Z[i,j];
        else
          x[c] = Z[i,j]*√2;
        end
        c = c + 1;
      end
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

  return SymWoodbury(J, w, 1.)

end

function nestod_sdc(z,s)

  # Nesterov-Todd Scaling Matrix for the Semidefinite Cone
  # Matrix which satisfies the properties
  # W*z = inv(W)*sb

  Ls  = cholesky(mat(s)).L
  Lz  = cholesky(mat(z)).L
  F   = svd(Lz'*Ls)
  U   = F.U
  Λ   = F.S
  R = inv(Lz)'*U*spdiagm(0 => sqrt.(Λ))
  return VecCongurance(R)

end

function maxstep_rp(x,d)

  # Assume x in R+.
  # Returns maximum α such that x + α*d in R+.

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
  μ  = ((β + d[1])/(xbar[1] + 1))[1]
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

  # Maximum step to Semidefinite cone
  X     = mat(x)
  # If X is not positive definite, return Inf
  λX    = eigvals(Symmetric(X))
  if any(λX .<= 0)
    return Inf
  end
  Xih   = X^(-1/2)
  D     = mat(d)
  XDX   = Xih*D*Xih
  XDX   = 0.5*(XDX + XDX')
  Λ     = eigvals(XDX)
  Λn    = Λ .< 0
  if all(Λn)
    return Inf
  else
    return 1/maximum(Λ[.!Λn])
  end

end

function maxstep_sdc(x,d::Nothing)

  # Maximum step to Semidefinite cone
  X = mat(x)
  Λ = eigvals(X)
  minΛ  = minimum(Λ)
  return all(minΛ .> 0) ? 0 : -1 + minΛ

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

  n = round(Int, sqrt(size(x,1)))
  X = mat(x); Y = mat(y)
  o[:] = vecm(lyap(Y,-X))

end

function xsdc!(x, y, o)

  X = mat(x); Y = mat(y)
  o[:] = vecm(X*Y + Y*X)

end

# ──────────────────────────────────────────────────────────────
#  Interior Point
# ──────────────────────────────────────────────────────────────

"""
    Solution

Return type of [`conicIP`](@ref) and [`preprocess_conicIP`](@ref).

# Fields
- `y::Matrix` -- primal variables
- `w::Matrix` -- dual variables for equality constraints (Gy = d)
- `v::Matrix` -- dual variables for inequality constraints (Ay ≥_K b)
- `status::Symbol` -- `:Optimal`, `:Infeasible`, `:Unbounded`, `:Abandoned`, or `:Error`
- `Iter::Integer` -- number of interior-point iterations
- `Mu::Real` -- final complementarity gap parameter
- `prFeas::Real` -- primal feasibility residual
- `duFeas::Real` -- dual feasibility residual
- `muFeas::Real` -- complementarity residual
- `pobj::Real` -- primal objective value
- `dobj::Real` -- dual objective value
"""
mutable struct Solution

  y      :: Matrix  # primal
  w      :: Matrix  # dual (linear equality)
  v      :: Matrix  # dual (linear inequality)
  status :: Symbol  # :Optimal, :Infeasible
  Iter   :: Integer # number of iterations
  Mu     :: Real    # optimality conditions
  prFeas :: Real
  duFeas :: Real
  muFeas :: Real
  pobj   :: Real
  dobj   :: Real

end

"""
  conicIP(Q, c, A, b, cone_dims, G, d;
  solve3x3gen = solve3x3gen_sparse,
  optTol = 1e-5,
  DTB = 0.01,
  verbose = true,
  maxRefinementSteps = 3,
  maxIters = 100,
  cache_nestodd = false,
  refinementThreshold = optTol/1e7)

Interior point solver for the system

```
minimize    ½yᵀQy - cᵀy
s.t         Ay >= b
            Gy  = d
```

Q,c,A,b,G,d are matrices (c,b,d are NOT vectors)

cone_dims is an array of tuples (Cone Type, Dimension)

```
e.g. [("R",2),("Q",4)] means
(y₁, y₂)          in  R+
(y₃, y₄, y₅, y₆)  in  Q
```

SDP Cones are NOT supported and purely experimental at this
point.

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
  Q, c::Matrix,

  # Ax ≧ b
  A, b::Matrix, cone_dims,

  # Gx = d
  G = spzeros(0,length(c)), d = zeros(0,1);

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
  kktsolver = kktsolver_qr,

  optTol = 1e-6,           # Optimal Tolerance
  DTB = 0.01,              # Distance to Boundary
  verbose = true,          # Verbose Output
  maxRefinementSteps = 3,  # Maximum number of IR Steps
  maxIters = 100,          # Maximum number of interior iterations
  cache_nestodd = false,   # Set to true if there are many small blocks
  infeasTol = optTol,      # Infeasibility threshold (this shouldn't need to be tweaked,
                           # but set it small if the program returns infeasible/unbounded when
                           # you are sure it isn't)
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

  normc = norm(c)
  normd = isempty(d) ? -Inf : norm(d)
  normb = isempty(b) ? -Inf : norm(b)

  # Sanity Checks
  ◂ = nothing
  size(Q,1) != size(Q,2) ? error("Q is not square") : ◂
  size(b,1) != m        ? error("Inconsistency in inequalities") : ◂
  size(c,1) != n        ? error("Inconsistency in inequalities/objective") : ◂
  size(d,1) != p        ? error("Inconsistency in equalities") : ◂
  size(G,2) != n        ? error("Inconsistency in equalities/objective") : ◂

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
  e = zeros(m,1)
  for (btype, I, i) = block_data
    m_i = length(I)
    if btype == "R"; e[I] = ones(m_i,1);             end
    if btype == "Q"; e[I] = [1; zeros(m_i-1,1)];     end
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
    # λ = F*x = F\y

    B = Block(size(block_sizes,1));

    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I);
      if btype == "R"; B[i] = Diagonal(sqrt.(yI./xI)); end
      if btype == "Q"; B[i] = nestod_soc(xI, yI); end
      if btype == "S"; B[i] = nestod_sdc(xI, yI); end
    end

    return B;

  end

  function ÷(x,y)

    # Group division x ○\ y

    o = zeros(length(x),1)
    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I); oI = view(o,I)
      if btype == "R"; drp!(xI, yI, oI);  end
      if btype == "Q"; dsoc!(xI, yI, oI); end
      if btype == "S"; dsdc!(xI, yI, oI); end
    end
    return o;

  end

  function ∘(x,y)

    # Group product x ○ y

    o = zeros(length(x),1)
    @inbounds for (btype, I, i) = block_data
      xI = view(x,I); yI = view(y,I); oI = view(o,I)
      if btype == "R"; xrp!(xI, yI, oI);  end
      if btype == "Q"; xsoc!(xI, yI, oI); end
      if btype == "S"; xsdc!(xI, yI, oI); end
    end
    return o;

  end

  solve3x3gen = kktsolver(Q,A,G,cone_dims)

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

      t1 = F'*(r.s ÷ λ)
      (Δy, Δw, Δv)  = solve3x3(r.y, r.w, r.v + t1)
      axpy!(-1, F'*(F*Δv), t1) # > Δs = t1 - F*(F*Δv)
      return v4x1(Δy,Δw,Δv,t1)

    end

  end

  if verbose
      print("\n > INTERIOR POINT SOLVER v0.7 (July 2016)\n\n")
  end

  # ────────────────────────────────────────────────────────────
  #  Initial Point
  # ────────────────────────────────────────────────────────────

  I  = Block([Diagonal(ones(i)) for i = block_sizes])
  r0 = v4x1(c, d, b, zeros(m,1))
  z  = solve4x4gen(e,I,I)(r0)

  α_v = maxstep(z.v, nothing)
  α_s = maxstep(z.s, nothing)

  # Change to +
  z.v = z.v - α_v*e
  z.s = z.s - α_s*e

  if verbose
      println("            Optimality                      Objective              Infeasibility       ")
      println()
      ξ1()=@printf("\x1b[1m %-6s  │  %-8s  %-8s  %-8s │  %-8s  %-8s  │  %-8s  %-8s │  %-8s \x1b[0m\n",
                  "  Iter","prFeas","duFeas","muFeas","pobj","dobj","icertp","icertd","refine");ξ1()
  end

  # ────────────────────────────────────────────────────────────
  #  Iterate Loop
  # ────────────────────────────────────────────────────────────

  sol     = Solution(z.y, z.w, z.v, :None, 0, 0, Inf, Inf, Inf, Inf, -Inf)
  optBest = Inf
  rStep   = 0
  rnorm   = 0
  for Iter = 1:maxIters

    F    = nt_scaling(z.v, z.s)   # Nesterov-Todd Scaling Matrix
    F⁻ᵀ  = inv(F)'
    λ    = F*z.v;                 # This is also F⁻ᵀ*z.s.

    solve = solve4x4gen(λ,F,F⁻ᵀ)   # Caches 4x4 solver
                                   # (used a few times, at least 2)

    #         ┌                   ┐ ┌     ┐
    # rleft = │ Q   G'   -A'      │ │ z.y │
    #         │ G                 │ │ z.w │  V = block(λ)*F⁻ᵀ
    #         │ A              -I │ │ z.v │    = block(λ)*λ
    #         │           S     V │ │ z.s │
    #         └                   ┘ └     ┘
    rleft = v4x1( Q*z.y + Gᵀ*z.w - Aᵀ*z.v ,
                  G*z.y                   ,
                  A*z.y - z.s             ,
                  λ ∘ λ                   )

    # True Residual of nonlinear KKT System
    r0 = v4x1(rleft.y - c, rleft.w - d, rleft.v - b, rleft.s);

    # Gap
    μbar = dot(z.v,z.s)
    μ    = μbar/conedim

    # ────────────────────────────────────────────────────────────
    #  Print iterate status, save best iterate
    # ────────────────────────────────────────────────────────────

    cᵀy = dot(c,z.y)
    rDu = norm(r0.y)/(1+norm(c))
    rPr = normsafe(r0.v)/(1+normsafe(b))
    rCp = normsafe(r0.s)/(1+abs(cᵀy));

    if max(rDu, rPr, rCp) < optBest
      sol.y[:] = z.y; sol.w[:] = z.w; sol.v[:] = z.v
      sol.Iter = Iter; sol.Mu = μ;
      sol.duFeas = rDu; sol.prFeas = rPr; sol.muFeas = rCp
      optBest = max(rDu, rPr, rCp)
    end

    pobj = 0.5*dot(z.y, Q*z.y) - dot(c, z.y)
    dobj = pobj + dot(z.w, r0.w) + dot(z.v, r0.v) - dot(z.v, z.s)

    sol.pobj = pobj
    sol.dobj = dobj

    # ────────────────────────────────────────────────────────────
    # Convergence Checks (on previous iterate)
    # ────────────────────────────────────────────────────────────

    # Optimality
    if max(rDu, rPr, rCp) < optTol
      sol.status = :Optimal
    end

    if !(p == 0 && m == 0)

      # Primal Infeasibility
      #
      # Certificate: ∃w,v such that
      #   Gᵀw + Aᵀv = 0
      #   bᵀv + dᵀw < 0
      #   w ≧ 0
      #
      # The program returns a certificate w,v that satisfies
      #
      #  CVXOPT style         ECOS Style
      #  -------------------------------------------
      #   Gᵀw + Aᵀv = 0        Gᵀw + Aᵀv = 0
      #   bᵀv + dᵀw = -1       bᵀv + dᵀw < 0
      #   w ≧ 0                w ≧ 0
      #                        norm(v) + norm(w) = 1
      #
      dᵀy_bᵀv  = dot(d,z.w) - dot(b,z.v)

      p_infeas_unscaled = norm(Gᵀ*z.w - Aᵀ*z.v)
      p_infeas_cvx = dᵀy_bᵀv < 0 ? p_infeas_unscaled/(normsafe(z.y) + normsafe(z.v)) : NaN
      p_infeas_ecos = dᵀy_bᵀv < 0 ? p_infeas_unscaled/(max(1,normc)*abs(dᵀy_bᵀv)) : NaN
      p_infeas = max(p_infeas_cvx, p_infeas_ecos)

      if p_infeas < infeasTol
        sol.y[:] = 0*sol.y[:]/0; sol.w[:] = z.w/-dᵀy_bᵀv; sol.v[:] = z.v/-dᵀy_bᵀv;
        sol.status = :Infeasible
      end

      # Dual Infeasiblity
      #
      # Certificate: ∃y,s such that
      #   Ay - s = 0   (d_infeas1)
      #   Gy = 0       (d_infeas2)
      #   Qy = 0       (d_infeas3)
      #   cᵀy > 0
      #   s ≧ 0
      #
      # The program returns a certificate y that satisfies
      #
      #  CVXOPT style    ECOS style
      #  ------------------------------
      #   Ay ≧ 0          Ay ≧ 0
      #   Gy = 0          Gy = 0
      #   Qy = 0          Qy = 0
      #   cᵀy = 1         cᵀy > 0
      #                   norm(y) = 1
      #
      d_infeas1 = isempty(A) ? -Inf : norm(A*z.y - z.s)
      d_infeas2 = isempty(G) ? -Inf : norm(G*z.y)
      d_infeas3 = all(isfinite.(z.y)) ? norm(Q*z.y) : NaN

      d_infeas_cvx = cᵀy > 0 ? max(d_infeas1/max(1,normb), d_infeas2/max(1,normd), d_infeas3/max(1,normc))/abs(cᵀy) : NaN
      d_infeas_ecos = cᵀy > 0 ? max(d_infeas1, d_infeas2, d_infeas3)/norm(z.y) : NaN
      d_infeas = abs(max(d_infeas_cvx, d_infeas_ecos))

      if d_infeas < infeasTol
        sol.y[:] = z.y/abs(cᵀy); sol.v[:] = 0*sol.v[:]/0; sol.w[:] = 0*sol.w[:]/0
        sol.status = :Unbounded
      end

    end

    if verbose
      if rnorm > 0.001; print("\x1b[1m\x1b[31m"); end
      ξ2()=@printf(" %6i  │  %-8.1e  %-8.1e  %-8.1e │  % -8.1e  % -8.1e  │  %-8.1e  %-8.1e │  %i\n",
                  Iter, rDu, rPr, rCp, pobj, dobj, p_infeas, d_infeas, rStep);ξ2()
      if rnorm > 0.001; print("\x1b[0m"); end
    end

    if verbose
      if sol.status == :Infeasible; print("\n > EXIT -- Certificate of Infeasiblity Found!\n\n"); end
      if sol.status == :Unbounded;  print("\n > EXIT -- Certificate of Dual Infeasibility Found!\n\n"); end
      if sol.status == :Optimal;    print("\n > EXIT -- Below Tolerance!\n\n"); end
    end

    if sol.status != :None; return sol; end

    # Cause of Divergence Unknown
    if !all(isfinite.([μ, rDu, rPr, rCp]))
      if verbose; print("\n > EXIT -- Error!\n\n"); end
      sol.status = :Error; return sol
    end

    # ────────────────────────────────────────────────────────────
    #  Predictor
    # ────────────────────────────────────────────────────────────

    d_aff   = solve(r0)

    α_aff_v = min( maxstep( z.v, d_aff.v ) , 1 )
    α_aff_s = min( maxstep( z.s, d_aff.s ) , 1 )
    α_aff   = min( α_aff_v , α_aff_s )

    # >> ρ  = (z.v - α_aff*d_aff.v)'*(z.s - α_aff*d_aff.s)/μbar
    ρ  = fts(z.v, α_aff, d_aff.v, z.s, α_aff,d_aff.s)/μbar
    σ  = max(0,min(1,ρ))^3

    # ────────────────────────────────────────────────────────────
    #  Corrector
    # ────────────────────────────────────────────────────────────

    F⁻ᵀdfs = F⁻ᵀ*d_aff.s
    Fdfs   = F*d_aff.v

    # >> lc = -(F⁻ᵀdfs ∘ Fdfs) + (σ*μ)[1]*e;
    lc = (F⁻ᵀdfs ∘ Fdfs); axpy!(-(σ*μ)[1], e, lc);
    scal!(length(e), -1., lc, 1)

    r  =  v4x1(r0.y, r0.w, r0.v, rleft.s - lc)

    # ────────────────────────────────────────────────────────────
    #  Take newton step, with iterative refinement
    # ────────────────────────────────────────────────────────────

    Δz  = solve(r);
    rStep = 1;
    for rStep = 1:maxRefinementSteps
      rkkt  = v4x1( Q*Δz.y  + Gᵀ*Δz.w  - Aᵀ*Δz.v , # y
                    G*Δz.y                       , # w
                    A*Δz.y - Δz.s                , # v
                    λ∘(F*Δz.v) + λ∘(F⁻ᵀ*Δz.s) )    # s
      rIr = r - rkkt
      rnorm = norm(rIr)/(n + 2*m)
      if rnorm < refinementThreshold; break; end
      Δzr = solve(rIr)
      Δz  = Δz + Δzr
    end

    # ────────────────────────────────────────────────────────────
    # Make Step
    # ────────────────────────────────────────────────────────────

    α_v = min( maxstep(z.v, Δz.v/(1-DTB)), 1 )
    α_s = min( maxstep(z.s, Δz.s/(1-DTB)), 1 )
    α   = min( α_v, α_s )

    # >> z = z - α*Δz;
    axpy4!(-α, Δz, z)

  end

  sol.status = :Abandoned
  return sol

end

include("preprocessor.jl")
include("MOI_wrapper.jl")

end
