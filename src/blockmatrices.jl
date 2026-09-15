import Base: +, *, -, \, ^, getindex, setindex!, show, print

# ****************************************************************
# Square Block Diagonal Matrix
# ****************************************************************

export Block, size, block_idx, broadcastf, mul_adjoint!,
    copy, getindex, setindex!, +, -, *, \, inv, square;

# Union of all block element types — Julia's union-splitting handles small unions efficiently.
# SymWoodbury{Float64} matches all concrete SymWoodbury variants (Vector vs Matrix B,
# scalar vs Matrix D) produced by nestod_soc and SymWoodbury products.
const BlockElem = Union{Diagonal{Float64, Vector{Float64}},
                        SymWoodbury{Float64},
                        VecCongurance, Matrix{Float64}}

# The two block types the hot path actually sees, spelled out concretely:
# `BlockElem` holds them behind non-concrete union members, so code that
# only knows `Blk::BlockElem` dispatches dynamically and any `view` built
# for it escapes onto the heap. Narrowing with `isa` first keeps the common
# cases statically dispatched and allocation-free.
const DiagBlock = Diagonal{Float64,Vector{Float64}}
# What nestod_soc builds, and what inv/adjoint of it return.
const SOCBlock  = SymWoodbury{Float64, DiagBlock, Vector{Float64}, Float64, Float64}

"""
    Block(size::Int)
    Block(Blk::Vector)

Block diagonal matrix type. Each diagonal block can be a different
matrix type (`Diagonal`, `SymWoodbury`, `VecCongurance`, or dense `Matrix`).

Used internally to represent the Nesterov-Todd scaling matrix, where
each block corresponds to a cone in the cone specification.

Supports arithmetic (`*`, `+`, `-`, `inv`, `adjoint`, `^`),
conversion to `sparse` and `Matrix`, and block-wise function
application via [`broadcastf`](@ref).

# Indexing
- `B[i]` returns the `i`-th diagonal block
- `B[i] = M` sets the `i`-th diagonal block
"""
mutable struct Block <: AbstractMatrix{Real}

  Blocks::Vector{BlockElem}

  Block(size::Int) = new(Vector{BlockElem}(undef, size))
  Block(Blk::Vector{BlockElem}) = new(Blk)
  Block(Blk::AbstractVector) = new(BlockElem[b for b in Blk])

end

function Base.size(A::Block)
  if length(A.Blocks) == 0; return (0,0); end
  n = sum([size(B,1) for B in A.Blocks])
  return (n,n)
end

Base.size(A::Block, i::Integer)    = (i == 1 || i == 2) ? size(A)[1] : 1
getindex(A::Block, i::Int)         = A.Blocks[i]
setindex!(A::Block, B::BlockElem, i::Int) = begin; A.Blocks[i] = B; end
# NB: any block type outside BlockElem is SILENTLY densified here. If a
# block operation starts returning a new type (e.g. a plain Woodbury from
# a SymWoodbury product), extend BlockElem rather than paying an O(k²)
# dense conversion on every assignment.
setindex!(A::Block, B, i::Int) = begin; A.Blocks[i] = Matrix{Float64}(B); end

"""
    block_idx(A::Block)

Return a vector of `UnitRange{Int}` giving the row/column index ranges
for each diagonal block of `A`.
"""
function block_idx(A::Block)

  k = size(A.Blocks,1)
  IColl = Vector{UnitRange{Int}}(undef, k)

  cum_count = 1
  for i = 1:k
    blk_size = size(A.Blocks[i],1);
    IColl[i] = cum_count:(cum_count + blk_size - 1)
    cum_count += blk_size
  end

  return IColl

end


"""
    broadcastf(op, A::Block)
    broadcastf(op, A::Block, B::Block)
    broadcastf(op, A::Block, x::Union{Vector,Matrix})

Apply function `op` block-wise to the diagonal blocks of `A`
(and optionally `B` or the corresponding segments of `x`).
"""
function broadcastf(op::Function, A::Block)

  B = copy(A)
  for i = 1:length(A.Blocks)
    B[i] = op(A[i])
  end
  return B

end

function broadcastf(op::Function, A::Block, B::Block)

  C = copy(A)
  for i = 1:length(A.Blocks)
    C[i] = op(A[i], B[i])
  end
  return C

end

function broadcastf(op::Function, A::Block, x::Vector)

  y = similar(x)
  i = 1
  @inbounds for I = block_idx(A)
    xI = view(x,I);
    y[I] = op(A.Blocks[i], xI)
    i += 1;
  end
  return y;

end

function broadcastf(op::Function, A::Block, X::Matrix)

  Y = similar(X)
  i = 1
  @inbounds for I = block_idx(A)
    XI = view(X,I,:);
    Y[I,:] = op(A.Blocks[i],XI)
    i += 1;
  end
  return Y

end

# Direct sparse conversion for SymWoodbury: A + B*D*B'
# Avoids dense Matrix(W) intermediate allocation.
function SparseArrays.sparse(W::SymWoodbury)
  A_sp = sparse(W.A)
  B_mat = W.B isa Vector ? reshape(W.B, :, 1) : W.B
  B_sp = sparse(B_mat)
  D_sp = W.D isa Number ? W.D : sparse(W.D)
  return A_sp + B_sp * D_sp * B_sp'
end

function SparseArrays.sparse(A::Block)

  I₊, J₊, V₊ = Int[], Int[], Float64[]
  @inbounds for (I,Blk) = zip(block_idx(A), A.Blocks)
    Aᵢ = sparse(Blk)
    rows = rowvals(Aᵢ)
    vals = nonzeros(Aᵢ)
    m, n = size(Aᵢ)
    for i = 1:n
       for j in nzrange(Aᵢ, i)
          row = rows[j]; val = vals[j]
          push!(J₊, i + I[1] - 1); push!(I₊, row + I[1] - 1); push!(V₊, val)
       end
    end
  end
  return sparse(I₊,J₊,V₊);

end

function Base.Matrix(A::Block)

  O = zeros(size(A))
  for (I,Blk) = zip(block_idx(A), A.Blocks)
    O[I,I] = Matrix(Blk)
  end
  return O;

end


*(A::Block, X::Array{Float64,2}) = broadcastf(*,A,X)
*(A::Adjoint{<:Any,Block}, X::Array{Float64,2}) = broadcastf((a,b) -> a'*b, parent(A), X)

*(A::Block, X::Vector) = broadcastf(*,A,X)
*(A::Adjoint{<:Any,Block}, X::Vector) = broadcastf((a,b) -> a'*b, parent(A), X)

# ****************************************************************
# In-place block-diagonal matrix-vector products
#
#   mul!(y, F, x)   and   mul!(y, F', x)
#
# The allocating `*` methods above stay for the code (and tests) that use
# them; the solver's hot path goes through these. Each per-block kernel
# performs EXACTLY the floating-point operations of the corresponding
# `Blk*xI` / `Blk'*xI` above, in the same order, so a direction computed
# through `mul!` is bit-identical to one computed through `*`.
#
# `Diagonal`, `SymWoodbury` and `Matrix` blocks allocate nothing.
# `VecCongurance` (SDP) still allocates inside `mat`/`vecm`; those blocks
# are small and off the hot path.
# ****************************************************************

# Diagonal blocks are real, so the adjoint is the same operator.
@inline function _blk_mul!(y, Blk::Diagonal{Float64,Vector{Float64}}, x, ::Bool)
  d = Blk.diag
  @inbounds for i in eachindex(y, x, d)
    y[i] = d[i] * x[i]
  end
  return y
end

# SymWoodbury{<:Real} is Hermitian: WoodburyMatrices defines adjoint(W) = W,
# so the adjoint flag is ignored. This is the body of
# `*(W::AbstractWoodbury, x::AbstractVector)` with the final `tmpN1 + tmpN2`
# written into `y` instead of a fresh vector.
@inline function _blk_mul!(y, W::SymWoodbury{Float64}, x, ::Bool)
  mul!(W.tmpN1, W.A, x)
  mul!(W.tmpk1, W.V, x)
  mul!(W.tmpk2, W.C, W.tmpk1)
  mul!(W.tmpN2, W.U, W.tmpk2)
  t1 = W.tmpN1; t2 = W.tmpN2
  @inbounds for i in eachindex(y, t1, t2)
    y[i] = t1[i] + t2[i]
  end
  return y
end

@inline function _blk_mul!(y, Blk::Matrix{Float64}, x, adj::Bool)
  adj ? mul!(y, adjoint(Blk), x) : mul!(y, Blk, x)
  return y
end

function _blk_mul!(y, Blk::VecCongurance, x, adj::Bool)
  copyto!(y, adj ? adjoint(Blk) * x : Blk * x)
  return y
end

function _block_mul!(y::AbstractVector, A::Block, x::AbstractVector, adj::Bool)
  length(y) == length(x) ||
    throw(DimensionMismatch("Block product: destination has length $(length(y)), source $(length(x))"))
  off = 0
  @inbounds for i in eachindex(A.Blocks)
    Blk = A.Blocks[i]
    # ::Int matters: `size(::SymWoodbury{Float64}, 1)` is inferred as Any
    # (the type is not concrete), which would make `off` — and with it every
    # index below — dynamic, and box the loop.
    k = size(Blk, 1)::Int
    if Blk isa DiagBlock
      d = Blk.diag
      for t = 1:k
        y[off+t] = d[t] * x[off+t]
      end
    elseif Blk isa SOCBlock
      _blk_mul!(view(y, (off+1):(off+k)), Blk, view(x, (off+1):(off+k)), adj)
    else
      rng = (off+1):(off+k)
      _blk_mul!(view(y, rng), Blk, view(x, rng), adj)
    end
    off += k
  end
  off == length(y) ||
    throw(DimensionMismatch("Block product: blocks span $off rows, destination has $(length(y))"))
  return y
end

LinearAlgebra.mul!(y::AbstractVector, A::Block, x::AbstractVector) =
  _block_mul!(y, A, x, false)
LinearAlgebra.mul!(y::AbstractVector, A::Adjoint{<:Any,Block}, x::AbstractVector) =
  _block_mul!(y, parent(A), x, true)

"""
    mul_adjoint!(y, A::Block, x)

Write `A'x` into `y` without forming the adjoint `Block`. Equivalent to
`mul!(y, A', x)`, and provided because `A'` on a `Block` is an eager
`broadcastf` that copies the block vector.
"""
mul_adjoint!(y::AbstractVector, A::Block, x::AbstractVector) =
  _block_mul!(y, A, x, true)

Base.copy(A::Block)        = Block(copy(A.Blocks))
Base.deepcopy(A::Block)    = Block(deepcopy(A.Blocks))
+(A::Block, B::Block)      = Block(A.Blocks + B.Blocks)
-(A::Block, B::Block)      = A + (-B)
Base.inv(A::Block)         = broadcastf(inv, A)
-(A::Block)                = broadcastf(-, A)
Base.adjoint(A::Block)     = broadcastf(adjoint, A)

"""
    inv_adjoint!(dest::Block, src::Block)

Compute `adjoint(inv(src))` block-wise, reusing the `dest` Block shell.
Avoids allocating two intermediate Blocks for `inv(F)'`.
"""
function inv_adjoint!(dest::Block, src::Block)
    for i = 1:length(src.Blocks)
        inv_adjoint_block!(dest, src, i)
    end
    return dest
end

"""
    inv_adjoint_block!(dest::Block, src::Block, i)

One block of [`inv_adjoint!`](@ref). A `Diagonal` destination of the right
length is overwritten in place — `inv(::Diagonal)` would otherwise allocate
a vector per cone per iteration — with `inv`'s singularity check kept. Any
other block type is rebuilt.
"""
function inv_adjoint_block!(dest::Block, src::Block, i::Integer)
    S = src.Blocks[i]
    if S isa DiagBlock && isassigned(dest.Blocks, Int(i))
        D = dest.Blocks[i]
        if D isa DiagBlock && length(D.diag) == length(S.diag)
            d = D.diag; sd = S.diag
            @inbounds for t in eachindex(d, sd)
                iszero(sd[t]) && throw(SingularException(t))
                d[t] = inv(sd[t])
            end
            return dest
        end
    end
    dest.Blocks[i] = adjoint(inv(S))
    return dest
end
*(A::Block, B::Block)      = broadcastf(*, A, B)
*(A::Adjoint{<:Any,Block}, B::Block) = broadcastf((a,b) -> a'*b, parent(A), B)

ViewTypes = Union{SubArray}
VectorTypes = Union{Matrix, Vector, ViewTypes}

# Extra functions for dealing with views and stuff

function +(A::Diagonal, B::Block)
  i = 1
  B0 = Block(length(B.Blocks))
  for I = block_idx(B)
    dI = A.diag[I];
    B0[i] = B[i] + Diagonal(dI);
    i = i + 1
  end
  return B0
end

^(A::Block,n::Integer) = broadcastf(x -> ^(x,n), A);
