import Base: +, *, -, \, ^, getindex, setindex!, show, print

# ****************************************************************
# Square Block Diagonal Matrix
# ****************************************************************

export Block, size, block_idx, broadcastf,
    copy, getindex, setindex!, +, -, *, \, inv, square;

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

  Blocks::Vector{Any}

  Block(size::Int) = new(Vector{Any}(undef, size))
  Block(Blk::Vector{Any}) = new(Blk)
  Block(Blk::Array) = new(convert(Vector{Any}, Blk))

end

function Base.size(A::Block)
  if length(A.Blocks) == 0; return (0,0); end
  n = sum([size(B,1) for B in A.Blocks])
  return (n,n)
end

Base.size(A::Block, i::Integer)    = (i == 1 || i == 2) ? size(A)[1] : 1
getindex(A::Block, i::Integer)     = A.Blocks[i]
setindex!(A::Block, B, i::Integer) = begin; A.Blocks[i] = B; end

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

Base.copy(A::Block)        = Block(copy(A.Blocks))
Base.deepcopy(A::Block)    = Block(deepcopy(A.Blocks))
+(A::Block, B::Block)      = Block(A.Blocks + B.Blocks)
-(A::Block, B::Block)      = A + (-B)
Base.inv(A::Block)         = broadcastf(inv, A)
-(A::Block)                = broadcastf(-, A)
Base.adjoint(A::Block)     = broadcastf(adjoint, A)
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
