# ──────────────────────────────────────────────────────────────
#  Various KKT Solvers
# ──────────────────────────────────────────────────────────────

"""
Solves the 3x3 system
```
┌             ┐ ┌    ┐   ┌   ┐
│ Q   G'  -A' │ │ y' │ = │ y │
│ G           │ │ w' │   │ w │
│ A       FᵀF │ │ v' │   │ v │
└             ┘ └    ┘   └   ┘
```
by the double QR method described in CVXOPT
http://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
section 10.2
"""
function kktsolver_qr(Q, A, G, cone_dims)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  F = qr(Matrix(G'))
  Q0 = F.Q * Matrix{Float64}(LinearAlgebra.I, size(G',1), size(G',1))  # materialize full Q
  R1 = F.R
  Q1      = Q0[:,1:p]
  Q2      = Q0[:,p+1:end]

  function solve3x3gen(F, F⁻ᵀ)

    F⁻ᵀ     = Matrix(inv(F))'
    Atil    = F⁻ᵀ*Matrix(A)
    QpAᵀA   = Q + Atil'Atil
    L = qr(Q2'*(QpAᵀA)*Q2)

    function solve3x3(bx, by, bz)

      Q1ᵀx = R1'\by
      Q2ᵀx = L \ ( Q2'*(bx + Atil'*(F⁻ᵀ*bz)) -
                   Q2'*((QpAᵀA)*(Q1*(Q1ᵀx))) )
      y    = R1 \ ( Q1'*(bx + Atil'*(F⁻ᵀ*bz))    -
                    Q1'*(QpAᵀA*(Q1*Q1ᵀx)) -
                    Q1'*(QpAᵀA*(Q2*Q2ᵀx)) )
      x    = Q0'\[Q1ᵀx; Q2ᵀx]
      Fz   = ( F⁻ᵀ*bz -
               Atil*(Q1*(Q1ᵀx)) - Atil*(Q2*(Q2ᵀx)) )
      z    = inv(F)*Fz

      return (x,y,z)

    end

    return solve3x3

  end

end

function lift(F::Block)

  d = zeros(0)

  IA, JA, VA = Int[], Int[], Float64[]
  IB, JB, VB = Int[], Int[], Float64[]
  ID, JD, VD = Int[], Int[], Float64[]

  n = block_idx(F)[end][end]
  Ir = 0   # Index of top right coordinate for expansion

  for (In,Blk) = zip(block_idx(F), F.Blocks)

    if isa(Blk, SymWoodbury)

      for i = 1:length(Blk.A.diag)
        push!(IA,In[i]); push!(JA,In[i]); push!(VA,Blk.A.diag[i])
      end

      for i = 1:size(Blk.B,1), j = 1:size(Blk.B,2)
        push!(IB,In[i]); push!(JB,Ir+j); push!(VB,Blk.B[i,j])
      end

      for i = 1:size(Blk.D,1), j = 1:size(Blk.D,2)
        invD = inv(Blk.D)
        if Blk.D[i,j] != 0
          push!(ID,Ir+i); push!(JD,Ir+j); push!(VD,-invD[i,j])
        end
      end

      Ir = Ir + size(Blk.B,2)

    end

    if isa(Blk, Diagonal)

      for i = 1:length(Blk.diag)
        push!(IA, In[i]); push!(JA, In[i]); push!(VA, Blk.diag[i])
      end

    end

  end
  return (sparse(IA,JA,VA), sparse(IB,JB,VB,n,Ir), sparse(ID,JD,VD));

end

"""
Estimates for the number of nonzeros of lift(F)
"""
function count_lift(cone_dims)
  n = 0
  for (btype, k) = cone_dims
    if btype == "Q"; n = n + k + 2*(2*k) + 4;  end
    if btype == "R"; n = n + k; end
    if btype == "S"; n = n + k^2; end
  end
  return n
end

"""
Estimates the number of nonzeros of F
"""
function count_dense(cone_dims)
  n = 0
  for (btype, k) = cone_dims
    if btype == "Q"; n = n + k^2;  end
    if btype == "R"; n = n + k; end
    if btype == "S"; n = n + k^2; end
  end
  return n
end

"""
Creates a matrix with the same sparsity structure as F
"""
function placeholder(cone_dims)
  num_cones = length(cone_dims)
  B = Block(num_cones);
  for i = 1:num_cones
    (ctype, k) = cone_dims[i]
    if ctype == "R"; B[i] = Diagonal(2*rand(k)); end
    if ctype == "Q"; B[i] = SymWoodbury(Diagonal(3*rand(k)), rand(k), 1.); end
    if ctype == "S"; B[i] = ConicIP.VecCongurance(ConicIP.mat(rand(k)) + LinearAlgebra.I); end
  end
  return B
end

"""
Checks if two sparse matrices have the same sparse structure
"""
function identical_sparse_structure(A::SparseMatrixCSC,B::SparseMatrixCSC)
  if length(A.nzval) != length(B.nzval)
    return false
  end
  if ( all(i -> (A.rowval[i] == B.rowval[i]), 1:length(A.rowval)) &&
       all(i -> (A.colptr[i] == B.colptr[i]), 1:length(A.colptr)) )
    return true
  end
  return false
end

"""
Solves the 3x3 system
```
┌             ┐ ┌    ┐   ┌   ┐
│ Q   G'  -A' │ │ y' │ = │ y │
│ G           │ │ w' │   │ w │
│ A       FᵀF │ │ v' │   │ v │
└             ┘ └    ┘   └   ┘
```

By lifting the large diagonal plus rank 3 blocks of FᵀF

Intelligently chooses between solve3x3gen_sparse_lift and
solve3x3gen_sparse_dense by approximating the number of non-zeros in
both and choosing the form with more sparsity. The former is better
for large second order cones, while the latter is better if the
constraints are the product of many small cones.
"""
function kktsolver_sparse(Q, A, G, cone_dims)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  Q = sparse(Q) # TODO: remove the need for this ?
  A = sparse(A)
  G = sparse(G)

  Fp = placeholder(cone_dims)

  if count_lift(cone_dims) < count_dense(cone_dims)

    # precompute sparse structure and analyze it
    (FᵀFA, FᵀFB, invFᵀFD) = lift(Fp'Fp); r = size(invFᵀFD,1)
    Z = [ Q             G'            -A'            spzeros(n,r)
          G             spzeros(p,p)   spzeros(p,m)  spzeros(p,r)
          A             spzeros(m,p)   FᵀFA          FᵀFB
          spzeros(r,n)  spzeros(r,p)   FᵀFB'         invFᵀFD        ]
    Zᶠ  = lu(Z)
    Z.nzval[:] = 1:length(Z.nzval)
    I₁₁ = round.( Int, Z[n+p+1:n+p+m,n+p+1:n+p+m].nzval )
    I₁₂ = round.( Int, Z[n+p+1:n+p+m,n+p+m+1:end].nzval )
    I₂₁ = round.( Int, Z[n+p+m+1:end,n+p+1:n+p+m].nzval )
    I₂₂ = round.( Int, Z[n+p+m+1:end,n+p+m+1:end].nzval )

    function solve3x3gen_lift(F, F⁻ᵀ)

      (FᵀFA, FᵀFB, invFᵀFD) = lift(F'F); r = size(invFᵀFD,1)
      # In the first iteration, FᵀF is the identity. This
      # detects that
      if r == 0
        Z₀ = [ Q        G'             -A'
               G        spzeros(p,p)   spzeros(p,m)
               A        spzeros(m,p)   FᵀFA         ]
        Z₀ᶠ = lu(Z₀)
        function solve3x3I(Δy, Δw, Δv)
          z = Z₀ᶠ\[Δy; Δw; Δv]
          return (z[1:n,:], z[n+1:n+p,:], z[n+p+1:end,:])
        end
        return solve3x3I
      else
        # If the sparsity structure is the same, you can reuse
        # the symbolic factorization.
        Zᶠ_lu = Zᶠ
        # Build a fresh sparse matrix with the updated values
        Z_new = [ Q             G'            -A'            spzeros(n,r)
                  G             spzeros(p,p)   spzeros(p,m)  spzeros(p,r)
                  A             spzeros(m,p)   FᵀFA          FᵀFB
                  spzeros(r,n)  spzeros(r,p)   FᵀFB'         invFᵀFD        ]
        Zᶠ_lu = lu(Z_new)
        function solve3x3lift(Δy, Δw, Δv)
          z = Zᶠ_lu\[Δy; Δw; Δv; zeros(r,1)]
          return (z[1:n,:], z[n+1:n+p,:], z[(n+p+1):(n+m+p),:])
        end
        return solve3x3lift
      end
    end

    return solve3x3gen_lift

  else

    # Compute sparse structure and analyze it
    FᵀFp = sparse(Fp'Fp)
    Z = [ Q        G'             -A'
          G        spzeros(p,p)   spzeros(p,m)
          A        spzeros(m,p)   FᵀFp         ]

    function solve3x3gen_nolift(F, F⁻ᵀ)

      FᵀF = sparse(F'F)

      Z₀ = [ Q        G'             -A'
              G        spzeros(p,p)   spzeros(p,m)
              A        spzeros(m,p)   FᵀF          ]
      Z₀ᶠ = lu(Z₀)
      function solve3x3_nolift(Δy, Δw, Δv)
        z = Z₀ᶠ\[Δy; Δw; Δv]
        return (z[1:n,:], z[n+1:n+p,:], z[n+p+1:end,:])
      end
      return solve3x3_nolift

    end

    return solve3x3gen_nolift

  end

end

"""
Solves the 2x2 system
```
┌                   ┐ ┌    ┐   ┌   ┐
│ Q + A'F⁻¹F⁻ᵀA  G' │ │ y' │ = │ y │
│ G                 │ │ w' │   │ w │
└                   ┘ └    ┘   └   ┘
```
"""
function kktsolver_2x2(Q, A, G, cone_dims)

  n = size(Q,1) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  function solve2x2gen(F, F⁻ᵀ)

    F⁻ᵀ = sparse(F⁻ᵀ)
    AᵀF⁻¹F⁻ᵀA = A'*(F⁻ᵀ'*(F⁻ᵀ*A))

    Z = [ Q + AᵀF⁻¹F⁻ᵀA   G'
          G               spzeros(p,p) ]

    Z = lu(Z)

    function solve2x2(Δy, Δw)

      z = Z\[Δy; Δw]
      return (z[1:n,:], z[n+1:end,:])

    end

    return solve2x2

  end

  return solve2x2gen

end

"""
Wrapper around solve2xegen to solve 3x3 systems by pivoting
on the third component.
"""
function pivotgen(kktsolver_2x2,Q,A,G,cone_dims)

  solve2x2gen = kktsolver_2x2(Q,A,G,cone_dims)

  function solve3x3gen(F, F⁻ᵀ)

    solve2x2 = solve2x2gen(F, F⁻ᵀ)

    function solve3x3(y, w, v)

      t1 = F⁻ᵀ*(F⁻ᵀ*v)
      (Δy, Δw) = solve2x2(y + A'*t1, w)
      axpy!(-1, F⁻ᵀ*(F⁻ᵀ*(A*Δy)), t1)  # Δv = F⁻²*(v - A*Δy)

      return(Δy, Δw, t1)

    end

  end

  return solve3x3gen

end

pivot(kktsolver_2x2) = (Q,A,G,cone_dims) -> pivotgen(kktsolver_2x2,Q,A,G,cone_dims)
