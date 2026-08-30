# ──────────────────────────────────────────────────────────────
#  Various KKT Solvers
# ──────────────────────────────────────────────────────────────

# Structural nonzero count, independent of storage type (MOI hands over
# SparseMatrixCSC even for effectively dense data, so never classify by type).
_structural_nnz(M::AbstractSparseMatrix) = nnz(M)
_structural_nnz(M::Diagonal)             = count(!iszero, M.diag)
_structural_nnz(M::AbstractMatrix)       = count(!iszero, M)

"""
    choose_kktsolver(Q, A, G, cone_dims; nnz_per_col_max = 10, size_min = 1000)

Pick a KKT solver from the problem's cone mix, size, and sparsity
(issue #10). Returns one of the solver constructors, chosen by:

1. any SDP cone ⇒ [`kktsolver_qr`](@ref) — the dense double-QR method is
   the numerically robust choice for the dense SDP scaling blocks;
2. `n + m + p < size_min` ⇒ `kktsolver_qr` — dense factorization wins at
   small sizes and matches the historical default exactly;
3. average structural nonzeros per column of `[Q; A; G]` above
   `nnz_per_col_max` ⇒ `kktsolver_qr` — sparse-typed but dense-ish data
   (e.g. many small SOCs with a 10%-dense `A`) factors faster densely;
4. otherwise ⇒ [`kktsolver_sparse`](@ref).

The decision is by *structural* nonzero counts, never by storage type.
"""
function choose_kktsolver(Q, A, G, cone_dims;
                          nnz_per_col_max = 10, size_min = 1000)
  if any(cd[1] == "S" for cd in cone_dims)
    return kktsolver_qr
  end
  n = size(Q,1)
  if n + size(A,1) + size(G,1) < size_min
    return kktsolver_qr
  end
  total = _structural_nnz(Q) + _structural_nnz(A) + _structural_nnz(G)
  if total > nnz_per_col_max * n
    return kktsolver_qr
  end
  return kktsolver_sparse
end

"""
    default_kktsolver(Q, A, G, cone_dims)

The default `kktsolver` for [`conicIP`](@ref): dispatches to the solver
picked by [`choose_kktsolver`](@ref). Satisfies the standard kktsolver
interface, so it can be passed anywhere a concrete solver can.
"""
default_kktsolver(Q, A, G, cone_dims) =
  choose_kktsolver(Q, A, G, cone_dims)(Q, A, G, cone_dims)

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

  if p > n
    throw(ArgumentError(
      "kktsolver_qr requires p ≤ n (got p = $p equality rows, n = $n " *
      "variables): G must have independent rows. Remove redundant rows " *
      "first, e.g. via preprocess_conicIP."))
  end

  # Setup (once): thin QR of G' gives G = R1'Q1' with Q0 = [Q1 Q2]
  # orthogonal. Only setup materializes an n×n dense matrix; the
  # per-iteration and per-solve work below never densifies the m×m NT
  # block or the m×n constraint matrix (issue #10).
  F = qr(Matrix(G'))
  Q0 = F.Q * Matrix{Float64}(LinearAlgebra.I, n, n)
  R1 = F.R
  Q1 = @view Q0[:, 1:p]
  Q2 = @view Q0[:, p+1:end]

  # Constants across iterations
  AQ2 = A * Q2           # m×(n−p) dense
  S22 = Q2' * (Q * Q2)   # (n−p)×(n−p), the Q part of the reduced Hessian

  function solve3x3gen(F, F⁻ᵀ)

    # Reduced Hessian Q2'(Q + A'F⁻¹F⁻ᵀA)Q2 = S22 + W'W, W = F⁻ᵀ(AQ2).
    # F⁻ᵀ is the cached block-diagonal inverse the caller passes in —
    # never densify it (it used to be shadowed by a dense m×m inverse).
    W = F⁻ᵀ * AQ2
    Lmat = S22 + W'W
    L = try
      cholesky(Symmetric(Lmat))   # SPD by construction (CVXOPT §10.2)
    catch err
      err isa LinearAlgebra.PosDefException || rethrow()
      qr(Lmat)                    # marginal PD: fall back to QR
    end

    # H*u = (Q + A'F⁻¹F⁻ᵀA)u via sparse matvecs and block applies;
    # F⁻¹ = (F⁻ᵀ)' holds for every scaling block (no symmetry assumed).
    F⁻¹ = F⁻ᵀ'
    Hmul(u) = Q*u + A'*(F⁻¹*(F⁻ᵀ*(A*u)))

    function solve3x3(bx, by, bz)

      u1 = R1' \ by                       # G y' = by on range(Q1)
      y1 = Q1 * u1
      t1 = Hmul(y1)
      g  = bx + A'*(F⁻¹*(F⁻ᵀ*bz))
      u2 = L \ (Q2'*(g - t1))             # reduced system on ker(G)
      y2 = Q2 * u2
      x  = y1 + y2
      y  = R1 \ (Q1'*(g - t1 - Hmul(y2))) # equality duals
      z  = F⁻¹*(F⁻ᵀ*(bz - A*x))           # v' = F⁻¹F⁻ᵀ(bz − Ay')

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

      invD = inv(Blk.D)
      for i = 1:size(Blk.D,1), j = 1:size(Blk.D,2)
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

  Q = sparse(Q)
  A = sparse(A)
  G = sparse(G)

  # Symbolic-factorization reuse: once the NT block's sparsity pattern
  # stabilizes (typically from the second interior-point iteration on),
  # lu! refactorizes numerically inside the cached UMFPACK object,
  # skipping the symbolic analysis. Falls back to a fresh lu whenever
  # the pattern changes (e.g. identity scaling at the initial point, or
  # exact cancellation dropping an entry).
  Zfact = nothing
  Zpat  = nothing
  function factor!(Z)
    if Zfact !== nothing && identical_sparse_structure(Z, Zpat)
      lu!(Zfact, Z)
    else
      Zfact = lu(Z)
      Zpat  = Z
    end
    return Zfact
  end

  # lift() can only represent Diagonal and SymWoodbury blocks; an SDP
  # (VecCongurance) block would contribute nothing and leave a
  # structurally singular system, so any "S" cone forces the no-lift form.
  use_lift = count_lift(cone_dims) < count_dense(cone_dims) &&
             !any(cd[1] == "S" for cd in cone_dims)

  if use_lift

    function solve3x3gen_lift(F, F⁻ᵀ)

      (FᵀFA, FᵀFB, invFᵀFD) = lift(F'F); r = size(invFᵀFD,1)
      # At the initial point FᵀF is the identity (no low-rank part)
      if r == 0
        Z₀ = [ Q        G'             -A'
               G        spzeros(p,p)   spzeros(p,m)
               A        spzeros(m,p)   FᵀFA         ]
        Z₀ᶠ = lu(Z₀)
        function solve3x3I(Δy, Δw, Δv)
          z = Z₀ᶠ\[Δy; Δw; Δv]
          return (z[1:n], z[n+1:n+p], z[n+p+1:end])
        end
        return solve3x3I
      else
        Z = [ Q             G'            -A'            spzeros(n,r)
              G             spzeros(p,p)   spzeros(p,m)  spzeros(p,r)
              A             spzeros(m,p)   FᵀFA          FᵀFB
              spzeros(r,n)  spzeros(r,p)   FᵀFB'         invFᵀFD      ]
        Zᶠ = factor!(Z)
        function solve3x3lift(Δy, Δw, Δv)
          z = Zᶠ\[Δy; Δw; Δv; zeros(r)]
          return (z[1:n], z[n+1:n+p], z[(n+p+1):(n+m+p)])
        end
        return solve3x3lift
      end
    end

    return solve3x3gen_lift

  else

    function solve3x3gen_nolift(F, F⁻ᵀ)

      FᵀF = sparse(F'F)
      Z₀ = [ Q        G'             -A'
             G        spzeros(p,p)   spzeros(p,m)
             A        spzeros(m,p)   FᵀF          ]
      Z₀ᶠ = factor!(Z₀)
      function solve3x3_nolift(Δy, Δw, Δv)
        z = Z₀ᶠ\[Δy; Δw; Δv]
        return (z[1:n], z[n+1:n+p], z[n+p+1:end])
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
      return (z[1:n], z[n+1:end])

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

      # F⁻¹F⁻ᵀ = (F⁻ᵀ)'F⁻ᵀ — the adjoint matters for SDP scaling
      # blocks, which are not self-adjoint.
      t1 = F⁻ᵀ'*(F⁻ᵀ*v)
      (Δy, Δw) = solve2x2(y + A'*t1, w)
      axpy!(-1, F⁻ᵀ'*(F⁻ᵀ*(A*Δy)), t1)  # Δv = F⁻¹F⁻ᵀ*(v - A*Δy)

      return(Δy, Δw, t1)

    end

  end

  return solve3x3gen

end

"""
    pivot(kktsolver_2x2)

Wrap a 2-by-2 KKT solver into a 3-by-3 solver by pivoting on the
third component. The inner solver handles the Schur complement system;
`pivot` reconstructs the full solution.

See also [`conicIP`](@ref) for the KKT solver interface specification.
"""
pivot(kktsolver_2x2) = (Q,A,G,cone_dims) -> pivotgen(kktsolver_2x2,Q,A,G,cone_dims)
