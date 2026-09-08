# ──────────────────────────────────────────────────────────────
#  Various KKT Solvers
# ──────────────────────────────────────────────────────────────

# Structural nonzero count, independent of storage type (MOI hands over
# SparseMatrixCSC even for effectively dense data, so never classify by type).
_structural_nnz(M::AbstractSparseMatrix) = nnz(M)
_structural_nnz(M::Diagonal)             = count(!iszero, M.diag)
_structural_nnz(M::AbstractMatrix)       = count(!iszero, M)

"""
    dense_kkt_bytes(n, m, p)

Routing estimate of the *persistent* dense arrays [`kktsolver_qr`](@ref)
holds at once, used by [`choose_kktsolver`](@ref) to keep hopeless sizes
off the dense path: the n×n orthogonal factor and the n×p dense copy of
`Gᵀ` at setup, the m×(n−p) `A*Q2` and the (n−p)² reduced Hessian, plus
one more of each per iteration (`W` and `Lmat`). Float64 throughout.

This is not a peak-memory bound: LAPACK workspaces, the temporaries of
`F.Q * I` and `Q2' * (Q * Q2)`, the Cholesky (or fallback QR) copy of
the reduced Hessian, and any fill in the caller's own matrices are all
excluded. Peak usage can be a small multiple of this figure.
"""
dense_kkt_bytes(n, m, p) = 8 * (2n^2 + 2m*max(n - p, 0) + 2max(n - p, 0)^2)

"""
    dense_kkt_flops(n, m, p; nnzA = 0, nnzQ = 0, iters = 25)

Per-iteration flop estimate for [`kktsolver_qr`](@ref), comparable with
the per-factorization estimate `Σⱼ nnz(L₍:,ⱼ₎)²` used for
[`kktsolver_ldl`](@ref). With `r = n − p` it counts

- per iteration: `W = F⁻ᵀ(AQ2)` (`2mr`), the reduced Hessian
  `S22 + WᵀW` (`mr²`) and its Cholesky (`r³/3`), plus three solves
  (predictor, corrector, one refinement) that each apply `Q1`, `Q1ᵀ`,
  `Q2`, `Q2ᵀ` (`4np + 4nr`), the Cholesky and `R1` triangles (`2r² + 2p²`)
  and two `Hmul` products (`2(nnzQ + 2nnzA)`);
- setup, amortized over `iters` iterations: the Householder QR of `Gᵀ`
  (`2np² − 2p³/3`), materializing the n×n `Q0 = F.Q * I` (`4n²p`),
  `A*Q2` (`2·nnzA·r`) and `Q2' * (Q * Q2)` (`2·nnzQ·r + 2nr²`).

The setup terms matter when equalities nearly determine the variables:
at `p = n` the per-iteration terms vanish but the n×n QR and the dense
`Q0` are still paid, so the old model (`mr² + r³/3` alone) reported zero
cost and routed `n = m = p` problems to the dense solver.

`nnzA` and `nnzQ` are structural nonzero counts; when omitted those
terms are dropped, which only makes the estimate optimistic for the
dense path.
"""
function dense_kkt_flops(n, m, p; nnzA = 0, nnzQ = 0, iters = 25)
  p > n && return Inf # QR requires independent equality rows.
  r = n - p
  iters = max(iters, 1)
  # Per iteration: NT-scaled constraint block, reduced Hessian, Cholesky.
  per_iter = 2m*r + m*r^2 + r^3/3
  # Per solve (nominally three per iteration): the orthogonal applies are
  # dense n×p and n×r products, whatever the sparsity of the data.
  per_solve = 4n*p + 4n*r + 2r^2 + 2p^2 + 2(nnzQ + 2nnzA)
  # Setup, paid once: QR of Gᵀ, the dense Q0, A*Q2 and Q2'(Q Q2).
  setup = (2n*p^2 - 2p^3/3) + 4n^2*p + 2nnzA*r + 2nnzQ*r + 2n*r^2
  return per_iter + 3per_solve + setup/iters
end

# Per-iteration flop estimate for kktsolver_ldl from the symbolic
# factorization: Σⱼ (column count of L)², the cost of the rank-1 updates.
# The column counts come from the elimination tree alone (O(nnz + N) time
# and memory); a logical `qdldl` would also allocate the pattern of L,
# which for a badly filling matrix is gigabytes spent on an estimate.
function _ldl_flops(pat)
  K = pat.K
  Kp = pat.perm === nothing ? K :
       first(QDLDL.permute_symmetric(K, invperm(pat.perm)))
  N = size(Kp, 1)
  Lnz = zeros(Int, N); etree = zeros(Int, N); work = zeros(Int, N)
  s = QDLDL.QDLDL_etree!(N, Kp.colptr, Kp.rowval, work, Lnz, etree)
  s < 0 && return Inf                    # structurally deficient diagonal
  return sum(abs2, Float64.(Lnz))
end

"""
    choose_kktsolver(Q, A, G, cone_dims;
                     size_min = 200, dense_bytes_max = 4 * 2^30,
                     ldl_flop_weight = 10.0)

Pick a KKT solver from the problem's cone mix, size, and predicted
factorization cost (issue #10). Returns one of the solver constructors,
chosen by:

0. the dense solver's storage estimate [`dense_kkt_bytes`](@ref) above
   `dense_bytes_max` ⇒ [`kktsolver_ldl`](@ref), whatever the rules
   below would say — dense QR at that size is an out-of-memory error,
   not a slow solve. If the problem also has a semidefinite cone an
   `ArgumentError` is thrown instead: the sparse solver materializes
   each k×k SDP scaling block as a dense k²×k² block, so it would trade
   one huge allocation for another. Such problems currently need the
   dense path; raise the budget explicitly with
   `kktsolver = (Q, A, G, cd) -> choose_kktsolver(Q, A, G, cd; dense_bytes_max = …)(Q, A, G, cd)`
   or pass `kktsolver = kktsolver_qr` directly;
1. any SDP cone ⇒ [`kktsolver_qr`](@ref) — the dense double-QR method is
   the numerically robust choice for the dense SDP scaling blocks, and
   the sparse solver's SDP path is dense in `k(k+1)/2`;
2. `n + m + p < size_min` ⇒ `kktsolver_qr` — dense factorization wins at
   small sizes (the symbolic analysis below would cost more than it
   saves there);
3. otherwise the two per-iteration flop estimates decide:
   `kktsolver_qr` if [`dense_kkt_flops`](@ref) is below
   `ldl_flop_weight` times the LDLᵀ estimate `Σⱼ nnz(L₍:,ⱼ₎)²` taken from
   a symbolic analysis of the quasi-definite KKT pattern, else
   `kktsolver_ldl`. The weight accounts for the dense path running in
   BLAS and the LDLᵀ in scalar code; 10 reproduces the measured
   crossover on the benchmark set (many small SOCs over a 10%-dense `A`
   go dense, banded and block-structured problems go sparse).

[`kktsolver_sparse`](@ref) (UMFPACK LU) is no longer selected
automatically; it remains available explicitly.
"""
choose_kktsolver(Q, A, G, cone_dims; kw...) =
  _choose_kktsolver(Q, A, G, cone_dims; kw...)[1]

function _choose_kktsolver(Q, A, G, cone_dims;
                           size_min = 200, dense_bytes_max = 4 * 2^30,
                           ldl_flop_weight = 10.0)
  n = size(Q,1); m = size(A,1); p = size(G,1)
  has_sdp = any(cd -> isequal(cd[1], "S"), cone_dims)
  bytes = dense_kkt_bytes(n, m, p)
  if bytes > dense_bytes_max
    if has_sdp
      GiB = 2.0^30
      throw(ArgumentError(
        "choose_kktsolver: the dense KKT solver needs an estimated " *
        "$(round(bytes / GiB; sigdigits = 3)) GiB of persistent storage " *
        "(n = $n, m = $m, p = $p), above the dense_bytes_max budget of " *
        "$(round(dense_bytes_max / GiB; sigdigits = 3)) GiB, and the problem " *
        "has semidefinite cones, which the sparse LDLᵀ solver would " *
        "materialize as dense k²×k² blocks (O(k⁴) memory per block). " *
        "Semidefinite problems currently need the dense path: raise the " *
        "budget with `kktsolver = (Q, A, G, cd) -> choose_kktsolver(Q, A, G, cd; " *
        "dense_bytes_max = <bytes>)(Q, A, G, cd)`, or pass " *
        "`kktsolver = kktsolver_qr` directly."))
    end
    return (kktsolver_ldl, nothing)
  end
  if has_sdp
    return (kktsolver_qr, nothing)
  end
  # Dense QR cannot factor an overdetermined equality block, regardless
  # of its predicted cost. LDL regularizes dependent equality rows.
  p > n && return (kktsolver_ldl, nothing)
  if n + m + p < size_min
    return (kktsolver_qr, nothing)
  end
  pat = _ldl_pattern(Q, A, G, cone_dims)
  flops_qr = dense_kkt_flops(n, m, p; nnzA = _structural_nnz(A),
                             nnzQ = _structural_nnz(Q))
  if flops_qr < ldl_flop_weight * _ldl_flops(pat)
    return (kktsolver_qr, nothing)
  end
  return (kktsolver_ldl, pat)
end

"""
    kkt_diagnostics(solve3x3) -> diagnostics or nothing

Hook through which a KKT solver reports per-factorization diagnostics to
`conicIP`. `solve3x3` is the object a solver's `solve3x3gen(F, F⁻ᵀ)`
returned; the default answers `nothing` (no diagnostics). A solver that
returns an object with integer fields `repaired` and `refactors` (for the
current factorization) and `repaired_total` and `refactors_total` (summed
over the solve) has the former printed in the verbose `kkt` column as
`repaired/refactors` and the latter stored in `Solution.kkt_repaired`
and `Solution.kkt_refactors`. [`kktsolver_ldl`](@ref) implements it with
`LDLDiagnostics`. Not exported.
"""
kkt_diagnostics(::Any) = nothing

"""
    default_kktsolver(Q, A, G, cone_dims)

The default `kktsolver` for [`conicIP`](@ref): dispatches to the solver
picked by [`choose_kktsolver`](@ref), reusing the KKT pattern the choice
analysed when the answer is [`kktsolver_ldl`](@ref). Satisfies the
standard kktsolver interface, so it can be passed anywhere a concrete
solver can.
"""
function default_kktsolver(Q, A, G, cone_dims)
  (ks, pat) = _choose_kktsolver(Q, A, G, cone_dims)
  return pat === nothing ? ks(Q, A, G, cone_dims) :
                           kktsolver_ldl(Q, A, G, cone_dims; pattern = pat)
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

  Q = sparse(Q); A = sparse(A); G = sparse(G)

  # Symbolic-factorization reuse, as in kktsolver_sparse: the Schur
  # complement's pattern is fixed once the scaling has left the identity.
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

  function solve2x2gen(F, F⁻ᵀ)

    F⁻ᵀ = sparse(F⁻ᵀ)
    AᵀF⁻¹F⁻ᵀA = A'*(F⁻ᵀ'*(F⁻ᵀ*A))

    Z = [ Q + AᵀF⁻¹F⁻ᵀA   G'
          G               spzeros(p,p) ]

    Zᶠ = factor!(Z)

    function solve2x2(Δy, Δw)

      z = Zᶠ\[Δy; Δw]
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
