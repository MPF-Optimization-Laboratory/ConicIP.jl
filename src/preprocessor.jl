"""
  imcols(A, b; ϵ = 1e-10)

Removes redundant inequalities in a system of equations

Ax = b

and checks if the equations are consistent.
"""
function imcols(A, b, ϵ = 1e-8)

  A = sparse(A)
  nA = norm(A); A = A/nA; b = b/nA

  if isempty(A); return ([], true); end

  F = qr(sparse(A'))
  R_mat = F.R
  n_r = min(size(R_mat)...)
  diag_R = [abs(R_mat[i,i]) for i in 1:n_r]
  piv = F.pcol  # column permutation
  R = sort(piv[findall(diag_R .> ϵ)])

  if isempty(R); return ([], true); end

  return (norm(A*(A[R,:]\b[R]) - b, Inf) < ϵ) ? (R, true) : ([], false)

end

"""
ConicIP with preprocessing to ensure the following
rank constraints

Primal equailty constraints : Gx = d
Rank condition              : rank(G) = size(G,1)

Dual equality constraints   : [ Q A' G'] = c
Rank condition              : rank([Q A' G']) = size(Q,1)
"""
function preprocess_conicIP(Q, c::AbstractVector,
  A, b::AbstractVector, cone_dims,
  G = spzeros(0,length(c)), d = zeros(0);
  verbose = false,
  options...)

  if verbose == true
    println()
    println(" > INTERIOR POINT PREPROCESSOR v0.7.1 (Aug 2016)")
    println()
  end

  n = length(c) # Number of variables
  m = size(A,1) # Number of inequality constraints
  p = size(G,1) # Number of equality constraints

  normd = isempty(d) ? -Inf : norm(d)

  (IP, pconsistent) = imcols(G, d)
  (ID, dconsistent) = imcols([Q A' G[IP,:]'], c)

  if !(pconsistent && dconsistent)
    return ConicIP.Solution(zeros(n)/0, zeros(p)/0,zeros(m)/0,
      :Infeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN)
  end

  if (verbose == true) && (length(IP) != p)
    println("   - Removing $(p - length(IP)) redundant primal constraints ");
  end

  if (verbose == true) && (length(ID) != n)
    println("   - Augmenting $(n - length(ID) ) dual constraints");
  end

  if (verbose == true) &&  (length(ID) == n) && (length(IP) == p)
    println("   - No changes made")
  end

  z = ones(n); z[ID] .= 0; Z = spdiagm(0 => z)

       # Augmented constraints
       #             |
  sol = conicIP(Q + Z, c, A, b, cone_dims, G[IP,:], d[IP];
    verbose = verbose,         #                   |
    options...)                # Removed redundant linear constraints
                               # TODO : (use view?)


  # Argument the dual variables with 0's corresponding to the redundant
  # constraints

  w = zeros(size(G,1)); w[IP] = sol.w; sol.w = w

  return sol


end
