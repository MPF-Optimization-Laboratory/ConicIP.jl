# ──────────────────────────────────────────────────────────────
#  SDP Problem Generators
#
#  Each generator returns a NamedTuple with fields:
#    Q, c, A, b, cone_dims       — required (direct API)
#    G, d                        — optional equality constraints
#    known_status                 — expected :Optimal / :Infeasible / :Unbounded
#    known_obj                   — expected objective (nothing if N/A)
#    known_X                     — expected primal PSD matrix (nothing if N/A)
#    description                 — human-readable description
# ──────────────────────────────────────────────────────────────

using LinearAlgebra, SparseArrays

# Helpers
_vecm(Z) = ConicIP.vecm(Z)
_mat(x)  = ConicIP.mat(x)
_vdim(n) = div(n * (n + 1), 2)   # vectorized dimension for n×n matrix

"""
    sdp_psd_projection(; n=6)

Project a diagonal matrix with mixed signs onto the PSD cone.
Known solution: clip negative eigenvalues to zero.
"""
function sdp_psd_projection(; n=6)
    eigs = vcat(ones(div(n, 2)), -ones(n - div(n, 2)))
    target = diagm(0 => eigs)
    expected = diagm(0 => max.(eigs, 0.0))

    k = _vdim(n)
    Q = Matrix{Float64}(I, k, k)
    c = reshape(_vecm(target), :, 1)
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    return (Q=Q, c=c, A=A, b=b, cone_dims=[("S", k)],
            G=spzeros(0, k), d=zeros(0, 1),
            known_status=:Optimal, known_obj=nothing,
            known_X=expected,
            description="PSD projection (n=$n): clip negative eigenvalues")
end

"""
    sdp_trace_minimization(; n=4)

Minimize tr(X) subject to X ≽ C where C is a fixed PSD matrix.
Known solution: X* = C, obj = tr(C).
"""
function sdp_trace_minimization(; n=4)
    # C = random PSD matrix
    Random.seed!(42)
    M = randn(n, n)
    C = M' * M / n   # well-conditioned PSD

    k = _vdim(n)

    # min tr(X) = min vecm(I)' * x  subject to  x ≽ vecm(C)
    # In ConicIP form: min ½x'Qx - c'x with Q=0
    # Cone constraint: A*y ≥ b  →  I*y ≥ vecm(C)
    # But ConicIP minimizes ½y'Qy - c'y, so to minimize vecm(I)'y
    # we set c = -vecm(I) (since it becomes -(-vecm(I))'y = vecm(I)'y)
    Q = spzeros(k, k)
    c_vec = -reshape(_vecm(Matrix{Float64}(I, n, n)), :, 1)
    A = sparse(1.0I, k, k)
    b = reshape(_vecm(C), :, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=spzeros(0, k), d=zeros(0, 1),
            known_status=:Optimal, known_obj=tr(C),
            known_X=C,
            description="Trace minimization: min tr(X) s.t. X ≽ C (n=$n)")
end

"""
    sdp_max_cut_relaxation(; n=5, seed=123)

Goemans-Williamson max-cut SDP relaxation for a random graph.
  max  ¼ tr(L * X)  s.t.  diag(X) = 1, X ≽ 0
where L is the graph Laplacian.

Reformulated for ConicIP's minimization form.
"""
function sdp_max_cut_relaxation(; n=5, seed=123)
    Random.seed!(seed)

    # Random adjacency matrix (symmetric, no self-loops)
    W = zeros(n, n)
    for i = 1:n, j = i+1:n
        w = rand() > 0.4 ? rand() : 0.0
        W[i, j] = w
        W[j, i] = w
    end
    L = diagm(0 => vec(sum(W, dims=2))) - W

    k = _vdim(n)

    # Maximize ¼ tr(L*X) → minimize -¼ tr(L*X) = -¼ vecm(L)'x
    # ConicIP: min ½y'Qy - c'y  →  c = ¼ vecm(L) (negated in the objective)
    Q = spzeros(k, k)
    c_vec = reshape(0.25 * _vecm(L), :, 1)

    # Constraint: diag(X) = 1  →  n equality constraints
    # Extract diagonal entries from vectorized form
    G = spzeros(n, k)
    d = ones(n, 1)
    for i = 1:n
        # Position of diagonal entry (i,i) in vecm ordering
        # vecm uses row-major upper triangular: entries for row i start at
        # position sum_{r=1}^{i-1}(n-r+1) + 1 = i + (i-1)*(2n-i)/2
        pos = round(Int, k - (n - i + 2) * (n - i + 1) / 2 + 1)
        G[i, pos] = 1.0
    end

    # X ≽ 0: cone constraint
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=G, d=d,
            known_status=:Optimal, known_obj=nothing,
            known_X=nothing,
            description="Max-cut SDP relaxation (n=$n)")
end

"""
    sdp_lovasz_theta(; n=5, seed=456)

Lovász theta number of a random graph.
  max  tr(J * X)  s.t.  tr(X) = 1, X_ij = 0 for (i,j) ∈ E, X ≽ 0
where J is the all-ones matrix.

The theta number upper-bounds the independence number.
"""
function sdp_lovasz_theta(; n=5, seed=456)
    Random.seed!(seed)

    # Random graph with ~40% edge density
    edges = Tuple{Int,Int}[]
    for i = 1:n, j = i+1:n
        if rand() < 0.4
            push!(edges, (i, j))
        end
    end

    k = _vdim(n)

    # Maximize tr(J*X) → minimize -tr(J*X) = -vecm(J)'x
    # ConicIP: c = vecm(J)
    J = ones(n, n)
    Q = spzeros(k, k)
    c_vec = reshape(_vecm(J), :, 1)

    # Equality constraints:
    # 1. tr(X) = 1
    # 2. X_ij = 0 for each edge (i,j)
    n_eq = 1 + length(edges)
    G = spzeros(n_eq, k)
    d = zeros(n_eq, 1)

    # tr(X) = 1: sum of diagonal entries
    I_mat = Matrix{Float64}(I, n, n)
    G[1, :] = _vecm(I_mat)'
    d[1] = 1.0

    # X_ij = 0 for each edge
    for (idx, (i, j)) in enumerate(edges)
        E_ij = zeros(n, n)
        E_ij[i, j] = 1.0
        E_ij[j, i] = 1.0
        G[1 + idx, :] = _vecm(E_ij)'
        d[1 + idx] = 0.0
    end

    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=sparse(G), d=d,
            known_status=:Optimal, known_obj=nothing,
            known_X=nothing,
            description="Lovász theta (n=$n, |E|=$(length(edges)))")
end

"""
    sdp_nearest_correlation(; n=5, seed=789)

Find the nearest correlation matrix (PSD + unit diagonal) to a given
symmetric matrix.

  min  ‖X - C‖²_F  s.t.  diag(X) = 1, X ≽ 0

This is a convex QP with SDP constraint.
"""
function sdp_nearest_correlation(; n=5, seed=789)
    Random.seed!(seed)

    # Create a symmetric matrix that is NOT a correlation matrix
    M = randn(n, n)
    C = (M + M') / 2
    # Normalize to have unit diagonal
    D = diagm(0 => 1.0 ./ sqrt.(abs.(diag(C)) .+ 1.0))
    C = D * C * D
    # Perturb to make it non-PSD
    C = C - 0.5I

    k = _vdim(n)

    # min ½‖x - vecm(C)‖² = ½x'Ix - vecm(C)'x + const
    Q = Matrix{Float64}(I, k, k)
    c_vec = reshape(_vecm(C), :, 1)

    # diag(X) = 1: equality constraints
    G = spzeros(n, k)
    d = ones(n, 1)
    for i = 1:n
        pos = round(Int, k - (n - i + 2) * (n - i + 1) / 2 + 1)
        G[i, pos] = 1.0
    end

    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=G, d=d,
            known_status=:Optimal, known_obj=nothing,
            known_X=nothing,
            description="Nearest correlation matrix (n=$n)")
end

"""
    sdp_multiple_blocks(; n1=3, n2=4)

Problem with two SDP blocks: min tr(X₁) + tr(X₂) s.t. X₁ ≽ C₁, X₂ ≽ C₂.
Tests that the Block type correctly handles multiple VecCongurance entries.
"""
function sdp_multiple_blocks(; n1=3, n2=4)
    Random.seed!(99)
    M1 = randn(n1, n1); C1 = M1' * M1 / n1
    M2 = randn(n2, n2); C2 = M2' * M2 / n2

    k1 = _vdim(n1)
    k2 = _vdim(n2)
    k  = k1 + k2

    Q = spzeros(k, k)
    I1_vec = _vecm(Matrix{Float64}(I, n1, n1))
    I2_vec = _vecm(Matrix{Float64}(I, n2, n2))
    c_vec  = -vcat(reshape(I1_vec, :, 1), reshape(I2_vec, :, 1))

    A = sparse(1.0I, k, k)
    b = vcat(reshape(_vecm(C1), :, 1), reshape(_vecm(C2), :, 1))

    return (Q=Q, c=c_vec, A=A, b=b,
            cone_dims=[("S", k1), ("S", k2)],
            G=spzeros(0, k), d=zeros(0, 1),
            known_status=:Optimal, known_obj=tr(C1) + tr(C2),
            known_X=nothing,
            description="Multiple SDP blocks (n1=$n1, n2=$n2)")
end

"""
    sdp_mixed_cones(; n_r=4, n_q=3, n_s=3)

Problem with R+, SOC, and SDP cones simultaneously.
min ½‖y‖² - c'y  s.t.  y_R ≥ 0, y_Q ∈ SOC, y_S ∈ PSD

Tests heterogeneous Block with Diagonal, SymWoodbury, and VecCongurance.
"""
function sdp_mixed_cones(; n_r=4, n_q=3, n_s=3)
    Random.seed!(77)
    k_s = _vdim(n_s)
    n = n_r + (n_q + 1) + k_s   # total variables

    Q = sparse(1.0I, n, n)
    c_target = randn(n, 1)
    c = Q * c_target   # so optimal is projection of c_target onto feasible set

    # A = I, b = 0 — project onto cone product
    A = sparse(1.0I, n, n)
    b = zeros(n, 1)

    cone_dims = [("R", n_r), ("Q", n_q + 1), ("S", k_s)]

    return (Q=Q, c=c, A=A, b=b, cone_dims=cone_dims,
            G=spzeros(0, n), d=zeros(0, 1),
            known_status=:Optimal, known_obj=nothing,
            known_X=nothing,
            description="Mixed cones: R($n_r) + Q($(n_q+1)) + S($k_s)")
end

"""
    sdp_with_equality(; n=4)

SDP with equality constraints: min tr(C*X) s.t. tr(Aᵢ*X) = bᵢ, X ≽ 0.
Standard dual-form SDP used in textbooks.
"""
function sdp_with_equality(; n=4)
    Random.seed!(321)

    k = _vdim(n)

    # Objective: min tr(C*X) where C is random symmetric
    M = randn(n, n)
    C = (M + M') / 2
    Q = spzeros(k, k)
    c_vec = reshape(_vecm(C), :, 1)

    # Equality constraints: tr(A_i * X) = b_i
    # Use 3 random symmetric constraint matrices
    n_eq = 3
    G = zeros(n_eq, k)
    d = zeros(n_eq, 1)

    # Build a strictly feasible X0 to ensure feasibility
    X0 = Matrix{Float64}(I, n, n) + 0.1 * randn(n, n)
    X0 = X0' * X0   # PSD and strictly feasible

    for i = 1:n_eq
        Ai = randn(n, n)
        Ai = (Ai + Ai') / 2
        G[i, :] = _vecm(Ai)'
        d[i] = tr(Ai * X0)   # feasible at X0
    end

    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=sparse(G), d=reshape(d, :, 1),
            known_status=:Optimal, known_obj=nothing,
            known_X=nothing,
            description="SDP with equality constraints (n=$n, m=$n_eq)")
end

"""
    sdp_larger(; n=10)

Larger SDP to test scaling behavior: PSD projection of a matrix with
known eigenvalues.
"""
function sdp_larger(; n=10)
    Random.seed!(555)

    # Create matrix with known eigenstructure
    U, _ = qr(randn(n, n))
    U = Matrix(U)
    eigs = collect(range(-1.0, 2.0, length=n))
    target = U * diagm(0 => eigs) * U'
    target = (target + target') / 2
    expected = U * diagm(0 => max.(eigs, 0.0)) * U'
    expected = (expected + expected') / 2

    k = _vdim(n)
    Q = Matrix{Float64}(I, k, k)
    c_vec = reshape(_vecm(target), :, 1)
    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=spzeros(0, k), d=zeros(0, 1),
            known_status=:Optimal, known_obj=nothing,
            known_X=expected,
            description="Larger PSD projection (n=$n)")
end

"""
    sdp_identity_feasible(; n=3)

Trivially feasible SDP: minimize 0 s.t. X ≽ I.
Solution: X = I.
"""
function sdp_identity_feasible(; n=3)
    k = _vdim(n)
    Q = spzeros(k, k)
    c_vec = zeros(k, 1)
    A = sparse(1.0I, k, k)
    b = reshape(_vecm(Matrix{Float64}(I, n, n)), :, 1)

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=spzeros(0, k), d=zeros(0, 1),
            known_status=:Optimal, known_obj=nothing,
            known_X=Matrix{Float64}(I, n, n),
            description="Trivially feasible: min 0 s.t. X ≽ I (n=$n)")
end

"""
    sdp_rank_one(; n=4)

SDP with rank-1 solution: min tr(C*X) s.t. tr(X) = 1, X ≽ 0.
Solution is X = vv' where v is eigenvector of smallest eigenvalue of C.
"""
function sdp_rank_one(; n=4)
    Random.seed!(111)
    M = randn(n, n)
    C = (M + M') / 2

    k = _vdim(n)
    Q = spzeros(k, k)
    c_vec = reshape(_vecm(C), :, 1)

    # Equality: tr(X) = 1
    I_mat = Matrix{Float64}(I, n, n)
    G = reshape(_vecm(I_mat)', 1, k)
    d = ones(1, 1)

    A = sparse(1.0I, k, k)
    b = zeros(k, 1)

    # Known solution: eigenvector of minimum eigenvalue
    λ, V = eigen(Symmetric(C))
    v = V[:, 1]
    known_obj = λ[1]  # minimum eigenvalue = objective

    return (Q=Q, c=c_vec, A=A, b=b, cone_dims=[("S", k)],
            G=sparse(G), d=d,
            known_status=:Optimal, known_obj=known_obj,
            known_X=v * v',
            description="Rank-1 SDP: min tr(C*X) s.t. tr(X)=1 (n=$n)")
end
