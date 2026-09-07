# SDP problem generators for the semidefinite test suite.
#
# Every generator returns a NamedTuple with the fields
#
#   Q, c, A, b, cone_dims   direct-API problem data
#   G, d                    equality constraints (0×n / empty if unused)
#   known_status            expected solver status
#   known_obj               expected `sol.pobj`   (nothing if unknown)
#   known_X                 expected PSD matrix   (nothing if unknown)
#   description             human-readable label
#
# and optionally
#
#   known_y                 expected primal `sol.y` in full
#   known_V                 expected PSD matrix in the *dual* `sol.v`, for
#                           instances whose matrix variable is the dual
#   unique_y = false        the optimal set is not a singleton, so solvers
#                           may legitimately return different `y`
#   checks                  `"name" => (sol, tol) -> Bool` structural checks
#
# read by the test loop through `get(prob, :field, default)`.
#
# ConicIP solves   min ½y'Qy - c'y   s.t.   Ay ⪰_K b,  Gy = d,
# so an objective `min tr(C*X)` is encoded as `c = -vecm(C)` and then
# `sol.pobj == tr(C*X⋆)`.  The modeled matrix variable is `sol.y`; the cone
# slack is `sol.s = A*y - b`, which coincides with `y` only when `A = I`
# and `b = 0`.  Every generator here uses `A = I` except
# `sdp_affine_eigmax`, whose `A` is a single column and whose matrix
# variable is therefore the dual `sol.v` (reported as `known_V`).
#
# Generators draw from a local RNG (`Random.Xoshiro(seed)`) so that they
# never perturb the global stream shared with the rest of the test suite.

using LinearAlgebra, SparseArrays

# Length of the vectorized form of an n×n symmetric matrix.
_vdim(n) = div(n * (n + 1), 2)

# Index of the diagonal entry (i,i) inside a vecm'd n×n matrix.  vecm
# walks the upper triangle row by row, so row i starts at this position.
# vecm leaves diagonal entries unscaled, so the coefficient there is 1.
_diagpos(n, i) = div(n * (n + 1), 2) - div((n - i + 2) * (n - i + 1), 2) + 1

# Euclidean projection onto the second-order cone {y : y₁ ≥ ‖y₂:ₘ‖}.
function _proj_soc(y)
    t  = y[1]
    z  = y[2:end]
    nz = norm(z)
    nz <= t  && return collect(y)
    nz <= -t && return zeros(length(y))
    return vcat((t + nz) / 2, ((t + nz) / (2 * nz)) .* z)
end

# Euclidean projection onto the PSD cone, in vecm coordinates (vecm is an
# isometry, so clipping the eigenvalues is the projection).
function _proj_psd(x)
    λ, V = eigen(Symmetric(ConicIP.mat(x)))
    return ConicIP.vecm(V * diagm(0 => max.(λ, 0.0)) * V')
end

# Reusable structural checks on the (single) SDP block of a solution.
_chk_psd(k)     = (sol, tol) -> eigmin(Symmetric(ConicIP.mat(sol.y[1:k]))) > -tol
_chk_unitdiag() = (sol, tol) -> all(abs.(diag(ConicIP.mat(sol.y)) .- 1) .< tol)

"""
    sdp_psd_projection(; n=6)

Project a diagonal matrix with mixed signs onto the PSD cone:
`min ½‖X - T‖²_F  s.t.  X ⪰ 0`.  The solution clips the negative
eigenvalues of `T` to zero.
"""
function sdp_psd_projection(; n = 6)
    eigs     = vcat(ones(div(n, 2)), -ones(n - div(n, 2)))
    target   = diagm(0 => eigs)
    expected = diagm(0 => max.(eigs, 0.0))

    k = _vdim(n)
    Q = Matrix{Float64}(I, k, k)
    c = ConicIP.vecm(target)
    A = sparse(1.0I, k, k)
    b = zeros(k)

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = spzeros(0, k), d = zeros(0),
            known_status = :Optimal, known_obj = nothing,
            known_X = expected,
            description = "PSD projection (n=$n): clip negative eigenvalues")
end

"""
    sdp_trace_minimization(; n=4)

`min tr(X)  s.t.  X ⪰ C`, with `C` a random PSD matrix.
Solution: `X⋆ = C`, objective `tr(C)`.
"""
function sdp_trace_minimization(; n = 4)
    rng = Random.Xoshiro(42)
    M = randn(rng, n, n)
    C = M' * M / n

    k = _vdim(n)
    Q = spzeros(k, k)
    c = -ConicIP.vecm(Matrix{Float64}(I, n, n))   # -c'y = tr(X)
    A = sparse(1.0I, k, k)
    b = ConicIP.vecm(C)                            # X - C ⪰ 0

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = spzeros(0, k), d = zeros(0),
            known_status = :Optimal, known_obj = tr(C),
            known_X = C,
            description = "Trace minimization: min tr(X) s.t. X ⪰ C (n=$n)")
end

"""
    sdp_max_cut_relaxation()

Goemans-Williamson max-cut relaxation on a fixed weighted 5-node graph,
`max ¼tr(L*X)  s.t.  diag(X) = 1, X ⪰ 0`, with `L` the Laplacian.
Maximization gives `c = +¼vecm(L)`.  The optimal value is not known in
closed form, but `diag(X) = 1` and `X ⪰ 0` pin the solution structurally
-- the unit-diagonal check in particular fixes the scaling of the `G`
rows.  The graph is hardcoded (rather than drawn) so the instance cannot
change with the RNG stream.
"""
function sdp_max_cut_relaxation()
    n = 5
    W = [0.0 0.8 0.0 0.3 0.0
         0.8 0.0 0.5 0.0 0.2
         0.0 0.5 0.0 0.7 0.0
         0.3 0.0 0.7 0.0 0.6
         0.0 0.2 0.0 0.6 0.0]
    L = diagm(0 => vec(sum(W, dims = 2))) - W

    k = _vdim(n)
    Q = spzeros(k, k)
    c = 0.25 * ConicIP.vecm(L)

    G = spzeros(n, k)
    d = ones(n)
    for i = 1:n
        G[i, _diagpos(n, i)] = 1.0
    end

    A = sparse(1.0I, k, k)
    b = zeros(k)

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = sparse(G), d = d,
            known_status = :Optimal, known_obj = nothing,
            known_X = nothing, unique_y = false,
            checks = ["diag(X) = 1" => _chk_unitdiag(),
                      "X ⪰ 0"       => _chk_psd(k)],
            description = "Max-cut SDP relaxation (n=$n)")
end

"""
    sdp_lovasz_theta_c5()

Lovász theta number of the 5-cycle:
`max tr(J*X)  s.t.  tr(X) = 1, X_ij = 0 for (i,j) ∈ E, X ⪰ 0`.
Lovász's classical result gives `θ(C₅) = √5` exactly.  With `c = vecm(J)`
the solver maximizes `tr(J*X)`, so `sol.pobj = -c'y = -√5`.
"""
function sdp_lovasz_theta_c5()
    n     = 5
    edges = [(1, 2), (2, 3), (3, 4), (4, 5), (5, 1)]

    k = _vdim(n)
    Q = spzeros(k, k)
    c = ConicIP.vecm(ones(n, n))                   # -c'y = -tr(J*X)

    G = spzeros(1 + length(edges), k)
    d = zeros(1 + length(edges))
    G[1, :] = ConicIP.vecm(Matrix{Float64}(I, n, n))'
    d[1]    = 1.0
    for (idx, (i, j)) in enumerate(edges)
        E = zeros(n, n)
        E[i, j] = 1.0
        E[j, i] = 1.0
        G[1+idx, :] = ConicIP.vecm(E)'
    end

    A = sparse(1.0I, k, k)
    b = zeros(k)

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = sparse(G), d = d,
            known_status = :Optimal, known_obj = -sqrt(5.0),
            known_X = nothing, unique_y = false,
            checks = ["X ⪰ 0" => _chk_psd(k)],
            description = "Lovász theta of C₅ (θ = √5)")
end

"""
    sdp_nearest_correlation(; n=5, seed=789)

Nearest correlation matrix: `min ½‖X - C‖²_F  s.t.  diag(X) = 1, X ⪰ 0`,
a QP with an SDP constraint (`Q = I`, `c = vecm(C)`).  Strictly convex,
so the solution is unique.
"""
function sdp_nearest_correlation(; n = 5, seed = 789)
    rng = Random.Xoshiro(seed)

    M = randn(rng, n, n)
    C = (M + M') / 2
    D = diagm(0 => 1.0 ./ sqrt.(abs.(diag(C)) .+ 1.0))
    C = D * C * D - 0.5I                            # perturb away from PSD

    k = _vdim(n)
    Q = Matrix{Float64}(I, k, k)
    c = ConicIP.vecm(C)

    G = spzeros(n, k)
    d = ones(n)
    for i = 1:n
        G[i, _diagpos(n, i)] = 1.0
    end

    A = sparse(1.0I, k, k)
    b = zeros(k)

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = sparse(G), d = d,
            known_status = :Optimal, known_obj = nothing,
            known_X = nothing,
            checks = ["diag(X) = 1" => _chk_unitdiag(),
                      "X ⪰ 0"       => _chk_psd(k)],
            description = "Nearest correlation matrix (n=$n)")
end

"""
    sdp_multiple_blocks(; n1=3, n2=4)

Two SDP blocks: `min tr(X₁) + tr(X₂)  s.t.  X₁ ⪰ C₁, X₂ ⪰ C₂`.
Exercises a `Block` holding several `VecCongurance` scalings.  Since
`tr(Cᵢ + S) = tr(Cᵢ) + tr(S) > tr(Cᵢ)` for any nonzero `S ⪰ 0`, the
solution `Xᵢ⋆ = Cᵢ` is unique.
"""
function sdp_multiple_blocks(; n1 = 3, n2 = 4)
    rng = Random.Xoshiro(99)
    M1 = randn(rng, n1, n1); C1 = M1' * M1 / n1
    M2 = randn(rng, n2, n2); C2 = M2' * M2 / n2

    k1 = _vdim(n1)
    k2 = _vdim(n2)
    k  = k1 + k2

    Q = spzeros(k, k)
    c = -vcat(ConicIP.vecm(Matrix{Float64}(I, n1, n1)),
              ConicIP.vecm(Matrix{Float64}(I, n2, n2)))
    A = sparse(1.0I, k, k)
    b = vcat(ConicIP.vecm(C1), ConicIP.vecm(C2))

    return (Q = Q, c = c, A = A, b = b,
            cone_dims = [("S", k1), ("S", k2)],
            G = spzeros(0, k), d = zeros(0),
            known_status = :Optimal, known_obj = tr(C1) + tr(C2),
            known_X = nothing,
            known_y = vcat(ConicIP.vecm(C1), ConicIP.vecm(C2)),
            description = "Multiple SDP blocks (n1=$n1, n2=$n2)")
end

"""
    sdp_mixed_cones(; n_r=4, n_q=3, n_s=3)

Projection of a fixed point onto `R₊ × Q × S`:
`min ½‖y - c‖²  s.t.  y ⪰_K 0`.  Exercises a heterogeneous `Block` of
`Diagonal`, `SymWoodbury`, and `VecCongurance` scalings.  The solution is
the blockwise Euclidean projection `P_K(c)`, computed here in closed form,
and is unique because the objective is strictly convex.

The drawn point is asserted to sit clear of every projection boundary, so
a change in the RNG stream fails loudly here rather than silently eroding
the accuracy of `known_y`.
"""
function sdp_mixed_cones(; n_r = 4, n_q = 3, n_s = 3)
    rng = Random.Xoshiro(77)
    k_s = _vdim(n_s)
    m_q = n_q + 1
    n   = n_r + m_q + k_s

    Q = sparse(1.0I, n, n)
    c = randn(rng, n)
    A = sparse(1.0I, n, n)
    b = zeros(n)

    c_r = c[1:n_r]
    c_q = c[n_r+1:n_r+m_q]
    c_s = c[n_r+m_q+1:end]

    # Stay clear of the kinks: no near-zero R₊ coordinate, no SOC point
    # near either boundary ray, no near-zero eigenvalue in the S block.
    @assert minimum(abs, c_r) > 0.05
    @assert abs(norm(c_q[2:end]) - abs(c_q[1])) > 0.05
    @assert minimum(abs, eigvals(Symmetric(ConicIP.mat(c_s)))) > 0.05

    known_y = vcat(max.(c_r, 0.0), _proj_soc(c_q), _proj_psd(c_s))

    return (Q = Q, c = c, A = A, b = b,
            cone_dims = [("R", n_r), ("Q", m_q), ("S", k_s)],
            G = spzeros(0, n), d = zeros(0),
            known_status = :Optimal, known_obj = nothing,
            known_X = nothing,
            known_y = known_y,
            description = "Mixed cones: R($n_r) + Q($m_q) + S($k_s)")
end

"""
    sdp_with_equality(; n=4)

Standard-form SDP `min tr(C*X)  s.t.  tr(Aᵢ*X) = dᵢ, X ⪰ 0`.  The first
constraint matrix is the identity, which fixes `tr(X)` and so keeps the
feasible set compact (and the objective bounded).
"""
function sdp_with_equality(; n = 4)
    rng = Random.Xoshiro(321)

    k = _vdim(n)
    M = randn(rng, n, n)
    C = (M + M') / 2
    Q = spzeros(k, k)
    c = -ConicIP.vecm(C)                            # -c'y = tr(C*X)

    # Strictly feasible reference point, used to set the right-hand side.
    X0 = Matrix{Float64}(I, n, n) + 0.1 * randn(rng, n, n)
    X0 = X0' * X0

    n_eq = 3
    G = zeros(n_eq, k)
    d = zeros(n_eq)
    for i = 1:n_eq
        Ri = randn(rng, n, n)
        Ai = i == 1 ? Matrix{Float64}(I, n, n) : (Ri + Ri') / 2
        G[i, :] = ConicIP.vecm(Ai)'
        d[i]    = tr(Ai * X0)
    end
    G = sparse(G)

    A = sparse(1.0I, k, k)
    b = zeros(k)

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = G, d = d,
            known_status = :Optimal, known_obj = nothing,
            known_X = nothing, unique_y = false,
            checks = ["G*y = d" => (sol, tol) -> norm(G * sol.y - d, Inf) < tol,
                      "X ⪰ 0"   => _chk_psd(k)],
            description = "SDP with equality constraints (n=$n, m=$n_eq)")
end

"""
    sdp_larger(; n=11)

Larger PSD projection, with a dense target of known eigenstructure so the
solution is still available in closed form.  The spectrum is chosen to
avoid an exact zero eigenvalue: a zero in the target makes the optimum
non-strictly-complementary and costs about an order of magnitude of
accuracy in the recovered matrix.
"""
function sdp_larger(; n = 11)
    rng  = Random.Xoshiro(555)
    U, _ = qr(randn(rng, n, n))
    U    = Matrix(U)
    eigs = collect(range(-1.0, 2.0, length = n))    # step 0.3, no zero
    @assert minimum(abs, eigs) > 0.05
    target   = Symmetric(U * diagm(0 => eigs) * U') |> Matrix
    expected = Symmetric(U * diagm(0 => max.(eigs, 0.0)) * U') |> Matrix

    k = _vdim(n)
    Q = Matrix{Float64}(I, k, k)
    c = ConicIP.vecm(target)
    A = sparse(1.0I, k, k)
    b = zeros(k)

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = spzeros(0, k), d = zeros(0),
            known_status = :Optimal, known_obj = nothing,
            known_X = expected,
            description = "Larger PSD projection (n=$n)")
end

"""
    sdp_identity_feasible(; n=3)

Trivially feasible SDP `min 0  s.t.  X ⪰ I`.  The objective is identically
zero, so every feasible `X` is optimal and only the constraint says
anything: the returned `X` must satisfy `X ⪰ I`.
"""
function sdp_identity_feasible(; n = 3)
    k = _vdim(n)
    Q = spzeros(k, k)
    c = zeros(k)
    A = sparse(1.0I, k, k)
    b = ConicIP.vecm(Matrix{Float64}(I, n, n))

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = spzeros(0, k), d = zeros(0),
            known_status = :Optimal, known_obj = nothing,
            known_X = nothing, unique_y = false,
            checks = ["X ⪰ I" =>
                      (sol, tol) -> eigmin(Symmetric(ConicIP.mat(sol.y))) > 1 - tol],
            description = "Trivially feasible: min 0 s.t. X ⪰ I (n=$n)")
end

"""
    sdp_rank_one()

`min tr(C*X)  s.t.  tr(X) = 1, X ⪰ 0` on a 4×4 `C` built with a fixed,
well-separated spectrum, so neither the optimal value nor the eigengap
depends on the RNG stream.  The solution is the rank-one matrix `vv'` for
`v` the eigenvector of the smallest eigenvalue of `C`, and the optimal
value is that eigenvalue.
"""
function sdp_rank_one()
    n    = 4
    rng  = Random.Xoshiro(111)
    U, _ = qr(randn(rng, n, n))
    U    = Matrix(U)
    spec = [-3.0, -0.5, 1.0, 2.5]
    C    = Symmetric(U * diagm(0 => spec) * U') |> Matrix

    k = _vdim(n)
    Q = spzeros(k, k)
    c = -ConicIP.vecm(C)                            # -c'y = tr(C*X)

    G = sparse(reshape(ConicIP.vecm(Matrix{Float64}(I, n, n)), 1, k))
    d = ones(1)

    A = sparse(1.0I, k, k)
    b = zeros(k)

    λ, V = eigen(Symmetric(C))
    v = V[:, argmin(λ)]

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = G, d = d,
            known_status = :Optimal, known_obj = minimum(spec),
            known_X = v * v',
            description = "Rank-1 SDP: min tr(C*X) s.t. tr(X)=1 (n=$n)")
end

"""
    sdp_affine_eigmax()

Largest eigenvalue as an SDP: `min t  s.t.  t*I - C ⪰ 0`, with
`C = I + 2uu'` and `u = [1,2,3]/√14`, so `λ(C) = (3, 1, 1)` and
`t⋆ = λmax(C) = 3` with `u` the (simple) leading eigenvector.

Unlike every other generator here, `A` is a single column and `b` has
nonzero off-diagonal entries, so the matrix variable is *not* `sol.y`
(which is the scalar `t`): it is the inequality dual `sol.v`, which at
the optimum is the spectral projector `uu'`.  The instance therefore
pins the vecm scaling of an off-diagonal in `b` *and* the off-diagonal
entries of a recovered dual.

This is a coverage fixture, not regression evidence for any particular
bug: it passes against the pre-Jordan-product `xsdc!`/`dsdc!` as well.
"""
function sdp_affine_eigmax()
    u = [1.0, 2.0, 3.0] / sqrt(14.0)
    C = Matrix{Float64}(I, 3, 3) + 2 * (u * u')

    k = _vdim(3)
    Q = zeros(1, 1)
    c = [-1.0]                                      # -c'y = t
    A = reshape(ConicIP.vecm(Matrix{Float64}(I, 3, 3)), k, 1)
    b = ConicIP.vecm(C)                             # t*I - C ⪰ 0

    return (Q = Q, c = c, A = A, b = b, cone_dims = [("S", k)],
            G = spzeros(0, 1), d = zeros(0),
            known_status = :Optimal, known_obj = 3.0,
            known_X = nothing, known_y = [3.0], known_V = u * u',
            description = "Max eigenvalue: min t s.t. t*I ⪰ C (n=3)")
end

"""
    sdp_all_problems()

The standard list of SDP instances used by the test suite.  Only one PSD
projection appears: `sdp_psd_projection(n=6)` duplicates the existing
"SDP - Projection onto PSD Matrix" test in `runtests.jl`, which asserts
strictly more.
"""
sdp_all_problems() = [
    sdp_psd_projection(n = 8),
    sdp_trace_minimization(n = 4),
    sdp_nearest_correlation(n = 5),
    sdp_max_cut_relaxation(),
    sdp_lovasz_theta_c5(),
    sdp_rank_one(),
    sdp_identity_feasible(n = 3),
    sdp_multiple_blocks(n1 = 3, n2 = 4),
    sdp_mixed_cones(n_r = 4, n_q = 3, n_s = 3),
    sdp_with_equality(n = 4),
    sdp_affine_eigmax(),
    sdp_larger(n = 11),
]
