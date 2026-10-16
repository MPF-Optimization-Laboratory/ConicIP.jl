# exp-activeset: shared helpers.  No file in the repository is modified.
#
# Iterate capture strategy (option (c) of the brief): re-run conicIP with
# maxIters = k for k = 1,2,...  and read the returned Solution.  The solver
# stores into `sol` the *best-measure-so-far* iterate, so we also record
# sol.Iter; when sol.Iter == k the returned point IS iterate k.

using ConicIP, SparseArrays, LinearAlgebra, Printf, Random
const QD = ConicIP.QDLDL

# ───────────────────────── instances ─────────────────────────

# verbatim from benchmark/suite.jl
function lp_band(n; w = 5, seed = 1)
    Random.seed!(seed)
    I_ = Int[]; J_ = Int[]; V_ = Float64[]
    for i in 1:n, j in max(1, i - w):min(n, i + w)
        push!(I_, i); push!(J_, j); push!(V_, randn())
    end
    B = sparse(I_, J_, V_, n, n)
    A = [B; sparse(1.0I, n, n); -sparse(1.0I, n, n)]
    b = [-rand(n); fill(-10.0, n); fill(-10.0, n)]
    return (Q = spzeros(n, n), c = randn(n), A = A, b = b,
            cone_dims = [("R", 3n)], G = spzeros(0, n), d = zeros(0))
end

# Matrix-Market reader (LPnetlib arrays / coordinate)
function readmm(path)
    lines = readlines(path)
    hdr = lines[1]
    body = filter(l -> !startswith(l, '%') && !isempty(strip(l)), lines[2:end])
    dims = parse.(Int, split(strip(body[1])))
    if occursin("array", hdr)
        vals = [parse(Float64, strip(l)) for l in body[2:end]]
        return reshape(vals, dims[1], dims[2])
    else
        I_ = Int[]; J_ = Int[]; V_ = Float64[]
        for l in body[2:end]
            t = split(strip(l))
            push!(I_, parse(Int, t[1])); push!(J_, parse(Int, t[2]))
            push!(V_, parse(Float64, t[3]))
        end
        return sparse(I_, J_, V_, dims[1], dims[2])
    end
end

# LPnetlib: min cᵀx s.t. Ax = b, lo ≤ x ≤ hi  (hi = 1e308 means +∞)
# ConicIP: min ½yᵀQy − c_cipᵀy  s.t. Gy = d, A_cip y − s = b_cip, s ≥ 0
function load_netlib(dir, name)
    p = joinpath(dir, name)
    A = readmm(joinpath(p, "$(name).mtx"))
    b = vec(readmm(joinpath(p, "$(name)_b.mtx")))
    c = vec(readmm(joinpath(p, "$(name)_c.mtx")))
    lo = vec(readmm(joinpath(p, "$(name)_lo.mtx")))
    hi = vec(readmm(joinpath(p, "$(name)_hi.mtx")))
    m, n = size(A)
    finlo = findall(isfinite.(lo) .& (lo .> -1e30))
    finhi = findall(isfinite.(hi) .& (hi .< 1e30))
    Alo = sparse(1:length(finlo), finlo, ones(length(finlo)), length(finlo), n)
    Ahi = sparse(1:length(finhi), finhi, -ones(length(finhi)), length(finhi), n)
    Acip = [Alo; Ahi]
    bcip = [lo[finlo]; -hi[finhi]]
    return (Q = spzeros(n, n), c = -c, A = Acip, b = bcip,
            cone_dims = [("R", size(Acip, 1))], G = sparse(A), d = b,
            nlo = length(finlo), nhi = length(finhi), mA = m, nx = n)
end

# Degenerate LP with a *known* optimal partition.
#   min −c'y  s.t. y ≥ 0 (n rows) and Gy = d.
# Construct y* with support B, dual v* with support N, and deliberately
# leave `ndeg` indices with y*_i = 0 AND v*_i = 0 (strict complementarity
# fails there), plus primal degeneracy from extra tight rows.
function degen_lp(n, p; nB = div(n, 3), ndeg = 5, seed = 7)
    rng = Random.MersenneTwister(seed)
    G = sprandn(rng, p, n, 0.3)
    G = G + sparse(1:p, 1:p, ones(p), p, n)     # keep rank p
    ystar = zeros(n)
    Bset = 1:nB
    ystar[Bset] .= rand(rng, nB) .+ 0.1
    d = G * ystar
    w = randn(rng, p)
    # v* ≥ 0 supported on N \ degenerate set; c = G'w − A'v with A = I
    vstar = zeros(n)
    Nset = (nB + 1):n
    for i in Nset; vstar[i] = rand(rng) + 0.1; end
    for i in (nB + 1):(nB + ndeg); vstar[i] = 0.0; end   # not strictly complementary
    c = G' * w - vstar                                    # A = I so A'v = v
    return (Q = spzeros(n, n), c = Vector(c), A = sparse(1.0I, n, n),
            b = zeros(n), cone_dims = [("R", n)], G = G, d = d,
            ystar = ystar, vstar = vstar, ndeg = ndeg, nB = nB)
end

function socp_sum_of_norms(k::Int; d::Int = max(2, div(k, 2)), seed::Int = 0)
    rng = Random.MersenneTwister(seed)
    n = d + 3k
    ti(i) = d + i
    ri(j) = d + k + j
    Ig, Jg, Vg = Int[], Int[], Float64[]
    dvec = zeros(2k)
    for j in 1:2k
        push!(Ig, j); push!(Jg, ri(j)); push!(Vg, 1.0)
        push!(Ig, j); push!(Jg, mod1(j, d)); push!(Vg, -randn(rng))
        for _ in 1:2
            push!(Ig, j); push!(Jg, rand(rng, 1:d)); push!(Vg, -randn(rng))
        end
        dvec[j] = -randn(rng)
    end
    G = sparse(Ig, Jg, Vg, 2k, n)
    Ia, Ja, Va = Int[], Int[], Float64[]
    for i in 1:k
        base = 3(i - 1)
        push!(Ia, base + 1); push!(Ja, ti(i));      push!(Va, 1.0)
        push!(Ia, base + 2); push!(Ja, ri(2i - 1)); push!(Va, 1.0)
        push!(Ia, base + 3); push!(Ja, ri(2i));     push!(Va, 1.0)
    end
    A = sparse(Ia, Ja, Va, 3k, n)
    b = zeros(3k)
    cone_dims = [("Q", 3) for _ in 1:k]
    Q = spzeros(n, n); c = zeros(n); c[d+1:d+k] .= -1.0
    return (Q = Q, c = c, A = A, b = b, cone_dims = cone_dims, G = G, d = dvec)
end

# ───────────────────────── capture ─────────────────────────

solve1(prob; kw...) = conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                              prob.G, prob.d; verbose = false, certFallback = false, kw...)

"""Return (sols, tdefault, iters) : sols[k] is the Solution returned with maxIters=k."""
function capture(prob; optTol = 1e-8, T = 60)
    sols = ConicIP.Solution[]
    for k in 1:T
        s = solve1(prob; maxIters = k, optTol = optTol)
        push!(sols, s)
        s.status == :Optimal && break
    end
    t = @elapsed sfin = solve1(prob; maxIters = T, optTol = optTol)
    return sols, t, sfin
end

# ───────────────────── partition / indicators (LP) ─────────────────────

"predicted B (s_i dominant ⇒ inequality inactive) from one iterate"
ratio_B(s, v) = s .> v

function tapia_B(s1, s0, v1, v0)
    Ts = s1 ./ max.(s0, 1e-300)
    Tv = v1 ./ max.(v0, 1e-300)
    return Ts .> Tv, Ts, Tv
end

# ───────────────────── Ye projection (LP) ─────────────────────

# min ‖x − x0‖ s.t. M x = r, via the quasi-definite system
#   [ I   Mᵀ ] [dx]   [ 0       ]
#   [ M  −δI ] [λ ] = [ r − M x0]
# factorized with QDLDL (the solver's own backend), one step of refinement.
function minnorm_proj(M::SparseMatrixCSC, r::Vector, x0::Vector; δ = 1e-12, nref = 2)
    n = size(M, 2); q = size(M, 1)
    K = [sparse(1.0I, n, n)  spzeros(n, q);
         M                   -δ * sparse(1.0I, q, q)]
    K = sparse(Symmetric(SparseMatrixCSC(K), :L))
    F = QD.qdldl(K; Dsigns = [ones(Int, n); -ones(Int, q)])
    rhs = [zeros(n); r - M * x0]
    z = copy(rhs); QD.solve!(F, z)
    for _ in 1:nref
        res = rhs - K * z
        e = copy(res); QD.solve!(F, e); z .+= e
    end
    return x0 + z[1:n], F
end

function minnorm_proj_timed(M, r, x0; kw...)
    t = @elapsed out = minnorm_proj(M, r, x0; kw...)
    return out[1], t
end

relres(x, scale) = norm(x) / (1 + scale)
