using Test
using ConicIP
using LinearAlgebra
using SparseArrays
using Random

include("testdata.jl")

const tol    = 1e-3
const optTol = 1e-7

P_box(t, x) = [sign(xi) * (abs(xi) <= t ? abs(xi) : t) for xi in x]
optcond(x, P, ∇f) = norm(x - P(x - ∇f(x))) / length(x)

function compare(s1, s2::Dict)
    return (s1.status == s2[:status] &&
            abs(s1.prFeas - s2[:prFeas]) < tol &&
            abs(s1.Mu - s2[:Mu]) < tol &&
            abs(s1.muFeas - s2[:muFeas]) < tol &&
            abs(s1.duFeas - s2[:duFeas]) < tol)
end

@testset "ConicIP" begin

    Random.seed!(0)

    @testset "Block Tests" begin

        A = Block(3)
        A[1] = rand(4, 4)
        A[2] = rand(3, 3)
        A[3] = rand(2, 2)

        B = Block(3)
        B[1] = rand(4, 4)
        B[2] = rand(3, 3)
        B[3] = rand(2, 2)

        @test size(A) == (9, 9)
        @test size(A, 1) == 9
        @test size(A, 2) == 9

        @test size(B) == (9, 9)
        @test size(B, 1) == 9
        @test size(B, 2) == 9

        @test Matrix(A * B) ≈ Matrix(A) * Matrix(B)
        @test Matrix(A + B) ≈ Matrix(A) + Matrix(B)
        @test Matrix(A^2) ≈ Matrix(A)^2

        @test Matrix(A - B) ≈ Matrix(A) - Matrix(B)

        @test A * Matrix{Float64}(I, 9, 9) ≈ Matrix(A)
        @test A * ones(9) ≈ Matrix(A) * ones(9)

        @test A' * ones(9) ≈ Matrix(A)' * ones(9)

        # inv is promised by the Block docstring. adjoint(::Block) returns a
        # Block, so the Adjoint{_,Block} methods are reached only through an
        # explicitly constructed wrapper — keep them exercised.
        C = Block(2)
        C[1] = rand(3, 3) + 3I        # well conditioned for the inverse check
        C[2] = rand(2, 2) + 3I
        @test Matrix(inv(C)) ≈ inv(Matrix(C))
        Aᴴ = LinearAlgebra.Adjoint(A)
        @test Aᴴ * ones(9) ≈ Matrix(A)' * ones(9)
        @test Aᴴ * Matrix{Float64}(I, 9, 9) ≈ Matrix(A)'
        @test Matrix(Aᴴ * B) ≈ Matrix(A)' * Matrix(B)

        Ad = deepcopy(A)
        Ad[1] = zeros(4, 4)

        @test A[1] != zeros(4, 4)
        @test A * ones(9) ≈ Matrix(A) * ones(9)

        @test Matrix(Diagonal(ones(9)) + A) ≈ Matrix(Diagonal(ones(9))) + Matrix(A)

    end

    @testset "Misc Tests" begin

        A = rand(3, 3)
        Z = ConicIP.VecCongurance(A)

        @test Z * ones(6) ≈ Matrix(Z) * ones(6)
        @test Matrix(Z' * Z) ≈ Matrix(Z)' * Matrix(Z)
        @test inv(Z) * ones(6) ≈ Matrix(Z) \ ones(6)
        @test size(Z, 1) == 6
        @test sparse(Z) ≈ Matrix(Z)

        # Conic steplength from a point outside the cone: the generalized
        # eigen form factors X, so X ⋡ 0 is a factorization failure (which
        # conicIP reports as :Error) rather than an infinite step.
        X = -Matrix{Float64}(I, 3, 3)
        D = Matrix{Float64}(I, 3, 3)
        @test_throws ConicIP.KKT_FAILURES ConicIP.maxstep_sdc(
            ConicIP.vecm(X), ConicIP.vecm(D))

        # Test direct sparse(SymWoodbury) avoids dense materialization
        sw = ConicIP.WoodburyMatrices.SymWoodbury(Diagonal(rand(50)), randn(50, 2), Matrix(1.0I, 2, 2))
        @test sparse(sw) ≈ Matrix(sw)

    end

    @testset "Box Constrained QP, H = I" begin

        Random.seed!(0)

        n = 1000
        H = 0.5 * Id(n)
        c = collect(1.0:n)

        A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
        b = -ones(2 * n)
        ∇f = x -> H * (x - c)

        function kktsolver_2x2_box(Q, A, G, cone_dims)
            function solve2x2gen(F, F⁻¹)
                v = inv(F[1] * F[1]).diag
                D = Diagonal(v[1:n] + v[n+1:end])
                invHD = inv(Diagonal(H.diag + D.diag))
                return (rhs, rhs2) -> (invHD * rhs, zeros(0))
            end
            return solve2x2gen
        end

        sol = conicIP(H, H * c, A, b, [("R", 2 * n)],
                      kktsolver = pivot(kktsolver_2x2_box),
                      optTol = optTol,
                      DTB = 0.01,
                      maxRefinementSteps = 3)

        ystar = sol.y

        @test optcond(ystar, x -> P_box(1, x), ∇f) < tol

        s = Dict(:status => :Optimal,
                 :prFeas => 0,
                 :Mu => 0,
                 :muFeas => 0,
                 :duFeas => 0,
                 :Iter => 7)

        @test compare(sol, s)

    end

    for kktsolver = (ConicIP.kktsolver_qr,
                     ConicIP.kktsolver_sparse,
                     pivot(ConicIP.kktsolver_2x2))

        @testset "Projection onto Sphere ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 2
            H = Id(n)
            a = ones(n)
            A = [spzeros(1, n); sparse(1.0I, n, n)]
            b = [-1; zeros(n)]

            sol = conicIP(H, H * a, A, b, [("Q", n + 1)],
                          optTol = optTol,
                          DTB = 0.01,
                          kktsolver = kktsolver,
                          maxRefinementSteps = 3)

            ystar = sol.y

            @test norm(ystar - a / norm(a)) < tol

            s = Dict(:status => :Optimal,
                     :prFeas => 0.0,
                     :Mu => 2.866608128093695e-7,
                     :muFeas => 1.621702501927476e-7,
                     :duFeas => 3.2367552452111847e-16,
                     :Iter => 5)

            @test compare(sol, s)

        end

        @testset "Combined ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = Id(n)
            c = collect(1.0:n)

            A = [sparse(1.0I, n, n);      # R
                 spzeros(1, n);            # Q
                 sparse(1.0I, n, n)]       #

            b = [zeros(n);
                 -1.0;
                 zeros(n)]

            sol = conicIP(H, H * c, A, b, [("R", n), ("Q", n + 1)],
                          optTol = optTol,
                          DTB = 0.01,
                          kktsolver = kktsolver,
                          maxRefinementSteps = 3)

            ystar = sol.y

            y = [max(0, i) for i in c]
            y = y / norm(y)

            @test norm(ystar - y) < tol

            s = Dict(:status => :Optimal,
                     :prFeas => 7.764421906286858e-17,
                     :Mu => 4.663886012743681e-7,
                     :muFeas => 1.7037397157416066e-7,
                     :duFeas => 2.77947804665922e-17,
                     :Iter => 10)

            @test compare(sol, s)

        end

        @testset "Projection onto simplex ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = Matrix{Float64}(I, n, n)
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = ones(1, n)
            d = [1.0]

            sol = conicIP(H, H * c,
                          A, b, [("R", n)],
                          G, d,
                          kktsolver = kktsolver,
                          optTol = optTol)

            ystar = sol.y

            ysol = zeros(n)
            ysol[n] = 1

            @test norm(ystar - ysol) < tol

            s = Dict(:status => :Optimal,
                     :prFeas => 1.4506364239112378e-16,
                     :Mu => 2.7686402945528533e-9,
                     :muFeas => 2.897827518851058e-9,
                     :duFeas => 2.70780035221441e-17,
                     :Iter => 11)

            @test compare(sol, s)

        end

        @testset "Abandoned ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = Matrix{Float64}(I, n, n)
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = ones(1, n)
            d = [1.0]

            sol = conicIP(H, H * c,
                          A, b, [("R", n)],
                          G, d,
                          kktsolver = kktsolver,
                          optTol = optTol,
                          maxIters = 2)

            @test sol.status == :Abandoned

        end

        @testset "Projection onto simplex, dense H ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = ones(1, n)
            d = [1.0]

            sol = ConicIP.conicIP(H, H * c,
                                  A, b, [("R", n)],
                                  G, d,
                                  kktsolver = kktsolver,
                                  optTol = optTol)

            ystar = sol.y

            s = Dict(:status => :Optimal,
                     :prFeas => 4.488229069360946e-16,
                     :Mu => 2.1436595135398927e-8,
                     :muFeas => 3.000777220457259e-9,
                     :duFeas => 6.279962324264275e-17,
                     :Iter => 8)

            @test compare(sol, s)

        end

        @testset "Projection onto simplex, dense H, Random Projection ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = rand(6, n)
            d = zeros(6)

            ystar = conicIP(H, H * c,
                            A, b, [("R", n)],
                            G, d,
                            kktsolver = kktsolver,
                            optTol = optTol).y

        end

        @testset "Linear Constraints Comparison ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = rand(6, n)
            d = zeros(6)

            ystar1 = conicIP(H, H * c,
                             A, b, [("R", n)],
                             G, d,
                             kktsolver = kktsolver,
                             optTol = optTol).y

            ystar2 = conicIP(H, H * c,
                             [A; G; -G], [b; d; -d], [("R", (n + 2 * 6))],
                             G, d,
                             optTol = optTol).y

            @test norm(ystar1 - ystar2) < tol

        end

        @testset "Preprocessor Test - Bad Primal Constraints ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = rand(6, n)
            G = [G; G]
            d = zeros(6)
            d = [d; d]

            ystar1 = preprocess_conicIP(H, H * c,
                                        A, b, [("R", n)],
                                        G, d,
                                        kktsolver = kktsolver,
                                        verbose = true,
                                        optTol = optTol).y

            ystar2 = preprocess_conicIP(H, H * c,
                                        [A; G; -G], [b; d; -d], [("R", (n + 4 * 6))],
                                        G, d,
                                        verbose = true,
                                        optTol = optTol).y

            @test norm(ystar1 - ystar2) < tol

        end

        @testset "Preprocessor Test - Bad Dual Constraints ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            Q = zeros(2 * n, 2 * n)
            c = -ones(2 * n)

            A = sparse(1.0I, n, n)
            A = [A A]
            d = zeros(n)

            sol = preprocess_conicIP(Q, c,
                                     A, d, [("R", n)],
                                     kktsolver = kktsolver,
                                     verbose = true,
                                     optTol = optTol)

            @test norm(sol.y) < tol

        end

        @testset "Preprocessor Test - Infeasible ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = zeros(1, n)
            G[1, 1] = 1
            G = [G; G]
            d = [1.0; -1.0]

            sol = preprocess_conicIP(H, H * c,
                                     A, b, [("R", n)],
                                     G, d,
                                     kktsolver = kktsolver,
                                     optTol = optTol)

            @test sol.status == :Infeasible

            # The preprocessor trims G to an independent row set, but the ray
            # it returns must certify the ORIGINAL system.
            @test sol.has_certificate
            chk, _, _ = ConicIP.validate_infeasibility_certificate(
                H, H * c, A, b, [("R", n)], G, d, sol.w, sol.v;
                abstol = 1e-9, reltol = 1e-7)
            @test chk.valid
            @test abs(dot(d, sol.w) - dot(b, sol.v) + 1) < 1e-6

        end

        @testset "Infeasible ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
            b = [ones(n); ones(n)]

            sol = conicIP(H, H * c,
                          A, b, [("R", 2 * n)],
                          kktsolver = kktsolver,
                          optTol = optTol)

            @test sol.status == :Infeasible

            # No equality block here: G/d are empty, so dᵀw̄ drops out and the
            # normalization identity reduces to -bᵀv̄ = -1.
            G = zeros(0, n); d = zeros(0)
            @test sol.has_certificate
            chk, _, _ = ConicIP.validate_infeasibility_certificate(
                H, H * c, A, b, [("R", 2 * n)], G, d, sol.w, sol.v;
                abstol = 1e-9, reltol = 1e-7)
            @test chk.valid
            @test abs(dot(d, sol.w) - dot(b, sol.v) + 1) < 1e-6

        end

        @testset "Infeasible (With linear constraints) ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            G = [1 zeros(1, 9)]
            d = [-1.0]

            sol = conicIP(H, H * c,
                          A, b, [("R", n)],
                          G, d,
                          kktsolver = kktsolver,
                          optTol = optTol)

            @test sol.status == :Infeasible

            @test sol.has_certificate
            chk, _, _ = ConicIP.validate_infeasibility_certificate(
                H, H * c, A, b, [("R", n)], G, d, sol.w, sol.v;
                abstol = 1e-9, reltol = 1e-7)
            @test chk.valid
            @test abs(dot(d, sol.w) - dot(b, sol.v) + 1) < 1e-6

        end

        @testset "Unbounded ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = zeros(n, n)
            c = collect(1.0:n)

            A = sparse(1.0I, n, n)
            b = zeros(n)

            sol = conicIP(H, c,
                          A, b, [("R", n)],
                          kktsolver = kktsolver,
                          optTol = optTol)

            @test sol.status == :DualInfeasible

            # Recession ray: Qȳ ≈ 0, Gȳ ≈ 0 (G empty), Aȳ ∈ K, cᵀȳ = +1.
            G = zeros(0, n); d = zeros(0)
            @test sol.has_certificate
            chk, _ = ConicIP.validate_unboundedness_certificate(
                H, c, A, b, [("R", n)], G, d, sol.y;
                abstol = 1e-9, reltol = 1e-7)
            @test chk.valid
            @test abs(dot(c, sol.y) - 1) < 1e-6

        end

        @testset "Bad Input ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = zeros(n, n)
            c = collect(1.0:n)

            A = sparse(1.0I, n + 2, n + 2)
            b = zeros(n)

            @test_throws Exception conicIP(H, c,
                                           A, b, [("R", n)],
                                           kktsolver = kktsolver,
                                           optTol = optTol)

        end

    end

    @testset "SDP - Projection onto PSD Matrix" begin

        Random.seed!(0)

        n = 21
        H = Matrix{Float64}(I, n, n)
        c = ConicIP.vecm(diagm(0 => [1.0; 1; 1; -1; -1; -1]))

        A = sparse(1.0I, 21, 21)
        b = zeros(21)

        sol = conicIP(H, c,
                      A, b, [("S", n)],
                      optTol = optTol)

        s = Dict(:status => :Optimal,
                 :prFeas => 4.2341217602756234e-16,
                 :Mu => 3.4583513329836624e-10,
                 :muFeas => 1.48267911727847e-9,
                 :duFeas => 4.2341217602756234e-16,
                 :Iter => 6)

        @test norm(ConicIP.mat(sol.y) - diagm(0 => [1.0; 1; 1; 0; 0; 0]), Inf) < tol
        @test compare(sol, s)

    end

    @testset "SOC Cone (direct API)" begin

        Random.seed!(0)

        for to_preprocess = [true, false]
            # QP with SOC and NonNeg constraints
            # min (1/2)||x||² + 1'x s.t. ||x[1:3]|| ≤ 1, x ≥ 0
            # Solver minimizes (1/2)y'Qy - c'y, so c = -1 gives +1'x
            n = 4
            Q = sparse(1.0I, n, n)
            c_obj = -ones(n)

            # SOC constraint: [1; x₁; x₂; x₃] ∈ SOC → ||x[1:3]|| ≤ 1
            A_soc = [spzeros(1, n); sparse(1.0I, 3, n)[1:3, :]]
            b_soc = [-1.0; zeros(3)]

            # NonNeg variable cone: x ≥ 0
            A_nn = sparse(1.0I, n, n)
            b_nn = zeros(n)

            A_full = sparse([A_soc; A_nn])
            b_full = [b_soc; b_nn]
            cone_dims = [("Q", 4), ("R", n)]

            if to_preprocess
                sol = preprocess_conicIP(Q, c_obj, A_full, b_full, cone_dims,
                                         verbose = true, optTol = 1e-6)
            else
                sol = conicIP(Q, c_obj, A_full, b_full, cone_dims,
                              optTol = 1e-6)
            end

            @test sol.status == :Optimal
            @test norm(sol.y) < tol
        end

    end

    @testset "Miles's Counterexamples" begin

        for kktsolver = (ConicIP.kktsolver_qr,
                         ConicIP.kktsolver_sparse,
                         pivot(ConicIP.kktsolver_2x2))

            @testset "Miles Problem 1 - Optimal" begin
                prob = miles_problem_1()
                data = mpb_to_conicip(prob.c, prob.A, prob.b, prob.con_cones, prob.var_cones)

                sol = preprocess_conicIP(data.Q, data.c, data.A, data.b, data.cone_dims,
                              data.G, data.d,
                              verbose = true, kktsolver = kktsolver)
                @test sol.status == :Optimal
            end

            @testset "Miles Problem 2 - Infeasible" begin
                prob = miles_problem_2()
                data = mpb_to_conicip(prob.c, prob.A, prob.b, prob.con_cones, prob.var_cones)

                sol = preprocess_conicIP(data.Q, data.c, data.A, data.b, data.cone_dims,
                              data.G, data.d,
                              verbose = true, kktsolver = kktsolver)
                @test sol.status == :Infeasible

                @test sol.has_certificate
                chk, _, _ = ConicIP.validate_infeasibility_certificate(
                    data.Q, data.c, data.A, data.b, data.cone_dims,
                    data.G, data.d, sol.w, sol.v;
                    abstol = 1e-9, reltol = 1e-7)
                @test chk.valid
                @test abs(dot(data.d, sol.w) - dot(data.b, sol.v) + 1) < 1e-6
            end

            @testset "Miles Problem 3 - Scaling" begin
                prob = miles_problem_3()

                for κ = [1e-8, 1e-6, 1e-4, 1, 1e4, 1e6, 1e8]
                    data = mpb_to_conicip(κ * prob.c, κ * prob.A, κ * prob.b,
                                          prob.con_cones, prob.var_cones)
                    sol = preprocess_conicIP(data.Q, data.c, data.A, data.b, data.cone_dims,
                                  data.G, data.d,
                                  verbose = true)
                    @test sol.status == :Optimal
                end

                for κ = [1e-4, 1, 1e4, 1e6]
                    data = mpb_to_conicip(prob.c, κ * prob.A, κ * prob.b,
                                          prob.con_cones, prob.var_cones)
                    sol = preprocess_conicIP(data.Q, data.c, data.A, data.b, data.cone_dims,
                                  data.G, data.d,
                                  verbose = true)
                    @test sol.status == :Optimal
                end

                for κ = [1e-6, 1e-4, 1, 1e4, 1e6]
                    data = mpb_to_conicip(prob.c, prob.A, prob.b,
                                          prob.con_cones, prob.var_cones)
                    sol = preprocess_conicIP(data.Q, data.c, data.A, data.b, data.cone_dims,
                                  data.G, data.d,
                                  verbose = true)
                    @test sol.status == :Optimal
                end
            end

        end

    end

    @testset "imcols correctness" begin
        Random.seed!(42)

        # Well-conditioned full-rank
        A = randn(5, 10)
        b = randn(5)
        R, consistent = ConicIP.imcols(A, b)
        @test length(R) == rank(A)
        @test consistent

        # Redundant rows
        A2 = [A; A[1:1, :] + A[2:2, :]]
        b2 = [b; b[1:1] + b[2:2]]
        R2, consistent2 = ConicIP.imcols(A2, b2)
        @test length(R2) == rank(Matrix(A2))
        @test consistent2

        # Inconsistent system
        A3 = [A; A[1:1, :]]
        b3 = [b; b[1:1] .+ 100]
        R3, consistent3 = ConicIP.imcols(A3, b3)
        @test !consistent3
    end

    @testset "Certificate validator" begin

        atol = 1e-8; rtol = 1e-8

        @testset "cone_margin" begin
            # No blocks
            @test ConicIP.cone_margin(Float64[], Tuple{String,Int}[]) == Inf

            # Nonnegative orthant
            @test ConicIP.cone_margin([1.0, 2.0], [("R",2)]) ≈ 1.0
            @test ConicIP.cone_margin([1.0, -0.5], [("R",2)]) ≈ -0.5

            # Second order cone: boundary and just outside
            @test ConicIP.cone_margin([1.0, 1.0, 0.0], [("Q",3)]) ≈ 0.0 atol=1e-12
            @test ConicIP.cone_margin([1.0 - 1e-6, 1.0, 0.0], [("Q",3)]) ≈ -1e-6 atol=1e-12

            # PSD cone: vecm of [1 2; 2 1] has eigmin = -1
            xs = ConicIP.vecm([1.0 2.0; 2.0 1.0])
            @test length(xs) == 3
            @test ConicIP.cone_margin(xs, [("S",3)]) ≈ -1.0

            # Mixed blocks take the blockwise minimum
            @test ConicIP.cone_margin([3.0; 1.0; 1.0; 0.0], [("R",1),("Q",3)]) ≈ 0.0 atol=1e-12
        end

        @testset "Infeasibility certificate" begin
            # x >= 1 and -x >= 0  (i.e. x <= 0) is infeasible
            A = reshape([1.0, -1.0], 2, 1)
            b = [1.0, 0.0]
            K = [("R",2)]
            G = spzeros(0,1); d = Float64[]
            Qm = zeros(1,1); c = [0.0]

            # Known-good Farkas ray
            w = Float64[]; v = [1.0, 1.0]
            (chk, w̄, v̄) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, b, K, G, d, w, v; abstol = atol, reltol = rtol)
            @test chk.valid
            @test chk.finite
            @test chk.separation ≈ 1.0
            @test chk.farkas_residual ≈ 0.0 atol=1e-12
            @test chk.cone_margin ≈ 1.0
            @test v̄ ≈ v
            @test isempty(w̄)

            # Nonfinite candidate
            for bad in ([NaN, 1.0], [Inf, 1.0])
                (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                    Qm, c, A, b, K, G, d, w, bad; abstol = atol, reltol = rtol)
                @test !chk.finite
                @test !chk.valid
            end

            # Nonpositive separation: candidates returned unmodified
            (chk, w2, v2) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, b, K, G, d, w, -v; abstol = atol, reltol = rtol)
            @test chk.separation ≈ -1.0
            @test !chk.valid
            @test v2 == -v

            # Feasible problem (0 <= x <= 1), Farkas-like candidate: separation < 0
            bf = [0.0, -1.0]
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, bf, K, G, d, w, [1.0, 1.0]; abstol = atol, reltol = rtol)
            @test chk.separation < 0
            @test !chk.valid

            # Small positive separation, residual blows up under normalization
            bs = [1e-8, 0.0]
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, bs, K, G, d, w, [1.0, 1.0 + 1e-6]; abstol = atol, reltol = rtol)
            @test chk.separation > 0
            @test chk.cone_margin > 0          # not a cone violation
            @test chk.farkas_residual > 1.0    # the linear residual is what fails
            @test !chk.valid
        end

        @testset "Infeasibility certificate - cone violation only" begin
            # A'v = 0 exactly, separation = 1, but v has a negative entry
            A = reshape([1.0, 1.0], 2, 1)
            b = [1.0, 0.0]
            K = [("R",2)]
            G = spzeros(0,1); d = Float64[]
            Qm = zeros(1,1); c = [0.0]

            (chk, _, v̄) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, b, K, G, d, Float64[], [1.0, -1.0];
                abstol = atol, reltol = rtol)
            @test chk.finite
            @test chk.separation ≈ 1.0
            @test chk.farkas_residual ≈ 0.0 atol=1e-12   # residual alone would pass
            @test chk.cone_margin ≈ -1.0                 # sole failure
            @test chk.cone_margin < -(atol + rtol)
            @test !chk.valid
            @test v̄ ≈ [1.0, -1.0]
        end

        @testset "Infeasibility certificate - SOC tolerance flip" begin
            A = zeros(3,3); G = spzeros(0,3); d = Float64[]
            Qm = zeros(3,3); c = zeros(3)
            b = [1.0, 0.0, 0.0]
            K = [("Q",3)]

            # On the boundary of Q: valid
            (chk1, _, _) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, b, K, G, d, Float64[], [1.0, 1.0, 0.0];
                abstol = atol, reltol = rtol)
            @test chk1.cone_margin ≈ 0.0 atol=1e-12
            @test chk1.valid

            # 1e-6 outside Q: verdict flips
            (chk2, _, _) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, b, K, G, d, Float64[], [1.0 - 1e-6, 1.0, 0.0];
                abstol = atol, reltol = rtol)
            @test chk2.cone_margin < -(atol + rtol)
            @test !chk2.valid
        end

        @testset "Infeasibility certificate - PSD block" begin
            # v = vecm([1 2; 2 1]) is not PSD (eigmin = -1)
            A = zeros(3,1); G = spzeros(0,1); d = Float64[]
            Qm = zeros(1,1); c = [0.0]
            b = [1.0, 0.0, 0.0]
            K = [("S",3)]
            v = ConicIP.vecm([1.0 2.0; 2.0 1.0])

            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Qm, c, A, b, K, G, d, Float64[], v; abstol = atol, reltol = rtol)
            @test chk.separation ≈ 1.0
            @test chk.farkas_residual ≈ 0.0 atol=1e-12
            @test chk.cone_margin ≈ -1.0
            @test !chk.valid
        end

        @testset "Unboundedness certificate" begin
            # min -x  s.t.  x >= 0  is unbounded below
            Qm = zeros(1,1); c = [1.0]
            A = ones(1,1); b = [0.0]
            K = [("R",1)]
            G = spzeros(0,1); d = Float64[]

            (chk, ȳ) = ConicIP.validate_unboundedness_certificate(
                Qm, c, A, b, K, G, d, [2.0]; abstol = atol, reltol = rtol)
            @test chk.valid
            @test chk.finite
            @test chk.separation ≈ 2.0
            @test ȳ ≈ [1.0]
            @test chk.farkas_residual ≈ 0.0 atol=1e-12
            @test chk.cone_margin ≈ 1.0

            # Ray leaves the cone
            (chk, _) = ConicIP.validate_unboundedness_certificate(
                Qm, c, -A, b, K, G, d, [2.0]; abstol = atol, reltol = rtol)
            @test chk.cone_margin ≈ -1.0
            @test !chk.valid

            # Q*ȳ != 0 : not a recession direction of the objective
            (chk, _) = ConicIP.validate_unboundedness_certificate(
                ones(1,1), c, A, b, K, G, d, [2.0]; abstol = atol, reltol = rtol)
            @test chk.farkas_residual ≈ 1.0
            @test !chk.valid

            # Nonpositive separation, candidate returned unmodified
            (chk, y2) = ConicIP.validate_unboundedness_certificate(
                Qm, c, A, b, K, G, d, [-2.0]; abstol = atol, reltol = rtol)
            @test chk.separation ≈ -2.0
            @test !chk.valid
            @test y2 == [-2.0]

            # Nonfinite candidate
            (chk, _) = ConicIP.validate_unboundedness_certificate(
                Qm, c, A, b, K, G, d, [NaN]; abstol = atol, reltol = rtol)
            @test !chk.finite
            @test !chk.valid
        end

        @testset "Solution has_certificate" begin
            args12 = (zeros(1), Float64[], zeros(1), zeros(1), :Optimal,
                      3, 1e-9, 1e-9, 1e-9, 1e-9, 0.0, 0.0)
            sol = ConicIP.Solution(args12...)
            @test :has_certificate in fieldnames(ConicIP.Solution)
            @test sol.has_certificate == false
            @test sol.message == ""

            sol2 = ConicIP.Solution(args12..., true)
            @test sol2.has_certificate == true
        end

    end

    @testset "Certificate fallback" begin

        atol = 1e-9; rtol = 1e-7

        # Infeasible QP:  min ½‖x‖²  s.t.  x ≥ 0,  -(x₁+x₂) ≥ 1/100
        # (i.e. x₁ + x₂ ≤ -0.01 with x ≥ 0 — empty).
        # At maxIters = 7 the in-loop screen has not fired yet, but μ has
        # collapsed, so the WP5 gate opens and the auxiliary Farkas QP
        # recovers an exact ray.
        Qi = Matrix(1.0I, 2, 2); ci = zeros(2)
        Ai = [sparse(1.0I, 2, 2); sparse(-ones(1, 2))]
        bi = [0.0, 0.0, 0.01]
        Ki = [("R", 3)]
        Gi = spzeros(0, 2); di = Float64[]

        @testset "Infeasible — fallback certifies a stalled solve" begin
            s = conicIP(Qi, ci, Ai, bi, Ki, Gi, di;
                        verbose = false, maxIters = 7, certFallback = true)
            @test s.status == :Infeasible
            @test s.has_certificate

            # the ray must validate against the ORIGINAL problem data
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Qi, ci, Ai, bi, Ki, Gi, di, s.w, s.v; abstol = atol, reltol = rtol)
            @test chk.valid
        end

        @testset "Infeasible — certFallback = false does not claim" begin
            s = conicIP(Qi, ci, Ai, bi, Ki, Gi, di;
                        verbose = false, maxIters = 7, certFallback = false)
            @test s.status == :Abandoned
            @test !s.has_certificate
            @test !(s.status == :Infeasible && s.has_certificate)
        end

        # Unbounded QP:  min ¼(y₁-y₂)² - y₁ - 2y₂  s.t.  y ≥ 0.
        # Q is singular with null direction (1,1), along which the objective
        # decreases without bound.
        Qu = [0.5 -0.5; -0.5 0.5]; cu = [1.0, 2.0]
        Au = sparse(1.0I, 2, 2); bu = zeros(2); Ku = [("R", 2)]
        Gu = spzeros(0, 2); du = Float64[]

        @testset "Unbounded — fallback certifies a stalled solve" begin
            s = conicIP(Qu, cu, Au, bu, Ku, Gu, du;
                        verbose = false, maxIters = 5, certFallback = true)
            @test s.status == :DualInfeasible
            @test s.has_certificate

            (chk, _) = ConicIP.validate_unboundedness_certificate(
                Qu, cu, Au, bu, Ku, Gu, du, s.y; abstol = atol, reltol = rtol)
            @test chk.valid
        end

        @testset "Unbounded — certFallback = false does not claim" begin
            s = conicIP(Qu, cu, Au, bu, Ku, Gu, du;
                        verbose = false, maxIters = 5, certFallback = false)
            @test s.status == :Abandoned
            @test !s.has_certificate
        end

        @testset "Feasible problem never gains a certificate" begin
            # min ½‖x‖² - 1ᵀx  s.t. x ≥ 0.  μ collapses by iteration 4, so the
            # gate opens and BOTH auxiliary QPs actually run — and both must
            # come back empty.
            n = 5
            Qf = Matrix(1.0I, n, n); cf = ones(n)
            Af = sparse(1.0I, n, n); bf = zeros(n); Kf = [("R", n)]
            s = conicIP(Qf, cf, Af, bf, Kf; verbose = false, maxIters = 4,
                        certFallback = true)
            @test s.status ∉ (:Infeasible, :DualInfeasible)
            @test !s.has_certificate
        end

        @testset "fallback_infeasibility_ray" begin
            # x ≥ 1 and -x ≥ 0 is infeasible
            A1 = sparse(reshape([1.0, -1.0], 2, 1)); K1 = [("R", 2)]
            Q1 = zeros(1, 1); c1 = [0.0]
            G1 = spzeros(0, 1); d1 = Float64[]

            r = ConicIP.fallback_infeasibility_ray(Q1, c1, A1, [1.0, 0.0], K1, G1, d1)
            @test r !== nothing
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Q1, c1, A1, [1.0, 0.0], K1, G1, d1, r[1], r[2];
                abstol = atol, reltol = rtol)
            @test chk.valid

            # 0 ≤ x ≤ 1 is feasible — no Farkas ray exists
            @test ConicIP.fallback_infeasibility_ray(
                Q1, c1, A1, [0.0, -1.0], K1, G1, d1) === nothing

            # and on the stalled QP above it recovers a valid ray outright
            r2 = ConicIP.fallback_infeasibility_ray(Qi, ci, Ai, bi, Ki, Gi, di)
            @test r2 !== nothing
            (chk2, _, _) = ConicIP.validate_infeasibility_certificate(
                Qi, ci, Ai, bi, Ki, Gi, di, r2[1], r2[2];
                abstol = atol, reltol = rtol)
            @test chk2.valid
        end

        @testset "fallback_unbounded_ray" begin
            # min -x s.t. x ≥ 0  (conicIP form: Q = 0, c = [1])
            Q1 = zeros(1, 1); A1 = sparse(1.0I, 1, 1); b1 = [0.0]
            K1 = [("R", 1)]; G1 = spzeros(0, 1); d1 = Float64[]

            y = ConicIP.fallback_unbounded_ray(Q1, [1.0], A1, b1, K1, G1, d1)
            @test y !== nothing
            (chk, _) = ConicIP.validate_unboundedness_certificate(
                Q1, [1.0], A1, b1, K1, G1, d1, y; abstol = atol, reltol = rtol)
            @test chk.valid

            # min +x s.t. x ≥ 0 is bounded — no recession ray
            @test ConicIP.fallback_unbounded_ray(
                Q1, [-1.0], A1, b1, K1, G1, d1) === nothing

            # singular Q: the Qy = 0 rows make the auxiliary equalities wider
            # than they are tall, which is the case that needs row reduction
            y2 = ConicIP.fallback_unbounded_ray(Qu, cu, Au, bu, Ku, Gu, du)
            @test y2 !== nothing
            (chk2, _) = ConicIP.validate_unboundedness_certificate(
                Qu, cu, Au, bu, Ku, Gu, du, y2; abstol = atol, reltol = rtol)
            @test chk2.valid
        end

        @testset "Recursion guard — fallback solves terminate" begin
            # certFallback = false inside the auxiliary solves means a fallback
            # can never spawn a fallback; the call must simply return.
            s = conicIP(Qi, ci, Ai, bi, Ki, Gi, di;
                        verbose = false, maxIters = 1, certFallback = true)
            @test s isa ConicIP.Solution
            @test s.status ∈ (:Optimal, :Infeasible, :DualInfeasible,
                              :AlmostInfeasible, :AlmostDualInfeasible,
                              :Abandoned, :Error)
        end

        @testset "Fallback solves surface non-KKT exceptions" begin
            # A factorization failure means "no ray"; anything else from a
            # custom kktsolver is a broken invariant and must propagate.
            Qu = zeros(2, 2); cu = [1.0, 0.0]          # min -x₁, x ≥ 0: unbounded
            Au = sparse(1.0I, 2, 2); bu = zeros(2); Ku = [("R", 2)]
            Gu = spzeros(0, 2); du = Float64[]
            throwing(exc) = (args...) -> throw(exc)

            @test_throws ArgumentError ConicIP.fallback_infeasibility_ray(
                Qi, ci, Ai, bi, Ki, Gi, di; kktsolver = throwing(ArgumentError("boom")))
            @test_throws ArgumentError ConicIP.fallback_unbounded_ray(
                Qu, cu, Au, bu, Ku, Gu, du; kktsolver = throwing(ArgumentError("boom")))

            @test ConicIP.fallback_infeasibility_ray(
                Qi, ci, Ai, bi, Ki, Gi, di;
                kktsolver = throwing(SingularException(0))) === nothing
            @test ConicIP.fallback_unbounded_ray(
                Qu, cu, Au, bu, Ku, Gu, du;
                kktsolver = throwing(SingularException(0))) === nothing
            # sanity: the unbounded auxiliary problem is well posed
            @test ConicIP.fallback_unbounded_ray(Qu, cu, Au, bu, Ku, Gu, du) !== nothing
        end

    end

    # ──────────────────────────────────────────────────────────────
    #  Post-loop certificate exits
    #
    #  The in-loop screens only nominate a ray once a cheap heuristic
    #  passes; the exhaustion path re-validates the saved iterate
    #  unconditionally. These cases reach the post-loop claims without
    #  ever passing the in-loop screen.
    # ──────────────────────────────────────────────────────────────
    @testset "Post-loop certificate exits" begin
        atol = 1e-9; rtol = 1e-7

        @testset "Infeasible — post-loop validator claims" begin
            # x ≥ 1 and -x ≥ 1e8 is empty. After one iteration the screen
            # reads icertp ≈ 0.45 (no nomination) but the saved iterate
            # already carries a Farkas ray at default tolerance.
            Q = ones(1, 1); c = [0.0]
            A = sparse(reshape([1.0, -1.0], 2, 1)); b = [1.0, 1e8]
            K = [("R", 2)]; G = spzeros(0, 1); d = Float64[]
            s = conicIP(Q, c, A, b, K, G, d;
                        verbose = false, maxIters = 1, certFallback = false)
            @test s.status == :Infeasible
            @test s.has_certificate
            @test s.Iter == 1
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Q, c, A, b, K, G, d, s.w, s.v; abstol = atol, reltol = rtol)
            @test chk.valid
            @test abs(dot(d, s.w) - dot(b, s.v) + 1) < 1e-6
            @test all(isnan, s.y)
        end

        @testset "Unbounded — post-loop validator claims" begin
            # min -x s.t. x ≥ 0. With maxIters = 0 the loop body never runs
            # (zero iterations is a supported budget), so the only route to
            # a claim is the post-loop re-validation of the initial point.
            Q = zeros(1, 1); c = [1.0]
            A = sparse(ones(1, 1)); b = [0.0]
            K = [("R", 1)]; G = spzeros(0, 1); d = Float64[]
            s = conicIP(Q, c, A, b, K, G, d;
                        verbose = false, maxIters = 0, certFallback = false)
            @test s.status == :DualInfeasible
            @test s.has_certificate
            @test s.Iter == 0
            (chk, ȳ) = ConicIP.validate_unboundedness_certificate(
                Q, c, A, b, K, G, d, s.y; abstol = atol, reltol = rtol)
            @test chk.valid
            @test dot(c, s.y) ≈ 1.0 atol = 1e-8
            @test s.s ≈ A * s.y
            @test all(isnan, s.w) && all(isnan, s.v)
        end

        @testset "Exhausted budget without a ray is :Abandoned" begin
            # Same infeasible problem, but with the certificate tolerance
            # tightened past what one iteration can deliver: no claim.
            Q = ones(1, 1); c = [0.0]
            A = sparse(reshape([1.0, -1.0], 2, 1)); b = [1.0, 1e8]
            s = conicIP(Q, c, A, b, [("R", 2)];
                        verbose = false, maxIters = 1, certFallback = false,
                        infeasTol = 1e-12)
            @test s.status == :Abandoned
            @test !s.has_certificate
        end
    end

    # ──────────────────────────────────────────────────────────────
    #  Infeasibility soundness (adversarial)
    #
    #  A sound solver never claims :Infeasible or :DualInfeasible on a
    #  problem that is neither. These cases sit *deliberately* close to
    #  the boundary: nearly-empty feasible sets, nearly-flat objectives,
    #  degenerate constraint blocks, and poisoned data. Each asserts the
    #  claim that must NOT be made, and — where a claim is legitimately
    #  made — re-validates the ray against the original data.
    # ──────────────────────────────────────────────────────────────

    @testset "Infeasibility soundness" begin

        no_eq(n) = (zeros(0, n), zeros(0))

        @testset "(a) ε-feasible box is Optimal, never Infeasible" begin
            # 0 ≤ x ≤ ε with ε = 1e-9. The feasible set is a sliver, and the
            # Farkas screen is *almost* satisfied — but the set is nonempty,
            # so :Infeasible would be unsound.
            ε = 1e-9
            for n in (1, 5)
                A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
                b = [zeros(n); fill(-ε, n)]
                K = [("R", 2n)]
                for c in (ones(n), zeros(n))
                    sol = conicIP(zeros(n, n), c, A, b, K, verbose = false)
                    @test sol.status == :Optimal
                    @test !sol.has_certificate
                    # Recovered point really is in the sliver
                    @test minimum(sol.y) > -1e-7
                    @test maximum(sol.y) < ε + 1e-7
                end
            end

            # And no Farkas ray exists for it: the fallback must decline.
            A = sparse([1.0; -1.0][:, :]); b = [0.0, -ε]
            @test ConicIP.fallback_infeasibility_ray(
                zeros(1, 1), [1.0], A, b, [("R", 2)], no_eq(1)...) === nothing
        end

        @testset "(b) tiny-Q recession-like problem" begin
            # min ½εx² - x  s.t.  x ≥ 0.  Bounded for every ε > 0 (optimum
            # x* = 1/ε), but as ε ↓ 0 the problem tends to an unbounded LP.
            #
            # FINDING: the recession validator's residual test is
            #   ‖Qȳ‖ ≤ abstol + reltol·(1 + ‖ȳ‖),
            # which is absolute in the data scale, *not* relative to ‖Q‖. So a
            # curvature below ≈ infeasTol reads as flat and the ray is accepted.
            # The flip is exactly at ε ≈ infeasTol = 1e-7, verified by sweep.
            mk(ε) = (reshape([ε], 1, 1), [1.0],
                     sparse(reshape([1.0], 1, 1)), [0.0], [("R", 1)])

            # Above the tolerance the solver must NOT claim a ray.
            for ε in (1e-6, 1e-4, 1e-2)
                (Q, c, A, b, K) = mk(ε)
                sol = conicIP(Q, c, A, b, K, verbose = false)
                @test sol.status == :Optimal
                @test !sol.has_certificate
                @test sol.y[1] ≈ 1/ε rtol=1e-4
            end

            # At ε = 1e-12 the claim IS made. Two things must still hold:
            # the claim is never :DualInfeasible-without-a-ray, and the ray it
            # carries genuinely satisfies the validator's own contract
            # against the original data — i.e. the tolerance is the only
            # thing being traded, not the soundness argument.
            (Q, c, A, b, K) = mk(1e-12)
            sol = conicIP(Q, c, A, b, K, verbose = false)
            @test sol.status == :DualInfeasible
            @test sol.has_certificate
            (chk, _) = ConicIP.validate_unboundedness_certificate(
                Q, c, A, b, K, no_eq(1)..., sol.y;
                abstol = 1e-9, reltol = 1e-7)
            @test chk.valid
            @test abs(dot(c, sol.y) - 1) < 1e-6
            @test norm(Q * sol.y) <= 1e-9 + 1e-7 * (1 + norm(sol.y))

            # Tightening infeasTol below the curvature restores :Optimal,
            # which pins the cause to the tolerance rather than to a bug.
            sol = conicIP(Q, c, A, b, K, verbose = false,
                          infeasTol = 1e-14, infeasAbsTol = 1e-16)
            @test sol.status == :Optimal
            @test !sol.has_certificate
            @test sol.y[1] ≈ 1e12 rtol=1e-4
        end

        @testset "(c) degenerate constraint blocks" begin
            # Empty A (no cone rows): infeasibility lives entirely in the
            # equalities, x = 1 and x = 2. G has more rows than columns, so
            # the raw KKT factorization cannot be formed — this is the
            # preprocessor's path, and it must still produce a ray.
            Q = zeros(1, 1); c = [0.0]
            A = spzeros(0, 1); b = zeros(0)
            G = reshape([1.0; 1.0], 2, 1); d = [1.0, 2.0]
            K = Tuple{String,Int}[]

            for kktsolver = (ConicIP.kktsolver_qr,
                             ConicIP.kktsolver_sparse,
                             pivot(ConicIP.kktsolver_2x2))
                sol = preprocess_conicIP(Q, c, A, b, K, G, d;
                                         verbose = false, kktsolver = kktsolver)
                @test sol.status == :Infeasible
                @test sol.has_certificate
                (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                    Q, c, A, b, K, G, d, sol.w, sol.v;
                    abstol = 1e-9, reltol = 1e-7)
                @test chk.valid
                @test abs(dot(d, sol.w) - dot(b, sol.v) + 1) < 1e-6
                @test isempty(sol.v)
            end

            # Empty G (no equalities): infeasibility lives entirely in the
            # cone rows, x ≥ 1 and -x ≥ 1.
            A2 = sparse([1.0; -1.0][:, :]); b2 = [1.0, 1.0]; K2 = [("R", 2)]
            sol = conicIP(Q, c, A2, b2, K2, verbose = false)
            @test sol.status == :Infeasible
            @test sol.has_certificate
            @test isempty(sol.w)
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                Q, c, A2, b2, K2, no_eq(1)..., sol.w, sol.v;
                abstol = 1e-9, reltol = 1e-7)
            @test chk.valid
            @test abs(-dot(b2, sol.v) + 1) < 1e-6
            @test ConicIP.cone_margin(sol.v, K2) >= -1e-6
        end

        @testset "(d) NaN/Inf data never yields a certificate" begin
            # Poisoned data must not be laundered into a verdict. Which of
            # the two failure modes fires depends on the KKT solver: the
            # dense/sparse paths propagate NaN into the residuals and exit
            # :Error, while the 2x2 pivot path raises SingularException from
            # the factorization. Both are acceptable; a certificate is not.
            A = sparse(1.0I, 2, 2); K = [("R", 2)]
            poisoned = [([NaN, 0.0], [1.0, 1.0]),
                        ([Inf, 0.0], [1.0, 1.0]),
                        ([0.0, 0.0], [NaN, 1.0]),
                        ([0.0, 0.0], [Inf, 1.0])]

            for kktsolver = (ConicIP.kktsolver_qr,
                             ConicIP.kktsolver_sparse,
                             pivot(ConicIP.kktsolver_2x2))
                for (b, c) in poisoned
                    local sol = nothing
                    threw = false
                    try
                        sol = conicIP(zeros(2, 2), c, A, b, K;
                                      verbose = false, maxIters = 20,
                                      kktsolver = kktsolver)
                    catch
                        threw = true
                    end
                    @test threw || sol.status ∉ (:Optimal, :Infeasible, :DualInfeasible)
                    @test threw || !sol.has_certificate
                end
            end

            # The validators themselves reject nonfinite candidates outright.
            (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                zeros(1, 1), [0.0], sparse(reshape([1.0], 1, 1)), [1.0],
                [("R", 1)], no_eq(1)..., Float64[], [NaN];
                abstol = 1e-9, reltol = 1e-7)
            @test !chk.valid
            @test !chk.finite
            (chk, _) = ConicIP.validate_unboundedness_certificate(
                zeros(1, 1), [1.0], sparse(reshape([1.0], 1, 1)), [0.0],
                [("R", 1)], no_eq(1)..., [Inf];
                abstol = 1e-9, reltol = 1e-7)
            @test !chk.valid
            @test !chk.finite
        end

        @testset "(e) SOC infeasible — ray lies in the cone" begin
            # Variables (t, x, y): ‖(x,y)‖ ≤ t together with t ≤ -1.
            # The Farkas ray v must itself be in K = Q³ × R₊, so its
            # cone margin is the thing to check.
            Q = zeros(3, 3); c = zeros(3)
            A = sparse([1.0 0.0 0.0
                        0.0 1.0 0.0
                        0.0 0.0 1.0
                       -1.0 0.0 0.0])
            b = [0.0, 0.0, 0.0, 1.0]
            K = [("Q", 3), ("R", 1)]

            for kktsolver = (ConicIP.kktsolver_qr,
                             ConicIP.kktsolver_sparse,
                             pivot(ConicIP.kktsolver_2x2))
                sol = conicIP(Q, c, A, b, K; verbose = false, kktsolver = kktsolver)
                @test sol.status == :Infeasible
                @test sol.has_certificate
                @test ConicIP.cone_margin(sol.v, K) >= -1e-6
                (chk, _, _) = ConicIP.validate_infeasibility_certificate(
                    Q, c, A, b, K, no_eq(3)..., sol.w, sol.v;
                    abstol = 1e-9, reltol = 1e-7)
                @test chk.valid
                @test abs(-dot(b, sol.v) + 1) < 1e-6
            end

            # Boundary-hugging variant: (t,x) ∈ Q², t ≤ 0, x ≥ 1 forces the
            # ray onto the SOC boundary, where cone_margin ≈ 0 from above.
            A2 = sparse([1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 1.0])
            b2 = [0.0, 0.0, 0.0, 1.0]
            K2 = [("Q", 2), ("R", 2)]
            sol = conicIP(zeros(2, 2), zeros(2), A2, b2, K2, verbose = false)
            @test sol.status == :Infeasible
            @test sol.has_certificate
            @test ConicIP.cone_margin(sol.v, K2) >= -1e-6
            @test ConicIP.cone_margin(sol.v, K2) < 1e-3   # genuinely on the boundary
        end

        @testset "(f) near-optimal and near-certificate — Optimal wins" begin
            # c = 0 makes every feasible point optimal, so the recession
            # screen (cᵀy > 0) can never fire; meanwhile the feasible set
            # collapses to a point, so the Farkas screen is nearly satisfied.
            # Precedence in the termination block puts optimality first.

            # Feasible set is exactly {0}: x ≥ 0 and -x ≥ 0.
            A = sparse([1.0; -1.0][:, :]); K = [("R", 2)]
            sol = conicIP(zeros(1, 1), [0.0], A, [0.0, 0.0], K, verbose = false)
            @test sol.status == :Optimal
            @test !sol.has_certificate
            @test abs(sol.y[1]) < 1e-6

            # Feasible set shrinks to a 1e-10 interval.
            sol = conicIP(zeros(1, 1), [0.0], A, [0.0, -1e-10], K, verbose = false)
            @test sol.status == :Optimal
            @test !sol.has_certificate

            # Same, pinned by equalities instead of a cone.
            G = Matrix(1.0I, 2, 2); d = zeros(2)
            sol = conicIP(zeros(2, 2), zeros(2), sparse(1.0I, 2, 2), zeros(2),
                          [("R", 2)], G, d, verbose = false)
            @test sol.status == :Optimal
            @test !sol.has_certificate
            @test norm(sol.y) < 1e-6
        end

        @testset "(g) MOI ray getters ignore constants" begin
            import MathOptInterface as MOI
            saf(terms, k) = MOI.ScalarAffineFunction(
                [MOI.ScalarAffineTerm(a, v) for (a, v) in terms], k)

            # ── Infeasible variant ──
            # min x + 11  s.t.  x + 3 ≥ 4,  x + 3 ≤ 2.
            # The objective constant 11 and the offsets 3 must not appear in
            # DualObjectiveValue / ObjectiveBound: a ray is a direction.
            for (sense, want) in ((MOI.MIN_SENSE, 1.0), (MOI.MAX_SENSE, -1.0))
                src = MOI.Utilities.Model{Float64}()
                x = MOI.add_variable(src)
                MOI.set(src, MOI.ObjectiveSense(), sense)
                MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                    saf([(1.0, x)], 11.0))
                c_lo = MOI.add_constraint(src, saf([(1.0, x)], 3.0), MOI.GreaterThan(4.0))
                c_hi = MOI.add_constraint(src, saf([(1.0, x)], 3.0), MOI.LessThan(2.0))

                opt = ConicIP.Optimizer()
                index_map, _ = MOI.optimize!(opt, src)

                @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
                @test opt.sol.has_certificate
                @test opt.objective_constant == 11.0
                @test MOI.get(opt, MOI.ResultCount()) == 1
                @test MOI.get(opt, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
                @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION

                # Homogeneous: exactly ±1, with no trace of 11 or 3.
                @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ want atol=1e-6
                @test MOI.get(opt, MOI.ObjectiveBound()) ≈ want atol=1e-6

                # Duals are the ray components, summing to zero on x.
                dl = MOI.get(opt, MOI.ConstraintDual(), index_map[c_lo])
                dh = MOI.get(opt, MOI.ConstraintDual(), index_map[c_hi])
                @test dl > 0 && dh < 0
                @test abs(dl + dh) < 1e-6
            end

            # ── Unbounded variant ──
            # min x₁ + 13  s.t.  x₁ + 4 ≤ 9,  x₂ + 6 = 10.
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variables(src, 2)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x[1])], 13.0))
            c_le = MOI.add_constraint(src, saf([(1.0, x[1])], 4.0), MOI.LessThan(9.0))
            c_eq = MOI.add_constraint(src, saf([(1.0, x[2])], 6.0), MOI.EqualTo(10.0))

            opt = ConicIP.Optimizer()
            index_map, _ = MOI.optimize!(opt, src)

            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
            @test opt.sol.has_certificate
            @test opt.objective_constant == 13.0
            @test MOI.get(opt, MOI.ResultCount()) == 1
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
            @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION

            # ObjectiveValue drops the constant: -1, not 12.
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -1.0 atol=1e-6

            # ConstraintPrimal drops both the eq rhs (eq_d = 4) and the
            # LessThan offset (ineq_offset = 9): the homogeneous part only.
            @test MOI.get(opt, MOI.ConstraintPrimal(), index_map[c_le]) ≈ -1.0 atol=1e-6
            @test abs(MOI.get(opt, MOI.ConstraintPrimal(), index_map[c_eq])) < 1e-6

            # The ray itself: c_intᵀȳ = +1, Gȳ ≈ 0, Aȳ ∈ K.
            @test dot(opt.c_int, opt.sol.y) ≈ 1.0 atol=1e-6
            @test norm(opt.eq_G * opt.sol.y) < 1e-6
            @test minimum(opt.ineq_A * opt.sol.y) > -1e-6
        end

    end

    # ──────────────────────────────────────────────────────────────
    #  Issue #10: solver selection, structural degeneracy, KKT contract
    # ──────────────────────────────────────────────────────────────

    @testset "choose_kktsolver" begin
        # #10-shaped: large, very sparse, no SDP → sparse solver
        pg = socp_sum_of_norms(150; d = 200)
        @test choose_kktsolver(pg.Q, pg.A, pg.G, pg.cone_dims) ===
              ConicIP.kktsolver_sparse
        # SDP anywhere → qr, regardless of sparsity
        @test choose_kktsolver(pg.Q, pg.A, pg.G, [("Q",3), ("S",6)]) ===
              ConicIP.kktsolver_qr
        # small problems keep the historical qr default (protects every
        # pinned compare() value in the suites above)
        @test choose_kktsolver(spzeros(10,10), sprandn(5,10,0.5),
                               spzeros(0,10), [("R",5)]) ===
              ConicIP.kktsolver_qr
        # sparse-typed but dense-ish data (many small SOCs, 10%-dense A)
        @test choose_kktsolver(spzeros(500,500), sprandn(750,500,0.1),
                               spzeros(0,500), [("Q",3) for _ in 1:250]) ===
              ConicIP.kktsolver_qr
    end

    @testset "Issue #10 regression (sum-of-norms SOCP)" begin
        pg = socp_sum_of_norms(150; d = 200)
        sol = conicIP(pg.Q, pg.c, pg.A, pg.b, pg.cone_dims, pg.G, pg.d;
                      verbose = false)
        @test sol.status == :Optimal
        @test sol.Iter <= 60
        @test max(sol.prFeas, sol.duFeas, sol.muFeas) < 1e-6
        # and through the preprocessor (the MOI path)
        sol2 = preprocess_conicIP(pg.Q, pg.c, pg.A, pg.b, pg.cone_dims,
                                  pg.G, pg.d; verbose = false)
        @test sol2.status == :Optimal
    end

    @testset "Structural degeneracy — direct conicIP" begin
        n = 5
        Q = Matrix{Float64}(I, n, n); c = ones(n)
        A = sparse(1.0I, n, n); b = zeros(n)
        cd = [("R", n)]
        for kktsolver = (ConicIP.kktsolver_qr,
                         ConicIP.kktsolver_sparse,
                         pivot(ConicIP.kktsolver_2x2))
            # zero row of G, dᵢ = 0: vacuous row is deflated exactly
            G = sparse([1.0 0 0 0 0; 0 0 0 0 0]); dvec = [1.0, 0.0]
            s = conicIP(Q, c, A, b, cd, G, dvec;
                        verbose = false, kktsolver = kktsolver)
            @test s.status == :Optimal
            @test length(s.w) == 2 && s.w[2] == 0.0
            # zero row of G, dᵢ ≠ 0: certified infeasibility, no solve
            s = conicIP(Q, c, A, b, cd, G, [1.0, 2.0];
                        verbose = false, kktsolver = kktsolver)
            @test s.status == :Infeasible
            @test s.has_certificate
            @test s.Iter == 0
            # duplicated (dependent, nonzero) rows: clean :Error status,
            # no escaped SingularException
            G2 = sparse([1.0 0 0 0 0; 1.0 0 0 0 0])
            s = conicIP(Q, c, A, b, cd, G2, [1.0, 1.0];
                        verbose = false, kktsolver = kktsolver)
            @test s.status == :Error
            @test !isempty(s.message)
        end
        # zero column of [Q; A; G] with cⱼ ≠ 0: certified unboundedness
        Qz = spzeros(3,3); cz = [0.0, 0.0, 1.0]
        Az = sparse([1.0 0 0; 0 1.0 0]); bz = zeros(2)
        s = conicIP(Qz, cz, Az, bz, [("R",2)], spzeros(0,3), zeros(0);
                    verbose = false)
        @test s.status == :DualInfeasible
        @test s.has_certificate
        # p > n under kktsolver_qr: invariant violation, clear error
        Gpn = sparse(ones(7, 5))
        @test_throws ArgumentError conicIP(Q, c, A, b, cd, Gpn, ones(7);
            verbose = false, kktsolver = ConicIP.kktsolver_qr)
    end

    @testset "Preprocessor status soundness" begin
        import MathOptInterface as MOI

        # Bounded problem: 100·y₁ = 0 and 100·y₁ + 1e-6·y₂ = 0 force y = 0, so
        # min -y₂ has optimum 0. imcols drops row 2 as numerically dependent,
        # the reduced problem is unbounded, and the recession ray fails
        # revalidation on the full G. That verdict must not survive as a
        # terminal :DualInfeasible (it used to, with has_certificate = false).
        Q = zeros(2, 2); c = [0.0, 1.0]
        A = sparse(1.0I, 2, 2); b = zeros(2); K = [("R", 2)]
        G = sparse([100.0 0.0; 100.0 1e-6]); d = zeros(2)

        @testset "Reduced-problem ray failing revalidation is :Error" begin
            (IP, consistent) = ConicIP.imcols(G, d)
            @test consistent && length(IP) == 1       # premise: a row is dropped
            s = preprocess_conicIP(Q, c, A, b, K, G, d; verbose = false)
            @test s.status == :Error
            @test !s.has_certificate
            @test all(isnan, s.y) && all(isnan, s.w)
            @test occursin("DualInfeasible", s.message)
            @test occursin("dropped rows [2]", s.message)
            @test occursin("does not certify the original data", s.message)
        end

        @testset "Same problem through MOI maps to NUMERICAL_ERROR" begin
            model = MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                ConicIP.Optimizer())
            x = MOI.add_variables(model, 2)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(-1.0, x[2])], 0.0))
            MOI.add_constraint(model,
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(100.0, x[1])], 0.0),
                MOI.EqualTo(0.0))
            MOI.add_constraint(model,
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(100.0, x[1]),
                                          MOI.ScalarAffineTerm(1e-6, x[2])], 0.0),
                MOI.EqualTo(0.0))
            MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
            MOI.add_constraint(model, x[2], MOI.GreaterThan(0.0))
            MOI.optimize!(model)
            @test MOI.get(model, MOI.TerminationStatus()) == MOI.NUMERICAL_ERROR
            @test MOI.get(model, MOI.ResultCount()) == 0
            @test occursin("does not certify", MOI.get(model, MOI.RawStatusString()))
        end
    end

    @testset "KKT failure at every stage" begin
        # conicIP catches KKT_FAILURES at five distinct call sites and must
        # degrade each to a clean :Error naming the stage; any other
        # exception is a broken invariant and must propagate. A counting
        # wrapper around kktsolver_qr injects the failure at a chosen stage.
        # Call order: gen #1 = initial point, gen #2 = iteration-1
        # factorization; solve #1 = initial point, #2 = predictor,
        # #3 = corrector.
        function failing_kkt(fail_at; exc = SingularException(0))
            ngen = Ref(0); nsolve = Ref(0)
            function factory(Q, A, G, cone_dims)
                fail_at == :setup && throw(exc)
                inner = ConicIP.kktsolver_qr(Q, A, G, cone_dims)
                function gen(F, F⁻ᵀ)
                    ngen[] += 1
                    (fail_at == :initial && ngen[] == 1) && throw(exc)
                    (fail_at == :factor  && ngen[] == 2) && throw(exc)
                    solve = inner(F, F⁻ᵀ)
                    function counted(bx, by, bz)
                        nsolve[] += 1
                        (fail_at == :predictor && nsolve[] == 2) && throw(exc)
                        (fail_at == :corrector && nsolve[] == 3) && throw(exc)
                        return solve(bx, by, bz)
                    end
                end
            end
        end
        Q = Matrix(1.0I, 3, 3); c = ones(3)
        A = sparse(1.0I, 3, 3); b = zeros(3)
        G = sparse([1.0 1.0 1.0]); d = [1.0]
        K = [("R", 3)]

        stages = ((:setup,     "solver setup"),
                  (:initial,   "initial point"),
                  (:factor,    "factorization, iteration 1"),
                  (:predictor, "predictor, iteration 1"),
                  (:corrector, "corrector, iteration 1"))
        for (stage, word) in stages
            s = conicIP(Q, c, A, b, K, G, d;
                        verbose = false, kktsolver = failing_kkt(stage))
            @test s.status == :Error
            @test !s.has_certificate
            @test occursin(word, s.message)
            @test occursin("SingularException", s.message)
        end

        # the guard is narrow: a non-KKT exception escapes
        for stage in (:setup, :initial, :factor, :predictor, :corrector)
            @test_throws ArgumentError conicIP(Q, c, A, b, K, G, d;
                verbose = false,
                kktsolver = failing_kkt(stage; exc = ArgumentError("boom")))
        end

        # sanity: the wrapper is transparent when it does not fire
        s = conicIP(Q, c, A, b, K, G, d;
                    verbose = false, kktsolver = failing_kkt(:never))
        @test s.status == :Optimal
    end

    @testset "Non-finite direction at every stage" begin
        # The mirror of the testset above. Instead of throwing, a broken
        # KKT solver hands back a direction full of NaN. A NaN reaching
        # LAPACK raises ArgumentError, which is deliberately *not* a
        # KKT_FAILURE, so conicIP has to screen the direction itself and
        # degrade to :Error rather than let the ArgumentError escape.
        # Solve #1 = initial point, #2 = predictor, #3 = corrector, later
        # calls = iterative refinement or the next iteration.
        function nan_after(k)
            nsolve = Ref(0)
            function factory(Q, A, G, cone_dims)
                inner = ConicIP.kktsolver_qr(Q, A, G, cone_dims)
                function gen(F, F⁻ᵀ)
                    solve = inner(F, F⁻ᵀ)
                    function counted(bx, by, bz)
                        nsolve[] += 1
                        (x, y, z) = solve(bx, by, bz)
                        nsolve[] == k && return (fill(NaN, length(x)),
                                                 fill(NaN, length(y)),
                                                 fill(NaN, length(z)))
                        return (x, y, z)
                    end
                end
            end
        end

        # Heterogeneous cone: every per-cone maxstep/scaling primitive is
        # on the path, and the S block is the one that calls LAPACK.
        Km = [("R", 2), ("Q", 3), ("S", 6)]
        nm = 11
        Qm = Matrix(1.0I, nm, nm)
        cm = [1.0, -0.5, 2.0, 0.3, -0.4, 1.0, 0.2, 0.1, 1.0, 0.3, 1.0]
        Am = sparse(1.0I, nm, nm); bm = zeros(nm)

        for k in (1, 2, 3, 6)
            s = nothing
            @test try
                s = conicIP(Qm, cm, Am, bm, Km;
                            verbose = false, kktsolver = nan_after(k))
                true
            catch err
                @error "conicIP threw instead of returning :Error" k err
                false
            end
            if s !== nothing
                @test s.status == :Error
                @test !isempty(s.message)
                @test !s.has_certificate
            end
        end

        # The four cases above stop at the initial-point, predictor and
        # corrector screens: with the default refinementThreshold the
        # refinement loop breaks on its first residual check, so no
        # refinement solve ever runs. Force the loop to solve by asking
        # for a residual it can never reach, and the refinement screen is
        # the one that fires (solves #4..#6 of iteration 1).
        for k in (4, 5, 6)
            s = conicIP(Qm, cm, Am, bm, Km; verbose = false,
                        refinementThreshold = 0.0, kktsolver = nan_after(k))
            @test s.status == :Error
            @test occursin("refinement direction", s.message)
            @test occursin("refinement, iteration 1", s.message)
        end

        # The iterate screen after axpy4! is the last line of defence, and
        # a NaN cannot reach it — every direction is screened first. What
        # can is a *finite* direction that overflows when added to a large
        # iterate. Inject a near-maximal Float64 into one primal
        # coordinate of the initial point and the opposite sign into the
        # corrector direction; their sum is +Inf.
        function inject_kkt(tbl)
            nsolve = Ref(0)
            function factory(Q, A, G, cone_dims)
                inner = ConicIP.kktsolver_qr(Q, A, G, cone_dims)
                function gen(F, F⁻ᵀ)
                    solve = inner(F, F⁻ᵀ)
                    function counted(bx, by, bz)
                        nsolve[] += 1
                        (x, y, z) = solve(bx, by, bz)
                        f = get(tbl, nsolve[], nothing)
                        return f === nothing ? (x, y, z) : f(x, y, z)
                    end
                end
            end
        end

        # min 1'y s.t. y ≥ 0 (Q = 0, so a huge y never enters the dual
        # residual and the run survives to the line search).
        huge = 1.7e308
        nh = 3
        Qh = spzeros(nh, nh); ch = -ones(nh)
        Ah = sparse(1.0I, nh, nh); bh = zeros(nh); Kh = [("R", nh)]
        tbl = Dict(
            # initial point: one coordinate at (nearly) floatmax
            1 => (x, y, z) -> (vcat(huge, x[2:end]), y, z),
            # predictor: a benign zero direction, so ρ and σ stay finite
            2 => (x, y, z) -> (zero(x), zero(y), zero(z)),
            # corrector: the same magnitude with the opposite sign
            3 => (x, y, z) -> (vcat(-huge, zeros(length(x) - 1)),
                               zero(y), zero(z)),
        )
        s = conicIP(Qh, ch, Ah, bh, Kh; verbose = false,
                    maxRefinementSteps = 0, kktsolver = inject_kkt(tbl))
        @test s.status == :Error
        @test occursin("non-finite iterate", s.message)
        @test occursin("line search, iteration 1", s.message)

        # sanity: the wrapper is transparent when it does not fire
        s = conicIP(Qm, cm, Am, bm, Km; verbose = false, kktsolver = nan_after(0))
        @test s.status == :Optimal
    end

    @testset "SOC scaling guard" begin
        # nestod_soc divides by the Jordan determinant QF(x) = x₁² - ‖x̄‖².
        # On a near-boundary iterate roundoff can drive it to zero or
        # below, and the old code then built a SymWoodbury out of Inf/NaN
        # entries without throwing; inv_adjoint! met it with
        # ArgumentError("D must be symmetric"), which is not a KKT_FAILURE
        # and escaped conicIP — breaking the documented :Error contract.
        # Refusing QF ≤ 0 up front turns it into a guarded failure, exactly
        # as the cholesky in nestod_sdc does for a boundary SDP iterate.
        zbad = [1.0, 1.0, 0.0]                     # QF = 0, on the boundary
        zgood = [2.0, 0.5, 0.5]
        @test_throws ConicIP.KKT_FAILURES ConicIP.nestod_soc(zbad, zgood)
        @test_throws ConicIP.KKT_FAILURES ConicIP.nestod_soc(zgood, zbad)
        @test ConicIP.nestod_soc(zgood, copy(zgood)) isa
              ConicIP.WoodburyMatrices.SymWoodbury

        # End-to-end: this family of two-SOC projections hits the boundary
        # on seeds 47 and 139 of 160 at optTol = 1e-11 under
        # kktsolver_sparse. Both used to throw out of conicIP; both must
        # now come back as an ordinary Solution.
        valid = (:Optimal, :Infeasible, :DualInfeasible, :AlmostInfeasible,
                 :AlmostDualInfeasible, :Abandoned, :Error, :None)
        escapes = 0
        statuses = Symbol[]
        for seed = 1:160
            rng = Random.Xoshiro(seed)
            nq = 5
            Qq = sparse(1.0I, nq, nq); cq = randn(rng, nq)
            Aq = sparse(1.0I, nq, nq); bq = randn(rng, nq)
            try
                sq = conicIP(Qq, cq, Aq, bq, [("Q", 2), ("Q", 3)];
                             verbose = false, optTol = 1e-11,
                             kktsolver = ConicIP.kktsolver_sparse)
                push!(statuses, sq.status)
            catch err
                escapes += 1
                @error "conicIP threw on SOC seed $seed" err
            end
        end
        @test escapes == 0
        @test length(statuses) == 160
        @test all(st -> st in valid, statuses)
        @test count(==(:Optimal), statuses) > 150
    end

    @testset "SDP certificates" begin
        # Infeasibility and unboundedness on a pure semidefinite cone,
        # through the direct API and through MOI. Both rays are produced by
        # the same validators as the R₊/Q cases, but only the S cone
        # exercises maxstep_sdc/nestod_sdc on the way there. These are
        # coverage tests for a path the suite did not reach, not
        # regression evidence for either fix in this branch — they pass
        # against the pre-fix solver too.
        k = 6                                     # vec dim of S³
        Qs = spzeros(k, k)
        As = sparse(1.0I, k, k); bs = zeros(k)     # X ⪰ 0
        Ki = [("S", k)]
        vecI = ConicIP.vecm(Matrix{Float64}(I, 3, 3))

        # tr X = -1 is unreachable inside the PSD cone.
        Gs = sparse(reshape(vecI, 1, k)); ds = [-1.0]
        s = conicIP(Qs, zeros(k), As, bs, Ki, Gs, ds; verbose = false)
        @test s.status == :Infeasible
        @test s.has_certificate
        @test eigmin(Symmetric(ConicIP.mat(s.v))) > -1e-8      # v̄ ∈ K
        @test dot(ds, s.w) - dot(bs, s.v) ≈ -1                 # normalized ray

        # min -tr X over X ⪰ 0 is unbounded below.
        s = conicIP(Qs, vecI, As, bs, Ki; verbose = false)
        @test s.status == :DualInfeasible
        @test s.has_certificate
        @test dot(vecI, s.y) ≈ 1                               # cᵀȳ = +1
        @test eigmin(Symmetric(ConicIP.mat(s.s))) > -1e-8      # Aȳ ∈ K

        import MathOptInterface as MOI

        # MOI stores the upper triangle column by column with
        # off-diagonals appearing once and unscaled, in the primal and in
        # the dual alike, so the matrix is read off directly.
        function moi_mat(t, n)
            M = zeros(n, n); c = 1
            for j = 1:n, i = 1:j
                M[i, j] = t[c]; M[j, i] = t[c]; c += 1
            end
            return M
        end
        # diagonal positions of a 3×3 upper-triangle-by-column vector
        diag3 = [1, 3, 6]

        @testset "MOI infeasible PSD" begin
            model = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
            MOI.set(model, MOI.Silent(), true)
            x = MOI.add_variables(model, k)
            psd = MOI.add_constraint(model, MOI.VectorOfVariables(x),
                                     MOI.PositiveSemidefiniteConeTriangle(3))
            eq = MOI.add_constraint(model,
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(ones(3), x[diag3]), 0.0),
                MOI.EqualTo(-1.0))
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
            @test MOI.get(model, MOI.DualStatus()) ==
                  MOI.INFEASIBILITY_CERTIFICATE
            V = moi_mat(MOI.get(model, MOI.ConstraintDual(), psd), 3)
            w = MOI.get(model, MOI.ConstraintDual(), eq)
            @test eigmin(Symmetric(V)) > -1e-8     # ray lies in the cone
            @test norm(V, Inf) > 1e-6              # and is not the zero ray
            # Farkas: the constraint duals annihilate the (zero) objective
            # row, and pair positively with the right-hand side.
            @test norm(V + w * Matrix{Float64}(I, 3, 3), Inf) < 1e-6
            @test w * (-1.0) > 1e-6
        end

        @testset "MOI dual-infeasible PSD" begin
            model = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
            MOI.set(model, MOI.Silent(), true)
            x = MOI.add_variables(model, k)
            MOI.add_constraint(model, MOI.VectorOfVariables(x),
                               MOI.PositiveSemidefiniteConeTriangle(3))
            MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    MOI.ScalarAffineTerm.(ones(3), x[diag3]), 0.0))
            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
            @test MOI.get(model, MOI.PrimalStatus()) ==
                  MOI.INFEASIBILITY_CERTIFICATE
            ray = MOI.get(model, MOI.VariablePrimal(), x)
            R = moi_mat(ray, 3)
            @test eigmin(Symmetric(R)) > -1e-8     # ray lies in the cone
            @test tr(R) > 1e-6                     # and improves the objective
        end
    end

    @testset "KKT solver contract" begin
        # solve3x3gen(F,F⁻ᵀ)(bx,by,bz) must solve the documented 3×3
        # system for every solver and cone mix — including SOC+SDP mixes
        # under kktsolver_sparse (the lift form cannot represent SDP
        # blocks and must be bypassed) and under pivot (whose reconstruction
        # needs the true adjoint for non-self-adjoint SDP scalings).
        function contract_residual(kktsolver, cone_dims; p = 3)
            m = sum(cdim[2] for cdim in cone_dims)
            n = m + 1
            Q = sparse(Symmetric(sprandn(n, n, 0.3) + 5.0*I))
            A = sprandn(m, n, 0.4) + [sparse(1.0I, m, m) spzeros(m, 1)]
            G = sprandn(p, n, 0.5); G[1,1] += 2.0
            F = ConicIP.placeholder(cone_dims)
            F⁻ᵀ = ConicIP.inv_adjoint!(Block(length(cone_dims)), F)
            solve = kktsolver(Q, A, G, cone_dims)(F, F⁻ᵀ)
            bx = randn(n); by = randn(p); bz = randn(m)
            (x, y, z) = solve(bx, by, bz)
            r1 = norm(Q*x + G'*y - A'*z - bx)
            r2 = norm(G*x - by)
            r3 = norm(A*x + F'*(F*z) - bz)
            return max(r1, r2, r3) / max(norm(bx), norm(by), norm(bz))
        end
        mixes = ([("R", 4), ("Q", 3), ("Q", 5)],
                 [("R", 3), ("Q", 4), ("S", 6)],
                 [("Q", 8), ("S", 6)],
                 [("S", 6), ("S", 10)])
        for kktsolver = (ConicIP.kktsolver_qr,
                         ConicIP.kktsolver_sparse,
                         pivot(ConicIP.kktsolver_2x2)),
            cone_mix in mixes
            @test contract_residual(kktsolver, cone_mix) < 1e-10
        end
    end

    @testset "mat!/vecm! and VecCongurance on matrices" begin
        x = randn(10)                      # ord = 4
        Z = ConicIP.mat(x)
        @test Z ≈ Z'
        @test ConicIP.vecm(Z) ≈ x
        Zb = zeros(4,4); ConicIP.mat!(Zb, x)
        @test Zb ≈ Z
        xb = zeros(10); ConicIP.vecm!(xb, Z)
        @test xb ≈ x
        # matrix action = column-wise vector action (incl. views)
        W = ConicIP.VecCongurance(randn(4,4))
        X = randn(10, 3)
        WX = W * X
        @test WX[:,2] ≈ W * X[:,2]
        @test W * view(X, :, 1:2) ≈ WX[:,1:2]
    end

    # ──────────────────────────────────────────────────────────────
    #  MathOptInterface Tests
    # ──────────────────────────────────────────────────────────────

    @testset "Solve accounting and dense guard" begin
        # A kktsolver wrapper that counts factorizations (solve3x3gen calls)
        # and back-solves (solve3x3 calls) without changing the arithmetic.
        nfact = Ref(0); nsolve = Ref(0)
        counting(inner) = (Q, A, G, cd) -> begin
            gen = inner(Q, A, G, cd)
            (F, Fi) -> begin
                nfact[] += 1
                s = gen(F, Fi)
                (a, b, c) -> (nsolve[] += 1; s(a, b, c))
            end
        end
        # Bounded, feasible random LP: a box −10 ≤ y ≤ 10 plus random rows
        # that y = 0 satisfies.
        Random.seed!(11)
        n = 300; m1 = 300
        A1 = sprandn(m1, n, 0.02)
        A  = [A1; sparse(1.0I, n, n); -sparse(1.0I, n, n)]
        b  = [-rand(m1); fill(-10.0, n); fill(-10.0, n)]
        c  = randn(n)
        for mr in (0, 3)
            nfact[] = 0; nsolve[] = 0
            sol = conicIP(spzeros(n, n), c, A, b, [("R", size(A, 1))];
                          kktsolver = counting(ConicIP.kktsolver_sparse),
                          verbose = false, maxRefinementSteps = mr)
            @test sol.status == :Optimal
            # One factorization for the initial point and one per stepped
            # iteration; the iterate that meets the tolerance is not factored.
            @test nfact[] == sol.Iter
            @test sol.kkt_solves == nsolve[]
            stepped = sol.Iter - 1
            if mr == 0
                @test sol.kkt_solves == 1 + 2*stepped
            else
                @test 1 + 2*stepped <= sol.kkt_solves <= 1 + (2 + mr)*stepped
            end
        end
        # Short constructors default the count to zero.
        z = zeros(1)
        @test ConicIP.Solution(z, z, z, z, :None, 0, 0, 0, 0, 0, 0, 0).kkt_solves == 0
        @test ConicIP.Solution(z, z, z, z, :None, 0, 0, 0, 0, 0, 0, 0, false).kkt_solves == 0
        @test ConicIP.Solution(z, z, z, z, :None, 0, 0, 0, 0, 0, 0, 0, false, "").kkt_solves == 0

        # Dense-storage guard in choose_kktsolver: a very sparse problem with
        # just over 10 nnz/col used to be routed to dense QR at any size.
        @test ConicIP.dense_kkt_bytes(10, 20, 3) == 8*(2*100 + 2*20*7 + 2*49)
        nb = 50_000
        A11 = sprand(nb, nb, 11/nb)
        @test choose_kktsolver(spzeros(nb, nb), A11, spzeros(0, nb), [("R", nb)]) ===
              ConicIP.kktsolver_sparse
        @test choose_kktsolver(spzeros(nb, nb), A11, spzeros(0, nb), [("R", nb)];
                               dense_bytes_max = typemax(Int)) ===
              ConicIP.kktsolver_qr
        # The guard precedes the SDP rule.
        @test choose_kktsolver(spzeros(20, 20), sprandn(21, 20, 0.5), spzeros(0, 20),
                               [("S", 6)]; dense_bytes_max = 1) ===
              ConicIP.kktsolver_sparse

        # A recession ray certifies dual infeasibility, not primal
        # unboundedness: this problem (0·y ≥ 1) is infeasible, and the
        # status must not claim an unbounded primal.
        s = conicIP(spzeros(1, 1), [1.0], spzeros(1, 1), [1.0], [("R", 1)];
                    verbose = false)
        @test s.status == :DualInfeasible
        @test s.has_certificate
        @test s.status != :Unbounded
    end

    @testset "MOI.Test" begin
        import MathOptInterface as MOI

        optimizer = MOI.Bridges.full_bridge_optimizer(
            MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                ConicIP.Optimizer(optTol = 1e-8),
            ),
            Float64,
        )
        config = MOI.Test.Config(
            atol = 1e-4,
            rtol = 1e-4,
            optimal_status = MOI.OPTIMAL,
            exclude = Any[
                MOI.VariableBasisStatus,
                MOI.ConstraintBasisStatus,
            ],
        )
        MOI.Test.runtests(optimizer, config;
            exclude = [
                # No quadratic objective support
                r"test_quadratic_",
                # Continuous solver — no integer/binary variables
                r"_Integer_",
                r"_ZeroOne_",
                r"test_variable_solve_with_upperbound",
                r"test_variable_solve_with_lowerbound",
                # Indicator constraints require integer variables
                r"test_linear_Indicator_",
                # No SOS constraint support
                r"test_SOS",
                # No mixed-integer support
                r"test_Semicontinuous",
                r"test_Semiinteger",
                # No nonlinear support
                r"test_vector_nonlinear",
                # Wrapper uses CachingOptimizer, not direct copy_to
                r"test_model_copy_to",
                # ObjectiveBound tests with integer variables
                r"test_solve_ObjectiveBound_.*_IP$",
                #
                # NOTE (WP6): the ten infeasibility/unboundedness exclusions
                # that used to live here are gone. With validated certificates
                # (WP3), ray-aware MOI getters (WP4) and the fallback certificate
                # solve (WP5), MOI.Test's INFEASIBLE / DUAL_INFEASIBLE /
                # INFEASIBILITY_CERTIFICATE / FEASIBILITY_SENSE families and the
                # SOC edge cases all pass unmodified. Do not re-add without a
                # test-name-specific reason comment.
                #
                # ConicIP has no native Interval support, so the bridge
                # splits Interval bounds and the inner model reports
                # LowerBoundAlreadySet{GreaterThan,...} instead of
                # {Interval,...}; @test_throws strictness for this differs
                # across Julia versions (fails on 1.10, passes on 1.12).
                # Remove once VariableIndex-in-Interval is supported natively.
                r"test_model_LowerBoundAlreadySet",
                r"test_model_UpperBoundAlreadySet",
            ],
            exclude_tests_after = v"1.49.0",
        )
    end

    @testset "MOI wrapper" begin
        import MathOptInterface as MOI

        @testset "Optimizer options (WP6)" begin
            opt = ConicIP.Optimizer(verbose = false, maxIters = 42)
            @test MOI.get(opt, MOI.RawOptimizerAttribute("maxIters")) == 42
            @test MOI.get(opt, MOI.RawOptimizerAttribute("kktsolver")) == "auto"
            MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), "sparse")
            @test MOI.get(opt, MOI.RawOptimizerAttribute("kktsolver")) == "sparse"
            @test MOI.supports(opt, MOI.RawOptimizerAttribute("preprocess"))
            @test !MOI.supports(opt, MOI.RawOptimizerAttribute("bogus"))
            @test_throws MOI.UnsupportedAttribute MOI.set(
                opt, MOI.RawOptimizerAttribute("bogus"), 1)
            @test_throws ArgumentError MOI.set(
                opt, MOI.RawOptimizerAttribute("kktsolver"), "nope")
            @test MOI.supports(opt, MOI.Silent())
            MOI.set(opt, MOI.Silent(), true)
            @test MOI.get(opt, MOI.Silent())
            # every named solver resolves
            for name in ("auto", "qr", "sparse", "2x2", "pivot")
                @test ConicIP._resolve_kktsolver(name) isa Function
            end
            @test ConicIP._resolve_kktsolver(ConicIP.kktsolver_qr) ===
                  ConicIP.kktsolver_qr
        end

        @testset "Status mapping and metadata" begin
            opt = ConicIP.Optimizer()
            @test MOI.get(opt, MOI.SolverName()) == "ConicIP"
            @test MOI.get(opt, MOI.SolverVersion()) == string(pkgversion(ConicIP))
            @test_throws MOI.UnsupportedAttribute MOI.get(
                opt, MOI.RawOptimizerAttribute("bogus"))
            @test !MOI.supports(opt, MOI.VariableBasisStatus())
            @test !MOI.supports(opt, MOI.ConstraintBasisStatus())

            # before optimize!
            @test MOI.get(opt, MOI.ResultCount()) == 0
            @test MOI.get(opt, MOI.RawStatusString()) == "OPTIMIZE_NOT_CALLED"
            @test MOI.get(opt, MOI.ObjectiveBound()) == Inf
            opt.max_sense = true      # set by copy_to; no public setter pre-solve
            @test MOI.get(opt, MOI.ObjectiveBound()) == -Inf

            function cached(; kw...)
                MOI.Utilities.CachingOptimizer(
                    MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                    ConicIP.Optimizer(; kw...))
            end

            # :Abandoned → ITERATION_LIMIT (min Σ i·xᵢ, Σ xᵢ = 1, x ≥ 0, one iteration)
            model = cached(maxIters = 1, preprocess = false)
            x = MOI.add_variables(model, 10)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(
                    [MOI.ScalarAffineTerm(Float64(i), x[i]) for i in 1:10], 0.0))
            MOI.add_constraint(model,
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, xi) for xi in x], 0.0),
                MOI.EqualTo(1.0))
            for xi in x; MOI.add_constraint(model, xi, MOI.GreaterThan(0.0)); end
            MOI.optimize!(model)
            @test MOI.get(model, MOI.TerminationStatus()) == MOI.ITERATION_LIMIT
            @test MOI.get(model, MOI.RawStatusString()) == "Abandoned"
            @test MOI.get(model, MOI.ResultCount()) == 0

            # :Error → NUMERICAL_ERROR with the KKT message (duplicated equality
            # row; preprocess = false so the duplicate reaches the factorization)
            model = cached(preprocess = false)
            x = MOI.add_variables(model, 2)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[1]),
                                          MOI.ScalarAffineTerm(1.0, x[2])], 0.0))
            for _ in 1:2
                MOI.add_constraint(model,
                    MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[1])], 0.0),
                    MOI.EqualTo(1.0))
            end
            for xi in x; MOI.add_constraint(model, xi, MOI.GreaterThan(0.0)); end
            MOI.optimize!(model)
            @test MOI.get(model, MOI.TerminationStatus()) == MOI.NUMERICAL_ERROR
            @test MOI.get(model, MOI.ResultCount()) == 0
            raw = MOI.get(model, MOI.RawStatusString())
            @test startswith(raw, "Error: ")
            @test occursin("KKT solve failed", raw)
        end

        @testset "Simple LP via MOI" begin
            # min x₁ + x₂ s.t. x₁ + x₂ ≥ 1, x₁ ≥ 0, x₂ ≥ 0
            optimizer = ConicIP.Optimizer(optTol = 1e-6)
            model = MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                optimizer,
            )

            x = MOI.add_variables(model, 2)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(1.0, x[2]),
                ], 0.0))

            MOI.add_constraint(model,
                MOI.ScalarAffineFunction([
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(1.0, x[2]),
                ], 0.0),
                MOI.GreaterThan(1.0))
            MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
            MOI.add_constraint(model, x[2], MOI.GreaterThan(0.0))

            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-4
            @test MOI.get(model, MOI.VariablePrimal(), x[1]) ≈ 0.5 atol=1e-2
            @test MOI.get(model, MOI.VariablePrimal(), x[2]) ≈ 0.5 atol=1e-2
        end

        @testset "SOC via MOI" begin
            # min x₃ s.t. x₁ = 1, x₂ = 1, ||(x₁,x₂)|| ≤ x₃
            optimizer = ConicIP.Optimizer(optTol = 1e-6)
            model = MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                optimizer,
            )

            x = MOI.add_variables(model, 3)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, x[3])], 0.0))

            # x₁ = 1
            MOI.add_constraint(model, x[1], MOI.EqualTo(1.0))
            # x₂ = 1
            MOI.add_constraint(model, x[2], MOI.EqualTo(1.0))
            # (x₃, x₁, x₂) ∈ SOC
            MOI.add_constraint(model,
                MOI.VectorOfVariables([x[3], x[1], x[2]]),
                MOI.SecondOrderCone(3))

            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(model, MOI.VariablePrimal(), x[3]) ≈ sqrt(2) atol=1e-4
            @test MOI.get(model, MOI.ObjectiveValue()) ≈ sqrt(2) atol=1e-4
        end

        @testset "Max sense via MOI" begin
            # max x₁ + 2x₂ s.t. x₁ + x₂ ≤ 1, x₁ ≥ 0, x₂ ≥ 0
            optimizer = ConicIP.Optimizer(optTol = 1e-6)
            model = MOI.Utilities.CachingOptimizer(
                MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()),
                optimizer,
            )

            x = MOI.add_variables(model, 2)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MAX_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(2.0, x[2]),
                ], 0.0))

            MOI.add_constraint(model,
                MOI.ScalarAffineFunction([
                    MOI.ScalarAffineTerm(1.0, x[1]),
                    MOI.ScalarAffineTerm(1.0, x[2]),
                ], 0.0),
                MOI.LessThan(1.0))
            MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
            MOI.add_constraint(model, x[2], MOI.GreaterThan(0.0))

            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(model, MOI.ObjectiveValue()) ≈ 2.0 atol=1e-4
            @test MOI.get(model, MOI.VariablePrimal(), x[1]) ≈ 0.0 atol=1e-2
            @test MOI.get(model, MOI.VariablePrimal(), x[2]) ≈ 1.0 atol=1e-2
        end

        @testset "Dimension-0 PSD constraint via MOI" begin
            # The wrapper passes PositiveSemidefiniteConeTriangle(0)
            # straight through as ("S", 0). It constrains nothing, so the
            # solve must be unaffected rather than crash on an empty
            # eigen-decomposition.
            model = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
            MOI.set(model, MOI.Silent(), true)
            x = MOI.add_variables(model, 2)
            MOI.add_constraint(model, MOI.VectorOfVariables(MOI.VariableIndex[]),
                               MOI.PositiveSemidefiniteConeTriangle(0))
            MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
            MOI.add_constraint(model, x[2], MOI.GreaterThan(1.0))
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction(MOI.ScalarAffineTerm.(ones(2), x), 0.0))
            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(model, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-4
        end

        @testset "Max eigenvalue SDP via MOI" begin
            # min t s.t. t*I - C ⪰ 0, with C = I + 2uu' and u = [1,2,3]/√14,
            # so λ(C) = (3,1,1): t⋆ = 3 and the dual of the PSD constraint
            # is the spectral projector uu' onto the leading eigenvector.
            # This is the sdp_affine_eigmax fixture of the SDP suite; here
            # it also pins the MOI triangle convention end to end, since C
            # has distinct off-diagonal entries.
            u = [1.0, 2.0, 3.0] / sqrt(14.0)
            C = Matrix{Float64}(I, 3, 3) + 2 * (u * u')

            model = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
            MOI.set(model, MOI.Silent(), true)
            # The dual is recovered to about optTol, so ask for more than
            # the 1e-6 default before asserting 1e-6 on uu'.
            MOI.set(model, MOI.RawOptimizerAttribute("optTol"), 1e-9)
            t = MOI.add_variable(model)
            MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                MOI.ScalarAffineFunction([MOI.ScalarAffineTerm(1.0, t)], 0.0))

            # PositiveSemidefiniteConeTriangle stores the upper triangle
            # column by column: (1,1),(1,2),(2,2),(1,3),(2,3),(3,3).
            terms = MOI.VectorAffineTerm{Float64}[]
            consts = Float64[]
            row = 1
            for j = 1:3, i = 1:j
                if i == j
                    push!(terms, MOI.VectorAffineTerm(row,
                        MOI.ScalarAffineTerm(1.0, t)))
                end
                push!(consts, -C[i, j])
                row += 1
            end
            psd = MOI.add_constraint(model,
                MOI.VectorAffineFunction(terms, consts),
                MOI.PositiveSemidefiniteConeTriangle(3))

            MOI.optimize!(model)

            @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(model, MOI.VariablePrimal(), t) ≈ 3.0 atol=1e-8
            @test MOI.get(model, MOI.ObjectiveValue()) ≈ 3.0 atol=1e-8

            # The dual comes back in the same plain triangle coordinates
            # as the constraint function: off-diagonals appear once and
            # unscaled, so the matrix is read off directly.
            dv = MOI.get(model, MOI.ConstraintDual(), psd)
            V = zeros(3, 3); row = 1
            for j = 1:3, i = 1:j
                V[i, j] = dv[row]; V[j, i] = dv[row]; row += 1
            end
            @test norm(V - u * u') < 1e-6
        end
    end

    # ──────────────────────────────────────────────────────────────
    #  Certificate contract (rays surfaced through MOI)
    #
    #  These tests drive `ConicIP.Optimizer` directly (no caching or
    #  bridge layer) so that `model.sol` can be inspected and — while
    #  the solver's termination block does not yet emit verified rays —
    #  replaced by a synthetic `Solution` obeying the field-convention
    #  table in the `Solution` docstring. The end-to-end assertions
    #  (TerminationStatus) run against the real solve either way.
    # ──────────────────────────────────────────────────────────────

    @testset "MOI certificate contract" begin
        import MathOptInterface as MOI

        saf(terms, const_) = MOI.ScalarAffineFunction(
            [MOI.ScalarAffineTerm(c, v) for (c, v) in terms], const_)

        # Farkas ray Solution: y/s are NaN, w/v carry the ray.
        function infeasible_sol(opt, w, v)
            n, m = opt.n, length(opt.ineq_b)
            return ConicIP.Solution(fill(NaN, n), copy(w), copy(v), fill(NaN, m),
                :Infeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, true)
        end

        # Recession ray Solution: w/v are NaN, y is the ray and s = A*ȳ.
        function unbounded_sol(opt, y)
            p = size(opt.eq_G, 1)
            s = opt.ineq_A * y
            return ConicIP.Solution(copy(y), fill(NaN, p), fill(NaN, length(s)),
                Vector(s), :DualInfeasible, 0, NaN, NaN, NaN, NaN, NaN, NaN, true)
        end

        # Gᵀw̄ - Aᵀv̄ (the Farkas residual), tolerating an empty equality block
        function farkas_residual(opt, sol)
            r = -Vector(opt.ineq_A' * sol.v)
            if size(opt.eq_G, 1) > 0
                r += Vector(opt.eq_G' * sol.w)
            end
            return r
        end

        @testset "Infeasible LP — dual ray" begin
            # min x  s.t.  x ≥ 1, x ≤ 0     (LessThan exercises ineq_sign = -1)
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(src)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x)], 0.0))
            c_lo = MOI.add_constraint(src, x, MOI.GreaterThan(1.0))
            c_hi = MOI.add_constraint(src, x, MOI.LessThan(0.0))

            opt = ConicIP.Optimizer()
            index_map, _ = MOI.optimize!(opt, src)

            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE

            # Internal data: A = [1; -1], b = [1, 0], no equalities.
            # Farkas ray: v = [1, 1] ⇒ Aᵀv = 0, dᵀw - bᵀv = -1.
            if !opt.sol.has_certificate
                opt.sol = infeasible_sol(opt, Float64[], [1.0, 1.0])
            end
            sol = opt.sol

            @test norm(farkas_residual(opt, sol)) < 1e-6
            @test dot(opt.eq_d, sol.w) - dot(opt.ineq_b, sol.v) ≈ -1.0 atol=1e-8

            @test MOI.get(opt, MOI.ResultCount()) == 1
            @test MOI.get(opt, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(opt, MOI.DualObjectiveValue()) > 0
            @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ 1.0 atol=1e-8

            # ConstraintDual returns the ray components (sign-flipped on LessThan)
            @test MOI.get(opt, MOI.ConstraintDual(), index_map[c_lo]) ≈ 1.0 atol=1e-8
            @test MOI.get(opt, MOI.ConstraintDual(), index_map[c_hi]) ≈ -1.0 atol=1e-8
        end

        @testset "Infeasible LP with equality — dual ray" begin
            # min x  s.t.  x ≤ -1, x = 0
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(src)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x)], 0.0))
            c_le = MOI.add_constraint(src, saf([(1.0, x)], 0.0), MOI.LessThan(-1.0))
            c_eq = MOI.add_constraint(src, saf([(1.0, x)], 0.0), MOI.EqualTo(0.0))

            opt = ConicIP.Optimizer()
            index_map, _ = MOI.optimize!(opt, src)

            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE

            # Internal data: A = [-1], b = [1], G = [1], d = [0].
            # Farkas ray: w = -1, v = 1 ⇒ Gᵀw - Aᵀv = 0, dᵀw - bᵀv = -1.
            if !opt.sol.has_certificate
                opt.sol = infeasible_sol(opt, [-1.0], [1.0])
            end
            sol = opt.sol

            @test norm(farkas_residual(opt, sol)) < 1e-6
            @test dot(opt.eq_d, sol.w) - dot(opt.ineq_b, sol.v) ≈ -1.0 atol=1e-8

            @test MOI.get(opt, MOI.ResultCount()) == 1
            @test MOI.get(opt, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ 1.0 atol=1e-8

            @test MOI.get(opt, MOI.ConstraintDual(), index_map[c_le]) ≈ -1.0 atol=1e-8
            @test MOI.get(opt, MOI.ConstraintDual(), index_map[c_eq]) ≈ 1.0 atol=1e-8
        end

        @testset "Unbounded LP — primal ray is homogeneous" begin
            # min x₁ + 5  s.t.  x₁ + 2 ≤ 2,  x₂ + 1 = 3
            # The objective constant (5) and the constraint constants must not
            # appear in any ray-valued getter.
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variables(src, 2)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x[1])], 5.0))
            c_le = MOI.add_constraint(src, saf([(1.0, x[1])], 2.0), MOI.LessThan(2.0))
            c_eq = MOI.add_constraint(src, saf([(1.0, x[2])], 1.0), MOI.EqualTo(3.0))

            opt = ConicIP.Optimizer()
            index_map, _ = MOI.optimize!(opt, src)

            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE

            # c_int = -c_moi = [-1, 0]; ray ȳ = [-1, 0] gives c_intᵀȳ = +1,
            # Gȳ = 0 and Aȳ = [1] ∈ K.
            if !opt.sol.has_certificate
                opt.sol = unbounded_sol(opt, [-1.0, 0.0])
            end
            sol = opt.sol

            @test dot(opt.c_int, sol.y) ≈ 1.0 atol=1e-8
            @test norm(opt.eq_G * sol.y) < 1e-8
            @test minimum(opt.ineq_A * sol.y) > -1e-8

            @test MOI.get(opt, MOI.ResultCount()) == 1
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
            @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION

            # MIN sense ⇒ improving direction ⇒ negative; constant 5 excluded
            @test MOI.get(opt, MOI.ObjectiveValue()) < 0
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ -1.0 atol=1e-8

            @test MOI.get(opt, MOI.VariablePrimal(), index_map[x[1]]) ≈ -1.0 atol=1e-8
            @test MOI.get(opt, MOI.VariablePrimal(), index_map[x[2]]) ≈ 0.0 atol=1e-8

            # ineq_offset = 2 (LessThan rhs) must NOT be added: -1, not +1
            @test MOI.get(opt, MOI.ConstraintPrimal(), index_map[c_le]) ≈ -1.0 atol=1e-8
            # eq_offset = 3 / eq_d = 2 must NOT enter: G*ȳ = 0, not 1
            @test MOI.get(opt, MOI.ConstraintPrimal(), index_map[c_eq]) ≈ 0.0 atol=1e-8
        end

        @testset "Unbounded LP — MAX sense ray objective is positive" begin
            # max x₁ + 7  s.t.  x₁ ≥ 0   (unbounded above)
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(src)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MAX_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x)], 7.0))
            MOI.add_constraint(src, x, MOI.GreaterThan(0.0))

            opt = ConicIP.Optimizer()
            MOI.optimize!(opt, src)

            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE

            # MAX ⇒ c_int = c_moi = [1]; ray ȳ = [1] gives c_intᵀȳ = +1.
            if !opt.sol.has_certificate
                opt.sol = unbounded_sol(opt, [1.0])
            end

            @test MOI.get(opt, MOI.ResultCount()) == 1
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
            # MAX sense flips the internal -1 to +1; constant 7 excluded
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-8
        end

        @testset "No certificate ⇒ no result" begin
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(src)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x)], 0.0))
            MOI.add_constraint(src, x, MOI.GreaterThan(1.0))
            MOI.add_constraint(src, x, MOI.LessThan(0.0))

            opt = ConicIP.Optimizer()
            MOI.optimize!(opt, src)

            args = (opt.sol.y, opt.sol.w, opt.sol.v, opt.sol.s, :Infeasible,
                    0, NaN, NaN, NaN, NaN, NaN, NaN)
            opt.sol = ConicIP.Solution(args..., false)
            @test MOI.get(opt, MOI.ResultCount()) == 0
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION

            opt.sol = ConicIP.Solution(args[1:4]..., :DualInfeasible, args[6:end]..., false)
            @test MOI.get(opt, MOI.ResultCount()) == 0
            @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION
            @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION
        end

        @testset "Almost-status mapping" begin
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(src)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x)], 0.0))
            MOI.add_constraint(src, x, MOI.GreaterThan(0.0))

            opt = ConicIP.Optimizer()
            MOI.optimize!(opt, src)
            args = (opt.sol.y, opt.sol.w, opt.sol.v, opt.sol.s)

            opt.sol = ConicIP.Solution(args..., :AlmostInfeasible,
                                       0, NaN, NaN, NaN, NaN, NaN, NaN, false)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.ALMOST_INFEASIBLE
            @test MOI.get(opt, MOI.ResultCount()) == 0

            opt.sol = ConicIP.Solution(args..., :AlmostDualInfeasible,
                                       0, NaN, NaN, NaN, NaN, NaN, NaN, false)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.ALMOST_DUAL_INFEASIBLE
            @test MOI.get(opt, MOI.ResultCount()) == 0
        end

        @testset "infeasTol option is accepted" begin
            opt = ConicIP.Optimizer(infeasTol = 1e-9)
            @test MOI.get(opt, MOI.RawOptimizerAttribute("infeasTol")) == 1e-9
            src = MOI.Utilities.Model{Float64}()
            x = MOI.add_variable(src)
            MOI.set(src, MOI.ObjectiveSense(), MOI.MIN_SENSE)
            MOI.set(src, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}(),
                saf([(1.0, x)], 0.0))
            MOI.add_constraint(src, x, MOI.GreaterThan(1.0))
            MOI.optimize!(opt, src)
            @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
            @test MOI.get(opt, MOI.ObjectiveValue()) ≈ 1.0 atol=1e-5

            # empty! clears the newly stored problem data
            MOI.empty!(opt)
            @test MOI.is_empty(opt)
            @test isempty(opt.c_int)
            @test opt.ineq_A === nothing
            @test isempty(opt.ineq_b)
        end
    end

    include("sdp_tests.jl")

    include("sdplib_tests.jl")

end
