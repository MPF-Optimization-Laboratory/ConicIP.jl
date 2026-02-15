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

        Ad = deepcopy(A)
        Ad[1] = zeros(4, 4)

        @test A[1] != zeros(4, 4)
        @test A * ones(9) ≈ Matrix(A) * ones(9)

        @test Matrix(Diagonal(ones(9)) + A) ≈ Matrix(Diagonal(ones(9))) + Matrix(A)

    end

    @testset "Misc Tests" begin

        A = rand(3, 3)
        Z = ConicIP.VecCongurance(A)

        @test Z * ones(6, 1) ≈ Matrix(Z) * ones(6, 1)
        @test Matrix(Z' * Z) ≈ Matrix(Z)' * Matrix(Z)
        @test inv(Z) * ones(6, 1) ≈ Matrix(Z) \ ones(6, 1)
        @test size(Z, 1) == 6
        @test sparse(Z) ≈ Matrix(Z)

        # Test conic steplength - if steplength is infinity
        X = -Matrix{Float64}(I, 3, 3)
        D = Matrix{Float64}(I, 3, 3)
        @test ConicIP.maxstep_sdc(ConicIP.vecm(X), ConicIP.vecm(D)) == Inf

    end

    @testset "Box Constrained QP, H = I" begin

        Random.seed!(0)

        n = 1000
        H = 0.5 * Id(n)
        c = reshape(collect(1.0:n), :, 1)

        A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
        b = -ones(2 * n, 1)
        ∇f = x -> H * (x - c)

        function kktsolver_2x2_box(Q, A, G, cone_dims)
            function solve2x2gen(F, F⁻¹)
                v = inv(F[1] * F[1]).diag
                D = Diagonal(v[1:n] + v[n+1:end])
                invHD = inv(Diagonal(H.diag + D.diag))
                return (rhs, rhs2) -> (invHD * rhs, zeros(0, 1))
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
            a = ones(n, 1)
            A = [spzeros(1, n); sparse(1.0I, n, n)]
            b = [-1; zeros(n, 1)]

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
            c = reshape(collect(1.0:n), :, 1)

            A = [sparse(1.0I, n, n);      # R
                 spzeros(1, n);            # Q
                 sparse(1.0I, n, n)]       #

            b = [zeros(n, 1);
                 -1.0;
                 zeros(n, 1)]

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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = ones(1, n)
            d = ones(1, 1)

            sol = conicIP(H, H * c,
                          A, b, [("R", n)],
                          G, d,
                          kktsolver = kktsolver,
                          optTol = optTol)

            ystar = sol.y

            ysol = zeros(n, 1)
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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = ones(1, n)
            d = ones(1, 1)

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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = ones(1, n)
            d = ones(1, 1)

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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = rand(6, n)
            d = zeros(6, 1)

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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = rand(6, n)
            d = zeros(6, 1)

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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = rand(6, n)
            G = [G; G]
            d = zeros(6, 1)
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
            c = -ones(2 * n, 1)

            A = sparse(1.0I, n, n)
            A = [A A]
            d = zeros(n, 1)

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
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = zeros(1, n)
            G[1, 1] = 1
            G = [G; G]
            d = reshape([1; -1], :, 1)

            ystatus = preprocess_conicIP(H, H * c,
                                         A, b, [("R", n)],
                                         G, d,
                                         kktsolver = kktsolver,
                                         optTol = optTol).status

            @test ystatus == :Infeasible

        end

        @testset "Infeasible ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = reshape(collect(1.0:n), :, 1)

            A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
            b = [ones(n, 1); ones(n, 1)]

            sol = conicIP(H, H * c,
                          A, b, [("R", 2 * n)],
                          kktsolver = kktsolver,
                          optTol = optTol)

            @test sol.status == :Infeasible

        end

        @testset "Infeasible (With linear constraints) ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = randn(n)
            H = H * H'
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            G = [1 zeros(1, 9)]
            d = -ones(1, 1)

            sol = conicIP(H, H * c,
                          A, b, [("R", n)],
                          G, d,
                          kktsolver = kktsolver,
                          optTol = optTol)

            @test sol.status == :Infeasible

        end

        @testset "Unbounded ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = zeros(n, n)
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n, n)
            b = zeros(n, 1)

            sol = conicIP(H, c,
                          A, b, [("R", n)],
                          kktsolver = kktsolver,
                          optTol = optTol)

            @test sol.status == :Unbounded

        end

        @testset "Bad Input ($(nameof(typeof(kktsolver))))" begin

            Random.seed!(0)

            n = 10
            H = zeros(n, n)
            c = reshape(collect(1.0:n), :, 1)

            A = sparse(1.0I, n + 2, n + 2)
            b = zeros(n, 1)

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
        c = reshape(ConicIP.vecm(diagm(0 => [1.0; 1; 1; -1; -1; -1])), :, 1)

        A = sparse(1.0I, 21, 21)
        b = zeros(21, 1)

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
            c_obj = -ones(n, 1)

            # SOC constraint: [1; x₁; x₂; x₃] ∈ SOC → ||x[1:3]|| ≤ 1
            A_soc = [spzeros(1, n); sparse(1.0I, 3, n)[1:3, :]]
            b_soc = [-1.0; zeros(3, 1)]

            # NonNeg variable cone: x ≥ 0
            A_nn = sparse(1.0I, n, n)
            b_nn = zeros(n, 1)

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
        b = randn(5, 1)
        R, consistent = ConicIP.imcols(A, b)
        @test length(R) == rank(A)
        @test consistent

        # Redundant rows
        A2 = [A; A[1:1, :] + A[2:2, :]]
        b2 = [b; b[1:1, :] + b[2:2, :]]
        R2, consistent2 = ConicIP.imcols(A2, b2)
        @test length(R2) == rank(Matrix(A2))
        @test consistent2

        # Inconsistent system
        A3 = [A; A[1:1, :]]
        b3 = [b; b[1:1, :] .+ 100]
        R3, consistent3 = ConicIP.imcols(A3, b3)
        @test !consistent3
    end

    # ──────────────────────────────────────────────────────────────
    #  MathOptInterface Tests
    # ──────────────────────────────────────────────────────────────

    @testset "MOI wrapper" begin
        import MathOptInterface as MOI

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
    end

end
