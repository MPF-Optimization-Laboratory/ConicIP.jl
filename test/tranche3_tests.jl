# Tests for the Tranche 3 preparation work (LDLᵀ diagnostics and retry,
# Solution fields, centrality correctors). Included from runtests.jl inside
# the outer testset.
@testset "Tranche 3 preparation" begin
  MOI = ConicIP.MOI

  # The KKT-contract problem generator of runtests.jl, with a feasible and
  # bounded QP on the same data (y0 random, s0 the cone identity).
  function t3_contract_case(cone_dims; p = 3)
    m = sum(cdim[2] for cdim in cone_dims)
    n = m + 1
    Q = sparse(Symmetric(sprandn(n, n, 0.3) + 5.0*I))
    A = sprandn(m, n, 0.4) + [sparse(1.0I, m, m) spzeros(m, 1)]
    G = sprandn(p, n, 0.5); G[1,1] += 2.0
    F = ConicIP.placeholder(cone_dims)
    F⁻ᵀ = ConicIP.inv_adjoint!(Block(length(cone_dims)), F)
    bx = randn(n); by = randn(p); bz = randn(m)
    y0 = randn(n)
    s0 = zeros(m)
    off = 0
    for (t, k) in cone_dims
      if t == "R"; s0[off+1:off+k] .= 1.0
      elseif t == "Q"; s0[off+1] = 2.0
      else
        r = round(Int, (sqrt(1 + 8k) - 1) / 2)
        s0[off+1:off+k] = ConicIP.vecm(Matrix(1.0I, r, r))
      end
      off += k
    end
    return (; Q, A, G, b = A*y0 - s0, d = G*y0, c = randn(n), F, F⁻ᵀ,
              bx, by, bz, n, m, p)
  end
  function t3_contract_residual(solve, cs)
    (x, y, z) = solve(cs.bx, cs.by, cs.bz)
    r1 = norm(cs.Q*x + cs.G'*y - cs.A'*z - cs.bx)
    r2 = norm(cs.G*x - cs.by)
    r3 = norm(cs.A*x + cs.F'*(cs.F*z) - cs.bz)
    return max(r1, r2, r3) / max(norm(cs.bx), norm(cs.by), norm(cs.bz))
  end
  t3_mixes = ([("R", 4), ("Q", 3), ("Q", 5)],
              [("R", 3), ("Q", 4), ("S", 6)],
              [("Q", 8), ("S", 6)],
              [("S", 6), ("S", 10)],
              [("Q", 12), ("Q", 7), ("R", 5)],
              [("Q", 30)])

  @testset "LDL diagnostics and retry" begin
    # Default hook: no diagnostics for an arbitrary level-3 callable
    @test ConicIP.kkt_diagnostics(x -> x) === nothing

    # Well-conditioned R + small Q (dense) + lifted Q mix: nothing repaired,
    # no bumps, inertia = n positive Q pivots + one +1 pivot per lift.
    Random.seed!(31)
    cone_dims = [("R", 4), ("Q", 3), ("Q", 8)]
    cs = t3_contract_case(cone_dims)
    gen = ConicIP.kktsolver_ldl(cs.Q, cs.A, cs.G, cone_dims)
    s3 = gen(cs.F, cs.F⁻ᵀ)
    @test s3 isa ConicIP.LDLSolve3x3
    dg = ConicIP.kkt_diagnostics(s3)
    @test dg isa ConicIP.LDLDiagnostics
    @test dg.repaired == 0 && dg.refactors == 0
    @test dg.repaired_total == 0 && dg.refactors_total == 0
    @test dg.pos_inertia == cs.n + 1                 # one lifted cone
    @test dg.δp == 1e-8 && dg.δe == 1e-8 && dg.δc == 0.0
    @test t3_contract_residual(s3, cs) < 1e-10
    @test isfinite(dg.last_residual)

    # The contract holds after a manual bump too, on every cone mix, and
    # the bump multiplies the base shift by retry_factor.
    for mix in t3_mixes
      csm = t3_contract_case(mix)
      sm = ConicIP.kktsolver_ldl(csm.Q, csm.A, csm.G, mix)(csm.F, csm.F⁻ᵀ)
      sm.bump!()
      dm = ConicIP.kkt_diagnostics(sm)
      @test dm.refactors == 1 && dm.refactors_total == 1
      @test dm.δp == 1e-7 && dm.δe == 1e-7
      @test t3_contract_residual(sm, csm) < 1e-10
    end

    # Forced repairs: Q = 0 with the natural ordering puts n zero pivots
    # first, each replaced by dynamic regularization.
    n = 6
    Q0 = spzeros(n, n); G0 = spzeros(0, n); A0 = sparse(1.0I, n, n)
    cd0 = [("R", n)]
    F0 = ConicIP.placeholder(cd0)
    F0⁻ᵀ = ConicIP.inv_adjoint!(Block(1), F0)
    pat0() = ConicIP._ldl_pattern(Q0, A0, G0, cd0; perm_hint = collect(1:2n))
    sr = ConicIP.kktsolver_ldl(Q0, A0, G0, cd0; static_reg = 0.0,
                               pattern = pat0())(F0, F0⁻ᵀ)
    dr = ConicIP.kkt_diagnostics(sr)
    @test dr.repaired >= n
    @test dr.repaired_total == dr.repaired
    @test dr.δp == 0.0 && dr.refactors == 0

    # Retry: a large dynamic_delta makes the repaired factorization a poor
    # preconditioner; with retry_max = 2 the shifts are bumped from zero
    # to shift_floor and the solve improves.
    mk(retry) = ConicIP.kktsolver_ldl(Q0, A0, G0, cd0; static_reg = 0.0,
                                      dynamic_delta = 1e-2, retry_max = retry,
                                      pattern = pat0())
    bx = randn(n); by = Float64[]; bz = randn(n)
    s_off = mk(0)(F0, F0⁻ᵀ)
    (x0, _, z0) = s_off(bx, by, bz)
    d_off = ConicIP.kkt_diagnostics(s_off)
    @test d_off.refactors == 0 && d_off.refactors_total == 0
    @test d_off.δp == 0.0
    @test d_off.last_residual > 1e-13 * (1 + norm([bx; bz]))   # retry would trigger

    gen2 = mk(2)
    s_on = gen2(F0, F0⁻ᵀ)
    (x1, _, z1) = s_on(bx, by, bz)
    d_on = ConicIP.kkt_diagnostics(s_on)
    @test d_on.refactors >= 1
    @test d_on.refactors <= 2
    @test d_on.refactors_total == d_on.refactors
    @test d_on.last_residual <= d_off.last_residual
    @test 0 < d_on.δp <= 1e-4 && d_on.δp == d_on.δe
    @test d_on.repaired == 0                          # bumped pivots need no repair
    # The returned solution satisfies the unregularized system
    res1 = max(norm(Q0*x1 - A0'*z1 - bx), norm(A0*x1 + F0'*(F0*z1) - bz)) /
           max(norm(bx), norm(bz))
    @test res1 < 1e-10
    @test res1 <= max(norm(Q0*x0 - A0'*z0 - bx), norm(A0*x0 + F0'*(F0*z0) - bz)) /
                  max(norm(bx), norm(bz))

    # Bumped shifts do not persist: the next factorization restarts at the
    # base shift, while the solve totals carry on.
    s_on2 = gen2(F0, F0⁻ᵀ)
    d_on2 = ConicIP.kkt_diagnostics(s_on2)
    @test d_on2 === d_on                              # one record per solver
    @test d_on2.δp == 0.0 && d_on2.δe == 0.0
    @test d_on2.refactors == 0
    @test d_on2.refactors_total >= 1
    # A zero base shift starts at shift_floor and is capped at shift_max
    s_on2.bump!()
    @test d_on2.δp == 1e-8 && d_on2.refactors == 1
    for _ in 1:10; s_on2.bump!(); end
    @test d_on2.δp == 1e-4 && d_on2.δe == 1e-4

    # Retry off (the default) means no code-path change: on the contract
    # mixes, with and without equilibration, the bare default and a solver
    # with `retry_max = 0` passed explicitly follow the same trajectory
    # (status, Iter, kkt_solves), return the same point, and never
    # refactorize.
    # Historical evidence (2026-09-07, macOS arm64, Julia 1.12.7, this seed
    # and generator): before the diagnostics and retry code existed the six
    # mixes gave, for equilibrate = true / false, (status, Iter, kkt_solves)
    # of (Optimal, 7, 13)/(Optimal, 7, 13), (7, 13)/(7, 13), (8, 17)/(7, 13),
    # (7, 15)/(7, 15), (8, 15)/(8, 15), (7, 14)/(7, 13), and the default run
    # reproduced every `y` bitwise. Those literal values are not asserted:
    # they depend on the RNG stream and on the platform's floating point.
    Random.seed!(20260907)
    ks_off = (Q, A, G, cd) -> ConicIP.kktsolver_ldl(Q, A, G, cd; retry_max = 0)
    for mix in t3_mixes
      csb = t3_contract_case(mix)
      ConicIP.kktsolver_ldl(csb.Q, csb.A, csb.G, mix)(csb.F, csb.F⁻ᵀ)(csb.bx, csb.by, csb.bz)
      for eqb in (true, false)
        sol = conicIP(csb.Q, csb.c, csb.A, csb.b, mix, csb.G, csb.d;
                      kktsolver = ConicIP.kktsolver_ldl, equilibrate = eqb,
                      verbose = false)
        sol0 = conicIP(csb.Q, csb.c, csb.A, csb.b, mix, csb.G, csb.d;
                       kktsolver = ks_off, equilibrate = eqb, verbose = false)
        @test sol.status == sol0.status == :Optimal
        @test sol.Iter == sol0.Iter
        @test sol.kkt_solves == sol0.kkt_solves
        @test isapprox(sol.y, sol0.y; rtol = 1e-6)
        @test sol.kkt_refactors == 0 && sol0.kkt_refactors == 0
        @test sol.kkt_repaired == sol0.kkt_repaired
      end
    end
  end

  @testset "Solution fields" begin
    # 12/13/14/15-argument constructors default the new tail
    a12 = (zeros(2), zeros(1), zeros(1), zeros(1), :Optimal, 1, 0.0,
           0.0, 0.0, 0.0, 0.0, 0.0)
    for s in (ConicIP.Solution(a12...),
              ConicIP.Solution(a12..., false),
              ConicIP.Solution(a12..., false, "msg"),
              ConicIP.Solution(a12..., false, "msg", 4))
      @test isnan(s.rEq) && isnan(s.rGap)
      @test s.kkt_repaired == 0 && s.kkt_refactors == 0
    end
    @test ConicIP.Solution(a12..., false, "msg", 4).kkt_solves == 4

    # Independent recomputation of the two residuals with the named formulas
    function t3_recompute(sol, Q, c, A, b, G, d; offset = 0.0)
      y, v, s = sol.y, sol.v, sol.s
      pobj = 0.5 * dot(y, Q * y) - dot(c, y)
      rGap = abs(dot(v, s)) / (1 + abs(pobj + offset))
      rEq  = isempty(d) ? 0.0 :
             norm(G * y - d) / (1 + max(norm(d), norm(abs.(G) * abs.(y))))
      return (rEq, rGap)
    end
    relclose(a, b) = abs(a - b) <= 1e-12 * max(abs(a), abs(b), 1e-300) ||
                     (a == 0 && b == 0)

    Random.seed!(7)
    n = 6
    Q = sparse(Symmetric(sprandn(n, n, 0.4) + 3.0*I))
    A = sparse(1.0I, n, n); b = zeros(n)          # y ≥ 0
    G = sparse([1.0 1.0 1.0 0.0 0.0 0.0; 0.0 0.0 1.0 1.0 1.0 1.0])
    d = [1.0, 2.0]
    c = randn(n)
    cone_dims = [("R", n)]
    for eqb in (false, true)
      sol = conicIP(Q, c, A, b, cone_dims, G, d; equilibrate = eqb,
                    verbose = false, kktsolver = ConicIP.kktsolver_ldl)
      @test sol.status == :Optimal
      @test sol.rEq <= sol.prFeas
      @test sol.rGap < 1e-6
      (rEq, rGap) = t3_recompute(sol, Q, c, A, b, G, d)
      @test relclose(sol.rEq, rEq)
      @test relclose(sol.rGap, rGap)
      @test sol.kkt_repaired == 0 && sol.kkt_refactors == 0
    end

    # Through the presolve, with a singleton equality row fixing y₁
    Gs = sparse([1.0 0.0 0.0 0.0 0.0 0.0; 1.0 1.0 1.0 1.0 1.0 1.0])
    ds = [0.5, 3.0]
    for eqb in (false, true)
      sol = preprocess_conicIP(Q, c, A, b, cone_dims, Gs, ds;
                               equilibrate = eqb, verbose = false)
      @test sol.status == :Optimal
      @test abs(sol.y[1] - 0.5) < 1e-8
      @test sol.rEq <= sol.prFeas
      @test sol.rGap < 1e-6
      (rEq, rGap) = t3_recompute(sol, Q, c, A, b, Gs, ds)
      @test relclose(sol.rEq, rEq)
      @test relclose(sol.rGap, rGap)
    end

    # Deflation path: variable 3 is absent from Q, A and G with c₃ = 0
    Qd = sparse(Diagonal([1.0, 2.0, 0.0]))
    Ad = sparse([1.0 0.0 0.0; 0.0 1.0 0.0])
    bd = [-1.0, -1.0]
    cd = [1.0, -1.0, 0.0]
    for eqb in (false, true)
      sol = conicIP(Qd, cd, Ad, bd, [("R", 2)]; equilibrate = eqb, verbose = false)
      @test sol.status == :Optimal
      @test isfinite(sol.rEq) && isfinite(sol.rGap)
      @test sol.rGap < 1e-6
      @test sol.y[3] == 0.0
    end

    # Verbose output carries the kkt column and a 0/0 cell under kktsolver_ldl
    out = mktemp() do path, io
      redirect_stdout(io) do
        conicIP(Q, c, A, b, cone_dims, G, d; verbose = true,
                kktsolver = ConicIP.kktsolver_ldl)
      end
      flush(io)
      read(path, String)
    end
    @test occursin("kkt", out)
    @test occursin("refine", out)
    @test occursin("0/0", out)
    # and the dense solver leaves the column blank
    out_qr = mktemp() do path, io
      redirect_stdout(io) do
        conicIP(Q, c, A, b, cone_dims, G, d; verbose = true,
                kktsolver = ConicIP.kktsolver_qr)
      end
      flush(io)
      read(path, String)
    end
    @test occursin("kkt", out_qr)
    @test !occursin("0/0", out_qr)

    # MOI: assemble_only stops after assembly; RelativeGap is sol.rGap
    opt = ConicIP.Optimizer()
    model = MOI.Utilities.CachingOptimizer(
      MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), opt)
    MOI.set(model, MOI.Silent(), true)
    MOI.set(model, MOI.RawOptimizerAttribute("assemble_only"), true)
    x = MOI.add_variables(model, 2)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    f = MOI.ScalarQuadraticFunction(
      [MOI.ScalarQuadraticTerm(2.0, x[1], x[1]),
       MOI.ScalarQuadraticTerm(1.0, x[1], x[2]),
       MOI.ScalarQuadraticTerm(2.0, x[2], x[2])],
      [MOI.ScalarAffineTerm(-1.0, x[2])], 0.5)
    MOI.set(model, MOI.ObjectiveFunction{typeof(f)}(), f)
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.EqualTo(1.0))
    MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
    MOI.add_constraint(model, x[2], MOI.GreaterThan(0.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.ResultCount()) == 0
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMIZE_NOT_CALLED
    @test occursin("assembly only", MOI.get(model, MOI.RawStatusString()))
    @test opt.Q_int !== nothing && size(opt.Q_int) == (2, 2)
    @test opt.Q_int[1, 1] == 2.0 && opt.Q_int[1, 2] == 1.0 && opt.Q_int[2, 2] == 2.0
    @test opt.c_int == [0.0, 1.0]
    @test size(opt.eq_G) == (1, 2) && opt.eq_d == [1.0]
    @test size(opt.ineq_A) == (2, 2) && opt.ineq_b == [0.0, 0.0]
    @test opt.cone_dims == [("R", 2)]
    @test isfinite(opt.assembly_time)
    @test MOI.supports(opt, MOI.RelativeGap())
    @test isnan(MOI.get(model, MOI.RelativeGap()))
    MOI.set(model, MOI.RawOptimizerAttribute("assemble_only"), false)
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    gap = MOI.get(model, MOI.RelativeGap())
    @test isfinite(gap) && gap < 1e-6
    @test gap == opt.sol.rGap
  end

  @testset "Centrality correctors" begin
    βmin = ConicIP.GONDZIO_βmin; βmax = ConicIP.GONDZIO_βmax
    @test ConicIP.GONDZIO_δα == 0.1 && ConicIP.GONDZIO_γ == 0.1
    @test βmin == 0.1 && βmax == 10.0

    # ── clip_spectral! against dense references ──
    Random.seed!(41)
    clip(w, lo, hi, cd) = ConicIP.clip_spectral!(similar(w), w, lo, hi, cd)

    # R: entrywise clamp; identity inside the box is exact
    w = randn(7)
    @test clip(w, -0.5, 0.5, [("R", 7)]) == clamp.(w, -0.5, 0.5)
    @test clip(w, -10.0, 10.0, [("R", 7)]) == w
    @test_throws ArgumentError clip(w, 1.0, 0.0, [("R", 7)])

    # Q: the Jordan eigenvalues are the extreme eigenvalues of the arrow
    # matrix Arw(w) = [w₁ w̄ᵀ; w̄ w₁I]; the clipped element keeps the frame
    # (w̄ direction) and has the clamped extreme eigenvalues.
    arrow(w) = [w[1] w[2:end]'; w[2:end] w[1]*Matrix(1.0I, length(w)-1, length(w)-1)]
    for k in (2, 3, 6)
      w = randn(k); w[1] = 0.3 * norm(w[2:end])           # outside the cone
      E = eigen(Symmetric(arrow(w)))
      λlo, λhi = E.values[1], E.values[end]
      @test λlo ≈ w[1] - norm(w[2:end]) && λhi ≈ w[1] + norm(w[2:end])
      lo, hi = 0.05, 0.6 * λhi
      o = clip(w, lo, hi, [("Q", k)])
      Eo = eigen(Symmetric(arrow(o)))
      @test Eo.values[1] ≈ clamp(λlo, lo, hi) atol = 1e-12
      @test Eo.values[end] ≈ clamp(λhi, lo, hi) atol = 1e-12
      @test norm(o[2:end] / norm(o[2:end]) - w[2:end] / norm(w[2:end])) < 1e-12
      # eigenvectors of the extreme eigenvalues coincide up to sign
      @test abs(abs(dot(E.vectors[:, 1], Eo.vectors[:, 1])) - 1) < 1e-10
      @test abs(abs(dot(E.vectors[:, end], Eo.vectors[:, end])) - 1) < 1e-10
      # idempotent, and identity inside the box (exact)
      @test clip(o, lo, hi, [("Q", k)]) ≈ o atol = 1e-14
      @test clip(o, lo - 1, hi + 1, [("Q", k)]) == o
    end
    # both eigenvalues clipped to the same value: the result is a multiple
    # of the cone identity; a zero w̄ is handled
    wq = [1.0, 0.2, -0.1]
    @test clip(wq, 3.0, 3.0, [("Q", 3)]) ≈ [3.0, 0.0, 0.0]
    @test clip([2.0, 0.0, 0.0], 0.5, 1.0, [("Q", 3)]) == [1.0, 0.0, 0.0]
    @test clip([2.0], 0.5, 1.0, [("Q", 1)]) == [1.0]

    # S: eigenvalues of mat before/after
    for r in (2, 4)
      Ws = Symmetric(randn(r, r)); Ws = Matrix(Ws)
      w = ConicIP.vecm(Ws)
      Λ = eigvals(Symmetric(ConicIP.mat(w)))
      lo, hi = -0.2, 0.7
      k = length(w)
      o = clip(w, lo, hi, [("S", k)])
      @test eigvals(Symmetric(ConicIP.mat(o))) ≈ clamp.(Λ, lo, hi) atol = 1e-12
      # same eigenvectors: mat(o) and mat(w) commute
      @test norm(ConicIP.mat(o) * ConicIP.mat(w) - ConicIP.mat(w) * ConicIP.mat(o)) < 1e-12
      @test clip(o, lo, hi, [("S", k)]) ≈ o atol = 1e-13         # idempotent
      @test clip(w, minimum(Λ) - 1, maximum(Λ) + 1, [("S", k)]) == w   # exact identity
    end

    # mixed cone product in one call
    cdm = [("R", 2), ("Q", 3), ("S", 3)]
    wm = [-1.0, 5.0, 1.0, 0.2, -0.1, ConicIP.vecm([2.0 0.3; 0.3 -1.0])...]
    om = clip(wm, 0.0, 1.0, cdm)
    @test om[1:2] == [0.0, 1.0]
    @test om[3:5] ≈ clip(wm[3:5], 0.0, 1.0, [("Q", 3)])
    @test om[6:8] ≈ clip(wm[6:8], 0.0, 1.0, [("S", 3)])

    # centrality_correction!: Π_box(w) − w with the cap, per frame
    w = randn(6) .* 3
    lo, hi, cap = 0.1, 1.0, 0.5
    Δ = ConicIP.centrality_correction!(similar(w), w, lo, hi, cap, [("R", 6)])
    @test Δ == max.(clamp.(w, lo, hi) .- w, -cap)
    @test ConicIP.centrality_correction!(similar(w), w, -100.0, 100.0, cap, [("R", 6)]) == zeros(6)
    Ws = Symmetric(randn(4, 4)); Ws = Matrix(Ws)
    w = ConicIP.vecm(Ws); Λ = eigvals(Symmetric(Ws))
    Δ = ConicIP.centrality_correction!(similar(w), w, lo, hi, cap, [("S", 10)])
    @test eigvals(Symmetric(ConicIP.mat(Δ))) ≈ sort(max.(clamp.(Λ, lo, hi) .- Λ, -cap)) atol = 1e-12
    @test_throws ArgumentError ConicIP.centrality_correction!(similar(w), w, lo, hi, -1.0, [("S", 10)])

    # ── corrector sign on an LP (mirrors the derivation in ConicIP.jl) ──
    # With z ← z − αΔz, a direction solving the 4×4 system with r_s = −Δw
    # must move the trial complementarity toward the box; +Δw must move it
    # away. The 4×4 solve is replicated from solve4x4 on kktsolver_qr.
    Random.seed!(3)
    let n = 8, m = 12, p = 2, DTB = 0.01
      Q = spzeros(n, n)
      A = sprandn(m, n, 0.6) + [sparse(1.0I, n, n); sprandn(m - n, n, 0.5)]
      G = sprandn(p, n, 0.7)
      y = randn(n); s = 0.5 .+ rand(m); v = 0.5 .+ rand(m); wq = randn(p)
      b = A*y - s + 0.3*randn(m); d = G*y + 0.2*randn(p)
      c = A'*v - G'*randn(p) + 0.3*randn(n)
      cd = [("R", m)]
      F   = Block([Diagonal(sqrt.(s) ./ sqrt.(v))])
      F⁻ᵀ = ConicIP.inv_adjoint!(Block(1), F)
      λ   = F*v
      @test λ ≈ F⁻ᵀ*s
      solve3x3 = ConicIP.kktsolver_qr(Q, A, G, cd)(F, F⁻ᵀ)
      function solve4(r)
        # (local names: a closure assignment to `Δw` would capture the
        # correction vector defined below)
        t1 = F'*(r.s ./ λ)
        (dy, dw, dv) = solve3x3(r.y, r.w, r.v + t1)
        return ConicIP.v4x1(dy, dw, dv, t1 - F'*(F*dv))
      end
      # 4×4 contract check of the replica
      Fm = Matrix(F.Blocks[1]); Fim = Matrix(F⁻ᵀ.Blocks[1])
      K4 = [Matrix(Q) Matrix(G') -Matrix(A') zeros(n, m);
            Matrix(G) zeros(p, p) zeros(p, m) zeros(p, m);
            Matrix(A) zeros(m, p) zeros(m, m) -Matrix(1.0I, m, m);
            zeros(m, n) zeros(m, p) Diagonal(λ)*Fm Diagonal(λ)*Fim]
      vec4(z) = [z.y; z.w; z.v; z.s]
      r0 = ConicIP.v4x1(Q*y + G'*wq - A'*v - c, G*y - d, A*y - s - b, λ .* λ)
      d_aff = solve4(r0)
      @test norm(K4*vec4(d_aff) - vec4(r0)) < 1e-10
      ms(x, dd) = ConicIP.maxstep_rp(x, dd)
      α_aff = min(1, ms(v, d_aff.v), ms(s, d_aff.s))
      μbar = dot(v, s); μ = μbar/m
      σ = max(0, min(1, dot(v - α_aff*d_aff.v, s - α_aff*d_aff.s)/μbar))^3
      lc = -((F⁻ᵀ*d_aff.s) .* (F*d_aff.v)) .+ σ*μ
      Δz = solve4(ConicIP.v4x1(r0.y, r0.w, r0.v, λ .* λ - lc))
      α = min(1, (1-DTB)*min(ms(v, Δz.v), ms(s, Δz.s)))
      @test 0 < α < 1
      σμ = σ*μ; lo = βmin*σμ; hi = βmax*σμ
      boxdist(wv) = norm(max.(lo .- wv, 0) .+ max.(wv .- hi, 0))
      trial(Δ, a) = (λ .- a .* (F*Δ.v)) .* (λ .- a .* (F⁻ᵀ*Δ.s))
      α̃ = min(1, α + ConicIP.GONDZIO_δα)
      wt = trial(Δz, α̃)
      Δw = ConicIP.centrality_correction!(zeros(m), wt, lo, hi, βmax*σμ, cd)
      @test boxdist(wt) > 0 && norm(Δw) > 0
      dist = Dict{Float64,Float64}()
      for sgn in (-1.0, 1.0)
        rc = ConicIP.v4x1(zeros(n), zeros(p), zeros(m), sgn .* Δw)
        Δc = solve4(rc)
        @test norm(K4*vec4(Δc) - vec4(rc)) < 1e-10
        cand = ConicIP.v4x1(Δz.y + Δc.y, Δz.w + Δc.w, Δz.v + Δc.v, Δz.s + Δc.s)
        dist[sgn] = boxdist(trial(cand, α̃))
        # The fourth block row gives the first-order change of the trial
        # complementarity: −α̃·(λ∘FΔv_c + λ∘F⁻ᵀΔs_c) = −α̃·r_s.
        @test λ .* (F*Δc.v) .+ λ .* (F⁻ᵀ*Δc.s) ≈ sgn .* Δw atol = 1e-10
      end
      @test dist[-1.0] < boxdist(wt)      # r_s = −Δw: strictly closer to the box
      @test dist[1.0]  > boxdist(wt)      # r_s = +Δw: farther away
    end

    # ── corrector sign on Q, S, and mixed cone products ──
    # Same construction with the NT scaling of each cone (nestod_soc,
    # nestod_sdc), the Jordan products, and the box distance measured on
    # the Jordan eigenvalues (w₁ ± ‖w̄‖ for Q, eigvals(mat(w)) for S, so the
    # vecm √2 convention is exercised). The fourth block row is linear, so
    # the first-order identity λ∘FΔv_c + λ∘F⁻ᵀΔs_c = r_s holds per cone.
    let
      cinterior(cd, rng) = begin
        s = Float64[]
        for (t, k) in cd
          if t == "R"; append!(s, 0.5 .+ rand(rng, k))
          elseif t == "Q"; u = randn(rng, k - 1); append!(s, [norm(u) + 0.5 + rand(rng); u])
          else r = round(Int, (sqrt(1 + 8k) - 1)/2); M = randn(rng, r, r); append!(s, ConicIP.vecm(M*M' + I))
          end
        end
        s
      end
      jeigs(w, cd) = begin
        out = Float64[]; off = 0
        for (t, k) in cd
          wI = w[off+1:off+k]
          if t == "R"; append!(out, wI)
          elseif t == "Q"; nb = norm(wI[2:end]); append!(out, [wI[1] - nb, wI[1] + nb])
          else append!(out, eigvals(Symmetric(ConicIP.mat(wI))))
          end
          off += k
        end
        out
      end
      ntF(v, s, cd) = begin
        blks = Any[]; off = 0
        for (t, k) in cd
          I = off+1:off+k
          push!(blks, t == "R" ? Diagonal(sqrt.(s[I]) ./ sqrt.(v[I])) :
                      t == "Q" ? ConicIP.nestod_soc(v[I], s[I]) : ConicIP.nestod_sdc(v[I], s[I]))
          off += k
        end
        Block(blks)
      end
      percone(f_rp, f_soc, f_sdc, cd, x, yv) = begin
        o = zeros(length(x)); off = 0
        for (t, k) in cd
          I = off+1:off+k
          (t == "R" ? f_rp : t == "Q" ? f_soc : f_sdc)(view(x, I), view(yv, I), view(o, I))
          off += k
        end
        o
      end
      cprod(x, yv, cd) = percone(ConicIP.xrp!, ConicIP.xsoc!, ConicIP.xsdc!, cd, x, yv)
      cdiv(x, yv, cd)  = percone(ConicIP.drp!, ConicIP.dsoc!, ConicIP.dsdc!, cd, x, yv)
      mstep(x, dd, cd) = begin
        a = Inf; off = 0
        for (t, k) in cd
          I = off+1:off+k
          a = min(a, t == "R" ? ConicIP.maxstep_rp(view(x, I), view(dd, I)) :
                     t == "Q" ? ConicIP.maxstep_soc(view(x, I), view(dd, I)) :
                                ConicIP.maxstep_sdc(view(x, I), view(dd, I)))
          off += k
        end
        a
      end
      cone_e(cd) = begin
        e = Float64[]
        for (t, k) in cd
          if t == "R"; append!(e, ones(k))
          elseif t == "Q"; append!(e, [1.0; zeros(k - 1)])
          else r = round(Int, (sqrt(1 + 8k) - 1)/2); append!(e, ConicIP.vecm(Matrix(1.0I, r, r)))
          end
        end
        e
      end
      cdim(cd) = sum(t == "R" ? k : t == "Q" ? 1 : round(Int, (sqrt(1 + 8k) - 1)/2) for (t, k) in cd)
      for cd in ([("Q", 5), ("Q", 3)], [("S", 6), ("S", 10)], [("R", 3), ("Q", 4), ("S", 6)])
        rng = MersenneTwister(11)
        m = sum(last, cd); n = m + 2; p = 2; DTB = 0.01
        Q = sparse(0.5I, n, n)
        A = sparse(randn(rng, m, n)); G = sparse(randn(rng, p, n))
        y = randn(rng, n); s = cinterior(cd, rng); v = cinterior(cd, rng); wq = randn(rng, p)
        b = A*y - s + 0.3*randn(rng, m); d = G*y + 0.2*randn(rng, p)
        c = A'*v - G'*wq + 0.3*randn(rng, n)
        F = ntF(v, s, cd); F⁻ᵀ = ConicIP.inv_adjoint!(Block(length(cd)), F)
        λ = F*v
        @test λ ≈ F⁻ᵀ*s
        solve3x3 = ConicIP.kktsolver_qr(Q, A, G, cd)(F, F⁻ᵀ)
        solve4(r) = begin
          t1 = F'*cdiv(r.s, λ, cd)
          (dy, dw, dv) = solve3x3(r.y, r.w, r.v + t1)
          ConicIP.v4x1(dy, dw, dv, t1 - F'*(F*dv))
        end
        res4(Δ, r) = max(norm(Q*Δ.y + G'*Δ.w - A'*Δ.v - r.y), norm(G*Δ.y - r.w),
                         norm(A*Δ.y - Δ.s - r.v),
                         norm(cprod(λ, F*Δ.v, cd) + cprod(λ, F⁻ᵀ*Δ.s, cd) - r.s))
        r0 = ConicIP.v4x1(Q*y + G'*wq - A'*v - c, G*y - d, A*y - s - b, cprod(λ, λ, cd))
        d_aff = solve4(r0)
        @test res4(d_aff, r0) < 1e-10
        α_aff = min(1, mstep(v, d_aff.v, cd), mstep(s, d_aff.s, cd))
        μbar = dot(v, s); μ = μbar/cdim(cd)
        σ = max(0, min(1, dot(v - α_aff*d_aff.v, s - α_aff*d_aff.s)/μbar))^3
        lc = -cprod(F⁻ᵀ*d_aff.s, F*d_aff.v, cd) .+ σ*μ .* cone_e(cd)
        Δz = solve4(ConicIP.v4x1(r0.y, r0.w, r0.v, cprod(λ, λ, cd) - lc))
        α = min(1, (1-DTB)*min(mstep(v, Δz.v, cd), mstep(s, Δz.s, cd)))
        @test 0 < α < 1
        σμ = σ*μ; lo = βmin*σμ; hi = βmax*σμ
        boxdist(wv) = (ev = jeigs(wv, cd); norm(max.(lo .- ev, 0) .+ max.(ev .- hi, 0)))
        trial(Δ, a) = cprod(λ .- a .* (F*Δ.v), λ .- a .* (F⁻ᵀ*Δ.s), cd)
        α̃ = min(1, α + ConicIP.GONDZIO_δα)
        wt = trial(Δz, α̃)
        Δw = ConicIP.centrality_correction!(zeros(m), wt, lo, hi, βmax*σμ, cd)
        @test boxdist(wt) > 0 && norm(Δw) > 0
        dist = Dict{Float64,Float64}()
        for sgn in (-1.0, 1.0)
          rc = ConicIP.v4x1(zeros(n), zeros(p), zeros(m), sgn .* Δw)
          Δc = solve4(rc)
          @test res4(Δc, rc) < 1e-10
          @test cprod(λ, F*Δc.v, cd) .+ cprod(λ, F⁻ᵀ*Δc.s, cd) ≈ sgn .* Δw atol = 1e-10
          cand = ConicIP.v4x1(Δz.y + Δc.y, Δz.w + Δc.w, Δz.v + Δc.v, Δz.s + Δc.s)
          dist[sgn] = boxdist(trial(cand, α̃))
        end
        @test dist[-1.0] < boxdist(wt)
        @test dist[1.0]  > boxdist(wt)
      end
    end

    # ── default 0: the corrector path is not entered ──
    Random.seed!(20260907)
    csd = t3_contract_case(t3_mixes[1])
    args = (csd.Q, csd.c, csd.A, csd.b, t3_mixes[1], csd.G, csd.d)
    s_unset = conicIP(args...; verbose = false, kktsolver = ConicIP.kktsolver_qr)
    s_zero  = conicIP(args...; verbose = false, kktsolver = ConicIP.kktsolver_qr,
                      centralityCorrectors = 0)
    @test s_unset.status == s_zero.status == :Optimal
    @test s_unset.Iter == s_zero.Iter
    @test s_unset.kkt_solves == s_zero.kkt_solves
    @test all(s_unset.y .=== s_zero.y)
    capture(f) = mktemp() do path, io
      redirect_stdout(io) do; f(); end
      flush(io); read(path, String)
    end
    out0 = capture(() -> conicIP(args...; verbose = true, kktsolver = ConicIP.kktsolver_qr))
    @test occursin("cc", out0)                                 # header
    @test !occursin(r"\d+/\d+", out0)                          # no cc (or kkt) cells
    out2 = capture(() -> conicIP(args...; verbose = true, kktsolver = ConicIP.kktsolver_qr,
                                 centralityCorrectors = 2))
    cells = [m.match for m in eachmatch(r"(\d+)/(\d+)\s*$"m, out2)]
    @test length(cells) >= 2
    @test all(c -> begin
                     (a, t) = parse.(Int, split(strip(c), '/'))
                     0 <= a <= t <= 2
                   end, cells)
    @test any(c -> strip(c) != "0/0", cells)
    @test_throws ArgumentError conicIP(args...; verbose = false, centralityCorrectors = -1)

    # ── centralityCorrectors = 2: Optimal, no more iterations, bounded solves ──
    # Band generators (copies of benchmark/suite.jl's lp_band / qp_band).
    function t3_lp_band(n; w = 5, seed = 1)
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
    function t3_qp_band(n; w = 3, seed = 1)
      Random.seed!(seed)
      I_ = Int[]; J_ = Int[]; V_ = Float64[]
      for i in 1:n, j in max(1, i - w):i
        push!(I_, i); push!(J_, j); push!(V_, randn())
      end
      L = sparse(I_, J_, V_, n, n)
      Q = L*L' + sparse(1.0I, n, n)
      A = [sparse(1.0I, n, n); -sparse(1.0I, n, n)]
      b = fill(-1.0, 2n)
      return (Q = Q, c = randn(n), A = A, b = b,
              cone_dims = [("R", 2n)], G = spzeros(0, n), d = zeros(0))
    end
    Random.seed!(20260907)
    probs = Any[socp_sum_of_norms(150; d = 200), t3_lp_band(2000), t3_qp_band(2000)]
    for mix in t3_mixes
      cs = t3_contract_case(mix)
      push!(probs, (Q = cs.Q, c = cs.c, A = cs.A, b = cs.b, cone_dims = mix, G = cs.G, d = cs.d))
    end
    # Bounds are exact invariants or carry slack: on this platform K = 2
    # saves 0–3 iterations per problem (three of the nine tie), so a strict
    # `Iter ≤ Iter₀` would hinge on rounding elsewhere on the CI matrix; the
    # corrector solves are counted from the verbose `cc` column (tried), at
    # most K per iteration by construction; the budget against the
    # uncorrected run allows for refinement counts that differ between the
    # two trajectories (one seeded mix sits exactly at `+ 2·Iter₀`).
    for P in probs
      s0 = conicIP(P.Q, P.c, P.A, P.b, P.cone_dims, P.G, P.d; verbose = false)
      s2 = conicIP(P.Q, P.c, P.A, P.b, P.cone_dims, P.G, P.d; verbose = false,
                   centralityCorrectors = 2)
      out = capture(() -> conicIP(P.Q, P.c, P.A, P.b, P.cone_dims, P.G, P.d;
                                  verbose = true, centralityCorrectors = 2))
      tried = sum(parse(Int, split(strip(m.match), '/')[2])
                  for m in eachmatch(r"\d+/\d+\s*$"m, out); init = 0)
      @test s0.status == :Optimal
      @test s2.status == :Optimal
      @test s2.Iter <= s0.Iter + 1
      @test 0 < tried <= 2 * s2.Iter
      @test s2.kkt_solves <= s0.kkt_solves + 2 * s0.Iter + 4
      @test abs(s2.pobj - s0.pobj) <= 1e-5 * (1 + abs(s0.pobj))
    end

    # ── no corrector solve when α == 1 ──
    # min ½‖y‖² − 1ᵀy over y ≥ −100: the optimum y = 1 is far from the
    # bounds and the first Newton step is full (α = 1 exactly). With
    # maxIters = 1 the corrector loop is skipped, so the solve count and
    # the iterate match the default path bitwise. From iteration 2 on the
    # step is boundary-limited (α = 1 − DTB) and each iteration tries one
    # corrector, which cannot lengthen the step and is rejected, so the
    # trajectory is the default one plus one solve per such iteration
    # (measured: 7 iterations, cells 0/0, 0/0, then 0/1 throughout).
    fs = (Matrix(1.0I, 3, 3), [1.0, 1.0, 1.0], sparse(1.0I, 3, 3), fill(-100.0, 3), [("R", 3)])
    f0 = conicIP(fs...; verbose = false, maxIters = 1)
    f2 = conicIP(fs...; verbose = false, maxIters = 1, centralityCorrectors = 2)
    @test f0.kkt_solves == f2.kkt_solves
    @test all(f0.y .=== f2.y)
    g0 = conicIP(fs...; verbose = false, maxIters = 2)
    g2 = conicIP(fs...; verbose = false, maxIters = 2, centralityCorrectors = 2)
    @test g0.kkt_solves <= g2.kkt_solves <= g0.kkt_solves + 1
    outf = capture(() -> conicIP(fs...; verbose = true, centralityCorrectors = 2))
    cellsf = [strip(m.match) for m in eachmatch(r"\d+/\d+\s*$"m, outf)]
    @test cellsf[1] == "0/0" && cellsf[2] == "0/0"     # initial point, iteration 1
    @test all(in(("0/0", "0/1")), cellsf[3:end])       # never more than one tried, none accepted
    @test count(==("0/1"), cellsf[3:end]) >= length(cellsf[3:end]) - 1

    # ── MOI option round trip ──
    opt = ConicIP.Optimizer()
    attr = MOI.RawOptimizerAttribute("centralityCorrectors")
    @test MOI.supports(opt, attr)
    @test MOI.get(opt, attr) == 0
    MOI.set(opt, attr, 2)
    @test MOI.get(opt, attr) == 2
    model = MOI.Utilities.CachingOptimizer(
      MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), opt)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variables(model, 2)
    MOI.set(model, MOI.ObjectiveSense(), MOI.MIN_SENSE)
    fobj = MOI.ScalarQuadraticFunction(
      [MOI.ScalarQuadraticTerm(2.0, x[1], x[1]), MOI.ScalarQuadraticTerm(2.0, x[2], x[2])],
      [MOI.ScalarAffineTerm(-1.0, x[2])], 0.0)
    MOI.set(model, MOI.ObjectiveFunction{typeof(fobj)}(), fobj)
    MOI.add_constraint(model, 1.0 * x[1] + 1.0 * x[2], MOI.EqualTo(1.0))
    MOI.add_constraint(model, x[1], MOI.GreaterThan(0.0))
    MOI.add_constraint(model, x[2], MOI.GreaterThan(0.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.OPTIMAL
    MOI.set(opt, attr, -1)
    @test_throws ArgumentError MOI.optimize!(model)

    # ── infeasibility soundness verdicts are unchanged with correctors on ──
    no_eq(n) = (zeros(0, n), zeros(0))
    for K in (0, 2)
      # (a) ε-feasible box is Optimal
      ε = 1e-9
      for nn in (1, 5), cc in (ones(nn), zeros(nn))
        Ab = [sparse(1.0I, nn, nn); -sparse(1.0I, nn, nn)]
        bb = [zeros(nn); fill(-ε, nn)]
        sol = conicIP(zeros(nn, nn), cc, Ab, bb, [("R", 2nn)];
                      verbose = false, centralityCorrectors = K)
        @test sol.status == :Optimal && !sol.has_certificate
        @test minimum(sol.y) > -1e-7 && maximum(sol.y) < ε + 1e-7
      end
      # (b) tiny-Q: bounded above the tolerance, a validated ray far below it
      mk(ε) = (reshape([ε], 1, 1), [1.0], sparse(reshape([1.0], 1, 1)), [0.0], [("R", 1)])
      for ε in (1e-6, 1e-4, 1e-2)
        sol = conicIP(mk(ε)...; verbose = false, centralityCorrectors = K)
        @test sol.status == :Optimal && !sol.has_certificate
        @test sol.y[1] ≈ 1/ε rtol = 1e-4
      end
      sol = conicIP(mk(1e-12)...; verbose = false, centralityCorrectors = K)
      @test sol.status == :DualInfeasible && sol.has_certificate
      (chk, _) = ConicIP.validate_unboundedness_certificate(
        mk(1e-12)..., no_eq(1)..., sol.y; abstol = 1e-9, reltol = 1e-7)
      @test chk.valid
      # (c) degenerate blocks: equality-only and cone-only infeasibility
      Q1 = zeros(1, 1); c1 = [0.0]
      sol = preprocess_conicIP(Q1, c1, spzeros(0, 1), zeros(0), Tuple{String,Int}[],
                               reshape([1.0; 1.0], 2, 1), [1.0, 2.0];
                               verbose = false, centralityCorrectors = K)
      @test sol.status == :Infeasible && sol.has_certificate
      A2 = sparse([1.0; -1.0][:, :]); b2 = [1.0, 1.0]; K2 = [("R", 2)]
      sol = conicIP(Q1, c1, A2, b2, K2; verbose = false, centralityCorrectors = K)
      @test sol.status == :Infeasible && sol.has_certificate
      (chk, _, _) = ConicIP.validate_infeasibility_certificate(
        Q1, c1, A2, b2, K2, no_eq(1)..., sol.w, sol.v; abstol = 1e-9, reltol = 1e-7)
      @test chk.valid
      # (e) SOC infeasible, ray in the cone
      A3 = sparse([1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0; -1.0 0.0 0.0])
      b3 = [0.0, 0.0, 0.0, 1.0]; K3 = [("Q", 3), ("R", 1)]
      sol = conicIP(zeros(3, 3), zeros(3), A3, b3, K3; verbose = false,
                    centralityCorrectors = K)
      @test sol.status == :Infeasible && sol.has_certificate
      @test ConicIP.cone_margin(sol.v, K3) >= -1e-6
      A4 = sparse([1.0 0.0; 0.0 1.0; -1.0 0.0; 0.0 1.0])
      b4 = [0.0, 0.0, 0.0, 1.0]; K4 = [("Q", 2), ("R", 2)]
      sol = conicIP(zeros(2, 2), zeros(2), A4, b4, K4; verbose = false,
                    centralityCorrectors = K)
      @test sol.status == :Infeasible && sol.has_certificate
      # (f) near-optimal and near-certificate: Optimal wins
      A5 = sparse([1.0; -1.0][:, :]); K5 = [("R", 2)]
      sol = conicIP(zeros(1, 1), [0.0], A5, [0.0, 0.0], K5; verbose = false,
                    centralityCorrectors = K)
      @test sol.status == :Optimal && !sol.has_certificate
      sol = conicIP(zeros(1, 1), [0.0], A5, [0.0, -1e-10], K5; verbose = false,
                    centralityCorrectors = K)
      @test sol.status == :Optimal && !sol.has_certificate
      sol = conicIP(zeros(2, 2), zeros(2), sparse(1.0I, 2, 2), zeros(2), [("R", 2)],
                    Matrix(1.0I, 2, 2), zeros(2); verbose = false,
                    centralityCorrectors = K)
      @test sol.status == :Optimal && !sol.has_certificate
      @test norm(sol.y) < 1e-6
    end
  end

  @testset "Review fixes" begin
    # ── certificate returns carry no stale iterate residuals ──
    # claim_infeasible!/claim_dual_infeasible! discard the iterate (y, s or
    # w, v are NaN, pobj too); rEq and rGap described that iterate and are
    # NaN as well, through MOI's RelativeGap included.
    s = conicIP(zeros(1, 1), [0.0], sparse([1.0; -1.0][:, :]), [1.0, 1.0], [("R", 2)]; verbose = false)
    @test s.status == :Infeasible && s.has_certificate
    @test isnan(s.rEq) && isnan(s.rGap) && isnan(s.pobj)
    s = conicIP(zeros(1, 1), [1.0], sparse(ones(1, 1)), [0.0], [("R", 1)]; verbose = false)
    @test s.status == :DualInfeasible && s.has_certificate
    @test isnan(s.rEq) && isnan(s.rGap)
    opt = ConicIP.Optimizer()
    model = MOI.Utilities.CachingOptimizer(
      MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}()), opt)
    MOI.set(model, MOI.Silent(), true)
    x = MOI.add_variable(model)
    MOI.add_constraint(model, x, MOI.GreaterThan(1.0))
    MOI.add_constraint(model, x, MOI.LessThan(0.0))
    MOI.optimize!(model)
    @test MOI.get(model, MOI.TerminationStatus()) == MOI.INFEASIBLE
    @test MOI.get(model, MOI.ResultCount()) == 1
    @test isnan(MOI.get(model, MOI.RelativeGap()))

    # ── deflation propagates the KKT diagnostics ──
    # Variable n+1 is absent from Q, A, G with c = 0 and is deflated; the
    # reduced LP with Q = 0 and the natural ordering repairs n pivots and,
    # with a large dynamic_delta and retry on, refactorizes. The totals of
    # the deflated solve equal those of the same problem without the
    # deflated column.
    n = 6
    Qz = spzeros(n + 1, n + 1); Az = [sparse(1.0I, n, n) spzeros(n, 1)]; cz = [ones(n); 0.0]
    ksr = (Q, A, G, cd) -> ConicIP.kktsolver_ldl(Q, A, G, cd; static_reg = 0.0,
              dynamic_delta = 1e-2, retry_max = 2,
              pattern = ConicIP._ldl_pattern(Q, A, G, cd;
                          perm_hint = collect(1:(size(Q, 1) + size(A, 1) + size(G, 1)))))
    sd = conicIP(Qz, cz, Az, zeros(n), [("R", n)]; verbose = false, kktsolver = ksr,
                 equilibrate = false)
    sf = conicIP(Qz[1:n, 1:n], cz[1:n], Az[:, 1:n], zeros(n), [("R", n)]; verbose = false,
                 kktsolver = ksr, equilibrate = false)
    @test sd.status == sf.status == :DualInfeasible
    @test sd.kkt_repaired == sf.kkt_repaired >= n
    @test sd.kkt_refactors == sf.kkt_refactors >= 1
    @test sd.kkt_solves == sf.kkt_solves

    # ── row-wise feasibility is part of the optimality test ──
    # The aggregate 2-norm test accepted a point violating a small-scale
    # row: x ≥ 1 at weight w₁ against x ≤ 0.5 at weight w₂ (infeasible)
    # ended :Optimal at x ≈ 0.5 for (w₁, w₂) = (1e-4, 1e4) unequilibrated,
    # with prFeas = 4e-8 and a row-wise residual of 1.5e-4. With the
    # row-wise test the pair is certified infeasible at every weight,
    # equilibrated or not.
    for (w1, w2) in ((1e-4, 1e4), (1e-6, 1e6), (1e-3, 1e3), (1.0, 1.0))
      A = sparse([w1; -w2][:, :]); b = [w1, -0.5*w2]
      for eqb in (false, true)
        s = conicIP(spzeros(1, 1), [0.0], A, b, [("R", 2)]; verbose = false, equilibrate = eqb)
        @test s.status == :Infeasible && s.has_certificate
      end
    end
    # The helper against the dense formula, with and without a row scaling
    r = [1e-3, -2.0, 0.5]; bb = [10.0, 0.0, 1e4]; pr = [1.0, 2.0, 3.0]
    ss = [0.1, 0.2, 0.3]; D = [2.0, 0.5, 4.0]
    @test ConicIP._rowwise_max(r, bb, pr, ss) ≈
          maximum(abs.(r) ./ (1 .+ abs.(bb) .+ pr .+ abs.(ss)))
    @test ConicIP._rowwise_max(r, bb, pr, nothing) ≈ maximum(abs.(r) ./ (1 .+ abs.(bb) .+ pr))
    @test ConicIP._rowwise_max(r, bb, pr, ss, D) ≈
          maximum((abs.(r) ./ D) ./ (1 .+ abs.(bb) ./ D .+ pr .+ abs.(ss) ./ D))
    @test ConicIP._rowwise_max(Float64[], Float64[], Float64[], nothing) == 0.0
    # An :Optimal point passes the row-wise test on the original data,
    # equilibrated or not, through the presolve or not (the same iterate
    # passes the aggregate test, so iteration counts are what they were).
    function rowwise_max(P, s)
      absA = ConicIP._absmat(P.A); absG = ConicIP._absmat(P.G)
      rv = P.A*s.y - s.s - P.b; rw = P.G*s.y - P.d
      rp = isempty(P.b) ? 0.0 : maximum(abs.(rv) ./ (1 .+ abs.(P.b) .+ absA*abs.(s.y) .+ abs.(s.s)))
      re = isempty(P.d) ? 0.0 : maximum(abs.(rw) ./ (1 .+ abs.(P.d) .+ absG*abs.(s.y)))
      return max(rp, re)
    end
    Random.seed!(20260907)
    P = let cs = t3_contract_case([("R", 4), ("Q", 3), ("S", 6)])
      (Q = cs.Q, c = cs.c, A = cs.A, b = cs.b, cone_dims = [("R", 4), ("Q", 3), ("S", 6)], G = cs.G, d = cs.d)
    end
    for entry in (conicIP, preprocess_conicIP), eqb in (false, true)
      s = entry(P.Q, P.c, P.A, P.b, P.cone_dims, P.G, P.d; verbose = false, equilibrate = eqb)
      @test s.status == :Optimal
      @test rowwise_max(P, s) < 1e-6
    end
    # Badly scaled but feasible: rows and columns over twelve orders of
    # magnitude solve with and without equilibration (no false stall). The
    # row-wise test keeps the unit floor of every residual normalization
    # ("1 +"), so a row of scale 1e-6 is held to an absolute 1e-6, which is
    # 1e-4 relative to its own scale: the tolerance below is that, not
    # optTol.
    nb = 8
    rs = 10.0 .^ range(-6, 6; length = nb); cs = 10.0 .^ range(6, -6; length = nb)
    Ab = Diagonal(rs) * sparse(1.0I, nb, nb) * Diagonal(cs); bbv = Diagonal(rs) * ones(nb)
    cb = Diagonal(cs) * ones(nb)
    for eqb in (false, true)
      s = conicIP(spzeros(nb, nb), -cb, Ab, bbv, [("R", nb)]; verbose = false, equilibrate = eqb)
      @test s.status == :Optimal
      @test s.y .* cs ≈ ones(nb) atol = 1e-3
    end
  end
end
