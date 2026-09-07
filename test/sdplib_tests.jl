# SDPLIB validation gate
# ======================
#
# End-to-end checks against a six-instance subset of SDPLIB 1.2 (Borchers
# 1999), read through `MOI.FileFormats.SDPA` and solved through ConicIP's
# MOI wrapper.  Nothing is redistributed: the instances are fetched at test
# time from a pinned upstream commit, sha256-verified against the table in
# `test/sdplib/instances.jl`, and cached in a ConicIP-owned scratch space.
# Provenance, hashes and the reference objective values are recorded in
# `test/sdplib/README.md`.  With no network the whole testset is skipped, so
# `Pkg.test()` stays green offline; a *hash mismatch* is never skipped.
#
# Why these instances validate anything the rest of the suite does not: they
# are third-party data with independently published optimal values, they mix
# many small PSD blocks with linear blocks, and they exercise the whole path
# (SDPA reader → bridges → preprocessor → solver → MOI getters).  Every
# assertion below is checked twice: once against what ConicIP reports, and
# once against residuals recomputed here from the *source* model data, so a
# wrapper that lies about its own solution cannot pass.
#
# ── Objective sign convention ─────────────────────────────────────────────
#
# `MOI.FileFormats.SDPA` imports a `.dat-s` file in *geometric* conic form,
#
#     min bᵀy   s.t.   Σᵢ Aᵢ yᵢ − C ∈ K,
#
# and that is exactly SDPA's own *primal*
#
#     P:  min cᵀx   s.t.   Σᵢ Fᵢ xᵢ − F₀ ⪰ 0
#
# (the file's `c` vector becomes the MOI objective, the `Fᵢ` become the
# constraint matrices).  The SDPLIB README reports optimal values "based on
# the SDPA conventions", i.e. for that same primal.  So the imported model's
# objective needs **no sign flip**: `MOI.ObjectiveValue()` equals the SDPLIB
# reference directly.  Verified on `truss1`, whose published value is
# −8.999996 and whose solved `ObjectiveValue()` is −8.9999963137.
#
# The same identification fixes the two infeasible instances.  Because the
# MOI model *is* SDPA's primal:
#
#   * `infp1` (SDPA-primal infeasible) ⇒ the MOI model is primal infeasible
#     ⇒ `TerminationStatus() == INFEASIBLE`, `DualStatus()` is a certificate;
#   * `infd1` (SDPA-dual infeasible)   ⇒ the MOI model is unbounded
#     ⇒ `TerminationStatus() == DUAL_INFEASIBLE`, `PrimalStatus()` is a ray.
#
# Both are re-derived here from the constraint data (Farkas conditions for
# `infp1`, an improving homogeneous ray for `infd1`) rather than taken on
# ConicIP's word.
#
# ── Triangle conventions ──────────────────────────────────────────────────
#
# `MOI.PositiveSemidefiniteConeTriangle` vectorizes the *upper* triangle in
# column-major order — (1,1), (1,2), (2,2), (1,3), (2,3), (3,3), … — with
# off-diagonal entries stored **unscaled**.  The scalar product that pairs a
# constraint function with its dual therefore counts off-diagonals twice;
# `_sdplib_weights` builds those weights and every inner product below uses
# them.  Getting this wrong is not subtle: the stationarity residual jumps
# from 1e-10 to O(1).

import MathOptInterface as MOI
using Scratch, Downloads, SHA

# Pinned-commit URLs, sha256 table and the shared scratch cache. Nothing is
# vendored: the six instances are fetched once and cached.
include(joinpath(@__DIR__, "sdplib", "instances.jl"))

# Fetch all six up front, so the offline decision is made once rather than
# per testset. A hash mismatch is NOT a skip — it means the cached or served
# file is wrong, and the gate must fail loudly.
const SDPLIB_PATHS = Dict{String,String}()
const SDPLIB_UNAVAILABLE = Ref{Union{Nothing,String}}(nothing)
let
    dir = sdplib_cache_dir()
    try
        for name in SDPLIB_TEST_INSTANCES
            SDPLIB_PATHS[name] = sdplib_fetch!(name; dir = dir)
        end
    catch e
        # Only a failed download is skippable (no network, upstream down).
        # Everything else — a hash mismatch, a missing hash entry, a cache
        # directory that cannot be created — is a local defect and must
        # fail the gate loudly rather than silently skip it.
        e isa Downloads.RequestError || rethrow()
        SDPLIB_UNAVAILABLE[] = sprint(showerror, e)
        @warn """
              SDPLIB instances could not be fetched — skipping the SDPLIB testset.
              Base URL: $SDPLIB_BASE_URL
              Cache:    $(sdplib_cache_dir())
              """ exception = e
    end
end

# Optimal objective values quoted verbatim from the SDPLIB 1.2 README
# (vsdp/SDPLIB @ fa11b45c1d8c896a6abad2648d5dad46d8ecefaa).
const SDPLIB_REF = Dict(
    "truss1"   => -8.999996,
    "hinf1"    =>  2.0326,
    "control1" =>  1.778463e1,
    "theta1"   =>  2.3e1,
)

# Relative objective tolerance, |obj − ref| / (1 + |ref|), for the instances
# solved to the default optTol = 1e-8.  `hinf1` is not one of them; it has its
# own testset and its own tolerances below.
const SDPLIB_OBJTOL = 1e-5

# Residual tolerance for the independently recomputed KKT quantities: a
# hundred times the requested optimality tolerance.  The same ratio is used
# for the `hinf1` solve at optTol = 1e-4.
sdplib_restol(optTol) = 100 * optTol
const SDPLIB_RESTOL = sdplib_restol(1e-8)

# ── helpers ───────────────────────────────────────────────────────────────

# Read one cached instance and solve it through the MOI wrapper.  The
# optimizer is configured the way the `MOI.Test` block in runtests.jl
# configures it (optTol = 1e-8), but through `RawOptimizerAttribute` so the
# attribute path is exercised too.
function sdplib_solve(name; optTol = 1e-8)
    model = MOI.FileFormats.Model(format = MOI.FileFormats.FORMAT_SDPA)
    MOI.read_from_file(model, SDPLIB_PATHS[name])
    opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
    MOI.set(opt, MOI.Silent(), true)
    MOI.set(opt, MOI.RawOptimizerAttribute("optTol"), optTol)
    index_map = MOI.copy_to(opt, model)
    MOI.optimize!(opt)
    return model, opt, index_map
end

# Weights of the scalar product MOI pairs a constraint with its dual:
# off-diagonal entries of a PSD triangle count twice, everything else once.
function _sdplib_weights(s::MOI.PositiveSemidefiniteConeTriangle)
    n = MOI.side_dimension(s)
    w = ones(MOI.dimension(s))
    k = 0
    for j in 1:n, i in 1:j
        k += 1
        i != j && (w[k] = 2.0)
    end
    return w
end
_sdplib_weights(s::MOI.Nonnegatives) = ones(MOI.dimension(s))

# Unpack a column-major upper-triangle vector into a full symmetric matrix.
function _sdplib_unpack(v)
    d = length(v)
    n = div(isqrt(8d + 1) - 1, 2)
    @assert n * (n + 1) == 2d
    M = zeros(n, n)
    k = 0
    for j in 1:n, i in 1:j
        k += 1
        M[i, j] = v[k]
        M[j, i] = v[k]
    end
    return M
end

# How far inside its cone a vector sits (negative ⇒ violation).
_sdplib_margin(v, ::MOI.PositiveSemidefiniteConeTriangle) =
    isempty(v) ? Inf : eigmin(Symmetric(_sdplib_unpack(v)))
_sdplib_margin(v, ::MOI.Nonnegatives) = isempty(v) ? Inf : minimum(v)

# (index, function, set) for every constraint of the *source* model.
function _sdplib_constraints(model)
    out = Any[]
    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F,S}())
            push!(out, (ci,
                        MOI.get(model, MOI.ConstraintFunction(), ci),
                        MOI.get(model, MOI.ConstraintSet(), ci)))
        end
    end
    return out
end

# Objective vector b and constant of the source model.
function _sdplib_objective(model)
    f = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
    b = zeros(MOI.get(model, MOI.NumberOfVariables()))
    for t in f.terms
        b[t.variable.value] += t.coefficient
    end
    return b, f.constant
end

# Recompute the KKT quantities of an optimal solve from the source data.
#
#   stat   ‖b − Σⱼ Aⱼᵀ dⱼ‖∞ / (1 + ‖b‖∞)   (weighted adjoint; stationarity)
#   pmarg  min cone margin of f(y)          (primal feasibility)
#   dmarg  min cone margin of d             (dual feasibility, cones self-dual)
#   gap    |bᵀy − Σⱼ ⟨−constant(fⱼ), dⱼ⟩| / (1 + |pobj|)
#   comp   |Σⱼ ⟨fⱼ(y), dⱼ⟩| / (1 + |pobj|)  (complementary slackness)
#   cprim  ‖ConstraintPrimal − f(y)‖∞       (wrapper self-consistency)
#   pobj   bᵀy + constant, recomputed
function sdplib_residuals(model, opt, index_map)
    vars = MOI.get(model, MOI.ListOfVariableIndices())
    y = [MOI.get(opt, MOI.VariablePrimal(), index_map[v]) for v in vars]
    b, b0 = _sdplib_objective(model)
    stat = copy(b)
    pmarg = dmarg = Inf
    dualobj = b0
    comp = 0.0
    cprim = 0.0
    for (ci, f, s) in _sdplib_constraints(model)
        w = _sdplib_weights(s)
        fy = MOI.Utilities.eval_variables(v -> y[v.value], f)
        d = MOI.get(opt, MOI.ConstraintDual(), index_map[ci])
        cprim = max(cprim, norm(MOI.get(opt, MOI.ConstraintPrimal(), index_map[ci]) - fy, Inf))
        pmarg = min(pmarg, _sdplib_margin(fy, s))
        dmarg = min(dmarg, _sdplib_margin(d, s))
        for t in f.terms
            stat[t.scalar_term.variable.value] -=
                w[t.output_index] * t.scalar_term.coefficient * d[t.output_index]
        end
        dualobj -= sum(w .* MOI.constant(f) .* d)
        comp += sum(w .* fy .* d)
    end
    pobj = dot(b, y) + b0
    return (stat  = norm(stat, Inf) / (1 + norm(b, Inf)),
            pmarg = pmarg,
            dmarg = dmarg,
            gap   = abs(pobj - dualobj) / (1 + abs(pobj)),
            comp  = abs(comp) / (1 + abs(pobj)),
            cprim = cprim,
            pobj  = pobj,
            dobj  = dualobj)
end

# Farkas certificate of primal infeasibility for  min bᵀy  s.t. Aⱼy + gⱼ ∈ Kⱼ.
# A dual ray d proves infeasibility when d ∈ K* (= K here), Σⱼ Aⱼᵀ dⱼ = 0 and
# ⟨g, d⟩ < 0: for any feasible y, ⟨Aⱼy + gⱼ, dⱼ⟩ ≥ 0 summed gives
# ⟨g,d⟩ ≥ −yᵀ(Σ Aⱼᵀ dⱼ) = 0, a contradiction.  MOI reports −⟨g,d⟩ as
# `DualObjectiveValue`, so the ray value must be strictly positive.
function sdplib_farkas(model, opt, index_map)
    nv = MOI.get(model, MOI.NumberOfVariables())
    adj = zeros(nv)
    dmarg = Inf
    dnorm = 0.0
    value = 0.0
    for (ci, f, s) in _sdplib_constraints(model)
        w = _sdplib_weights(s)
        d = MOI.get(opt, MOI.ConstraintDual(), index_map[ci])
        dnorm = max(dnorm, norm(d, Inf))
        dmarg = min(dmarg, _sdplib_margin(d, s))
        for t in f.terms
            adj[t.scalar_term.variable.value] +=
                w[t.output_index] * t.scalar_term.coefficient * d[t.output_index]
        end
        value -= sum(w .* MOI.constant(f) .* d)
    end
    return (adj = norm(adj, Inf) / (1 + dnorm), dmarg = dmarg,
            value = value, dnorm = dnorm)
end

# Improving ray certifying dual infeasibility (an unbounded primal): ȳ must
# satisfy the *homogeneous* constraints Aⱼȳ ∈ Kⱼ (constants dropped — a ray
# is a direction) and decrease the objective, bᵀȳ < 0 for a minimization.
function sdplib_ray(model, opt, index_map)
    vars = MOI.get(model, MOI.ListOfVariableIndices())
    ybar = [MOI.get(opt, MOI.VariablePrimal(), index_map[v]) for v in vars]
    b, _ = _sdplib_objective(model)
    margin = Inf
    for (_, f, s) in _sdplib_constraints(model)
        hom = MOI.Utilities.eval_variables(v -> ybar[v.value], f) .- MOI.constant(f)
        margin = min(margin, _sdplib_margin(hom, s))
    end
    return (margin = margin, value = dot(b, ybar), norm = norm(ybar, Inf))
end

# ── the gate ──────────────────────────────────────────────────────────────

@testset "SDPLIB" begin

if SDPLIB_UNAVAILABLE[] !== nothing

    # No network (or the mirror is down). Record one skipped test so the
    # suite reports the gate as not-run rather than silently passing.
    @info "SDPLIB testset skipped: $(SDPLIB_UNAVAILABLE[])"
    @test_skip "SDPLIB instances unavailable"

else

    @testset "$name" for name in ("truss1", "control1", "theta1")
        ref = SDPLIB_REF[name]
        model, opt, index_map = sdplib_solve(name)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(opt, MOI.DualStatus()) == MOI.FEASIBLE_POINT

        obj = MOI.get(opt, MOI.ObjectiveValue())
        # No sign flip: the imported model is SDPA's primal (file header).
        @test abs(obj - ref) / (1 + abs(ref)) < SDPLIB_OBJTOL

        r = sdplib_residuals(model, opt, index_map)
        @test r.stat  < SDPLIB_RESTOL          # stationarity  b = Σ Aⱼᵀdⱼ
        @test r.pmarg > -SDPLIB_RESTOL         # f(y) ∈ K
        @test r.dmarg > -SDPLIB_RESTOL         # d ∈ K*
        @test r.gap   < SDPLIB_RESTOL          # zero duality gap
        @test r.comp  < SDPLIB_RESTOL          # complementary slackness
        @test r.cprim < SDPLIB_RESTOL          # ConstraintPrimal is f(y)
        # The recomputed objective agrees with what the wrapper reports, and
        # so does the recomputed dual objective.
        @test r.pobj ≈ obj rtol = 1e-9
        @test r.dobj ≈ MOI.get(opt, MOI.DualObjectiveValue()) rtol = 1e-6
    end

    # `hinf1` gets its own testset because it cannot be solved to 1e-8.
    #
    # The hinf family lacks strict complementarity: the optimal primal and
    # dual slacks are both singular on a common face, so σ(X)·σ(Z) has no
    # gap to close.  The Nesterov–Todd scaling then blows up as the iterates
    # approach that face — the KKT system becomes ill-conditioned once
    # feasibility reaches ~1e-6, and the iteration stagnates instead of
    # converging.  This is a property of the problem, not a defect to fix:
    # the same behaviour shows up across the family and across kktsolvers
    # (`qr`, `sparse`, `2x2` all stagnate identically; without the
    # preprocessor the KKT solve fails outright at iteration 67).
    #
    # Sweep over the family on this tree (2026-09-06), by requested optTol:
    #
    #   1e-8: hinf1 ITERATION_LIMIT, hinf2 NUMERICAL_ERROR,
    #         hinf3 NUMERICAL_ERROR
    #   1e-6: hinf1 ITERATION_LIMIT, hinf2 OPTIMAL (2.2e-5, 31 it),
    #         hinf3 NUMERICAL_ERROR
    #   1e-4: hinf1 OPTIMAL (8.9e-4, 14 it), hinf2 OPTIMAL (6.2e-5),
    #         hinf3 OPTIMAL (9.6e-4)
    #
    # So the contract worth pinning is two-sided: at 1e-8 the solve must
    # fail *cleanly* (a status and a message, never an exception), and at
    # 1e-4 it must actually converge to the published optimum.
    @testset "hinf1 — degenerate, tolerance-limited" begin
        ref = SDPLIB_REF["hinf1"]

        # (i) At optTol = 1e-8 the solve does not throw and reports one of
        #     the honest non-convergence statuses with a diagnostic string.
        #     Before the SDP line-search fix this threw a raw
        #     `PosDefException` out of `nestod_sdc`.
        _, opt8, _ = sdplib_solve("hinf1"; optTol = 1e-8)
        @test MOI.get(opt8, MOI.TerminationStatus()) in
              (MOI.OPTIMAL, MOI.ALMOST_OPTIMAL,
               MOI.ITERATION_LIMIT, MOI.NUMERICAL_ERROR)
        @test !isempty(MOI.get(opt8, MOI.RawStatusString()))

        # (ii) At optTol = 1e-4 it converges, and every independently
        #      recomputed residual holds at the matching looser tolerance.
        optTol = 1e-4
        δ = sdplib_restol(optTol)
        model, opt, index_map = sdplib_solve("hinf1"; optTol = optTol)

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.OPTIMAL
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.FEASIBLE_POINT
        @test MOI.get(opt, MOI.DualStatus()) == MOI.FEASIBLE_POINT

        obj = MOI.get(opt, MOI.ObjectiveValue())
        @test abs(obj - ref) / (1 + abs(ref)) < 2e-3

        r = sdplib_residuals(model, opt, index_map)
        @test r.stat  < δ
        @test r.pmarg > -δ
        @test r.dmarg > -δ
        @test r.gap   < δ
        @test r.comp  < δ
        @test r.cprim < δ
        @test r.pobj ≈ obj rtol = 1e-9
    end

    # `infp1` is SDPA-primal infeasible, and the imported model *is* SDPA's
    # primal, so MOI must report INFEASIBLE with a dual (Farkas) ray.
    @testset "infp1 — primal infeasible" begin
        model, opt, index_map = sdplib_solve("infp1")

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.INFEASIBLE
        @test MOI.get(opt, MOI.ResultCount()) == 1
        @test MOI.get(opt, MOI.DualStatus()) == MOI.INFEASIBILITY_CERTIFICATE
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.NO_SOLUTION

        c = sdplib_farkas(model, opt, index_map)
        @test c.dnorm > 0                      # a ray, not the zero vector
        @test c.adj    < SDPLIB_RESTOL         # Σ Aⱼᵀ dⱼ = 0
        @test c.dmarg  > -SDPLIB_RESTOL        # d ∈ K*
        @test c.value  > 1e-6                  # ⟨g,d⟩ < 0, strictly
        @test MOI.get(opt, MOI.DualObjectiveValue()) ≈ c.value rtol = 1e-6
    end

    # `infd1` is SDPA-dual infeasible, so the imported (primal) model is
    # unbounded: MOI must report DUAL_INFEASIBLE with a primal improving ray.
    @testset "infd1 — dual infeasible (primal unbounded)" begin
        model, opt, index_map = sdplib_solve("infd1")

        @test MOI.get(opt, MOI.TerminationStatus()) == MOI.DUAL_INFEASIBLE
        @test MOI.get(opt, MOI.ResultCount()) == 1
        @test MOI.get(opt, MOI.PrimalStatus()) == MOI.INFEASIBILITY_CERTIFICATE
        @test MOI.get(opt, MOI.DualStatus()) == MOI.NO_SOLUTION

        c = sdplib_ray(model, opt, index_map)
        @test c.norm > 0                       # a ray, not the zero vector
        @test c.margin > -SDPLIB_RESTOL        # Aȳ ∈ K (homogeneous)
        @test c.value  < -1e-6                 # bᵀȳ < 0, strictly
        @test MOI.get(opt, MOI.ObjectiveValue()) ≈ c.value rtol = 1e-6
    end

end  # SDPLIB_UNAVAILABLE

end
