import MathOptInterface as MOI

"""
    Optimizer(; kwargs...)

MathOptInterface optimizer wrapping the ConicIP interior-point solver.
Use as a JuMP solver via `Model(ConicIP.Optimizer)`.

# Options

Settable as constructor keywords or through
`MOI.RawOptimizerAttribute` / JuMP's `set_attribute`:

- `verbose::Bool` -- print solver iterations (default: `false`)
- `optTol::Float64` -- optimality tolerance (default: `1e-6`)
- `maxIters::Int` -- maximum iterations (default: `100`)
- `infeasTol::Float64` -- infeasibility/unboundedness certificate tolerance
  (default: `1e-7`)
- `kktsolver` -- `"auto"` (default; picks by cone mix and sparsity via
  [`choose_kktsolver`](@ref)), `"qr"`, `"sparse"`, `"2x2"`, or a solver
  function
- `preprocess::Bool` -- remove redundant equality rows via
  [`preprocess_conicIP`](@ref) before solving (default: `true`)
- plus `infeasAbsTol`, `DTB`, `maxRefinementSteps`, `staticReg`,
  `certFallback`, `certFallbackIters`, `cache_nestodd` — forwarded to
  [`conicIP`](@ref)

`MOI.Silent` is supported and overrides `verbose`.

# Supported Constraints
- **Vector:** `Zeros`, `Nonnegatives`, `Nonpositives`, `SecondOrderCone`,
  `PositiveSemidefiniteConeTriangle`
- **Scalar:** `EqualTo`, `GreaterThan`, `LessThan`
"""
mutable struct Optimizer <: MOI.AbstractOptimizer
    sol::Union{Nothing, Solution}
    max_sense::Bool
    objective_constant::Float64
    n::Int
    c_int::Vector{Float64}         # internal objective vector handed to the solver
    # Constraint row tracking for primal/dual recovery
    eq_ci_map::Vector{Pair{Any, UnitRange{Int}}}
    eq_offset::Vector{Float64}     # 0 for Zeros, rhs for EqualTo
    eq_is_scalar::Vector{Bool}
    eq_G::Union{Nothing, SparseMatrixCSC{Float64, Int}}  # equality constraint matrix
    eq_d::Vector{Float64}                                 # equality constraint RHS
    ineq_ci_map::Vector{Pair{Any, UnitRange{Int}}}
    ineq_sign::Vector{Float64}     # +1 or -1 (Nonpositive/LessThan flip)
    ineq_offset::Vector{Float64}   # 0 for vector, lower/upper for scalar
    ineq_is_scalar::Vector{Bool}
    ineq_is_psd::Vector{Bool}     # true for PSD constraints (√2 scaling)
    ineq_A::Union{Nothing, SparseMatrixCSC{Float64, Int}}  # inequality constraint matrix
    ineq_b::Vector{Float64}                                # inequality constraint RHS
    # Timing
    solve_time::Float64
    # Solver options: only the ones explicitly set are stored, so the
    # solver's own defaults (and the preprocessor's dynamic staticReg
    # opt-in) stay in charge of everything else.
    options::Dict{String, Any}
    silent::Bool
end

# Options settable via MOI.RawOptimizerAttribute (and Optimizer kwargs)
const _SUPPORTED_OPTIONS = (
    "verbose", "optTol", "maxIters", "infeasTol", "infeasAbsTol", "DTB",
    "maxRefinementSteps", "staticReg", "certFallback", "certFallbackIters",
    "cache_nestodd", "kktsolver", "preprocess",
)

# Map a kktsolver name to the solver constructor. Accepts the constructor
# itself, or "auto" | "qr" | "sparse" | "2x2"/"pivot".
function _resolve_kktsolver(v)
    v isa Function && return v
    s = lowercase(string(v))
    s == "auto"             && return default_kktsolver
    s == "qr"               && return kktsolver_qr
    s == "sparse"           && return kktsolver_sparse
    s in ("2x2", "pivot")   && return pivot(kktsolver_2x2)
    throw(ArgumentError(
        "unknown kktsolver \"$v\" (expected \"auto\", \"qr\", \"sparse\", " *
        "\"2x2\", or a solver function)"))
end

function Optimizer(; kwargs...)
    model = Optimizer(
        nothing, false, 0.0, 0, Float64[],
        Pair{Any, UnitRange{Int}}[], Float64[], Bool[], nothing, Float64[],
        Pair{Any, UnitRange{Int}}[], Float64[], Float64[], Bool[], Bool[], nothing, Float64[],
        NaN,
        Dict{String, Any}(), false,
    )
    for (k, v) in kwargs
        MOI.set(model, MOI.RawOptimizerAttribute(string(k)), v)
    end
    return model
end

MOI.supports(::Optimizer, attr::MOI.RawOptimizerAttribute) =
    attr.name in _SUPPORTED_OPTIONS

function MOI.set(model::Optimizer, attr::MOI.RawOptimizerAttribute, value)
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    if attr.name == "kktsolver"
        _resolve_kktsolver(value)   # validate eagerly
    end
    model.options[attr.name] = value
    return
end

function MOI.get(model::Optimizer, attr::MOI.RawOptimizerAttribute)
    if !MOI.supports(model, attr)
        throw(MOI.UnsupportedAttribute(attr))
    end
    defaults = Dict{String, Any}(
        "verbose" => false, "optTol" => 1e-6, "maxIters" => 100,
        "infeasTol" => 1e-7, "infeasAbsTol" => 1e-9, "DTB" => 0.01,
        "maxRefinementSteps" => 3, "staticReg" => 0.0,
        "certFallback" => true, "certFallbackIters" => 50,
        "cache_nestodd" => false, "kktsolver" => "auto",
        "preprocess" => true)
    return get(model.options, attr.name, defaults[attr.name])
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(model::Optimizer, ::MOI.Silent, value::Bool) = (model.silent = value; nothing)
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

function MOI.empty!(model::Optimizer)
    model.sol = nothing
    model.max_sense = false
    model.objective_constant = 0.0
    model.n = 0
    model.solve_time = NaN
    empty!(model.c_int)
    empty!(model.eq_ci_map)
    empty!(model.eq_offset)
    empty!(model.eq_is_scalar)
    model.eq_G = nothing
    empty!(model.eq_d)
    empty!(model.ineq_ci_map)
    empty!(model.ineq_sign)
    empty!(model.ineq_offset)
    empty!(model.ineq_is_scalar)
    empty!(model.ineq_is_psd)
    model.ineq_A = nothing
    empty!(model.ineq_b)
end

function MOI.is_empty(model::Optimizer)
    return model.sol === nothing && model.n == 0
end

MOI.get(::Optimizer, ::MOI.SolverName) = "ConicIP"
MOI.get(::Optimizer, ::MOI.SolverVersion) = "0.2"

# Interior-point solver — no simplex basis information
MOI.supports(::Optimizer, ::MOI.VariableBasisStatus) = false
MOI.supports(::Optimizer, ::MOI.ConstraintBasisStatus) = false

# Supported objective
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},MOI.VariableIndex}}
    return true
end

# Supported constraints
const SupportedVectorSets = Union{
    MOI.Zeros,
    MOI.Nonnegatives,
    MOI.Nonpositives,
    MOI.SecondOrderCone,
    MOI.PositiveSemidefiniteConeTriangle,
}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.VectorAffineFunction{Float64},MOI.VectorOfVariables}},
    ::Type{<:SupportedVectorSets},
)
    return true
end

const SupportedScalarSets = Union{
    MOI.EqualTo{Float64},
    MOI.GreaterThan{Float64},
    MOI.LessThan{Float64},
}

function MOI.supports_constraint(
    ::Optimizer,
    ::Type{<:Union{MOI.ScalarAffineFunction{Float64},MOI.VariableIndex}},
    ::Type{<:SupportedScalarSets},
)
    return true
end

# ──────────────────────────────────────────────────────────────
#  Extract constraint rows from MOI functions
# ──────────────────────────────────────────────────────────────

function _extract_vector_constraint(f, n)
    if f isa MOI.VectorOfVariables
        dim = length(f.variables)
        Ai = spzeros(dim, n)
        bi = zeros(dim)
        for (i, vi) in enumerate(f.variables)
            Ai[i, vi.value] = 1.0
        end
        return Ai, bi
    else  # VectorAffineFunction
        dim = MOI.output_dimension(f)
        Ai = spzeros(dim, n)
        bi = collect(Float64, f.constants)
        for term in f.terms
            row = term.output_index
            col = term.scalar_term.variable.value
            Ai[row, col] += term.scalar_term.coefficient
        end
        return Ai, bi
    end
end

function _extract_scalar_constraint(f, n)
    Ai = spzeros(1, n)
    bi = 0.0
    if f isa MOI.VariableIndex
        Ai[1, f.value] = 1.0
    else  # ScalarAffineFunction
        bi = f.constant
        for term in f.terms
            Ai[1, term.variable.value] += term.coefficient
        end
    end
    return Ai, bi
end

# ──────────────────────────────────────────────────────────────
#  PSD triangle reordering + √2 scaling helpers
#
#  MOI uses column-major upper triangle:  (1,1),(1,2),(2,2),(1,3),(2,3),(3,3),…
#  ConicIP vecm uses row-major upper triangle: (1,1),(1,2),(1,3),(2,2),(2,3),(3,3),…
#  Additionally, vecm scales off-diagonal entries by √2.
# ──────────────────────────────────────────────────────────────

"""
Return `(perm, is_offdiag)` where `perm[moi_k]` is the vecm position
for MOI triangle position `moi_k`, and `is_offdiag[moi_k]` is true
when position `moi_k` corresponds to an off-diagonal entry.
"""
function _psd_moi_vecm_info(d::Int)
    n = round(Int, (sqrt(1 + 8*d) - 1) / 2)
    perm = zeros(Int, d)
    is_offdiag = falses(d)
    moi_k = 0
    for j in 1:n          # MOI: column-major
        for i in 1:j
            moi_k += 1
            # vecm position for (i,j) in row-major upper triangle
            before_i = (i - 1) * n - (i - 1) * (i - 2) ÷ 2
            vecm_k = before_i + (j - i + 1)
            perm[moi_k] = vecm_k
            is_offdiag[moi_k] = (i != j)
        end
    end
    return perm, is_offdiag
end

"""
Reorder rows of `Ai` and entries of `bi` from MOI triangle order to vecm
order, and scale off-diagonal rows by √2.
"""
function _psd_scale_input!(Ai::SparseMatrixCSC, bi::Vector{Float64}, dim::Int)
    perm, is_offdiag = _psd_moi_vecm_info(dim)
    Ai_copy = copy(Ai)
    bi_copy = copy(bi)
    s2 = √2
    for moi_k in 1:dim
        vecm_k = perm[moi_k]
        scale = is_offdiag[moi_k] ? s2 : 1.0
        Ai[vecm_k, :] = scale * Ai_copy[moi_k, :]
        bi[vecm_k] = scale * bi_copy[moi_k]
    end
end

"""
Convert a vector from vecm order (solver convention) to MOI triangle order,
dividing off-diagonal entries by √2.
"""
function _psd_vecm_to_moi(x::AbstractVector)
    d = length(x)
    perm, is_offdiag = _psd_moi_vecm_info(d)
    out = similar(x, Float64)
    s2inv = 1 / √2
    for moi_k in 1:d
        vecm_k = perm[moi_k]
        out[moi_k] = is_offdiag[moi_k] ? s2inv * x[vecm_k] : x[vecm_k]
    end
    return out
end

# ──────────────────────────────────────────────────────────────
#  optimize!
# ──────────────────────────────────────────────────────────────

function MOI.optimize!(dest::Optimizer, src::MOI.ModelLike)
    MOI.empty!(dest)

    model = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    index_map = MOI.copy_to(model, src)

    n = MOI.get(model, MOI.NumberOfVariables())
    dest.n = n

    # ── Objective ──
    sense = MOI.get(model, MOI.ObjectiveSense())
    dest.max_sense = (sense == MOI.MAX_SENSE)

    c_moi = zeros(n)
    obj_constant = 0.0
    obj_type = MOI.get(model, MOI.ObjectiveFunctionType())
    if obj_type == MOI.ScalarAffineFunction{Float64}
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        obj_constant = obj.constant
        for term in obj.terms
            c_moi[term.variable.value] += term.coefficient
        end
    elseif obj_type == MOI.VariableIndex
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
        c_moi[obj.value] = 1.0
    end
    dest.objective_constant = obj_constant

    # ConicIP minimizes (1/2)y'Qy - c'y
    # For min c_moi'x: set c_int = -c_moi  → minimizes -(-c_moi)'x = c_moi'x
    # For max c_moi'x: set c_int = c_moi   → minimizes -(c_moi)'x = -c_moi'x
    c_int = dest.max_sense ? c_moi : -c_moi
    dest.c_int = c_int
    Q = spzeros(n, n)

    # ── Constraints ──
    G_rows = Any[]
    d_vals = Float64[]
    A_rows = Any[]
    b_vals = Float64[]
    cone_dims = Tuple{String, Int}[]
    eq_row = 0
    ineq_row = 0

    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
            f = MOI.get(model, MOI.ConstraintFunction(), ci)
            s = MOI.get(model, MOI.ConstraintSet(), ci)

            if F <: Union{MOI.VectorAffineFunction{Float64}, MOI.VectorOfVariables}
                Ai, bi = _extract_vector_constraint(f, n)
                dim = size(Ai, 1)

                if S <: MOI.Zeros
                    # Ai*x + bi = 0 → G = Ai, d = -bi
                    push!(G_rows, Ai)
                    append!(d_vals, -bi)
                    push!(dest.eq_ci_map, ci => (eq_row+1):(eq_row+dim))
                    push!(dest.eq_offset, 0.0)
                    push!(dest.eq_is_scalar, false)
                    eq_row += dim
                elseif S <: MOI.Nonnegatives
                    # Ai*x + bi ≥ 0 → A_int = Ai, b_int = -bi
                    push!(A_rows, Ai)
                    append!(b_vals, -bi)
                    push!(cone_dims, ("R", dim))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+dim))
                    push!(dest.ineq_sign, 1.0)
                    push!(dest.ineq_offset, 0.0)
                    push!(dest.ineq_is_scalar, false)
                    push!(dest.ineq_is_psd, false)
                    ineq_row += dim
                elseif S <: MOI.Nonpositives
                    # Ai*x + bi ≤ 0 → -Ai*x - bi ≥ 0 → A_int = -Ai, b_int = bi
                    push!(A_rows, -Ai)
                    append!(b_vals, bi)
                    push!(cone_dims, ("R", dim))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+dim))
                    push!(dest.ineq_sign, -1.0)
                    push!(dest.ineq_offset, 0.0)
                    push!(dest.ineq_is_scalar, false)
                    push!(dest.ineq_is_psd, false)
                    ineq_row += dim
                elseif S <: MOI.SecondOrderCone
                    push!(A_rows, Ai)
                    append!(b_vals, -bi)
                    push!(cone_dims, ("Q", dim))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+dim))
                    push!(dest.ineq_sign, 1.0)
                    push!(dest.ineq_offset, 0.0)
                    push!(dest.ineq_is_scalar, false)
                    push!(dest.ineq_is_psd, false)
                    ineq_row += dim
                elseif S <: MOI.PositiveSemidefiniteConeTriangle
                    # MOI uses unscaled triangle; solver uses vecm (√2 off-diag)
                    _psd_scale_input!(Ai, bi, dim)
                    push!(A_rows, Ai)
                    append!(b_vals, -bi)
                    push!(cone_dims, ("S", dim))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+dim))
                    push!(dest.ineq_sign, 1.0)
                    push!(dest.ineq_offset, 0.0)
                    push!(dest.ineq_is_scalar, false)
                    push!(dest.ineq_is_psd, true)
                    ineq_row += dim
                end

            elseif F <: Union{MOI.ScalarAffineFunction{Float64}, MOI.VariableIndex}
                Ai, bi = _extract_scalar_constraint(f, n)

                if S <: MOI.EqualTo{Float64}
                    # Ai*x + bi = rhs → Ai*x = rhs - bi
                    rhs = MOI.constant(s)
                    push!(G_rows, Ai)
                    push!(d_vals, rhs - bi)
                    push!(dest.eq_ci_map, ci => (eq_row+1):(eq_row+1))
                    push!(dest.eq_offset, rhs)
                    push!(dest.eq_is_scalar, true)
                    eq_row += 1
                elseif S <: MOI.GreaterThan{Float64}
                    # Ai*x + bi ≥ lower → Ai*x ≥ lower - bi
                    lower = MOI.constant(s)
                    push!(A_rows, Ai)
                    push!(b_vals, lower - bi)
                    push!(cone_dims, ("R", 1))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+1))
                    push!(dest.ineq_sign, 1.0)
                    push!(dest.ineq_offset, lower)
                    push!(dest.ineq_is_scalar, true)
                    push!(dest.ineq_is_psd, false)
                    ineq_row += 1
                elseif S <: MOI.LessThan{Float64}
                    # Ai*x + bi ≤ upper → upper - Ai*x - bi ≥ 0
                    # (-Ai)*x - (bi - upper) ≥ 0 → A_int = -Ai, b_int = bi - upper
                    upper = MOI.constant(s)
                    push!(A_rows, -Ai)
                    push!(b_vals, bi - upper)
                    push!(cone_dims, ("R", 1))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+1))
                    push!(dest.ineq_sign, -1.0)
                    push!(dest.ineq_offset, upper)
                    push!(dest.ineq_is_scalar, true)
                    push!(dest.ineq_is_psd, false)
                    ineq_row += 1
                end
            end
        end
    end

    # ── Assemble matrices ──
    if isempty(G_rows)
        G = spzeros(0, n)
        d = zeros(0)
    else
        G = sparse(vcat(G_rows...))
        d = Float64.(d_vals)
    end
    dest.eq_G = G
    dest.eq_d = d

    if isempty(A_rows)
        A = spzeros(0, n)
        b = zeros(0)
    else
        A = sparse(vcat(A_rows...))
        b = Float64.(b_vals)
    end
    dest.ineq_A = A
    dest.ineq_b = b

    # ── Solve ──
    do_preprocess = get(dest.options, "preprocess", true)
    verbose = dest.silent ? false : get(dest.options, "verbose", false)
    solver = _resolve_kktsolver(get(dest.options, "kktsolver", "auto"))
    kw = (; (Symbol(k) => v for (k, v) in dest.options
             if k ∉ ("preprocess", "kktsolver", "verbose"))...)
    entry = do_preprocess ? preprocess_conicIP : conicIP
    t0 = time()
    dest.sol = entry(Q, c_int, A, b, cone_dims, G, d;
        verbose = verbose, kktsolver = solver, kw...)
    dest.solve_time = time() - t0

    return index_map, false
end

# ──────────────────────────────────────────────────────────────
#  Result getters
# ──────────────────────────────────────────────────────────────

# A `Solution` carries a ray only when the solver verified one
# (`has_certificate`). Per the `Solution` field-convention table:
#   :Unbounded + certificate → sol.y is the primal ray ȳ (cᵀȳ = +1),
#                              sol.s = A*ȳ, and sol.w/sol.v are NaN
#   :Infeasible + certificate → sol.w/sol.v are the Farkas ray
#                              (dᵀw̄ - bᵀv̄ = -1), and sol.y/sol.s are NaN
_is_primal_ray(model::Optimizer) =
    model.sol !== nothing && model.sol.status == :Unbounded && model.sol.has_certificate

_is_dual_ray(model::Optimizer) =
    model.sol !== nothing && model.sol.status == :Infeasible && model.sol.has_certificate

function MOI.get(model::Optimizer, ::MOI.TerminationStatus)
    if model.sol === nothing
        return MOI.OPTIMIZE_NOT_CALLED
    end
    status = model.sol.status
    if status == :Optimal
        return MOI.OPTIMAL
    elseif status == :Infeasible
        return MOI.INFEASIBLE
    elseif status == :Unbounded
        return MOI.DUAL_INFEASIBLE
    elseif status == :AlmostInfeasible
        return MOI.ALMOST_INFEASIBLE
    elseif status == :AlmostUnbounded
        return MOI.ALMOST_DUAL_INFEASIBLE
    elseif status == :Abandoned
        return MOI.ITERATION_LIMIT
    elseif status == :Error
        return MOI.NUMERICAL_ERROR
    else
        return MOI.OTHER_ERROR
    end
end

function MOI.get(model::Optimizer, attr::MOI.PrimalStatus)
    if model.sol === nothing || attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    status = model.sol.status
    if status == :Optimal
        return MOI.FEASIBLE_POINT
    elseif status == :Unbounded
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        return MOI.NO_SOLUTION
    end
end

function MOI.get(model::Optimizer, attr::MOI.DualStatus)
    if model.sol === nothing || attr.result_index > MOI.get(model, MOI.ResultCount())
        return MOI.NO_SOLUTION
    end
    status = model.sol.status
    if status == :Optimal
        return MOI.FEASIBLE_POINT
    elseif status == :Infeasible
        return MOI.INFEASIBILITY_CERTIFICATE
    else
        return MOI.NO_SOLUTION
    end
end

function MOI.get(model::Optimizer, ::MOI.ResultCount)
    if model.sol === nothing
        return 0
    end
    status = model.sol.status
    if status == :Optimal
        return 1
    elseif status in (:Infeasible, :Unbounded) && model.sol.has_certificate
        return 1
    end
    return 0
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.sol === nothing
        return "OPTIMIZE_NOT_CALLED"
    end
    if isempty(model.sol.message)
        return string(model.sol.status)
    end
    return string(model.sol.status, ": ", model.sol.message)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    # pobj = (1/2)y'Qy - c_int'y
    # MIN: c_int = -c_moi → pobj = c_moi'y (correct)
    # MAX: c_int = c_moi  → pobj = -c_moi'y (negate)
    if _is_primal_ray(model)
        # Homogeneous ray value: (1/2)ȳ'Qȳ - c_int'ȳ with Q ≡ 0, so -c_int'ȳ.
        # The ray is normalized to c_int'ȳ = +1, hence the internal value is -1;
        # the objective constant is *not* added (a ray is a direction).
        val = -dot(model.c_int, model.sol.y)
        return model.max_sense ? -val : val
    end
    val = model.sol.pobj
    if model.max_sense
        val = -val
    end
    return val + model.objective_constant
end

# On a dual ray (:Infeasible with certificate) ResultCount is 1, so the primal
# getters below pass `check_result_index_bounds`, but sol.y/sol.s are NaN by the
# `Solution` field convention. We deliberately do NOT throw: PrimalStatus is
# NO_SOLUTION, which is the documented signal that no primal point/ray exists,
# and MOI (and MOI.Test) does not query primal values in that state. A caller
# that ignores PrimalStatus gets NaN rather than an exception.
function MOI.get(
    model::Optimizer,
    attr::MOI.VariablePrimal,
    vi::MOI.VariableIndex,
)
    MOI.check_result_index_bounds(model, attr)
    return model.sol.y[vi.value]
end

# ConstraintPrimal: return f(x) for constraint f(x) ∈ S
#
# Inequality constraints use sol.s (cone slack: s = A_int*y - b_int).
#   Nonneg/SOC/PSD: f(x) = s           (sign=+1, offset=0)
#   Nonpositive:    f(x) = -s          (sign=-1, offset=0)
#   GreaterThan(L): f(x) = s + L       (sign=+1, offset=L)
#   LessThan(U):    f(x) = U - s       (sign=-1, offset=U)
# General formula: f(x) = sign * s + offset
#
# Equality constraints are approximately satisfied:
#   Zeros:       f(x) ≈ 0    (offset=0)
#   EqualTo(r):  f(x) ≈ r    (offset=r)
#
# On a primal ray (:Unbounded with certificate) the value is the *homogeneous*
# part only: the constant terms (eq_d, ineq_offset) are dropped, since a ray is
# a direction rather than a point.
function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    ray = _is_primal_ray(model)
    for (i, (ci_stored, rows)) in enumerate(model.eq_ci_map)
        if ci_stored == ci
            # f(x) = G[rows,:]*y - d[rows] + offset  (ray: G[rows,:]*ȳ)
            residual = ray ? model.eq_G[rows, :] * model.sol.y :
                model.eq_G[rows, :] * model.sol.y - model.eq_d[rows]
            if model.eq_is_scalar[i]
                return ray ? residual[1] : residual[1] + model.eq_offset[i]
            else
                return Vector(residual)
            end
        end
    end
    for (i, (ci_stored, rows)) in enumerate(model.ineq_ci_map)
        if ci_stored == ci
            sgn = model.ineq_sign[i]
            off = ray ? 0.0 : model.ineq_offset[i]
            if model.ineq_is_scalar[i]
                return sgn * model.sol.s[rows[1]] + off
            else
                val = Vector(sgn .* model.sol.s[rows])
                if model.ineq_is_psd[i]
                    return _psd_vecm_to_moi(val)
                end
                return val
            end
        end
    end
    error("Constraint index $ci not found")
end

# ConstraintDual: return MOI dual for constraint f(x) ∈ S
#
# The solver's v ∈ K* satisfies: Qy - c_int + A_int'v + G'w = 0
# For sets mapped with sign flip (Nonpositive, LessThan), the MOI dual
# is negated relative to v. For MAX_SENSE, all duals are negated.
# Formula: dual = sign * sense_sign * v  (ineq)
#          dual = sense_sign * w         (eq)
#
# This is already correct on a dual ray (:Infeasible with certificate): the
# Farkas ray lives in sol.w/sol.v with the same sign and PSD scaling
# conventions as the optimal duals, and no constant enters the formula.
function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintDual,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    # The KKT stationarity is Qy - c + G'w - A'v = 0, so:
    #   eq_dual = -w    (sign from -A' in KKT)
    #   ineq_dual = ineq_sign * v   (ineq_sign accounts for Nonpos/LessThan flip)
    # The conic dual convention is sense-independent (dual ∈ S*).
    for (i, (ci_stored, rows)) in enumerate(model.eq_ci_map)
        if ci_stored == ci
            if model.eq_is_scalar[i]
                return -model.sol.w[rows[1]]
            else
                return Vector(-1.0 .* model.sol.w[rows])
            end
        end
    end
    for (i, (ci_stored, rows)) in enumerate(model.ineq_ci_map)
        if ci_stored == ci
            sgn = model.ineq_sign[i]
            if model.ineq_is_scalar[i]
                return sgn * model.sol.v[rows[1]]
            else
                val = Vector(sgn .* model.sol.v[rows])
                if model.ineq_is_psd[i]
                    return _psd_vecm_to_moi(val)
                end
                return val
            end
        end
    end
    error("Constraint index $ci not found")
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return model.n
end

MOI.supports(::Optimizer, ::MOI.SolveTimeSec) = true
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

# Homogeneous dual objective along a Farkas ray. The ray is normalized so that
# dᵀw̄ - bᵀv̄ = -1, hence bᵀv̄ - dᵀw̄ = +1 — positive, matching the MOI
# convention that the dual objective improves without bound along the ray of a
# MIN problem. NOTE: the overall sign convention here is the risky part; if
# MOI.Test's Farkas-dual checks disagree, a single global flip is the fix.
function _dual_ray_objective(model::Optimizer)
    val = -(dot(model.eq_d, model.sol.w) - dot(model.ineq_b, model.sol.v))
    return model.max_sense ? -val : val
end

MOI.supports(::Optimizer, ::MOI.ObjectiveBound) = true
function MOI.get(model::Optimizer, ::MOI.ObjectiveBound)
    if model.sol === nothing
        return model.max_sense ? -Inf : Inf
    elseif _is_dual_ray(model)
        return _dual_ray_objective(model)
    end
    val = model.sol.dobj
    if model.max_sense
        val = -val
    end
    return val + model.objective_constant
end

MOI.supports(::Optimizer, ::MOI.DualObjectiveValue) = true
function MOI.get(model::Optimizer, attr::MOI.DualObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    if _is_dual_ray(model)
        # Ray value: no objective constant (a ray is a direction).
        return _dual_ray_objective(model)
    end
    val = model.sol.dobj
    if model.max_sense
        val = -val
    end
    return val + model.objective_constant
end
