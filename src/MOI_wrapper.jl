import MathOptInterface as MOI

"""
    Optimizer(; verbose=false, optTol=1e-6, maxIters=100)

MathOptInterface optimizer wrapping the ConicIP interior-point solver.
Use as a JuMP solver via `Model(ConicIP.Optimizer)`.

# Keyword Arguments
- `verbose::Bool` -- print solver iterations (default: `false`)
- `optTol::Float64` -- optimality tolerance (default: `1e-6`)
- `maxIters::Int` -- maximum iterations (default: `100`)

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
    # Constraint row tracking for primal/dual recovery
    eq_ci_map::Vector{Pair{Any, UnitRange{Int}}}
    eq_offset::Vector{Float64}     # 0 for Zeros, rhs for EqualTo
    eq_is_scalar::Vector{Bool}
    ineq_ci_map::Vector{Pair{Any, UnitRange{Int}}}
    ineq_sign::Vector{Float64}     # +1 or -1 (Nonpositive/LessThan flip)
    ineq_offset::Vector{Float64}   # 0 for vector, lower/upper for scalar
    ineq_is_scalar::Vector{Bool}
    # Timing
    solve_time::Float64
    # Solver options
    verbose::Bool
    optTol::Float64
    maxIters::Int
end

function Optimizer(; verbose::Bool = false, optTol::Float64 = 1e-6, maxIters::Int = 100)
    return Optimizer(
        nothing, false, 0.0, 0,
        Pair{Any, UnitRange{Int}}[], Float64[], Bool[],
        Pair{Any, UnitRange{Int}}[], Float64[], Float64[], Bool[],
        NaN,
        verbose, optTol, maxIters,
    )
end

function MOI.empty!(model::Optimizer)
    model.sol = nothing
    model.max_sense = false
    model.objective_constant = 0.0
    model.n = 0
    model.solve_time = NaN
    empty!(model.eq_ci_map)
    empty!(model.eq_offset)
    empty!(model.eq_is_scalar)
    empty!(model.ineq_ci_map)
    empty!(model.ineq_sign)
    empty!(model.ineq_offset)
    empty!(model.ineq_is_scalar)
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
                    ineq_row += dim
                elseif S <: MOI.SecondOrderCone
                    push!(A_rows, Ai)
                    append!(b_vals, -bi)
                    push!(cone_dims, ("Q", dim))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+dim))
                    push!(dest.ineq_sign, 1.0)
                    push!(dest.ineq_offset, 0.0)
                    push!(dest.ineq_is_scalar, false)
                    ineq_row += dim
                elseif S <: MOI.PositiveSemidefiniteConeTriangle
                    push!(A_rows, Ai)
                    append!(b_vals, -bi)
                    push!(cone_dims, ("S", dim))
                    push!(dest.ineq_ci_map, ci => (ineq_row+1):(ineq_row+dim))
                    push!(dest.ineq_sign, 1.0)
                    push!(dest.ineq_offset, 0.0)
                    push!(dest.ineq_is_scalar, false)
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

    if isempty(A_rows)
        A = spzeros(0, n)
        b = zeros(0)
    else
        A = sparse(vcat(A_rows...))
        b = Float64.(b_vals)
    end

    # ── Solve ──
    t0 = time()
    dest.sol = preprocess_conicIP(Q, c_int, A, b, cone_dims, G, d;
        verbose = dest.verbose,
        optTol = dest.optTol,
        maxIters = dest.maxIters,
    )
    dest.solve_time = time() - t0

    return index_map, false
end

# ──────────────────────────────────────────────────────────────
#  Result getters
# ──────────────────────────────────────────────────────────────

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
    elseif status == :Abandoned
        return MOI.ITERATION_LIMIT
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
    return model.sol.status == :Optimal ? 1 : 0
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.sol === nothing
        return "OPTIMIZE_NOT_CALLED"
    end
    return string(model.sol.status)
end

function MOI.get(model::Optimizer, attr::MOI.ObjectiveValue)
    MOI.check_result_index_bounds(model, attr)
    # pobj = (1/2)y'Qy - c_int'y
    # MIN: c_int = -c_moi → pobj = c_moi'y (correct)
    # MAX: c_int = c_moi  → pobj = -c_moi'y (negate)
    val = model.sol.pobj
    if model.max_sense
        val = -val
    end
    return val + model.objective_constant
end

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
function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    for (i, (ci_stored, rows)) in enumerate(model.eq_ci_map)
        if ci_stored == ci
            if model.eq_is_scalar[i]
                return model.eq_offset[i]
            else
                return zeros(length(rows))
            end
        end
    end
    for (i, (ci_stored, rows)) in enumerate(model.ineq_ci_map)
        if ci_stored == ci
            sgn = model.ineq_sign[i]
            off = model.ineq_offset[i]
            if model.ineq_is_scalar[i]
                return sgn * model.sol.s[rows[1]] + off
            else
                return Vector(sgn .* model.sol.s[rows])
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
                return Vector(sgn .* model.sol.v[rows])
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

MOI.supports(::Optimizer, ::MOI.ObjectiveBound) = true
function MOI.get(model::Optimizer, ::MOI.ObjectiveBound)
    if model.sol === nothing
        return model.max_sense ? -Inf : Inf
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
    val = model.sol.dobj
    if model.max_sense
        val = -val
    end
    return val + model.objective_constant
end
