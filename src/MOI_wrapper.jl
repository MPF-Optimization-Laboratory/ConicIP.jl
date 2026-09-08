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
- `kktsolver` -- `"auto"` (default; picks by cone mix and predicted
  factorization cost via [`choose_kktsolver`](@ref)), `"ldl"`, `"qr"`,
  `"sparse"`, `"2x2"`, or any callable solver object (a function or an
  instance such as `cached_kktsolver_ldl()`)
- `preprocess::Bool` -- remove redundant equality rows via
  [`preprocess_conicIP`](@ref) before solving (default: `true`)
- `equilibrate::Bool` -- Ruiz-scale the data before solving (default: `true`)
- `timeLimit::Float64` -- wall-clock budget in seconds (also `MOI.TimeLimitSec`)
- `assemble_only::Bool` -- stop `optimize!` once the solver's matrices
  (`Q_int`, `c_int`, `ineq_A`, `ineq_b`, `cone_dims`, `eq_G`, `eq_d`) are
  assembled, without calling the solver; the model then reports
  `OPTIMIZE_NOT_CALLED` and `ResultCount == 0` (default: `false`)
- plus `infeasAbsTol`, `DTB`, `maxRefinementSteps`, `refineRelTol`,
  `refineAbsTol`, `staticReg`, `certFallback`, `certFallbackIters`,
  `cache_nestodd` — forwarded to [`conicIP`](@ref)

`MOI.Silent` is supported and overrides `verbose`.

# Supported Objectives
Affine and convex quadratic (`ScalarQuadraticFunction`), both handled
natively: a quadratic objective becomes the solver's `Q` rather than a
second-order-cone reformulation, so positive semidefinite but singular
Hessians are fine. A Hessian that is not positive semidefinite for the
given sense (a nonconvex QP) is rejected before the solve with
`TerminationStatus == INVALID_MODEL` and no result.

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
    cone_dims::Vector{Tuple{String, Int}} # original assembled cone partition
    c_int::Vector{Float64}         # internal objective vector handed to the solver
    Q_int::Union{Nothing, SparseMatrixCSC{Float64, Int}}  # internal Hessian (nothing = 0)
    # Constraint row tracking for primal/dual recovery. Each constraint
    # index maps to a slot i; the per-slot vectors below are indexed by i.
    eq_ci_map::Dict{MOI.ConstraintIndex, Int}
    eq_rows::Vector{UnitRange{Int}}
    eq_offset::Vector{Float64}     # 0 for Zeros, rhs for EqualTo
    eq_is_scalar::Vector{Bool}
    eq_G::Union{Nothing, SparseMatrixCSC{Float64, Int}}  # equality constraint matrix
    eq_d::Vector{Float64}                                 # equality constraint RHS
    eq_Gy::Vector{Float64}         # G*y of the returned point, computed once
    ineq_ci_map::Dict{MOI.ConstraintIndex, Int}
    ineq_rows::Vector{UnitRange{Int}}
    ineq_sign::Vector{Float64}     # +1 or -1 (Nonpositive/LessThan flip)
    ineq_offset::Vector{Float64}   # 0 for vector, lower/upper for scalar
    ineq_is_scalar::Vector{Bool}
    ineq_is_psd::Vector{Bool}     # true for PSD constraints (√2 scaling)
    ineq_A::Union{Nothing, SparseMatrixCSC{Float64, Int}}  # inequality constraint matrix
    ineq_b::Vector{Float64}                                # inequality constraint RHS
    ineq_Ay::Vector{Float64}       # A*y of the returned point, computed once
    # Timing: solve_time is the whole optimize! (assembly included);
    # assembly_time is the part before the solver is called.
    solve_time::Float64
    assembly_time::Float64
    # Solver options: only the ones explicitly set are stored, so the
    # solver's own defaults (and the preprocessor's dynamic staticReg
    # opt-in) stay in charge of everything else.
    options::Dict{String, Any}
    silent::Bool
    # True when the last optimize! stopped after assembly (`assemble_only`);
    # `sol` is then `nothing`, as before any optimize! call.
    assembled_only::Bool
end

# Options settable via MOI.RawOptimizerAttribute (and Optimizer kwargs)
const _SUPPORTED_OPTIONS = (
    "verbose", "optTol", "maxIters", "infeasTol", "infeasAbsTol", "DTB",
    "maxRefinementSteps", "refineRelTol", "refineAbsTol", "staticReg",
    "certFallback", "certFallbackIters", "cache_nestodd", "kktsolver",
    "preprocess", "rank_check", "fix_singletons", "timeLimit", "equilibrate",
    "assemble_only",
)

# Map a kktsolver name to the solver constructor. Accepts the name
# "auto" | "ldl" | "qr" | "sparse" | "2x2"/"pivot", or any callable solver
# object — a function such as `kktsolver_ldl` or a callable struct such as
# `cached_kktsolver_ldl()` — which is returned as is.
function _resolve_kktsolver(v)
    v isa Union{AbstractString, Symbol} || return v
    s = lowercase(string(v))
    s == "auto"             && return default_kktsolver
    s == "ldl"              && return kktsolver_ldl
    s == "qr"               && return kktsolver_qr
    s == "sparse"           && return kktsolver_sparse
    s in ("2x2", "pivot")   && return pivot(kktsolver_2x2)
    throw(ArgumentError(
        "unknown kktsolver \"$v\" (expected \"auto\", \"ldl\", \"qr\", " *
        "\"sparse\", \"2x2\", or a callable solver object)"))
end

# Convexity guard for the objective: the solver assumes ½yᵀQy is convex,
# so `Q` (already sign-adjusted for the sense) must be positive
# semidefinite. A Cholesky attempt on Q + δI with δ tiny relative to the
# entries accepts singular PSD Hessians (a diagonal with zeros, or the
# Maros–Mészáros rank-deficient QPs) and rejects indefinite ones.
function _is_psd(Q::SparseMatrixCSC{Float64, Int})
    nnz(Q) == 0 && return true
    all(isfinite, nonzeros(Q)) || return false
    dq = diag(Q)
    any(<(0), dq) && return false
    # For a PSD matrix, a zero diagonal implies a zero row and column.
    # Congruence-scale the positive diagonal to one before shifting;
    # a large, unrelated block must not hide negative curvature elsewhere.
    for j in axes(Q, 2), t in nzrange(Q, j)
        i = rowvals(Q)[t]
        if (dq[i] == 0 || dq[j] == 0) && nonzeros(Q)[t] != 0
            return false
        end
    end
    scale = [x > 0 ? 1 / sqrt(x) : 1.0 for x in dq]
    Qs = Diagonal(scale) * Q * Diagonal(scale)
    F = cholesky(Symmetric(Qs); shift = 2e-10, check = false)
    return issuccess(F)
end

const _NONCONVEX_MESSAGE =
    "objective Hessian is not positive semidefinite for the given sense " *
    "(nonconvex QP); ConicIP solves convex problems only"

# A `Solution` standing in for a solve that was never run because the model
# is outside the solver's class. `:InvalidModel` maps to
# `MOI.INVALID_MODEL` with `ResultCount == 0`; the vectors are NaN so an
# accidental read is visibly meaningless.
_invalid_model_solution(n::Int, mA::Int, mG::Int, message::String) =
    Solution(fill(NaN, n), fill(NaN, mG), fill(NaN, mA), fill(NaN, mA),
             :InvalidModel, 0, NaN, NaN, NaN, NaN, NaN, NaN, false, message, 0)

function Optimizer(; kwargs...)
    model = Optimizer(
        nothing, false, 0.0, 0, Tuple{String, Int}[], Float64[], nothing,
        Dict{MOI.ConstraintIndex, Int}(), UnitRange{Int}[], Float64[], Bool[],
        nothing, Float64[], Float64[],
        Dict{MOI.ConstraintIndex, Int}(), UnitRange{Int}[], Float64[], Float64[],
        Bool[], Bool[], nothing, Float64[], Float64[],
        NaN, NaN,
        Dict{String, Any}(), false, false,
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
        "maxRefinementSteps" => 3, "refineRelTol" => 1e-13,
        "refineAbsTol" => 1e-12, "staticReg" => 0.0,
        "certFallback" => true, "certFallbackIters" => 50,
        "cache_nestodd" => false, "kktsolver" => "auto",
        "preprocess" => true, "rank_check" => "auto", "fix_singletons" => true,
        "timeLimit" => Inf, "equilibrate" => true, "assemble_only" => false)
    return get(model.options, attr.name, defaults[attr.name])
end

# MOI.TimeLimitSec is the "timeLimit" option; `nothing` clears it.
MOI.supports(::Optimizer, ::MOI.TimeLimitSec) = true
function MOI.set(model::Optimizer, ::MOI.TimeLimitSec, value::Union{Nothing, Real})
    if value === nothing
        delete!(model.options, "timeLimit")
    else
        model.options["timeLimit"] = Float64(value)
    end
    return
end
function MOI.get(model::Optimizer, ::MOI.TimeLimitSec)
    v = get(model.options, "timeLimit", Inf)
    return isfinite(v) ? v : nothing
end

MOI.supports(::Optimizer, ::MOI.Silent) = true
MOI.set(model::Optimizer, ::MOI.Silent, value::Bool) = (model.silent = value; nothing)
MOI.get(model::Optimizer, ::MOI.Silent) = model.silent

function MOI.empty!(model::Optimizer)
    model.sol = nothing
    model.assembled_only = false
    model.max_sense = false
    model.objective_constant = 0.0
    model.n = 0
    model.solve_time = NaN
    model.assembly_time = NaN
    empty!(model.cone_dims)
    empty!(model.c_int)
    model.Q_int = nothing
    empty!(model.eq_ci_map)
    empty!(model.eq_rows)
    empty!(model.eq_offset)
    empty!(model.eq_is_scalar)
    model.eq_G = nothing
    empty!(model.eq_d)
    empty!(model.eq_Gy)
    empty!(model.ineq_ci_map)
    empty!(model.ineq_rows)
    empty!(model.ineq_sign)
    empty!(model.ineq_offset)
    empty!(model.ineq_is_scalar)
    empty!(model.ineq_is_psd)
    model.ineq_A = nothing
    empty!(model.ineq_b)
    empty!(model.ineq_Ay)
end

function MOI.is_empty(model::Optimizer)
    return model.sol === nothing && model.n == 0
end

MOI.get(::Optimizer, ::MOI.SolverName) = "ConicIP"
MOI.get(::Optimizer, ::MOI.SolverVersion) = string(pkgversion(@__MODULE__))

# Interior-point solver — no simplex basis information
MOI.supports(::Optimizer, ::MOI.VariableBasisStatus) = false
MOI.supports(::Optimizer, ::MOI.ConstraintBasisStatus) = false

# Supported objective: affine and (convex) quadratic, both native
MOI.supports(::Optimizer, ::MOI.ObjectiveSense) = true
function MOI.supports(
    ::Optimizer,
    ::MOI.ObjectiveFunction{F},
) where {F<:Union{MOI.ScalarAffineFunction{Float64},
                  MOI.ScalarQuadraticFunction{Float64},
                  MOI.VariableIndex}}
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

# Triplet accumulator for one constraint matrix. Every constraint appends
# its rows here; the matrix is built once by `sparse(I, J, V, m, n)` at the
# end (duplicates summed), so assembly is linear in the number of
# nonzeros rather than one small CSC object per constraint.
struct _Triplets
    I::Vector{Int}
    J::Vector{Int}
    V::Vector{Float64}
    rhs::Vector{Float64}
end
_Triplets() = _Triplets(Int[], Int[], Float64[], Float64[])
_nrows(t::_Triplets) = length(t.rhs)

# Append the rows of `sign * f(x) + rhs_shift` with `f` a vector function.
# `rowmap[k]` gives the local row for output index k and `rowscale[k]` a
# per-row factor (the PSD triangle permutation and √2 scaling); both are
# `nothing` for the identity. The constants of `f` are appended to the
# accumulator's rhs *negated* (the solver's convention is A y ≥ b,
# G y = d with the constant moved to the right-hand side).
function _append_vector!(t::_Triplets, f, dim::Int, sign::Float64;
                         rowmap = nothing, rowscale = nothing)
    r0 = _nrows(t)
    resize!(t.rhs, r0 + dim)
    @inbounds for k in 1:dim
        t.rhs[r0 + k] = 0.0
    end
    if f isa MOI.VectorOfVariables
        for (k, vi) in enumerate(f.variables)
            row = rowmap === nothing ? k : rowmap[k]
            sc  = rowscale === nothing ? 1.0 : rowscale[k]
            push!(t.I, r0 + row); push!(t.J, vi.value); push!(t.V, sign * sc)
        end
    else  # VectorAffineFunction
        for term in f.terms
            k   = term.output_index
            row = rowmap === nothing ? k : rowmap[k]
            sc  = rowscale === nothing ? 1.0 : rowscale[k]
            push!(t.I, r0 + row)
            push!(t.J, term.scalar_term.variable.value)
            push!(t.V, sign * sc * term.scalar_term.coefficient)
        end
        for (k, ck) in enumerate(f.constants)
            row = rowmap === nothing ? k : rowmap[k]
            sc  = rowscale === nothing ? 1.0 : rowscale[k]
            t.rhs[r0 + row] = -sign * sc * ck
        end
    end
    return (r0 + 1):(r0 + dim)
end

# Append one row `sign * f(x)` with `f` scalar; the constant of `f` and the
# set's bound go to the rhs as `sign * (bound − constant)`.
function _append_scalar!(t::_Triplets, f, sign::Float64, bound::Float64)
    r = _nrows(t) + 1
    const_f = 0.0
    if f isa MOI.VariableIndex
        push!(t.I, r); push!(t.J, f.value); push!(t.V, sign)
    else
        const_f = f.constant
        for term in f.terms
            push!(t.I, r); push!(t.J, term.variable.value)
            push!(t.V, sign * term.coefficient)
        end
    end
    push!(t.rhs, sign * (bound - const_f))
    return r:r
end

_assemble(t::_Triplets, n::Int) =
    (sparse(t.I, t.J, t.V, _nrows(t), n), copy(t.rhs))

# Append an orthant block of `dim` rows to `cone_dims`, merging it into a
# preceding orthant block: the cone product is separable over R₊, and one
# block instead of thousands keeps the per-block loops in the solver short.
function _push_orthant!(cone_dims, dim::Int)
    if !isempty(cone_dims) && cone_dims[end][1] == "R"
        cone_dims[end] = ("R", cone_dims[end][2] + dim)
    else
        push!(cone_dims, ("R", dim))
    end
    return cone_dims
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
    t_start = time()

    model = MOI.Utilities.UniversalFallback(MOI.Utilities.Model{Float64}())
    index_map = MOI.copy_to(model, src)

    n = MOI.get(model, MOI.NumberOfVariables())
    dest.n = n

    # ── Objective ──
    sense = MOI.get(model, MOI.ObjectiveSense())
    dest.max_sense = (sense == MOI.MAX_SENSE)

    # MOI objective: ½xᵀQx + aᵀx + const (Q symmetric; a diagonal quadratic
    # term with coefficient q means ½·q·xᵢ², an off-diagonal one q·xᵢxⱼ).
    c_moi = zeros(n)
    obj_constant = 0.0
    QI = Int[]; QJ = Int[]; QV = Float64[]
    # A feasibility model ignores any objective retained in the cache.
    obj_type = sense == MOI.FEASIBILITY_SENSE ? Nothing : MOI.get(model, MOI.ObjectiveFunctionType())
    if obj_type == MOI.ScalarAffineFunction{Float64}
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarAffineFunction{Float64}}())
        obj_constant = obj.constant
        for term in obj.terms
            c_moi[term.variable.value] += term.coefficient
        end
    elseif obj_type == MOI.ScalarQuadraticFunction{Float64}
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.ScalarQuadraticFunction{Float64}}())
        obj_constant = obj.constant
        for term in obj.affine_terms
            c_moi[term.variable.value] += term.coefficient
        end
        for term in obj.quadratic_terms
            i = term.variable_1.value; j = term.variable_2.value
            push!(QI, i); push!(QJ, j); push!(QV, term.coefficient)
            if i != j
                push!(QI, j); push!(QJ, i); push!(QV, term.coefficient)
            end
        end
    elseif obj_type == MOI.VariableIndex
        obj = MOI.get(model, MOI.ObjectiveFunction{MOI.VariableIndex}())
        c_moi[obj.value] = 1.0
    end
    dest.objective_constant = obj_constant

    # ConicIP minimizes (1/2)y'Qy - c'y
    # MIN ½xᵀQx + aᵀx: Q_int = Q,  c_int = -a
    # MAX ½xᵀQx + aᵀx: Q_int = -Q, c_int = a   (Q negative semidefinite)
    c_int = dest.max_sense ? c_moi : -c_moi
    dest.c_int = c_int
    if isempty(QI)
        Q = spzeros(n, n)
        dest.Q_int = nothing
    else
        Q = sparse(QI, QJ, dest.max_sense ? -QV : QV, n, n)
        dest.Q_int = Q
    end

    # ── Constraints ──
    tG = _Triplets()          # equality rows,   G y = d
    tA = _Triplets()          # cone rows,       A y ≥_K b
    cone_dims = Tuple{String, Int}[]

    function record_eq!(ci, rows, offset, scalar)
        push!(dest.eq_rows, rows); push!(dest.eq_offset, offset)
        push!(dest.eq_is_scalar, scalar)
        dest.eq_ci_map[ci] = length(dest.eq_rows)
    end
    function record_ineq!(ci, rows, sign, offset, scalar, psd)
        push!(dest.ineq_rows, rows); push!(dest.ineq_sign, sign)
        push!(dest.ineq_offset, offset); push!(dest.ineq_is_scalar, scalar)
        push!(dest.ineq_is_psd, psd)
        dest.ineq_ci_map[ci] = length(dest.ineq_rows)
    end

    for (F, S) in MOI.get(model, MOI.ListOfConstraintTypesPresent())
        for ci in MOI.get(model, MOI.ListOfConstraintIndices{F, S}())
            f = MOI.get(model, MOI.ConstraintFunction(), ci)
            s = MOI.get(model, MOI.ConstraintSet(), ci)

            if F <: Union{MOI.VectorAffineFunction{Float64}, MOI.VectorOfVariables}
                dim = MOI.output_dimension(f)
                if S <: MOI.Zeros
                    # f(x) = 0:  G = A_f, d = -const
                    rows = _append_vector!(tG, f, dim, 1.0)
                    record_eq!(ci, rows, 0.0, false)
                elseif S <: MOI.Nonnegatives
                    rows = _append_vector!(tA, f, dim, 1.0)
                    _push_orthant!(cone_dims, dim)
                    record_ineq!(ci, rows, 1.0, 0.0, false, false)
                elseif S <: MOI.Nonpositives
                    # f(x) ≤ 0  →  -f(x) ≥ 0
                    rows = _append_vector!(tA, f, dim, -1.0)
                    _push_orthant!(cone_dims, dim)
                    record_ineq!(ci, rows, -1.0, 0.0, false, false)
                elseif S <: MOI.SecondOrderCone
                    rows = _append_vector!(tA, f, dim, 1.0)
                    push!(cone_dims, ("Q", dim))
                    record_ineq!(ci, rows, 1.0, 0.0, false, false)
                elseif S <: MOI.PositiveSemidefiniteConeTriangle
                    # MOI's column-major unscaled triangle → vecm's row-major
                    # triangle with √2 on the off-diagonal rows, applied to
                    # the triplets directly.
                    perm, is_offdiag = _psd_moi_vecm_info(dim)
                    scale = [od ? √2 : 1.0 for od in is_offdiag]
                    rows = _append_vector!(tA, f, dim, 1.0;
                                           rowmap = perm, rowscale = scale)
                    push!(cone_dims, ("S", dim))
                    record_ineq!(ci, rows, 1.0, 0.0, false, true)
                end

            elseif F <: Union{MOI.ScalarAffineFunction{Float64}, MOI.VariableIndex}
                if S <: MOI.EqualTo{Float64}
                    rhs = MOI.constant(s)
                    rows = _append_scalar!(tG, f, 1.0, rhs)
                    record_eq!(ci, rows, rhs, true)
                elseif S <: MOI.GreaterThan{Float64}
                    lower = MOI.constant(s)
                    rows = _append_scalar!(tA, f, 1.0, lower)
                    _push_orthant!(cone_dims, 1)
                    record_ineq!(ci, rows, 1.0, lower, true, false)
                elseif S <: MOI.LessThan{Float64}
                    # f(x) ≤ u  →  -f(x) ≥ -u
                    upper = MOI.constant(s)
                    rows = _append_scalar!(tA, f, -1.0, upper)
                    _push_orthant!(cone_dims, 1)
                    record_ineq!(ci, rows, -1.0, upper, true, false)
                end
            end
        end
    end

    # ── Assemble matrices (one sparse() per matrix) ──
    G, d = _assemble(tG, n)
    A, b = _assemble(tA, n)
    dest.cone_dims = cone_dims
    dest.eq_G = G;   dest.eq_d = d
    dest.ineq_A = A; dest.ineq_b = b
    dest.assembly_time = time() - t_start

    # Assembly only: the matrices are available through the fields above;
    # `sol` stays `nothing`, so every result getter answers as if
    # optimize! had not been called.
    if get(dest.options, "assemble_only", false)
        dest.assembled_only = true
        dest.solve_time = time() - t_start
        return index_map, false
    end

    # ── Solve ──
    do_preprocess = get(dest.options, "preprocess", true)
    verbose = dest.silent ? false : get(dest.options, "verbose", false)
    solver = _resolve_kktsolver(get(dest.options, "kktsolver", "auto"))
    skip = do_preprocess ? ("preprocess", "kktsolver", "verbose", "assemble_only") :
                           ("preprocess", "kktsolver", "verbose", "assemble_only",
                            "rank_check", "fix_singletons")
    kw = (; (Symbol(k) => v for (k, v) in dest.options if k ∉ skip)...)
    entry = do_preprocess ? preprocess_conicIP : conicIP
    if dest.Q_int !== nothing && !_is_psd(Q)
        # Nonconvex objective: the solver would report a stationary point
        # as optimal, so it is not called at all.
        dest.sol = _invalid_model_solution(n, size(A, 1), size(G, 1), _NONCONVEX_MESSAGE)
    else
        # Charge all front-end work, including the Hessian check, before
        # starting the solver's budget. Inner wrappers charge their own
        # presolve/equilibration work in the same way.
        if haskey(kw, :timeLimit) && isfinite(kw.timeLimit)
            kw = merge(kw, (; timeLimit = kw.timeLimit - (time() - t_start)))
        end
        dest.sol = entry(Q, c_int, A, b, cone_dims, G, d;
            verbose = verbose, kktsolver = solver, kw...)
    end

    # Products needed by the result getters, formed once. The inequality
    # product is used instead of the slack `sol.s` so that ConstraintPrimal
    # is f(y) at the returned point even when A y − b ≠ s.
    y = dest.sol.y
    finite_y = all(isfinite, y)
    dest.eq_Gy   = (size(G, 1) > 0 && finite_y) ? Vector(G * y) : fill(NaN, size(G, 1))
    dest.ineq_Ay = (size(A, 1) > 0 && finite_y) ? Vector(A * y) : fill(NaN, size(A, 1))
    dest.solve_time = time() - t_start

    return index_map, false
end

# ──────────────────────────────────────────────────────────────
#  Result getters
# ──────────────────────────────────────────────────────────────

# A `Solution` carries a ray only when the solver verified one
# (`has_certificate`). Per the `Solution` field-convention table:
#   :DualInfeasible + certificate → sol.y is the primal ray ȳ (cᵀȳ = +1),
#                              sol.s = A*ȳ, and sol.w/sol.v are NaN
#   :Infeasible + certificate → sol.w/sol.v are the Farkas ray
#                              (dᵀw̄ - bᵀv̄ = -1), and sol.y/sol.s are NaN
_is_primal_ray(model::Optimizer) =
    model.sol !== nothing && model.sol.status == :DualInfeasible && model.sol.has_certificate

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
    elseif status == :DualInfeasible
        return MOI.DUAL_INFEASIBLE
    elseif status == :AlmostInfeasible
        return MOI.ALMOST_INFEASIBLE
    elseif status == :AlmostDualInfeasible
        return MOI.ALMOST_DUAL_INFEASIBLE
    elseif status == :Abandoned
        return MOI.ITERATION_LIMIT
    elseif status == :TimeLimit
        return MOI.TIME_LIMIT
    elseif status == :Error
        return MOI.NUMERICAL_ERROR
    elseif status == :InvalidModel
        return MOI.INVALID_MODEL
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
    elseif status == :DualInfeasible
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
    elseif status in (:Infeasible, :DualInfeasible) && model.sol.has_certificate
        return 1
    end
    return 0
end

function MOI.get(model::Optimizer, ::MOI.RawStatusString)
    if model.sol === nothing
        return model.assembled_only ?
               "OPTIMIZE_NOT_CALLED: assembly only (assemble_only = true)" :
               "OPTIMIZE_NOT_CALLED"
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
# Inequality rows are stored as A_int y ≥_K b_int with, per constraint,
# r = A_int[rows,:] y − b_int[rows]:
#   Nonneg/SOC/PSD: f(x) = r           (sign=+1, offset=0)
#   Nonpositive:    f(x) = -r          (sign=-1, offset=0)
#   GreaterThan(L): f(x) = r + L       (sign=+1, offset=L)
#   LessThan(U):    f(x) = U - r       (sign=-1, offset=U)
# General formula: f(x) = sign * r + offset.
# `r` is evaluated from the cached product A_int y (`ineq_Ay`), not from
# the cone slack `sol.s`: the two agree only at convergence (A y − s = b
# is a residual the solver drives to zero), and reporting the slack would
# hide the primal violation of a non-converged iterate.
#
# Equality constraints are approximately satisfied:
#   Zeros:       f(x) ≈ 0    (offset=0)
#   EqualTo(r):  f(x) ≈ r    (offset=r)
#
# On a primal ray (:DualInfeasible with certificate) the value is the *homogeneous*
# part only: the constant terms (eq_d, ineq_b, ineq_offset) are dropped, since a
# ray is a direction rather than a point (sol.s holds A_int ȳ there as well).
function MOI.get(
    model::Optimizer,
    attr::MOI.ConstraintPrimal,
    ci::MOI.ConstraintIndex,
)
    MOI.check_result_index_bounds(model, attr)
    ray = _is_primal_ray(model)
    i = get(model.eq_ci_map, ci, 0)
    if i > 0
        rows = model.eq_rows[i]
        # f(x) = G[rows,:]*y - d[rows] + offset  (ray: G[rows,:]*ȳ)
        residual = ray ? model.eq_Gy[rows] : model.eq_Gy[rows] - model.eq_d[rows]
        if model.eq_is_scalar[i]
            return ray ? residual[1] : residual[1] + model.eq_offset[i]
        else
            return residual
        end
    end
    i = get(model.ineq_ci_map, ci, 0)
    if i > 0
        rows = model.ineq_rows[i]
        sgn = model.ineq_sign[i]
        off = ray ? 0.0 : model.ineq_offset[i]
        # r = A[rows,:]*y - b[rows]  (ray: A[rows,:]*ȳ)
        if model.ineq_is_scalar[i]
            r = ray ? model.ineq_Ay[rows[1]] : model.ineq_Ay[rows[1]] - model.ineq_b[rows[1]]
            return sgn * r + off
        else
            r = ray ? model.ineq_Ay[rows] : model.ineq_Ay[rows] - model.ineq_b[rows]
            val = sgn .* r
            if model.ineq_is_psd[i]
                return _psd_vecm_to_moi(val)
            end
            return val
        end
    end
    error("Constraint index $ci not found")
end

# ConstraintDual: return MOI dual for constraint f(x) ∈ S
#
# The solver's v ∈ K* satisfies the stationarity condition written out
# below; for sets mapped with a sign flip (Nonpositive, LessThan) the MOI
# dual is negated relative to v. The formulas are sense-independent: the
# objective sense is already folded into Q and c_int, and the conic dual
# convention (dual ∈ S*) does not change with the sense.
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
    i = get(model.eq_ci_map, ci, 0)
    if i > 0
        rows = model.eq_rows[i]
        if model.eq_is_scalar[i]
            return -model.sol.w[rows[1]]
        else
            return -1.0 .* model.sol.w[rows]
        end
    end
    i = get(model.ineq_ci_map, ci, 0)
    if i > 0
        rows = model.ineq_rows[i]
        sgn = model.ineq_sign[i]
        if model.ineq_is_scalar[i]
            return sgn * model.sol.v[rows[1]]
        else
            val = sgn .* model.sol.v[rows]
            if model.ineq_is_psd[i]
                return _psd_vecm_to_moi(val)
            end
            return val
        end
    end
    error("Constraint index $ci not found")
end

function MOI.get(model::Optimizer, ::MOI.NumberOfVariables)
    return model.n
end

MOI.supports(::Optimizer, ::MOI.SolveTimeSec) = true
MOI.get(model::Optimizer, ::MOI.SolveTimeSec) = model.solve_time

MOI.supports(::Optimizer, ::MOI.BarrierIterations) = true
MOI.get(model::Optimizer, ::MOI.BarrierIterations) =
    model.sol === nothing ? 0 : Int(model.sol.Iter)

# The wrapper is its own solver object: `model.sol` is the full `Solution`
# (iteration and KKT-solve counts, residuals, message) behind the MOI
# attributes, and benchmark/suite.jl reads it through this attribute.
MOI.get(model::Optimizer, ::MOI.RawSolver) = model

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

# Relative duality gap |vᵀs| / (1 + |pobj + offset|) of the returned point,
# the quantity the solver's gap test measures (`Solution.rGap`). NaN when
# there is no result to describe.
MOI.supports(::Optimizer, ::MOI.RelativeGap) = true
function MOI.get(model::Optimizer, ::MOI.RelativeGap)
    sol = model.sol
    (sol === nothing || MOI.get(model, MOI.ResultCount()) == 0) && return NaN
    return Float64(sol.rGap)
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
