# Opt-in per-phase timing.
#
# A caller that wants a breakdown constructs a `PhaseTimes` and passes it as
# `timing = pt` (direct API keyword, or the MOI option `"timing"`). Every
# instrumented site is written as
#
#     @phase timing t_scaling b_scaling begin ... end
#
# which expands to the untouched expression when `timing === nothing` and to
# the expression bracketed by `time_ns()` / allocation probes otherwise. The
# off path is therefore the original code plus one pointer comparison per site.
#
# Contract (frozen 2026-09-14; WP1/WP2/WP3 only read it):
#
# * All `t_*` fields are nanoseconds (`UInt64`), all `b_*` fields are bytes
#   allocated by Julia (`Int64`, process-wide; the probe itself writes a
#   shared Ref and allocates nothing), all `n_*` fields are counts (`Int`).
# * Whole-call phases are ADDITIVE: t_frontend + t_presolve + t_equilibrate +
#   t_setup + t_init + t_loop + t_final + t_fallback + t_postsolve ≈ wall time
#   of the outermost call (the remainder is `t_other`, computed by the harness,
#   and must not be materially negative).
# * Loop children are ADDITIVE within t_loop: t_scaling + t_residuals +
#   t_kktupdate + t_direction + t_rhs + t_linesearch ≈ t_loop.
# * Direction children are INCLUSIVE diagnostics within t_direction, never
#   summed with it: t_dir_base (the first solve of predictor/corrector),
#   t_dir_refine (whole `refine!` calls, including their correction solves),
#   t_dir_refine_resid (the `step_residual!` evaluations inside them).
# * LDL children are INCLUSIVE diagnostics reported by `kktsolver_ldl`
#   through `kkt_attach_timing!`: t_ldl_factor (numeric refactorization, a
#   child of t_kktupdate), t_ldl_solve (triangular solves) and t_ldl_resid
#   (residual products of the internal refinement), both children of the
#   solve that called them. Custom KKT callables leave them zero.
# * Counts: n_passes = loop passes including the final convergence check,
#   n_steps = completed iterate updates, n_kktupdate = factorizations
#   (initial one included), n_solve = calls to the 4×4 solve (initial point,
#   predictor, refinement, corrector, correctors), n_refine_attempt =
#   outer refinement corrections attempted (rolled-back ones included),
#   n_refine_resid = outer residual evaluations, n_ldl_factor / n_ldl_solve /
#   n_ldl_resid = the LDL backend's own counts. Retries and the deflation
#   recursion ACCUMULATE into the same object; the fallback solves are timed
#   as a whole (t_fallback) and pass `timing = nothing` inwards.
# * t_gc is `Base.gc_time_ns()` over the outermost call, recorded separately
#   and never added to the phases (a pause is charged to whichever phase it
#   interrupts).
# * One `PhaseTimes` per measured call: the harness constructs a fresh one
#   per repetition and keeps the whole record of the fastest repetition.

"""
    PhaseTimes()

Accumulator for the opt-in per-phase instrumentation of [`conicIP`](@ref):
pass one as `timing = pt` (or the MOI option `"timing"`) and the solver adds
wall time in nanoseconds (`t_*` fields), allocation bytes (`b_*`) and counts
(`n_*`) for each phase; see the header of `src/timing.jl` for the field
contract. Use one object per measured call, [`reset!`](@ref) to reuse it,
and [`phase_table`](@ref) to print it. `nothing` (the default) leaves the
solver uninstrumented.
"""
mutable struct PhaseTimes
  # whole-call phases (additive)
  t_frontend::UInt64
  t_presolve::UInt64
  t_equilibrate::UInt64
  t_setup::UInt64
  t_init::UInt64
  t_loop::UInt64
  t_final::UInt64
  t_fallback::UInt64
  t_postsolve::UInt64
  # loop children (additive within t_loop)
  t_scaling::UInt64
  t_residuals::UInt64
  t_kktupdate::UInt64
  t_direction::UInt64
  t_rhs::UInt64
  t_linesearch::UInt64
  # direction children (inclusive diagnostics)
  t_dir_base::UInt64
  t_dir_refine::UInt64
  t_dir_refine_resid::UInt64
  # LDL backend children (inclusive diagnostics)
  t_ldl_factor::UInt64
  t_ldl_solve::UInt64
  t_ldl_resid::UInt64
  # allocation bytes for the loop children and the whole-call phases that matter
  b_presolve::Int64
  b_equilibrate::Int64
  b_setup::Int64
  b_scaling::Int64
  b_residuals::Int64
  b_kktupdate::Int64
  b_direction::Int64
  b_rhs::Int64
  b_linesearch::Int64
  # counts
  n_passes::Int
  n_steps::Int
  n_kktupdate::Int
  n_solve::Int
  n_refine_attempt::Int
  n_refine_resid::Int
  n_ldl_factor::Int
  n_ldl_solve::Int
  n_ldl_resid::Int
  # GC time over the outermost call, kept apart from the phases
  t_gc::UInt64
end

PhaseTimes() = PhaseTimes(ntuple(_ -> 0, fieldcount(PhaseTimes))...)

"""
    reset!(pt::PhaseTimes)

Zero every field in place.
"""
function reset!(pt::PhaseTimes)
  for f in fieldnames(PhaseTimes)
    setfield!(pt, f, zero(fieldtype(PhaseTimes, f)))
  end
  return pt
end

const _GC_BYTES_REF = Ref{Int64}(0)

# Bytes allocated so far by this process. The Ref form is the non-deprecated
# one; the single shared Ref is written, never read across a task boundary
# (the solver is single-threaded).
@inline function _gc_bytes()
  Base.gc_bytes(_GC_BYTES_REF)
  return _GC_BYTES_REF[]
end

"""
    @phase timing t_field b_field expr
    @phase timing t_field expr

Evaluate `expr`, and when `timing !== nothing` add its wall time to
`timing.t_field` (and its allocated bytes to `timing.b_field` when given).
Timestamps are local to the expansion, so nested `@phase` blocks are safe.

Restrictions. `expr` is expanded twice (once per branch), so it must not
define a named local function (the timed branch would see the method
overwritten and throw `UndefVarError`); use an anonymous function or hoist
the definition. A `return`, `break` or `continue` inside `expr` keeps its
control flow but leaves the TIMED span unaccounted, because the update runs
only when `expr` completes normally; where a span can exit early, use
[`@phase_start`](@ref) / `@phase_stop` and close the span on each
exit path.
"""
macro phase(timing, tfield, bfield, expr)
  quote
    local _pt = $(esc(timing))
    if _pt === nothing
      $(esc(expr))
    else
      local _t0 = time_ns()
      local _b0 = _gc_bytes()
      local _val = $(esc(expr))
      setfield!(_pt, $(QuoteNode(tfield)), getfield(_pt, $(QuoteNode(tfield))) + (time_ns() - _t0))
      setfield!(_pt, $(QuoteNode(bfield)), getfield(_pt, $(QuoteNode(bfield))) + (_gc_bytes() - _b0))
      _val
    end
  end
end

macro phase(timing, tfield, expr)
  quote
    local _pt = $(esc(timing))
    if _pt === nothing
      $(esc(expr))
    else
      local _t0 = time_ns()
      local _val = $(esc(expr))
      setfield!(_pt, $(QuoteNode(tfield)), getfield(_pt, $(QuoteNode(tfield))) + (time_ns() - _t0))
      _val
    end
  end
end

"""
    (t0, b0) = @phase_start timing
    @phase_stop timing t_field b_field t0 b0
    @phase_stop timing t_field t0

Manual form of [`@phase`](@ref) for spans that can exit early: open once,
close on every exit path. The stamps are `(UInt64(0), Int64(0))` when
`timing === nothing`, and no clock is read.
"""
macro phase_start(timing)
  quote
    local _pt = $(esc(timing))
    _pt === nothing ? (UInt64(0), Int64(0)) : (time_ns(), _gc_bytes())
  end
end

macro phase_stop(timing, tfield, bfield, t0, b0)
  quote
    local _pt = $(esc(timing))
    if _pt !== nothing
      setfield!(_pt, $(QuoteNode(tfield)), getfield(_pt, $(QuoteNode(tfield))) + (time_ns() - $(esc(t0))))
      setfield!(_pt, $(QuoteNode(bfield)), getfield(_pt, $(QuoteNode(bfield))) + (_gc_bytes() - $(esc(b0))))
    end
    nothing
  end
end

macro phase_stop(timing, tfield, t0)
  quote
    local _pt = $(esc(timing))
    if _pt !== nothing
      setfield!(_pt, $(QuoteNode(tfield)), getfield(_pt, $(QuoteNode(tfield))) + (time_ns() - $(esc(t0))))
    end
    nothing
  end
end

"""
    gc_start(timing) -> UInt64
    gc_stop!(timing, gc0, x) -> x

`t_gc` bookkeeping for an entry point (`conicIP`, `preprocess_conicIP`,
`_preprocess_core`, the MOI `optimize!`): read `Base.gc_time_ns()` on entry
and ASSIGN the delta on every exit. Nested entry points each assign, and the
outermost assigns last, so the recorded value is the outermost call's.
"""
@inline gc_start(timing) = timing === nothing ? UInt64(0) : Base.gc_time_ns()
@inline function gc_stop!(timing, gc0, x)
  timing === nothing || (timing.t_gc = Base.gc_time_ns() - gc0)
  return x
end

"""
    kkt_attach_timing!(solve3x3, pt::PhaseTimes)

Ask a KKT solver object to report its internal timings and counts into `pt`
(the `t_ldl_*` / `n_ldl_*` fields). The default does nothing; `kktsolver_ldl`
implements it for both the generator returned by `kktsolver(Q, A, G, cone_dims)`
and the per-factorization object returned by `solve3x3gen(F, F⁻ᵀ)`. The main
loop calls it on the generator right after construction (so the initial
factorization is counted) and on every factorization, only when timing is on.
"""
kkt_attach_timing!(::Any, ::PhaseTimes) = nothing

"""
    phase_table(pt::PhaseTimes) -> Vector{Pair{Symbol,Float64}}

Seconds per whole-call phase and per loop child, in the contract's order,
for printing and CSV export. Diagnostics and counts are not included.
"""
function phase_table(pt::PhaseTimes)
  fs = (:t_frontend, :t_presolve, :t_equilibrate, :t_setup, :t_init, :t_loop,
        :t_final, :t_fallback, :t_postsolve,
        :t_scaling, :t_residuals, :t_kktupdate, :t_direction, :t_rhs, :t_linesearch)
  return [f => getfield(pt, f) / 1e9 for f in fs]
end
