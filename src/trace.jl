# Structured solver trace.
#
# The same events the log prints (header, iteration, note, exit, final; see
# logging.jl) are also available as data: pass `trace = f` and `f` is called
# with one NamedTuple per event. `SolverTrace()` collects them in memory, a
# path String (or ENV["CONICIP_TRACE"]) appends them to a JSON Lines file,
# one object per line. The trace is independent of `verbose` and of
# `MOI.Silent`: it is data, not printing.
#
# Schema "conicip-trace/1" (bump TRACE_SCHEMA when a field changes meaning).
# Every record has `event`, `depth` (0 for the top-level solve) and `t`
# (seconds since the entry point started). Units: residuals are the
# relative measures of the stopping test; times are seconds.
#
#   header  version, n, m, p, nnzQ, nnzA, nnzG, cones, nR, nQ, nS,
#           kktsolver, routed, nnz_per_col, staticReg, equilibrated,
#           optTol, infeasTol, maxIters, centralityCorrectors, timeLimit
#           (+ schema)
#   iter    iter, rPr, rDu, rCp, rEq, rGap, rPrRow, rEqRow, optMeasure,
#           pobj, dobj, mu, icertp, icertd, alpha, sigma, refine, rnorm,
#           kkt_repaired, kkt_refactors, cc_acc, cc_try, kkt_solves, best,
#           refine_warn
#   note    source, msg
#   exit    kind, status, detail
#   final   status, message, iters, time, pobj, dobj, prFeas, duFeas,
#           muFeas, rGap, has_certificate, kkt_solves, kkt_repaired,
#           kkt_refactors, changed_from, phases
#
# Row k of `iter` is iterate k; alpha/sigma/refine/rnorm/cc_* describe the
# step that produced it (NaN, 0 or -1 at the initial point). -1 in
# kkt_repaired/kkt_refactors means the KKT solver reports no diagnostics,
# in cc_acc/cc_try that correctors are off. An icert value is NaN when the
# sign test that makes a ray meaningful fails. In JSON, NaN and ±Inf are
# written as null.

const TRACE_SCHEMA = "conicip-trace/1"

"""
    SolverTrace()

In-memory collector for the solver trace: pass `trace = tr` to
[`conicIP`](@ref), [`preprocess_conicIP`](@ref) or the MOI option
`"trace"`, and `tr.events` receives one `NamedTuple` per event (`header`,
`iter`, `note`, `exit`, `final`; the schema is documented in
`src/trace.jl` and in AGENTS.md). [`iterations`](@ref) extracts the
iteration records and [`write_jsonl`](@ref) saves the whole trace.

```julia
tr = ConicIP.SolverTrace()
sol = conicIP(Q, c, A, b, cone_dims; verbose = false, trace = tr)
[r.rnorm for r in ConicIP.iterations(tr)]
```
"""
struct SolverTrace
  events :: Vector{Any}
end
SolverTrace() = SolverTrace(Any[])
(tr::SolverTrace)(rec) = (push!(tr.events, rec); nothing)

"""
    iterations(tr::SolverTrace) -> Vector

The `iter` records of a trace, in order.
"""
iterations(tr::SolverTrace) = [e for e in tr.events if e.event === :iter]

"""
    jsonl_trace(io::IO)

A trace sink that writes each event to `io` as one line of JSON. Pass it
as `trace = ConicIP.jsonl_trace(io)`; the caller owns `io`.
"""
jsonl_trace(io::IO) = rec -> (_write_json(io, rec); print(io, '\n'); nothing)

"""
    write_jsonl(io_or_path, tr::SolverTrace)

Write a collected trace as JSON Lines, one event per line; a path is
created or overwritten.
"""
function write_jsonl(io::IO, tr::SolverTrace)
  sink = jsonl_trace(io)
  foreach(sink, tr.events)
  return nothing
end
write_jsonl(path::AbstractString, tr::SolverTrace) = open(io -> write_jsonl(io, tr), path, "w")

# The trace an owner uses when the caller passed none: the file named by
# ENV["CONICIP_TRACE"], if set.
_env_trace() = (p = get(ENV, "CONICIP_TRACE", ""); isempty(p) ? nothing : p)

# (sink, file to close): a path opens a file in append mode, so that
# several solves in one process (or a presolve retry) share one file.
function _open_trace(trace)
  trace === nothing && return (nothing, nothing)
  if trace isa AbstractString
    io = open(trace, "a")
    return (jsonl_trace(io), io)
  end
  return (trace, nothing)
end

# ──────────────────────────────────────────────────────────────
#  JSON (dependency-free; only what the records contain)
# ──────────────────────────────────────────────────────────────

_write_json(io::IO, x::Nothing) = print(io, "null")
_write_json(io::IO, x::Bool) = print(io, x ? "true" : "false")
_write_json(io::IO, x::Integer) = print(io, x)
_write_json(io::IO, x::Real) = (y = Float64(x); isfinite(y) ? print(io, y) : print(io, "null"))
_write_json(io::IO, x::Symbol) = _write_json(io, string(x))
function _write_json(io::IO, s::AbstractString)
  print(io, '"')
  for ch in s
    if ch == '"'
      print(io, "\\\"")
    elseif ch == '\\'
      print(io, "\\\\")
    elseif ch == '\n'
      print(io, "\\n")
    elseif ch == '\r'
      print(io, "\\r")
    elseif ch == '\t'
      print(io, "\\t")
    elseif ch < ' '
      print(io, "\\u", string(UInt16(ch), base = 16, pad = 4))
    else
      print(io, ch)
    end
  end
  print(io, '"')
end
function _write_json(io::IO, x::Union{NamedTuple, AbstractDict})
  print(io, '{')
  first = true
  for (k, v) in pairs(x)
    first || print(io, ',')
    first = false
    _write_json(io, string(k)); print(io, ':'); _write_json(io, v)
  end
  print(io, '}')
end
function _write_json(io::IO, x::Union{AbstractVector, Tuple})
  print(io, '[')
  for (i, v) in enumerate(x)
    i > 1 && print(io, ',')
    _write_json(io, v)
  end
  print(io, ']')
end
_write_json(io::IO, x) = _write_json(io, string(x))
