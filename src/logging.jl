# Solver log.
#
# Every line the solver prints goes through this file. The solver never
# formats or prints anything itself: at each event (header, iteration,
# note, exit) it hands a record to one of the `_log_*!` functions below,
# guarded as
#
#     log.on && _log_iter!(log, (; iter = Iter, ...))
#
# so that with the log off nothing is built and the cost is one field load
# per site. The renderers are @noinline and live outside `_conicIP`, which
# is already very large.
#
# Ownership. The outermost entry point (`conicIP`, `preprocess_conicIP`)
# creates the `SolverLog` and hands it down as the internal keyword `_log`;
# nested calls (presolve → conicIP → _conicIP, the deflation recursion)
# reuse it. Only the owner reports the final verdict, after postsolve and
# certificate revalidation, so the last thing printed is the status that
# is actually returned.
#
# Styles. `:classic` is the original fixed-width ASCII table; its text is
# pinned by test/golden. `:rich` is a Unicode table with colour, a
# per-row progress bar and an exit card with sparklines. Both are
# append-only (no cursor movement), so captured output reads the same as
# the terminal. `:auto` picks rich on an interactive terminal and classic
# anywhere else (pipes, files, Documenter, CI).
#
# One concrete type for every style: `_conicIP` specializes on the type of
# each keyword, and a type per renderer would compile it once per style.

# An iteration row is flagged when the scaled KKT residual of the step,
# ‖r - K Δz‖/(1 + ‖r‖), is still above this after the iterative-refinement
# loop. It flags an ill-conditioned KKT system.
const REFINE_WARN_NORM = 1e-3

# Per-iteration refinement statistics of the solver loop (see `_it` in
# `_conicIP`): corrections applied and the relative KKT residual of the
# step, ‖r − KΔz‖/(1 + ‖r‖).
mutable struct IterStats
  nref  :: Int
  rnorm :: Float64
end

# Per-solve history kept for the rich exit card (one `_conicIP`; reset by
# each header, so a presolve retry starts afresh).
mutable struct _LogHistory
  m      :: Vector{Float64}   # full stopping measure (incl. row-wise tests)
  rPr    :: Vector{Float64}   # max(rPr, rEq), the reported prFeas
  rDu    :: Vector{Float64}
  rCp    :: Vector{Float64}
  mu     :: Vector{Float64}
  alpha  :: Vector{Float64}
  icertp :: Vector{Float64}
  icertd :: Vector{Float64}
  refine :: Int
  warn   :: Vector{Int}       # iterations whose step failed REFINE_WARN_NORM
end
_LogHistory() = _LogHistory(Float64[], Float64[], Float64[], Float64[], Float64[],
                            Float64[], Float64[], Float64[], 0, Int[])

mutable struct SolverLog
  on      :: Bool      # anything consumes the events (printing or a trace)
  show    :: Bool      # print to `io`
  sink    :: Any       # trace callable (trace.jl), or nothing
  closeio :: Union{Nothing,IOStream}   # trace file the owner closes, or nothing
  style   :: Symbol    # resolved style: :classic or :rich
  io      :: IO
  color   :: Bool      # emit ANSI colour (rich style)
  width   :: Int       # columns available to the rich table
  depth   :: Int       # 0 for the top-level solve
  t0      :: Float64   # time() when the owner created the log
  # The last verdict `_conicIP` reported (_log_exit!), against which the
  # owner's final status is compared; :None until there is one.
  exit_kind   :: Symbol
  exit_status :: Symbol
  exit_detail :: String
  # Rich-style state of the current `_conicIP`. The renderers read three
  # values of its header record; they are kept as concrete fields (holding
  # the defaults shown before any header) rather than read out of an
  # untyped record.
  optTol    :: Float64 # header optTol
  infeasTol :: Float64 # header infeasTol
  ccorr     :: Int     # header centralityCorrectors
  nsolves :: Int       # headers seen (a presolve retry is the second)
  t_hdr   :: Float64   # time() at its header
  has_kkt :: Bool      # kkt column shown
  hist    :: _LogHistory
end

SolverLog(on, show, style, io, depth) =
  SolverLog(on, show, nothing, nothing, style, io, false, 120, depth, time(),
            :none, :None, "", 1e-6, 1e-7, 0, 0, 0.0, false, _LogHistory())

# The log of a solve that prints nothing. Never mutated.
const _NOLOG = SolverLog(false, false, :classic, devnull, 0)

_is_tty(io::IOContext) = _is_tty(io.io)
_is_tty(io::IO) = io isa Base.TTY

# The style a solve prints in: `log_style` if given, else the environment
# variable CONICIP_LOG_STYLE, else rich on an interactive terminal.
function _check_log_style(log_style)
  s = log_style isa Union{Symbol, AbstractString} ? Symbol(log_style) : :invalid
  s in (:auto, :classic, :rich) || throw(ArgumentError(
    "log_style must be :auto, :classic or :rich (got $(repr(log_style)))"))
  return s
end

function _resolve_style(log_style, io::IO)
  s = _check_log_style(log_style)
  if s === :auto
    e = Symbol(lowercase(get(ENV, "CONICIP_LOG_STYLE", "auto")))
    s = e in (:classic, :rich) ? e : :auto
  end
  s === :auto || return s
  return _is_tty(io) && get(ENV, "TERM", "") != "dumb" ? :rich : :classic
end

# The log of an owner. `trace` is a sink accepted by `_open_trace`; the
# caller closes the log with `_close_log!` when the solve is over.
function _make_log(verbose::Bool; log_style = :auto, log_io::IO = stdout, trace = nothing)
  # The style is checked even when nothing is printed, so that a typo
  # does not wait for the first verbose run to surface.
  _check_log_style(log_style)
  (verbose || trace !== nothing) || return _NOLOG
  log = SolverLog(true, verbose, verbose ? _resolve_style(log_style, log_io) : :classic,
                  log_io, 0)
  (log.sink, log.closeio) = _open_trace(trace)
  if verbose
    log.color = get(log_io, :color, false)::Bool && !haskey(ENV, "NO_COLOR")
    log.width = _is_tty(log_io) ? displaysize(log_io)[2] :
                get(log_io, :displaysize, (24, 120))[2]
  end
  return log
end

function _close_log!(log::SolverLog)
  io = log.closeio
  io === nothing || close(io)
  return nothing
end

# Hand an event to the trace sink, stamped with its kind, depth and time.
function _emit(log::SolverLog, event::Symbol, rec::NamedTuple)
  log.sink === nothing && return nothing
  log.sink(merge((; event, depth = log.depth, t = time() - log.t0), rec))
  return nothing
end

# The status each exit kind returns.
const _EXIT_STATUS = Dict{Symbol,Symbol}(
  :optimal => :Optimal, :infeasible => :Infeasible, :dual_infeasible => :DualInfeasible,
  :struct_infeasible => :Infeasible, :struct_dual_infeasible => :DualInfeasible,
  :error => :Error, :diverged => :Error, :time_limit => :TimeLimit,
  :abandoned => :Abandoned, :almost_infeasible => :AlmostInfeasible,
  :almost_dual_infeasible => :AlmostDualInfeasible)

# ──────────────────────────────────────────────────────────────
#  Events
# ──────────────────────────────────────────────────────────────

# Presolve banner (preprocess_conicIP).
@noinline function _log_presolve_start!(log::SolverLog)
  log.show || return nothing
  log.style === :rich && return nothing      # its notes carry the label
  print(log.io, "\n > INTERIOR POINT PREPROCESSOR\n\n")
  return nothing
end

# A remark from one of the stages around the iteration. `source` is one of
# :presolve :equilibrate :kkt :stall :fallback :retry :deflate :postsolve.
# The classic table shows the presolve and stall notes only.
@noinline function _log_note!(log::SolverLog, source::Symbol, msg::AbstractString)
  _emit(log, :note, (; source, msg = String(msg)))
  log.show || return nothing
  io = log.io
  if log.style === :rich
    _rich_note(log, source, msg)
  elseif source === :presolve
    print(io, "   - ", msg, "\n")
  elseif source === :stall
    print(io, "\n > Stalled: ", msg, "\n")
  end
  return nothing
end

# Problem summary, once per `_conicIP` that reaches the iteration (after the
# KKT solver is built). `h` is the header record (see the trace schema).
@noinline function _log_header!(log::SolverLog, h::NamedTuple)
  log.optTol = h.optTol; log.infeasTol = h.infeasTol; log.ccorr = h.centralityCorrectors
  log.nsolves += 1
  log.t_hdr = time()
  log.hist = _LogHistory()
  log.exit_kind = :none; log.exit_status = :None; log.exit_detail = ""
  _emit(log, :header, merge((; schema = TRACE_SCHEMA), h))
  log.show || return nothing
  log.style === :rich && return _rich_header(log, h)
  io = log.io
  print(io, "\n > INTERIOR POINT SOLVER (ConicIP v$(h.version))\n\n")
  if h.routed
    @printf(io, " > KKT solver: %s (auto, %.1f nnz/col)\n\n", h.kktsolver, h.nnz_per_col)
  end
  return nothing
end

# Column heads, after the initial point. `has_kkt`: the KKT solver
# reports regularization diagnostics (the kkt column).
@noinline function _log_table_start!(log::SolverLog, has_kkt::Bool)
  log.has_kkt = has_kkt
  log.show || return nothing
  log.style === :rich && return _rich_table_start(log)
  io = log.io
  println(io, "            Optimality                      Objective              Infeasibility       ")
  println(io)
  printstyled(io, @sprintf(" %-6s  │  %-8s  %-8s  %-8s │  %-8s  %-8s  │  %-8s  %-8s │  %-6s  %-8s  %-5s\n",
              "  Iter","prFeas","duFeas","muFeas","pobj","dobj","icertp","icertd","refine","kkt","cc");
              bold = true)
  return nothing
end

# One row per pass of the loop: iterate `r.iter`, and the step that
# produced it (see the trace schema for the fields).
@noinline function _log_iter!(log::SolverLog, r::NamedTuple)
  h = log.hist
  push!(h.m, max(r.optMeasure, r.rPrRow, r.rEqRow))
  push!(h.rPr, max(r.rPr, r.rEq)); push!(h.rDu, r.rDu); push!(h.rCp, r.rCp)
  push!(h.mu, r.mu); push!(h.alpha, r.alpha)
  push!(h.icertp, r.icertp); push!(h.icertd, r.icertd)
  h.refine += r.refine
  r.refine_warn && push!(h.warn, r.iter)
  _emit(log, :iter, r)
  log.show || return nothing
  log.style === :rich && return _rich_iter(log, r)
  # The kkt cell is repaired/refactors of the factorization that produced
  # this iterate; blank when the solver reports no diagnostics. The cc cell
  # is accepted/tried centrality correctors of the step that produced this
  # iterate; blank when the option is off.
  kkt = r.kkt_repaired < 0 ? "" : @sprintf("%d/%d", r.kkt_repaired, r.kkt_refactors)
  cc  = r.cc_try < 0 ? "" : @sprintf("%d/%d", r.cc_acc, r.cc_try)
  row = @sprintf(" %6i  │  %-8.1e  %-8.1e  %-8.1e │  % -8.1e  % -8.1e  │  %-8.1e  %-8.1e │  %-6i  %-8s  %s\n",
                 r.iter, r.rPr, r.rDu, r.rCp, r.pobj, r.dobj, r.icertp, r.icertd,
                 r.refine, kkt, cc)
  # A row is highlighted in red when the KKT step is still inaccurate after
  # iterative refinement (see REFINE_WARN_NORM).
  if r.refine_warn
    printstyled(log.io, row; bold = true, color = :red)
  else
    print(log.io, row)
  end
  return nothing
end

# How `_conicIP` left: `kind` is a key of _EXIT_STATUS, and `detail` the
# reason shown with it. The rich style shows it on the owner's exit card.
@noinline function _log_exit!(log::SolverLog, kind::Symbol, detail::AbstractString = "")
  log.exit_kind = kind
  log.exit_status = _EXIT_STATUS[kind]
  log.exit_detail = detail
  _emit(log, :exit, (; kind, status = log.exit_status, detail = String(detail)))
  log.show || return nothing
  log.style === :rich && return nothing
  text = kind === :optimal                ? "Below Tolerance!" :
         kind === :infeasible             ? "Certificate of Infeasibility Found!" :
         kind === :dual_infeasible        ? "Certificate of Dual Infeasibility Found!" :
         kind === :struct_infeasible      ? "Structurally infeasible ($detail)" :
         kind === :struct_dual_infeasible ? "Structurally dual infeasible ($detail)" :
         kind === :error                  ? "Error! ($detail)" :
         kind === :diverged               ? "Error!" :
         kind === :time_limit             ? "Time limit reached ($detail)" :
         kind === :abandoned              ? "No verdict ($detail)" :
         kind === :almost_infeasible      ? "Almost infeasible ($detail)" :
         kind === :almost_dual_infeasible ? "Almost dual infeasible ($detail)" :
                                            string(kind)
  print(log.io, "\n > EXIT -- ", text, "\n\n")
  return nothing
end

# The verdict the caller receives, reported once by the log's owner after
# postsolve and certificate revalidation. In the classic table it adds a
# line only when that verdict is not the one the solver's own exit line
# announced (or when no solve ran, as for a presolve verdict); the rich
# style prints the exit card. `timing` (a PhaseTimes or nothing) adds a
# time breakdown to the card.
@noinline function _log_final!(log::SolverLog, sol, timing = nothing)
  changed = log.exit_status !== :None && sol.status !== log.exit_status
  _emit(log, :final, (; status = sol.status, message = sol.message, iters = sol.Iter,
                        time = time() - log.t0, sol.pobj, sol.dobj, sol.prFeas,
                        sol.duFeas, sol.muFeas, sol.rGap, sol.has_certificate,
                        sol.kkt_solves, sol.kkt_repaired, sol.kkt_refactors,
                        changed_from = changed ? log.exit_status : nothing,
                        phases = timing === nothing ? nothing : (; phase_table(timing)...)))
  log.show || return nothing
  log.style === :rich && return _rich_final(log, sol, timing)
  msg = isempty(sol.message) ? "" : " ($(sol.message))"
  if log.exit_status === :None
    print(log.io, "\n > EXIT -- ", sol.status, msg, "\n\n")
  elseif sol.status !== log.exit_status
    print(log.io, "\n > POSTSOLVE -- status changed: ", log.exit_status, " → ",
          sol.status, msg, "\n\n")
  end
  return nothing
end

# ──────────────────────────────────────────────────────────────
#  Rich style
# ──────────────────────────────────────────────────────────────

# Write `s`, styled when the log has colour.
function _put(log::SolverLog, s::AbstractString; color::Symbol = :normal,
              bold::Bool = false, dim::Bool = false)
  if log.color && (color !== :normal || bold || dim)
    # `dim` is :light_black rather than the faint attribute, which many
    # terminals render the same as normal text.
    printstyled(log.io, s; color = dim ? :light_black : color, bold = bold)
  else
    print(log.io, s)
  end
  return nothing
end

_e1(x) = isfinite(x) ? @sprintf("%.1e", x) : string(x)          # 7 wide
_e3(x) = isfinite(x) ? @sprintf("% .3e", x) : string(x)          # 10 wide
_g(x)  = @sprintf("%g", x)

# 0.1 ms, 41 ms, 1.2 s, 2m03s
function _dur(t)
  t < 1e-3 ? @sprintf("%.2f ms", 1e3t) :
  t < 1    ? @sprintf("%.1f ms", 1e3t) :
  t < 60   ? @sprintf("%.2f s", t) :
             @sprintf("%dm%02ds", floor(Int, t / 60), floor(Int, t % 60))
end

const _SPARK = ('▁', '▂', '▃', '▄', '▅', '▆', '▇', '█')
const _EIGHTHS = ('▏', '▎', '▍', '▌', '▋', '▊', '▉', '█')

# _sparkline(xs; width = 40, scale = :log, floor = 0.0) -> String
#
# One glyph `▁…█` per value of `xs` (downsampled to `width` by taking the
# largest value of each bucket), scaled between the smallest and largest
# finite value on a log₁₀ (`scale = :log`) or linear (`:lin`, clamped to
# [0, 1]) axis. On the log axis, values below `floor` are drawn at the
# floor, which is then the bottom of the axis: residuals at rounding level
# read as `▁` rather than as noise. Nonfinite values are drawn as `·`.
function _sparkline(xs::AbstractVector{<:Real}; width::Int = 40, scale::Symbol = :log,
                    floor::Real = 0.0)
  isempty(xs) && return ""
  # (single-assignment locals: a closure would box a reassigned one)
  ys0 = Float64.(xs)
  ys = if length(ys0) > width
    edges = round.(Int, range(0, length(ys0), length = width + 1))
    Float64[maximum(view(ys0, edges[k]+1:edges[k+1])) for k in 1:width]
  else
    ys0
  end
  f(x) = scale === :log ? (isnan(x) ? NaN : (x = max(x, floor); x > 0 ? log10(x) : NaN)) :
                          clamp(x, 0.0, 1.0)
  vs = f.(ys)
  fin = filter(isfinite, vs)
  lo0, hi = scale === :log ? (isempty(fin) ? (0.0, 0.0) : extrema(fin)) : (0.0, 1.0)
  lo = floor > 0 && scale === :log ? Float64(log10(floor)) : lo0
  # A flat series is drawn mid-height, or at the bottom when it sits on
  # the floor.
  flat = floor > 0 && scale === :log && hi <= lo ? 1 : 4
  return String([isfinite(v) ? _SPARK[hi > lo ? 1 + round(Int, 7 * (v - lo) / (hi - lo)) : flat] : '·'
                 for v in vs])
end

# A bar of `cells` cells filled to `frac` ∈ [0, 1] in eighths.
function _bar(frac::Float64, cells::Int)
  u = round(Int, clamp(frac, 0.0, 1.0) * 8cells)
  return "█"^(u ÷ 8) * (u % 8 == 0 ? "" : string(_EIGHTHS[u % 8]))
end

const _NOTE_COLOR = Dict(:stall => :yellow, :retry => :yellow, :fallback => :magenta)

function _rich_note(log::SolverLog, source::Symbol, msg::AbstractString)
  _put(log, " › ", dim = true)
  _put(log, rpad(string(source), 12); color = get(_NOTE_COLOR, source, :normal),
       dim = !haskey(_NOTE_COLOR, source))
  _put(log, msg * "\n"; dim = !haskey(_NOTE_COLOR, source))
end

function _rich_header(log::SolverLog, h::NamedTuple)
  log.nsolves > 1 && _put(log, "\n ── solve $(log.nsolves) " * "─"^40 * "\n"; dim = true)
  print(log.io, "\n ")
  _put(log, "ConicIP $(h.version)"; bold = true)
  _put(log, " · primal–dual interior point · NT scaling · predictor–corrector\n"; dim = true)
  lab(s) = _put(log, " " * rpad(s, 10); dim = true)
  lab("problem")
  print(log.io, "$(h.n) variables · $(h.m) cone rows ($(h.cones)) · $(h.p) equalities",
        " · nnz Q $(h.nnzQ) · A $(h.nnzA) · G $(h.nnzG)\n")
  lab("kkt")
  print(log.io, h.kktsolver)
  h.routed && @printf(log.io, " (auto, %.1f nnz/col)", h.nnz_per_col)
  print(log.io, h.equilibrated ? " · Ruiz equilibrated" : " · not equilibrated")
  print(log.io, h.staticReg > 0 ? " · static reg $(_g(h.staticReg))" : " · no static reg", "\n")
  lab("stop")
  print(log.io, "optTol $(_g(h.optTol)) · infeasTol $(_g(h.infeasTol)) · maxIters $(h.maxIters)")
  h.centralityCorrectors > 0 && print(log.io, " · correctors $(h.centralityCorrectors)")
  isfinite(h.timeLimit) && print(log.io, " · time limit $(_g(h.timeLimit)) s")
  print(log.io, "\n\n")
  return nothing
end

# Column layout: (head, width, right-aligned). kkt and cc only when they
# carry information; time and progress only when the table fits.
function _rich_columns(log::SolverLog)
  cols = Tuple{String,Int}[("iter", 4), ("primal", 7), ("dual", 7), ("compl", 7),
                           ("gap", 7), ("pobj", 10), ("dobj", 10), ("step", 5), ("ref", 3)]
  log.has_kkt && push!(cols, ("kkt", 5))
  log.ccorr > 0 && push!(cols, ("cc", 3))
  w = sum(c[2] + 2 for c in cols) + 1
  if w + 9 + 14 <= log.width
    push!(cols, ("time", 7)); push!(cols, ("progress", 12))
  end
  return cols
end

function _rich_table_start(log::SolverLog)
  cols = _rich_columns(log)
  heads = join((c[1] == "progress" ? rpad(c[1], c[2]) : lpad(c[1], c[2]) for c in cols), "  ")
  rule  = join(("─"^c[2] for c in cols), "  ")
  _put(log, " " * heads * "\n"; bold = true)
  _put(log, " " * rule * "\n"; dim = true)
  return nothing
end

function _rich_iter(log::SolverLog, r::NamedTuple)
  tol = log.optTol
  cols = _rich_columns(log)
  names = first.(cols)
  res(x) = (str = _e1(x); !isfinite(x) ? (str, :red) : x < tol ? (str, :green) : (str, :normal))
  print(log.io, " ", lpad(r.iter, 4))
  for (x, w) in ((max(r.rPr, r.rEq), 7), (r.rDu, 7), (r.rCp, 7), (r.rGap, 7))
    (s, c) = res(x)
    print(log.io, "  "); _put(log, lpad(s, w); color = c)
  end
  print(log.io, "  ", lpad(_e3(r.pobj), 10), "  ", lpad(_e3(r.dobj), 10))
  # step: the length of the step that produced this iterate
  a = r.alpha
  print(log.io, "  ")
  if isnan(a)
    _put(log, lpad("·", 5); dim = true)
  else
    _put(log, lpad(a >= 0.01 ? @sprintf("%.2f", a) : @sprintf("%.0e", a), 5);
         color = a < 0.1 ? :yellow : :normal)
  end
  print(log.io, "  ")
  if r.refine_warn
    _put(log, lpad("$(r.refine)!", 3); color = :red, bold = true)
  else
    print(log.io, lpad(r.refine, 3))
  end
  if "kkt" in names
    print(log.io, "  ")
    s = r.kkt_repaired < 0 ? "" : "$(r.kkt_repaired)/$(r.kkt_refactors)"
    _put(log, lpad(s, 5); color = s == "0/0" || s == "" ? :normal : :yellow)
  end
  "cc" in names && print(log.io, "  ", lpad(r.cc_try < 0 ? "" : "$(r.cc_acc)/$(r.cc_try)", 3))
  if "time" in names
    print(log.io, "  ")
    _put(log, lpad(_dur(time() - log.t_hdr), 7); dim = true)
    print(log.io, "  ")
    _rich_progress(log, r, tol)
  end
  print(log.io, "\n")
  return nothing
end

# The progress cell: a bar of how far the full stopping measure has fallen
# from its first value toward optTol, on a log scale. It gives way to the
# infeasibility screen once a ray is close to being nominated, and flags
# the case where the aggregate residuals pass but a row-wise test does not.
function _rich_progress(log::SolverLog, r::NamedTuple, tol)
  m  = max(r.optMeasure, r.rPrRow, r.rEqRow)
  m1 = log.hist.m[1]
  # the closer of the two infeasibility screens (NaN when its sign test fails)
  scr, which = isnan(r.icertd) || r.icertp <= r.icertd ? (r.icertp, "p") : (r.icertd, "d")
  itol = log.infeasTol
  if m < tol
    _put(log, _bar(1.0, 10) * " ✓"; color = :green)
  elseif isfinite(scr) && scr < 1e-2 && m > 1e2 * tol
    _put(log, "$which $(_e1(scr))" * (scr < itol ? " ⇢" : ""); color = :magenta)
  elseif r.optMeasure < tol
    _put(log, "rows $(_e1(max(r.rPrRow, r.rEqRow)))"; color = :yellow)
  else
    frac = m1 > tol && isfinite(m) && m > 0 ? log10(m1 / m) / log10(m1 / tol) : 0.0
    _put(log, _bar(frac, 10); color = :cyan)
  end
end

const _STATUS_TITLE = Dict(
  :Optimal => "OPTIMAL", :Infeasible => "INFEASIBLE", :DualInfeasible => "DUAL INFEASIBLE",
  :AlmostInfeasible => "ALMOST INFEASIBLE", :AlmostDualInfeasible => "ALMOST DUAL INFEASIBLE",
  :Abandoned => "ABANDONED", :TimeLimit => "TIME LIMIT", :Error => "ERROR", :None => "NONE")

function _status_mark(st::Symbol)
  st === :Optimal && return ("✓", :green)
  st in (:Infeasible, :DualInfeasible) && return ("✗", :magenta)
  st === :Error && return ("✗", :red)
  return ("⚠", :yellow)
end

function _rich_final(log::SolverLog, sol, timing)
  io = log.io
  h = log.hist
  st = sol.status
  # (concrete whatever the declared types of the Solution fields)
  iter_best = Int(sol.Iter)
  pobj = Float64(sol.pobj); dobj = Float64(sol.dobj); rGap = Float64(sol.rGap)
  prFeas = Float64(sol.prFeas); duFeas = Float64(sol.duFeas); muFeas = Float64(sol.muFeas)
  (mark, col) = _status_mark(st)
  title = get(_STATUS_TITLE, st, string(st))
  changed = log.exit_status !== :None && st !== log.exit_status
  changed && (title = get(_STATUS_TITLE, log.exit_status, "") * " → " * title)
  iters = isempty(h.m) ? iter_best : length(h.m)
  # (after a header with no table the header's blank line separates)
  print(io, log.nsolves > 0 && isempty(h.m) ? " " : "\n ")
  _put(log, "$mark $title"; color = col, bold = true)
  _put(log, "  $iters iteration$(iters == 1 ? "" : "s") · $(_dur(time() - log.t0)) · " *
            "$(sol.kkt_solves) KKT solve$(sol.kkt_solves == 1 ? "" : "s")\n"; dim = true)
  lab(s) = _put(log, "   " * rpad(s, 12); dim = true)
  tol = log.optTol
  itol = log.infeasTol
  if changed
    lab("postsolve")
    print(io, "the solver reported $(log.exit_status); checks on the original data changed it\n")
  end
  if !isempty(sol.message)
    lab("message"); print(io, sol.message, "\n")
  elseif log.exit_kind in (:struct_infeasible, :struct_dual_infeasible) && !isempty(log.exit_detail)
    lab("reason"); print(io, "structural: ", log.exit_detail, "\n")
  end
  if st === :Infeasible
    lab("certificate")
    print(io, sol.has_certificate ? "Farkas ray (w, v), validated on the original data\n" :
                                    "no validated ray\n")
    isempty(h.icertp) || (lab("screen"); _spark_value(log, "icertp", h.icertp, itol / 100);
                          print(io, "  (infeasTol $(_g(itol)))\n"))
  elseif st === :DualInfeasible
    lab("certificate")
    print(io, sol.has_certificate ? "recession ray y, validated on the original data\n" :
                                    "no validated ray\n")
    isempty(h.icertd) || (lab("screen"); _spark_value(log, "icertd", h.icertd, itol / 100);
                          print(io, "  (infeasTol $(_g(itol)))\n"))
  else
    if isfinite(pobj) || isfinite(dobj)
      lab("objective")
      @printf(io, "pobj % .6e   dobj % .6e   rel. gap %s\n", pobj, dobj, _e1(rGap))
    end
    if st !== :Optimal && iters > 0
      lab("best")
      print(io, "iteration $iter_best: ")
      for (name, x) in (("primal", prFeas), ("dual", duFeas), ("compl", muFeas),
                        ("gap", rGap))
        isfinite(x) || continue
        print(io, name, " ")
        _put(log, _e1(x); color = x < tol ? :green : :red)
        print(io, "   ")
      end
      print(io, "(optTol $(_g(tol)))\n")
    end
  end
  if !isempty(h.m)
    # residuals mean nothing beside a certified ray
    if !(st in (:Infeasible, :DualInfeasible) && sol.has_certificate)
      lab("residuals")
      _spark_value(log, "primal", h.rPr, tol / 100); print(io, "   ")
      _spark_value(log, "dual", h.rDu, tol / 100); print(io, "   ")
      _spark_value(log, "compl", h.rCp, tol / 100); print(io, "\n")
    end
    lab("path")
    _spark_value(log, "μ", h.mu); print(io, "   ")
    steps = filter(!isnan, h.alpha)
    if !isempty(steps)
      print(io, "step ")
      _put(log, _sparkline(steps; scale = :lin); color = :cyan)
      print(io, " min ", @sprintf("%.2g", minimum(steps)), "   ")
    end
    print(io, "refine $(h.refine)")
    log.has_kkt && print(io, " · repaired $(sol.kkt_repaired) · refactors $(sol.kkt_refactors)")
    print(io, "\n")
  end
  if !isempty(h.warn)
    lab("warnings")
    _put(log, "$(length(h.warn)) step(s) with KKT residual > $(_g(REFINE_WARN_NORM)) after " *
              "refinement (iteration $(join(first(h.warn, 5), ", "))" *
              (length(h.warn) > 5 ? ", …" : "") * "); try staticReg or another kktsolver\n";
         color = :yellow)
  end
  if timing !== nothing
    lab("time")
    print(io, _phase_shares(timing), "\n")
  end
  print(io, "\n")
  return nothing
end

function _spark_value(log::SolverLog, name, xs, floor = 0.0)
  print(log.io, name, " ")
  _put(log, _sparkline(xs; floor = floor); color = :cyan)
  print(log.io, " ", _e1(xs[end]))
end

# "kkt 41% · direction 22% · …": the largest phases of a PhaseTimes, as
# shares of the whole call.
function _phase_shares(pt)
  fs = (:t_presolve => "presolve", :t_equilibrate => "equilibrate", :t_setup => "setup",
        :t_init => "init", :t_scaling => "scaling", :t_residuals => "residuals",
        :t_kktupdate => "kkt", :t_direction => "direction", :t_rhs => "rhs",
        :t_linesearch => "linesearch", :t_final => "final", :t_fallback => "fallback",
        :t_postsolve => "postsolve")
  tot = sum(Float64(getfield(pt, f)) for f in (:t_frontend, :t_presolve, :t_equilibrate,
            :t_setup, :t_init, :t_loop, :t_final, :t_fallback, :t_postsolve))
  tot > 0 || return "not recorded"
  sh = sort([(getfield(pt, f) / tot, s) for (f, s) in fs]; rev = true)
  return join(("$s $(round(Int, 100x))%" for (x, s) in sh if x >= 0.01), " · ")
end

# ──────────────────────────────────────────────────────────────
#  Records
# ──────────────────────────────────────────────────────────────

# The equilibration note: the objective scale and the ranges of the
# column, row and equality scalings (see equilibrate.jl).
function _equilibrate_msg(eq)
  rng(D) = isempty(D) ? "—" : @sprintf("%.1e…%.1e", extrema(D)...)
  return @sprintf("Ruiz: objective scale %.2g · columns %s · rows %s · equalities %s",
                  eq.σ, rng(eq.Dc), rng(eq.Dr), rng(eq.De))
end

# The display name of a KKT solver: a function's name, or a callable
# object's type name.
_kkt_name(k::Function) = string(nameof(k)::Symbol)
_kkt_name(k) = string(nameof(typeof(k))::Symbol)

# "ℝ₊ 120 · SOC 4×5, 2×11 · PSD 1×(10×10)": the orthant as one total, the
# second-order and semidefinite cones as count×size per distinct size, in
# order of first appearance.
function _cone_summary(cone_dims)
  nR = 0
  soc = Pair{Int,Int}[]; psd = Pair{Int,Int}[]
  bump!(v, k) = (i = findfirst(x -> x.first == k, v);
                 i === nothing ? push!(v, k => 1) : (v[i] = k => v[i].second + 1))
  for (t, k) in cone_dims
    t == "R" ? (nR += k) :
    t == "Q" ? bump!(soc, k) :
    t == "S" ? bump!(psd, round(Int, (sqrt(8k + 1) - 1) / 2)) : nothing
  end
  parts = String[]
  nR > 0 && push!(parts, "ℝ₊ $nR")
  isempty(soc) || push!(parts, "SOC " * join(("$(c)×$(k)" for (k, c) in soc), ", "))
  isempty(psd) || push!(parts, "PSD " * join(("$(c)×($(k)×$(k))" for (k, c) in psd), ", "))
  return isempty(parts) ? "none" : join(parts, " · ")
end

# The header record of one `_conicIP` (trace schema, event "header").
@noinline function _header_record(Q, A, G, cone_dims, n, m, p, kktsolver, routed,
                                  staticReg, scaling, optTol, infeasTol, maxIters,
                                  centralityCorrectors, timeLimit)
  nnzQ = _structural_nnz(Q); nnzA = _structural_nnz(A); nnzG = _structural_nnz(G)
  count_type(t) = count(x -> x[1] == t, cone_dims)
  return (; version = string(pkgversion(@__MODULE__)), n, m, p, nnzQ, nnzA, nnzG,
            cones = _cone_summary(cone_dims),
            nR = count_type("R"), nQ = count_type("Q"), nS = count_type("S"),
            kktsolver = _kkt_name(routed === nothing ? kktsolver : routed[1]),
            routed = routed !== nothing,
            nnz_per_col = (nnzQ + nnzA + nnzG) / max(n, 1),
            staticReg = Float64(staticReg), equilibrated = scaling !== nothing,
            optTol = Float64(optTol), infeasTol = Float64(infeasTol),
            maxIters = Int(maxIters), centralityCorrectors = Int(centralityCorrectors),
            timeLimit = Float64(timeLimit))
end
