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

mutable struct SolverLog
  on      :: Bool      # anything consumes the events (printing or a trace)
  show    :: Bool      # print to `io`
  style   :: Symbol    # resolved style: :classic
  io      :: IO
  depth   :: Int       # 0 for the top-level solve
  t0      :: Float64   # time() when the owner created the log
  # The last verdict `_conicIP` reported (_log_exit!), against which the
  # owner's final status is compared; :None until there is one.
  exit_kind   :: Symbol
  exit_status :: Symbol
end

SolverLog(on, show, style, io, depth) =
  SolverLog(on, show, style, io, depth, time(), :none, :None)

# The log of a solve that prints nothing. Never mutated.
const _NOLOG = SolverLog(false, false, :classic, devnull, 0)

function _make_log(verbose::Bool)
  verbose || return _NOLOG
  return SolverLog(true, true, :classic, stdout, 0)
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
  print(log.io, "\n > INTERIOR POINT PREPROCESSOR\n\n")
  return nothing
end

# A remark from one of the stages around the iteration. `source` is one of
# :presolve :equilibrate :kkt :stall :fallback :retry :deflate :postsolve.
# The classic table shows the presolve and stall notes only.
@noinline function _log_note!(log::SolverLog, source::Symbol, msg::AbstractString)
  log.show || return nothing
  io = log.io
  if source === :presolve
    print(io, "   - ", msg, "\n")
  elseif source === :stall
    print(io, "\n > Stalled: ", msg, "\n")
  end
  return nothing
end

# Problem summary, once per `_conicIP` that reaches the iteration (after the
# KKT solver is built). `h` is the header record (see the trace schema).
@noinline function _log_header!(log::SolverLog, h::NamedTuple)
  log.show || return nothing
  io = log.io
  print(io, "\n > INTERIOR POINT SOLVER (ConicIP v$(h.version))\n\n")
  if h.routed
    @printf(io, " > KKT solver: %s (auto, %.1f nnz/col)\n\n", h.kktsolver, h.nnz_per_col)
  end
  return nothing
end

# Column heads, after the initial point.
@noinline function _log_table_start!(log::SolverLog)
  log.show || return nothing
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
  log.show || return nothing
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
# reason shown with it.
@noinline function _log_exit!(log::SolverLog, kind::Symbol, detail::AbstractString = "")
  log.exit_kind = kind
  log.exit_status = _EXIT_STATUS[kind]
  log.show || return nothing
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
# announced (or when no solve ran, as for a presolve verdict).
@noinline function _log_final!(log::SolverLog, sol)
  log.show || return nothing
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
#  Records
# ──────────────────────────────────────────────────────────────

# The display name of a KKT solver: a function's name, or a callable
# object's type name.
_kkt_name(k::Function) = string(nameof(k))
_kkt_name(k) = string(nameof(typeof(k)))

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
