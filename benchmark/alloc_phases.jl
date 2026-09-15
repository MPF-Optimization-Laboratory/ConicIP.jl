# Allocation attribution per source site and per phase (WP5, private)
# ===================================================================
# Complements the per-phase timing study: `PhaseTimes` says how many bytes
# each phase allocates, this script says WHICH source lines allocate them.
# For every instance it
#
#   1. builds the direct-API tuple with suite.jl's loader (file instances are
#      assembled through the MOI wrapper with `assemble_only`, then read back),
#   2. warms the exact timed path, records the `b_*` counters of a fresh
#      `PhaseTimes` run and a warmed `@allocated` reference of the same call,
#   3. records EVERY allocation of one more identical call with
#      `Profile.Allocs` (sample_rate = 1), assigns each allocation once to the
#      innermost frame under this package's `src/`, or to an "other" bucket
#      when no such frame exists, and guesses a phase from the nearest
#      enclosing function name (table PHASE_GUESS below),
#   4. prints the top sites, the Σ-recorded vs `@allocated` cross-check, the
#      per-phase `b_*` next to the phase-guess aggregation, and per-pass
#      normalizations.
#
#   julia --project benchmark/alloc_phases.jl                    # nb, chainsing-1000-1, lp-band-20000
#   julia --project benchmark/alloc_phases.jl --quick            # suite.jl's --quick subset
#   julia --project benchmark/alloc_phases.jl --only lp-band-2000,sambal
#   julia --project benchmark/alloc_phases.jl --top 15 --out alloc.md
#   julia --project benchmark/alloc_phases.jl --in-process       # no child process per instance
#   julia --project benchmark/alloc_phases.jl --raw              # conicIP instead of preprocess_conicIP
#   julia --project benchmark/alloc_phases.jl --sample-rate 0.1  # when a full record is too large
#
# Each instance runs in a fresh child process by default (this file with
# `--one name`): a full allocation record holds one decoded stack trace per
# allocation, which for a large instance is more memory than the solve
# itself, and profiler state must not leak between instances. Timings are
# not reported; allocation counts do not depend on machine load.
#
# Known discrepancies between Σ recorded bytes and `@allocated`: the
# profiler records Julia-heap allocations only (BLAS/UMFPACK/QDLDL scratch
# obtained through malloc outside the GC is invisible to both, but the
# profiler also misses `Base.gc_bytes` bookkeeping of realloc growth and of
# allocations made while the profiler itself allocates its record); the
# `@allocated` reference is a separate call and can differ by GC-driven
# variation in the number of refinement steps. Both numbers are printed.
#
# Not part of CI. Never leaves `dev`.

using Profile      # stdlib, reachable through @stdlib in the default LOAD_PATH
using ConicIP, SparseArrays, LinearAlgebra, Printf, Dates
using MathOptInterface
const MOI = MathOptInterface

include(joinpath(@__DIR__, "suite.jl"))   # INSTANCES, direct_kwargs, arg_value, select_instances

const SRC_DIR = normpath(dirname(pathof(ConicIP)))
const DEFAULT_SET = ["nb", "chainsing-1000-1", "lp-band-20000"]

# ──────────────────────────────────────────────────────────────
#  Phase guess: nearest enclosing function name → phase
# ──────────────────────────────────────────────────────────────
# Walking outward from the allocating frame, the first frame whose function
# name matches an entry decides the phase. Order matters only when two
# patterns could match the same name. Anything under src/ that matches no
# entry is "core?" (the closures of the main loop that allocate in an
# anonymous `do` block reach `_conicIP` before any named function).
const PHASE_GUESS = [
    # scaling: Nesterov–Todd scaling assembly and its per-cone pieces
    (r"^(nt_scaling|nestod_soc|nestod_sdc|inv_adjoint!|QFunit|_absmat)$", "scaling"),
    # refine: outer iterative refinement and the LDL backend's own residuals
    (r"^(refine!|step_residual!|residual!|residual_core!|_symmul!|bump!)$", "refine"),
    # direction: the 4×4 / 3×3 solves and their Block products
    (r"^(solve4x4|solve4x4gen|solve3x3|solve_loaded!|ldl_solve!|solve2x2|solve3x3gen_lift|solve3x3gen_nolift)$", "direction"),
    (r"^(adjoint|broadcastf|block_idx|inv|copy|deepcopy|\*|\\|\+)$", "direction"),   # Block algebra (F', F'*x, F\x)
    # kktupdate: numeric refactorization and scaling-block assembly
    (r"^(numeric_factor!|numeric_factor_core!|write_static!|solve3x3gen|solve2x2gen|factor!|lift|soc_uv|_dense_FtF|_nzindex)$", "kktupdate"),
    # linesearch: max step to the boundary, centering, correctors
    (r"^(maxstep|maxstep_rp|maxstep_soc|maxstep_sdc|interior|centrality_correction!|spectral_map!|_spectral_map_soc!|_spectral_map_sdc!|clip_spectral!)$", "linesearch"),
    # rhs / cone algebra used by the corrector right-hand side
    (r"^(cone_div!|cone_prod!|dsoc!|xsoc!|drp!|xrp!|dsdc!|xsdc!|÷|∘|mat|mat!|vecm|vecm!|ord)$", "rhs"),
    # residuals: primal/dual residual products and their norms
    (r"^(_absprod_norm!|_rowwise_max|kkt_error|normsafe|norminfsafe|adjmulsafe|cone_margin)$", "residuals"),
    # setup: KKT solver construction, pattern, ordering, routing
    (r"^(kktsolver_ldl|_ldl_pattern|_ldl_flops|_ldl_structure_key|cached_kktsolver_ldl|kktsolver_qr|kktsolver_sparse|kktsolver_2x2|pivotgen|_choose_kktsolver|default_kktsolver|choose_kktsolver|placeholder|count_lift|count_dense|identical_sparse_structure|structurally_zero_rows|structurally_zero_cols|dense_kkt_bytes|dense_kkt_flops|_structural_nnz|_stamp_kkt!)$", "setup"),
    # equilibrate / postsolve (unequilibrate and certificate revalidation)
    (r"^(equilibrate_conicIP|_col_infnorms|_row_infnorms)$", "equilibrate"),
    (r"^(unequilibrate!|_refresh_point!|_check_postsolve!|_revalidate_certificate!|retract!|validate_infeasibility_certificate|validate_unboundedness_certificate|claim_infeasible!|claim_dual_infeasible!)$", "postsolve"),
    # presolve
    (r"^(preprocess_conicIP|_preprocess_core|imcols|_singleton_fixings)$", "presolve"),
    # fallback rays
    (r"^(fallback_infeasibility_ray|fallback_unbounded_ray)$", "fallback"),
    # the core itself: allocation in an anonymous loop closure, or setup code
    (r"^(_conicIP|conicIP|guarded|exit_loop|nonfinite!)$", "core?"),
]

# `#f#N` is the keyword body of `f`
_plain(name) = replace(name, r"^#(.+)#\d+$" => s"\1")

function name_guess(frames)
    for fr in frames
        name = _plain(String(fr.func))
        for (pat, ph) in PHASE_GUESS
            occursin(pat, name) && return ph
        end
    end
    return "core?"
end

# ──────────────────────────────────────────────────────────────
#  Phase segments: the `@phase` sites themselves
# ──────────────────────────────────────────────────────────────
# The `@phase` macros are the ground truth, so a frame of a phase-owning
# function (`_conicIP`, `conicIP`, `preprocess_conicIP`, `_preprocess_core`)
# at line L is attributed to the `@phase` block containing L. Blocks are
# read off the source by indentation: a `@phase ... begin|do|try` line
# extends to the first later line at the same indentation that starts with
# `end`; a one-expression `@phase` extends over its continuation lines
# (deeper indentation); a manual `@phase_start` … `@phase_stop` span runs
# from the line binding its stamp to the last `@phase_stop` naming it. The
# innermost (latest-starting) block containing L wins. Lines of `_conicIP`
# outside every block fall back to the region they lie in: before `t_init0`
# → setup, up to the `for Iter` line → init, inside the loop → loop-other,
# after it → final-other; for the other owners the fallback is
# `<owner>-other`. The tables are rebuilt from the files at load time, so
# they follow the source as the instrumentation moves.
_phase_name(t) = t == "t_dir_base" || t == "t_dir_refine" || t == "t_dir_refine_resid" ? "direction" :
                 replace(t, r"^t_" => "")
_indent(s) = length(s) - length(lstrip(s))

function phase_segments(path)
    lines = readlines(path)
    segs = Tuple{Int, Int, String}[]
    core_start = something(findfirst(l -> startswith(l, "function _conicIP("), lines), length(lines))
    loop_start = something(findfirst(l -> occursin(r"^\s+for Iter = 1:maxIters", l), lines), length(lines))
    loop_end = something(findfirst(l -> occursin(r"^\s+exit_loop\(nothing\)", l), lines), length(lines))
    init_start = something(findfirst(l -> occursin(r"^\s+t_init0 = ", l), lines), loop_start)
    # manual spans: `(t0, b0) = @phase_start timing` … `@phase_stop timing t_X [b_X] t0 [b0]`;
    # the span runs from the line binding t0 to the LAST stop that names it
    spans = Dict{Tuple{Int, String}, Int}()
    for (i, l) in enumerate(lines)
        m = match(r"@phase_stop timing (t_\w+)(?: b_\w+)? (\w+)", l)
        m === nothing && continue
        t0var = m.captures[2]
        j = findprev(l2 -> occursin(Regex("^\\s*\\(?\\b" * t0var * "\\b[^=]*=\\s*(@phase_start|timing === nothing)"), l2),
                     lines, i)
        j === nothing && continue
        key = (j, _phase_name(m.captures[1]))
        spans[key] = max(get(spans, key, 0), i)
    end
    for ((j, ph), i) in spans
        push!(segs, (j, i, ph))
    end
    for (i, l) in enumerate(lines)
        m = match(r"@phase timing (t_\w+)", l)
        m === nothing && continue
        ph = _phase_name(m.captures[1])
        ind = _indent(l)
        j = i
        if occursin(r"(\bbegin|\bdo|\btry)\s*(#.*)?$", l)
            j = something(findnext(l2 -> _indent(l2) == ind && startswith(lstrip(l2), "end"),
                                   lines, i + 1), i)
        else
            while j + 1 <= length(lines) && !isempty(strip(lines[j + 1])) && _indent(lines[j + 1]) > ind
                j += 1
            end
        end
        push!(segs, (i, j, ph))
    end
    sort!(segs)
    return (segs = segs, core_start = core_start, init_start = init_start,
            loop_start = loop_start, loop_end = loop_end)
end

const SEGMENTS = Dict(f => phase_segments(joinpath(SRC_DIR, f)) for f in ("ConicIP.jl", "preprocessor.jl"))

# Phase-owning functions: (function-name pattern, file, fallback outside every block).
# `#f#N` is the keyword body of `f`; the plain `f` frame is the forwarding stub.
const OWNERS = [
    (r"^#?_conicIP(#\d+)?$", "ConicIP.jl", :core),
    (r"^#?conicIP(#\d+)?$", "ConicIP.jl", "frontend-other"),
    (r"^#?(preprocess_conicIP|_preprocess_core)(#\d+)?$", "preprocessor.jl", "presolve-other"),
]

function segment_phase(file::String, L::Int, fallback)
    S = SEGMENTS[file]
    best = nothing            # innermost (latest-starting) segment containing L
    for (lo, hi, ph) in S.segs
        lo <= L <= hi && (best === nothing || lo >= best[1]) && (best = (lo, hi, ph))
    end
    best !== nothing && return best[3]
    fallback === :core || return fallback
    L < S.core_start && return "frontend-other"
    L < S.init_start && return "setup"
    L < S.loop_start && return "init"
    L <= S.loop_end && return "loop-other"
    return "final-other"
end

const REFINE_NAMES = r"^(refine!|step_residual!)$"

function phase_guess(frames)
    # 1. the innermost phase-owning frame decides by @phase block
    for fr in frames
        name = String(fr.func); file = basename(String(fr.file))
        for (pat, ofile, fallback) in OWNERS
            (file == ofile && occursin(pat, name)) || continue
            ph = segment_phase(file, fr.line, fallback)
            if ph == "direction" && any(f -> occursin(REFINE_NAMES, String(f.func)), frames)
                ph = "direction/refine"
            end
            return ph
        end
    end
    # 2. otherwise (equilibrate, postsolve helpers reached from MOI, …) by function name
    return name_guess(frames)
end

# ──────────────────────────────────────────────────────────────
#  Problem tuples
# ──────────────────────────────────────────────────────────────

function load_tuple(inst::Instance)
    inst.kind == :direct && return inst.load()
    path = inst.load()
    src = MOI.FileFormats.Model(format = inst.format, filename = path)
    MOI.read_from_file(src, path)
    opt = MOI.instantiate(ConicIP.Optimizer; with_bridge_type = Float64)
    MOI.set(opt, MOI.Silent(), true)
    MOI.set(opt, MOI.RawOptimizerAttribute("assemble_only"), true)
    MOI.copy_to(opt, src)
    MOI.optimize!(opt)
    raw = MOI.get(opt, MOI.RawSolver())::ConicIP.Optimizer
    n = raw.n
    return (Q = raw.Q_int === nothing ? spzeros(n, n) : raw.Q_int,
            c = copy(raw.c_int),
            A = raw.ineq_A === nothing ? spzeros(0, n) : raw.ineq_A,
            b = raw.ineq_A === nothing ? zeros(0) : copy(raw.ineq_b),
            cone_dims = copy(raw.cone_dims),
            G = raw.eq_G === nothing ? spzeros(0, n) : raw.eq_G,
            d = raw.eq_G === nothing ? zeros(0) : copy(raw.eq_d))
end

# The measured call: suite.jl's route (preprocess_conicIP, default solver)
# unless --raw, with a fresh PhaseTimes each time.
function timed_solve(prob, pt; raw = false)
    entry = raw ? conicIP : preprocess_conicIP
    return entry(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims, prob.G, prob.d;
                 verbose = false, timing = pt, direct_kwargs()...)
end

# ──────────────────────────────────────────────────────────────
#  Attribution
# ──────────────────────────────────────────────────────────────

mutable struct Site
    file  :: String
    line  :: Int
    func  :: String
    phase :: String
    bytes :: Int
    count :: Int
end

# Innermost Julia frame whose file lives under src/ (inlined frames are
# kept; C frames are skipped). Returns the index or 0.
function innermost_src(st)
    for (i, fr) in enumerate(st)
        fr.from_c && continue
        f = String(fr.file)
        isempty(f) && continue
        startswith(normpath(f), SRC_DIR) && return i
    end
    return 0
end

function attribute(allocs)
    # keyed by site AND phase: one source line (e.g. a Block product) serves
    # several phases, and each allocation is assigned once to its own phase
    sites = Dict{Tuple{String, Int, String, String}, Site}()
    other = Dict{String, Site}()          # innermost non-C frame, for the sub-table
    other_total = 0; other_count = 0
    total = 0
    for a in allocs
        total += a.size
        st = a.stacktrace
        i = innermost_src(st)
        if i == 0
            other_total += a.size; other_count += 1
            j = findfirst(fr -> !fr.from_c, st)
            key = j === nothing ? "(no Julia frame)" :
                  string(basename(String(st[j].file)), ":", st[j].line, " ", st[j].func)
            s = get!(other, key) do
                Site(key, 0, j === nothing ? "" : String(st[j].func), "other", 0, 0)
            end
            s.bytes += a.size; s.count += 1
        else
            fr = st[i]
            ph = phase_guess(@view st[i:end])
            key = (relpath(normpath(String(fr.file)), SRC_DIR), fr.line, String(fr.func), ph)
            s = get!(sites, key) do
                Site(key[1], key[2], key[3], ph, 0, 0)
            end
            s.bytes += a.size; s.count += 1
        end
    end
    return (sites = collect(values(sites)), other = collect(values(other)),
            other_total = other_total, other_count = other_count, total = total)
end

# ──────────────────────────────────────────────────────────────
#  Reporting
# ──────────────────────────────────────────────────────────────

const LOOP_B = [:b_scaling, :b_residuals, :b_kktupdate, :b_direction, :b_rhs, :b_linesearch]
const CALL_B = [:b_presolve, :b_equilibrate, :b_setup]

# b_* field ↔ phase-guess label
const B_LABEL = Dict(:b_presolve => "presolve", :b_equilibrate => "equilibrate",
                     :b_setup => "setup", :b_scaling => "scaling", :b_residuals => "residuals",
                     :b_kktupdate => "kktupdate", :b_direction => "direction",
                     :b_rhs => "rhs", :b_linesearch => "linesearch")

human(b) = b >= 2^30 ? @sprintf("%.2f GiB", b / 2^30) :
           b >= 2^20 ? @sprintf("%.1f MiB", b / 2^20) :
           b >= 2^10 ? @sprintf("%.1f KiB", b / 2^10) : string(round(Int, b), " B")
pct(x, t) = t == 0 ? "–" : @sprintf("%.1f%%", 100x / t)

function report_instance(io, inst::Instance, prob, top::Int; raw = false, sample_rate = 1.0)
    n = length(prob.c); m = size(prob.A, 1); p = size(prob.G, 1)
    println(io, "## ", inst.name, "  (", inst.family, "; n=", n, ", m=", m, ", p=", p,
            ", cones=", length(prob.cone_dims), ", route=", raw ? "conicIP" : "preprocess_conicIP", ")\n")

    # 1. warm-up on the exact timed path, then the PhaseTimes record
    sol0 = timed_solve(prob, ConicIP.PhaseTimes(); raw = raw)
    pt = ConicIP.PhaseTimes()
    sol = timed_solve(prob, pt; raw = raw)
    GC.gc()
    ref_bytes = @allocated timed_solve(prob, ConicIP.PhaseTimes(); raw = raw)
    npass = max(pt.n_passes, 1)
    println(io, "status=`", sol.status, "` (warm-up `", sol0.status, "`)  iters=", sol.Iter,
            "  n_passes=", pt.n_passes, "  n_steps=", pt.n_steps, "  n_kktupdate=", pt.n_kktupdate,
            "  n_solve=", pt.n_solve, "  n_refine_attempt=", pt.n_refine_attempt,
            "  kktsolver=", kkt_solver_name(prob), "\n")

    # 2. profile one more identical call (profiling kept apart from timing)
    Profile.Allocs.clear()
    GC.gc()
    if sample_rate == 1.0
        Profile.Allocs.@profile sample_rate=1 timed_solve(prob, ConicIP.PhaseTimes(); raw = raw)
    else
        Profile.Allocs.@profile sample_rate=sample_rate timed_solve(prob, ConicIP.PhaseTimes(); raw = raw)
    end
    res = Profile.Allocs.fetch()
    nalloc = length(res.allocs)
    att = attribute(res.allocs)
    Profile.Allocs.clear()
    scale = 1 / sample_rate            # bytes are scaled back up when sampling
    total = att.total * scale

    # 3. cross-check Σ recorded vs @allocated
    println(io, "### Cross-check\n")
    println(io, "| quantity | bytes | |\n|---|---:|---|")
    @printf(io, "| Σ recorded (profiler, %d allocs%s) | %d | %s |\n", nalloc,
            sample_rate == 1.0 ? "" : @sprintf(", sample_rate=%g, scaled", sample_rate), round(Int, total), human(total))
    @printf(io, "| `@allocated` reference (warmed, separate call) | %d | %s |\n", ref_bytes, human(ref_bytes))
    @printf(io, "| ratio recorded / reference | %.3f | |\n", total / max(ref_bytes, 1))
    b_sum = sum(getfield(pt, f) for f in vcat(CALL_B, LOOP_B))
    @printf(io, "| Σ b_* of the PhaseTimes run (presolve+equilibrate+setup+loop children) | %d | %s (%s of reference) |\n",
            b_sum, human(b_sum), pct(b_sum, ref_bytes))
    @printf(io, "| other (no src/ frame: Base/stdlib/deps) | %d | %s of recorded, %d allocs |\n\n",
            round(Int, att.other_total * scale), pct(att.other_total, att.total), att.other_count)

    # 4. top sites
    sort!(att.sites; by = s -> -s.bytes)
    println(io, "### Top ", top, " sites (of ", length(att.sites), " under src/)\n")
    println(io, "| # | bytes | % | count | bytes/pass | site | function | phase guess |")
    println(io, "|--:|--:|--:|--:|--:|---|---|---|")
    for (k, s) in enumerate(first(att.sites, top))
        @printf(io, "| %d | %s | %s | %d | %s | `%s:%d` | `%s` | %s |\n", k, human(s.bytes * scale),
                pct(s.bytes, att.total), round(Int, s.count * scale), human(s.bytes * scale / npass),
                s.file, s.line, s.func, s.phase)
    end
    @printf(io, "| | %s | %s | %d | %s | other (Base/stdlib/deps, no src/ frame) | | other |\n\n",
            human(att.other_total * scale), pct(att.other_total, att.total),
            round(Int, att.other_count * scale), human(att.other_total * scale / npass))
    if !isempty(att.other)
        sort!(att.other; by = s -> -s.bytes)
        println(io, "<details><summary>largest \"other\" frames</summary>\n")
        println(io, "| bytes | count | innermost Julia frame |\n|--:|--:|---|")
        for s in first(att.other, 8)
            @printf(io, "| %s | %d | `%s` |\n", human(s.bytes * scale), round(Int, s.count * scale), s.file)
        end
        println(io, "\n</details>\n")
    end

    # 5. per-phase: PhaseTimes b_* vs phase-guess aggregation
    byphase = Dict{String, Int}()
    for s in att.sites
        ph = s.phase == "direction/refine" ? "direction" : s.phase   # refine is inclusive in b_direction
        byphase[ph] = get(byphase, ph, 0) + s.bytes
        s.phase == "direction/refine" && (byphase["refine"] = get(byphase, "refine", 0) + s.bytes)
    end
    println(io, "### Per phase: PhaseTimes `b_*` vs phase-guess aggregation\n")
    println(io, "| phase | b_* (PhaseTimes) | b_*/pass | Σ sites by guess | guess/pass | guess − b_* |")
    println(io, "|---|--:|--:|--:|--:|--:|")
    for f in vcat(CALL_B, LOOP_B)
        lab = B_LABEL[f]; b = getfield(pt, f); g = get(byphase, lab, 0) * scale
        perpass = f in LOOP_B
        @printf(io, "| %s | %s | %s | %s | %s | %+.1f MiB |\n", lab, human(b),
                perpass ? human(b / npass) : "", human(g), perpass ? human(g / npass) : "",
                (g - b) / 2^20)
    end
    for lab in ("refine", "init", "loop-other", "final-other", "postsolve", "fallback", "core?")
        g = get(byphase, lab, 0) * scale
        g == 0 && continue
        note = lab == "refine" ? "; inclusive, already inside direction" :
               lab == "loop-other" ? "; loop lines outside every @phase block" : ""
        @printf(io, "| %s (no b_* field%s) | | | %s | %s | |\n", lab, note, human(g),
                lab in ("refine", "loop-other") ? human(g / npass) : "")
    end
    println(io, "\nΣ loop children b_* per pass: ", human(sum(getfield(pt, f) for f in LOOP_B) / npass),
            " (n_passes=", pt.n_passes, "). Phases are guessed from the `@phase` block that ",
            "contains the owning frame's line (segment table in the header), else from function names. ",
            "A nonzero last column marks either a ",
            "site the guess misplaces or bytes the `b_*` probe sees that the profiler does not ",
            "(the probe counts `Base.gc_bytes`, which includes realloc growth).\n")
    return nothing
end

# ──────────────────────────────────────────────────────────────
#  Driver
# ──────────────────────────────────────────────────────────────

function run_one_instance(io, inst::Instance, top; raw, sample_rate)
    try
        prob = load_tuple(inst)
        report_instance(io, inst, prob, top; raw = raw, sample_rate = sample_rate)
    catch err
        println(io, "## ", inst.name, "\n\nFAILED: ", sprint(showerror, err)[1:min(end, 400)], "\n")
    end
end

function child_markdown(inst::Instance, extra_args)
    cmd = `$(Base.julia_cmd()) --project=$(Base.active_project()) $(@__FILE__) --one $(inst.name) $extra_args`
    out = IOBuffer(); err = IOBuffer()
    ok = success(pipeline(ignorestatus(cmd); stdout = out, stderr = err))
    md = String(take!(out))
    ok || (md *= "\n\nchild failed:\n```\n" * String(take!(err))[1:min(end, 2000)] * "\n```\n")
    return md
end

function select_set(args)
    if "--only" in args
        names = split(arg_value(args, "--only"), ',')
        return [INSTANCES[findfirst(i -> i.name == n, INSTANCES)] for n in names]
    elseif "--quick" in args
        return filter(i -> i.quick, INSTANCES)
    else
        return [INSTANCES[findfirst(i -> i.name == n, INSTANCES)] for n in DEFAULT_SET]
    end
end

function header()
    return string("# Allocation attribution per site and phase\n\n", environment_header(),
                  "\n\nprofiler=Profile.Allocs, src=`", SRC_DIR, "`\n\n",
                  "Phase guess, rule 1 (frames of `_conicIP`, `conicIP`, `preprocess_conicIP`, ",
                  "`_preprocess_core`): the `@phase` block containing the frame's line. ",
                  "`_conicIP` lines outside every block: setup < ", SEGMENTS["ConicIP.jl"].init_start,
                  " ≤ init < ", SEGMENTS["ConicIP.jl"].loop_start, " ≤ loop-other ≤ ",
                  SEGMENTS["ConicIP.jl"].loop_end, " < final-other.\n\n",
                  "| file | lines | phase |\n|---|---|---|\n",
                  join(["| $(f) | $(lo)–$(hi) | $(ph) |" for f in ("ConicIP.jl", "preprocessor.jl")
                        for (lo, hi, ph) in SEGMENTS[f].segs], "\n"),
                  "\n\nPhase guess, rule 2 (no owner frame): nearest enclosing function name, ",
                  "innermost match wins:\n\n",
                  "| pattern | phase |\n|---|---|\n",
                  join(["| `" * replace(p.pattern, "|" => "\\|") * "` | " * ph * " |" for (p, ph) in PHASE_GUESS], "\n"),
                  "\n\n")
end

function main(args)
    "--opt" in args && set_opts!(arg_value(args, "--opt"))
    top = parse(Int, arg_value(args, "--top", "25"))
    raw = "--raw" in args
    sample_rate = parse(Float64, arg_value(args, "--sample-rate", "1"))
    if "--one" in args
        name = arg_value(args, "--one")
        inst = INSTANCES[findfirst(i -> i.name == name, INSTANCES)]
        run_one_instance(stdout, inst, top; raw = raw, sample_rate = sample_rate)
        return
    end
    extra = String["--top", string(top), "--sample-rate", string(sample_rate)]
    raw && push!(extra, "--raw")
    isempty(OPT_STRING[]) || append!(extra, ["--opt", OPT_STRING[]])
    inproc = "--in-process" in args
    md = IOBuffer()
    print(md, header())
    println(md, "mode: ", inproc ? "in-process" : "one child process per instance", "\n")
    for inst in select_set(args)
        println(stderr, "▸ ", inst.name); flush(stderr)
        if inproc
            run_one_instance(md, inst, top; raw = raw, sample_rate = sample_rate)
        else
            print(md, child_markdown(inst, extra))
        end
    end
    text = String(take!(md))
    print(text)
    if "--out" in args
        path = arg_value(args, "--out")
        mkpath(dirname(abspath(path)))
        write(path, text)
        println(stderr, "wrote ", path)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
