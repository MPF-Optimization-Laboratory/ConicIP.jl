# Compare per-phase timing CSVs (private)
# ========================================
# Joins the ConicIP and Clarabel outputs of phases.jl on `instance` and
# prints a Markdown report:
#
#   1. per instance: wall, iterations and the comparable phases side by side,
#      the ABSOLUTE differences ConicIP − Clarabel per phase, and the
#      attribution line  Δwall = Σ_phase Δ + Δother  (checked numerically);
#   2. aggregate over the common instances: total Δwall and the share of it
#      per phase (absolute sums, which add up to 100 %), then the shifted
#      geometric mean ratio per phase (secondary: ratios do NOT add up);
#   3. ConicIP only: t_other_call and t_other_loop as a share of wall / loop,
#      flagged when < −1 % (double counting) or > 10 % (coverage gap),
#      refinement share of the direction time, LDL internal shares, GC share,
#      and bytes per pass for each loop child;
#   4. counts: n_passes, n_steps, n_kktupdate, n_solve against Clarabel's
#      iterations, to settle the "one fewer iteration" question.
#
#   julia --project=benchmark benchmark/compare_phases.jl phase-conicip.csv phase-clarabel.csv
#         [--md report.md] [--shift 0.01]        # shift (s) of the phase sgm
#
# `read_csv`, `sgm`, `num`, `fmtnum` come from compare.jl (include-safe).

include(joinpath(@__DIR__, "compare.jl"))

# Comparable whole-call phases: ConicIP's t_final + t_fallback have no
# Clarabel counterpart and are shown as one column (0 for Clarabel), so the
# nine additive phases plus t_other_call still sum to wall for both.
const CALL_PHASES = [
    ("frontend", r -> num(r, "t_frontend")),
    ("presolve", r -> num(r, "t_presolve")),
    ("equilibrate", r -> num(r, "t_equilibrate")),
    ("setup", r -> num(r, "t_setup")),
    ("init", r -> num(r, "t_init")),
    ("loop", r -> num(r, "t_loop")),
    ("final+fallback", r -> num(r, "t_final") + num(r, "t_fallback")),
    ("postsolve", r -> num(r, "t_postsolve")),
    ("other", r -> num(r, "t_other_call")),
]
const LOOP_PHASES = [
    ("scaling", r -> num(r, "t_scaling")),
    ("kktupdate", r -> num(r, "t_kktupdate")),
    ("direction", r -> num(r, "t_direction")),
    ("loop_rest", r -> num(r, "t_loop_rest")),
    ("other_loop", r -> num(r, "t_other_loop")),
]
# Shown alongside, not part of the additive set.
kkt_total(r) = num(r, "t_kkt_total")

# Seconds: fixed 4 decimals down to 0.1 ms, scientific below (quick instances).
fmts(x) = isnan(x) ? "" : abs(x) >= 1e-3 || x == 0 ? @sprintf("%.4f", x) : @sprintf("%.2e", x)
fmtd(x) = isnan(x) ? "" : abs(x) >= 1e-3 || x == 0 ? @sprintf("%+.4f", x) : @sprintf("%+.2e", x)
fmtpct(x) = isnan(x) ? "" : @sprintf("%.1f%%", 100x)
fmtspct(x) = isnan(x) ? "" : @sprintf("%+.1f%%", 100x)
fmtratio(x) = isnan(x) || isinf(x) ? "" : @sprintf("%.2f", x)
# Ratio of shifted geometric means; blank when the denominator is (numerically) zero.
sgm_ratio(a, b; shift) = (gb = sgm(b; shift = shift); gb <= 1e-12 ? NaN : sgm(a; shift = shift) / gb)
fmtmib(x) = isnan(x) ? "" : @sprintf("%.2f", x / 2^20)

function load_phase_csv(path)
    header, rows = read_csv(path)
    return header, Dict(r["instance"] => r for r in rows)
end

# `Δwall = Σ Δphase + Δother`, phases in `set` (`other` is a member).
function attribution_line(c, k, set)
    Δwall = num(c, "wall") - num(k, "wall")
    parts = [(name, f(c) - f(k)) for (name, f) in set]
    Σ = sum(last, parts)
    return Δwall, parts, Σ
end

function main(args)
    paths = String[]; mdpath = nothing; shift = 0.01
    i = 1
    while i <= length(args)
        if args[i] == "--md"
            mdpath = args[i + 1]; i += 2
        elseif args[i] == "--shift"
            shift = parse(Float64, args[i + 1]); i += 2
        else
            push!(paths, args[i]); i += 1
        end
    end
    length(paths) == 2 || error("compare_phases.jl needs exactly two CSV files: conicip, clarabel")
    io = mdpath === nothing ? stdout : IOBuffer()

    hc, C = load_phase_csv(paths[1])
    hk, K = load_phase_csv(paths[2])
    names = String[]
    for r in (C, K), n in keys(r)
        n in names || push!(names, n)
    end
    sort!(names; by = n -> (get(C, n, nothing) === nothing ? Inf : -num(C[n], "wall")))
    common = [n for n in names if haskey(C, n) && haskey(K, n) &&
              solved(C[n]) && solved(K[n]) && !isnan(num(C[n], "wall")) && !isnan(num(K[n], "wall"))]

    println(io, "# Per-phase comparison: ConicIP vs Clarabel\n")
    println(io, "- **conicip**: `", paths[1], "`  \n  ", hc)
    println(io, "- **clarabel**: `", paths[2], "`  \n  ", hk)
    println(io, "\nInstances: ", length(names), "; solved by both: ", length(common),
            ". Times in seconds; Δ = ConicIP − Clarabel.\n")

    # ── 1. per instance ─────────────────────────────────────────
    println(io, "## 1. Per instance\n")
    println(io, "### Wall, iterations, status\n")
    println(io, "| instance | C status | K status | C verified | K verified | C wall | K wall | Δwall | ratio | C passes | C steps | K iters |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names
        c = get(C, n, nothing); k = get(K, n, nothing)
        cells = [n,
                 c === nothing ? "—" : get(c, "status", ""), k === nothing ? "—" : get(k, "status", ""),
                 c === nothing ? "" : (isverified(c) ? "yes" : "no"),
                 k === nothing ? "" : (isverified(k) ? "yes" : "no"),
                 c === nothing ? "" : fmts(num(c, "wall")), k === nothing ? "" : fmts(num(k, "wall")),
                 (c === nothing || k === nothing) ? "" : fmtd(num(c, "wall") - num(k, "wall")),
                 (c === nothing || k === nothing) ? "" : fmtratio(num(c, "wall") / num(k, "wall")),
                 c === nothing ? "" : get(c, "iters", ""), c === nothing ? "" : get(c, "n_steps", ""),
                 k === nothing ? "" : get(k, "iters", "")]
        println(io, "| ", join(cells, " | "), " |")
    end

    for (title, set) in (("Whole-call phases", CALL_PHASES), ("Loop children", LOOP_PHASES))
        println(io, "\n### ", title, " (C | K | Δ per phase)\n")
        extra = title == "Loop children" ? " | kkt_total C | K | Δ" : ""
        println(io, "| instance | ", join(("$(nm) C | K | Δ" for (nm, _) in set), " | "), extra, " |")
        println(io, "|---|", join(("---|---|---" for _ in set), "|"), (isempty(extra) ? "" : "|---|---|---"), "|")
        for n in common
            c = C[n]; k = K[n]
            cells = String[]
            for (_, f) in set
                append!(cells, [fmts(f(c)), fmts(f(k)), fmtd(f(c) - f(k))])
            end
            if !isempty(extra)
                append!(cells, [fmts(kkt_total(c)), fmts(kkt_total(k)), fmtd(kkt_total(c) - kkt_total(k))])
            end
            println(io, "| ", n, " | ", join(cells, " | "), " |")
        end
    end

    println(io, "\n### Attribution: Δwall = Σ_phase Δ + Δother\n")
    println(io, "One line per instance over the whole-call phases (`other` included); the residual is Δwall − Σ and should be 0 up to rounding.\n")
    for n in common
        Δwall, parts, Σ = attribution_line(C[n], K[n], CALL_PHASES)
        terms = join((@sprintf("%s %+.4f", nm, d) for (nm, d) in parts), " ")
        @printf(io, "- **%s**: Δwall %+.4f = %s  (residual %+.1e)\n", n, Δwall, terms, Δwall - Σ)
    end

    # ── 2. aggregate ───────────────────────────────────────────
    println(io, "\n## 2. Aggregate over the ", length(common), " common instances\n")
    if isempty(common)
        println(io, "(no instance solved by both files)")
    else
        ΣΔwall = sum(num(C[n], "wall") - num(K[n], "wall") for n in common)
        ΣC = sum(num(C[n], "wall") for n in common); ΣK = sum(num(K[n], "wall") for n in common)
        @printf(io, "Total wall: ConicIP %.4f s, Clarabel %.4f s, ΣΔwall %+.4f s (ratio of totals %.2f).\n\n",
                ΣC, ΣK, ΣΔwall, ΣC / ΣK)
        for (title, set) in (("Whole-call phases", CALL_PHASES), ("Loop children (within Δloop)", LOOP_PHASES))
            ref = title == "Whole-call phases" ? ΣΔwall :
                  sum(num(C[n], "t_loop") - num(K[n], "t_loop") for n in common)
            println(io, "### ", title, ": absolute sums (shares add up)\n")
            println(io, "| phase | Σ ConicIP | Σ Clarabel | ΣΔ | share of ", title == "Whole-call phases" ? "ΣΔwall" : "ΣΔloop", " | sgm ratio C/K (shift ", shift, " s) |")
            println(io, "|---|---|---|---|---|---|")
            for (nm, f) in set
                sc = sum(f(C[n]) for n in common); sk = sum(f(K[n]) for n in common)
                @printf(io, "| %s | %.4f | %.4f | %+.4f | %s | %s |\n", nm, sc, sk, sc - sk,
                        ref == 0 ? "" : fmtpct((sc - sk) / ref),
                        fmtratio(sgm_ratio([f(C[n]) for n in common], [f(K[n]) for n in common]; shift = shift)))
            end
            if title != "Whole-call phases"
                sc = sum(kkt_total(C[n]) for n in common); sk = sum(kkt_total(K[n]) for n in common)
                @printf(io, "| kkt_total (kktupdate+direction, not additive) | %.4f | %.4f | %+.4f | %s | %s |\n",
                        sc, sk, sc - sk, ref == 0 ? "" : fmtpct((sc - sk) / ref),
                        fmtratio(sgm_ratio([kkt_total(C[n]) for n in common],
                                           [kkt_total(K[n]) for n in common]; shift = shift)))
            end
            println(io)
        end
        @printf(io, "sgm wall ratio C/K (shift 0.1 s as in compare.jl): %s. ",
                fmtratio(sgm_ratio([num(C[n], "wall") for n in common],
                                   [num(K[n], "wall") for n in common]; shift = 0.1)))
        println(io, "Per-phase sgm ratios are secondary: they do not add up to the wall ratio, and small phases are dominated by the shift.\n")
    end

    # ── 3. ConicIP only ────────────────────────────────────────
    println(io, "## 3. ConicIP instrumentation checks\n")
    println(io, "Flags: `DOUBLE` when a remainder is < −1 % (double counting), `GAP` when > 10 % (coverage gap).\n")
    println(io, "| instance | wall | other_call | % wall | loop | other_loop | % loop | flags | refine/direction | ldl_factor/kktupdate | ldl_solve/direction | ldl_resid/direction | gc/wall |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    cnames = [n for n in names if haskey(C, n) && !isnan(num(C[n], "wall"))]
    for n in cnames
        c = C[n]
        wall = num(c, "wall"); oc = num(c, "t_other_call"); loop = num(c, "t_loop"); ol = num(c, "t_other_loop")
        fc = oc / wall; fl = loop > 0 ? ol / loop : NaN
        flags = String[]
        fc < -0.01 && push!(flags, "DOUBLE(call)"); fc > 0.10 && push!(flags, "GAP(call)")
        !isnan(fl) && fl < -0.01 && push!(flags, "DOUBLE(loop)"); !isnan(fl) && fl > 0.10 && push!(flags, "GAP(loop)")
        dir = num(c, "t_direction"); upd = num(c, "t_kktupdate")
        println(io, "| ", join([n, fmts(wall), fmtd(oc), fmtspct(fc), fmts(loop), fmtd(ol), fmtspct(fl),
                                isempty(flags) ? "ok" : join(flags, " "),
                                fmtpct(num(c, "t_dir_refine") / dir), fmtpct(num(c, "t_ldl_factor") / upd),
                                fmtpct(num(c, "t_ldl_solve") / dir), fmtpct(num(c, "t_ldl_resid") / dir),
                                fmtpct(num(c, "t_gc") / wall)], " | "), " |")
    end
    println(io, "\n### Bytes per pass (MiB), ConicIP loop children\n")
    bcols = ["b_scaling", "b_residuals", "b_kktupdate", "b_direction", "b_rhs", "b_linesearch"]
    println(io, "| instance | passes | ", join(bcols, " | "), " | Σ loop MiB/pass |")
    println(io, "|---|---|", join(("---" for _ in bcols), "|"), "|---|")
    for n in cnames
        c = C[n]; np = num(c, "iters")
        per = [num(c, b) / np for b in bcols]
        println(io, "| ", n, " | ", fmtint(np), " | ", join(fmtmib.(per), " | "), " | ", fmtmib(sum(per)), " |")
    end

    # ── 4. counts ──────────────────────────────────────────────
    println(io, "\n## 4. Iteration counts\n")
    println(io, "ConicIP numbers passes from 1 and counts the final convergence check; Clarabel counts only passes that reach a KKT update (`info.iterations`).\n")
    println(io, "| instance | C passes | C steps | C kktupdate | C solves | C refine att. | C BarrierIterations | K iters | K BarrierIterations | passes − K | kktupdate − K | steps − K |")
    println(io, "|---|---|---|---|---|---|---|---|---|---|---|---|")
    for n in names
        c = get(C, n, nothing); k = get(K, n, nothing)
        g(r, key) = r === nothing ? "" : get(r, key, "")
        ki = k === nothing ? NaN : num(k, "iters")
        d(key) = (c === nothing || isnan(ki)) ? "" : @sprintf("%+d", round(Int, num(c, key) - ki))
        println(io, "| ", join([n, g(c, "iters"), g(c, "n_steps"), g(c, "n_kktupdate"), g(c, "n_solve"),
                                g(c, "n_refine_attempt"), g(c, "iters_reported"), g(k, "iters"),
                                g(k, "iters_reported"), d("iters"), d("n_kktupdate"), d("n_steps")], " | "), " |")
    end

    if mdpath !== nothing
        write(mdpath, String(take!(io)))
        println("wrote ", mdpath)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
