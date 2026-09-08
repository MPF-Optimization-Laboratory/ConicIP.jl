# Compare benchmark CSVs
# ======================
# Joins two or more suite.jl / baselines.jl result files on the instance
# name and prints a Markdown table (status, iterations, time, verified per
# file) followed by solved/verified counts and shifted geometric means
#
#   sgm(x; s) = exp(mean(log(x + s))) − s
#
# of time and iterations, both over the instances every file solved and
# over each file's own solved set.
#
#   julia --project=benchmark benchmark/compare.jl a.csv b.csv [c.csv ...]
#         [--shift-time 0.1] [--shift-iter 5]

using Printf, Statistics

# `(header_comment, rows::Vector{Dict{String,String}})`; `#` lines are skipped.
function read_csv(path)
    header = ""; columns = String[]; rows = Dict{String, String}[]
    for line in eachline(path)
        isempty(strip(line)) && continue
        if startswith(line, '#')
            isempty(header) && (header = strip(line[2:end]))
            continue
        end
        fields = String.(split(line, ','; keepempty = true))
        if isempty(columns)
            columns = fields
        else
            length(fields) < length(columns) && append!(fields, fill("", length(columns) - length(fields)))
            push!(rows, Dict(columns[i] => fields[i] for i in eachindex(columns)))
        end
    end
    return header, rows
end

# Label for a file: its `solver` column when uniform, else the file stem.
function label_for(path, rows)
    solvers = unique(get(r, "solver", "") for r in rows)
    length(solvers) == 1 && !isempty(solvers[1]) && return solvers[1]
    return first(splitext(basename(path)))
end

solved(r) = lowercase(get(r, "status", "")) == "optimal"
isverified(r) = get(r, "verified", "") == "true"
num(r, k) = something(tryparse(Float64, get(r, k, "")), NaN)

sgm(xs; shift) = isempty(xs) ? NaN : exp(mean(log.(xs .+ shift))) - shift

fmtnum(x) = isnan(x) ? "" : x >= 100 ? @sprintf("%.0f", x) : @sprintf("%.3g", x)
fmtint(x) = isnan(x) ? "" : string(round(Int, x))

function main(args)
    shift_time = 0.1; shift_iter = 5.0
    paths = String[]
    i = 1
    while i <= length(args)
        if args[i] == "--shift-time"
            shift_time = parse(Float64, args[i + 1]); i += 2
        elseif args[i] == "--shift-iter"
            shift_iter = parse(Float64, args[i + 1]); i += 2
        else
            push!(paths, args[i]); i += 1
        end
    end
    length(paths) >= 2 || error("compare.jl needs at least two CSV files")

    tables = [read_csv(p) for p in paths]
    byname = [Dict(r["name"] => r for r in rows) for (_, rows) in tables]
    labels = [label_for(paths[k], tables[k][2]) for k in eachindex(paths)]
    names = String[]
    for (_, rows) in tables, r in rows
        r["name"] in names || push!(names, r["name"])
    end

    for k in eachindex(paths)
        println("- **", labels[k], "**: `", paths[k], "`  \n  ", tables[k][1])
    end
    println()
    cols = ["status", "iters", "time", "verified"]
    println("| instance | ", join(("$(l) $(c)" for l in labels for c in cols), " | "), " |")
    println("|---|", join(("---" for _ in labels for _ in cols), "|"), "|")
    for name in names
        cells = String[]
        for k in eachindex(paths)
            r = get(byname[k], name, nothing)
            if r === nothing
                append!(cells, ["—", "", "", ""])
            else
                append!(cells, [get(r, "status", ""), get(r, "iters", ""),
                                fmtnum(num(r, "t_total")), isverified(r) ? "yes" : "no"])
            end
        end
        println("| ", name, " | ", join(cells, " | "), " |")
    end
    println()

    solved_sets = [Set(n for n in names if haskey(byname[k], n) && solved(byname[k][n]))
                   for k in eachindex(paths)]
    common = reduce(intersect, solved_sets)
    println("Instances: ", length(names), "; solved by every file: ", length(common),
            " (shifts: time +", shift_time, ", iterations +", shift_iter, ")")
    println()
    println("| file | solved | verified | sgm time (common) | sgm iters (common) | sgm time (own) | sgm iters (own) |")
    println("|---|---|---|---|---|---|---|")
    for k in eachindex(paths)
        rows_k = byname[k]
        nver = count(n -> haskey(rows_k, n) && isverified(rows_k[n]), names)
        t_common = [num(rows_k[n], "t_total") for n in common]
        i_common = [num(rows_k[n], "iters") for n in common]
        t_own = [num(rows_k[n], "t_total") for n in solved_sets[k]]
        i_own = [num(rows_k[n], "iters") for n in solved_sets[k]]
        @printf("| %s | %d/%d | %d | %s | %s | %s | %s |\n", labels[k],
                length(solved_sets[k]), length(names), nver,
                fmtnum(sgm(filter(!isnan, t_common); shift = shift_time)),
                fmtnum(sgm(filter(!isnan, i_common); shift = shift_iter)),
                fmtnum(sgm(filter(!isnan, t_own); shift = shift_time)),
                fmtnum(sgm(filter(!isnan, i_own); shift = shift_iter)))
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
