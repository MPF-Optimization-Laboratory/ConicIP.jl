using Documenter
using ConicIP

DocMeta.setdocmeta!(ConicIP, :DocTestSetup, :(using ConicIP); recursive=true)

makedocs(;
    modules  = [ConicIP],
    sitename = "ConicIP.jl",
    authors  = "MPF Optimization Laboratory",
    format   = Documenter.HTML(;
        canonical = "https://MPF-Optimization-Laboratory.github.io/ConicIP.jl",
        edit_link = "master",
    ),
    pages = [
        "Home" => "index.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo      = "github.com/MPF-Optimization-Laboratory/ConicIP.jl",
    devbranch = "master",
    push_preview = true,
)
