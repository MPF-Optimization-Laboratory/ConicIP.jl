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
        "Installation" => "installation.md",
        "Tutorials" => [
            "Getting Started" => "tutorials/getting_started.md",
            "Linear Programs" => "tutorials/lp.md",
            "Quadratic Programs" => "tutorials/qp.md",
            "Second-Order Cone" => "tutorials/socp.md",
            "Semidefinite (Experimental)" => "tutorials/sdp.md",
        ],
        "JuMP Integration" => "jump.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo      = "github.com/MPF-Optimization-Laboratory/ConicIP.jl",
    devbranch = "master",
    push_preview = true,
)
