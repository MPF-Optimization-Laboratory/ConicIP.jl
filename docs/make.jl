using Documenter
using Literate
using ConicIP

DocMeta.setdocmeta!(ConicIP, :DocTestSetup, :(using ConicIP); recursive=true)

# Generate tutorial markdown from Literate.jl scripts
tutorial_src = joinpath(@__DIR__, "src", "tutorials")
tutorial_out = joinpath(@__DIR__, "src", "tutorials", "generated")
mkpath(tutorial_out)

for file in readdir(tutorial_src)
    endswith(file, ".jl") || continue
    Literate.markdown(joinpath(tutorial_src, file), tutorial_out;
                      documenter=true, credit=false)
end

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
            "Getting Started" => "tutorials/generated/getting_started.md",
            "Linear Programs" => "tutorials/generated/lp.md",
            "Quadratic Programs" => "tutorials/generated/qp.md",
            "Second-Order Cone" => "tutorials/generated/socp.md",
            "Semidefinite (Experimental)" => "tutorials/generated/sdp.md",
        ],
        "How-To Guides" => [
            "JuMP Integration" => "guides/jump.md",
            "KKT Solvers" => "guides/kkt_solvers.md",
            "Preprocessing" => "guides/preprocessing.md",
        ],
        "Mathematical Background" => "background.md",
        "API Reference" => "api.md",
    ],
)

deploydocs(;
    repo      = "github.com/MPF-Optimization-Laboratory/ConicIP.jl",
    devbranch = "master",
    push_preview = true,
)
