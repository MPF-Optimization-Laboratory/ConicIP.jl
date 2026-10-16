include(joinpath(@__DIR__, "lib.jl"))
nb = joinpath(@__DIR__, "netlib")
probs = Dict("lp-band-2000"=>lp_band(2000), "afiro"=>load_netlib(nb,"lp_afiro"),
             "degen2"=>load_netlib(nb,"lp_degen2"), "socp-sumnorms-150"=>socp_sum_of_norms(150; d=200))
for (nm,pr) in probs
    bad = Int[]
    s0 = solve1(pr; optTol=1e-10, maxIters=200)
    for k in 1:s0.Iter
        s = solve1(pr; optTol=1e-10, maxIters=k)
        s.Iter == k || push!(bad, k)
    end
    println(nm, ": iterations=", s0.Iter, "  truncated runs where sol.Iter != maxIters: ", isempty(bad) ? "none" : bad)
end
