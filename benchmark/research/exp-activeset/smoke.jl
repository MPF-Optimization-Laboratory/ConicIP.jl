include(joinpath(@__DIR__, "lib.jl"))
prob = lp_band(200)
sols, t, sfin = capture(prob; optTol=1e-8, T=40)
for (k,s) in enumerate(sols)
    @printf("k=%2d status=%-10s Iter=%d mu=%.2e pr=%.2e du=%.2e gap=%.2e\n",
            k, String(s.status), s.Iter, s.Mu, s.prFeas, s.duFeas, s.rGap)
end
ref = solve1(prob; optTol=1e-11, maxIters=200)
@printf("ref: %s Iter=%d pr=%.2e du=%.2e gap=%.2e pobj=%.16e\n", ref.status, ref.Iter, ref.prFeas, ref.duFeas, ref.rGap, ref.pobj)
s,v = ref.s, ref.v
@printf("m=%d  #(s>v)=%d  min max(s,v)=%.3e  #both<1e-7=%d\n", length(s), sum(s.>v), minimum(max.(s,v)), sum(max.(s,v).<1e-7))
# netlib
nb = joinpath(@__DIR__, "netlib")
p2 = load_netlib(nb, "lp_afiro")
println("afiro: n=", length(p2.c), " Grows=", size(p2.G,1), " conerows=", size(p2.A,1))
r2 = solve1(p2; optTol=1e-10, maxIters=200)
@printf("afiro ref: %s Iter=%d pr=%.2e du=%.2e gap=%.2e obj(nl)=%.10e\n", r2.status, r2.Iter, r2.prFeas, r2.duFeas, r2.rGap, -r2.pobj)
