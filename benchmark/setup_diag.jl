# Attribution of ConicIP's `t_setup` phase (private)
# ==================================================
# `t_setup` runs in `_conicIP` from function entry to just after the KKT
# solver is constructed. This script rebuilds the data `_conicIP` actually
# sees on a harness instance — the *equilibrated* tuple, since the MOI route
# runs `preprocess_conicIP` → `conicIP` → `equilibrate_conicIP` → `_conicIP` —
# and then times each candidate component of that region separately.
#
#   julia --project benchmark/setup_diag.jl [instance ...]
#
# Default instances: lp-band-20000 qp-band-20000. Each component is timed
# `reps` times and the FASTEST repetition is reported (other agents may be
# running on the same machine), with the allocation count of one call.
# Allocation bytes are the stable signal; wall times are indicative.

include(joinpath(@__DIR__, "suite.jl"))

using ConicIP
using ConicIP: _absmat, structurally_zero_rows, structurally_zero_cols,
               _ldl_pattern, _ldl_flops, _choose_kktsolver, choose_kktsolver,
               kktsolver_ldl, kktsolver_qr, default_kktsolver,
               dense_kkt_bytes, dense_kkt_flops, _structural_nnz,
               equilibrate_conicIP, Id
using QDLDL, AMD, SparseArrays, LinearAlgebra, Printf

const REPS = 3

# Fastest of REPS repetitions, plus the bytes of one call.
function best(f, reps = REPS)
    f()                              # compile / warm
    t = Inf
    for _ in 1:reps
        t = min(t, @elapsed f())
    end
    b = @allocated f()
    return (t, b)
end

row(label, t, b) = @printf("  %-42s %9.4f s  %12.0f B\n", label, t, b)

function diagnose(name, prob)
    (; Q, c, A, b, cone_dims, G, d) = prob
    eq = equilibrate_conicIP(Q, c, A, b, cone_dims, G, d)
    Qe, ce, Ae, be, Ge, de = eq.Q, eq.c, eq.A, eq.b, eq.G, eq.d
    n = length(ce); m = size(Ae, 1); p = size(Ge, 1)
    @printf("\n=== %s ===  n=%d m=%d p=%d  nnz(Q)=%d nnz(A)=%d  typeof(Q)=%s\n",
            name, n, m, p, _structural_nnz(Qe), _structural_nnz(Ae),
            string(typeof(Qe)))

    # ── whole phase, for reference ──
    pt = ConicIP.PhaseTimes()
    ConicIP._conicIP(Qe, ce, Ae, be, cone_dims, Ge, de;
                     verbose = false, timing = pt, maxIters = 1)
    pt = ConicIP.PhaseTimes()
    ConicIP._conicIP(Qe, ce, Ae, be, cone_dims, Ge, de;
                     verbose = false, timing = pt, maxIters = 1)
    @printf("  %-42s %9.4f s  %12.0f B   (whole phase)\n",
            "t_setup", pt.t_setup / 1e9, pt.b_setup)

    # ── components ──
    row("_absmat(Q)",  best(() -> _absmat(Qe))...)
    row("_absmat(A)",  best(() -> _absmat(Ae))...)
    row("_absmat(G)",  best(() -> _absmat(Ge))...)
    row("structurally_zero_rows(G)", best(() -> structurally_zero_rows(Ge))...)
    row("structurally_zero_cols(Q)", best(() -> structurally_zero_cols(Qe))...)
    row("structurally_zero_cols(A)", best(() -> structurally_zero_cols(Ae))...)
    row("structurally_zero_cols(G)", best(() -> structurally_zero_cols(Ge))...)
    row("Zc = 3 × cols .& .&", best(() ->
        structurally_zero_cols(Qe) .& structurally_zero_cols(Ae) .&
        structurally_zero_cols(Ge))...)
    row("norm(Q, Inf)", best(() -> norm(Qe, Inf))...)
    row("e / block_data / buffers (zeros(m)×9)",
        best(() -> (zeros(m), zeros(m), zeros(m), zeros(m), zeros(m),
                    zeros(m), zeros(m), zeros(n), zeros(p)))...)

    bytes = dense_kkt_bytes(n, m, p)
    @printf("  dense_kkt_bytes = %.3g  (budget %.3g) → %s\n", bytes, 4 * 2.0^30,
            bytes > 4 * 2^30 ? "LDL, no pattern from choose" : "pattern built in choose")
    row("_choose_kktsolver", best(() -> _choose_kktsolver(Qe, Ae, Ge, cone_dims))...)
    println("     chosen = ", nameof(_choose_kktsolver(Qe, Ae, Ge, cone_dims)[1]))

    # ── inside _ldl_pattern ──
    row("_ldl_pattern (total)", best(() -> _ldl_pattern(Qe, Ae, Ge, cone_dims))...)
    # Reference costs of the input handling the pattern assembly can avoid:
    # a copy of each already-sparse input, and a triplet listing of it.
    row("  [ref] sparse(Q)+sparse(A)+sparse(G)",
        best(() -> (sparse(Qe), sparse(Ae), sparse(Ge)))...)
    Qs = sparse(Qe); As = sparse(Ae); Gs = sparse(Ge)
    row("  diag(Qs)", best(() -> Vector{Float64}(diag(Qs)))...)
    row("  [ref] findnz(Q)+findnz(A)+findnz(G)",
        best(() -> (findnz(Qs), findnz(As), findnz(Gs)))...)
    pat = _ldl_pattern(Qe, Ae, Ge, cone_dims)
    K = pat.K
    @printf("     N=%d nnz(K)=%d nlift=%d\n", pat.N, nnz(K), pat.nlift)
    row("  amd(K)", best(() -> amd(K))...)
    row("  _ldl_pattern with perm_hint (no amd)",
        best(() -> _ldl_pattern(Qe, Ae, Ge, cone_dims; perm_hint = pat.perm))...)
    row("  _ldl_flops(pat)", best(() -> _ldl_flops(pat))...)
    row("  permute_symmetric(K, invperm)",
        best(() -> QDLDL.permute_symmetric(K, invperm(pat.perm)))...)

    # ── the rest of kktsolver_ldl ──
    row("kktsolver_ldl(; pattern = pat)",
        best(() -> kktsolver_ldl(Qe, Ae, Ge, cone_dims; pattern = pat))...)
    row("  qdldl(K; perm, logical = true)",
        best(() -> qdldl(copy(K); perm = pat.perm, logical = true,
                         Dsigns = pat.Dsigns, regularize_eps = 1e-13,
                         regularize_delta = 2e-7))...)
    row("default_kktsolver (choose + build)",
        best(() -> default_kktsolver(Qe, Ae, Ge, cone_dims))...)
    return nothing
end

const NAMED = Dict("lp-band-2000"   => () -> lp_band(2_000),
                   "lp-band-20000"  => () -> lp_band(20_000),
                   "lp-band-200000" => () -> lp_band(200_000),
                   "qp-band-2000"   => () -> qp_band(2_000),
                   "qp-band-20000"  => () -> qp_band(20_000),
                   "qp-band-200000" => () -> qp_band(200_000))

function main(args)
    names = isempty(args) ? ["lp-band-20000", "qp-band-20000"] : args
    for nm in names
        haskey(NAMED, nm) || (println("unknown instance $nm"); continue)
        diagnose(nm, NAMED[nm]())
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main(ARGS)
end
