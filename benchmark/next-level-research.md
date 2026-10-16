# What would take ConicIP to the next level — research phase 2026-09-20 (private)

**Status:** research only; no code changed. Read after `RESUME.md` when deciding the lane
after v0.6.0. This file is on the strip list (`strip-release.sh`) and never goes upstream: it
names other solvers and cites private harness numbers.

**Setting.** dev `97f1111` (= v0.6.0 + private harness). On the 23-instance harness
(dev 3881ac2 study, `results/*-3881ac2.*`) ConicIP and Clarabel both solve and verify 23/23;
totals 8.94 s vs 8.78 s, shifted-geometric-mean wall ratio 1.16; iteration counts within
−2…+3 except sched_50_50_scaled (+10). The maintainer's question: what changes bring the code
clearly beyond that class — specialized linear algebra, iterative solves with preconditioning,
finite termination / active-set / crossover, or new mathematics?

**Method.** Six Opus subagents, three literature surveys (KKT linear algebra; iteration-count
and finite-termination algorithms; SDP at scale) and three read-only experiments on the
solver's own iterates (factorization headroom vs BLAS-3 back ends; Krylov solves with
stale-factor and incomplete preconditioners; Tapia / Ye identification). The full reports,
scripts, results, and scratch environments are kept in `benchmark/research/` (see its README;
the 161 MB of captured KKT sequences are regenerable and were not kept). §8 summarizes the
capture methods.

---

## 1. Headline

Within the architecture ConicIP now shares with the best open conic IPMs (sparse
quasi-definite LDLᵀ of the augmented system, NT scaling, Mehrotra PC), the remaining headroom
on LP/QP/SOCP of the harness kind is **about 1.3–1.5× and is engineering, not mathematics**.
A qualitative jump requires changing what the solver is asked to do: SDP at scale,
warm-started sequences, certified high-accuracy output, or GPU. Three of the four named
directions are ruled out by measurements on ConicIP's own Newton systems:

| direction | verdict on this problem class | evidence |
|---|---|---|
| Krylov / inexact Newton KKT solves | **no**, at any accuracy | §3; consecutive KKT matrices differ by 52–99 % in norm; stale-factor GMRES needs 10–42 its vs a 4–13 break-even; literature: IPX 3.67× slower than a commercial barrier over 164 LPs |
| Supernodal / BLAS-3 / threaded factorization | **only as a fill-triggered dispatch** | §2; 20/23 KKT matrices have no supernodes; −13 % sgm total, 95 % of it one instance (sum-of-norms, 13.5× there) |
| Finite termination / crossover | **accuracy feature, not speed** | §4; median saving 0.5 of 15 iterations, projection costs 1.3–3.2 iterations; but gap → exactly 0, residuals → 1e-13…1e-16 |
| Iteration-reducing algorithms | nothing > 20 % on cold solves | §5; correctors are the exception and must be gated on the measured fac/solve ratio |

Where a real jump exists: **SDP at scale** (§6.1, largest capability gap, well-mapped),
**warm-started conic sequences** (§6.2, 50–75 % iteration reductions documented, unsolved in
production), **certified-exact LP output and SOCP face identification** (§6.3, distinctive,
research content), **GPU direct solves** (§6.4, the only documented order of magnitude, not a
pure-Julia project).

---

## 2. Linear algebra: where the time is and what BLAS-3 buys (exp-factor)

### 2.1 Decomposition of the 8.94 s (dev 3881ac2, phase timers, LDL rows are inclusive diagnostics)

| block | seconds | % of total | lever |
|---|---|---|---|
| factorization (`t_ldl_factor`) | 3.03 | 33.9 | fill-triggered supernodal dispatch; pays only on high-fill instances |
| back-solves (`t_ldl_solve`) | 1.61 | 18.0 | no kernel headroom (±40 % across four libraries): *fewer* solves |
| refinement residual evaluations (`t_ldl_resid` + `t_dir_refine_resid`) | 1.41 | 15.7 | cheaper 4×4 residual evaluation (roadmap (b')), not a screen |
| outside the loop (frontend, presolve, equilibrate, setup, init, postsolve) | 1.74 | 19.5 | setup is 2.06× Clarabel's (QDLDL symbolic memory, AMD); 50 % of qp-band-200000 |
| residuals + rhs + line search (`loop_rest`) | 0.76 | 8.6 | SOC allocations (roadmap item (a)) |
| kkt glue | 0.40 | 4.5 | |
| NT scaling | 0.07 | 0.8 | done |

ConicIP spends as much time checking and correcting its solves (34 %) as computing the
factorizations (34 %). K assembly is 0.6 % / 5.1 % of `kktupdate` (sumnorms-1500,
lp-band-200000): already free.

### 2.2 Ceilings (23 instances; sum 8.937 s, sgm shift 0.1 = 0.1194 s)

| lever | sum | sgm |
|---|---|---|
| measured supernodal back end, 1 BLAS thread | 7.520 (−15.9 %) | −10.8 % |
| measured supernodal back end, 6 BLAS threads | 7.376 (−17.5 %) | −13.2 % |
| factorization halved everywhere (hypothetical) | 7.424 (−16.9 %) | −10.2 % |
| back-solves + refinement halved | 7.430 (−16.9 %) | −10.9 % |
| everything outside `kktupdate + direction` free | **6.319 (−29.3 %)** | **−24.6 %** |

Ceiling (iii) is largest on 22 of 23 instances. Only socp-sumnorms-1500 is factorization-bound
(91 % of wall).

### 2.3 Measured back ends on the captured mid-run KKT matrices (ms, best of 5, 1 Julia thread)

| instance | n_K | fill nnz(L)/nnz(triu K) | flop frac in chains ≥16 | CHOLMOD auto picks | QDLDL refactor! | CHOLMOD Cholesky proxy, same pattern 1thr / 6thr | speedup | UMFPACK lu! |
|---|---|---|---|---|---|---|---|---|
| lp-band-20000 | 80 000 | 1.29 | 0.00 | simplicial | 3.75 | 8.98 / 8.87 | 0.42× | 35.0 |
| lp-band-200000 | 800 000 | 1.29 | 0.00 | simplicial | 36.97 | 86.4 / 87.1 | 0.43× | 296 |
| qp-band-200000 | 600 000 | 0.62 | 0.00 | simplicial | 8.36 | 26.1 / 28.1 | 0.32× | 174 |
| socp-sumnorms-1500 | 14 000 | **24.05** | **0.43** | supernodal | **142.4** | **23.0 / 10.5** | **6.2× / 13.5×** | 169.9 |
| nb | 4 889 | 1.98 | 0.01 | supernodal | 16.0 | 11.8 / 11.8 | 1.36× | 27.5 |
| chainsing-1000-1 | 31 940 | 0.72 | 0.00 | simplicial | 0.48 | 1.38 / 1.37 | 0.35× | 8.0 |
| qssp30 | 15 038 | 2.39 | 0.64 | supernodal | 2.82 | 2.82 / 2.65 | 1.06× | 10.7 |
| issue10 | 17 130 | 0.79 | 0.00 | simplicial | 0.26 | 0.79 / 0.79 | 0.34× | 6.1 |

Notes. The proxy is `|K|` off-diagonal with diagonal `1 + Σ|K_ij|` (SPD, identical pattern):
the honest upper bound for a signed supernodal LDLᵀ. CHOLMOD's ordering carries 1.18–1.81×
more nnz(L) than ConicIP's, so per flop the speedups are understated. CHOLMOD's *simplicial*
`ldlt!` of the genuine quasi-definite K succeeded on all eight (static regularization keeps
the signed pivots) and was 2.3–6.4× slower than QDLDL. LDLFactorizations.jl lands within
10–30 % of QDLDL. Forcing supernodal on low-fill matrices makes them 1.8–2.8× worse
(qp-band-200000 26 → 73 ms). Threading changes nothing except on the one supernodal matrix.
Back-solves: QDLDL within ±40 % of CHOLMOD and UMFPACK everywhere; a supernodal factor's
back-solve is 2–7× slower on low-fill matrices.

**Conclusion.** QDLDL is at or ahead of the state of the art for scalar sparse LDLᵀ on these
matrices. The literature's threshold (CHOLMOD `supernodal_switch = 40` flops per nnz(L),
Clarabel-Rust uses the same rule for its faer back end) agreed with the measurement on all
eight. A fill-triggered dispatch turns socp-sumnorms-1500 from 1.77 s into ≈0.28 s and changes
nothing else. The suite under-represents high-fill problems; any SDP or dense-column instance
would show the large win.

### 2.4 Literature (lit-linalg)

- Supernodal vs simplicial: CHOLMOD Table I gives 3.3–10× peak GFlop/s, realized only above
  flops/nnz(L) ≈ 30–45 (1 thread), 80–120 (2 threads). No published QDLDL-vs-supernodal
  timing on IPM KKT matrices exists; the Clarabel paper describes the switch but gives no A/B.
- Threaded multifrontal (HiPO in HiGHS, 2025): ≈2× typical, >4× on a few of 46 Mittelmann
  LPs at 8 cores; parallelism from the tree, BLAS serial for bit-reproducibility; they warn
  the profile changes completely with Apple's BLAS.
- Normal equations vs augmented: HiPO picks NE on ~75 % of Mittelmann LPs, AS on ~25 %, AS
  only feasible on PyPSA energy LPs (MBs vs >128 GB). Moot for a conic solver: NT scaling of
  SOC/PSD blocks does not collapse to AD⁻¹Aᵀ.
- GPU (cuDSS): CuClarabel ≈10× vs multithreaded Rust Clarabel on portfolio QP, up to 40× on
  multistage SOCP, ~2× vs Gurobi, ~4× vs MOSEK; QOCO-GPU 50–70× vs own CPU. Caveats: nothing on
  small/sparse; setup 47–66 % of GPU runtime; two SOCPs lost; PSD only uniform dims ≤ 32.
  Mixed precision FP32 factor + FP64 refinement: +2×, needs static reg 1e-8 → 3.45e-4.
- No pure-Julia supernodal LDLᵀ for quasi-definite matrices exists (CHOLMOD via stdlib
  ignores Dsigns; MA57/Pardiso are binary deps; RipQP.jl sits on simplicial
  LDLFactorizations).
- Adaptive corrector count (HiPO rule): with E_s per solve, E_f per factorization, f
  refinement steps, choose k from `(1+k)(1+f/2)E_s ≈ E_f`. See §5.

---

## 3. Krylov solves on ConicIP's own Newton systems (exp-krylov)

Capture: wrapper around the real `ConicIP.kktsolver_ldl` with a prebuilt `_ldl_pattern`,
snapshotting `pat.K.nzval` after each `gen(F, F⁻ᵀ)` — the exact regularized matrix QDLDL
factorizes (δ_p = δ_e = 1e-8, δ_c = 0, dynamic eps 1e-13 / delta 2e-7). Runs match
`suite-3881ac2.csv` iteration-for-iteration. `nlift = 0` everywhere (all SOCs < dim 6, lifted
path not exercised). Krylov.jl 0.10.10, KrylovKit 0.10.4, LimitedLDLFactorizations 0.5.2.

### 3.1 Cost model: Krylov iterations one factorization buys

| instance | N | nnz(L) | factor ms | back-solve ms | **fac/bs** | κ(K₁) → κ(K_T) |
|---|---|---|---|---|---|---|
| lp-band-20000 | 80 000 | 439 927 | 8.63 | 0.936 | 9.2 | 4.8 → 9.1e12 |
| qp-band-20000 | 60 000 | 99 994 | 3.52 | 0.408 | 8.6 | 3.6 → 1.7e10 |
| issue10 | 17 130 | 31 065 | 0.68 | 0.130 | 5.2 | 31 → 1.8e16 |
| chainsing-1000-1 | 31 940 | 52 897 | 1.14 | 0.263 | 4.4 | 29 → 2.9e9 |
| nql30 | 9 080 | 111 051 | 2.58 | 0.199 | 12.9 | 3.0e3 → 2.5e17 |
| socp-sumnorms-1500 | 14 000 | 841 528 | 161.7 | 1.131 | **143** | 1.2e3 → 2.3e10 |

κ(N_t) for the normal equations is ~2 orders better and grows at the same rate. nql30 and
issue10 end above 1/ε_mach: no unpreconditioned Krylov method can reach 1e-8 there.

### 3.2 Results

- **Unpreconditioned / Jacobi MINRES:** converges only at t = 1 (33–93 its). From t ≥ 2 nothing
  reaches 1e-8 in 2000 iterations. Jacobi is consistently worse than none.
- **Stale exact factorization of K_{t−k}:** `‖K_t − K_{t−k}‖_F / ‖K_t‖_F` = 0.52–0.99 at k = 1,
  ≥ 0.95 for k ≥ 2. Plain iterative refinement with it **diverges on every instance**.
  GMRES(50) right-preconditioned, iterations to 1e-8 / 1e-12 at k = 1: lp-band-20000 14–42 /
  20–69; qp-band-20000 32–61 / 61–93; issue10 10–31 / 13–47; chainsing 11–38 / 17–50; nql30
  30–41 / 40–170; sumnorms-1500 28–40 / 40–151. k ≥ 3 hopeless except chainsing.
- **Incomplete LDLᵀ (lldl, memory ≤ 30):** a null test on 4/6 instances (reproduces exact L at
  this fill). Where fill is real: sumnorms-1500 27 its at t = 6 (10× win there), fails at
  t = 10, 12; nql30 never converges. Works early, fails exactly at the late accuracy-critical
  iterations.
- **Accounting, "factor every 2nd iteration" vs direct:** lp-band-20000 6.3× slower;
  qp-band-20000 8.8×; issue10 2.6×; chainsing 1.3×; nql30 4.5×; sumnorms-1500 0.93× (the one
  gain, 7 %, only because its scalar factorization is worth 143 back-solves; the measured fix
  there is BLAS-3, §2).

### 3.3 Verdict

Factorization reuse, the standard argument for Krylov inside an IPM, is not available: the NT
block moves by the step length in every coordinate each iteration. A preconditioner would need
≤ 3–5 back-solve-equivalents per apply, cost less than a factorization to build (sparser than L,
nonexistent at this fill), be uniform in μ while FᵀF spreads 10–14 orders, and be robust at
κ ≈ 1e16. Only structure-based saddle-point preconditioners (Rees–Greif; multigrid/DD on a
PDE-structured A) qualify, and a general conic solver cannot assume them. Revisit only where
`t_fac / t_bs > 500` (dense-column LPs, PDE SOCPs, SDP Schur complements). Literature agrees:
IPX (Schork–Gondzio 2020, in HiGHS) is geo-mean 3.67× slower than Gurobi barrier over 164 LPs
(faster on 22); HiPO (direct, 2025) beats IPX sgm 274 vs 338 s. Every Krylov win is on a named
class (huge sparse LPs with fill explosion, PDE control, operator-only A, spectrally decaying
kernels).

**Roadmap consequence:** drop Krylov/inexact Newton from Tranche 5 for LP/QP/SOCP; keep
operator inputs only.

---

## 4. Finite termination, identification, crossover (exp-activeset)

Capture: no per-iteration callback exists; iterates obtained by re-running `conicIP` with
`maxIters = k` (`verbose=false, certFallback=false`), reading `y, w, v, s` (original
coordinates). Caveat: `sol` holds the best-measure iterate; `sol.Iter` ≠ `maxIters` only in
afiro's stagnant early phase (k = 2, 3, 4). Instances: lp-band-2000/20000; Netlib afiro,
adlittle, share2b, scagr7, degen2 (LPnetlib via sparse.tamu.edu, `G = A`, cone rows
`[I_lo; −I_hi]`); a synthetic non-strictly-complementary LP; SOCPs socp-sumnorms-150, issue10.

### 4.1 LP results

| instance | n | m | it@1e-6 | tTapia | exact Ye projection at | ms/IPM-it | ms/projection | maxres 1e-6 → proj | gap 1e-6 → proj |
|---|---|---|---|---|---|---|---|---|---|
| lp-band-2000 | 2000 | 6000 | 19 | 19 | 19 (+0) | 1.58 | 2.88 | 6.3e-10 → 1.7e-16 | 1.4e-7 → 0 |
| lp-band-20000 | 20000 | 60000 | 21 | 21 | 21 (+0) | 17.6 | 55.6 | 7.2e-10 → 1.6e-16 | 1.7e-7 → 0 |
| afiro | 51 | 51 | 12 | 10 | 10 (+2) | 0.025 | 0.045 | 7.1e-14 → 1.7e-13 | 2.0e-8 → 0 |
| adlittle | 138 | 138 | 16 | 15 | 16 (+0) | 0.099 | 0.128 | 8.5e-11 → 3.6e-13 | 8.5e-8 → 0 |
| share2b | 162 | 162 | 15 | 13 | 14 (+1) | 0.086 | 0.162 | 4.3e-11 → 3.1e-13 | 1.6e-8 → 0 |
| scagr7 | 185 | 185 | 15 | 15 | 16 (−1) | 0.073 | 0.175 | 1.0e-7 → 2.2e-13 | 9.6e-8 → 0 |
| degen2 | 757 | 757 | 14 | 13 | 13 (+1) | 0.902 | 2.67 | 1.2e-8 → 4.8e-14 | 6.2e-8 → 0 |
| degen-synth | 300 | 300 | 9 | 7 | 7 (+2) | 0.330 | 1.81 | 5.0e-8 → 4.2e-14 | 4.9e-7 → 0 |

- **Speed: no.** Median saving +0.5 of 15 iterations (≈4 %); one projection (two
  quasi-definite factorizations, primal and dual, on the solver's own backend) costs 1.3–3.2
  IPM iterations. ConicIP's 1e-6 test already stops at μ ≈ 1e-7, where identification becomes
  exact; there is no tail to harvest. The `s_i > v_i` ratio indicator is exact only at/after
  the last iteration (lp-band-2000: 1463 false positives of 6000 at t = 7, 103 at t = 14).
  Tapia is 1–2 iterations sharper but not monotone (exact at 17, wrong at 18, exact at 19), so
  the projection must verify. The "partition stable 3 iterations" gate fires at t = 3 (wrong)
  or never.
- **Accuracy: yes, strongly.** From the default 1e-6 iterate, the projection gives `vᵀs = 0`
  exactly, residuals 1e-13…1e-16, objective exact to rounding. Not obtainable by tightening
  `optTol`. Argues for an optional `finite_termination` post-step costing one factorization
  pair, returning a certified-exact point independent of `optTol`.
- **Crossover to a basis: expensive where it matters.** Only the synthetic banded LPs land on a
  unique nondegenerate vertex. Real Netlib LPs converge to the analytic centre of an optimal
  face of dimension 2–45 (adlittle 16 of 138; degen-synth 45 of 300) or a vertex with 209
  redundant tight rows (degen2). A real simplex phase, not a projection. Not a next-level item.

### 4.2 SOCP cone-state identification

| instance | SOC blocks | it@1e-6 | ratio rule exact | **Tapia exact at** | interior / boundary / zero |
|---|---|---|---|---|---|
| socp-sumnorms-150 | 150 | 10 | never | **7** | 0 / 81 / 69 |
| issue10 | 1000 | 25 | never | **17** | 0 / 600 / 400 |

States are sharp (median λ₋/λ₊ on boundary blocks 3.9e-12 / 3.9e-13) and identified ≈30 %
before termination, a better margin than any LP here. Not monotone (issue10 exact at 17,
799/1000 at 18, 600/1000 at 19–22, exact from 23). The ±2-decade ratio rule fails on 96 of
600 boundary blocks (scale artefact); a scale-free rule should compare λ₋(s)/λ₊(s) with
λ₋(v)/λ₊(v). Obstacle to exploiting it: the SOC boundary face `{s : s₀ = ‖s̄‖}` is a nonlinear
manifold, so there is no linear least-squares projection; the analogue is Newton on the
reduced system (Mohammad-Nezhad–Terlaky). Realistic uses: cone-block elimination (drop
issue10's 400 provably-zero blocks from the KKT from iteration 17 on), warm starts.

### 4.3 Literature (lit-algorithms)

Ye 1992, Mehrotra–Ye 1993 (O(n³L) optimal-face point), Tapia–Zhang 1991, El-Bakry–Tapia–Zhang
1994 (indicators converge quadratically faster than the iterate, fragile under degeneracy).
Crossover: Bixby–Saltzman 1994 (CPLEX), Gurobi push/cleanup; cost 2 % of IPX runtime (basis
comes from its preconditioner) vs "over a quarter" in commercial practice (Ge–Wang–Xiong–Ye
2024). Conic: theory only (Mohammad-Nezhad–Terlaky SOCP OMS 2019 / MP 2021, SDP INFOR 2020,
rounding ORL 2019; Yıldırım JOTA 2004; AHO 1997 genericity). **No production conic solver
does finite termination**: Gurobi has no crossover for QP/QCP; MOSEK basis identification is
LP-only.

---

## 5. Iteration count (lit-algorithms)

Nothing cuts Mehrotra PC + NT by more than ~20 % on cold general LP/QP/SOCP solves.

- **Gondzio correctors** are the exception and ConicIP's negative sweep (13.5 → 11.7 sgm
  iterations, slower wall, `centralityCorrectors=K` off by default) is the textbook case.
  Colombo–Gondzio 2008 on 101 Netlib/Kennington: PCx 2114 its / 4849 solves; HOPDM 1871 / 6043;
  weighted HOPDM 1445 / 5717 (−22.8 % iterations, 3.95 solves/it). PCx enabled correctors on
  4 of 101 because its Cholesky was cheap relative to a solve. Time savings 10–30 % on large
  instances (45 % rail2586, 65 % rail4284), negative on 4. The corrector count must be chosen
  from the **measured** factorization:back-solve ratio (HOPDM rule, §2.4). Measured ratios
  here: 4–13 on five of six instances (→ k = 0/1), 143 on sumnorms-1500 (→ k > 1). Gating is
  cheap: the flop count is available from `_ldl_pattern`, and the correctors exist.
- Cartis 2009: Mehrotra-type corrector can fail to converge; remedy is weighting by stepsize²
  or multiple correctors.
- Arc-search (Yang 2014; COAP 2024 with Nesterov restart): fewer iterations on 40/51 Netlib,
  −5.4 % time, MATLAB vs MATLAB, 100-iteration cap. Ai–Zhang 2005 wide neighbourhood: theory
  only. Potra superlinearity: tail only.
- **HSD embedding is robustness, not iterations.** No ablation exists (Mészáros 2015,
  Goulart–Chen 2024). Clarabel's QP advantage (Maros–Mészáros sgm 1.0 vs ECOS 15.5, MOSEK 32.7)
  comes from keeping xᵀPx native rather than an SOC lift, which ConicIP already has. MOSEK's
  homogeneous-model whitepaper: HSD warm starts are "virtually never" available in practice.
  → Weighs on `hsd-design.md` §7 go/no-go (see §6.2).
- Hybrids: Rothberg 2026 PDHG → IPM warm start −50…70 % IPM iterations, 83 % success on 2066
  models, cold IPM best overall on his smaller set. ABIP+ 5.8× over ABIP, behind commercial at
  1e-6. cuPDLP/HPR-LP/PDCS win at 1e-4; barrier wins at 1e-8.
- Gondzio 2025 survey: practical counts "almost constant… O(log n)"; iterations track
  conditioning (Ordóñez–Freund); the survey's implementation chapters are about linear algebra,
  not correctors.

---

## 6. Candidate next-level projects, ranked

### 6.1 SDP at scale (largest capability gap; well-mapped; months)

Forced by arithmetic: an explicit symmetric-Kronecker block has (k(k+1)/2)² entries — 3.2 GB
at k = 200, **125 GB at k = 500**, 2 TB at k = 1000. The current `VecCongurance` → dense-QR path
has a hard ceiling near k ≈ 150–200 regardless of factorization speed (Clarabel's `skron!`
into the KKT block has the same ceiling). *Correction 2026-10-14:* that ceiling applies to
PSD-variable models (n = k(k+1)/2); the QR path itself never formed the Kronecker block and
was already a dense reduced Hessian in LMI form. Addressed on dev by `kktsolver_sdpschur` and
the trace-form Gram kernel (`src/schur.jl`); see the roadmap's Tranche 4 status. The reduced Schur complement
`S_ij = tr(B_i T⁻¹ B_j T⁻¹)` (n×n, 800 MB / 3.3e11 flops at n = 10⁴, independent of k) is what
SDPT3, SDPA, CSDP, SeDuMi, MOSEK do; assembly O(n k³ + n² k²) then dominates, and the
Fujisawa–Kojima–Nakata F1/F2/F3 per-column dispatch (or DSDP-style low-rank `B_j`) is what
collapses the n²k² term — so the named family must have sparse or low-rank `B_j`.

Order: **reduced Schur complement first, chordal decomposition second.** Decomposition is a
reformulation layer (can live in presolve/MOI), a no-op on dense aggregate sparsity
(theta/control/qap identical on/off in Clarabel's table), and harmful without clique merging
(mcp500-3 7.66 → 22.69 s; mcp500-4 11.63 → 51.37 s; SparseCoLO thetaG51 587.8 vs 505.3 s).
Added first it only moves the k⁴ wall to max-clique (qpG51 max clique 304; rs* up to 418). On
chordally sparse families it dominates everything: Clarabel arch0/2/4/8 60–154×
(8.17 s vs 496 s); CDCS 9×/43×/66× vs undecomposed ADMM on maxG11/maxG32/qpG11. Keep NT
(the roadmap's reduced-Hessian formula is the NT one; MOSEK's profiling shows the NT update
is the SDP cost but it is O(k³), small next to assembly). Iterative Schur solves only with
structure (Loraine's low-rank preconditioner; Toh 2004: 127 600 constraints to 1e-6 in 9.5 h).

Free item, independent of the above (roadmap Tranche 4 already notes it): `dsdc!` runs
`lyap(Y, −2X)`, an O(k³) Schur-based Sylvester solve, on `Y = mat(λ)` which `nestod_sdc` makes
diagonal (Λ = `F.S`, discarded). Closed form used by CVXOPT and Clarabel:
`Z_ij = 2 X_ij / (λ_i + λ_j)`, O(k²), allocation-free. Take `λ` as a vector at scaling time;
`mat(λ)` is diagonal only to rounding. The existing comment in `dsdc!` cites overflow scaling
as the reason for `lyap`; with diagonal Λ the Hadamard form has no such issue unless
`λ_i + λ_j` underflows.

Benchmarks: SDPLIB is retired from Mittelmann's sparse-SDP board (75 new instances, updated
2026-04-25): scaled sgm COPT 1.00, cuLoRADS 3.01 (H100), MOSEK 3.67, SDPT3 5.32, CSDP 5.40,
SDPA 7.99, SeDuMi 29.9. Sanity target on a 2026 laptop: maxG11 (m = 800, k = 800) under ~10 s,
thetaG51 (m = 6910, k = 1001) under ~150 s.

### 6.2 Warm-started conic sequences (largest documented iteration win; research content)

Skajaa–Andersen–Ye 2013 on perturbed Netlib (10 % of entries, ≤ 20 elements): geometric-mean
iterations −34–52 % primal-only, **−50–75 % primal-dual**; conic-quadratic portfolio sequence
−25 % / −60 %. Yıldırım–Wright 2002, John–Yıldırım 2008: iterations always fall for small
perturbations, only one strategy beats cold start in wall time. WARP 2026 (AC-OPF, IPOPT): full
primal-dual-barrier prediction 22.6 → 5.4 iterations; **primal-only prediction gives no
improvement**. Conic warm start is unsolved in production; Chen–Goulart–Jones 2025
(arXiv:2512.00693) give a central-path smoothing recipe prioritising centrality (tables not
visible). JuMP users re-solving parametric models are the audience.

**Design consequence for `hsd-design.md` §7:** HSD and warm starts pull against each other
(MOSEK whitepaper: needs small residuals, small μ⁰, and centrality
`min(x_j s_j, τκ) ≥ βμ⁰`, "virtually never known"). ConicIP's infeasible-start method is an
asset for this lane. A go on HSD should be weighed against it; the two are not incompatible
but HSD makes the warm start harder.

The API must take duals and a barrier parameter, not just `y`.

### 6.3 Certified-exact LP output and SOCP face identification (distinctive; cheap for LP)

- LP: optional `finite_termination` post-step (§4.1): one factorization pair, verified
  projection onto the Tapia-predicted face, returns gap exactly 0 and residuals ≈1e-15 or
  falls back to the interior point. No production conic solver offers it.
- SOCP: identification is sharp at ~70 % of the run (§4.2); the missing mathematics is a
  projection onto the nonlinear optimal face (Newton on the reduced system, Mohammad-Nezhad–
  Terlaky OMS 2019 gives quadratic convergence without strict complementarity under primal and
  dual nondegeneracy). Nearer-term use: cone-block elimination in the last third of the run.
- Not crossover: real LPs need a simplex phase (§4.1).

### 6.4 GPU direct solves (the only documented order of magnitude; not pure Julia)

CuClarabel / QOCO-GPU numbers in §2.4. Only for large structured problems; setup eats half the
GPU time; CUDA.jl + cuDSS + batched cone kernels. Record here as the honest answer to "what
beats the CPU class by 10×", not as a plan.

---

## 7. Bounded items that need no new mathematics

1. **Fill-triggered supernodal dispatch** (§2.3): after `_ldl_pattern`, compute
   `Σ colcount²` / nnz(L); above ≈40 route to a CHOLMOD path (signed pivots survive with the
   static regularization; CHOLMOD ignores Dsigns, so regularization + refinement carry the
   pivots). Expected: socp-sumnorms-1500 1.77 → ≈0.28 s, nothing else changes. Harness rule
   applies (23/23 verified, iteration counts unchanged).
2. **Adaptive corrector gate** (§5): `k` from `(1+k)(1+f/2) E_s ≈ E_f` with `E_f`, `E_s`
   measured on the first factorization/solve (or predicted from the flop count and nnz(L)).
   Turns correctors on exactly on sumnorms-type fill.
3. **Diagonal-λ closed form in `dsdc!`** (§6.1).
4. The already-ranked engineering list (roadmap "Next, in order"): (a) `maxstep_soc`/`soc_uv`
   allocation-free; (b') cheaper 4×4 residual evaluation; (c) presolve pattern plumbing; the
   outside-loop 19.5 %.

Together these are the ~1.3–1.5× on the current suite. After them, stop optimizing for this
suite: the evidence says the class is saturated.

---

## 8. How the experiments captured the solver's internals (no `src/` edits)

- **KKT matrices (exp-factor):** `kktsolver_ldl` returns an `LDLGen` whose `.gen` is the
  `solve3x3gen` closure; captured variables are fields, so `getfield(gen.gen, :K)` and
  `getfield(gen.gen, :Fact)` give the assembled quasi-definite `K` and the live QDLDL
  factorization. Pass a wrapper as `conicIP(...; kktsolver = w)` or
  `MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), w)` (`_resolve_kktsolver` passes
  non-`String` values through).
- **Per-iteration KKT sequence (exp-krylov):** `pat = ConicIP._ldl_pattern(Q, A, G, cone_dims;
  lift_min = 6)`, `gen = ConicIP.kktsolver_ldl(Q, A, G, cone_dims; pattern = pat,
  static_reg = 1e-8)`; snapshot `pat.K.nzval` after each `gen(F, F⁻ᵀ)`; record rhs/solutions
  by wrapping the returned `solve3x3`.
- **Iterates (exp-activeset):** re-run with `maxIters = k`, `certFallback = false`; check
  `sol.Iter == k`. A per-iteration callback keyword would make this O(T) instead of O(T²) and
  is worth adding if the finite-termination or warm-start lane is pursued.
- Scratch Julia environments: copy `benchmark/Project.toml`/`Manifest.toml`,
  `Pkg.develop(path = <worktree>)`, add Krylov/KrylovKit/LimitedLDLFactorizations/
  LDLFactorizations as needed. Never touch the repo's Project/Manifest.
- Netlib LPs: `https://sparse.tamu.edu/MM/LPnetlib/lp_<name>.tar.gz` (Matrix Market, with
  bounds), mapped to `G = A`, cone rows `[I_lo; −I_hi]`.

## 9. Not verified / open

- No published QDLDL-vs-supernodal A/B on IPM KKT matrices; the local measurement (§2.3) is
  the only one. faer vs CHOLMOD: nothing citable.
- MOSEK/COPT barrier internals (refinement, NE vs AS on conic, chordal decomposition):
  undocumented.
- Exact HOPDM corrector thresholds (Gondzio 1996 §4) and Chen–Goulart–Jones 2025 tables.
- FKN 1997 F1/F2/F3 cost constants quoted via a 2025 SDPT3 exposition; read the original.
- Loraine's exact speedups and preconditioner form.
- The lifted-SOC path (`nlift > 0`) was not exercised by any captured instance; the Krylov
  conclusions may differ marginally on large SOCs.
- Whether a cheaper 4×4 residual evaluation can recover the 15.7 % in refinement residuals
  without a screen (the screen was removed as unsound, see roadmap "Adversarial review").

## 10. Key references (full lists were in the scratchpad reports)

Linear algebra: Chen–Davis–Hager–Rajamanickam, CHOLMOD, ACM TOMS 2008 · Goulart–Chen,
Clarabel, MPC 2026, arXiv:2405.12762 · Zanetti–Gondzio, HiPO, arXiv:2508.04370 ·
Chen–Tse–Nobel–Goulart–Boyd, CuClarabel, ACM TOMS 2026, arXiv:2412.19027 · QOCO-GPU, MPC
2026, arXiv:2603.29197 · Schork–Gondzio, IPX, MPC 12 (2020) · Gondzio, matrix-free IPM, COAP
51 (2012) · Zhao–Xu–Huang–Chow–Xi, Nyström IP-PMM, SISC 2026 · Bergamaschi et al., NLAA 2018,
SIOPT 2015 · Gondzio–Sobral, COAP 74 (2019) · Gould–Hu–Scott, ACM TOMS 2007.

Algorithms: Colombo–Gondzio, COAP 41 (2008) · Gondzio, COAP 6 (1996) · Gondzio, "IPMs in the
year 2025", EJCO 13 (2025) · Cartis, ANM 59 (2009) · Ye, MP 57 (1992) · Mehrotra–Ye, MP 62
(1993) · El-Bakry–Tapia–Zhang, SIAM Review 36 (1994) · Bixby–Saltzman, ORL 15 (1994) ·
Ge–Wang–Xiong–Ye, IJOC 2024, arXiv:2102.09420 · Mohammad-Nezhad–Terlaky, OMS 34 (2019), MP 189
(2021), INFOR 58 (2020) · Skajaa–Andersen–Ye, MPC 5 (2013) · Yıldırım–Wright, SIOPT 12 (2002) ·
Chen–Goulart–Jones, arXiv:2512.00693 · Suri–Hilmarsson–Bose, WARP, arXiv:2605.05728 ·
Rothberg, arXiv:2603.03150, 2511.13894, 2510.24429 · Mészáros, CEJOR 23 (2015) · Yang,
arXiv:1406.4539 and COAP 2024 · Deng et al., ABIP+, arXiv:2209.01793 · MOSEK homolo
whitepaper.

SDP: Fujisawa–Kojima–Nakata, MP 79 (1997) · Tütüncü–Toh–Todd, SDPT3, MP 95 (2003) · Yamashita
et al., SDPA 7, Optimization Online 2010 · Dahl, MOSEK SDO slides, ISMP 2012 · Benson–Ye,
DSDP5, ACM TOMS 2008 · Gao–Ge–Ye, HDSDP, arXiv:2207.13862 · Vandenberghe, CVXOPT cone
programs, 2010 · Toh, SIOPT 14 (2004) · Habibi–Kočvara–Stingl, Loraine, OMS 2023 ·
Fukuda–Kojima–Murota–Nakata, SIOPT 11 (2001) · Kim–Kojima–Mevissen–Yamashita, MP 129 (2011) ·
Vandenberghe–Andersen, FnT Optim. 2015 · Zheng et al., CDCS, MP 180 (2020) ·
Garstka–Cannon–Goulart, COSMO, JOTA 2021 · Zhang, chordal conversion complexity, MP 2024 ·
Burer–Monteiro, MP 95 (2003) · cuLoRADS, arXiv:2407.15049 · HALLaR, arXiv:2401.12490 ·
Yang–Sun–Toh, SDPNAL+, MPC 7 (2015) · Mittelmann, plato.asu.edu/ftp/sparse_sdp.html
(2026-04-25).
