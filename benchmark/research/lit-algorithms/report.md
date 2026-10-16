# Algorithmic ideas that cut IPM iterations or terminate finitely (lit-algorithms, 2026-09-20)

Source PDFs/text extracts: `lit-algorithms/pdf/` (gondzio2025, colombo-gondzio, arcsearch-nesterov, say-warmstart, ipmbasis, clarabel, mosek-homolo, concurrent-crossover, partition-soco).

## (a) Verdict

Nothing in the literature reliably cuts Mehrotra-PC + NT iteration counts on a general LP/QP/SOCP set by more than ~20%. The one technique with a long reproducible record, Gondzio's multiple centrality correctors, pays off exactly when factorization is expensive relative to a back-solve. ConicIP's negative result (13.5 → 11.7 sgm iterations, slower wall) is the textbook case: PCx, with a good Ng–Peyton Cholesky, enabled correctors on only 4 of 101 Netlib/Kennington problems for the same reason (Colombo & Gondzio 2008). The actionable conclusion is to gate correctors on the factorization:back-solve ratio measured after symbolic analysis (HOPDM's rule), not to drop them.

Buckets:
- Documented practical wins, but not on a cold general solve: **warm starts** (30–75% iteration reduction on perturbed re-solves) and **crossover** (accuracy device, LP only).
- Theory-only or class-specific: Ai–Zhang wide neighbourhoods, arc-search (−5.4% time, MATLAB vs MATLAB), Potra superlinearity (tail only), ABIP+ (5.8× over ABIP, behind commercial at 1e-6), GPU first-order (1e-4 regime).
- Robustness, not iterations: **HSD embedding**. No ablation exists. Clarabel's measured advantage comes from keeping the quadratic objective out of an SOC epigraph lift, which ConicIP already has natively. MOSEK's whitepaper says HSD warm starts are effectively impossible.
- New with numbers: Rothberg 2026 PDHG→IPM warm start, −50…70% IPM iterations, 83% success on 2066 models, cold IPM still best overall on his smaller set.

Ranked recommendations: (1) adaptive correctors keyed to measured factorization:back-solve ratio; (2) a conic warm-start interface taking duals and μ (Chen–Goulart–Jones 2025 recipe; WARP 2026 shows primal-only warm start gives zero benefit); (3) LP-only crossover if users need vertices; (4) ignore arc-search, wide neighbourhoods, hybrid FOM/IPM unless targeting a class.

## (b) Table

| Idea | Claimed effect | Evidence | LP/SOCP/SDP | Effort | Refs |
|---|---|---|---|---|---|
| Gondzio weighted correctors | −22.8% iterations (1871→1445, 101 probs); 10–30% time on large, 45–65% extremes, ~0 small | Strong | LP, QP, conic | done; needs adaptive gate | Gondzio 1996; Colombo–Gondzio 2008 |
| Crossover / basis identification | exact vertex; 2% of IPX runtime avg; ~25% of runtime in commercial practice | Strong | LP only | very high | Bixby–Saltzman 1994; Megiddo 1991; Schork–Gondzio 2020; Ge et al. 2024 |
| Ye finite termination / projection | exact optimal-face point in O(n³L) | strong theory, little practice | LP | medium | Ye 1992; Mehrotra–Ye 1993 |
| Tapia / El-Bakry–Tapia–Zhang indicators | identify zeros early; indicators converge quadratically faster than iterate | strong theory; fragile under degeneracy | LP (conic analogues) | low | Tapia–Zhang 1991; EBTZ 1994 |
| Optimal-partition identification SOCP | quadratic Newton convergence under nondegeneracy | theory only | SOCP | high | Mohammad-Nezhad–Terlaky 2019, 2021; Yıldırım 2004 |
| Optimal-partition / rounding SDP | approx partition + rounding | theory only | SDP | high | Mohammad-Nezhad–Terlaky 2020; AHO 1997 |
| Arc-search | fewer iterations 40/51 Netlib; with Nesterov restart −5.4% time | weak–medium | LP, QP | medium-high | Yang 2014; COAP 2024 |
| Ai–Zhang wide neighbourhood | O(√n L) long-step | theory only | LP/LCP/SDP | high | Ai–Zhang 2005 |
| HSD embedding | certificates by convergence; no iteration claim | strong robustness, no ablation | all | high | YTM 1994; XHY 1996; Mészáros 2015; Goulart–Chen 2024 |
| Warm start (perturbed re-solve) | 34–52% (primal) / 50–75% (primal-dual) iterations | Strong | LP + conic quadratic | medium | Skajaa–Andersen–Ye 2013; Yıldırım–Wright 2002; John–Yıldırım 2008; Engau et al. 2010 |
| Conic warm start via central-path smoothing | reduces iterations/time (no numbers visible) | weak (2025 preprint) | conic | medium | Chen–Goulart–Jones 2025 |
| PDHG → IPM warm start | −50…70% IPM iterations; 83% success | medium | LP | high | Rothberg 2026 |
| ABIP / ABIP+ | 5.8× vs ABIP; behind commercial at high accuracy | medium | LP/SOCP/SDP | very high | Lin et al. 2021; Deng et al. 2024 |
| GPU first-order (cuPDLP, HPR-LP, PDCS) | competitive at 1e-4; barrier wins at 1e-8 | strong, other regime | LP, SOCP | very high | Lu–Yang 2024; Xiong et al. 2025 |
| Quasi-Newton / low-rank KKT updates | polynomial complexity preserved | thin numerics | LP, QP | high | Gondzio–Sobral 2022 |

## (c) Details

### 1. Finite termination, identification, crossover
LP theory complete: Ye 1992 (exact optimal-face solution), Mehrotra–Ye 1993 (relative-interior point of optimal face in O(n³L), same order as the LP). Tapia–Zhang 1991, El-Bakry–Tapia–Zhang 1994: indicators converge to the 0/1 pattern quadratically faster than the iterate under nondegeneracy; degeneracy is what killed them in practice.

Crossover shipped: Bixby–Saltzman 1994 (CPLEX 2.2), Gurobi push/cleanup, Megiddo 1991 correctness. Cost: IPX/HiGHS 2% of runtime average (preconditioner basis already near-final; basis after push optimal in 150/165 cases; ≤6 extra IPM iterations on 60% of models); commercial practice "over a quarter" (Maes et al. 2014 as quoted by Ge et al. 2024, who get 15–17× from structure-exploiting crossover on network/OT).

Conic: theory only. Mohammad-Nezhad–Terlaky SOCP (OMS 2019: quadratic Newton convergence to the optimum without strict complementarity under primal/dual nondegeneracy; MP 2021 parametric), SDP (INFOR 2020; rounding ORL 2019), Yıldırım JOTA 2004 unified partition. AHO 1997: nondegeneracy generic for SDP, so procedures work on random and fail on structured instances. **No production conic solver does finite termination**: Gurobi has no crossover for QP/QCP; MOSEK basis identification is LP-only.

### 2. Correctors and higher order
Colombo–Gondzio 2008, 101 problems: PCx 2114 its / 4849 solves; HOPDM 1871 / 6043 (3.23 solves/it); weighted HOPDM 1445 / 5717 (3.95 solves/it), −22.77% iterations. Corrector count chosen from factorization:back-solve cost ratio; PCx enabled correctors on 4/101. Time savings 10–30% on large, 45% rail2586, 65% rail4284, negative on 4 large instances. This is ConicIP's result restated: direction dominated by ldl_solve 40–55% + refinement ~30%, factorization 0.8× Clarabel's → low ratio → correctors lose. Gate them.

Cartis 2009: Mehrotra-type corrector can fail to converge; remedies: weight corrector by stepsize², or multiple correctors.

Arc-search (Yang): 51 Netlib, beats MPC on 40, loses 6, no CPU comparison; 2024 Nesterov-restart version: −5.4% average time on >30 s instances, 100-iteration cap. Ai–Zhang 2005: O(√n L) long-step, no implementation beating Mehrotra PC in practice found.

### 3. HSD embedding
Mészáros 2015: infeasible-start detects infeasibility by divergence; HSD by convergence, at higher dimension. Goulart–Chen: Lemma 2.2 is status determination; no ablation; Clarabel wins on QP (Maros–Mészáros sgm 1.0 vs ECOS 15.49, MOSEK 32.67; optimal control 1.0 vs 197.6/511.0; SDPLIB 1.0 vs MOSEK 9.95; Netlib 1.0 vs Gurobi 1.3, MOSEK 1.2) from native xᵀPx in the embedding rather than an SOC lift — ConicIP already has this. MOSEK homolo whitepaper: warm start of HSD needs small residuals, small μ⁰, centrality min(x_j s_j, τκ) ≥ βμ⁰, "virtually never known".

### 4. Warm starts
Skajaa–Andersen–Ye 2013 (HSD): perturbed Netlib (10% entries, ≤20 elements): geometric mean iterations −34–52% (primal-only w_p), −50–75% (primal-dual w_pd); conic-quadratic portfolio sequence −25% / −60%. Yıldırım–Wright 2002, John–Yıldırım 2008: every strategy cuts iterations for small perturbations, only one beats cold start in wall time. Engau–Anjos–Vannelli ~50% on perturbed Netlib and max-cut/TSP relaxation sequences. Colombo–Gondzio–Grothey 2011 for stochastic LP. Chen–Goulart–Jones 2025 (arXiv:2512.00693): smoothing operator mapping previous optimum onto the central path, conic-ready, parallelizable; numbers not visible. WARP 2026 (AC-OPF, IPOPT): full primal-dual-barrier prediction 22.6 → 5.4 iterations; primal-only 21.1 (no gain); earlier 30–46% claims measured against non-default flat start.

### 5. Hybrid first-order / IPM, 2020–2026
Rothberg 2026 (arXiv:2603.03150): PDHG to low accuracy, re-centre on Ruiz-scaled model, then IPM: −50% (loose) to −70% (tight) IPM iterations, >half finish ≤10 IPM iterations vs 17% cold; 35/43 Mittelmann, 83% of 2066 models; cold IPM best overall on smaller set. Rothberg 2025 "Backing PDHG into a corner": crossover push steps ×0.005 but ~4× PDHG iterations, net win ~20% of models. ABIP+ (Deng et al.): 5.8× over ABIP on 105 Netlib; behind commercial at high accuracy. cuPDLP/cuPDLPx/HPR-LP/PDCS: win at 1e-4 on huge instances; barrier wins at 1e-8. Gondzio–Sobral quasi-Newton: complexity preserved, no general-set wins. Anderson acceleration of IPM iterates: nothing found.

### 6. Theory vs practice
Gondzio 2025 survey (EJCO 13, 100105): short-step is "completely impractical"; practical methods use wide neighbourhoods and aggressive damped Newton, O(n log 1/ε) guarantee, "best behaviour known to date"; practical count "almost constant… O(log n)"; tropical geometry rules out log-barrier path following for strong polynomiality; the survey's implementation chapters are about linear algebra, barely correctors — a statement about where the field sees remaining gains. Ordóñez–Freund 2003: iterations track log C(d) of the post-processed instance (~42% variance). Potra 2004/2008: superlinear order ≥1.5 without strict complementarity, quadratic with; affects the tail only.

## (d) Not verified
1. Exact HOPDM corrector-gating thresholds (Gondzio 1996 §4).
2. Clarabel per-instance iteration counts; no HSD ablation exists.
3. Mészáros 2015 numbers (paywalled).
4. Chen–Goulart–Jones 2025 tables.
5. Xu–Hung–Ye 1996 HSD-vs-OB1.60 figures.
6. Maes et al. 2014 primary source for the "quarter of running time".
7. Any solver doing SOCP finite termination (none found; COPT docs not read in detail).
8. Anderson acceleration on IPM iterates.

## References
- Alizadeh, Haeberly, Overton. Complementarity and nondegeneracy in SDP. MP 77 (1997). doi:10.1007/BF02614432
- Alizadeh, Goldfarb. Second-order cone programming. MP 95 (2003). doi:10.1007/s10107-002-0339-5
- Ai, Zhang. O(√nL) wide-neighbourhood path-following for monotone LCP. SIOPT 16 (2005). doi:10.1137/S1052623403423114
- Bixby, Saltzman. Recovering an optimal LP basis from an interior point solution. ORL 15 (1994)
- Cartis. Some disadvantages of a Mehrotra-type corrector. Appl. Numer. Math. 59 (2009)
- Chen, Goulart, Jones. A warmstarting technique for general conic optimization in IPMs. arXiv:2512.00693 (2025)
- Colombo, Gondzio. Further development of multiple centrality correctors. COAP 41 (2008). doi:10.1007/s10589-007-9106-0
- Colombo, Gondzio, Grothey. Warm-start for large-scale stochastic LP. MP 127 (2011)
- Deng et al. Enhanced ADMM-based IPM (ABIP+). arXiv:2209.01793
- El-Bakry, Tapia, Zhang. Indicators for identifying zero variables in IPMs. SIAM Review 36 (1994). doi:10.1137/1036003
- Engau, Anjos, Vannelli. IPM warmstarts for linear and combinatorial optimization. SIOPT 20 (2010)
- Ge, Wang, Xiong, Ye. From an interior point to a corner point: smart crossover. IJOC 2024. arXiv:2102.09420
- Gondzio. Multiple centrality corrections. COAP 6 (1996). doi:10.1007/BF00249643
- Gondzio. Interior point methods in the year 2025. EJCO 13 (2025) 100105
- Gondzio, Sobral. Polynomial complexity of quasi-Newton primal-dual IPMs for LP. arXiv:2208.08771
- Goulart, Chen. Clarabel. MPC 2026. arXiv:2405.12762
- John, Yıldırım. Warm-start strategies in IPMs for LP in fixed dimension. COAP 41 (2008)
- Lin, Ma, Ye, Zhang. ADMM-based IPM for large-scale LP. arXiv:1805.12344
- Lu, Yang. cuPDLP.jl. Operations Research 2024. arXiv:2311.12180
- Megiddo. On finding primal- and dual-optimal bases. ORSA J. Comput. 3 (1991)
- Mehrotra. On the implementation of a primal-dual IPM. SIOPT 2 (1992)
- Mehrotra, Ye. Finding an interior point in the optimal face of LPs. MP 62 (1993). doi:10.1007/BF01585180
- Mészáros. Practical behavior of HSD formulations in IPMs. CEJOR 23 (2015). doi:10.1007/s10100-013-0336-1
- Mohammad-Nezhad, Terlaky. A rounding procedure for SDO. ORL 47 (2019); Quadratic convergence for SOCO without strict complementarity. OMS 34 (2019); Identification of the optimal partition for SDO. INFOR 58 (2020); Sensitivity of the optimal partition for parametric SOCO. MP 189 (2021). arXiv:1910.03684
- Ordóñez, Freund. Condition measures for linear optimization. SIOPT 14 (2003)
- Potra. Superlinearly convergent predictor-corrector for degenerate LCP. MP 100 (2004); MP 111 (2008)
- Rothberg. Backing PDHG into a corner. arXiv:2511.13894; Concurrent crossover for PDHG. arXiv:2510.24429; Hybridizing PDHG and IPMs. arXiv:2603.03150 (2026)
- Schork, Gondzio. IPM with basis preconditioning (IPX). MPC 12 (2020)
- Skajaa, Andersen, Ye. Warmstarting the HSD IPM for linear and conic quadratic problems. MPC 5 (2013). doi:10.1007/s12532-012-0046-z
- Suri, Hilmarsson, Bose. WARP: benchmark for primal-dual warm-starting of IP solvers. arXiv:2605.05728 (2026)
- Tapia, Zhang. Optimal-basis identification for IP LP algorithms. LAA 152 (1991)
- Xiong et al. PDCS. arXiv:2505.00311 (2025)
- Xu, Hung, Ye. Simplified HSD LP algorithm and implementation. Annals OR 62 (1996)
- Yang. Arc-search infeasible IPM for LP. arXiv:1406.4539; CRC book 2020; arc-search with Nesterov restart, COAP 2024, arXiv:2303.01666
- Ye. On the finite convergence of IP algorithms for LP. MP 57 (1992)
- Ye, Todd, Mizuno. O(√nL) HSD LP algorithm. MOR 19 (1994)
- Yıldırım. Unifying optimal partition approach to sensitivity analysis in conic optimization. JOTA 122 (2004)
- Yıldırım, Wright. Warm-start strategies in IPMs for LP. SIOPT 12 (2002)
- HPR-LP. arXiv:2408.12179
- MOSEK. The homogeneous and self-dual model and algorithm for LO. docs.mosek.com/whitepapers/homolo.pdf
- Gurobi reference manual: Crossover, BarQCPConvTol
