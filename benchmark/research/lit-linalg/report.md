# Literature survey: KKT linear algebra in conic IPMs (lit-linalg, 2026-09-20)

## (a) Verdict

Only two directions have documented, reproducible, general-purpose wins over the baseline (one sparse quasi-definite LDLᵀ of the augmented system per iteration, scalar simplicial kernel, single thread, static+dynamic regularization, iterative refinement):

(i) replacing the scalar kernel with a **supernodal/multifrontal BLAS-3 factorization, switched on by a fill heuristic**: nothing below `flops/nnz(L) ≈ 30–45`, roughly **3–10× in factorization throughput** above it (CHOLMOD Table I), plus **~2× more from 8-thread tree/node parallelism** (HiPO in HiGHS);

(ii) **moving the whole iteration to a GPU direct solver (cuDSS)**: **4–40×** on large dense-ish conic problems, **50–70×** at 10⁷–10⁸ KKT nonzeros, a further **~2× from FP32 factorization + FP64 refinement**; nothing (often a loss) on small or very sparse problems.

Normal equations vs augmented system is a per-instance choice (HiPO: NE on ~75% of Mittelmann LPs, AS on ~25%; AS is the only one that fits in memory on hard energy LPs) and largely moot for a conic solver because NT scaling does not collapse to `AD⁻¹Aᵀ`.

**No Krylov-based IPM has been shown to beat a good direct barrier on a broad suite.** IPX (Schork–Gondzio, in HiGHS) is a geometric-mean **3.67× slower than Gurobi barrier over 164 LPs**, winning on 22; its direct successor HiPO beats it (sgm 274 s vs 338 s). Quasi-Newton / factorization-reuse schemes cut factorization *counts* in research codes with no published wall-clock suite win.

The cheapest real lever needs no dependency: **size the corrector count from the measured solve/factorization cost ratio** (HiPO's rule `(1+k)(1+f/2)E_s ≈ E_f α_F`).

## (b) Table

| Direction | Best documented speedup | Class | Cost / deps for pure Julia | Refs |
|---|---|---|---|---|
| Supernodal BLAS-3 LDLᵀ vs simplicial | 3.3–10× factor GFlop/s (0.20–0.41 → 1.34–3.93, 1 thread); zero or negative below flops/nnz(L)≈30–45 | high-fill KKT: 3-D/PDE, dense-ish SOCP, SDP | CHOLMOD in stdlib (`ldlt`) but ignores Dsigns; MA57/Pardiso binary deps; **no pure-Julia supernodal LDLᵀ exists**; LDLFactorizations.jl is simplicial | 1,2,3,4,14 |
| Runtime simplicial↔supernodal switch on flops/nnz(L) | takes max of the two; threshold 40 platform-stable | all | cheap: flop count already in symbolic phase | 1,2 |
| Threaded multifrontal (tree+node) | ~2× typical, >4× a few, 8 cores, 46 Mittelmann LPs | large LPs, deep etrees | high: needs multifrontal code | 3 |
| Normal equations vs augmented | instance-dependent; NE 75% / AS 25%; AS only feasible on PyPSA (MBs vs >128 GB) | LP with m≪n; AS for QP/dense cols/SOCP | moderate, low value for conic | 3,11 |
| Dense-column Schur complement | removes catastrophic blow-up; MOSEK, Gurobi | LP/QP few dense cols | only if forming NE | 11,12,13 |
| GPU cuDSS | 10× vs Rust Clarabel+faer (portfolio QP); 40× multistage SOCP; ~2× Gurobi, ~4× MOSEK; 50–70× (QOCO-GPU) | large, structured | very high; setup 47–66% of GPU runtime on sparse | 5,6,7 |
| Mixed precision FP32 factor + FP64 refinement | +2× (598→334 s, n=25 000) | well-conditioned, large, GPU | moderate; static reg 1e-8 → 3.45e-4 | 5,15 |
| Krylov + basis preconditioner (IPX) | **3.67× slower** geo-mean vs Gurobi over 164 LPs; faster on 22/164 | very large LPs with fill explosion | very high (~10k lines) | 8 |
| Matrix-free partial Cholesky | PCG 5–20 its, rank 2–50; no wall-clock win vs direct | operator-only A | high | 9 |
| Randomized Nyström (Nys-IP-PMM) | 2× vs CG-IPM, 3× vs partial Cholesky | fast spectral decay | moderate; baseline iterative | 10 |
| Inner-iteration Krylov (Cui et al.) | "more robust than SeDuMi/SDPT3, same CPU" 138 LPs | degenerate LPs | high; baselines not SOTA | 16 |
| Quasi-Newton / low-rank KKT factor updates | fewer factorizations; no wall-clock suite result | QP, matrix-free | moderate | 17,18,19 |
| Adaptive corrector count from solve/factor ratio | trades (1+k)(1+f/2) solves vs 1 factorization | all | **low, scalar heuristic** | 3,20,21 |

## (c) Details

### 1. Supernodal / multifrontal BLAS-3, and what solvers use
CHOLMOD Table I [1]: up-looking simplicial (twin of QDLDL) vs supernodal peak GFlop/s: 0.38 vs 3.76 (P4/Goto), 0.20 vs 1.77–1.80 (P4M), 0.41 vs 1.34–1.47 (Core Duo 1 thread; 2.47 at 2), 0.38 vs 2.56–3.93 (Opteron 1 thread; 6.65 at 2), 0.06 vs 0.43 (UltraSPARC). Asymptotic factor **3.3–10×**, realized only with enough dense work per nonzero. Measured crossover flops/nnz(L): **30–45 one thread, 80–120 two threads**. CHOLMOD default `supernodal_switch` = 40.

Clarabel copies this: Rust build auto-chooses QDLDL vs faer supernodal by flops/nnz(L) ≈ 40 [2]. Clarabel.jl exposes `direct_solve_method = :auto | :qdldl | :cholmod | :ma57 | :mkl | :panua` [4]; its CHOLMOD backend ignores `Dsigns` (generic `ldlt(Symmetric(K); check=false)` with `ldlt!` reuse). The Clarabel paper gives **no QDLDL-vs-faer A/B timings**.

ECOS: Davis SparseLDL + AMD, dynamic reg, refinement, single-thread C [14]. HiGHS: IPX (Krylov) and since 2025 **HiPO**, multifrontal supernodal LDLᵀ, METIS ND, dgemm/dtrsm, modified Bunch–Kaufman for regularized quasi-definite pivots, tree+node parallelism [3]. Gurobi: supernodal Cholesky of scaled AAᵀ, `BarOrder` AMD vs ND, all cores [12]. MOSEK: homogeneous algorithm, Andersen modified Schur complement for dense columns, parallel linear algebra [11,13]. Ipopt: MA27/57/86/97, MUMPS, PARDISO, SPRAL, WSMP; Tasseff et al. find serial MA57/MUMPS best overall [22].

Julia: `SparseArrays.ldlt` → CHOLMOD (supernodal, BLAS-parallel only); LDLFactorizations.jl (simplicial); HSL.jl (MA57/MA97, licence); Pardiso.jl; MUMPS.jl. **No pure-Julia supernodal LDLᵀ for quasi-definite matrices.** RipQP.jl [31] also sits on simplicial LDLFactorizations.

### 2. Normal equations vs augmented
HiPO [3]: symbolic AS first, abandon NE if fill exceeds; dense columns force AS. 49 Mittelmann LPs: ~25% AS, ~75% NE. 45 PyPSA instances: all AS; NE factor >128 GB vs "a few hundred MB"; growth exponent n^1.85 (IPX) → n^0.97 (HiPO). Dense columns: Andersen modified Schur complement [13], MOSEK [11], Gurobi [12]. For conic solvers NT scaling of SOC/PSD blocks does not give a clean AD⁻¹Aᵀ, so Clarabel, ECOS, ConicIP stay augmented.

### 3. Threading and GPU
HiPO Fig. 7: 8-core factorization speedup mostly ≈2×, a few >4× over 46 Mittelmann LPs; OpenBLAS run serially for bit-compatibility, parallelism from the tree; warns profile changes completely with Apple BLAS on Mac [3].

CuClarabel [5,6]: cuDSS LDLᵀ + batched cone ops; RTX 4090 vs Xeon w9-3475X: ~10× over Rust Clarabel+multithreaded faer on portfolio QP, up to 40× multistage portfolio SOCP, ~2× vs Gurobi, ~4× vs MOSEK, 1.5–4× vs MOSEK on SDP; 25 000-var QP 11.8 s vs 549 s. Caveats: no benefit small; setup 47–66% of GPU runtime on sparse; two SOCPs lost; SDP only uniform PSD dims ≤ 32. QOCO-GPU [7]: same cuDSS design, 50–70× over own CPU, "only for sufficiently large problems". Mixed precision: FP32 factor + FP64 refinement up to +2× (598→334 s) with static reg 1e-8 → √ε ≈ 3.45e-4, off by default (may fail on ill-conditioned K) [5].

### 4. Krylov / inexact-Newton IPMs — decisive evidence
IPX [8]: conjugate-residual on NE preconditioned by a maximum-volume basis maintained with simplex-type pivots and Forrest–Tomlin updates. Table 4 vs Gurobi 7.0 barrier:

| subset | n | geo-mean IPX/Gurobi | IPX faster | Gurobi faster |
|---|---|---|---|---|
| >1 s | 164 | 3.67 | 22 | 142 |
| >10 s | 89 | 3.94 | 16 | 73 |
| >100 s | 41 | 6.29 | 6 | 35 |

Authors: "considerably slower than a state-of-the-art Cholesky factorization on average, faster for relevant problems". Runtime: preconditioner 22%, iterative solver 45%, crossover 2%. Wins at m ≥ 100 000 with fill explosion. IPX failed 5 instances (LU fill). HiPO (2025) beats IPX on Mittelmann sgm 274 vs 338 s [3]. IPX side benefit: hands crossover a basis, can reach high accuracy by eliminating converged variables.

Others: Bergamaschi–Gondzio–Zilli [23]; Oliveira–Sorensen splitting preconditioner [24]; Bocanegra–Campos–Oliveira hybrid [25]; Rees–Greif [26]; Chai–Toh [27]; Gondzio matrix-free partial Cholesky rank 2–50, PCG 5–15 steps [9]; Pougkakiotis–Gondzio–Pearson IP-PMM preconditioners [28]; Nys-IP-PMM 2× vs CG-IPM, 3× vs partial Cholesky on portfolio [10]; Cui et al. robustness on 138 LPs vs SeDuMi/SDPT3/PDCO [16]; SDSL-Solver distributed 2026 [29].

**No published Krylov-based IPM has beaten a good direct solver on a broad LP/QP/SOCP suite.** Wins are class-specific: huge sparse LPs (IPX), PDE-constrained control, operator-only A, spectrally decaying kernels/portfolios.

### 5. Factorization reuse
Gondzio–Sobral quasi-Newton IPM for QP [17]: fewer factorizations, no wall-clock vs tuned direct. Bergamaschi et al. BFGS-like/low-rank updates of constraint preconditioners across IPM iterations [18,19]: cleanest "factor once, reuse as preconditioner" formalization; no broad-suite win. Risk: Krylov iterations from a stale preconditioner cost more back-solves than the saved factorization.

### 6. Fewer back-solves
Mehrotra PC = 1 factor + 2 solves [20]; Gondzio correctors +1 solve each [21]. HiPO [3]: with E_s per solve (∝ 2 nnz(L)), E_f factorization, f refinement steps (1–2), α_F parallel efficiency, choose k from `(1+k)(1+f/2)E_s ≈ E_f α_F`, floor 1; total 2–4 solves/iteration. Explains ConicIP's finding that K=2 correctors cost time on a low-fill suite: E_f/E_s small there. Refinement is standard everywhere (ECOS, Clarabel reltol 1e-13/abstol 1e-12/max 10/stop_ratio 5, static 1e-8, dynamic eps 1e-13 delta 2e-7; CuClarabel ≤10 steps to 1e-12 ℓ∞; HiPO 1–2 steps, static 1e-10–1e-12).

## (d) Not verified
- No published QDLDL-vs-supernodal timing on IPM KKT matrices; CHOLMOD numbers are SPD SuiteSparse peak rates. Must be measured locally (the socp-sumnorms-1500 point, QDLDL 142 ms vs UMFPACK 115 ms at 6 threads, suggests fill near the crossover).
- faer-rs vs CHOLMOD head-to-head: none.
- MOSEK internals (refinement, NE vs AS on conic), COPT barrier linear algebra: undocumented.
- Gurobi dense-column threshold/kernel: qualitative only.
- Tasseff et al. per-thread curves; Pougkakiotis et al. and Cui et al. tables: from abstracts/secondary.
- Nothing quantifies parallelizing the non-factorization part of a conic iteration (residuals, NT scaling, line search), which is 1.67× Clarabel's cost in ConicIP.

## References
1. Chen, Davis, Hager, Rajamanickam. Algorithm 887: CHOLMOD. ACM TOMS 35(3), 2008. doi:10.1145/1391989.1391995
2. Goulart, Chen. Clarabel: an interior-point solver for conic programs with quadratic objectives. Math. Program. Comput. 2026. doi:10.1007/s12532-026-00320-7, arXiv:2405.12762
3. Zanetti, Gondzio. A factorisation-based regularised interior point method using the augmented system (HiPO). arXiv:2508.04370, 2025
4. Clarabel docs, Linear System Solvers. https://clarabel.org/stable/user_guide_linsolvers/ ; Clarabel.jl src/kktsolvers/direct-ldl/directldl_cholmod.jl
5. Chen, Tse, Nobel, Goulart, Boyd. CuClarabel: GPU acceleration for a conic optimization solver. ACM TOMS 2026. doi:10.1145/3815420, arXiv:2412.19027
6. https://github.com/cvxgrp/CuClarabel
7. QOCO-GPU. Math. Program. Comput. 2026. doi:10.1007/s12532-026-00338-x, arXiv:2603.29197
8. Schork, Gondzio. Implementation of an interior point method with basis preconditioning. MPC 12, 603–635, 2020. doi:10.1007/s12532-020-00181-8
9. Gondzio. Matrix-free interior point method. COAP 51, 457–480, 2012. doi:10.1007/s10589-010-9361-3
10. Zhao, Xu, Huang, Chow, Xi. Randomized Nyström preconditioned IP-PMM. SISC 2026. doi:10.1137/24M1654968, arXiv:2404.14524
11. Andersen, Andersen. The MOSEK interior point optimizer for LP. High Performance Optimization, Springer 1999. doi:10.1007/978-1-4757-3216-0_8
12. Gurobi Reference Manual, Barrier logging / parameters. https://docs.gurobi.com/projects/optimizer/en/current/concepts/logging/barrier.html
13. K. D. Andersen. A modified Schur-complement method for handling dense columns in IPMs for LP. ACM TOMS 22(3), 1996. doi:10.1145/232826.232937
14. Domahidi, Chu, Boyd. ECOS: an SOCP solver for embedded systems. ECC 2013. doi:10.23919/ECC.2013.6669541
15. Higham, Mary. Mixed precision algorithms in numerical linear algebra. Acta Numerica 31, 2022. doi:10.1017/S0962492922000022
16. Cui, Morikuni, Tsuchiya, Hayami. IPMs for LP based on Krylov solvers with inner-iteration preconditioning. COAP 74, 2019. doi:10.1007/s10589-019-00103-y
17. Gondzio, Sobral. Quasi-Newton approaches to IPMs for quadratic problems. COAP 74, 93–120, 2019. doi:10.1007/s10589-019-00102-z
18. Bergamaschi, De Simone, di Serafino, Martínez. BFGS-like updates of constraint preconditioners for KKT sequences in QP. NLAA 25(5), 2018. doi:10.1002/nla.2144
19. Same authors. Updating constraint preconditioners for KKT systems via low-rank corrections. SIOPT 25(3), 2015. arXiv:1312.0047
20. Mehrotra. On the implementation of a primal-dual interior point method. SIOPT 2(4), 1992. doi:10.1137/0802028
21. Gondzio. Multiple centrality corrections in a primal-dual method for LP. COAP 6(2), 1996. doi:10.1007/BF00249643
22. Tasseff, Coffrin, Wächter, Laird. Exploring benefits of linear solver parallelism on modern nonlinear optimization applications. arXiv:1909.08104, 2019
23. Bergamaschi, Gondzio, Zilli. Preconditioning indefinite systems in IPMs. COAP 28, 2004
24. Oliveira, Sorensen. A new class of preconditioners for large-scale linear systems from IPMs for LP. LAA 394, 2005
25. Bocanegra, Campos, Oliveira. Hybrid preconditioner for IPM linear systems. COAP 36, 2007
26. Rees, Greif. A preconditioner for linear systems arising from IPMs. SISC 29(5), 2007
27. Chai, Toh. Preconditioning and iterative solution of symmetric indefinite systems from IPMs for LP. COAP 36, 2007
28. Gondzio, Pougkakiotis, Pearson. General-purpose preconditioning for regularized IPMs. COAP 83, 2022. arXiv:2107.06822
29. Yang, Wang, Cheng, Tan. SDSL-Solver: scalable distributed sparse linear solvers for large-scale IPMs. arXiv:2604.23979, 2026
30. LDLFactorizations.jl, HSL.jl, Pardiso.jl (JuliaSmoothOptimizers / JuliaSparse)
31. Leconte, Orban. RipQP: multi-precision regularized predictor-corrector for convex QP. GERAD 2024
32. Gould, Hu, Scott. A numerical evaluation of sparse direct solvers for large sparse symmetric systems. ACM TOMS 33(2), 2007 (CHOLMOD fastest or within 10% on 73/87; worst outliers are IPM LP matrices with dense rows/cols, dominated by AMD time)
