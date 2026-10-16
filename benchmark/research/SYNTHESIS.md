# ConicIP "next level" — synthesis of the 2026-09-20 research phase

Inputs: `lit-linalg/report.md`, `lit-algorithms/report.md`, `lit-sdp/report.md`, `exp-factor/report.md`,
`exp-krylov/report.md`, `exp-activeset/report.md` (all Opus subagents; experiments read-only on dev 97f1111).

## Headline

Within the architecture ConicIP now shares with the best open conic IPMs (sparse quasi-definite LDLᵀ of the
augmented system, NT scaling, Mehrotra PC), the remaining headroom on LP/QP/SOCP of the harness kind is
bounded at roughly 1.3–1.5× and is engineering, not mathematics. A qualitative jump requires changing what the
solver is asked to do: SDP at scale, warm-started sequences, high-accuracy/finite-termination output, or GPU.

## What the evidence rules out (for this problem class)

1. **Krylov KKT solves.** Measured on ConicIP's own Newton systems: consecutive KKT matrices differ by
   52–99 % in norm, so a stale factorization is useless as a preconditioner (plain refinement diverges;
   GMRES needs 10–42 iterations against a 4–13 back-solve break-even). Unpreconditioned/Jacobi MINRES
   converges only at t=1. Incomplete LDLᵀ reproduces exact L at this fill (null test) and fails at the late,
   accuracy-critical iterations where fill is real. Literature agrees: no Krylov IPM has beaten a good direct
   barrier on a broad suite (IPX 3.67× slower than Gurobi over 164 LPs; HiPO direct beats IPX). Revisit only
   where t_fac/t_bs > 500 (dense-column LPs, PDE SOCPs, SDP Schur complements).
2. **Iteration-count algorithms.** Nothing cuts Mehrotra PC + NT by >20 % on cold general solves. Arc-search
   −5 % time; wide neighbourhoods theory-only; HSD is robustness not iterations (no ablation exists);
   ABIP/PDHG hybrids are 1e-4-regime or LP-only with 83 % success.
3. **Finite termination as a speed device.** Ye projection on 8 LPs: median saving 0.5 of 15 iterations,
   projection costs 1.3–3.2 iterations. Tapia indicators are exact only at/after the last iteration on LPs.
4. **Supernodal BLAS-3 factorization as a general lever.** 20/23 harness KKT matrices have thin elimination
   trees (0 % of flops in chains ≥16, CHOLMOD itself picks simplicial). Aggregate −13 % sgm, 95 % of the
   saving from one instance (socp-sumnorms-1500: QDLDL 142 ms vs CHOLMOD 10.5 ms at 6 threads).

## Where the remaining 1.3–1.5× on the current suite sits (exp-factor, phase data 3881ac2)

| block | share of total 8.94 s | lever |
|---|---|---|
| back-solves + refinement | 34 % | fewer 4×4 residual evaluations (cheaper evaluation, not a screen: see roadmap (b')); solves/iteration policy |
| outside the loop (frontend, presolve, equilibrate, setup, init) | 19.5 % | setup 2.06× Clarabel's (QDLDL symbolic memory, AMD) |
| residuals + rhs + line search | 8.6 % | SOC allocation items (a) in roadmap |
| factorization | ≈33 % | fill-triggered supernodal dispatch; only pays on sumnorms-type fill |

## Cheap, evidence-backed items (each bounded, no new math)

- **Fill-triggered supernodal dispatch**: `flops/nnz(L) > ~40` (CHOLMOD's `supernodal_switch`; Clarabel-Rust uses the
  same rule with faer) → CHOLMOD `ldlt`/Cholesky-proxy path. Turns socp-sumnorms-1500 1.77 s → ~0.28 s; no effect
  elsewhere. Dsigns are ignored by CHOLMOD; regularization/refinement must carry the pivots.
- **Adaptive Gondzio corrector count** from the measured fac/bs ratio (HOPDM rule `(1+k)(1+f/2)E_s ≈ E_f`).
  Measured ratios: 4–13 on most harness instances (→ k=0/1), 143 on sumnorms-1500 (→ k>1). Explains the
  earlier negative sweep; correctors are already implemented.
- **Diagonal-λ closed form in `dsdc!`** (`Z_ij = 2X_ij/(λ_i+λ_j)`, O(k²) vs the O(k³) Sylvester solve, twice per
  iteration). Already noted in roadmap Tranche 4; `nestod_sdc` computes Λ and discards it.

## Candidate "next level" projects, ranked

1. **SDP at scale** (largest capability gap; well-mapped). Replace the explicit congruence (skron block is
   (k(k+1)/2)² entries: 3.2 GB at k=200, 125 GB at k=500) with the reduced Schur complement
   `S_ij = tr(B_i T⁻¹ B_j T⁻¹)` assembled with Fujisawa–Kojima–Nakata F1/F2/F3 dispatch and a dense Cholesky of the
   n×n reduced Hessian (800 MB / 3.3e11 flops at n=1e4, independent of k). Chordal decomposition second, with
   clique merging (naive decomposition is measurably worse: mcp500-3 7.66→22.69 s), 60–154× on chordally
   sparse families (Clarabel arch*). Keep NT. Iterative Schur only with structure (Loraine). Months.
2. **Warm-started conic sequences** (largest documented iteration win: 50–75 % primal-dual on perturbed
   Netlib, ~60 % on conic-quadratic sequences; primal-only gives zero, WARP 2026). Research component: conic
   warm start is unsolved in production (Chen–Goulart–Jones 2025 central-path smoothing is a preprint).
   Design consequence for hsd-design §7: HSD and warm starts pull against each other (MOSEK whitepaper), so
   ConicIP's infeasible-start method is an *asset* here; a go on HSD should be weighed against this.
3. **High-accuracy / finite-termination mode** (distinctive, cheap for LP, research for SOCP). Measured: Ye
   projection from the default 1e-6 iterate gives gap identically 0, residuals 1e-13…1e-16, at the cost of
   ~2 factorizations. No production conic solver offers this (Gurobi no crossover for QP/QCP, MOSEK basis
   identification LP-only). SOCP: Tapia identifies every cone state exactly at ~70 % of the run (t=7/10,
   17/25), boundary blocks with λ₋/λ₊ ≈ 1e-12; the missing piece is a projection onto the nonlinear optimal
   face (Newton on the reduced system, Mohammad-Nezhad–Terlaky) — a real research lane. Real Netlib LPs land in
   optimal faces of dimension 2–45, so "crossover" would need a simplex phase; the projection does not.
4. **GPU direct solve (cuDSS)**: the only documented order of magnitude on the same problem class
   (CuClarabel 10× vs multithreaded Rust Clarabel, 40× on multistage SOCP; QOCO-GPU 50–70×), but only for
   large structured problems, setup 47–66 % of GPU time, very high dependency cost. Not a pure-Julia project.

## Decisions this suggests for the maintainer

- Drop Krylov/inexact-Newton from the roadmap for LP/QP/SOCP (Tranche 5 stays operator-inputs only).
- Reframe the "1.6× / parity" chase: remaining suite headroom is ~30 % engineering; stop optimizing for it
  once items (a)/(b') land.
- Choose between SDP (capability) and warm start + finite termination (distinctiveness, research content) as
  the next multi-month lane; both are compatible with keeping the classic infeasible-start method.
