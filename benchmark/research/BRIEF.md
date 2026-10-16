# Shared brief for the ConicIP "next level" research phase (2026-09-20)

## Hard rules
- RESEARCH PHASE. Do NOT edit any file under the git repository (`/Users/mpf/projects/Software/ConicIP.jl` or any `.worktrees/*`). No commits, no branches, no `git stash`.
- All scripts, CSVs, notes go in the scratchpad directory:
  `/private/tmp/claude-502/-Users-mpf-projects-Software-ConicIP-jl/8d6f2e95-bb86-4902-8603-0c2255592e24/scratchpad/<your-agent-name>/`
- Code under test: the `dev` branch, checked out in the worktree
  `/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap` (HEAD 97f1111 = v0.6.0 + private benchmark harness). Run Julia from that directory.
- Julia 1.12.7. Two environments:
  - `julia --project=.` (the package itself)
  - `julia --project=benchmark` (has Clarabel.jl 0.11, ECOS.jl, and ConicIP dev'd in; Manifest present). Use this when you need Clarabel or the harness helpers.
- Wrap every Julia run in `timeout` (e.g. `timeout 900 julia ...`). Keep total machine time per agent under ~40 minutes. Machine: Apple M3 Pro, use 1 Julia thread unless the experiment is about threads.
- Never mention other solvers by name in anything that could end up in public repo text; here in the scratchpad it is fine.

## What ConicIP is
Pure-Julia primal-dual interior-point method (Mehrotra predictor-corrector, Nesterov–Todd scaling) for
  min ½yᵀQy + cᵀy  s.t.  Gy = d (p equalities),  Ay − s = b, s ∈ K (m cone rows),  K = product of R₊ ("R"), second-order ("Q"), PSD ("S") cones.
Newton system solved per iteration through an LDLᵀ factorization (QDLDL.jl backend, `src/kktsolver_ldl.jl`) of the quasi-definite matrix
  [Q+δI  Gᵀ  −Aᵀ; G  −δI  0; −A  0  −(FᵀF+δI)]  with SOC blocks "lifted" (diag + rank-2 → sparse), static+dynamic regularization, iterative refinement against the unregularized system (`refine!`), Ruiz equilibration, singleton presolve, original-coordinate termination, certificate screens + fallback solves for infeasibility (no homogeneous embedding yet; designed in `hsd-design.md`, awaiting go/no-go). Gondzio multiple centrality correctors exist (`centralityCorrectors=K`) but are off by default: they cut iterations 13.5→11.7 sgm but cost more time on this set.
SDP cones go through a dense QR path (`kktsolver_qr`) that forms the congruence explicitly: O(k⁴) memory, O(k⁵) work; SDP is "experimental" and far from scale.

Key source files (in the worktree): `src/ConicIP.jl` (main loop, `_conicIP`, `solve4x4gen` at ~line 1567; the direct API accepts `kktsolver = f(Q,A,G,cone_dims) -> solve3x3gen(F, F⁻ᵀ) -> solve3x3(rhs...)`, so a *wrapper* around `ConicIP.kktsolver_ldl` can capture every KKT matrix/rhs/solution without touching src), `src/kktsolver_ldl.jl` (`kktsolver_ldl`, `_ldl_pattern`, `LDLDiagnostics`, `cached_kktsolver_ldl`), `src/kktsolvers.jl` (dense QR, UMFPACK sparse, 2x2, `choose_kktsolver`), `src/correctors.jl`, `src/equilibrate.jl`, `src/timing.jl` (`timing=` keyword gives per-phase timers), `docs/src/guides/kkt_solvers.md` (callback contract).
Harness (private, in worktree): `benchmark/suite.jl` (23 instances: lp-band-{2000,20000,200000}, qp-band-{...}, socp-sumnorms-{150,1500}, issue10 SOCP, CBLIB nb/nql30/qssp30/sched_50_50_scaled/chainsing-1000-1, Maros–Mészáros hs21/hs35/hs76/qafiro/dual1/primal1/qpcblend/sambal/cvxqp1_s), `benchmark/moi_model.jl`, `benchmark/baselines.jl`, `benchmark/phases.jl`. Instance construction functions live in `suite.jl`; downloads are cached in `benchmark/.cache/` (CBLIB `.cbf.gz`, more CBLIB files than the 5 used are cached). `benchmark/RESUME.md` documents the commands.

## Where we stand (measured 2026-09-15/16, dev 3881ac2, M3 Pro, 1 thread)
- 23/23 instances solved and independently verified at 1e-6 for both ConicIP and Clarabel. Totals 8.94 s vs 8.78 s; shifted-geometric-mean wall ratio 1.16 (ConicIP/Clarabel). ConicIP is faster on the large banded LP/QP (0.90–0.95), at parity on sum-of-norms SOCPs, 1.5–2.2× slower on the CBLIB SOCPs (remaining allocations in `maxstep_soc`/`soc_uv`), 4× on one 1-ms instance.
- Iteration counts are within −2…+3 of Clarabel's except sched_50_50_scaled (+10).
- Inside the loop, factorization (QDLDL) is ~0.8× Clarabel's; back-solves + refinement dominate `direction` (ldl_solve 40–55% of direction, refinement residual evaluation ~30%); residual/rhs/line-search overhead is 1.67× Clarabel's.
- The QDLDL factor is scalar (no BLAS-3): on socp-sumnorms-1500 (fill 24×, nnz(L)=0.84M) QDLDL 142 ms vs UMFPACK 115 ms with 6 BLAS threads. Factorization is 95% of the loop on that instance.
- Existing data files: `benchmark/results/phase-conicip-3881ac2.csv`, `phase-clarabel-3881ac2.csv`, `compare-phases-3881ac2.md`, `suite-3881ac2.csv` (columns include kkt_nnz, lu_nnz, iters, kkt_solves, t_total ...). Roadmap and HSD design copied to the scratchpad root: `roadmap.md`, `hsd-design.md`, `suite.jl`.

## The question this phase answers
ConicIP is now roughly at parity with the best open conic IPMs of the same architecture (sparse LDLᵀ of the augmented system, NT scaling, Mehrotra PC). What would take it to the *next level*, i.e. clearly beyond that class? Candidate directions the maintainer named:
1. specialized linear algebra for the Newton solves (supernodal/BLAS-3, threading, normal equations vs augmented, dense-column handling, GPU, mixed precision, factorization reuse);
2. iterative (Krylov) solves, which need preconditioning;
3. finite-termination devices: active-set identification, crossover, optimal-partition identification (LP; also what exists for SOCP/SDP);
4. new mathematical ideas that cut iteration counts or the per-iteration cost (higher-order/arc-search, quasi-Newton scaling updates, warm starts, homogeneous embedding, hybrid first-order/IPM, etc.);
5. SDP at scale (reduced Schur complement, chordal decomposition, low-rank).
Report evidence, not opinions: cite papers/solvers precisely, and for experiments give tables with the exact commands to reproduce.
