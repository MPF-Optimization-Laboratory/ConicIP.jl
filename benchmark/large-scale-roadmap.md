# ConicIP.jl large-scale roadmap

**Date:** 2026-09-06 (revised the same day after an independent review, then
rebased onto v0.4.0)
**Baseline commit:** `119fb5e` (master, v0.4.0)
**Status:** planning document. No tranche has started. Update the Status line of a
tranche when work on it lands, and keep the `file:line` references current.

`docs/src/index.md:82` positions ConicIP as a solver for **moderate-size**
LP/QP/SOCP/SDP problems and sends large-scale users to COSMO, Hypatia, SCS, and ECOS. This document
records what would have to change to drop that qualifier honestly, and in what order.

Claims below marked **verified** were checked against the code at the baseline commit.
Timing predictions are hypotheses until the Tranche 0 harness measures them.

## What "large-scale" means here

For a primal-dual interior-point method, large-scale means the Clarabel/ECOS class:
sparse LP/QP/SOCP with 10⁵–10⁶ variables and 10⁶–10⁷ nonzeros, memory proportional to
nnz plus factorization fill, and each iteration costing one sparse numeric
factorization. It does not mean the SCS class (first-order methods at 10⁷+ nonzeros),
which is a different algorithm with different accuracy trade-offs. Sparse factorization
fill remains a fundamental limit in either case; an LDLᵀ solver removes an implementation
limit, not that one.

### Baseline

From the issue #10 close-out (PR #28), on that issue's 6010-variable SOCP
(`benchmark/issue10.jl` downloads and reproduces it):

| configuration | time | iterations | status |
|---|---|---|---|
| old default (dense QR) | ~11 s/iteration, ≳5 min/solve | — | did not converge usefully |
| current default (auto → sparse), with preprocessing | 0.78 s | 29 | Optimal |
| ECOS (yardstick) | 0.17 s | 23 | Optimal |

The two timings were taken under different conditions and are not a controlled
comparison. Do not read the gap as purely per-iteration constants or purely iteration
count: scaling, regularization, and solve accuracy all change step lengths in finite
precision.

## Diagnosis: what caps scale today, ranked by impact

### 1. KKT linear algebra

- **No LDLᵀ / quasi-definite path.** `kktsolver_sparse` (`src/kktsolvers.jl:255-342`)
  factors the KKT matrix with UMFPACK's unsymmetric LU. UMFPACK does compute a
  fill-reducing ordering and does pick a symmetric strategy for symmetric patterns, and
  the code already reuses the symbolic analysis via `lu!` (`src/kktsolvers.jl:273-278`).
  So the cost of LU over LDLᵀ is a constant factor (roughly 2× on comparable patterns is
  the usual heuristic, unmeasured here), not an asymptotic one. What LU cannot give is
  prescribed pivot signs, dynamic regularization on the fly, and a quasi-definite
  guarantee. **Verified.**
- **The assembled system is not symmetric as written.** `src/kktsolvers.jl:325-329` puts
  `−Aᵀ` above the diagonal and `+A` below it, with `+FᵀF` in the (3,3) block. An LDLᵀ
  needs the third block row and its right-hand side negated (see Tranche 1). **Verified.**
- **Regularization touches only Q.** `staticReg` perturbs the (1,1) block
  (`src/ConicIP.jl:944-945`). If `Gᵀw = 0` has a nonzero solution, the null vector
  `(0, w, 0)` survives any such perturbation, so a rank-deficient G is fatal without the
  presolve. A custom callback could assemble a regularized system today; the built-in
  solvers hardcode a zero equality block. **Verified.**
- **Dense QR is the default for `n+m+p < 1000`, for every SDP, and for any problem with
  more than 10 structural nonzeros per column** (`src/kktsolvers.jl:28-42`). The last
  rule has no size guard: a million-variable problem with 11 nnz/col is routed to a
  solver that materializes an n×n dense orthogonal factor (`src/kktsolvers.jl:85`). That
  is a hazard, not a tuning question. The per-iteration cost is a dense (n−p)² Cholesky
  plus m×(n−p) dense products (`src/kktsolvers.jl:99-106`). **Verified.** The rule also
  misfires well below the memory guard: `benchmark/suite.jl`'s banded LP at n=2000
  (13 nnz/col once the box-constraint identity rows are counted, factor fill 1.8× the
  KKT nonzeros) takes about 5 s and 6 GiB of allocation through dense QR; forcing
  `kktsolver_sparse` on the same instance takes 0.27 s at the same iteration count, a
  23× difference. **Measured.**
- **`kktsolver_2x2`** re-runs symbolic and numeric `lu` and re-forms an explicit Schur
  complement every iteration (`src/kktsolvers.jl:361-367`). No caching. **Verified.**
- **`lift` is not reusable as-is for a quasi-definite system.** It inserts `−inv(D)` for
  the low-rank part (`src/kktsolvers.jl:158-161`) and selects entries by the sparsity of
  `D` while storing entries of `inv(D)`. The SOC scaling `FᵀF` is diagonal plus a rank-2
  term with one negative sign (`nestod_soc` builds `J = Diagonal([−β; β…])` plus a
  rank-1 update, `src/ConicIP.jl:242-244`), so the lifted auxiliary block is indefinite
  and pivot signs are not the ones LDLᵀ expects. The lifted pattern also changes at the
  initial point (`src/kktsolvers.jl:294-304`). **Verified.**

### 2. Solve accuracy, refinement, and termination

- **Refinement runs, but is neither instrumented nor applied where it matters.**
  Measured at the baseline with a counting KKT wrapper on random sparse LPs: about
  2.7–2.9 KKT solves per iteration, i.e. predictor, corrector, and on average under one
  refinement correction. The stopping test is
  `(‖r_y‖+‖r_w‖+‖r_v‖+‖r_s‖)/(n+p+2m) < optTol/1e7` (`src/ConicIP.jl:64`, `:1338`),
  which gets *easier* to pass as the problem grows for comparable component residuals.
  The predictor solve is never refined (`src/ConicIP.jl:1280`), the residual is not
  recomputed after the last correction, and the verbose "refine" column always prints 1
  because the loop variable shadows the outer `rStep` (`src/ConicIP.jl:1325-1326`).
  Once LDLᵀ regularization is introduced, refinement against the unregularized system is
  what recovers accuracy, so this needs a real policy. **Verified.**
- **Complementarity is tested by 2-norm, not aggregate gap.** `rCp` uses
  `‖λ∘λ‖₂/(1+|cᵀy|)` (`src/ConicIP.jl:1136`); the aggregate gap `⟨v,s⟩` is √m larger for
  equal components, so the enforced gap accuracy drifts with size. There is no explicit
  absolute or relative gap test. Equality feasibility enters the termination test but is
  not stored in `prFeas` (`src/ConicIP.jl:1145`). **Verified.**
- **A factorization happens before the termination check** (`src/ConicIP.jl:1086-1107`
  precedes `:1160`), so every solve pays one unnecessary final factorization and can fail
  numerically on an iterate that already met the tolerance. The initial point costs
  another factorization (`src/ConicIP.jl:1014`). **Verified.**
- **The `:Unbounded` status over-claims.** `certificates.jl:155-165` checks `Qy = 0`,
  `Gy = 0`, `Ay ∈ K`, `cᵀy > 0`, which certifies dual infeasibility, not primal
  unboundedness (that also needs primal feasibility). Counterexample: `Q = 0, c = 1,
  A = 0, b = 1, K = R₊` is infeasible, yet the structural zero-column branch
  (`src/ConicIP.jl:801-813`) returns `:Unbounded`. The MOI mapping to `DUAL_INFEASIBLE`
  (`src/MOI_wrapper.jl:520-521`) is correct; the direct-API label and the verbose
  message are not. **Verified.**
- **Products are recomputed.** The screens redo `Gᵀw`, `Aᵀv`, `Ay`, `Gy`, `Qy` each
  iteration (`src/ConicIP.jl:1192`, `:1225-1227`) and the objective recomputes `Qy`
  (`:1139`). Cheap now; not once factors are cheap.

### 3. SDP cones

- **The explicit congruence matrix is O(k⁴) memory and O(k⁵) work**, but only on the
  paths that build it. `Matrix(::VecCongurance)` applies the O(k³) congruence to each of
  k(k+1)/2 basis columns (`src/ConicIP.jl:100-112`); `sparse(F'F)` in the no-lift sparse
  path and `sparse(F⁻ᵀ)` in the 2×2 path hit it. The default dense-QR path applies
  `F⁻ᵀ` column-wise to `AQ2` and never forms it (`src/kktsolvers.jl:91-100`). At k=100
  one such matrix is 204 MB, and the construction allocates an identity and the output
  besides. **Verified.**
- Per iteration per block: two Cholesky factorizations and one SVD in `nestod_sdc`
  (`src/ConicIP.jl:248-268`; v0.4.0 replaced the explicit inverse with a triangular
  solve), a general Lyapunov solve in `dsdc!` (`:414-424`) where the scaled variable λ is
  diagonal and a closed form exists, and one generalized symmetric eigenproblem in
  `maxstep_sdc` (`:331-360`; v0.4.0 replaced the inverse square root and now raises on a
  non-positive-definite iterate instead of returning `Inf`) called four times.
  **Verified.**
- Large-SDP solvers (SDPA, SeDuMi, Mosek, Clarabel with chordal decomposition) never form
  the congruence operator; see Tranche 4 for the reduced-system form in this package's
  notation.

### 4. Time limit and the exhaustion tail

- No wall-clock check anywhere. `MOI.TimeLimitSec` is unimplemented (only
  `SolveTimeSec`, `src/MOI_wrapper.jl:724-725`). The only escape is `maxIters = 100`.
- On exhaustion, `certFallback` can launch up to two more solves of 50 iterations each
  (`src/ConicIP.jl:1419-1452`), each of which calls `imcols` on an auxiliary system
  (`src/fallback.jl:80`, `:149`) and does not receive the caller's `kktsolver`.
  `preprocess_conicIP` retries the whole solve with `staticReg = 1e-8` on `:Error`
  (`src/preprocessor.jl:171-175`). The multiplier on hard instances is unbounded in
  principle. **Verified.**

### 5. Presolve and scaling

- `imcols` runs SPQR `qr(sparse(Aᵀ))` (`src/preprocessor.jl:29`) plus a second
  factorization for the consistency test (`:42`). `preprocess_conicIP` applies it to `G`
  and to the n×(n+m+p) concatenation `[Q Aᵀ Gᵀ]` (`:89`, `:115`). Sparse QR fill can
  dwarf the KKT factorization. ECOS and Clarabel do not rank-detect; they regularize.
- No data equilibration. `docs/src/background.md:217` asks the user to scale the data.
  Pivot thresholds and regularization cannot be tuned independently of data scale, so
  this belongs with the LDLᵀ work, not after it.
- No presolve beyond structural zero rows/columns: no fixed variables, singleton
  equalities, bound tightening, duplicate rows, or empty cone blocks, and no reversible
  transformation record.

### 6. MOI front end

- One `spzeros(dim, n)` per constraint, each carrying an (n+1)-entry column-pointer
  array, filled by scalar CSC insertion and held until concatenation
  (`src/MOI_wrapper.jl:206-240`, `:459-475`). Memory is O(C·n) for C constraint objects,
  which is quadratic when C and n grow together, and can exhaust memory before the
  solver starts. PSD scaling does `dim` row assignments into CSC (`:277-288`); a full
  `UniversalFallback` model copy precedes extraction (`:310-314`). **Verified.**
- Result queries linear-scan `eq_ci_map` and `ineq_ci_map` per constraint and re-slice
  `eq_G[rows,:]` per call (`src/MOI_wrapper.jl:643-717`): retrieving all duals is
  O(C²). **Verified.**
- No quadratic objective through MOI (`Q = spzeros(n,n)` at `src/MOI_wrapper.jl:343`);
  JuMP QP users pay a bridge. The MOI solve timer starts after assembly
  (`src/MOI_wrapper.jl:487`), so assembly cost is invisible to `SolveTimeSec`.

### 7. Secondary constants

- `nt_scaling` allocates a new `Block` per iteration (`src/ConicIP.jl:889-905`); every
  block application allocates its output and per-block temporaries; adjoints construct
  transformed blocks (`src/blockmatrices.jl:111-133`, `:187-200`); `block_idx` and
  `size(::Block)` allocate per call (`:45-49`, `:66-80`). There is no `mul!` or in-place
  solve for `Block`, and a `Block` cannot multiply a `SparseMatrixCSC`.
- `Solution` has abstract-typed numeric fields (`src/ConicIP.jl:491-497`). Minor.
- No multithreading in the package. UMFPACK's factorization is not itself parallel;
  only its dense BLAS kernels are. Not a priority until profiling says otherwise.

### 8. No evidence base

- The largest LP/QP/SOCP in CI has n=650 (`socp_sum_of_norms(150; d = 200)` in
  `test/runtests.jl`). The largest tutorial has 12 variables. The 6010-variable
  issue-#10 instance is download-only. v0.4.0 added an SDPLIB subset fetched on demand
  as a CI gate (`test/sdplib_tests.jl`) and `benchmark/sdplib.jl`; that is the only
  standard-library coverage.
- No runs against Maros–Mészáros (QP), CBLIB (SOCP), or SDPLIB, and no harness with
  phase timings, fill statistics, or independently evaluated residuals.

## Roadmap, in tranches

Tranche 0 is cheap and must precede everything else so the later tranches can be
measured. Tranche 1 is the numerical core and includes equilibration, refinement, and
safeguards, because none of those can be tuned separately. Tranches 2 and 3 are
independent of each other. Tranches 4 and 5 are separate research-grade directions.
The documentation change waits for Tranche 0's harness to produce numbers after
Tranches 1–2.

### Tranche 0: harness, safety, and status correctness

**Status:** done on branch `large-scale-roadmap` (2026-09-06): `benchmark/suite.jl`,
the `dense_bytes_max` guard in `choose_kktsolver`, factorization after the termination
checks, `:DualInfeasible`, `Solution.kkt_solves`, the refine counter, and cached
products. **Estimate was:** 1 week.

1. `benchmark/suite.jl`: a small reproducible set (the issue-#10 instance, a few
   Maros–Mészáros QPs, a few CBLIB SOCPs, synthetic families at several sizes) with
   phase timings (assembly, presolve, setup, per-iteration factor and solve, fallback),
   input nnz, KKT nnz, `nnz(L)`, peak RSS in a fresh worker process, iteration counts,
   and residuals recomputed from original data. Record hardware, Julia and package
   versions, thread counts, and compilation policy.
2. Guard the dense path: a hard memory bound on n² and m(n−p) storage before
   `choose_kktsolver` may return `kktsolver_qr`, independent of nnz/col.
3. Skip the factorization when the termination test already passes: move the residual
   and optimality check ahead of `nt_scaling` and `solve4x4gen` in the loop.
4. Rename the over-claiming status: `:Unbounded` becomes `:DualInfeasible` (with a
   deprecation alias), claim primal unboundedness only with feasibility evidence, and
   fix the verbose message. Keep the MOI mapping.
5. Fix the refine counter, count actual correction solves, recompute the residual after
   the last correction, and expose the per-iteration solve count in `Solution`.
6. Cache `Qy`, `Ay`, `Gy`, `Aᵀv`, `Gᵀw` once per iteration and reuse them in the
   objective, residuals, and screens.

**Verification targets**

- The harness runs end to end on a laptop in under 10 minutes and writes a table that
  a later tranche can diff against.
- A million-variable, 11-nnz/col LP no longer selects dense QR (unit test on
  `choose_kktsolver` with a synthetic pattern; no solve needed).
- The infeasible counterexample above returns an infeasibility status, not
  `:Unbounded`.

### Tranche 1: numerical core

**Status:** done on branch `large-scale-roadmap` (2026-09-07), all eleven items:
`kktsolver_ldl` with the lifted SOC form and QDLDL backend; flop-based default
selection replacing the nnz/col rule, dense gate at 200, `kktsolver_2x2` with a
factorization cache; predictor and corrector refinement with relative/absolute
tolerances; Ruiz equilibration with cone-uniform row scaling, an objective rescale only
outside `[1e-3, 1e3]`, and termination on original-coordinate residuals; relative gap
test and `prFeas` including equalities; `timeLimit` / `MOI.TimeLimitSec`; verified
interiority with geometric back-off and stall detection in the line search; and the
callback contract in the KKT-solver guide.

**Measured 2026-09-07** (`benchmark/suite.jl`, full set, Apple M3 Pro, one Julia
thread, six BLAS threads; each instance in a fresh process, so RSS includes about
1.3 GB of compilation):

| instance | n | iterations | KKT solves | time | peak RSS |
|---|---|---|---|---|---|
| lp-band | 2 000 | 19 | 42 | 0.04 s | 1.4 GB |
| lp-band | 20 000 | 21 | 57 | 0.63 s | 1.8 GB |
| lp-band | 200 000 | 23 | 66 | 5.7 s | 3.2 GB |
| qp-band | 200 000 | 9 | 17 | 1.6 s | 2.4 GB |
| sum-of-norms SOCP | 6 500 | 12 | 23 | 2.1 s | 1.7 GB |
| issue #10 | 6 010 | 25 | 91 | 0.34 s | 1.4 GB |
| chainsing-1000-1 (CBLIB) | 12 976 | 11 | 21 | 0.88 s | 2.0 GB |
| nb, nql30, qssp30, sched_50_50_scaled (CBLIB) | 2 383–7 565 | 16–32 | 55–97 | 0.36–2.0 s | ≤ 1.9 GB |
| Maros–Mészáros hs21…qpcblend (7 of 8) | 3–326 | 10–32 | 23–75 | ≤ 0.03 s | |

Every solved instance reports original-coordinate residuals and relative gap below
`1e-6`. Memory grows linearly across each banded family (about 1.9 GB above baseline
at n = 200 000 with 1.4 M nonzeros in A). Issue #10 went from 0.78 s / 29 iterations
to 0.34 s / 25, exactly 2× ECOS's 0.17 s / 23. The one failure, `cvxqp1_s`, is not the
solver's: its Hessian is positive semidefinite but singular, and MOI's
quadratic-to-SOC bridge refuses it, which is the native-QP item in Tranche 2. The
n = 10⁶ point of the verification list has not been run.
Measured so far: LDLᵀ is 5–25× faster than UMFPACK LU on banded LP/QP and large
single SOCs at equal iteration counts, and 1.7× *slower* on the sum-of-norms SOCP at
n = 6500 where fill is heavy (QDLDL is scalar code, UMFPACK's kernels are BLAS). The
old absolute refinement threshold was hurting well-scaled problems: the Miles scaling
instance at κ = 1 went from 89 iterations to 15 with the relative tolerance, and the
κ = 10⁶–10⁸ variants from 30 to 21–23 with equilibration. A mixed SOC/LP with rows
scaled over ten orders of magnitude stalls at 100 iterations unequilibrated and solves
in 13 equilibrated. Two lessons for the remaining items: the retained best iterate must
be judged on feasibility and complementarity only (a gap term hides the late iterate
the certificate fallback needs), and predictor and corrector need separate refinement
budgets.
**Estimate was:** 4–8 focused weeks for one experienced contributor, of which the LDLᵀ
prototype is 1–2 weeks. Scaling, safeguards, and validation are the rest.

1. **Formulation.** Negate the third block row and its right-hand side and factor the
   symmetric quasi-definite matrix

   ```
   K_δ = [ Q + δ_p I     Gᵀ        −Aᵀ           ]      rhs = [  b_y ]
         [ G            −δ_e I      0            ]            [  b_w ]
         [ −A            0        −(FᵀF + δ_c I) ]            [ −b_v ]
   ```

   with `δ_p, δ_e > 0` and `δ_c ≥ 0`. This is the standard augmented conic form after
   grouping `[G; −A]` as the constraint block. Keep the reduced form
   `[Q + δ_p I + Aᵀ(FᵀF + δ_c I)⁻¹A   Gᵀ; G  −δ_e I]` as a structure-dependent option
   chosen by predicted fill, not as the default: constraint rows become cliques and the
   Gram term can worsen conditioning.
2. **SOC blocks.** Derive the lifted representation of `FᵀF = D + u uᵀ − v vᵀ` with the
   auxiliary rows placed so the extended matrix stays quasi-definite with known pivot
   signs (as in the ECOS paper), and prove it before coding. Do not reuse `lift`. Keep
   the pattern fixed across iterations, including at the identity-scaled initial point,
   by storing structural zeros.
3. **Backend.** QDLDL.jl first: value updates, numeric refactorization into a fixed
   pattern, prescribed pivot signs, and dynamic regularization are exactly the needed
   operations, and it keeps the pure-Julia claim. CHOLMOD `ldlt` is a comparison
   backend only; it has no numerical pivoting and is not a rescue path. Do not write
   another sparse LDLᵀ.
4. **Regularization policy.** Equilibrate first, then apply positive primal and negative
   dual shifts in scaled coordinates; carry expected pivot signs through the
   permutation; record repaired pivots and the inertia; evaluate residuals against the
   original Newton system; detect stagnation, worsening, and nonfinite corrections;
   allow bounded refactorization retries with larger shifts.
5. **Refinement policy.** Refine both the predictor and the corrector against the
   unregularized system, with independent absolute and relative linear-residual
   tolerances and a backward-error diagnostic. Refinement is a contraction only when
   `‖K_δ⁻¹E‖ < 1`; treat failure to contract as a signal to refactor with different
   shifts, not as noise.
6. **Equilibration.** Ruiz scaling of `[Q Aᵀ Gᵀ; A; G]` with uniform scaling inside
   each SOC and SDP block (row scaling inside a block changes the cone), applied to
   `Q, c, A, b, G, d`, undone on primal and dual variables, slacks, objective values,
   and certificates. Available to the direct API, not tied to rank-checking presolve.
7. **Termination.** Add explicit absolute and relative gap tests alongside the residual
   tests; store `rEq` in the solution; document what each field measures.
8. **Defaults.** `kktsolver_ldl` becomes the default for all non-SDP problems. The
   dense path is chosen only when its storage fits the Tranche 0 memory guard and its
   predicted flop count is lower. Give `kktsolver_2x2` the factorization cache or retire
   it.
9. **Time limit.** A `timeLimit` keyword covering presolve, initialization, retries, and
   fallback, checked once per iteration, returning `:TimeLimit` with the best iterate
   assessed so far (or a documented result when none has been). Wire `MOI.TimeLimitSec`
   and `MOI.TIME_LIMIT`. Note that a single factorization can overrun the deadline.
10. **Step safeguards.** Keep `DTB` (`src/ConicIP.jl:1358-1360`) and the v0.4.0
    finiteness guards, and add verified interiority after each step, a backtracking or
    refactor-and-retry path on failure (today a boundary iterate is a terminal
    `:Error`), and tiny-step termination.
11. **Callback contract.** Document the `solve3x3gen(F, F⁻ᵀ)` interface's sign
    convention, who owns regularization, what residual is guaranteed, how failure is
    reported, and workspace lifetime. Add an in-place variant without breaking the
    existing one.

**Verification targets**

- `Pkg.test()` green, with SDP tests still routed through `kktsolver_qr`.
- Dependent and nearly dependent equality rows (including the issue-#10 zero row) solve
  to tolerance with `rank_check = :never` and no presolve retry. These are central
  acceptance cases, not exclusions.
- `benchmark/issue10.jl` on the Tranche 0 harness: time and iteration count recorded
  under stated conditions. The hypothesis is within 2.5× of ECOS at equal accuracy;
  treat a miss as data, not failure.
- Synthetic sparse LP and SOCP families at n = 10⁴, 10⁵, 10⁶ with bounded fill: peak
  RSS tracks `nnz(L)` plus workspace across the family, no n² term; solve to
  `optTol = 1e-8` with residuals recomputed in original coordinates.
- Deliberately badly scaled variants (rows and whole SOC blocks scaled by 10⁶) reach
  the same original-coordinate accuracy with iteration and time degradation under 25 %.
- `timeLimit = 1` on the n = 10⁵ instance returns `:TimeLimit` with a finite best
  iterate, and the total wall time including presolve is under 1 s plus one
  factorization.
- `jetls check` shows no new inference failures.

### Tranche 2: front end and presolve

**Status:** done on branch `large-scale-roadmap` (2026-09-07): triplet assembly with one
`sparse` call per matrix, PSD permutation and √2 applied to the triplets, adjacent
orthant blocks merged, O(1) constraint-result lookup with `G*y` computed once,
`SolveTimeSec` covering assembly (`Optimizer.assembly_time` separately); native
quadratic objectives through MOI (`cvxqp1_s`, refused by the SOC bridge, now solves to
its reference objective); singleton-equality presolve with dual and objective postsolve
and certified conflicts (`fix_singletons`); `rank_check = :auto | :always | :never`
with `:auto` skipping the two sparse QRs whenever the LDLᵀ solver is used;
`cached_kktsolver_ldl` reusing the AMD ordering across same-structure solves. Two
termination fixes fell out of the presolve work: the gap is measured as ⟨v,s⟩ rather
than `pobj − dobj` (whose residual products floor it when duals are large), and
feasibility residuals are normalized by the data norm times the iterate norm as well
as the right-hand side, since a homogeneous equality row under a 10⁸ scale was an
absolute test. Not done: bound tightening, duplicate-row detection, and general
singleton substitution beyond equality rows. **Estimate was:** 2–4 weeks.

**Measured 2026-09-07** (full harness, same machine as the Tranche 1 table; all 23
instances solve, residuals and gaps below `1e-6` in original coordinates):

| instance | Tranche 1 | Tranche 2 |
|---|---|---|
| issue #10 (6 010 vars) | 25 it, 0.34 s | 25 it, 0.21 s |
| CBLIB nb | 19 it, 2.03 s | 19 it, 0.76 s |
| CBLIB chainsing-1000-1 | 11 it, 0.88 s | 11 it, 0.22 s |
| CBLIB nql30 / qssp30 / sched_50_50 | 0.36 / 0.65 / 0.36 s | 0.18 / 0.30 / 0.16 s |
| Maros–Mészáros primal1 | 32 it | 13 it |
| Maros–Mészáros qafiro / qpcblend / dual1 | 23 / 24 / 17 it | 14 / 17 / 12 it |
| Maros–Mészáros cvxqp1_s | bridge failure | 10 it, 0.001 s |
| banded LP / QP families | | unchanged |

The CBLIB gains come from the front end (triplet assembly, no rank detection before
an LDLᵀ solve), the Maros–Mészáros iteration drops from the native quadratic objective
replacing the SOC epigraph. Issue #10 now stands at 1.2× ECOS.

**Independent review of Tranches 0–2 (2026-09-07), fixes landed the same day:** the
`Xmax·‖iterate‖` residual normalization could certify an infeasible QP as `:Optimal`
(now componentwise `‖|A||y|‖` etc.); `:Optimal` could return an earlier saved iterate;
certificates found on scaled data were not revalidated on the original data (one in
eight failed on badly scaled infeasible LPs); duplicate singleton rows broke the dual
postsolve; the presolve dropped the fixed variables' objective constant from the gap
criterion; `soc_uv`'s degeneracy threshold was scale-sensitive; nonconvex quadratic
objectives through MOI returned `OPTIMAL`; inequality `ConstraintPrimal` read the slack;
refinement kept worsening corrections; the dense flop model scored `p = n` as free; an
over-budget SDP fell through to LDLᵀ; `nestod_soc` overflowed on jointly extreme scales;
assembly time was not deducted from `TimeLimitSec`. Cleared by the review: LDLᵀ signs
and pivot-sign vector, fixed pattern, `unequilibrate!` maps, MOI dual signs, lifted
columns in the flop count. A subsequent Codex review added original-data checks after singleton and
rank-reduction postsolve: dropped rows must satisfy `optTol`, and lifted rays must
validate against the full data. `rank_check = :auto` now also recognizes the cached
LDLᵀ callable. Certificate validators reject nonfinite normalized rays; structural
certificate failures no longer fall through to row/column deletion. Bounded Ruiz
sweeps now apply the same increments to the working matrices and scaling record.
Time accounting includes singleton reduction, equilibration, MOI Hessian validation,
and fallback setup, with an expiry check on loop exhaustion as well.
Dual objective reporting now uses the stationary quadratic dual formula; the MOI
convexity guard uses diagonal congruence scaling to prevent an unrelated large block
from masking negative curvature. Feasibility-sense models ignore cached objectives.
Still open from the review: peak-memory estimate for the
dense path (the 4 GiB figure is a routing estimate, not a bound).

**Against the last release (v0.4.0, commit `0430b9e`), measured 2026-09-07 after the
review fixes (commit `3aff869`), same machine, sequential runs, the harness run in a
v0.4.0 worktree with two compatibility shims (`kkt_solves`, `RawSolver` absent there):**

| instance | v0.4.0 it / s / RSS | now it / s / RSS | speedup |
|---|---|---|---|
| lp-band-2000 | 22 / 3.72 / 1.8 GB | 19 / 0.033 / 1.5 GB | 110× |
| lp-band-20000 | 23 / 2520 / 6.9 GB | 21 / 0.57 / 1.7 GB | 4400× |
| lp-band-200000 | child process died | 23 / 5.3 / 2.7 GB | — |
| qp-band-2000 / 20000 / 200000 | 6–7 / 0.021 / 0.39 / 3.36 | 8–9 / 0.0085 / 0.085 / 1.41 | 2.4× / 4.6× / 2.4× |
| sum-of-norms SOCP 150 / 1500 | 9 / 0.026, 10 / 1.18 (3.6 GB) | 10 / 0.011, 12 / 1.79 (1.4 GB) | 2.4× / **0.7×** |
| issue #10 | 29 / 0.69 | 25 / 0.34 | 2.0× |
| Maros–Mészáros hs21 hs35 hs76 qafiro dual1 primal1 qpcblend sambal | 7–30 it, 2–47 ms | 5–17 it, 0.4–5.7 ms | 1.0–9.4× |
| Maros–Mészáros cvxqp1_s | bridge failure | 10 / 0.0014 | — |
| CBLIB chainsing-1000-1 / nb / nql30 / qssp30 | 0.97 / 4.72 / 0.33 / 1.64 | 0.39 / 0.77 / 0.18 / 0.46 | 2.5× / 6.1× / 1.8× / 3.6× |
| CBLIB sched_50_50_scaled | 23 / 7.08 / 3.4 GB | 32 / 0.17 / 1.5 GB | 42× |

v0.4.0's residuals on the same instances are typically 1e-8 to 1e-10 (relative, its own
normalization); the current tree reports 1e-13 to 1e-16 in original coordinates. The one
regression is the 1500-variable sum-of-norms SOCP: v0.4.0 took the dense path (3.6 GB),
the current tree the lifted LDLᵀ with two more iterations; the LDLᵀ route wins on memory
and loses 0.6 s. The 200 000-variable LP under v0.4.0 exited within a minute with no error
text (memory, most likely).

1. MOI assembly directly into the global `A` and `G` by triplet accumulation (one
   `sparse(I, J, V, m, n)` per matrix, not per constraint object); merge compatible
   orthant blocks; PSD input scaling as a diagonal multiply; result lookup through
   typed per-constraint-type maps with O(1) access; cache `G*y` and `A*y` once per solve;
   start the solve timer before assembly or report assembly time separately.
2. Native quadratic objectives through MOI.
3. Reversible presolve: fixed variables, singleton equalities, bound tightening,
   duplicate rows, empty or inconsistent cone blocks, with a transformation record for
   postsolve of primal, dual, and certificate vectors. Free variables already live as the
   unconstrained `y`; keep that representation and never split them.
4. Make full-matrix `imcols` rank detection opt-in (`rank_check = :auto | :always |
   :never`). Drop the `staticReg` retry once LDLᵀ carries its own regularization.
5. Repeated-solve workspace reuse: keep the symbolic analysis and buffers when the
   structure is unchanged. Numerical warm starts are a separate, later decision.

**Verification targets**

- MOI `copy_to` plus solve on a 10⁵-constraint, 10⁵-variable LP: assembly memory
  linear in nnz (no O(C·n) term), assembly time under the first factorization, and
  retrieval of all duals under one second.
- Postsolve round-trips primal, dual, and certificate vectors on the harness set to the
  same original-coordinate accuracy as the unpresolved solve.

### Tranche 3: robustness and iteration count

**Status:** not started. **Estimate:** homogeneous embedding is a separate multiweek
effort; correctors are 1–2 weeks including measurement.

1. Homogeneous self-dual embedding, compatible with a quadratic objective (Clarabel's
   formulation is the reference), replacing the certificate screens and the
   `certFallback` solves. This is a robustness project: it gives infeasibility detection
   the same convergence machinery as optimality. Decide the architecture during
   Tranche 1 so the KKT assembly leaves room for the extra row and column.
2. Gondzio multiple centrality correctors, measured, not assumed. Factor reuse is real,
   but each corrector adds cone operations and step computations, and SOC/SDP centrality
   needs the Jordan-algebraic construction, not LP-style clipping. Enable by default
   only if total time falls on the harness.

**Verification targets**

- Infeasible and dual-infeasible harness instances terminate with certificates in
  bounded iterations, without fallback solves. The random unbounded SOCP that today runs
  35 iterations to objective 10⁻²⁹ before a singular factorization is the smoke test.
- Correctors: total time on the harness set decreases; iteration count is reported, not
  targeted.

### Tranche 4: SDP at scale

**Status:** not started. **Estimate:** research-grade, months; milestone-based. Only
worth funding around a named family (dense-block SDPs, or chordally sparse SDPs).

In this package's notation `A` maps n free variables to m cone coordinates. For an SDP
block with row range I, the coefficient matrix of variable j is `B_j = mat(A[I, j])`, a
column of the conic map reshaped to a symmetric matrix. With `VecCongurance(R)` and
`T = R Rᵀ`, the scaling acts as `FᵀF: X ↦ T X T` and its inverse as `X ↦ T⁻¹ X T⁻¹`, so
the block's contribution to the n×n reduced Hessian is `S_ij = tr(B_i T⁻¹ B_j T⁻¹)`.
That reduced Hessian still carries `Q` and the equality block `G`; classical SDP
notation calls its dimension m, which is this package's n.

Specify before coding: the primal/dual formulation solved and when to dualize; the exact
`svec` ordering and √2 weights (the package already has one); NT versus HKM direction,
since the reduced-Hessian formula must match; sparse or low-rank `B_j` handling and
block incidence; storage versus recomputation of transformed coefficients; chordal
decomposition and clique merging when aggregate sparsity allows; direction recovery,
regularization, refinement, and original-data residual checks.

Independent of the reduced system: exploit the diagonal λ in `dsdc!` instead of a
general Lyapunov solve (v0.4.0 already removed the explicit inverse in `nestod_sdc` and
moved the step length to a generalized eigenproblem). An explicit symmetric-Kronecker
operator in the augmented matrix is still O(k⁴) when dense; `svec` improves constants
only.

Build on the v0.4.0 SDP test suite (`test/sdp_tests.jl`, `test/sdp_problems.jl`), the
on-demand SDPLIB subset (`test/sdplib_tests.jl`), and `benchmark/sdplib.jl`. SDP left
experimental status in v0.4.0; the scale claim for it stays separate from the
LP/QP/SOCP one until this tranche delivers.

**Verification targets**

- For the chosen family, peak memory has no k⁴ term (measured across k at fixed n) and
  the reduced-Hessian assembly time is reported separately from its factorization.
  Note that at n = 10⁴ one dense reduced Hessian is 800 MB and its Cholesky is about
  3×10¹¹ flops; the target is bounded by that, not by k.
- The vendored SDPLIB subset matches reference objectives to 1e-6 relative *and*
  satisfies primal and dual feasibility, gap, and cone membership at the same
  tolerance, with reliable reference statuses.

### Tranche 5: operator inputs and structured iterative KKT

**Status:** not started. **Estimate:** operator support 2–3 weeks; a competitive
iterative KKT solver is a research project tied to a named application.

Keep operator support as an architectural goal: the main loop touches `Q`, `A`, `G` only
through products, the certificate validators mostly do too, and the blockers are
`structurally_zero_cols` (needs CSC), `norm(Q, Inf)`, the deflation sub-indexing, the
default dispatch, presolve, and fallback construction. A generic MINRES with a
block-diagonal preconditioner is not a competitive-solver plan: near the boundary the
NT-scaled system is badly conditioned and the preconditioner is the research problem.
Ship a bundled iterative solver only with an application whose structure gives a
preconditioner and a memory advantage, and measure Krylov iterations, matvecs,
preconditioner cost, and total time, not just outer iterations.

**Verification targets**

- A problem given with `Q`, `A`, `G` as operators solves to `optTol` through a
  user-supplied KKT callback with no call to `sparse` or `Matrix` on them.
- For the named application: total time and peak memory beat the direct solver on that
  family at equal accuracy.

### Documentation change

Rewrite `docs/src/index.md:76-92` and the README positioning only after the Tranche 0
harness, run after Tranches 1–2, shows: a fixed instance set with a stated solved set,
equal accuracy enforced in original coordinates, and both pairwise time ratios against
Clarabel.jl and a Dolan–Moré profile. "Within 2× of Clarabel on 90 % of a broad suite"
is a stretch hypothesis, not a milestone: pure Julia is not the obstacle (Clarabel.jl
is pure Julia), but shared QDLDL machinery does not confer shared robustness, assembly
efficiency, or cone-implementation quality. Benchmark native QPs against native-QP
solvers; report the ECOS reformulation cost separately. Include presolve and MOI
assembly in end-to-end times.

## Regimes the harness must cover

n ≫ m, m ≫ n, many small cones, few large cones, dense columns, dependent and nearly
dependent equalities, degenerate optima (nonunique solutions, lack of strict
complementarity, near-boundary optima), badly scaled data, infeasible and
dual-infeasible instances, and repeated solves with unchanged structure.

## Explicitly not planned

- Threading, GPU, or reduced precision before profiling on the Tranche 0 harness
  identifies a kernel that would benefit. Not a ban; a sequencing decision.
- Dropping the "moderate-size" phrase before the numbers above exist.
- A hand-written sparse LDLᵀ.
