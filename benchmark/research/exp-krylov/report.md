# Krylov KKT solves for ConicIP — empirical study on the solver's own Newton systems

Agent `exp-krylov`, 2026-09-20. Apple M3 Pro, 1 Julia thread, `BLAS.set_num_threads(1)`.
Code under test: worktree `.worktrees/large-scale-roadmap`, HEAD `97f1111` (ConicIP v0.6.0 + private harness).
**No file in the repository was modified.** Everything is in `scratchpad/exp-krylov/`.

## 1. Method

### Capture (`capture.jl`)

The `kktsolver` callback was wrapped without touching `src/`:

```julia
function ks(Q, A, G, cone_dims)
    pat = ConicIP._ldl_pattern(Q, A, G, cone_dims; lift_min = 6)
    gen = ConicIP.kktsolver_ldl(Q, A, G, cone_dims; pattern = pat, static_reg = 1e-8)
    # snapshot pat.K.nzval after each gen(F, F⁻ᵀ); record every rhs and timing
end
```

`kktsolver_ldl` accepts a prebuilt `pattern` and rewrites **that object's**
`K.nzval` in place before every numeric refactorization, so snapshotting
`pat.K.nzval` right after `gen(F, F⁻ᵀ)` returns gives the exact assembled matrix
the backend factorizes, with the exact regularization it uses. The real
`ConicIP.kktsolver_ldl` does all the work, so the run is unchanged: iteration
counts and statuses match `benchmark/results/suite-3881ac2.csv` on every instance.

Recorded per iteration `t`: `K.nzval` (upper triangle, fixed pattern), the AMD
permutation, every right-hand side in the lifted layout `rhs = (bx, by, −bz, 0)`,
the wall time of the factorization call and of each `solve3x3` call, the index `t`.

Regularization actually in force on all six instances:

| symbol | value | where |
|---|---|---|
| `δ_p` | `1e-8` | `(1,1)` block, `Q + δ_p I` |
| `δ_e` | `1e-8` | `(2,2)` block, `−δ_e I` |
| `δ_c` | `0` | `(3,3)` block, `−(FᵀF + δ_c I)` |
| dynamic | `eps 1e-13`, `delta 2e-7` | QDLDL pivot repair |

`K_t` below is the **regularized** `K_δ` (that is what gets factorized). ConicIP
additionally refines each solve against `K₀ = K_δ − Diagonal(shift)` and refines
the 4×4 step on top; the backend's internal target is
`refine_tol = 1e-13·(1+‖rhs‖)`, hence the 1e-12 column everywhere below.

`nlift = 0` on all six instances: every second-order cone here has dimension < 6,
so the lifted-SOC path was **not** exercised.

### Experiments (`experiments.jl`, scratch env `exp-krylov/Project.toml`)

- Conditioning: `|λ|max(K_t)` and `|λ|max(K_t⁻¹)` by KrylovKit Lanczos
  (`tol=1e-4`, `krylovdim=24`), `κ ≈ |λ|max·|λ|max(K⁻¹)`; `K_t⁻¹` applied by a
  QDLDL factorization of `K_t`. Same for `N_t` with a CHOLMOD Cholesky.
- Normal equations, LP/QP only (`p = 0`, `FᵀF` diagonal):
  **`N_t = Q + δ_p I + Aᵀ(FᵀF + δ_c I)⁻¹A`**, rhs `b₁ − Aᵀ D⁻¹ b₃`, from
  eliminating `z` in `[Q+δ_pI, −Aᵀ; −A, −D]`. (With `p > 0` the term would be
  `+ GᵀG/δ_e`, numerically useless at `δ_e = 1e-8`, so the SOCPs stayed on `K_t`.)
- Krylov: Krylov.jl, `atol = 0`, `history = true`; iteration counts read off the
  residual history at relative 1e-8 and 1e-12.
- Stale preconditioner: exact QDLDL `LDLᵀ` of `K_{t−k}` (same AMD permutation and
  `Dsigns`) used (a) as the stationary iteration `x ← x + L⁻ᵀD⁻¹L⁻¹(b − K_t x)`
  and (b) as a right preconditioner for restarted GMRES(50), cap 200.
- Incomplete: `LimitedLDLFactorizations.lldl`, `memory ∈ {0,5,10,20,30}`; PCG on
  `N_t` (LP/QP), right-preconditioned GMRES on `K_t` (SOCP).

### Exact commands

```
# capture (from the worktree)
cd /Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap
timeout 900 julia --project=. <scratch>/exp-krylov/capture.jl

# experiments (from the scratch dir)
cd <scratch>/exp-krylov
timeout 1800 julia --project=. experiments.jl \
  qp-band-20000 lp-band-20000 issue10 chainsing-1000-1 socp-sumnorms-1500 nql30 \
  > full.log 2>&1
```

Versions: Julia 1.12.7; ConicIP 0.6.0 (`97f1111`); QDLDL 0.4.1; AMD 0.5.4;
Krylov 0.10.10; KrylovKit 0.10.4; LimitedLDLFactorizations 0.5.2; LinearOperators 2.14.3.

Raw output `full.log`; per-instance CSVs `res-<instance>.csv`; captured systems
`data/<instance>.jls` (144 MB, Julia `Serialization`); accounting `accounting.txt`.

## 2. Per-instance cost model (the yardstick)

Measured on `K_T`: bare QDLDL numeric factorization, one triangular back-solve,
one sparse matvec. A preconditioned Krylov iteration costs about one back-solve
plus one matvec, so **`t_fac / t_bs` is the number of Krylov iterations a
factorization buys**.

| instance | n / m / p | N | nnz(L) | factor (ms) | back-solve (ms) | matvec (ms) | **fac / bs** |
|---|---|---|---|---|---|---|---|
| lp-band-20000 | 20000/60000/0 | 80000 | 439927 | 8.63 | 0.936 | 0.344 | **9.2** |
| qp-band-20000 | 20000/40000/0 | 60000 | 99994 | 3.52 | 0.408 | 0.182 | **8.6** |
| issue10 | 6010/7020/4100 | 17130 | 31065 | 0.68 | 0.130 | 0.042 | **5.2** |
| chainsing-1000-1 | 12976/9982/8982 | 31940 | 52897 | 1.14 | 0.263 | 0.081 | **4.4** |
| nql30 | 3601/2700/2779 | 9080 | 111051 | 2.58 | 0.199 | 0.030 | **12.9** |
| socp-sumnorms-1500 | 6500/4500/3000 | 14000 | 841528 | **161.7** | 1.131 | 0.036 | **143.0** |

On five of six instances a Krylov solve must converge in **under ~5–13
iterations** to beat refactorizing. Only socp-sumnorms-1500, where the scalar
QDLDL factorization has 24× fill and costs 162 ms, leaves room.

## 3. Conditioning along the iterates

| instance | t=1 | t≈T/2 | t=T−2 | t=T |
|---|---|---|---|---|
| lp-band-20000 κ(K) | 4.8e+00 | 1.6e+06 | 4.9e+12 | 9.1e+12 |
| lp-band-20000 κ(N) | 9.1e+00 | 3.5e+04 | 7.6e+10 | 1.4e+11 |
| qp-band-20000 κ(K) | 3.6e+00 | 6.8e+03 | 3.0e+07 | 1.7e+10 |
| qp-band-20000 κ(N) | 2.5e+00 | 7.8e+02 | 1.8e+06 | 9.9e+08 |
| issue10 κ(K) | 3.1e+01 | 8.9e+08 | 2.0e+13 | 1.8e+16 |
| chainsing-1000-1 κ(K) | 2.9e+01 | 2.1e+03 | 2.4e+06 | 2.9e+09 |
| nql30 κ(K) | 3.0e+03 | 4.3e+10 | 3.9e+15 | **2.5e+17** |
| socp-sumnorms-1500 κ(K) | 1.2e+03 | 4.5e+05 | 5.0e+08 | 2.3e+10 |

`κ(K_t)` grows 9–14 orders of magnitude over a run; `κ(N_t)` is ~2 orders better
but grows at the same rate. nql30 (2.5e+17) and issue10 (1.8e+16) end above
`1/ε_mach`: at those iterations an unpreconditioned Krylov method cannot reach
1e-8 in *any* number of iterations in double precision. The direct solver only
survives there because of regularize-and-refine.

## 4. Unpreconditioned and Jacobi MINRES on `K_t`

Iterations to relative residual 1e-8, cap 2000; `—` = not reached (best relative
residual attained in the cap shown).

| instance | t | plain @1e-8 | plain best | Jacobi @1e-8 | Jacobi best |
|---|---|---|---|---|---|
| lp-band-20000 | 1 | 49 | — | 83 | — |
| lp-band-20000 | 10 | — | 1.0e-03 | — | 4.8e-04 |
| lp-band-20000 | 21 | — | 3.4e-05 | — | 8.2e-01 |
| qp-band-20000 | 1 | 33 | 4.9e-10 | 33 | 4.9e-10 |
| qp-band-20000 | 9 | — | 4.0e-04 | — | 7.1e-01 |
| issue10 | 1 | 35 | 2.7e-12 | — | 3.9e-05 |
| issue10 | 25 | — | 1.3e-04 | — | 8.4e-01 |
| chainsing-1000-1 | 1 | 93 | 2.2e-09 | — | 3.2e-05 |
| chainsing-1000-1 | 11 | — | 4.1e-06 | — | 5.5e-02 |
| nql30 | 1…16 | — | 2.1e-08 … 3.9e-05 | — | 3.0e-03 … 7.4e-01 |
| socp-sumnorms-1500 | 1…12 | — | 2.3e-08 … 2.7e-05 | — | 5.4e-05 … 7.8e-01 |

Dead on arrival. Only at `t = 1` (identity scaling) does anything converge, and
then in 33–93 iterations — already 4–10× a factorization. From `t ≥ 2` no
instance reaches 1e-8 in 2000 iterations; 1e-12 is reached nowhere. Jacobi
(`M = diag|K|⁻¹`) is consistently **worse** than no preconditioner: it destroys
the 2×2 saddle structure the indefinite spectrum needs.

## 5. Stale-factorization preconditioner (the practically interesting one)

Exact `LDLᵀ` of `K_{t−k}` applied to `K_t`. GMRES(50), right preconditioned,
cap 200. Entries are **iterations to 1e-8 / 1e-12**, `—` = cap reached.

| instance | t | k=1 | k=2 | k=3 | k=5 |
|---|---|---|---|---|---|
| lp-band-20000 | 10 | 42 / 69 | 98 / 164 | 169 / — | — / — |
| lp-band-20000 | 19 | 31 / 41 | 91 / 178 | — / — | — / — |
| lp-band-20000 | 21 | 14 / 20 | 34 / 42 | 118 / — | — / — |
| qp-band-20000 | 7 | 38 / 67 | 178 / — | — / — | — / — |
| qp-band-20000 | 9 | 32 / 61 | — / — | — / — | — / — |
| issue10 | 12 | 31 / 47 | 92 / 149 | 148 / — | — / — |
| issue10 | 23 | 10 / 13 | 13 / 17 | 18 / 26 | 44 / 81 |
| issue10 | 25 | 20 / 45 | 60 / 178 | 49 / — | 97 / — |
| chainsing-1000-1 | 9 | 17 / 23 | 24 / 35 | 45 / 87 | 141 / — |
| chainsing-1000-1 | 11 | 11 / 17 | 15 / 26 | 28 / 41 | 47 / 121 |
| nql30 | 8 | 31 / 44 | 88 / 140 | — / — | — / — |
| nql30 | 16 | 41 / 170 | 94 / 132 | — / — | — / — |
| socp-sumnorms-1500 | 6 | 28 / 40 | 83 / 128 | 183 / — | — / — |
| socp-sumnorms-1500 | 10 | 40 / 151 | 136 / — | — / — | — / — |
| socp-sumnorms-1500 | 12 | 33 / 47 | 47 / 85 | 129 / — | — / — |

**Simple iterative refinement never converged — any `t`, any `k`, any
instance.** Its best residual is 0.02–5e+5 times `‖b‖` (`IR_best` in `full.log`);
the spectral radius of `I − K_{t−k}⁻¹K_t` exceeds 1 almost everywhere. That kills
the cheapest "reuse the factor" scheme outright.

Why: `relΔK = ‖K_t − K_{t−k}‖_F/‖K_t‖_F` is **0.52–0.99 already at `k = 1`** and
`≥ 0.95` for `k ≥ 2` throughout. A one-iteration-old factorization is not a small
perturbation of the current matrix — the NT scaling block moves by a factor of
the step length in every coordinate each iteration.

GMRES is far more robust than IR but still needs 10–42 iterations at `k = 1` to
1e-8 and 13–170 to 1e-12, against a break-even of 4–13. `k ≥ 3` is hopeless
except on chainsing.

The cheap "keep `L`, update only `D`" variant was **not** run: QDLDL's `D` is the
pivot sequence of a permuted elimination, so a diagonal change in `K` is not a
diagonal change in `D` once fill has propagated — not "simple" in the sense the
brief allowed, and the `relΔK` numbers say it would not help.

## 6. Incomplete / limited-memory factorization

**Caveat that voids most of this section.** On four of six instances `lldl` with
`memory ≥ 10` reproduces the **complete** factorization — `nnz(L)` matches QDLDL
exactly (issue10 31065 = 31065; chainsing 52897 = 52897) or, for the banded
`N_t`, IC(0) on a banded matrix *is* the Cholesky factor, and PCG converges in
**1** iteration. These systems have too little fill for "incomplete" to mean
anything; those 1–2 iteration counts are an artefact.

The informative cases are the two instances with real fill:

| instance | t | memory | nnz(L_P) | nnz(L_exact) | setup (s) | GMRES @1e-8 | @1e-12 |
|---|---|---|---|---|---|---|---|
| socp-sumnorms-1500 | 1 | 30 | 55094 | 841528 | 0.006 | 9 | 14 |
| socp-sumnorms-1500 | 6 | 30 | 82321 | 841528 | 0.010 | **27** | 40 |
| socp-sumnorms-1500 | 10 | 30 | 82160 | 841528 | 0.009 | — | — |
| socp-sumnorms-1500 | 12 | 30 | 82166 | 841528 | 0.009 | — | — |
| nql30 | 1, 8, 14, 16 | 0 / 10 / 30 | 18–77 k | 111051 | 0.002–0.004 | — | — |

The pattern is decisive: the limited-memory factor works *early*, when `κ(K_t)`
is modest, and stops working exactly where it would have to pay off. On
socp-sumnorms-1500 it is a 10× win at `t = 6` (0.016 s vs 0.162 s) and a total
failure at `t = 10, 12`. On nql30 it never converges at all.

LP/QP normal equations, plain CG and Jacobi-PCG for reference:

| instance | t | CG @1e-8 | Jacobi-PCG @1e-8 | Cholesky factor (ms) | Cholesky solve (ms) |
|---|---|---|---|---|---|
| lp-band-20000 | 1 | 27 | 26 | 25.3 | 0.462 |
| lp-band-20000 | 10 | 1203 | 784 | 9.9 | 0.457 |
| lp-band-20000 | 19 | — | — | 9.7 | 0.457 |
| lp-band-20000 | 21 | — | — | 9.3 | 0.460 |
| qp-band-20000 | 4 | 156 | 41 | 3.8 | 0.313 |
| qp-band-20000 | 7 | — | 35 | 3.7 | 0.314 |
| qp-band-20000 | 9 | — | 24 | 4.3 | 0.315 |

Jacobi-PCG on `N_t` is the only cheap preconditioner here that survives to the
last iteration (qp-band: 24 iterations at `t = 9`), and it is still 6× the 4 ms
Cholesky. On lp-band it dies at `t ≥ 19`.

## 7. Accounting

Baseline = captured wall time of the run's factorization calls (`T_fac`,
including the per-iteration `FᵀF` value assembly) plus its `solve3x3` calls
(`T_solve`, including the backend's internal refinement). Iterative scheme =
**"factor every 2nd iteration"**: `⌈T/2⌉` full factorizations, fresh iterations
solved directly, stale ones by GMRES right-preconditioned with the previous
factor at the median `k = 1` iteration count, one back-solve + one matvec each;
the value assembly is charged to both (the matvecs need it too).

| instance | direct (ms) | factorizations | stale-half @1e-12 (ms) | ratio | @1e-8 (ms) | ratio |
|---|---|---|---|---|---|---|
| lp-band-20000 | 256.7 | 21 → 11 | 1609.0 | **6.3×** | 1261.5 | 4.9× |
| qp-band-20000 | 37.9 | 9 → 5 | 332.8 | **8.8×** | 203.6 | 5.4× |
| issue10 | 191.1 | 25 → 13 | 502.4 | **2.6×** | 314.6 | 1.6× |
| chainsing-1000-1 | 177.8 | 11 → 6 | 238.2 | **1.3×** | 218.5 | 1.2× |
| nql30 | 76.9 | 16 → 8 | 342.7 | **4.5×** | 253.4 | 3.3× |
| socp-sumnorms-1500 | 1746.8 | 12 → 6 | 1629.8 | **0.93×** | 1441.9 | 0.83× |

Ratios > 1 mean the iterative scheme is slower. Halving the number of
factorizations costs 1.2–8.8× more time on five of six instances. The single
exception, socp-sumnorms-1500, gains 7% at 1e-12 and 17% at 1e-8 — only because
its factorization is worth 143 back-solves.

## 8. Verdict

**On these problem classes, no Krylov approach is competitive with the direct
solver, at any accuracy.** In order of strength:

1. **The per-iteration budget is too small.** A factorization costs 4–13
   back-solves on five of six instances. A preconditioner would have to deliver
   1e-12 in under ~10 iterations *and* cost less to build than a factorization.
   Nothing tested comes close.
2. **The matrices are not "nearly the same" between iterations.** `relΔK ≈
   0.52–0.99` at `k = 1`. Factorization reuse — the standard argument for Krylov
   inside an IPM — is not available: IR with a one-iteration-old exact factor
   diverges on every instance, and GMRES with the same factor needs 10–42
   iterations.
3. **Conditioning forbids cheap preconditioners.** `κ(K_t)` reaches 1e+13–1e+17.
   Anything short of a near-exact factorization cannot deliver 1e-8, let alone
   the 1e-12 the backend's refinement targets. Jacobi is actively harmful.
4. **The one place with headroom is a linear-algebra problem, not a Krylov
   problem.** socp-sumnorms-1500 is the only instance where an iterative scheme
   is not clearly worse, and only because QDLDL's scalar factorization there has
   24× fill and costs 162 ms. The measured fix for that instance is a
   supernodal/BLAS-3 factorization (roadmap item 1) — a 1.2–1.4× win with no
   accuracy risk — versus the iterative scheme's 1.07–1.2× with a convergence
   cliff at the late iterations.

**What a preconditioner would have to be** to change this answer:

- Applied in **≤ 3–5 back-solve-equivalents** and built in less than one
  factorization, so a 10-iteration Krylov solve stays inside budget. That means a
  preconditioner *sparser* than `L`, which on these low-fill instances does not
  exist.
- **Uniform in the barrier parameter**: not degrading as `FᵀF` spreads over 10–14
  orders of magnitude. The known constructions of that kind are constraint /
  block preconditioners built from the *structure* of `A` (Rees–Greif-type
  block-diagonal saddle-point preconditioners; multigrid or domain decomposition
  on a PDE-structured `A`). None of these six instances has such structure, and a
  general conic solver cannot assume it.
- **Robust at `κ ≈ 1e+16`**, which in double precision means an approximate
  inverse accurate to ~1e-8 on the relevant subspace — an exact factorization by
  another name.

What this study does **not** refute: problems far larger than anything here,
where `nnz(L)` grows superlinearly (dense-column LPs, PDE-constrained SOCPs, SDP
Schur complements) and one factorization costs thousands of back-solves. There
the §2 break-even column flips, and the socp-sumnorms-1500 row is the first hint
of it. The natural next experiment is not "a better preconditioner" but
"instances where `t_fac/t_bs > 500`". For the instances ConicIP is aimed at
today, the per-iteration effort belongs in the factorization — supernodal,
BLAS-3, threading — not in a Krylov loop.
