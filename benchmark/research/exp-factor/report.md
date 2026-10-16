# exp-factor — headroom in the KKT linear algebra of ConicIP (dev)

**Date** 2026-09-20 · **Code** worktree `.worktrees/large-scale-roadmap`, HEAD `97f1111`
(ConicIP v0.6.0 + private benchmark harness); Part A reads results measured on `3881ac2`.
**Machine** Apple M3 Pro, 12 cores (6 performance), macOS arm64-apple-darwin25.5.0.
**Julia** 1.12.7, 1 Julia thread, `libopenblas64_` (ILP64), BLAS threads as stated per row.
**Packages** QDLDL 0.4.1, SparseArrays 1.12.0 (CHOLMOD/UMFPACK from SuiteSparse_jll),
LDLFactorizations 0.10.2, MathOptInterface 1.53.0.
**Scratch environment** `exp-factor/env` — a copy of `benchmark/Project.toml`/`Manifest.toml`
with `Pkg.develop(path=<worktree>)`, `Pkg.add("LDLFactorizations")`, `Pkg.add(name="QDLDL", version="0.4.1")`.
No file inside the repository or any worktree was modified.

## Exact commands

```
# Part A (no solves; reads benchmark/results/phase-conicip-3881ac2.csv)
python3 exp-factor/partA.py                 # -> partA-out.md, partA-decomp.csv
python3 exp-factor/partD.py                 # -> partD-out.md  (ceiling iv)

# Part B (8 instances, each in a fresh process, from the worktree root)
for i in lp-band-20000 socp-sumnorms-1500 nb chainsing-1000-1 qssp30 \
         issue10 qp-band-200000 lp-band-200000; do
  timeout 900 julia --project=exp-factor/env exp-factor/capture.jl $i exp-factor
done
python3 exp-factor/partB_tables.py          # -> partB-out.md

# Part C
for i in socp-sumnorms-1500 lp-band-200000; do
  timeout 900 julia --project=exp-factor/env exp-factor/partC.jl $i exp-factor
done
```

`capture.jl` needs no `src/` change: `kktsolver_ldl` returns an `LDLGen` whose `.gen`
field is the closure `solve3x3gen`, and a Julia closure's captured variables are its
fields, so `getfield(gen.gen, :K)` and `getfield(gen.gen, :Fact)` hand over the
assembled quasi-definite `K` and the live QDLDL factorization. A wrapper callable is
passed as `conicIP(...; kktsolver = w)` (direct API) or
`MOI.set(opt, MOI.RawOptimizerAttribute("kktsolver"), w)` (file instances) — the MOI
wrapper's `_resolve_kktsolver` returns a non-`String` value unchanged. The matrix is
copied at factorization ⌊iters/2⌋.

## How to read the decomposition

`t_ldl_factor`, `t_ldl_solve`, `t_ldl_resid` are *inclusive diagnostics* over the whole
call (`src/timing.jl` contract), so the initial factorization and the initial-point
solve — charged to `t_init`, i.e. to "outside loop" — are counted in them as well. The
`Σparts − wall` column makes the overlap visible; it is below 3 ms everywhere except
socp-sumnorms-1500 (+0.123 s = exactly one factorization). "ref.resid" is
`t_ldl_resid + t_dir_refine_resid` (the backend's internal refinement residual plus the
outer `refine!` `step_residual!` evaluations; the two are disjoint code paths).
"kkt glue" is `t_kktupdate + t_direction` minus those three primitives, floored at 0.

# Part A — decomposition and ceilings (23 instances, dev 3881ac2)

## A1. Wall-time decomposition (seconds; LDL rows are inclusive diagnostics over the whole call)

| instance | wall | it | fact | solve | ref.resid | NT scal | loop_rest | kkt glue | outside loop | Σparts−wall |
|---|---|---|---|---|---|---|---|---|---|---|
| lp-band-200000 | 4.2810 | 23 | 0.8490 | 1.0937 | 0.9263 | 0.0269 | 0.4682 | 0.2190 | 0.6673 | -0.0306 |
| socp-sumnorms-1500 | 1.7666 | 12 | 1.6040 | 0.0507 | 0.0055 | 0.0036 | 0.0111 | 0.0000 | 0.2144 | +0.1227 |
| qp-band-200000 | 1.1271 | 9 | 0.0875 | 0.1407 | 0.1574 | 0.0075 | 0.1232 | 0.0343 | 0.5686 | -0.0079 |
| nb | 0.5788 | 19 | 0.2756 | 0.0835 | 0.1328 | 0.0030 | 0.0161 | 0.0005 | 0.0671 | -0.0002 |
| lp-band-20000 | 0.3420 | 21 | 0.0755 | 0.0925 | 0.0733 | 0.0025 | 0.0420 | 0.0175 | 0.0360 | -0.0028 |
| qssp30 | 0.2005 | 18 | 0.0472 | 0.0300 | 0.0203 | 0.0070 | 0.0211 | 0.0392 | 0.0354 | -0.0004 |
| nql30 | 0.1521 | 16 | 0.0271 | 0.0344 | 0.0189 | 0.0028 | 0.0087 | 0.0148 | 0.0452 | -0.0002 |
| issue10 | 0.1245 | 25 | 0.0065 | 0.0207 | 0.0252 | 0.0050 | 0.0180 | 0.0300 | 0.0186 | -0.0006 |
| sched_50_50_scaled | 0.1158 | 32 | 0.0306 | 0.0237 | 0.0197 | 0.0009 | 0.0153 | 0.0118 | 0.0133 | -0.0005 |
| chainsing-1000-1 | 0.1125 | 11 | 0.0047 | 0.0132 | 0.0072 | 0.0080 | 0.0218 | 0.0258 | 0.0314 | -0.0005 |
| qp-band-20000 | 0.0799 | 9 | 0.0069 | 0.0140 | 0.0120 | 0.0007 | 0.0131 | 0.0042 | 0.0281 | -0.0008 |
| lp-band-2000 | 0.0292 | 19 | 0.0065 | 0.0072 | 0.0058 | 0.0002 | 0.0037 | 0.0012 | 0.0044 | -0.0003 |
| socp-sumnorms-150 | 0.0071 | 10 | 0.0022 | 0.0010 | 0.0003 | 0.0003 | 0.0009 | 0.0008 | 0.0016 | -0.0000 |
| qp-band-2000 | 0.0070 | 8 | 0.0006 | 0.0012 | 0.0010 | 0.0001 | 0.0011 | 0.0002 | 0.0028 | -0.0001 |
| primal1 | 0.0045 | 13 | 0.0019 | 0.0006 | 0.0003 | 0.0000 | 0.0002 | 0.0000 | 0.0016 | +0.0001 |
| dual1 | 0.0038 | 12 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0002 | 0.0015 | 0.0020 | -0.0000 |
| sambal | 0.0013 | 14 | 0.0000 | 0.0000 | 0.0001 | 0.0000 | 0.0001 | 0.0009 | 0.0002 | -0.0000 |
| qpcblend | 0.0010 | 17 | 0.0001 | 0.0003 | 0.0002 | 0.0000 | 0.0001 | 0.0001 | 0.0002 | -0.0000 |
| cvxqp1_s | 0.0009 | 10 | 0.0001 | 0.0002 | 0.0001 | 0.0000 | 0.0001 | 0.0001 | 0.0003 | -0.0000 |
| qafiro | 0.0007 | 14 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0004 | 0.0002 | -0.0000 |
| hs76 | 0.0001 | 7 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | -0.0000 |
| hs21 | 0.0001 | 9 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | -0.0000 |
| hs35 | 0.0001 | 5 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0001 | -0.0000 |
| **sum** | 8.9368 |  | 3.0261 | 1.6075 | 1.4065 | 0.0685 | 0.7648 | 0.4023 | 1.7389 |
| **% of sum** | 100 |  | 33.9 | 18.0 | 15.7 | 0.8 | 8.6 | 4.5 | 19.5 |

## A2. Same, as a percentage of that instance's wall

| instance | wall s | fact | solve | ref.resid | NT scal | loop_rest | kkt glue | outside loop | Σparts−wall |
|---|---|---|---|---|---|---|---|---|---|
| lp-band-200000 | 4.2810 | 19.8 | 25.5 | 21.6 | 0.6 | 10.9 | 5.1 | 15.6 |
| socp-sumnorms-1500 | 1.7666 | 90.8 | 2.9 | 0.3 | 0.2 | 0.6 | 0.0 | 12.1 |
| qp-band-200000 | 1.1271 | 7.8 | 12.5 | 14.0 | 0.7 | 10.9 | 3.0 | 50.4 |
| nb | 0.5788 | 47.6 | 14.4 | 22.9 | 0.5 | 2.8 | 0.1 | 11.6 |
| lp-band-20000 | 0.3420 | 22.1 | 27.0 | 21.4 | 0.7 | 12.3 | 5.1 | 10.5 |
| qssp30 | 0.2005 | 23.5 | 14.9 | 10.1 | 3.5 | 10.5 | 19.6 | 17.7 |
| nql30 | 0.1521 | 17.8 | 22.6 | 12.4 | 1.8 | 5.7 | 9.7 | 29.7 |
| issue10 | 0.1245 | 5.2 | 16.6 | 20.2 | 4.0 | 14.4 | 24.1 | 15.0 |
| sched_50_50_scaled | 0.1158 | 26.4 | 20.5 | 17.0 | 0.8 | 13.2 | 10.2 | 11.5 |
| chainsing-1000-1 | 0.1125 | 4.2 | 11.7 | 6.4 | 7.1 | 19.4 | 23.0 | 27.9 |
| qp-band-20000 | 0.0799 | 8.7 | 17.5 | 15.0 | 0.9 | 16.4 | 5.3 | 35.2 |
| lp-band-2000 | 0.0292 | 22.1 | 24.7 | 19.8 | 0.7 | 12.6 | 4.1 | 15.0 |
| socp-sumnorms-150 | 0.0071 | 30.4 | 13.8 | 4.2 | 4.2 | 12.3 | 11.6 | 23.2 |
| qp-band-2000 | 0.0070 | 8.7 | 16.8 | 14.4 | 0.8 | 15.2 | 3.2 | 39.8 |
| primal1 | 0.0045 | 42.1 | 12.5 | 6.2 | 0.1 | 5.1 | 0.0 | 35.9 |
| dual1 | 0.0038 | 0.0 | 0.0 | 3.1 | 0.2 | 4.4 | 38.5 | 53.3 |
| sambal | 0.0013 | 0.0 | 0.0 | 6.3 | 3.0 | 8.8 | 66.0 | 15.1 |
| qpcblend | 0.0010 | 12.9 | 25.3 | 17.3 | 0.7 | 9.9 | 8.6 | 23.4 |
| cvxqp1_s | 0.0009 | 16.1 | 17.7 | 13.0 | 0.6 | 8.5 | 6.4 | 35.9 |
| qafiro | 0.0007 | 0.0 | 0.0 | 2.5 | 0.6 | 5.2 | 61.2 | 29.2 |
| hs76 | 0.0001 | 0.0 | 0.0 | 1.8 | 1.4 | 5.8 | 24.3 | 63.8 |
| hs21 | 0.0001 | 0.0 | 0.0 | 2.2 | 1.4 | 7.7 | 30.3 | 54.3 |
| hs35 | 0.0001 | 0.0 | 0.0 | 1.4 | 1.3 | 5.1 | 19.1 | 70.8 |

## A3. Ceilings: predicted wall and saving

| instance | wall | (i) fact/2 | save % | (ii) solve+ref/2 | save % | (iii) only kktupdate+direction | save % | largest |
|---|---|---|---|---|---|---|---|---|
| lp-band-200000 | 4.2810 | 3.8565 | 9.9 | 3.2710 | 23.6 | 3.0880 | 27.9 | iii |
| socp-sumnorms-1500 | 1.7666 | 0.9646 | 45.4 | 1.7385 | 1.6 | 1.5373 | 13.0 | i |
| qp-band-200000 | 1.1271 | 1.0833 | 3.9 | 0.9780 | 13.2 | 0.4200 | 62.7 | iii |
| nb | 0.5788 | 0.4410 | 23.8 | 0.4707 | 18.7 | 0.4925 | 14.9 | i |
| lp-band-20000 | 0.3420 | 0.3043 | 11.0 | 0.2591 | 24.2 | 0.2588 | 24.3 | iii |
| qssp30 | 0.2005 | 0.1769 | 11.8 | 0.1754 | 12.5 | 0.1366 | 31.9 | iii |
| nql30 | 0.1521 | 0.1386 | 8.9 | 0.1254 | 17.5 | 0.0952 | 37.4 | iii |
| issue10 | 0.1245 | 0.1213 | 2.6 | 0.1016 | 18.4 | 0.0824 | 33.9 | iii |
| sched_50_50_scaled | 0.1158 | 0.1006 | 13.2 | 0.0941 | 18.7 | 0.0858 | 25.9 | iii |
| chainsing-1000-1 | 0.1125 | 0.1102 | 2.1 | 0.1024 | 9.0 | 0.0509 | 54.8 | iii |
| qp-band-20000 | 0.0799 | 0.0764 | 4.3 | 0.0669 | 16.3 | 0.0372 | 53.5 | iii |
| lp-band-2000 | 0.0292 | 0.0260 | 11.1 | 0.0227 | 22.2 | 0.0207 | 29.2 | iii |
| socp-sumnorms-150 | 0.0071 | 0.0060 | 15.2 | 0.0065 | 9.0 | 0.0043 | 40.0 | iii |
| qp-band-2000 | 0.0070 | 0.0066 | 4.3 | 0.0059 | 15.6 | 0.0030 | 57.0 | iii |
| primal1 | 0.0045 | 0.0036 | 21.0 | 0.0041 | 9.3 | 0.0026 | 41.5 | iii |
| dual1 | 0.0038 | 0.0038 | 0.0 | 0.0037 | 1.6 | 0.0016 | 58.4 | iii |
| sambal | 0.0013 | 0.0013 | 0.0 | 0.0013 | 3.1 | 0.0010 | 27.7 | iii |
| qpcblend | 0.0010 | 0.0009 | 6.5 | 0.0008 | 21.3 | 0.0006 | 35.8 | iii |
| cvxqp1_s | 0.0009 | 0.0008 | 8.1 | 0.0008 | 15.3 | 0.0005 | 46.8 | iii |
| qafiro | 0.0007 | 0.0007 | 0.0 | 0.0007 | 1.3 | 0.0004 | 36.3 | iii |
| hs76 | 0.0001 | 0.0001 | 0.0 | 0.0001 | 0.9 | 0.0000 | 74.0 | iii |
| hs21 | 0.0001 | 0.0001 | 0.0 | 0.0001 | 1.1 | 0.0000 | 67.5 | iii |
| hs35 | 0.0001 | 0.0001 | 0.0 | 0.0001 | 0.7 | 0.0000 | 79.5 | iii |

## A4. Aggregate

| aggregate | baseline | (i) fact/2 | (ii) solve+ref/2 | (iii) kkt-only |
|---|---|---|---|---|
| sum (s) | 8.937 | 7.424 (-16.9%) | 7.430 (-16.9%) | 6.319 (-29.3%) |
| sgm, shift 0.1 (s) | 0.1194 | 0.1072 (-10.2%) | 0.1065 (-10.9%) | 0.0900 (-24.6%) |

## A5. Per-family saving (sum of seconds saved)

| family | n | wall | (i) | (ii) | (iii) |
|---|---|---|---|---|---|
| lp-band | 3 | 4.652 | 0.466 (10%) | 1.099 (24%) | 1.285 (28%) |
| socp-sumnorms | 2 | 1.774 | 0.803 (45%) | 0.029 (2%) | 0.232 (13%) |
| qp-band | 3 | 1.214 | 0.048 (4%) | 0.163 (13%) | 0.754 (62%) |
| cblib | 5 | 1.160 | 0.193 (17%) | 0.192 (17%) | 0.299 (26%) |
| issue10 | 1 | 0.125 | 0.003 (3%) | 0.023 (18%) | 0.042 (34%) |
| maros-meszaros | 9 | 0.013 | 0.001 (9%) | 0.001 (7%) | 0.006 (46%) |

---

# Part B — measured factorization back ends on the captured KKT matrices

## B1. Captured KKT matrix (mid-run factorization)

| instance | n_K | nnz(triu K) | nnz(L) QDLDL | fill nnz(L)/nnz(K) | flops Σcolcount² | max supernode chain | flop frac in chains ≥16 | CHOLMOD nsuper (forced) | CHOLMOD auto picks |
|---|---|---|---|---|---|---|---|---|---|
| lp-band-20000 | 80000 | 339970 | 439927 | 1.29 | 4.04e+06 | 10 | 0.00 | 42488 | simplicial |
| lp-band-200000 | 800000 | 3399970 | 4399927 | 1.29 | 4.04e+07 | 10 | 0.00 | 424988 | simplicial |
| qp-band-200000 | 600000 | 1599994 | 999994 | 0.62 | 2.2e+06 | 1 | 0.00 | 412500 | simplicial |
| socp-sumnorms-1500 | 14000 | 34993 | 841528 | 24.05 | 6.49e+08 | 668 | 0.43 | 1524 | supernodal |
| nb | 4889 | 201170 | 398498 | 1.98 | 4.78e+07 | 123 | 0.01 | 798 | supernodal |
| chainsing-1000-1 | 31940 | 73858 | 52897 | 0.72 | 9.48e+04 | 3 | 0.00 | 3992 | simplicial |
| qssp30 | 15038 | 67016 | 159972 | 2.39 | 7.11e+06 | 96 | 0.64 | 1961 | supernodal |
| issue10 | 17130 | 39150 | 31065 | 0.79 | 6.83e+04 | 2 | 0.00 | 2119 | simplicial |

## B2. Numeric factorization time, ms (best of 5, warmed, 1 Julia thread)

| instance | QDLDL refactor! | CHOLMOD ldlt!(K) 1thr | CHOLMOD ldlt!(K) 6thr | CHOLMOD chol! proxy auto 1thr | CHOLMOD chol! proxy auto 6thr | CHOLMOD chol! proxy SUPER 1thr | CHOLMOD chol! proxy SUPER 6thr | UMFPACK lu! 1thr | UMFPACK lu! 6thr | LDLFactorizations |
|---|---|---|---|---|---|---|---|---|---|---|
| lp-band-20000 | 3.75 | 9.74 | 9.90 | 8.98 | 8.87 | 11.40 | 11.35 | 35.00 | 33.98 | 4.93 |
| lp-band-200000 | 36.97 | 88.31 | 88.52 | 86.35 | 87.05 | 110.77 | 110.77 | 296.26 | 365.45 | 50.24 |
| qp-band-200000 | 8.36 | 27.47 | 27.29 | 26.11 | 28.06 | 73.21 | 73.60 | 174.31 | 190.95 | 16.42 |
| socp-sumnorms-1500 | 142.38 | 147.66 | 146.35 | 22.97 | 10.52 | 23.74 | 12.70 | 169.93 | 169.23 | 140.73 |
| nb | 15.98 | 19.78 | 19.82 | 11.81 | 11.79 | 11.85 | 11.81 | 27.46 | 37.49 | 19.18 |
| chainsing-1000-1 | 0.48 | 1.36 | 1.32 | 1.38 | 1.37 | 2.59 | 2.57 | 8.03 | 7.95 | 0.00 |
| qssp30 | 2.82 | 4.12 | 4.05 | 2.82 | 2.65 | 2.81 | 2.68 | 10.69 | 10.21 | 3.36 |
| issue10 | 0.26 | 0.79 | 0.77 | 0.79 | 0.79 | 1.67 | 1.63 | 6.06 | 5.91 | 0.66 |

## B3. Best BLAS-3 speedup over QDLDL on the same pattern

| instance | QDLDL ms | best CHOLMOD-Cholesky ms (threads) | speedup | nnz(L) CHOLMOD / QDLDL |
|---|---|---|---|---|
| lp-band-20000 | 3.75 | 8.87 (6) | 0.42× | 1.18 |
| lp-band-200000 | 36.97 | 86.35 (1) | 0.43× | 1.18 |
| qp-band-200000 | 8.36 | 26.11 (1) | 0.32× | 1.60 |
| socp-sumnorms-1500 | 142.38 | 10.52 (6) | 13.53× | 1.42 |
| nb | 15.98 | 11.79 (6) | 1.36× | 1.53 |
| chainsing-1000-1 | 0.48 | 1.37 (6) | 0.35× | 1.60 |
| qssp30 | 2.82 | 2.65 (6) | 1.06× | 1.81 |
| issue10 | 0.26 | 0.79 (6) | 0.34× | 1.55 |

## B4. One back-solve, ms

| instance | QDLDL solve! | CHOLMOD \ simplicial proxy | CHOLMOD \ supernodal proxy | CHOLMOD \ ldlt(K) | UMFPACK \ |
|---|---|---|---|---|---|
| lp-band-20000 | 0.924 | 1.129 | 3.710 | 1.075 | 0.918 |
| lp-band-200000 | 9.159 | 11.196 | 36.306 | 10.041 | 8.622 |
| qp-band-200000 | 4.257 | 5.004 | 31.716 | 4.036 | 5.040 |
| socp-sumnorms-1500 | 1.135 | 0.849 | 0.771 | 0.730 | 1.739 |
| nb | 0.475 | 0.344 | 0.346 | 0.552 | 0.482 |
| chainsing-1000-1 | 0.287 | 0.384 | 0.906 | 0.257 | 0.313 |
| qssp30 | 0.313 | 0.416 | 0.407 | 0.300 | 0.287 |
| issue10 | 0.140 | 0.155 | 0.546 | 0.128 | 0.184 |

### What was timed, precisely

1. **QDLDL `refactor!`** — exactly what ConicIP does: `qdldl(K; perm = <ConicIP's pattern
   ordering>, Dsigns = <ConicIP's signs>, regularize_eps = 1e-13, regularize_delta = 2e-7)`
   once, then `refactor!(Fact)` timed. Numeric phase only; QDLDL is pure Julia and scalar.
2. **CHOLMOD `ldlt!(F, Symmetric(K))`** — CHOLMOD's *simplicial* LDLᵀ applied to the
   genuine quasi-definite `K`. It did **not** fail on any of the eight matrices: with
   ConicIP's static regularization the signed pivots survive without pivoting. Symbolic
   analysis done once, numeric phase timed. (`nnz(sparse(F.L))` throws for an LDLt
   factor in this SparseArrays version — `"supported only for :LD"` — so the fill of
   this factor is not reported; the timings are unaffected.)
3. **CHOLMOD `cholesky!` on an SPD proxy with the identical sparsity pattern.** The
   proxy is `|K|` off the diagonal with diagonal entry `1 + Σ_j |K_ij|` — same pattern
   as `K`, symmetric, strictly diagonally dominant, hence SPD with no pivoting needed.
   This is the honest upper bound on what a supernodal BLAS-3 *signed* Cholesky of the
   quasi-definite matrix could reach, because a quasi-definite LDLᵀ with a fixed pivot
   order does exactly the same arithmetic. Reported twice: with CHOLMOD's own choice
   ("auto"; `Common.supernodal = 1`) and with supernodal forced (`= 2`). CHOLMOD picks
   its own fill-reducing ordering, which here gives 1.18–1.81× **more** nonzeros in L
   than ConicIP's ordering — so the speedups below are, if anything, understated per flop.
4. **UMFPACK `lu!(F, K)`** — numeric refactorization with the symbolic analysis reused.
5. **LDLFactorizations 0.10.2** `ldl_analyze` once, `ldl_factorize!` timed. It is a pure-
   Julia scalar LDLᵀ like QDLDL and lands within 10–30% of it, except on chainsing-1000-1
   where the measured 0.00 ms is not credible and should be ignored.
6. **Back-solve**, one dense RHS, each factorization's own triangular solve.

### Reading of Part B

* **Fill is the whole story.** Seven of the eight matrices have fill ratio `nnz(L)/nnz(triu K)`
  between 0.62 and 2.4, a longest identical-pattern column chain of 1–123, and **0–1% of
  the flop proxy in chains of ≥ 16 columns**. There is no dense structure for BLAS-3 to
  exploit, CHOLMOD's own heuristic agrees (it selects simplicial on five of them), and
  forcing supernodal makes things 1.8–2.8× *worse* (qp-band-200000: 26 → 73 ms). On these,
  scalar QDLDL is the fastest thing measured — 2.3–6.4× faster than CHOLMOD's simplicial
  LDLᵀ of the same matrix and 9–23× faster than UMFPACK.
* **socp-sumnorms-1500 is the exception and it is dramatic.** Fill ratio 24, longest chain
  668, 43% of the flop proxy in chains ≥ 16, CHOLMOD chooses supernodal unprompted.
  QDLDL 142.4 ms against 23.0 ms (1 BLAS thread) and 10.5 ms (6 BLAS threads) — **6.2×
  and 13.5×** — while CHOLMOD is carrying 42% more nonzeros in L. Per unit of the flop
  proxy that is roughly a 12× and 27× higher rate: QDLDL runs this factorization at about
  4.6 G(proxy-flop)/s, the supernodal code at about 57 and 124.
* **Threading buys nothing anywhere else.** 6 BLAS threads change no other row by more
  than noise, because the kernels are scalar sparse loops, not BLAS-3 calls. Even
  UMFPACK is flat or slightly worse with 6 threads.
* **Back-solves have no headroom.** QDLDL's triangular solve is within ±40% of CHOLMOD's
  and UMFPACK's everywhere, and a supernodal factor's back-solve is 2–7× *slower* on the
  low-fill matrices. Ceiling (ii) is therefore not reachable by changing the back end;
  it would have to come from *fewer* solves and residual evaluations.

---

# Part C — assembly vs QDLDL numeric phase inside `kktupdate`

Two independent measurements per instance: the package's own `timing = PhaseTimes()`
counters, and a micro-benchmark that calls the captured generator
`solve3x3gen(F, F⁻ᵀ)` (value update + numeric refactorization) against `refactor!(Fact)`
alone (numeric only), best of 7 each.

| instance | t_loop | t_kktupdate | t_ldl_factor (whole call) | n_factorizations | `solve3x3gen` | `refactor!` | assembly | assembly share |
|---|---|---|---|---|---|---|---|---|
| socp-sumnorms-1500 | 1528.3 ms | 1448.7 ms | 1572.8 ms | 12 | 135.43 ms | 134.56 ms | **0.88 ms** | **0.6%** |
| lp-band-200000 | 3797.2 ms | 840.3 ms | 833.1 ms | 23 | 35.89 ms | 34.06 ms | **1.83 ms** | **5.1%** |

The phase counters agree: `n_kktupdate` counts the initial factorization but
`t_kktupdate` does not contain it (it is inside `t_init`, 133.9 ms ≈ one 135 ms
factorization on socp-sumnorms-1500; 70.8 ms ≈ one 36 ms factorization plus other init
work on lp-band-200000). Netting that out gives an assembly share of ~0.5% and ~5%,
matching the micro-benchmark.

**Conclusion.** The KKT value update / assembly is already negligible. Anything spent on
a faster `K` assembly — cheaper NT-block writes, better `update_values!` indexing — is
capped at 1–2 ms per factorization. `kktupdate` *is* the numeric factorization.

---

## D. Ceiling (iv): measured supernodal/BLAS-3 factorization, applied to Part A

| instance | wall s | factorization s | measured speedup 1thr | wall after 1thr | measured speedup 6thr | wall after 6thr | saving 6thr % |
|---|---|---|---|---|---|---|---|
| lp-band-200000 | 4.2810 | 0.8490 | 1.00× | 4.2810 | 1.00× | 4.2810 | 0.0 |
| socp-sumnorms-1500 | 1.7666 | 1.6040 | 6.20× | 0.4214 | 13.53× | 0.2811 | 84.1 |
| qp-band-200000 | 1.1271 | 0.0875 | 1.00× | 1.1271 | 1.00× | 1.1271 | 0.0 |
| nb | 0.5788 | 0.2756 | 1.35× | 0.5069 | 1.36× | 0.5066 | 12.5 |
| lp-band-20000 | 0.3420 | 0.0755 | 1.00× | 0.3420 | 1.00× | 0.3420 | 0.0 |
| qssp30 | 0.2005 | 0.0472 | 1.00× | 0.2005 | 1.06× | 0.1977 | 1.4 |
| nql30 | 0.1521 | 0.0271 | 1.00× | 0.1521 | 1.00× | 0.1521 | 0.0 |
| issue10 | 0.1245 | 0.0065 | 1.00× | 0.1245 | 1.00× | 0.1245 | 0.0 |
| sched_50_50_scaled | 0.1158 | 0.0306 | 1.00× | 0.1158 | 1.00× | 0.1158 | 0.0 |
| chainsing-1000-1 | 0.1125 | 0.0047 | 1.00× | 0.1125 | 1.00× | 0.1125 | 0.0 |
| qp-band-20000 | 0.0799 | 0.0069 | 1.00× | 0.0799 | 1.00× | 0.0799 | 0.0 |
| lp-band-2000 | 0.0292 | 0.0065 | 1.00× | 0.0292 | 1.00× | 0.0292 | 0.0 |
| socp-sumnorms-150 | 0.0071 | 0.0022 | 1.00× | 0.0071 | 1.00× | 0.0071 | 0.0 |
| qp-band-2000 | 0.0070 | 0.0006 | 1.00× | 0.0070 | 1.00× | 0.0070 | 0.0 |
| primal1 | 0.0045 | 0.0019 | 1.00× | 0.0045 | 1.00× | 0.0045 | 0.0 |
| qpcblend | 0.0010 | 0.0001 | 1.00× | 0.0010 | 1.00× | 0.0010 | 0.0 |
| cvxqp1_s | 0.0009 | 0.0001 | 1.00× | 0.0009 | 1.00× | 0.0009 | 0.0 |

| aggregate | baseline | BLAS-3 1 thread | BLAS-3 6 threads |
|---|---|---|---|
| sum (s) | 8.937 | 7.520 (-15.9%) | 7.376 (-17.5%) |
| sgm 0.1 (s) | 0.1194 | 0.1066 (-10.8%) | 0.1037 (-13.2%) |

Ceiling (iv) applies the **measured** Part B speedup to each instance's Part A
factorization time; the 15 instances that were not measured get a factor of 1.0 (no
credit), and a measured factor below 1 is clipped to 1 (you would not switch back ends
where the scalar code wins).

---

# Verdict

**How much end-to-end time could a BLAS-3 factorization, threading, or cheaper solves
plausibly save on this 23-instance suite?**

| lever | basis | sum, 8.937 s | sgm (shift 0.1), 0.1194 s |
|---|---|---|---|
| supernodal/BLAS-3 factorization, 1 BLAS thread | **measured** (Part B) | 7.520 s (**−15.9%**) | 0.1066 s (−10.8%) |
| supernodal/BLAS-3 factorization, 6 BLAS threads | **measured** | 7.376 s (**−17.5%**) | 0.1037 s (−13.2%) |
| factorization halved everywhere (hypothetical) | ceiling (i) | 7.424 s (−16.9%) | 0.1072 s (−10.2%) |
| back-solves + refinement residuals halved | ceiling (ii) | 7.430 s (−16.9%) | 0.1065 s (−10.9%) |
| everything outside `kktupdate + direction` free | ceiling (iii) | 6.319 s (**−29.3%**) | 0.0900 s (**−24.6%**) |

**Which ceiling is largest, and where.** Ceiling (iii) — deleting every second that is
not `kktupdate + direction` — is the largest on 22 of 23 instances and in aggregate. The
exception is socp-sumnorms-1500, the only instance where the factorization dominates
(91% of wall). Ceiling (i) is largest on the two high-fill instances
(socp-sumnorms-1500 45%, nb 24%); ceiling (ii) never wins outright but is worth 22–27%
of wall on the whole banded-LP family, where `t_ldl_solve + refinement residuals` is
1.6× the factorization time.

**But the measured version of ceiling (i) is almost entirely one instance.** Of the
1.56 s that a real supernodal back end would save at 6 threads, **1.486 s (95%) comes
from socp-sumnorms-1500 alone** and 0.073 s from nb; qssp30 contributes 3 ms and the
other 20 instances contribute nothing (their measured factor is ≤ 1). Remove
socp-sumnorms-1500 and the whole "specialized linear algebra" lever is worth 1.1% of the
remaining 7.17 s.

**Specifically:**

1. **Supernodal / BLAS-3 is not the next level for this suite as it stands.** It is a
   6–13× win on exactly one instance family (high-fill SOCPs, the sum-of-norms problems,
   fill ratio 24) and a 2–3× *loss* everywhere else. A defensible product decision is a
   *dispatch*: keep QDLDL, and route to a supernodal back end when the symbolic analysis
   predicts fill above a threshold (CHOLMOD's own `supernodal_switch = 40` — the ratio of
   flops to nnz(L) — is a ready-made and well-tested criterion, and its "auto" choice
   agreed with the measurements on all eight matrices). That is a bounded piece of work
   and it converts socp-sumnorms-1500 from 1.77 s to ~0.28 s.
2. **Threading is worth nothing without supernodes.** Every 6-thread number here equals
   its 1-thread number except on the one supernodal matrix (23.0 → 10.5 ms). Threading a
   scalar sparse LDLᵀ is a separate and much harder project (task-parallel etree
   scheduling), and the etrees here are long thin chains — max chain length 1–10 on the
   banded families — which is the worst case for it.
3. **The back-solve back end has no headroom at all** (±40% across four libraries).
   Ceiling (ii) is real money — 1.1 s of the suite, 24% of the banded-LP family — but it
   has to be bought with *fewer* refinement iterations and residual evaluations, not with
   faster triangular solves. `t_ldl_solve + t_ldl_resid + t_dir_refine_resid` is 34% of
   the suite total against 34% for the factorization: ConicIP currently spends as much
   time checking and correcting its solves as computing them.
4. **The single largest block of recoverable time is not in the KKT solver at all.**
   "Outside loop" (frontend, presolve, equilibrate, setup, init, postsolve) is 19.5% of
   the suite total and 50% of qp-band-200000, 36% of qp-band-20000, 40% of
   qp-band-2000, 29% of chainsing-1000-1, 30% of nql30; `loop_rest`
   (residual + rhs + line search) is another 8.6%, and "kkt glue" 4.5%.
5. **The assembly of `K` is already free** (0.6% and 5.1% of `kktupdate`), so there is
   nothing to win between the NT scaling and the factorization.

**Recommendation.** "Specialized linear algebra for the Newton solves" is *not* where
the next level is for this suite, with one caveat worth acting on: a fill-triggered
supernodal path. Its whole measured value is 1.56 s of 8.94 s, and 95% of that is one
instance. The suite's factorizations are, on the evidence of Part B, already at or ahead
of the state of the art for *scalar* sparse LDLᵀ (QDLDL beats CHOLMOD's simplicial LDLᵀ
of the same matrix by 2.3–6.4× on seven of eight matrices). If the goal is a clearly
better solver on this class of problems, the evidence points at the iteration count and
the per-iteration *non-factorization* work — the refinement policy, the residual/rhs/
line-search loop, and the whole-call overhead — not at the factorization kernel. The one
caveat is that this suite may under-represent the problems where BLAS-3 matters: every
instance with real fill (socp-sumnorms-1500, and by extension any SDP or dense-column
problem) shows a large win, and SDP is exactly where ConicIP is currently far from scale.
A suite with more high-fill instances would shift this verdict.

## Files

- `partA.py`, `partA-out.md`, `partA-decomp.csv` — Part A decomposition and ceilings (i)–(iii)
- `capture.jl`, `partB-<instance>.csv`, `partB-<instance>.stats`, `partB_tables.py`, `partB-out.md` — Part B
- `partC.jl`, `partC-<instance>.txt` — Part C
- `partD.py`, `partD-out.md` — ceiling (iv), measured speedups applied to Part A
- `probe.jl` — the field-introspection probe used to find `LDLGen.gen.K` / `.Fact`
- `env/` — scratch Julia environment (repo Project/Manifest copy + LDLFactorizations + QDLDL)
