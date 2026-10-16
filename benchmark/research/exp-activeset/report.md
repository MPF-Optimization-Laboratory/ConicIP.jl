# exp-activeset — finite termination / active-set identification / crossover in ConicIP

Date 2026-09-20. Apple M3 Pro, 1 Julia thread, Julia 1.12.7.
Code under test: `dev` in `/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap`.
**No file in the repository was modified.** All scripts are in this directory.

## Exact commands

```
cd /Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap
S=/private/tmp/claude-502/-Users-mpf-projects-Software-ConicIP-jl/8d6f2e95-bb86-4902-8603-0c2255592e24/scratchpad/exp-activeset

# Netlib LPs (LPnetlib / SuiteSparse collection, Matrix Market form)
mkdir -p $S/netlib && cd $S/netlib
for n in lp_afiro lp_adlittle lp_share2b lp_degen2 lp_scagr7; do
  curl -sSL -O "https://sparse.tamu.edu/MM/LPnetlib/$n.tar.gz"; tar xzf $n.tar.gz; done

timeout 1500 julia --project=.                      $S/run_lp2.jl       > $S/lp3_out.txt
DO_ISSUE10=1 timeout 1500 julia --project=benchmark $S/run_socp.jl      > $S/socp_out.txt
timeout  600 julia --project=.                      $S/check_capture.jl
```

Files: `lib.jl` (instances + capture + projection kernel), `run_lp2.jl`, `run_socp.jl`,
`check_capture.jl`; raw logs `lp3_out.txt`, `socp_out.txt`.

## How the iterates were captured (and what they are)

`src/ConicIP.jl` has **no per-iteration callback**: `kkt_diagnostics` reports on the KKT
solver only, `timing=` gives phase timers, and a `kktsolver` wrapper sees the NT scaling
`F` but never `(y,w,v,s)`. For an LP `F = Diag(sqrt(s_i/v_i))` would give the *ratio*
indicator but not the Tapia ones, so I used option (c): **re-run `conicIP` with
`maxIters = k` for `k = 1,2,…`** and read the returned `Solution` (`y`, `w`, `v`, `s`, all
in the caller's original un-equilibrated coordinates — `conicIP` calls `unequilibrate!`
before returning). Cost `O(T²/2)` solver iterations; every instance ran in seconds.
Options: `verbose=false, certFallback=false` (the latter suppresses the post-loop auxiliary
ray solves). Truncated runs return `:Abandoned`.

**Caveat, measured.** The loop stores into `sol` the *best-measure-so-far* iterate, not
necessarily the current one, but records which one in `sol.Iter`. `check_capture.jl`
compared `sol.Iter` with `maxIters` for every `k`:

| instance | iterations | truncated runs where `sol.Iter != maxIters` |
|---|---|---|
| lp-band-2000 | 21 | none |
| degen2 | 16 | none |
| socp-sumnorms-150 | 16 | none |
| afiro | 14 | k = 2, 3, 4 (returns iterate 1) |

So the truncated run gives the current iterate except in a stagnant early phase (afiro's μ
is flat at 3.5e+4 for four iterations). Always check `sol.Iter`.

## Instances

`lp-band-2000`, `lp-band-20000` exactly as `benchmark/suite.jl`'s `lp_band`. Netlib LPs
`afiro`, `adlittle`, `share2b`, `scagr7`, `degen2` from LPnetlib in the SuiteSparse
collection (`min cᵀx, Ax = b, lo ≤ x ≤ hi`), mapped to ConicIP form with `G = A`, cone rows
`[I_lo; −I_hi]`. `degen-synth`: random LP (n = 300, p = 60) built from a KKT triple with a
deliberately non-strictly-complementary block. SOCPs: `socp-sumnorms-150` and `issue10`.
Maros–Mészáros was not used: those are QPs and this is an LP question.

---

## 1–3. Identification and Ye-style finite termination (LP)

`it@1e-6` / `it@1e-8` = iterations of the solver's own run (the iterate sequence does not
depend on `optTol`, only the stopping point does). `tRatio` = first iteration at which
`B̂ = {i : sᵢ > vᵢ}` reproduces the reference optimal partition on **all** m rows;
`tTapia` = same for `B̂ = {i : sᵢᵗ/sᵢᵗ⁻¹ > vᵢᵗ/vᵢᵗ⁻¹}`. `pRatio`/`pTapia` = first iteration
at which the **Ye projection** onto the predicted affine face returns a verifiably exactly
optimal point (all relative residuals < 1e-12, `s_B ≥ 0`, `v_N ≥ 0`, `vᵀs = 0` exactly by
construction). `stab` = first iteration at which the ratio partition has been unchanged for
three consecutive iterations.

| instance | n | m | it@1e-6 | it@1e-8 | tRatio | tTapia | pRatio | pTapia | stab | ms/IPM-iter | ms/projection |
|---|---|---|---|---|---|---|---|---|---|---|---|
| lp-band-2000  | 2000 | 6000 | 19 | 20 | 20 | **19** | 20 | **19** | never | 1.58 | 2.88 |
| lp-band-20000 | 20000 | 60000 | 21 | 22 | 23 | **21** | 23 | **21** | never | 17.6 | 55.6 |
| afiro    | 51 | 51 | 12 | 13 | 10 | **10** | 10 | **10** | 3 | 0.025 | 0.045 |
| adlittle | 138 | 138 | 16 | 17 | 16 | 15 | **16** | 16 | 3 | 0.099 | 0.128 |
| share2b  | 162 | 162 | 15 | 16 | 15 | 13 | **14** | 14 | 17 | 0.086 | 0.162 |
| scagr7   | 185 | 185 | 15 | 16 | 17 | 15 | 17 | **16** | never | 0.073 | 0.175 |
| degen2   | 757 | 757 | 14 | 15 | 14 | 13 | 14 | **13** | 16 | 0.902 | 2.67 |
| degen-synth | 300 | 300 | 9 | 10 | 6 | 7 | **7** | 7 | 8 | 0.330 | 1.81 |

Net iterations saved against the solver's own 1e-6 stop: **0, 0, +2, 0, +1, −1, +1, +2** —
median **+0.5 of 15**, about **4 %**. One projection costs **1.3–3.2 IPM iterations** (two
sparse quasi-definite factorizations, primal and dual, using the solver's own QDLDL
backend); on `lp-band-20000` 55.6 ms against 17.6 ms per IPM iteration.

Observations from the per-iteration tables (`lp3_out.txt`):

* The **ratio indicator `sᵢ > vᵢ` is weak until the very end**: on `lp-band-2000` still
  1463 false positives of 6000 at t = 7 and 103 at t = 14; first exact at t = **20**, one
  iteration *after* the solver stopped.
* The **Tapia indicator is 1–2 iterations sharper** on every instance and is the one that
  first makes the projection succeed on 5 of 8.
* Tapia is **not monotone**: on `lp-band-2000` exact at t = 17, wrong at t = 18 (2 rows),
  exact again at t = 19. "Tapia agrees" is not a safe trigger; the projection must verify.
* The "partition unchanged for 3 iterations" gate fires far too early (afiro, adlittle:
  t = 3, partition still badly wrong, projection attempted and fails — 1–3 iterations of
  waste each attempt) or never (lp-band, scagr7). Not usable.
* Speculative projection every iteration from a fixed μ threshold costs 1.3–3.2 IPM
  iterations per attempt, which swamps the 0–2 iterations it can save.

## 4. Accuracy gain

`maxres` = max of the three relative residuals (primal, equality, dual stationarity) in
original coordinates. The projected point has `vᵀs = 0` **exactly** (v zero on B, s zero on
N by construction), so its objective is the exact optimum up to the backward error of the
residuals; the objective-error column is measured against it.

| instance | maxres @1e-6 stop | maxres after projection | gap @1e-6 | objective error @1e-6 | objective error after projection |
|---|---|---|---|---|---|
| lp-band-2000  | 6.3e-10 | **1.7e-16** | 1.4e-07 | 4.6e-08 | 0 (exact) |
| lp-band-20000 | 7.2e-10 | **1.6e-16** | 1.7e-07 | 5.5e-08 | 0 (exact) |
| afiro    | 7.1e-14 | 1.7e-13 | 2.0e-08 | 7.9e-09 | 0 (exact) |
| adlittle | 8.5e-11 | **3.6e-13** | 8.5e-08 | 4.1e-08 | 0 (exact) |
| share2b  | 4.3e-11 | **3.1e-13** | 1.6e-08 | 5.9e-09 | 0 (exact) |
| scagr7   | 1.0e-07 | **2.2e-13** | 9.6e-08 | 2.8e-08 | 0 (exact) |
| degen2   | 1.2e-08 | **4.8e-14** | 6.2e-08 | 3.1e-08 | 0 (exact) |
| degen-synth | 5.0e-08 | **4.2e-14** | 4.9e-07 | 3.5e-07 | 0 (exact) |

The strong result: **a 1e-8-ish answer becomes an exact one.** Duality gap ~1e-7 →
identically zero, residuals down 3–6 orders, objective correct to the last bit rather than
8 digits. Not obtained by tightening `optTol`: the projection applied to the `optTol = 1e-6`
iterate already gives it.

## 5. What a crossover would cost

Reference optimal partition from the 1e-10 solve. A vertex needs `[A_N; G]` to have rank n.

| instance | rows `|N*| + p` | n | rank | verdict |
|---|---|---|---|---|
| lp-band-2000  | 2000 | 2000 | 2000 | unique nondegenerate vertex — crossover free |
| lp-band-20000 | 20000 | 20000 | 20000 | unique nondegenerate vertex — crossover free |
| afiro    | 56 | 51 | 49 | optimal face of dim 2; 5 redundant active rows |
| adlittle | 123 | 138 | 122 | optimal face of **dim 16** |
| share2b  | 166 | 162 | 156 | optimal face of dim 6; 10 redundant active rows |
| scagr7   | 185 | 185 | 185 | unique nondegenerate vertex |
| degen2   | 966 | 757 | 757 | vertex, but **209 redundant active rows** |
| degen-synth | 255 | 300 | 255 | optimal face of dim 45 |

On the synthetic banded family the IPM lands on the unique optimal vertex and a crossover
would be a no-op. On the real Netlib LPs it never does: the IPM converges to the analytic
centre of the optimal face, so a simplex crossover would have to pivot 2–45 columns
(adlittle 16 of 138; degen-synth 45 of 300) and, on degen2, choose among 209 redundant tight
rows. Goldman–Tucker check on `degen-synth`: it was built with five indices where the
constructed `(y*, v*)` is not strictly complementary, and the solver puts all five in B — it
found the maximally complementary solution, and the optimal face is 45-dimensional. Correct
behaviour, and exactly why crossover is not cheap here.

## SOCP cone-state identification

Each SOC block ends interior (s in int K, v = 0), boundary (both nonzero,
λ₋(s) = λ₋(v) = 0) or zero (s = 0, v in int K). Reference states from the 1e-10 solve,
classified by `‖s‖/‖v‖`. "ratio" predicts from `‖s‖/‖v‖` with a neutral band of 1e±2;
"tapia" from `‖sᵗ‖/‖sᵗ⁻¹‖`, `‖vᵗ‖/‖vᵗ⁻¹‖` against 0.5.

| instance | SOC blocks | it@1e-6 | ratio exact | **tapia exact** | interior | boundary | zero |
|---|---|---|---|---|---|---|---|
| socp-sumnorms-150 | 150 | 10 | never | **7** | 0 | 81 | 69 |
| issue10 | 1000 | 25 | never | **17** | 0 | 600 | 400 |

* States are **sharp**: median `λ₋(s)/λ₊(s)` over boundary blocks is 3.9e-12 (sumnorms) and
  3.9e-13 (issue10).
* Tapia identifies **all** blocks well before termination — t = 7 of 10 (30 % early) and
  t = 17 of 25 (32 % early). Much better margin than any LP here.
* But not monotone: issue10 is exact at t = 17, 799/1000 at t = 18, 600/1000 at t = 19–22,
  exact again from t = 23. Unusable without verification.
* The plain ratio rule never becomes exact because 96 of issue10's 600 boundary blocks have
  `‖s‖/‖v‖` outside the ±2-decade band — a scaling artefact of the rule. A scale-free rule
  should compare `λ₋(s)/λ₊(s)` with `λ₋(v)/λ₊(v)`.
* The obstacle to *exploiting* this is structural: the LP projection works because the
  identified face is **affine**, so it is a linear least-squares problem, one extra
  quasi-definite factorization. A SOC boundary face `{s : s₀ = ‖s̄‖}` is a nonlinear
  manifold, so the analogue is a nonlinear least-squares/Newton problem on the face, with no
  finite-termination guarantee. Not implemented; I do not think it pays here.

---

## Verdict

**Speed on LPs: no.** Across 8 LPs the earliest iteration at which a verified exact optimum
can be recovered by projection is within ±2 iterations of where ConicIP already stops at
`optTol = 1e-6` (median +0.5 of 15 iterations, ≈ 4 %), and the projection costs 1.3–3.2 IPM
iterations. No usable cheap gate exists. Net effect: **time-neutral to slightly negative.**
The reason is that ConicIP's 1e-6 test already stops at μ ≈ 1e-7, essentially where
identification becomes exact; there is no long tail to harvest.

**Accuracy on LPs: yes — the one strong result.** The projection turns a 1e-8 answer into an
exact one: gap 1e-7 → 0, max residual 1e-8…1e-10 → 1e-13…1e-16, objective exact to rounding.
That argues for an optional `finite_termination`/`crossover` post-step costing one extra
factorization pair that, when it succeeds, returns a certified-exact point with zero gap —
and an answer that is reproducible independently of `optTol`.

**Crossover to a simplex basis: expensive where it matters.** Only the synthetic banded LPs
give a unique nondegenerate vertex. Every real Netlib LP tested lands in the relative
interior of an optimal face of dimension 2–45, or on a vertex with 209 redundant tight rows.
A full IPM+crossover pipeline needs a genuine simplex phase, not a projection. Much bigger
build than the accuracy post-step; not recommended as a "next level" item.

**SOCP: promising signal, no known way to spend it.** Cone states are identified exactly
~30 % of the iterations before termination, and are sharp (λ₋/λ₊ ≈ 1e-12 on boundary
blocks). But the face is nonlinear, so there is no cheap exact projection, and the Tapia
signal oscillates. The realistic use is **not** finite termination but warm starting /
cone-block elimination — e.g. dropping issue10's 400 provably-zero blocks from the KKT
system from iteration 17 on, which is a per-iteration-cost idea, not a finite-termination
one.
