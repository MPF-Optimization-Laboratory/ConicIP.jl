# Homogeneous self-dual embedding for ConicIP — design (Tranche 3, item 1)

**Date:** 2026-09-07. **Status:** design only; no source changes. Numerical checks
in this document were run with a scratch script against the current tree (v0.4.0 +
Tranches 0–2); the "prototype" is a 100-line dense R-cone Mehrotra HSD written in
ConicIP's sign conventions, with `validate_*_certificate` as the only arbiter.

References: Goulart & Chen, *Clarabel: An interior-point solver for conic programs
with quadratic objectives* (2024) for the QP embedding and the τ elimination;
Andersen & Ye (1999) / Ye, Todd & Mizuno (1994) for the τ/κ dynamics.

## 1. Embedding in ConicIP notation

ConicIP solves

```
min ½yᵀQy − cᵀy   s.t.  Ay − s = b,  s ∈ K,  Gy = d
dual:  max −½yᵀQy − dᵀw + bᵀv   s.t.  Qy + Gᵀw − Aᵀv = c,  v ∈ K
gap:   pobj − dobj = vᵀs  at a primal–dual feasible pair
```

Unknowns of the embedding: `(y, w, v, τ)` free/cone/scalar and the slacks `(s, κ)`,
with `s ∈ K`, `v ∈ K`, `τ ≥ 0`, `κ ≥ 0`. Homogenizing (`y → y/τ`, …) and adding a
row for the objective gap gives the linear part

```
[ 0   Gᵀ  −Aᵀ  −c ] [y]   [0]
[−G   0    0    d ] [w] = [0]
[ A   0    0   −b ] [v]   [s]
[ cᵀ −dᵀ   bᵀ   0 ] [τ]   [κ]
```

The matrix `M` is skew-symmetric (`‖M + Mᵀ‖ = 0`, checked), so `uᵀMu = 0` for
every `u = (y, w, v, τ)`, i.e. `vᵀs + τκ = 0` for every exact solution of the
linear part.

**Quadratic terms** (Clarabel's QP extension, with Clarabel's `q = −c`): add `+Qy`
to the first block row and `−yᵀQy/τ` to the last, so that

```
r_y := Qy + Gᵀw − Aᵀv − cτ                       (n)   dual feasibility
r_w := Gy − dτ                                   (p)   equalities   (sign of the KKT row, see below)
r_v := Ay − s − bτ                               (m)   cone rows
r_τ := cᵀy − dᵀw + bᵀv − yᵀQy/τ − κ              (1)   objective row
complementarity:  v ∘ s = μ e,  τκ = μ,   μ := (vᵀs + τκ)/(conedim + 1)
```

`conedim` is the existing barrier degree (`Σ` orthant dims + one per SOC + order per
SDP block). Note `r_w` keeps ConicIP's `Gy − d` orientation (the second KKT row of
`solve3x3` is `GΔy = r.w`); the skew form has `−Gy + dτ`. With that orientation the
identity that replaces `uᵀMu = 0` reads, for *any* point,

```
vᵀs + τκ = −(yᵀr_y − wᵀr_w + vᵀr_v)                       (checked to 1e-14)
```

The `Qy` and `−yᵀQy/τ` terms cancel in `uᵀ(·)u`, so the identity is exact for the QP
embedding too: when the three linear residuals vanish, `vᵀs + τκ = 0`, and since all
four quantities are nonnegative each pair is complementary.

**Sign of κ.** Dividing the last row by `τ` at a point with `r_y = r_w = r_v = 0`
and writing `ŷ = y/τ` etc.:

```
κ/τ = cᵀŷ − dᵀŵ + bᵀv̂ − ŷᵀQŷ = dobj(ŷ,ŵ,v̂) − pobj(ŷ)        (checked to 1e-14)
```

So **κ ≥ 0 means the de-homogenized point is super-optimal, `dobj ≥ pobj`**, exactly
as in Ye–Todd–Mizuno (`κ = bᵀy − cᵀx`) and Clarabel. The task brief asked for the
sign that makes `κ ≥ 0 ⇔ pobj − dobj ≥ 0`; that sign is not available: negating the
last row breaks skew-symmetry and turns `vᵀs + τκ = 0` into `vᵀs = τκ`, which is not
a complementarity condition. In ConicIP's `−cᵀy` convention the last row is
`cᵀy − dᵀw + bᵀv`, not `−cᵀy + …`, and the interior iterates carry `κ > 0`, i.e.
an infeasible point with `dobj > pobj`. This is consistent with Tranche 2's decision
to measure the gap as `⟨v,s⟩` rather than `pobj − dobj`.

**Mapping to Clarabel** (`min ½xᵀPx + qᵀx, Ax + s = b, s ∈ K`; dual
`Px + Aᵀz + q = 0`):

| Clarabel | `P` | `q` | `A` | `b` | `z` | `s` | `κ` |
|---|---|---|---|---|---|---|---|
| ConicIP | `Q` | `−c` | `[−G; −A]` | `[−d; −b]` | `[−w; v]` | `[0; s]` | same |

Two conventions made the transcription awkward: the equality multiplier flips sign
(`z_eq = −w`), and ConicIP's cone rows are `Ay − s = b` rather than `Ax + s = b`,
so every `A`, `b` acquires a minus. With that mapping Clarabel's
`κ = −qᵀx − bᵀz − xᵀPx/τ` equals the expression above (checked to 1e-14).

**Linearization of the objective row.** `∂/∂y(−yᵀQy/τ) = −2Qy/τ` and
`∂/∂τ(−yᵀQy/τ) = +yᵀQy/τ²` (finite differences agree to 1e-8).

## 2. Newton system with the existing factorization

In ConicIP's convention (`K Δz = r`, `z ← z − αΔz`) the linearized embedding is

```
Q Δy + Gᵀ Δw − Aᵀ Δv          − c Δτ                    = r_y
G Δy                          − d Δτ                    = r_w
A Δy          − Δs            − b Δτ                    = r_v
λ∘(F Δv) + λ∘(F⁻ᵀ Δs)                                   = r_s      (existing 4th row)
gᵀΔy − dᵀΔw + bᵀΔv + (ξᵀQξ) Δτ − Δκ                     = r_τ
κ Δτ + τ Δκ                                             = r_κ
```

with `ξ := y/τ` and `g := c − 2Qξ` (the `−2Qy/τ` and `+yᵀQy/τ²` terms of §1 written
through ξ). The τ column is dense (`c, d, b`) and the τ row is *not* the negative
transpose of the τ column (`gᵀ` versus `−cᵀ`), so the augmented matrix is neither
skew nor symmetric; folding it into a symmetric factorization would need this
elimination anyway. Eliminate `Δτ`:

```
(Δy, Δw, Δv, Δs) = Δz_r + Δτ · Δz_c
   Δz_r := solve4x4(r_y, r_w, r_v, r_s)          per right-hand side
   Δz_c := solve4x4(c,   d,   b,   0)            once per factorization
φ(Δz) := gᵀΔz.y − dᵀΔz.w + bᵀΔz.v
Δτ = ( r_τ − φ(Δz_r) + r_κ/τ ) / ( φ(Δz_c) + ξᵀQξ + κ/τ )
Δκ = ( r_κ − κ Δτ ) / τ
```

Checked against a direct dense solve of the full `(n+p+m+m+2)` system: agreement to
5e-14. The denominator is positive whenever `κ/τ > 0`: with `Δz_c = (y_c, w_c, v_c, s_c)`
the fourth row gives `s_c = −FᵀF v_c`, and the first three rows give
`cᵀy_c − dᵀw_c + bᵀv_c = y_cᵀQy_c + ‖Fv_c‖²`, hence

```
φ(Δz_c) + ξᵀQξ + κ/τ = (y_c − ξ)ᵀQ(y_c − ξ) + ‖F v_c‖² + κ/τ  > 0    (checked to 3e-14)
```

This is Clarabel's completion of squares in ConicIP's variables; it is what makes
the scalar step well defined for a singular `Q`.

**Right-hand sides.** The constant column `(c, d, b, 0)` is exactly the classic
initial-point right-hand side `r_init`, so `Δz_c` at the identity scaling is the
classic unshifted initial point. Predictor: `r_s = λ∘λ`, `r_κ = τκ`. Corrector, in
the form the code already uses (`r.s = λ∘λ + (F⁻ᵀΔs_aff ∘ FΔv_aff) − σμe`):
`r_κ = τκ + Δτ_aff Δκ_aff − σμ`. The centering ratio becomes
`ρ = [(v−αΔv)ᵀ(s−αΔs) + (τ−αΔτ)(κ−αΔκ)] / (vᵀs + τκ)` with `σ = clamp(ρ,0,1)³` as now.

**Solves per iteration.** Classic: predictor + corrector + refinements
(measured 2.7–2.9). HSD: constant column + predictor + corrector + refinements, i.e.
one extra back-solve per factorization, ≈ 3.7–3.9 expected. `Δz_c` and `Δz_r` are
each refined with the existing `refine!`/`step_residual!` against the 4×4 (the
constant column with its own budget); the scalar rows are then satisfied exactly by
construction, so the 6-block residual is bounded by the two 4×4 residuals times
`(1 + |Δτ|)`. `kkt_solves` counts the constant-column solve.

**Callback contract.** `solve3x3gen(F, F⁻ᵀ)` and `solve4x4gen(λ, F, F⁻ᵀ)` are
unchanged; the τ elimination lives entirely in the main loop. No built-in or custom
KKT solver sees τ. Folding τ into the callback was rejected: it would add a dense row
and column to every factorization (fill in LDLᵀ, a rank-one border in QR), change the
contract for all four built-in solvers and every user callback, and — because the
border is unsymmetric — require this same elimination inside each solver.

## 3. Termination and certificates from τ, κ

The de-homogenized point is `ẑ = z/τ` (`ŷ, ŵ, v̂, ŝ`); its classic residuals are
`r̂_y = r_y/τ`, `r̂_w = r_w/τ`, `r̂_v = r_v/τ`, and `v̂ᵀŝ = vᵀs/τ²`.

1. **Optimal.** Run the existing termination block (`rDu, rPr, rEq, rCp, rGap` with
   the componentwise backward-error normalizations, `objective_offset`, and the
   equilibration mapping) on `ẑ` verbatim; `:Optimal` iff `max(...) < optTol`. This
   gives test-for-test parity with `:classic`. Optimality is checked first and wins.
2. **Nominate → validate → claim.** On a non-optimal iterate:
   - if `dᵀw − bᵀv < 0`, nominate `(w, v)` to `validate_infeasibility_certificate`;
   - else if `cᵀy > 0`, nominate `y` to `validate_unboundedness_certificate`;
   - a valid verdict claims `:Infeasible` / `:DualInfeasible` via the existing
     `claim_*!` (validator-normalized ray), in that precedence.
   The validators are scale-invariant (they normalize by the separation), so the raw
   homogeneous `(w, v)` and `y` are passed; no division by τ. Under equilibration the
   outer `conicIP` revalidates against the original data exactly as today.
   Why this is sound and complete: as `τ → 0` with `κ > 0`, boundedness of
   `yᵀQy/τ` forces `Qy → 0`, so the limit satisfies `Gᵀw − Aᵀv = 0`, `Gy = 0`,
   `Ay = s ∈ K`, and `κ = cᵀy + (bᵀv − dᵀw) > 0`, so at least one of the two
   nominations is a genuine ray (both, when primal and dual are both infeasible;
   prototype instance E6 returns `:Infeasible`, the stated precedence).
   **Cost gate.** The sign tests are `O(n+m+p)`; a validator call is two or three
   products plus `cone_margin` (an eigendecomposition per SDP block). Nominate only
   when `τ < max(κ, τ_nom)` with `τ_nom = 1e-2`. The `τ < κ` term alone is late when
   the certificate's separation is small: on the infeasible QP of the "Certificate
   fallback" testset the prototype reaches `κ* ≈ 4e-3` and `τ < κ` fires one
   iteration after the ray already validates. The gate is a cost control, never a
   soundness device: the validator decides.
3. **Exhaustion** (`maxIters`, stall, `timeLimit`): validate the *last* iterate's
   rays at `infeasTol` (claim) and at `100·infeasTol` (`:AlmostInfeasible` /
   `:AlmostDualInfeasible`), corroborated by `τ < κ` in place of today's
   `μ_collapsed`. Return the best de-homogenized point (tracked as today by
   `bestMeasure` on `ẑ`, but only over iterates with `τ ≥ κ`; once `τ < κ` the
   iterate is a ray in the making and the point fields are those of the last
   `τ ≥ κ` iterate). If both `τ` and `κ` fall below `1e-8·max(1, τ₀, κ₀)` with no
   ray validating even at `100×`, the status is `:Abandoned` with a message naming
   the ill-posed (weakly infeasible / no strictly complementary) case; `:Almost*` is
   reserved for a ray that validates at the relaxed tolerance, so that its documented
   meaning is unchanged (§7, item 6).
4. **Errors.** Nonfinite `τ`, `κ`, or `Δτ`, `Δκ` join the existing `nonfinite!`
   paths; a nonpositive scalar denominator (impossible with `κ/τ > 0`, but guard it)
   is `:Error` with a message.

| status | when (HSD) | fields |
|---|---|---|
| `:Optimal` | classic test on `ẑ` passes | `ẑ`, `has_certificate = false` |
| `:Infeasible` | `(w, v)` validates at `infeasTol` (in loop or post-loop) | ray, as today |
| `:DualInfeasible` | `y` validates at `infeasTol` | ray, as today |
| `:AlmostInfeasible` / `:AlmostDualInfeasible` | exhaustion, ray validates only at `100·infeasTol`, `τ < κ` | best point |
| `:Abandoned` | exhaustion, no ray; includes τ, κ both vanished (message) | best point |
| `:TimeLimit` | unchanged | best point |
| `:Error` | unchanged plus nonfinite / nonpositive scalar step | current point |

**Finding from the prototype (termination, not HSD).** On the badly scaled
infeasible LP of the "Certificate revalidation after equilibration" testset
(`xᵢ ≥ 1`, `−Σx ≥ 0`, rows scaled `1e-6…1e6`, columns reversed), run *unequilibrated*,
the HSD iterate at iteration 3 passes the classic termination block on `ẑ`
(`rPr = 2e-11`, `rCp = 8e-10`, `rGap = 8e-10`) at a point with `x'₁ = −7.2` (the
constraint is `x'₁ ≥ 1`). The violated row has weight `1e-6`, `‖b‖ ≈ 3e4`, and the
2-norm relative residual is `7e-13`; the *row-wise* relative residual
`|r_i| / (1 + |b_i| + (|A||ŷ|)_i + |ŝ_i|)` is `8e-6`. The classic path returns
`:Infeasible` in 16 iterations on the same data only because its trajectory never
visits such a point; equilibrated, HSD certifies infeasibility in 2 iterations. So
the aggregate relative test can certify a point on an infeasible problem when the
data span twelve orders of magnitude, independent of the method. Options in §7,
item 8.

## 4. Interactions

- **Equilibration.** `_conicIP` runs the embedding on the scaled data. With
  `ỹ = Dc⁻¹y`, `w̃ = σDe⁻¹w`, `ṽ = σDr⁻¹v`, `s̃ = Dr s` the embedding is homogeneous in
  the same τ: `r̃_y = σDc r_y`, `r̃_w = De r_w`, `r̃_v = Dr r_v`, and
  `κ̃ = σκ`, `μ̃ = σμ`, `τ̃ = τ`. The termination block's `scaling` branch already maps
  `r̃/(σDc)`, `r̃/Dr`, `r̃/De` and `μ̃/σ` back; it is applied to `ẑ = z̃/τ` unchanged.
  `unequilibrate!` gains `sol.kappa /= σ`; `sol.tau` is invariant. The initial
  `τ = κ = 1` is taken in scaled coordinates (so `κ₀ = 1/σ` in the original ones —
  a legitimate choice, since only ratios matter). `_revalidate_certificate!` and the
  original-data revalidation in `conicIP` are unchanged.
- **Presolve.** `preprocess_conicIP` forwards `kwargs` (`rest...`) to `conicIP`, so
  `method` passes through; `_check_postsolve!` / `_refresh_point!` act on the
  de-homogenized point. The structural-deflation recursion inside `_conicIP` passes
  keywords explicitly and must add `method`. Singleton fixings, `imcols`, and the
  `staticReg` retry are untouched.
- **Time limit.** Checked once per iteration as now. Under `:hsd` the exhaustion
  tail has no fallback solves, so the overrun is bounded by one factorization plus
  the validator calls.
- **Initial point.** The classic `solve4x4gen(e, I, I)(r_init)` plus the cone shifts
  of `v, s`, then `τ = κ = 1`. Prototype E1 (feasible QP) converged in 7 iterations
  against 8 for the classic path from this start; Clarabel initializes the same way.
- **Line search.** `α = min(1, (1−DTB)·min(maxstep(v,Δv), maxstep(s,Δs), τ/Δτ, κ/Δκ))`
  with the scalar terms only when `Δτ > 0` / `Δκ > 0` (the minus convention of
  `maxstep_rp`). The predictor `α_aff` takes the same four terms without `DTB`. The
  verified-interiority loop adds `τ − αΔτ > 0` and `κ − αΔκ > 0` to `interior(·)`
  before accepting a step; the geometric back-off is unchanged.
- **Stall detection.** `α < 1e-8` three times breaks the loop as now. Add a second
  exit: `τ` and `κ` both below the §3 threshold, message "τ and κ vanished".
- **`objective_offset`.** Enters `rGap`'s denominator as `|pobj(ŷ) + offset|`,
  exactly the existing formula on `ẑ`; the embedding never sees the constant.
- **Verbose table.** Under `:hsd` the `icertp`/`icertd` columns become `τ`/`κ`
  (the screens no longer exist); the `refine` and `kkt` columns are unchanged.
- **`Solution`.** Two trailing fields `tau::Real`, `kappa::Real` (`NaN` under
  `:classic` and in the short constructors), so a caller can see how a
  `:Abandoned` run was heading.

## 5. Test plan

1. **Parity.** For the `t3_mixes` generator of `test/tranche3_tests.jl` (LP/SOC/SDP
   mixes) and the `test/testdata.jl` LP/QP/SOCP/SDP instances, with
   `equilibrate ∈ {true, false}` and `kktsolver ∈ {ldl, qr}`: same status,
   `Iter_hsd ≤ Iter_classic + 2`, `‖y_hsd − y_classic‖ ≤ 1e-5(1 + ‖y‖)`,
   `kkt_solves_hsd ≤ kkt_solves_classic + Iter_hsd·(1 + maxRefinementSteps)`.
   The bitwise `:classic` baseline in that file must still pass: the `:classic`
   branch must not reorder a single floating-point operation.
2. **Certificate testsets under `method = :hsd`, `certFallback = false`.**
   "Infeasibility soundness" (a)–(f) unchanged in expectation (no false claims; the
   sliver box `0 ≤ x ≤ 1e-9` returned `:Optimal` in the prototype with
   `max|y| = 7.6e-9`). "Post-loop certificate exits": the `maxIters = 1` and
   `maxIters = 0` cases exercise the post-loop validation of the last iterate. "Certificate
   fallback": the infeasible and unbounded QPs must reach `:Infeasible` /
   `:DualInfeasible` with `has_certificate` under `maxIters = 30`
   (prototype: 8 and 6 iterations, 22 and 16 solves) and
   `kkt_solves ≤ 1 + Iter·(3 + 2·maxRefinementSteps)`; the two `certFallback = false`
   cases that expect `:Abandoned` at `maxIters = 7`/`5` are `:classic`-specific and
   are kept under `:classic` only. "Feasible problem never gains a certificate":
   unchanged.
3. **Unbounded SOCP smoke test.** The roadmap's instance ("35 iterations to objective
   1e-29, then a singular factorization") is not pinned in the repository. A
   generator that is feasible and unbounded by construction — recession direction
   `ȳ` with `Aȳ ∈ int K` and `cᵀȳ = 1`, `b = Ay₀ − s₀` with `s₀ ∈ int K`, two
   SOC blocks of dimension 8, `n = 20` — was solved by the classic path with
   certificates in 1–3 iterations for five seeds, so it is not the hard case. Test:
   `:DualInfeasible` with certificate in ≤ 30 iterations for ten seeds of that
   generator *and* for the original instance once it is recovered (§7, item 9).
4. **MOI.Test** with `MOI.RawOptimizerAttribute("method") => "hsd"` in the existing
   `MOI.Test.runtests` configuration, same exclusions as today; plus the "MOI
   certificate contract" testset under `"hsd"`.
5. **Badly scaled infeasible LPs** ("Certificate revalidation after equilibration"):
   `equilibrate = true` must give `:Infeasible` with a ray valid on the original data
   (prototype: 2 iterations). `equilibrate = false` depends on §7, item 8: with the
   current termination block the expected status is `:Optimal` (relative residual
   below `optTol`) *or* `:Infeasible`, and the test should assert only
   "no invalid certificate and, if `:Optimal`, the returned point meets `optTol`
   under the block's own normalization"; with a row-wise test it stays
   `:Infeasible`. The `rs5/cs5` instance keeps its consistency assertions.
6. **Harness.** `benchmark/suite.jl --quick` and the full set under both methods:
   iterations within +2 and time within 1.3× on every solved instance; identical
   original-coordinate residuals to `1e-6`.

## 6. Migration

- `method::Symbol = :classic` keyword on `_conicIP` (validated: `:classic | :hsd`),
  forwarded by `conicIP`, `preprocess_conicIP` (through `rest...`), the deflation
  recursion, and the MOI option `"method" ∈ ("classic", "hsd")` added to
  `_SUPPORTED_OPTIONS` and the defaults table.
- `:classic` stays the default until §5 items 1, 2, 4 and 6 pass; then flip the
  default in a minor release with a changelog entry; `:classic` remains selectable
  for one further minor release.
- Under `:hsd`, `certFallback` and `certFallbackIters` are accepted and ignored
  (documented; a verbose note when set explicitly).
- Implementation shape: one loop, not two. `τ, κ` are scalars beside `z::v4x1`
  (`v4x1` is untouched — it is the KKT residual type). Branches on `hsd::Bool` at
  five points: constant-column solve, scalar step, line-search terms, residual
  de-homogenization, nomination. Under `:classic` every branch is skipped and `τ ≡ 1`
  is never multiplied in, which is what keeps the bitwise baseline.
- **Reused as is:** `nt_scaling`, `cone_div!`/`cone_prod!`, `maxstep`, `interior`,
  `solve4x4gen`, `step_residual!`/`refine!`, the termination block and its `scaling`
  branch, the validators, `claim_*!`, `Solution`, `equilibrate.jl`,
  `preprocessor.jl`, `timeLimit`, `kkt_diagnostics`, the MOI status map.
- **Retired under `:hsd`:** the two in-loop screens (`p_infeas`, `d_infeas` and
  their CVXOPT/ECOS scalings), `μ_history` / `μ_collapsed` / `μ_diverged`, and the
  `fallback_*_ray` calls. `src/fallback.jl` stays while `:classic` exists; the
  two-model coexistence adds no new file.
- **Effort** (one contributor): core loop 1–1.5 weeks; parity and certificate suites
  1 week; equilibration mapping, `Solution` fields, MOI option, verbose, docs
  0.5 week; SDP verification on `test/sdp_tests.jl` and the SDPLIB subset 0.5 week;
  harness measurement and the default decision 0.5 week. About 4 weeks, in that
  order, with the plumbing (keyword, fields, no behaviour change) as the first
  commit so that the baseline test guards everything after it.

## 7. Decisions for the maintainer

1. **Go / no-go** on Tranche 3 item 1 with the design above (≈ 4 weeks).
2. **Coexist vs replace.** Recommended: coexist behind `method`, `:classic` default,
   flip after §5; retire screens and `fallback.jl` one minor release after the flip.
3. **τ elimination by two 4×4 solves** (recommended; contract unchanged, one extra
   back-solve per factorization, denominator provably positive) **vs folding τ into
   the KKT callback** (dense unsymmetric border, contract change for every solver).
4. **SDP in the first cut.** Nothing in the embedding is cone-specific
   (`maxstep_sdc`, `nestod_sdc`, `cone_margin` for `"S"` are reused), so the
   recommendation is yes, gated by `test/sdp_tests.jl` and the SDPLIB subset under
   `:hsd`; the only SDP-specific cost is `cone_margin`'s eigendecomposition inside the
   nomination, which the §3 gate bounds.
5. **Meaning of κ.** Confirm the documentation wording "κ ≥ 0: the de-homogenized
   iterate satisfies `dobj ≥ pobj`"; the brief's `pobj − dobj ≥ 0` reading is the
   opposite sign and is not achievable (§1).
6. **`:Almost*` semantics** when τ and κ both vanish with no ray validating at
   `100·infeasTol`: `:Abandoned` with a message (recommended, keeps the documented
   meaning of `:Almost*`) vs extending `:Almost*` to this case.
7. **Nomination gate** `τ < max(κ, 1e-2)`: accept the constant, or run the validators
   unconditionally whenever the sign test passes (two to three products per
   iteration; SDP eigendecompositions per block).
8. **Termination normalization** exposed by the prototype (§3): keep the aggregate
   2-norm test and relax the `equilibrate = false` expectation of the badly scaled
   testset, or add a row-wise relative residual test `max_i |r_i| / (1 + |b_i| +
   (|A||y|)_i + |s_i|) < optTol` alongside it (a `:classic`-affecting change, to be
   measured on the harness before HSD lands).
   **Decided (review round, 2026-09-07):** the row-wise test is in `:classic` for the
   cone and equality rows (`src/ConicIP.jl`, termination block). The point that
   terminates the aggregate test passes it on every `--quick` instance and on the
   Miles κ-scaling family (largest value 4.4e-11), so no iteration count changed; the
   classic solver itself was caught by it on a two-row LP (`x ≥ 1` at weight 1e-4,
   `x ≤ 0.5` at weight 1e4, unequilibrated: `:Optimal` at x ≈ 0.5, now `:Infeasible`).
   The stationarity-row analogue is *not* added: it fails on 9 of 15 Miles-3 κ points
   by two to three orders of magnitude (up to 1.1e-3 at κ = 1e8, an instance that
   already needs 40 iterations), because a free variable's row with a tiny cost
   coefficient turns it into an absolute test. HSD inherits the decision.
9. **Smoke-test instance.** Recover and pin the roadmap's 35-iteration unbounded
   SOCP (seed and generator), or accept the §5 generator family in its place.
10. **`Solution.tau` / `Solution.kappa`** as new trailing fields (recommended) vs
    exposing them only through `sol.message`.

Checkpoint: implementation does not start until the maintainer answers the items in §7.
