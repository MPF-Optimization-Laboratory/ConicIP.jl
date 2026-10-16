# SDP at scale: interior-point linear algebra and the state of practice

Agent `lit-sdp`, 2026-09-20. Survey for Tranche 4 of the ConicIP roadmap.
Notation: `k` = PSD block order; `n` = number of free variables in ConicIP's `A: R^n -> R^m`
(this is what the SDP literature calls `m`, the number of constraints); `B_j = mat(A[I,j])`.
Where a source uses the classical convention I say so explicitly.

---

## (a) Verdict

**The minimum change that makes SDP credible at k ~ 500-2000 is to stop putting a
symmetric-Kronecker operator into the KKT matrix, and to assemble the reduced Schur
complement `S_ij = tr(B_i T^-1 B_j T^-1)` instead --- with a per-column dispatch over the
Fujisawa-Kojima-Nakata F1/F2/F3 formulae --- then factor the n x n reduced Hessian by
Cholesky.** This is not a tuning decision; it is forced by arithmetic. An explicit
`skron` block for one k x k cone has (k(k+1)/2)^2 entries: 3.2 GB at k = 200, **125 GB at
k = 500, 2.0 TB at k = 1000, 32 TB at k = 2000**. The current `VecCongurance` -> dense-QR
path therefore has a hard ceiling near k ~ 150-200 regardless of how fast the factorization
is, and the same is true of any augmented-system code that materialises W (x) W --- including
the one ConicIP is benchmarked against, whose `coneops_psdtrianglecone.jl` calls
`skron!(f.Hs, Symmetric(RRt))` and packs the result into the KKT block. By contrast the
reduced Hessian at n = 10^4 is **800 MB with a 3.3e11-flop Cholesky** --- seconds per
iteration on one M3 Pro --- and its size does not depend on k at all. Assembly, not
factorization, becomes the cost: O(n k^3 + n^2 k^2) dense, which at (n, k) = (10^4, 500) is
1.25e12 + 2.5e13 flops per iteration and is only tolerable if the F1/F2/F3 (or
DSDP-style low-rank) dispatch actually collapses the n^2 k^2 term on the target family.
That is precisely the condition on which "a named family" in the roadmap should be chosen.

**Reduced Schur complement first; chordal decomposition second.** Three reasons. (i) The
reduced complement is unconditional --- it helps every SDP, sparse or dense --- whereas chordal
decomposition is a no-op the moment the aggregate sparsity pattern is dense (Zheng et al.
report `theta1`/`theta2` as single-clique, zero benefit; Clarabel's own table shows
`control*`, `theta1-5`, `qap*` identical with decomposition on and off, to three significant
figures). (ii) Chordal decomposition is a *reformulation layer*: it rewrites one PSD cone of
order k into many cones of order |C_i| plus consensus equalities, and then hands the result
to whatever cone algebra the solver already has. It can be built entirely above the solver
core (in presolve/MOI) and retrofitted at any time --- and if it is added *first*, it merely
moves the k^4 wall from k = 200 to max-clique = 200, which on `qpG51` (max clique 304 after
extension) and on the nonchordal `rs*` family (max clique up to 418) is not far enough.
(iii) Decomposition without a competent medium-block path is actively harmful: naive
clique-tree conversion without merging is **3.0x and 4.4x slower than no decomposition at
all** on `mcp500-3` and `mcp500-4` (7.66 -> 22.69 s; 11.63 -> 51.37 s), because the consensus
equalities inflate n and the fixed per-block overhead swamps the cubic saving. The two
compose well in the other order: decomposition produces many medium blocks with sparse
`B_j`, which is exactly the regime F2/F3 were designed for.

Two riders. First, if the named family turns out to be *chordally sparse* (power-flow,
truss, graph relaxations), chordal decomposition is worth more than everything else combined
--- Clarabel reports **60.7x / 131x / 154x / 134x on arch0/arch2/arch4/arch8** (8.17 s vs 496 s,
etc.) purely from decomposition, and CDCS reports 9x/43x/66x vs undecomposed ADMM on
maxG11/maxG32/qpG11. Second, `dsdc!` is a free O(k^3) -> O(k^2) win that is independent of
everything above: `cone_div!` is called with `y = lambda = F*z.v`, and by construction
`nestod_sdc` makes `mat(lambda) = Lambda` diagonal, so `lyap(Y, -2X)` is running LAPACK's
Schur-based Sylvester solver on a diagonal matrix. The closed form every leading code uses is
a Hadamard division, `Z = X .* Gamma` with `Gamma_ij = 2/(lambda_i + lambda_j)`.
`nestod_sdc` already computes `Lambda` (as `F.S`) and throws it away.

---

## (b) Comparison table

| Approach | Per-iteration complexity | Memory | Where it wins | Reference implementations | Refs |
|---|---|---|---|---|---|
| Explicit symmetric-Kronecker in the augmented system (ConicIP's sparse/LDL paths; *not* the default QR path, see correction below) | O(k^6) dense factor, or O(k^4) nnz in a sparse LDL^T | **O(k^4)**: 125 GB at k=500 | k <~ 150; keeps one KKT code path for all cones | ConicIP `kktsolver_sparse`/`kktsolver_ldl`; Clarabel `skron!` into `Hs` | [8], [16] |
| Reduced Schur complement, dense `B_j`, NT or HKM | assembly O(n k^3 + n^2 k^2); Cholesky O(n^3/3) | **O(n^2 + k^2)**: 800 MB at n=10^4 | the default; k large, n <~ 2e4 | SDPT3, CSDP, SDPA, SeDuMi, MOSEK | [1],[2],[3],[4],[5] |
| ... + F1/F2/F3 sparse dispatch per column | F1 `n_p f_j + n_p^3`; F2 `n_p(f_j+\|I\|)`; F3 `2 f_j \|I\|` | same | sparse `B_j` (graph/combinatorial SDPs) | SDPA 3.0+ (1997), CSDP, SDPT3 | [1],[3],[4] |
| ... + low-rank `B_j = sum_r a_r a_r^T` | O(n * rank * k^2) or O(n^2 * rank^2 * k) | same | truss, sensor nets, Lasserre/MAXCUT | DSDP5 (M1/M2), Loraine, HDSDP | [6],[7],[12] |
| Chordal / clique decomposition (domain- or range-space conversion) + any of the above | sum_cliques O(\|C_i\|^3) instead of O(k^3); n grows by consensus rows | O(sum \|C_i\|^2) | sparse aggregate pattern, small cliques | SparseCoLO, CHOMPACK/SMCP, COSMO, CDCS, Clarabel | [13]-[21] |
| Nonsymmetric chordal-barrier IPM (never forms the dense cone) | multifrontal recursions on the clique tree | O(nnz of pattern) | very sparse, chordal, moderate n | SMCP + CHOMPACK | [17],[18] |
| Iterative (PCG / PSQMR) Schur or reduced-augmented solve | matvec-only; Krylov count grows near the boundary | O(n) + preconditioner | low-rank `B_j`; n far beyond dense-Cholesky range | Loraine, Toh's PSQMR codes, SDPT3 fallback | [9],[10],[11],[12] |
| Dual scaling | one Cholesky of S per iteration; X never formed | lower than primal-dual | sparse + low-rank combinatorial SDPs | DSDP5, HDSDP (2022) | [6],[7] |
| ADMM / operator splitting (+ chordal) | O(k^3) eigendecomposition per cone per iteration; 10^2-10^3 iterations | O(k^2) | k huge, accuracy 1e-3 to 1e-4 | SCS, COSMO, CDCS | [19],[20],[21] |
| Burer-Monteiro / low-rank factorization | O(nnz * r) per gradient; no Schur complement at all | O(k*r) | k >> 10^4; accuracy 1e-3 to 1e-5 | SDPLR, LoRADS/cuLoRADS, HALLaR, SketchyCGAL | [23]-[27] |
| ALM + semismooth Newton-CG | inner CG on a generalized Hessian | O(k^2) | large DNN/QAP relaxations at ~1e-6 | SDPNAL, SDPNAL+ | [28],[29] |

**Correction (2026-10-14).** ConicIP's default SDP path, `kktsolver_qr`, never forms the
symmetric-Kronecker block: it applies the congruence column by column and was already a dense
reduced Hessian O(n k³ + n² k² + n³) in the affine (LMI) form. The O(k⁶) arises from the
*formulation*: a PSD variable (`@variable(X, PSD)`) makes n = k(k+1)/2, and the same dense
path then factors an (n−p)² matrix. Fixed on dev by `kktsolver_sdpschur` (equality Schur
complement for standard-form problems) and a shared trace-form Gram kernel (`src/schur.jl`);
see `large-scale-roadmap.md` Tranche 4.

---

## (c) Details

### 1. The standard reduced Schur complement

Every classical code eliminates to a single n x n (classically m x m) system. SDPT3 forms
`M = (A^s)^T (E^s)^-1 F^s A^s + ...` with the NT block `M^p_ij = <a^p_i, W^p a^p_j W^p>`,
`W^p = (G^p)^-2`, and the HKM block `M^p_ij = <a^p_i, x^p a^p_j (z^p)^-1>` [2],[3]. SDPA states
the cost model explicitly: **ELEMENTS O(m n^3 + m^2 n^2), CHOLESKY O(m^3), OTHERS O(n^3)** in
classical notation (their Table 3) --- i.e. in ConicIP's notation O(n k^3 + n^2 k^2), O(n^3),
O(k^3) [4]. The Fujisawa-Kojima-Nakata dispatch [1] decides per row which of three formulae to
use, based on `f^b_k = nnz(F^b_k)`:

- **F-1** (both dense): `B_ij += (Y^b F^b_i (X^b)^-1) . F^b_j`. Cost `n_p f_j + n_p^3`.
- **F-2** (`F_i` dense, `F_j` sparse): double sum over the nonzeros of `F^b_j` against rows of
  `Y^b F^b_i (X^b)^-1`. Cost `n_p (f_j + |I|)`.
- **F-3** (both sparse): quadruple sum over the nonzero patterns of both. Cost `2 f_j |I|`.

SDPA 7 confirms these "have also been adopted in CSDP and SDPT3" [4]; SDPT3's own paper cites
[1] for its sparsity handling [2]. MOSEK does the same thing with a different arrangement: it
computes `A Pi A^T = sum_k P_k^T (P_k A_{:,k} Pi_k A_{:,k}^T P_k^T) P_k`, sorting columns by
nnz and evaluating `R R^T A_j R R^T = Mhat M + M^T Mhat^T` element-by-element when `A_i` is
sparse and by level-3 BLAS when dense, following SeDuMi --- described in the slides as "more
general than SDPA" [5]. DSDP is the odd one out: it exploits **rank**, not sparsity,
decomposing `A_i = sum_r alpha_{i,r} a_{i,r} a_{i,r}^T` and using technique M1
(`w <- S^-1 a; M_ij += alpha_j w^T A_j w`) or M2 (accumulate `V <- V + alpha w w^T`, then
`M_ij += <A_i, V>`), "instead of the sparsity structure that is exploited by other
interior-point solvers [Fujisawa et al. 1997]" [6]. HDSDP (2022) revives this with a
homogeneous self-dual embedding [7].

**NT vs HKM.** SDPA (all versions since 1.0), CSDP, and SDPT3's SDP default use HKM [2],[3],[4].
MOSEK, SeDuMi, CVXOPT and Clarabel use NT [5],[8],[16]. The practical difference is not the
Schur assembly (same cost either way) but the scaling update: MOSEK's own profiling across 27
SDPLIB instances concludes that "the update step (updating variables, neighborhood check, new
NT scaling) is very cheap for LP and SOCP, but expensive for SDP, mainly due to NT scaling;
**HKM scaling would give an improvement**" [5]. On `maxG11` the update step is 45.0% of total
time vs 2.7% for the Schur formation. The split is strongly instance-dependent: `control11` is
78.8% Schur / 18.3% factor, `theta6` is 21.4% Schur / 74.5% factor.
**Recommendation for ConicIP: keep NT.** The reduced-Hessian formula the roadmap already
writes down is the NT one, NT keeps a single code path with the R/Q cones, and the NT update
is 2 Choleskys + 1 SVD = O(k^3), which is small next to an O(n k^3 + n^2 k^2) assembly at the
scales in question.

### 2. Chordal decomposition

The completion theorem is Grone-Johnson-Sa-Wolkowicz (1984); the dual "sum of clique-supported
PSD matrices" direction is Agler-Helton-McCullough-Rodman (1988). Fukuda-Kojima-Murota-Nakata
[13] gave two conversions: clique-tree / matrix-completion (domain-space) and a
completion-inside-the-IPM variant; Nakata et al. [14] implemented them. Kim-Kojima-Mevissen-
Yamashita [15] generalized to domain- and range-space sparsity with four conversions, shipped
as SparseCoLO. Vandenberghe-Andersen's monograph [16] is the standard reference for the
structure theory and the clique-tree cost model (O(sum |C_i|^3) vs O(k^3));
Andersen-Dahl-Vandenberghe [17] built the only production IPM that factors the chordal
structure *inside* its own linear algebra (SMCP + CHOMPACK [18]) rather than reformulating and
handing off.

Measured payoffs: CDCS vs SCS on the undecomposed problem, **maxG11 10.9 s vs 93.9 s (9x),
maxG32 56 s vs 2433 s (43x), qpG11 16 s vs 1065 s (66x)**; on nonchordal `rs*` instances
CDCS-dual solves `rs1555` in 317 s where SCS needs 139,314 s without converging [19].
Clarabel, which reuses COSMO's clique-graph merge with edge weight
`e(C_i,C_j) = |C_i|^3 + |C_j|^3 - |C_i u C_j|^3` and the range-space conversion of [15],
reports **arch0 8.17 s vs 496 s, arch2 3.36 s vs 441 s, arch4 2.85 s vs 440 s, arch8 3.35 s vs
448 s** with decomposition on vs off [20]. Chordal decomposition is **on by default** in
Clarabel (`chordal_decomposition_enable = true`, merge method `:clique_graph`).

Failure modes, all measured: (i) dense aggregate pattern -> one clique ~ k, pure overhead;
(ii) many small overlapping cliques with no merging -> consensus equalities inflate n and fixed
per-block overhead dominates (`mcp500-3` 7.66 -> 22.69 s, `mcp500-4` 11.63 -> 51.37 s;
SparseCoLO on `thetaG51` 587.79 s vs 505.33 s undecomposed --- worse than not decomposing) [21];
(iii) large max cliques after extension (`qpG51`, max clique 304) --- CDCS hits its iteration
cap there [19]; (iv) primal degeneracy of the decomposed problem whenever the original SDP has
a low-rank optimum. Zhang's complexity results [22] make the theory precise: chordal conversion
+ IPM gives O(m+n) time per iteration over O(sqrt(m+n) log 1/eps) iterations *if the extended
aggregate sparsity graph has small treewidth* --- and he proves small treewidth of the plain
aggregate graph is not sufficient.

### 3. Where IPMs stop being competitive

The crossover is about **accuracy**, not only size. First-order splitting (SCS, COSMO) is
standardly run at 1e-3 to 1e-4; Burer-Monteiro GPU codes are explicitly two-phase (cuLoRADS
ships with phase-1 tol 1e-3, phase-2 1e-5) [24]. In exchange the scale is not comparable:
cuLoRADS solved a Max-Cut instance with a **1.7e8 x 1.7e8 matrix variable and 1.7e8 constraints
in ~158 s on one H100** [24]; HALLaR solves max-stable-set at (n,m) ~ (1e6, 1e7) to 1e-5
relative accuracy in under 20 minutes [25]; SketchyCGAL handles matrix variables with >1e14
entries via Nystrom sketching [26]. The theoretical basis for Burer-Monteiro is the
Barvinok-Pataki bound `r(r+1)/2 <= m` plus Boumal-Voroninski-Bandeira: once `p(p+1)/2 > m`,
second-order critical points are global optima --- **deterministically**, but only for SDPs
whose factored constraint set is a smooth manifold [27]. Outside that class spurious local
minima exist. SDPNAL+ (ALM + semismooth Newton-CG) sits in between, reaching ~1e-6 on DNN
relaxations of QAP/BIQ that primal-dual IPMs cannot touch [28],[29].

For a dense-block IPM the practical envelope in 2026 is roughly k up to a few thousand with a
reduced Schur complement, and n up to ~2e4 before the dense Cholesky (2.7e12 flops at n = 2e4)
and 3.2 GB of M become the binding constraint. Beyond that: sparse Cholesky of M if M is sparse
(SDPA 7.3.1 adopts MUMPS when `sparse_cost < m^3 * sd_ratio`, decided once [4]), then chordal
conversion, then a different algorithm class.

### 4. Iterative Schur solves: do they work?

Rarely, and only with structure. Toh's PSQMR on the *reduced augmented* system (rather than the
Schur complement, which becomes hopelessly ill-conditioned) solved an SDP with **127,600
constraints to 1e-6 relative gap in about 9.5 hours** [10] --- a scale unreachable by dense
Cholesky, at a cost no one would pay otherwise. Toh-Kojima [9] used conjugate residuals on the
Schur system; Chai-Toh [11] set out the two-phase strategy (PCG on normal equations early,
preconditioned SQMR on the indefinite reduced system once the normal matrix degrades). The
structural obstacle is universal: the NT/HKM scaling's condition number grows by many orders of
magnitude as the iterates approach the boundary, so a preconditioner tuned early fails late.
Every code that actually ships Krylov solves restricts the problem class. Loraine [12] is the
current best example --- a genuine primal-dual IPM using PCG on the dual Schur complement with a
preconditioner built from the **low rank of the data matrices `B_i`**, targeted at truss
topology, sensor-network localization and Lasserre MAXCUT relaxations. This is the honest
reading of the roadmap's Tranche 5 caveat: a generic MINRES/CG on the NT-scaled SDP Schur
complement is not a plan.

### 5. svec, symmetric Kronecker, and the Lyapunov solve

The conventions the leading codes use are uniform. SDPT3 defines
`svec(x) = (x11, sqrt2 x12, x22, sqrt2 x13, sqrt2 x23, x33, ...)`, "the sqrt(2) to make the
operation an isometry" [2] --- the same convention ConicIP already has. The scaling block is
`E^s = P (x) P^-T z`, `F^s = P x (x) P^-T`, with `(x)` the symmetric Kronecker product of
Todd-Toh-Tutuncu [30].

The NT scaling point has a standard factored form. CVXOPT's derivation [8]: compute
`S = L1 L1^T`, `Z = L2 L2^T`, then the SVD `L2^T L1 = U Lambda V^T`, set `lambda = diag(Lambda)`
and `R = L1 V Lambda^(-1/2) = L2^(-T) U Lambda^(1/2)`, whence `R^T Z R = Lambda` and
`R^T S^-1 R = Lambda^-1`. Clarabel implements exactly this (`chol1`, `chol2`,
`factor!(f.SVD, L2'*L1)`, `f.lambda .= f.SVD.s`, `f.R = L1*Vt'*Lisqrt`) [16].
**ConicIP's `nestod_sdc` already does this computation** --- two Choleskys, `svd(Lz'*Ls)`,
`R = (Lz' \ U) .* sqrt.(Lambda)'` --- and discards `Lambda`.

The consequence for `dsdc!`: in lambda-coordinates the Jordan product is
`lambda o u = (Lambda * mat(u) + mat(u) * Lambda)/2` with `Lambda` **diagonal**, so its inverse
is `lambda <> v = mat(v) .* Gamma`, `Gamma_ij = 2/(lambda_i + lambda_j)` --- stated in exactly
that form in [8], and coded as `X[i,j] = 2*Z[i,j]/(lam[i] + lam[j])` in Clarabel's
`lambda_inv_circ_op!` [16]. ConicIP calls `cone_div!(_div_buf, r.s, lambda)` with
`lambda = F*z.v`, i.e. the argument *is* Lambda; the current `lyap(Y, -2X)` runs an O(k^3)
Schur-decomposition-based Sylvester solve on a diagonal matrix, twice per iteration. Replacing
it with the Hadamard form is O(k^2), allocation-free, and independent of the Schur-complement
work. (Floating point caveat: `mat(lambda)` is diagonal only to rounding, so take `diag`
explicitly rather than assuming the off-diagonals are exactly zero; store `lambda` as a vector
at scaling time as Clarabel does.)

### 6. Benchmarks and what "good" looks like

**SDPLIB is no longer the benchmark.** Mittelmann's live sparse-SDP page (updated
**2026-04-25**) has 75 instances, a 40,000 s limit, on an AMD Ryzen 9 5900X (12 cores, 128 GB),
and **contains none of maxG11, maxG32, thetaG51, control11, theta1-6, equalG11, qap*, truss*,
hinf*, mcp***; the theta family now runs theta12/theta102/theta123 and the max-cut family
G40_mb/G48mc/G55mc/G59mc/G60_mb/G60mc [31]. Scaled shifted geometric means (COPT = 1.00):
**COPT 1.00 (75/75 solved), cuLoRADS 3.01 (71/75, on an H100), MOSEK 3.67 (73/75), SDPT3 5.32
(69/75), CSDP 5.40 (70/75), SDPA 7.99 (61/75), SeDuMi 29.9 (62/75)**. Failure is DIMACS error
> 1e-2; a flag for "at least one DIMACS error > 1e-4" is reported separately. The DIMACS error
measures themselves (err1-err6: primal residual, primal cone violation, dual residual, dual cone
violation, relative gap, complementarity) are defined on Mittelmann's DIMACS page and in [32].

For SDPLIB reference points, the best-documented numbers remain SDPA 7's table (Xeon X5550
2.67 GHz, 8 threads, 2009) [4]: **maxG11 (m=800, k=800) SDPA 6.6 s / CSDP 7.1 / SDPT3 7.6 /
SeDuMi 92.9; thetaG51 (m=6910, k=1001) 86.8 / 256.3 / 517.5 / 2579.1; control11 (m=1596, blocks
110+55) 35.9 / 49.4 / 52.8 / 86.9; theta6 (m=4375, k=300) 14.2 / 23.6 / 23.8 / 326.7; maxG60
(m=7000, k=7000) 1483.8 / 1445.5 / 2411.2 / timeout**; and MOSEK 7.0's own single-thread table
(Xeon E3-1270 3.4 GHz, 2012) [5]: maxG11 16 s, maxG32 262 s, maxG51 57 s, thetaG51 137 s,
control11 38 s, theta6 27 s. A useful sanity target: **anything slower than ~10 s on maxG11 or
~150 s on thetaG51 on a 2026 laptop is not in the game.**

SDPLIB dimensions verified against Borchers' own tables [33]: maxG11 m=800 n=800; maxG51
m=1000 n=1000; maxG32 m=2000 n=2000; maxG60 m=7000 n=7000; thetaG51 m=6910 n=1001; thetaG11
m=2401 n=801; control11 m=1596 n=165 (blocks 110 + 55 diagonal); theta6 m=4375 n=300; equalG11
m=801 n=801; equalG51 m=1001 n=1001.

Maintenance status as of 2026: MOSEK 11.2.x actively developed; SDPA at 7.4.4; CSDP 6.2.0;
SDPT3 4.0 last substantive fix 2017; SeDuMi official 1.05R5 (2003) with an unofficial fork;
DSDP effectively dead (superseded by HDSDP); COSMO.jl in maintenance mode; Clarabel, Hypatia.jl
and Loraine.jl actively developed.

---

## (d) What I could not verify

- **Exact FKN 1997 cost expressions.** The F1/F2/F3 cost formulas quoted
  (`n_p f_j + n_p^3`, `n_p(f_j+|I|)`, `2 f_j |I|`) come from a 2025 exposition of SDPT3 [3],
  not from Fujisawa-Kojima-Nakata's own text, which is paywalled. The formula *statements* are
  verified verbatim from SDPA 7's paper [4]. Read [1] directly before quoting constants.
- **Nakata et al. 2003 (matrix completion II) numerical tables** --- not retrieved; the
  characterization above is abstract-level only.
- **Loraine's exact speedup factors** over MOSEK/SDPT3 and the algebraic form of its
  preconditioner --- the PDF resisted extraction. "2-10x" is a secondary characterization, not
  a verified table row.
- **SDPNAL+ table numbers** (theta/BIQ/QAP) and its current maintenance status --- unverified;
  check Toh's group page directly.
- **Whether MOSEK performs chordal decomposition internally** --- not found in the official docs
  either way. Do not claim absence.
- **Modern (2024-2026) timings for the legacy SDPLIB instances** --- nobody publishes them; all
  SDPLIB numbers above are 2009/2012 hardware.
- **cuHALLaR (arXiv:2505.13719)** numeric tables --- not extracted.
- One task item named the completion theorem as "Agler/Griffin/Grone-Johnson-Marcus-Rodman";
  the correct attributions are **Grone-Johnson-Sa-Wolkowicz (1984)** for completion and
  **Agler-Helton-McCullough-Rodman (1988)** for the clique-sum decomposition.

---

## References

1. K. Fujisawa, M. Kojima, K. Nakata, "Exploiting sparsity in primal-dual interior-point methods for semidefinite programming," *Math. Program.* 79:235-253, 1997. https://doi.org/10.1007/BF02614319
2. R.H. Tutuncu, K.C. Toh, M.J. Todd, "Solving semidefinite-quadratic-linear programs using SDPT3," *Math. Program.* 95:189-217, 2003. https://people.orie.cornell.edu/miketodd/tttmpbc.pdf
3. "A Study on the Algorithm and Implementation of SDPT3," arXiv:2512.24623, 2025. https://arxiv.org/html/2512.24623
4. M. Yamashita, K. Fujisawa, K. Nakata, M. Nakata, M. Fukuda, K. Kobayashi, K. Goto, "A high-performance software package for semidefinite programs: SDPA 7," Optimization Online, 2010. https://optimization-online.org/wp-content/uploads/2010/01/2531.pdf
5. J. Dahl, "Semidefinite Optimization using MOSEK," ISMP 2012 slides, MOSEK ApS. https://docs.mosek.com/slides/ismp2012/sdo.pdf
6. S.J. Benson, Y. Ye, "Algorithm 875: DSDP5 --- software for semidefinite programming," *ACM TOMS* 34(3), 2008. https://web.stanford.edu/~yyye/DSDP5-Paper.pdf
7. W. Gao, D. Ge, Y. Ye, "HDSDP: Software for Semidefinite Programming," arXiv:2207.13862, 2022/2023. https://arxiv.org/abs/2207.13862
8. L. Vandenberghe, "The CVXOPT linear and quadratic cone program solvers," 2010. https://www.seas.ucla.edu/~vandenbe/publications/coneprog.pdf
9. K.C. Toh, M. Kojima, "Solving some large scale semidefinite programs via the conjugate residual method," *SIAM J. Optim.* 12(3):669-691, 2002.
10. K.C. Toh, "Solving large scale semidefinite programs via an iterative solver on the augmented systems," *SIAM J. Optim.* 14:670-698, 2004. https://optimization-online.org/2003/01/596/
11. Y.Y. Chai, K.C. Toh, "Preconditioning and iterative solution of symmetric indefinite linear systems arising from interior point methods for linear programming," *COAP* 36:221-247, 2007. https://doi.org/10.1007/s10589-006-9006-8
12. A. Habibi, M. Kocvara, M. Stingl, "Loraine --- an interior-point solver for low-rank semidefinite programming," *Optim. Methods Softw.*, 2023/2024. https://doi.org/10.1080/10556788.2023.2250522 and https://github.com/kocvara/Loraine.jl
13. M. Fukuda, M. Kojima, K. Murota, K. Nakata, "Exploiting sparsity in semidefinite programming via matrix completion I: general framework," *SIAM J. Optim.* 11(3):647-674, 2001. https://doi.org/10.1137/S1052623400366218
14. K. Nakata, K. Fujisawa, M. Fukuda, M. Kojima, K. Murota, "... II: implementation and numerical results," *Math. Program.* 95:303-327, 2003. https://doi.org/10.1007/s10107-002-0351-9
15. S. Kim, M. Kojima, M. Mevissen, M. Yamashita, "Exploiting sparsity in linear and nonlinear matrix inequalities via positive semidefinite matrix completion," *Math. Program.* 129:33-68, 2011. https://doi.org/10.1007/s10107-010-0402-6 (SparseCoLO)
16. L. Vandenberghe, M.S. Andersen, "Chordal Graphs and Semidefinite Optimization," *Foundations and Trends in Optimization* 1(4):241-433, 2015. https://doi.org/10.1561/2400000006 --- and Clarabel.jl source, `src/cones/coneops_psdtrianglecone.jl`, `src/chordal/`, https://github.com/oxfordcontrol/Clarabel.jl
17. M.S. Andersen, J. Dahl, L. Vandenberghe, "Implementation of nonsymmetric interior-point methods for linear optimization over sparse matrix cones," *Math. Program. Comput.* 2(3-4):167-201, 2010.
18. M.S. Andersen, L. Vandenberghe, "CHOMPACK: a Python package for chordal matrix computations," 2015.
19. Y. Zheng, G. Fantuzzi, A. Papachristodoulou, P. Goulart, A. Wynn, "Chordal decomposition in operator-splitting methods for sparse semidefinite programs," *Math. Program.* 180:489-532, 2020. arXiv:1707.05058 (CDCS)
20. P. Goulart, Y. Chen, "Clarabel: an interior-point solver for conic programs with quadratic objectives," *Math. Program. Comput.*, 2026. arXiv:2405.12762
21. M. Garstka, M. Cannon, P. Goulart, "A clique graph based merging strategy for decomposable SDPs," IFAC-PapersOnLine, 2020. arXiv:1911.05615 --- and "COSMO: a conic operator splitting method for convex conic problems," *JOTA* 190(3):779-810, 2021. arXiv:1901.10887
22. R.Y. Zhang, "Complexity of chordal conversion for sparse semidefinite programs with small treewidth," *Math. Program.*, 2024/2025. arXiv:2306.15288 --- and R.Y. Zhang, J. Lavaei, arXiv:1710.03475
23. S. Burer, R.D.C. Monteiro, "A nonlinear programming algorithm for solving semidefinite programs via low-rank factorization," *Math. Program.* 95:329-357, 2003. https://doi.org/10.1007/s10107-002-0352-8 (SDPLR)
24. Q. Han, Z. Lin, H. Liu, C. Chen, Q. Deng, D. Ge, Y. Ye, "Accelerating low-rank factorization-based semidefinite programming algorithms on GPU," arXiv:2407.15049, 2024. https://github.com/COPT-Public/cuLoRADS
25. R.D.C. Monteiro, A. Sujanani, D. Cifuentes, "A low-rank augmented Lagrangian method for large-scale semidefinite programming based on a hybrid convex-nonconvex approach," arXiv:2401.12490, 2024 (HALLaR)
26. A. Yurtsever, J. Tropp, O. Fercoq, M. Udell, V. Cevher, "Scalable semidefinite programming," *SIAM J. Math. Data Sci.* 3(1):171-200, 2021. arXiv:1912.02949 (SketchyCGAL)
27. N. Boumal, V. Voroninski, A. Bandeira, "Deterministic guarantees for Burer-Monteiro factorizations of smooth semidefinite programs," *CPAM* 73(3):581-608, 2020. arXiv:1804.02008 --- and NeurIPS 2016, arXiv:1606.04970
28. X.-Y. Zhao, D. Sun, K.-C. Toh, "A Newton-CG augmented Lagrangian method for semidefinite programming," *SIAM J. Optim.* 20(4):1737-1765, 2010
29. L. Yang, D. Sun, K.-C. Toh, "SDPNAL+: a majorized semismooth Newton-CG augmented Lagrangian method for SDP with nonnegative constraints," *Math. Program. Comput.* 7:331-366, 2015. arXiv:1406.0942
30. M.J. Todd, K.C. Toh, R.H. Tutuncu, "On the Nesterov-Todd direction in semidefinite programming," *SIAM J. Optim.* 8(3):769-796, 1998. https://ecommons.cornell.edu/entities/publication/0f9e262e-5755-49ce-ba87-addb29e68f9f
31. H.D. Mittelmann, "Sparse SDP benchmark," https://plato.asu.edu/ftp/sparse_sdp.html, updated 2026-04-25 (accessed 2026-09-20); index https://plato.asu.edu/bench.html
32. H.D. Mittelmann, "An independent benchmarking of SDP and SOCP solvers," *Math. Program.* 95:407-430, 2003; DIMACS error measures at https://plato.asu.edu/dimacs/node3.html
33. B. Borchers, "SDPLIB 1.2, a library of semidefinite programming test problems," *Optim. Methods Softw.* 11(1):683-716, 1999.
