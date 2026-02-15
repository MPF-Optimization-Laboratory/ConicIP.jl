# ConicIP.jl Performance Report

**Date:** 2026-02-14
**Julia:** 1.12.3 (aarch64-apple-darwin)
**ConicIP:** v0.2.0

## 1. Executive Summary

**Top 3 findings:**

1. **`Block.Blocks::Vector{Any}` causes pervasive type instability** — every `F*x`, `inv(F)`, `F'*x` call incurs dynamic dispatch per block. The `@code_warntype` audit shows `::ANY` annotations on all block element accesses and operations. This is the single largest architectural bottleneck.

2. **KKT solver choice dominates wall time by 10–100x** — `kktsolver_qr` materializes full Q (O(n²) dense matrix at `kktsolvers.jl:25`), making it 55x slower than `pivot(2x2)` on sparse problems (0.89s vs 0.007s for Box QP sparse n=1000). Conversely, `pivot(2x2)` is 83x slower on single large SOC (0.82s vs 0.01s) due to `sparse(F'F)` materialization of dense SymWoodbury blocks.

3. **Allocation pressure is moderate but solver-dependent** — GC overhead reaches 16% on `kktsolver_qr` with large sparse problems (1.26 GB allocated for 7 iterations). The `sparse(SymWoodbury Block)` call alone allocates 56 MB for a 500-dim block. Per-cone primitives (`drp!`, `xrp!`, `dsoc!`, `xsoc!`) are zero-allocation, but `÷` and `∘` allocate a fresh `zeros(m,1)` every call.

**Estimated total speedup potential:** 2–5x on typical problems from Tier 1 fixes alone; up to 50x on pathological solver-mismatch cases from better solver selection.

## 2. Methodology

### Problem Suite

| # | Name | Cone | n | m | p | Key Stress |
|---|------|------|---|---|---|------------|
| 1 | Box QP, dense Q | R | 500 | 1000 | 0 | Dense Hessian mat-vec |
| 2 | Box QP, sparse Q | R | 1000 | 2000 | 0 | Sparse KKT advantage |
| 3 | Single large SOC | Q | 500 | 501 | 0 | SymWoodbury scaling |
| 4 | Many small SOCs | Q×250 | 500 | 750 | 0 | Block iteration overhead |
| 5 | Small SDP (k=10) | S | 55 | 55 | 0 | mat/vecm, VecCongurance |
| 6 | Larger SDP (k=30) | S | 465 | 465 | 0 | SDP scaling stress |
| 7 | Mixed R+Q + eq. | R,Q | 200 | 251 | 10 | Mixed Block, equality path |
| 8 | Mixed R+Q+S | R,Q,S | 86 | 86 | 0 | All three cone types |

### Tools
- `@timed` (3 trials, report median) for macro timing
- `@profile` with `format=:flat` for statistical profiling
- `@allocated` for micro allocation measurement
- `@code_warntype` for type stability audit

### Protocol
- `Random.seed!(42)` for all problem generation
- Warmup call with `maxIters=3` before each timed run
- `GC.gc()` before each trial

## 3. Results

### 3a. Macro Timing Table

| Problem | Solver | Time (s) | Alloc (MB) | GC% | Iters | Status |
|---------|--------|----------|------------|-----|-------|--------|
| Box QP dense (500) | kktsolver_qr | 0.149 | 321.9 | 0.8 | 8 | Optimal |
| Box QP dense (500) | kktsolver_sparse | 0.430 | 670.0 | 1.3 | 8 | Optimal |
| Box QP dense (500) | pivot(2x2) | 0.083 | 529.2 | 2.7 | 8 | Optimal |
| Box QP sparse (1000) | kktsolver_qr | **0.894** | **1262.4** | **16.0** | 7 | Optimal |
| Box QP sparse (1000) | kktsolver_sparse | 0.016 | 43.2 | 0.0 | 7 | Optimal |
| Box QP sparse (1000) | pivot(2x2) | **0.007** | 22.5 | 0.0 | 7 | Optimal |
| Single SOC (500) | kktsolver_qr | 0.076 | 164.2 | 0.0 | 6 | Optimal |
| Single SOC (500) | kktsolver_sparse | **0.010** | 19.0 | 0.0 | 6 | Optimal |
| Single SOC (500) | pivot(2x2) | **0.821** | **772.9** | 1.3 | 6 | Optimal |
| Many SOCs (250×3) | kktsolver_qr | 0.140 | 305.8 | 1.9 | 9 | Optimal |
| Many SOCs (250×3) | kktsolver_sparse | **1.008** | 566.3 | 0.5 | 9 | Optimal |
| Many SOCs (250×3) | pivot(2x2) | 0.201 | 589.9 | 1.9 | 9 | Optimal |
| Small SDP (k=10) | kktsolver_qr | 0.002 | 4.1 | 0.0 | 5 | Optimal |
| Small SDP (k=10) | kktsolver_sparse | 0.002 | 4.5 | 0.0 | 5 | Optimal |
| Small SDP (k=10) | pivot(2x2) | 0.001 | 3.0 | 0.0 | 5 | Optimal |
| Larger SDP (k=30) | kktsolver_qr | 0.098 | 204.4 | 3.2 | 5 | Optimal |
| Larger SDP (k=30) | kktsolver_sparse | 0.067 | 193.6 | 0.0 | 5 | Optimal |
| Larger SDP (k=30) | pivot(2x2) | 0.027 | 115.4 | 0.0 | 5 | Optimal |
| Mixed R+Q+eq (200) | kktsolver_qr | 0.025 | 52.4 | 0.0 | 11 | Optimal |
| Mixed R+Q+eq (200) | kktsolver_sparse | 0.048 | 39.6 | 0.0 | 11 | Optimal |
| Mixed R+Q+eq (200) | pivot(2x2) | 0.040 | 161.9 | 0.0 | 11 | Optimal |
| Mixed R+Q+S (86) | kktsolver_qr | 0.005 | 7.8 | 0.0 | 8 | Optimal |

**Key observations:**
- **No single solver is best across all problems.** `kktsolver_qr` is worst for sparse R+ problems (55x slower), `pivot(2x2)` is worst for single large SOC (83x slower), `kktsolver_sparse` is worst for many small SOCs (7x slower).
- **GC is only significant for `kktsolver_qr` on large sparse problems** (16% at n=1000), driven by full Q materialization allocating 1.26 GB.
- **SDP problems are small and fast** — profiler collected 0 samples for k=10 SDP. Even k=30 SDP completes in 27ms.

### 3b. Statistical Profile Highlights

**Box QP sparse (kktsolver_qr) — 177 total samples, 50% utilization:**

| Samples | Function | Location |
|---------|----------|----------|
| 147 | `GenericMemory` (allocation) | `boot.jl:588` |
| 127 | `solve3x3gen` (QR KKT build) | `kktsolvers.jl:33` — `Atil = F⁻ᵀ*Matrix(A)` |
| 126 | `Array` (sparse→dense) | `sparsematrix.jl:974` |
| 24 | `solve3x3gen` line 34 | `kktsolvers.jl:34` — `QpAᵀA = Q + Atil'Atil` |
| 24 | `*` (matmul) | `matmul.jl:136` |
| 16 | `Matrix(A::Block)` | `blockmatrices.jl:148` |

The QR solver spends **72% of samples** inside `solve3x3gen` (lines 32–34), which materializes `F⁻ᵀ` as a dense matrix and forms `Atil = F⁻ᵀ*Matrix(A)` every iteration.

**Single SOC — 18 total samples (problem too fast for good profiling):**
- `solve4x4gen` and `conicIP` dominate, with `*` (matmul) and `solve3x3gen` appearing.

### 3c. Allocation Hotspots

| Operation | Bytes | Notes |
|-----------|-------|-------|
| `sparse(SymWoodbury Block(500))` | **56.1 MB** | Materializes dense 500×500 matrix then converts |
| `Diagonal Block(500) * Matrix` | 8,576 | Small: just output allocation |
| `SymWoodbury Block(500) * Matrix` | 17,088 | Moderate: WoodburyMatrices multiply |
| `VecCongurance Block(55) * Matrix` | 4,160 | Small |
| `inv(SymWoodbury Block)` | 25,952 | Creates new SymWoodbury |
| `inv(SymWoodbury Block)'` | 39,312 | inv + adjoint = 2 allocations |
| `inv(VecCongurance Block)'` | 7,456 | inv + adjoint |
| `drp!`, `xrp!`, `dsoc!`, `xsoc!` | **0** | In-place, zero-alloc — well designed |
| `dsdc!(55)` | 12,560 | `mat`/`vecm` + `lyap` allocations |
| `xsdc!(55)` | 5,264 | `mat` + matrix multiply |
| `mat(k=30)` | 8,272 | Scalar loop builds dense matrix |
| `vecm(k=30)` | 4,176 | Scalar loop builds vector |

**Per-iteration allocation budget (estimated for Box QP sparse, n=1000):**
- `÷` and `∘` each allocate `zeros(2000,1)` = 16 KB × ~6 calls/iter = ~96 KB/iter
- `inv(F)'` recomputed each iteration (line 690): allocates new Block
- `F*x` calls through `broadcastf` allocate output each time
- KKT solver rebuilds factorization each iteration

### 3d. Type Stability Findings

**`Block.Blocks::Vector{Any}` (blockmatrices.jl:30)** is the root cause of all type instability:

```
%22 = Base.getindex(%20, %21)::ANY       ← block element access returns Any
%24 = (op)(%22, %23)::ANY                ← operation result is Any
```

This pattern appears in every `broadcastf` call, which underlies:
- `F * x` (Block multiply)
- `inv(F)` (Block inversion)
- `F'` (Block adjoint)
- `F' * x` (adjoint multiply)

The `::ANY` return type forces Julia to use dynamic dispatch for every block operation at every iteration. With 250 blocks (many small SOCs case), this means 250 dynamic dispatches per `F*x` call.

**`typeof(F.Blocks) = Vector{Any}`** — confirmed by runtime inspection.

## 4. Bottleneck Analysis — 3 Tiers

### Tier 1: Critical (≥20% of time or pervasive type instability)

**1a. `Block.Blocks::Vector{Any}` type instability** (`blockmatrices.jl:30`)
- **Impact:** Every `F*x`, `inv(F)`, `F'*x` call goes through dynamic dispatch. Affects ALL problem types on EVERY iteration.
- **Evidence:** `@code_warntype` shows `::ANY` on all block element access. With 250 blocks (many SOC case), this is 250 dynamic dispatches per block multiply.
- **Root cause:** The `Block` struct stores heterogeneous block types (Diagonal, SymWoodbury, VecCongurance) in `Vector{Any}`.

**1b. KKT solver materializations**
- **`kktsolver_qr` full Q materialization** (`kktsolvers.jl:25`): `F.Q * Matrix(I, n, n)` creates an n×n dense matrix at KKT solver construction time. Then at every iteration, `F⁻ᵀ = Matrix(inv(F))'` and `Atil = F⁻ᵀ*Matrix(A)` (lines 32–33) re-densify. This dominates 72% of profiler samples on sparse problems.
- **`sparse(F'F)` in kktsolver_sparse** (`kktsolvers.jl:252`): For SymWoodbury blocks, `sparse()` first calls `Matrix()` (column-by-column), creating a dense 500×500 matrix (56 MB for n=500) and then converting to sparse. Called every iteration.
- **`sparse(F⁻ᵀ)` in kktsolver_2x2** (`kktsolvers.jl:289`): Same materialization issue as above.

### Tier 2: High (5–20% of time or 10–30% of allocations)

**2a. `÷(x,y)` and `∘(x,y)` allocate `zeros(m,1)` per call** (`ConicIP.jl:599,614`)
- Called 4–10 times per iteration (predictor, corrector, refinement steps).
- For m=2000 (Box QP sparse), each call allocates 16 KB. Over 7 iterations with ~6 calls each: ~672 KB. Not dominant but unnecessary.

**2b. `inv(F)'` recomputed every iteration** (`ConicIP.jl:690`)
- `F⁻ᵀ = inv(F)'` creates a new Block (via `broadcastf(inv, F)`) then takes adjoint (via `broadcastf(adjoint, ...)`). Two full Block traversals with dynamic dispatch per block.
- For SymWoodbury: `inv` allocates 26 KB, then `adjoint` allocates 39 KB → 65 KB per iteration.

**2c. String-based cone dispatch** in `÷`/`∘`/`maxstep`/`nt_scaling`
- Uses `if btype == "R"; ... elseif btype == "Q"; ...` chains. String comparison is cheap but prevents compiler specialization and creates unnecessary allocation of string iterators.

### Tier 3: Low (<5% of time, fix after Tiers 1–2)

**3a. `mat()`/`vecm()` scalar loops for SDP** (`ConicIP.jl:89–147`)
- Scalar element-by-element loops with conditional branches. Could use BLAS-style vectorized operations.
- Only relevant for SDP problems, which are already fast at tested sizes.

**3b. `VecCongurance` column-by-column materialization** (`ConicIP.jl:67–75`)
- `Matrix(W::VecCongurance)` applies W to each column of an identity matrix. O(n³) with n full matrix multiplies.
- Only called when `sparse(Block)` triggers `sparse(VecCongurance)`.

**3c. Redundant norm computations** (`ConicIP.jl:719–720`)
- `norm(c)` and `normsafe(b)` computed every iteration, but `normc`/`normb` are already computed at lines 520–522.

**3d. `v4x1` arithmetic allocates new structs** (`ConicIP.jl:37–38`)
- `+(a::v4x1, b::v4x1)` and `-(a::v4x1, b::v4x1)` create new v4x1 with 4 new Matrix allocations each. Used in iterative refinement (line 872: `Δz = Δz + Δzr`).
- `axpy4!` exists and is used for the main step (line 884) but not for refinement accumulation.

## 5. Improvement Plan

### Phase 1: Type-Stable Block (Tier 1a) — Highest Impact

**Goal:** Eliminate dynamic dispatch in all block operations.

**Approach:** Replace `Vector{Any}` with a type-stable container. Two options:

- **Option A (minimal change):** Use a `Tuple` of homogeneous `Vector`s grouped by block type:
  ```julia
  struct TypedBlock
      diag_blocks::Vector{Diagonal{Float64, Vector{Float64}}}
      diag_ranges::Vector{UnitRange{Int}}
      sw_blocks::Vector{SymWoodbury{...}}
      sw_ranges::Vector{UnitRange{Int}}
      vc_blocks::Vector{VecCongurance}
      vc_ranges::Vector{UnitRange{Int}}
  end
  ```
  Each `broadcastf` iterates over typed vectors — no dynamic dispatch within each group.

- **Option B (more invasive):** Parameterize `Block{T}` and require all blocks to be the same type. Works well for pure R+ or pure SOC problems but requires union-splitting or separate storage for mixed problems.

**Recommendation:** Option A — it handles mixed cone types naturally and requires no changes to the solver loop.

### Phase 2: KKT Solver Allocation Reduction (Tier 1b)

1. **`kktsolver_qr`:** Avoid `Matrix(G')` full Q materialization. Use `lmul!` / `rmul!` with the QR object directly instead of extracting Q as a dense matrix.
2. **`kktsolver_sparse`:** Implement `sparse(::SymWoodbury)` directly using the Woodbury structure (diagonal + rank-1 update) rather than going through dense `Matrix()`. This eliminates the 56 MB allocation.
3. **`kktsolver_2x2`:** Same — implement `sparse(F⁻ᵀ)` directly.
4. **Cache KKT symbolic factorization** across iterations when sparsity structure doesn't change (it often doesn't for R+ cones).

### Phase 3: Per-Iteration Allocation (Tier 2)

1. **Pre-allocate `÷`/`∘` output buffers** — allocate once before the iteration loop, reuse each call. Change from returning `zeros(m,1)` to writing into a pre-allocated buffer.
2. **Cache `inv(F)'`** — compute `F⁻ᵀ` once per iteration and store it, avoiding redundant `inv` + `adjoint` calls. Or better, implement `ldiv!` for Block to avoid materializing the inverse entirely.
3. **Use `axpy4!` for refinement accumulation** — replace `Δz = Δz + Δzr` (line 872) with `axpy4!(1.0, Δzr, Δz)`.

### Phase 4: Minor Optimizations (Tier 3)

1. Replace `norm(c)` at line 719 with pre-computed `normc` (line 520).
2. Vectorize `mat()`/`vecm()` using triangular indexing.
3. Implement direct `sparse()` for `VecCongurance` using Kronecker product structure.

### Sequencing

| Phase | Target | Expected Impact | Complexity |
|-------|--------|----------------|------------|
| 1 | TypedBlock | 2–3x on mixed/many-block problems | Medium |
| 2 | KKT materialization | 2–10x on sparse problems with wrong solver | Medium |
| 3 | Allocation reduction | 10–30% across all problems | Low |
| 4 | Minor opts | <5% | Low |

Phases 1 and 2 are independent and can be done in parallel. Phase 3 depends on Phase 1 (buffer pre-allocation needs stable types). Phase 4 can be done anytime.
