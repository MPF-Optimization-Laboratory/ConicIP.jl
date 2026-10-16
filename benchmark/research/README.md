# Research-phase artifacts, 2026-09-20 (private, never upstream)

Companion to `../next-level-research.md`, which is the synthesis. This directory keeps the
full subagent reports, the scripts, and the results they produced, so the experiments can be
rerun and the literature surveys re-read without the session scratchpad.

| directory | contents |
|---|---|
| `SYNTHESIS.md`, `BRIEF.md` | the coordinator's synthesis and the brief every subagent read |
| `lit-linalg/` | KKT linear-algebra survey (supernodal, GPU, Krylov, factorization reuse), 32 refs |
| `lit-algorithms/` | iteration-count and finite-termination survey (correctors, crossover, HSD, warm starts, hybrids) |
| `lit-sdp/` | SDP-at-scale survey (reduced Schur complement, chordal, low-rank, benchmarks); `clarabel_psd.jl` is the Clarabel PSD-cone excerpt consulted |
| `exp-factor/` | factorization headroom: `capture.jl` (KKT capture via `LDLGen.gen` fields), Part A/B/C/D scripts, per-instance CSV/stats, `env/` scratch Project+Manifest |
| `exp-krylov/` | Krylov viability: `capture.jl` (pattern-snapshot capture), `experiments.jl`, per-instance `res-*.csv`, `accounting.txt`, `full.log`, scratch Project+Manifest |
| `exp-activeset/` | Tapia / Ye identification: `lib.jl` (instances, capture, projection kernel), `run_lp2.jl`, `run_socp.jl`, `check_capture.jl`, raw logs |

**Not kept** (regenerable or third-party): the 161 MB of captured KKT sequences
(`exp-krylov/data/*.jls`, rebuilt by `exp-krylov/capture.jl`), the downloaded paper PDFs and
text extracts, the Netlib tarballs (`https://sparse.tamu.edu/MM/LPnetlib/lp_<name>.tar.gz`),
and the copies of `roadmap.md` / `hsd-design.md` / `suite.jl` that the brief pointed at.

**Paths.** The scripts and the "Exact commands" blocks in the reports refer to the session
scratchpad (`/private/tmp/claude-502/…/scratchpad/<agent>`). Substitute this directory. Scratch
environments: `julia --project=benchmark/research/exp-krylov` (or `exp-factor/env`) after
`Pkg.develop(path=".")` from the worktree root; they were built from
`benchmark/Project.toml`/`Manifest.toml` plus Krylov, KrylovKit, LimitedLDLFactorizations,
LDLFactorizations, QDLDL 0.4.1. Instances are built the way `benchmark/suite.jl` does.
