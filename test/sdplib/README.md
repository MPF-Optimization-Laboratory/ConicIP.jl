# SDPLIB instances

A six-instance subset of **SDPLIB 1.2**, used by `test/sdplib_tests.jl` as an
end-to-end validation gate: third-party SDP data with independently published
optimal values, read through `MOI.FileFormats.SDPA` and solved through
ConicIP's MOI wrapper.

**No instance data is stored in this repository.** The files are fetched at
test time from the pinned upstream commit below, verified against the sha256
table, and cached in a ConicIP-owned scratch space
(`<depot>/scratchspaces/d92ec50d-.../sdplib`), so each is downloaded at most
once per machine and shared with `benchmark/sdplib.jl`. `instances.jl` in
this directory holds the URLs, the hash table and the fetch logic; it is
included by both entry points.

With no network the whole testset is skipped and `Pkg.test()` stays green. A
sha256 mismatch is never skipped: the bad file is deleted and the testset
fails, because a wrong instance must never be solved and reported as a pass.

## Citation

> B. Borchers. "SDPLIB 1.2, a library of semidefinite programming test
> problems." *Optimization Methods and Software* **11** (1999) 683–690.

## Provenance

- Source: <https://github.com/vsdp/SDPLIB>, `data/<name>.dat-s`
- Pinned commit: `fa11b45c1d8c896a6abad2648d5dad46d8ecefaa`
- Raw URL pattern (pinned, not `master`, so the gate cannot change meaning
  because upstream moved):
  `https://raw.githubusercontent.com/vsdp/SDPLIB/fa11b45c1d8c896a6abad2648d5dad46d8ecefaa/data/<name>.dat-s`
- Hashes recorded 2026-09-06

Set `CONICIP_SDPLIB_URL` to override the base URL (used to exercise the
offline skip path).

The `vsdp/SDPLIB` repository is a modern re-hosting of Brian Borchers' original
SDPLIB 1.2, formerly at `http://euler.nmt.edu/~brian/sdplib/sdplib.html`.

### Checksums

| File              | sha256                                                             |
| ----------------- | ------------------------------------------------------------------ |
| `control1.dat-s`  | `482528bb128e64dad102fab88e4e8b7074efdfa22e396ebec586d832b1545bcb` |
| `hinf1.dat-s`     | `a2d3e9f340f304fe59147e5f7d8b3c54c8169cebe946d81009796c184164ab77` |
| `infd1.dat-s`     | `4cbb4dcd44caa57c6970db23905971ed144f1046b663dfb828decda51d12acd8` |
| `infp1.dat-s`     | `c81f23ce297cd489c0500076677d6c70727fb1e761ca21d53398498e8192dd45` |
| `theta1.dat-s`    | `e957517b2284f24eba158db56a0ae34ecc07d24fa299a31f732dad3d4a54ea34` |
| `truss1.dat-s`    | `07bfaa5beaee8d2df2188a7aff80abe307a176466824211d68ffe68764c6efca` |

These are the six the test gate uses; `instances.jl` additionally records
hashes for the twelve further instances `benchmark/sdplib.jl` solves
(`arch0`, `control2`, `control3`, `gpp100`, `hinf2`, `hinf3`, `mcp100`,
`mcp124-1`, `qap5`, `qap6`, `theta2`, `truss2`–`truss4`). Every fetch is
checked; `instances.jl` is the authoritative table and this one is a copy for
readers.

Verify a populated cache (any platform) with

```
julia --project=test -e 'using Scratch, SHA
    d = get_scratch!(Base.UUID("d92ec50d-21ac-4ea5-850a-0588cd9e47b8"), "sdplib")
    for f in filter(endswith(".dat-s"), readdir(d))
        println(bytes2hex(open(sha256, joinpath(d, f))), "  ", f)
    end'
```

## Reference optimal objective values

Quoted verbatim from the SDPLIB 1.2 README at the commit above. `m` is the
number of variables of the SDPA primal, `n` the total order of the block
diagonal matrix.

| Problem  |   m |  n | Optimal objective value | Family                       |
| -------- | --: | -: | ----------------------: | ---------------------------- |
| control1 |  21 | 15 |           `1.778463e+01` | control / system theory      |
| hinf1    |  13 | 14 |             `2.0326e+00` | LMI from control engineering |
| infd1    |  10 | 30 |         *dual infeasible* | Todd's infeasible pair       |
| infp1    |  10 | 30 |       *primal infeasible* | Todd's infeasible pair       |
| theta1   | 104 | 50 |           `2.300000e+01` | Lovász theta                 |
| truss1   |   6 | 13 |          `-8.999996e+00` | truss topology design        |

The SDPLIB README states these values are "based on the SDPA conventions", so
"primal" and "dual" above mean SDPA's primal and dual.

### Sign and dual conventions

SDPA's *primal* is the geometric conic form

```
P:  min cᵀx   s.t.   Σᵢ Fᵢ xᵢ − F₀ ⪰ 0
```

which is exactly what `MOI.FileFormats.SDPA` imports (as
`VectorAffineFunction`-in-`PositiveSemidefiniteConeTriangle` /
`Nonnegatives`, with a `MIN_SENSE` affine objective). The imported model is
therefore SDPA's primal, not its dual, and:

- `MOI.ObjectiveValue()` matches the table above **with no sign flip**;
- `infp1` (SDPA-primal infeasible) makes the MOI model `INFEASIBLE`;
- `infd1` (SDPA-dual infeasible) makes the MOI model `DUAL_INFEASIBLE`
  (i.e. the imported primal is unbounded).

`test/sdplib_tests.jl` re-derives all three of these from the constraint data
rather than trusting the solver.

## Solver behaviour by tolerance

Measured 2026-09-06 on the solver-fix branch (the SDP line-search and
`:Error` contract fix, and the Jordan-product `vecm(I)` fix), via
`julia --project benchmark/sdplib.jl`. These numbers are referenced by the
docs; regenerate them with the same script if the solver changes.

`rel.err` is `|obj − ref| / (1 + |ref|)` against the reference table above and
the wider SDPLIB table embedded in `benchmark/sdplib.jl`.

### optTol = 1e-8 (the default, and what the CI gate uses)

| instance |   m |   n | status          | iter | rel.err |
| -------- | --: | --: | --------------- | ---: | ------: |
| control1 |  21 |  15 | OPTIMAL         |   44 | 1.7e-07 |
| control2 |  66 |  30 | NUMERICAL_ERROR |   42 |         |
| control3 | 136 |  45 | ITERATION_LIMIT |   45 |         |
| hinf1    |  13 |  14 | ITERATION_LIMIT |   31 |         |
| hinf2    |  13 |  16 | NUMERICAL_ERROR |   25 |         |
| hinf3    |  13 |  16 | NUMERICAL_ERROR |   28 |         |
| truss1   |   6 |  13 | OPTIMAL         |   16 | 2.8e-08 |
| truss2   |  58 | 133 | OPTIMAL         |   22 | 3.5e-07 |
| truss3   |  27 |  31 | OPTIMAL         |   13 | 1.2e-08 |
| truss4   |  12 |  19 | OPTIMAL         |   12 | 2.1e-08 |
| theta1   | 104 |  50 | OPTIMAL         |   13 | 1.4e-09 |
| theta2   | 498 | 100 | OPTIMAL         |   12 | 2.2e-08 |
| mcp100   | 100 | 100 | OPTIMAL         |   11 | 2.1e-07 |
| mcp124-1 | 124 | 124 | OPTIMAL         |   12 | 1.6e-07 |
| gpp100   | 101 | 100 | NUMERICAL_ERROR |   18 |         |
| qap5     | 136 |  26 | OPTIMAL         |   15 | 1.0e-10 |
| qap6     | 229 |  37 | ITERATION_LIMIT |   52 |         |
| arch0    | 174 | 335 | OPTIMAL         |   31 | 2.0e-07 |

Twelve of eighteen solve to 1e-7 or better. The six that do not are the
degenerate ones, and they now fail *cleanly* — a status and a diagnostic
message, never an exception.

### The seven hard instances at looser tolerances

| instance | optTol = 1e-6            | optTol = 1e-4            |
| -------- | ------------------------ | ------------------------ |
| control2 | OPTIMAL, 3.3e-07 (41 it) | OPTIMAL, 3.9e-05 (39 it) |
| control3 | OPTIMAL, 2.2e-07 (45 it) | OPTIMAL, 3.8e-05 (43 it) |
| hinf1    | ITERATION_LIMIT          | OPTIMAL, 8.9e-04 (14 it) |
| hinf2    | OPTIMAL, 2.2e-05 (31 it) | OPTIMAL, 6.2e-05 (21 it) |
| hinf3    | NUMERICAL_ERROR          | OPTIMAL, 9.6e-04 (22 it) |
| gpp100   | OPTIMAL, 1.9e-07 (17 it) | OPTIMAL, 3.3e-05 (9 it)  |
| qap6     | ITERATION_LIMIT          | OPTIMAL, 3.3e-04 (15 it) |

Every one of them converges once the requested tolerance is relaxed, so
these are conditioning limits rather than wrong answers. `control2`,
`control3` and `gpp100` reach full 1e-7 accuracy at optTol = 1e-6; only the
hinf family and `qap6` need 1e-4.

The hinf family lacks strict complementarity — the optimal primal and dual
slacks are singular on a common face — so the Nesterov–Todd scaling blows up
as the iterates approach it and the KKT system becomes ill-conditioned once
feasibility reaches ~1e-6. `test/sdplib_tests.jl` pins both halves of that
contract for `hinf1`: a clean non-convergence status at 1e-8, and actual
convergence to the published optimum at 1e-4.

## Licensing

**ConicIP redistributes no SDPLIB data.** This repository stores only URLs,
hashes and reference values; the instances are fetched by the user's own
machine at test time, from upstream, and cached outside the repository. That
is the reason for the download-on-demand design, and it is why the note below
is informational rather than a constraint on this package.

For reference: the `vsdp/SDPLIB` repository is published under the **GNU
General Public License v3.0** (`LICENSE` at its root; GitHub reports
`spdx_id: GPL-3.0`), while ConicIP.jl is MIT-licensed (see `LICENSE.md`). The
original SDPLIB 1.2 distribution by Borchers carried no explicit licence, and
the files are numerical problem instances rather than program code. Anyone
redistributing the instances themselves — rather than fetching them, as this
package does — should satisfy themselves about those terms.
