#!/usr/bin/env python3
"""Part A: decompose ConicIP wall time and compute three ceilings.

Input : benchmark/results/phase-conicip-3881ac2.csv  (dev 3881ac2, M3 Pro, 1 thread)
Output: partA-decomp.csv, partA-ceilings.csv, and markdown tables on stdout.
"""
import csv, math, sys, os

W = "/Users/mpf/projects/Software/ConicIP.jl/.worktrees/large-scale-roadmap"
OUT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(W, "benchmark/results/phase-conicip-3881ac2.csv")

raw = open(SRC).read().split("\n", 1)[1]
rows = list(csv.DictReader(raw.splitlines()))
f = lambda r, k: float(r[k])

SHIFT = 0.1
def sgm(xs, shift=SHIFT):
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift

rec = []
for r in rows:
    d = {}
    d["instance"] = r["instance"]
    wall = f(r, "wall")
    d["wall"] = wall
    d["iters"] = int(r["iters"])
    # whole-call phases
    loop = f(r, "t_loop")
    outside = wall - loop                      # frontend+presolve+equil+setup+init+final+fallback+postsolve+other
    d["t_outside_loop"] = outside
    for k in ("t_frontend", "t_presolve", "t_equilibrate", "t_setup", "t_init",
              "t_postsolve", "t_other_call", "t_loop", "t_scaling", "t_kktupdate",
              "t_direction", "t_loop_rest", "t_other_loop", "t_kkt_total",
              "t_dir_base", "t_dir_refine", "t_dir_refine_resid",
              "t_ldl_factor", "t_ldl_solve", "t_ldl_resid",
              "t_residuals", "t_rhs", "t_linesearch"):
        d[k] = f(r, k)
    for k in ("n_kktupdate", "n_ldl_factor", "n_ldl_solve", "n_ldl_resid",
              "n_refine_resid", "n_solve"):
        d[k] = int(r[k])
    # LDL primitives (inclusive diagnostics, whole call: the initial
    # factorization/solve sit in t_setup/t_init, not in the loop children)
    d["fact"] = d["t_ldl_factor"]
    d["solve"] = d["t_ldl_solve"]
    d["refine_resid"] = d["t_ldl_resid"] + d["t_dir_refine_resid"]
    # glue inside kktupdate+direction that is neither factor, solve nor residual
    d["kkt_glue"] = d["t_kkt_total"] - (d["t_kktupdate"] and 0) - 0  # placeholder
    d["kkt_glue"] = max(d["t_kkt_total"] - (d["fact"] + d["solve"] + d["refine_resid"]), 0.0)
    d["assembly"] = max(d["t_kktupdate"] - d["fact"], 0.0)   # value update / K assembly (loop part)
    rec.append(d)

# ---------------- ceilings ----------------
for d in rec:
    w = d["wall"]
    d["c1_save"] = 0.5 * d["fact"]                                  # BLAS-3 halves factorization
    d["c2_save"] = 0.5 * (d["solve"] + d["refine_resid"])           # halve back-solves + refinement
    d["c3_save"] = max(w - d["t_kkt_total"], 0.0)                   # everything outside kktupdate+direction free
    for c in ("c1", "c2", "c3"):
        d[c + "_wall"] = max(w - d[c + "_save"], 0.0)
        d[c + "_frac"] = d[c + "_save"] / w

walls = [d["wall"] for d in rec]
tot = sum(walls)
print("# Part A — decomposition and ceilings (23 instances, dev 3881ac2)\n")
print("## A1. Wall-time decomposition (seconds; LDL rows are inclusive diagnostics over the whole call)\n")
hdr = ["instance", "wall", "it", "fact", "solve", "ref.resid", "NT scal",
       "loop_rest", "kkt glue", "outside loop", "Σparts−wall"]
print("| " + " | ".join(hdr) + " |")
print("|" + "---|" * len(hdr))
for d in sorted(rec, key=lambda x: -x["wall"]):
    d["overlap"] = (d["fact"] + d["solve"] + d["refine_resid"] + d["t_scaling"] +
                    d["t_loop_rest"] + d["kkt_glue"] + d["t_outside_loop"] - d["wall"])
    print("| {instance} | {wall:.4f} | {iters} | {fact:.4f} | {solve:.4f} | {refine_resid:.4f} | "
          "{t_scaling:.4f} | {t_loop_rest:.4f} | {kkt_glue:.4f} | {t_outside_loop:.4f} | "
          "{overlap:+.4f} |".format(**d))
agg = {k: sum(d[k] for d in rec) for k in
       ("wall", "fact", "solve", "refine_resid", "t_scaling", "t_loop_rest", "kkt_glue", "t_outside_loop")}
print("| **sum** | {wall:.4f} |  | {fact:.4f} | {solve:.4f} | {refine_resid:.4f} | {t_scaling:.4f} | "
      "{t_loop_rest:.4f} | {kkt_glue:.4f} | {t_outside_loop:.4f} |".format(**agg))
print("| **% of sum** | 100 |  | " + " | ".join(
    "{:.1f}".format(100 * agg[k] / agg["wall"]) for k in
    ("fact", "solve", "refine_resid", "t_scaling", "t_loop_rest", "kkt_glue", "t_outside_loop")) + " |")

print("\n## A2. Same, as a percentage of that instance's wall\n")
print("| " + " | ".join(hdr[:1] + ["wall s"] + hdr[3:]) + " |")
print("|" + "---|" * (len(hdr) - 1))
for d in sorted(rec, key=lambda x: -x["wall"]):
    print("| {} | {:.4f} | ".format(d["instance"], d["wall"]) + " | ".join(
        "{:.1f}".format(100 * d[k] / d["wall"]) for k in
        ("fact", "solve", "refine_resid", "t_scaling", "t_loop_rest", "kkt_glue", "t_outside_loop")) + " |")

print("\n## A3. Ceilings: predicted wall and saving\n")
h2 = ["instance", "wall", "(i) fact/2", "save %", "(ii) solve+ref/2", "save %",
      "(iii) only kktupdate+direction", "save %", "largest"]
print("| " + " | ".join(h2) + " |")
print("|" + "---|" * len(h2))
for d in sorted(rec, key=lambda x: -x["wall"]):
    best = max(("i", d["c1_save"]), ("ii", d["c2_save"]), ("iii", d["c3_save"]), key=lambda t: t[1])[0]
    print("| {} | {:.4f} | {:.4f} | {:.1f} | {:.4f} | {:.1f} | {:.4f} | {:.1f} | {} |".format(
        d["instance"], d["wall"], d["c1_wall"], 100 * d["c1_frac"], d["c2_wall"], 100 * d["c2_frac"],
        d["c3_wall"], 100 * d["c3_frac"], best))

print("\n## A4. Aggregate\n")
print("| aggregate | baseline | (i) fact/2 | (ii) solve+ref/2 | (iii) kkt-only |")
print("|---|---|---|---|---|")
s0 = tot
s1, s2, s3 = (sum(d[c + "_wall"] for d in rec) for c in ("c1", "c2", "c3"))
print("| sum (s) | {:.3f} | {:.3f} ({:+.1f}%) | {:.3f} ({:+.1f}%) | {:.3f} ({:+.1f}%) |".format(
    s0, s1, -100 * (s0 - s1) / s0, s2, -100 * (s0 - s2) / s0, s3, -100 * (s0 - s3) / s0))
g0 = sgm(walls)
g1, g2, g3 = (sgm([d[c + "_wall"] for d in rec]) for c in ("c1", "c2", "c3"))
print("| sgm, shift 0.1 (s) | {:.4f} | {:.4f} ({:+.1f}%) | {:.4f} ({:+.1f}%) | {:.4f} ({:+.1f}%) |".format(
    g0, g1, -100 * (g0 - g1) / g0, g2, -100 * (g0 - g2) / g0, g3, -100 * (g0 - g3) / g0))

# family rollup
fam = {}
def family(n):
    for p in ("lp-band", "qp-band", "socp-sumnorms"):
        if n.startswith(p): return p
    if n in ("nb", "nql30", "qssp30", "sched_50_50_scaled", "chainsing-1000-1"): return "cblib"
    if n == "issue10": return "issue10"
    return "maros-meszaros"
for d in rec:
    fam.setdefault(family(d["instance"]), []).append(d)
print("\n## A5. Per-family saving (sum of seconds saved)\n")
print("| family | n | wall | (i) | (ii) | (iii) |")
print("|---|---|---|---|---|---|")
for k, v in sorted(fam.items(), key=lambda kv: -sum(d["wall"] for d in kv[1])):
    w = sum(d["wall"] for d in v)
    print("| {} | {} | {:.3f} | {:.3f} ({:.0f}%) | {:.3f} ({:.0f}%) | {:.3f} ({:.0f}%) |".format(
        k, len(v), w, sum(d["c1_save"] for d in v), 100 * sum(d["c1_save"] for d in v) / w,
        sum(d["c2_save"] for d in v), 100 * sum(d["c2_save"] for d in v) / w,
        sum(d["c3_save"] for d in v), 100 * sum(d["c3_save"] for d in v) / w))

cols = ["instance", "wall", "iters", "fact", "solve", "refine_resid", "t_scaling",
        "t_loop_rest", "kkt_glue", "t_outside_loop", "t_kktupdate", "t_direction",
        "assembly", "n_ldl_factor", "n_ldl_solve", "n_ldl_resid",
        "c1_save", "c2_save", "c3_save", "c1_wall", "c2_wall", "c3_wall"]
with open(os.path.join(OUT, "partA-decomp.csv"), "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    for d in rec: w.writerow(d)
print("\nwrote partA-decomp.csv", file=sys.stderr)
