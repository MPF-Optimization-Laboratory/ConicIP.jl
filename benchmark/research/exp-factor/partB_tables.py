#!/usr/bin/env python3
"""Assemble the Part B tables from partB-*.csv / partB-*.stats."""
import csv, glob, os, sys

D = os.path.dirname(os.path.abspath(__file__))
ORDER = ["lp-band-20000", "lp-band-200000", "qp-band-200000", "socp-sumnorms-1500",
         "nb", "chainsing-1000-1", "qssp30", "issue10"]

stats, times = {}, {}
for n in ORDER:
    sp = os.path.join(D, "partB-%s.stats" % n)
    if not os.path.exists(sp):
        continue
    stats[n] = dict(l.split("=", 1) for l in open(sp).read().split())
    times[n] = {}
    # the `what` labels contain commas, so split from the ends, not with csv
    for line in open(os.path.join(D, "partB-%s.csv" % n)).read().splitlines()[1:]:
        p = line.split(",")
        note = p[-1]
        what = ",".join(p[1:-3])
        times[n][(what, p[-3])] = (float(p[-2]), note)

def t(n, what, bl="1"):
    v = times.get(n, {}).get((what, bl))
    if v is None or v[0] != v[0]:
        return None
    return v[0]

def f(x, d=2):
    return "—" if x is None else ("%.*f" % (d, 1e3 * x))

print("## B1. Captured KKT matrix (mid-run factorization)\n")
h = ["instance", "n_K", "nnz(triu K)", "nnz(L) QDLDL", "fill nnz(L)/nnz(K)",
     "flops Σcolcount²", "max supernode chain", "flop frac in chains ≥16",
     "CHOLMOD nsuper (forced)", "CHOLMOD auto picks"]
print("| " + " | ".join(h) + " |")
print("|" + "---|" * len(h))
for n in ORDER:
    if n not in stats: continue
    s = stats[n]
    print("| {} | {} | {} | {} | {:.2f} | {:.3g} | {} | {:.2f} | {} | {} |".format(
        n, s["N"], s["nnz_triuK"], s["nnz_L_qdldl"], float(s["fill_ratio_L_over_triuK"]),
        float(s["flops_sum_colcount2"]), s["max_chain_len"],
        float(s["flop_frac_in_chains_ge16"]), s["cholmod_nsuper_forced"],
        "supernodal" if s["cholmod_is_super"] == "1" else "simplicial"))

print("\n## B2. Numeric factorization time, ms (best of 5, warmed, 1 Julia thread)\n")
cols = [("QDLDL refactor!", "qdldl refactor!", "1"),
        ("CHOLMOD ldlt!(K) 1thr", "cholmod ldlt! (K, numeric only)", "1"),
        ("CHOLMOD ldlt!(K) 6thr", "cholmod ldlt! (K, numeric only)", "6"),
        ("CHOLMOD chol! proxy auto 1thr", "cholmod cholesky! (SPD proxy, auto)", "1"),
        ("CHOLMOD chol! proxy auto 6thr", "cholmod cholesky! (SPD proxy, auto)", "6"),
        ("CHOLMOD chol! proxy SUPER 1thr", "cholmod cholesky! (SPD proxy, SUPERNODAL)", "1"),
        ("CHOLMOD chol! proxy SUPER 6thr", "cholmod cholesky! (SPD proxy, SUPERNODAL)", "6"),
        ("UMFPACK lu! 1thr", "umfpack lu! refactorization", "1"),
        ("UMFPACK lu! 6thr", "umfpack lu! refactorization", "6"),
        ("LDLFactorizations", "LDLFactorizations ldl_factorize!", "1")]
print("| instance | " + " | ".join(c[0] for c in cols) + " |")
print("|" + "---|" * (len(cols) + 1))
for n in ORDER:
    if n not in times: continue
    print("| " + n + " | " + " | ".join(f(t(n, w, b)) for _, w, b in cols) + " |")

print("\n## B3. Best BLAS-3 speedup over QDLDL on the same pattern\n")
print("| instance | QDLDL ms | best CHOLMOD-Cholesky ms (threads) | speedup | nnz(L) CHOLMOD / QDLDL |")
print("|---|---|---|---|---|")
for n in ORDER:
    if n not in times: continue
    q = t(n, "qdldl refactor!")
    cands = [(t(n, "cholmod cholesky! (SPD proxy, auto)", b), b) for b in ("1", "6")] + \
            [(t(n, "cholmod cholesky! (SPD proxy, SUPERNODAL)", b), b) for b in ("1", "6")]
    cands = [c for c in cands if c[0] is not None]
    if not q or not cands:
        continue
    bt, bb = min(cands)
    s = stats[n]
    ratio = float(s["nnz_L_cholmod_spdproxy"]) / float(s["nnz_L_qdldl"])
    print("| {} | {:.2f} | {:.2f} ({}) | {:.2f}× | {:.2f} |".format(n, 1e3 * q, 1e3 * bt, bb, q / bt, ratio))

print("\n## B4. One back-solve, ms\n")
bs = [("QDLDL solve!", "qdldl solve! (1 rhs)", "1"),
      ("CHOLMOD \\ simplicial proxy", "cholmod \\ (1 rhs, SPD proxy)", "1"),
      ("CHOLMOD \\ supernodal proxy", "cholmod \\ (1 rhs, SUPERNODAL proxy)", "1"),
      ("CHOLMOD \\ ldlt(K)", "cholmod \\ (1 rhs, ldlt of K)", "1"),
      ("UMFPACK \\", "umfpack \\ (1 rhs)", "1")]
print("| instance | " + " | ".join(c[0] for c in bs) + " |")
print("|" + "---|" * (len(bs) + 1))
for n in ORDER:
    if n not in times: continue
    print("| " + n + " | " + " | ".join(f(t(n, w, b), 3) for _, w, b in bs) + " |")
