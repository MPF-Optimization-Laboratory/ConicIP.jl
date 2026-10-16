#!/usr/bin/env python3
"""Ceiling (iv): the MEASURED BLAS-3 factorization speedups of Part B applied
to the per-instance factorization time of Part A.  No credit for instances we
did not measure."""
import csv, math, os

D = os.path.dirname(os.path.abspath(__file__))
SHIFT = 0.1
def sgm(xs, shift=SHIFT):
    return math.exp(sum(math.log(x + shift) for x in xs) / len(xs)) - shift

rows = list(csv.DictReader(open(os.path.join(D, "partA-decomp.csv"))))
# measured best CHOLMOD-Cholesky speedup over QDLDL on the same pattern
SPEED1 = {"lp-band-20000": 3.75/8.98, "lp-band-200000": 36.97/86.35,
          "qp-band-200000": 8.36/26.11, "socp-sumnorms-1500": 142.38/22.97,
          "nb": 15.98/11.81, "chainsing-1000-1": 0.48/1.38,
          "qssp30": 2.82/2.82, "issue10": 0.26/0.79}          # 1 BLAS thread
SPEED6 = {"lp-band-20000": 3.75/8.87, "lp-band-200000": 36.97/86.35,
          "qp-band-200000": 8.36/26.11, "socp-sumnorms-1500": 142.38/10.52,
          "nb": 15.98/11.79, "chainsing-1000-1": 0.48/1.37,
          "qssp30": 2.82/2.65, "issue10": 0.26/0.79}          # 6 BLAS threads

out = []
for r in rows:
    w = float(r["wall"]); fa = float(r["fact"]); n = r["instance"]
    s1 = max(SPEED1.get(n, 1.0), 1.0)
    s6 = max(SPEED6.get(n, 1.0), 1.0)
    out.append((n, w, fa, w - fa * (1 - 1/s1), w - fa * (1 - 1/s6), s1, s6))

print("## D. Ceiling (iv): measured supernodal/BLAS-3 factorization, applied to Part A\n")
print("| instance | wall s | factorization s | measured speedup 1thr | wall after 1thr | "
      "measured speedup 6thr | wall after 6thr | saving 6thr % |")
print("|---|---|---|---|---|---|---|---|")
for n, w, fa, w1, w6, s1, s6 in sorted(out, key=lambda t: -t[1]):
    if fa / w < 0.02 and s6 <= 1.0:
        continue
    print("| {} | {:.4f} | {:.4f} | {:.2f}× | {:.4f} | {:.2f}× | {:.4f} | {:.1f} |".format(
        n, w, fa, s1, w1, s6, w6, 100 * (w - w6) / w))
W = [t[1] for t in out]; W1 = [t[3] for t in out]; W6 = [t[4] for t in out]
print("\n| aggregate | baseline | BLAS-3 1 thread | BLAS-3 6 threads |")
print("|---|---|---|---|")
print("| sum (s) | {:.3f} | {:.3f} ({:+.1f}%) | {:.3f} ({:+.1f}%) |".format(
    sum(W), sum(W1), -100*(sum(W)-sum(W1))/sum(W), sum(W6), -100*(sum(W)-sum(W6))/sum(W)))
print("| sgm 0.1 (s) | {:.4f} | {:.4f} ({:+.1f}%) | {:.4f} ({:+.1f}%) |".format(
    sgm(W), sgm(W1), -100*(sgm(W)-sgm(W1))/sgm(W), sgm(W6), -100*(sgm(W)-sgm(W6))/sgm(W)))
