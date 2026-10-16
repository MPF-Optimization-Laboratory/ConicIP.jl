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
