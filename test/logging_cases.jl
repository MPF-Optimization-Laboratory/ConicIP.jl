# Fixed problems whose solver logs are pinned by test/logging_tests.jl.
# Each case is `name => f`, where `f(; kw...)` runs one solve with the log
# on and its output going to `stdout` (the keywords are passed to the
# solver); the harness captures and compares it.
# Shared with the local before/after diff used when the log code changes,
# so keep every case deterministic.

using ConicIP, LinearAlgebra, SparseArrays

# Raises a KKT failure from the first factorization (the initial point).
_failing_kktsolver(Q, A, G, cone_dims) =
  (F, F⁻ᵀ) -> throw(LinearAlgebra.SingularException(1))

const LOG_QP = (Q = [2.0 0.5 0.0; 0.5 2.0 0.5; 0.0 0.5 2.0], c = [1.0, 2.0, 3.0],
                A = sparse(1.0I, 3, 3), b = zeros(3), K = [("R", 3)])

const LOG_CASES = Pair{String,Function}[
  # The QP of docs/src/tutorials/verbose.jl
  "qp_optimal" => (; kw...) -> conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K;
                                verbose = true, kw...),
  # kkt column filled by kktsolver_ldl
  "qp_ldl" => (; kw...) -> conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K;
                            verbose = true, kw..., kktsolver = ConicIP.kktsolver_ldl),
  # cc column
  "qp_correctors" => (; kw...) -> conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K;
                                   verbose = true, kw..., kktsolver = ConicIP.kktsolver_qr,
                                   centralityCorrectors = 2),
  # Iteration limit: :Abandoned
  "qp_abandoned" => (; kw...) -> conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K;
                                  verbose = true, kw..., maxIters = 2, certFallback = false),
  # max y₁ + y₂ over the unit second-order cone ball
  "soc" => (; kw...) -> conicIP(zeros(2, 2), [1.0, 1.0],
                         sparse([0.0 0.0; 1.0 0.0; 0.0 1.0]), [-1.0, 0.0, 0.0],
                         [("Q", 3)]; verbose = true, kw...),
  # min ⟨C, X⟩ over tr X = 1, X ⪰ 0 (2×2, vecm form)
  "sdp" => (; kw...) -> conicIP(zeros(3, 3), -[2.0, sqrt(2.0), 3.0], sparse(1.0I, 3, 3),
                         zeros(3), [("S", 3)], sparse([1.0 0.0 1.0]), [1.0];
                         verbose = true, kw...),
  # y ≥ 1 and −y ≥ 0
  "infeasible" => (; kw...) -> conicIP(zeros(1, 1), [0.0], sparse(reshape([1.0, -1.0], 2, 1)),
                                [1.0, 0.0], [("R", 2)]; verbose = true, kw...),
  # min −y over y ≥ 0
  "dual_infeasible" => (; kw...) -> conicIP(zeros(1, 1), [1.0], sparse(reshape([1.0], 1, 1)),
                                     [0.0], [("R", 1)]; verbose = true, kw...),
  # 0ᵀy = 1
  "struct_infeasible" => (; kw...) -> conicIP(zeros(2, 2), [1.0, 1.0], sparse(1.0I, 2, 2),
                                       zeros(2), [("R", 2)], spzeros(1, 2), [1.0];
                                       verbose = true, kw...),
  # Factorization failure at the initial point
  "kkt_error" => (; kw...) -> conicIP(LOG_QP.Q, LOG_QP.c, LOG_QP.A, LOG_QP.b, LOG_QP.K;
                               verbose = true, kw..., kktsolver = _failing_kktsolver),
  # Singleton row (fixes y₁) and a redundant pair of rows
  "preprocess" => (; kw...) -> preprocess_conicIP(Matrix(1.0I, 3, 3), ones(3),
                                           sparse(1.0I, 3, 3), zeros(3), [("R", 3)],
                                           sparse([1.0 0.0 0.0; 0.0 1.0 1.0; 0.0 2.0 2.0]),
                                           [1.0, 1.0, 2.0];
                                           verbose = true, kw..., rank_check = :always),
  # Inconsistent equalities y₁ + y₂ = 1, y₁ + y₂ = 2: a presolve verdict
  "preprocess_infeasible" => (; kw...) -> preprocess_conicIP(Matrix(1.0I, 2, 2), ones(2),
                                     sparse(1.0I, 2, 2), zeros(2), [("R", 2)],
                                     sparse([1.0 1.0; 1.0 1.0]), [1.0, 2.0];
                                     verbose = true, kw..., rank_check = :always),
]

# Captured stdout of `f()`.
function capture_log(f)
  mktemp() do path, io
    redirect_stdout(io) do
      f()
    end
    flush(io)
    read(path, String)
  end
end

# The structure of a log with the numbers taken out: every number (and
# NaN/Inf) becomes `#`, whitespace runs collapse, blank lines go, and
# consecutive identical lines are merged. Iteration counts and digits
# depend on the BLAS and the Julia version; the skeleton does not.
function log_skeleton(s::AbstractString)
  s = replace(s, r"\e\[[0-9;]*m" => "")
  # an exception's text depends on the Julia version
  s = replace(s, r"(KKT solve failed \([^)]*\)): .*\)$"m => s"\1: <exception>)")
  s = replace(s, r"[-+]?(?:\d+(?:\.\d+)*(?:[eE][-+]?\d+)?|\bNaN\b|\bInf\b)" => "#")
  out = String[]
  for line in split(s, '\n')
    l = strip(replace(line, r"\s+" => " "))
    isempty(l) && continue
    (isempty(out) || out[end] != l) && push!(out, l)
  end
  return join(out, '\n') * '\n'
end
