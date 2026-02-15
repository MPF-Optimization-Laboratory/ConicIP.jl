# ──────────────────────────────────────────────────────────────
#  SDP Evaluation Harness
#
#  Structured framework for running SDP problems through the
#  solver, collecting detailed diagnostics, and producing a
#  summary report.
# ──────────────────────────────────────────────────────────────

using Printf

"""
    SDPResult

Collected diagnostics from a single SDP solve attempt.
"""
struct SDPResult
    description  :: String
    kktsolver    :: String
    status       :: Symbol        # solver status
    iterations   :: Int
    solve_time   :: Float64       # seconds
    alloc_bytes  :: Int64
    primal_feas  :: Float64
    dual_feas    :: Float64
    mu_feas      :: Float64
    pobj         :: Float64
    dobj         :: Float64
    # Validation against known answers
    obj_error    :: Union{Float64, Nothing}   # |pobj - known_obj| if known
    X_error      :: Union{Float64, Nothing}   # ‖X - known_X‖∞   if known
    status_match :: Bool                       # solver status == known_status
    passed       :: Bool                       # all checks passed
    error_msg    :: Union{String, Nothing}     # exception message if solver threw
end

"""
    sdp_solve(prob; kktsolver, optTol, verbose, tol)

Run a single SDP problem through conicIP and validate results.
Returns an `SDPResult` with full diagnostics.
"""
function sdp_solve(prob;
        kktsolver = ConicIP.kktsolver_qr,
        optTol    = 1e-7,
        verbose   = false,
        tol       = 1e-3,
        maxIters  = 100)

    ks_name = _kktsolver_name(kktsolver)

    local sol
    local t, bytes
    error_msg = nothing

    try
        stats = @timed begin
            if isempty(prob.G) || size(prob.G, 1) == 0
                conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims;
                        kktsolver=kktsolver, optTol=optTol,
                        verbose=verbose, maxIters=maxIters)
            else
                conicIP(prob.Q, prob.c, prob.A, prob.b, prob.cone_dims,
                        prob.G, prob.d;
                        kktsolver=kktsolver, optTol=optTol,
                        verbose=verbose, maxIters=maxIters)
            end
        end
        sol = stats.value
        t = stats.time
        bytes = stats.bytes
    catch ex
        return SDPResult(
            prob.description, ks_name,
            :Error, 0, 0.0, 0,
            NaN, NaN, NaN, NaN, NaN,
            nothing, nothing,
            false, false,
            sprint(showerror, ex))
    end

    # Validate status
    status_match = (sol.status == prob.known_status)

    # Validate objective
    obj_error = nothing
    if prob.known_obj !== nothing && sol.status == :Optimal
        obj_error = abs(sol.pobj - prob.known_obj)
    end

    # Validate primal solution (PSD matrix)
    X_error = nothing
    if prob.known_X !== nothing && sol.status == :Optimal
        # Extract PSD solution from dual variable v
        # For a pure SDP with A=I, b=0: the variable v contains the SDP iterate
        # The primal y is the decision variable; the cone variable is in sol.v
        # But the standard test uses sol.y projected through mat()
        X_sol = _extract_sdp_matrix(sol, prob)
        if X_sol !== nothing
            X_error = norm(X_sol - prob.known_X, Inf)
        end
    end

    # Overall pass/fail
    passed = status_match
    if obj_error !== nothing
        passed = passed && (obj_error < tol)
    end
    if X_error !== nothing
        passed = passed && (X_error < tol)
    end

    return SDPResult(
        prob.description, ks_name,
        sol.status, sol.Iter, t, bytes,
        sol.prFeas, sol.duFeas, sol.muFeas, sol.pobj, sol.dobj,
        obj_error, X_error,
        status_match, passed,
        error_msg)
end

"""
    _extract_sdp_matrix(sol, prob)

Extract the PSD solution matrix from the solver output.
For problems where A=I and there is a single SDP block,
the dual variable `v` or the relationship `y = A\\(v + b)` gives us
the SDP iterate.
"""
function _extract_sdp_matrix(sol, prob)
    # For projection problems (Q=I, A=I, b=0), sol.y IS the vectorized solution
    # Check if there's a single SDP cone
    sdp_cones = filter(c -> c[1] == "S", prob.cone_dims)
    if length(sdp_cones) != 1
        return nothing
    end
    k = sdp_cones[1][2]
    # Find offset of SDP cone in the variable vector
    offset = 0
    for (ctype, cdim) in prob.cone_dims
        if ctype == "S"
            break
        end
        offset += cdim
    end
    # The primal iterate y gives us the SDP variable for QP formulations
    # For LP formulations (Q=0), the dual v gives the cone iterate
    if nnz(sparse(prob.Q)) > 0
        # QP: y is the primal, and for projection problems y is the answer
        x_vec = sol.y[offset+1:offset+k]
    else
        # LP: the cone variable is in v
        x_vec = sol.v[offset+1:offset+k]
    end
    return ConicIP.mat(x_vec)
end

"""
    _kktsolver_name(ks)

Human-readable name for a KKT solver.
"""
function _kktsolver_name(ks)
    if ks === ConicIP.kktsolver_qr
        return "qr"
    elseif ks === ConicIP.kktsolver_sparse
        return "sparse"
    else
        return "pivot(2x2)"
    end
end

"""
    run_sdp_harness(problems; kktsolver, optTol, verbose, tol)

Run a collection of SDP problems and return a vector of `SDPResult`s.
"""
function run_sdp_harness(problems;
        kktsolver = ConicIP.kktsolver_qr,
        optTol    = 1e-7,
        verbose   = false,
        tol       = 1e-3,
        maxIters  = 100)

    results = SDPResult[]
    for prob in problems
        r = sdp_solve(prob; kktsolver=kktsolver, optTol=optTol,
                      verbose=verbose, tol=tol, maxIters=maxIters)
        push!(results, r)
    end
    return results
end

"""
    run_sdp_harness_all_solvers(problems; optTol, verbose, tol)

Run all problems across all three KKT solvers. Returns a vector of SDPResults.
"""
function run_sdp_harness_all_solvers(problems;
        optTol  = 1e-7,
        verbose = false,
        tol     = 1e-3,
        maxIters = 100)

    solvers = [
        ConicIP.kktsolver_qr,
        ConicIP.kktsolver_sparse,
        pivot(ConicIP.kktsolver_2x2),
    ]
    results = SDPResult[]
    for ks in solvers
        for prob in problems
            r = sdp_solve(prob; kktsolver=ks, optTol=optTol,
                          verbose=verbose, tol=tol, maxIters=maxIters)
            push!(results, r)
        end
    end
    return results
end

"""
    print_sdp_report(results; io=stdout)

Print a formatted summary table of SDP evaluation results.
"""
function print_sdp_report(results; io=stdout)
    println(io)
    println(io, "=" ^ 120)
    println(io, "  SDP Evaluation Report")
    println(io, "=" ^ 120)
    @printf(io, "  %-45s │ %-10s │ %-9s │ %4s │ %8s │ %8s │ %8s │ %6s\n",
            "Problem", "KKT", "Status", "Iter", "ObjErr", "XErr", "Time(s)", "Pass")
    println(io, "─" ^ 120)

    n_pass = 0
    n_fail = 0
    n_error = 0

    for r in results
        obj_str = r.obj_error === nothing ? "    —   " : @sprintf("%8.1e", r.obj_error)
        X_str   = r.X_error === nothing   ? "    —   " : @sprintf("%8.1e", r.X_error)
        pass_str = r.passed ? "  OK" : (r.status == :Error ? " ERR" : "FAIL")

        @printf(io, "  %-45s │ %-10s │ %-9s │ %4d │ %s │ %s │ %8.4f │ %4s\n",
                first(r.description, 45),
                r.kktsolver,
                r.status,
                r.iterations,
                obj_str, X_str,
                r.solve_time,
                pass_str)

        if r.passed; n_pass += 1
        elseif r.status == :Error; n_error += 1
        else n_fail += 1
        end

        if r.error_msg !== nothing
            println(io, "    └─ ERROR: ", first(r.error_msg, 90))
        end
    end

    println(io, "─" ^ 120)
    @printf(io, "  Total: %d  |  Passed: %d  |  Failed: %d  |  Errors: %d\n",
            length(results), n_pass, n_fail, n_error)
    println(io, "=" ^ 120)
    println(io)

    return (total=length(results), passed=n_pass, failed=n_fail, errors=n_error)
end

# Utility to truncate strings
first(s::AbstractString, n::Int) = length(s) <= n ? rpad(s, n) : s[1:n-1] * "…"
