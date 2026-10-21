# Formulating Semidefinite Programs

The same semidefinite program can be written in JuMP in two ways. ConicIP
solves both efficiently in the common cases, but not every matrix-variable
model qualifies; this guide explains which do, and what to do with one that
does not.

## Two forms of one problem

**Matrix-variable form.** The matrix is a decision variable:

```julia
@variable(model, X[1:k, 1:k], PSD)
@constraint(model, [i = 1:k], X[i, i] == 1)
@objective(model, Max, dot(L, X) / 4)
```

**Linear-matrix-inequality form.** The matrix is an affine expression in a
few variables:

```julia
@variable(model, y[1:k])
@constraint(model, Symmetric(Matrix(Diagonal(y)) - L / 4) in PSDCone())
@objective(model, Min, sum(y))
```

These two models are duals of each other and have the same optimal value
(here, the max-cut relaxation of a graph with Laplacian `L`).

## How ConicIP solves each

ConicIP's variables are the model's variables. An order-`k` matrix
variable therefore brings `k(k+1)/2` of them, one per entry of the upper
triangle.

- **Linear-matrix-inequality form.** The `n` variables are few. Each
  iteration forms an `n × n` reduced matrix directly from the coefficient
  matrices of the inequality, exploiting their sparsity, and factors it.
  The dense worst case is about `n k³ + n² k²` operations.
- **Matrix-variable form in standard form.** ConicIP recognizes the
  standard form below and eliminates the matrix entries exactly. It then
  factors only a `p × p` matrix for the `p` equality constraints
  ([`kktsolver_sdpschur`](@ref ConicIP.kktsolver_sdpschur)), at a cost of
  about `p k³ + p² k² + p³` per iteration. The max-cut model above is in
  this form.
- **Matrix-variable form with a few extras.** Free variables, bounds or a
  second cone on some entries of `X`, or a quadratic objective on the
  other variables: ConicIP still eliminates the matrix entries, and
  factors a matrix whose size is the number of other variables plus the
  equality constraints plus the inequality rows that touch entries of `X`
  (the same solver, its *mixed form*). It does so when that size is at
  most half the variable count.
- **Matrix-variable form otherwise.** ConicIP factors a dense matrix in
  all `k(k+1)/2` variables. That costs on the order of `k⁶` per iteration
  and `k⁴` memory, and past roughly `k ≈ 150` the automatic KKT solver
  choice refuses the problem rather than allocate tens of gigabytes.

A model is in **standard form** when:

- the objective is linear;
- every variable is an entry of exactly one PSD (or nonnegative, or
  second-order cone) constraint written on variables
  (`@variable(model, X, PSD)`, `x[1:n] >= 0` with `x in Nonnegatives`);
- the remaining constraints are linear equalities.

The mixed form also covers free variables alongside the matrix (theta
with a free scalar, `min t` with `X[i, i] == t - 1`), a few bounds or a
second cone on entries of `X` (a Shor relaxation `[1 x'; x X] ⪰ 0` with
`0 ≤ x ≤ 1`), and a quadratic objective in the non-matrix variables.

What still leaves both: a quadratic objective in `X` itself, and
constraints on so many entries of `X` that the kept rows rival the
variable count — a doubly nonnegative relaxation, `X .>= 0` alongside
`X ⪰ 0`, keeps one row per entry.

## Outside both forms: let Dualization.jl hand ConicIP the dual

[Dualization.jl](https://jump.dev/Dualization.jl/stable/) builds the dual
of a JuMP model automatically. Wrapping the optimizer leaves the model as
written while ConicIP solves the dual, whose variables are the original
constraints:

```julia
using JuMP, ConicIP, Dualization

model = Model(dual_optimizer(ConicIP.Optimizer))
@variable(model, X[1:k, 1:k], PSD)
c = @constraint(model, [i = 1:k], X[i, i] == 1)
@objective(model, Max, dot(L, X) / 4)
optimize!(model)

value.(X)   # primal values, as for a direct solve
dual.(c)    # constraint duals map back the same way
```

Solver options pass through the wrapper:

```julia
model = Model(dual_optimizer(optimizer_with_attributes(ConicIP.Optimizer,
                                                       "optTol" => 1e-8)))
set_attribute(model, "maxIters", 200)
```

The test suite (`test/dualization_tests.jl`) checks that this route returns
the same optimum, primal values, equality and semidefinite constraint duals
as a direct solve, and that `INFEASIBLE` and `DUAL_INFEASIBLE` are reported
with their certificates mapped back to the original model.

## When dualizing does not help

- The model is already in linear-matrix-inequality form. Dualizing it
  produces a matrix-variable model.
- The model has about as many equality constraints as matrix entries, for
  example a relaxation with a constraint per pair of indices. Both forms
  are then large.
- The model mixes a few semidefinite blocks with many other variables and
  constraints. The dual then carries those too; compare both on a small
  instance first.
