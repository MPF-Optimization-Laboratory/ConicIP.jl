# Formulating Semidefinite Programs

The same semidefinite program can be written in JuMP in two ways, and for
ConicIP the choice decides how large a problem is practical. This guide
explains why and shows how to keep a model in the form that is natural to
write while ConicIP solves the cheap one.

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

## Why the form matters

ConicIP's variables are the model's variables. In matrix-variable form
every entry of the upper triangle of `X` is one, so an order-`k` matrix
gives `k(k+1)/2` solver variables. The semidefinite scaling block is dense,
so ConicIP factors a dense matrix of that size each iteration: the work
grows like `k⁶` and the memory like `k⁴`. Past roughly `k ≈ 150` the
automatic KKT solver choice refuses the problem rather than allocate tens
of gigabytes (see the [KKT Solvers](@ref) guide for the `dense_bytes_max`
budget).

In linear-matrix-inequality form the solver variables are the `y`, here
`k` of them, and each iteration costs about `n k³ + n² k²` for `n`
variables. The max-cut relaxation above solves at `k = 300` in a few seconds
on a laptop in this form. In matrix-variable form, `k = 100` already takes
about 40 times longer than in this form.

## Keep the model, change what the solver sees

[Dualization.jl](https://jump.dev/Dualization.jl/stable/) builds the dual of
a JuMP model automatically. Wrapping the optimizer leaves the model in
matrix-variable form while ConicIP solves the dual:

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
  produces a matrix-variable model, the expensive form.
- The model has about as many equality constraints as matrix entries, for
  example a semidefinite relaxation with a constraint per pair of indices.
  Both forms are then large; ConicIP has no sparse semidefinite
  specialization yet.
- The model mixes a few semidefinite blocks with many other variables and
  constraints. The dual then carries those too; compare both on a small
  instance first.
