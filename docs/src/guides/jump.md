# JuMP Integration

ConicIP provides a [MathOptInterface](https://jump.dev/MathOptInterface.jl/)
wrapper, so it can be used as a solver backend for
[JuMP](https://jump.dev/JuMP.jl/).

## Setup

```julia
using JuMP, ConicIP
model = Model(ConicIP.Optimizer)
```

## Solver Options

Pass options at construction via an anonymous function:

```julia
model = Model(() -> ConicIP.Optimizer(verbose=false, optTol=1e-8, maxIters=200))
```

The available options are:

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `verbose` | `Bool` | `false` | Print solver iterations |
| `optTol` | `Float64` | `1e-6` | Optimality tolerance |
| `maxIters` | `Int` | `100` | Maximum iterations |

## Example: Simple LP

```@example jump
using JuMP, ConicIP

model = Model(() -> ConicIP.Optimizer(verbose=false, optTol=1e-6))

@variable(model, x[1:2] >= 0)
@objective(model, Min, x[1] + x[2])
@constraint(model, x[1] + x[2] >= 1)

optimize!(model)
termination_status(model)
```

```@example jump
round(objective_value(model), digits=6)
```

## Example: SOC Constraint

```@example jump_soc
using JuMP, ConicIP

model = Model(() -> ConicIP.Optimizer(verbose=false, optTol=1e-6))

@variable(model, x[1:2])
@variable(model, t)
@objective(model, Min, t)
@constraint(model, x[1] == 1)
@constraint(model, x[2] == 1)
@constraint(model, [t; x] in SecondOrderCone())

optimize!(model)
termination_status(model)
```

The minimum norm is `√2`:

```@example jump_soc
round(objective_value(model), digits=4)
```

## Example: Maximization

```@example jump_max
using JuMP, ConicIP

model = Model(() -> ConicIP.Optimizer(verbose=false, optTol=1e-6))

@variable(model, x[1:2] >= 0)
@objective(model, Max, x[1] + 2x[2])
@constraint(model, x[1] + x[2] <= 1)

optimize!(model)
termination_status(model)
```

```@example jump_max
round(objective_value(model), digits=6)
```

## Supported Constraints

| Constraint type | JuMP syntax |
|----------------|-------------|
| Nonnegative | `@variable(model, x >= 0)` or `@constraint(model, x in MOI.Nonnegatives(n))` |
| Nonpositive | `@constraint(model, x in MOI.Nonpositives(n))` |
| Zero (equality) | `@constraint(model, x .== 0)` or `@constraint(model, x in MOI.Zeros(n))` |
| Second-order cone | `@constraint(model, [t; x] in SecondOrderCone())` |
| PSD (experimental) | `@constraint(model, X in PSDCone())` |
| Scalar equal | `@constraint(model, x == 1)` |
| Scalar greater | `@constraint(model, x >= 1)` |
| Scalar less | `@constraint(model, x <= 1)` |

## Limitations

!!! warning "No quadratic objectives through JuMP"
    The MOI wrapper currently supports only **linear objectives**. For quadratic
    programs, use the direct [`conicIP`](@ref) interface.

Other limitations:

- No integer variables
- No indicator or SOS constraints
- Semidefinite support is experimental
