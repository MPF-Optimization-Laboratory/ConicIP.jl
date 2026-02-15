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

**Vector sets:** `Zeros`, `Nonnegatives`, `Nonpositives`, `SecondOrderCone`,
`PositiveSemidefiniteConeTriangle`

**Scalar sets:** `EqualTo`, `GreaterThan`, `LessThan`

## Limitations

- No quadratic objectives in JuMP form (Q must be zero; use the direct
  `conicIP` interface for QPs)
- No integer variables
- No indicator or SOS constraints
