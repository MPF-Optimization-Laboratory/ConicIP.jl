# Installation

## Requirements

- Julia 1.10 or later

## Installing ConicIP.jl

ConicIP.jl is not yet registered in the Julia General registry. Install it
directly from GitHub:

```julia
using Pkg
Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/ConicIP.jl")
```

## Verification

After installation, verify that the package loads correctly:

```julia
using ConicIP
```

## Optional: JuMP Integration

To use ConicIP as a JuMP solver, install JuMP as well:

```julia
using Pkg
Pkg.add("JuMP")
```

Then create a model with:

```julia
using JuMP, ConicIP
model = Model(ConicIP.Optimizer)
```

See the [JuMP Integration](@ref) guide for details.
