ConicIP.jl: A Pure Julia Conic QP Solver
==

`ConicIP` (Conic **I**nterior **P**oint) is an interior-point solver inspired by [cvxopt](http://cvxopt.org/) for optimizing quadratic objectives with linear equality constraints, and polyhedral, second-order cone constraints. (Semidefinite cone constraints are available, but only supported as an experimental feature.) Because ConicIP is written in Julia, it allows abstract input and allows callbacks for its most computationally intensive internal routines.

### Installation

```julia
using Pkg
Pkg.add(url="https://github.com/MPF-Optimization-Laboratory/ConicIP.jl")
```

Requires Julia 1.10 or later.

### Basic Usage

ConicIP has the interface
```julia
sol = conicIP( Q , c , A , b , 𝐾 , G , d )
```
For the problem
```
minimize    ½yᵀQy - cᵀy
s.t         Ay ≧𝐾 b,  𝐾 = 𝐾₁  × ⋯ × 𝐾ⱼ
            Gy  = d
```

`𝐾` is a list of tuples of the form `(Cone Type ∈ {"R", "Q"}, Cone Dimension)` specifying the cone `𝐾ᵢ`. For example, the cone `𝐾 = 𝑅² × 𝑄³ × 𝑅²` has the following specification:

```julia
𝐾 = [ ("R",2) , ("Q",3),  ("R",2) ]
```

ConicIP returns `sol`, a structure containing error information (`sol.status`), the primal variables (`sol.y`), dual variables (`sol.v`, `sol.w`), and convergence information.

To solve the problem

```
minimize    ½yᵀQy - cᵀy
such that   y ≧ 0
```

for example, use `ConicIP` as follows

```julia
using ConicIP
using SparseArrays, LinearAlgebra

n = 1000

Q = sprandn(n, n, 0.1)
Q = Q'*Q
c = ones(n, 1)
A = sparse(1.0I, n, n)
b = zeros(n, 1)
𝐾 = [("R", n)]

sol = conicIP(Q, c, A, b, 𝐾, verbose=true);
```

### Usage with JuMP

ConicIP implements a [MathOptInterface](https://github.com/jump-dev/MathOptInterface.jl) wrapper, so it can be used as a solver in [JuMP](https://github.com/jump-dev/JuMP.jl).

```julia
using JuMP
using ConicIP

model = Model(ConicIP.Optimizer)
@variable(model, x[1:10] >= 0)
@constraint(model, sum(x) == 1.0)
@objective(model, Min, sum(x))
optimize!(model)
value.(x) # should be ≈ [0.1, 0.1, …, 0.1]
```
