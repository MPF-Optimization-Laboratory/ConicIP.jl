# Second-Order Cone Programs

A second-order cone (SOC) constraint has the form

```
‖x‖ ≤ t
```

or equivalently `(t, x) ∈ Q`, where `Q` is the second-order (Lorentz) cone.
In ConicIP, this is specified as `("Q", dim)` where `dim = 1 + length(x)`.
The first component of the cone block is the scalar bound `t`.

## Example: Projection onto the Unit Ball

Project a point onto the unit Euclidean ball `{y : ‖y‖ ≤ 1}`.

```@example socp
using ConicIP, SparseArrays, LinearAlgebra

n = 3
Q = sparse(1.0I, n, n)
a = ones(n, 1)  # point to project

# SOC constraint: (t, y) ∈ Q with t = 1
# Encode as A*y ≥ b with A = [0...0; I], b = [-1; 0...0]
A = [spzeros(1, n); sparse(1.0I, n, n)]
b = [-ones(1, 1); zeros(n, 1)]
cone_dims = [("Q", n + 1)]

sol = conicIP(Q, Q * a, A, b, cone_dims;
              verbose=false, optTol=1e-7)
sol.status
```

```@example socp
round.(sol.y, digits=4)
```

The solution is the normalized vector `a / ‖a‖`:

```@example socp
round.(a ./ norm(a), digits=4)
```

## Example: Mixed Cones

Combine nonnegative (`"R"`) and second-order cone (`"Q"`) constraints.
Here we minimize a QP subject to both `y ≥ 0` and `‖y‖ ≤ 1`:

```@example socp_mixed
using ConicIP, SparseArrays, LinearAlgebra, Random
Random.seed!(42)

n = 5
Q = sparse(1.0I, n, n)
c = randn(n, 1)

# Stack constraints: first n rows are R+ (y ≥ 0),
# next n+1 rows are SOC (‖y‖ ≤ 1)
A = [sparse(1.0I, n, n);          # y ≥ 0
     spzeros(1, n);                # t = 1 (SOC slack)
     sparse(1.0I, n, n)]           # y components in SOC
b = [zeros(n, 1);                  # R+ bound
     -ones(1, 1);                  # t ≥ 1
     zeros(n, 1)]                  # SOC body
cone_dims = [("R", n), ("Q", n + 1)]

sol = conicIP(Q, c, A, b, cone_dims; verbose=false)
sol.status
```

```@example socp_mixed
round.(sol.y, digits=4)
```
