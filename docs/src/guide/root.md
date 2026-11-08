# Periodic points and continuation

## Periodic points

[`find_cycle`](@ref) locates a periodic point of period ``n`` of a map
``T`` by solving ``T^n(y) - y = 0`` with the analytic Jacobian
``DT^n(y) - I`` supplied by the map. The problem is a
`SciMLBase.NonlinearProblem`; the default algorithm is `SimpleTrustRegion()`
from SimpleNonlinearSolve, and any algorithm of SimpleNonlinearSolve or
NonlinearSolve can be passed as `alg`.

```julia
using MappyDS, StaticArrays

henon = Mapping(2) do y, p
    x, u = y
    return SVector(1 - p.a * x^2 + u, p.b * x)
end
hmap = Diffeomorphism(henon)
p = (a = 1.4, b = 0.3)
fp = find_cycle(hmap, [0.6, 0.2], p)
fp.y        # ≈ [0.631354, 0.189406]
fp.eigvals  # ≈ [-1.923739, 0.155946]
two = find_cycle(hmap, [1.0, -0.1], p; period = 2)
```

The [`CycleResult`](@ref) carries the periodic point, the characteristic
multipliers (eigenvalues of ``DT^n``) and their eigenvectors, the number of
map evaluations and the final residual; `success` is `false` when the solver
did not converge (then `y`, `eigvals` and `eigvecs` are `nothing`).

## Natural-parameter continuation

[`trace_cycle`](@ref) follows a periodic point while one parameter is swept
linearly from its current value to an end value, starting each search from
the previous point:

```julia
steps = trace_cycle(hmap, fp.y, p, :a, 1.2; resolution = 21)
[(s.params.a, s.y, s.info[:eigvals]) for s in steps]
```

Each [`ContinuationStep`](@ref) holds the point, the parameters at that step
and, in `info`, the multipliers, eigenvectors, evaluation count and
residual. The sweep stops at the first parameter value where the search
fails, so the returned vector may be shorter than `resolution`.

The generic driver [`continuation`](@ref) sweeps a parameter for any step
function returning a [`StepFunctionResult`](@ref); `trace_cycle` is built on
it. Pseudo-arclength continuation (through folds) and the continuation of
bifurcation points arrive in MappyDS 0.2 and 0.3.
