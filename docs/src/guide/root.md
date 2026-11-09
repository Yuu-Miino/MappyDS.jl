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
it.

## Pseudo-arclength continuation

Natural-parameter continuation fails at a fold, where the branch turns back
in the parameter. [`trace_cycle_arclength`](@ref) treats the periodic point
and the parameter together as a point ``u = (y, λ)`` on the curve
``T^n(y, λ) - y = 0`` and advances along the curve itself (Keller's
pseudo-arclength method), using the analytic tangent from `jac` and `jac_p`:

```julia
f = Mapping((y, p) -> y[1] + p.p - y[1]^2, 1)   # fixed points ±√p, fold at p = 0
steps = trace_cycle_arclength(Diffeomorphism(f), [0.5], (p = 0.25,), :p;
    direction = -1, step = 0.05, step_max = 0.1, bounds = (-1.0, 0.25))
[(s.params.p, s.y[1], s.info[:events]) for s in steps]
```

For a Poincaré map of an ODE, integrate with tolerances well below the
corrector's `newton_tol` (default `1e-10`), e.g. `reltol = 1e-8,
abstol = 1e-10`: the noise floor of the map limits how far the Newton
correction can converge.

Between consecutive points the sign changes of ``\det(DT^n - I)``,
``\det(DT^n + I)`` and of the number of multipliers outside the unit circle
flag `:fold`, `:branch_point`, `:period_doubling` and `:neimark_sacker`
events in `info[:events]`; the continuation of the bifurcation points
themselves arrives in MappyDS 0.3. The generic algorithm is
[`arclength_continuation`](@ref), which follows the zero set of any
``F : \mathbb{R}^{m+1} \to \mathbb{R}^m`` given `fun(u) -> (F, J)`.
