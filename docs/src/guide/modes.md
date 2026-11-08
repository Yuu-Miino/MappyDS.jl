# Concept of modes

A hybrid dynamical system is described by *modes*, smooth primitive pieces
of the dynamics, and *transition rules* that connect them. One iteration of
the Poincaré map starts in an initial mode, follows the rules from mode to
mode and ends when an end mode is reached.

## System functions

Every mode is built on a user-written function `f(y, p)` of the state `y`
and the parameters `p`, wrapped as one of three [`SystemFunction`](@ref)s:

| Wrapper | Meaning | Python |
| --- | --- | --- |
| [`VectorField(f, dim)`](@ref VectorField) | right-hand side of an autonomous ODE ``\dot y = f(y, p)`` | `@vector_field(dim)` |
| [`Mapping(f, dom_dim, cod_dim)`](@ref Mapping) | a map ``y \mapsto f(y, p)`` | `@mapping(dom_dim, cod_dim)` |
| [`Border(q, dim; direction)`](@ref Border) | a section ``q(y, p) = 0`` that terminates a continuous mode | `@border(dim, direction)` |

The functions are evaluated on `SVector`s and differentiated with
ForwardDiff, so they must be written generically:

```julia
using MappyDS, StaticArrays

henon = Mapping(2) do y, p
    x, u = y
    return SVector(1 - p.a * x^2 + u, p.b * x)
end
henon([0.0, 0.0], (a = 1.4, b = 0.3))          # [1.0, 0.0]
jacobian(henon, [0.0, 0.0], (a = 1.4, b = 0.3)) # [0.0 1.0; 0.3 0.0]
```

Return an `SVector`, a tuple or a vector of the declared length (a scalar
for a one-dimensional codomain), do not fix element types to `Float64`, and
read parameters from whatever container you pass as `p` (a `NamedTuple` is
recommended; `p` may be `nothing` when the function has no parameters).

## Continuous and discrete modes

A [`ContinuousMode`](@ref) integrates its vector field until one of its
borders is crossed in the declared direction. Borders are
[`Border`](@ref)s of the same dimension with distinct names; a mode without
borders is the time-`max_interval` flow map. A [`DiscreteMode`](@ref)
applies its map once.

```julia
izhikevich = VectorField(2) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
fire = Border((y, p) -> y[1] - 30, 2)
A = ContinuousMode(:A, izhikevich; borders = (fire,))
B = DiscreteMode(:B, Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2))
```

Both kinds expose [`evolve`](@ref), which returns a [`StepResult`](@ref)
with the end state, the time spent and the index of the border that fired,
and on request the Jacobian, the Hessian and the trajectory within the mode:

```julia
params = (a = 0.2, b = 0.2, c = -50.0, d = 2.0, I = 10.0)
res = evolve(A, [-50.0, -1.7], params; jacobian = true)
res.border, res.t, res.y[1]   # 1, ≈ 2.37, 30.0
```

The ODE solver is any OrdinaryDiffEq algorithm (`alg`, default `Tsit5()`),
with `reltol`, `abstol` and further keyword arguments forwarded to `solve`.
For high accuracy use e.g. `alg = Vern9()` from OrdinaryDiffEqVerner with
`reltol = 1e-10, abstol = 1e-12`.

## Transition rules and the Poincaré map

A [`PoincareMap`](@ref) takes the modes, the transition rules, the initial
mode and optionally the end modes. A rule gives the successor of a mode:

- a mode name, for a discrete mode, a borderless continuous mode or a
  continuous mode with a single border;
- a vector of mode names, one per border in the order of `borders`;
- a `Dict(:border_name => :mode_name)` covering every border.

```julia
pmap = PoincareMap(
    [A, B, DiscreteMode(:P, Mapping((y, p) -> y[2], 2, 1)),
        DiscreteMode(:Pinv, Mapping((u, p) -> SVector(30.0, u[1]), 1, 2))],
    Dict(:A => [:P], :B => :A, :P => :Pinv, :Pinv => :B),
    :Pinv,
)
```

The map is validated at construction: unknown names raise
[`UnknownModeError`](@ref); rules that do not fit their mode, missing rules
for reachable modes, dimension mismatches between a mode and its successor
and unreachable end modes raise [`TransitionError`](@ref). At run time a
continuous mode that reaches `max_interval` without hitting a border raises
[`ModeNotTerminatedError`](@ref), and `max_transitions` bounds the number of
transitions per iteration.

[`image`](@ref) computes the `n`-th image of a point, [`image_detail`](@ref)
additionally the derivatives, the mode history and the trajectory, and
[`trajectory`](@ref) the trajectory alone. When several end modes are given
(a system whose section can be reached in different modes), the result's
[`end_mode`](@ref) tells where the iteration ended and the `mode` keyword of
`image_detail` continues from there.

A smooth discrete map needs no transition rules: [`Diffeomorphism`](@ref)
wraps a [`Mapping`](@ref) with equal domain and codomain dimensions.
