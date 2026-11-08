# MappyDS.jl

[![CI](https://github.com/Yuu-Miino/MappyDS.jl/actions/workflows/CI.yml/badge.svg)](https://github.com/Yuu-Miino/MappyDS.jl/actions/workflows/CI.yml)
[![Stable docs](https://img.shields.io/badge/docs-stable-blue.svg)](https://yuu-miino.github.io/doc.mappy/julia/stable/)
[![Dev docs](https://img.shields.io/badge/docs-dev-blue.svg)](https://yuu-miino.github.io/doc.mappy/julia/dev/)
[![ColPrac: Contributor's Guide on Collaborative Practices for Community Packages](https://img.shields.io/badge/ColPrac-Contributor's%20Guide-blueviolet)](https://github.com/SciML/ColPrac)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

Poincaré maps, periodic points and bifurcations of **hybrid (smooth / non-smooth)
dynamical systems**. The Julia counterpart of the Python package
[`mappy-ds`](https://github.com/Yuu-Miino/mappy).

`MappyDS` implements the *mode concept*: a dynamical system is split into smooth
primitive pieces — **modes** — connected by transition rules. From the modes the
package automatically builds the Poincaré map of the system as a composite
diffeomorphism together with its **Jacobian matrix** and **Hessian tensor**, which
feed periodic-point search, local bifurcation analysis and parameter continuation.

> Y. Miino, "Mode concept to construct diffeomorphisms representing smooth/non-smooth
> dynamical systems", *Nonlinear Theory and Its Applications, IEICE*, 15(2), 284–298, 2024.
> [doi:10.1587/nolta.15.284](https://doi.org/10.1587/nolta.15.284)

> **Status:** MappyDS 0.1 covers the features of `mappy-ds` 0.1.0 (modes,
> Poincaré maps, Jacobians and Hessians, periodic points and natural-parameter
> continuation). Parameter derivatives, pseudo-arclength continuation, local
> bifurcations and the Makie viewer follow in 0.2 and 0.3; see the plan in
> [`docs/JULIA_RELEASE_PLAN.md`](https://github.com/Yuu-Miino/mappy/blob/master/docs/JULIA_RELEASE_PLAN.md)
> of the Python repository.

## Installation

```julia
pkg> add MappyDS
```

Until the package appears in the General registry:

```julia
pkg> add https://github.com/Yuu-Miino/MappyDS.jl
```

## Quick start

The Izhikevich neuron model fires (jumps) when the membrane potential reaches
`v = 30`. Describe the ODE, the border, the jump and the Poincaré section as
modes and let the package build the map:

```julia
using MappyDS, StaticArrays

izhikevich = VectorField(2) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
fire = Border((y, p) -> y[1] - 30, 2)
firing = Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2)
reduce = Mapping((y, p) -> y[2], 2, 1)                 # section (v = 30) -> u
revert = Mapping((u, p) -> SVector(30.0, u[1]), 1, 2)  # u -> section (v = 30)

pmap = PoincareMap(
    [
        ContinuousMode(:A, izhikevich; borders = (fire,)),
        DiscreteMode(:B, firing),
        DiscreteMode(:P, reduce),
        DiscreteMode(:Pinv, revert),
    ],
    Dict(:A => [:P], :B => :A, :P => :Pinv, :Pinv => :B),
    :Pinv,
)

params = (a = 0.2, b = 0.2, c = -50.0, d = 2.0, I = 10.0)
res = image_detail(pmap, -3.0, params; hessian = true)
res.y, res.jac, res.hes  # ≈ [-3.47838], [0.27080;;], [-0.19326;;;]
res.mode_history         # [:Pinv, :B, :A, :P, :Pinv]
```

Find a fixed point and follow it while a parameter changes:

```julia
fp = find_cycle(pmap, -4.5, merge(params, (I = 8.0,)))
fp.y, fp.eigvals  # ≈ [-4.64542], [0.22196 + 0.0im]

branch = trace_cycle(pmap, fp.y, merge(params, (I = 8.0,)), :I, 4.5; resolution = 50)
```

## Features

- **Modes**: `ContinuousMode` (an ODE integrated with OrdinaryDiffEq until a
  border is hit) and `DiscreteMode` (a map), freely composed through
  transition rules into a `PoincareMap`; `Diffeomorphism` for a smooth map.
- **Derivatives**: system functions are differentiated with ForwardDiff; the
  Jacobian and Hessian of every mode map, including the correction for the
  state-dependent hitting time at a border, and of their composition come
  from the variational equations.
- **Periodic points**: `find_cycle` solves `Tⁿ(y) = y` with the analytic
  Jacobian; `trace_cycle` and `continuation` follow solutions along a
  parameter.

## Documentation

<https://yuu-miino.github.io/doc.mappy/julia/stable/> (the Python documentation is
at <https://yuu-miino.github.io/doc.mappy/latest/>).

## Development

```julia
pkg> activate .
pkg> instantiate
pkg> test
```

Formatting follows [Runic](https://github.com/fredrikekre/Runic.jl); CI checks it.

## License

MIT — see [LICENSE](LICENSE).
