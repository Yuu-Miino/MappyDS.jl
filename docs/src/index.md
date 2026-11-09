```@meta
CurrentModule = MappyDS
```

# MappyDS.jl

Poincaré maps, periodic points and bifurcations of **hybrid (smooth /
non-smooth) dynamical systems**: the Julia counterpart of the Python package
[mappy-ds](https://yuu-miino.github.io/doc.mappy/latest/).

`MappyDS` implements the *mode concept* [miino2024](@cite): a dynamical system
is split into smooth primitive pieces, **modes**, connected by transition
rules. From the modes the package builds the Poincaré map of the system as a
composite diffeomorphism together with its **Jacobian matrix** and **Hessian
tensor**, which feed periodic-point search, local bifurcation analysis and
parameter continuation.

!!! note "Status"
    MappyDS 0.3 covers the features of `mappy-ds` 0.3.0: modes, Poincaré maps,
    Jacobians, Hessians, parameter and mixed derivatives, periodic points,
    natural-parameter and pseudo-arclength continuation, local bifurcations
    and their continuation, and the Makie viewer. See [Python package](@ref)
    for the feature matrix.

## Installation

```julia
pkg> add MappyDS
```

The package is not registered yet; until then use
`pkg> add https://github.com/Yuu-Miino/MappyDS.jl`.

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
fire = Border((y, p) -> y[1] - 30, 2)                      # v = 30
firing = Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2)  # (v, u) -> (c, u + d)
reduce = Mapping((y, p) -> y[2], 2, 1)                     # section (v = 30) -> u
revert = Mapping((u, p) -> SVector(30.0, u[1]), 1, 2)      # u -> section (v = 30)

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
res.y, res.jac, res.hes      # ≈ [-3.47838], [0.27080;;], [-0.19326;;;]
res.mode_history             # [:Pinv, :B, :A, :P, :Pinv]
```

Find a fixed point and follow it while a parameter changes:

```julia
fp = find_cycle(pmap, -4.5, merge(params, (I = 8.0,)))
fp.y, fp.eigvals             # ≈ [-4.64542], [0.22196 + 0.0im]

branch = trace_cycle(pmap, fp.y, merge(params, (I = 8.0,)), :I, 4.5; resolution = 50)
```

Smooth systems are a special case: a discrete map is a single mode
([`Diffeomorphism`](@ref)), and a periodically forced oscillator becomes
autonomous by adding time as a state and resetting it on a section.

## Features

- **Modes**: [`ContinuousMode`](@ref) (an ODE integrated with OrdinaryDiffEq
  until a border is hit) and [`DiscreteMode`](@ref) (a map), freely composed
  through transition rules.
- **Derivatives**: system functions are differentiated with ForwardDiff; the
  Jacobian and Hessian of every mode map, including the correction for the
  state-dependent hitting time at a border, and of their composition are
  computed from the variational equations.
- **Periodic points**: [`find_cycle`](@ref) solves ``T^n(y) = y`` with the
  analytic Jacobian; [`trace_cycle`](@ref) and [`continuation`](@ref) follow
  solutions along a parameter.

## Contents

```@contents
Pages = ["guide/modes.md", "guide/derivatives.md", "guide/root.md", "guide/plotting.md", "api.md", "python.md"]
Depth = 1
```
