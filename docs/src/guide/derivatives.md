# Derivatives

## Conventions

For a map ``y_0 \mapsto y`` the Jacobian matrix and the Hessian tensor are

```math
\mathtt{jac}[i, j] = \frac{\partial y_i}{\partial y_{0,j}}, \qquad
\mathtt{hes}[i, j, k] = \frac{\partial^2 y_i}{\partial y_{0,j}\, \partial y_{0,k}},
```

stored as a `Matrix{Float64}` of size `(cod_dim, dom_dim)` and an
`Array{Float64, 3}` of size `(cod_dim, dom_dim, dom_dim)`. The same
conventions hold for [`jacobian`](@ref) and [`hessian`](@ref) of a system
function, for the `jac` and `hes` fields of a [`StepResult`](@ref) and for
those of a [`MapResult`](@ref).

## System functions

The derivatives of a system function are computed with ForwardDiff on
`SVector`s: the Jacobian with `ForwardDiff.jacobian` and the Hessian as the
Jacobian of the vectorised Jacobian (nested dual numbers). Unlike the SymPy
tracing of the Python package this works for arbitrary Julia code, including
branches on the state, as long as the function is differentiable where it is
evaluated and does not fix element types.

## Continuous modes

When the Jacobian or the Hessian of a [`ContinuousMode`](@ref) is requested,
the variational equations

```math
\dot J = Df(\varphi)\, J, \qquad
\dot H_{ijk} = \sum_{l, m} (D^2 f)_{ilm} J_{lj} J_{mk} + \sum_l (Df)_{il} H_{ljk}
```

are integrated together with the state ``\varphi`` (as one augmented ODE),
and the result is corrected for the dependence of the hitting time ``\tau``
on the initial state. With ``f`` the vector field, ``q`` the border function
and ``s = \nabla q \cdot f`` at the hitting point, the mode map
``y_0 \mapsto \varphi(\tau(y_0), y_0)`` has Jacobian ``B J`` with

```math
B = I - \frac{f \otimes \nabla q}{s},
```

and its Hessian follows by differentiating once more with respect to
``y_0``, using ``\partial\tau/\partial y_0 = -(\nabla q\, J)/s``
[miino2024](@cite). The composition of modes uses the chain rule

```math
D(g \circ f) = Dg\, Df, \qquad
D^2(g \circ f) = D^2 g[Df, Df] + Dg\, D^2 f.
```

Integrating the variational equations changes the adaptive step sizes of
the solver, so the image computed with derivatives differs from the one
computed without by an amount of the order of the solver tolerance.

## Parameter derivatives

With `parameter_jacobian = true`, [`evolve`](@ref) and
[`image_detail`](@ref) also return `jac_p[i, m] = ∂y_i/∂p_m`, one column per
key of the parameter `NamedTuple` in its order (`MapResult.param_keys`).
For a system function, [`parameter_jacobian`](@ref) gives ``∂f/∂p`` by
ForwardDiff over the parameter values. For a continuous mode the
sensitivity equations

```math
\dot P = Df(\varphi)\, P + \partial_p f(\varphi), \qquad P(0) = 0,
```

are integrated with the state, and the border correction becomes
``B P + G`` with ``G = -f \otimes \partial_p q / s``, so borders that depend
on the parameters are handled. Across modes the chain rule reads
``\partial_p (g \circ f) = Dg\, \partial_p f + \partial_p g``.

## Checking derivatives

Finite differences are a convenient check:

```julia
using MappyDS, StaticArrays, OrdinaryDiffEqVerner

mode = ContinuousMode(:A, izhikevich; borders = (fire,), alg = Vern9(), reltol = 1e-10, abstol = 1e-12)
res = evolve(mode, y0, params; hessian = true)
f(y) = evolve(mode, y, params).y
h = 1e-4
fd_jac = reduce(hcat, [(f(y0 + h * e) - f(y0 - h * e)) / 2h for e in eachcol(I(2))])
res.jac ≈ fd_jac  # within ~1e-6
```
