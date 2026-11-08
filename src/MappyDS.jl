"""
    MappyDS

Poincaré maps, periodic points and bifurcations of hybrid (smooth / non-smooth)
dynamical systems: the Julia counterpart of the Python package
[`mappy-ds`](https://github.com/Yuu-Miino/mappy).

`MappyDS` implements the *mode concept* (Miino, NOLTA, IEICE 15(2), 2024): a
dynamical system is split into smooth primitive pieces, *modes*, connected by
transition rules. From the modes the package builds the Poincaré map of the
system as a composite diffeomorphism together with its Jacobian matrix and
Hessian tensor, which in turn feed periodic-point search, local bifurcation
analysis and parameter continuation.

# Example

The Izhikevich neuron model with a firing threshold at `v = 30`:

```julia
using MappyDS, StaticArrays

izhikevich = VectorField(2) do y, p
    v, u = y
    return SVector(0.04 * v^2 + 5 * v + 140 - u + p.I, p.a * (p.b * v - u))
end
fire = Border((y, p) -> y[1] - 30, 2)
firing = Mapping((y, p) -> y + SVector(p.c - 30, p.d), 2)
reduce = Mapping((y, p) -> y[2], 2, 1)
revert = Mapping((u, p) -> SVector(30.0, u[1]), 1, 2)

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
res.y, res.jac, res.hes   # ≈ [-3.47838], [0.27080;;], [-0.19326;;;]
res.mode_history          # [:Pinv, :B, :A, :P, :Pinv]
```

The implementation plan and the API mapping from the Python package are in
`docs/JULIA_RELEASE_PLAN.md` of the `mappy` repository.
"""
module MappyDS

using LinearAlgebra: LinearAlgebra, I, dot
using StaticArrays: SVector, SArray
import ForwardDiff
import OrdinaryDiffEqTsit5
import PrecompileTools
import Printf
import SciMLBase
import SimpleNonlinearSolve

include("exceptions.jl")
include("functions.jl")
include("trajectory.jl")
include("modes.jl")
include("maps.jl")
include("continuation.jl")
include("root/cycle.jl")
include("plot_api.jl")

# Milestones (see the release plan, §5):
# 0.2.0: parameter derivatives (jac_p), arclength_continuation, trace_cycle_arclength
# 0.3.0: mixed derivatives (hes_p), find_local_bf, trace_local_bf(_arclength), Makie extension

export MappyError, UnknownModeError, TransitionError, ModeNotTerminatedError,
    DerivativeError
export SystemFunction, VectorField, Mapping, Border, dom_dim, cod_dim, jacobian, hessian,
    as_state
export TrajectorySegment, Trajectory, states, dim
export Mode, ContinuousMode, DiscreteMode, StepResult, evolve, border_names
export AbstractMap, PoincareMap, Diffeomorphism, MapResult, end_mode, image, image_detail,
    trajectory
export ContinuationStep, StepFunctionResult, continuation
export CycleResult, find_cycle, trace_cycle
export plot2d, plot_trajectory

end # module
